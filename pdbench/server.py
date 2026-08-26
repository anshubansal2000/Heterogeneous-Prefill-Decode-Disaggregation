"""Launch and manage vLLM OpenAI-compatible servers for each experiment config.

Config A  Aggregated NVIDIA          : one `vllm serve` on the GPU.
Config E  CPU prefill -> GPU decode  : a disaggregated pair (CPU producer + GPU
                                       consumer) behind a proxy, KV via a connector.
Config C  Same-vendor NVIDIA disagg  : two GPU engines (producer + consumer).

For the first milestone we implement A end-to-end and provide the disaggregated
launch scaffolding (E/C) with the exact engine args; those are enabled once the
Phase-0 connector spike passes on the pod.
"""
from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.request


def _wait_healthy(base_url: str, timeout: float = 1200.0, log_path: str | None = None) -> bool:
    url = base_url.rstrip("/") + "/health"
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False


class VLLMServer:
    """A single `vllm serve` subprocess."""

    def __init__(self, model: str, port: int, engine_args: list[str] | None = None,
                 env: dict | None = None, log_path: str | None = None):
        self.model = model
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}"
        self.engine_args = engine_args or []
        self.env = {**os.environ, **(env or {})}
        self.log_path = log_path
        self.proc: subprocess.Popen | None = None

    def start(self) -> "VLLMServer":
        cmd = ["vllm", "serve", self.model, "--port", str(self.port)] + self.engine_args
        if self.log_path:
            os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        logf = open(self.log_path, "w") if self.log_path else subprocess.DEVNULL
        print("  launching:", " ".join(shlex.quote(c) for c in cmd), flush=True)
        self.proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                     env=self.env)
        return self

    def wait_ready(self, timeout: float = 1200.0) -> bool:
        if _wait_healthy(self.base_url, timeout, self.log_path):
            return True
        # surface last log lines to help debugging
        if self.log_path and os.path.exists(self.log_path):
            with open(self.log_path, errors="replace") as f:
                tail = f.readlines()[-25:]
            print("  server did not become healthy; log tail:\n   " + "   ".join(tail))
        return False

    def stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.send_signal(signal.SIGINT)
                self.proc.wait(timeout=30)
            except Exception:
                self.proc.kill()

    def scrape_metrics(self) -> str:
        try:
            with urllib.request.urlopen(self.base_url + "/metrics", timeout=10) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:  # noqa: BLE001
            return f"# scrape failed: {e}\n"


# ------------------------------------------------------------------ config A ---
def launch_aggregated(model: str, port: int, gpu_mem_util: float, kv_dtype: str,
                      max_model_len: int, extra: list[str], log_path: str) -> VLLMServer:
    args = [
        "--gpu-memory-utilization", str(gpu_mem_util),
        "--max-model-len", str(max_model_len),
        "--no-enable-prefix-caching",   # match Moreh: isolate raw compute
    ] + extra
    if kv_dtype and kv_dtype != "auto":
        args = ["--kv-cache-dtype", kv_dtype] + args
    return VLLMServer(model, port, args, log_path=log_path).start()


# ------------------------------------------------------- config E / C (disagg) -
# Producer (prefill) and consumer (decode) each run vLLM with a KV connector.
# The proxy (examples/online_serving/disaggregated_serving) routes prefill->P,
# decode->D. We keep these as builders so run.py can enable them post-spike.

def disagg_engine_args(role: str, connector: str, block_size: int,
                       max_model_len: int, kv_dtype: str = "auto") -> list[str]:
    assert role in ("kv_producer", "kv_consumer")
    kv_cfg = f'{{"kv_connector":"{connector}","kv_role":"{role}"}}'
    args = [
        "--kv-transfer-config", kv_cfg,
        "--block-size", str(block_size),
        "--max-model-len", str(max_model_len),
        "--no-enable-prefix-caching",
    ]
    if kv_dtype and kv_dtype != "auto":
        args = ["--kv-cache-dtype", kv_dtype] + args
    return args


def launch_disagg_pair(model, prefill_port, decode_port, gpu_mem_util, max_model_len,
                       connector, block_size, log_dir, kv_dtype="auto",
                       prefill_gpu="0", decode_gpu="1",
                       side_channel_base=5600) -> tuple["VLLMServer", "VLLMServer"]:
    """Config C: prefill (producer) on one GPU, decode (consumer) on another.
    Same-vendor NVIDIA disaggregation; NIXL moves KV intra-node (NVLink/PCIe P2P)."""
    common = [f"--gpu-memory-utilization", str(gpu_mem_util)]
    p_args = common + disagg_engine_args("kv_producer", connector, block_size, max_model_len, kv_dtype)
    d_args = common + disagg_engine_args("kv_consumer", connector, block_size, max_model_len, kv_dtype)
    p_env = {"CUDA_VISIBLE_DEVICES": prefill_gpu,
             "VLLM_NIXL_SIDE_CHANNEL_PORT": str(side_channel_base)}
    d_env = {"CUDA_VISIBLE_DEVICES": decode_gpu,
             "VLLM_NIXL_SIDE_CHANNEL_PORT": str(side_channel_base + 1)}
    prefill = VLLMServer(model, prefill_port, p_args, env=p_env,
                         log_path=f"{log_dir}/C_prefill.log").start()
    decode = VLLMServer(model, decode_port, d_args, env=d_env,
                        log_path=f"{log_dir}/C_decode.log").start()
    return prefill, decode


class ProxyServer:
    """Runs pdbench.proxy as a subprocess in front of the prefill/decode pair."""

    def __init__(self, prefill_url, decode_url, port, log_path):
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}"
        self.log_path = log_path
        self.proc = None
        self._cmd = [sys.executable, "-m", "pdbench.proxy",
                     "--prefill", prefill_url, "--decode", decode_url,
                     "--port", str(port)]

    def start(self):
        logf = open(self.log_path, "w") if self.log_path else subprocess.DEVNULL
        self.proc = subprocess.Popen(self._cmd, stdout=logf,
                                     stderr=subprocess.STDOUT, env=os.environ.copy())
        return self

    def wait_ready(self, timeout=1200.0):
        return _wait_healthy(self.base_url, timeout)

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except Exception:
                self.proc.kill()


class DisaggBundle:
    """Config C/E bundle: prefill engine + decode engine + proxy, one interface.

    Exposes .base_url (the proxy), .wait_ready(), .stop(), .scrape_metrics() so
    run.py can treat it exactly like a single VLLMServer.
    """

    def __init__(self, prefill: VLLMServer, decode: VLLMServer, proxy: ProxyServer):
        self.prefill, self.decode, self.proxy = prefill, decode, proxy
        self.base_url = proxy.base_url

    def wait_ready(self, timeout=1800.0):
        if not self.prefill.wait_ready(timeout):
            print("  prefill engine not healthy"); return False
        if not self.decode.wait_ready(timeout):
            print("  decode engine not healthy"); return False
        self.proxy.start()
        return self.proxy.wait_ready(120)

    def stop(self):
        for s in (self.proxy, self.prefill, self.decode):
            try:
                s.stop()
            except Exception:
                pass

    def scrape_metrics(self) -> str:
        return ("# ---- prefill ----\n" + self.prefill.scrape_metrics()
                + "\n# ---- decode ----\n" + self.decode.scrape_metrics())
