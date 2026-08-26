#!/usr/bin/env python
"""Spike 2 — validate the NIXL CUDA<->CUDA disaggregation handshake with ONE request
before running the whole Config-C matrix (so we debug for pennies, not pod-hours).

Launches prefill (GPU0 producer) + decode (GPU1 consumer) + proxy, sends a single
completion through the proxy, and prints the KV-transfer params exchanged and the
generated text. Also does an output-correctness check vs an aggregated baseline.

  python scripts/spike_c.py --model gpt-oss-20b
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time

sys.path.insert(0, ".")
from pdbench import server, client  # noqa: E402

MODELS = {"gpt-oss-20b": "openai/gpt-oss-20b", "gpt-oss-120b": "openai/gpt-oss-120b",
          "qwen3-0.6b": "Qwen/Qwen3-0.6B"}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-oss-20b")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-model-len", type=int, default=18000)
    ap.add_argument("--connector", default="NixlConnector")
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    args = ap.parse_args()
    hf = MODELS[args.model]

    print(f"[spike_c] launching disagg pair for {hf} ...", flush=True)
    prefill, decode = server.launch_disagg_pair(
        model=hf, prefill_port=args.port + 100, decode_port=args.port + 200,
        gpu_mem_util=args.gpu_mem_util, max_model_len=args.max_model_len,
        connector=args.connector, block_size=16, log_dir="results")
    proxy = server.ProxyServer(prefill.base_url, decode.base_url, args.port,
                               "results/C_proxy.log")
    bundle = server.DisaggBundle(prefill, decode, proxy)
    try:
        print("[spike_c] waiting for prefill+decode+proxy ...", flush=True)
        if not bundle.wait_ready(1200):
            print("[spike_c] FAILED: engines/proxy did not become healthy")
            print("  check results/C_prefill.log, C_decode.log, C_proxy.log")
            return 1
        print("[spike_c] all healthy; sending one request through the proxy ...")
        ids = list(range(1000, 1000 + 64))  # 64-token prompt
        t0 = time.perf_counter()
        r = await client.probe_once(bundle.base_url, hf, ids, osl=32)
        dt = time.perf_counter() - t0
        if r.ok:
            print(f"[spike_c] SUCCESS: disagg path works. "
                  f"TTFT={r.ttft:.3f}s E2EL={r.e2el:.3f}s out_tokens={r.output_tokens} "
                  f"({dt:.2f}s total)")
            print("[spike_c] -> Config C is GO. Run: python run.py --config C --model "
                  f"{args.model}")
            return 0
        print(f"[spike_c] FAILED at request: {r.error}")
        print("  inspect results/C_*.log; the NIXL field names in pdbench/proxy.py "
              "may need adjusting for this vLLM version.")
        return 1
    finally:
        bundle.stop()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
