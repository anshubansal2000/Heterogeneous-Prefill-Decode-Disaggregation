"""Run manifest: capture everything needed to reproduce a result set."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import time


def _sh(cmd: str) -> str:
    try:
        return subprocess.check_output(cmd, shell=True, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def gpu_info() -> list[dict]:
    out = _sh("nvidia-smi --query-gpu=name,memory.total,driver_version "
              "--format=csv,noheader")
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            gpus.append({"name": parts[0], "memory": parts[1],
                         "driver": parts[2] if len(parts) > 2 else ""})
    return gpus


def cpu_info() -> dict:
    return {
        "processor": platform.processor(),
        "cores_logical": os.cpu_count(),
        "flags_avx512": "avx512" in _sh("cat /proc/cpuinfo 2>/dev/null").lower(),
        "flags_amx": "amx" in _sh("cat /proc/cpuinfo 2>/dev/null").lower(),
        "mem_total": _sh("grep MemTotal /proc/meminfo 2>/dev/null"),
    }


def build(config: str, model: str, args: dict, timestamp: str) -> dict:
    return {
        "config": config,
        "model": model,
        "timestamp_utc": timestamp,
        "args": args,
        "vllm_version": _sh("vllm --version") or _sh("python -c \"import vllm;print(vllm.__version__)\""),
        "torch_version": _sh("python -c \"import torch;print(torch.__version__)\""),
        "git_sha": _sh("git rev-parse --short HEAD"),
        "platform": platform.platform(),
        "gpu": gpu_info(),
        "cpu": cpu_info(),
    }


def save(path: str, manifest: dict, results: list[dict], metrics_scrapes: dict | None = None):
    payload = {"manifest": manifest, "results": results,
               "metrics_scrapes": metrics_scrapes or {}}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
