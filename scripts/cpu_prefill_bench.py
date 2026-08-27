#!/usr/bin/env python
"""Config E, Spike 1 — measure CPU prefill throughput for a dense model.

Prefill == one forward pass over ISL input tokens (produces the KV cache + first
logits). We time exactly that on the CPU with a bf16 transformers model, using all
cores, at a sweep of ISL. This is the core H2 number: how fast can this CPU prefill
an 8B model, and where does the ISL ceiling make it untenable.

Output feeds the *emulated* Config E:
    E_TTFT(ISL)  ~= CPU prefill time here
    E_ITL        ~= GPU decode ITL from the GPU-aggregated run
    E_E2EL       ~= E_TTFT + OSL * E_ITL

  python scripts/cpu_prefill_bench.py --model Qwen/Qwen3-8B \
      --isl 128 256 512 1024 2048 4096 --repeats 3
"""
from __future__ import annotations

import argparse
import json
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--isl", type=int, nargs="*", default=[128, 256, 512, 1024, 2048, 4096])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--threads", type=int, default=0, help="0 = all cores")
    ap.add_argument("--out", default="results/E_cpu_prefill.json")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoConfig

    threads = args.threads or os.cpu_count()
    torch.set_num_threads(threads)
    print(f"[cpu_prefill] model={args.model} threads={threads} "
          f"dtype=bfloat16 isl={args.isl}", flush=True)

    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    vocab = int(getattr(cfg, "vocab_size", 100000))
    n_params = None
    t0 = time.time()
    # low_cpu_mem_usage=False forces weights fully into RAM at load time, so the
    # forward pass is NOT bottlenecked by faulting mmap'd weights off a slow
    # (network) filesystem. We then .contiguous() every parameter to be certain.
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
        low_cpu_mem_usage=False).eval()
    # .clone() forces a real copy into fresh RAM (safetensors mmaps the file, and
    # .contiguous() would return the same mmap view — so every forward would refault
    # 16 GB off the filesystem). This materializes the weights once, up front.
    for p in model.parameters():
        p.data = p.data.clone()
    try:
        n_params = sum(p.numel() for p in model.parameters())
    except Exception:
        pass
    print(f"[cpu_prefill] loaded (into RAM) in {time.time() - t0:.0f}s "
          f"({(n_params or 0) / 1e9:.1f}B params)", flush=True)

    results = []
    with torch.inference_mode():
        for isl in args.isl:
            ids = torch.randint(1000, vocab - 100, (1, isl), dtype=torch.long)
            # warmup
            _ = model(ids, use_cache=True)
            times = []
            for _ in range(args.repeats):
                t = time.perf_counter()
                _ = model(ids, use_cache=True)
                times.append(time.perf_counter() - t)
            med = sorted(times)[len(times) // 2]
            tps = isl / med
            results.append({"isl": isl, "prefill_s_median": round(med, 4),
                            "prefill_tokens_per_s": round(tps, 1),
                            "all_times_s": [round(x, 4) for x in times]})
            print(f"[cpu_prefill] ISL {isl:>5}: {med:.3f}s  -> {tps:>7.1f} tok/s", flush=True)

    payload = {
        "model": args.model, "device": "cpu", "threads": threads,
        "n_params_b": round((n_params or 0) / 1e9, 2),
        "cpu_model": _cpu_model(), "results": results,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[cpu_prefill] saved -> {args.out}")


def _cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return ""


if __name__ == "__main__":
    main()
