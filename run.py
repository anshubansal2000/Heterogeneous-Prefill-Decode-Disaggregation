#!/usr/bin/env python
"""
pdbench — Prefill/Decode disaggregation benchmark harness
=========================================================
Self-contained benchmark for heterogeneous PD-disaggregation serving, built for
the project plan in this repo. Measures TTFT / ITL / E2EL / throughput / goodput
across an ISL/OSL x concurrency matrix, probes maximum extent (max prefill ISL,
decode OSL, concurrency before OOM), saves everything to JSON, and plots graphs.

Configs
  A   Aggregated single GPU (baseline)          [implemented]
  E   CPU prefill -> GPU decode                  [scaffolded; enable post-spike]
  C   Same-vendor NVIDIA disaggregation (2 GPU)  [scaffolded; enable post-spike]
  B/D AMD / cross-vendor                         [skipped — no AMD on RunPod]

Model presets (both MoE):
  gpt-oss-20b   ~21B total / ~3.6B active, ~16 GB MXFP4  (fits ~24 GB GPU)
  gpt-oss-120b  ~117B total / ~5.1B active, ~63 GB MXFP4 (needs 80 GB GPU)

Usage
  python run.py --config A --model gpt-oss-20b --quick
  python run.py --config A --model gpt-oss-120b
  python run.py --config A --model gpt-oss-20b --stage maxextent
  python run.py --plots-only --results results/A_gpt-oss-20b.json

Everything is written to results/ .
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

from pdbench import client, maxextent, metrics, plots, server, workload

# ---- model presets ----------------------------------------------------------
MODELS = {
    "gpt-oss-20b": {
        "hf": "openai/gpt-oss-20b",
        "gpu_mem_util": 0.90,
        "min_gpu_gb": 24,
    },
    "gpt-oss-120b": {
        "hf": "openai/gpt-oss-120b",
        "gpu_mem_util": 0.92,
        "min_gpu_gb": 80,
    },
    # a tiny open model for local smoke tests of the harness itself (no gpt-oss)
    "qwen3-0.6b": {
        "hf": "Qwen/Qwen3-0.6B",
        "gpu_mem_util": 0.85,
        "min_gpu_gb": 4,
    },
}


def get_vocab_size(hf_model: str) -> int:
    """Vocab size for building token-id prompts. Falls back to a safe default."""
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(hf_model, trust_remote_code=True)
        return int(getattr(cfg, "vocab_size", 0) or 100000)
    except Exception as e:  # noqa: BLE001
        print(f"  (vocab lookup failed: {e}; using 100000)")
        return 100000


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


async def bench_matrix(base_url, hf_model, vocab, cells, reps, slo, warmup=2):
    results = []
    for i, cell in enumerate(cells):
        ids = workload.build_prompt_token_ids(cell.isl, vocab, seed=cell.isl)
        total = max(cell.concurrency * reps, cell.concurrency)
        # warmup (discarded)
        if warmup:
            await client.run_closed_loop(base_url, hf_model, ids, min(cell.osl, 16),
                                         cell.concurrency, min(cell.concurrency, warmup))
        res, wall = await client.run_closed_loop(base_url, hf_model, ids, cell.osl,
                                                 cell.concurrency, total)
        summ = metrics.summarize(res, wall, cell.isl, cell.osl, cell.concurrency, slo)
        results.append(summ)
        log(f"  [{i+1}/{len(cells)}] {cell.name}: "
            f"TTFT_p95={summ['ttft_s']['p95']:.3f}s "
            f"ITL_p95={summ['itl_ms']['p95']:.1f}ms "
            f"out_tps={summ['output_tps']:.0f} "
            f"good={summ['goodput_rps']:.2f}rps "
            f"err={summ['n_err']}")
    return results


def build_server(config, preset, args, log_path):
    if config == "A":
        return server.launch_aggregated(
            model=preset["hf"], port=args.port,
            gpu_mem_util=preset["gpu_mem_util"], kv_dtype=args.kv_dtype,
            max_model_len=args.max_model_len, extra=args.engine_args or [],
            log_path=log_path)
    raise SystemExit(f"config {config} not yet enabled in this build "
                     f"(A is implemented; E/C are scaffolded in pdbench/server.py)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="A", choices=["A", "C", "E"])
    ap.add_argument("--model", default="gpt-oss-20b", choices=list(MODELS))
    ap.add_argument("--stage", default="all",
                    choices=["all", "matrix", "maxextent", "serve-only"])
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--kv-dtype", default="auto")
    ap.add_argument("--max-model-len", type=int, default=18000)
    ap.add_argument("--reps", type=int, default=3, help="requests-per-worker per cell")
    ap.add_argument("--concurrency", type=int, nargs="*", default=None)
    ap.add_argument("--isl-osl", type=str, default=None,
                    help="comma list like 1024x1024,8192x1024")
    ap.add_argument("--slo-ttft", type=float, default=2.0)
    ap.add_argument("--slo-itl-ms", type=float, default=50.0)
    ap.add_argument("--engine-args", type=str, nargs=argparse.REMAINDER, default=[],
                    help="extra args passed verbatim to `vllm serve` (put LAST)")
    ap.add_argument("--quick", action="store_true", help="tiny smoke matrix")
    ap.add_argument("--no-server", action="store_true",
                    help="assume a server is already running at --port")
    ap.add_argument("--plots-only", action="store_true")
    ap.add_argument("--results", type=str, default=None, help="for --plots-only")
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    tag = f"{args.config}_{args.model}"
    results_path = args.results or f"results/{tag}.json"
    plots_dir = f"results/plots"

    # -------- plots-only -----------------------------------------------------
    if args.plots_only:
        payload = json.load(open(results_path, encoding="utf-8"))
        made = plots.plot_config(payload["results"], plots_dir, payload["manifest"]["config"])
        log(f"wrote {len(made)} plots to {plots_dir}")
        return

    preset = MODELS[args.model]
    hf_model = preset["hf"]

    # workload
    isl_osl = None
    if args.isl_osl:
        isl_osl = [tuple(int(x) for x in p.split("x")) for p in args.isl_osl.split(",")]
    conc = args.concurrency
    if args.quick:
        isl_osl = isl_osl or [(256, 128), (1024, 256)]
        conc = conc or [1, 4]
        args.reps = min(args.reps, 2)
    cells = workload.build_matrix(isl_osl, conc)
    slo = metrics.SLO(args.slo_ttft, args.slo_itl_ms)

    log(f"config={args.config} model={hf_model} cells={len(cells)} results={results_path}")
    vocab = get_vocab_size(hf_model)
    log(f"vocab_size={vocab}")

    base_url = f"http://127.0.0.1:{args.port}"
    srv = None
    log_path = f"results/{tag}.server.log"
    if not args.no_server:
        srv = build_server(args.config, preset, args, log_path)
        log(f"waiting for server to become healthy (log: {log_path}) ...")
        if not srv.wait_ready():
            srv.stop()
            sys.exit("server failed to start — see log above")
        log("server healthy.")

    if args.stage == "serve-only":
        log(f"serve-only: server at {base_url} — Ctrl+C to stop")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass
        if srv:
            srv.stop()
        return

    from pdbench import manifest as _manifest
    man = _manifest.build(args.config, hf_model, vars(args),
                          time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    all_results = []
    extent = {}
    try:
        if args.stage in ("all", "matrix"):
            log("=== matrix sweep ===")
            all_results = asyncio.run(bench_matrix(base_url, hf_model, vocab, cells,
                                                   args.reps, slo))
        if args.stage in ("all", "maxextent"):
            log("=== max-extent probe (this can take a while) ===")
            extent = asyncio.run(maxextent.probe_all(base_url, hf_model, vocab))
            log(f"  max_isl={extent['max_isl']} max_osl={extent['max_osl']} "
                f"max_concurrency={extent['max_concurrency']}")
    finally:
        scrapes = {"server": srv.scrape_metrics()} if srv else {}
        man["max_extent"] = extent
        _manifest.save(results_path, man, all_results, scrapes)
        log(f"saved results -> {results_path}")
        if srv:
            srv.stop()
            log("server stopped.")

    if all_results:
        made = plots.plot_config(all_results, plots_dir, args.config)
        log(f"wrote {len(made)} plots -> {plots_dir}")


if __name__ == "__main__":
    main()
