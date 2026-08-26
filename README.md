# Heterogeneous Prefill/Decode Disaggregation — benchmark harness

Self-contained harness to benchmark **prefill/decode (PD) disaggregation** serving
against aggregated serving, on open-source components only (vLLM + an OpenAI-style
load generator). Built for the project plan in
[`pd-disaggregation-project-plan.md.pdf`](pd-disaggregation-project-plan.md.pdf).

We measure the metrics that matter for PD serving and graph them so configs can be
compared directly.

## Configs

| ID | Name | Prefill | Decode | Status |
|---|---|---|---|---|
| **A** | Aggregated (baseline) | GPU | same GPU | ✅ implemented |
| **E** | CPU prefill → GPU decode | CPU (vLLM CPU backend) | GPU | 🧩 scaffolded (enable after Phase-0 spike) |
| **C** | Same-vendor disagg | NVIDIA | NVIDIA | 🧩 scaffolded |
| B / D | AMD / cross-vendor | — | — | ⛔ skipped (no AMD on RunPod) |

## Models (both MoE)

| key | HF id | total / active | size (MXFP4) | GPU |
|---|---|---|---|---|
| `gpt-oss-20b` | `openai/gpt-oss-20b` | ~21B / ~3.6B | ~16 GB | ≥24 GB |
| `gpt-oss-120b` | `openai/gpt-oss-120b` | ~117B / ~5.1B | ~63 GB | 80 GB (H100) |

## Metrics (per ISL/OSL × concurrency cell, p50/p95/p99)

- **TTFT** — time to first token (prefill health)
- **ITL / TPOT** — inter-token latency (decode health)
- **E2EL** — end-to-end latency (headline)
- **output-TPS / total-TPS** — throughput
- **goodput** — req/s meeting an SLO (TTFT ≤ 2 s ∧ ITL ≤ 50 ms; both configurable)
- **max-extent** — largest prefill ISL, decode OSL, and concurrency before OOM

All results → `results/<config>_<model>.json` (+ a reproducible run manifest:
git SHA, vLLM/torch versions, GPU/CPU info, engine args). Graphs → `results/plots/`.

## Run

**On a RunPod GPU pod** (see `scripts/setup_runpod.sh`):

```bash
bash scripts/setup_runpod.sh gpt-oss-20b        # installs vLLM + downloads weights
python run.py --config A --model gpt-oss-20b --quick     # ~2-min smoke
python run.py --config A --model gpt-oss-20b             # full matrix + max-extent
python run.py --config A --model gpt-oss-120b            # headline run (80 GB GPU)
```

**Locally** you can validate the client/metrics/plots with the mock server in
`scripts/` (no GPU needed) — this is how the data path was tested for $0.

Re-plot from saved JSON without re-running:
```bash
python run.py --plots-only --results results/A_gpt-oss-20b.json
```

## Workload matrix

ISL/OSL ∈ {1K/1K, 1K/8K, 8K/1K, 8K/8K} (Moreh's matrix), concurrency ∈ {1,4,8,16,32}.
Config E adds short-ISL cells (128/256/512) where CPU prefill lives or dies.
OSL is pinned exactly with `max_tokens` + `ignore_eos`; ISL is pinned by sending the
prompt as a token-id list. Prefix caching is disabled to isolate raw compute.

## Layout

```
run.py                 CLI orchestrator (launch server → sweep → save → plot)
pdbench/
  workload.py          exact-length prompts + the ISL/OSL × concurrency matrix
  client.py            async streaming load generator (per-request TTFT/ITL/E2EL)
  metrics.py           aggregation → p50/p95/p99, throughput, goodput
  server.py            vLLM server launch/manage (A implemented; E/C scaffolded)
  maxextent.py         probe max ISL / OSL / concurrency before OOM
  plots.py             matplotlib graphs + cross-config crossover charts
  manifest.py          reproducible run manifest
scripts/setup_runpod.sh   provision a RunPod pod
diagrams/                 topology + memory-hierarchy figures (for the report)
```

## Cost discipline (RunPod)

Develop locally for free; rent one pod, run a tight matrix, stop it immediately.
`gpt-oss-20b` runs on a ~$0.8–1.2/hr GPU; `gpt-oss-120b` needs an H100 80 GB
(~$2.5/hr). Target: stay under a fixed budget.
