"""Generate the graphs we compare across configs. All matplotlib, saved as PNG.

For a single config's result set we plot, per (ISL,OSL) workload:
  - TTFT p50/p95 vs concurrency          (prefill health)
  - ITL/TPOT p50/p95 vs concurrency      (decode health)
  - E2EL p50 vs concurrency              (headline latency)
  - output-TPS vs concurrency            (headline throughput)
  - goodput vs concurrency               (honest metric)
For cross-config comparison we overlay E2EL / output-TPS per config (crossover).
"""
from __future__ import annotations

import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7", "#e34948"]


def _by_workload(results: list[dict]):
    groups = defaultdict(list)
    for r in results:
        groups[(r["isl"], r["osl"])].append(r)
    for k in groups:
        groups[k].sort(key=lambda r: r["concurrency"])
    return groups


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, color="#e8e8e8")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot_config(results: list[dict], out_dir: str, config: str):
    os.makedirs(out_dir, exist_ok=True)
    groups = _by_workload(results)
    metrics = [
        ("ttft_s", "p95", "TTFT p95 (s)", "TTFT vs concurrency (prefill health)"),
        ("tpot_ms", "p95", "TPOT p95 (ms)", "Inter-token latency vs concurrency (decode health)"),
        ("e2el_s", "p50", "E2EL p50 (s)", "End-to-end latency vs concurrency"),
    ]
    made = []
    # latency panels
    for key, pct, ylabel, title in metrics:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for i, ((isl, osl), rows) in enumerate(sorted(groups.items())):
            xs = [r["concurrency"] for r in rows]
            ys = [r[key][pct] for r in rows]
            ax.plot(xs, ys, "-o", color=PALETTE[i % len(PALETTE)],
                    label=f"ISL{isl}/OSL{osl}", markersize=4)
        _style(ax, f"[{config}] {title}", "concurrency", ylabel)
        ax.legend(fontsize=8, frameon=False)
        p = os.path.join(out_dir, f"{config}_{key}.png")
        fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig); made.append(p)
    # throughput + goodput
    for key, ylabel, title in [
        ("output_tps", "output tokens/s", "Output throughput vs concurrency"),
        ("goodput_rps", "goodput (req/s meeting SLO)", "Goodput vs concurrency"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for i, ((isl, osl), rows) in enumerate(sorted(groups.items())):
            xs = [r["concurrency"] for r in rows]
            ys = [r[key] for r in rows]
            ax.plot(xs, ys, "-o", color=PALETTE[i % len(PALETTE)],
                    label=f"ISL{isl}/OSL{osl}", markersize=4)
        _style(ax, f"[{config}] {title}", "concurrency", ylabel)
        ax.legend(fontsize=8, frameon=False)
        p = os.path.join(out_dir, f"{config}_{key}.png")
        fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig); made.append(p)
    return made


def plot_crossover(config_results: dict[str, list[dict]], out_dir: str,
                   isl: int, osl: int, metric: str = "e2el_s", pct: str = "p50"):
    """Overlay one metric across configs for a fixed workload — the crossover chart."""
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, (cfg, results) in enumerate(sorted(config_results.items())):
        rows = sorted([r for r in results if r["isl"] == isl and r["osl"] == osl],
                      key=lambda r: r["concurrency"])
        if not rows:
            continue
        xs = [r["concurrency"] for r in rows]
        ys = [(r[metric][pct] if isinstance(r[metric], dict) else r[metric]) for r in rows]
        ax.plot(xs, ys, "-o", color=PALETTE[i % len(PALETTE)], label=cfg, markersize=4)
    ylabel = f"{metric} {pct}" if metric.endswith(("_s", "_ms")) else metric
    _style(ax, f"Crossover — ISL{isl}/OSL{osl}", "concurrency", ylabel)
    ax.legend(fontsize=8, frameon=False)
    p = os.path.join(out_dir, f"crossover_isl{isl}_osl{osl}_{metric}.png")
    fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)
    return p
