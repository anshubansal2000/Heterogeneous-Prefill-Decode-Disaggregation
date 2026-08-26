"""Aggregate per-request results into the metrics we report and graph.

Primary metrics (Moreh-style):
  TTFT, TPOT/ITL, E2EL           -> p50/p95/p99, prefill vs decode health
  output_tps, total_tps          -> throughput (generated / generated+prompt)
  goodput                        -> req/s meeting an SLO (TTFT & ITL thresholds)
  request_throughput             -> completed req/s
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


@dataclass
class SLO:
    ttft_s: float = 2.0      # first token within 2 s
    itl_ms: float = 50.0     # inter-token latency under 50 ms


def summarize(results, wall_s: float, isl: int, osl: int, concurrency: int,
              slo: SLO | None = None) -> dict:
    slo = slo or SLO()
    ok = [r for r in results if r.ok]
    n_ok, n_err = len(ok), len(results) - len(ok)

    ttfts = [r.ttft for r in ok]
    tpots = [r.tpot for r in ok if r.itls]
    e2els = [r.e2el for r in ok]
    all_itls = [x for r in ok for x in r.itls]

    out_tokens = sum(r.output_tokens for r in ok)
    prompt_tokens = sum(r.prompt_tokens for r in ok)

    def stat(xs, scale=1.0):
        if not xs:
            return {"mean": 0, "p50": 0, "p95": 0, "p99": 0}
        return {
            "mean": statistics.fmean(xs) * scale,
            "p50": _pct(xs, 0.50) * scale,
            "p95": _pct(xs, 0.95) * scale,
            "p99": _pct(xs, 0.99) * scale,
        }

    # goodput: requests that met BOTH the TTFT and ITL SLOs
    good = sum(1 for r in ok
               if r.ttft <= slo.ttft_s and (r.tpot * 1000.0) <= slo.itl_ms)

    return {
        "isl": isl, "osl": osl, "concurrency": concurrency,
        "n_ok": n_ok, "n_err": n_err, "wall_s": round(wall_s, 4),
        "ttft_s": stat(ttfts),
        "tpot_ms": stat(tpots, 1000.0),
        "itl_ms": stat(all_itls, 1000.0),
        "e2el_s": stat(e2els),
        "output_tps": round(out_tokens / wall_s, 2) if wall_s else 0,
        "total_tps": round((out_tokens + prompt_tokens) / wall_s, 2) if wall_s else 0,
        "request_throughput_rps": round(n_ok / wall_s, 4) if wall_s else 0,
        "goodput_rps": round(good / wall_s, 4) if wall_s else 0,
        "slo": {"ttft_s": slo.ttft_s, "itl_ms": slo.itl_ms},
        "errors_sample": [r.error for r in results if not r.ok][:3],
    }
