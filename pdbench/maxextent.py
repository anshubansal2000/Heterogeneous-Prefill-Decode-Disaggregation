"""Probe the maximum extent of prefill (ISL), decode (OSL), and concurrency
before the server OOMs or a latency ceiling is exceeded.

We grow each dimension geometrically until a single request fails (OOM / server
error) or a latency threshold is crossed, then binary-search the boundary.
"""
from __future__ import annotations

import asyncio

from .client import probe_once, run_closed_loop
from .workload import build_prompt_token_ids


async def _ok_isl(base_url, model, vocab, isl, osl, ttft_ceiling_s) -> bool:
    ids = build_prompt_token_ids(isl, vocab)
    r = await probe_once(base_url, model, ids, osl)
    return r.ok and (ttft_ceiling_s <= 0 or r.ttft <= ttft_ceiling_s)


async def _ok_osl(base_url, model, vocab, isl, osl) -> bool:
    ids = build_prompt_token_ids(isl, vocab)
    r = await probe_once(base_url, model, ids, osl)
    return r.ok


async def _boundary(pred, lo: int, hi_start: int, cap: int):
    """Return the largest value <= cap for which `pred(value)` is True.
    Geometric growth from hi_start until failure, then binary search."""
    hi = hi_start
    last_ok = 0
    if await pred(lo):
        last_ok = lo
    else:
        return 0
    # grow
    v = hi
    while v <= cap and await pred(v):
        last_ok = v
        v *= 2
    fail = min(v, cap + 1)
    lo, hi = last_ok, min(v, cap)
    # binary search between last_ok and first fail
    while hi - lo > max(1, last_ok // 20):
        mid = (lo + hi) // 2
        if await pred(mid):
            lo = mid
        else:
            hi = mid
    return lo


async def max_isl(base_url, model, vocab, osl=8, cap=131072, ttft_ceiling_s=0.0):
    async def pred(isl):
        return await _ok_isl(base_url, model, vocab, isl, osl, ttft_ceiling_s)
    return await _boundary(pred, 512, 2048, cap)


async def max_osl(base_url, model, vocab, isl=128, cap=131072):
    async def pred(osl):
        return await _ok_osl(base_url, model, vocab, isl, osl)
    return await _boundary(pred, 128, 1024, cap)


async def max_concurrency(base_url, model, vocab, isl=1024, osl=128, cap=512,
                          fail_error_frac=0.1):
    """Largest concurrency where <fail_error_frac of requests error."""
    ids = build_prompt_token_ids(isl, vocab)

    async def pred(c):
        results, _ = await run_closed_loop(base_url, model, ids, osl, c, c)
        errs = sum(1 for r in results if not r.ok)
        return errs / max(1, len(results)) <= fail_error_frac

    return await _boundary(pred, 1, 8, cap)


async def probe_all(base_url, model, vocab, ttft_ceiling_s=0.0):
    mi = await max_isl(base_url, model, vocab, ttft_ceiling_s=ttft_ceiling_s)
    mo = await max_osl(base_url, model, vocab)
    mc = await max_concurrency(base_url, model, vocab)
    return {"max_isl": mi, "max_osl": mo, "max_concurrency": mc,
            "ttft_ceiling_s": ttft_ceiling_s}
