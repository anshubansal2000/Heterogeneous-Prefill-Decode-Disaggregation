"""Async load generator that measures per-request TTFT / ITL / E2EL against an
OpenAI-compatible vLLM server, under closed-loop concurrency control.

TTFT  = time from send to the first generated token.
ITL   = inter-token latencies (gaps between successive tokens); TPOT = mean ITL.
E2EL  = time from send to the last token.

We stream /v1/completions so we can timestamp every token as it arrives.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field

import httpx


@dataclass
class RequestResult:
    ok: bool
    ttft: float = 0.0                 # seconds
    e2el: float = 0.0                 # seconds
    itls: list[float] = field(default_factory=list)
    output_tokens: int = 0
    prompt_tokens: int = 0
    error: str = ""

    @property
    def tpot(self) -> float:
        return sum(self.itls) / len(self.itls) if self.itls else 0.0


async def _one_request(client: httpx.AsyncClient, url: str, model: str,
                       prompt_token_ids: list[int], osl: int,
                       temperature: float, timeout: float) -> RequestResult:
    payload = {
        "model": model,
        "prompt": prompt_token_ids,          # vLLM accepts token-id lists
        "max_tokens": osl,
        "min_tokens": osl,                   # force exact OSL (vLLM extension)
        "ignore_eos": True,                  # force exact OSL
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    ttft = 0.0
    last_tok_t = None
    itls: list[float] = []
    out_toks = 0
    prompt_toks = len(prompt_token_ids)
    try:
        async with client.stream("POST", url, json=payload, timeout=timeout) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode("utf-8", "replace")[:200]
                return RequestResult(False, error=f"HTTP {resp.status_code}: {body}")
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                # usage-only final chunk
                choices = obj.get("choices") or []
                text = choices[0].get("text", "") if choices else ""
                if obj.get("usage"):
                    u = obj["usage"]
                    out_toks = u.get("completion_tokens", out_toks)
                    prompt_toks = u.get("prompt_tokens", prompt_toks)
                if text:
                    now = time.perf_counter()
                    if ttft == 0.0:
                        ttft = now - t0
                    else:
                        itls.append(now - last_tok_t)
                    last_tok_t = now
        e2el = time.perf_counter() - t0
        if ttft == 0.0:
            return RequestResult(False, error="no tokens streamed")
        # Fall back to counting streamed tokens if usage was absent.
        if out_toks == 0:
            out_toks = len(itls) + 1
        return RequestResult(True, ttft=ttft, e2el=e2el, itls=itls,
                             output_tokens=out_toks, prompt_tokens=prompt_toks)
    except Exception as e:  # noqa: BLE001
        return RequestResult(False, error=f"{type(e).__name__}: {e}")


async def run_closed_loop(base_url: str, model: str, prompt_token_ids: list[int],
                          osl: int, concurrency: int, total_requests: int,
                          temperature: float = 0.0, timeout: float = 1800.0):
    """`concurrency` workers each loop send→await→send until `total_requests` done.

    Returns (results, wall_seconds).
    """
    url = base_url.rstrip("/") + "/v1/completions"
    results: list[RequestResult] = []
    sent = 0
    lock = asyncio.Lock()
    limits = httpx.Limits(max_connections=concurrency + 4,
                          max_keepalive_connections=concurrency + 4)

    async with httpx.AsyncClient(limits=limits) as client:
        async def worker():
            nonlocal sent
            while True:
                async with lock:
                    if sent >= total_requests:
                        return
                    sent += 1
                r = await _one_request(client, url, model, prompt_token_ids,
                                       osl, temperature, timeout)
                results.append(r)

        t0 = time.perf_counter()
        await asyncio.gather(*[worker() for _ in range(concurrency)])
        wall = time.perf_counter() - t0
    return results, wall


async def probe_once(base_url: str, model: str, prompt_token_ids: list[int],
                     osl: int, timeout: float = 1800.0) -> RequestResult:
    """A single request — used by the max-extent prober."""
    url = base_url.rstrip("/") + "/v1/completions"
    async with httpx.AsyncClient() as client:
        return await _one_request(client, url, model, prompt_token_ids, osl, 0.0, timeout)
