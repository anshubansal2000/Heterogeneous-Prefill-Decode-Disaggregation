"""OpenAI-compatible disaggregated-prefill proxy for vLLM's NIXL connector.

Config C / D flow, per request:
  1) POST to the PREFILL engine with max_tokens=1 and
       kv_transfer_params = {"do_remote_decode": true}
     The prefill engine computes the KV cache, registers it with NIXL, and returns
     kv_transfer_params describing where the decode engine can pull it from
     (remote_engine_id / remote_block_ids / remote_host / remote_port).
  2) POST to the DECODE engine with the full request and
       kv_transfer_params = {"do_remote_prefill": true, ...the values from step 1}
     The decode engine pulls the KV over NIXL and streams the generation.

This mirrors vLLM's shipped disaggregated_serving example. Field names follow the
vLLM v1 NixlConnector contract; if a vLLM version differs, only _prefill_params /
_decode_params below need adjusting (validated by scripts/spike_c.py before a run).

Run:
  python -m pdbench.proxy --prefill http://127.0.0.1:8100 \
                          --decode  http://127.0.0.1:8200 --port 8000
"""
from __future__ import annotations

import argparse
import json

import httpx
from aiohttp import web

PREFILL_URL = ""
DECODE_URL = ""
_client: httpx.AsyncClient | None = None


def _prefill_params() -> dict:
    return {"do_remote_decode": True}


def _decode_params(from_prefill: dict) -> dict:
    p = {"do_remote_prefill": True}
    # carry through whatever the prefill side handed back
    for k in ("remote_engine_id", "remote_block_ids", "remote_host",
              "remote_port", "remote_kv_url"):
        if k in from_prefill:
            p[k] = from_prefill[k]
    return p


async def handle_completions(request: web.Request) -> web.StreamResponse:
    body = await request.json()

    # ---- 1) prefill (produce KV) ----
    pf = dict(body)
    pf["max_tokens"] = 1
    pf["min_tokens"] = 1
    pf["stream"] = False
    pf["kv_transfer_params"] = _prefill_params()
    r = await _client.post(PREFILL_URL + "/v1/completions", json=pf, timeout=600)
    if r.status_code != 200:
        return web.json_response({"error": f"prefill {r.status_code}: {r.text[:200]}"},
                                 status=502)
    pf_out = r.json()
    ktp = pf_out.get("kv_transfer_params") or {}

    # ---- 2) decode (consume KV), streamed ----
    dec = dict(body)
    dec["stream"] = True
    dec.setdefault("stream_options", {"include_usage": True})
    dec["kv_transfer_params"] = _decode_params(ktp)

    resp = web.StreamResponse(status=200,
                              headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    async with _client.stream("POST", DECODE_URL + "/v1/completions",
                              json=dec, timeout=3600) as up:
        if up.status_code != 200:
            body_txt = (await up.aread()).decode("utf-8", "replace")[:200]
            await resp.write(f"data: {json.dumps({'error': body_txt})}\n\n".encode())
            await resp.write_eof()
            return resp
        async for chunk in up.aiter_raw():
            if chunk:
                await resp.write(chunk)
    await resp.write_eof()
    return resp


async def handle_health(request: web.Request) -> web.Response:
    # healthy only when both upstreams are healthy
    try:
        for u in (PREFILL_URL, DECODE_URL):
            hr = await _client.get(u + "/health", timeout=5)
            if hr.status_code != 200:
                return web.Response(status=503, text="upstream not ready")
    except Exception as e:  # noqa: BLE001
        return web.Response(status=503, text=str(e))
    return web.Response(text="ok")


async def _on_start(app):
    global _client
    _client = httpx.AsyncClient()


async def _on_stop(app):
    if _client:
        await _client.aclose()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefill", required=True)
    ap.add_argument("--decode", required=True)
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    global PREFILL_URL, DECODE_URL
    PREFILL_URL = args.prefill.rstrip("/")
    DECODE_URL = args.decode.rstrip("/")

    app = web.Application()
    app.router.add_post("/v1/completions", handle_completions)
    app.router.add_get("/health", handle_health)
    app.on_startup.append(_on_start)
    app.on_cleanup.append(_on_stop)
    web.run_app(app, host="127.0.0.1", port=args.port, print=None)


if __name__ == "__main__":
    main()
