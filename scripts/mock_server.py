# A tiny fake OpenAI /v1/completions streaming server to smoke-test the pdbench
# client/metrics/plots WITHOUT a GPU or vLLM. Simulates TTFT (prefill ~ ISL) and
# per-token ITL (decode), so we can validate the whole harness locally for $0.
import asyncio, json, time, sys
from aiohttp import web

async def completions(request):
    body = await request.json()
    prompt = body.get("prompt", [])
    isl = len(prompt) if isinstance(prompt, list) else 100
    osl = int(body.get("max_tokens", 16))
    # simulate: prefill time grows with ISL, decode ~ fixed per token, mild load penalty
    prefill_s = 0.0002 * isl
    itl_s = 0.004
    resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    await asyncio.sleep(prefill_s)
    for i in range(osl):
        chunk = {"choices": [{"text": "x", "index": 0}]}
        await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())
        await asyncio.sleep(itl_s)
    usage = {"choices": [], "usage": {"prompt_tokens": isl, "completion_tokens": osl}}
    await resp.write(f"data: {json.dumps(usage)}\n\n".encode())
    await resp.write(b"data: [DONE]\n\n")
    await resp.write_eof()
    return resp

async def health(request):
    return web.Response(text="ok")

app = web.Application()
app.router.add_post("/v1/completions", completions)
app.router.add_get("/health", health)
web.run_app(app, host="127.0.0.1", port=8111, print=None)
