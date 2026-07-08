from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from functools import cache
from typing import Any

import httpx
import structlog
import tiktoken
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from penny_pincher import lms
from penny_pincher.settings import Settings

log = structlog.get_logger(__name__)

LOCAL_MODEL_ALIAS = "local"
_HOP_BY_HOP = frozenset(
    {"host", "content-length", "transfer-encoding", "connection", "keep-alive", "te", "trailers", "upgrade"}
)
# Rough approximations for non-text content blocks — cl100k_base can't tokenize
# images/tool payloads faithfully, so use fixed costs in the ballpark of what
# Anthropic reports. Good enough for context budgeting.
_IMAGE_BLOCK_TOKENS = 1500
_TOOL_USE_BLOCK_TOKENS = 200


@cache
def _encoding() -> tiktoken.Encoding:
    # Lazy so `import penny_pincher` never requires network — tiktoken fetches
    # the encoding file from a CDN on first call and caches it locally.
    return tiktoken.get_encoding("cl100k_base")


def create_app(settings: Settings, http_client: httpx.AsyncClient | None = None) -> FastAPI:
    """Build the penny-pincher FastAPI app.

    The app manages its own `httpx.AsyncClient` via lifespan by default —
    Starlette ≥0.32 propagates the lifespan of mounted sub-apps, so
    `parent.mount("/proxy", create_app(settings))` works without extra wiring.

    `http_client` is an escape hatch: pass one to share a client with the
    parent, or to inject a mock transport in tests.
    """
    if http_client is None:

        @asynccontextmanager
        async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
            async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
                app.state.http = client
                yield

        app = FastAPI(title="penny-pincher", lifespan=lifespan)
    else:
        app = FastAPI(title="penny-pincher")
        app.state.http = http_client

    @app.post("/v1/messages")
    async def messages(request: Request) -> Any:
        body: dict[str, Any] = await request.json()
        model: str = body.get("model", "")
        streaming: bool = body.get("stream", False)
        client: httpx.AsyncClient = request.app.state.http

        log.debug(
            "request_received",
            model=model,
            streaming=streaming,
            num_messages=len(body.get("messages", [])),
            path=str(request.url),
            has_system=bool(body.get("system")),
        )

        upstream_headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP}

        if model == LOCAL_MODEL_ALIAS and settings.local_configured:
            return await _handle_local(client, body, streaming, upstream_headers, settings)
        if model == LOCAL_MODEL_ALIAS:
            log.debug("local_unconfigured_fallback", fallback_model=settings.fallback_model)
            body = {**body, "model": settings.fallback_model}
        return await _handle_anthropic(client, body, streaming, upstream_headers, settings)

    @app.post("/v1/messages/count_tokens")
    async def count_tokens(request: Request) -> Any:
        body: dict[str, Any] = await request.json()
        model: str = body.get("model", "")
        client: httpx.AsyncClient = request.app.state.http

        if model == LOCAL_MODEL_ALIAS and settings.local_configured:
            tokens = _count_tokens_local(body)
            log.debug("count_tokens_local", tokens=tokens)
            return JSONResponse({"input_tokens": tokens})

        if model == LOCAL_MODEL_ALIAS:
            body = {**body, "model": settings.fallback_model}

        upstream_headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP}
        url = f"{settings.anthropic_base_url}/v1/messages/count_tokens"
        resp = await client.post(url, json=body, headers=upstream_headers)
        log.debug("count_tokens_anthropic", status_code=resp.status_code, model=body.get("model"))
        return JSONResponse(content=resp.json(), status_code=resp.status_code)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


async def _handle_local(
    client: httpx.AsyncClient,
    body: dict[str, Any],
    streaming: bool,
    headers: dict[str, str],
    settings: Settings,
) -> Any:
    assert settings.local_model is not None and settings.lm_studio_url is not None
    try:
        await lms.ensure_loaded(settings.local_model, settings.local_model_context_length)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # Replace "local" alias with the real model ID; forward everything else as-is.
    # LM Studio 0.4.1+ supports the Anthropic /v1/messages format natively.
    forwarded_body = {**body, "model": settings.local_model}
    url = f"{settings.lm_studio_url}/v1/messages"
    # LM Studio ignores the api key value but requires a non-empty header when auth is enabled.
    forward_headers = {k: v for k, v in headers.items() if k.lower() != "x-api-key"}
    forward_headers["x-api-key"] = "lmstudio"

    log.debug(
        "routing_local",
        url=url,
        model=settings.local_model,
        streaming=streaming,
        context_length=settings.local_model_context_length,
    )

    if streaming:
        return StreamingResponse(
            _proxy_stream(client, url, forward_headers, forwarded_body),
            media_type="text/event-stream",
        )

    resp = await client.post(url, json=forwarded_body, headers=forward_headers)
    log.debug("local_response", status_code=resp.status_code, content_length=len(resp.content))
    if resp.status_code != 200:
        log.warning("local_error", status_code=resp.status_code, body=resp.text[:500])
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return JSONResponse(resp.json())


async def _proxy_stream(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
) -> AsyncGenerator[bytes, None]:
    log.debug("stream_start", url=url)
    chunks = 0
    async with client.stream("POST", url, headers=headers, json=body) as resp:
        log.debug("stream_connected", status_code=resp.status_code)
        if resp.status_code != 200:
            error_body = await resp.aread()
            log.warning("stream_error", status_code=resp.status_code, body=error_body[:500])
            raise HTTPException(status_code=resp.status_code, detail=error_body.decode())
        async for chunk in resp.aiter_bytes():
            chunks += 1
            yield chunk
    log.debug("stream_done", chunks=chunks)


async def _handle_anthropic(
    client: httpx.AsyncClient,
    body: dict[str, Any],
    streaming: bool,
    headers: dict[str, str],
    settings: Settings,
) -> Any:
    url = f"{settings.anthropic_base_url}/v1/messages"
    log.debug("routing_anthropic", url=url, model=body.get("model"), streaming=streaming)

    if streaming:
        return StreamingResponse(
            _proxy_stream(client, url, headers, body),
            media_type="text/event-stream",
        )

    resp = await client.post(url, json=body, headers=headers)
    log.debug("anthropic_response", status_code=resp.status_code)
    return JSONResponse(content=resp.json(), status_code=resp.status_code)


def _count_tokens_local(body: dict[str, Any]) -> int:
    total = 0
    system = body.get("system")
    if isinstance(system, str):
        total += len(_encoding().encode(system))
    elif isinstance(system, list):
        total += _count_blocks(system)

    for message in body.get("messages", []):
        content = message.get("content", "")
        if isinstance(content, str):
            total += len(_encoding().encode(content))
        elif isinstance(content, list):
            total += _count_blocks(content)

    for tool in body.get("tools", []) or []:
        # Tool schemas end up in the model's system prompt; approximate by
        # encoding the JSON-ish string form of the schema.
        total += len(_encoding().encode(str(tool)))

    return total


def _count_blocks(blocks: list[dict[str, Any]]) -> int:
    total = 0
    for block in blocks:
        block_type = block.get("type")
        if block_type == "text":
            total += len(_encoding().encode(block.get("text", "")))
        elif block_type == "image":
            total += _IMAGE_BLOCK_TOKENS
        elif block_type in ("tool_use", "tool_result"):
            total += _TOOL_USE_BLOCK_TOKENS
            content = block.get("content")
            if isinstance(content, str):
                total += len(_encoding().encode(content))
            elif isinstance(content, list):
                total += _count_blocks(content)
        else:
            total += len(_encoding().encode(str(block)))
    return total
