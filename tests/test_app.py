import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from penny_pincher import Settings, create_app


def _make_client(settings: Settings, handler: Callable[[httpx.Request], httpx.Response]) -> TestClient:
    """Build a TestClient wired to a MockTransport so no real network is touched."""
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return TestClient(create_app(settings, http_client=http))


def _anthropic_stub(calls: list[httpx.Request]) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path.endswith("/count_tokens"):
            return httpx.Response(200, json={"input_tokens": 42})
        return httpx.Response(
            200,
            json={"id": "msg_1", "type": "message", "role": "assistant", "content": [], "model": "stub"},
        )

    return handler


def test_health() -> None:
    tc = _make_client(Settings(), lambda _r: httpx.Response(500))
    resp = tc.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_messages_non_local_forwards_to_anthropic() -> None:
    calls: list[httpx.Request] = []
    tc = _make_client(Settings(), _anthropic_stub(calls))
    resp = tc.post(
        "/v1/messages",
        json={"model": "claude-sonnet-4-5", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    assert len(calls) == 1
    assert json.loads(calls[0].content)["model"] == "claude-sonnet-4-5"
    assert calls[0].url.host == "api.anthropic.com"


def test_messages_local_falls_back_when_unconfigured() -> None:
    calls: list[httpx.Request] = []
    tc = _make_client(Settings(fallback_model="claude-haiku-4-5"), _anthropic_stub(calls))
    resp = tc.post(
        "/v1/messages",
        json={"model": "local", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    assert json.loads(calls[0].content)["model"] == "claude-haiku-4-5"
    assert calls[0].url.host == "api.anthropic.com"


def test_messages_local_routes_to_lm_studio_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _noop_ensure_loaded(model_id: str, ctx: int) -> None:
        return None

    monkeypatch.setattr("penny_pincher.app.lms.ensure_loaded", _noop_ensure_loaded)

    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"id": "msg_1", "content": [], "model": "qwen"})

    tc = _make_client(Settings(local_model="qwen", lm_studio_url="http://localhost:1234"), handler)
    resp = tc.post(
        "/v1/messages",
        json={"model": "local", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert resp.status_code == 200
    assert calls[0].url.host == "localhost"
    assert calls[0].url.port == 1234
    assert json.loads(calls[0].content)["model"] == "qwen"
    assert calls[0].headers["x-api-key"] == "lmstudio"


def test_count_tokens_local_uses_tiktoken() -> None:
    def handler(_r: httpx.Request) -> httpx.Response:
        raise AssertionError("local count_tokens must not hit upstream")

    tc = _make_client(Settings(local_model="qwen", lm_studio_url="http://localhost:1234"), handler)
    resp = tc.post(
        "/v1/messages/count_tokens",
        json={"model": "local", "messages": [{"role": "user", "content": "hello world"}]},
    )
    assert resp.status_code == 200
    tokens = resp.json()["input_tokens"]
    assert isinstance(tokens, int)
    assert tokens > 0


def test_count_tokens_local_counts_system_and_tools() -> None:
    def handler(_r: httpx.Request) -> httpx.Response:
        raise AssertionError("local count_tokens must not hit upstream")

    tc = _make_client(Settings(local_model="qwen", lm_studio_url="http://localhost:1234"), handler)
    body: dict[str, Any] = {
        "model": "local",
        "system": "you are helpful",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hi there"}]},
            {"role": "assistant", "content": "hello"},
        ],
        "tools": [{"name": "get_weather", "description": "gets the weather"}],
    }
    resp = tc.post("/v1/messages/count_tokens", json=body)
    assert resp.status_code == 200
    assert resp.json()["input_tokens"] > 5


def test_count_tokens_local_image_block_adds_fixed_cost() -> None:
    def handler(_r: httpx.Request) -> httpx.Response:
        raise AssertionError("local count_tokens must not hit upstream")

    tc = _make_client(Settings(local_model="qwen", lm_studio_url="http://localhost:1234"), handler)
    resp = tc.post(
        "/v1/messages/count_tokens",
        json={
            "model": "local",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}},
                    ],
                }
            ],
        },
    )
    assert resp.json()["input_tokens"] >= 1500


def test_count_tokens_fallback_proxies_to_anthropic() -> None:
    calls: list[httpx.Request] = []
    tc = _make_client(Settings(fallback_model="claude-haiku-4-5"), _anthropic_stub(calls))
    resp = tc.post(
        "/v1/messages/count_tokens",
        json={"model": "local", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    assert resp.json()["input_tokens"] == 42
    assert calls[0].url.path.endswith("/count_tokens")
    assert json.loads(calls[0].content)["model"] == "claude-haiku-4-5"


def test_count_tokens_non_local_proxies_to_anthropic() -> None:
    calls: list[httpx.Request] = []
    tc = _make_client(Settings(), _anthropic_stub(calls))
    resp = tc.post(
        "/v1/messages/count_tokens",
        json={"model": "claude-sonnet-4-5", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    assert json.loads(calls[0].content)["model"] == "claude-sonnet-4-5"


def test_mounted_as_subapp() -> None:
    def handler(_r: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "msg_1", "content": [], "model": "claude-haiku-4-5"})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    pp = create_app(Settings(fallback_model="claude-haiku-4-5"), http_client=http)

    parent = FastAPI()
    parent.mount("/proxy", pp)

    with TestClient(parent) as tc:
        assert tc.get("/proxy/health").status_code == 200
        resp = tc.post(
            "/proxy/v1/messages",
            json={"model": "local", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200
