# penny-pincher

Local Claude API proxy. Exposes `/v1/messages` and `/v1/messages/count_tokens` (Anthropic format) and routes requests to LM Studio or Anthropic based on the request's `model` field.

## Routing

| Request `model` | Local backend configured? | Destination |
|---|---|---|
| `"local"` | yes | LM Studio at `lm_studio_url` (with the real model ID substituted) |
| `"local"` | no | Anthropic API, rewritten to `fallback_model` (default `claude-haiku-4-5`) |
| anything else | — | Anthropic API, passed through unchanged |

The local backend is considered configured when both `local_model` and `lm_studio_url` are set. If either is missing, `"local"` requests fall through to Anthropic — useful when a parent app wants to run entirely on the API without any local model.

`count_tokens` uses the same routing: local requests are counted with `tiktoken` (`cl100k_base` encoding, a rough approximation suitable for context budgeting); everything else is proxied to Anthropic's real `count_tokens` endpoint.

## Install

```bash
# Library only (for mounting into another FastAPI app)
uv add penny-pincher

# With the CLI (`penny-pincher serve`)
uv add "penny-pincher[cli]"
```

## As a library — mounted sub-app

`create_app()` returns a `FastAPI` instance. Mounting it into a parent app works out of the box: Starlette ≥0.32 propagates the sub-app's lifespan, so the internal `httpx.AsyncClient` is created and torn down as part of the parent's lifecycle.

```python
from fastapi import FastAPI
from penny_pincher import Settings, create_app

parent = FastAPI()
parent.mount(
    "/proxy",
    create_app(
        Settings(
            local_model="qwen3.5-9b-mlx",
            lm_studio_url="http://localhost:1234",
        )
    ),
)
```

Settings can come from anywhere — env vars (`PP_*` prefix), a `.env` file, or explicit constructor args. Explicit args win over env vars.

### Sharing an `httpx.AsyncClient`

If you want the parent to own the HTTP client (e.g. to share a connection pool with other outbound traffic), pass one in:

```python
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI
from penny_pincher import Settings, create_app

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with httpx.AsyncClient(timeout=300.0) as http:
        app.mount("/proxy", create_app(Settings(...), http_client=http))
        yield

parent = FastAPI(lifespan=lifespan)
```

## Standalone CLI

```bash
uv sync --extra cli
uv run penny-pincher serve --local-model qwen3.5-9b-mlx --lm-studio-url http://localhost:1234
```

## Configuration

All settings are readable from environment variables (`PP_` prefix) or a `.env` file, and overridable via CLI flags or constructor args.

| Env var | CLI flag | Default | Description |
|---|---|---|---|
| `PP_HOST` | `--host` | `127.0.0.1` | Bind address (CLI only) |
| `PP_PORT` | `--port` | `8082` | Bind port (CLI only) |
| `PP_LOCAL_MODEL` | `--local-model` | *(none)* | LM Studio model ID; unset → local disabled |
| `PP_LM_STUDIO_URL` | `--lm-studio-url` | *(none)* | LM Studio base URL; unset → local disabled |
| `PP_LOCAL_MODEL_CONTEXT_LENGTH` | `--context-length` | `32768` | Context window for local model |
| `PP_FALLBACK_MODEL` | `--fallback-model` | `claude-haiku-4-5` | Model used when `"local"` is requested but local is unconfigured |
| `PP_ANTHROPIC_BASE_URL` | — | `https://api.anthropic.com` | Anthropic base URL |

Anthropic API keys are read from the incoming request's `x-api-key` / `authorization` headers — the proxy never stores one.

## Development

```bash
uv sync --extra cli --group dev
uv run pytest
prek run   # ruff format/lint + mypy
```
