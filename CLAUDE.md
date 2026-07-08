# penny-pincher

Local Claude API proxy. Exposes `/v1/messages` (Anthropic format) and routes requests to either LM Studio or Anthropic based on the model name.

## Architecture

```
src/penny_pincher/
├── settings.py   # pydantic-settings config (env prefix: PP_)
├── lms.py        # LM Studio process management via `lms` CLI
├── logging.py    # structlog dev console configuration
├── app.py        # FastAPI app and route handlers
└── cli.py        # Typer CLI entrypoint
```

**Routing logic:**
- `model == "local"` → LM Studio at `PP_LM_STUDIO_URL` (default `http://localhost:1234`)
  - Wakes the model via `lms load` if not already running
  - Replaces `model: "local"` with the real model ID (`PP_LOCAL_MODEL`) and forwards as-is
  - LM Studio 0.4.1+ accepts Anthropic `/v1/messages` format natively — no translation needed
- Any other model → forwarded as-is to Anthropic

## Configuration

All settings are readable from environment variables (prefix `PP_`) or a `.env` file, and overridable via CLI flags.

| Env var | CLI flag | Default | Description |
|---|---|---|---|
| `PP_HOST` | `--host` | `127.0.0.1` | Bind address |
| `PP_PORT` | `--port` | `8082` | Bind port |
| `PP_LOCAL_MODEL` | `--local-model` | `mlx-community/qwen3.5-pb-mlx-8bit` | LM Studio model ID |
| `PP_LOCAL_MODEL_CONTEXT_LENGTH` | `--context-length` | `32000` | Context window for local model |
| `PP_LM_STUDIO_URL` | `--lm-studio-url` | `http://localhost:1234` | LM Studio base URL |
| `PP_ANTHROPIC_API_KEY` | `--api-key` | *(required for Anthropic routing)* | Anthropic API key |
| `PP_ANTHROPIC_BASE_URL` | — | `https://api.anthropic.com` | Anthropic base URL |
| `PP_LOCAL_MODEL_ALIAS` | `--local-model-alias` | `local` | Model name that routes requests to LM Studio |
| `PP_LMS_PATH` | `--lms-path` | `lms` | Path to the `lms` CLI binary |

## Development

```bash
uv sync
uv run penny-pincher serve
```

Pre-commit hooks (ruff format/lint + mypy):

```bash
prek run
```

## Translation layer

LM Studio speaks OpenAI format; the proxy converts:

- **Request:** Anthropic `system` + `messages` (with content blocks) → OpenAI `messages` array
- **Response (non-streaming):** OpenAI `choices[0].message` → Anthropic `content` blocks
- **Response (streaming):** OpenAI `data: {...delta...}` chunks → Anthropic SSE event sequence
  (`message_start` → `content_block_start` → `content_block_delta`× → `content_block_stop` → `message_delta` → `message_stop`)

Tool use and image content blocks are not translated — text-only for now.
