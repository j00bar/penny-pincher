from typing import Annotated

import typer
import uvicorn

from penny_pincher.app import create_app
from penny_pincher.logging import configure_logging
from penny_pincher.settings import Settings

app = typer.Typer(name="penny-pincher", help="Local Claude API proxy with LM Studio support.")


@app.command()
def serve(
    host: Annotated[str | None, typer.Option("--host", help="Bind host (PP_HOST)")] = None,
    port: Annotated[int | None, typer.Option("--port", help="Bind port (PP_PORT)")] = None,
    local_model: Annotated[
        str | None, typer.Option("--local-model", help="LM Studio model ID (PP_LOCAL_MODEL)")
    ] = None,
    local_model_context_length: Annotated[
        int | None,
        typer.Option("--context-length", help="Context length for local model (PP_LOCAL_MODEL_CONTEXT_LENGTH)"),
    ] = None,
    lm_studio_url: Annotated[
        str | None, typer.Option("--lm-studio-url", help="LM Studio base URL (PP_LM_STUDIO_URL)")
    ] = None,
    fallback_model: Annotated[
        str | None,
        typer.Option("--fallback-model", help="Model to use when local is unconfigured (PP_FALLBACK_MODEL)"),
    ] = None,
) -> None:
    overrides = {
        k: v
        for k, v in {
            "host": host,
            "port": port,
            "local_model": local_model,
            "local_model_context_length": local_model_context_length,
            "lm_studio_url": lm_studio_url,
            "fallback_model": fallback_model,
        }.items()
        if v is not None
    }
    settings = Settings(**overrides)  # type: ignore[arg-type]
    configure_logging()
    fastapi_app = create_app(settings)

    typer.echo(f"Starting penny-pincher on http://{settings.host}:{settings.port}")
    uvicorn.run(fastapi_app, host=settings.host, port=settings.port)
