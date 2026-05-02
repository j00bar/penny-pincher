import asyncio

import structlog

log = structlog.get_logger(__name__)


async def _run(args: list[str]) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    return proc.returncode or 0, stdout.decode(), stderr.decode()


async def is_model_loaded(model_id: str) -> bool:
    log.debug("checking_model", model_id=model_id)
    code, stdout, stderr = await _run(["lms", "ps"])
    log.debug("lms_ps", returncode=code, stdout=stdout.strip(), stderr=stderr.strip())
    loaded = model_id in stdout
    log.debug("model_status", model_id=model_id, loaded=loaded)
    return loaded


async def load_model(model_id: str, context_length: int) -> None:
    log.info("loading_model", model_id=model_id, context_length=context_length)
    code, _, stderr = await _run(
        ["lms", "load", model_id, "--context-length", str(context_length), "--gpu", "max", "-y"]
    )
    if code != 0:
        log.error("load_failed", model_id=model_id, returncode=code, stderr=stderr.strip())
        raise RuntimeError(f"lms load failed for {model_id!r}: {stderr.strip()}")
    log.info("model_ready", model_id=model_id)


async def ensure_loaded(model_id: str, context_length: int) -> None:
    if not await is_model_loaded(model_id):
        await load_model(model_id, context_length)
