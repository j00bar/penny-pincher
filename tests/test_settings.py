import pytest

from penny_pincher import Settings


@pytest.mark.parametrize(
    ("local_model", "lm_studio_url", "expected"),
    [
        (None, None, False),
        ("qwen", None, False),
        (None, "http://localhost:1234", False),
        ("qwen", "http://localhost:1234", True),
    ],
)
def test_local_configured(local_model: str | None, lm_studio_url: str | None, expected: bool) -> None:
    settings = Settings(local_model=local_model, lm_studio_url=lm_studio_url)
    assert settings.local_configured is expected


def test_defaults_are_unconfigured() -> None:
    assert Settings().local_configured is False
    assert Settings().fallback_model == "claude-haiku-4-5"
