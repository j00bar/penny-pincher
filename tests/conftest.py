import os

import pytest


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip PP_* env vars so tests don't inherit a developer's local config."""
    for key in list(os.environ):
        if key.startswith("PP_"):
            monkeypatch.delenv(key, raising=False)
