"""Shared test fixtures."""

import pytest


@pytest.fixture(autouse=True)
def _no_ambient_otari_api_base(monkeypatch):
    """Keep the developer's own Otari base URL out of tests that read the environment."""
    for name in ("OTARI_API_BASE", "GATEWAY_API_BASE"):
        monkeypatch.delenv(name, raising=False)
