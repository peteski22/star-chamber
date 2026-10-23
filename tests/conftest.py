"""Shared test fixtures."""

import pytest

from star_chamber.transport import ProviderResponse
from star_chamber.types import OtariConfig, ProviderConfig
from tests.helpers import SYNTHESIS


@pytest.fixture(autouse=True)
def _no_ambient_otari_api_base(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's own Otari base URL out of tests that read the environment."""
    for name in ("OTARI_API_BASE", "GATEWAY_API_BASE"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _canned_synthesis_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    """Answer the closing assessment locally, so no test reaches a provider for one."""

    async def reply(
        config: ProviderConfig,
        prompt: str,
        timeout: float | None = None,
        otari: OtariConfig | None = None,
    ) -> ProviderResponse:
        return ProviderResponse(
            provider=config.provider,
            display_name=config.display_name or config.provider,
            model=config.model,
            success=True,
            content=SYNTHESIS,
        )

    monkeypatch.setattr("star_chamber.synthesis.send_to_provider", reply)
