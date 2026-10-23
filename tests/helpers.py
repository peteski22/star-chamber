"""Helpers shared by the council tests."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from star_chamber.council import _build_code_review_result, _build_design_result
from star_chamber.synthesis import Synthesis, Synthesizer
from star_chamber.transport import ProviderResponse
from star_chamber.types import CodeReviewResult, DesignQuestionResult, ProviderConfig

SYNTHESIS = "The members broadly agreed."
SYNTHESIS_AUTHOR = "scribe"


async def stub_synthesizer(material: str, answered: Sequence[ProviderConfig]) -> Synthesis:
    """Stand in for the member that writes the closing assessment.

    Asks nobody when nobody answered, as the orchestrator's own synthesizer does.
    """
    if not answered:
        return Synthesis()
    return Synthesis(author=SYNTHESIS_AUTHOR, text=SYNTHESIS)


def build_code_review_result(
    responses: Sequence[ProviderResponse],
    *,
    files_reviewed: tuple[str, ...],
    threshold: int = 2,
    synthesizer: Synthesizer = stub_synthesizer,
) -> CodeReviewResult:
    """Build a code review result without reaching a provider for the synthesis."""
    return asyncio.run(
        _build_code_review_result(
            exchanges(responses),
            files_reviewed=files_reviewed,
            threshold=threshold,
            synthesizer=synthesizer,
        )
    )


def build_design_result(
    responses: Sequence[ProviderResponse],
    *,
    prompt: str,
    synthesizer: Synthesizer = stub_synthesizer,
) -> DesignQuestionResult:
    """Build a design question result without reaching a provider for the synthesis."""
    return asyncio.run(_build_design_result(exchanges(responses), prompt=prompt, synthesizer=synthesizer))


def exchanges(
    responses: Sequence[ProviderResponse],
) -> list[tuple[ProviderConfig, ProviderResponse]]:
    """Pair each reply with a member configured the way the reply describes it.

    Stands in for what the fan-out returns, so a test can hand replies alone to
    anything that now takes pairs.
    """
    return [
        (
            ProviderConfig(provider=response.provider, model=response.model, display_name=response.display_name),
            response,
        )
        for response in responses
    ]
