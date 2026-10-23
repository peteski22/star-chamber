"""Helpers shared by the council tests."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from star_chamber.council import _build_code_review_result, _build_design_result
from star_chamber.synthesis import Synthesis, Synthesizer
from star_chamber.transport import ProviderResponse
from star_chamber.types import CodeReviewResult, DesignQuestionResult

SYNTHESIS = "The members broadly agreed."


async def stub_synthesizer(material: str, answered: Sequence[str]) -> Synthesis:
    """Stand in for the member that writes the closing assessment."""
    return Synthesis(text=SYNTHESIS)


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
            responses,
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
    return asyncio.run(_build_design_result(responses, prompt=prompt, synthesizer=synthesizer))
