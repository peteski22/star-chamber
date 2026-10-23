"""Closing assessment of a finished council session, written by one of its members."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from star_chamber.prompt import render_synthesis_prompt
from star_chamber.transport import send_to_provider
from star_chamber.types import (
    Approach,
    ClassificationResult,
    OtariConfig,
    ProviderConfig,
    ProviderDesignAdvice,
    ProviderError,
    ProviderReview,
)

_EMPTY_REPLY = "The member returned no assessment."


@dataclass(frozen=True)
class Synthesis:
    """A council session's closing assessment, or the reason there is none.

    Attributes:
        text: The assessment in prose. Empty when no member wrote one.
        error: The member asked and why it wrote nothing. None when text is
            present, and None when no member could be asked.
    """

    text: str = ""
    error: ProviderError | None = None


Synthesizer = Callable[[str, Sequence[str]], Awaitable[Synthesis]]
"""Takes council material and the display names that answered, and returns the assessment."""


def render_code_review_material(
    reviews: Sequence[ProviderReview],
    classification: ClassificationResult,
) -> str:
    """Render a code review session as material to assess.

    Args:
        reviews: Each answering member's parsed review.
        classification: The issues grouped by how many members raised them.

    Returns:
        Markdown carrying the assessments, the ratings, and the grouped concerns.

    NOTE: The material names no member, so an assessment written from it cannot
    attribute a point to one.
    """
    sections: list[tuple[str, Sequence[str]]] = [
        ("Quality ratings given", [review.quality_rating for review in reviews]),
        ("Each assessment of its own", [review.summary for review in reviews if review.summary]),
        ("Concerns every member raised", [issue.description for issue in classification.consensus_issues]),
        ("Concerns more than one member raised", [issue.description for issue in classification.majority_issues]),
        (
            "Concerns one member raised",
            [issue.description for issues in classification.individual_issues.values() for issue in issues],
        ),
    ]
    return _render_sections(sections)


def render_design_material(
    advices: Sequence[ProviderDesignAdvice],
    approaches: Sequence[Approach],
) -> str:
    """Render a design question session as material to assess.

    Args:
        advices: Each answering member's parsed advice.
        approaches: The approaches grouped across the members.

    Returns:
        Markdown carrying the assessments, the recommendations, and the approaches.

    NOTE: The material names no member, so an assessment written from it cannot
    attribute a point to one.
    """
    sections: list[tuple[str, Sequence[str]]] = [
        ("Each assessment of its own", [advice.summary for advice in advices if advice.summary]),
        ("Each recommendation made", [advice.recommendation for advice in advices if advice.recommendation]),
        (
            "Approaches named, and how many members recommended each",
            [f"{approach.name} (recommended by {approach.recommended_by})" for approach in approaches],
        ),
    ]
    return _render_sections(sections)


def select_synthesizer(
    configs: Sequence[ProviderConfig],
    answered: Sequence[str],
) -> ProviderConfig | None:
    """Return the member that writes the closing assessment, or None when none can.

    Args:
        configs: The configured roster, in configuration order.
        answered: Display names of the members that returned usable output.

    Returns:
        The first configured member that answered, or None when none did.

    NOTE: Selection follows roster order rather than reply order, so one roster
    always asks the same member.
    """
    names = set(answered)
    for config in configs:
        if config.identity in names:
            return config
    return None


async def synthesize(
    material: str,
    config: ProviderConfig,
    *,
    timeout: float | None = None,
    otari: OtariConfig | None = None,
) -> Synthesis:
    """Ask one member to write the session's closing assessment.

    Args:
        material: The council material to assess, carrying no member identities.
        config: The member to ask.
        timeout: Optional call timeout in seconds.
        otari: Optional Otari gateway routing configuration.

    Returns:
        The assessment, or an empty one carrying the reason it is missing.
    """
    response = await send_to_provider(config, render_synthesis_prompt(material), timeout=timeout, otari=otari)

    if not response.success:
        return Synthesis(error=ProviderError(provider=config.identity, error=response.error))

    text = response.content.strip()
    if not text:
        return Synthesis(error=ProviderError(provider=config.identity, error=_EMPTY_REPLY))

    return Synthesis(text=text)


def _render_sections(sections: Sequence[tuple[str, Sequence[str]]]) -> str:
    """Render each heading and its bullets, dropping a section that has no entries."""
    parts: list[str] = []
    for heading, entries in sections:
        if not entries:
            continue
        bullets = "\n".join(f"- {entry}" for entry in entries)
        parts.append(f"## {heading}\n{bullets}")
    return "\n\n".join(parts)
