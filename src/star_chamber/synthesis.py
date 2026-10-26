"""Closing assessment of a finished council session, written by one of its members."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from star_chamber.prompt import render_synthesis_prompt
from star_chamber.transport import sanitize_error, send_to_provider
from star_chamber.types import (
    Approach,
    ClassificationResult,
    OtariConfig,
    ProviderConfig,
    ProviderDesignAdvice,
    ProviderError,
    ProviderReview,
    QualityRating,
)

# A reply long enough to be prose rather than the sentence or two asked for.
MAX_ASSESSMENT_CHARS = 2000

_BOTH_STATES = "An assessment cannot carry both prose and a failure."
_NO_AUTHOR = "An assessment and a failure each need the member they came from."
_EMPTY_REPLY = "The member returned no assessment."
_OVERLONG_REPLY = f"The member returned more than {MAX_ASSESSMENT_CHARS} characters instead of a sentence or two."

_WHITESPACE_RUN = re.compile(r"\s+")


@dataclass(frozen=True)
class Synthesis:
    """A council session's closing assessment, or the reason there is none.

    Attributes:
        author: The member asked for the assessment. Empty when none was asked.
        text: The assessment in prose. Empty when no member wrote one.
        error: The member asked and why it wrote nothing. None when text is
            present, and None when no assessment was requested.

    Raises:
        ValueError: If prose and a failure are given together, or if either
            arrives without an author.

    NOTE: An empty assessment with no failure means none was requested, either
    because no member answered or because there was nothing to assess. A reader
    tells those apart by whether any member answered at all.
    """

    author: str = ""
    text: str = ""
    error: ProviderError | None = None

    def __post_init__(self) -> None:
        """Reject a state the attributes above say cannot exist."""
        if self.text and self.error is not None:
            raise ValueError(_BOTH_STATES)
        if (self.text or self.error is not None) and not self.author:
            raise ValueError(_NO_AUTHOR)


Synthesizer = Callable[[str, Sequence[ProviderConfig]], Awaitable[Synthesis]]
"""Takes council material and the members that answered, and returns the assessment."""


def render_code_review_material(
    reviews: Sequence[ProviderReview],
    classification: ClassificationResult,
) -> str:
    """Render a code review session as material to assess.

    Args:
        reviews: Each answering member's parsed review.
        classification: The issues grouped by how many members raised them.

    Returns:
        Markdown carrying the verdict tally, the assessments, and the grouped concerns.

    NOTE: The material names no member and preserves no member ordering, so an
    assessment written from it can neither attribute a point to a member nor
    count how many points each member raised.
    """
    sections: list[tuple[str, Sequence[str]]] = [
        ("Verdicts given", _rating_tally(reviews)),
        ("Each assessment of its own", [review.summary for review in reviews]),
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

    NOTE: The material names no member and preserves no member ordering, so an
    assessment written from it can neither attribute a point to a member nor
    count how many points each member raised.
    """
    sections: list[tuple[str, Sequence[str]]] = [
        ("Each assessment of its own", [advice.summary for advice in advices]),
        ("Each recommendation made", [advice.recommendation for advice in advices]),
        (
            "Approaches named, and how many members recommended each",
            [f"{approach.name} (recommended by {approach.recommended_by})" for approach in approaches],
        ),
    ]
    return _render_sections(sections)


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

    NOTE: Blank material is nothing to assess, so no member is asked and the
    assessment comes back empty. Asking would invite prose invented from
    nothing.
    NOTE: The member writing the assessment also wrote part of the material, so
    it can recognize its own prose even though the material names nobody.
    NOTE: A failure nothing anticipated becomes the assessment's own failure
    rather than an exception, because the session's whole output is built after
    this call and an exception here discards every member's reply.
    """
    if not material.strip():
        return Synthesis()

    try:
        response = await send_to_provider(config, render_synthesis_prompt(material), timeout=timeout, otari=otari)
    except Exception as exc:
        error = ProviderError(
            provider=config.identity,
            model=config.model,
            error=f"Unexpected failure asking {config.identity} for the assessment: {sanitize_error(str(exc))}",
            raw_content=None,
        )
        return Synthesis(author=config.identity, error=error)

    if not response.success:
        error = ProviderError(
            provider=config.identity, model=config.model, error=response.error, raw_content=response.content
        )
        return Synthesis(author=config.identity, error=error)

    text = _one_line(response.content)
    if not text:
        error = ProviderError(
            provider=config.identity, model=config.model, error=_EMPTY_REPLY, raw_content=response.content
        )
        return Synthesis(author=config.identity, error=error)
    if len(text) > MAX_ASSESSMENT_CHARS:
        error = ProviderError(
            provider=config.identity, model=config.model, error=_OVERLONG_REPLY, raw_content=response.content
        )
        return Synthesis(author=config.identity, error=error)

    return Synthesis(author=config.identity, text=text)


def _one_line(text: str) -> str:
    """Return text as one line, dropping characters that cannot be displayed."""
    displayable = "".join(character for character in text if character.isprintable() or character.isspace())
    return _WHITESPACE_RUN.sub(" ", displayable).strip()


def _rating_tally(reviews: Sequence[ProviderReview]) -> list[str]:
    """Return one entry counting how many members gave each verdict, worst verdict first.

    A single tally rather than one entry per member, so the material carries
    nothing a reader could line up against another section.
    """
    counts = Counter(_one_line(review.quality_rating) for review in reviews)
    # QualityRating is declared worst first, so a reader meets the harshest verdict before the kindest.
    ordered = [rating for rating in QualityRating if counts[rating]]
    if not ordered:
        return []
    return [", ".join(f"{rating}: {counts[rating]}" for rating in ordered)]


def _render_sections(sections: Sequence[tuple[str, Sequence[str]]]) -> str:
    """Render each heading and its bullets, dropping a section that has no entries.

    Each entry is reduced to one line, then empty entries are dropped, then the
    rest are sorted. That order matters: sorting first would let leading
    whitespace decide the order, which only looks deterministic.
    """
    parts: list[str] = []
    for heading, entries in sections:
        bullets = sorted(entry for entry in (_one_line(entry) for entry in entries) if entry)
        if not bullets:
            continue
        rendered = "\n".join(f"- {bullet}" for bullet in bullets)
        parts.append(f"## {heading}\n{rendered}")
    return "\n\n".join(parts)
