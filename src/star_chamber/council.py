"""Top-level council orchestrator tying together prompt, transport, parsing, and consensus."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from star_chamber.config import load_config
from star_chamber.consensus import classify
from star_chamber.parsing import ParseError, parse_code_review, parse_design_advice
from star_chamber.prompt import augment_with_synthesis, render_code_review_prompt, render_design_prompt
from star_chamber.transport import ProviderResponse, fan_out, resolve_api_keys
from star_chamber.types import (
    Approach,
    CodeReviewResult,
    CouncilConfig,
    DebateMetadata,
    DesignQuestionResult,
    ProviderDesignAdvice,
    ProviderError,
    ProviderReview,
)

_VALID_MODES = frozenset({"code-review", "design-question"})


def _synthesize_round(responses: Sequence[ProviderResponse]) -> str:
    """Build an anonymous Chatham House synthesis from round responses.

    Produces a plain-text summary of what providers said without attributing
    responses to specific providers.

    Args:
        responses: Provider responses from a single fan-out round.

    Returns:
        Anonymous synthesis text suitable for augmenting the next round's prompt.
    """
    successful = [r for r in responses if r.success and r.content]
    parts: list[str] = []
    for idx, response in enumerate(successful, start=1):
        parts.append(f"- Reviewer {idx}: {response.content[:500]}")
    return "\n".join(parts)


def _build_code_review_result(
    responses: Sequence[ProviderResponse],
    threshold: int = 2,
    debate: DebateMetadata | None = None,
) -> CodeReviewResult:
    """Parse responses, classify issues, and build a CodeReviewResult.

    Args:
        responses: Provider responses from the final fan-out round.
        threshold: Consensus threshold for classification.
        debate: Optional debate metadata to attach to the result.

    Returns:
        A fully populated CodeReviewResult.
    """
    reviews: list[ProviderReview] = []
    failed: list[ProviderError] = []
    providers_used: list[str] = []

    for response in responses:
        providers_used.append(response.provider)
        if not response.success:
            failed.append(ProviderError(provider=response.provider, error=response.error))
            continue
        try:
            review = parse_code_review(response.content, response.provider, response.model)
            reviews.append(review)
        except ParseError as exc:
            failed.append(ProviderError(provider=response.provider, error=str(exc)))

    # Classify issues across all successful reviews.
    classification = classify(reviews, threshold=threshold)

    # Build quality ratings map.
    quality_ratings = {r.provider: r.quality_rating for r in reviews}

    # Build summary from individual review summaries.
    summaries = [r.summary for r in reviews if r.summary]
    summary = " ".join(summaries) if summaries else ""

    return CodeReviewResult(
        mode="code-review",
        providers_used=tuple(providers_used),
        failed_providers=tuple(failed),
        reviews=tuple(reviews),
        consensus_issues=classification.consensus_issues,
        majority_issues=classification.majority_issues,
        individual_issues=classification.individual_issues,
        quality_ratings=quality_ratings,
        summary=summary,
        debate=debate,
    )


def _build_design_result(
    responses: Sequence[ProviderResponse],
    prompt: str,
    debate: DebateMetadata | None = None,
) -> DesignQuestionResult:
    """Parse responses, aggregate approaches, and build a DesignQuestionResult.

    Args:
        responses: Provider responses from the final fan-out round.
        prompt: The original design question.
        debate: Optional debate metadata to attach to the result.

    Returns:
        A fully populated DesignQuestionResult.
    """
    advices: list[ProviderDesignAdvice] = []
    failed: list[ProviderError] = []
    providers_used: list[str] = []

    for response in responses:
        providers_used.append(response.provider)
        if not response.success:
            failed.append(ProviderError(provider=response.provider, error=response.error))
            continue
        try:
            advice = parse_design_advice(response.content, response.provider, response.model)
            advices.append(advice)
        except ParseError as exc:
            failed.append(ProviderError(provider=response.provider, error=str(exc)))

    # Aggregate all approaches from all successful providers.
    all_approaches: list[Approach] = []
    for advice in advices:
        all_approaches.extend(advice.approaches)

    # Check for consensus recommendation.
    recommendations = [a.recommendation for a in advices]
    consensus_recommendation: str | None = None
    if recommendations and all(r == recommendations[0] for r in recommendations):
        consensus_recommendation = recommendations[0]

    # Build summary from individual advice summaries.
    summaries = [a.summary for a in advices if a.summary]
    summary = " ".join(summaries) if summaries else ""

    return DesignQuestionResult(
        mode="design-question",
        prompt=prompt,
        providers_used=tuple(providers_used),
        failed_providers=tuple(failed),
        approaches=tuple(all_approaches),
        consensus_recommendation=consensus_recommendation,
        summary=summary,
        debate=debate,
    )


async def run_council(
    *,
    files: dict[str, str] | None = None,
    config: CouncilConfig | None = None,
    prompt: str = "",
    mode: str = "code-review",
    context: str = "",
    debate: bool = False,
    rounds: int = 1,
) -> CodeReviewResult | DesignQuestionResult:
    """Run a multi-LLM council session.

    Orchestrates the full council flow: resolve API keys, build prompt,
    fan-out to providers, parse responses, and classify/aggregate results.

    Args:
        files: Mapping of file paths to source content (required for code-review mode).
        config: Council configuration. Auto-loaded from providers.json when None.
        prompt: The design question (required for design-question mode).
        mode: Council mode, either "code-review" or "design-question".
        context: Optional project-specific context string.
        debate: Whether to run in debate mode with multiple rounds.
        rounds: Number of debate rounds (only used when debate is True).

    Returns:
        CodeReviewResult for code-review mode, DesignQuestionResult for design-question mode.

    Raises:
        ValueError: If required parameters for the chosen mode are missing, or
            if an invalid mode is specified.
    """
    # Validate mode.
    if mode not in _VALID_MODES:
        msg = f"Invalid mode: {mode}. Must be one of: {', '.join(sorted(_VALID_MODES))}"
        raise ValueError(msg)

    # Validate mode-specific parameters.
    if mode == "code-review" and not files:
        msg = "The files parameter is required for code-review mode."
        raise ValueError(msg)
    if mode == "design-question" and not prompt:
        msg = "The prompt parameter is required for design-question mode."
        raise ValueError(msg)

    # Auto-load config when not provided.
    if config is None:
        config = load_config()

    # Resolve API keys.
    use_platform = config.platform is not None
    resolved_providers = resolve_api_keys(config.providers, use_platform=use_platform)

    # Build the initial prompt.
    if mode == "code-review":
        rendered_prompt = render_code_review_prompt(files, context=context)  # type: ignore[arg-type]
    else:
        rendered_prompt = render_design_prompt(prompt, context=context)

    # Determine effective round count.
    effective_rounds = rounds if debate else 1

    # Execute rounds.
    current_prompt = rendered_prompt
    last_responses: list[ProviderResponse] = []

    for round_num in range(1, effective_rounds + 1):
        last_responses = await fan_out(
            configs=resolved_providers,
            prompt=current_prompt,
            timeout=float(config.timeout_seconds),
        )

        # If not the final round, augment the prompt with synthesis.
        if round_num < effective_rounds:
            synthesis = _synthesize_round(last_responses)
            current_prompt = augment_with_synthesis(rendered_prompt, synthesis, round_number=round_num)

    # Build debate metadata if applicable.
    debate_meta: DebateMetadata | None = None
    if debate:
        debate_meta = DebateMetadata(rounds_completed=effective_rounds, converged=False)

    # Build and return the appropriate result type.
    if mode == "code-review":
        return _build_code_review_result(
            last_responses,
            threshold=config.consensus_threshold,
            debate=debate_meta,
        )

    return _build_design_result(
        last_responses,
        prompt=prompt,
        debate=debate_meta,
    )


def run_council_sync(
    *,
    files: dict[str, str] | None = None,
    config: CouncilConfig | None = None,
    prompt: str = "",
    mode: str = "code-review",
    context: str = "",
    debate: bool = False,
    rounds: int = 1,
) -> CodeReviewResult | DesignQuestionResult:
    """Synchronous wrapper around run_council().

    Args:
        files: Mapping of file paths to source content (required for code-review mode).
        config: Council configuration. Auto-loaded from providers.json when None.
        prompt: The design question (required for design-question mode).
        mode: Council mode, either "code-review" or "design-question".
        context: Optional project-specific context string.
        debate: Whether to run in debate mode with multiple rounds.
        rounds: Number of debate rounds (only used when debate is True).

    Returns:
        CodeReviewResult for code-review mode, DesignQuestionResult for design-question mode.
    """
    return asyncio.run(
        run_council(
            files=files,
            config=config,
            prompt=prompt,
            mode=mode,
            context=context,
            debate=debate,
            rounds=rounds,
        )
    )
