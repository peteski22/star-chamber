"""Top-level council orchestrator tying together prompt, transport, parsing, and consensus."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from star_chamber.config import load_config
from star_chamber.consensus import classify
from star_chamber.parsing import ParseError, parse_code_review, parse_design_advice
from star_chamber.prompt import render_code_review_prompt, render_design_prompt
from star_chamber.transport import ProviderResponse, fan_out, resolve_api_keys
from star_chamber.types import (
    Approach,
    CodeReviewResult,
    CouncilConfig,
    DesignQuestionResult,
    ProviderDesignAdvice,
    ProviderError,
    ProviderReview,
)

_VALID_MODES = frozenset({"code-review", "design-question"})


def _build_code_review_result(
    responses: Sequence[ProviderResponse],
    threshold: int = 2,
) -> CodeReviewResult:
    """Parse responses, classify issues, and build a CodeReviewResult.

    Args:
        responses: Provider responses from the fan-out round.
        threshold: Consensus threshold for classification.

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
    )


def _build_design_result(
    responses: Sequence[ProviderResponse],
    prompt: str,
) -> DesignQuestionResult:
    """Parse responses, aggregate approaches, and build a DesignQuestionResult.

    Args:
        responses: Provider responses from the fan-out round.
        prompt: The original design question.

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
    )


async def run_council(
    *,
    files: dict[str, str] | None = None,
    config: CouncilConfig | None = None,
    prompt: str = "",
    mode: str = "code-review",
    context: str = "",
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

    # Fan out to all providers.
    responses = await fan_out(
        configs=resolved_providers,
        prompt=rendered_prompt,
        timeout=float(config.timeout_seconds),
    )

    # Build and return the appropriate result type.
    if mode == "code-review":
        return _build_code_review_result(
            responses,
            threshold=config.consensus_threshold,
        )

    return _build_design_result(
        responses,
        prompt=prompt,
    )


def run_council_sync(
    *,
    files: dict[str, str] | None = None,
    config: CouncilConfig | None = None,
    prompt: str = "",
    mode: str = "code-review",
    context: str = "",
) -> CodeReviewResult | DesignQuestionResult:
    """Synchronous wrapper around run_council().

    Args:
        files: Mapping of file paths to source content (required for code-review mode).
        config: Council configuration. Auto-loaded from providers.json when None.
        prompt: The design question (required for design-question mode).
        mode: Council mode, either "code-review" or "design-question".
        context: Optional project-specific context string.

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
        )
    )
