"""Top-level council orchestrator tying together prompt, transport, parsing, and consensus."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from star_chamber.config import load_config
from star_chamber.consensus import aggregate_approaches, classify, consensus_recommendation
from star_chamber.parsing import ParseError, parse_code_review, parse_design_advice
from star_chamber.prompt import render_code_review_prompt, render_design_prompt
from star_chamber.synthesis import (
    Synthesis,
    Synthesizer,
    render_code_review_material,
    render_design_material,
    synthesize,
)
from star_chamber.transport import ProviderResponse, fan_out, resolve_api_keys, resolve_otari
from star_chamber.types import (
    CodeReviewResult,
    CodeReviewSummary,
    CouncilConfig,
    DesignQuestionResult,
    DesignSummary,
    ProviderConfig,
    ProviderDesignAdvice,
    ProviderError,
    ProviderReview,
)

_VALID_MODES = frozenset({"code-review", "design-question"})


async def _build_code_review_result(
    exchanges: Sequence[tuple[ProviderConfig, ProviderResponse]],
    files_reviewed: tuple[str, ...],
    threshold: int = 2,
    *,
    synthesizer: Synthesizer,
) -> CodeReviewResult:
    """Parse responses, classify issues, and build a CodeReviewResult.

    Args:
        exchanges: Each member of the fan-out round paired with its reply.
        files_reviewed: Paths of the files sent for review.
        threshold: Consensus threshold for classification.
        synthesizer: Writes the session's closing assessment.

    Returns:
        A fully populated CodeReviewResult.

    NOTE: Every response must land in either failed_providers or reviews, never both and never neither.
    providers_responded is therefore the exact complement of failed_providers within providers_used.
    """
    reviews: list[ProviderReview] = []
    failed: list[ProviderError] = []
    providers_used: list[str] = []
    answered: list[ProviderConfig] = []

    for config, response in exchanges:
        providers_used.append(response.display_name)
        if not response.success:
            failed.append(
                ProviderError(
                    provider=response.display_name,
                    model=response.model,
                    error=response.error,
                    raw_content=response.content,
                )
            )
            continue
        try:
            review = parse_code_review(response.content, response.display_name, response.model)
        except ParseError as exc:
            failed.append(
                ProviderError(
                    provider=response.display_name,
                    model=response.model,
                    error=str(exc),
                    raw_content=response.content,
                )
            )
            continue
        reviews.append(review)
        answered.append(config)

    # Classify issues across all successful reviews.
    classification = classify(reviews, threshold=threshold)

    # Build quality ratings map.
    quality_ratings = {r.provider: r.quality_rating for r in reviews}

    synthesis = await synthesizer(render_code_review_material(reviews, classification), answered)

    total_issues = (
        len(classification.consensus_issues)
        + len(classification.majority_issues)
        + sum(len(issues) for issues in classification.individual_issues.values())
    )

    return CodeReviewResult(
        mode="code-review",
        files_reviewed=files_reviewed,
        providers_used=tuple(providers_used),
        providers_responded=tuple(r.provider for r in reviews),
        failed_providers=tuple(failed),
        reviews=tuple(reviews),
        consensus_issues=classification.consensus_issues,
        majority_issues=classification.majority_issues,
        individual_issues=classification.individual_issues,
        quality_ratings=quality_ratings,
        summary=CodeReviewSummary(
            total_issues=total_issues,
            consensus_count=len(classification.consensus_issues),
            majority_count=len(classification.majority_issues),
            synthesis=synthesis.text,
            synthesis_error=synthesis.error,
            synthesized_by=synthesis.author or None,
        ),
    )


async def _build_design_result(
    exchanges: Sequence[tuple[ProviderConfig, ProviderResponse]],
    prompt: str,
    *,
    synthesizer: Synthesizer,
) -> DesignQuestionResult:
    """Parse responses, aggregate approaches, and build a DesignQuestionResult.

    Args:
        exchanges: Each member of the fan-out round paired with its reply.
        prompt: The original design question.
        synthesizer: Writes the session's closing assessment.

    Returns:
        A fully populated DesignQuestionResult.

    NOTE: Every response must land in either failed_providers or the parsed advice, never both and never neither.
    providers_responded is therefore the exact complement of failed_providers within providers_used.
    """
    advices: list[ProviderDesignAdvice] = []
    failed: list[ProviderError] = []
    providers_used: list[str] = []
    answered: list[ProviderConfig] = []

    for config, response in exchanges:
        providers_used.append(response.display_name)
        if not response.success:
            failed.append(
                ProviderError(
                    provider=response.display_name,
                    model=response.model,
                    error=response.error,
                    raw_content=response.content,
                )
            )
            continue
        try:
            advice = parse_design_advice(response.content, response.display_name, response.model)
        except ParseError as exc:
            failed.append(
                ProviderError(
                    provider=response.display_name,
                    model=response.model,
                    error=str(exc),
                    raw_content=response.content,
                )
            )
            continue
        advices.append(advice)
        answered.append(config)

    approaches = aggregate_approaches(advices)

    synthesis = await synthesizer(render_design_material(advices, approaches), answered)

    return DesignQuestionResult(
        mode="design-question",
        prompt=prompt,
        providers_used=tuple(providers_used),
        providers_responded=tuple(a.provider for a in advices),
        failed_providers=tuple(failed),
        approaches=approaches,
        consensus_recommendation=consensus_recommendation(advices, approaches),
        summary=DesignSummary(
            synthesis=synthesis.text,
            synthesis_error=synthesis.error,
            synthesized_by=synthesis.author or None,
        ),
    )


async def run_council(
    *,
    files: dict[str, str] | None = None,
    config: CouncilConfig | None = None,
    prompt: str = "",
    mode: str = "code-review",
    context: str = "",
    council_context: str = "",
) -> CodeReviewResult | DesignQuestionResult:
    """Run a multi-LLM council session.

    Orchestrates the full council flow: resolve API keys, build prompt,
    fan-out to providers, parse responses, classify/aggregate results, and ask
    one member that answered for the session's closing assessment.

    NOTE: The closing assessment costs one model call beyond the fan-out, and
    it runs after them, so the worst-case wall time is two timeouts rather than
    one. A config that turns the assessment off spends neither.

    Args:
        files: Mapping of file paths to source content (required for code-review mode).
        config: Council configuration. Auto-loaded from providers.json when None.
        prompt: The design question (required for design-question mode).
        mode: Council mode, either "code-review" or "design-question".
        context: Optional project-specific context string.
        council_context: Optional prior council round feedback for debate mode.

    Returns:
        CodeReviewResult for code-review mode, DesignQuestionResult for design-question mode.

    Raises:
        ValueError: If required parameters for the chosen mode are missing, or
            if an invalid mode is specified.
        ConfigError: If the config cannot be loaded, or its Otari base URL is malformed or ends in an API path.
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

    # Resolve API keys and Otari env-var references.
    resolved_providers = resolve_api_keys(config.providers)
    resolved_otari = resolve_otari(config.otari)

    # Build the initial prompt.
    if mode == "code-review":
        rendered_prompt = render_code_review_prompt(
            files,
            context=context,
            council_context=council_context,  # type: ignore[arg-type]
        )
    else:
        rendered_prompt = render_design_prompt(prompt, context=context, council_context=council_context)

    # Fan out to all providers.
    exchanges = await fan_out(
        configs=resolved_providers,
        prompt=rendered_prompt,
        timeout=float(config.timeout_seconds),
        otari=resolved_otari,
    )

    async def synthesizer(material: str, answered: Sequence[ProviderConfig]) -> Synthesis:
        """Ask the earliest member that answered for the closing assessment."""
        if not config.synthesis or not answered:
            return Synthesis()
        return await synthesize(
            material,
            answered[0],
            timeout=float(config.timeout_seconds),
            otari=resolved_otari,
        )

    if mode == "code-review":
        return await _build_code_review_result(
            exchanges,
            files_reviewed=tuple(files),  # type: ignore[arg-type]
            threshold=config.consensus_threshold,
            synthesizer=synthesizer,
        )

    return await _build_design_result(
        exchanges,
        prompt=prompt,
        synthesizer=synthesizer,
    )


def run_council_sync(
    *,
    files: dict[str, str] | None = None,
    config: CouncilConfig | None = None,
    prompt: str = "",
    mode: str = "code-review",
    context: str = "",
    council_context: str = "",
) -> CodeReviewResult | DesignQuestionResult:
    """Synchronous wrapper around run_council().

    Args:
        files: Mapping of file paths to source content (required for code-review mode).
        config: Council configuration. Auto-loaded from providers.json when None.
        prompt: The design question (required for design-question mode).
        mode: Council mode, either "code-review" or "design-question".
        context: Optional project-specific context string.
        council_context: Optional prior council round feedback for debate mode.

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
            council_context=council_context,
        )
    )
