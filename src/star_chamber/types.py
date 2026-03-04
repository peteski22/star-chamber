"""Frozen dataclass types for all council protocol wire formats."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderConfig:
    """Configuration for a single LLM provider.

    Attributes:
        provider: Provider identifier (e.g. "openai", "anthropic").
        model: Model name to use with this provider.
        api_key: Optional API key override.
        api_base: Optional base URL override.
        max_tokens: Optional maximum token limit.
        local: Whether this provider runs locally.
    """

    provider: str
    model: str
    api_key: str | None = None
    api_base: str | None = None
    max_tokens: int | None = None
    local: bool = False


@dataclass(frozen=True)
class CouncilConfig:
    """Full council configuration.

    Attributes:
        providers: Tuple of provider configurations.
        timeout_seconds: Per-provider timeout in seconds.
        consensus_threshold: Minimum providers that must agree for consensus.
        platform: Optional platform identifier.
    """

    providers: tuple[ProviderConfig, ...]
    timeout_seconds: int = 60
    consensus_threshold: int = 2
    platform: str | None = None


@dataclass(frozen=True)
class Issue:
    """A single review issue.

    Attributes:
        severity: Issue severity level (e.g. "error", "warning").
        location: File and line location string.
        category: Issue category (e.g. "security", "style", "bug").
        description: Human-readable description of the issue.
        suggestion: Suggested fix or improvement.
    """

    severity: str
    location: str
    category: str
    description: str
    suggestion: str


@dataclass(frozen=True)
class MajorityIssue:
    """An issue flagged by multiple (but not all) providers.

    Attributes:
        severity: Issue severity level.
        location: File and line location string.
        category: Issue category.
        description: Human-readable description of the issue.
        suggestion: Suggested fix or improvement.
        provider_count: Number of providers that flagged this issue.
        flagged_by: Names of providers that flagged this issue.
    """

    severity: str
    location: str
    category: str
    description: str
    suggestion: str
    provider_count: int
    flagged_by: tuple[str, ...]


@dataclass(frozen=True)
class ProviderError:
    """A provider that failed during a council round.

    Attributes:
        provider: Name of the provider that failed.
        error: Error message or description.
    """

    provider: str
    error: str


@dataclass(frozen=True)
class ProviderReview:
    """Parsed review from a single provider.

    Attributes:
        provider: Name of the provider.
        model: Model used for this review.
        quality_rating: Overall quality rating string.
        issues: Tuple of issues found by this provider.
        praise: Tuple of positive observations.
        summary: Provider's summary of the review.
        raw_content: Raw response content from the provider.
    """

    provider: str
    model: str
    quality_rating: str
    issues: tuple[Issue, ...]
    praise: tuple[str, ...]
    summary: str
    raw_content: str


@dataclass(frozen=True)
class Approach:
    """A design approach from a provider.

    Attributes:
        name: Name of the approach.
        recommended_by: Number of providers recommending this approach.
        pros: Tuple of advantages.
        cons: Tuple of disadvantages.
        risk_level: Risk assessment string.
        fit_rating: Optional fitness rating.
    """

    name: str
    recommended_by: int
    pros: tuple[str, ...]
    cons: tuple[str, ...]
    risk_level: str
    fit_rating: str | None = None


@dataclass(frozen=True)
class ProviderDesignAdvice:
    """Parsed design advice from a single provider.

    Attributes:
        provider: Name of the provider.
        model: Model used for this advice.
        recommendation: Provider's recommended approach.
        approaches: Tuple of design approaches considered.
        summary: Provider's summary of the advice.
        raw_content: Raw response content from the provider.
    """

    provider: str
    model: str
    recommendation: str
    approaches: tuple[Approach, ...]
    summary: str
    raw_content: str


@dataclass(frozen=True)
class ClassificationResult:
    """Result of consensus classification.

    Attributes:
        consensus_issues: Issues all providers agree on.
        majority_issues: Issues flagged by most but not all providers.
        individual_issues: Issues keyed by provider name.
    """

    consensus_issues: tuple[Issue, ...]
    majority_issues: tuple[MajorityIssue, ...]
    individual_issues: dict[str, tuple[Issue, ...]]


@dataclass(frozen=True)
class DebateMetadata:
    """Metadata about a debate session.

    Attributes:
        rounds_completed: Number of debate rounds completed.
        converged: Whether the debate reached convergence.
    """

    rounds_completed: int
    converged: bool


@dataclass(frozen=True)
class CodeReviewResult:
    """Final aggregated result from a code review council.

    Attributes:
        mode: Council mode (e.g. "code-review").
        providers_used: Names of providers that participated.
        failed_providers: Providers that failed during the session.
        reviews: Individual provider reviews.
        consensus_issues: Issues all providers agree on.
        majority_issues: Issues flagged by most providers.
        individual_issues: Issues keyed by provider name.
        quality_ratings: Quality ratings keyed by provider name.
        summary: Aggregated summary of all reviews.
        debate: Optional debate session metadata.
    """

    mode: str
    providers_used: tuple[str, ...]
    failed_providers: tuple[ProviderError, ...]
    reviews: tuple[ProviderReview, ...]
    consensus_issues: tuple[Issue, ...]
    majority_issues: tuple[MajorityIssue, ...]
    individual_issues: dict[str, tuple[Issue, ...]]
    quality_ratings: dict[str, str]
    summary: str
    debate: DebateMetadata | None = None


@dataclass(frozen=True)
class DesignQuestionResult:
    """Final aggregated result from a design question council.

    Attributes:
        mode: Council mode (e.g. "design-question").
        prompt: The original design question.
        providers_used: Names of providers that participated.
        failed_providers: Providers that failed during the session.
        approaches: Aggregated design approaches.
        consensus_recommendation: Recommendation all providers agreed on.
        summary: Aggregated summary of design advice.
        debate: Optional debate session metadata.
    """

    mode: str
    prompt: str
    providers_used: tuple[str, ...]
    failed_providers: tuple[ProviderError, ...]
    approaches: tuple[Approach, ...]
    consensus_recommendation: str | None = None
    summary: str = ""
    debate: DebateMetadata | None = None
