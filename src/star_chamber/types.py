"""Frozen dataclass types and closed-set vocabularies for the council protocol wire formats."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from star_chamber.spec import SPEC_VERSION


class Ranked(StrEnum):
    """A closed set whose declaration order carries meaning."""

    @property
    def rank(self) -> int:
        """Position in declaration order, so an ordered set needs no table beside it.

        NOTE: A set holds at most four members, so the scan costs less than caching it would.
        """
        return list(type(self)).index(self)


class Severity(Ranked):
    """How urgent an issue is. Declared most urgent first, which is the order issues sort in."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class IssueCategory(StrEnum):
    """What kind of concern an issue raises."""

    ARCHITECTURE = "architecture"
    CORRECTNESS = "correctness"
    CRAFTSMANSHIP = "craftsmanship"
    INVARIANTS = "invariants"
    MAINTAINABILITY = "maintainability"


class QualityRating(Ranked):
    """A member's verdict on the code it reviewed. Declared worst first, which is the order a tally reads in."""

    NEEDS_WORK = "needs-work"
    FAIR = "fair"
    GOOD = "good"
    EXCELLENT = "excellent"


class RiskLevel(Ranked):
    """How risky an approach is. Declared most cautious first, which is how a tie resolves.

    NOTE: UNKNOWN is the council's own value for an approach no member rated.
    A member may not send it.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"

    @property
    def stated(self) -> bool:
        """Whether a member stated this level, which UNKNOWN reports that none did."""
        return self is not RiskLevel.UNKNOWN


class FitRating(Ranked):
    """How well an approach fits the project. Declared most cautious first, which is how a tie resolves."""

    POOR = "poor"
    FAIR = "fair"
    GOOD = "good"
    EXCELLENT = "excellent"


@dataclass(frozen=True)
class ProviderConfig:
    """Configuration for a single LLM provider.

    Attributes:
        provider: Provider identifier (e.g. "openai", "anthropic").
        model: Model name to use with this provider.
        api_key: Optional API key override.
        api_base: Optional base URL override.
        max_tokens: Optional budget in tokens for this provider's reply, rather
            than a capability read from the model.
        local: Whether this provider runs locally.
        display_name: Optional unique identity for council output, used in
            place of the provider name. Lets multiple entries share one
            provider (e.g. several models behind an OpenAI-compatible gateway)
            while remaining distinct voters in consensus classification,
            quality ratings, and individual-issue buckets. When unset, output
            falls back to the provider name.
    """

    provider: str
    model: str
    api_key: str | None = None
    api_base: str | None = None
    max_tokens: int | None = None
    local: bool = False
    display_name: str | None = None

    @property
    def identity(self) -> str:
        """Return the name this member is known by in council output."""
        return self.display_name or self.provider


@dataclass(frozen=True)
class OtariConfig:
    """Optional Otari gateway routing.

    When set on a CouncilConfig, non-local providers are dispatched through
    Otari, an OpenAI-compatible gateway. The same api_base and api_key apply to
    all routed providers.

    Attributes:
        api_base: Otari gateway origin with no path, or ${ENV_VAR} reference.
            When None, the SDK's OtariProvider resolves it from its own env vars.
        api_key: Otari API key or ${ENV_VAR} reference. When None, the
            SDK's OtariProvider auto-detects credentials from its own env
            vars (OTARI_API_KEY for platform mode, GATEWAY_API_KEY for
            self-hosted mode).
    """

    api_base: str | None = None
    api_key: str | None = None


@dataclass(frozen=True)
class CouncilConfig:
    """Full council configuration.

    Attributes:
        providers: Tuple of provider configurations.
        timeout_seconds: Per-provider timeout in seconds.
        consensus_threshold: Minimum providers that must agree for consensus.
        otari: Optional otari routing for non-local providers.
        synthesis: Whether to ask a member for the session's closing
            assessment. False spends no call on one and leaves the summary
            saying none was requested.
    """

    providers: tuple[ProviderConfig, ...]
    timeout_seconds: int = 60
    consensus_threshold: int = 2
    otari: OtariConfig | None = None
    synthesis: bool = True


def _require_member(value: object, allowed: type[StrEnum], field: str) -> None:
    """Reject a value the field's closed set does not hold.

    Args:
        value: What the caller supplied.
        allowed: The set the field declares.
        field: Field name, for the error message.

    Raises:
        TypeError: If the value is not one of the set's members.
    """
    if not isinstance(value, allowed):
        msg = f"{field} must be a {allowed.__name__}, got {type(value).__name__}"
        raise TypeError(msg)


@dataclass(frozen=True)
class Issue:
    """A single review issue.

    Attributes:
        severity: How urgent the issue is.
        location: File and line location string.
        category: What kind of concern the issue raises.
        description: Human-readable description of the issue.
        suggestion: Suggested fix or improvement.
    """

    severity: Severity
    location: str
    category: IssueCategory
    description: str
    suggestion: str

    def __post_init__(self) -> None:
        """Reject a value the attributes above say the field cannot hold."""
        _require_member(self.severity, Severity, "severity")
        _require_member(self.category, IssueCategory, "category")


@dataclass(frozen=True)
class MajorityIssue:
    """An issue flagged by multiple (but not all) providers.

    Attributes:
        severity: How urgent the issue is.
        location: File and line location string.
        category: What kind of concern the issue raises.
        description: Human-readable description of the issue.
        suggestion: Suggested fix or improvement.
        provider_count: Number of providers that flagged this issue.
        flagged_by: Names of providers that flagged this issue.
    """

    severity: Severity
    location: str
    category: IssueCategory
    description: str
    suggestion: str
    provider_count: int
    flagged_by: tuple[str, ...]

    def __post_init__(self) -> None:
        """Reject a value the attributes above say the field cannot hold."""
        _require_member(self.severity, Severity, "severity")
        _require_member(self.category, IssueCategory, "category")


@dataclass(frozen=True)
class ProviderError:
    """A provider that failed during a council round.

    Attributes:
        provider: Name of the provider that failed.
        model: Model that was asked, which the provider name does not give when
            several members share one provider.
        error: Error message or description.
        raw_content: What the provider sent, when it sent anything, so the
            failure can be diagnosed from the reply rather than from its
            description alone. None when the call produced no reply at all.
            Empty when a reply arrived carrying no content.

    NOTE: raw_content has no default. A site that failed to state it would
    claim no reply arrived, which is the evidence the field exists to keep.
    """

    provider: str
    model: str
    error: str
    raw_content: str | None


@dataclass(frozen=True)
class ProviderReview:
    """Parsed review from a single provider.

    Attributes:
        provider: Name of the provider.
        model: Model used for this review.
        quality_rating: This member's verdict on the code it reviewed.
        issues: Tuple of issues found by this provider.
        praise: Tuple of positive observations.
        summary: Provider's summary of the review.
        raw_content: Raw response content from the provider.
    """

    provider: str
    model: str
    quality_rating: QualityRating
    issues: tuple[Issue, ...]
    praise: tuple[str, ...]
    summary: str
    raw_content: str

    def __post_init__(self) -> None:
        """Reject a value the attributes above say the field cannot hold."""
        _require_member(self.quality_rating, QualityRating, "quality_rating")


@dataclass(frozen=True)
class Approach:
    """A design approach from a provider.

    Attributes:
        name: Name of the approach.
        recommended_by: Number of providers recommending this approach.
        pros: Tuple of advantages.
        cons: Tuple of disadvantages.
        risk_level: How risky the approach is.
        fit_rating: How well the approach fits, or None where none was stated.

    NOTE: fit_rating has no default. A site that failed to state it would claim
    nobody rated the approach, which is what a null here reports.
    """

    name: str
    recommended_by: int
    pros: tuple[str, ...]
    cons: tuple[str, ...]
    risk_level: RiskLevel
    fit_rating: FitRating | None

    def __post_init__(self) -> None:
        """Reject a value the attributes above say the field cannot hold."""
        _require_member(self.risk_level, RiskLevel, "risk_level")
        if self.fit_rating is not None:
            _require_member(self.fit_rating, FitRating, "fit_rating")


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
class CodeReviewSummary:
    """Counts and prose summarizing a code review council session.

    Attributes:
        total_issues: Number of distinct issues found, across every bucket.
        consensus_count: Number of issues every member that answered flagged.
        majority_count: Number of issues more than one but not all of them flagged.
        synthesis: Overall assessment in prose. Empty when no member wrote one.
        synthesis_error: The member asked for the assessment and why it wrote
            none. Null when the assessment is present, and null when no member
            could be asked.
        synthesized_by: The member asked for the assessment, whether or not it
            wrote one. Null when no member was asked.
    """

    total_issues: int
    consensus_count: int
    majority_count: int
    synthesis: str
    synthesis_error: ProviderError | None = None
    synthesized_by: str | None = None


@dataclass(frozen=True)
class DesignSummary:
    """Prose summarizing a design question council session.

    Attributes:
        synthesis: Overall assessment in prose. Empty when no member wrote one.
        synthesis_error: The member asked for the assessment and why it wrote
            none. Null when the assessment is present, and null when no member
            could be asked.
        synthesized_by: The member asked for the assessment, whether or not it
            wrote one. Null when no member was asked.
    """

    synthesis: str
    synthesis_error: ProviderError | None = None
    synthesized_by: str | None = None


@dataclass(frozen=True)
class CodeReviewResult:
    """Final aggregated result from a code review council.

    Attributes:
        mode: Council mode (e.g. "code-review").
        files_reviewed: Paths of the files sent for review.
        providers_used: Names of every council member consulted, including those that failed.
        providers_responded: Names of the members that returned a usable review.
            Consensus, majority, and individual classifications were computed
            over these members alone.
        failed_providers: Providers that failed during the session.
        reviews: Individual provider reviews.
        consensus_issues: Issues all providers agree on.
        majority_issues: Issues flagged by most providers.
        individual_issues: Issues keyed by provider name.
        quality_ratings: Each member's verdict, keyed by provider name.
        summary: Counts and prose for the session.
        spec_version: The protocol version that produced this document, for a
            consumer to compare against the version its instructions were
            written against. Not a constructor argument, because a document
            states the version its producer implements and no other.
    """

    mode: str
    files_reviewed: tuple[str, ...]
    providers_used: tuple[str, ...]
    providers_responded: tuple[str, ...]
    failed_providers: tuple[ProviderError, ...]
    reviews: tuple[ProviderReview, ...]
    consensus_issues: tuple[Issue, ...]
    majority_issues: tuple[MajorityIssue, ...]
    individual_issues: dict[str, tuple[Issue, ...]]
    quality_ratings: dict[str, QualityRating]
    summary: CodeReviewSummary
    spec_version: str = field(default=SPEC_VERSION, init=False)

    def __post_init__(self) -> None:
        """Reject a value the attributes above say the field cannot hold."""
        for member, rating in self.quality_ratings.items():
            _require_member(rating, QualityRating, f"quality_ratings[{member!r}]")


@dataclass(frozen=True)
class DesignQuestionResult:
    """Final aggregated result from a design question council.

    Attributes:
        mode: Council mode (e.g. "design-question").
        prompt: The original design question.
        providers_used: Names of every council member consulted, including those that failed.
        providers_responded: Names of the members that returned usable advice.
            The consensus recommendation was computed over these members alone.
        failed_providers: Providers that failed during the session.
        approaches: Aggregated design approaches.
        consensus_recommendation: The approach a majority of the providers
            that answered recommended. None when no approach held one.
        summary: Prose for the session.
        spec_version: The protocol version that produced this document, for a
            consumer to compare against the version its instructions were
            written against. Not a constructor argument, because a document
            states the version its producer implements and no other.
    """

    mode: str
    prompt: str
    providers_used: tuple[str, ...]
    providers_responded: tuple[str, ...]
    failed_providers: tuple[ProviderError, ...]
    approaches: tuple[Approach, ...]
    summary: DesignSummary
    consensus_recommendation: str | None = None
    spec_version: str = field(default=SPEC_VERSION, init=False)
