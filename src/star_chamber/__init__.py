"""Star-chamber: Multi-LLM council protocol SDK."""

from importlib.metadata import PackageNotFoundError, version

from star_chamber.council import run_council, run_council_sync
from star_chamber.schema import get_schema, list_schemas, strict_variant
from star_chamber.spec import SPEC_VERSION
from star_chamber.types import (
    CodeReviewResult,
    CouncilConfig,
    DesignQuestionResult,
    Finding,
    FitRating,
    Issue,
    IssueCategory,
    IssueGroup,
    OtariConfig,
    ProviderConfig,
    ProviderError,
    ProviderReview,
    QualityRating,
    RiskLevel,
    Severity,
)

try:
    __version__ = version("star-chamber")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

__all__ = [
    "CodeReviewResult",
    "CouncilConfig",
    "DesignQuestionResult",
    "Finding",
    "FitRating",
    "Issue",
    "IssueCategory",
    "IssueGroup",
    "OtariConfig",
    "ProviderConfig",
    "ProviderError",
    "ProviderReview",
    "QualityRating",
    "RiskLevel",
    "SPEC_VERSION",
    "Severity",
    "__version__",
    "get_schema",
    "list_schemas",
    "run_council",
    "run_council_sync",
    "strict_variant",
]
