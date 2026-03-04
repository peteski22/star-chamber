"""Star-chamber: Multi-LLM council protocol SDK."""

from importlib.metadata import PackageNotFoundError, version

from star_chamber.council import run_council, run_council_sync
from star_chamber.types import (
    CodeReviewResult,
    CouncilConfig,
    DesignQuestionResult,
    Issue,
    ProviderConfig,
    ProviderError,
    ProviderReview,
)

try:
    __version__ = version("star-chamber")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

__all__ = [
    "CouncilConfig",
    "CodeReviewResult",
    "DesignQuestionResult",
    "Issue",
    "ProviderConfig",
    "ProviderError",
    "ProviderReview",
    "__version__",
    "run_council",
    "run_council_sync",
]
