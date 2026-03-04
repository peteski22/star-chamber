"""Consensus classification for multi-provider code review issues.

Implements the grouping and classification algorithm from spec section 9.
Issues from multiple providers are grouped by file, line proximity, and
category, then classified as consensus, majority, or individual.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from star_chamber.types import ClassificationResult, Issue, MajorityIssue, ProviderReview

# Line numbers within this tolerance are considered the same location.
_LINE_TOLERANCE = 5

# Sort order for severity levels (lower value = higher priority).
_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def _parse_location(location: str) -> tuple[str, int | None]:
    """Parse a "file:line" location string into file path and optional line number.

    Args:
        location: Location string in the form "path/to/file.py:42" or "file.py".

    Returns:
        Tuple of (file_path, line_number) where line_number is None if absent.
    """
    idx = location.rfind(":")
    if idx == -1:
        return (location, None)
    file_part = location[:idx]
    line_part = location[idx + 1 :]
    try:
        return (file_part, int(line_part))
    except ValueError:
        return (location, None)


def _issue_key(issue: Issue) -> tuple[str, int | None, str]:
    """Build a grouping key from an issue.

    Args:
        issue: The issue to extract a key from.

    Returns:
        Tuple of (file, line_number, category).
    """
    file, line = _parse_location(issue.location)
    return (file, line, issue.category)


def _issues_match(key_a: tuple[str, int | None, str], key_b: tuple[str, int | None, str]) -> bool:
    """Determine whether two issue keys refer to the same logical issue.

    Two keys match when they share the same file and category, and their line
    numbers are within +-_LINE_TOLERANCE of each other.  Both lines must be
    present (non-None) for tolerance comparison; if both are None they match,
    but if only one is None they do not.

    Args:
        key_a: First issue key.
        key_b: Second issue key.

    Returns:
        True if the keys represent the same logical issue.
    """
    file_a, line_a, cat_a = key_a
    file_b, line_b, cat_b = key_b

    if file_a != file_b or cat_a != cat_b:
        return False

    # Both None => match.  One None => no match.
    if line_a is None and line_b is None:
        return True
    if line_a is None or line_b is None:
        return False

    return abs(line_a - line_b) <= _LINE_TOLERANCE


def _severity_sort_key(issue: Issue | MajorityIssue) -> int:
    """Return a numeric sort key for an issue's severity.

    Args:
        issue: An Issue or MajorityIssue to sort.

    Returns:
        Integer sort value where lower means higher severity.
    """
    return _SEVERITY_ORDER.get(issue.severity, 999)


def classify(reviews: Sequence[ProviderReview], threshold: int = 2) -> ClassificationResult:
    """Classify issues from multiple providers into consensus, majority, and individual buckets.

    Args:
        reviews: Sequence of provider reviews to classify.
        threshold: Minimum number of providers that must agree for consensus.

    Returns:
        ClassificationResult with consensus, majority, and individual issues.
    """
    num_providers = len(reviews)
    if num_providers == 0:
        return ClassificationResult(
            consensus_issues=(),
            majority_issues=(),
            individual_issues={},
        )

    # Clamp threshold so it never exceeds the number of providers.
    effective_threshold = min(threshold, num_providers)

    # Each group tracks: the representative issue, its key, and the set of providers.
    groups: list[tuple[Issue, tuple[str, int | None, str], set[str]]] = []

    for review in reviews:
        for issue in review.issues:
            key = _issue_key(issue)
            matched = False
            for group in groups:
                _, group_key, providers = group
                if _issues_match(key, group_key) and review.provider not in providers:
                    providers.add(review.provider)
                    matched = True
                    break
            if not matched:
                groups.append((issue, key, {review.provider}))

    # Classify each group.
    consensus: list[Issue] = []
    majority: list[MajorityIssue] = []
    individual_map: dict[str, list[Issue]] = defaultdict(list)

    for representative, _key, providers in groups:
        count = len(providers)
        all_agree = count == num_providers
        meets_threshold = count >= effective_threshold

        if all_agree and meets_threshold:
            consensus.append(representative)
        elif count >= 2:
            majority.append(
                MajorityIssue(
                    severity=representative.severity,
                    location=representative.location,
                    category=representative.category,
                    description=representative.description,
                    suggestion=representative.suggestion,
                    provider_count=count,
                    flagged_by=tuple(sorted(providers)),
                )
            )
        else:
            # Exactly one provider.
            provider_name = next(iter(providers))
            individual_map[provider_name].append(representative)

    # Sort each bucket by severity.
    consensus.sort(key=_severity_sort_key)
    majority.sort(key=_severity_sort_key)
    for issues in individual_map.values():
        issues.sort(key=_severity_sort_key)

    return ClassificationResult(
        consensus_issues=tuple(consensus),
        majority_issues=tuple(majority),
        individual_issues={k: tuple(v) for k, v in individual_map.items()},
    )
