"""Consensus classification for multi-provider code review issues.

Implements the grouping and classification algorithm from spec section 9.
Issues from multiple providers are grouped by file, line proximity, and
category, then classified as consensus, majority, or individual.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence

from star_chamber.types import ClassificationResult, Issue, MajorityIssue, ProviderReview

# Line numbers within this tolerance are considered the same location.
_LINE_TOLERANCE = 5

# Matches a location's trailing line number or line range, e.g. "23" or "203-207".
_LINE_SPAN_RE = re.compile(r"(\d+)(?:-(\d+))?$")

# An inclusive first and last line.  A single line is a span of one.
_LineSpan = tuple[int, int]

# A grouping key: file path, line span, and category.
_IssueKey = tuple[str, _LineSpan | None, str]

# Sort order for severity levels (lower value = higher priority).
_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def _parse_location(location: str) -> tuple[str, _LineSpan | None]:
    """Parse a location string into a file path and an optional line span.

    Accepts a single line ("file.py:42"), a line range ("file.py:203-207"), or
    a bare path ("file.py").  A single line yields a span of one line, so one
    comparison covers both forms.  A trailing segment that is neither leaves the
    location whole, which keeps a path containing a colon intact.

    Args:
        location: Location string such as "path/to/file.py:42" or "file.py:203-207".

    Returns:
        Tuple of (file_path, line_span), where line_span is None if absent.
    """
    idx = location.rfind(":")
    if idx == -1:
        return (location, None)

    match = _LINE_SPAN_RE.fullmatch(location[idx + 1 :])
    if match is None:
        return (location, None)

    first = int(match.group(1))
    last = first if match.group(2) is None else int(match.group(2))
    return (location[:idx], (min(first, last), max(first, last)))


def _issue_key(issue: Issue) -> _IssueKey:
    """Build a grouping key from an issue.

    Args:
        issue: The issue to extract a key from.

    Returns:
        Tuple of (file, line_span, category).
    """
    file, span = _parse_location(issue.location)
    return (file, span, issue.category)


def _spans_match(span_a: _LineSpan, span_b: _LineSpan) -> bool:
    """Determine whether two line spans refer to the same place.

    Spans match when they overlap, or when the gap between them is within
    _LINE_TOLERANCE.  A single line is a span of one, so two single lines match
    on the same tolerance that separated them before spans existed.

    Args:
        span_a: First inclusive line span.
        span_b: Second inclusive line span.

    Returns:
        True if the spans refer to the same place.
    """
    first_a, last_a = span_a
    first_b, last_b = span_b
    return first_a - _LINE_TOLERANCE <= last_b and first_b - _LINE_TOLERANCE <= last_a


def _issues_match(key_a: _IssueKey, key_b: _IssueKey) -> bool:
    """Determine whether two issue keys refer to the same logical issue.

    Two keys match when they share the same file and category, and their line
    spans refer to the same place.  Both spans must be present for that
    comparison; if both are absent they match, but if only one is absent they
    do not.

    Args:
        key_a: First issue key.
        key_b: Second issue key.

    Returns:
        True if the keys represent the same logical issue.
    """
    file_a, span_a, cat_a = key_a
    file_b, span_b, cat_b = key_b

    if file_a != file_b or cat_a != cat_b:
        return False

    # Both absent => match.  One absent => no match.
    if span_a is None and span_b is None:
        return True
    if span_a is None or span_b is None:
        return False

    return _spans_match(span_a, span_b)


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
    groups: list[tuple[Issue, _IssueKey, set[str]]] = []

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
