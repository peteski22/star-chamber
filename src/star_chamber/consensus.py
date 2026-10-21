"""Consensus algorithms that combine what several providers reported.

Implements the grouping and classification algorithm from spec section 9.
Issues from multiple providers are grouped by file, line proximity, and
category, then classified as consensus, majority, or individual.
Design approaches are grouped by name, and the count of providers that
recommended each one is derived from the recommendation each provider gave.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence

from star_chamber.types import (
    Approach,
    ClassificationResult,
    Issue,
    MajorityIssue,
    ProviderDesignAdvice,
    ProviderReview,
)

# Line numbers within this tolerance are considered the same location.
_LINE_TOLERANCE = 5

# Matches a location's trailing line number or line range, e.g. "23" or "203-207".
# The digit runs are bounded so the pattern cannot accept a number int() would reject.
_LINE_SPAN_RE = re.compile(r"(\d{1,7})(?:-(\d{1,7}))?$", re.ASCII)

# The most lines a span may cover, counting both ends, and still mean one place.
# A wider span cites a file-level concern.
# Grouping such a span by overlap would let it absorb unrelated findings, whose
# descriptions the group then discards, so it is treated as no span instead.
_MAX_SPAN_LINES = 50

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

    A span wider than _MAX_SPAN_LINES yields the file path and no span, because
    it cites the file rather than one place within it.

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
    first, last = min(first, last), max(first, last)

    if last - first + 1 > _MAX_SPAN_LINES:
        return (location[:idx], None)

    return (location[:idx], (first, last))


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


# Characters outside a name's words, so a name matches regardless of case and punctuation.
_NON_WORD_RE = re.compile(r"[^a-z0-9]+", re.ASCII)

# Symbols that tell one approach name from another, spelled out so that stripping
# punctuation cannot merge names such as C++, C# and C.
_NAME_SYMBOL_WORDS = {"#": " sharp ", "+": " plus "}

# Fit ratings, most cautious first, so a tie resolves to the more cautious rating.
_FIT_RATINGS_BY_CAUTION = ("poor", "fair", "good", "excellent")

# Risk levels, most cautious first, so a tie resolves to the more cautious level.
# "unknown" is absent because it means no provider stated a level, not that one was chosen.
_RISK_LEVELS_BY_CAUTION = ("high", "medium", "low")


def _approach_name_key(name: str) -> str:
    """Reduce an approach name to the form two providers must share to mean the same approach."""
    text = name.casefold()
    for symbol, word in _NAME_SYMBOL_WORDS.items():
        text = text.replace(symbol, word)
    return _NON_WORD_RE.sub(" ", text).strip()


def _own_name_keys(advice: ProviderDesignAdvice) -> dict[str, None]:
    """Return a key for each approach this provider named, in the order it named them."""
    keys = (_approach_name_key(approach.name) for approach in advice.approaches)
    return {key: None for key in keys if key}


def _names_mentioned_in(text: str, name_keys: Iterable[str]) -> set[str]:
    """Return the approach names the text names, matching whole words only."""
    haystack = f" {_approach_name_key(text)} "
    return {key for key in name_keys if f" {key} " in haystack}


def _most_cautious_of_the_most_common(values: Sequence[str], by_caution: Sequence[str]) -> str | None:
    """Return the value the most providers stated, resolving a tie toward the more cautious value.

    Args:
        values: What each provider stated, including repeats.
        by_caution: The permitted values, most cautious first.

    Returns:
        The winning value, or None when no provider stated a permitted one.
    """
    counts = Counter(value for value in values if value in by_caution)
    if not counts:
        return None
    most = max(counts.values())
    return next(value for value in by_caution if counts.get(value) == most)


def aggregate_approaches(advices: Sequence[ProviderDesignAdvice]) -> tuple[Approach, ...]:
    """Group the approaches every provider named, and count who recommended each one.

    A provider cannot report how many providers recommended an approach, so the
    count is derived from the recommendation each provider gave. A provider is
    counted only for the approaches it named itself, and votes once per approach
    on the aggregated risk level and fit rating.

    Args:
        advices: Parsed advice from every provider that answered.

    Returns:
        One approach per distinct name, ordered as the providers first named them.
    """
    display_names: dict[str, str] = {}
    pros: dict[str, dict[str, None]] = {}
    cons: dict[str, dict[str, None]] = {}
    risk_levels: dict[str, list[str]] = defaultdict(list)
    fit_ratings: dict[str, list[str]] = defaultdict(list)
    recommended_by = Counter[str]()

    for advice in advices:
        stated_risks: dict[str, list[str]] = defaultdict(list)
        stated_fits: dict[str, list[str]] = defaultdict(list)

        for approach in advice.approaches:
            key = _approach_name_key(approach.name)
            if not key:
                continue
            display_names.setdefault(key, approach.name)
            pros.setdefault(key, {}).update(dict.fromkeys(approach.pros))
            cons.setdefault(key, {}).update(dict.fromkeys(approach.cons))
            stated_risks[key].append(approach.risk_level)
            if approach.fit_rating is not None:
                stated_fits[key].append(approach.fit_rating)

        # One ballot per provider per approach, so naming an approach twice does not vote twice.
        for key, levels in stated_risks.items():
            level = _most_cautious_of_the_most_common(levels, _RISK_LEVELS_BY_CAUTION)
            if level is not None:
                risk_levels[key].append(level)
        for key, ratings in stated_fits.items():
            rating = _most_cautious_of_the_most_common(ratings, _FIT_RATINGS_BY_CAUTION)
            if rating is not None:
                fit_ratings[key].append(rating)

        recommended_by.update(_names_mentioned_in(advice.recommendation, _own_name_keys(advice)))

    return tuple(
        Approach(
            name=display_names[key],
            recommended_by=recommended_by[key],
            pros=tuple(pros[key]),
            cons=tuple(cons[key]),
            risk_level=_most_cautious_of_the_most_common(risk_levels[key], _RISK_LEVELS_BY_CAUTION) or "unknown",
            fit_rating=_most_cautious_of_the_most_common(fit_ratings[key], _FIT_RATINGS_BY_CAUTION),
        )
        for key in display_names
    )


def consensus_recommendation(
    advices: Sequence[ProviderDesignAdvice],
    approaches: Sequence[Approach],
) -> str | None:
    """Report the one direction every provider that answered recommended.

    Args:
        advices: Parsed advice from every provider that answered.
        approaches: The grouped approaches those providers named.

    Returns:
        The shared recommendation, or None when the providers named no single direction.

    NOTE: Providers that recommend different variants of one direction yield None,
    because recognizing the shared direction behind differing prose needs a synthesis step.
    """
    if not advices:
        return None

    recommendations = [advice.recommendation for advice in advices]
    if all(text == recommendations[0] for text in recommendations):
        return recommendations[0]

    display_names = {_approach_name_key(approach.name): approach.name for approach in approaches}
    picks = [_names_mentioned_in(advice.recommendation, _own_name_keys(advice)) for advice in advices]
    if any(len(pick) != 1 for pick in picks):
        return None
    shared = {next(iter(pick)) for pick in picks}
    if len(shared) != 1:
        return None
    return display_names[next(iter(shared))]
