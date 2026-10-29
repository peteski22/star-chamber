"""Consensus algorithms that combine what several providers reported.

Implements the grouping and classification algorithm from spec section 9.
Issues from multiple providers are grouped by file and line proximity, each
group keeping every provider's own account, then classified as consensus,
majority, or individual.
Design approaches are grouped by name, and the count of providers that
recommended each one is derived from the recommendation each provider gave.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from typing import TypeVar

from star_chamber.types import (
    Approach,
    ClassificationResult,
    Finding,
    FitRating,
    IssueGroup,
    ProviderDesignAdvice,
    ProviderReview,
    Ranked,
    RiskLevel,
)

_RankedT = TypeVar("_RankedT", bound=Ranked)

# Line numbers within this tolerance are considered the same location.
_LINE_TOLERANCE = 5

# Matches a location's trailing line number or line range, e.g. "23" or "203-207".
# The digit runs are bounded so the pattern cannot accept a number int() would reject.
_LINE_SPAN_RE = re.compile(r"(\d{1,7})(?:-(\d{1,7}))?$", re.ASCII)

# The most lines a span may cover, counting both ends, and still mean one place.
# A wider span cites the file rather than a place within it, so it is treated as
# no span and placed the way a bare file path is.
_MAX_SPAN_LINES = 50

# An inclusive first and last line.  A single line is a span of one.
_LineSpan = tuple[int, int]

# A cited place: file path and an optional line span.
_Location = tuple[str, _LineSpan | None]


def _parse_location(location: str) -> _Location:
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


def _locations_match(place_a: _Location, place_b: _Location) -> bool:
    """Determine whether two cited places refer to the same place.

    Two places match when they name the same file and their line spans refer
    to the same place.  A citation carrying no span names the file as a whole,
    so it matches every span within that file.

    The category a member chose does not take part, because two members can
    read one defect as two kinds of concern, and a group carries every category
    its findings name.

    Args:
        place_a: First cited place.
        place_b: Second cited place.

    Returns:
        True if the places refer to the same place.
    """
    file_a, span_a = place_a
    file_b, span_b = place_b

    if file_a != file_b:
        return False

    if span_a is None or span_b is None:
        return True

    return _spans_match(span_a, span_b)


def _is_file_level(place: _Location) -> bool:
    """Determine whether a citation names a file rather than a place within it."""
    return place[1] is None


def _place(groups: list[tuple[_Location, list[Finding]]], place: _Location, finding: Finding) -> None:
    """Add a finding to the first group whose place matches, or open a group for it.

    Args:
        groups: The groups collected so far, which this may extend.
        place: The place the finding cited.
        finding: The finding to place.
    """
    for group_place, findings in groups:
        if _locations_match(place, group_place):
            findings.append(finding)
            return
    groups.append((place, [finding]))


def _severity_sort_key(group: IssueGroup) -> int:
    """Return a numeric sort key for a grouped issue's severity.

    Args:
        group: A grouped issue to sort.

    Returns:
        Integer sort value where lower means higher severity.
    """
    return group.severity.rank


def _bucketed(groups: Sequence[IssueGroup], num_providers: int, threshold: int) -> ClassificationResult:
    """Sort each grouped issue into the bucket its number of contributing providers names.

    Args:
        groups: The grouped issues to sort.
        num_providers: How many providers answered, all of which consensus needs.
        threshold: Minimum providers that must agree, clamped to that roster.

    Returns:
        ClassificationResult holding each group in one bucket, ordered by severity.
    """
    effective_threshold = min(threshold, num_providers)
    consensus: list[IssueGroup] = []
    majority: list[IssueGroup] = []
    individual_map: dict[str, list[IssueGroup]] = defaultdict(list)

    for group in groups:
        count = len(group.flagged_by)
        all_agree = count == num_providers
        meets_threshold = count >= effective_threshold

        if all_agree and meets_threshold:
            consensus.append(group)
        elif count >= 2:
            majority.append(group)
        else:
            # Exactly one provider.
            individual_map[group.flagged_by[0]].append(group)

    consensus.sort(key=_severity_sort_key)
    majority.sort(key=_severity_sort_key)
    for issues in individual_map.values():
        issues.sort(key=_severity_sort_key)

    return ClassificationResult(
        consensus_issues=tuple(consensus),
        majority_issues=tuple(majority),
        individual_issues={provider: tuple(individual_map[provider]) for provider in sorted(individual_map)},
    )


def classify(reviews: Sequence[ProviderReview], threshold: int = 2) -> ClassificationResult:
    """Classify issues from multiple providers into consensus, majority, and individual buckets.

    Every issue joins the first group whose place matches its own, and each
    group keeps the account its contributor gave, so a group that draws several
    providers reports all of them rather than one on behalf of the rest.

    A citation naming a file rather than a place within it joins one group of
    that file rather than merging the places the providers raised separately.

    Args:
        reviews: Sequence of provider reviews to classify.
        threshold: Minimum number of providers that must agree for consensus.

    Returns:
        ClassificationResult with consensus, majority, and individual issues.

    NOTE: Matching is not transitive, because three citations can overlap in
    pairs without naming one place.  Where that happens, the order the reviews
    arrive in decides which group a citation joins.
    """
    num_providers = len(reviews)
    if num_providers == 0:
        return ClassificationResult(
            consensus_issues=(),
            majority_issues=(),
            individual_issues={},
        )

    # A group keeps the place that opened it, rather than widening to cover what joined.
    groups: list[tuple[_Location, list[Finding]]] = []
    file_level: list[tuple[_Location, Finding]] = []

    for review in reviews:
        for issue in review.issues:
            place = _parse_location(issue.location)
            finding = Finding(
                provider=review.provider,
                severity=issue.severity,
                location=issue.location,
                category=issue.category,
                description=issue.description,
                suggestion=issue.suggestion,
            )
            if _is_file_level(place):
                file_level.append((place, finding))
            else:
                _place(groups, place, finding)

    # A file-level citation matches every place in its file, so placing it first
    # would collect the whole file into whichever group it happened to open.
    for place, finding in file_level:
        _place(groups, place, finding)

    return _bucketed([IssueGroup(findings=tuple(findings)) for _, findings in groups], num_providers, threshold)


# Characters outside a name's words, so a name matches regardless of case and punctuation.
# The pattern is Unicode aware, because an ASCII one empties the key of a name written
# in another script and the approach then disappears from the result.
_NON_WORD_RE = re.compile(r"[\W_]+")

# Symbols that tell one approach name from another, spelled out so that stripping
# punctuation cannot merge names such as C++, C# and C.
# NOTE: The spelling carries no spaces, so a symbol joined to a word stays part of that
# word. Expanding it into a word of its own would let "C" match a recommendation of "C++".
_NAME_SYMBOL_WORDS = {"#": "sharp", "+": "plus"}


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


def _word_spans(words: list[str], phrase: list[str]) -> list[tuple[int, int]]:
    """Return the start and stop word offsets of every whole-word run of the phrase."""
    width = len(phrase)
    return [(start, start + width) for start in range(len(words) - width + 1) if words[start : start + width] == phrase]


def _names_mentioned_in(text: str, name_keys: Iterable[str]) -> set[str]:
    """Return the approach names the text names, matching whole words only.

    NOTE: A name counts only where it stands on its own. One that appears solely
    inside a longer name the text also names was never named in its own right.
    """
    words = _approach_name_key(text).split()
    phrases = {key: key.split() for key in name_keys if key}
    spans = {key: _word_spans(words, phrase) for key, phrase in phrases.items()}

    named = set()
    for key, occurrences in spans.items():
        swallowing = [
            span for other, found in spans.items() if len(phrases[other]) > len(phrases[key]) for span in found
        ]
        if any(
            all(not (outer_start <= start and stop <= outer_stop) for outer_start, outer_stop in swallowing)
            for start, stop in occurrences
        ):
            named.add(key)
    return named


def _approaches_recommended_by(advice: ProviderDesignAdvice, name_keys: Iterable[str]) -> set[str]:
    """Return the keys of the approaches one provider's recommendation chose.

    Args:
        advice: Parsed advice from one provider.
        name_keys: Every grouped approach name, so that a longer name any provider
            raised can rule out a shorter one.

    Returns:
        The keys this provider both named itself and then recommended.
    """
    return _names_mentioned_in(advice.recommendation, name_keys) & _own_name_keys(advice).keys()


def _most_cautious_of_the_most_common(values: Sequence[_RankedT]) -> _RankedT | None:
    """Return the value the most providers stated, resolving a tie toward the more cautious value.

    Args:
        values: What each provider stated, including repeats.

    Returns:
        The winning value, or None when no provider stated one.
    """
    counts = Counter(values)
    if not counts:
        return None
    most = max(counts.values())
    return min((value for value, count in counts.items() if count == most), key=lambda value: value.rank)


def aggregate_approaches(advices: Sequence[ProviderDesignAdvice]) -> tuple[Approach, ...]:
    """Group the approaches every provider named, and count who recommended each one.

    A provider cannot report how many providers recommended an approach, so the
    count is derived from the recommendation each provider gave. The recommendation
    is read against every grouped name, and the result is then restricted to the
    approaches that provider named itself. A provider votes once per approach on
    the aggregated risk level and fit rating.

    Args:
        advices: Parsed advice from every provider that answered.

    Returns:
        One approach per distinct name, ordered as the providers first named them.

    NOTE: A recommendation that names an approach in order to reject it still counts
    toward it, because the count is derived from names alone and not from their polarity.
    """
    display_names: dict[str, str] = {}
    pros: dict[str, dict[str, None]] = {}
    cons: dict[str, dict[str, None]] = {}
    risk_levels: dict[str, list[RiskLevel]] = defaultdict(list)
    fit_ratings: dict[str, list[FitRating]] = defaultdict(list)
    recommended_by = Counter[str]()

    for advice in advices:
        stated_risks: dict[str, list[RiskLevel]] = defaultdict(list)
        stated_fits: dict[str, list[FitRating]] = defaultdict(list)

        for approach in advice.approaches:
            key = _approach_name_key(approach.name)
            if not key:
                continue
            display_names.setdefault(key, approach.name)
            pros.setdefault(key, {}).update(dict.fromkeys(approach.pros))
            cons.setdefault(key, {}).update(dict.fromkeys(approach.cons))
            if approach.risk_level.stated:
                stated_risks[key].append(approach.risk_level)
            if approach.fit_rating is not None:
                stated_fits[key].append(approach.fit_rating)

        # One ballot per provider per approach, so naming an approach twice does not vote twice.
        for key, levels in stated_risks.items():
            level = _most_cautious_of_the_most_common(levels)
            if level is not None:
                risk_levels[key].append(level)
        for key, ratings in stated_fits.items():
            rating = _most_cautious_of_the_most_common(ratings)
            if rating is not None:
                fit_ratings[key].append(rating)

    # The grouped names are complete only once every provider has been read, and a longer
    # name rules out a shorter one, so the count needs a pass of its own.
    for advice in advices:
        recommended_by.update(_approaches_recommended_by(advice, display_names))

    return tuple(
        Approach(
            name=display_names[key],
            recommended_by=recommended_by[key],
            pros=tuple(pros[key]),
            cons=tuple(cons[key]),
            risk_level=_most_cautious_of_the_most_common(risk_levels[key]) or RiskLevel.UNKNOWN,
            fit_rating=_most_cautious_of_the_most_common(fit_ratings[key]),
        )
        for key in display_names
    )


def consensus_recommendation(
    advices: Sequence[ProviderDesignAdvice],
    approaches: Sequence[Approach],
    threshold: int = 2,
) -> str | None:
    """Name the approach a majority of the providers that answered recommended.

    The value is always a name that appears in ``approaches``, so a consumer can
    look it up rather than having to tell prose and a name apart.

    Args:
        advices: Parsed advice from every provider that answered.
        approaches: The grouped approaches those providers named, which supply
            the names a recommendation is read against.
        threshold: Minimum providers that must recommend one approach. It can
            raise the majority toward unanimity but never lower it.

    Returns:
        The approach's name, or None when no approach holds a majority.

    NOTE: A recommendation casts a ballot only when it names exactly one
    approach and that provider named the approach itself.
    A name carries no polarity, so a recommendation naming two does not say
    which of them it endorsed.
    NOTE: Providers that express one direction in different words are tallied
    separately, and each wording must hold a majority on its own.
    """
    if not advices:
        return None

    display_names = {_approach_name_key(approach.name): approach.name for approach in approaches}
    ballots = Counter[str]()
    for advice in advices:
        mentioned = _names_mentioned_in(advice.recommendation, display_names)
        if len(mentioned) != 1:
            continue
        key = next(iter(mentioned))
        if key in _own_name_keys(advice):
            ballots[key] += 1

    # At most one approach can hold a majority, because two would need more
    # ballots than there are providers, so no tie is possible here.
    needed = max(len(advices) // 2 + 1, min(threshold, len(advices)))
    winner = next((key for key, count in ballots.items() if count >= needed), None)
    return None if winner is None else display_names[winner]
