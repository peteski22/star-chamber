from __future__ import annotations

from star_chamber.consensus import (
    _issue_key,
    _issues_match,
    _parse_location,
    classify,
)
from star_chamber.types import Issue, IssueCategory, ProviderReview, QualityRating, Severity

# ---------------------------------------------------------------------------
# Test helpers.
# ---------------------------------------------------------------------------


def _review(provider: str, issues: list[Issue], rating: str = "good") -> ProviderReview:
    return ProviderReview(
        provider=provider,
        model="test-model",
        quality_rating=QualityRating(rating),
        issues=tuple(issues),
        praise=(),
        summary="test summary",
        raw_content="",
    )


def _issue(location: str, category: str, severity: str = "high") -> Issue:
    """Build an issue from the wire spellings, so a fixture outside the protocol's sets fails here."""
    return Issue(
        severity=Severity(severity),
        location=location,
        category=IssueCategory(category),
        description=f"Issue at {location}",
        suggestion=f"Fix {category}",
    )


# ---------------------------------------------------------------------------
# _parse_location
# ---------------------------------------------------------------------------


class TestParseLocation:
    def test_file_and_line(self) -> None:
        assert _parse_location("auth.py:23") == ("auth.py", (23, 23))

    def test_file_only(self) -> None:
        assert _parse_location("auth.py") == ("auth.py", None)

    def test_line_number_parsing_with_path(self) -> None:
        assert _parse_location("backend/app/auth.py:23") == ("backend/app/auth.py", (23, 23))

    def test_empty_string(self) -> None:
        assert _parse_location("") == ("", None)


# ---------------------------------------------------------------------------
# _issue_key / _issues_match
# ---------------------------------------------------------------------------


class TestIssueKey:
    def test_basic_key(self) -> None:
        issue = _issue("auth.py:10", "architecture")
        assert _issue_key(issue) == ("auth.py", (10, 10), "architecture")

    def test_key_without_line(self) -> None:
        issue = _issue("auth.py", "architecture")
        assert _issue_key(issue) == ("auth.py", None, "architecture")


class TestIssuesMatch:
    def test_exact_match(self) -> None:
        key_a = ("auth.py", (10, 10), "architecture")
        key_b = ("auth.py", (10, 10), "architecture")
        assert _issues_match(key_a, key_b) is True

    def test_within_tolerance(self) -> None:
        key_a = ("auth.py", (10, 10), "architecture")
        key_b = ("auth.py", (15, 15), "architecture")
        assert _issues_match(key_a, key_b) is True

    def test_beyond_tolerance(self) -> None:
        key_a = ("auth.py", (10, 10), "architecture")
        key_b = ("auth.py", (16, 16), "architecture")
        assert _issues_match(key_a, key_b) is False

    def test_different_category(self) -> None:
        key_a = ("auth.py", (10, 10), "architecture")
        key_b = ("auth.py", (10, 10), "craftsmanship")
        assert _issues_match(key_a, key_b) is False

    def test_different_file(self) -> None:
        key_a = ("auth.py", (10, 10), "architecture")
        key_b = ("models.py", (10, 10), "architecture")
        assert _issues_match(key_a, key_b) is False

    def test_absent_spans_match(self) -> None:
        key_a = ("auth.py", None, "architecture")
        key_b = ("auth.py", None, "architecture")
        assert _issues_match(key_a, key_b) is True

    def test_one_absent_span_no_match(self) -> None:
        key_a = ("auth.py", None, "architecture")
        key_b = ("auth.py", (10, 10), "architecture")
        assert _issues_match(key_a, key_b) is False


# ---------------------------------------------------------------------------
# classify
# ---------------------------------------------------------------------------


class TestClassify:
    def test_all_providers_agree_is_consensus(self) -> None:
        issue = _issue("auth.py:10", "architecture")
        reviews = [
            _review("openai", [issue]),
            _review("anthropic", [issue]),
            _review("gemini", [issue]),
        ]

        result = classify(reviews)

        assert len(result.consensus_issues) == 1
        assert result.consensus_issues[0].category == "architecture"
        assert result.consensus_issues[0].location == "auth.py:10"
        assert len(result.majority_issues) == 0
        assert len(result.individual_issues) == 0

    def test_two_of_three_is_majority(self) -> None:
        issue = _issue("auth.py:10", "architecture")
        reviews = [
            _review("openai", [issue]),
            _review("anthropic", [issue]),
            _review("gemini", []),
        ]

        result = classify(reviews)

        assert len(result.consensus_issues) == 0
        assert len(result.majority_issues) == 1
        majority = result.majority_issues[0]
        assert majority.provider_count == 2
        assert set(majority.flagged_by) == {"openai", "anthropic"}
        assert len(result.individual_issues) == 0

    def test_single_provider_is_individual(self) -> None:
        issue = _issue("auth.py:10", "architecture")
        reviews = [
            _review("openai", [issue]),
            _review("anthropic", []),
            _review("gemini", []),
        ]

        result = classify(reviews)

        assert len(result.consensus_issues) == 0
        assert len(result.majority_issues) == 0
        assert "openai" in result.individual_issues
        assert len(result.individual_issues["openai"]) == 1
        assert result.individual_issues["openai"][0].category == "architecture"

    def test_line_tolerance_groups_nearby_lines(self) -> None:
        reviews = [
            _review("openai", [_issue("auth.py:23", "architecture")]),
            _review("anthropic", [_issue("auth.py:25", "architecture")]),
            _review("gemini", [_issue("auth.py:27", "architecture")]),
        ]

        result = classify(reviews)

        # All within +-5 tolerance of each other, so consensus.
        assert len(result.consensus_issues) == 1
        assert result.consensus_issues[0].category == "architecture"

    def test_different_category_not_grouped(self) -> None:
        reviews = [
            _review("openai", [_issue("auth.py:10", "architecture")]),
            _review("anthropic", [_issue("auth.py:10", "craftsmanship")]),
        ]

        result = classify(reviews)

        # Different categories mean separate groups, each individual.
        assert len(result.consensus_issues) == 0
        assert len(result.majority_issues) == 0
        assert "openai" in result.individual_issues
        assert "anthropic" in result.individual_issues

    def test_different_file_not_grouped(self) -> None:
        reviews = [
            _review("openai", [_issue("auth.py:10", "architecture")]),
            _review("anthropic", [_issue("models.py:10", "architecture")]),
        ]

        result = classify(reviews)

        # Different files mean separate groups, each individual.
        assert len(result.consensus_issues) == 0
        assert len(result.majority_issues) == 0
        assert "openai" in result.individual_issues
        assert "anthropic" in result.individual_issues

    def test_empty_reviews(self) -> None:
        result = classify([])

        assert result.consensus_issues == ()
        assert result.majority_issues == ()
        assert result.individual_issues == {}

    def test_single_provider(self) -> None:
        issue = _issue("auth.py:10", "architecture")
        reviews = [_review("openai", [issue])]

        result = classify(reviews)

        # Single provider trivially agrees with itself => consensus.
        assert len(result.consensus_issues) == 1
        assert result.consensus_issues[0].category == "architecture"

    def test_all_providers_no_issues(self) -> None:
        reviews = [
            _review("openai", []),
            _review("anthropic", []),
        ]

        result = classify(reviews)

        assert result.consensus_issues == ()
        assert result.majority_issues == ()
        assert result.individual_issues == {}

    def test_consensus_threshold(self) -> None:
        issue = _issue("auth.py:10", "architecture")
        reviews = [
            _review("openai", [issue]),
            _review("anthropic", [issue]),
            _review("gemini", []),
        ]

        # With threshold=3, two providers agreeing is not enough for consensus.
        result = classify(reviews, threshold=3)

        assert len(result.consensus_issues) == 0
        assert len(result.majority_issues) == 1
        assert result.majority_issues[0].provider_count == 2

    def test_ordering_high_before_low(self) -> None:
        reviews = [
            _review(
                "openai",
                [
                    _issue("auth.py:10", "architecture", severity="low"),
                    _issue("auth.py:20", "correctness", severity="high"),
                ],
            ),
            _review(
                "anthropic",
                [
                    _issue("auth.py:10", "architecture", severity="low"),
                    _issue("auth.py:20", "correctness", severity="high"),
                ],
            ),
            _review(
                "gemini",
                [
                    _issue("auth.py:10", "architecture", severity="low"),
                    _issue("auth.py:20", "correctness", severity="high"),
                ],
            ),
        ]

        result = classify(reviews)

        assert len(result.consensus_issues) == 2
        # High severity should come first.
        assert result.consensus_issues[0].severity == "high"
        assert result.consensus_issues[1].severity == "low"

    def test_multiple_issues_mixed_classification(self) -> None:
        # Issue that all three agree on.
        consensus_issue = _issue("auth.py:10", "architecture")
        # Issue that two of three agree on.
        majority_issue = _issue("models.py:30", "correctness")
        # Issue from only one provider.
        individual_issue = _issue("views.py:5", "craftsmanship")

        reviews = [
            _review("openai", [consensus_issue, majority_issue, individual_issue]),
            _review("anthropic", [consensus_issue, majority_issue]),
            _review("gemini", [consensus_issue]),
        ]

        result = classify(reviews)

        assert len(result.consensus_issues) == 1
        assert result.consensus_issues[0].category == "architecture"

        assert len(result.majority_issues) == 1
        assert result.majority_issues[0].category == "correctness"
        assert result.majority_issues[0].provider_count == 2

        assert "openai" in result.individual_issues
        assert len(result.individual_issues["openai"]) == 1
        assert result.individual_issues["openai"][0].category == "craftsmanship"

    def test_line_number_parsing_with_path(self) -> None:
        reviews = [
            _review("openai", [_issue("backend/app/auth.py:23", "architecture")]),
            _review("anthropic", [_issue("backend/app/auth.py:24", "architecture")]),
        ]

        result = classify(reviews)

        assert len(result.consensus_issues) == 1
        assert result.consensus_issues[0].location == "backend/app/auth.py:23"


# ---------------------------------------------------------------------------
# Labelled gateway members.
# ---------------------------------------------------------------------------


class TestLabelledGatewayMembers:
    """Members routed through one gateway provider stay distinct via display names.

    When multiple council members share a single provider (e.g. several models
    behind an OpenAI-compatible gateway such as OpenRouter), the per-entry
    display name becomes the review identity, so each member counts as a
    separate voter instead of collapsing into one provider-name vote.
    """

    def test_distinct_labels_are_distinct_voters(self) -> None:
        reviews = [
            _review("gpt-5.2", [_issue("app.py:10", "correctness")]),
            _review("gemini-3.1-pro", [_issue("app.py:12", "correctness")]),
            _review("grok-4.3", [_issue("app.py:11", "correctness")]),
        ]

        result = classify(reviews, threshold=2)

        assert len(result.consensus_issues) == 1
        assert result.majority_issues == ()

    def test_two_of_three_labelled_members_form_majority(self) -> None:
        reviews = [
            _review("gpt-5.2", [_issue("other.py:5", "correctness")]),
            _review("gemini-3.1-pro", [_issue("app.py:12", "correctness")]),
            _review("grok-4.3", [_issue("app.py:11", "correctness")]),
        ]

        result = classify(reviews, threshold=2)

        assert len(result.majority_issues) == 1
        assert result.majority_issues[0].flagged_by == ("gemini-3.1-pro", "grok-4.3")


# ---------------------------------------------------------------------------
# Line spans
# ---------------------------------------------------------------------------


class TestLineSpans:
    """A line range must group with the single lines and ranges it covers."""

    def test_range_parses_into_a_span(self) -> None:
        assert _parse_location("auth.py:203-207") == ("auth.py", (203, 207))

    def test_single_line_parses_into_a_span_of_one(self) -> None:
        assert _parse_location("auth.py:205") == ("auth.py", (205, 205))

    def test_reversed_range_is_normalized(self) -> None:
        assert _parse_location("auth.py:207-203") == ("auth.py", (203, 207))

    def test_non_numeric_tail_keeps_the_whole_string(self) -> None:
        assert _parse_location("auth.py:somewhere") == ("auth.py:somewhere", None)

    def test_drive_letter_path_is_not_split(self) -> None:
        assert _parse_location(r"C:\src\auth.py") == (r"C:\src\auth.py", None)

    def test_overlapping_ranges_match(self) -> None:
        assert _issues_match(("auth.py", (203, 207), "correctness"), ("auth.py", (203, 205), "correctness"))

    def test_range_matches_a_single_line_inside_it(self) -> None:
        assert _issues_match(("auth.py", (203, 207), "correctness"), ("auth.py", (205, 205), "correctness"))

    def test_range_matches_a_single_line_within_tolerance_of_its_edge(self) -> None:
        assert _issues_match(("auth.py", (203, 207), "correctness"), ("auth.py", (212, 212), "correctness"))

    def test_distant_ranges_do_not_match(self) -> None:
        assert not _issues_match(("auth.py", (203, 207), "correctness"), ("auth.py", (300, 310), "correctness"))

    def test_ranges_in_different_files_do_not_match(self) -> None:
        assert not _issues_match(("auth.py", (203, 207), "correctness"), ("db.py", (203, 207), "correctness"))

    def test_unanimous_finding_cited_in_three_formats_is_consensus(self) -> None:
        reviews = [
            _review("openai", [_issue("auth.py:203-207", "correctness")]),
            _review("anthropic", [_issue("auth.py:203-205", "correctness")]),
            _review("gemini", [_issue("auth.py:205", "correctness")]),
        ]

        result = classify(reviews, threshold=2)

        assert len(result.consensus_issues) == 1
        assert len(result.majority_issues) == 0
        assert result.individual_issues == {}

    def test_single_line_tolerance_is_unchanged(self) -> None:
        reviews = [
            _review("openai", [_issue("auth.py:203", "correctness")]),
            _review("anthropic", [_issue("auth.py:208", "correctness")]),
        ]

        result = classify(reviews, threshold=2)

        assert len(result.consensus_issues) == 1

    def test_single_lines_beyond_tolerance_stay_individual(self) -> None:
        reviews = [
            _review("openai", [_issue("auth.py:203", "correctness")]),
            _review("anthropic", [_issue("auth.py:209", "correctness")]),
        ]

        result = classify(reviews, threshold=2)

        assert len(result.consensus_issues) == 0
        assert {k: len(v) for k, v in result.individual_issues.items()} == {"openai": 1, "anthropic": 1}


class TestSpanWidthLimit:
    """A span too wide to be one place is a file-level citation, not a line-level one."""

    def test_file_wide_span_yields_no_span(self) -> None:
        assert _parse_location("app.py:1-9999") == ("app.py", None)

    def test_file_wide_span_keeps_the_file_path_recoverable(self) -> None:
        file, span = _parse_location("backend/app.py:1-9999")
        assert file == "backend/app.py"
        assert span is None

    def test_span_at_the_limit_is_kept(self) -> None:
        assert _parse_location("app.py:100-149") == ("app.py", (100, 149))

    def test_span_past_the_limit_is_dropped(self) -> None:
        assert _parse_location("app.py:100-150") == ("app.py", None)

    def test_file_wide_span_does_not_absorb_a_specific_line(self) -> None:
        assert not _issues_match(("app.py", None, "correctness"), ("app.py", (50, 50), "correctness"))

    def test_file_wide_citations_still_group_together(self) -> None:
        assert _issues_match(("app.py", None, "correctness"), ("app.py", None, "correctness"))

    def test_a_wide_citation_cannot_suppress_other_findings(self) -> None:
        reviews = [
            _review("openai", [_issue("app.py:1-9999", "correctness")]),
            _review("anthropic", [_issue("app.py:50", "correctness")]),
            _review("gemini", [_issue("app.py:800", "correctness")]),
        ]

        result = classify(reviews, threshold=2)

        surviving = (
            len(result.consensus_issues)
            + len(result.majority_issues)
            + sum(len(v) for v in result.individual_issues.values())
        )
        assert surviving == 3
        assert len(result.consensus_issues) == 0


class TestPathologicalLocations:
    """An unusable location must not take the run down with it."""

    def test_absurdly_long_digit_run_does_not_raise(self) -> None:
        location = "app.py:" + "1" * 5000
        assert _parse_location(location) == (location, None)

    def test_absurdly_long_range_does_not_raise(self) -> None:
        location = "app.py:1-" + "9" * 5000
        assert _parse_location(location) == (location, None)

    def test_non_ascii_digits_are_not_treated_as_line_numbers(self) -> None:
        assert _parse_location("app.py:٢٣") == ("app.py:٢٣", None)

    def test_classify_survives_a_pathological_location(self) -> None:
        reviews = [
            _review("openai", [_issue("app.py:" + "1" * 5000, "correctness")]),
            _review("anthropic", [_issue("app.py:50", "correctness")]),
        ]

        result = classify(reviews, threshold=2)

        assert sum(len(v) for v in result.individual_issues.values()) == 2
