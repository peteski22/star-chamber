from __future__ import annotations

from star_chamber.consensus import (
    _issue_key,
    _issues_match,
    _parse_location,
    classify,
)
from star_chamber.types import Issue, ProviderReview

# ---------------------------------------------------------------------------
# Test helpers.
# ---------------------------------------------------------------------------


def _review(provider: str, issues: list[Issue], rating: str = "good") -> ProviderReview:
    return ProviderReview(
        provider=provider,
        model="test-model",
        quality_rating=rating,
        issues=tuple(issues),
        praise=(),
        summary="test summary",
        raw_content="",
    )


def _issue(location: str, category: str, severity: str = "high") -> Issue:
    return Issue(
        severity=severity,
        location=location,
        category=category,
        description=f"Issue at {location}",
        suggestion=f"Fix {category}",
    )


# ---------------------------------------------------------------------------
# _parse_location
# ---------------------------------------------------------------------------


class TestParseLocation:
    def test_file_and_line(self):
        assert _parse_location("auth.py:23") == ("auth.py", 23)

    def test_file_only(self):
        assert _parse_location("auth.py") == ("auth.py", None)

    def test_line_number_parsing_with_path(self):
        assert _parse_location("backend/app/auth.py:23") == ("backend/app/auth.py", 23)

    def test_empty_string(self):
        assert _parse_location("") == ("", None)


# ---------------------------------------------------------------------------
# _issue_key / _issues_match
# ---------------------------------------------------------------------------


class TestIssueKey:
    def test_basic_key(self):
        issue = _issue("auth.py:10", "security")
        assert _issue_key(issue) == ("auth.py", 10, "security")

    def test_key_without_line(self):
        issue = _issue("auth.py", "security")
        assert _issue_key(issue) == ("auth.py", None, "security")


class TestIssuesMatch:
    def test_exact_match(self):
        key_a = ("auth.py", 10, "security")
        key_b = ("auth.py", 10, "security")
        assert _issues_match(key_a, key_b) is True

    def test_within_tolerance(self):
        key_a = ("auth.py", 10, "security")
        key_b = ("auth.py", 15, "security")
        assert _issues_match(key_a, key_b) is True

    def test_beyond_tolerance(self):
        key_a = ("auth.py", 10, "security")
        key_b = ("auth.py", 16, "security")
        assert _issues_match(key_a, key_b) is False

    def test_different_category(self):
        key_a = ("auth.py", 10, "security")
        key_b = ("auth.py", 10, "style")
        assert _issues_match(key_a, key_b) is False

    def test_different_file(self):
        key_a = ("auth.py", 10, "security")
        key_b = ("models.py", 10, "security")
        assert _issues_match(key_a, key_b) is False

    def test_none_lines_match(self):
        key_a = ("auth.py", None, "security")
        key_b = ("auth.py", None, "security")
        assert _issues_match(key_a, key_b) is True

    def test_one_none_line_no_match(self):
        key_a = ("auth.py", None, "security")
        key_b = ("auth.py", 10, "security")
        assert _issues_match(key_a, key_b) is False


# ---------------------------------------------------------------------------
# classify
# ---------------------------------------------------------------------------


class TestClassify:
    def test_all_providers_agree_is_consensus(self):
        issue = _issue("auth.py:10", "security")
        reviews = [
            _review("openai", [issue]),
            _review("anthropic", [issue]),
            _review("gemini", [issue]),
        ]

        result = classify(reviews)

        assert len(result.consensus_issues) == 1
        assert result.consensus_issues[0].category == "security"
        assert result.consensus_issues[0].location == "auth.py:10"
        assert len(result.majority_issues) == 0
        assert len(result.individual_issues) == 0

    def test_two_of_three_is_majority(self):
        issue = _issue("auth.py:10", "security")
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

    def test_single_provider_is_individual(self):
        issue = _issue("auth.py:10", "security")
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
        assert result.individual_issues["openai"][0].category == "security"

    def test_line_tolerance_groups_nearby_lines(self):
        reviews = [
            _review("openai", [_issue("auth.py:23", "security")]),
            _review("anthropic", [_issue("auth.py:25", "security")]),
            _review("gemini", [_issue("auth.py:27", "security")]),
        ]

        result = classify(reviews)

        # All within +-5 tolerance of each other, so consensus.
        assert len(result.consensus_issues) == 1
        assert result.consensus_issues[0].category == "security"

    def test_different_category_not_grouped(self):
        reviews = [
            _review("openai", [_issue("auth.py:10", "security")]),
            _review("anthropic", [_issue("auth.py:10", "style")]),
        ]

        result = classify(reviews)

        # Different categories mean separate groups, each individual.
        assert len(result.consensus_issues) == 0
        assert len(result.majority_issues) == 0
        assert "openai" in result.individual_issues
        assert "anthropic" in result.individual_issues

    def test_different_file_not_grouped(self):
        reviews = [
            _review("openai", [_issue("auth.py:10", "security")]),
            _review("anthropic", [_issue("models.py:10", "security")]),
        ]

        result = classify(reviews)

        # Different files mean separate groups, each individual.
        assert len(result.consensus_issues) == 0
        assert len(result.majority_issues) == 0
        assert "openai" in result.individual_issues
        assert "anthropic" in result.individual_issues

    def test_empty_reviews(self):
        result = classify([])

        assert result.consensus_issues == ()
        assert result.majority_issues == ()
        assert result.individual_issues == {}

    def test_single_provider(self):
        issue = _issue("auth.py:10", "security")
        reviews = [_review("openai", [issue])]

        result = classify(reviews)

        # Single provider trivially agrees with itself => consensus.
        assert len(result.consensus_issues) == 1
        assert result.consensus_issues[0].category == "security"

    def test_all_providers_no_issues(self):
        reviews = [
            _review("openai", []),
            _review("anthropic", []),
        ]

        result = classify(reviews)

        assert result.consensus_issues == ()
        assert result.majority_issues == ()
        assert result.individual_issues == {}

    def test_consensus_threshold(self):
        issue = _issue("auth.py:10", "security")
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

    def test_ordering_high_before_low(self):
        reviews = [
            _review(
                "openai",
                [
                    _issue("auth.py:10", "security", severity="low"),
                    _issue("auth.py:20", "bug", severity="high"),
                ],
            ),
            _review(
                "anthropic",
                [
                    _issue("auth.py:10", "security", severity="low"),
                    _issue("auth.py:20", "bug", severity="high"),
                ],
            ),
            _review(
                "gemini",
                [
                    _issue("auth.py:10", "security", severity="low"),
                    _issue("auth.py:20", "bug", severity="high"),
                ],
            ),
        ]

        result = classify(reviews)

        assert len(result.consensus_issues) == 2
        # High severity should come first.
        assert result.consensus_issues[0].severity == "high"
        assert result.consensus_issues[1].severity == "low"

    def test_multiple_issues_mixed_classification(self):
        # Issue that all three agree on.
        consensus_issue = _issue("auth.py:10", "security")
        # Issue that two of three agree on.
        majority_issue = _issue("models.py:30", "bug")
        # Issue from only one provider.
        individual_issue = _issue("views.py:5", "style")

        reviews = [
            _review("openai", [consensus_issue, majority_issue, individual_issue]),
            _review("anthropic", [consensus_issue, majority_issue]),
            _review("gemini", [consensus_issue]),
        ]

        result = classify(reviews)

        assert len(result.consensus_issues) == 1
        assert result.consensus_issues[0].category == "security"

        assert len(result.majority_issues) == 1
        assert result.majority_issues[0].category == "bug"
        assert result.majority_issues[0].provider_count == 2

        assert "openai" in result.individual_issues
        assert len(result.individual_issues["openai"]) == 1
        assert result.individual_issues["openai"][0].category == "style"

    def test_line_number_parsing_with_path(self):
        reviews = [
            _review("openai", [_issue("backend/app/auth.py:23", "security")]),
            _review("anthropic", [_issue("backend/app/auth.py:24", "security")]),
        ]

        result = classify(reviews)

        assert len(result.consensus_issues) == 1
        assert result.consensus_issues[0].location == "backend/app/auth.py:23"
