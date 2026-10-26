from __future__ import annotations

import json
from pathlib import Path

import pytest

from star_chamber.parsing import ParseError, extract_json, parse_code_review, parse_design_advice
from star_chamber.types import (
    Approach,
    FitRating,
    Issue,
    IssueCategory,
    ProviderDesignAdvice,
    ProviderReview,
    QualityRating,
    RiskLevel,
    Severity,
)

FIXTURES = Path(__file__).parent / "testdata" / "provider_responses"


# -- extract_json --------------------------------------------------------------


class TestExtractJson:
    def test_direct_json_object(self) -> None:
        raw = '{"quality_rating": "good", "issues": []}'
        result = extract_json(raw)
        assert result == {"quality_rating": "good", "issues": []}

    def test_json_code_block(self) -> None:
        raw = '```json\n{"key": "value"}\n```'
        result = extract_json(raw)
        assert result == {"key": "value"}

    def test_bare_code_block(self) -> None:
        raw = '```\n{"key": "value"}\n```'
        result = extract_json(raw)
        assert result == {"key": "value"}

    def test_code_block_with_surrounding_text(self) -> None:
        raw = 'Here is my analysis:\n\n```json\n{"rating": "good"}\n```\n\nHope that helps!'
        result = extract_json(raw)
        assert result == {"rating": "good"}

    def test_empty_string(self) -> None:
        result = extract_json("")
        assert result is None

    def test_no_json(self) -> None:
        result = extract_json("This is just plain text with no JSON at all.")
        assert result is None

    def test_array_json(self) -> None:
        raw = '[{"a": 1}, {"b": 2}]'
        result = extract_json(raw)
        assert result == [{"a": 1}, {"b": 2}]


# -- parse_code_review ---------------------------------------------------------


class TestParseCodeReview:
    def test_parse_from_code_block(self) -> None:
        content = (FIXTURES / "openai_code_review.txt").read_text()
        review = parse_code_review(content, provider="openai", model="gpt-4o")

        assert isinstance(review, ProviderReview)
        assert review.provider == "openai"
        assert review.model == "gpt-4o"
        assert review.quality_rating == "fair"
        assert len(review.issues) == 1
        assert review.issues[0] == Issue(
            severity="high",
            location="auth.py:23",
            category="correctness",
            description="SHA-256 for password hashing is insecure.",
            suggestion="Use bcrypt.",
        )
        assert review.praise == ("Clean separation of concerns.",)
        assert review.summary == "Critical security flaw found."
        assert review.raw_content == content

    def test_parse_direct_json(self) -> None:
        content = (FIXTURES / "anthropic_code_review.txt").read_text()
        review = parse_code_review(content, provider="anthropic", model="claude-sonnet-4-20250514")

        assert isinstance(review, ProviderReview)
        assert review.provider == "anthropic"
        assert review.model == "claude-sonnet-4-20250514"
        assert review.quality_rating == "good"
        assert review.issues == ()
        assert review.praise == ("Well structured.",)
        assert review.summary == "No issues found."

    def test_malformed_raises_parse_error(self) -> None:
        content = (FIXTURES / "malformed.txt").read_text()
        with pytest.raises(ParseError, match="Failed to extract JSON"):
            parse_code_review(content, provider="openai", model="gpt-4o")

    def test_missing_quality_rating_raises_parse_error(self) -> None:
        content = json.dumps({"issues": [], "praise": [], "summary": "ok"})
        with pytest.raises(ParseError, match="quality_rating"):
            parse_code_review(content, provider="openai", model="gpt-4o")

    def test_provider_from_wrapper_overrides_body(self) -> None:
        content = json.dumps(
            {
                "provider": "wrong-provider",
                "quality_rating": "good",
                "issues": [],
                "praise": [],
                "summary": "ok",
            }
        )
        review = parse_code_review(content, provider="anthropic", model="claude-sonnet-4-20250514")
        assert review.provider == "anthropic"


# -- parse_design_advice -------------------------------------------------------


class TestParseDesignAdvice:
    def test_parse_design_advice(self) -> None:
        content = (FIXTURES / "openai_design.txt").read_text()
        advice = parse_design_advice(content, provider="openai", model="gpt-4o")

        assert isinstance(advice, ProviderDesignAdvice)
        assert advice.provider == "openai"
        assert advice.model == "gpt-4o"
        assert advice.recommendation == "Use the repository pattern for data access."
        assert len(advice.approaches) == 2

        repo_approach = advice.approaches[0]
        assert isinstance(repo_approach, Approach)
        assert repo_approach.name == "Repository Pattern"
        assert repo_approach.pros == ("Clean separation of concerns", "Testable")
        assert repo_approach.cons == ("More boilerplate",)
        assert repo_approach.risk_level == "low"
        assert repo_approach.fit_rating == "excellent"

        active_record = advice.approaches[1]
        assert active_record.name == "Active Record"
        assert active_record.risk_level == "medium"

        assert advice.summary == "Repository pattern is the best fit for this codebase."
        assert advice.raw_content == content

    def test_malformed_raises_parse_error(self) -> None:
        content = (FIXTURES / "malformed.txt").read_text()
        with pytest.raises(ParseError, match="Failed to extract JSON"):
            parse_design_advice(content, provider="openai", model="gpt-4o")


# -- unparsable response diagnostics -------------------------------------------


class TestUnparsableDiagnostics:
    """The failure message must separate a cut-off reply from prose from an empty one."""

    def test_empty_response_is_named_as_empty(self) -> None:
        with pytest.raises(ParseError, match="empty"):
            parse_code_review("", provider="anthropic", model="claude-opus-5")

    def test_cut_off_json_is_named_as_incomplete(self) -> None:
        content = '{"quality_rating": "good", "issues": [{"severity": "high", "location": "a.py:1'
        with pytest.raises(ParseError, match="ends mid-structure"):
            parse_code_review(content, provider="anthropic", model="claude-opus-5")

    def test_cut_off_json_inside_a_code_fence_is_named_as_incomplete(self) -> None:
        content = '```json\n{"quality_rating": "good", "issues": [{"severity": "hi'
        with pytest.raises(ParseError, match="ends mid-structure"):
            parse_code_review(content, provider="anthropic", model="claude-opus-5")

    def test_prose_is_named_as_carrying_no_json(self) -> None:
        content = "I cannot review this code because the request is unclear."
        with pytest.raises(ParseError, match="no JSON found"):
            parse_code_review(content, provider="anthropic", model="claude-opus-5")

    def test_complete_but_invalid_json_is_named_as_malformed(self) -> None:
        content = '{"quality_rating": "good",,}'
        with pytest.raises(ParseError, match="malformed"):
            parse_code_review(content, provider="anthropic", model="claude-opus-5")

    def test_message_reports_the_response_length(self) -> None:
        content = "x" * 4096
        with pytest.raises(ParseError, match="4096 characters"):
            parse_code_review(content, provider="anthropic", model="claude-opus-5")

    def test_design_advice_reports_the_same_detail(self) -> None:
        content = '{"recommendation": "repository", "approaches": [{"name": "Repo'
        with pytest.raises(ParseError, match="ends mid-structure"):
            parse_design_advice(content, provider="anthropic", model="claude-opus-5")

    def test_existing_message_prefix_is_preserved(self) -> None:
        with pytest.raises(ParseError, match=r"Failed to extract JSON from anthropic response"):
            parse_code_review("prose only", provider="anthropic", model="claude-opus-5")


_ISSUE = {
    "severity": "high",
    "location": "app.py:10",
    "category": "correctness",
    "description": "Off-by-one.",
    "suggestion": "Use <=.",
}

_NON_TEXT = [
    pytest.param(None, "null", id="null"),
    pytest.param(12, "a whole number", id="number"),
    pytest.param(True, "a boolean", id="boolean"),
    pytest.param([], "a list", id="list"),
    pytest.param({}, "an object", id="object"),
]


def _review(**overrides) -> str:
    body = {"quality_rating": "good", "issues": [dict(_ISSUE)], "praise": ["Clear."], "summary": "Fine."}
    body.update(overrides)
    return json.dumps(body)


def _advice(**overrides) -> str:
    body = {"recommendation": "Use a repository.", "approaches": [{"name": "Repository"}], "summary": "Fine."}
    body.update(overrides)
    return json.dumps(body)


def _parse_review(content: str) -> ProviderReview:
    return parse_code_review(content, provider="anthropic", model="claude-opus-5")


def _parse_advice(content: str) -> ProviderDesignAdvice:
    return parse_design_advice(content, provider="anthropic", model="claude-opus-5")


class TestCodeReviewFieldTypes:
    """A member's reply is untrusted, so every field it supplies is checked here.

    A field of the wrong shape raises ParseError, which puts that member in
    failed_providers rather than ending the session.
    """

    @pytest.mark.parametrize("field", sorted(_ISSUE))
    @pytest.mark.parametrize(("value", "shape"), _NON_TEXT)
    def test_non_text_issue_field_is_rejected(self, field, value, shape) -> None:
        issue = dict(_ISSUE) | {field: value}

        with pytest.raises(ParseError, match=rf"Field {field} in an issue .* must be text, got {shape}"):
            _parse_review(_review(issues=[issue]))

    @pytest.mark.parametrize("field", sorted(_ISSUE))
    def test_omitted_issue_field_is_rejected(self, field) -> None:
        issue = {k: v for k, v in _ISSUE.items() if k != field}

        with pytest.raises(ParseError, match=rf"Missing required field {field} in an issue"):
            _parse_review(_review(issues=[issue]))

    @pytest.mark.parametrize(("value", "shape"), _NON_TEXT)
    def test_non_text_quality_rating_is_rejected(self, value, shape) -> None:
        with pytest.raises(ParseError, match=rf"Field quality_rating .* must be text, got {shape}"):
            _parse_review(_review(quality_rating=value))

    @pytest.mark.parametrize(("value", "shape"), _NON_TEXT)
    def test_non_text_summary_is_rejected(self, value, shape) -> None:
        with pytest.raises(ParseError, match=rf"Field summary .* must be text, got {shape}"):
            _parse_review(_review(summary=value))

    def test_issues_must_be_a_list(self) -> None:
        with pytest.raises(ParseError, match="Field issues .* must be a list, got text"):
            _parse_review(_review(issues="none"))

    def test_an_issue_must_be_an_object(self) -> None:
        with pytest.raises(ParseError, match="Every entry of issues .* must be an object, got text"):
            _parse_review(_review(issues=["none"]))

    def test_praise_must_be_a_list(self) -> None:
        """A string would otherwise become one praise entry per character."""
        with pytest.raises(ParseError, match="Field praise .* must be a list, got text"):
            _parse_review(_review(praise="Clear."))

    def test_a_praise_entry_must_be_text(self) -> None:
        with pytest.raises(ParseError, match="Every entry of praise .* must be text, got null"):
            _parse_review(_review(praise=[None]))

    def test_omitted_optional_fields_keep_their_defaults(self) -> None:
        review = _parse_review(json.dumps({"quality_rating": "good"}))

        assert review.issues == ()
        assert review.praise == ()
        assert review.summary == ""


class TestDesignFieldTypes:
    """The same checks for design advice, where fit_rating may legitimately be null."""

    @pytest.mark.parametrize(("value", "shape"), _NON_TEXT)
    def test_non_text_approach_name_is_rejected(self, value, shape) -> None:
        with pytest.raises(ParseError, match=rf"Field name in an approach .* must be text, got {shape}"):
            _parse_advice(_advice(approaches=[{"name": value}]))

    def test_omitted_approach_name_is_rejected(self) -> None:
        with pytest.raises(ParseError, match="Missing required field name in an approach"):
            _parse_advice(_advice(approaches=[{}]))

    @pytest.mark.parametrize(("value", "shape"), _NON_TEXT)
    def test_non_text_recommendation_is_rejected(self, value, shape) -> None:
        with pytest.raises(ParseError, match=rf"Field recommendation .* must be text, got {shape}"):
            _parse_advice(_advice(recommendation=value))

    def test_null_risk_level_is_rejected(self) -> None:
        with pytest.raises(ParseError, match="Field risk_level in an approach .* must be text, got null"):
            _parse_advice(_advice(approaches=[{"name": "Repository", "risk_level": None}]))

    def test_non_text_fit_rating_is_rejected(self) -> None:
        with pytest.raises(ParseError, match="Field fit_rating in an approach .* must be text, got a whole number"):
            _parse_advice(_advice(approaches=[{"name": "Repository", "fit_rating": 3}]))

    def test_null_fit_rating_is_accepted(self) -> None:
        """The output schema admits a null fit_rating, so the parser must too."""
        advice = _parse_advice(_advice(approaches=[{"name": "Repository", "fit_rating": None}]))

        assert advice.approaches[0].fit_rating is None

    def test_non_numeric_recommended_by_is_rejected(self) -> None:
        with pytest.raises(ParseError, match="Field recommended_by in an approach .* must be a whole number, got text"):
            _parse_advice(_advice(approaches=[{"name": "Repository", "recommended_by": "two"}]))

    def test_boolean_recommended_by_is_rejected(self) -> None:
        """A JSON boolean is an int in Python but serializes back as true."""
        with pytest.raises(ParseError, match="must be a whole number, got a boolean"):
            _parse_advice(_advice(approaches=[{"name": "Repository", "recommended_by": True}]))

    def test_approaches_must_be_a_list(self) -> None:
        with pytest.raises(ParseError, match="Field approaches .* must be a list, got text"):
            _parse_advice(_advice(approaches="none"))

    def test_an_approach_must_be_an_object(self) -> None:
        with pytest.raises(ParseError, match="Every entry of approaches .* must be an object, got text"):
            _parse_advice(_advice(approaches=["none"]))

    def test_a_pros_entry_must_be_text(self) -> None:
        with pytest.raises(ParseError, match="Every entry of pros .* must be text, got null"):
            _parse_advice(_advice(approaches=[{"name": "Repository", "pros": [None]}]))

    def test_omitted_optional_fields_keep_their_defaults(self) -> None:
        advice = _parse_advice(_advice(approaches=[{"name": "Repository"}], summary=""))
        approach = advice.approaches[0]

        assert approach.recommended_by == 0
        assert approach.pros == ()
        assert approach.cons == ()
        assert approach.risk_level == "unknown"
        assert approach.fit_rating is None


# What a member may send for each closed-set field, which is not every member of the set for risk_level.
_MEMBER_CHOICES = {
    "category": tuple(IssueCategory),
    "fit_rating": tuple(FitRating),
    "quality_rating": tuple(QualityRating),
    "risk_level": tuple(level for level in RiskLevel if level.stated),
    "severity": tuple(Severity),
}


def _carrying(field: str, value: object) -> Issue | ProviderReview | Approach:
    """Parse a reply carrying one value, from whichever shape declares that field."""
    if field in ("category", "severity"):
        return _parse_review(_review(issues=[dict(_ISSUE) | {field: value}])).issues[0]
    if field == "quality_rating":
        return _parse_review(_review(**{field: value}))
    return _parse_advice(_advice(approaches=[{"name": "Repository", field: value}])).approaches[0]


class TestClosedSets:
    """Five fields name a closed set, so a member's own vocabulary is a parse failure.

    The council would otherwise report that member as a success and emit a
    document that fails the schema the package itself ships.
    """

    @pytest.mark.parametrize(
        ("field", "value"),
        [(field, value) for field, values in sorted(_MEMBER_CHOICES.items()) for value in values],
    )
    def test_a_value_the_protocol_names_is_accepted(self, field: str, value: str) -> None:
        assert getattr(_carrying(field, value), field) == value

    @pytest.mark.parametrize("field", sorted(_MEMBER_CHOICES))
    def test_a_value_outside_the_set_is_rejected(self, field: str) -> None:
        allowed = ", ".join(_MEMBER_CHOICES[field])

        with pytest.raises(ParseError, match=rf"Field {field} in .* must be one of {allowed}, got 'solid'"):
            _carrying(field, "solid")

    def test_a_member_may_not_state_the_council_s_own_risk_level(self) -> None:
        """UNKNOWN reports that no member rated the approach, so no member can claim it."""
        with pytest.raises(ParseError, match=r"Field risk_level in .* must be one of high, medium, low, got 'unknown'"):
            _carrying("risk_level", RiskLevel.UNKNOWN.value)

    def test_an_overlong_value_is_reported_by_length(self) -> None:
        """A member can put a whole reply in a field the protocol says holds one short word."""
        with pytest.raises(ParseError, match=r"must be one of .*, got 5000 characters of text"):
            _carrying("severity", "x" * 5000)

    def test_an_unrated_approach_keeps_the_council_s_own_risk_level(self) -> None:
        approach = _parse_advice(_advice(approaches=[{"name": "Repository"}])).approaches[0]

        assert approach.risk_level is RiskLevel.UNKNOWN
        assert approach.fit_rating is None
