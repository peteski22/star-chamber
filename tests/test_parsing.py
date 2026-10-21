from __future__ import annotations

import json
from pathlib import Path

import pytest

from star_chamber.parsing import ParseError, extract_json, parse_code_review, parse_design_advice
from star_chamber.types import Approach, Issue, ProviderDesignAdvice, ProviderReview

FIXTURES = Path(__file__).parent / "testdata" / "provider_responses"


# -- extract_json --------------------------------------------------------------


class TestExtractJson:
    def test_direct_json_object(self):
        raw = '{"quality_rating": "good", "issues": []}'
        result = extract_json(raw)
        assert result == {"quality_rating": "good", "issues": []}

    def test_json_code_block(self):
        raw = '```json\n{"key": "value"}\n```'
        result = extract_json(raw)
        assert result == {"key": "value"}

    def test_bare_code_block(self):
        raw = '```\n{"key": "value"}\n```'
        result = extract_json(raw)
        assert result == {"key": "value"}

    def test_code_block_with_surrounding_text(self):
        raw = 'Here is my analysis:\n\n```json\n{"rating": "good"}\n```\n\nHope that helps!'
        result = extract_json(raw)
        assert result == {"rating": "good"}

    def test_empty_string(self):
        result = extract_json("")
        assert result is None

    def test_no_json(self):
        result = extract_json("This is just plain text with no JSON at all.")
        assert result is None

    def test_array_json(self):
        raw = '[{"a": 1}, {"b": 2}]'
        result = extract_json(raw)
        assert result == [{"a": 1}, {"b": 2}]


# -- parse_code_review ---------------------------------------------------------


class TestParseCodeReview:
    def test_parse_from_code_block(self):
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

    def test_parse_direct_json(self):
        content = (FIXTURES / "anthropic_code_review.txt").read_text()
        review = parse_code_review(content, provider="anthropic", model="claude-sonnet-4-20250514")

        assert isinstance(review, ProviderReview)
        assert review.provider == "anthropic"
        assert review.model == "claude-sonnet-4-20250514"
        assert review.quality_rating == "good"
        assert review.issues == ()
        assert review.praise == ("Well structured.",)
        assert review.summary == "No issues found."

    def test_malformed_raises_parse_error(self):
        content = (FIXTURES / "malformed.txt").read_text()
        with pytest.raises(ParseError, match="Failed to extract JSON"):
            parse_code_review(content, provider="openai", model="gpt-4o")

    def test_missing_quality_rating_raises_parse_error(self):
        content = json.dumps({"issues": [], "praise": [], "summary": "ok"})
        with pytest.raises(ParseError, match="quality_rating"):
            parse_code_review(content, provider="openai", model="gpt-4o")

    def test_provider_from_wrapper_overrides_body(self):
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
    def test_parse_design_advice(self):
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

    def test_malformed_raises_parse_error(self):
        content = (FIXTURES / "malformed.txt").read_text()
        with pytest.raises(ParseError, match="Failed to extract JSON"):
            parse_design_advice(content, provider="openai", model="gpt-4o")


# -- unparsable response diagnostics -------------------------------------------


class TestUnparsableDiagnostics:
    """The failure message must separate a cut-off reply from prose from an empty one."""

    def test_empty_response_is_named_as_empty(self):
        with pytest.raises(ParseError, match="empty"):
            parse_code_review("", provider="anthropic", model="claude-opus-5")

    def test_cut_off_json_is_named_as_incomplete(self):
        content = '{"quality_rating": "good", "issues": [{"severity": "high", "location": "a.py:1'
        with pytest.raises(ParseError, match="ends mid-structure"):
            parse_code_review(content, provider="anthropic", model="claude-opus-5")

    def test_cut_off_json_inside_a_code_fence_is_named_as_incomplete(self):
        content = '```json\n{"quality_rating": "good", "issues": [{"severity": "hi'
        with pytest.raises(ParseError, match="ends mid-structure"):
            parse_code_review(content, provider="anthropic", model="claude-opus-5")

    def test_prose_is_named_as_carrying_no_json(self):
        content = "I cannot review this code because the request is unclear."
        with pytest.raises(ParseError, match="no JSON found"):
            parse_code_review(content, provider="anthropic", model="claude-opus-5")

    def test_complete_but_invalid_json_is_named_as_malformed(self):
        content = '{"quality_rating": "good",,}'
        with pytest.raises(ParseError, match="malformed"):
            parse_code_review(content, provider="anthropic", model="claude-opus-5")

    def test_message_reports_the_response_length(self):
        content = "x" * 4096
        with pytest.raises(ParseError, match="4096 characters"):
            parse_code_review(content, provider="anthropic", model="claude-opus-5")

    def test_design_advice_reports_the_same_detail(self):
        content = '{"recommendation": "repository", "approaches": [{"name": "Repo'
        with pytest.raises(ParseError, match="ends mid-structure"):
            parse_design_advice(content, provider="anthropic", model="claude-opus-5")

    def test_existing_message_prefix_is_preserved(self):
        with pytest.raises(ParseError, match=r"Failed to extract JSON from anthropic response"):
            parse_code_review("prose only", provider="anthropic", model="claude-opus-5")
