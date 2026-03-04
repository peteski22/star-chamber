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
