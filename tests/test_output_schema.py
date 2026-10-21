"""Validate serialized council output against the schema the package ships.

Both artifacts are hand written, so only a test that serializes a real result
and validates it keeps them from drifting apart.
"""

from __future__ import annotations

import dataclasses
import json

import jsonschema
import pytest

from star_chamber.council import _build_code_review_result, _build_design_result
from star_chamber.schema import get_schema
from star_chamber.transport import ProviderResponse


@pytest.fixture
def schema() -> dict:
    return json.loads(get_schema("council-output"))


def _as_document(result: object) -> dict:
    """Serialize a result exactly as the CLI's JSON output does."""
    return json.loads(json.dumps(dataclasses.asdict(result)))  # type: ignore[call-overload]


def _review_body(issues: list[dict] | None = None) -> str:
    return json.dumps(
        {
            "quality_rating": "good",
            "issues": issues or [],
            "praise": ["Clear naming."],
            "summary": "Solid overall.",
        }
    )


def _design_body() -> str:
    return json.dumps(
        {
            "recommendation": "Use a repository.",
            "approaches": [{"name": "Repository", "pros": ["Testable"], "cons": ["Indirection"], "risk_level": "low"}],
            "summary": "Repository fits best.",
        }
    )


_ISSUE = {
    "severity": "high",
    "location": "app.py:10",
    "category": "correctness",
    "description": "Off-by-one.",
    "suggestion": "Use <=.",
}


class TestCodeReviewOutput:
    def test_full_result_validates(self, schema):
        responses = [
            ProviderResponse("openai", "openai", "gpt-4o", True, _review_body([_ISSUE])),
            ProviderResponse("anthropic", "anthropic", "claude-3", True, _review_body([_ISSUE])),
            ProviderResponse("gemini", "gemini", "g-2", False, error="Timeout."),
        ]
        result = _build_code_review_result(responses, files_reviewed=("app.py",))

        jsonschema.validate(_as_document(result), schema)

    def test_empty_result_validates(self, schema):
        result = _build_code_review_result([], files_reviewed=())

        jsonschema.validate(_as_document(result), schema)

    def test_summary_carries_counts_and_synthesis(self, schema):
        responses = [ProviderResponse("openai", "openai", "gpt-4o", True, _review_body([_ISSUE]))]
        result = _build_code_review_result(responses, files_reviewed=("app.py",))
        document = _as_document(result)

        jsonschema.validate(document, schema)
        assert document["summary"]["total_issues"] == 1
        assert document["summary"]["consensus_count"] == 1
        assert document["summary"]["majority_count"] == 0
        assert document["summary"]["synthesis"] == "Solid overall."

    def test_files_reviewed_is_emitted(self, schema):
        result = _build_code_review_result([], files_reviewed=("b.py", "a.py"))

        jsonschema.validate(_as_document(result), schema)
        assert _as_document(result)["files_reviewed"] == ["b.py", "a.py"]


class TestDesignOutput:
    def test_full_result_validates(self, schema):
        responses = [
            ProviderResponse("openai", "openai", "gpt-4o", True, _design_body()),
            ProviderResponse("anthropic", "anthropic", "claude-3", False, error="Timeout."),
        ]
        result = _build_design_result(responses, prompt="Which data layer?")

        jsonschema.validate(_as_document(result), schema)

    def test_empty_result_validates(self, schema):
        result = _build_design_result([], prompt="Which data layer?")

        jsonschema.validate(_as_document(result), schema)

    def test_summary_carries_synthesis(self, schema):
        responses = [ProviderResponse("openai", "openai", "gpt-4o", True, _design_body())]
        result = _build_design_result(responses, prompt="Which data layer?")
        document = _as_document(result)

        jsonschema.validate(document, schema)
        assert document["summary"]["synthesis"] == "Repository fits best."

    def test_absent_consensus_recommendation_validates(self, schema):
        """Providers that disagree leave consensus_recommendation null."""
        responses = [
            ProviderResponse("openai", "openai", "gpt-4o", True, _design_body()),
            ProviderResponse(
                "anthropic",
                "anthropic",
                "claude-3",
                True,
                json.dumps(
                    {
                        "recommendation": "Use an ORM.",
                        "approaches": [{"name": "ORM"}],
                        "summary": "ORM fits best.",
                    }
                ),
            ),
        ]
        result = _build_design_result(responses, prompt="Which data layer?")

        assert result.consensus_recommendation is None
        jsonschema.validate(_as_document(result), schema)

    def test_approach_without_a_fit_rating_validates(self, schema):
        """A provider that does not assess fit leaves fit_rating null."""
        responses = [
            ProviderResponse(
                "openai",
                "openai",
                "gpt-4o",
                True,
                json.dumps(
                    {
                        "recommendation": "Use a repository.",
                        "approaches": [{"name": "Repository"}],
                        "summary": "Fits best.",
                    }
                ),
            )
        ]
        result = _build_design_result(responses, prompt="Which data layer?")

        assert result.approaches[0].fit_rating is None
        jsonschema.validate(_as_document(result), schema)
