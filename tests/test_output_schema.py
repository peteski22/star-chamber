"""Validate serialized council output against the schema the package ships.

Both artifacts are hand written, so only a test that serializes a real result
and validates it keeps them from drifting apart.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from importlib.resources import files

import jsonschema
import pytest

from star_chamber.parsing import extract_json
from star_chamber.schema import get_schema
from star_chamber.synthesis import Synthesis
from star_chamber.transport import ProviderResponse
from star_chamber.types import ProviderError
from tests.helpers import SYNTHESIS, build_code_review_result, build_design_result

# The schema each example's recorded provider bodies answer to, keyed by example package.
_EXAMPLE_BODY_SCHEMAS = {
    "code_review": "code-review-result",
    "design_question": "design-advice-result",
}


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


async def _failing_synthesizer(material: str, answered: Sequence[str]) -> Synthesis:
    """Stand in for a member that was asked for an assessment and did not answer."""
    return Synthesis(error=ProviderError(provider="openai", error="Timeout."))


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
        result = build_code_review_result(responses, files_reviewed=("app.py",))

        jsonschema.validate(_as_document(result), schema)

    def test_empty_result_validates(self, schema):
        result = build_code_review_result([], files_reviewed=())

        jsonschema.validate(_as_document(result), schema)

    def test_summary_carries_counts_and_synthesis(self, schema):
        responses = [ProviderResponse("openai", "openai", "gpt-4o", True, _review_body([_ISSUE]))]
        result = build_code_review_result(responses, files_reviewed=("app.py",))
        document = _as_document(result)

        jsonschema.validate(document, schema)
        assert document["summary"]["total_issues"] == 1
        assert document["summary"]["consensus_count"] == 1
        assert document["summary"]["majority_count"] == 0
        assert document["summary"]["synthesis"] == SYNTHESIS

    def test_files_reviewed_is_emitted(self, schema):
        result = build_code_review_result([], files_reviewed=("b.py", "a.py"))

        jsonschema.validate(_as_document(result), schema)
        assert _as_document(result)["files_reviewed"] == ["b.py", "a.py"]

    def test_summary_carries_a_null_synthesis_error_when_the_synthesis_arrived(self, schema):
        responses = [ProviderResponse("openai", "openai", "gpt-4o", True, _review_body())]
        result = build_code_review_result(responses, files_reviewed=("app.py",))
        document = _as_document(result)

        jsonschema.validate(document, schema)
        assert document["summary"]["synthesis_error"] is None

    def test_summary_carries_the_member_that_owed_a_synthesis(self, schema):
        responses = [ProviderResponse("openai", "openai", "gpt-4o", True, _review_body())]
        result = build_code_review_result(responses, files_reviewed=("app.py",), synthesizer=_failing_synthesizer)
        document = _as_document(result)

        jsonschema.validate(document, schema)
        assert document["summary"]["synthesis"] == ""
        assert document["summary"]["synthesis_error"] == {"provider": "openai", "error": "Timeout."}


class TestDesignOutput:
    def test_full_result_validates(self, schema):
        responses = [
            ProviderResponse("openai", "openai", "gpt-4o", True, _design_body()),
            ProviderResponse("anthropic", "anthropic", "claude-3", False, error="Timeout."),
        ]
        result = build_design_result(responses, prompt="Which data layer?")

        jsonschema.validate(_as_document(result), schema)

    def test_empty_result_validates(self, schema):
        result = build_design_result([], prompt="Which data layer?")

        jsonschema.validate(_as_document(result), schema)

    def test_summary_carries_synthesis(self, schema):
        responses = [ProviderResponse("openai", "openai", "gpt-4o", True, _design_body())]
        result = build_design_result(responses, prompt="Which data layer?")
        document = _as_document(result)

        jsonschema.validate(document, schema)
        assert document["summary"]["synthesis"] == SYNTHESIS

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
        result = build_design_result(responses, prompt="Which data layer?")

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
        result = build_design_result(responses, prompt="Which data layer?")

        assert result.approaches[0].fit_rating is None
        jsonschema.validate(_as_document(result), schema)


class TestReviewsItems:
    """An item of reviews carries a member's whole parsed review.

    A consumer reads those items directly, so one that omits a field is as
    unusable as one the schema never described.
    """

    def _document(self) -> dict:
        responses = [ProviderResponse("openai", "openai", "gpt-4o", True, _review_body([_ISSUE]))]
        return _as_document(build_code_review_result(responses, files_reviewed=("app.py",)))

    def test_every_field_of_a_real_review_is_declared(self, schema):
        document = self._document()

        jsonschema.validate(document, schema)
        assert set(document["reviews"][0]) == {
            "provider",
            "model",
            "quality_rating",
            "issues",
            "praise",
            "summary",
            "raw_content",
        }

    def test_empty_review_is_rejected(self, schema):
        document = self._document()
        document["reviews"] = [{}]

        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(document, schema)

    @pytest.mark.parametrize(
        "field",
        ["issues", "model", "praise", "provider", "quality_rating", "raw_content", "summary"],
    )
    def test_review_missing_a_field_is_rejected(self, schema, field):
        document = self._document()
        del document["reviews"][0][field]

        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(document, schema)

    def test_review_issue_keeps_the_issue_shape(self, schema):
        document = self._document()
        document["reviews"][0]["issues"][0]["severity"] = "critical"

        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(document, schema)

    def test_design_output_may_not_carry_reviews(self, schema):
        responses = [ProviderResponse("openai", "openai", "gpt-4o", True, _design_body())]
        document = _as_document(build_design_result(responses, prompt="Which data layer?"))
        document["reviews"] = []

        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(document, schema)


class TestShippedExamples:
    """Every example ships inside the package as a canonical document.

    An example that fails its own schema teaches a consumer the wrong shape.
    """

    def _example(self, package: str, filename: str):
        resource = files(f"star_chamber.spec.examples.{package}").joinpath(filename)
        return json.loads(resource.read_text(encoding="utf-8"))

    @pytest.mark.parametrize("package", sorted(_EXAMPLE_BODY_SCHEMAS))
    def test_council_output_validates(self, schema, package):
        jsonschema.validate(self._example(package, "council-output.json"), schema)

    @pytest.mark.parametrize("package", sorted(_EXAMPLE_BODY_SCHEMAS))
    def test_request_validates(self, package):
        request_schema = json.loads(get_schema("review-request"))

        jsonschema.validate(self._example(package, "request.json"), request_schema)

    @pytest.mark.parametrize("package", sorted(_EXAMPLE_BODY_SCHEMAS))
    def test_provider_response_envelopes_validate(self, package):
        envelope_schema = json.loads(get_schema("provider-response"))

        for response in self._example(package, "provider-responses.json"):
            jsonschema.validate(response, envelope_schema)

    @pytest.mark.parametrize(("package", "body_schema"), sorted(_EXAMPLE_BODY_SCHEMAS.items()))
    def test_recorded_bodies_validate(self, package, body_schema):
        response_schema = json.loads(get_schema(body_schema))

        for response in self._example(package, "provider-responses.json"):
            if not response["success"]:
                continue
            jsonschema.validate(extract_json(response["content"]), response_schema)
