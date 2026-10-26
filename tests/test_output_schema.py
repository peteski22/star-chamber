"""Validate serialized council output against the schema the package ships.

Both artifacts are hand written, so only a test that serializes a real result
and validates it keeps them from drifting apart. The package ships the schema
open to keys it does not declare, and the producer answers to the strict
variant derived from it, so these tests validate against the derived variant.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import re
from collections.abc import Callable, Iterator, Sequence
from importlib.resources import files
from typing import Any

import jsonschema
import pytest

from star_chamber.parsing import extract_json
from star_chamber.schema import get_schema, list_schemas, strict_variant
from star_chamber.spec import SPEC_VERSION
from star_chamber.synthesis import Synthesis
from star_chamber.transport import ProviderResponse
from star_chamber.types import ProviderConfig, ProviderError
from tests.helpers import SYNTHESIS, SYNTHESIS_AUTHOR, build_code_review_result, build_design_result

# The schema each example's recorded provider bodies answer to, keyed by example package.
_EXAMPLE_BODY_SCHEMAS = {
    "code_review": "code-review-result",
    "design_question": "design-advice-result",
}

# Objects whose keys a member's display name supplies, so no schema can declare them.
_DYNAMIC_MAPS = ("individual_issues", "quality_ratings")
_DYNAMIC_MAP_PATHS = tuple(f"$.{name}" for name in _DYNAMIC_MAPS)

_RECEIVED_SCHEMAS = sorted(set(list_schemas()) - {"council-output"})

# A key no spec declares, used to find an object the derivation failed to reach.
_CANARY = "a_key_no_spec_declares"

# Every form in which SPEC.md states its own version, each capturing that version.
_VERSION_STATEMENTS = {
    "heading": r"# Council Protocol Specification v([0-9]+\.[0-9]+)",
    "status": r"\*\*Status:\*\* Draft \(v([0-9]+\.[0-9]+)\)",
    "versioning": r"current version is \*\*([0-9]+\.[0-9]+)\*\*",
}

# Keywords the spec forbids the derivation from entering, because a subschema
# they hold constrains the same instance as the schema that carries it.
_IN_PLACE_KEYWORDS = ("allOf", "anyOf", "dependentSchemas", "else", "if", "not", "oneOf", "then")

# The fields a summary reports whether or not a member was asked for the assessment.
_REPORTING_FIELDS = ("synthesis_error", "synthesized_by")

# A recorded failure, as a document carries it, for probing the pairing rules.
_SYNTHESIS_FAILURE = {"provider": "openai", "model": "gpt-4o", "error": "Timeout.", "raw_content": None}


@pytest.fixture
def published() -> dict[str, Any]:
    """The schema as the package ships it, open to keys it does not declare."""
    return json.loads(get_schema("council-output"))


@pytest.fixture
def schema(published: dict[str, Any]) -> dict[str, Any]:
    """The strict variant the producer answers to."""
    return strict_variant(published)


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


async def _failing_synthesizer(material: str, answered: Sequence[ProviderConfig]) -> Synthesis:
    """Stand in for a member that was asked for an assessment and did not answer."""
    return Synthesis(
        author="openai", error=ProviderError(provider="openai", model="gpt-4o", error="Timeout.", raw_content=None)
    )


_ISSUE = {
    "severity": "high",
    "location": "app.py:10",
    "category": "correctness",
    "description": "Off-by-one.",
    "suggestion": "Use <=.",
}

_OTHER_ISSUE = {
    "severity": "low",
    "location": "app.py:40",
    "category": "maintainability",
    "description": "Name is vague.",
    "suggestion": "Rename it.",
}


def _full_code_review_document() -> dict:
    responses = [
        ProviderResponse("openai", "openai", "gpt-4o", True, _review_body([_ISSUE])),
        ProviderResponse("anthropic", "anthropic", "claude-3", True, _review_body([_ISSUE])),
        ProviderResponse("gemini", "gemini", "g-2", False, error="Timeout."),
    ]
    return _as_document(build_code_review_result(responses, files_reviewed=("app.py",)))


def _majority_code_review_document() -> dict:
    """A document carrying a majority issue, raised by two of the three that answered."""
    responses = [
        ProviderResponse("openai", "openai", "gpt-4o", True, _review_body([_ISSUE, _OTHER_ISSUE])),
        ProviderResponse("anthropic", "anthropic", "claude-3", True, _review_body([_ISSUE, _OTHER_ISSUE])),
        ProviderResponse("gemini", "gemini", "g-2", True, _review_body([_ISSUE])),
    ]
    return _as_document(build_code_review_result(responses, files_reviewed=("app.py",)))


def _debating_code_review_document() -> dict:
    """A document carrying the debate object, which a calling orchestrator adds.

    NOTE: This SDK never emits the object, so nothing else probes the shape the
    schema declares for it.
    """
    document = _full_code_review_document()
    document["debate"] = {"rounds_completed": 2, "converged": False}
    return document


def _failed_synthesis_document() -> dict:
    responses = [ProviderResponse("openai", "openai", "gpt-4o", True, _review_body())]
    result = build_code_review_result(responses, files_reviewed=("app.py",), synthesizer=_failing_synthesizer)
    return _as_document(result)


def _unasked_synthesis_document() -> dict:
    """A council no member answered, so none was asked for the assessment."""
    return _as_document(build_code_review_result([], files_reviewed=()))


def _full_design_document() -> dict:
    responses = [
        ProviderResponse("openai", "openai", "gpt-4o", True, _design_body()),
        ProviderResponse("anthropic", "anthropic", "claude-3", False, error="Timeout."),
    ]
    return _as_document(build_design_result(responses, prompt="Which data layer?"))


def _shipped_example(package: str) -> dict:
    resource = files(f"star_chamber.spec.examples.{package}").joinpath("council-output.json")
    return json.loads(resource.read_text(encoding="utf-8"))


# Every document the strict variant is probed against, so each declared object is reached.
_STRICT_DOCUMENTS = {
    "code_review": _full_code_review_document,
    "code_review_example": lambda: _shipped_example("code_review"),
    "debate": _debating_code_review_document,
    "design_question": _full_design_document,
    "design_question_example": lambda: _shipped_example("design_question"),
    "failed_synthesis": _failed_synthesis_document,
    "majority": _majority_code_review_document,
    "unasked_synthesis": _unasked_synthesis_document,
}

# Every object the schema declares a fixed shape for, as a document path reaches it.
_DECLARED_OBJECTS = (
    "$",
    "$.approaches[",
    "$.consensus_issues[",
    "$.debate",
    "$.failed_providers[",
    "$.individual_issues.",
    "$.majority_issues[",
    "$.reviews[",
    "$.summary",
    "$.summary.synthesis_error",
)


class TestCodeReviewOutput:
    def test_full_result_validates(self, schema) -> None:
        responses = [
            ProviderResponse("openai", "openai", "gpt-4o", True, _review_body([_ISSUE])),
            ProviderResponse("anthropic", "anthropic", "claude-3", True, _review_body([_ISSUE])),
            ProviderResponse("gemini", "gemini", "g-2", False, error="Timeout."),
        ]
        result = build_code_review_result(responses, files_reviewed=("app.py",))

        jsonschema.validate(_as_document(result), schema)

    def test_empty_result_validates(self, schema) -> None:
        result = build_code_review_result([], files_reviewed=())

        jsonschema.validate(_as_document(result), schema)

    def test_summary_carries_counts_and_synthesis(self, schema) -> None:
        responses = [ProviderResponse("openai", "openai", "gpt-4o", True, _review_body([_ISSUE]))]
        result = build_code_review_result(responses, files_reviewed=("app.py",))
        document = _as_document(result)

        jsonschema.validate(document, schema)
        assert document["summary"]["total_issues"] == 1
        assert document["summary"]["consensus_count"] == 1
        assert document["summary"]["majority_count"] == 0
        assert document["summary"]["synthesis"] == SYNTHESIS

    def test_files_reviewed_is_emitted(self, schema) -> None:
        result = build_code_review_result([], files_reviewed=("b.py", "a.py"))

        jsonschema.validate(_as_document(result), schema)
        assert _as_document(result)["files_reviewed"] == ["b.py", "a.py"]

    def test_summary_carries_a_null_synthesis_error_when_the_synthesis_arrived(self, schema) -> None:
        responses = [ProviderResponse("openai", "openai", "gpt-4o", True, _review_body())]
        result = build_code_review_result(responses, files_reviewed=("app.py",))
        document = _as_document(result)

        jsonschema.validate(document, schema)
        assert document["summary"]["synthesis_error"] is None

    def test_summary_names_the_member_that_wrote_the_synthesis(self, schema) -> None:
        responses = [ProviderResponse("openai", "openai", "gpt-4o", True, _review_body())]
        result = build_code_review_result(responses, files_reviewed=("app.py",))
        document = _as_document(result)

        jsonschema.validate(document, schema)
        assert document["summary"]["synthesized_by"] == SYNTHESIS_AUTHOR

    def test_summary_names_no_member_when_none_was_asked(self, schema) -> None:
        result = build_code_review_result([], files_reviewed=())
        document = _as_document(result)

        jsonschema.validate(document, schema)
        assert document["summary"]["synthesized_by"] is None

    def test_summary_carries_the_member_that_owed_a_synthesis(self, schema) -> None:
        responses = [ProviderResponse("openai", "openai", "gpt-4o", True, _review_body())]
        result = build_code_review_result(responses, files_reviewed=("app.py",), synthesizer=_failing_synthesizer)
        document = _as_document(result)

        jsonschema.validate(document, schema)
        assert document["summary"]["synthesis"] == ""
        assert document["summary"]["synthesis_error"] == {
            "provider": "openai",
            "model": "gpt-4o",
            "error": "Timeout.",
            "raw_content": None,
        }


class TestDesignOutput:
    def test_full_result_validates(self, schema) -> None:
        responses = [
            ProviderResponse("openai", "openai", "gpt-4o", True, _design_body()),
            ProviderResponse("anthropic", "anthropic", "claude-3", False, error="Timeout."),
        ]
        result = build_design_result(responses, prompt="Which data layer?")

        jsonschema.validate(_as_document(result), schema)

    def test_empty_result_validates(self, schema) -> None:
        result = build_design_result([], prompt="Which data layer?")

        jsonschema.validate(_as_document(result), schema)

    def test_summary_carries_synthesis(self, schema) -> None:
        responses = [ProviderResponse("openai", "openai", "gpt-4o", True, _design_body())]
        result = build_design_result(responses, prompt="Which data layer?")
        document = _as_document(result)

        jsonschema.validate(document, schema)
        assert document["summary"]["synthesis"] == SYNTHESIS

    def test_absent_consensus_recommendation_validates(self, schema) -> None:
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

    def test_approach_nobody_rated_reports_a_null_fit_rating(self, schema) -> None:
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

    def test_approach_that_omits_the_fit_rating_is_rejected(
        self,
        published: dict[str, Any],
        schema: dict[str, Any],
    ) -> None:
        """A null states that nobody rated the approach, and an absent key states nothing."""
        document = _full_design_document()
        del document["approaches"][0]["fit_rating"]

        for variant in (published, schema):
            with pytest.raises(jsonschema.ValidationError):
                jsonschema.validate(document, variant)


class TestReviewsItems:
    """An item of reviews carries a member's whole parsed review.

    A consumer reads those items directly, so one that omits a field is as
    unusable as one the schema never described.
    """

    def _document(self) -> dict:
        responses = [ProviderResponse("openai", "openai", "gpt-4o", True, _review_body([_ISSUE]))]
        return _as_document(build_code_review_result(responses, files_reviewed=("app.py",)))

    def test_every_field_of_a_real_review_is_declared(self, schema) -> None:
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

    def test_empty_review_is_rejected(self, schema) -> None:
        document = self._document()
        document["reviews"] = [{}]

        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(document, schema)

    @pytest.mark.parametrize(
        "field",
        ["issues", "model", "praise", "provider", "quality_rating", "raw_content", "summary"],
    )
    def test_review_missing_a_field_is_rejected(self, schema, field) -> None:
        document = self._document()
        del document["reviews"][0][field]

        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(document, schema)

    def test_review_issue_keeps_the_issue_shape(self, schema) -> None:
        document = self._document()
        document["reviews"][0]["issues"][0]["severity"] = "critical"

        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(document, schema)

    def test_design_output_may_not_carry_reviews(self, schema) -> None:
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
    def test_council_output_validates(self, schema, package) -> None:
        jsonschema.validate(self._example(package, "council-output.json"), schema)

    @pytest.mark.parametrize("package", sorted(_EXAMPLE_BODY_SCHEMAS))
    def test_request_validates(self, package) -> None:
        request_schema = json.loads(get_schema("review-request"))

        jsonschema.validate(self._example(package, "request.json"), request_schema)

    @pytest.mark.parametrize("package", sorted(_EXAMPLE_BODY_SCHEMAS))
    def test_provider_response_envelopes_validate(self, package) -> None:
        envelope_schema = json.loads(get_schema("provider-response"))

        for response in self._example(package, "provider-responses.json"):
            jsonschema.validate(response, envelope_schema)

    @pytest.mark.parametrize(("package", "body_schema"), sorted(_EXAMPLE_BODY_SCHEMAS.items()))
    def test_recorded_bodies_validate(self, package, body_schema) -> None:
        response_schema = json.loads(get_schema(body_schema))

        for response in self._example(package, "provider-responses.json"):
            if not response["success"]:
                continue
            jsonschema.validate(extract_json(response["content"]), response_schema)


def _objects(document: object, path: str = "$") -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield every JSON object within a document, with the path that reaches it."""
    if isinstance(document, dict):
        yield path, document
        for key, value in document.items():
            yield from _objects(value, f"{path}.{key}")
    elif isinstance(document, list):
        for index, item in enumerate(document):
            yield from _objects(item, f"{path}[{index}]")


def _canaried(document: dict) -> Iterator[tuple[str, dict]]:
    """Yield a copy of the document per object in it, each carrying one undeclared key.

    An object keyed by member name accepts any key by design, so it is skipped.
    """
    for index, (path, _) in enumerate(_objects(document)):
        if path in _DYNAMIC_MAP_PATHS:
            continue
        candidate = copy.deepcopy(document)
        _, node = list(_objects(candidate))[index]
        node[_CANARY] = "value"
        yield path, candidate


class TestPublishedSchema:
    """The package ships council-output open, so a consumer on an older copy keeps working.

    A closed schema rejects a document from a newer producer on the fields it
    has never heard of, which every field the protocol has added has done.
    """

    def test_no_object_is_closed(self, published: dict[str, Any]) -> None:
        assert [path for path, node in _objects(published) if node.get("additionalProperties") is False] == []

    def test_a_key_a_newer_producer_adds_is_accepted(self, published: dict[str, Any]) -> None:
        document = _as_document(build_code_review_result([], files_reviewed=()))
        document[_CANARY] = "value"
        document["summary"][_CANARY] = 1

        jsonschema.validate(document, published)

    def test_a_dynamic_map_declares_the_shape_of_its_values(self, published: dict[str, Any]) -> None:
        for name in _DYNAMIC_MAPS:
            assert isinstance(published["properties"][name]["additionalProperties"], dict)

    @pytest.mark.parametrize("name", _RECEIVED_SCHEMAS)
    def test_a_schema_the_council_receives_stays_closed(self, name: str) -> None:
        """Strictness guards untrusted input, so deriving it changes nothing here."""
        received = json.loads(get_schema(name))

        assert strict_variant(received) == received

    def test_a_received_schema_was_discovered(self) -> None:
        """The guard above runs no case at all if the schema names ever change."""
        assert _RECEIVED_SCHEMAS


class TestDerivedStrictSchema:
    """The producer answers to the strict variant the package derives.

    A second hand-written schema can disagree with the published one without
    any test failing, so the strict variant is derived from it.
    """

    @pytest.mark.parametrize("document_name", sorted(_STRICT_DOCUMENTS))
    def test_every_object_rejects_an_undeclared_key(
        self,
        published: dict[str, Any],
        schema: dict[str, Any],
        document_name: str,
    ) -> None:
        """Probe each object of a real document, so no object goes unreached.

        Naming the paths that survived the probe says which object the
        derivation failed to close, which a single assertion per path does not.
        """
        document = _STRICT_DOCUMENTS[document_name]()
        strictly = jsonschema.Draft202012Validator(schema)
        survived: list[str] = []
        probed = 0

        for path, candidate in _canaried(document):
            jsonschema.validate(candidate, published)
            if not any(_CANARY in error.message for error in strictly.iter_errors(candidate)):
                survived.append(path)
            probed += 1

        assert survived == []
        assert probed, f"{document_name} carries no object to probe"

    def test_every_declared_object_is_probed(self) -> None:
        """A declared object no document carries would never have its strictness tested."""
        probed = {path for build in _STRICT_DOCUMENTS.values() for path, _ in _canaried(build())}

        assert [shape for shape in _DECLARED_OBJECTS if not any(p.startswith(shape) for p in probed)] == []

    @pytest.mark.parametrize("keyword", _IN_PLACE_KEYWORDS)
    def test_a_branch_constraining_the_same_instance_is_never_closed(self, keyword: str) -> None:
        """Such a branch describes part of an instance another branch also describes."""
        branch = {"type": "object", "properties": {"a": {"type": "string"}}}
        if keyword == "dependentSchemas":
            held: object = {"trigger": branch}
        elif keyword in ("allOf", "anyOf", "oneOf"):
            held = [branch]
        else:
            held = branch

        derived = strict_variant({"type": "object", "properties": {"b": {"type": "string"}}, keyword: held})

        carried = derived[keyword]
        if keyword == "dependentSchemas":
            reached = carried["trigger"]
        elif isinstance(carried, list):
            reached = carried[0]
        else:
            reached = carried
        assert "additionalProperties" not in reached

    def test_a_nullable_object_is_closed(self) -> None:
        """A union of types still declares an object, so strictness still applies."""
        nullable = {
            "type": "object",
            "properties": {"maybe": {"type": ["object", "null"], "properties": {"a": {"type": "string"}}}},
        }

        assert strict_variant(nullable)["properties"]["maybe"]["additionalProperties"] is False

    def test_an_object_the_producer_may_extend_is_left_open(self) -> None:
        """additionalProperties:true is how an object opts out, so the derivation leaves it."""
        extensible = {"type": "object", "properties": {"a": {"type": "string"}}, "additionalProperties": True}

        assert strict_variant(extensible)["additionalProperties"] is True

    def test_a_dynamic_map_is_left_alone(self, published: dict[str, Any]) -> None:
        derived = strict_variant(published)

        for name in _DYNAMIC_MAPS:
            values = published["properties"][name]["additionalProperties"]
            assert derived["properties"][name]["additionalProperties"] == values

    def test_instance_data_is_left_alone(self) -> None:
        """A default, example, const or enum holds an instance rather than a subschema."""
        shape = {"type": "object", "properties": {"a": {"type": "string"}}}
        derived = strict_variant(
            {
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "default": dict(shape),
                "examples": [dict(shape)],
                "const": dict(shape),
                "enum": [dict(shape)],
            }
        )

        assert "additionalProperties" not in derived["default"]
        assert "additionalProperties" not in derived["examples"][0]
        assert "additionalProperties" not in derived["const"]
        assert "additionalProperties" not in derived["enum"][0]

    def test_deriving_a_strict_schema_again_changes_nothing(self, published: dict[str, Any]) -> None:
        once = strict_variant(published)

        assert strict_variant(once) == once


class TestCompositionReach:
    """Strictness at a base object cannot see a property declared only in a branch.

    `additionalProperties` is composition blind, so a branch states which
    properties are required and never introduces one. These tests hold the
    schema to that, because the derivation is only safe while it holds.
    """

    @pytest.mark.parametrize("owner", ["$", "$.summary"])
    def test_a_branch_introduces_no_property(self, published: dict[str, Any], owner: str) -> None:
        carrier = {"$": published, "$.summary": published["properties"]["summary"]}[owner]
        base = set(carrier["properties"])

        for rule in carrier.get("allOf", []):
            for branch in ("if", "then", "else"):
                carried = rule.get(branch, {})
                assert set(carried.get("properties", {})) <= base
                assert set(carried.get("required", [])) <= base

    def test_a_mode_branch_requires_no_undeclared_summary_field(self, published: dict[str, Any]) -> None:
        declared = set(published["properties"]["summary"]["properties"])

        for rule in published["allOf"]:
            summary = rule.get("then", {}).get("properties", {}).get("summary", {})
            assert set(summary.get("required", [])) <= declared


class TestSynthesisReportingIsRequired:
    """Both synthesis fields are required and nullable, so a document cannot skip the question.

    A null reports an absence the schema documents per field. An absent key
    reports nothing, and a requirement rather than the strictness derivation
    carries that rule, so the copy a consumer holds rejects the omission as
    well.
    """

    @pytest.mark.parametrize("document_name", sorted(_STRICT_DOCUMENTS))
    def test_the_producer_reports_both(self, document_name: str) -> None:
        document = _STRICT_DOCUMENTS[document_name]()

        assert set(_REPORTING_FIELDS) <= set(document["summary"])

    @pytest.mark.parametrize(
        ("field", "build"),
        [
            ("synthesis_error", _full_code_review_document),
            ("synthesized_by", _unasked_synthesis_document),
        ],
        ids=_REPORTING_FIELDS,
    )
    def test_a_null_the_producer_emits_is_accepted(
        self,
        published: dict[str, Any],
        schema: dict[str, Any],
        field: str,
        build: Callable[[], dict],
    ) -> None:
        """Pin nullability against a document the spec permits, so an edit that dropped it fails."""
        document = build()

        assert document["summary"][field] is None
        jsonschema.validate(document, published)
        jsonschema.validate(document, schema)

    @pytest.mark.parametrize("field", _REPORTING_FIELDS)
    @pytest.mark.parametrize("document_name", sorted(_STRICT_DOCUMENTS))
    def test_a_document_that_omits_one_is_rejected(
        self,
        published: dict[str, Any],
        document_name: str,
        field: str,
    ) -> None:
        """Validate against the published copy, because a requirement is not a strictness."""
        document = _STRICT_DOCUMENTS[document_name]()
        del document["summary"][field]

        with pytest.raises(jsonschema.ValidationError) as rejection:
            jsonschema.validate(document, published)

        assert rejection.value.validator == "required"
        assert list(rejection.value.absolute_path) == ["summary"]
        assert field in rejection.value.message

    @pytest.mark.parametrize(
        ("synthesis", "error", "author", "permitted"),
        [
            pytest.param("", None, None, True, id="nothing_was_requested"),
            pytest.param("", None, SYNTHESIS_AUTHOR, False, id="an_author_with_nothing_to_show"),
            pytest.param("", _SYNTHESIS_FAILURE, None, False, id="a_failure_against_no_member"),
            pytest.param("", _SYNTHESIS_FAILURE, SYNTHESIS_AUTHOR, True, id="an_author_that_failed"),
            pytest.param(SYNTHESIS, None, None, False, id="an_assessment_without_an_author"),
            pytest.param(SYNTHESIS, None, SYNTHESIS_AUTHOR, True, id="an_assessment_by_its_author"),
            pytest.param(SYNTHESIS, _SYNTHESIS_FAILURE, None, False, id="a_failure_and_an_assessment"),
            pytest.param(SYNTHESIS, _SYNTHESIS_FAILURE, SYNTHESIS_AUTHOR, False, id="a_failure_beside_an_assessment"),
        ],
    )
    def test_every_combination_of_the_three_fields_is_judged(
        self,
        published: dict[str, Any],
        schema: dict[str, Any],
        synthesis: str,
        error: dict[str, Any] | None,
        author: str | None,
        permitted: bool,
    ) -> None:
        """Enumerate the state space, so a rule covering only part of it fails here."""
        document = _full_code_review_document()
        document["summary"] |= {"synthesis": synthesis, "synthesis_error": error, "synthesized_by": author}

        for variant in (published, schema):
            if permitted:
                jsonschema.validate(document, variant)
                continue

            with pytest.raises(jsonschema.ValidationError) as rejection:
                jsonschema.validate(document, variant)

            assert list(rejection.value.absolute_path)[:1] == ["summary"]

    def test_an_author_named_by_nobody_is_rejected(
        self,
        published: dict[str, Any],
        schema: dict[str, Any],
    ) -> None:
        """An empty name passes a type check and still identifies no member."""
        document = _failed_synthesis_document()
        document["summary"]["synthesized_by"] = ""

        for variant in (published, schema):
            with pytest.raises(jsonschema.ValidationError):
                jsonschema.validate(document, variant)

    def test_the_requirement_is_not_duplicated_into_a_mode_branch(self, published: dict[str, Any]) -> None:
        assert set(_REPORTING_FIELDS) <= set(published["properties"]["summary"]["required"])

        for rule in published["allOf"]:
            branch = rule.get("then", {}).get("properties", {}).get("summary", {})
            assert not set(_REPORTING_FIELDS) & set(branch.get("required", []))


class TestSpecVersion:
    """Output carries the protocol version so a consumer can tell its reading is stale.

    The field is optional, because a document without it predates the field and
    that absence is the answer rather than missing data.
    """

    @pytest.mark.parametrize("document_name", sorted(_STRICT_DOCUMENTS))
    def test_the_producer_emits_the_version_it_implements(
        self,
        schema: dict[str, Any],
        document_name: str,
    ) -> None:
        document = _STRICT_DOCUMENTS[document_name]()

        jsonschema.validate(document, schema)
        assert document["spec_version"] == SPEC_VERSION

    def test_a_document_without_a_version_still_validates(self, published: dict[str, Any]) -> None:
        """A consumer reads the absence, so the schema must not require the field."""
        document = _as_document(build_code_review_result([], files_reviewed=()))
        del document["spec_version"]

        jsonschema.validate(document, published)
        jsonschema.validate(document, strict_variant(published))

    def test_the_version_is_constrained_no_further_than_a_string(self, published: dict[str, Any]) -> None:
        """A pattern here would let a consumer reject the document that says it is stale."""
        declared = published["properties"]["spec_version"]

        assert declared["type"] == "string"
        assert set(declared) == {"description", "type"}

    def test_a_version_the_schema_has_never_seen_is_accepted(self, published: dict[str, Any]) -> None:
        document = _as_document(build_code_review_result([], files_reviewed=()))
        document["spec_version"] = "99.1"

        jsonschema.validate(document, published)
        jsonschema.validate(document, strict_variant(published))


class TestSpecVersionAgreement:
    """Every statement of the protocol version has to agree with the others.

    The spec states its version in three places and the package states it in a
    fourth. A release that advances one and not the rest leaves a document
    claiming a version whose rules it was not built from, and nothing about the
    document reveals it.
    """

    def _spec(self) -> str:
        return files("star_chamber.spec").joinpath("SPEC.md").read_text(encoding="utf-8")

    @pytest.mark.parametrize("template", sorted(_VERSION_STATEMENTS))
    def test_the_spec_states_the_version_the_package_implements(self, template: str) -> None:
        assert re.search(_VERSION_STATEMENTS[template], self._spec()) is not None

    def test_the_spec_states_no_other_version_of_itself(self) -> None:
        """A stale statement left beside a new one is the failure this guards against."""
        spec = self._spec()
        stated = {version for pattern in _VERSION_STATEMENTS.values() for version in re.findall(pattern, spec)}

        assert stated == {SPEC_VERSION}

    @pytest.mark.parametrize("package", sorted(_EXAMPLE_BODY_SCHEMAS))
    def test_a_shipped_example_claims_the_current_version(self, package: str) -> None:
        assert _shipped_example(package)["spec_version"] == SPEC_VERSION


class TestDocumentedFields:
    """Section 10 of the spec is the field list a second orchestrator implements from.

    A field the schema declares and that section omits is emitted by nobody who
    works from the spec, and a field that section describes and the schema does
    not declare is rejected by every consumer. Either way the two artifacts
    disagree and only a reader of both would notice.
    """

    def _documented(self, heading: str) -> set[str]:
        """Return the field names one output section lists, without their shape suffix."""
        spec = files("star_chamber.spec").joinpath("SPEC.md").read_text(encoding="utf-8")
        body = spec.split(heading, 1)[1].split("**Markdown presentation", 1)[0]
        return {name.rstrip("[]{}") for name in re.findall(r"^- `([^`]+)`", body, re.MULTILINE)}

    def _declared(self, published: dict[str, Any]) -> dict[str, set[str]]:
        """Return the field names the schema declares for each mode."""
        every = set(published["properties"])
        declared = {}
        for rule in published["allOf"]:
            mode = rule["if"]["properties"]["mode"]["const"]
            withheld = {entry["required"][0] for entry in rule["then"]["not"]["anyOf"]}
            declared[mode] = every - withheld
        return declared

    @pytest.mark.parametrize(
        ("heading", "mode"),
        [
            ("### 10.1 Code Review Output", "code-review"),
            ("### 10.2 Design Question Output", "design-question"),
        ],
    )
    def test_the_spec_lists_every_field_the_schema_declares(
        self,
        published: dict[str, Any],
        heading: str,
        mode: str,
    ) -> None:
        assert self._documented(heading) == self._declared(published)[mode]


class TestMemberVocabulary:
    """A member answering in its own vocabulary cannot make the document non-conformant.

    That member is recorded as a failure instead, so the document still
    validates against the copy a consumer holds.
    """

    def test_a_review_outside_the_protocol_s_sets_fails_its_member(
        self,
        published: dict[str, Any],
        schema: dict[str, Any],
    ) -> None:
        body = json.dumps(
            {
                "quality_rating": "solid",
                "issues": [
                    {
                        "severity": "critical",
                        "location": "app.py:1",
                        "category": "security",
                        "description": "Off-by-one.",
                        "suggestion": "Use <=.",
                    }
                ],
                "praise": [],
                "summary": "Fine.",
            }
        )
        responses = [ProviderResponse("openai", "openai", "gpt-4o", True, body)]
        document = _as_document(build_code_review_result(responses, files_reviewed=("app.py",)))

        jsonschema.validate(document, published)
        jsonschema.validate(document, schema)
        assert document["providers_responded"] == []
        assert "quality_rating" in document["failed_providers"][0]["error"]

    def test_advice_outside_the_protocol_s_sets_fails_its_member(
        self,
        published: dict[str, Any],
        schema: dict[str, Any],
    ) -> None:
        body = json.dumps(
            {
                "recommendation": "Use a repository.",
                "approaches": [{"name": "Repository", "risk_level": "severe", "fit_rating": "great"}],
                "summary": "Fine.",
            }
        )
        responses = [ProviderResponse("openai", "openai", "gpt-4o", True, body)]
        document = _as_document(build_design_result(responses, prompt="Which data layer?"))

        jsonschema.validate(document, published)
        jsonschema.validate(document, schema)
        assert document["providers_responded"] == []
        assert "risk_level" in document["failed_providers"][0]["error"]


class TestFailureKeepsTheReply:
    """A failed member's reply reaches the output document, which is its only copy.

    An orchestrator holds a reply for the length of one session and nothing
    else persists it, so a failure that records only its own description
    cannot be diagnosed afterwards at all.
    """

    def test_an_unparsed_reply_is_kept_whole(self, schema: dict[str, Any]) -> None:
        truncated = '{"quality_rating": "goo'
        responses = [ProviderResponse("openai", "openai", "gpt-4o", True, truncated)]
        document = _as_document(build_code_review_result(responses, files_reviewed=("app.py",)))

        jsonschema.validate(document, schema)
        assert document["failed_providers"][0]["raw_content"] == truncated

    def test_an_unparsed_reply_is_kept_for_a_design_question(self, schema: dict[str, Any]) -> None:
        truncated = '{"recommendation": "Use a rep'
        responses = [ProviderResponse("openai", "openai", "gpt-4o", True, truncated)]
        document = _as_document(build_design_result(responses, prompt="Which data layer?"))

        jsonschema.validate(document, schema)
        assert document["failed_providers"][0]["raw_content"] == truncated

    def test_a_failure_before_any_reply_reports_null(self, schema: dict[str, Any]) -> None:
        responses = [ProviderResponse("gemini", "gemini", "g-2", False, error="Timeout.")]
        document = _as_document(build_code_review_result(responses, files_reviewed=("app.py",)))

        jsonschema.validate(document, schema)
        assert document["failed_providers"][0]["raw_content"] is None

    def test_a_reply_that_arrived_empty_reports_empty_rather_than_null(self, schema: dict[str, Any]) -> None:
        """An empty reply and no reply are different facts about the member."""
        responses = [ProviderResponse("openai", "openai", "gpt-4o", True, "")]
        document = _as_document(build_code_review_result(responses, files_reviewed=("app.py",)))

        jsonschema.validate(document, schema)
        assert document["failed_providers"][0]["raw_content"] == ""

    def test_a_cut_short_reply_that_arrived_empty_reports_empty(self, schema: dict[str, Any]) -> None:
        """A provider that returned a reply object carrying nothing did still reply."""
        responses = [ProviderResponse("openai", "openai", "gpt-4o", False, content="", error="Token limit.")]
        document = _as_document(build_code_review_result(responses, files_reviewed=("app.py",)))

        jsonschema.validate(document, schema)
        assert document["failed_providers"][0]["raw_content"] == ""

    def test_a_cut_short_reply_is_kept_on_the_failure(self, schema: dict[str, Any]) -> None:
        partial = '{"quality_rating": "goo'
        responses = [ProviderResponse("openai", "openai", "gpt-4o", False, content=partial, error="Token limit.")]
        document = _as_document(build_code_review_result(responses, files_reviewed=("app.py",)))

        jsonschema.validate(document, schema)
        assert document["failed_providers"][0]["raw_content"] == partial

    def test_a_failure_omitting_the_reply_is_rejected(self, schema: dict[str, Any]) -> None:
        responses = [ProviderResponse("gemini", "gemini", "g-2", False, error="Timeout.")]
        document = _as_document(build_code_review_result(responses, files_reviewed=("app.py",)))
        del document["failed_providers"][0]["raw_content"]

        with pytest.raises(jsonschema.ValidationError, match="raw_content"):
            jsonschema.validate(document, schema)

    def test_the_published_schema_requires_it_too(self, published: dict[str, Any]) -> None:
        """Opening the schema for consumers did not relax what a failure must state."""
        responses = [ProviderResponse("gemini", "gemini", "g-2", False, error="Timeout.")]
        document = _as_document(build_code_review_result(responses, files_reviewed=("app.py",)))
        del document["failed_providers"][0]["raw_content"]

        with pytest.raises(jsonschema.ValidationError, match="raw_content"):
            jsonschema.validate(document, published)
