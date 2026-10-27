"""Hold each shape council-output declares inline to the shipped shape it duplicates.

The output schema declares a member's review, an issue, and an approach inline,
because each one is a shipped shape plus the fields the council adds to it.
Nothing ties the two descriptions together, so a change to one side drifts
silently, and the two sides sit on opposite sides of the open-closed boundary,
where a difference in openness hides a difference in content. These tests pin
the fields a shape declares, the fields it requires, and the values its enums
allow.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

import pytest

from star_chamber.prompt import render_code_review_prompt, render_design_prompt
from star_chamber.schema import get_schema
from star_chamber.types import FitRating, IssueCategory, QualityRating, RiskLevel, Severity


@dataclass(frozen=True)
class _InlineCopy:
    """One shape council-output declares inline, and the shipped shape it duplicates.

    NOTE: A comparison covers the fields a shape declares and the values its
    enums allow. It leaves the rest of a field's subschema alone.
    """

    inline_path: tuple[str, ...]
    source_schema: str
    source_path: tuple[str, ...]
    required_adds: frozenset[str]
    optional_adds: frozenset[str] = frozenset()
    widens: tuple[tuple[str, frozenset[str | None]], ...] = ()


# The shapes council-output duplicates, with the fields the council adds to each.
_INLINE_COPIES = {
    "approach": _InlineCopy(
        inline_path=("properties", "approaches", "items"),
        source_schema="design-advice-result",
        source_path=("properties", "approaches", "items"),
        required_adds=frozenset({"recommended_by"}),
        widens=(("fit_rating", frozenset({None})), ("risk_level", frozenset({"unknown"}))),
    ),
    "issue": _InlineCopy(
        inline_path=("$defs", "issue"),
        source_schema="code-review-result",
        source_path=("properties", "issues", "items"),
        required_adds=frozenset(),
    ),
    "majority_issue": _InlineCopy(
        inline_path=("properties", "majority_issues", "items"),
        source_schema="council-output",
        source_path=("$defs", "issue"),
        required_adds=frozenset({"flagged_by", "provider_count"}),
    ),
    "review": _InlineCopy(
        inline_path=("properties", "reviews", "items"),
        source_schema="code-review-result",
        source_path=(),
        required_adds=frozenset({"model", "raw_content"}),
    ),
}


@dataclass(frozen=True)
class _Vocabulary:
    """One closed set this package names, and the shipped schema that defines what a member may send."""

    members: tuple[str, ...]
    source_schema: str
    source_path: tuple[str, ...]


# Each closed set the package names in Python, beside the schema that defines it.
_VOCABULARIES = {
    "category": _Vocabulary(
        members=tuple(IssueCategory),
        source_schema="code-review-result",
        source_path=("properties", "issues", "items", "properties", "category"),
    ),
    "fit_rating": _Vocabulary(
        members=tuple(FitRating),
        source_schema="design-advice-result",
        source_path=("properties", "approaches", "items", "properties", "fit_rating"),
    ),
    "quality_rating": _Vocabulary(
        members=tuple(QualityRating),
        source_schema="code-review-result",
        source_path=("properties", "quality_rating"),
    ),
    "risk_level": _Vocabulary(
        members=tuple(level for level in RiskLevel if level.stated),
        source_schema="design-advice-result",
        source_path=("properties", "approaches", "items", "properties", "risk_level"),
    ),
    "severity": _Vocabulary(
        members=tuple(Severity),
        source_schema="code-review-result",
        source_path=("properties", "issues", "items", "properties", "severity"),
    ),
}


def _shape(name: str, path: tuple[str, ...]) -> dict[str, Any]:
    """Load a shipped schema and walk to the shape the path names."""
    node = json.loads(get_schema(name))
    for key in path:
        node = node[key]
    return node


class TestInlineCopies:
    """A copy declares the source plus named fields, so drift on either side fails here."""

    @pytest.mark.parametrize("copied", sorted(_INLINE_COPIES))
    def test_the_copy_declares_the_source_plus_what_the_council_adds(self, copied: str) -> None:
        duplicate = _INLINE_COPIES[copied]
        inline = _shape("council-output", duplicate.inline_path)
        source = _shape(duplicate.source_schema, duplicate.source_path)

        added = duplicate.required_adds | duplicate.optional_adds

        assert set(inline["properties"]) == set(source["properties"]) | added

    @pytest.mark.parametrize("copied", sorted(_INLINE_COPIES))
    def test_the_copy_requires_what_the_source_requires(self, copied: str) -> None:
        duplicate = _INLINE_COPIES[copied]
        inline = _shape("council-output", duplicate.inline_path)
        source = _shape(duplicate.source_schema, duplicate.source_path)

        assert set(inline["required"]) == set(source["required"]) | duplicate.required_adds

    @pytest.mark.parametrize("copied", sorted(_INLINE_COPIES))
    def test_the_copy_allows_the_source_values_and_what_aggregation_adds(self, copied: str) -> None:
        """Aggregation adds a value for a verdict no member gave, and adds nothing else."""
        duplicate = _INLINE_COPIES[copied]
        inline = _shape("council-output", duplicate.inline_path)
        source = _shape(duplicate.source_schema, duplicate.source_path)
        widened = dict(duplicate.widens)

        for name, declared in source["properties"].items():
            carried = inline["properties"][name]
            if "enum" not in declared and "enum" not in carried:
                continue

            allowed = set(declared.get("enum", ())) | set(widened.get(name, ()))

            assert set(carried.get("enum", ())) == allowed, name

    def test_the_rating_map_allows_the_ratings_a_member_reports(self) -> None:
        """The map keyed by member name repeats the rating set rather than referencing it."""
        rated = _shape("council-output", ("properties", "quality_ratings", "additionalProperties"))
        source = _shape("code-review-result", ("properties", "quality_rating"))

        assert set(rated["enum"]) == set(source["enum"])

    @pytest.mark.parametrize("copied", sorted(_INLINE_COPIES))
    def test_the_added_fields_are_the_council_s_own(self, copied: str) -> None:
        """The added set is the contribution, so a name already in the source is a mistake."""
        duplicate = _INLINE_COPIES[copied]
        source = _shape(duplicate.source_schema, duplicate.source_path)

        assert duplicate.required_adds.isdisjoint(source["properties"])


class TestVocabularies:
    """A closed set is written in Python and in a shipped schema, so the two are pinned together.

    The parse boundary validates a member's value against the Python set, and
    a consumer validates the document against the schema, so the sets drifting
    apart means a reply one accepts the other rejects.
    """

    @pytest.mark.parametrize("named", sorted(_VOCABULARIES))
    def test_the_package_names_what_a_member_may_send(self, named: str) -> None:
        vocabulary = _VOCABULARIES[named]
        declared = _shape(vocabulary.source_schema, vocabulary.source_path)

        assert set(vocabulary.members) == set(declared["enum"])

    def test_the_council_adds_the_level_no_member_states(self) -> None:
        """The output carries one risk level a member may not send, for an approach nobody rated."""
        declared = _shape("council-output", ("properties", "approaches", "items", "properties", "risk_level"))

        assert set(RiskLevel) == set(declared["enum"])


class TestPromptVocabularies:
    """The request templates list the same closed sets, so they are pinned to the types as well.

    A value the template invites and the set omits now fails the whole reply,
    so the two drifting apart costs a member's whole review rather than one
    field.
    """

    @pytest.mark.parametrize("named", sorted(_VOCABULARIES))
    def test_the_template_invites_what_a_member_may_send(self, named: str) -> None:
        asked = render_code_review_prompt(files={"app.py": "x = 1"}) + render_design_prompt(question="Which layer?")
        listed = re.search(rf'"{named}": "([^"]+)"', asked)

        assert listed is not None, f"no template lists {named}"
        assert {value.strip() for value in listed.group(1).split("|")} == set(_VOCABULARIES[named].members)


class TestSpecNamesTheClosedSets:
    """The spec names the constrained fields in prose, which the schemas define.

    An orchestrator implements the rule from that sentence, so a field whose
    schema gains an enum and whose name never reaches the sentence is a rule
    nobody outside this package can follow.
    """

    def _constrained(self, name: str) -> set[str]:
        """Each field the schema constrains to a closed set."""
        found: set[str] = set()
        stack: list[Any] = [json.loads(get_schema(name))]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                for key, value in node.items():
                    if isinstance(value, dict) and "enum" in value:
                        found.add(key)
                    stack.append(value)
            elif isinstance(node, list):
                stack.extend(node)
        return found

    def test_the_spec_names_every_field_a_member_schema_constrains(self) -> None:
        spec = (files("star_chamber.spec") / "SPEC.md").read_text(encoding="utf-8")
        section = spec.split("### Closed Sets")[1].split("###")[0]
        named = set(re.findall(r"`([a-z_]+)`", section.split("\n\n")[1]))
        constrained = self._constrained("code-review-result") | self._constrained("design-advice-result")

        assert constrained <= named
