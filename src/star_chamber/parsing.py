"""JSON extraction and typed parsing for LLM provider responses."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from enum import StrEnum
from typing import Any, TypeVar

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

_ChoiceT = TypeVar("_ChoiceT", bound=StrEnum)

# Matches ```json ... ``` or ``` ... ``` code blocks.
_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*\n(.*?)\n```", re.DOTALL)

# Matches an opening fence alone, so a reply cut off before its closing fence still reads as JSON.
_OPEN_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n")

_JSON_START_RE = re.compile(r"[{\[]")

# A plausible wrong value is quoted back whole, and anything longer is reported by its length.
_MAX_ECHOED_VALUE_CHARS = 40

# The risk levels a member may send. UNKNOWN is the council's own value for a level nobody stated.
_MEMBER_RISK_LEVELS = tuple(level for level in RiskLevel if level.stated)

# JSON shape names, so a parse failure describes what the provider sent in protocol terms.
_JSON_TYPE_NAMES: dict[type, str] = {
    bool: "a boolean",
    dict: "an object",
    float: "a number",
    int: "a whole number",
    list: "a list",
    str: "text",
    type(None): "null",
}


class ParseError(Exception):
    """Raised when response parsing fails."""


def _choose(text: str, field: str, context: str, permitted: Iterable[_ChoiceT]) -> _ChoiceT:
    """Return the permitted value the text names.

    Args:
        text: What the member sent.
        field: Field name, as the protocol spells it.
        context: Where the field sits, for the error message.
        permitted: The values a member may send.

    Returns:
        The permitted value the text names.

    Raises:
        ParseError: If the text names no permitted value.
    """
    allowed = tuple(permitted)
    for value in allowed:
        if text == value:
            return value
    msg = f"Field {field} in {context} must be one of {', '.join(allowed)}, got {_describe_value(text)}"
    raise ParseError(msg)


def _describe_unparsable(content: str) -> str:
    """Describe a response that yielded no JSON.

    Distinguishes a reply the provider cut short from one that refused in prose,
    so a caller can act on the failure instead of only seeing that parsing failed.

    Args:
        content: Raw LLM response text that produced no JSON.

    Returns:
        A short phrase naming the response length and its shape.
    """
    if not content.strip():
        return "empty"

    body = _OPEN_FENCE_RE.sub("", content).strip().removesuffix("```").strip()
    length = len(content)

    if not body or _JSON_START_RE.search(body) is None:
        return f"{length} characters, no JSON found"
    if not body.endswith(("}", "]")):
        return f"{length} characters, ends mid-structure"
    return f"{length} characters, malformed JSON"


def _describe_value(text: str) -> str:
    """Quote a rejected value back, reporting its length instead once it is long.

    A member can put a whole completion in a field the protocol says holds one
    short word, and this value reaches the terminal and the output document.

    Args:
        text: What the member sent.

    Returns:
        The value quoted, or a phrase naming its length.
    """
    if len(text) <= _MAX_ECHOED_VALUE_CHARS:
        return repr(text)
    return f"{len(text)} characters of text"


def _json_type_name(value: object) -> str:
    """Name the JSON shape of a value, for a parse failure a reader can act on."""
    return _JSON_TYPE_NAMES.get(type(value), "an unexpected value")


def _object_list(container: dict[str, Any], field: str, context: str) -> list[dict[str, Any]]:
    """Return a list-of-objects field, empty when absent.

    Args:
        container: Decoded JSON object that may hold the field.
        field: Field name, as the protocol spells it.
        context: Where the field sits, for the error message.

    Returns:
        The field's items, or an empty list when the field is absent.

    Raises:
        ParseError: If the field is not a list, or any item is not an object.
    """
    if field not in container:
        return []
    value = container[field]
    if not isinstance(value, list):
        msg = f"Field {field} in {context} must be a list, got {_json_type_name(value)}"
        raise ParseError(msg)
    for item in value:
        if not isinstance(item, dict):
            msg = f"Every entry of {field} in {context} must be an object, got {_json_type_name(item)}"
            raise ParseError(msg)
    return value


def _optional_choice(
    container: dict[str, Any],
    field: str,
    context: str,
    permitted: Iterable[_ChoiceT],
    default: _ChoiceT,
) -> _ChoiceT:
    """Return a closed-set field, or ``default`` when absent.

    Args:
        container: Decoded JSON object that may hold the field.
        field: Field name, as the protocol spells it.
        context: Where the field sits, for the error message.
        permitted: The values a member may send.
        default: Value to use when the field is absent.

    Returns:
        The value the field names, or the default.

    Raises:
        ParseError: If the field is present and names no permitted value.
    """
    if field not in container:
        return default
    return _require_choice(container, field, context, permitted)


def _optional_choice_or_none(
    container: dict[str, Any],
    field: str,
    context: str,
    permitted: Iterable[_ChoiceT],
) -> _ChoiceT | None:
    """Return a closed-set field that may be null, None when absent or null.

    Args:
        container: Decoded JSON object that may hold the field.
        field: Field name, as the protocol spells it.
        context: Where the field sits, for the error message.
        permitted: The values a member may send.

    Returns:
        The value the field names, or None.

    Raises:
        ParseError: If the field is present and names no permitted value.
    """
    if container.get(field) is None:
        return None
    return _require_choice(container, field, context, permitted)


def _optional_text(container: dict[str, Any], field: str, context: str, default: str) -> str:
    """Return a text field, or ``default`` when absent.

    Args:
        container: Decoded JSON object that may hold the field.
        field: Field name, as the protocol spells it.
        context: Where the field sits, for the error message.
        default: Value to use when the field is absent.

    Returns:
        The field's text, or the default.

    Raises:
        ParseError: If the field is present and is not text.
    """
    if field not in container:
        return default
    return _require_text(container, field, context)


def _optional_text_list(container: dict[str, Any], field: str, context: str) -> tuple[str, ...]:
    """Return a text-list field, empty when absent.

    Args:
        container: Decoded JSON object that may hold the field.
        field: Field name, as the protocol spells it.
        context: Where the field sits, for the error message.

    Returns:
        The field's entries, or an empty tuple when the field is absent.

    Raises:
        ParseError: If the field is not a list, or any entry is not text.
    """
    if field not in container:
        return ()
    value = container[field]
    if not isinstance(value, list):
        msg = f"Field {field} in {context} must be a list, got {_json_type_name(value)}"
        raise ParseError(msg)
    for entry in value:
        if not isinstance(entry, str):
            msg = f"Every entry of {field} in {context} must be text, got {_json_type_name(entry)}"
            raise ParseError(msg)
    return tuple(value)


def _optional_text_or_none(container: dict[str, Any], field: str, context: str) -> str | None:
    """Return a text field that may be null, None when absent or null.

    Args:
        container: Decoded JSON object that may hold the field.
        field: Field name, as the protocol spells it.
        context: Where the field sits, for the error message.

    Returns:
        The field's text, or None.

    Raises:
        ParseError: If the field is present and is neither text nor null.
    """
    value = container.get(field)
    if value is None:
        return None
    return _require_text(container, field, context)


def _optional_whole_number(container: dict[str, Any], field: str, context: str, default: int) -> int:
    """Return a whole-number field, or ``default`` when absent.

    Args:
        container: Decoded JSON object that may hold the field.
        field: Field name, as the protocol spells it.
        context: Where the field sits, for the error message.
        default: Value to use when the field is absent.

    Returns:
        The field's value, or the default.

    Raises:
        ParseError: If the field is present and is not a whole number.
    """
    if field not in container:
        return default
    value = container[field]
    # A JSON boolean is an int in Python, and serializes back as true rather than a number.
    if not isinstance(value, int) or isinstance(value, bool):
        msg = f"Field {field} in {context} must be a whole number, got {_json_type_name(value)}"
        raise ParseError(msg)
    return value


def _require_choice(
    container: dict[str, Any],
    field: str,
    context: str,
    permitted: Iterable[_ChoiceT],
) -> _ChoiceT:
    """Return a required field whose value must name one of a closed set.

    Args:
        container: Decoded JSON object that should hold the field.
        field: Field name, as the protocol spells it.
        context: Where the field sits, for the error message.
        permitted: The values a member may send, which is not always every member of the set.

    Returns:
        The value the field names.

    Raises:
        ParseError: If the field is absent, is not text, or names no permitted value.
    """
    return _choose(_require_text(container, field, context), field, context, permitted)


def _require_text(container: dict[str, Any], field: str, context: str) -> str:
    """Return a required text field.

    Args:
        container: Decoded JSON object that should hold the field.
        field: Field name, as the protocol spells it.
        context: Where the field sits, for the error message.

    Returns:
        The field's text.

    Raises:
        ParseError: If the field is absent or is not text.
    """
    if field not in container:
        msg = f"Missing required field {field} in {context}"
        raise ParseError(msg)
    value = container[field]
    if not isinstance(value, str):
        msg = f"Field {field} in {context} must be text, got {_json_type_name(value)}"
        raise ParseError(msg)
    return value


def extract_json(content: str) -> dict[str, Any] | list[Any] | None:
    """Extract JSON from an LLM response string.

    Tries direct ``json.loads`` first, then falls back to regex extraction
    from fenced code blocks.

    Args:
        content: Raw LLM response text.

    Returns:
        Parsed JSON as a dict or list, or ``None`` if no valid JSON is found.
    """
    if not content:
        return None

    # Try direct parse first.
    try:
        parsed = json.loads(content)
        if isinstance(parsed, (dict, list)):
            return parsed
    except json.JSONDecodeError:
        pass

    # Try extracting from code blocks.
    match = _CODE_BLOCK_RE.search(content)
    if match:
        try:
            parsed = json.loads(match.group(1))
            if isinstance(parsed, (dict, list)):
                return parsed
        except json.JSONDecodeError:
            pass

    return None


def parse_code_review(content: str, provider: str, model: str) -> ProviderReview:
    """Parse an LLM response into a typed code review.

    The ``provider`` parameter is authoritative and overrides any provider
    field present in the JSON body (per protocol spec section 8).

    Args:
        content: Raw LLM response text.
        provider: Authoritative provider identifier.
        model: Model name used for this review.

    Returns:
        A frozen ``ProviderReview`` dataclass.

    Raises:
        ParseError: If JSON extraction fails or required fields are missing.
    """
    data = extract_json(content)
    if data is None or not isinstance(data, dict):
        msg = f"Failed to extract JSON from {provider} response ({_describe_unparsable(content)})"
        raise ParseError(msg)

    context = f"{provider} response"
    issue_context = f"an issue in {provider} response"

    quality_rating = _require_choice(data, "quality_rating", context, QualityRating)
    issues = tuple(
        Issue(
            severity=_require_choice(issue, "severity", issue_context, Severity),
            location=_require_text(issue, "location", issue_context),
            category=_require_choice(issue, "category", issue_context, IssueCategory),
            description=_require_text(issue, "description", issue_context),
            suggestion=_require_text(issue, "suggestion", issue_context),
        )
        for issue in _object_list(data, "issues", context)
    )

    return ProviderReview(
        provider=provider,
        model=model,
        quality_rating=quality_rating,
        issues=issues,
        praise=_optional_text_list(data, "praise", context),
        summary=_optional_text(data, "summary", context, ""),
        raw_content=content,
    )


def parse_design_advice(content: str, provider: str, model: str) -> ProviderDesignAdvice:
    """Parse an LLM response into typed design advice.

    The ``provider`` parameter is authoritative and overrides any provider
    field present in the JSON body (per protocol spec section 8).

    Args:
        content: Raw LLM response text.
        provider: Authoritative provider identifier.
        model: Model name used for this advice.

    Returns:
        A frozen ``ProviderDesignAdvice`` dataclass.

    Raises:
        ParseError: If JSON extraction fails or required fields are missing.
    """
    data = extract_json(content)
    if data is None or not isinstance(data, dict):
        msg = f"Failed to extract JSON from {provider} response ({_describe_unparsable(content)})"
        raise ParseError(msg)

    if "recommendation" not in data:
        msg = f"Missing required field recommendation in {provider} response"
        raise ParseError(msg)

    if "approaches" not in data:
        msg = f"Missing required field approaches in {provider} response"
        raise ParseError(msg)

    context = f"{provider} response"
    approach_context = f"an approach in {provider} response"

    recommendation = _require_text(data, "recommendation", context)
    approaches = tuple(
        Approach(
            name=_require_text(approach, "name", approach_context),
            recommended_by=_optional_whole_number(approach, "recommended_by", approach_context, 0),
            pros=_optional_text_list(approach, "pros", approach_context),
            cons=_optional_text_list(approach, "cons", approach_context),
            risk_level=_optional_choice(
                approach, "risk_level", approach_context, _MEMBER_RISK_LEVELS, RiskLevel.UNKNOWN
            ),
            fit_rating=_optional_choice_or_none(approach, "fit_rating", approach_context, FitRating),
        )
        for approach in _object_list(data, "approaches", context)
    )

    return ProviderDesignAdvice(
        provider=provider,
        model=model,
        recommendation=recommendation,
        approaches=approaches,
        summary=_optional_text(data, "summary", context, ""),
        raw_content=content,
    )
