"""JSON extraction and typed parsing for LLM provider responses."""

from __future__ import annotations

import json
import re
from typing import Any

from star_chamber.types import Approach, Issue, ProviderDesignAdvice, ProviderReview

# Matches ```json ... ``` or ``` ... ``` code blocks.
_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*\n(.*?)\n```", re.DOTALL)


class ParseError(Exception):
    """Raised when response parsing fails."""


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
        msg = f"Failed to extract JSON from {provider} response"
        raise ParseError(msg)

    if "quality_rating" not in data:
        msg = f"Missing required field quality_rating in {provider} response"
        raise ParseError(msg)

    issues = tuple(
        Issue(
            severity=issue["severity"],
            location=issue["location"],
            category=issue["category"],
            description=issue["description"],
            suggestion=issue["suggestion"],
        )
        for issue in data.get("issues", [])
    )

    return ProviderReview(
        provider=provider,
        model=model,
        quality_rating=data["quality_rating"],
        issues=issues,
        praise=tuple(data.get("praise", [])),
        summary=data.get("summary", ""),
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
        msg = f"Failed to extract JSON from {provider} response"
        raise ParseError(msg)

    if "recommendation" not in data:
        msg = f"Missing required field recommendation in {provider} response"
        raise ParseError(msg)

    if "approaches" not in data:
        msg = f"Missing required field approaches in {provider} response"
        raise ParseError(msg)

    approaches = tuple(
        Approach(
            name=approach["name"],
            recommended_by=approach.get("recommended_by", 0),
            pros=tuple(approach.get("pros", [])),
            cons=tuple(approach.get("cons", [])),
            risk_level=approach.get("risk_level", "unknown"),
            fit_rating=approach.get("fit_rating"),
        )
        for approach in data["approaches"]
    )

    return ProviderDesignAdvice(
        provider=provider,
        model=model,
        recommendation=data["recommendation"],
        approaches=approaches,
        summary=data.get("summary", ""),
        raw_content=content,
    )
