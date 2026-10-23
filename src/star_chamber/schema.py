"""Access council protocol spec files shipped as package data."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from importlib.resources import files
from typing import Any

_SCHEMAS_PKG = "star_chamber.spec.schemas"

# Keywords holding a mapping of name to subschema, each describing its own instance.
_SUBSCHEMA_MAPPINGS = ("$defs", "patternProperties", "properties")

# Keywords holding a list of subschemas, each describing its own instance.
_SUBSCHEMA_SEQUENCES = ("prefixItems",)

# Keywords holding one subschema describing its own instance.
_SUBSCHEMA_VALUES = (
    "additionalProperties",
    "contains",
    "items",
    "propertyNames",
    "unevaluatedProperties",
)


def list_schemas() -> list[str]:
    """List available schema names (without .schema.json suffix).

    Returns:
        Sorted list of schema names.
    """
    schemas_dir = files(_SCHEMAS_PKG)
    names: list[str] = []
    for item in schemas_dir.iterdir():
        name = item.name
        if name.endswith(".schema.json"):
            names.append(name.removesuffix(".schema.json"))
    return sorted(names)


def get_schema(name: str) -> str:
    """Read a schema file by name and return its contents as a string.

    Args:
        name: Schema name without the .schema.json suffix
            (e.g. "code-review-result").

    Returns:
        The JSON schema file contents as a string.

    Raises:
        FileNotFoundError: If the schema does not exist.
    """
    schemas_dir = files(_SCHEMAS_PKG)
    filename = f"{name}.schema.json"
    schema_file = schemas_dir.joinpath(filename)
    if not schema_file.is_file():
        available = ", ".join(list_schemas())
        msg = f"Schema '{name}' not found. Available: {available}"
        raise FileNotFoundError(msg)
    return schema_file.read_text(encoding="utf-8")


def strict_variant(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Derive the variant of a published schema that a producer answers to.

    A published schema that describes council output states no
    `additionalProperties`, so a consumer validating against an older copy
    accepts a document from a newer producer. This derives the strict reading
    of the same schema, which rejects any key the schema does not declare.

    Args:
        schema: A schema the package publishes.

    Returns:
        A copy stating `additionalProperties: false` on every subschema that
        declares `properties` for an instance of its own and states no
        `additionalProperties` already.

    NOTE: The derivation never enters a subschema that constrains the same
    instance as the schema carrying it, such as an `if`, `then` or `allOf`
    branch. Such a branch describes part of an instance that another branch
    also describes, so closing it rejects the keys that other branch declares.
    A schema therefore gains no strictness for a shape it declares only inside
    such a branch, and must declare the shape where the instance is described
    to gain it.
    """
    variant = copy.deepcopy(dict(schema))
    _close(variant)
    return variant


def _close(node: Any) -> None:
    """State strictness on a subschema describing its own instance, and recurse.

    NOTE: Recursion follows the keywords above and no others. A keyword that
    constrains the instance in place, and one whose contents are an instance
    rather than a subschema, are both left alone because neither is listed.
    """
    if not isinstance(node, dict):
        return

    if "properties" in node and "additionalProperties" not in node:
        node["additionalProperties"] = False

    for keyword in _SUBSCHEMA_MAPPINGS:
        held = node.get(keyword)
        if isinstance(held, dict):
            for subschema in held.values():
                _close(subschema)

    for keyword in _SUBSCHEMA_SEQUENCES:
        held = node.get(keyword)
        if isinstance(held, list):
            for subschema in held:
                _close(subschema)

    for keyword in _SUBSCHEMA_VALUES:
        held = node.get(keyword)
        if isinstance(held, dict):
            _close(held)
