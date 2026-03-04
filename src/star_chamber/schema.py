"""Access council protocol spec files shipped as package data."""

from __future__ import annotations

from importlib.resources import files

_SCHEMAS_PKG = "star_chamber.spec.schemas"


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
