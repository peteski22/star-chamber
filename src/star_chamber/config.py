"""Config loading and validation for star-chamber providers."""

from __future__ import annotations

import json
import os
from pathlib import Path

from star_chamber.types import CouncilConfig, ProviderConfig


class ConfigError(Exception):
    """Raised for invalid or missing configuration."""


def _default_config_path() -> Path:
    """Return the default config file path.

    Uses the ``STAR_CHAMBER_CONFIG`` environment variable when set,
    otherwise falls back to ``~/.config/star-chamber/providers.json``.

    Returns:
        Resolved path to the configuration file.
    """
    env = os.environ.get("STAR_CHAMBER_CONFIG")
    if env:
        return Path(env)
    return Path.home() / ".config" / "star-chamber" / "providers.json"


def _parse_provider(raw: dict) -> ProviderConfig:
    """Build a ProviderConfig from a raw dict, validating required fields.

    Args:
        raw: Dictionary parsed from the providers list entry.

    Returns:
        A validated ProviderConfig instance.

    Raises:
        ConfigError: If required fields are missing.
    """
    missing = [f for f in ("provider", "model") if f not in raw]
    if missing:
        msg = f"Provider entry missing required fields: {', '.join(missing)}"
        raise ConfigError(msg)

    return ProviderConfig(
        provider=raw["provider"],
        model=raw["model"],
        api_key=raw.get("api_key"),
        api_base=raw.get("api_base"),
        max_tokens=raw.get("max_tokens"),
        local=raw.get("local", False),
    )


def load_config(path: Path | None = None) -> CouncilConfig:
    """Load and validate a providers.json configuration file.

    Args:
        path: Explicit path to the config file.  When ``None`` the path
            is resolved via ``_default_config_path()``.

    Returns:
        A validated CouncilConfig instance.

    Raises:
        ConfigError: If the file is missing, contains invalid JSON, or
            fails structural validation.
    """
    if path is None:
        path = _default_config_path()

    if not path.exists():
        msg = f"Config file not found: {path}"
        raise ConfigError(msg)

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"Invalid JSON in {path}: {exc}"
        raise ConfigError(msg) from exc

    if "providers" not in raw:
        msg = f"Config missing required key 'providers' in {path}"
        raise ConfigError(msg)

    providers_raw = raw["providers"]
    if not isinstance(providers_raw, list):
        msg = f"'providers' must be a list in {path}"
        raise ConfigError(msg)

    if len(providers_raw) == 0:
        msg = f"'providers' must contain at least one entry in {path}"
        raise ConfigError(msg)

    providers = tuple(_parse_provider(p) for p in providers_raw)

    return CouncilConfig(
        providers=providers,
        timeout_seconds=raw.get("timeout_seconds", 60),
        consensus_threshold=raw.get("consensus_threshold", 2),
        platform=raw.get("platform"),
    )
