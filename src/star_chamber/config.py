"""Config loading and validation for star-chamber providers."""

from __future__ import annotations

import json
import os
from pathlib import Path

from star_chamber.types import CouncilConfig, OtariConfig, ProviderConfig


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
        ConfigError: If required fields are missing or display_name is invalid.
    """
    missing = [f for f in ("provider", "model") if f not in raw]
    if missing:
        msg = f"Provider entry missing required fields: {', '.join(missing)}"
        raise ConfigError(msg)

    display_name = raw.get("display_name")
    if display_name is not None and (not isinstance(display_name, str) or not display_name.strip()):
        msg = "Provider entry has an invalid 'display_name': must be a non-empty string"
        raise ConfigError(msg)

    return ProviderConfig(
        provider=raw["provider"],
        model=raw["model"],
        api_key=raw.get("api_key"),
        api_base=raw.get("api_base"),
        max_tokens=raw.get("max_tokens"),
        local=raw.get("local", False),
        display_name=display_name,
    )


def _validate_distinct_identities(providers: tuple[ProviderConfig, ...]) -> None:
    """Reject a display name that collides with another member's identity.

    Members sharing a provider without a display name keep loading (pre-existing
    behaviour), but an explicit display name must yield a unique identity —
    colliding members would silently merge again in consensus classification
    and the aggregation maps, which is exactly what display names exist to
    prevent.

    Args:
        providers: Parsed provider configurations.

    Raises:
        ConfigError: If an explicit display name collides with another
            member's identity.
    """
    identities = [p.display_name or p.provider for p in providers]
    duplicates = sorted(
        {p.display_name for p in providers if p.display_name is not None and identities.count(p.display_name) > 1}
    )
    if duplicates:
        msg = f"Duplicate provider identities are not allowed: {', '.join(duplicates)}"
        raise ConfigError(msg)


def _parse_otari(raw: dict) -> OtariConfig:
    """Build an OtariConfig from a raw dict.

    Args:
        raw: Dictionary parsed from the top-level "otari" key.

    Returns:
        A validated OtariConfig instance.

    Raises:
        ConfigError: If unexpected fields are present.
    """
    allowed = {"api_base", "api_key"}
    extra = set(raw) - allowed
    if extra:
        msg = f"Unknown 'otari' fields: {', '.join(sorted(extra))}"
        raise ConfigError(msg)
    return OtariConfig(api_base=raw.get("api_base"), api_key=raw.get("api_key"))


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

    if not isinstance(raw, dict):
        msg = f"Config must be a JSON object in {path}"
        raise ConfigError(msg)

    if "platform" in raw:
        msg = (
            f"'platform' is no longer supported in {path}. "
            "Replace it with a top-level 'otari' object — see the README and SPEC for the migration."
        )
        raise ConfigError(msg)

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
    _validate_distinct_identities(providers)

    otari_raw = raw.get("otari")
    if otari_raw is not None and not isinstance(otari_raw, dict):
        msg = f"'otari' must be an object in {path}"
        raise ConfigError(msg)
    otari = _parse_otari(otari_raw) if otari_raw is not None else None

    return CouncilConfig(
        providers=providers,
        timeout_seconds=raw.get("timeout_seconds", 60),
        consensus_threshold=raw.get("consensus_threshold", 2),
        otari=otari,
    )
