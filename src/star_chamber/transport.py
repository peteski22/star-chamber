"""Provider transport layer with parallel fan-out across LLM providers."""

from __future__ import annotations

import asyncio
import dataclasses
import os
import re
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

from star_chamber.config import ConfigError
from star_chamber.types import OtariConfig, ProviderConfig

# Default maximum token limit when none is configured.
DEFAULT_MAX_TOKENS = 16384

# Patterns for redacting API keys from error messages.
_API_KEY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"Bearer\s+\S+"),
    re.compile(r"sk-[A-Za-z0-9_-]{10,}"),
    re.compile(r"key-[A-Za-z0-9_-]{10,}"),
    re.compile(r"api[_-]?key[=:]\s*\S+", re.IGNORECASE),
    re.compile(r"x{0}[A-Za-z0-9]{32,}", re.IGNORECASE),
]

_ENV_VAR_REFERENCE = re.compile(r"\$\{([^}]+)}")

# The gateway client reads these, in order, when it receives no base URL.
_OTARI_API_BASE_ENV_VARS = ("OTARI_API_BASE", "GATEWAY_API_BASE")

# Longest first, so a base that ends in /api/v1 loses the whole API path.
_OTARI_API_PATHS = ("/api/v1", "/v1")


@dataclass(frozen=True)
class ProviderResponse:
    """Response from a single LLM provider.

    Attributes:
        provider: The provider configured for this member (e.g. "openai",
            "openrouter"), preserved from the request. Not necessarily the
            effective route: under Otari routing the call is dispatched through
            Otari while this field keeps the configured provider.
        display_name: Member identity used as the key in council output.
            Equals the configured display name, or the provider name when none
            is configured.
        model: Model name used.
        success: Whether the call succeeded.
        content: Response content on success.
        error: Error message on failure.
    """

    provider: str
    display_name: str
    model: str
    success: bool
    content: str = ""
    error: str = ""


def _sanitize_error(message: str) -> str:
    """Redact API keys from an error message.

    Args:
        message: Raw error message that may contain secrets.

    Returns:
        Sanitized message with keys replaced by REDACTED.
    """
    result = message
    for pattern in _API_KEY_PATTERNS:
        result = pattern.sub("REDACTED", result)
    return result


def _is_auth_error(error_msg: str) -> bool:
    """Check whether an error message indicates an authentication failure.

    Args:
        error_msg: Error message to inspect.

    Returns:
        True if the message looks like an auth error.
    """
    lowered = error_msg.lower()
    indicators = ("api_key", "unauthorized", "401", "api key", "apikey")
    return any(indicator in lowered for indicator in indicators)


async def send_to_provider(
    config: ProviderConfig,
    prompt: str,
    timeout: float | None = None,
    otari: OtariConfig | None = None,
) -> ProviderResponse:
    """Send a prompt to a single provider.

    When ``otari`` is set and ``config.local`` is False, the call is routed
    through Otari using Otari's api_base/api_key instead of the provider's own
    routing. Local providers always bypass Otari.

    Args:
        config: Provider configuration.
        prompt: The prompt to send.
        timeout: Optional per-call timeout in seconds.
        otari: Optional Otari gateway routing configuration.

    Returns:
        A ProviderResponse indicating success or failure.
    """
    # The output identity: the configured display name, or the provider name.
    identity = config.display_name or config.provider

    try:
        import any_llm  # noqa: F811
    except ImportError:
        return ProviderResponse(
            provider=config.provider,
            display_name=identity,
            model=config.model,
            success=False,
            error="any_llm package is not installed. Install it with: pip install any-llm-sdk",
        )

    max_tok = config.max_tokens or DEFAULT_MAX_TOKENS

    routed_via_otari = otari is not None and not config.local
    effective_provider = "otari" if routed_via_otari else config.provider

    kwargs: dict[str, object] = {
        "model": config.model,
        "provider": effective_provider,
        "messages": [{"role": "user", "content": prompt}],
    }

    # OpenAI uses max_completion_tokens; others (including Otari) use max_tokens.
    if effective_provider == "openai":
        kwargs["max_completion_tokens"] = max_tok
    else:
        kwargs["max_tokens"] = max_tok

    if routed_via_otari:
        assert otari is not None  # narrowing for type checker
        if otari.api_key is not None:
            kwargs["api_key"] = otari.api_key
        if otari.api_base is not None:
            kwargs["api_base"] = otari.api_base
    else:
        if config.api_key is not None:
            kwargs["api_key"] = config.api_key
        if config.api_base is not None:
            kwargs["api_base"] = config.api_base

    if timeout is not None:
        kwargs["timeout"] = timeout

    try:
        response = await any_llm.acompletion(**kwargs)
    except TimeoutError:
        return ProviderResponse(
            provider=config.provider,
            display_name=identity,
            model=config.model,
            success=False,
            error=f"Timeout: provider {config.provider} did not respond in time.",
        )
    except Exception as exc:
        error_msg = str(exc)
        if _is_auth_error(error_msg):
            sanitized = _sanitize_error(error_msg)
            if routed_via_otari:
                detail = "Authentication failed at the Otari gateway. Check your Otari API key"
            elif config.local:
                detail = f"Authentication failed locally. Check your API key for {config.provider}"
            else:
                detail = f"Authentication failed for cloud provider. Check your API key for {config.provider}"
            return ProviderResponse(
                provider=config.provider,
                display_name=identity,
                model=config.model,
                success=False,
                error=f"{detail}: {sanitized}",
            )
        return ProviderResponse(
            provider=config.provider,
            display_name=identity,
            model=config.model,
            success=False,
            error=_sanitize_error(error_msg),
        )

    if not response.choices:
        return ProviderResponse(
            provider=config.provider,
            display_name=identity,
            model=config.model,
            success=False,
            error=f"No response content (empty choices) from {config.provider}.",
        )

    content = response.choices[0].message.content
    return ProviderResponse(
        provider=config.provider,
        display_name=identity,
        model=config.model,
        success=True,
        content=content,
    )


async def fan_out(
    configs: tuple[ProviderConfig, ...],
    prompt: str,
    timeout: float | None = None,
    otari: OtariConfig | None = None,
) -> list[ProviderResponse]:
    """Send a prompt to all providers in parallel.

    Args:
        configs: Tuple of provider configurations.
        prompt: The prompt to broadcast.
        timeout: Optional per-provider timeout in seconds.
        otari: Optional otari routing configuration.

    Returns:
        List of ProviderResponse objects, one per provider.
    """
    tasks = [send_to_provider(cfg, prompt, timeout=timeout, otari=otari) for cfg in configs]
    return list(await asyncio.gather(*tasks))


def resolve_api_keys(
    configs: tuple[ProviderConfig, ...],
) -> tuple[ProviderConfig, ...]:
    """Resolve ``${ENV_VAR}`` references in provider API keys.

    Always returns NEW ProviderConfig objects; never mutates input.

    Args:
        configs: Tuple of provider configurations.

    Returns:
        Tuple of new ProviderConfig objects with resolved keys.
    """
    resolved: list[ProviderConfig] = []
    for cfg in configs:
        api_key = cfg.api_key
        if api_key is not None:
            api_key = _expand_env_var(api_key)
        resolved.append(dataclasses.replace(cfg, api_key=api_key))
    return tuple(resolved)


def resolve_otari(otari: OtariConfig | None) -> OtariConfig | None:
    """Resolve an Otari configuration against the environment.

    Returns a new OtariConfig; never mutates input.  Returns None when
    ``otari`` is None.  An explicit ``api_base`` or ``api_key`` may use a
    ``${ENV_VAR}`` reference, which is expanded.  When either field is
    None, it stays None so the SDK's OtariProvider can auto-detect
    credentials from its own env vars (OTARI_API_KEY, GATEWAY_API_KEY,
    etc.).

    Args:
        otari: Optional Otari configuration.

    Returns:
        A new OtariConfig with resolved fields, or None.

    Raises:
        ConfigError: If the effective base URL is malformed or ends in an API path.
    """
    if otari is None:
        return None

    validate_otari_api_base(otari)

    api_base = None if otari.api_base is None else _expand_env_var(otari.api_base)
    api_key = None if otari.api_key is None else _expand_env_var(otari.api_key)

    return OtariConfig(api_base=api_base, api_key=api_key)


def validate_otari_api_base(otari: OtariConfig | None) -> None:
    """Reject an Otari base URL that already ends in an API path.

    The gateway client adds the API path itself, so such a base fails every call as not found.
    Rejecting it up front gives one error that names the setting to change and its corrected value.
    The check applies to the base the gateway client would use, including one read from the environment.

    Args:
        otari: Optional Otari configuration, before environment expansion.

    Raises:
        ConfigError: If the effective base URL is malformed or ends in an API path.
    """
    if otari is None:
        return

    source = _otari_api_base_source(otari)
    if source is None:
        return

    setting, api_base = source
    try:
        origin = _without_api_path(api_base)
    except ValueError as exc:
        msg = (
            f"Otari API base '{api_base}' from {setting} is not a valid URL ({exc}). "
            f"Set {setting} to the gateway origin."
        )
        raise ConfigError(msg) from exc
    if origin is None:
        return

    msg = (
        f"Otari API base '{api_base}' from {setting} ends in an API path. "
        "The gateway client adds the API path itself, so the base must be the gateway origin. "
        f"Set {setting} to '{origin}'."
    )
    raise ConfigError(msg)


def _otari_api_base_source(otari: OtariConfig) -> tuple[str, str] | None:
    """Return the setting that supplies the Otari base URL and its value, or None when none does."""
    if otari.api_base:
        reference = _ENV_VAR_REFERENCE.fullmatch(otari.api_base)
        if reference is None:
            return "'otari.api_base' in the config", otari.api_base
        value = os.environ.get(reference.group(1))
        if value:
            return reference.group(1), value

    # An empty base, including an unset reference, leaves the gateway client to read the environment.
    for name in _OTARI_API_BASE_ENV_VARS:
        value = os.environ.get(name)
        if value:
            return name, value
    return None


def _without_api_path(url: str) -> str | None:
    """Return ``url`` without its trailing API path, or None when it has none."""
    parts = urlsplit(url)
    path = parts.path.rstrip("/")
    for api_path in _OTARI_API_PATHS:
        if path.endswith(api_path):
            return urlunsplit(parts._replace(path=path.removesuffix(api_path)))
    return None


def _expand_env_var(value: str) -> str:
    """Expand a ``${VAR}`` template from the environment.

    Args:
        value: A string that may contain a ``${VAR}`` reference.

    Returns:
        The resolved value, or an empty string if the variable is not set.
        Non-template strings are returned unchanged.
    """
    match = _ENV_VAR_REFERENCE.fullmatch(value)
    if match:
        return os.environ.get(match.group(1), "")
    return value
