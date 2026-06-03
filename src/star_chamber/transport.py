"""Provider transport layer with parallel fan-out across LLM providers."""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass

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


@dataclass(frozen=True)
class ProviderResponse:
    """Response from a single LLM provider.

    Attributes:
        provider: Provider identifier.
        model: Model name used.
        success: Whether the call succeeded.
        content: Response content on success.
        error: Error message on failure.
    """

    provider: str
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
    try:
        import any_llm  # noqa: F811
    except ImportError:
        return ProviderResponse(
            provider=config.provider,
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
                model=config.model,
                success=False,
                error=f"{detail}: {sanitized}",
            )
        return ProviderResponse(
            provider=config.provider,
            model=config.model,
            success=False,
            error=_sanitize_error(error_msg),
        )

    if not response.choices:
        return ProviderResponse(
            provider=config.provider,
            model=config.model,
            success=False,
            error=f"No response content (empty choices) from {config.provider}.",
        )

    content = response.choices[0].message.content
    return ProviderResponse(
        provider=config.provider,
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
        resolved.append(
            ProviderConfig(
                provider=cfg.provider,
                model=cfg.model,
                api_key=api_key,
                api_base=cfg.api_base,
                max_tokens=cfg.max_tokens,
                local=cfg.local,
            )
        )
    return tuple(resolved)


def resolve_otari(otari: OtariConfig | None) -> OtariConfig | None:
    """Resolve an Otari configuration against the environment.

    Returns a new OtariConfig; never mutates input.  Returns None when
    ``otari`` is None.  An explicit ``api_key`` may use a ``${ENV_VAR}``
    reference, which is expanded.  When ``api_base`` or ``api_key`` is None,
    it falls back to the ``OTARI_API_BASE`` / ``OTARI_API_KEY``
    environment variable; an unset variable leaves the field None.

    Args:
        otari: Optional Otari configuration.

    Returns:
        A new OtariConfig with resolved fields, or None.
    """
    if otari is None:
        return None

    api_base = otari.api_base
    if api_base is None:
        api_base = os.environ.get("OTARI_API_BASE")

    api_key = otari.api_key
    api_key = os.environ.get("OTARI_API_KEY") if api_key is None else _expand_env_var(api_key)

    return OtariConfig(api_base=api_base, api_key=api_key)


def _expand_env_var(value: str) -> str:
    """Expand a ``${VAR}`` template from the environment.

    Args:
        value: A string that may contain a ``${VAR}`` reference.

    Returns:
        The resolved value, or an empty string if the variable is not set.
        Non-template strings are returned unchanged.
    """
    match = re.fullmatch(r"\$\{([^}]+)}", value)
    if match:
        return os.environ.get(match.group(1), "")
    return value
