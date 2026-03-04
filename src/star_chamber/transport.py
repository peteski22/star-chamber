"""Provider transport layer with fan-out via any-llm."""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass

from star_chamber.types import ProviderConfig

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
) -> ProviderResponse:
    """Send a prompt to a single provider via any_llm.acompletion.

    Args:
        config: Provider configuration.
        prompt: The prompt to send.
        timeout: Optional per-call timeout in seconds.

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

    kwargs: dict[str, object] = {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
    }

    # OpenAI uses max_completion_tokens; others use max_tokens.
    if config.provider == "openai":
        kwargs["max_completion_tokens"] = max_tok
    else:
        kwargs["max_tokens"] = max_tok

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
            location = "locally" if config.local else "for cloud provider"
            sanitized = _sanitize_error(error_msg)
            return ProviderResponse(
                provider=config.provider,
                model=config.model,
                success=False,
                error=f"Authentication failed {location}. Check your API key for {config.provider}: {sanitized}",
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
) -> list[ProviderResponse]:
    """Send a prompt to all providers in parallel.

    Args:
        configs: Tuple of provider configurations.
        prompt: The prompt to broadcast.
        timeout: Optional per-provider timeout in seconds.

    Returns:
        List of ProviderResponse objects, one per provider.
    """
    tasks = [send_to_provider(cfg, prompt, timeout=timeout) for cfg in configs]
    return list(await asyncio.gather(*tasks))


def resolve_api_keys(
    configs: tuple[ProviderConfig, ...],
    use_platform: bool,
    any_llm_key: str = "",
) -> tuple[ProviderConfig, ...]:
    """Resolve environment variable references in API keys.

    In direct mode (use_platform=False), expands ``${ENV_VAR}`` patterns
    using ``os.environ``.  In platform mode the keys are left as-is.

    Always returns NEW ProviderConfig objects; never mutates input.

    Args:
        configs: Tuple of provider configurations.
        use_platform: When True, skip env-var expansion.
        any_llm_key: Optional platform API key (reserved for future use).

    Returns:
        Tuple of new ProviderConfig objects with resolved keys.
    """
    resolved: list[ProviderConfig] = []
    for cfg in configs:
        api_key = cfg.api_key
        if not use_platform and api_key is not None:
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
