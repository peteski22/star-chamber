from __future__ import annotations

import asyncio
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from star_chamber.transport import (
    ProviderResponse,
    _is_auth_error,
    _sanitize_error,
    fan_out,
    resolve_api_keys,
    send_to_provider,
)
from star_chamber.types import ProviderConfig

# -- helpers ------------------------------------------------------------------


def _make_mock_any_llm(response_content: str = "Review looks good."):
    """Build a mock any_llm module with an acompletion coroutine."""
    mock_module = types.ModuleType("any_llm")

    # Build a response shaped like any_llm's ModelResponse.
    mock_choice = MagicMock()
    mock_choice.message.content = response_content

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_module.acompletion = AsyncMock(return_value=mock_response)  # type: ignore[attr-defined]
    return mock_module


def _make_empty_response_module():
    """Build a mock any_llm module that returns an empty choices list."""
    mock_module = types.ModuleType("any_llm")

    mock_response = MagicMock()
    mock_response.choices = []

    mock_module.acompletion = AsyncMock(return_value=mock_response)  # type: ignore[attr-defined]
    return mock_module


# -- ProviderResponse ---------------------------------------------------------


class TestProviderResponse:
    def test_frozen(self):
        pr = ProviderResponse(provider="openai", model="gpt-4", success=True, content="ok")
        with pytest.raises(AttributeError):
            pr.content = "changed"  # type: ignore[misc]

    def test_defaults(self):
        pr = ProviderResponse(provider="openai", model="gpt-4", success=False)
        assert pr.content == ""
        assert pr.error == ""


# -- _sanitize_error ----------------------------------------------------------


class TestSanitizeError:
    def test_redacts_bearer_token(self):
        msg = "Authorization: Bearer sk-abc123xyz failed"
        result = _sanitize_error(msg)
        assert "sk-abc123xyz" not in result
        assert "REDACTED" in result

    def test_redacts_sk_key(self):
        msg = "Invalid key: sk-1234567890abcdef"
        result = _sanitize_error(msg)
        assert "sk-1234567890abcdef" not in result

    def test_passthrough_safe_message(self):
        msg = "Connection refused to host"
        assert _sanitize_error(msg) == msg


# -- _is_auth_error -----------------------------------------------------------


class TestIsAuthError:
    @pytest.mark.parametrize(
        "msg",
        [
            "Invalid api_key provided",
            "Unauthorized access",
            "HTTP 401 response",
            "Missing API key for provider",
            "Check your apikey",
        ],
    )
    def test_detects_auth_errors(self, msg: str):
        assert _is_auth_error(msg) is True

    def test_non_auth_error(self):
        assert _is_auth_error("Connection timeout after 30s") is False


# -- send_to_provider ---------------------------------------------------------


class TestSendToProvider:
    def test_successful_call(self):
        mock_module = _make_mock_any_llm("Great code!")
        config = ProviderConfig(provider="anthropic", model="claude-3")

        with patch.dict(sys.modules, {"any_llm": mock_module}):
            result = asyncio.run(send_to_provider(config, "Review this code."))

        assert result.success is True
        assert result.content == "Great code!"
        assert result.provider == "anthropic"
        assert result.model == "claude-3"

    def test_provider_passed_as_separate_kwarg(self):
        """The any-llm SDK uses a separate provider kwarg for routing."""
        mock_module = _make_mock_any_llm("Looks good.")
        config = ProviderConfig(provider="openai", model="gpt-4o")

        with patch.dict(sys.modules, {"any_llm": mock_module}):
            asyncio.run(send_to_provider(config, "Review this."))

        call_kwargs = mock_module.acompletion.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o"
        assert call_kwargs["provider"] == "openai"

    def test_openai_uses_max_completion_tokens(self):
        mock_module = _make_mock_any_llm()
        config = ProviderConfig(provider="openai", model="gpt-4o", max_tokens=8192)

        with patch.dict(sys.modules, {"any_llm": mock_module}):
            asyncio.run(send_to_provider(config, "Review this."))

        call_kwargs = mock_module.acompletion.call_args.kwargs
        assert "max_completion_tokens" in call_kwargs
        assert call_kwargs["max_completion_tokens"] == 8192
        assert "max_tokens" not in call_kwargs

    def test_non_openai_uses_max_tokens(self):
        mock_module = _make_mock_any_llm()
        config = ProviderConfig(provider="anthropic", model="claude-3", max_tokens=4096)

        with patch.dict(sys.modules, {"any_llm": mock_module}):
            asyncio.run(send_to_provider(config, "Review this."))

        call_kwargs = mock_module.acompletion.call_args.kwargs
        assert "max_tokens" in call_kwargs
        assert call_kwargs["max_tokens"] == 4096
        assert "max_completion_tokens" not in call_kwargs

    def test_api_key_passed_when_set(self):
        mock_module = _make_mock_any_llm()
        config = ProviderConfig(
            provider="anthropic",
            model="claude-3",
            api_key="test-key-not-real",  # pragma: allowlist secret
        )

        with patch.dict(sys.modules, {"any_llm": mock_module}):
            asyncio.run(send_to_provider(config, "Review this."))

        call_kwargs = mock_module.acompletion.call_args.kwargs
        assert call_kwargs["api_key"] == "test-key-not-real"  # pragma: allowlist secret

    def test_api_key_omitted_when_none(self):
        mock_module = _make_mock_any_llm()
        config = ProviderConfig(provider="anthropic", model="claude-3")

        with patch.dict(sys.modules, {"any_llm": mock_module}):
            asyncio.run(send_to_provider(config, "Review this."))

        call_kwargs = mock_module.acompletion.call_args.kwargs
        assert "api_key" not in call_kwargs

    def test_api_base_passed_when_set(self):
        mock_module = _make_mock_any_llm()
        config = ProviderConfig(
            provider="ollama",
            model="llama3",
            api_base="http://localhost:11434",
        )

        with patch.dict(sys.modules, {"any_llm": mock_module}):
            asyncio.run(send_to_provider(config, "Review this."))

        call_kwargs = mock_module.acompletion.call_args.kwargs
        assert call_kwargs["api_base"] == "http://localhost:11434"

    def test_timeout_handling(self):
        mock_module = _make_mock_any_llm()
        mock_module.acompletion = AsyncMock(  # type: ignore[attr-defined]
            side_effect=TimeoutError("timed out")
        )
        config = ProviderConfig(provider="openai", model="gpt-4")

        with patch.dict(sys.modules, {"any_llm": mock_module}):
            result = asyncio.run(send_to_provider(config, "Review this.", timeout=5.0))

        assert result.success is False
        assert "timed out" in result.error.lower() or "timeout" in result.error.lower()

    def test_import_error(self):
        config = ProviderConfig(provider="openai", model="gpt-4")

        # Ensure any_llm is not importable.
        with patch.dict(sys.modules, {"any_llm": None}):
            result = asyncio.run(send_to_provider(config, "Review this."))

        assert result.success is False
        assert "any_llm" in result.error.lower() or "install" in result.error.lower()

    def test_auth_error_message(self):
        mock_module = _make_mock_any_llm()
        mock_module.acompletion = AsyncMock(  # type: ignore[attr-defined]
            side_effect=Exception("Unauthorized: invalid api_key")
        )
        config = ProviderConfig(provider="openai", model="gpt-4")

        with patch.dict(sys.modules, {"any_llm": mock_module}):
            result = asyncio.run(send_to_provider(config, "Review this."))

        assert result.success is False
        # Auth errors should mention checking credentials.
        assert "check" in result.error.lower() or "key" in result.error.lower()

    def test_no_choices_returns_error(self):
        mock_module = _make_empty_response_module()
        config = ProviderConfig(provider="anthropic", model="claude-3")

        with patch.dict(sys.modules, {"any_llm": mock_module}):
            result = asyncio.run(send_to_provider(config, "Review this."))

        assert result.success is False
        assert "no response" in result.error.lower() or "empty" in result.error.lower()


# -- fan_out ------------------------------------------------------------------


class TestFanOut:
    def test_parallel_calls(self):
        mock_module = _make_mock_any_llm("Looks good.")
        configs = (
            ProviderConfig(provider="openai", model="gpt-4"),
            ProviderConfig(provider="anthropic", model="claude-3"),
        )

        with patch.dict(sys.modules, {"any_llm": mock_module}):
            results = asyncio.run(fan_out(configs, "Review this.", timeout=30.0))

        assert len(results) == 2
        assert all(r.success for r in results)
        providers = {r.provider for r in results}
        assert providers == {"openai", "anthropic"}

    def test_mixed_success_and_failure(self):
        mock_module = _make_mock_any_llm("Looks good.")
        call_count = 0

        original_acompletion = mock_module.acompletion

        async def _side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if kwargs.get("model", "").startswith("gpt"):
                raise Exception("Rate limited")
            return await original_acompletion(*args, **kwargs)

        mock_module.acompletion = AsyncMock(side_effect=_side_effect)  # type: ignore[attr-defined]

        configs = (
            ProviderConfig(provider="openai", model="gpt-4"),
            ProviderConfig(provider="anthropic", model="claude-3"),
        )

        with patch.dict(sys.modules, {"any_llm": mock_module}):
            results = asyncio.run(fan_out(configs, "Review this.", timeout=30.0))

        assert len(results) == 2
        by_provider = {r.provider: r for r in results}
        assert by_provider["openai"].success is False
        assert by_provider["anthropic"].success is True


# -- resolve_api_keys ---------------------------------------------------------


class TestResolveApiKeys:
    def test_env_var_expansion(self, monkeypatch):
        monkeypatch.setenv("MY_API_KEY", "resolved-key-value")
        configs = (
            ProviderConfig(
                provider="openai",
                model="gpt-4",
                api_key="${MY_API_KEY}",
            ),
        )

        result = resolve_api_keys(configs, use_platform=False)

        assert result[0].api_key == "resolved-key-value"  # pragma: allowlist secret

    def test_missing_env_var(self, monkeypatch):
        monkeypatch.delenv("NONEXISTENT_KEY", raising=False)
        configs = (
            ProviderConfig(
                provider="openai",
                model="gpt-4",
                api_key="${NONEXISTENT_KEY}",
            ),
        )

        result = resolve_api_keys(configs, use_platform=False)

        # Missing env var should result in empty string or None.
        assert result[0].api_key == "" or result[0].api_key is None

    def test_non_template_key_unchanged(self):
        configs = (
            ProviderConfig(
                provider="openai",
                model="gpt-4",
                api_key="literal-key-value",  # pragma: allowlist secret
            ),
        )

        result = resolve_api_keys(configs, use_platform=False)

        assert result[0].api_key == "literal-key-value"  # pragma: allowlist secret

    def test_returns_new_objects(self):
        configs = (
            ProviderConfig(
                provider="openai",
                model="gpt-4",
                api_key="literal-key",  # pragma: allowlist secret
            ),
        )

        result = resolve_api_keys(configs, use_platform=False)

        # Must return new objects, never mutate originals.
        assert result is not configs
        assert result[0] is not configs[0]

    def test_platform_mode_skips_expansion(self, monkeypatch):
        monkeypatch.setenv("MY_API_KEY", "resolved-key-value")
        configs = (
            ProviderConfig(
                provider="openai",
                model="gpt-4",
                api_key="${MY_API_KEY}",
            ),
        )

        result = resolve_api_keys(configs, use_platform=True)

        # In platform mode, keys are not expanded.
        assert result[0].api_key == "${MY_API_KEY}"
