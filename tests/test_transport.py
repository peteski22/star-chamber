from __future__ import annotations

import asyncio
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from star_chamber.config import ConfigError
from star_chamber.transport import (
    DEFAULT_MAX_TOKENS,
    ProviderResponse,
    _is_auth_error,
    _sanitize_error,
    fan_out,
    resolve_api_keys,
    resolve_otari,
    send_to_provider,
    validate_otari_api_base,
)
from star_chamber.types import OtariConfig, ProviderConfig

# -- helpers ------------------------------------------------------------------


def _make_mock_any_llm(response_content: str = "Review looks good.", finish_reason: str = "stop"):
    """Build a mock any_llm module with an acompletion coroutine."""
    mock_module = types.ModuleType("any_llm")

    # Build a response shaped like any_llm's ModelResponse.
    mock_choice = MagicMock()
    mock_choice.message.content = response_content
    mock_choice.finish_reason = finish_reason

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
    def test_frozen(self) -> None:
        pr = ProviderResponse(provider="openai", display_name="openai", model="gpt-4", success=True, content="ok")
        with pytest.raises(AttributeError):
            pr.content = "changed"  # type: ignore[misc]

    def test_defaults(self) -> None:
        pr = ProviderResponse(provider="openai", display_name="openai", model="gpt-4", success=False)
        assert pr.content == ""
        assert pr.error == ""


# -- _sanitize_error ----------------------------------------------------------


class TestSanitizeError:
    def test_redacts_bearer_token(self) -> None:
        msg = "Authorization: Bearer sk-abc123xyz failed"
        result = _sanitize_error(msg)
        assert "sk-abc123xyz" not in result
        assert "REDACTED" in result

    def test_redacts_sk_key(self) -> None:
        msg = "Invalid key: sk-1234567890abcdef"
        result = _sanitize_error(msg)
        assert "sk-1234567890abcdef" not in result

    def test_passthrough_safe_message(self) -> None:
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
    def test_detects_auth_errors(self, msg: str) -> None:
        assert _is_auth_error(msg) is True

    def test_non_auth_error(self) -> None:
        assert _is_auth_error("Connection timeout after 30s") is False


# -- send_to_provider ---------------------------------------------------------


class TestSendToProvider:
    def test_successful_call(self) -> None:
        mock_module = _make_mock_any_llm("Great code!")
        config = ProviderConfig(provider="anthropic", model="claude-3")

        with patch.dict(sys.modules, {"any_llm": mock_module}):
            result = asyncio.run(send_to_provider(config, "Review this code."))

        assert result.success is True
        assert result.content == "Great code!"
        assert result.provider == "anthropic"
        assert result.model == "claude-3"

    def test_display_name_defaults_to_provider(self) -> None:
        mock_module = _make_mock_any_llm("ok")
        config = ProviderConfig(provider="openrouter", model="x-ai/grok-4.3")

        with patch.dict(sys.modules, {"any_llm": mock_module}):
            result = asyncio.run(send_to_provider(config, "Review this."))

        assert result.provider == "openrouter"
        assert result.display_name == "openrouter"

    def test_display_name_carries_configured_value(self) -> None:
        """The response keeps the underlying provider AND the display identity."""
        mock_module = _make_mock_any_llm("ok")
        config = ProviderConfig(provider="openrouter", model="openai/gpt-5.2", display_name="gpt-5.2")

        with patch.dict(sys.modules, {"any_llm": mock_module}):
            result = asyncio.run(send_to_provider(config, "Review this."))

        assert result.provider == "openrouter"  # underlying provider preserved
        assert result.display_name == "gpt-5.2"  # identity carries the display name

    def test_failure_keeps_provider_and_display_name_aligned(self) -> None:
        """On failure, the response provider and the diagnostic name the same underlying provider."""
        mock_module = _make_mock_any_llm()
        mock_module.acompletion = AsyncMock(  # type: ignore[attr-defined]
            side_effect=TimeoutError("timed out")
        )
        config = ProviderConfig(provider="openrouter", model="openai/gpt-5.2", display_name="gpt-5.2")

        with patch.dict(sys.modules, {"any_llm": mock_module}):
            result = asyncio.run(send_to_provider(config, "Review this.", timeout=5.0))

        assert result.success is False
        assert result.provider == "openrouter"
        assert result.display_name == "gpt-5.2"
        # The diagnostic names the underlying provider, matching result.provider.
        assert "openrouter" in result.error

    def test_provider_passed_as_separate_kwarg(self) -> None:
        """The SDK uses a separate provider kwarg for routing."""
        mock_module = _make_mock_any_llm("Looks good.")
        config = ProviderConfig(provider="openai", model="gpt-4o")

        with patch.dict(sys.modules, {"any_llm": mock_module}):
            asyncio.run(send_to_provider(config, "Review this."))

        call_kwargs = mock_module.acompletion.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o"
        assert call_kwargs["provider"] == "openai"

    def test_openai_uses_max_completion_tokens(self) -> None:
        mock_module = _make_mock_any_llm()
        config = ProviderConfig(provider="openai", model="gpt-4o", max_tokens=8192)

        with patch.dict(sys.modules, {"any_llm": mock_module}):
            asyncio.run(send_to_provider(config, "Review this."))

        call_kwargs = mock_module.acompletion.call_args.kwargs
        assert "max_completion_tokens" in call_kwargs
        assert call_kwargs["max_completion_tokens"] == 8192
        assert "max_tokens" not in call_kwargs

    def test_non_openai_uses_max_tokens(self) -> None:
        mock_module = _make_mock_any_llm()
        config = ProviderConfig(provider="anthropic", model="claude-3", max_tokens=4096)

        with patch.dict(sys.modules, {"any_llm": mock_module}):
            asyncio.run(send_to_provider(config, "Review this."))

        call_kwargs = mock_module.acompletion.call_args.kwargs
        assert "max_tokens" in call_kwargs
        assert call_kwargs["max_tokens"] == 4096
        assert "max_completion_tokens" not in call_kwargs

    def test_api_key_passed_when_set(self) -> None:
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

    def test_api_key_omitted_when_none(self) -> None:
        mock_module = _make_mock_any_llm()
        config = ProviderConfig(provider="anthropic", model="claude-3")

        with patch.dict(sys.modules, {"any_llm": mock_module}):
            asyncio.run(send_to_provider(config, "Review this."))

        call_kwargs = mock_module.acompletion.call_args.kwargs
        assert "api_key" not in call_kwargs

    def test_api_base_passed_when_set(self) -> None:
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

    def test_timeout_handling(self) -> None:
        mock_module = _make_mock_any_llm()
        mock_module.acompletion = AsyncMock(  # type: ignore[attr-defined]
            side_effect=TimeoutError("timed out")
        )
        config = ProviderConfig(provider="openai", model="gpt-4")

        with patch.dict(sys.modules, {"any_llm": mock_module}):
            result = asyncio.run(send_to_provider(config, "Review this.", timeout=5.0))

        assert result.success is False
        assert "timed out" in result.error.lower() or "timeout" in result.error.lower()

    def test_import_error(self) -> None:
        config = ProviderConfig(provider="openai", model="gpt-4")

        # Ensure any_llm is not importable.
        with patch.dict(sys.modules, {"any_llm": None}):
            result = asyncio.run(send_to_provider(config, "Review this."))

        assert result.success is False
        assert "any_llm" in result.error.lower() or "install" in result.error.lower()

    def test_auth_error_message(self) -> None:
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

    def test_no_choices_returns_error(self) -> None:
        mock_module = _make_empty_response_module()
        config = ProviderConfig(provider="anthropic", model="claude-3")

        with patch.dict(sys.modules, {"any_llm": mock_module}):
            result = asyncio.run(send_to_provider(config, "Review this."))

        assert result.success is False
        assert "no response" in result.error.lower() or "empty" in result.error.lower()

    def test_output_token_limit_fails_instead_of_returning_a_partial_reply(self) -> None:
        mock_module = _make_mock_any_llm('{"quality_rating": "go', finish_reason="length")
        config = ProviderConfig(provider="anthropic", model="claude-3", max_tokens=20000)

        with patch.dict(sys.modules, {"any_llm": mock_module}):
            result = asyncio.run(send_to_provider(config, "Review this."))

        assert result.success is False
        assert "20000" in result.error
        assert "max_tokens" in result.error

    def test_output_token_limit_error_names_the_default_limit_when_unconfigured(self) -> None:
        mock_module = _make_mock_any_llm("partial", finish_reason="length")
        config = ProviderConfig(provider="anthropic", model="claude-3")

        with patch.dict(sys.modules, {"any_llm": mock_module}):
            result = asyncio.run(send_to_provider(config, "Review this."))

        assert result.success is False
        assert str(DEFAULT_MAX_TOKENS) in result.error

    def test_blocked_response_fails_with_its_own_message(self) -> None:
        mock_module = _make_mock_any_llm("", finish_reason="content_filter")
        config = ProviderConfig(provider="anthropic", model="claude-3")

        with patch.dict(sys.modules, {"any_llm": mock_module}):
            result = asyncio.run(send_to_provider(config, "Review this."))

        assert result.success is False
        assert "blocked" in result.error.lower()
        assert "max_tokens" not in result.error

    def test_normal_stop_still_succeeds(self) -> None:
        mock_module = _make_mock_any_llm("Great code!", finish_reason="stop")
        config = ProviderConfig(provider="anthropic", model="claude-3")

        with patch.dict(sys.modules, {"any_llm": mock_module}):
            result = asyncio.run(send_to_provider(config, "Review this."))

        assert result.success is True
        assert result.content == "Great code!"

    def test_absent_content_becomes_an_empty_string(self) -> None:
        mock_module = _make_mock_any_llm(None, finish_reason="stop")
        config = ProviderConfig(provider="anthropic", model="claude-3")

        with patch.dict(sys.modules, {"any_llm": mock_module}):
            result = asyncio.run(send_to_provider(config, "Review this."))

        assert result.success is True
        assert result.content == ""

    def test_gemini_timeout_passed_as_kwarg(self) -> None:
        """Gemini receives timeout as a direct request kwarg."""
        mock_module = _make_mock_any_llm()
        config = ProviderConfig(provider="gemini", model="gemini-2.0-flash")

        with patch.dict(sys.modules, {"any_llm": mock_module}):
            asyncio.run(send_to_provider(config, "Review this.", timeout=30.0))

        call_kwargs = mock_module.acompletion.call_args.kwargs
        assert call_kwargs["timeout"] == 30.0

    def test_non_gemini_timeout_uses_request_kwarg(self) -> None:
        """Non-gemini providers receive timeout as a direct request kwarg."""
        mock_module = _make_mock_any_llm()
        config = ProviderConfig(provider="openai", model="gpt-4")

        with patch.dict(sys.modules, {"any_llm": mock_module}):
            asyncio.run(send_to_provider(config, "Review this.", timeout=30.0))

        call_kwargs = mock_module.acompletion.call_args.kwargs
        assert call_kwargs["timeout"] == 30.0


# -- fan_out ------------------------------------------------------------------


class TestFanOut:
    def test_parallel_calls(self) -> None:
        mock_module = _make_mock_any_llm("Looks good.")
        configs = (
            ProviderConfig(provider="openai", model="gpt-4"),
            ProviderConfig(provider="anthropic", model="claude-3"),
        )

        with patch.dict(sys.modules, {"any_llm": mock_module}):
            results = asyncio.run(fan_out(configs, "Review this.", timeout=30.0))

        assert len(results) == 2
        assert all(response.success for _, response in results)
        assert [config.provider for config, _ in results] == ["openai", "anthropic"]
        assert all(config.provider == response.provider for config, response in results)

    def test_mixed_success_and_failure(self) -> None:
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
        by_provider = {config.provider: response for config, response in results}
        assert by_provider["openai"].success is False
        assert by_provider["anthropic"].success is True


# -- resolve_api_keys ---------------------------------------------------------


class TestResolveApiKeys:
    def test_env_var_expansion(self, monkeypatch) -> None:
        monkeypatch.setenv("MY_API_KEY", "resolved-key-value")
        configs = (
            ProviderConfig(
                provider="openai",
                model="gpt-4",
                api_key="${MY_API_KEY}",
            ),
        )

        result = resolve_api_keys(configs)

        assert result[0].api_key == "resolved-key-value"  # pragma: allowlist secret

    def test_missing_env_var(self, monkeypatch) -> None:
        monkeypatch.delenv("NONEXISTENT_KEY", raising=False)
        configs = (
            ProviderConfig(
                provider="openai",
                model="gpt-4",
                api_key="${NONEXISTENT_KEY}",
            ),
        )

        result = resolve_api_keys(configs)

        # Missing env var should result in empty string or None.
        assert result[0].api_key == "" or result[0].api_key is None

    def test_non_template_key_unchanged(self) -> None:
        configs = (
            ProviderConfig(
                provider="openai",
                model="gpt-4",
                api_key="literal-key-value",  # pragma: allowlist secret
            ),
        )

        result = resolve_api_keys(configs)

        assert result[0].api_key == "literal-key-value"  # pragma: allowlist secret

    def test_returns_new_objects(self) -> None:
        configs = (
            ProviderConfig(
                provider="openai",
                model="gpt-4",
                api_key="literal-key",  # pragma: allowlist secret
            ),
        )

        result = resolve_api_keys(configs)

        # Must return new objects, never mutate originals.
        assert result is not configs
        assert result[0] is not configs[0]


class TestResolveOtari:
    def test_none_passes_through(self) -> None:
        assert resolve_otari(None) is None

    def test_env_var_expansion(self, monkeypatch) -> None:
        monkeypatch.setenv("OTARI_API_KEY", "resolved-gw-key")
        gw = OtariConfig(api_base="https://gw.example", api_key="${OTARI_API_KEY}")

        result = resolve_otari(gw)

        assert result is not None
        assert result.api_base == "https://gw.example"
        assert result.api_key == "resolved-gw-key"  # pragma: allowlist secret

    def test_api_base_env_var_expansion(self, monkeypatch) -> None:
        monkeypatch.setenv("OTARI_API_BASE", "https://resolved-gw.example")
        gw = OtariConfig(api_base="${OTARI_API_BASE}", api_key="literal")  # pragma: allowlist secret

        result = resolve_otari(gw)

        assert result is not None
        assert result.api_base == "https://resolved-gw.example"

    def test_returns_new_object(self) -> None:
        gw = OtariConfig(api_base="https://gw.example", api_key="literal")  # pragma: allowlist secret
        result = resolve_otari(gw)
        assert result is not gw

    def test_none_fields_stay_none_for_sdk_auto_detection(self) -> None:
        """None fields are left for the SDK's OtariProvider to resolve."""
        gw = OtariConfig()  # both None

        result = resolve_otari(gw)

        assert result is not None
        assert result.api_base is None
        assert result.api_key is None

    def test_explicit_values_are_preserved(self) -> None:
        gw = OtariConfig(api_base="https://explicit.example", api_key="explicit-key")  # pragma: allowlist secret

        result = resolve_otari(gw)

        assert result is not None
        assert result.api_base == "https://explicit.example"
        assert result.api_key == "explicit-key"  # pragma: allowlist secret

    def test_api_path_in_base_is_rejected(self) -> None:
        with pytest.raises(ConfigError, match="ends in an API path"):
            resolve_otari(OtariConfig(api_base="https://gw.example/v1"))


class TestValidateOtariApiBase:
    @pytest.mark.parametrize(
        ("api_base", "origin"),
        [
            ("https://api.otari.ai/v1", "https://api.otari.ai"),
            ("https://api.otari.ai/v1/", "https://api.otari.ai"),
            ("https://api.otari.ai/api/v1", "https://api.otari.ai"),
            ("https://proxy.example/otari/v1", "https://proxy.example/otari"),
        ],
    )
    def test_config_value_with_api_path_names_the_origin(self, api_base, origin) -> None:
        with pytest.raises(ConfigError) as exc_info:
            validate_otari_api_base(OtariConfig(api_base=api_base))

        assert f"Set 'otari.api_base' in the config to '{origin}'." in str(exc_info.value)

    def test_referenced_env_var_with_api_path_is_named(self, monkeypatch) -> None:
        monkeypatch.setenv("OTARI_API_BASE", "https://api.otari.ai/v1")

        with pytest.raises(ConfigError, match=r"Set OTARI_API_BASE to 'https://api\.otari\.ai'\."):
            validate_otari_api_base(OtariConfig(api_base="${OTARI_API_BASE}"))

    @pytest.mark.parametrize("name", ["OTARI_API_BASE", "GATEWAY_API_BASE"])
    def test_fallback_env_var_with_api_path_is_named(self, monkeypatch, name) -> None:
        monkeypatch.setenv(name, "https://gw.example/v1")

        with pytest.raises(ConfigError, match=rf"Set {name} to 'https://gw\.example'\."):
            validate_otari_api_base(OtariConfig())

    @pytest.mark.parametrize("api_base", ["${UNSET_OTARI_BASE}", ""])
    def test_empty_base_falls_back_to_env_var(self, monkeypatch, api_base) -> None:
        monkeypatch.delenv("UNSET_OTARI_BASE", raising=False)
        monkeypatch.setenv("GATEWAY_API_BASE", "https://gw.example/v1")

        with pytest.raises(ConfigError, match="Set GATEWAY_API_BASE"):
            validate_otari_api_base(OtariConfig(api_base=api_base))

    def test_config_value_takes_precedence_over_env_var(self, monkeypatch) -> None:
        monkeypatch.setenv("OTARI_API_BASE", "https://gw.example/v1")

        validate_otari_api_base(OtariConfig(api_base="https://gw.example"))

    @pytest.mark.parametrize(
        "api_base",
        [None, "https://api.otari.ai", "https://api.otari.ai/", "https://proxy.example/otari"],
    )
    def test_base_without_api_path_is_accepted(self, api_base) -> None:
        validate_otari_api_base(OtariConfig(api_base=api_base))

    def test_missing_otari_config_is_accepted(self) -> None:
        validate_otari_api_base(None)

    def test_malformed_base_is_a_config_error_naming_its_setting(self, monkeypatch) -> None:
        monkeypatch.setenv("OTARI_API_BASE", "https://[::1/v1")

        with pytest.raises(ConfigError, match=r"from OTARI_API_BASE is not a valid URL"):
            validate_otari_api_base(OtariConfig())


class TestOtariRouting:
    def test_otari_overrides_provider_routing(self) -> None:
        mock_module = _make_mock_any_llm("ok")
        config = ProviderConfig(provider="anthropic", model="claude-3")
        gw = OtariConfig(api_base="https://gw.example", api_key="gw-key")  # pragma: allowlist secret

        with patch.dict(sys.modules, {"any_llm": mock_module}):
            asyncio.run(send_to_provider(config, "Review this.", otari=gw))

        call_kwargs = mock_module.acompletion.call_args.kwargs
        assert call_kwargs["provider"] == "otari"
        assert call_kwargs["model"] == "claude-3"
        assert call_kwargs["api_base"] == "https://gw.example"
        assert call_kwargs["api_key"] == "gw-key"  # pragma: allowlist secret

    def test_local_provider_bypasses_otari(self) -> None:
        mock_module = _make_mock_any_llm("ok")
        config = ProviderConfig(
            provider="ollama",
            model="llama3",
            api_base="http://localhost:11434",
            local=True,
        )
        gw = OtariConfig(api_base="https://gw.example", api_key="gw-key")  # pragma: allowlist secret

        with patch.dict(sys.modules, {"any_llm": mock_module}):
            asyncio.run(send_to_provider(config, "Review this.", otari=gw))

        call_kwargs = mock_module.acompletion.call_args.kwargs
        # Local providers keep their own routing even when a otari is configured.
        assert call_kwargs["provider"] == "ollama"
        assert call_kwargs["api_base"] == "http://localhost:11434"
        assert "api_key" not in call_kwargs

    def test_otari_with_none_fields_omits_overrides(self) -> None:
        """send_to_provider omits None otari fields from the call kwargs.

        Env-var resolution happens upstream in resolve_otari; by the time a
        field is still None here, it is left to the SDK's own resolution."""
        mock_module = _make_mock_any_llm("ok")
        config = ProviderConfig(provider="anthropic", model="claude-3")
        gw = OtariConfig()  # both None

        with patch.dict(sys.modules, {"any_llm": mock_module}):
            asyncio.run(send_to_provider(config, "Review this.", otari=gw))

        call_kwargs = mock_module.acompletion.call_args.kwargs
        assert call_kwargs["provider"] == "otari"
        assert "api_base" not in call_kwargs
        assert "api_key" not in call_kwargs

    def test_otari_auth_error_points_at_otari(self) -> None:
        """An auth failure on a otari-routed call blames the otari, not the provider."""
        mock_module = _make_mock_any_llm()
        mock_module.acompletion = AsyncMock(  # type: ignore[attr-defined]
            side_effect=Exception("Unauthorized: invalid api_key")
        )
        config = ProviderConfig(provider="anthropic", model="claude-3")
        gw = OtariConfig(api_base="https://gw.example", api_key="gw-key")  # pragma: allowlist secret

        with patch.dict(sys.modules, {"any_llm": mock_module}):
            result = asyncio.run(send_to_provider(config, "Review this.", otari=gw))

        assert result.success is False
        assert "otari" in result.error.lower()
        # Must not misdirect users to the upstream provider's own key.
        assert "key for anthropic" not in result.error.lower()

    def test_fan_out_forwards_otari(self) -> None:
        mock_module = _make_mock_any_llm("ok")
        configs = (
            ProviderConfig(provider="openai", model="gpt-4"),
            ProviderConfig(provider="ollama", model="llama3", local=True, api_base="http://localhost:11434"),
        )
        gw = OtariConfig(api_base="https://gw.example", api_key="gw-key")  # pragma: allowlist secret

        with patch.dict(sys.modules, {"any_llm": mock_module}):
            asyncio.run(fan_out(configs, "Review this.", timeout=30.0, otari=gw))

        # Two calls were dispatched; the cloud one through Otari, the local one direct.
        providers_used = [call.kwargs["provider"] for call in mock_module.acompletion.call_args_list]
        assert sorted(providers_used) == ["ollama", "otari"]


# -- resolve_api_keys field preservation --------------------------------------


class TestResolveApiKeysPreservesFields:
    def test_display_name_and_optional_fields_survive(self, monkeypatch) -> None:
        monkeypatch.setenv("MY_API_KEY", "resolved-key-value")
        cfg = ProviderConfig(
            provider="openrouter",
            model="openai/gpt-5.2",
            api_key="${MY_API_KEY}",
            api_base="https://gw.example",
            max_tokens=1024,
            display_name="gpt-5.2",
        )

        result = resolve_api_keys((cfg,))

        assert result[0].api_key == "resolved-key-value"  # pragma: allowlist secret
        assert result[0].display_name == "gpt-5.2"
        assert result[0].api_base == "https://gw.example"
        assert result[0].max_tokens == 1024
