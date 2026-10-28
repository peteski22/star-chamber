from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from star_chamber.config import ConfigError, _default_config_path, load_config
from star_chamber.schema import get_schema
from star_chamber.transport import DEFAULT_MAX_TOKENS
from star_chamber.types import CouncilConfig, ProviderConfig

TESTDATA = Path(__file__).parent / "testdata"


class TestLoadValidConfig:
    def test_load_valid_config(self) -> None:
        cfg = load_config(TESTDATA / "providers.json")

        assert len(cfg.providers) == 2
        assert cfg.providers[0].provider == "openai"
        assert cfg.providers[0].model == "gpt-4o"
        assert cfg.providers[0].api_key == "${OPENAI_API_KEY}"
        assert cfg.providers[0].max_tokens == 16384
        assert cfg.providers[1].provider == "anthropic"
        assert cfg.providers[1].model == "claude-sonnet-4-20250514"
        assert cfg.providers[1].api_key == "${ANTHROPIC_API_KEY}"
        assert cfg.timeout_seconds == 90
        assert cfg.consensus_threshold == 2

    def test_load_otari_config(self) -> None:
        cfg = load_config(TESTDATA / "providers_otari.json")

        assert cfg.otari is not None
        assert cfg.otari.api_base == "https://otari.example"
        assert cfg.otari.api_key == "${OTARI_API_KEY}"
        assert len(cfg.providers) == 2
        assert cfg.providers[0].provider == "openai"
        assert cfg.providers[1].provider == "anthropic"

    def test_load_direct_config_with_local(self) -> None:
        cfg = load_config(TESTDATA / "providers_direct.json")

        assert len(cfg.providers) == 2
        # First provider is remote with an API key.
        assert cfg.providers[0].provider == "openai"
        assert cfg.providers[0].api_key == "${OPENAI_API_KEY}"
        assert cfg.providers[0].local is False
        # Second provider is local with an api_base override.
        assert cfg.providers[1].provider == "ollama"
        assert cfg.providers[1].api_base == "http://localhost:11434"
        assert cfg.providers[1].local is True


class TestConfigErrors:
    def test_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises(ConfigError, match="not found"):
            load_config(tmp_path / "does-not-exist.json")

    def test_invalid_json_raises(self, tmp_path) -> None:
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{not valid json!!")
        with pytest.raises(ConfigError, match="Invalid JSON"):
            load_config(bad_file)

    def test_missing_providers_key_raises(self, tmp_path) -> None:
        no_providers = tmp_path / "no_providers.json"
        no_providers.write_text(json.dumps({"timeout_seconds": 60}))
        with pytest.raises(ConfigError, match="providers"):
            load_config(no_providers)

    @pytest.mark.parametrize("root", ["null", "42", '"a string"', "[]"])
    def test_non_object_root_raises(self, tmp_path, root) -> None:
        bad = tmp_path / "non_object.json"
        bad.write_text(root)
        with pytest.raises(ConfigError, match="must be a JSON object"):
            load_config(bad)

    def test_providers_not_array_raises(self) -> None:
        with pytest.raises(ConfigError, match="must be a list"):
            load_config(TESTDATA / "invalid_config.json")

    def test_empty_providers_raises(self, tmp_path) -> None:
        empty = tmp_path / "empty.json"
        empty.write_text(json.dumps({"providers": []}))
        with pytest.raises(ConfigError, match="at least one"):
            load_config(empty)

    def test_provider_missing_required_fields_raises(self, tmp_path) -> None:
        missing_model = tmp_path / "missing_model.json"
        missing_model.write_text(json.dumps({"providers": [{"provider": "openai"}]}))
        with pytest.raises(ConfigError, match="model"):
            load_config(missing_model)

    def test_legacy_platform_key_rejected(self, tmp_path) -> None:
        legacy = tmp_path / "legacy.json"
        legacy.write_text(
            json.dumps(
                {
                    "platform": "any-llm",
                    "providers": [{"provider": "openai", "model": "gpt-4o"}],
                }
            )
        )
        with pytest.raises(ConfigError, match="'platform' is no longer supported"):
            load_config(legacy)

    def test_otari_unknown_field_rejected(self, tmp_path) -> None:
        bad = tmp_path / "bad_gw.json"
        bad.write_text(
            json.dumps(
                {
                    "otari": {"api_base": "https://x", "extra_field": 1},
                    "providers": [{"provider": "openai", "model": "gpt-4o"}],
                }
            )
        )
        with pytest.raises(ConfigError, match="Unknown 'otari' fields"):
            load_config(bad)

    def test_otari_not_object_rejected(self, tmp_path) -> None:
        bad = tmp_path / "bad_gw_type.json"
        bad.write_text(
            json.dumps(
                {
                    "otari": "not-an-object",
                    "providers": [{"provider": "openai", "model": "gpt-4o"}],
                }
            )
        )
        with pytest.raises(ConfigError, match="'otari' must be an object"):
            load_config(bad)


class TestDefaults:
    def test_defaults_applied(self, tmp_path) -> None:
        minimal = tmp_path / "minimal.json"
        minimal.write_text(json.dumps({"providers": [{"provider": "openai", "model": "gpt-4o"}]}))
        cfg = load_config(minimal)

        assert cfg.timeout_seconds == 60
        assert cfg.consensus_threshold == 2
        assert cfg.otari is None
        assert cfg.synthesis is True

    def test_synthesis_can_be_declined(self, tmp_path) -> None:
        declined = tmp_path / "declined.json"
        declined.write_text(json.dumps({"providers": [{"provider": "openai", "model": "gpt-4o"}], "synthesis": False}))

        assert load_config(declined).synthesis is False

    def test_non_boolean_synthesis_raises(self, tmp_path) -> None:
        bad = tmp_path / "bad_synthesis.json"
        bad.write_text(json.dumps({"providers": [{"provider": "openai", "model": "gpt-4o"}], "synthesis": "no"}))

        with pytest.raises(ConfigError, match="'synthesis' must be true or false"):
            load_config(bad)

    def test_default_config_path(self, monkeypatch) -> None:
        # When STAR_CHAMBER_CONFIG is set, use it.
        monkeypatch.setenv("STAR_CHAMBER_CONFIG", "/tmp/custom.json")
        assert _default_config_path() == Path("/tmp/custom.json")

    def test_default_config_path_fallback(self, monkeypatch) -> None:
        # When STAR_CHAMBER_CONFIG is unset, fall back to ~/.config path.
        monkeypatch.delenv("STAR_CHAMBER_CONFIG", raising=False)
        result = _default_config_path()
        assert result == Path.home() / ".config" / "star-chamber" / "providers.json"

    def test_load_config_uses_default_path(self, tmp_path, monkeypatch) -> None:
        config_file = tmp_path / "providers.json"
        config_file.write_text(json.dumps({"providers": [{"provider": "openai", "model": "gpt-4o"}]}))
        with patch("star_chamber.config._default_config_path", return_value=config_file):
            cfg = load_config()

        assert len(cfg.providers) == 1
        assert cfg.providers[0].provider == "openai"


class TestProviderDisplayName:
    def test_display_name_parsed_and_optional(self, tmp_path) -> None:
        config = {
            "providers": [
                {"provider": "openrouter", "model": "openai/gpt-5.2", "display_name": "gpt-5.2"},
                {"provider": "openai", "model": "gpt-4o"},
            ]
        }
        path = tmp_path / "providers.json"
        path.write_text(json.dumps(config))

        cfg = load_config(path)

        assert cfg.providers[0].display_name == "gpt-5.2"
        assert cfg.providers[1].display_name is None

    def test_duplicate_display_names_rejected(self, tmp_path) -> None:
        config = {
            "providers": [
                {"provider": "openrouter", "model": "openai/gpt-5.2", "display_name": "member-a"},
                {"provider": "openrouter", "model": "x-ai/grok-4.3", "display_name": "member-a"},
            ]
        }
        path = tmp_path / "providers.json"
        path.write_text(json.dumps(config))

        with pytest.raises(ConfigError, match="member-a"):
            load_config(path)

    def test_display_name_colliding_with_bare_provider_rejected(self, tmp_path) -> None:
        config = {
            "providers": [
                {"provider": "openai", "model": "gpt-4o"},
                {"provider": "openrouter", "model": "openai/gpt-5.2", "display_name": "openai"},
            ]
        }
        path = tmp_path / "providers.json"
        path.write_text(json.dumps(config))

        with pytest.raises(ConfigError, match="openai"):
            load_config(path)

    def test_same_provider_members_without_display_name_still_load(self, tmp_path) -> None:
        # Pre-existing behaviour stays intact: same provider twice, no display names.
        config = {
            "providers": [
                {"provider": "openrouter", "model": "openai/gpt-5.2"},
                {"provider": "openrouter", "model": "x-ai/grok-4.3"},
            ]
        }
        path = tmp_path / "providers.json"
        path.write_text(json.dumps(config))

        cfg = load_config(path)

        assert len(cfg.providers) == 2
        assert cfg.providers[0].display_name is None
        assert cfg.providers[1].display_name is None

    def test_blank_display_name_rejected(self, tmp_path) -> None:
        config = {
            "providers": [
                {"provider": "openrouter", "model": "openai/gpt-5.2", "display_name": "   "},
            ]
        }
        path = tmp_path / "providers.json"
        path.write_text(json.dumps(config))

        with pytest.raises(ConfigError, match="display_name"):
            load_config(path)

    def test_non_string_display_name_rejected(self, tmp_path) -> None:
        config = {
            "providers": [
                {"provider": "openrouter", "model": "openai/gpt-5.2", "display_name": ["gpt-5.2"]},
            ]
        }
        path = tmp_path / "providers.json"
        path.write_text(json.dumps(config))

        with pytest.raises(ConfigError, match="display_name"):
            load_config(path)


class TestDocumentedDefaults:
    """The shipped schema states the defaults a config author reads.

    A stated default that drifts from the code teaches the wrong number to
    every author who omits the field, and nothing else would catch it.
    """

    def test_the_schema_states_the_max_tokens_budget_the_transport_applies(self) -> None:
        schema = json.loads(get_schema("provider-config"))

        assert schema["properties"]["max_tokens"]["default"] == DEFAULT_MAX_TOKENS

    def test_the_schema_states_whether_a_session_asks_for_a_synthesis(self) -> None:
        schema = json.loads(get_schema("council-config"))
        loaded = CouncilConfig(providers=(ProviderConfig(provider="openai", model="gpt-4o"),))

        assert schema["properties"]["synthesis"]["default"] is loaded.synthesis


class TestConsensusThresholdValidation:
    """A threshold the comparison cannot use is a configuration error, not a traceback."""

    def test_a_string_threshold_is_a_config_error(self, tmp_path) -> None:
        path = tmp_path / "providers.json"
        path.write_text(
            json.dumps({"providers": [{"provider": "openai", "model": "gpt-4o"}], "consensus_threshold": "2"})
        )

        with pytest.raises(ConfigError, match="'consensus_threshold' must be a whole number"):
            load_config(path)

    def test_a_boolean_threshold_is_a_config_error(self, tmp_path) -> None:
        path = tmp_path / "providers.json"
        path.write_text(
            json.dumps({"providers": [{"provider": "openai", "model": "gpt-4o"}], "consensus_threshold": True})
        )

        with pytest.raises(ConfigError, match="'consensus_threshold' must be a whole number"):
            load_config(path)
