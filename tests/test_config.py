from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from star_chamber.config import ConfigError, _default_config_path, load_config

TESTDATA = Path(__file__).parent / "testdata"


class TestLoadValidConfig:
    def test_load_valid_config(self):
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

    def test_load_platform_config(self):
        cfg = load_config(TESTDATA / "providers_platform.json")

        assert cfg.platform == "any-llm"
        assert len(cfg.providers) == 2
        assert cfg.providers[0].provider == "openai"
        assert cfg.providers[1].provider == "anthropic"

    def test_load_direct_config_with_local(self):
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
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            load_config(tmp_path / "does-not-exist.json")

    def test_invalid_json_raises(self, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{not valid json!!")
        with pytest.raises(ConfigError, match="Invalid JSON"):
            load_config(bad_file)

    def test_missing_providers_key_raises(self, tmp_path):
        no_providers = tmp_path / "no_providers.json"
        no_providers.write_text(json.dumps({"timeout_seconds": 60}))
        with pytest.raises(ConfigError, match="providers"):
            load_config(no_providers)

    def test_providers_not_array_raises(self):
        with pytest.raises(ConfigError, match="must be a list"):
            load_config(TESTDATA / "invalid_config.json")

    def test_empty_providers_raises(self, tmp_path):
        empty = tmp_path / "empty.json"
        empty.write_text(json.dumps({"providers": []}))
        with pytest.raises(ConfigError, match="at least one"):
            load_config(empty)

    def test_provider_missing_required_fields_raises(self, tmp_path):
        missing_model = tmp_path / "missing_model.json"
        missing_model.write_text(json.dumps({"providers": [{"provider": "openai"}]}))
        with pytest.raises(ConfigError, match="model"):
            load_config(missing_model)


class TestDefaults:
    def test_defaults_applied(self, tmp_path):
        minimal = tmp_path / "minimal.json"
        minimal.write_text(json.dumps({"providers": [{"provider": "openai", "model": "gpt-4o"}]}))
        cfg = load_config(minimal)

        assert cfg.timeout_seconds == 60
        assert cfg.consensus_threshold == 2
        assert cfg.platform is None

    def test_default_config_path(self, monkeypatch):
        # When STAR_CHAMBER_CONFIG is set, use it.
        monkeypatch.setenv("STAR_CHAMBER_CONFIG", "/tmp/custom.json")
        assert _default_config_path() == Path("/tmp/custom.json")

    def test_default_config_path_fallback(self, monkeypatch):
        # When STAR_CHAMBER_CONFIG is unset, fall back to ~/.config path.
        monkeypatch.delenv("STAR_CHAMBER_CONFIG", raising=False)
        result = _default_config_path()
        assert result == Path.home() / ".config" / "star-chamber" / "providers.json"

    def test_load_config_uses_default_path(self, tmp_path, monkeypatch):
        config_file = tmp_path / "providers.json"
        config_file.write_text(json.dumps({"providers": [{"provider": "openai", "model": "gpt-4o"}]}))
        with patch("star_chamber.config._default_config_path", return_value=config_file):
            cfg = load_config()

        assert len(cfg.providers) == 1
        assert cfg.providers[0].provider == "openai"
