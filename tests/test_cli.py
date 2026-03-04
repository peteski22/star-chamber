from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

from click.testing import CliRunner

from star_chamber.cli import main
from star_chamber.transport import ProviderResponse
from star_chamber.types import CouncilConfig, ProviderConfig

# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------

_OPENAI_CONFIG = ProviderConfig(provider="openai", model="gpt-4o")
_ANTHROPIC_CONFIG = ProviderConfig(provider="anthropic", model="claude-3")

_TWO_PROVIDER_CONFIG = CouncilConfig(
    providers=(_OPENAI_CONFIG, _ANTHROPIC_CONFIG),
    timeout_seconds=30,
    consensus_threshold=2,
)


def _code_review_json(
    quality: str = "good",
    issues: list[dict] | None = None,
    praise: list[str] | None = None,
    summary: str = "Looks solid overall.",
) -> str:
    return json.dumps(
        {
            "quality_rating": quality,
            "issues": issues or [],
            "praise": praise or ["Clean code."],
            "summary": summary,
        }
    )


def _design_json(
    recommendation: str = "Use approach A.",
    approaches: list[dict] | None = None,
    summary: str = "Approach A is recommended.",
) -> str:
    return json.dumps(
        {
            "recommendation": recommendation,
            "approaches": approaches
            or [
                {
                    "name": "Approach A",
                    "pros": ["Simple"],
                    "cons": ["Limited"],
                    "risk_level": "low",
                    "fit_rating": "good",
                }
            ],
            "summary": summary,
        }
    )


def _success_response(provider: str, model: str, content: str) -> ProviderResponse:
    return ProviderResponse(provider=provider, model=model, success=True, content=content)


# ---------------------------------------------------------------------------
# review command.
# ---------------------------------------------------------------------------


class TestReviewCommand:
    def test_review_with_temp_file(self, tmp_path: Path):
        src = tmp_path / "hello.py"
        src.write_text("print('hello')\n")

        responses = [
            _success_response("openai", "gpt-4o", _code_review_json()),
            _success_response("anthropic", "claude-3", _code_review_json()),
        ]

        with (
            patch("star_chamber.cli._load_config", return_value=_TWO_PROVIDER_CONFIG),
            patch("star_chamber.council.resolve_api_keys", return_value=_TWO_PROVIDER_CONFIG.providers),
            patch("star_chamber.council.fan_out", new_callable=AsyncMock, return_value=responses),
        ):
            runner = CliRunner()
            result = runner.invoke(main, ["review", str(src)])

        assert result.exit_code == 0
        assert "Looks solid overall" in result.output

    def test_review_format_json(self, tmp_path: Path):
        src = tmp_path / "app.py"
        src.write_text("x = 1\n")

        responses = [
            _success_response("openai", "gpt-4o", _code_review_json()),
            _success_response("anthropic", "claude-3", _code_review_json()),
        ]

        with (
            patch("star_chamber.cli._load_config", return_value=_TWO_PROVIDER_CONFIG),
            patch("star_chamber.council.resolve_api_keys", return_value=_TWO_PROVIDER_CONFIG.providers),
            patch("star_chamber.council.fan_out", new_callable=AsyncMock, return_value=responses),
        ):
            runner = CliRunner()
            result = runner.invoke(main, ["review", "--format", "json", str(src)])

        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["mode"] == "code-review"

    def test_review_nonexistent_file(self):
        runner = CliRunner()
        result = runner.invoke(main, ["review", "/nonexistent/file.py"])

        assert result.exit_code == 2

    def test_review_with_provider_flag_filters(self, tmp_path: Path):
        src = tmp_path / "filtered.py"
        src.write_text("y = 2\n")

        responses = [
            _success_response("openai", "gpt-4o", _code_review_json()),
        ]

        # Config has two providers, but we only request openai.
        with (
            patch("star_chamber.cli._load_config", return_value=_TWO_PROVIDER_CONFIG),
            patch("star_chamber.council.resolve_api_keys", return_value=(_OPENAI_CONFIG,)),
            patch("star_chamber.council.fan_out", new_callable=AsyncMock, return_value=responses) as mock_fan_out,
        ):
            runner = CliRunner()
            result = runner.invoke(main, ["review", "-p", "openai", str(src)])

        assert result.exit_code == 0
        # fan_out should have been called with only the openai provider.
        call_kwargs = mock_fan_out.call_args
        configs_arg = call_kwargs.kwargs.get("configs") or call_kwargs[0][0]
        assert len(configs_arg) == 1
        assert configs_arg[0].provider == "openai"

    def test_review_output_to_file(self, tmp_path: Path):
        src = tmp_path / "out_test.py"
        src.write_text("a = 1\n")
        output_file = tmp_path / "result.json"

        responses = [
            _success_response("openai", "gpt-4o", _code_review_json()),
        ]
        single_config = CouncilConfig(providers=(_OPENAI_CONFIG,))

        with (
            patch("star_chamber.cli._load_config", return_value=single_config),
            patch("star_chamber.council.resolve_api_keys", return_value=single_config.providers),
            patch("star_chamber.council.fan_out", new_callable=AsyncMock, return_value=responses),
        ):
            runner = CliRunner()
            result = runner.invoke(main, ["review", "--output", str(output_file), str(src)])

        assert result.exit_code == 0
        assert output_file.exists()
        parsed = json.loads(output_file.read_text())
        assert parsed["mode"] == "code-review"


# ---------------------------------------------------------------------------
# ask command.
# ---------------------------------------------------------------------------


class TestAskCommand:
    def test_ask_design_question(self):
        responses = [
            _success_response("openai", "gpt-4o", _design_json()),
            _success_response("anthropic", "claude-3", _design_json()),
        ]

        with (
            patch("star_chamber.cli._load_config", return_value=_TWO_PROVIDER_CONFIG),
            patch("star_chamber.council.resolve_api_keys", return_value=_TWO_PROVIDER_CONFIG.providers),
            patch("star_chamber.council.fan_out", new_callable=AsyncMock, return_value=responses),
        ):
            runner = CliRunner()
            result = runner.invoke(main, ["ask", "Should we use Redis?"])

        assert result.exit_code == 0
        assert "Approach A" in result.output

    def test_ask_format_json(self):
        responses = [
            _success_response("openai", "gpt-4o", _design_json()),
        ]
        single_config = CouncilConfig(providers=(_OPENAI_CONFIG,))

        with (
            patch("star_chamber.cli._load_config", return_value=single_config),
            patch("star_chamber.council.resolve_api_keys", return_value=single_config.providers),
            patch("star_chamber.council.fan_out", new_callable=AsyncMock, return_value=responses),
        ):
            runner = CliRunner()
            result = runner.invoke(main, ["ask", "--format", "json", "Should we use Redis?"])

        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["mode"] == "design-question"
        assert parsed["prompt"] == "Should we use Redis?"


# ---------------------------------------------------------------------------
# --context-file flag.
# ---------------------------------------------------------------------------


class TestContextFileFlag:
    def test_review_with_context_file(self, tmp_path: Path):
        src = tmp_path / "app.py"
        src.write_text("x = 1\n")
        ctx = tmp_path / "context.txt"
        ctx.write_text("This project uses strict typing.\n")

        responses = [
            _success_response("openai", "gpt-4o", _code_review_json()),
        ]
        single_config = CouncilConfig(providers=(_OPENAI_CONFIG,))

        with (
            patch("star_chamber.cli._load_config", return_value=single_config),
            patch("star_chamber.council.resolve_api_keys", return_value=single_config.providers),
            patch("star_chamber.council.fan_out", new_callable=AsyncMock, return_value=responses) as mock_fan_out,
        ):
            runner = CliRunner()
            result = runner.invoke(main, ["review", "--context-file", str(ctx), str(src)])

        assert result.exit_code == 0
        # Context should appear in the prompt sent to providers.
        call_kwargs = mock_fan_out.call_args
        prompt_arg = call_kwargs.kwargs.get("prompt") or call_kwargs[0][1]
        assert "strict typing" in prompt_arg

    def test_ask_with_context_file(self, tmp_path: Path):
        ctx = tmp_path / "context.txt"
        ctx.write_text("Monorepo with shared packages.\n")

        responses = [
            _success_response("openai", "gpt-4o", _design_json()),
        ]
        single_config = CouncilConfig(providers=(_OPENAI_CONFIG,))

        with (
            patch("star_chamber.cli._load_config", return_value=single_config),
            patch("star_chamber.council.resolve_api_keys", return_value=single_config.providers),
            patch("star_chamber.council.fan_out", new_callable=AsyncMock, return_value=responses) as mock_fan_out,
        ):
            runner = CliRunner()
            result = runner.invoke(main, ["ask", "--context-file", str(ctx), "Should we use Redis?"])

        assert result.exit_code == 0
        call_kwargs = mock_fan_out.call_args
        prompt_arg = call_kwargs.kwargs.get("prompt") or call_kwargs[0][1]
        assert "shared packages" in prompt_arg

    def test_review_context_file_not_found(self, tmp_path: Path):
        src = tmp_path / "app.py"
        src.write_text("x = 1\n")

        runner = CliRunner()
        result = runner.invoke(main, ["review", "--context-file", "/nonexistent/ctx.txt", str(src)])
        assert result.exit_code != 0

    def test_review_without_context_file(self, tmp_path: Path):
        """Context defaults to empty when --context-file is not provided."""
        src = tmp_path / "app.py"
        src.write_text("x = 1\n")

        responses = [
            _success_response("openai", "gpt-4o", _code_review_json()),
        ]
        single_config = CouncilConfig(providers=(_OPENAI_CONFIG,))

        with (
            patch("star_chamber.cli._load_config", return_value=single_config),
            patch("star_chamber.council.resolve_api_keys", return_value=single_config.providers),
            patch("star_chamber.council.fan_out", new_callable=AsyncMock, return_value=responses) as mock_fan_out,
        ):
            runner = CliRunner()
            result = runner.invoke(main, ["review", str(src)])

        assert result.exit_code == 0
        # Prompt should contain the "no context" fallback.
        call_kwargs = mock_fan_out.call_args
        prompt_arg = call_kwargs.kwargs.get("prompt") or call_kwargs[0][1]
        assert "No project-specific context" in prompt_arg


# ---------------------------------------------------------------------------
# list-providers command.
# ---------------------------------------------------------------------------


class TestListProvidersCommand:
    def test_list_providers_shows_names(self):
        with patch("star_chamber.cli._load_config", return_value=_TWO_PROVIDER_CONFIG):
            runner = CliRunner()
            result = runner.invoke(main, ["list-providers"])

        assert result.exit_code == 0
        assert "openai" in result.output
        assert "anthropic" in result.output
        assert "gpt-4o" in result.output
        assert "claude-3" in result.output

    def test_list_providers_shows_local_status(self):
        local_provider = ProviderConfig(provider="ollama", model="llama3", local=True)
        config = CouncilConfig(providers=(local_provider,))

        with patch("star_chamber.cli._load_config", return_value=config):
            runner = CliRunner()
            result = runner.invoke(main, ["list-providers"])

        assert result.exit_code == 0
        assert "ollama" in result.output
        assert "local" in result.output.lower()

    def test_list_providers_shows_platform_status(self):
        config = CouncilConfig(
            providers=(_OPENAI_CONFIG,),
            platform="any-llm",
        )

        with patch("star_chamber.cli._load_config", return_value=config):
            runner = CliRunner()
            result = runner.invoke(main, ["list-providers"])

        assert result.exit_code == 0
        assert "platform" in result.output.lower()


# ---------------------------------------------------------------------------
# Config error handling.
# ---------------------------------------------------------------------------


class TestConfigErrorHandling:
    def test_config_error_produces_nonzero_exit(self):
        from star_chamber.config import ConfigError

        with patch("star_chamber.cli._load_config", side_effect=ConfigError("Config file not found: /bad/path.json")):
            runner = CliRunner()
            result = runner.invoke(main, ["list-providers"])

        assert result.exit_code != 0
        assert "Config file not found" in result.output

    def test_review_config_error(self, tmp_path: Path):
        from star_chamber.config import ConfigError

        src = tmp_path / "err.py"
        src.write_text("pass\n")

        with patch("star_chamber.cli._load_config", side_effect=ConfigError("Bad config")):
            runner = CliRunner()
            result = runner.invoke(main, ["review", str(src)])

        assert result.exit_code != 0
        assert "Bad config" in result.output


# ---------------------------------------------------------------------------
# schema command.
# ---------------------------------------------------------------------------


class TestSchemaCommand:
    def test_schema_list(self):
        runner = CliRunner()
        result = runner.invoke(main, ["schema", "list"])
        assert result.exit_code == 0
        assert "code-review-result" in result.output
        assert "council-config" in result.output

    def test_schema_get(self):
        runner = CliRunner()
        result = runner.invoke(main, ["schema", "code-review-result"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, dict)

    def test_schema_unknown(self):
        runner = CliRunner()
        result = runner.invoke(main, ["schema", "nonexistent"])
        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "Error" in result.output
