"""Tests for the council orchestrator module."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from unittest.mock import AsyncMock, patch

import pytest

from star_chamber.council import run_council
from star_chamber.synthesis import Synthesis
from star_chamber.transport import ProviderResponse
from star_chamber.types import (
    CodeReviewResult,
    CouncilConfig,
    DesignQuestionResult,
    ProviderConfig,
    ProviderError,
)
from tests.helpers import SYNTHESIS, build_code_review_result, build_design_result, exchanges

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


_GOOD_ISSUE = {
    "severity": "high",
    "location": "app.py:10",
    "category": "correctness",
    "description": "Off-by-one.",
    "suggestion": "Use <=.",
}


def _code_review_json(
    quality: str = "good",
    issues: list[dict] | None = None,
    praise: list[str] | None = None,
    summary: str = "Looks solid overall.",
) -> str:
    """Build a valid code-review JSON response string."""
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
    """Build a valid design-question JSON response string."""
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


def _success_response(provider: str, model: str, content: str, display_name: str | None = None) -> ProviderResponse:
    """Build a successful ProviderResponse."""
    return ProviderResponse(
        provider=provider, display_name=display_name or provider, model=model, success=True, content=content
    )


def _error_response(provider: str, model: str, error: str, display_name: str | None = None) -> ProviderResponse:
    """Build a failed ProviderResponse."""
    return ProviderResponse(
        provider=provider, display_name=display_name or provider, model=model, success=False, error=error
    )


# ---------------------------------------------------------------------------
# _build_code_review_result
# ---------------------------------------------------------------------------


class TestBuildCodeReviewResult:
    def test_successful_two_providers(self) -> None:
        issue_data = {
            "severity": "medium",
            "location": "main.py:10",
            "category": "craftsmanship",
            "description": "Missing docstring.",
            "suggestion": "Add a docstring.",
        }
        responses = [
            _success_response("openai", "gpt-4o", _code_review_json(issues=[issue_data])),
            _success_response("anthropic", "claude-3", _code_review_json(issues=[issue_data])),
        ]

        result = build_code_review_result(responses, files_reviewed=("app.py",), threshold=2)

        assert isinstance(result.reviews, tuple)
        assert len(result.reviews) == 2
        assert len(result.failed_providers) == 0
        # Both providers flagged the same issue, so it should be consensus.
        assert len(result.consensus_issues) == 1
        assert result.consensus_issues[0].location == "main.py:10"

    def test_parse_error_recorded_as_failed_provider(self) -> None:
        responses = [
            _success_response("openai", "gpt-4o", _code_review_json()),
            _success_response("anthropic", "claude-3", "not valid json at all"),
        ]

        result = build_code_review_result(responses, files_reviewed=("app.py",), threshold=2)

        assert len(result.reviews) == 1
        assert result.reviews[0].provider == "openai"
        assert len(result.failed_providers) == 1
        assert result.failed_providers[0].provider == "anthropic"
        assert "parse" in result.failed_providers[0].error.lower() or "json" in result.failed_providers[0].error.lower()

    def test_transport_error_recorded_as_failed_provider(self) -> None:
        responses = [
            _success_response("openai", "gpt-4o", _code_review_json()),
            _error_response("anthropic", "claude-3", "Timeout: provider anthropic did not respond in time."),
        ]

        result = build_code_review_result(responses, files_reviewed=("app.py",), threshold=2)

        assert len(result.reviews) == 1
        assert len(result.failed_providers) == 1
        assert result.failed_providers[0].provider == "anthropic"
        assert "timeout" in result.failed_providers[0].error.lower()

    def test_quality_ratings_mapped_by_provider(self) -> None:
        responses = [
            _success_response("openai", "gpt-4o", _code_review_json(quality="excellent")),
            _success_response("anthropic", "claude-3", _code_review_json(quality="good")),
        ]

        result = build_code_review_result(responses, files_reviewed=("app.py",), threshold=2)

        assert result.quality_ratings["openai"] == "excellent"
        assert result.quality_ratings["anthropic"] == "good"


# ---------------------------------------------------------------------------
# _build_design_result
# ---------------------------------------------------------------------------


class TestBuildDesignQuestionResult:
    def test_successful_two_providers(self) -> None:
        responses = [
            _success_response("openai", "gpt-4o", _design_json()),
            _success_response("anthropic", "claude-3", _design_json(recommendation="Use approach B.")),
        ]

        result = build_design_result(responses, prompt="How should we structure the API?")

        assert isinstance(result.approaches, tuple)
        assert len(result.approaches) == 1
        assert result.approaches[0].recommended_by == 1
        assert len(result.failed_providers) == 0
        assert result.prompt == "How should we structure the API?"

    def test_parse_error_recorded_as_failed_provider(self) -> None:
        responses = [
            _success_response("openai", "gpt-4o", _design_json()),
            _success_response("anthropic", "claude-3", "garbage response"),
        ]

        result = build_design_result(responses, prompt="Design question?")

        assert len(result.approaches) == 1
        assert len(result.failed_providers) == 1
        assert result.failed_providers[0].provider == "anthropic"

    def test_consensus_recommendation_when_all_agree(self) -> None:
        responses = [
            _success_response("openai", "gpt-4o", _design_json(recommendation="Use Approach A.")),
            _success_response("anthropic", "claude-3", _design_json(recommendation="Use Approach A.")),
        ]

        result = build_design_result(responses, prompt="How to improve perf?")

        assert result.consensus_recommendation == "Approach A"

    def test_no_consensus_when_providers_disagree(self) -> None:
        responses = [
            _success_response("openai", "gpt-4o", _design_json(recommendation="Use caching.")),
            _success_response("anthropic", "claude-3", _design_json(recommendation="Optimize queries.")),
        ]

        result = build_design_result(responses, prompt="How to improve perf?")

        assert result.consensus_recommendation is None


# ---------------------------------------------------------------------------
# run_council — code-review mode.
# ---------------------------------------------------------------------------


class TestRunCouncilCodeReview:
    async def test_successful_code_review(self) -> None:
        issue_data = {
            "severity": "medium",
            "location": "app.py:5",
            "category": "craftsmanship",
            "description": "Unused import.",
            "suggestion": "Remove the import.",
        }
        mock_responses = [
            _success_response("openai", "gpt-4o", _code_review_json(issues=[issue_data])),
            _success_response("anthropic", "claude-3", _code_review_json(issues=[issue_data])),
        ]

        with (
            patch("star_chamber.council.resolve_api_keys", return_value=_TWO_PROVIDER_CONFIG.providers),
            patch("star_chamber.council.fan_out", new_callable=AsyncMock, return_value=exchanges(mock_responses)),
        ):
            result = await run_council(
                files={"app.py": "import os\nprint('hello')"},
                config=_TWO_PROVIDER_CONFIG,
                mode="code-review",
            )

        assert isinstance(result, CodeReviewResult)
        assert result.mode == "code-review"
        assert set(result.providers_used) == {"openai", "anthropic"}
        assert len(result.reviews) == 2
        assert len(result.failed_providers) == 0

    async def test_partial_failure(self) -> None:
        mock_responses = [
            _success_response("openai", "gpt-4o", _code_review_json()),
            _error_response("anthropic", "claude-3", "Rate limit exceeded."),
        ]

        with (
            patch("star_chamber.council.resolve_api_keys", return_value=_TWO_PROVIDER_CONFIG.providers),
            patch("star_chamber.council.fan_out", new_callable=AsyncMock, return_value=exchanges(mock_responses)),
        ):
            result = await run_council(
                files={"app.py": "print('hello')"},
                config=_TWO_PROVIDER_CONFIG,
                mode="code-review",
            )

        assert isinstance(result, CodeReviewResult)
        assert len(result.reviews) == 1
        assert len(result.failed_providers) == 1
        assert result.failed_providers[0].provider == "anthropic"

    async def test_all_providers_fail(self) -> None:
        mock_responses = [
            _error_response("openai", "gpt-4o", "Service unavailable."),
            _error_response("anthropic", "claude-3", "Rate limit exceeded."),
        ]

        with (
            patch("star_chamber.council.resolve_api_keys", return_value=_TWO_PROVIDER_CONFIG.providers),
            patch("star_chamber.council.fan_out", new_callable=AsyncMock, return_value=exchanges(mock_responses)),
        ):
            result = await run_council(
                files={"app.py": "print('hello')"},
                config=_TWO_PROVIDER_CONFIG,
                mode="code-review",
            )

        assert isinstance(result, CodeReviewResult)
        assert len(result.reviews) == 0
        assert len(result.failed_providers) == 2

    async def test_consensus_classification_flows_through(self) -> None:
        issue_data = {
            "severity": "high",
            "location": "auth.py:15",
            "category": "architecture",
            "description": "SQL injection risk.",
            "suggestion": "Use parameterized queries.",
        }
        mock_responses = [
            _success_response("openai", "gpt-4o", _code_review_json(issues=[issue_data])),
            _success_response("anthropic", "claude-3", _code_review_json(issues=[issue_data])),
        ]

        with (
            patch("star_chamber.council.resolve_api_keys", return_value=_TWO_PROVIDER_CONFIG.providers),
            patch("star_chamber.council.fan_out", new_callable=AsyncMock, return_value=exchanges(mock_responses)),
        ):
            result = await run_council(
                files={"auth.py": "query = f'SELECT * FROM users WHERE id={uid}'"},
                config=_TWO_PROVIDER_CONFIG,
                mode="code-review",
            )

        assert len(result.consensus_issues) == 1
        assert result.consensus_issues[0].category == "architecture"

    async def test_parse_error_becomes_failed_provider(self) -> None:
        mock_responses = [
            _success_response("openai", "gpt-4o", _code_review_json()),
            _success_response("anthropic", "claude-3", "this is not json"),
        ]

        with (
            patch("star_chamber.council.resolve_api_keys", return_value=_TWO_PROVIDER_CONFIG.providers),
            patch("star_chamber.council.fan_out", new_callable=AsyncMock, return_value=exchanges(mock_responses)),
        ):
            result = await run_council(
                files={"app.py": "print('hello')"},
                config=_TWO_PROVIDER_CONFIG,
                mode="code-review",
            )

        assert len(result.reviews) == 1
        assert len(result.failed_providers) == 1
        assert result.failed_providers[0].provider == "anthropic"

    async def test_auto_loads_config_when_none(self) -> None:
        mock_responses = [
            _success_response("openai", "gpt-4o", _code_review_json()),
        ]
        auto_config = CouncilConfig(providers=(_OPENAI_CONFIG,))

        with (
            patch("star_chamber.council.load_config", return_value=auto_config),
            patch("star_chamber.council.resolve_api_keys", return_value=auto_config.providers),
            patch("star_chamber.council.fan_out", new_callable=AsyncMock, return_value=exchanges(mock_responses)),
        ):
            result = await run_council(
                files={"app.py": "print('hello')"},
                config=None,
                mode="code-review",
            )

        assert isinstance(result, CodeReviewResult)
        assert len(result.reviews) == 1


# ---------------------------------------------------------------------------
# run_council — design-question mode.
# ---------------------------------------------------------------------------


class TestRunCouncilDesignQuestion:
    async def test_design_question_returns_design_result(self) -> None:
        mock_responses = [
            _success_response("openai", "gpt-4o", _design_json()),
            _success_response("anthropic", "claude-3", _design_json(recommendation="Use approach B.")),
        ]

        with (
            patch("star_chamber.council.resolve_api_keys", return_value=_TWO_PROVIDER_CONFIG.providers),
            patch("star_chamber.council.fan_out", new_callable=AsyncMock, return_value=exchanges(mock_responses)),
        ):
            result = await run_council(
                prompt="Should we use a monorepo?",
                config=_TWO_PROVIDER_CONFIG,
                mode="design-question",
            )

        assert isinstance(result, DesignQuestionResult)
        assert result.mode == "design-question"
        assert result.prompt == "Should we use a monorepo?"
        assert set(result.providers_used) == {"openai", "anthropic"}
        assert len(result.approaches) == 1

    async def test_design_question_partial_failure(self) -> None:
        mock_responses = [
            _success_response("openai", "gpt-4o", _design_json()),
            _error_response("anthropic", "claude-3", "Connection refused."),
        ]

        with (
            patch("star_chamber.council.resolve_api_keys", return_value=_TWO_PROVIDER_CONFIG.providers),
            patch("star_chamber.council.fan_out", new_callable=AsyncMock, return_value=exchanges(mock_responses)),
        ):
            result = await run_council(
                prompt="How to handle auth?",
                config=_TWO_PROVIDER_CONFIG,
                mode="design-question",
            )

        assert isinstance(result, DesignQuestionResult)
        assert len(result.failed_providers) == 1
        assert result.failed_providers[0].provider == "anthropic"


# ---------------------------------------------------------------------------
# run_council_sync
# ---------------------------------------------------------------------------


class TestRunCouncilSync:
    def test_sync_wrapper_works(self) -> None:
        from star_chamber.council import run_council_sync

        mock_responses = [
            _success_response("openai", "gpt-4o", _code_review_json()),
        ]
        single_config = CouncilConfig(providers=(_OPENAI_CONFIG,))

        with (
            patch("star_chamber.council.resolve_api_keys", return_value=single_config.providers),
            patch("star_chamber.council.fan_out", new_callable=AsyncMock, return_value=exchanges(mock_responses)),
        ):
            result = run_council_sync(
                files={"app.py": "print('hello')"},
                config=single_config,
                mode="code-review",
            )

        assert isinstance(result, CodeReviewResult)
        assert result.mode == "code-review"

    def test_sync_wrapper_design_mode(self) -> None:
        from star_chamber.council import run_council_sync

        mock_responses = [
            _success_response("openai", "gpt-4o", _design_json()),
        ]
        single_config = CouncilConfig(providers=(_OPENAI_CONFIG,))

        with (
            patch("star_chamber.council.resolve_api_keys", return_value=single_config.providers),
            patch("star_chamber.council.fan_out", new_callable=AsyncMock, return_value=exchanges(mock_responses)),
        ):
            result = run_council_sync(
                prompt="How to design this?",
                config=single_config,
                mode="design-question",
            )

        assert isinstance(result, DesignQuestionResult)
        assert result.mode == "design-question"


# ---------------------------------------------------------------------------
# run_council — council_context passthrough.
# ---------------------------------------------------------------------------


class TestRunCouncilCouncilContext:
    async def test_council_context_appears_in_code_review_prompt(self) -> None:
        mock_responses = [
            _success_response("openai", "gpt-4o", _code_review_json()),
        ]
        single_config = CouncilConfig(providers=(_OPENAI_CONFIG,))
        mock_fan_out = AsyncMock(return_value=exchanges(mock_responses))

        with (
            patch("star_chamber.council.resolve_api_keys", return_value=single_config.providers),
            patch("star_chamber.council.fan_out", mock_fan_out),
        ):
            await run_council(
                files={"app.py": "x = 1"},
                config=single_config,
                mode="code-review",
                council_context="Round 1: missing error handling.",
            )

        prompt_arg = mock_fan_out.call_args.kwargs.get("prompt") or mock_fan_out.call_args[0][1]
        assert "Previous Council Feedback" in prompt_arg
        assert "missing error handling" in prompt_arg

    async def test_council_context_absent_when_not_provided(self) -> None:
        mock_responses = [
            _success_response("openai", "gpt-4o", _code_review_json()),
        ]
        single_config = CouncilConfig(providers=(_OPENAI_CONFIG,))
        mock_fan_out = AsyncMock(return_value=exchanges(mock_responses))

        with (
            patch("star_chamber.council.resolve_api_keys", return_value=single_config.providers),
            patch("star_chamber.council.fan_out", mock_fan_out),
        ):
            await run_council(
                files={"app.py": "x = 1"},
                config=single_config,
                mode="code-review",
            )

        prompt_arg = mock_fan_out.call_args.kwargs.get("prompt") or mock_fan_out.call_args[0][1]
        assert "Previous Council Feedback" not in prompt_arg

    async def test_council_context_appears_in_design_prompt(self) -> None:
        mock_responses = [
            _success_response("openai", "gpt-4o", _design_json()),
        ]
        single_config = CouncilConfig(providers=(_OPENAI_CONFIG,))
        mock_fan_out = AsyncMock(return_value=exchanges(mock_responses))

        with (
            patch("star_chamber.council.resolve_api_keys", return_value=single_config.providers),
            patch("star_chamber.council.fan_out", mock_fan_out),
        ):
            await run_council(
                prompt="Should we use Redis?",
                config=single_config,
                mode="design-question",
                council_context="Council favoured Memcached.",
            )

        prompt_arg = mock_fan_out.call_args.kwargs.get("prompt") or mock_fan_out.call_args[0][1]
        assert "Previous Council Feedback" in prompt_arg
        assert "Council favoured Memcached." in prompt_arg


# ---------------------------------------------------------------------------
# run_council — single-round only.
# ---------------------------------------------------------------------------


class TestRunCouncilSingleRoundOnly:
    """Verify the SDK always does exactly one fan-out call."""

    async def test_single_fan_out_call(self) -> None:
        """run_council() always calls fan_out exactly once."""
        mock_responses = [
            _success_response("openai", "gpt-4o", _code_review_json()),
            _success_response("anthropic", "claude-3", _code_review_json()),
        ]
        mock_fan_out = AsyncMock(return_value=exchanges(mock_responses))

        with (
            patch("star_chamber.council.resolve_api_keys", return_value=_TWO_PROVIDER_CONFIG.providers),
            patch("star_chamber.council.fan_out", mock_fan_out),
        ):
            await run_council(
                files={"app.py": "print('hello')"},
                config=_TWO_PROVIDER_CONFIG,
                mode="code-review",
            )

        assert mock_fan_out.call_count == 1

    async def test_no_debate_parameter(self) -> None:
        """run_council() does not accept debate or rounds parameters."""
        with pytest.raises(TypeError):
            await run_council(
                files={"app.py": "x = 1"},
                config=_TWO_PROVIDER_CONFIG,
                mode="code-review",
                debate=True,
            )

    async def test_no_rounds_parameter(self) -> None:
        """run_council() does not accept a rounds parameter."""
        with pytest.raises(TypeError):
            await run_council(
                files={"app.py": "x = 1"},
                config=_TWO_PROVIDER_CONFIG,
                mode="code-review",
                rounds=3,
            )


# ---------------------------------------------------------------------------
# run_council — validation.
# ---------------------------------------------------------------------------


class TestRunCouncilValidation:
    async def test_code_review_requires_files(self) -> None:
        with pytest.raises(ValueError, match="files"):
            await run_council(
                config=_TWO_PROVIDER_CONFIG,
                mode="code-review",
            )

    async def test_design_question_requires_prompt(self) -> None:
        with pytest.raises(ValueError, match="prompt"):
            await run_council(
                config=_TWO_PROVIDER_CONFIG,
                mode="design-question",
            )

    async def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="mode"):
            await run_council(
                config=_TWO_PROVIDER_CONFIG,
                mode="invalid-mode",
            )


class TestAnsweringRoster:
    """providers_responded must carry only the members that returned a usable answer."""

    async def test_transport_failure_is_absent_from_the_answering_roster(self) -> None:
        mock_responses = [
            _success_response("openai", "gpt-4o", _code_review_json()),
            _error_response("anthropic", "claude-3", "Rate limit exceeded."),
        ]

        with (
            patch("star_chamber.council.resolve_api_keys", return_value=_TWO_PROVIDER_CONFIG.providers),
            patch("star_chamber.council.fan_out", new_callable=AsyncMock, return_value=exchanges(mock_responses)),
        ):
            result = await run_council(files={"app.py": "x = 1"}, config=_TWO_PROVIDER_CONFIG, mode="code-review")

        assert set(result.providers_used) == {"openai", "anthropic"}
        assert result.providers_responded == ("openai",)

    async def test_unparsable_reply_is_absent_from_the_answering_roster(self) -> None:
        mock_responses = [
            _success_response("openai", "gpt-4o", _code_review_json()),
            _success_response("anthropic", "claude-3", "I cannot answer that."),
        ]

        with (
            patch("star_chamber.council.resolve_api_keys", return_value=_TWO_PROVIDER_CONFIG.providers),
            patch("star_chamber.council.fan_out", new_callable=AsyncMock, return_value=exchanges(mock_responses)),
        ):
            result = await run_council(files={"app.py": "x = 1"}, config=_TWO_PROVIDER_CONFIG, mode="code-review")

        assert set(result.providers_used) == {"openai", "anthropic"}
        assert result.providers_responded == ("openai",)

    async def test_answering_roster_is_the_denominator_the_issues_were_classified_over(self) -> None:
        issue_data = {
            "severity": "high",
            "location": "app.py:5",
            "category": "craftsmanship",
            "description": "Unused import.",
            "suggestion": "Remove the import.",
        }
        mock_responses = [
            _success_response("openai", "gpt-4o", _code_review_json(issues=[issue_data])),
            _success_response("anthropic", "claude-3", _code_review_json(issues=[issue_data])),
            _error_response("gemini", "gemini-2.0", "Timeout."),
        ]
        config = CouncilConfig(
            providers=(
                ProviderConfig(provider="openai", model="gpt-4o"),
                ProviderConfig(provider="anthropic", model="claude-3"),
                ProviderConfig(provider="gemini", model="gemini-2.0"),
            ),
        )

        with (
            patch("star_chamber.council.resolve_api_keys", return_value=config.providers),
            patch("star_chamber.council.fan_out", new_callable=AsyncMock, return_value=exchanges(mock_responses)),
        ):
            result = await run_council(files={"app.py": "x = 1"}, config=config, mode="code-review")

        # Three members were asked, two answered, and both flagged the issue, so it is
        # consensus across the two that answered rather than across the three asked.
        assert len(result.providers_used) == 3
        assert set(result.providers_responded) == {"openai", "anthropic"}
        assert len(result.consensus_issues) == 1
        assert len(result.providers_responded) == len(result.reviews)

    async def test_answering_roster_uses_the_display_name(self) -> None:
        mock_responses = [
            _success_response("openai", "gpt-4o", _code_review_json(), display_name="scribe"),
        ]
        config = CouncilConfig(providers=(ProviderConfig(provider="openai", model="gpt-4o", display_name="scribe"),))

        with (
            patch("star_chamber.council.resolve_api_keys", return_value=config.providers),
            patch("star_chamber.council.fan_out", new_callable=AsyncMock, return_value=exchanges(mock_responses)),
        ):
            result = await run_council(files={"app.py": "x = 1"}, config=config, mode="code-review")

        assert result.providers_responded == ("scribe",)

    async def test_design_question_reports_its_answering_roster(self) -> None:
        mock_responses = [
            _success_response("openai", "gpt-4o", _design_json()),
            _error_response("anthropic", "claude-3", "Connection refused."),
        ]

        with (
            patch("star_chamber.council.resolve_api_keys", return_value=_TWO_PROVIDER_CONFIG.providers),
            patch("star_chamber.council.fan_out", new_callable=AsyncMock, return_value=exchanges(mock_responses)),
        ):
            result = await run_council(prompt="Monorepo?", config=_TWO_PROVIDER_CONFIG, mode="design-question")

        assert set(result.providers_used) == {"openai", "anthropic"}
        assert result.providers_responded == ("openai",)

    async def test_no_answers_leaves_the_answering_roster_empty(self) -> None:
        mock_responses = [
            _error_response("openai", "gpt-4o", "Timeout."),
            _error_response("anthropic", "claude-3", "Timeout."),
        ]

        with (
            patch("star_chamber.council.resolve_api_keys", return_value=_TWO_PROVIDER_CONFIG.providers),
            patch("star_chamber.council.fan_out", new_callable=AsyncMock, return_value=exchanges(mock_responses)),
        ):
            result = await run_council(files={"app.py": "x = 1"}, config=_TWO_PROVIDER_CONFIG, mode="code-review")

        assert result.providers_responded == ()


class TestRosterComplement:
    """Each response lands in exactly one roster, so the two rosters stay complementary."""

    @pytest.mark.parametrize(
        "responses",
        [
            pytest.param(
                [
                    _success_response("openai", "gpt-4o", _code_review_json()),
                    _error_response("anthropic", "claude-3", "Timeout."),
                    _success_response("gemini", "gemini-2.0", _code_review_json()),
                ],
                id="transport-failure",
            ),
            pytest.param(
                [
                    _success_response("openai", "gpt-4o", _code_review_json()),
                    _success_response("anthropic", "claude-3", '{"quality_rating": "go'),
                ],
                id="reply-cut-off-mid-json",
            ),
            pytest.param(
                [
                    _success_response("openai", "gpt-4o", _code_review_json()),
                    _success_response("anthropic", "claude-3", "I cannot answer that."),
                ],
                id="prose-instead-of-json",
            ),
            pytest.param(
                [
                    _success_response("openai", "gpt-4o", _code_review_json()),
                    _success_response("anthropic", "claude-3", json.dumps({"issues": []})),
                ],
                id="missing-required-field",
            ),
            pytest.param(
                [_error_response("openai", "gpt-4o", "Timeout.")],
                id="every-member-fails",
            ),
            pytest.param(
                [_success_response("openai", "gpt-4o", _code_review_json())],
                id="every-member-answers",
            ),
        ],
    )
    def test_code_review_rosters_partition_the_members_asked(self, responses) -> None:
        result = build_code_review_result(responses, files_reviewed=("app.py",))

        failed = {f.provider for f in result.failed_providers}
        assert tuple(p for p in result.providers_used if p not in failed) == result.providers_responded
        assert len(result.providers_responded) + len(result.failed_providers) == len(result.providers_used)

    @pytest.mark.parametrize(
        "responses",
        [
            pytest.param(
                [
                    _success_response("openai", "gpt-4o", _design_json()),
                    _error_response("anthropic", "claude-3", "Connection refused."),
                ],
                id="transport-failure",
            ),
            pytest.param(
                [
                    _success_response("openai", "gpt-4o", _design_json()),
                    _success_response("anthropic", "claude-3", '{"recommendation": "re'),
                ],
                id="reply-cut-off-mid-json",
            ),
            pytest.param(
                [_error_response("openai", "gpt-4o", "Timeout.")],
                id="every-member-fails",
            ),
        ],
    )
    def test_design_rosters_partition_the_members_asked(self, responses) -> None:
        result = build_design_result(responses, prompt="Monorepo?")

        failed = {f.provider for f in result.failed_providers}
        assert tuple(p for p in result.providers_used if p not in failed) == result.providers_responded
        assert len(result.providers_responded) + len(result.failed_providers) == len(result.providers_used)


class TestMalformedMemberFailsAlone:
    """A member that sends an unusable reply costs the council that member, not the session.

    Every field of a reply is attacker-shaped data, so the parser raises ParseError and the
    member lands in failed_providers alongside the members that timed out.
    """

    @pytest.mark.parametrize(
        ("label", "issue"),
        [
            ("null location", {**_GOOD_ISSUE, "location": None}),
            ("numeric location", {**_GOOD_ISSUE, "location": 12}),
            ("no severity", {k: v for k, v in _GOOD_ISSUE.items() if k != "severity"}),
            ("no location", {k: v for k, v in _GOOD_ISSUE.items() if k != "location"}),
        ],
    )
    def test_one_bad_issue_does_not_end_a_code_review(self, label, issue) -> None:
        responses = [
            _success_response("openai", "gpt-4o", _code_review_json(issues=[dict(_GOOD_ISSUE)])),
            _success_response("anthropic", "claude-opus-5", _code_review_json(issues=[issue])),
            _success_response("gemini", "gemini-2.0", _code_review_json(issues=[dict(_GOOD_ISSUE)])),
        ]

        result = build_code_review_result(responses, files_reviewed=("app.py",))

        assert result.providers_responded == ("openai", "gemini")
        assert [failure.provider for failure in result.failed_providers] == ["anthropic"]
        assert result.providers_used == ("openai", "anthropic", "gemini")

    def test_the_surviving_members_still_reach_consensus(self) -> None:
        bad = {**_GOOD_ISSUE, "location": None}
        responses = [
            _success_response("openai", "gpt-4o", _code_review_json(issues=[dict(_GOOD_ISSUE)])),
            _success_response("anthropic", "claude-opus-5", _code_review_json(issues=[bad])),
            _success_response("gemini", "gemini-2.0", _code_review_json(issues=[dict(_GOOD_ISSUE)])),
        ]

        result = build_code_review_result(responses, files_reviewed=("app.py",))

        assert len(result.consensus_issues) == 1
        assert result.summary.consensus_count == 1

    @pytest.mark.parametrize(
        ("label", "approaches"),
        [
            ("approaches not a list", "none"),
            ("approach not an object", ["none"]),
            ("approach without a name", [{}]),
            ("null approach name", [{"name": None}]),
        ],
    )
    def test_one_bad_approach_does_not_end_a_design_question(self, label, approaches) -> None:
        responses = [
            _success_response("openai", "gpt-4o", _design_json()),
            _success_response("anthropic", "claude-opus-5", _design_json(approaches=approaches)),
        ]

        result = build_design_result(responses, prompt="Which data layer?")

        assert result.providers_responded == ("openai",)
        assert [failure.provider for failure in result.failed_providers] == ["anthropic"]

    def test_the_failure_names_the_field_and_the_shape(self) -> None:
        bad = {**_GOOD_ISSUE, "location": None}
        responses = [_success_response("anthropic", "claude-opus-5", _code_review_json(issues=[bad]))]

        result = build_code_review_result(responses, files_reviewed=("app.py",))

        assert "location" in result.failed_providers[0].error
        assert "null" in result.failed_providers[0].error


# ---------------------------------------------------------------------------
# _build_design_result — approach aggregation.
# ---------------------------------------------------------------------------


def _approach(
    name: str,
    pros: list[str] | None = None,
    cons: list[str] | None = None,
    risk_level: str = "low",
    fit_rating: str | None = "good",
) -> dict:
    approach: dict = {
        "name": name,
        "pros": pros if pros is not None else ["Simple"],
        "cons": cons if cons is not None else ["Limited"],
        "risk_level": risk_level,
    }
    if fit_rating is not None:
        approach["fit_rating"] = fit_rating
    return approach


class TestDesignApproachAggregation:
    """Members name the same approach, so the council must report it once.

    A member cannot send recommended_by, so the council has to derive it from
    the primary recommendation each member gave.
    """

    def _result(self, *bodies: str):
        responses = [_success_response(f"member-{index}", "model", body) for index, body in enumerate(bodies)]
        return build_design_result(responses, prompt="Which data layer?")

    def test_same_approach_from_two_members_merges(self) -> None:
        result = self._result(
            _design_json(approaches=[_approach("Repository")]),
            _design_json(approaches=[_approach("Repository")]),
        )

        assert len(result.approaches) == 1
        assert result.approaches[0].name == "Repository"

    def test_merge_unions_pros_and_cons_without_duplicates(self) -> None:
        result = self._result(
            _design_json(approaches=[_approach("Repository", pros=["Testable"], cons=["Indirection"])]),
            _design_json(approaches=[_approach("Repository", pros=["Testable", "Swappable"], cons=["Boilerplate"])]),
        )

        assert result.approaches[0].pros == ("Testable", "Swappable")
        assert result.approaches[0].cons == ("Indirection", "Boilerplate")

    def test_recommended_by_counts_members_that_recommended_it(self) -> None:
        result = self._result(
            _design_json(recommendation="Use the Repository pattern.", approaches=[_approach("Repository")]),
            _design_json(recommendation="Go with Repository here.", approaches=[_approach("Repository")]),
        )

        assert result.approaches[0].recommended_by == 2

    def test_discussed_but_unrecommended_approach_counts_zero(self) -> None:
        result = self._result(
            _design_json(
                recommendation="Use the Repository pattern.",
                approaches=[_approach("Repository"), _approach("Active Record")],
            )
        )

        counts = {approach.name: approach.recommended_by for approach in result.approaches}
        assert counts == {"Repository": 1, "Active Record": 0}

    def test_name_match_ignores_case_and_spacing(self) -> None:
        result = self._result(
            _design_json(recommendation="use   a REPOSITORY layer", approaches=[_approach("Repository")])
        )

        assert result.approaches[0].recommended_by == 1

    def test_name_matches_on_word_boundaries_only(self) -> None:
        """An approach named ORM must not match the word normalize."""
        result = self._result(_design_json(recommendation="Normalize the schema first.", approaches=[_approach("ORM")]))

        assert result.approaches[0].recommended_by == 0

    def test_member_listing_an_approach_twice_counts_once(self) -> None:
        result = self._result(
            _design_json(
                recommendation="Use the Repository pattern.",
                approaches=[_approach("Repository"), _approach("Repository")],
            )
        )

        assert len(result.approaches) == 1
        assert result.approaches[0].recommended_by == 1

    def test_risk_level_takes_the_most_common_value(self) -> None:
        result = self._result(
            _design_json(approaches=[_approach("Repository", risk_level="low")]),
            _design_json(approaches=[_approach("Repository", risk_level="high")]),
            _design_json(approaches=[_approach("Repository", risk_level="high")]),
        )

        assert result.approaches[0].risk_level == "high"

    def test_tied_risk_level_takes_the_more_cautious_value(self) -> None:
        result = self._result(
            _design_json(approaches=[_approach("Repository", risk_level="low")]),
            _design_json(approaches=[_approach("Repository", risk_level="high")]),
        )

        assert result.approaches[0].risk_level == "high"

    def test_risk_level_is_unknown_when_no_member_stated_one(self) -> None:
        body = json.dumps(
            {
                "recommendation": "Use the Repository pattern.",
                "approaches": [{"name": "Repository", "pros": ["Testable"], "cons": ["Indirection"]}],
                "summary": "Repository fits.",
            }
        )

        result = self._result(body)

        assert result.approaches[0].risk_level == "unknown"

    def test_fit_rating_takes_the_most_common_value(self) -> None:
        result = self._result(
            _design_json(approaches=[_approach("Repository", fit_rating="excellent")]),
            _design_json(approaches=[_approach("Repository", fit_rating="good")]),
            _design_json(approaches=[_approach("Repository", fit_rating="good")]),
        )

        assert result.approaches[0].fit_rating == "good"

    def test_tied_fit_rating_takes_the_more_cautious_value(self) -> None:
        result = self._result(
            _design_json(approaches=[_approach("Repository", fit_rating="excellent")]),
            _design_json(approaches=[_approach("Repository", fit_rating="fair")]),
        )

        assert result.approaches[0].fit_rating == "fair"

    def test_fit_rating_is_null_when_no_member_assessed_fit(self) -> None:
        result = self._result(
            _design_json(approaches=[_approach("Repository", fit_rating=None)]),
            _design_json(approaches=[_approach("Repository", fit_rating=None)]),
        )

        assert result.approaches[0].fit_rating is None

    def test_consensus_fires_when_members_pick_the_same_approach_in_different_words(self) -> None:
        result = self._result(
            _design_json(recommendation="Use the Repository pattern.", approaches=[_approach("Repository")]),
            _design_json(recommendation="I would reach for Repository.", approaches=[_approach("Repository")]),
        )

        assert result.consensus_recommendation == "Repository"

    def test_no_consensus_when_members_pick_different_approaches(self) -> None:
        result = self._result(
            _design_json(
                recommendation="Use Repository.",
                approaches=[_approach("Repository"), _approach("Active Record")],
            ),
            _design_json(
                recommendation="Use Active Record.",
                approaches=[_approach("Repository"), _approach("Active Record")],
            ),
        )

        assert result.consensus_recommendation is None


class TestDesignAggregationVoting:
    """Each provider gets one vote per approach, and only for approaches it named.

    Counting an approach's entries instead of its providers lets one provider
    that names an approach twice outvote a provider that names it once.
    """

    def _result(self, *bodies: str):
        responses = [_success_response(f"member-{index}", "model", body) for index, body in enumerate(bodies)]
        return build_design_result(responses, prompt="Which data layer?")

    def test_provider_naming_an_approach_twice_casts_one_risk_vote(self) -> None:
        result = self._result(
            _design_json(
                approaches=[
                    _approach("Repository", risk_level="low"),
                    _approach("Repository", risk_level="low"),
                ]
            ),
            _design_json(approaches=[_approach("Repository", risk_level="high")]),
        )

        assert result.approaches[0].risk_level == "high"

    def test_provider_naming_an_approach_twice_casts_one_fit_vote(self) -> None:
        result = self._result(
            _design_json(
                approaches=[
                    _approach("Repository", fit_rating="excellent"),
                    _approach("Repository", fit_rating="excellent"),
                ]
            ),
            _design_json(approaches=[_approach("Repository", fit_rating="fair")]),
        )

        assert result.approaches[0].fit_rating == "fair"

    def test_provider_is_not_counted_for_an_approach_it_did_not_name(self) -> None:
        """A provider recommends from the approaches it offered, not from another provider's."""
        result = self._result(
            _design_json(recommendation="Use Active Record.", approaches=[_approach("Repository")]),
            _design_json(recommendation="Use Active Record.", approaches=[_approach("Active Record")]),
        )

        counts = {approach.name: approach.recommended_by for approach in result.approaches}
        assert counts == {"Repository": 0, "Active Record": 1}

    def test_names_that_differ_only_by_symbols_stay_distinct(self) -> None:
        """Stripping every symbol collapses C++, C# and C into one approach."""
        result = self._result(
            _design_json(
                approaches=[_approach("C++"), _approach("C#"), _approach("C")],
            )
        )

        assert [approach.name for approach in result.approaches] == ["C++", "C#", "C"]


class TestDesignApproachNameMatching:
    """A name has to survive normalization, and has to match only where it stands alone.

    Dropping the letters of a name empties its key, and the approach disappears
    from the result. Matching a short name inside a longer one credits an
    approach the member never chose.
    """

    def _result(self, *bodies: str):
        responses = [_success_response(f"member-{index}", "model", body) for index, body in enumerate(bodies)]
        return build_design_result(responses, prompt="Which data layer?")

    def test_a_name_outside_the_latin_alphabet_is_reported(self) -> None:
        result = self._result(_design_json(approaches=[_approach("事件溯源")]))

        assert [approach.name for approach in result.approaches] == ["事件溯源"]

    def test_a_name_outside_the_latin_alphabet_can_be_recommended(self) -> None:
        result = self._result(
            _design_json(recommendation="採用 事件溯源 吧。", approaches=[_approach("事件溯源")]),
        )

        assert result.approaches[0].recommended_by == 1

    def test_a_short_name_inside_a_longer_one_is_not_counted(self) -> None:
        result = self._result(
            _design_json(
                recommendation="Use CRUD with domain events.",
                approaches=[_approach("CRUD"), _approach("CRUD with domain events")],
            ),
        )

        counts = {approach.name: approach.recommended_by for approach in result.approaches}
        assert counts == {"CRUD": 0, "CRUD with domain events": 1}

    def test_a_short_name_named_on_its_own_is_counted(self) -> None:
        result = self._result(
            _design_json(
                recommendation="Use CRUD with domain events, because plain CRUD alone loses the history.",
                approaches=[_approach("CRUD"), _approach("CRUD with domain events")],
            ),
        )

        counts = {approach.name: approach.recommended_by for approach in result.approaches}
        assert counts == {"CRUD": 1, "CRUD with domain events": 1}

    def test_consensus_fires_when_every_member_names_the_longer_approach(self) -> None:
        result = self._result(
            _design_json(
                recommendation="Use CRUD with domain events.",
                approaches=[_approach("CRUD"), _approach("CRUD with domain events")],
            ),
            _design_json(
                recommendation="Go with CRUD with domain events.",
                approaches=[_approach("CRUD"), _approach("CRUD with domain events")],
            ),
        )

        assert result.consensus_recommendation == "CRUD with domain events"

    def test_a_member_recommending_an_approach_it_did_not_list_counts_for_nothing(self) -> None:
        """The longer name belongs to another member, so the short name is not what was chosen."""
        result = self._result(
            _design_json(
                recommendation="Use CRUD with domain events.",
                approaches=[_approach("CRUD")],
            ),
            _design_json(
                recommendation="Use CRUD with domain events.",
                approaches=[_approach("CRUD with domain events")],
            ),
        )

        counts = {approach.name: approach.recommended_by for approach in result.approaches}
        assert counts == {"CRUD": 0, "CRUD with domain events": 1}

    def test_no_consensus_when_a_member_recommends_an_approach_it_did_not_list(self) -> None:
        result = self._result(
            _design_json(
                recommendation="Use CRUD with domain events.",
                approaches=[_approach("CRUD")],
            ),
            _design_json(
                recommendation="Go with CRUD with domain events.",
                approaches=[_approach("CRUD with domain events")],
            ),
        )

        assert result.consensus_recommendation is None

    def test_a_symbol_name_in_the_recommendation_does_not_count_the_shorter_name(self) -> None:
        """Expanding C++ into words lets the word C match a recommendation that never named it."""
        result = self._result(
            _design_json(recommendation="Use C++ here.", approaches=[_approach("C")]),
        )

        assert result.approaches[0].recommended_by == 0

    def test_a_symbol_name_counts_only_the_approach_it_names(self) -> None:
        result = self._result(
            _design_json(
                recommendation="Use C++ here.",
                approaches=[_approach("C"), _approach("C++")],
            ),
        )

        counts = {approach.name: approach.recommended_by for approach in result.approaches}
        assert counts == {"C": 0, "C++": 1}


class TestDesignConsensusRecommendation:
    """The field carries one kind of value: the name of the approach every member chose.

    Returning a member's own prose instead makes one answer read as a council
    result, and gives a consumer two shapes it cannot tell apart.
    """

    def _result(self, *bodies: str):
        responses = [_success_response(f"member-{index}", "model", body) for index, body in enumerate(bodies)]
        return build_design_result(responses, prompt="Which data layer?")

    def test_identical_prose_naming_a_shared_approach_reports_the_approach_name(self) -> None:
        result = self._result(
            _design_json(recommendation="Use Repository.", approaches=[_approach("Repository")]),
            _design_json(recommendation="Use Repository.", approaches=[_approach("Repository")]),
        )

        assert result.consensus_recommendation == "Repository"

    def test_identical_prose_naming_no_approach_is_null(self) -> None:
        result = self._result(
            _design_json(recommendation="Do the simplest thing.", approaches=[_approach("Repository")]),
            _design_json(recommendation="Do the simplest thing.", approaches=[_approach("Repository")]),
        )

        assert result.consensus_recommendation is None

    def test_blank_recommendations_are_null(self) -> None:
        result = self._result(
            _design_json(recommendation="", approaches=[_approach("Repository")]),
            _design_json(recommendation="", approaches=[_approach("Repository")]),
        )

        assert result.consensus_recommendation is None

    def test_a_single_member_reports_the_approach_it_named(self) -> None:
        result = self._result(
            _design_json(recommendation="Use Repository, it fits.", approaches=[_approach("Repository")]),
        )

        assert result.consensus_recommendation == "Repository"


# ---------------------------------------------------------------------------
# Session synthesis.
# ---------------------------------------------------------------------------


class TestSessionSynthesis:
    """The closing assessment is written by a member, not assembled from the members."""

    async def _failing(self, material: str, answered: Sequence[ProviderConfig]) -> Synthesis:
        return Synthesis(
            author="openai", error=ProviderError(provider="openai", model="gpt-4o", error="Timeout.", raw_content=None)
        )

    def test_code_review_carries_what_the_member_wrote(self) -> None:
        responses = [_success_response("openai", "gpt-4o", _code_review_json(summary="Member prose."))]

        result = build_code_review_result(responses, files_reviewed=("app.py",))

        assert result.summary.synthesis == SYNTHESIS
        assert result.summary.synthesis_error is None

    def test_design_carries_what_the_member_wrote(self) -> None:
        responses = [_success_response("openai", "gpt-4o", _design_json(summary="Member prose."))]

        result = build_design_result(responses, prompt="Which data layer?")

        assert result.summary.synthesis == SYNTHESIS
        assert result.summary.synthesis_error is None

    def test_a_failed_call_leaves_the_assessment_empty_and_names_the_member(self) -> None:
        responses = [_success_response("openai", "gpt-4o", _code_review_json(summary="Member prose."))]

        result = build_code_review_result(responses, files_reviewed=("app.py",), synthesizer=self._failing)

        assert result.summary.synthesis == ""
        assert result.summary.synthesis_error == ProviderError(
            provider="openai", model="gpt-4o", error="Timeout.", raw_content=None
        )

    def test_a_failed_call_does_not_fall_back_to_the_member_summaries(self) -> None:
        responses = [
            _success_response("openai", "gpt-4o", _code_review_json(summary="First prose.")),
            _success_response("anthropic", "claude-3", _code_review_json(summary="Second prose.")),
        ]

        result = build_code_review_result(responses, files_reviewed=("app.py",), synthesizer=self._failing)

        assert "First prose." not in result.summary.synthesis
        assert "Second prose." not in result.summary.synthesis

    def test_the_material_withholds_member_names(self) -> None:
        seen: list[str] = []

        async def recording(material: str, answered: Sequence[ProviderConfig]) -> Synthesis:
            seen.append(material)
            return Synthesis(author="openai", text=SYNTHESIS)

        responses = [
            _success_response("openai", "gpt-4o", _code_review_json(issues=[_GOOD_ISSUE])),
            _success_response("anthropic", "claude-3", _code_review_json(issues=[_GOOD_ISSUE])),
        ]

        build_code_review_result(responses, files_reviewed=("app.py",), synthesizer=recording)

        assert "openai" not in seen[0]
        assert "anthropic" not in seen[0]

    def test_only_the_members_that_answered_are_offered_as_authors(self) -> None:
        seen: list[Sequence[ProviderConfig]] = []

        async def recording(material: str, answered: Sequence[ProviderConfig]) -> Synthesis:
            seen.append(answered)
            return Synthesis(author="openai", text=SYNTHESIS)

        responses = [
            _error_response("openai", "gpt-4o", "Timeout."),
            _success_response("anthropic", "claude-3", _code_review_json()),
        ]

        build_code_review_result(responses, files_reviewed=("app.py",), synthesizer=recording)

        assert [config.identity for config in seen[0]] == ["anthropic"]


class TestRunCouncilSynthesis:
    async def test_asks_the_first_member_that_answered(self) -> None:
        responses = [
            _error_response("openai", "gpt-4o", "Timeout."),
            _success_response("anthropic", "claude-3", _code_review_json()),
        ]
        send = AsyncMock(
            return_value=ProviderResponse(
                provider="anthropic", display_name="anthropic", model="claude-3", success=True, content="Agreed."
            )
        )

        with (
            patch("star_chamber.council.resolve_api_keys", return_value=_TWO_PROVIDER_CONFIG.providers),
            patch("star_chamber.council.fan_out", new_callable=AsyncMock, return_value=exchanges(responses)),
            patch("star_chamber.synthesis.send_to_provider", send),
        ):
            result = await run_council(
                files={"app.py": "x = 1"},
                config=_TWO_PROVIDER_CONFIG,
                mode="code-review",
            )

        assert send.await_count == 1
        assert send.await_args.args[0].provider == "anthropic"
        assert result.summary.synthesis == "Agreed."

    async def test_asks_nobody_when_the_council_declined_the_assessment(self) -> None:
        responses = [_success_response("openai", "gpt-4o", _code_review_json())]
        send = AsyncMock()
        declined = dataclasses.replace(_TWO_PROVIDER_CONFIG, synthesis=False)

        with (
            patch("star_chamber.council.resolve_api_keys", return_value=declined.providers),
            patch("star_chamber.council.fan_out", new_callable=AsyncMock, return_value=exchanges(responses)),
            patch("star_chamber.synthesis.send_to_provider", send),
        ):
            result = await run_council(
                files={"app.py": "x = 1"},
                config=declined,
                mode="code-review",
            )

        assert send.await_count == 0
        assert result.summary.synthesis == ""
        assert result.summary.synthesis_error is None
        assert result.summary.synthesized_by is None

    async def test_asks_nobody_when_no_member_answered(self) -> None:
        responses = [_error_response("openai", "gpt-4o", "Timeout.")]
        send = AsyncMock()

        with (
            patch("star_chamber.council.resolve_api_keys", return_value=_TWO_PROVIDER_CONFIG.providers),
            patch("star_chamber.council.fan_out", new_callable=AsyncMock, return_value=exchanges(responses)),
            patch("star_chamber.synthesis.send_to_provider", send),
        ):
            result = await run_council(
                files={"app.py": "x = 1"},
                config=_TWO_PROVIDER_CONFIG,
                mode="code-review",
            )

        assert send.await_count == 0
        assert result.summary.synthesis == ""
        assert result.summary.synthesis_error is None


class TestOneOddValueCostsAVote:
    """A value outside a closed set fails its member, which moves the denominator.

    The cost is not confined to that member. The roster that answered is what
    consensus is measured against, so dropping a member changes what the rest
    are reported to agree on.
    """

    _ISSUE = {
        "severity": "high",
        "location": "app.py:10",
        "category": "correctness",
        "description": "Off-by-one.",
        "suggestion": "Use <=.",
    }

    def _council(self, third: list[dict]) -> list[ProviderResponse]:
        """Two members that share one concern, and a third that raises its own."""
        return [
            _success_response("openai", "gpt-4o", _code_review_json(issues=[dict(self._ISSUE)])),
            _success_response("anthropic", "claude-3", _code_review_json(issues=[dict(self._ISSUE)])),
            _success_response("gemini", "g-2", _code_review_json(issues=third)),
        ]

    def test_failing_a_member_promotes_a_concern_it_did_not_share(self) -> None:
        theirs = dict(self._ISSUE) | {"location": "app.py:90"}
        invented = theirs | {"category": "defensive-design"}

        answered = build_code_review_result(self._council([theirs]), files_reviewed=("app.py",), threshold=2)
        failed = build_code_review_result(self._council([invented]), files_reviewed=("app.py",), threshold=2)

        assert answered.providers_responded == ("openai", "anthropic", "gemini")
        assert [issue.location for issue in answered.majority_issues] == ["app.py:10"]
        assert answered.consensus_issues == ()

        assert failed.providers_responded == ("openai", "anthropic")
        assert [issue.location for issue in failed.consensus_issues] == ["app.py:10"]
        assert failed.majority_issues == ()

    def test_the_failed_member_s_reply_is_still_readable(self) -> None:
        invented = dict(self._ISSUE) | {"category": "defensive-design"}
        responses = [_success_response("gemini", "g-2", _code_review_json(issues=[invented]))]

        result = build_code_review_result(responses, files_reviewed=("app.py",))

        assert "defensive-design" in result.failed_providers[0].error
        assert "defensive-design" in (result.failed_providers[0].raw_content or "")
