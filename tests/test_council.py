"""Tests for the council orchestrator module."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from star_chamber.council import _build_code_review_result, _build_design_result, run_council
from star_chamber.transport import ProviderResponse
from star_chamber.types import (
    CodeReviewResult,
    CouncilConfig,
    DesignQuestionResult,
    ProviderConfig,
)

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
    def test_successful_two_providers(self):
        issue_data = {
            "severity": "warning",
            "location": "main.py:10",
            "category": "style",
            "description": "Missing docstring.",
            "suggestion": "Add a docstring.",
        }
        responses = [
            _success_response("openai", "gpt-4o", _code_review_json(issues=[issue_data])),
            _success_response("anthropic", "claude-3", _code_review_json(issues=[issue_data])),
        ]

        result = _build_code_review_result(responses, files_reviewed=("app.py",), threshold=2)

        assert isinstance(result.reviews, tuple)
        assert len(result.reviews) == 2
        assert len(result.failed_providers) == 0
        # Both providers flagged the same issue, so it should be consensus.
        assert len(result.consensus_issues) == 1
        assert result.consensus_issues[0].location == "main.py:10"

    def test_parse_error_recorded_as_failed_provider(self):
        responses = [
            _success_response("openai", "gpt-4o", _code_review_json()),
            _success_response("anthropic", "claude-3", "not valid json at all"),
        ]

        result = _build_code_review_result(responses, files_reviewed=("app.py",), threshold=2)

        assert len(result.reviews) == 1
        assert result.reviews[0].provider == "openai"
        assert len(result.failed_providers) == 1
        assert result.failed_providers[0].provider == "anthropic"
        assert "parse" in result.failed_providers[0].error.lower() or "json" in result.failed_providers[0].error.lower()

    def test_transport_error_recorded_as_failed_provider(self):
        responses = [
            _success_response("openai", "gpt-4o", _code_review_json()),
            _error_response("anthropic", "claude-3", "Timeout: provider anthropic did not respond in time."),
        ]

        result = _build_code_review_result(responses, files_reviewed=("app.py",), threshold=2)

        assert len(result.reviews) == 1
        assert len(result.failed_providers) == 1
        assert result.failed_providers[0].provider == "anthropic"
        assert "timeout" in result.failed_providers[0].error.lower()

    def test_quality_ratings_mapped_by_provider(self):
        responses = [
            _success_response("openai", "gpt-4o", _code_review_json(quality="excellent")),
            _success_response("anthropic", "claude-3", _code_review_json(quality="good")),
        ]

        result = _build_code_review_result(responses, files_reviewed=("app.py",), threshold=2)

        assert result.quality_ratings["openai"] == "excellent"
        assert result.quality_ratings["anthropic"] == "good"


# ---------------------------------------------------------------------------
# _build_design_result
# ---------------------------------------------------------------------------


class TestBuildDesignQuestionResult:
    def test_successful_two_providers(self):
        responses = [
            _success_response("openai", "gpt-4o", _design_json()),
            _success_response("anthropic", "claude-3", _design_json(recommendation="Use approach B.")),
        ]

        result = _build_design_result(responses, prompt="How should we structure the API?")

        assert isinstance(result.approaches, tuple)
        assert len(result.approaches) == 1
        assert result.approaches[0].recommended_by == 1
        assert len(result.failed_providers) == 0
        assert result.prompt == "How should we structure the API?"

    def test_parse_error_recorded_as_failed_provider(self):
        responses = [
            _success_response("openai", "gpt-4o", _design_json()),
            _success_response("anthropic", "claude-3", "garbage response"),
        ]

        result = _build_design_result(responses, prompt="Design question?")

        assert len(result.approaches) == 1
        assert len(result.failed_providers) == 1
        assert result.failed_providers[0].provider == "anthropic"

    def test_consensus_recommendation_when_all_agree(self):
        responses = [
            _success_response("openai", "gpt-4o", _design_json(recommendation="Use caching.")),
            _success_response("anthropic", "claude-3", _design_json(recommendation="Use caching.")),
        ]

        result = _build_design_result(responses, prompt="How to improve perf?")

        assert result.consensus_recommendation == "Use caching."

    def test_no_consensus_when_providers_disagree(self):
        responses = [
            _success_response("openai", "gpt-4o", _design_json(recommendation="Use caching.")),
            _success_response("anthropic", "claude-3", _design_json(recommendation="Optimize queries.")),
        ]

        result = _build_design_result(responses, prompt="How to improve perf?")

        assert result.consensus_recommendation is None


# ---------------------------------------------------------------------------
# run_council — code-review mode.
# ---------------------------------------------------------------------------


class TestRunCouncilCodeReview:
    async def test_successful_code_review(self):
        issue_data = {
            "severity": "warning",
            "location": "app.py:5",
            "category": "style",
            "description": "Unused import.",
            "suggestion": "Remove the import.",
        }
        mock_responses = [
            _success_response("openai", "gpt-4o", _code_review_json(issues=[issue_data])),
            _success_response("anthropic", "claude-3", _code_review_json(issues=[issue_data])),
        ]

        with (
            patch("star_chamber.council.resolve_api_keys", return_value=_TWO_PROVIDER_CONFIG.providers),
            patch("star_chamber.council.fan_out", new_callable=AsyncMock, return_value=mock_responses),
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

    async def test_partial_failure(self):
        mock_responses = [
            _success_response("openai", "gpt-4o", _code_review_json()),
            _error_response("anthropic", "claude-3", "Rate limit exceeded."),
        ]

        with (
            patch("star_chamber.council.resolve_api_keys", return_value=_TWO_PROVIDER_CONFIG.providers),
            patch("star_chamber.council.fan_out", new_callable=AsyncMock, return_value=mock_responses),
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

    async def test_all_providers_fail(self):
        mock_responses = [
            _error_response("openai", "gpt-4o", "Service unavailable."),
            _error_response("anthropic", "claude-3", "Rate limit exceeded."),
        ]

        with (
            patch("star_chamber.council.resolve_api_keys", return_value=_TWO_PROVIDER_CONFIG.providers),
            patch("star_chamber.council.fan_out", new_callable=AsyncMock, return_value=mock_responses),
        ):
            result = await run_council(
                files={"app.py": "print('hello')"},
                config=_TWO_PROVIDER_CONFIG,
                mode="code-review",
            )

        assert isinstance(result, CodeReviewResult)
        assert len(result.reviews) == 0
        assert len(result.failed_providers) == 2

    async def test_consensus_classification_flows_through(self):
        issue_data = {
            "severity": "high",
            "location": "auth.py:15",
            "category": "security",
            "description": "SQL injection risk.",
            "suggestion": "Use parameterized queries.",
        }
        mock_responses = [
            _success_response("openai", "gpt-4o", _code_review_json(issues=[issue_data])),
            _success_response("anthropic", "claude-3", _code_review_json(issues=[issue_data])),
        ]

        with (
            patch("star_chamber.council.resolve_api_keys", return_value=_TWO_PROVIDER_CONFIG.providers),
            patch("star_chamber.council.fan_out", new_callable=AsyncMock, return_value=mock_responses),
        ):
            result = await run_council(
                files={"auth.py": "query = f'SELECT * FROM users WHERE id={uid}'"},
                config=_TWO_PROVIDER_CONFIG,
                mode="code-review",
            )

        assert len(result.consensus_issues) == 1
        assert result.consensus_issues[0].category == "security"

    async def test_parse_error_becomes_failed_provider(self):
        mock_responses = [
            _success_response("openai", "gpt-4o", _code_review_json()),
            _success_response("anthropic", "claude-3", "this is not json"),
        ]

        with (
            patch("star_chamber.council.resolve_api_keys", return_value=_TWO_PROVIDER_CONFIG.providers),
            patch("star_chamber.council.fan_out", new_callable=AsyncMock, return_value=mock_responses),
        ):
            result = await run_council(
                files={"app.py": "print('hello')"},
                config=_TWO_PROVIDER_CONFIG,
                mode="code-review",
            )

        assert len(result.reviews) == 1
        assert len(result.failed_providers) == 1
        assert result.failed_providers[0].provider == "anthropic"

    async def test_auto_loads_config_when_none(self):
        mock_responses = [
            _success_response("openai", "gpt-4o", _code_review_json()),
        ]
        auto_config = CouncilConfig(providers=(_OPENAI_CONFIG,))

        with (
            patch("star_chamber.council.load_config", return_value=auto_config),
            patch("star_chamber.council.resolve_api_keys", return_value=auto_config.providers),
            patch("star_chamber.council.fan_out", new_callable=AsyncMock, return_value=mock_responses),
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
    async def test_design_question_returns_design_result(self):
        mock_responses = [
            _success_response("openai", "gpt-4o", _design_json()),
            _success_response("anthropic", "claude-3", _design_json(recommendation="Use approach B.")),
        ]

        with (
            patch("star_chamber.council.resolve_api_keys", return_value=_TWO_PROVIDER_CONFIG.providers),
            patch("star_chamber.council.fan_out", new_callable=AsyncMock, return_value=mock_responses),
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

    async def test_design_question_partial_failure(self):
        mock_responses = [
            _success_response("openai", "gpt-4o", _design_json()),
            _error_response("anthropic", "claude-3", "Connection refused."),
        ]

        with (
            patch("star_chamber.council.resolve_api_keys", return_value=_TWO_PROVIDER_CONFIG.providers),
            patch("star_chamber.council.fan_out", new_callable=AsyncMock, return_value=mock_responses),
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
    def test_sync_wrapper_works(self):
        from star_chamber.council import run_council_sync

        mock_responses = [
            _success_response("openai", "gpt-4o", _code_review_json()),
        ]
        single_config = CouncilConfig(providers=(_OPENAI_CONFIG,))

        with (
            patch("star_chamber.council.resolve_api_keys", return_value=single_config.providers),
            patch("star_chamber.council.fan_out", new_callable=AsyncMock, return_value=mock_responses),
        ):
            result = run_council_sync(
                files={"app.py": "print('hello')"},
                config=single_config,
                mode="code-review",
            )

        assert isinstance(result, CodeReviewResult)
        assert result.mode == "code-review"

    def test_sync_wrapper_design_mode(self):
        from star_chamber.council import run_council_sync

        mock_responses = [
            _success_response("openai", "gpt-4o", _design_json()),
        ]
        single_config = CouncilConfig(providers=(_OPENAI_CONFIG,))

        with (
            patch("star_chamber.council.resolve_api_keys", return_value=single_config.providers),
            patch("star_chamber.council.fan_out", new_callable=AsyncMock, return_value=mock_responses),
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
    async def test_council_context_appears_in_code_review_prompt(self):
        mock_responses = [
            _success_response("openai", "gpt-4o", _code_review_json()),
        ]
        single_config = CouncilConfig(providers=(_OPENAI_CONFIG,))
        mock_fan_out = AsyncMock(return_value=mock_responses)

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

    async def test_council_context_absent_when_not_provided(self):
        mock_responses = [
            _success_response("openai", "gpt-4o", _code_review_json()),
        ]
        single_config = CouncilConfig(providers=(_OPENAI_CONFIG,))
        mock_fan_out = AsyncMock(return_value=mock_responses)

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

    async def test_council_context_appears_in_design_prompt(self):
        mock_responses = [
            _success_response("openai", "gpt-4o", _design_json()),
        ]
        single_config = CouncilConfig(providers=(_OPENAI_CONFIG,))
        mock_fan_out = AsyncMock(return_value=mock_responses)

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

    async def test_single_fan_out_call(self):
        """run_council() always calls fan_out exactly once."""
        mock_responses = [
            _success_response("openai", "gpt-4o", _code_review_json()),
            _success_response("anthropic", "claude-3", _code_review_json()),
        ]
        mock_fan_out = AsyncMock(return_value=mock_responses)

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

    async def test_no_debate_parameter(self):
        """run_council() does not accept debate or rounds parameters."""
        with pytest.raises(TypeError):
            await run_council(
                files={"app.py": "x = 1"},
                config=_TWO_PROVIDER_CONFIG,
                mode="code-review",
                debate=True,
            )

    async def test_no_rounds_parameter(self):
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
    async def test_code_review_requires_files(self):
        with pytest.raises(ValueError, match="files"):
            await run_council(
                config=_TWO_PROVIDER_CONFIG,
                mode="code-review",
            )

    async def test_design_question_requires_prompt(self):
        with pytest.raises(ValueError, match="prompt"):
            await run_council(
                config=_TWO_PROVIDER_CONFIG,
                mode="design-question",
            )

    async def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode"):
            await run_council(
                config=_TWO_PROVIDER_CONFIG,
                mode="invalid-mode",
            )


class TestAnsweringRoster:
    """providers_responded must carry only the members that returned a usable answer."""

    async def test_transport_failure_is_absent_from_the_answering_roster(self):
        mock_responses = [
            _success_response("openai", "gpt-4o", _code_review_json()),
            _error_response("anthropic", "claude-3", "Rate limit exceeded."),
        ]

        with (
            patch("star_chamber.council.resolve_api_keys", return_value=_TWO_PROVIDER_CONFIG.providers),
            patch("star_chamber.council.fan_out", new_callable=AsyncMock, return_value=mock_responses),
        ):
            result = await run_council(files={"app.py": "x = 1"}, config=_TWO_PROVIDER_CONFIG, mode="code-review")

        assert set(result.providers_used) == {"openai", "anthropic"}
        assert result.providers_responded == ("openai",)

    async def test_unparsable_reply_is_absent_from_the_answering_roster(self):
        mock_responses = [
            _success_response("openai", "gpt-4o", _code_review_json()),
            _success_response("anthropic", "claude-3", "I cannot answer that."),
        ]

        with (
            patch("star_chamber.council.resolve_api_keys", return_value=_TWO_PROVIDER_CONFIG.providers),
            patch("star_chamber.council.fan_out", new_callable=AsyncMock, return_value=mock_responses),
        ):
            result = await run_council(files={"app.py": "x = 1"}, config=_TWO_PROVIDER_CONFIG, mode="code-review")

        assert set(result.providers_used) == {"openai", "anthropic"}
        assert result.providers_responded == ("openai",)

    async def test_answering_roster_is_the_denominator_the_issues_were_classified_over(self):
        issue_data = {
            "severity": "high",
            "location": "app.py:5",
            "category": "style",
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
            patch("star_chamber.council.fan_out", new_callable=AsyncMock, return_value=mock_responses),
        ):
            result = await run_council(files={"app.py": "x = 1"}, config=config, mode="code-review")

        # Three members were asked, two answered, and both flagged the issue, so it is
        # consensus across the two that answered rather than across the three asked.
        assert len(result.providers_used) == 3
        assert set(result.providers_responded) == {"openai", "anthropic"}
        assert len(result.consensus_issues) == 1
        assert len(result.providers_responded) == len(result.reviews)

    async def test_answering_roster_uses_the_display_name(self):
        mock_responses = [
            _success_response("openai", "gpt-4o", _code_review_json(), display_name="scribe"),
        ]
        config = CouncilConfig(providers=(ProviderConfig(provider="openai", model="gpt-4o", display_name="scribe"),))

        with (
            patch("star_chamber.council.resolve_api_keys", return_value=config.providers),
            patch("star_chamber.council.fan_out", new_callable=AsyncMock, return_value=mock_responses),
        ):
            result = await run_council(files={"app.py": "x = 1"}, config=config, mode="code-review")

        assert result.providers_responded == ("scribe",)

    async def test_design_question_reports_its_answering_roster(self):
        mock_responses = [
            _success_response("openai", "gpt-4o", _design_json()),
            _error_response("anthropic", "claude-3", "Connection refused."),
        ]

        with (
            patch("star_chamber.council.resolve_api_keys", return_value=_TWO_PROVIDER_CONFIG.providers),
            patch("star_chamber.council.fan_out", new_callable=AsyncMock, return_value=mock_responses),
        ):
            result = await run_council(prompt="Monorepo?", config=_TWO_PROVIDER_CONFIG, mode="design-question")

        assert set(result.providers_used) == {"openai", "anthropic"}
        assert result.providers_responded == ("openai",)

    async def test_no_answers_leaves_the_answering_roster_empty(self):
        mock_responses = [
            _error_response("openai", "gpt-4o", "Timeout."),
            _error_response("anthropic", "claude-3", "Timeout."),
        ]

        with (
            patch("star_chamber.council.resolve_api_keys", return_value=_TWO_PROVIDER_CONFIG.providers),
            patch("star_chamber.council.fan_out", new_callable=AsyncMock, return_value=mock_responses),
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
    def test_code_review_rosters_partition_the_members_asked(self, responses):
        result = _build_code_review_result(responses, files_reviewed=("app.py",))

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
    def test_design_rosters_partition_the_members_asked(self, responses):
        result = _build_design_result(responses, prompt="Monorepo?")

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
    def test_one_bad_issue_does_not_end_a_code_review(self, label, issue):
        responses = [
            _success_response("openai", "gpt-4o", _code_review_json(issues=[dict(_GOOD_ISSUE)])),
            _success_response("anthropic", "claude-opus-5", _code_review_json(issues=[issue])),
            _success_response("gemini", "gemini-2.0", _code_review_json(issues=[dict(_GOOD_ISSUE)])),
        ]

        result = _build_code_review_result(responses, files_reviewed=("app.py",))

        assert result.providers_responded == ("openai", "gemini")
        assert [failure.provider for failure in result.failed_providers] == ["anthropic"]
        assert result.providers_used == ("openai", "anthropic", "gemini")

    def test_the_surviving_members_still_reach_consensus(self):
        bad = {**_GOOD_ISSUE, "location": None}
        responses = [
            _success_response("openai", "gpt-4o", _code_review_json(issues=[dict(_GOOD_ISSUE)])),
            _success_response("anthropic", "claude-opus-5", _code_review_json(issues=[bad])),
            _success_response("gemini", "gemini-2.0", _code_review_json(issues=[dict(_GOOD_ISSUE)])),
        ]

        result = _build_code_review_result(responses, files_reviewed=("app.py",))

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
    def test_one_bad_approach_does_not_end_a_design_question(self, label, approaches):
        responses = [
            _success_response("openai", "gpt-4o", _design_json()),
            _success_response("anthropic", "claude-opus-5", _design_json(approaches=approaches)),
        ]

        result = _build_design_result(responses, prompt="Which data layer?")

        assert result.providers_responded == ("openai",)
        assert [failure.provider for failure in result.failed_providers] == ["anthropic"]

    def test_the_failure_names_the_field_and_the_shape(self):
        bad = {**_GOOD_ISSUE, "location": None}
        responses = [_success_response("anthropic", "claude-opus-5", _code_review_json(issues=[bad]))]

        result = _build_code_review_result(responses, files_reviewed=("app.py",))

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
        return _build_design_result(responses, prompt="Which data layer?")

    def test_same_approach_from_two_members_merges(self):
        result = self._result(
            _design_json(approaches=[_approach("Repository")]),
            _design_json(approaches=[_approach("Repository")]),
        )

        assert len(result.approaches) == 1
        assert result.approaches[0].name == "Repository"

    def test_merge_unions_pros_and_cons_without_duplicates(self):
        result = self._result(
            _design_json(approaches=[_approach("Repository", pros=["Testable"], cons=["Indirection"])]),
            _design_json(approaches=[_approach("Repository", pros=["Testable", "Swappable"], cons=["Boilerplate"])]),
        )

        assert result.approaches[0].pros == ("Testable", "Swappable")
        assert result.approaches[0].cons == ("Indirection", "Boilerplate")

    def test_recommended_by_counts_members_that_recommended_it(self):
        result = self._result(
            _design_json(recommendation="Use the Repository pattern.", approaches=[_approach("Repository")]),
            _design_json(recommendation="Go with Repository here.", approaches=[_approach("Repository")]),
        )

        assert result.approaches[0].recommended_by == 2

    def test_discussed_but_unrecommended_approach_counts_zero(self):
        result = self._result(
            _design_json(
                recommendation="Use the Repository pattern.",
                approaches=[_approach("Repository"), _approach("Active Record")],
            )
        )

        counts = {approach.name: approach.recommended_by for approach in result.approaches}
        assert counts == {"Repository": 1, "Active Record": 0}

    def test_name_match_ignores_case_and_spacing(self):
        result = self._result(
            _design_json(recommendation="use   a REPOSITORY layer", approaches=[_approach("Repository")])
        )

        assert result.approaches[0].recommended_by == 1

    def test_name_matches_on_word_boundaries_only(self):
        """An approach named ORM must not match the word normalize."""
        result = self._result(_design_json(recommendation="Normalize the schema first.", approaches=[_approach("ORM")]))

        assert result.approaches[0].recommended_by == 0

    def test_member_listing_an_approach_twice_counts_once(self):
        result = self._result(
            _design_json(
                recommendation="Use the Repository pattern.",
                approaches=[_approach("Repository"), _approach("Repository")],
            )
        )

        assert len(result.approaches) == 1
        assert result.approaches[0].recommended_by == 1

    def test_risk_level_takes_the_most_common_value(self):
        result = self._result(
            _design_json(approaches=[_approach("Repository", risk_level="low")]),
            _design_json(approaches=[_approach("Repository", risk_level="high")]),
            _design_json(approaches=[_approach("Repository", risk_level="high")]),
        )

        assert result.approaches[0].risk_level == "high"

    def test_tied_risk_level_takes_the_more_cautious_value(self):
        result = self._result(
            _design_json(approaches=[_approach("Repository", risk_level="low")]),
            _design_json(approaches=[_approach("Repository", risk_level="high")]),
        )

        assert result.approaches[0].risk_level == "high"

    def test_risk_level_is_unknown_when_no_member_stated_one(self):
        body = json.dumps(
            {
                "recommendation": "Use the Repository pattern.",
                "approaches": [{"name": "Repository", "pros": ["Testable"], "cons": ["Indirection"]}],
                "summary": "Repository fits.",
            }
        )

        result = self._result(body)

        assert result.approaches[0].risk_level == "unknown"

    def test_fit_rating_takes_the_most_common_value(self):
        result = self._result(
            _design_json(approaches=[_approach("Repository", fit_rating="excellent")]),
            _design_json(approaches=[_approach("Repository", fit_rating="good")]),
            _design_json(approaches=[_approach("Repository", fit_rating="good")]),
        )

        assert result.approaches[0].fit_rating == "good"

    def test_tied_fit_rating_takes_the_more_cautious_value(self):
        result = self._result(
            _design_json(approaches=[_approach("Repository", fit_rating="excellent")]),
            _design_json(approaches=[_approach("Repository", fit_rating="fair")]),
        )

        assert result.approaches[0].fit_rating == "fair"

    def test_fit_rating_is_null_when_no_member_assessed_fit(self):
        result = self._result(
            _design_json(approaches=[_approach("Repository", fit_rating=None)]),
            _design_json(approaches=[_approach("Repository", fit_rating=None)]),
        )

        assert result.approaches[0].fit_rating is None

    def test_consensus_fires_when_members_pick_the_same_approach_in_different_words(self):
        result = self._result(
            _design_json(recommendation="Use the Repository pattern.", approaches=[_approach("Repository")]),
            _design_json(recommendation="I would reach for Repository.", approaches=[_approach("Repository")]),
        )

        assert result.consensus_recommendation == "Repository"

    def test_no_consensus_when_members_pick_different_approaches(self):
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
        return _build_design_result(responses, prompt="Which data layer?")

    def test_provider_naming_an_approach_twice_casts_one_risk_vote(self):
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

    def test_provider_naming_an_approach_twice_casts_one_fit_vote(self):
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

    def test_provider_is_not_counted_for_an_approach_it_did_not_name(self):
        """A provider recommends from the approaches it offered, not from another provider's."""
        result = self._result(
            _design_json(recommendation="Use Active Record.", approaches=[_approach("Repository")]),
            _design_json(recommendation="Use Active Record.", approaches=[_approach("Active Record")]),
        )

        counts = {approach.name: approach.recommended_by for approach in result.approaches}
        assert counts == {"Repository": 0, "Active Record": 1}

    def test_names_that_differ_only_by_symbols_stay_distinct(self):
        """Stripping every symbol collapses C++, C# and C into one approach."""
        result = self._result(
            _design_json(
                approaches=[_approach("C++"), _approach("C#"), _approach("C")],
            )
        )

        assert [approach.name for approach in result.approaches] == ["C++", "C#", "C"]
