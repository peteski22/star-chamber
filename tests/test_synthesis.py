"""Tests for the closing assessment of a council session."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from star_chamber.synthesis import (
    render_code_review_material,
    render_design_material,
    select_synthesizer,
    synthesize,
)
from star_chamber.transport import ProviderResponse
from star_chamber.types import (
    Approach,
    ClassificationResult,
    Issue,
    MajorityIssue,
    ProviderConfig,
    ProviderDesignAdvice,
    ProviderReview,
)


def _config(provider: str, display_name: str | None = None) -> ProviderConfig:
    return ProviderConfig(provider=provider, model="m", display_name=display_name)


def _review(provider: str, summary: str = "s", rating: str = "good") -> ProviderReview:
    return ProviderReview(
        provider=provider,
        model="m",
        quality_rating=rating,
        issues=(),
        praise=(),
        summary=summary,
        raw_content="{}",
    )


def _advice(provider: str, summary: str = "s", recommendation: str = "r") -> ProviderDesignAdvice:
    return ProviderDesignAdvice(
        provider=provider,
        model="m",
        recommendation=recommendation,
        approaches=(),
        summary=summary,
        raw_content="{}",
    )


def _issue(description: str) -> Issue:
    return Issue(severity="high", location="a.py:1", category="correctness", description=description, suggestion="fix")


class TestSelectSynthesizer:
    def test_picks_the_first_member_that_answered(self):
        configs = (_config("openai"), _config("anthropic"), _config("gemini"))

        chosen = select_synthesizer(configs, ["anthropic", "gemini"])

        assert chosen is not None
        assert chosen.provider == "anthropic"

    def test_follows_roster_order_not_answer_order(self):
        configs = (_config("openai"), _config("anthropic"))

        chosen = select_synthesizer(configs, ["anthropic", "openai"])

        assert chosen is not None
        assert chosen.provider == "openai"

    def test_matches_on_display_name(self):
        configs = (_config("openai", display_name="scribe"),)

        chosen = select_synthesizer(configs, ["scribe"])

        assert chosen is not None
        assert chosen.display_name == "scribe"

    def test_returns_none_when_nobody_answered(self):
        configs = (_config("openai"), _config("anthropic"))

        assert select_synthesizer(configs, []) is None


class TestSynthesize:
    async def test_returns_the_member_reply_stripped(self):
        reply = ProviderResponse(
            provider="openai",
            display_name="openai",
            model="m",
            success=True,
            content="  The council agreed on the hashing flaw.\n",
        )

        with patch("star_chamber.synthesis.send_to_provider", new_callable=AsyncMock, return_value=reply):
            result = await synthesize("material", _config("openai"))

        assert result.text == "The council agreed on the hashing flaw."
        assert result.error is None

    async def test_sends_the_material_to_the_member(self):
        reply = ProviderResponse(provider="openai", display_name="openai", model="m", success=True, content="Done.")
        send = AsyncMock(return_value=reply)

        with patch("star_chamber.synthesis.send_to_provider", send):
            await synthesize("## Each member's own assessment\n- solid", _config("openai"))

        prompt = send.call_args.args[1]
        assert "## Each member's own assessment" in prompt
        assert "- solid" in prompt

    async def test_reports_a_failed_call_and_leaves_the_text_empty(self):
        reply = ProviderResponse(
            provider="openai",
            display_name="scribe",
            model="m",
            success=False,
            error="Timeout: provider openai did not respond in time.",
        )

        with patch("star_chamber.synthesis.send_to_provider", new_callable=AsyncMock, return_value=reply):
            result = await synthesize("material", _config("openai", display_name="scribe"))

        assert result.text == ""
        assert result.error is not None
        assert result.error.provider == "scribe"
        assert "did not respond" in result.error.error

    async def test_reports_an_empty_reply_rather_than_passing_it_off_as_a_synthesis(self):
        reply = ProviderResponse(provider="openai", display_name="openai", model="m", success=True, content="   ")

        with patch("star_chamber.synthesis.send_to_provider", new_callable=AsyncMock, return_value=reply):
            result = await synthesize("material", _config("openai"))

        assert result.text == ""
        assert result.error is not None
        assert result.error.provider == "openai"


class TestCodeReviewMaterial:
    def test_withholds_member_names(self):
        reviews = (_review("openai", summary="Hashing is weak."), _review("anthropic", summary="Auth is broken."))
        classification = ClassificationResult(
            consensus_issues=(_issue("SHA-256 for passwords"),),
            majority_issues=(),
            individual_issues={"openai": (_issue("Unused import"),)},
        )

        material = render_code_review_material(reviews, classification)

        assert "openai" not in material
        assert "anthropic" not in material

    def test_carries_each_assessment_and_the_grouped_concerns(self):
        reviews = (_review("openai", summary="Hashing is weak.", rating="fair"),)
        classification = ClassificationResult(
            consensus_issues=(_issue("SHA-256 for passwords"),),
            majority_issues=(
                MajorityIssue(
                    severity="medium",
                    location="a.py:2",
                    category="maintainability",
                    description="No tests for the token path",
                    suggestion="add tests",
                    provider_count=2,
                    flagged_by=("openai", "anthropic"),
                ),
            ),
            individual_issues={"openai": (_issue("Unused import"),)},
        )

        material = render_code_review_material(reviews, classification)

        assert "Hashing is weak." in material
        assert "fair" in material
        assert "SHA-256 for passwords" in material
        assert "No tests for the token path" in material
        assert "Unused import" in material

    def test_omits_a_section_with_nothing_in_it(self):
        reviews = (_review("openai", summary="All good."),)
        classification = ClassificationResult(consensus_issues=(), majority_issues=(), individual_issues={})

        material = render_code_review_material(reviews, classification)

        assert "every member raised" not in material


class TestDesignMaterial:
    def test_withholds_member_names(self):
        advices = (
            _advice("openai", summary="Repository fits.", recommendation="Use Approach A."),
            _advice("anthropic", summary="Service fits.", recommendation="Use Approach B."),
        )
        approaches = (Approach(name="Approach A", recommended_by=1, pros=(), cons=(), risk_level="low"),)

        material = render_design_material(advices, approaches)

        assert "openai" not in material
        assert "anthropic" not in material

    def test_carries_the_recommendations_and_the_counts(self):
        advices = (_advice("openai", summary="Repository fits.", recommendation="Use Approach A."),)
        approaches = (
            Approach(name="Approach A", recommended_by=2, pros=(), cons=(), risk_level="low"),
            Approach(name="Approach B", recommended_by=0, pros=(), cons=(), risk_level="high"),
        )

        material = render_design_material(advices, approaches)

        assert "Repository fits." in material
        assert "Use Approach A." in material
        assert "Approach A" in material
        assert "Approach B" in material
        assert "2" in material
