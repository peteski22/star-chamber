from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from star_chamber.types import (
    Approach,
    ClassificationResult,
    CodeReviewResult,
    CouncilConfig,
    DebateMetadata,
    DesignQuestionResult,
    Issue,
    MajorityIssue,
    ProviderConfig,
    ProviderDesignAdvice,
    ProviderError,
    ProviderReview,
)

# -- helpers ------------------------------------------------------------------


def _round_trip(obj):
    """Serialize via asdict -> json.dumps -> json.loads and return the dict."""
    d = asdict(obj)
    return json.loads(json.dumps(d))


def _reconstruct(cls, data):
    """Reconstruct a flat frozen dataclass from a dict, converting lists to tuples for tuple-typed fields."""
    return cls(**data)


# -- ProviderConfig -----------------------------------------------------------


class TestProviderConfig:
    def test_all_fields_set(self):
        pc = ProviderConfig(
            provider="openai",
            model="gpt-4",
            api_key="test-key-not-real",  # pragma: allowlist secret
            api_base="https://api.openai.com",
            max_tokens=4096,
            local=True,
        )
        assert pc.provider == "openai"
        assert pc.model == "gpt-4"
        assert pc.api_key == "test-key-not-real"  # pragma: allowlist secret
        assert pc.api_base == "https://api.openai.com"
        assert pc.max_tokens == 4096
        assert pc.local is True

    def test_defaults(self):
        pc = ProviderConfig(provider="anthropic", model="claude-3")
        assert pc.api_key is None
        assert pc.api_base is None
        assert pc.max_tokens is None
        assert pc.local is False

    def test_frozen(self):
        pc = ProviderConfig(provider="openai", model="gpt-4")
        with pytest.raises(AttributeError):
            pc.provider = "anthropic"  # type: ignore[misc]

    def test_json_round_trip(self):
        pc = ProviderConfig(
            provider="openai",
            model="gpt-4",
            api_key="test-key-not-real",  # pragma: allowlist secret
            api_base="https://api.openai.com",
            max_tokens=4096,
            local=True,
        )
        data = _round_trip(pc)
        rebuilt = _reconstruct(ProviderConfig, data)
        assert rebuilt == pc


# -- CouncilConfig ------------------------------------------------------------


class TestCouncilConfig:
    def test_all_fields_set(self):
        p1 = ProviderConfig(provider="openai", model="gpt-4")
        p2 = ProviderConfig(provider="anthropic", model="claude-3")
        cc = CouncilConfig(
            providers=(p1, p2),
            timeout_seconds=120,
            consensus_threshold=3,
            platform="my-platform",
        )
        assert cc.providers == (p1, p2)
        assert cc.timeout_seconds == 120
        assert cc.consensus_threshold == 3
        assert cc.platform == "my-platform"

    def test_defaults(self):
        p1 = ProviderConfig(provider="openai", model="gpt-4")
        cc = CouncilConfig(providers=(p1,))
        assert cc.timeout_seconds == 60
        assert cc.consensus_threshold == 2
        assert cc.platform is None

    def test_frozen(self):
        p1 = ProviderConfig(provider="openai", model="gpt-4")
        cc = CouncilConfig(providers=(p1,))
        with pytest.raises(AttributeError):
            cc.timeout_seconds = 999  # type: ignore[misc]

    def test_json_round_trip(self):
        p1 = ProviderConfig(provider="openai", model="gpt-4", api_key="test-key-not-real")  # pragma: allowlist secret
        cc = CouncilConfig(providers=(p1,), timeout_seconds=90)
        data = _round_trip(cc)
        # Providers come back as list-of-dicts; rebuild manually.
        rebuilt = CouncilConfig(
            providers=tuple(ProviderConfig(**p) for p in data["providers"]),
            timeout_seconds=data["timeout_seconds"],
            consensus_threshold=data["consensus_threshold"],
            platform=data["platform"],
        )
        assert rebuilt == cc


# -- Issue ---------------------------------------------------------------------


class TestIssue:
    def test_all_fields_set(self):
        issue = Issue(
            severity="error",
            location="main.py:10",
            category="security",
            description="SQL injection",
            suggestion="Use parameterised queries",
        )
        assert issue.severity == "error"
        assert issue.location == "main.py:10"
        assert issue.category == "security"
        assert issue.description == "SQL injection"
        assert issue.suggestion == "Use parameterised queries"

    def test_frozen(self):
        issue = Issue(
            severity="error",
            location="main.py:10",
            category="security",
            description="SQL injection",
            suggestion="Use parameterised queries",
        )
        with pytest.raises(AttributeError):
            issue.severity = "warning"  # type: ignore[misc]

    def test_json_round_trip(self):
        issue = Issue(
            severity="warning",
            location="utils.py:42",
            category="style",
            description="Long line",
            suggestion="Break the line",
        )
        data = _round_trip(issue)
        rebuilt = _reconstruct(Issue, data)
        assert rebuilt == issue


# -- MajorityIssue ------------------------------------------------------------


class TestMajorityIssue:
    def test_all_fields_set(self):
        mi = MajorityIssue(
            severity="error",
            location="app.py:5",
            category="bug",
            description="Null deref",
            suggestion="Add null check",
            provider_count=2,
            flagged_by=("openai", "anthropic"),
        )
        assert mi.severity == "error"
        assert mi.provider_count == 2
        assert mi.flagged_by == ("openai", "anthropic")

    def test_frozen(self):
        mi = MajorityIssue(
            severity="error",
            location="app.py:5",
            category="bug",
            description="Null deref",
            suggestion="Add null check",
            provider_count=2,
            flagged_by=("openai",),
        )
        with pytest.raises(AttributeError):
            mi.provider_count = 5  # type: ignore[misc]

    def test_json_round_trip(self):
        mi = MajorityIssue(
            severity="warning",
            location="x.py:1",
            category="perf",
            description="Slow loop",
            suggestion="Vectorise",
            provider_count=3,
            flagged_by=("a", "b", "c"),
        )
        data = _round_trip(mi)
        rebuilt = MajorityIssue(
            severity=data["severity"],
            location=data["location"],
            category=data["category"],
            description=data["description"],
            suggestion=data["suggestion"],
            provider_count=data["provider_count"],
            flagged_by=tuple(data["flagged_by"]),
        )
        assert rebuilt == mi


# -- ProviderError -------------------------------------------------------------


class TestProviderError:
    def test_all_fields_set(self):
        pe = ProviderError(provider="openai", error="timeout")
        assert pe.provider == "openai"
        assert pe.error == "timeout"

    def test_frozen(self):
        pe = ProviderError(provider="openai", error="timeout")
        with pytest.raises(AttributeError):
            pe.error = "other"  # type: ignore[misc]

    def test_json_round_trip(self):
        pe = ProviderError(provider="openai", error="rate-limited")
        data = _round_trip(pe)
        rebuilt = _reconstruct(ProviderError, data)
        assert rebuilt == pe


# -- ProviderReview ------------------------------------------------------------


class TestProviderReview:
    @pytest.fixture()
    def review(self):
        return ProviderReview(
            provider="openai",
            model="gpt-4",
            quality_rating="good",
            issues=(
                Issue(
                    severity="warning",
                    location="a.py:1",
                    category="style",
                    description="bad name",
                    suggestion="rename",
                ),
            ),
            praise=("Clean code",),
            summary="Looks ok",
            raw_content='{"rating": "good"}',
        )

    def test_all_fields_set(self, review):
        assert review.provider == "openai"
        assert review.model == "gpt-4"
        assert review.quality_rating == "good"
        assert len(review.issues) == 1
        assert review.praise == ("Clean code",)
        assert review.summary == "Looks ok"

    def test_frozen(self, review):
        with pytest.raises(AttributeError):
            review.summary = "changed"  # type: ignore[misc]

    def test_json_round_trip(self, review):
        data = _round_trip(review)
        rebuilt = ProviderReview(
            provider=data["provider"],
            model=data["model"],
            quality_rating=data["quality_rating"],
            issues=tuple(Issue(**i) for i in data["issues"]),
            praise=tuple(data["praise"]),
            summary=data["summary"],
            raw_content=data["raw_content"],
        )
        assert rebuilt == review


# -- Approach ------------------------------------------------------------------


class TestApproach:
    def test_all_fields_set(self):
        a = Approach(
            name="Option A",
            recommended_by=3,
            pros=("fast", "simple"),
            cons=("limited",),
            risk_level="low",
            fit_rating="excellent",
        )
        assert a.name == "Option A"
        assert a.recommended_by == 3
        assert a.pros == ("fast", "simple")
        assert a.cons == ("limited",)
        assert a.risk_level == "low"
        assert a.fit_rating == "excellent"

    def test_defaults(self):
        a = Approach(
            name="Option B",
            recommended_by=1,
            pros=(),
            cons=(),
            risk_level="high",
        )
        assert a.fit_rating is None

    def test_frozen(self):
        a = Approach(
            name="Option A",
            recommended_by=1,
            pros=(),
            cons=(),
            risk_level="low",
        )
        with pytest.raises(AttributeError):
            a.name = "changed"  # type: ignore[misc]

    def test_json_round_trip(self):
        a = Approach(
            name="X",
            recommended_by=2,
            pros=("a",),
            cons=("b",),
            risk_level="medium",
            fit_rating="good",
        )
        data = _round_trip(a)
        rebuilt = Approach(
            name=data["name"],
            recommended_by=data["recommended_by"],
            pros=tuple(data["pros"]),
            cons=tuple(data["cons"]),
            risk_level=data["risk_level"],
            fit_rating=data["fit_rating"],
        )
        assert rebuilt == a


# -- ProviderDesignAdvice ------------------------------------------------------


class TestProviderDesignAdvice:
    def test_all_fields_set(self):
        approach = Approach(
            name="Microservices",
            recommended_by=2,
            pros=("scalable",),
            cons=("complex",),
            risk_level="medium",
        )
        pda = ProviderDesignAdvice(
            provider="anthropic",
            model="claude-3",
            recommendation="Use microservices",
            approaches=(approach,),
            summary="Go with microservices",
            raw_content="raw",
        )
        assert pda.provider == "anthropic"
        assert pda.model == "claude-3"
        assert pda.recommendation == "Use microservices"
        assert len(pda.approaches) == 1
        assert pda.summary == "Go with microservices"

    def test_frozen(self):
        pda = ProviderDesignAdvice(
            provider="openai",
            model="gpt-4",
            recommendation="rec",
            approaches=(),
            summary="sum",
            raw_content="raw",
        )
        with pytest.raises(AttributeError):
            pda.provider = "other"  # type: ignore[misc]

    def test_json_round_trip(self):
        approach = Approach(
            name="Monolith",
            recommended_by=1,
            pros=("simple",),
            cons=("scaling",),
            risk_level="low",
            fit_rating="fair",
        )
        pda = ProviderDesignAdvice(
            provider="openai",
            model="gpt-4",
            recommendation="Keep it simple",
            approaches=(approach,),
            summary="Monolith for now",
            raw_content="{}",
        )
        data = _round_trip(pda)
        rebuilt = ProviderDesignAdvice(
            provider=data["provider"],
            model=data["model"],
            recommendation=data["recommendation"],
            approaches=tuple(
                Approach(
                    name=a["name"],
                    recommended_by=a["recommended_by"],
                    pros=tuple(a["pros"]),
                    cons=tuple(a["cons"]),
                    risk_level=a["risk_level"],
                    fit_rating=a["fit_rating"],
                )
                for a in data["approaches"]
            ),
            summary=data["summary"],
            raw_content=data["raw_content"],
        )
        assert rebuilt == pda


# -- ClassificationResult -----------------------------------------------------


class TestClassificationResult:
    def test_all_fields_set(self):
        issue = Issue(
            severity="error",
            location="x.py:1",
            category="bug",
            description="desc",
            suggestion="fix",
        )
        mi = MajorityIssue(
            severity="warning",
            location="y.py:2",
            category="style",
            description="desc2",
            suggestion="fix2",
            provider_count=2,
            flagged_by=("a", "b"),
        )
        cr = ClassificationResult(
            consensus_issues=(issue,),
            majority_issues=(mi,),
            individual_issues={"openai": (issue,)},
        )
        assert cr.consensus_issues == (issue,)
        assert cr.majority_issues == (mi,)
        assert cr.individual_issues == {"openai": (issue,)}

    def test_frozen(self):
        cr = ClassificationResult(
            consensus_issues=(),
            majority_issues=(),
            individual_issues={},
        )
        with pytest.raises(AttributeError):
            cr.consensus_issues = ()  # type: ignore[misc]

    def test_json_round_trip(self):
        issue = Issue(
            severity="error",
            location="z.py:3",
            category="sec",
            description="vuln",
            suggestion="patch",
        )
        cr = ClassificationResult(
            consensus_issues=(issue,),
            majority_issues=(),
            individual_issues={"anthropic": (issue,)},
        )
        data = _round_trip(cr)
        rebuilt = ClassificationResult(
            consensus_issues=tuple(Issue(**i) for i in data["consensus_issues"]),
            majority_issues=tuple(
                MajorityIssue(
                    severity=m["severity"],
                    location=m["location"],
                    category=m["category"],
                    description=m["description"],
                    suggestion=m["suggestion"],
                    provider_count=m["provider_count"],
                    flagged_by=tuple(m["flagged_by"]),
                )
                for m in data["majority_issues"]
            ),
            individual_issues={k: tuple(Issue(**i) for i in v) for k, v in data["individual_issues"].items()},
        )
        assert rebuilt == cr


# -- DebateMetadata ------------------------------------------------------------


class TestDebateMetadata:
    def test_all_fields_set(self):
        dm = DebateMetadata(rounds_completed=3, converged=True)
        assert dm.rounds_completed == 3
        assert dm.converged is True

    def test_frozen(self):
        dm = DebateMetadata(rounds_completed=1, converged=False)
        with pytest.raises(AttributeError):
            dm.converged = True  # type: ignore[misc]

    def test_json_round_trip(self):
        dm = DebateMetadata(rounds_completed=2, converged=False)
        data = _round_trip(dm)
        rebuilt = _reconstruct(DebateMetadata, data)
        assert rebuilt == dm


# -- CodeReviewResult -------------------------------------------------------------


class TestCodeReviewResult:
    @pytest.fixture()
    def result(self):
        issue = Issue(
            severity="error",
            location="m.py:1",
            category="bug",
            description="crash",
            suggestion="fix it",
        )
        mi = MajorityIssue(
            severity="warning",
            location="m.py:2",
            category="style",
            description="naming",
            suggestion="rename",
            provider_count=2,
            flagged_by=("openai", "anthropic"),
        )
        review = ProviderReview(
            provider="openai",
            model="gpt-4",
            quality_rating="good",
            issues=(issue,),
            praise=("nice",),
            summary="ok",
            raw_content="raw",
        )
        return CodeReviewResult(
            mode="code-review",
            providers_used=("openai", "anthropic"),
            failed_providers=(ProviderError(provider="local", error="crash"),),
            reviews=(review,),
            consensus_issues=(issue,),
            majority_issues=(mi,),
            individual_issues={"openai": (issue,)},
            quality_ratings={"openai": "good"},
            summary="Overall good",
            debate=DebateMetadata(rounds_completed=2, converged=True),
        )

    def test_all_fields_set(self, result):
        assert result.mode == "code-review"
        assert result.providers_used == ("openai", "anthropic")
        assert len(result.failed_providers) == 1
        assert len(result.reviews) == 1
        assert len(result.consensus_issues) == 1
        assert len(result.majority_issues) == 1
        assert result.individual_issues == {"openai": result.consensus_issues}
        assert result.quality_ratings == {"openai": "good"}
        assert result.summary == "Overall good"
        assert result.debate is not None
        assert result.debate.converged is True

    def test_defaults(self):
        cr = CodeReviewResult(
            mode="code-review",
            providers_used=(),
            failed_providers=(),
            reviews=(),
            consensus_issues=(),
            majority_issues=(),
            individual_issues={},
            quality_ratings={},
            summary="empty",
        )
        assert cr.debate is None

    def test_frozen(self, result):
        with pytest.raises(AttributeError):
            result.mode = "other"  # type: ignore[misc]

    def test_json_round_trip(self, result):
        data = _round_trip(result)
        # Verify the serialized JSON contains all top-level keys.
        expected_keys = {
            "mode",
            "providers_used",
            "failed_providers",
            "reviews",
            "consensus_issues",
            "majority_issues",
            "individual_issues",
            "quality_ratings",
            "summary",
            "debate",
        }
        assert set(data.keys()) == expected_keys
        # Verify nested structures survive serialization.
        assert data["debate"]["converged"] is True
        assert data["debate"]["rounds_completed"] == 2
        assert len(data["reviews"]) == 1
        assert data["reviews"][0]["provider"] == "openai"


# -- DesignQuestionResult --------------------------------------------------------------


class TestDesignQuestionResult:
    @pytest.fixture()
    def design_result(self):
        approach = Approach(
            name="Option A",
            recommended_by=2,
            pros=("fast",),
            cons=("limited",),
            risk_level="low",
            fit_rating="good",
        )
        return DesignQuestionResult(
            mode="design-question",
            prompt="How should we structure the API?",
            providers_used=("openai", "anthropic"),
            failed_providers=(),
            approaches=(approach,),
            consensus_recommendation="Go with Option A",
            summary="Option A is best",
            debate=DebateMetadata(rounds_completed=1, converged=True),
        )

    def test_all_fields_set(self, design_result):
        assert design_result.mode == "design-question"
        assert design_result.prompt == "How should we structure the API?"
        assert design_result.providers_used == ("openai", "anthropic")
        assert design_result.failed_providers == ()
        assert len(design_result.approaches) == 1
        assert design_result.consensus_recommendation == "Go with Option A"
        assert design_result.summary == "Option A is best"
        assert design_result.debate is not None

    def test_defaults(self):
        dr = DesignQuestionResult(
            mode="design-question",
            prompt="What DB?",
            providers_used=(),
            failed_providers=(),
            approaches=(),
        )
        assert dr.consensus_recommendation is None
        assert dr.summary == ""
        assert dr.debate is None

    def test_frozen(self, design_result):
        with pytest.raises(AttributeError):
            design_result.mode = "other"  # type: ignore[misc]

    def test_json_round_trip(self, design_result):
        data = _round_trip(design_result)
        expected_keys = {
            "mode",
            "prompt",
            "providers_used",
            "failed_providers",
            "approaches",
            "consensus_recommendation",
            "summary",
            "debate",
        }
        assert set(data.keys()) == expected_keys
        assert data["approaches"][0]["name"] == "Option A"
        assert data["debate"]["converged"] is True
