from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

import pytest

from star_chamber.types import (
    Approach,
    ClassificationResult,
    CodeReviewResult,
    CodeReviewSummary,
    CouncilConfig,
    DesignQuestionResult,
    DesignSummary,
    Finding,
    FitRating,
    Issue,
    IssueCategory,
    IssueGroup,
    OtariConfig,
    ProviderConfig,
    ProviderDesignAdvice,
    ProviderError,
    ProviderReview,
    QualityRating,
    RiskLevel,
    Severity,
)

# -- helpers ------------------------------------------------------------------


def _round_trip(obj):
    """Serialize via asdict -> json.dumps -> json.loads and return the dict."""
    d = asdict(obj)
    return json.loads(json.dumps(d))


# The type each closed-set field carries, so a round trip restores it from text as a consumer must.
_CLOSED_SETS = {
    "category": IssueCategory,
    "fit_rating": FitRating,
    "quality_rating": QualityRating,
    "risk_level": RiskLevel,
    "severity": Severity,
}


def _reconstruct(cls, data):
    """Reconstruct a flat frozen dataclass from a dict, restoring each closed-set field to its type."""
    restored = {
        key: _CLOSED_SETS[key](value) if key in _CLOSED_SETS and value is not None else value
        for key, value in data.items()
    }
    return cls(**restored)


# -- ProviderConfig -----------------------------------------------------------


class TestProviderConfig:
    def test_all_fields_set(self) -> None:
        pc = ProviderConfig(
            provider="openai",
            model="gpt-4",
            api_key="test-key-not-real",  # pragma: allowlist secret
            api_base="https://api.openai.com",
            max_tokens=4096,
            local=True,
            display_name="primary",
        )
        assert pc.provider == "openai"
        assert pc.model == "gpt-4"
        assert pc.api_key == "test-key-not-real"  # pragma: allowlist secret
        assert pc.api_base == "https://api.openai.com"
        assert pc.max_tokens == 4096
        assert pc.local is True
        assert pc.display_name == "primary"

    def test_identity_is_the_display_name_when_one_is_set(self) -> None:
        pc = ProviderConfig(provider="openai", model="gpt-4", display_name="primary")

        assert pc.identity == "primary"

    def test_identity_falls_back_to_the_provider_name(self) -> None:
        pc = ProviderConfig(provider="openai", model="gpt-4")

        assert pc.identity == "openai"

    def test_defaults(self) -> None:
        pc = ProviderConfig(provider="anthropic", model="claude-3")
        assert pc.api_key is None
        assert pc.api_base is None
        assert pc.max_tokens is None
        assert pc.local is False
        assert pc.display_name is None

    def test_frozen(self) -> None:
        pc = ProviderConfig(provider="openai", model="gpt-4")
        with pytest.raises(AttributeError):
            pc.provider = "anthropic"  # type: ignore[misc]

    def test_json_round_trip(self) -> None:
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
    def test_all_fields_set(self) -> None:
        p1 = ProviderConfig(provider="openai", model="gpt-4")
        p2 = ProviderConfig(provider="anthropic", model="claude-3")
        gw = OtariConfig(api_base="https://gw.example", api_key="${OTARI_API_KEY}")
        cc = CouncilConfig(
            providers=(p1, p2),
            timeout_seconds=120,
            consensus_threshold=3,
            otari=gw,
        )
        assert cc.providers == (p1, p2)
        assert cc.timeout_seconds == 120
        assert cc.consensus_threshold == 3
        assert cc.otari == gw

    def test_defaults(self) -> None:
        p1 = ProviderConfig(provider="openai", model="gpt-4")
        cc = CouncilConfig(providers=(p1,))
        assert cc.timeout_seconds == 60
        assert cc.consensus_threshold == 2
        assert cc.otari is None

    def test_frozen(self) -> None:
        p1 = ProviderConfig(provider="openai", model="gpt-4")
        cc = CouncilConfig(providers=(p1,))
        with pytest.raises(AttributeError):
            cc.timeout_seconds = 999  # type: ignore[misc]

    def test_json_round_trip(self) -> None:
        p1 = ProviderConfig(provider="openai", model="gpt-4", api_key="test-key-not-real")  # pragma: allowlist secret
        gw = OtariConfig(api_base="https://gw.example", api_key="${OTARI_API_KEY}")
        cc = CouncilConfig(providers=(p1,), timeout_seconds=90, otari=gw)
        data = _round_trip(cc)
        # Providers come back as list-of-dicts; rebuild manually.
        rebuilt = CouncilConfig(
            providers=tuple(ProviderConfig(**p) for p in data["providers"]),
            timeout_seconds=data["timeout_seconds"],
            consensus_threshold=data["consensus_threshold"],
            otari=OtariConfig(**data["otari"]),
        )
        assert rebuilt == cc


class TestOtariConfig:
    def test_defaults(self) -> None:
        gw = OtariConfig()
        assert gw.api_base is None
        assert gw.api_key is None

    def test_frozen(self) -> None:
        gw = OtariConfig(api_base="x", api_key="y")
        with pytest.raises(AttributeError):
            gw.api_base = "z"  # type: ignore[misc]


# -- Issue ---------------------------------------------------------------------


class TestIssue:
    def test_all_fields_set(self) -> None:
        issue = Issue(
            severity=Severity.HIGH,
            location="main.py:10",
            category=IssueCategory.ARCHITECTURE,
            description="SQL injection",
            suggestion="Use parameterised queries",
        )
        assert issue.severity == "high"
        assert issue.location == "main.py:10"
        assert issue.category == "architecture"
        assert issue.description == "SQL injection"
        assert issue.suggestion == "Use parameterised queries"

    def test_frozen(self) -> None:
        issue = Issue(
            severity=Severity.HIGH,
            location="main.py:10",
            category=IssueCategory.ARCHITECTURE,
            description="SQL injection",
            suggestion="Use parameterised queries",
        )
        with pytest.raises(AttributeError):
            issue.severity = "warning"  # type: ignore[misc]

    def test_json_round_trip(self) -> None:
        issue = Issue(
            severity=Severity.MEDIUM,
            location="utils.py:42",
            category=IssueCategory.CRAFTSMANSHIP,
            description="Long line",
            suggestion="Break the line",
        )
        data = _round_trip(issue)
        rebuilt = _reconstruct(Issue, data)
        assert rebuilt == issue


# -- Finding ------------------------------------------------------------------


def _finding(provider: str, category: IssueCategory = IssueCategory.CORRECTNESS, **overrides: object) -> Finding:
    """Build a finding, so a test states only the fields it is about."""
    fields = {
        "provider": provider,
        "severity": Severity.HIGH,
        "location": "app.py:5",
        "category": category,
        "description": f"{provider} saw a null deref",
        "suggestion": "Add a null check",
    }
    return Finding(**{**fields, **overrides})


def _rebuild_group(data: dict[str, Any]) -> IssueGroup:
    """Rebuild a grouped issue from a document, which carries only its findings as input."""
    return IssueGroup(findings=tuple(_reconstruct(Finding, finding) for finding in data["findings"]))


class TestFinding:
    def test_all_fields_set(self) -> None:
        finding = _finding("openai")

        assert finding.provider == "openai"
        assert finding.severity == "high"
        assert finding.category == "correctness"

    def test_frozen(self) -> None:
        finding = _finding("openai")

        with pytest.raises(AttributeError):
            finding.provider = "anthropic"  # type: ignore[misc]

    def test_rejects_a_severity_outside_the_set(self) -> None:
        with pytest.raises(TypeError, match="severity must be a Severity"):
            _finding("openai", severity="critical")

    def test_rejects_a_category_outside_the_set(self) -> None:
        with pytest.raises(TypeError, match="category must be a IssueCategory"):
            _finding("openai", category="performance")

    def test_json_round_trip(self) -> None:
        finding = _finding("openai")

        assert _reconstruct(Finding, _round_trip(finding)) == finding


# -- IssueGroup ---------------------------------------------------------------


class TestIssueGroup:
    def test_the_findings_decide_the_derived_fields(self) -> None:
        group = IssueGroup(
            findings=(
                _finding("openai", severity=Severity.LOW, location="app.py:5-9"),
                _finding("anthropic", category=IssueCategory.ARCHITECTURE, severity=Severity.HIGH),
            )
        )

        assert group.location == "app.py:5-9"
        assert group.severity == "high"
        assert group.categories == ("architecture", "correctness")
        assert group.flagged_by == ("anthropic", "openai")

    def test_a_caller_cannot_change_the_findings_after_the_group_is_built(self) -> None:
        """The group holds its own tuple, so the derived fields cannot drift from the findings."""
        passed = [_finding("openai")]
        group = IssueGroup(findings=passed)  # type: ignore[arg-type]

        passed.append(_finding("gemini"))

        assert group.findings == (_finding("openai"),)
        assert group.flagged_by == ("openai",)

    def test_a_derived_field_is_not_a_constructor_argument(self) -> None:
        with pytest.raises(TypeError):
            IssueGroup(findings=(_finding("openai"),), severity=Severity.LOW)  # type: ignore[call-arg]

    def test_a_group_with_no_findings_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one finding"):
            IssueGroup(findings=())

    def test_frozen(self) -> None:
        group = IssueGroup(findings=(_finding("openai"),))

        with pytest.raises(AttributeError):
            group.findings = ()  # type: ignore[misc]

    def test_json_round_trip(self) -> None:
        group = IssueGroup(findings=(_finding("openai"), _finding("anthropic")))
        data = _round_trip(group)

        rebuilt = IssueGroup(findings=tuple(_reconstruct(Finding, finding) for finding in data["findings"]))

        assert rebuilt == group
        assert data["flagged_by"] == ["anthropic", "openai"]


# -- ProviderError -------------------------------------------------------------


class TestProviderError:
    def test_all_fields_set(self) -> None:
        pe = ProviderError(provider="openai", model="gpt-4o", error="timeout", raw_content=None)
        assert pe.provider == "openai"
        assert pe.error == "timeout"

    def test_frozen(self) -> None:
        pe = ProviderError(provider="openai", model="gpt-4o", error="timeout", raw_content=None)
        with pytest.raises(AttributeError):
            pe.error = "other"  # type: ignore[misc]

    def test_json_round_trip(self) -> None:
        pe = ProviderError(provider="openai", model="gpt-4o", error="rate-limited", raw_content=None)
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
            quality_rating=QualityRating.GOOD,
            issues=(
                Issue(
                    severity=Severity.MEDIUM,
                    location="a.py:1",
                    category=IssueCategory.CRAFTSMANSHIP,
                    description="bad name",
                    suggestion="rename",
                ),
            ),
            praise=("Clean code",),
            summary="Looks ok",
            raw_content='{"rating": "good"}',
        )

    def test_all_fields_set(self, review) -> None:
        assert review.provider == "openai"
        assert review.model == "gpt-4"
        assert review.quality_rating == "good"
        assert len(review.issues) == 1
        assert review.praise == ("Clean code",)
        assert review.summary == "Looks ok"

    def test_frozen(self, review) -> None:
        with pytest.raises(AttributeError):
            review.summary = "changed"  # type: ignore[misc]

    def test_json_round_trip(self, review) -> None:
        data = _round_trip(review)
        rebuilt = ProviderReview(
            provider=data["provider"],
            model=data["model"],
            quality_rating=QualityRating(data["quality_rating"]),
            issues=tuple(_reconstruct(Issue, i) for i in data["issues"]),
            praise=tuple(data["praise"]),
            summary=data["summary"],
            raw_content=data["raw_content"],
        )
        assert rebuilt == review


# -- Approach ------------------------------------------------------------------


class TestApproach:
    def test_all_fields_set(self) -> None:
        a = Approach(
            name="Option A",
            recommended_by=3,
            pros=("fast", "simple"),
            cons=("limited",),
            risk_level=RiskLevel.LOW,
            fit_rating=FitRating.EXCELLENT,
        )
        assert a.name == "Option A"
        assert a.recommended_by == 3
        assert a.pros == ("fast", "simple")
        assert a.cons == ("limited",)
        assert a.risk_level == "low"
        assert a.fit_rating == "excellent"

    def test_a_rating_nobody_stated_is_passed_explicitly(self) -> None:
        """No default, so a site cannot claim nobody rated the approach by omission."""
        a = Approach(
            name="Option B",
            recommended_by=1,
            pros=(),
            cons=(),
            risk_level=RiskLevel.HIGH,
            fit_rating=None,
        )
        assert a.fit_rating is None

    def test_frozen(self) -> None:
        a = Approach(
            name="Option A",
            recommended_by=1,
            pros=(),
            cons=(),
            risk_level=RiskLevel.LOW,
            fit_rating=None,
        )
        with pytest.raises(AttributeError):
            a.name = "changed"  # type: ignore[misc]

    def test_json_round_trip(self) -> None:
        a = Approach(
            name="X",
            recommended_by=2,
            pros=("a",),
            cons=("b",),
            risk_level=RiskLevel.MEDIUM,
            fit_rating=FitRating.GOOD,
        )
        data = _round_trip(a)
        rebuilt = Approach(
            name=data["name"],
            recommended_by=data["recommended_by"],
            pros=tuple(data["pros"]),
            cons=tuple(data["cons"]),
            risk_level=RiskLevel(data["risk_level"]),
            fit_rating=FitRating(data["fit_rating"]),
        )
        assert rebuilt == a


# -- ProviderDesignAdvice ------------------------------------------------------


class TestProviderDesignAdvice:
    def test_all_fields_set(self) -> None:
        approach = Approach(
            name="Microservices",
            recommended_by=2,
            pros=("scalable",),
            cons=("complex",),
            risk_level=RiskLevel.MEDIUM,
            fit_rating=FitRating.GOOD,
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

    def test_frozen(self) -> None:
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

    def test_json_round_trip(self) -> None:
        approach = Approach(
            name="Monolith",
            recommended_by=1,
            pros=("simple",),
            cons=("scaling",),
            risk_level=RiskLevel.LOW,
            fit_rating=FitRating.FAIR,
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
                    risk_level=RiskLevel(a["risk_level"]),
                    fit_rating=FitRating(a["fit_rating"]),
                )
                for a in data["approaches"]
            ),
            summary=data["summary"],
            raw_content=data["raw_content"],
        )
        assert rebuilt == pda


# -- ClassificationResult -----------------------------------------------------


class TestClassificationResult:
    def test_all_fields_set(self) -> None:
        alone = IssueGroup(findings=(_finding("openai"),))
        shared = IssueGroup(findings=(_finding("openai"), _finding("anthropic")))
        cr = ClassificationResult(
            consensus_issues=(shared,),
            majority_issues=(shared,),
            individual_issues={"openai": (alone,)},
        )
        assert cr.consensus_issues == (shared,)
        assert cr.majority_issues == (shared,)
        assert cr.individual_issues == {"openai": (alone,)}

    def test_frozen(self) -> None:
        cr = ClassificationResult(
            consensus_issues=(),
            majority_issues=(),
            individual_issues={},
        )
        with pytest.raises(AttributeError):
            cr.consensus_issues = ()  # type: ignore[misc]

    def test_json_round_trip(self) -> None:
        group = IssueGroup(findings=(_finding("anthropic", IssueCategory.INVARIANTS),))
        cr = ClassificationResult(
            consensus_issues=(group,),
            majority_issues=(),
            individual_issues={"anthropic": (group,)},
        )
        data = _round_trip(cr)
        rebuilt = ClassificationResult(
            consensus_issues=tuple(_rebuild_group(g) for g in data["consensus_issues"]),
            majority_issues=tuple(_rebuild_group(g) for g in data["majority_issues"]),
            individual_issues={
                name: tuple(_rebuild_group(g) for g in groups) for name, groups in data["individual_issues"].items()
            },
        )
        assert rebuilt == cr


# -- CodeReviewResult -------------------------------------------------------------


class TestCodeReviewResult:
    @pytest.fixture()
    def result(self):
        issue = Issue(
            severity=Severity.HIGH,
            location="m.py:1",
            category=IssueCategory.CORRECTNESS,
            description="crash",
            suggestion="fix it",
        )
        alone = IssueGroup(findings=(_finding("openai", location="m.py:1"),))
        shared = IssueGroup(
            findings=(
                _finding("openai", IssueCategory.CRAFTSMANSHIP, location="m.py:2"),
                _finding("anthropic", IssueCategory.CRAFTSMANSHIP, location="m.py:2"),
            )
        )
        review = ProviderReview(
            provider="openai",
            model="gpt-4",
            quality_rating=QualityRating.GOOD,
            issues=(issue,),
            praise=("nice",),
            summary="ok",
            raw_content="raw",
        )
        return CodeReviewResult(
            mode="code-review",
            files_reviewed=("app.py",),
            providers_used=("openai", "anthropic"),
            providers_responded=("openai",),
            failed_providers=(ProviderError(provider="local", model="m", error="crash", raw_content=None),),
            reviews=(review,),
            consensus_issues=(alone,),
            majority_issues=(shared,),
            individual_issues={"openai": (alone,)},
            quality_ratings={"openai": QualityRating.GOOD},
            summary=CodeReviewSummary(total_issues=1, consensus_count=1, majority_count=1, synthesis="Overall good"),
        )

    def test_all_fields_set(self, result) -> None:
        assert result.mode == "code-review"
        assert result.providers_used == ("openai", "anthropic")
        assert result.providers_responded == ("openai",)
        assert len(result.failed_providers) == 1
        assert len(result.reviews) == 1
        assert len(result.consensus_issues) == 1
        assert len(result.majority_issues) == 1
        assert result.individual_issues == {"openai": result.consensus_issues}
        assert result.quality_ratings == {"openai": "good"}
        assert result.summary.synthesis == "Overall good"
        assert result.files_reviewed == ("app.py",)

    def test_defaults(self) -> None:
        CodeReviewResult(
            mode="code-review",
            files_reviewed=(),
            providers_used=(),
            providers_responded=(),
            failed_providers=(),
            reviews=(),
            consensus_issues=(),
            majority_issues=(),
            individual_issues={},
            quality_ratings={},
            summary=CodeReviewSummary(total_issues=0, consensus_count=0, majority_count=0, synthesis=""),
        )

    def test_frozen(self, result) -> None:
        with pytest.raises(AttributeError):
            result.mode = "other"  # type: ignore[misc]

    def test_json_round_trip(self, result) -> None:
        data = _round_trip(result)
        # Verify the serialized JSON contains all top-level keys.
        expected_keys = {
            "mode",
            "files_reviewed",
            "providers_used",
            "providers_responded",
            "failed_providers",
            "reviews",
            "consensus_issues",
            "majority_issues",
            "individual_issues",
            "quality_ratings",
            "spec_version",
            "summary",
        }
        assert set(data.keys()) == expected_keys
        # Verify nested structures survive serialization.
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
            risk_level=RiskLevel.LOW,
            fit_rating=FitRating.GOOD,
        )
        return DesignQuestionResult(
            mode="design-question",
            prompt="How should we structure the API?",
            providers_used=("openai", "anthropic"),
            providers_responded=("openai", "anthropic"),
            failed_providers=(),
            approaches=(approach,),
            consensus_recommendation="Go with Option A",
            summary=DesignSummary(synthesis="Option A is best"),
        )

    def test_all_fields_set(self, design_result) -> None:
        assert design_result.mode == "design-question"
        assert design_result.prompt == "How should we structure the API?"
        assert design_result.providers_used == ("openai", "anthropic")
        assert design_result.providers_responded == ("openai", "anthropic")
        assert design_result.failed_providers == ()
        assert len(design_result.approaches) == 1
        assert design_result.consensus_recommendation == "Go with Option A"
        assert design_result.summary.synthesis == "Option A is best"

    def test_defaults(self) -> None:
        dr = DesignQuestionResult(
            mode="design-question",
            prompt="What DB?",
            providers_used=(),
            providers_responded=(),
            failed_providers=(),
            approaches=(),
            summary=DesignSummary(synthesis=""),
        )
        assert dr.consensus_recommendation is None
        assert dr.summary.synthesis == ""

    def test_frozen(self, design_result) -> None:
        with pytest.raises(AttributeError):
            design_result.mode = "other"  # type: ignore[misc]

    def test_json_round_trip(self, design_result) -> None:
        data = _round_trip(design_result)
        expected_keys = {
            "mode",
            "prompt",
            "providers_used",
            "providers_responded",
            "failed_providers",
            "approaches",
            "consensus_recommendation",
            "spec_version",
            "summary",
        }
        assert set(data.keys()) == expected_keys
        assert data["approaches"][0]["name"] == "Option A"


# -- ProviderConfig.display_name ----------------------------------------------


class TestProviderConfigDisplayName:
    def test_none_by_default(self) -> None:
        cfg = ProviderConfig(provider="openai", model="gpt-4o")
        assert cfg.display_name is None

    def test_holds_configured_value(self) -> None:
        cfg = ProviderConfig(provider="openrouter", model="openai/gpt-5.2", display_name="gpt-5.2")
        assert cfg.display_name == "gpt-5.2"


class TestCouncilConfigThreshold:
    """A threshold the comparison cannot use must fail before any provider is called."""

    def test_a_string_threshold_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="consensus_threshold must be a whole number, got str"):
            CouncilConfig(providers=(ProviderConfig(provider="openai", model="m"),), consensus_threshold="2")

    def test_a_boolean_threshold_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="consensus_threshold must be a whole number, got bool"):
            CouncilConfig(providers=(ProviderConfig(provider="openai", model="m"),), consensus_threshold=True)

    def test_a_whole_number_is_accepted(self) -> None:
        config = CouncilConfig(providers=(ProviderConfig(provider="openai", model="m"),), consensus_threshold=3)
        assert config.consensus_threshold == 3


class TestClosedSetFields:
    """A wire type refuses a value outside the set its field declares.

    A plain string is accepted by the language and then fails deep in the
    classifier, so the type rejects it where the caller supplied it.
    """

    def _issue(self, **overrides) -> Issue:
        fields = {
            "severity": Severity.HIGH,
            "location": "a.py:1",
            "category": IssueCategory.CORRECTNESS,
            "description": "Off-by-one.",
            "suggestion": "Use <=.",
        }
        return Issue(**(fields | overrides))

    def _approach(self, **overrides) -> Approach:
        fields = {
            "name": "Repository",
            "recommended_by": 1,
            "pros": (),
            "cons": (),
            "risk_level": RiskLevel.LOW,
            "fit_rating": FitRating.GOOD,
        }
        return Approach(**(fields | overrides))

    def test_an_issue_refuses_a_raw_severity(self) -> None:
        with pytest.raises(TypeError, match="severity must be a Severity, got str"):
            self._issue(severity="high")

    def test_an_issue_refuses_a_raw_category(self) -> None:
        with pytest.raises(TypeError, match="category must be a IssueCategory, got str"):
            self._issue(category="correctness")

    def test_a_finding_refuses_a_raw_severity(self) -> None:
        with pytest.raises(TypeError, match="severity must be a Severity, got str"):
            _finding("openai", severity="high")

    def test_a_review_refuses_a_raw_rating(self) -> None:
        with pytest.raises(TypeError, match="quality_rating must be a QualityRating, got str"):
            ProviderReview(
                provider="openai",
                model="gpt-4o",
                quality_rating="good",
                issues=(),
                praise=(),
                summary="Fine.",
                raw_content="{}",
            )

    def test_an_approach_refuses_a_raw_risk_level(self) -> None:
        with pytest.raises(TypeError, match="risk_level must be a RiskLevel, got str"):
            self._approach(risk_level="low")

    def test_an_approach_refuses_a_raw_fit_rating(self) -> None:
        with pytest.raises(TypeError, match="fit_rating must be a FitRating, got str"):
            self._approach(fit_rating="good")

    def test_a_result_refuses_a_raw_rating_in_its_map(self) -> None:
        """The map keyed by member name carries the vocabulary too."""
        with pytest.raises(TypeError, match=r"quality_ratings\['openai'\] must be a QualityRating, got str"):
            CodeReviewResult(
                mode="code-review",
                files_reviewed=(),
                providers_used=(),
                providers_responded=(),
                failed_providers=(),
                reviews=(),
                consensus_issues=(),
                majority_issues=(),
                individual_issues={},
                quality_ratings={"openai": "good"},
                summary=CodeReviewSummary(total_issues=0, consensus_count=0, majority_count=0, synthesis=""),
            )

    def test_an_approach_accepts_no_fit_rating(self) -> None:
        """Null reports that nobody rated the approach, so it is not a value outside the set."""
        assert self._approach(fit_rating=None).fit_rating is None
