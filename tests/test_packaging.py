from __future__ import annotations

from importlib.metadata import metadata

from packaging.requirements import Requirement

PACKAGE = "star-chamber"
PROVIDER_DEPENDENCY = "any-llm-sdk"


def _requirements() -> list[Requirement]:
    return [Requirement(raw) for raw in metadata(PACKAGE).get_all("Requires-Dist") or []]


def _base_requirement(name: str) -> Requirement:
    matches = [req for req in _requirements() if req.name == name and req.marker is None]
    assert len(matches) == 1, f"expected exactly one unconditional {name} requirement: {matches}"
    return matches[0]


def _requirement_for_extra(name: str, extra: str) -> Requirement:
    matches = [
        req
        for req in _requirements()
        if req.name == name and req.marker is not None and req.marker.evaluate({"extra": extra})
    ]
    assert len(matches) == 1, f"expected exactly one {name} requirement gated on '{extra}': {matches}"
    return matches[0]


# -- Otari extra --------------------------------------------------------------


class TestOtariExtra:
    """The 'otari' extra is part of the installable contract, not just docs.

    Otari-mode councils dispatch through any-llm's otari provider, which needs
    a package the base install omits. Declaring the extra here lets callers ask
    for Otari support by star-chamber's own name instead of reaching into a
    transitive dependency's extras.
    """

    def test_otari_extra_is_declared(self):
        extras = metadata(PACKAGE).get_all("Provides-Extra") or []
        assert "otari" in extras

    def test_otari_extra_requires_the_any_llm_otari_provider(self):
        gated = _requirement_for_extra(PROVIDER_DEPENDENCY, "otari")
        assert "otari" in gated.extras

    def test_otari_extra_mirrors_the_base_provider_pin(self):
        # The extra exists to add a provider client, not to widen or narrow the
        # version range the base install already resolves. Comparing the two
        # specifiers catches drift without restating the range as a literal,
        # which would otherwise need editing on every intentional bump.
        base = _base_requirement(PROVIDER_DEPENDENCY)
        gated = _requirement_for_extra(PROVIDER_DEPENDENCY, "otari")

        assert gated.specifier == base.specifier, f"extra {gated.specifier} drifted from base {base.specifier}"
