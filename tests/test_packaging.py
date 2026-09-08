from __future__ import annotations

from importlib.metadata import metadata

# -- Otari extra --------------------------------------------------------------


class TestOtariExtra:
    """The 'otari' extra is part of the installable contract, not just docs.

    Otari-mode councils dispatch through any-llm's otari provider, which needs
    a package the base install omits. Declaring the extra here lets callers ask
    for Otari support by star-chamber's own name instead of reaching into a
    transitive dependency's extras.
    """

    def test_otari_extra_is_declared(self):
        extras = metadata("star-chamber").get_all("Provides-Extra") or []
        assert "otari" in extras

    def test_otari_extra_requires_the_any_llm_otari_provider(self):
        requires = metadata("star-chamber").get_all("Requires-Dist") or []
        gated = [req for req in requires if 'extra == "otari"' in req]

        assert gated, f"no Requires-Dist entry is gated on the otari extra: {requires}"
        assert any("any-llm-sdk" in req and "[otari]" in req for req in gated), gated
