"""The return surface is chosen by resolved authority, never by directory luck.

``load_published_return_surface`` required the manifests directory to hold
exactly one file. That is a property of a fixture, not of a running system: a
real workspace accumulates immutable return surfaces as dates advance, so the
second publication made the Desk unusable rather than ambiguous.

These cases use the real surface from the shared workspace and derive variants
from it, so what is being selected between are genuinely well-formed surfaces
that differ only on the axes selection is supposed to care about.
"""

from __future__ import annotations

import pytest

from alphalattice.investment.risk_research.contracts import (
    CausalRiskReturnSurface,
    RiskUniverseEpoch,
    seal_contract,
)
from alphalattice.investment.risk_research.experiments.window import (
    select_return_surface_for_authority,
)
from alphalattice.protocols.research_authoring.contracts import (
    AuthoringError,
    ResolvedResearchAuthority,
)
from tests.researcher_methodology_surface.real_workspace import RealRiskWorkspace


def _authority(workspace: RealRiskWorkspace) -> ResolvedResearchAuthority:
    surface = workspace.return_surface
    return ResolvedResearchAuthority.create(
        data_snapshot_handle="current",
        universe_handle="us-current-index-research",
        panel_snapshot_hash=surface.epoch.panel_snapshot_hash,
        panel_manifest_ref=workspace.panel_manifest_ref,
        universe_revision_sha256=surface.epoch.universe_manifest_revision,
        ordered_listing_ids=tuple(surface.epoch.ordered_listing_ids),
        sessions=(surface.last_formation_session,),
        source_watermark_hash="0" * 64,
    )


def _variant(
    surface: CausalRiskReturnSurface, **epoch_overrides: object
) -> CausalRiskReturnSurface:
    """A second well-formed surface differing only on the named epoch axes.

    Re-sealed rather than hand-built, so the result is a real artifact that would
    verify on its own -- which is the point: selection has to reject valid
    surfaces that cannot answer for this authority, not merely malformed ones.
    """

    epoch_values = surface.epoch.model_dump(mode="json")
    epoch_values.pop("epoch_hash")
    epoch_values.update(epoch_overrides)
    epoch = seal_contract(RiskUniverseEpoch, "epoch_hash", **epoch_values)
    values = surface.model_dump(mode="json")
    values.pop("surface_hash")
    values["epoch"] = epoch
    return seal_contract(CausalRiskReturnSurface, "surface_hash", **values)


def test_the_matching_surface_is_chosen_from_several(
    real_risk_workspace: RealRiskWorkspace,
) -> None:
    """requirement: two surfaces coexist and the authority decides, not the order."""

    wanted = real_risk_workspace.return_surface
    other = _variant(wanted, panel_snapshot_hash="a" * 64)
    authority = _authority(real_risk_workspace)

    for surfaces in ((other, wanted), (wanted, other)):
        chosen = select_return_surface_for_authority(authority=authority, surfaces=surfaces)
        assert chosen.surface_hash == wanted.surface_hash


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("panel_snapshot_hash", "b" * 64),
        ("universe_manifest_revision", "c" * 64),
    ],
)
def test_a_same_shaped_surface_on_another_axis_is_refused(
    real_risk_workspace: RealRiskWorkspace,
    field: str,
    value: str,
) -> None:
    """Same shape, same asset count, wrong provenance -- and so unusable."""

    stray = _variant(real_risk_workspace.return_surface, **{field: value})
    with pytest.raises(AuthoringError, match="return_surface_unavailable_for_authority"):
        select_return_surface_for_authority(
            authority=_authority(real_risk_workspace), surfaces=(stray,)
        )


def test_a_permuted_listing_axis_is_refused(real_risk_workspace: RealRiskWorkspace) -> None:
    """A covariance surface with a permuted listing axis is refused even when its membership is
    unchanged."""

    surface = real_risk_workspace.return_surface
    listings = tuple(surface.epoch.ordered_listing_ids)
    permuted = _variant(surface, ordered_listing_ids=[listings[1], listings[0], *listings[2:]])
    assert set(permuted.epoch.ordered_listing_ids) == set(listings)
    with pytest.raises(AuthoringError, match="return_surface_unavailable_for_authority"):
        select_return_surface_for_authority(
            authority=_authority(real_risk_workspace), surfaces=(permuted,)
        )


def test_no_published_surface_is_refused(real_risk_workspace: RealRiskWorkspace) -> None:
    with pytest.raises(AuthoringError, match="return_surface_unavailable_for_authority"):
        select_return_surface_for_authority(authority=_authority(real_risk_workspace), surfaces=())


def test_two_exact_matches_are_refused(real_risk_workspace: RealRiskWorkspace) -> None:
    """Selecting a return surface refuses two exact matches rather than choosing by publication
    time."""

    surface = real_risk_workspace.return_surface
    # A genuinely different surface that still matches on every selection axis:
    # same Panel, same universe revision, same ordered listings, different source
    # watermark. That is what two publications of one epoch at different times
    # actually look like -- and it is why the watermark cannot serve as a
    # tie-breaker here, since it is precisely the thing that differs.
    values = surface.model_dump(mode="json")
    values.pop("surface_hash")
    values["source_watermark_hash"] = "d" * 64
    twin = seal_contract(CausalRiskReturnSurface, "surface_hash", **values)

    assert twin.surface_hash != surface.surface_hash
    assert twin.epoch.epoch_hash == surface.epoch.epoch_hash
    with pytest.raises(AuthoringError, match="return_surface_ambiguous_for_authority"):
        select_return_surface_for_authority(
            authority=_authority(real_risk_workspace), surfaces=(surface, twin)
        )
