"""The Host projects complete formulas and independent development decisions."""

from __future__ import annotations

from alphalattice.foundation.feature_engine.catalog.authoring_surface import (
    describe_factor_authoring_surface,
)
from alphalattice.foundation.feature_engine.catalog.contracts import desktop_core_feature_bundle
from alphalattice.foundation.feature_engine.producers.factors.catalog import (
    extension_factor_specs,
    installed_formula_specs,
)


def test_host_projects_every_entry_without_stopping_on_one_refusal() -> None:
    surface = describe_factor_authoring_surface()
    installed_ids = {item.factor_id for item in extension_factor_specs()}
    assert {item.factor_id for item in surface.entries} == installed_ids
    assert set(surface.admitted_factor_ids) == installed_ids - {"sector_leader_lag_5"}
    assert surface.refused_factor_ids == ("sector_leader_lag_5",)
    leader = next(value for value in surface.entries if value.factor_id == "sector_leader_lag_5")
    assert leader.formula_status == "COMPLETE"
    assert leader.development_disposition == "REFUSED"
    assert leader.development_reason == "LAGGED_SECTOR_MEMBERSHIP_UNAVAILABLE"


def test_the_installed_formulas_are_the_controls_and_the_extensions() -> None:
    controls = desktop_core_feature_bundle().factor_ids
    extension_ids = {item.factor_id for item in extension_factor_specs()}
    assert set(controls).isdisjoint(extension_ids)
    assert {item.factor_id for item in installed_formula_specs()} == set(controls) | extension_ids


def test_authoring_projection_is_deterministic() -> None:
    assert describe_factor_authoring_surface() == describe_factor_authoring_surface()
