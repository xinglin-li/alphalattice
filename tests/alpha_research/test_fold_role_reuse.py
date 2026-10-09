"""A role matrix cache cannot cross the scope that fitted its scaling state."""

from dataclasses import replace
from datetime import date, timedelta
from inspect import signature
from types import MappingProxyType

import numpy as np

from alphalattice.investment.alpha_research.inputs.panel_feature_materialization import (
    fold_role_materialization_key,
    materialize_panel_feature_projection,
    preflight_panel_feature_plan,
)
from alphalattice.investment.alpha_research.inputs.panel_feature_views import (
    PanelFeatureSourceArrays,
)


def test_shared_role_cache_mutates_source_recipe_and_consuming_windows() -> None:
    """Role-cache reuse matches a fresh materialization for each source recipe and consuming
    window while receipt-only labels preserve the cache entry."""
    sessions = tuple(date(2024, 1, 1) + timedelta(days=i) for i in range(60))
    ids = tuple(f"L{i}" for i in range(6))
    rng = np.random.default_rng(1729)

    def readonly(array):
        array.setflags(write=False)
        return array

    raw = readonly(rng.normal(size=(60, 6)))
    source = PanelFeatureSourceArrays(
        formation_sessions=sessions,
        holding_end_sessions=tuple(d + timedelta(days=1) for d in sessions),
        ordered_listing_ids=ids,
        ordered_factor_ids=("mom_252_21",),
        absolute_state_factor_ids=("mom_252_21",),
        ordered_sector_ids=("S0", "S1"),
        sector_by_listing_id=MappingProxyType({k: f"S{i % 2}" for i, k in enumerate(ids)}),
        raw_formula_values=readonly(raw[:, :, None]),
        total_return_target_z=raw,
        raw_log_execution_returns=raw,
        raw_simple_execution_returns=raw,
        sector_context_values=readonly(rng.normal(size=(60, 2, 5))),
        market_context_values=readonly(rng.normal(size=(60, 8))),
        source_identity_hashes=MappingProxyType({"panel": "a" * 64, "outcome": "b" * 64}),
    )
    plan = preflight_panel_feature_plan(
        source=source, selected_method_ids=("RELATIVE_CONTROL",), maximum_aggregation_span=1
    )
    cache = {}

    def project(training, transforms, shared, *, program="c" * 64):
        return materialize_panel_feature_projection(
            plan=plan,
            method_id="RELATIVE_CONTROL",
            program_hash=program,
            fold_index=0,
            boundary_id="OUTER",
            training_sessions=training,
            transform_sessions=transforms,
            role_materialization_cache=shared,
        )

    training, transforms = sessions[5:25], sessions[40:50]
    first = project(training, transforms, cache)
    entries = dict(cache)
    again = project(training, transforms, cache, program="d" * 64)
    assert cache.keys() == entries.keys()
    assert all(cache[k] is entries[k] for k in entries)
    np.testing.assert_array_equal(first.training_features, again.training_features)
    for changed_training, changed_transforms in (
        (sessions[5:30], transforms),
        (training, sessions[40:55]),
    ):
        expected = project(changed_training, changed_transforms, {})
        actual = project(changed_training, changed_transforms, cache)
        np.testing.assert_array_equal(actual.training_features, expected.training_features)
        np.testing.assert_array_equal(actual.transformed_features, expected.transformed_features)

    role = plan.selected_methods[0].roles[0]
    key = fold_role_materialization_key(plan, role, training, transforms)
    for name in source.source_identity_hashes:
        identities = dict(source.source_identity_hashes)
        identities[name] = "e" * 64
        changed = replace(
            plan, source=replace(source, source_identity_hashes=MappingProxyType(identities))
        )
        assert fold_role_materialization_key(changed, role, training, transforms) != key
    changed = replace(
        plan, preflight=plan.preflight.model_copy(update={"preflight_hash": "e" * 64})
    )
    assert fold_role_materialization_key(changed, role, training, transforms) != key
    changed_role = role.model_copy(update={"selection_hash": "e" * 64})
    assert fold_role_materialization_key(plan, changed_role, training, transforms) != key
    for control in ("threads", "workers", "batch", "cache"):
        assert control not in signature(fold_role_materialization_key).parameters
        assert control not in type(plan).__dataclass_fields__
        assert fold_role_materialization_key(plan, role, training, transforms) == key
