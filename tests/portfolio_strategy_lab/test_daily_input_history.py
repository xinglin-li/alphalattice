"""A new Task reuses only a freshly proved, immutable workspace prefix."""

from datetime import date
from types import SimpleNamespace

import numpy as np
import pytest

from alphalattice.control.product_host.composition.strategy_score_inputs import (
    build_workspace_score_inputs,
    prepare_workspace_component_inputs,
    workspace_observation_history_columns,
    workspace_score_source_identity,
)
from alphalattice.control.product_host.composition.strategy_scoring import (
    workspace_observation_history_reuse,
)
from alphalattice.foundation.feature_engine.producers.factors.session_observation import (
    SESSION_OBSERVATION_FACTOR_IDS,
)
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from alphalattice.investment.alpha_research.publication.artifacts import AlphaCurrentArtifactStore
from tests.researcher_methodology_surface.real_workspace import OBSERVED_AT
from tests.researcher_methodology_surface.session_fixtures import (
    offline_execution_environment,
    real_risk_workspace,
)
from tests.researcher_methodology_surface.session_workspace import copy_workspace

__all__ = ["offline_execution_environment", "real_risk_workspace"]

FIRST, LAST = date(2026, 7, 28), date(2026, 7, 31)


@pytest.fixture
def daily_workspace(real_risk_workspace, tmp_path):
    return copy_workspace(real_risk_workspace.workspace, tmp_path / "daily")


def _publish(store, prepared, reuse=None):
    assert prepared.dependency_prefix_hash is not None
    return store.publish_workspace_observation_history(
        scope_hash="1" * 64,
        selection_hash="2" * 64,
        dependency_prefix_hash=prepared.dependency_prefix_hash,
        formation_sessions=prepared.formation_sessions,
        ordered_listing_ids=prepared.ordered_listing_ids,
        stable_session_count=prepared.stable_session_count,
        columns=workspace_observation_history_columns(prepared),
        reuse=reuse,
        capacity=lambda count: None,
    )


def _capture(workspace, through, features=(), history=None):
    return prepare_workspace_component_inputs(
        workspace,
        through=through,
        observed_at=OBSERVED_AT,
        expected_source_hash=workspace_score_source_identity(workspace),
        ordered_feature_ids=features,
        history=history,
    )


def _same_preparation(left, right):
    assert left.formation_sessions == right.formation_sessions
    assert left.ordered_listing_ids == right.ordered_listing_ids
    for name in ("ohlcv", "observations", "formula_values"):
        expected, actual = getattr(left, name), getattr(right, name)
        assert set(expected) == set(actual)
        for key in expected:
            assert expected[key].dtype == actual[key].dtype
            assert expected[key].shape == actual[key].shape
            assert expected[key].tobytes() == actual[key].tobytes()
    assert left.raw_simple.tobytes() == right.raw_simple.tobytes()
    assert left.actions == right.actions


@pytest.mark.parametrize("with_formulas", [False, True])
def test_a_fresh_owner_appends_the_same_history_and_publishes_the_same_observations(
    daily_workspace, with_formulas
):
    features = (
        (
            "RELATIVE_STOCK_CROSS_SECTION::mom_252_21::identity",
            "NON_NEUTRAL_STOCK_CROSS_SECTION::gap::identity",
        )
        if with_formulas
        else ()
    )
    store = AlphaCurrentArtifactStore(daily_workspace / "artifacts")
    previous = _publish(store, _capture(daily_workspace, FIRST, features))
    # This is a new owner with no in-memory prepared value from the first Task.
    fresh = AlphaCurrentArtifactStore(daily_workspace / "artifacts")
    history = fresh.load_workspace_observation_history("1" * 64)
    assert history is not None
    appended = _capture(daily_workspace, LAST, features, history)
    full = _capture(daily_workspace, LAST, features)
    assert appended.reused_history_hash == previous.head_hash
    _same_preparation(full, appended)
    for day in (FIRST, LAST):
        request = dict(
            formation=day,
            observed_at=OBSERVED_AT,
            expected_source_hash=workspace_score_source_identity(daily_workspace),
            ordered_feature_ids=features,
        )
        ordinary = build_workspace_score_inputs(daily_workspace, **request)
        reused = build_workspace_score_inputs(daily_workspace, **request, prepared=appended)
        for name in (
            "open",
            "high",
            "low",
            "close",
            "volume",
            "market_context_values",
            "sector_trend_values",
            "reference_eligible",
            "nominal_member_count",
        ):
            np.testing.assert_array_equal(getattr(ordinary, name), getattr(reused, name))
        assert fresh.publish_frozen_observations(
            ordinary, disposition="RECORDED_INPUT_QA"
        ) == fresh.publish_frozen_observations(reused, disposition="RECORDED_INPUT_QA")


@pytest.mark.parametrize("correction", ["bar", "feature", "sector"])
def test_an_old_value_correction_rebuilds_the_history_even_without_a_revision(
    daily_workspace, correction
):
    store = AlphaCurrentArtifactStore(daily_workspace / "artifacts")
    previous = _publish(store, _capture(daily_workspace, FIRST))
    history = store.load_workspace_observation_history("1" * 64)
    assert history is not None
    market = MarketDataRepository(daily_workspace)
    listing = previous.ordered_listing_ids[0]
    old_identity = workspace_score_source_identity(daily_workspace)
    with market.database.connect(read_only=False) as connection:
        if correction == "bar":
            connection.execute(
                "UPDATE raw_daily_bar_current SET open = open + 0.01 "
                "WHERE listing_id = ? AND session_date = ?",
                [listing, FIRST],
            )
        elif correction == "feature":
            connection.execute(
                "UPDATE feature_daily_current SET dist_52w_high = dist_52w_high + 0.01 "
                "WHERE listing_id = ? AND session_date = ?",
                [listing, FIRST],
            )
        else:
            connection.execute(
                "UPDATE sector_classification_current SET sector_name = "
                "CASE WHEN sector_name = 'Energy' THEN 'Technology' ELSE 'Energy' END "
                "WHERE listing_id = ?",
                [listing],
            )
    assert workspace_score_source_identity(daily_workspace) == old_identity
    actual = _capture(daily_workspace, LAST, history=history)
    expected = _capture(daily_workspace, LAST)
    assert actual.reused_history_hash is None
    _same_preparation(expected, actual)


def test_a_future_action_is_checked_at_its_effective_cutoff_after_history_reload(daily_workspace):
    store = AlphaCurrentArtifactStore(daily_workspace / "artifacts")
    previous = _publish(store, _capture(daily_workspace, FIRST))
    history = store.load_workspace_observation_history("1" * 64)
    assert history is not None
    market = MarketDataRepository(daily_workspace)
    listing = previous.ordered_listing_ids[0]
    bar = market.raw_bars(listing, start=LAST, through=LAST)[0]
    with market.database.connect(read_only=False) as connection:
        connection.execute(
            "INSERT INTO corporate_action_current VALUES "
            "(?, ?, ?, 'CAPITAL_GAIN', NULL, NULL, TRUE, 'synthetic-future-action', "
            "?, 'ACTIVE', ?)",
            [listing, bar.provider, LAST, "a" * 64, OBSERVED_AT],
        )
    assert (
        _capture(daily_workspace, FIRST, history=history).reused_history_hash == previous.head_hash
    )
    with pytest.raises(ValueError, match="unsupported corporate action"):
        _capture(daily_workspace, LAST, history=history)


def test_a_reloaded_owner_appends_every_session_observation_with_the_same_public_snapshot(
    daily_workspace,
):
    """requirement: the verified daily prefix preserves every selected extension and readback."""

    features = tuple(
        f"NON_NEUTRAL_STOCK_CROSS_SECTION::{name}::identity"
        for name in SESSION_OBSERVATION_FACTOR_IDS
    )
    store = AlphaCurrentArtifactStore(daily_workspace / "artifacts")
    previous = _publish(store, _capture(daily_workspace, FIRST, features))
    fresh = AlphaCurrentArtifactStore(daily_workspace / "artifacts")
    history = fresh.load_workspace_observation_history("1" * 64)
    assert history is not None
    for through in (FIRST, LAST):
        appended = _capture(daily_workspace, through, features, history)
        full = _capture(daily_workspace, through, features)
        assert appended.reused_history_hash == previous.head_hash
        assert set(SESSION_OBSERVATION_FACTOR_IDS) <= appended.formula_values.keys()
        _same_preparation(full, appended)
        request = dict(
            formation=through,
            observed_at=OBSERVED_AT,
            expected_source_hash=workspace_score_source_identity(daily_workspace),
            ordered_feature_ids=features,
        )
        ordinary = build_workspace_score_inputs(daily_workspace, **request)
        reused = build_workspace_score_inputs(daily_workspace, **request, prepared=appended)
        assert set(ordinary.formula_values) == set(reused.formula_values)
        for name in ordinary.formula_values:
            assert ordinary.formula_values[name].dtype == reused.formula_values[name].dtype
            assert ordinary.formula_values[name].shape == reused.formula_values[name].shape
            assert ordinary.formula_values[name].tobytes() == reused.formula_values[name].tobytes()
        assert fresh.publish_frozen_observations(
            ordinary, disposition="RECORDED_INPUT_QA"
        ) == fresh.publish_frozen_observations(reused, disposition="RECORDED_INPUT_QA")


WIDE = (
    "RELATIVE_STOCK_CROSS_SECTION::mom_252_21::identity",
    "NON_NEUTRAL_STOCK_CROSS_SECTION::gap::identity",
)
"""A renewal day's Features: the model's plus a training factor the next day does not need."""


def _column_hashes(prepared):
    import hashlib

    return {
        name: hashlib.sha256(values.tobytes()).hexdigest()
        for name, values in workspace_observation_history_columns(prepared).items()
    }


def test_the_day_after_a_renewal_reuses_its_wider_history_as_a_cold_rebuild(daily_workspace):
    """The day after a renewal reuses its wider history as a cold rebuild."""
    store = AlphaCurrentArtifactStore(daily_workspace / "artifacts")
    previous = _publish(store, _capture(daily_workspace, FIRST, WIDE))
    history = AlphaCurrentArtifactStore(
        daily_workspace / "artifacts"
    ).load_workspace_observation_history("1" * 64)
    narrow = WIDE[:1]
    appended = _capture(daily_workspace, LAST, narrow, history)
    assert appended.reused_history_hash == previous.head_hash
    cold = _capture(daily_workspace, LAST, narrow)
    _same_preparation(cold, appended)
    assert _column_hashes(cold) == _column_hashes(appended)
    assert appended.dependency_prefix_hash == cold.dependency_prefix_hash
    # The publication the scoring owner makes of it: the renewal's wider parts are not shared,
    # the day's own columns are written, and the day after reuses them, sharing their parts.
    columns = workspace_observation_history_columns(appended)
    assert workspace_observation_history_reuse(history[0], appended, columns) is None
    with pytest.raises(Exception, match="workspace_observation_history_reuse_invalid"):
        _publish(store, appended, reuse=history[0])
    narrow_head = _publish(store, appended)
    again = _capture(
        daily_workspace, LAST, narrow, store.load_workspace_observation_history("1" * 64)
    )
    assert again.reused_history_hash == narrow_head.head_hash
    assert (
        workspace_observation_history_reuse(
            narrow_head, again, workspace_observation_history_columns(again)
        )
        == narrow_head
    )
    request = dict(
        formation=LAST,
        observed_at=OBSERVED_AT,
        expected_source_hash=workspace_score_source_identity(daily_workspace),
        ordered_feature_ids=narrow,
    )
    ordinary = build_workspace_score_inputs(daily_workspace, **request)
    reused = build_workspace_score_inputs(daily_workspace, **request, prepared=appended)
    for name in ("close", "volume", "market_context_values", "sector_trend_values"):
        np.testing.assert_array_equal(getattr(ordinary, name), getattr(reused, name))
    assert set(ordinary.formula_values) == set(reused.formula_values)
    for name, values in ordinary.formula_values.items():
        assert values.tobytes() == reused.formula_values[name].tobytes()


def test_a_kept_history_is_refused_for_more_columns_or_another_implementation(daily_workspace):
    """A kept history is refused for more columns or another implementation."""

    from alphalattice.control.product_host.composition.strategy_scoring import (
        workspace_observation_history_selection,
    )

    store = AlphaCurrentArtifactStore(daily_workspace / "artifacts")
    _publish(store, _capture(daily_workspace, FIRST, WIDE[:1]))
    history = store.load_workspace_observation_history("1" * 64)
    wider = _capture(daily_workspace, LAST, WIDE, history)
    assert wider.reused_history_hash is None
    _same_preparation(_capture(daily_workspace, LAST, WIDE), wider)
    binding = SimpleNamespace(
        strategy_package_id="package",
        component_id="component",
        source_kind="WORKSPACE_DATA_FEATURE",
        strategy_package_hash="3" * 64,
    )
    same = workspace_observation_history_selection(binding, implementation="a" * 64)
    assert same == workspace_observation_history_selection(binding, implementation="a" * 64)
    assert same != workspace_observation_history_selection(binding, implementation="b" * 64)
