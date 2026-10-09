"""Task-local input prefixes through the real Data, Feature and publication owners."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from alphalattice.control.product_host.composition import strategy_score_inputs
from alphalattice.control.product_host.composition.strategy_score_inputs import (
    build_workspace_score_inputs,
    prepare_workspace_component_inputs,
    read_workspace_component_inputs,
    workspace_score_source_identity,
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

DAYS = (date(2026, 7, 28), date(2026, 7, 30), date(2026, 7, 31))


@pytest.fixture
def input_workspace(real_risk_workspace, tmp_path):
    return copy_workspace(real_risk_workspace.workspace, tmp_path / "inputs")


@pytest.fixture
def input_features(real_risk_workspace):
    """Select a materialized recipe through the fixture's real public catalog."""
    stored = next(
        factor.factor_id
        for factor in real_risk_workspace.feature_catalog.factors
        if factor.factor_id == "mom_252_21"
    )
    return (
        f"RELATIVE_STOCK_CROSS_SECTION::{stored}::identity",
        "NON_NEUTRAL_STOCK_CROSS_SECTION::gap::identity",
        "NON_NEUTRAL_STOCK_CROSS_SECTION::session_dollar_volume::identity",
    )


def _same_source(left, right, publisher):
    assert left.formation_sessions == right.formation_sessions
    assert left.ordered_listing_ids == right.ordered_listing_ids
    assert left.sector_by_listing_id == right.sector_by_listing_id
    assert left.source_binding_hash == right.source_binding_hash
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
        np.testing.assert_array_equal(getattr(left, name), getattr(right, name))
    assert tuple(left.formula_values) == tuple(right.formula_values)
    for name in left.formula_values:
        np.testing.assert_array_equal(left.formula_values[name], right.formula_values[name])
    full = publisher.publish_frozen_observations(left, disposition="RECORDED_INPUT_QA")
    prefix = publisher.publish_frozen_observations(right, disposition="RECORDED_INPUT_QA")
    assert full == prefix
    return full


@pytest.mark.parametrize("with_training", [False, True])
def test_prepared_history_keeps_every_cutoff_array_and_observation_hash(
    input_workspace, input_features, with_training
):
    """Later captures censor mature outcomes and publish the identical earlier observations."""
    source_hash = workspace_score_source_identity(input_workspace)
    training_ids = (
        tuple(sorted(("gap", input_features[0].split("::")[1], "session_dollar_volume")))
        if with_training
        else ()
    )
    prepared = prepare_workspace_component_inputs(
        input_workspace,
        through=DAYS[-1],
        observed_at=OBSERVED_AT,
        expected_source_hash=source_hash,
        ordered_feature_ids=input_features,
        training_factor_ids=training_ids,
    )
    publisher = AlphaCurrentArtifactStore(input_workspace / "artifacts")
    hashes = []
    for day in DAYS:
        request = dict(
            formation=day,
            observed_at=OBSERVED_AT,
            expected_source_hash=source_hash,
            ordered_feature_ids=input_features,
            training_factor_ids=training_ids,
        )
        full, full_training = read_workspace_component_inputs(input_workspace, **request)
        prefix, prefix_training = read_workspace_component_inputs(
            input_workspace, **request, prepared=prepared
        )
        hashes.append(_same_source(full, prefix, publisher).snapshot_hash)
        if training_ids:
            assert full_training is not None and prefix_training is not None
            assert np.isfinite(full_training.raw_formula_values[-50:]).any()
            assert np.isfinite(full_training.raw_simple_execution_returns[-50:]).any()
            for name in (
                "formation_sessions",
                "holding_end_sessions",
                "ordered_listing_ids",
                "ordered_factor_ids",
                "absolute_state_factor_ids",
                "ordered_sector_ids",
                "sector_by_listing_id",
                "source_identity_hashes",
            ):
                assert getattr(full_training, name) == getattr(prefix_training, name)
            for name in (
                "raw_formula_values",
                "total_return_target_z",
                "raw_log_execution_returns",
                "raw_simple_execution_returns",
                "sector_context_values",
                "market_context_values",
                "reference_eligible",
            ):
                np.testing.assert_array_equal(
                    getattr(full_training, name), getattr(prefix_training, name)
                )
        else:
            assert full_training is prefix_training is None
    assert len(set(hashes)) == len(DAYS), "different cutoffs remain different snapshots"


def test_inference_selects_the_same_context_bytes_as_complete_training(
    input_workspace, input_features
):
    """requirement: selecting inference lanes preserves the full owner's saved observations."""
    training_ids = tuple(sorted(("gap", input_features[0].split("::")[1], "session_dollar_volume")))
    publisher = AlphaCurrentArtifactStore(input_workspace / "artifacts")
    request = dict(
        observed_at=OBSERVED_AT,
        expected_source_hash=workspace_score_source_identity(input_workspace),
        ordered_feature_ids=input_features,
    )
    for day in DAYS:
        complete, training = read_workspace_component_inputs(
            input_workspace, formation=day, training_factor_ids=training_ids, **request
        )
        inference, absent = read_workspace_component_inputs(
            input_workspace, formation=day, **request
        )
        assert training is not None and absent is None
        assert training.market_context_values.shape[1] == 17
        assert training.sector_context_values.shape[2] == 5
        for name in ("market_context_values", "sector_trend_values"):
            expected, actual = getattr(complete, name), getattr(inference, name)
            assert expected.dtype == actual.dtype == np.dtype(np.float64)
            assert expected.shape == actual.shape
            assert expected.tobytes(order="C") == actual.tobytes(order="C")
            # NumPy's persisted layout is Fortran only for an exclusively
            # Fortran-contiguous array; a strided Sector view saves in C order.
            assert (expected.flags.f_contiguous and not expected.flags.c_contiguous) == (
                actual.flags.f_contiguous and not actual.flags.c_contiguous
            )
        _same_source(complete, inference, publisher)


def test_prepared_history_is_immutable_and_projects_independent_callers(
    input_workspace, input_features
):
    """No consumer can change a held raw/Feature buffer or another formation's result."""
    source_hash = workspace_score_source_identity(input_workspace)
    prepared = prepare_workspace_component_inputs(
        input_workspace,
        through=DAYS[-1],
        observed_at=OBSERVED_AT,
        expected_source_hash=source_hash,
        ordered_feature_ids=input_features,
    )
    for arrays in (prepared.ohlcv, prepared.observations, prepared.formula_values):
        with pytest.raises(TypeError):
            arrays["another"] = prepared.raw_simple
        for values in arrays.values():
            assert not values.flags.writeable
            with pytest.raises(ValueError):
                values.setflags(write=True)
    with pytest.raises(ValueError):
        prepared.raw_simple.setflags(write=True)
    request = dict(
        formation=DAYS[0],
        observed_at=OBSERVED_AT,
        expected_source_hash=source_hash,
        ordered_feature_ids=input_features,
    )
    first = build_workspace_score_inputs(input_workspace, **request, prepared=prepared)
    baseline = build_workspace_score_inputs(input_workspace, **request)
    first.close[:] = 1.0
    first.market_context_values[:] = 2.0
    first.formula_values["gap"][:] = 3.0
    again = build_workspace_score_inputs(input_workspace, **request, prepared=prepared)
    publisher = AlphaCurrentArtifactStore(input_workspace / "artifacts")
    _same_source(baseline, again, publisher)
    # A successor authority can select another formula set. The unchanged full
    # owner supplies it, with the same publication and no extra prepared columns.
    extra = {
        **request,
        "ordered_feature_ids": ("RELATIVE_STOCK_CROSS_SECTION::close_to_close::identity",),
    }
    _same_source(
        build_workspace_score_inputs(input_workspace, **extra),
        build_workspace_score_inputs(input_workspace, **extra, prepared=prepared),
        publisher,
    )


def test_prepared_history_refuses_changed_source_without_a_revision_journal(input_workspace):
    """An unchanged coarse status cannot authorize bytes that were changed behind its journal."""
    source_hash = workspace_score_source_identity(input_workspace)
    prepared = prepare_workspace_component_inputs(
        input_workspace,
        through=DAYS[-1],
        observed_at=OBSERVED_AT,
        expected_source_hash=source_hash,
    )
    market = MarketDataRepository(input_workspace)
    # Deliberate source tamper through the public physical database port. Leave
    # payload and revision journals unchanged to expose status-only reuse.
    with market.database.connect(read_only=False) as connection:
        connection.execute(
            "UPDATE raw_daily_bar_current SET open = open + 0.01 "
            "WHERE listing_id = ? AND session_date = ?",
            [prepared.ordered_listing_ids[0], DAYS[0]],
        )
    assert workspace_score_source_identity(input_workspace) == source_hash
    with pytest.raises(ValueError, match=r"strategy_score.source_revision_changed"):
        build_workspace_score_inputs(
            input_workspace,
            formation=DAYS[0],
            observed_at=OBSERVED_AT,
            expected_source_hash=source_hash,
            prepared=prepared,
        )


def test_repeat_source_proofs_in_a_process_hash_no_unchanged_store_file(
    input_workspace, monkeypatch
):
    """regression: a process hashes a store file whole at its first proof and
    again only once its size or mtime moves (an edit is then refused, as the test above holds)."""
    source_hash = workspace_score_source_identity(input_workspace)
    prepared = prepare_workspace_component_inputs(
        input_workspace,
        through=DAYS[-1],
        observed_at=OBSERVED_AT,
        expected_source_hash=source_hash,
    )
    hashed: list[str] = []
    whole = strategy_score_inputs.file_digest

    def counted(handle, name):  # type: ignore[no-untyped-def]
        hashed.append(handle.name)
        return whole(handle, name)

    monkeypatch.setattr(strategy_score_inputs, "file_digest", counted)
    build_workspace_score_inputs(
        input_workspace,
        formation=DAYS[0],
        observed_at=OBSERVED_AT,
        expected_source_hash=source_hash,
        prepared=prepared,
    )
    assert hashed == []


def test_invalid_future_action_does_not_advance_an_earlier_formation_refusal(input_workspace):
    """A failed maximum capture leaves the ordinary earlier-cutoff admission intact."""
    market = MarketDataRepository(input_workspace)
    manifest = market.current_quality_filtered_research_manifest(
        market_profile_id="us-current-index-research"
    )
    assert manifest is not None
    listing = manifest.listings[0].listing_id
    bars = market.raw_bars(listing, start=DAYS[-1], through=DAYS[-1])
    assert bars
    with market.database.connect(read_only=False) as connection:
        connection.execute(
            "INSERT INTO corporate_action_current VALUES "
            "(?, ?, ?, 'CAPITAL_GAIN', NULL, NULL, TRUE, 'synthetic-future-action', "
            "?, 'ACTIVE', ?)",
            [listing, bars[0].provider, DAYS[-1], "a" * 64, OBSERVED_AT],
        )
    source_hash = workspace_score_source_identity(input_workspace)
    with pytest.raises(ValueError, match="unsupported corporate action"):
        prepare_workspace_component_inputs(
            input_workspace,
            through=DAYS[-1],
            observed_at=OBSERVED_AT,
            expected_source_hash=source_hash,
        )
    earlier = build_workspace_score_inputs(
        input_workspace,
        formation=DAYS[0],
        observed_at=OBSERVED_AT,
        expected_source_hash=source_hash,
    )
    assert earlier.formation_sessions[-1] == DAYS[0]
    with pytest.raises(ValueError, match="unsupported corporate action"):
        build_workspace_score_inputs(
            input_workspace,
            formation=DAYS[-1],
            observed_at=OBSERVED_AT,
            expected_source_hash=source_hash,
        )
