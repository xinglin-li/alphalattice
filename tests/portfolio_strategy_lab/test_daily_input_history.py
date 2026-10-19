"""A new Task reuses only a freshly proved, immutable workspace prefix."""

from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from alphalattice.control.product_host.composition.strategy_score_inputs import (
    PER_LISTING_EXTENSIONS,
    build_workspace_score_inputs,
    prepare_workspace_component_inputs,
    workspace_observation_history_columns,
    workspace_score_source_identity,
)
from alphalattice.control.product_host.composition.strategy_scoring import (
    workspace_observation_history_reuse,
)
from alphalattice.foundation.feature_engine.producers.factors.catalog import (
    default_extension_kernel_registry,
)
from alphalattice.foundation.feature_engine.producers.factors.session_liquidity import (
    session_liquidity_factor_specs,
)
from alphalattice.foundation.feature_engine.producers.factors.session_observation import (
    SESSION_OBSERVATION_FACTOR_IDS,
    session_observation_factor_specs,
)
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from alphalattice.investment.alpha_research.experiments.development_contracts import (
    seal_current_contract,
)
from alphalattice.investment.alpha_research.publication import artifacts
from alphalattice.investment.alpha_research.publication.artifacts import AlphaCurrentArtifactStore
from alphalattice.investment.alpha_research.publication.contracts import (
    FrozenScoreObservationSnapshot,
)
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


def _publish(store, prepared, reuse=None, **rebased):
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
        proofs=(prepared.axes_proof, prepared.listing_digests, prepared.listing_closed),
        **rebased,
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
    # The next head's proof, closed years carried or read, is a cold capture's.
    for name in ("dependency_prefix_hash", "axes_proof", "listing_digests", "listing_closed"):
        assert getattr(left, name) == getattr(right, name)


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


@pytest.mark.parametrize(
    "correction",
    ["bar", "feature", "two", "action", "closed_year", "sector", "cross_sectional"],
)
def test_an_old_value_correction_rebuilds_what_it_touched_even_without_a_revision(
    daily_workspace, correction, monkeypatch
):
    """A corrected bar, Feature, action or closed-year bar rebuilds its own listings alone, with
    Features and formula columns kept; a Sector change moves every listing's axes, and a
    cross-sectional kernel keeps the capture cold, whole."""
    from alphalattice.control.product_host.composition import strategy_score_inputs

    if correction == "cross_sectional":  # a kept kernel no longer proved per listing
        monkeypatch.setattr(strategy_score_inputs, "PER_LISTING_EXTENSIONS", frozenset())
    store = AlphaCurrentArtifactStore(daily_workspace / "artifacts")
    previous = _publish(store, _capture(daily_workspace, FIRST, WIDE))
    history = store.load_workspace_observation_history("1" * 64)
    assert history is not None
    market = MarketDataRepository(daily_workspace)
    ids = previous.ordered_listing_ids
    closed = max(day for day in previous.formation_sessions if day.year < FIRST.year)
    old_identity = workspace_score_source_identity(daily_workspace)
    with market.database.connect(read_only=False) as connection:
        bar = (
            "UPDATE raw_daily_bar_current SET open = open + 0.01 "
            "WHERE listing_id = ? AND session_date = ?"
        )
        values = (
            "UPDATE feature_daily_current SET dist_52w_high = dist_52w_high + 0.01, "
            "mom_252_21 = mom_252_21 + 0.01 WHERE listing_id = ? AND session_date = ?"
        )
        touched = {
            "bar": [(bar, ids[1], FIRST)],
            "cross_sectional": [(bar, ids[1], FIRST)],
            "feature": [(values, ids[1], FIRST)],
            "two": [(bar, ids[1], FIRST), (values, ids[2], FIRST)],
            "closed_year": [(bar, ids[1], closed)],
        }.get(correction, [])
        for statement, listing, day in touched:
            connection.execute(statement, [listing, day])
        if correction == "action":  # a dividend found late, inside the kept history
            connection.execute(
                "INSERT INTO corporate_action_current SELECT listing_id, provider, session_date, "
                "'CASH_DIVIDEND', NULL, 0.25, false, 'fixture', repeat('a', 64), 'ACTIVE', ? "
                "FROM raw_daily_bar_current WHERE listing_id = ? AND session_date = ?",
                [OBSERVED_AT.replace(tzinfo=None), ids[3], previous.formation_sessions[-2]],
            )
        elif correction == "sector":
            connection.execute(
                "UPDATE sector_classification_current SET sector_name = "
                "CASE WHEN sector_name = 'Energy' THEN 'Technology' ELSE 'Energy' END "
                "WHERE listing_id = ?",
                [ids[1]],
            )
    # Each edit but the inserted action leaves the source identity as it was (no revision).
    assert (workspace_score_source_identity(daily_workspace) == old_identity) != (
        correction == "action"
    )
    actual = _capture(daily_workspace, LAST, WIDE, history=history)
    expected = _capture(daily_workspace, LAST, WIDE)
    assert actual.reused_history_hash is None
    restated = {
        "bar": (ids[1],),
        "feature": (ids[1],),
        "two": (ids[1], ids[2]),
        "action": (ids[3],),
        "closed_year": (ids[1],),
    }.get(correction, ())
    assert actual.restated_listing_ids == restated
    _same_preparation(expected, actual)


KERNELS = (*session_observation_factor_specs(), *session_liquidity_factor_specs())


@pytest.mark.parametrize("spec", KERNELS, ids=lambda spec: spec.factor_id)
def test_a_reviewed_per_listing_kernel_gives_a_listing_alone_its_full_frame_bytes(spec):
    """The reviewed set is every kernel the capture computes, and each one computes a listing
    subset exactly as within the whole frame; a new kernel fails here until reviewed."""
    reviewed = {
        *("at_own_high_share_63", "close_to_close", "gap", "gap_amplitude", "high_extension"),
        *("intraday", "intraday_amplitude", "low_extension", "range_position", "session_span"),
        *("return_run_length_21", "session_dollar_volume", "sessions_since_252_high"),
        *(
            "previous_close_excursion_intraday_adjusted_square",
            "previous_close_excursion_intraday_cross",
        ),
        "previous_close_excursion_scaled_span_square",
    }
    assert {kernel.factor_id for kernel in KERNELS} == PER_LISTING_EXTENSIONS == reviewed
    days, names = 300, ("A", "B", "C")
    close = 100.0 + np.sin(np.arange(days * 3, dtype=np.float64) / 7.0).reshape(days, 3)
    close[[60, 250], 1] = (200.0, 220.0)
    frame = pd.DataFrame(
        {
            "session_date": np.repeat([date(2024, 1, 1) + timedelta(i) for i in range(days)], 3),
            "listing_id": np.tile(names, days),
            "open_split_adjusted": (close * 0.99).reshape(-1),
            "high_split_adjusted": (close * 1.02).reshape(-1),
            "low_split_adjusted": (close * 0.98).reshape(-1),
            "close_split_adjusted": close.reshape(-1),
            "close_raw": close.reshape(-1),
            "volume_raw": np.tile((1000.0, 1500.0, 900.0), days),
        }
    )
    compute = default_extension_kernel_registry().compute
    whole, mask = compute(frame, spec), (frame["listing_id"] == "B").to_numpy()
    assert compute(frame[mask], spec).to_numpy().tobytes() == whole[mask].to_numpy().tobytes()


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


def test_a_renewing_component_keeps_its_training_columns_on_a_day_it_does_not_renew(monkeypatch):
    """regression (the person's D2, 2026-10-15): a kept history held only that day's columns, so
    the renewal day rebuilt ten years cold; every capture now keeps the training columns."""
    from contextlib import nullcontext

    from alphalattice.control.product_host.composition import strategy_scoring as scoring
    from alphalattice.investment.alpha_research.scores.model_renewal import (
        AdmittedRenewingInference as Renewing,
    )

    inference, captured = object.__new__(Renewing), {}
    inference.authority = SimpleNamespace(
        training_factor_ids=("amihud_21",), supports=lambda _: True, model_set=SimpleNamespace()
    )
    inference.authority.model_set.ordered_feature_ids = ()

    def capture(_workspace, **kwargs):
        captured.update(kwargs)
        raise ValueError("causal execution split is invalid")  # the owner's own early return

    monkeypatch.setattr(scoring, "prepare_workspace_component_inputs", capture)
    owner = SimpleNamespace(
        _authority=lambda _: inference,
        store=SimpleNamespace(load_workspace_observation_history=lambda _: None),
        session=SimpleNamespace(reads=nullcontext, workspace=None),
        clock=lambda: OBSERVED_AT,
    )
    binding = SimpleNamespace(
        strategy_package_id="p", component_id="c", source_kind="WORKSPACE_DATA_FEATURE"
    )
    binding.strategy_package_hash = "3" * 64
    prepare = scoring.StrategyScoringApplication.prepare_inputs
    assert prepare(owner, binding, through=LAST, expected_source_hash="4" * 64) is None
    assert captured["training_factor_ids"] == ("amihud_21",)


def test_a_reference_adjusted_series_restated_keeps_the_kept_history_warm(daily_workspace):
    """regression (FINAL-022's real D2, 2026-10-15): the provider restated every adjusted series
    and the kept prefix went cold, though no captured value reads the reference's closes."""
    store = AlphaCurrentArtifactStore(daily_workspace / "artifacts")
    previous = _publish(store, _capture(daily_workspace, FIRST, WIDE[:1]))
    market = MarketDataRepository(daily_workspace)
    with market.database.transaction() as connection:
        connection.execute(
            "UPDATE provider_adjusted_close_current SET adjusted_close = adjusted_close * 1.001 "
            "WHERE listing_id IN (SELECT listing_id FROM market_reference_current)"
        )
    history = store.load_workspace_observation_history("1" * 64)
    appended = _capture(daily_workspace, LAST, WIDE[:1], history)
    assert appended.reused_history_hash == previous.head_hash
    _same_preparation(_capture(daily_workspace, LAST, WIDE[:1]), appended)


def test_a_head_without_listing_proofs_keeps_its_seal_and_reuses_whole(daily_workspace):
    """A head written without the optional proof parts is sealed as before (its JSON without
    them), verifies on load and reuses whole when its prefix proof holds. A head from before the
    proof's v3 kind never holds it, so the first update after upgrading is cold once."""
    from alphalattice.kernel.shared_kernel.identity import canonical_hash

    store = AlphaCurrentArtifactStore(daily_workspace / "artifacts")
    prepared = _capture(daily_workspace, FIRST, WIDE[:1])
    head = store.publish_workspace_observation_history(
        scope_hash="1" * 64,
        selection_hash="2" * 64,
        dependency_prefix_hash=prepared.dependency_prefix_hash,
        formation_sessions=prepared.formation_sessions,
        ordered_listing_ids=prepared.ordered_listing_ids,
        stable_session_count=prepared.stable_session_count,
        columns=workspace_observation_history_columns(prepared),
        capacity=lambda count: None,
    )
    apart = {"head_hash", "axes_proof", "listing_digests", "listing_closed"}
    assert head.head_hash == canonical_hash(head.model_dump(mode="json", exclude=apart))
    history = store.load_workspace_observation_history("1" * 64)
    assert history is not None and history[0] == head
    appended = _capture(daily_workspace, LAST, WIDE[:1], history)
    assert appended.reused_history_hash == head.head_hash and not appended.restated_listing_ids


def _legacy(store, snapshot):
    """The record the release before blocks wrote: one whole npz and no block list."""
    record = snapshot.model_dump(mode="json", exclude={"snapshot_hash", "array_blocks"})
    legacy = seal_current_contract(
        FrozenScoreObservationSnapshot,
        {**record, "array_content_hash": snapshot.array_files[0]},
        "snapshot_hash",
    )
    store._publish("frozen-observation-snapshots", legacy, "snapshot_hash")
    return legacy


@pytest.mark.parametrize("upgraded", [False, True])
def test_three_consecutive_days_write_only_their_last_frozen_block(
    daily_workspace, monkeypatch, upgraded
):
    store = AlphaCurrentArtifactStore(daily_workspace / "artifacts")
    folder = store.root / "current/frozen-observation-arrays"
    days = (date(2026, 7, 29), date(2026, 7, 30), date(2026, 7, 31))
    written = []
    for day in days:
        before = set(folder.glob("*.bin")) if folder.exists() else set()
        source = build_workspace_score_inputs(
            daily_workspace,
            formation=day,
            observed_at=OBSERVED_AT,
            expected_source_hash=workspace_score_source_identity(daily_workspace),
            ordered_feature_ids=(),
        )
        # Upgraded: the first day's snapshot is a stored whole-history file, as before blocks.
        legacy = upgraded and day == days[0]
        with monkeypatch.context() as patch:
            if legacy:
                patch.setattr(artifacts, "FROZEN_BLOCK_SESSIONS", len(source.formation_sessions))
            snapshot = store.publish_frozen_observations(source, disposition="RECORDED_INPUT_QA")
        if legacy:
            snapshot = _legacy(store, snapshot)
            assert snapshot.array_blocks is None and "array_blocks" not in snapshot.model_dump()
            store.verify_frozen_observations(snapshot.snapshot_hash)
        written.append(set(folder.glob("*.bin")) - before)
        loaded = store.load_frozen_observations(snapshot.snapshot_hash)
        for name in ("open", "close", "volume", "market_context_values", "sector_trend_values"):
            assert getattr(loaded, name).tobytes() == getattr(source, name).tobytes()
    # The first day of blocks (the upgrade's first day, after a stored whole file) writes every
    # block once; each later day only its last, the rest shared: about one block's bytes.
    first = int(upgraded)
    size = [sum(path.stat().st_size for path in w) for w in written]
    assert len(written[first]) == len(snapshot.array_files)
    assert [len(w) for w in written[first + 1 :]] == [1] * (2 - first)
    assert max(size[first + 1 :]) * len(snapshot.array_files) <= 2 * size[first]


def test_a_split_reaches_the_kept_history_the_frozen_read_and_the_next_day_as_stored(
    daily_workspace,
):
    """requirement: splits rebased in the store, a 3-for-1 then a 1-for-2 composed, read through
    the kept history, the frozen inputs and the next daily as the store's bars; each formula
    agrees with a cold capture."""
    from alphalattice.control.data_platform.data_decision_rule import DataDecisionRule
    from alphalattice.foundation.market_data_ops.storage.rebase import RawBarRebase, record

    store = AlphaCurrentArtifactStore(daily_workspace / "artifacts")
    first = _capture(daily_workspace, FIRST, WIDE)
    _publish(store, first)
    listing, receipts = first.ordered_listing_ids[1], []

    def split(ex_date: date, ratio: float) -> None:  # as the store rebases: then its row
        price, at = 1 / ratio, OBSERVED_AT.replace(tzinfo=None) + timedelta(days=len(receipts))
        with MarketDataRepository(daily_workspace).database.connect(read_only=False) as db:
            provider, start = db.execute(
                "SELECT provider, min(session_date) FROM raw_daily_bar_current"
                " WHERE listing_id = ? GROUP BY provider",
                [listing],
            ).fetchone()
            db.execute(
                "UPDATE raw_daily_bar_current SET open = open * ?, high = high * ?, low = low * ?,"
                " close = close * ?, volume = floor(volume * ? + 0.5)"
                " WHERE listing_id = ? AND session_date < ?",
                [price] * 4 + [ratio, listing, ex_date],
            )
            db.execute(
                "UPDATE provider_adjusted_close_current SET adjusted_close = adjusted_close * ?"
                " WHERE listing_id = ? AND session_date < ?",
                [price, listing, ex_date],
            )
            last = ex_date - timedelta(days=1)
            rebase = RawBarRebase(listing, provider, ex_date, start, last, price, ratio, 0.0, at)
            record(db, [rebase], "fixture")
        receipts.append(rebase.receipt_hash)

    for day, ratio in ((date(2026, 7, 30), 3.0), (LAST, 0.5)):  # the split day, the next daily
        split(day, ratio)
        history = store.load_workspace_observation_history("1" * 64)
        prepared = _capture(daily_workspace, day, WIDE, history)
        assert [item.receipt_hash for item in prepared.rebases] == receipts  # in store order
        # No research manifest declares a rule here: the shipped rule kept the rows, by name.
        assert prepared.rebases[-1].kept_rule_hash == DataDecisionRule().rule_hash
        _publish(store, prepared, rebases=prepared.rebases, kept_rows=prepared.kept_rows)
        request = dict(
            formation=day,
            observed_at=OBSERVED_AT,
            expected_source_hash=workspace_score_source_identity(daily_workspace),
            ordered_feature_ids=WIDE,
        )
        cold = build_workspace_score_inputs(daily_workspace, **request)
        kept = build_workspace_score_inputs(daily_workspace, **request, prepared=prepared)
        snapshot = store.publish_frozen_observations(kept, disposition="RECORDED_INPUT_QA")
        read = store.load_frozen_observations(snapshot.snapshot_hash)
        for name in ("open", "high", "low", "close", "volume"):
            assert getattr(read, name).tobytes() == getattr(cold, name).tobytes()  # as stored
            with pytest.raises(ValueError):
                getattr(read, name).setflags(write=True)
        for name, values in cold.formula_values.items():
            np.testing.assert_allclose(read.formula_values[name], values, rtol=1e-6, atol=1e-12)
