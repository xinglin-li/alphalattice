"""Properties that make the desktop feature policy economically bounded."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, seed, settings
from hypothesis import strategies as st

from alphalattice.foundation.feature_engine.catalog.contracts import FeatureCatalog
from alphalattice.foundation.feature_engine.contracts import (
    FactorSessionInvalidation,
    FactorSessionInvalidationPlan,
    FeatureBuildRequest,
    FeatureCatalogBinding,
    FeatureInvalidation,
    PanelInvalidationRange,
)
from alphalattice.foundation.feature_engine.producers.base_materializer import (
    BaseFeatureMaterializer,
)
from alphalattice.foundation.feature_engine.producers.preprocessing.robust_cross_section import (
    SOURCE_ELIGIBILITY_POLICY_HASH,
)
from alphalattice.foundation.feature_engine.runtime.factor_invalidation import (
    compile_listing_plan,
    compile_panel_plan,
    expand_ranges,
    restrict_plan_to_sessions,
    targets_by_session,
)
from alphalattice.foundation.feature_engine.runtime.service import (
    FeatureFoundationService,
)
from alphalattice.foundation.market_data_ops.storage.duckdb import (
    MarketDataRepository,
)
from alphalattice.kernel.data.calendar import materialize_calendar_schedule

CASE_ROOT = Path(__file__).resolve().parent
PLAYPEN_ROOT = CASE_ROOT.parents[1]


def _frame(*, periods: int = 560, start: str = "2023-01-03", seed_value: int = 7):
    dates = pd.bdate_range(start, periods=periods)
    rng = np.random.default_rng(seed_value)
    log_returns = rng.normal(0.0003, 0.012, periods)
    close = 80.0 * np.exp(np.cumsum(log_returns))
    spread = 0.008 + rng.uniform(0.0, 0.005, periods)
    open_ = close * np.exp(rng.normal(0.0, 0.003, periods))
    high = np.maximum(open_, close) * (1.0 + spread)
    low = np.minimum(open_, close) * (1.0 - spread)
    volume = rng.integers(500_000, 4_000_000, periods).astype(float)
    return pd.DataFrame(
        {
            "session_date": dates,
            "open_raw": open_,
            "high_raw": high,
            "low_raw": low,
            "close_raw": close,
            "volume_raw": volume,
            "open_split_adjusted": open_,
            "high_split_adjusted": high,
            "low_split_adjusted": low,
            "close_split_adjusted": close,
            "provider_adjusted_close": close,
            "close_total_return_adjusted": close,
        }
    )


def _values(asset: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    return (
        BaseFeatureMaterializer(FeatureCatalog.load())
        .materialize_listing(listing_id="fixture", projected_bars=asset, market_bars=market)
        .values
    )


def test_missing_interior_session_keeps_calendar_windows_and_preceding_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from alphalattice.foundation.feature_engine.panels.materialization_identity import (
        align_feature_source_sessions,
        feature_source_values_hash,
    )
    from alphalattice.foundation.feature_engine.runtime import service as feature_service
    from alphalattice.foundation.market_data_ops.sources.price_integrity import (
        build_trading_session_authority,
    )

    asset, market = _frame(), _frame(seed_value=13)
    sessions = tuple(asset.session_date.dt.date)
    sparse = asset.drop(index=400)
    calls = []

    def calendar(*args, **kwargs):
        calls.append((args, kwargs))
        return build_trading_session_authority(
            calendar_ids=kwargs["calendar_ids"], sessions=sessions
        )

    owner = object.__new__(FeatureFoundationService)
    owner.panel = SimpleNamespace(policy_hash=SOURCE_ELIGIBILITY_POLICY_HASH)
    owner.catalog = FeatureCatalog.load()  # reads no as-traded field, so asks for none (V395)
    owner.session_authority_resolver = calendar
    request = SimpleNamespace(history_start=sessions[0], as_of_session=sessions[-1])
    held = {}
    kwargs = dict(
        calendar_id="XNYS",
        history_start=request.history_start,
        as_of_session=request.as_of_session,
        observed_at=datetime.now(UTC),
        calendars=held,
    )
    actual_axis = owner._base_session_calendar(
        observed_sessions=tuple(sparse.session_date.dt.date), **kwargs
    )
    assert actual_axis == sessions
    assert owner._base_session_calendar(observed_sessions=sessions[10:], **kwargs) == sessions[10:]
    assert len(calls) == 1
    assert align_feature_source_sessions(asset, sessions) is asset
    aligned = align_feature_source_sessions(sparse, actual_axis)
    # The writer reads the listing's stored inputs, asking for no as-traded field (V395),
    # and the frame is projected from them and aligned to the calendar (V92).
    source = SimpleNamespace(raw_hash="a" * 64, action_hash="b" * 64)
    owner.feature_state = SimpleNamespace(
        feature_source_inputs=lambda *args, **kwargs: None if kwargs["as_traded"] else source
    )
    monkeypatch.setattr(
        feature_service,
        "projected_feature_rows",
        lambda inputs: sparse.to_dict("records") if inputs is source else [],
    )
    delivered, raw_hash, actions_hash = owner._base_source_frame(
        None,
        listing_id="fixture",
        through=sessions[-1],
        start=sessions[0],
        calendar=actual_axis,
        _connection=None,
    )
    pd.testing.assert_frame_equal(delivered, aligned, check_exact=True)
    assert (raw_hash, actions_hash) == ("a" * 64, "b" * 64)
    assert aligned.iloc[400].drop("session_date").isna().all()
    before, after = _values(asset, market), _values(aligned, market)
    pd.testing.assert_frame_equal(before.iloc[:400], after.iloc[:400], check_exact=True)
    # The missing date is newer than the skip-month momentum's last input.
    assert after.loc[410, "mom_126_21"] == before.loc[410, "mom_126_21"]
    assert pd.isna(after.loc[410, "beta_63"])
    compressed = _values(sparse, market).set_index("session_date")
    assert pd.notna(compressed.loc[sessions[410], "beta_63"])
    assert compressed.loc[sessions[410], "mom_126_21"] != after.loc[410, "mom_126_21"]
    assert feature_source_values_hash(
        sparse, ("provider_adjusted_close",)
    ) != feature_source_values_hash(aligned, ("provider_adjusted_close",))
    owner.session_authority_resolver = lambda **_: build_trading_session_authority(
        calendar_ids=("XNAS",), sessions=sessions
    )
    with pytest.raises(ValueError, match="source_window_calendar_invalid"):
        owner._base_session_calendar(observed_sessions=sessions, **{**kwargs, "calendars": {}})
    owner.panel = SimpleNamespace(policy_hash="legacy")
    assert owner._base_session_calendar(
        observed_sessions=tuple(sparse.session_date.dt.date), **kwargs
    ) == tuple(sparse.session_date.dt.date)


@seed(20260803)
@settings(max_examples=8, deadline=None)
@given(st.floats(min_value=0.05, max_value=20.0, allow_nan=False, allow_infinity=False))
def test_uniform_adjusted_scale_does_not_change_adjusted_return_features(scale: float) -> None:
    catalog = FeatureCatalog.load()
    asset = _frame(seed_value=11)
    market = _frame(seed_value=13)
    scaled_asset = asset.copy()
    scaled_market = market.copy()
    scaled_asset["provider_adjusted_close"] *= scale
    scaled_market["provider_adjusted_close"] *= scale * 1.7
    before = _values(asset, market)
    after = _values(scaled_asset, scaled_market)
    factors = tuple(
        contract.factor_id
        for contract in catalog.maintenance_contracts
        if "provider_adjusted_close" in contract.required_fields
    )
    np.testing.assert_allclose(
        before.loc[320:, factors], after.loc[320:, factors], rtol=1e-11, atol=1e-12
    )


@seed(20260803)
@settings(max_examples=8, deadline=None)
@given(st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False))
def test_share_basis_rebase_preserves_price_volume_semantics(scale: float) -> None:
    catalog = FeatureCatalog.load()
    asset = _frame(seed_value=17)
    market = _frame(seed_value=19)
    rebased = asset.copy()
    price_columns = (
        "open_raw",
        "high_raw",
        "low_raw",
        "close_raw",
        "open_split_adjusted",
        "high_split_adjusted",
        "low_split_adjusted",
        "close_split_adjusted",
        "provider_adjusted_close",
        "close_total_return_adjusted",
    )
    rebased.loc[:, price_columns] *= scale
    rebased["volume_raw"] /= scale
    before = _values(asset, market)
    after = _values(rebased, market)
    factors = tuple(
        contract.factor_id
        for contract in catalog.maintenance_contracts
        if contract.adjustment_transformation == "SPLIT_REBASE_OHLC_X_K_VOLUME_DIV_K"
    )
    np.testing.assert_allclose(
        before.loc[320:, factors], after.loc[320:, factors], rtol=1e-9, atol=1e-11
    )


def test_prefix_warmup_and_independent_formula_oracles() -> None:
    asset = _frame(seed_value=23)
    market = _frame(seed_value=29)
    values = _values(asset, market)
    position = len(values) - 1
    asset_return = np.log(asset["provider_adjusted_close"]).diff().to_numpy()
    market_return = np.log(market["provider_adjusted_close"]).diff().to_numpy()

    # Every window below ends at the observation session itself; only
    # ``residual_mom_252_21`` keeps a declared economic skip.
    r14 = asset_return[position - 13 : position + 1]
    average_gain = np.maximum(r14, 0.0).mean()
    average_loss = np.maximum(-r14, 0.0).mean()
    cutler = 100.0 if average_loss == 0.0 else 100.0 - 100.0 / (1.0 + average_gain / average_loss)
    assert np.isclose(values.iloc[position]["cutler_rsi_14"], cutler, rtol=1e-12)

    r63 = asset_return[position - 62 : position + 1]
    m63 = market_return[position - 62 : position + 1]
    beta = np.dot(r63 - r63.mean(), m63 - m63.mean()) / np.dot(m63 - m63.mean(), m63 - m63.mean())
    assert np.isclose(values.iloc[position]["beta_63"], beta, rtol=1e-12)
    assert np.isclose(
        values.iloc[position]["rev_21"],
        -asset_return[position - 20 : position + 1].sum(),
        rtol=1e-12,
    )

    estimation_asset = asset_return[position - 21 - 252 + 1 : position - 21 + 1]
    estimation_market = market_return[position - 21 - 252 + 1 : position - 21 + 1]
    centered_asset = estimation_asset - estimation_asset.mean()
    centered_market = estimation_market - estimation_market.mean()
    residual_beta = np.sum(centered_asset * centered_market) / np.sum(centered_market**2)
    residual_oracle = np.sum((centered_asset - residual_beta * centered_market)[-231:])
    assert np.isclose(
        values.iloc[position]["residual_mom_252_21"],
        residual_oracle,
        rtol=1e-14,
        atol=1e-15,
    )
    assert np.nanmedian(np.abs(values["residual_mom_252_21"])) > 1e-12

    high = asset["high_split_adjusted"].to_numpy()
    low = asset["low_split_adjusted"].to_numpy()
    close = asset["close_split_adjusted"].to_numpy()
    true_range = np.maximum.reduce(
        (
            high - low,
            np.abs(high - np.r_[np.nan, close[:-1]]),
            np.abs(low - np.r_[np.nan, close[:-1]]),
        )
    )
    up = np.r_[np.nan, np.diff(high)]
    down = np.r_[np.nan, -np.diff(low)]
    plus_dm = np.where((up > down) & (up > 0.0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0.0), down, 0.0)
    dx = np.full(len(close), np.nan)
    for end in range(13, len(close)):
        sl = slice(end - 13, end + 1)
        atr = np.mean(true_range[sl])
        plus_di = 100.0 * np.mean(plus_dm[sl]) / atr
        minus_di = 100.0 * np.mean(minus_dm[sl]) / atr
        if plus_di + minus_di:
            dx[end] = 100.0 * abs(plus_di - minus_di) / (plus_di + minus_di)
    directional = np.mean(dx[position - 13 : position + 1])
    assert np.isclose(values.iloc[position]["directional_strength_14"], directional, rtol=1e-12)

    prefix_asset = _frame(periods=300, start="2021-11-09", seed_value=31)
    prefix_market = _frame(periods=300, start="2021-11-09", seed_value=37)
    prefixed_asset = pd.concat([prefix_asset, asset], ignore_index=True)
    prefixed_market = pd.concat([prefix_market, market], ignore_index=True)
    prefixed = _values(prefixed_asset, prefixed_market).iloc[-len(asset) :].reset_index(drop=True)
    np.testing.assert_allclose(
        values.loc[320:, FeatureCatalog.load().factor_ids],
        prefixed.loc[320:, FeatureCatalog.load().factor_ids],
        rtol=1e-9,
        atol=1e-11,
        equal_nan=True,
    )


def test_catalog_has_no_exact_sign_duplicate_factor_columns() -> None:
    values = _values(_frame(seed_value=101), _frame(seed_value=103))
    factor_ids = FeatureCatalog.load().factor_ids
    for left_position, left_id in enumerate(factor_ids):
        left = values[left_id].to_numpy(dtype=float)
        for right_id in factor_ids[left_position + 1 :]:
            right = values[right_id].to_numpy(dtype=float)
            assert not np.array_equal(left, -right, equal_nan=True), (
                f"exact sign-duplicate factors: {left_id}, {right_id}"
            )


def test_window_local_materialization_is_bitwise_prefix_independent() -> None:
    catalog = FeatureCatalog.load()
    asset = _frame(periods=620, start="2021-01-04", seed_value=61)
    market = _frame(periods=620, start="2021-01-04", seed_value=67)
    calendar = tuple(pd.to_datetime(asset["session_date"]).dt.date)
    first_target = calendar[-80]
    warmup = max(
        item.lookback_sessions + item.formula_skip_sessions + 2
        for item in catalog.maintenance_contracts
    )
    input_start = FeatureFoundationService._bounded_input_start(
        calendar=calendar,
        first_target=first_target,
        warmup_sessions=warmup,
    )
    start_position = calendar.index(input_start)
    full = _values(asset, market)
    bounded = _values(
        asset.iloc[start_position:].reset_index(drop=True),
        market.iloc[start_position:].reset_index(drop=True),
    )
    full_target = full.loc[full["session_date"] >= first_target, list(catalog.factor_ids)].to_numpy(
        dtype=float
    )
    bounded_target = bounded.loc[
        bounded["session_date"] >= first_target, list(catalog.factor_ids)
    ].to_numpy(dtype=float)
    np.testing.assert_array_equal(full_target, bounded_target)

    sessions = pd.to_datetime(asset["session_date"])
    session_labels = sessions.dt.strftime("%Y-%m-%d").to_numpy()
    # The calendar-selected Formula reads the final close of the month twelve
    # back, so a session offset cannot express its cutoff and the harness has to
    # resolve it the same way the Formula does.
    monthly_last_position = (
        pd.DataFrame({"period": sessions.dt.to_period("M"), "position": range(len(sessions))})
        .groupby("period")["position"]
        .last()
    )
    calendar_ids = catalog.calendar_factor_ids
    assert calendar_ids == frozenset({"seasonality_12m"})

    def cutoff(spec, position: int) -> str | None:
        if spec.factor_id in calendar_ids:
            selected = monthly_last_position.reindex(
                [sessions.dt.to_period("M").iloc[position] - 12]
            ).to_numpy(dtype=float)[0]
            return None if not np.isfinite(selected) else session_labels[int(selected)]
        skip = spec.lag_sessions
        return session_labels[position - skip] if position >= skip else None

    for position in (0, 1, 20, 21, len(full) - 1):
        expected = json.dumps(
            {spec.factor_id: cutoff(spec, position) for spec in catalog.factors},
            sort_keys=True,
            separators=(",", ":"),
        )
        assert full.iloc[position]["input_cutoffs_json"] == expected


def test_catalog_identity_binds_independent_semantics() -> None:
    catalog = FeatureCatalog.load()
    binding = catalog.binding

    def rebind(**overrides: str) -> FeatureCatalogBinding:
        fields = {
            "stable_id": binding.stable_id,
            "catalog_content_hash": binding.catalog_content_hash,
            "formula_implementation_hash": binding.formula_implementation_hash,
            "materializer_policy_hash": binding.materializer_policy_hash,
            "formula_observation_policy_hash": binding.formula_observation_policy_hash,
            "source_availability_policy_hash": binding.source_availability_policy_hash,
            "source_authority_binding_hash": binding.source_authority_binding_hash,
        }
        return FeatureCatalogBinding.create(**{**fields, **overrides})

    assert rebind().catalog_hash == binding.catalog_hash
    assert rebind(catalog_content_hash="f" * 64).catalog_hash != binding.catalog_hash
    # The two clock authorities move the catalog independently: a Formula clock
    # change and a Provider publication change are different events and must not
    # be indistinguishable in the identity a Panel quotes.
    assert rebind(formula_observation_policy_hash="f" * 64).catalog_hash != binding.catalog_hash
    assert rebind(source_availability_policy_hash="e" * 64).catalog_hash != binding.catalog_hash
    assert (
        rebind(formula_observation_policy_hash="f" * 64).catalog_hash
        != rebind(source_availability_policy_hash="f" * 64).catalog_hash
    )
    # And a third, because "when a source publishes" and "which sources a Formula
    # reads" are also different events. A Provider moving its publication time
    # and a Formula gaining a Sector dependency must not produce the same catalog.
    assert rebind(source_authority_binding_hash="d" * 64).catalog_hash != binding.catalog_hash
    assert (
        rebind(source_availability_policy_hash="f" * 64).catalog_hash
        != rebind(source_authority_binding_hash="f" * 64).catalog_hash
    )
    residual = next(
        factor for factor in catalog.factors if factor.factor_id == "residual_mom_252_21"
    )
    assert residual.formula_ref == "factor.desktop.residual_mom_252_21"
    # The 21-session economic skip survives the observation-clock successor, so
    # the row requirement of the one Formula that declares the largest one does
    # not move either.
    assert residual.lag_sessions == 21
    assert residual.minimum_observations == 274


def test_return_delta_and_factor_specific_sparse_ranges() -> None:
    dates = tuple(pd.bdate_range("2024-01-02", periods=360).date)
    prior = {session: 100.0 + index for index, session in enumerate(dates[:-1])}
    appended = {**prior, dates[-1]: 100.0 + len(dates) - 1}
    assert MarketDataRepository._provider_adjusted_return_changes(prior, appended) == (dates[-1],)
    scaled = {session: value * 0.37 for session, value in prior.items()}
    assert MarketDataRepository._provider_adjusted_return_changes(prior, scaled) == ()
    corrected = dict(prior)
    corrected[dates[120]] *= 1.01
    assert MarketDataRepository._provider_adjusted_return_changes(prior, corrected) == (
        dates[120],
        dates[121],
    )

    catalog = FeatureCatalog.load()
    request = FeatureBuildRequest.create(
        manifest_revision="a" * 64,
        catalog=catalog.binding,
        spy_revision="b" * 64,
        history_start=dates[0],
        as_of_session=dates[-1],
    )
    plan = compile_listing_plan(
        catalog=catalog,
        request=request,
        listing_id="listing-a",
        sessions=dates,
        invalidations=(
            FeatureInvalidation(
                "adjusted_return_correction",
                listing_id="listing-a",
                affected_sessions=(dates[120],),
                source_fields=("provider_adjusted_close",),
                factor_ids=("rev_21", "rev_63"),
            ),
        ),
    )
    by_factor = {item.factor_id: expand_ranges(item.ranges, dates) for item in plan.items}
    # A Formula with no economic skip consumes the corrected session itself, so
    # invalidation now starts at that session rather than one after it.
    assert by_factor["rev_21"] == dates[120:142]
    assert by_factor["rev_63"] == dates[120:184]
    clipped = compile_listing_plan(
        catalog=catalog,
        request=request,
        listing_id="listing-a",
        sessions=dates[:-3],
        invalidations=(
            FeatureInvalidation(
                "ohlc_correction",
                listing_id="listing-a",
                affected_sessions=(dates[-3], dates[-2], dates[-1]),
                source_fields=("high_split_adjusted",),
            ),
        ),
    )
    assert clipped.items == ()
    action_only = compile_listing_plan(
        catalog=catalog,
        request=request,
        listing_id="listing-a",
        sessions=dates,
        invalidations=(FeatureInvalidation("action_evidence_change", listing_id="listing-a"),),
    )
    assert action_only.items == ()


def test_panel_binding_change_rebuilds_panel_without_base_feature_work() -> None:
    """A manifest revision reaches no base row and, over a compatible base, no Panel row.

    Under the session-cross-section rule a governance-only revision is a new
    relationship over the same rows: the composition reuses every partition
    whose cross-sections it still computes. Only without a compatible base
    does the binding change recompute the whole Panel.
    """

    sessions = tuple(pd.bdate_range("2026-07-01", periods=20).date)
    catalog = FeatureCatalog.load()
    invalidation = FeatureInvalidation(
        "panel_binding_change",
        earliest_session=sessions[0],
    )
    request = FeatureBuildRequest.create(
        manifest_revision="a" * 64,
        catalog=catalog.binding,
        spy_revision="b" * 64,
        history_start=sessions[0],
        as_of_session=sessions[-1],
        invalidations=(invalidation,),
    )

    listing_plan = compile_listing_plan(
        catalog=catalog,
        request=request,
        invalidations=request.invalidations,
        listing_id="listing-a",
        sessions=sessions,
    )
    panel_plan = compile_panel_plan(
        catalog=catalog,
        request=request,
        invalidations=request.invalidations,
        listing_plans=(listing_plan,),
        sessions=sessions,
    )

    assert listing_plan.items == ()
    assert panel_plan.items == ()
    rebuilt = compile_panel_plan(
        catalog=catalog,
        request=request,
        invalidations=request.invalidations,
        listing_plans=(listing_plan,),
        sessions=sessions,
        base_reusable=False,
    )
    assert rebuilt.factor_ids == catalog.factor_ids
    expected_range = (PanelInvalidationRange(sessions[0], sessions[-1]),)
    assert all(item.ranges == expected_range for item in rebuilt.items)

    sector = FeatureInvalidation("sector_revision_change", source_receipt_hash="c" * 64)
    arguments = dict(
        catalog=catalog,
        request=request,
        invalidations=(sector,),
        listing_plans=(),
        sessions=sessions,
        base_reusable=True,
    )
    legacy = compile_panel_plan(**arguments)
    assert all(item.ranges == expected_range for item in legacy.items)
    unchanged = compile_panel_plan(**arguments, row_identity_basis="SESSION_CROSS_SECTION")
    assert unchanged.items == ()
    assert unchanged.source_receipt_hashes == ("c" * 64,)
    changed = compile_panel_plan(
        **arguments, row_identity_basis="SESSION_CROSS_SECTION", forced_sessions=sessions[-2:]
    )
    assert set(targets_by_session(changed, sessions)) == set(sessions[-2:])


def test_full_catalog_plan_keeps_exact_dense_identity_and_targets() -> None:
    sessions = tuple(pd.bdate_range("2026-01-02", periods=12).date)
    catalog = FeatureCatalog.load()
    request = FeatureBuildRequest.create(
        manifest_revision="a" * 64,
        catalog=catalog.binding,
        spy_revision="b" * 64,
        history_start=sessions[0],
        as_of_session=sessions[-2],
        invalidations=(
            FeatureInvalidation(
                "catalog_binding_change",
                source_receipt_hash="c" * 64,
            ),
        ),
    )
    plan = compile_listing_plan(
        catalog=catalog,
        request=request,
        invalidations=request.invalidations,
        listing_id="listing-a",
        sessions=sessions,
    )
    active_sessions = sessions[:-1]
    full_range = (PanelInvalidationRange(active_sessions[0], active_sessions[-1]),)
    expected = FactorSessionInvalidationPlan.create(
        items=tuple(
            FactorSessionInvalidation(factor_id, full_range) for factor_id in catalog.factor_ids
        ),
        source_receipt_hashes=("c" * 64,),
    )

    assert plan == expected
    assert targets_by_session(plan, sessions) == {
        session: expected.factor_ids for session in active_sessions
    }


def test_real_calendar_warmup_preserves_seasonality_month_anchor() -> None:
    schedule = materialize_calendar_schedule(
        ("XNYS",),
        start=date(2020, 1, 1),
        end=date(2022, 3, 31),
        as_of_timestamp=datetime(2022, 4, 1, tzinfo=UTC),
    )
    sessions = tuple(row["session_date"] for row in schedule.to_pylist())
    target = date(2022, 3, 31)
    catalog = FeatureCatalog.load()
    warmup = max(
        item.lookback_sessions + item.formula_skip_sessions + 2
        for item in catalog.maintenance_contracts
    )
    start = FeatureFoundationService._bounded_input_start(
        calendar=sessions,
        first_target=target,
        warmup_sessions=warmup,
    )
    target_position = sessions.index(target)
    start_position = sessions.index(start)
    assert start.month == 2 and start.year == 2021
    assert target_position - start_position >= 276

    asset = _frame(periods=len(sessions), start="2020-01-02", seed_value=41)
    market = _frame(periods=len(sessions), start="2020-01-02", seed_value=43)
    asset["session_date"] = pd.to_datetime(sessions)
    market["session_date"] = pd.to_datetime(sessions)
    full = _values(asset, market)
    bounded = _values(
        asset.iloc[start_position:].reset_index(drop=True),
        market.iloc[start_position:].reset_index(drop=True),
    )
    assert np.isfinite(float(full.iloc[-1]["seasonality_12m"]))
    assert bounded.iloc[-1]["seasonality_12m"] == full.iloc[-1]["seasonality_12m"]


def test_source_perturbation_reach_is_covered_by_sparse_plan() -> None:
    catalog = FeatureCatalog.load()
    asset = _frame(periods=560, start="2022-01-03", seed_value=47)
    market = _frame(periods=560, start="2022-01-03", seed_value=53)
    sessions = tuple(pd.to_datetime(asset["session_date"]).dt.date)
    request = FeatureBuildRequest.create(
        manifest_revision="c" * 64,
        catalog=catalog.binding,
        spy_revision="d" * 64,
        history_start=sessions[0],
        as_of_session=sessions[-1],
    )
    baseline = _values(asset, market)
    correction_position = 300

    def materially_changed(left: object, right: object) -> bool:
        left_number = float(left)
        right_number = float(right)
        left_missing = np.isnan(left_number)
        right_missing = np.isnan(right_number)
        if left_missing or right_missing:
            return left_missing != right_missing
        return left_number != right_number

    cases: list[tuple[pd.DataFrame, FeatureInvalidation]] = []
    adjusted = asset.copy()
    adjusted.loc[correction_position, "provider_adjusted_close"] *= 1.013
    cases.append(
        (
            adjusted,
            FeatureInvalidation(
                "adjusted_return_correction",
                listing_id="fixture",
                affected_sessions=(
                    sessions[correction_position],
                    sessions[correction_position + 1],
                ),
                source_fields=("provider_adjusted_close",),
            ),
        )
    )
    ohlc = asset.copy()
    ohlc.loc[correction_position, "open_split_adjusted"] *= 1.002
    ohlc.loc[correction_position, "high_split_adjusted"] *= 1.007
    ohlc.loc[correction_position, "low_split_adjusted"] *= 0.993
    ohlc.loc[correction_position, "close_split_adjusted"] *= 1.001
    cases.append(
        (
            ohlc,
            FeatureInvalidation(
                "ohlc_correction",
                listing_id="fixture",
                affected_sessions=(sessions[correction_position],),
                source_fields=(
                    "close_split_adjusted",
                    "high_split_adjusted",
                    "low_split_adjusted",
                    "open_split_adjusted",
                ),
            ),
        )
    )
    volume = asset.copy()
    volume.loc[correction_position, "volume_raw"] *= 1.25
    cases.append(
        (
            volume,
            FeatureInvalidation(
                "volume_correction",
                listing_id="fixture",
                affected_sessions=(sessions[correction_position],),
                source_fields=("volume_raw",),
            ),
        )
    )

    for changed_frame, invalidation in cases:
        changed = _values(changed_frame, market)
        plan = compile_listing_plan(
            catalog=catalog,
            request=request,
            invalidations=(invalidation,),
            listing_id="fixture",
            sessions=sessions,
        )
        planned = {item.factor_id: set(expand_ranges(item.ranges, sessions)) for item in plan.items}
        for factor_id in catalog.factor_ids:
            actual_sessions = {
                sessions[position]
                for position, (left, right) in enumerate(
                    zip(baseline[factor_id], changed[factor_id], strict=True)
                )
                if materially_changed(left, right)
            }
            assert actual_sessions.issubset(planned.get(factor_id, set())), factor_id


def test_expand_ranges_bisects_a_calendar_and_keeps_an_unsorted_sequence_in_its_order() -> None:
    """requirement: the sessions inside any range, in the sequence's own order.

    A sorted calendar is sliced by bisection (the daily plan restriction
    expanded every factor's ranges by a linear scan of 2,500 sessions); an
    unsorted sequence still takes the linear filter, so both answer the same
    set, and the calendar answer is in calendar order.
    """

    from datetime import timedelta

    calendar = tuple(date(2026, 1, 5) + timedelta(days=offset) for offset in range(0, 40, 2))
    ranges = (
        PanelInvalidationRange(date(2026, 1, 9), date(2026, 1, 13)),
        PanelInvalidationRange(date(2025, 12, 30), date(2026, 1, 5)),
        PanelInvalidationRange(date(2026, 2, 8), date(2026, 3, 1)),
        PanelInvalidationRange(date(2026, 1, 12), date(2026, 1, 12)),
    )
    linear = tuple(
        session
        for session in calendar
        if any(item.first_session <= session <= item.last_session for item in ranges)
    )
    assert expand_ranges(ranges, calendar) == linear
    assert linear == (
        date(2026, 1, 5),
        date(2026, 1, 9),
        date(2026, 1, 11),
        date(2026, 1, 13),
        date(2026, 2, 8),
        date(2026, 2, 10),
        date(2026, 2, 12),
    )
    shuffled = calendar[::-1]
    assert expand_ranges(ranges, shuffled) == tuple(reversed(linear))
    assert expand_ranges((), calendar) == () and expand_ranges(ranges, ()) == ()


@settings(max_examples=300, deadline=None)
@given(
    offsets=st.lists(st.integers(0, 60), min_size=1, max_size=40),
    picked=st.sets(st.integers(0, 60), max_size=60),
    repeat=st.booleans(),
)
def test_planned_runs_of_selected_sessions_equal_the_calendar_scan(
    offsets: list[int], picked: set[int], repeat: bool
) -> None:
    """requirement: a plan's runs of selected sessions are found by position, as the scan's.

    The daily update planned one new session per factor by scanning the listing's whole calendar
    (53 factors, two plans a listing, 473 listings). Finding the runs by calendar position must
    give the scan's runs for any selection, and a calendar that repeats a session keeps the scan.
    """

    from datetime import timedelta

    def scanned(calendar: tuple[date, ...], selected: set[date]) -> tuple[object, ...]:
        ordered = [session for session in calendar if session in selected]
        if not ordered:
            return ()
        positions = {session: index for index, session in enumerate(calendar)}
        result = []
        first = previous = ordered[0]
        for session in ordered[1:]:
            if positions[session] != positions[previous] + 1:
                result.append(PanelInvalidationRange(first, previous))
                first = session
            previous = session
        result.append(PanelInvalidationRange(first, previous))
        return tuple(result)

    start = date(2026, 1, 5)
    calendar = tuple(start + timedelta(days=offset) for offset in sorted(set(offsets)))
    if repeat:
        middle = len(calendar) // 2
        calendar = (*calendar[: middle + 1], *calendar[middle:])
    whole = FactorSessionInvalidationPlan.create(
        items=(
            FactorSessionInvalidation(
                "ret_1d", (PanelInvalidationRange(min(calendar), max(calendar)),)
            ),
        )
    )
    selected = {start + timedelta(days=offset) for offset in picked}
    for chosen in (selected, set(sorted(selected)[:1])):
        restricted = restrict_plan_to_sessions(
            whole, calendar=calendar, allowed=chosen.__contains__
        )
        expected = scanned(calendar, chosen & set(calendar))
        assert tuple(item.ranges for item in restricted.items) == ((expected,) if expected else ())
