"""Named Context selection preserves full-history bytes and source admission."""

from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pytest

from alphalattice.investment.alpha_research.inputs.frozen_price_volume import (
    FrozenPriceVolumeInputs,
    prepare_frozen_price_volume_features,
    prepare_frozen_price_volume_history,
)
from alphalattice.investment.alpha_research.inputs.panel_feature_views import (
    BASE_MARKET_CONTEXT_SOURCE_IDS,
    OBSERVED_MARKET_SOURCE_IDS,
    SECTOR_CONTEXT_SOURCE_IDS,
    PanelFeatureBoundaryError,
    assemble_panel_context_arrays,
)
from alphalattice.investment.alpha_research.scores.heterogeneous_product import (
    INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY,
)
from alphalattice.kernel.quant.sector_history import SectorHistory, SectorReclassification

MARKET_IDS = (*BASE_MARKET_CONTEXT_SOURCE_IDS, *OBSERVED_MARKET_SOURCE_IDS)
INFERENCE_MARKET_IDS = (
    "market_drawdown_252",
    "observed_breadth_positive_share",
    "observed_new_high_low_share",
)
INFERENCE_SECTOR_IDS = ("sector_trend_20",)


def _sessions():
    days = []
    day = date(2024, 1, 2)
    while len(days) < 300:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return tuple(days)


@pytest.fixture(params=("all-members", "dated-members"))
def context_source(request):
    days = _sessions()
    names = tuple(f"listing-{index:02d}" for index in range(12))
    base = {name: f"sector-{index % 3}" for index, name in enumerate(names)}
    current = {**base, names[0]: "sector-1", names[7]: "sector-2"}
    sectors = SectorHistory(
        current_revision="b" * 64,
        current=current,
        reclassifications=(
            SectorReclassification(names[0], days[110], "sector-0", "sector-1"),
            SectorReclassification(names[7], days[210], "sector-1", "sector-2"),
        ),
    )
    rng = np.random.default_rng(1729)
    raw_log = rng.normal(0.0, 0.01, (300, 12))
    raw_log[73, 4] = np.nan
    raw_log[-4:, 2] = np.nan
    raw_simple = np.expm1(raw_log)
    observed = rng.normal(0.0, 0.012, (300, 12))
    observed[38, 2] = np.nan
    observed[71, 9] = np.inf
    observed[133, 8] = -np.inf
    volume_state = rng.normal(0.0, 0.1, (300, 12))
    volume_state[90, 4] = np.nan
    dollar_volume = rng.uniform(1e5, 1e6, (300, 12))
    dollar_volume[61, 11] = np.nan
    high_distance = -rng.uniform(0.0, 0.08, (300, 12))
    low_distance = rng.uniform(0.0, 0.08, (300, 12))
    high_distance[::11, :6] = 0.0
    low_distance[::17, 6:] = 0.0
    high_distance[55, 0] = np.nan
    low_distance[112, 8] = np.nan
    # Published formation-available states are independent of the outcome lanes.
    states = rng.normal(0.0, 0.01, (300, 3, 4))
    states[:10] = np.nan
    interactions = rng.normal(0.0, 0.01, (300, 2))
    reference = None
    if request.param == "dated-members":
        reference = np.ones((300, 12), dtype=np.bool_)
        reference[:100, :3] = False
        reference[100:200, -3:] = False
        reference[200:, (1, 5, 9)] = False
        reference.setflags(write=False)
    arrays = {
        "raw_log_execution_returns": raw_log,
        "raw_simple_execution_returns": raw_simple,
        "sector_state_values": states,
        "market_interaction_state_values": interactions,
        "observation_returns": observed,
        "observation_volume_state": volume_state,
        "observation_dollar_volume": dollar_volume,
        "observation_high_distance": high_distance,
        "observation_low_distance": low_distance,
    }
    for values in arrays.values():
        values.setflags(write=False)
    return {
        "formation_sessions": days,
        "holding_end_sessions": tuple(
            days[index + 2] if index < 296 else None for index in range(300)
        ),
        "ordered_listing_ids": names,
        "ordered_sector_ids": sectors.sectors,
        "sector_by_listing_id": sectors,
        "reference_eligible": reference,
        **arrays,
    }


def _same_bytes(expected, actual):
    assert actual.dtype == expected.dtype == np.dtype(np.float64)
    assert actual.shape == expected.shape
    assert actual.flags.c_contiguous
    assert not actual.flags.writeable
    assert actual.tobytes(order="C") == np.ascontiguousarray(expected).tobytes(order="C")
    if actual.size:
        with pytest.raises(ValueError, match="read-only"):
            actual.flat[0] = 0.0


@pytest.mark.parametrize(
    ("sector_ids", "market_ids"),
    (
        pytest.param(None, None, id="default-full"),
        pytest.param(SECTOR_CONTEXT_SOURCE_IDS, MARKET_IDS, id="explicit-full"),
        pytest.param(
            tuple(reversed(SECTOR_CONTEXT_SOURCE_IDS)),
            tuple(reversed(MARKET_IDS)),
            id="reordered",
        ),
        pytest.param(INFERENCE_SECTOR_IDS, INFERENCE_MARKET_IDS, id="inference"),
        pytest.param((), None, id="empty-sector"),
        pytest.param(None, (), id="empty-market"),
        pytest.param((), (), id="empty-both"),
    ),
)
def test_named_axes_equal_the_complete_owner_output(context_source, sector_ids, market_ids):
    full_sector, full_market = assemble_panel_context_arrays(**context_source)
    assert full_sector.shape == (300, 3, 5)
    assert full_market.shape == (300, 17)
    sector, market = assemble_panel_context_arrays(
        **context_source,
        selected_sector_source_ids=sector_ids,
        selected_market_source_ids=market_ids,
    )
    sector_positions = [
        SECTOR_CONTEXT_SOURCE_IDS.index(name)
        for name in (SECTOR_CONTEXT_SOURCE_IDS if sector_ids is None else sector_ids)
    ]
    market_positions = [
        MARKET_IDS.index(name) for name in (MARKET_IDS if market_ids is None else market_ids)
    ]
    _same_bytes(full_sector[:, :, sector_positions], sector)
    _same_bytes(full_market[:, market_positions], market)


@pytest.mark.parametrize(
    ("axis", "source_id"),
    (
        *(("sector", name) for name in SECTOR_CONTEXT_SOURCE_IDS),
        *(("market", name) for name in MARKET_IDS),
    ),
)
def test_each_source_lane_can_be_selected_without_its_neighbours(context_source, axis, source_id):
    full_sector, full_market = assemble_panel_context_arrays(**context_source)
    sector, market = assemble_panel_context_arrays(
        **context_source,
        selected_sector_source_ids=(source_id,) if axis == "sector" else (),
        selected_market_source_ids=(source_id,) if axis == "market" else (),
    )
    sector_positions = [SECTOR_CONTEXT_SOURCE_IDS.index(source_id)] if axis == "sector" else []
    market_positions = [MARKET_IDS.index(source_id)] if axis == "market" else []
    _same_bytes(full_sector[:, :, sector_positions], sector)
    _same_bytes(full_market[:, market_positions], market)


def test_unmatured_outcome_tail_does_not_enter_any_selected_or_full_lane(context_source):
    original = assemble_panel_context_arrays(**context_source)
    log = context_source["raw_log_execution_returns"].copy()
    simple = context_source["raw_simple_execution_returns"].copy()
    log[-4:] = 0.5
    simple[-4:] = np.expm1(0.5)
    changed = {
        **context_source,
        "raw_log_execution_returns": log,
        "raw_simple_execution_returns": simple,
    }
    for expected, actual in zip(original, assemble_panel_context_arrays(**changed), strict=True):
        _same_bytes(expected, actual)
    sector, market = assemble_panel_context_arrays(
        **changed,
        selected_sector_source_ids=INFERENCE_SECTOR_IDS,
        selected_market_source_ids=INFERENCE_MARKET_IDS,
    )
    sector_positions = [SECTOR_CONTEXT_SOURCE_IDS.index(name) for name in INFERENCE_SECTOR_IDS]
    _same_bytes(original[0][:, :, sector_positions], sector)
    _same_bytes(original[1][:, [MARKET_IDS.index(name) for name in INFERENCE_MARKET_IDS]], market)


@pytest.mark.parametrize("component_id", ("G6_R0_FAST_REBOUND", "G2_R0_TREND"))
def test_installed_component_feature_history_keeps_exact_selected_context_bytes(
    context_source, component_id
):
    component = INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY.component(component_id)
    full_sector, full_market = assemble_panel_context_arrays(**context_source)
    sector, market = assemble_panel_context_arrays(
        **context_source,
        selected_sector_source_ids=INFERENCE_SECTOR_IDS,
        selected_market_source_ids=INFERENCE_MARKET_IDS,
    )
    rng = np.random.default_rng(42)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.008, (300, 12)), axis=0))
    formula_values = {
        name.split("::")[1]: rng.normal(size=(300, 12))
        for name in component.ordered_feature_ids
        if name.startswith(("RELATIVE_STOCK_CROSS_SECTION::", "NON_NEUTRAL_STOCK_CROSS_SECTION::"))
    }
    common = {
        "formation_sessions": context_source["formation_sessions"],
        "ordered_listing_ids": context_source["ordered_listing_ids"],
        "sector_by_listing_id": context_source["sector_by_listing_id"],
        "open": close * 0.999,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": rng.uniform(1e5, 1e6, (300, 12)),
        "formula_values": formula_values,
        "reference_eligible": context_source["reference_eligible"],
        "source_binding_hash": "c" * 64,
    }
    baseline = FrozenPriceVolumeInputs(
        **common,
        market_context_values=full_market[
            :, [MARKET_IDS.index(name) for name in INFERENCE_MARKET_IDS]
        ],
        sector_trend_values=full_sector[:, :, SECTOR_CONTEXT_SOURCE_IDS.index("sector_trend_20")],
    )
    selected = FrozenPriceVolumeInputs(
        **common, market_context_values=market, sector_trend_values=sector[:, :, 0]
    )
    arguments = {
        "ordered_feature_ids": component.ordered_feature_ids,
        "through": context_source["formation_sessions"][-1],
    }
    expected = prepare_frozen_price_volume_history(baseline, **arguments)
    actual = prepare_frozen_price_volume_history(selected, **arguments)
    assert actual.shape == (300, 12, component.feature_count)
    _same_bytes(expected, actual)


@pytest.mark.parametrize("component_id", ("G6_R0_FAST_REBOUND", "G2_R0_TREND"))
def test_a_formation_row_from_its_window_is_the_full_pass_row_on_every_day(
    context_source, component_id, monkeypatch
):
    """A formation row from its window is the full pass row on every day."""
    component = INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY.component(component_id)
    sector, market = assemble_panel_context_arrays(
        **context_source,
        selected_sector_source_ids=INFERENCE_SECTOR_IDS,
        selected_market_source_ids=INFERENCE_MARKET_IDS,
    )
    rng = np.random.default_rng(7)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.008, (300, 12)), axis=0))
    close[:150, 3] = np.nan
    close[[40, 41, 200], 6] = np.nan
    days = context_source["formation_sessions"]
    source = FrozenPriceVolumeInputs(
        formation_sessions=days,
        ordered_listing_ids=context_source["ordered_listing_ids"],
        sector_by_listing_id=context_source["sector_by_listing_id"],
        open=close * 0.999,
        high=close * 1.01,
        low=close * 0.99,
        close=close,
        volume=rng.uniform(1e5, 1e6, (300, 12)),
        formula_values={
            name.split("::")[1]: rng.normal(size=(300, 12))
            for name in component.ordered_feature_ids
            if name.startswith(
                ("RELATIVE_STOCK_CROSS_SECTION::", "NON_NEUTRAL_STOCK_CROSS_SECTION::")
            )
        },
        reference_eligible=context_source["reference_eligible"],
        source_binding_hash="c" * 64,
        market_context_values=market,
        sector_trend_values=sector[:, :, 0],
    )
    compared = 0
    for day in days:
        arguments = {"ordered_feature_ids": component.ordered_feature_ids}
        try:
            expected = prepare_frozen_price_volume_history(source, **arguments, through=day)[-1]
        except PanelFeatureBoundaryError as error:
            # A history shorter than a kernel's window refuses alike either way.
            with pytest.raises(PanelFeatureBoundaryError, match=str(error)):
                prepare_frozen_price_volume_features(source, **arguments, formation_session=day)
            continue
        row = prepare_frozen_price_volume_features(source, **arguments, formation_session=day)
        assert row.tobytes() == np.ascontiguousarray(expected).tobytes(), day
        compared += 1
    assert compared > 200
    names = ("open", "high", "low", "volume")
    borrowed = tuple(getattr(source, name) for name in names)
    consumed = []
    asarray = np.asarray

    def count_rows(value, *args, **kwargs):
        if isinstance(value, np.ndarray) and any(
            np.shares_memory(value, lane) for lane in borrowed
        ):
            consumed.append(len(value))
        return asarray(value, *args, **kwargs)

    def bounded_formation():
        observed = max(consumed)
        assert observed <= 128, (
            f"COUNT STOP op=formation_rows observed={observed} budget=128 "
            "way_on=bound_price_volume_kernels"
        )

    with monkeypatch.context() as counts:
        counts.setattr(np, "asarray", count_rows)
        row = prepare_frozen_price_volume_features(source, **arguments, formation_session=days[-1])
        bounded_formation()
        consumed.clear()
        overworked = prepare_frozen_price_volume_history(source, **arguments, through=days[-1])[-1]
        assert overworked.tobytes() == row.tobytes()
        with pytest.raises(AssertionError, match="COUNT STOP"):
            bounded_formation()
    poisoned = {name: lane.copy() for name, lane in zip(names, borrowed, strict=True)}
    for lane in poisoned.values():
        lane[:-128] = np.nan
    assert (
        prepare_frozen_price_volume_features(
            replace(source, **poisoned), **arguments, formation_session=days[-1]
        ).tobytes()
        == row.tobytes()
    )


@pytest.mark.parametrize(
    ("selector", "source_ids", "code"),
    (
        ("selected_sector_source_ids", ("sector_trend_20", "sector_trend_20"), "axis_invalid"),
        (
            "selected_market_source_ids",
            ("market_drawdown_252", "market_drawdown_252"),
            "axis_invalid",
        ),
        ("selected_sector_source_ids", ("not_installed",), "source_not_installed"),
        ("selected_market_source_ids", ("not_installed",), "source_not_installed"),
    ),
)
def test_selection_refuses_unknown_or_repeated_source_ids(
    context_source, selector, source_ids, code
):
    with pytest.raises(PanelFeatureBoundaryError) as seen:
        assemble_panel_context_arrays(**context_source, **{selector: source_ids})
    assert str(seen.value) == f"alpha_research.feature_context_{code}"


def test_observed_sources_require_the_observation_axis(context_source):
    without_observations = {**context_source, "observation_returns": None}
    full_sector, full_market = assemble_panel_context_arrays(**context_source)
    sector, market = assemble_panel_context_arrays(**without_observations)
    _same_bytes(full_sector, sector)
    _same_bytes(full_market[:, : len(BASE_MARKET_CONTEXT_SOURCE_IDS)], market)
    selected_sector, selected_market = assemble_panel_context_arrays(
        **without_observations,
        selected_sector_source_ids=SECTOR_CONTEXT_SOURCE_IDS,
        selected_market_source_ids=tuple(reversed(BASE_MARKET_CONTEXT_SOURCE_IDS)),
    )
    _same_bytes(sector, selected_sector)
    _same_bytes(market[:, ::-1], selected_market)
    with pytest.raises(PanelFeatureBoundaryError) as seen:
        assemble_panel_context_arrays(
            **without_observations, selected_market_source_ids=INFERENCE_MARKET_IDS
        )
    assert str(seen.value) == "alpha_research.feature_context_source_not_installed"


@pytest.mark.parametrize(
    ("source_id", "absent"),
    (
        ("observed_volume_regime", "observation_volume_state"),
        ("observed_new_high_low_share", "observation_high_distance"),
        ("observed_new_high_low_share", "observation_low_distance"),
        ("observed_up_volume_share", "observation_dollar_volume"),
    ),
)
def test_an_absent_optional_observation_keeps_its_requested_lane_missing(
    context_source, source_id, absent
):
    source = {**context_source, absent: None}
    _, full_market = assemble_panel_context_arrays(**source)
    _, market = assemble_panel_context_arrays(
        **source, selected_sector_source_ids=(), selected_market_source_ids=(source_id,)
    )
    assert np.isnan(market).all()
    _same_bytes(full_market[:, [MARKET_IDS.index(source_id)]], market)


@pytest.mark.parametrize(
    "name",
    (
        "observation_returns",
        "observation_volume_state",
        "observation_dollar_volume",
        "observation_high_distance",
        "observation_low_distance",
    ),
)
def test_selected_mode_checks_every_provided_observation_even_when_its_lane_is_unused(
    context_source, name
):
    source = {**context_source, name: np.zeros((300, 11), dtype=np.float64)}
    with pytest.raises(PanelFeatureBoundaryError) as seen:
        assemble_panel_context_arrays(
            **source,
            selected_sector_source_ids=(),
            selected_market_source_ids=("market_drawdown_x_momentum__state",),
        )
    assert str(seen.value) == "alpha_research.feature_context_axis_invalid"


@pytest.mark.parametrize(
    ("name", "shape", "dtype", "code"),
    (
        ("raw_log_execution_returns", (300, 11), np.float64, "source_axis_invalid"),
        ("raw_simple_execution_returns", (299, 12), np.float64, "source_axis_invalid"),
        ("sector_state_values", (300, 3, 3), np.float64, "source_axis_invalid"),
        ("market_interaction_state_values", (300, 1), np.float64, "source_axis_invalid"),
        ("reference_eligible", (300, 11), np.bool_, "reference_axis_invalid"),
        ("reference_eligible", (300, 12), np.float64, "reference_axis_invalid"),
    ),
)
@pytest.mark.parametrize("selected", (False, True))
def test_selection_keeps_original_source_and_reference_axis_refusals(
    context_source, name, shape, dtype, code, selected
):
    source = {**context_source, name: np.zeros(shape, dtype=dtype)}
    selection = (
        {
            "selected_sector_source_ids": INFERENCE_SECTOR_IDS,
            "selected_market_source_ids": INFERENCE_MARKET_IDS,
        }
        if selected
        else {}
    )
    with pytest.raises(PanelFeatureBoundaryError) as seen:
        assemble_panel_context_arrays(**source, **selection)
    assert str(seen.value) == f"alpha_research.panel_context_{code}"


@pytest.mark.parametrize("selected", (False, True))
def test_selection_keeps_original_holding_clock_axis_refusal(context_source, selected):
    source = {**context_source, "holding_end_sessions": context_source["holding_end_sessions"][:-1]}
    selection = (
        {
            "selected_sector_source_ids": INFERENCE_SECTOR_IDS,
            "selected_market_source_ids": INFERENCE_MARKET_IDS,
        }
        if selected
        else {}
    )
    with pytest.raises(PanelFeatureBoundaryError) as seen:
        assemble_panel_context_arrays(**source, **selection)
    assert str(seen.value) == "alpha_research.panel_context_source_axis_invalid"
