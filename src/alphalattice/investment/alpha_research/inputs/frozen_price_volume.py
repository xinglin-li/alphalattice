"""Frozen price/volume component inputs, independent of future training labels.

The consumed arithmetic is ported from the admitted research owner (source
SHA256 3b13fd5a0f9ae42f472751f39fc2a4aee6f0208d6418017444aa595af66a0728).
Shared stock and Sector scaling remain in their original local owners. This
module selects no strategy, fits no model and consumes no target array.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from types import MappingProxyType
from typing import cast

import numpy as np
import numpy.typing as npt

from alphalattice.investment.alpha_research.inputs.panel_feature_materialization import (
    apply_frozen_temporal_scale,
    scale_alpha_stock_cross_section,
)
from alphalattice.investment.alpha_research.inputs.panel_feature_views import (
    PanelFeatureBoundaryError,
)
from alphalattice.investment.alpha_research.inputs.preprocessing.sector_context import (
    scale_mapped_sector_context_by_listing_session,
)
from alphalattice.kernel.quant.sector_history import (
    SectorHistory,
    sector_codes,
    sector_ids,
    sector_slices,
)

type FloatArray = npt.NDArray[np.float64]
type IntArray = npt.NDArray[np.int64]
type BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class FrozenPriceVolumeInputs:
    """Retain dated price/volume and context tensors on one declared source listing axis.

    Optional formula values retain named float64 session-by-listing cells. Optional reference
    eligibility and nominal member counts describe a declared population axis; the object validates
    shape and coverage declarations.
    """

    formation_sessions: tuple[date, ...]
    ordered_listing_ids: tuple[str, ...]
    sector_by_listing_id: Mapping[str, str]
    open: FloatArray
    high: FloatArray
    low: FloatArray
    close: FloatArray
    volume: FloatArray
    # The three declared Market observations, before lag/temporal scaling.
    market_context_values: FloatArray
    # The declared Sector trend observation, before lag/listing scaling.
    sector_trend_values: FloatArray
    source_binding_hash: str
    # Raw, cutoff-visible Formula values read from the Feature owner. They are
    # not pre-scaled model features or target lanes.
    formula_values: Mapping[str, FloatArray] = field(default_factory=dict)
    reference_eligible: BoolArray | None = None
    """Dated members with usable basic observations, separate from raw warm-up coverage.

    None is the legacy all-member contract. Covered non-members may receive
    features under the fitted reference but cannot enter its estimates or selection.
    """
    nominal_member_count: npt.NDArray[np.int64] | None = None
    """Coverage denominator before source exclusions; absent means the reference roster."""

    @property
    def ordered_sector_ids(self) -> tuple[str, ...]:
        """Derive the sorted classification axis: every Sector some session reads.

        Returns:
            Sorted unique sector identifiers, a Sector history's included (V346).
        """
        ordered: tuple[str, ...] = sector_ids(self.sector_by_listing_id)
        return ordered

    def __post_init__(self) -> None:
        """Require complete source/classification axes and matching input/population shapes.

        Raises:
            PanelFeatureBoundaryError: Sessions/listings/source identity, classification coverage,
                price/context/formula shapes or optional reference/nominal-population declarations
                are invalid.
        """
        sessions, listings = len(self.formation_sessions), len(self.ordered_listing_ids)
        if (
            not sessions
            or listings < 2
            or tuple(sorted(set(self.formation_sessions))) != self.formation_sessions
            or len(set(self.ordered_listing_ids)) != listings
            or set(self.sector_by_listing_id) != set(self.ordered_listing_ids)
            or len(self.source_binding_hash) != 64
        ):
            raise PanelFeatureBoundaryError("alpha_research.frozen_input_axis_invalid")
        for name in ("open", "high", "low", "close", "volume"):
            if getattr(self, name).shape != (sessions, listings):
                raise PanelFeatureBoundaryError("alpha_research.frozen_input_shape_invalid")
        if self.market_context_values.shape != (sessions, 3) or self.sector_trend_values.shape != (
            sessions,
            len(self.ordered_sector_ids),
        ):
            raise PanelFeatureBoundaryError("alpha_research.frozen_context_axis_invalid")
        if any(
            not name or values.shape != (sessions, listings) or values.dtype != np.float64
            for name, values in self.formula_values.items()
        ):
            raise PanelFeatureBoundaryError("alpha_research.frozen_formula_axis_invalid")
        if self.reference_eligible is not None and (
            self.reference_eligible.shape != (sessions, listings)
            or self.reference_eligible.dtype != np.bool_
        ):
            raise PanelFeatureBoundaryError("alpha_research.frozen_reference_axis_invalid")
        if self.nominal_member_count is not None and (
            self.reference_eligible is None
            or self.nominal_member_count.shape != (sessions,)
            or self.nominal_member_count.dtype != np.int64
            or np.any(self.nominal_member_count < self.reference_eligible.sum(axis=1))
            or np.any(self.nominal_member_count > listings)
            or np.any(self.nominal_member_count < 1)
        ):
            raise PanelFeatureBoundaryError("alpha_research.frozen_nominal_population_invalid")

    def require_unchanged_prefix(self, previous: FrozenPriceVolumeInputs) -> None:
        """Append admission is not correction admission, including before book inception."""
        size = len(previous.formation_sessions)
        if (
            self.formation_sessions[:size] != previous.formation_sessions
            or self.ordered_listing_ids != previous.ordered_listing_ids
            or _prefix_sectors(self, size) != _prefix_sectors(previous, size)
            or self.formula_values.keys() != previous.formula_values.keys()
            or (self.reference_eligible is None) != (previous.reference_eligible is None)
            or (self.nominal_member_count is None) != (previous.nominal_member_count is None)
            or (
                self.nominal_member_count is not None
                and not np.array_equal(
                    self.nominal_member_count[:size], previous.nominal_member_count
                )
            )
            or (
                self.reference_eligible is not None
                and not np.array_equal(self.reference_eligible[:size], previous.reference_eligible)
            )
            or any(
                not np.array_equal(values[:size], previous.formula_values[name], equal_nan=True)
                for name, values in self.formula_values.items()
            )
            or any(
                not np.array_equal(
                    getattr(self, name)[:size], getattr(previous, name), equal_nan=True
                )
                for name in (
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "market_context_values",
                    "sector_trend_values",
                )
            )
        ):
            raise PanelFeatureBoundaryError("alpha_research.frozen_input_prefix_changed")


def _rolling_windows(values: FloatArray, window: int) -> tuple[FloatArray, BoolArray]:
    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 2 or window < 1 or len(source) < window:
        raise PanelFeatureBoundaryError("alpha_research.candidate_rolling_axis_invalid")
    windows = np.lib.stride_tricks.sliding_window_view(source, window, axis=0)
    finite = np.isfinite(windows).all(axis=-1)
    return cast(FloatArray, windows), cast(BoolArray, finite)


def _rolling_stat(values: FloatArray, window: int, statistic: str) -> FloatArray:
    windows, finite = _rolling_windows(values, window)
    if statistic == "sum":
        measured = np.sum(np.where(np.isfinite(windows), windows, 0.0), axis=-1)
    elif statistic == "mean":
        measured = np.mean(np.where(np.isfinite(windows), windows, 0.0), axis=-1)
    elif statistic == "std":
        measured = np.std(np.where(np.isfinite(windows), windows, 0.0), axis=-1, ddof=1)
    elif statistic == "min":
        measured = np.min(np.where(np.isfinite(windows), windows, np.inf), axis=-1)
    elif statistic == "max":
        measured = np.max(np.where(np.isfinite(windows), windows, -np.inf), axis=-1)
    elif statistic == "median":
        measured = np.median(np.where(np.isfinite(windows), windows, 0.0), axis=-1)
    else:
        raise PanelFeatureBoundaryError("alpha_research.candidate_rolling_stat_invalid")
    output = np.full(values.shape, np.nan, dtype=np.float64)
    output[window - 1 :] = np.where(finite, measured, np.nan)
    return cast(FloatArray, output)


def _observation_log_returns(close: FloatArray) -> FloatArray:
    prior = np.vstack((np.full((1, close.shape[1]), np.nan), close[:-1]))
    valid = np.isfinite(close) & np.isfinite(prior) & (close > 0.0) & (prior > 0.0)
    output = np.full(close.shape, np.nan, dtype=np.float64)
    output[valid] = np.log(close[valid] / prior[valid])
    return cast(FloatArray, output)


def _ratio(mean: FloatArray, denominator: FloatArray) -> FloatArray:
    return cast(
        FloatArray,
        np.divide(
            mean,
            denominator,
            out=np.full(mean.shape, np.nan, dtype=np.float64),
            where=np.isfinite(mean) & np.isfinite(denominator) & (denominator > 0.0),
        ),
    )


def _shift_rows(values: FloatArray, periods: int) -> FloatArray:
    if periods < 1:
        raise PanelFeatureBoundaryError("alpha_research.candidate_shift_invalid")
    output = np.full(values.shape, np.nan, dtype=np.float64)
    output[periods:] = values[:-periods]
    return cast(FloatArray, output)


def _recovery(close: FloatArray, window: int) -> FloatArray:
    minimum = _rolling_stat(close, window, "min")
    return _ratio(close, minimum) - 1.0


def _ema(values: FloatArray, span: int) -> FloatArray:
    alpha = 2.0 / (span + 1.0)
    state = np.full(values.shape[1], np.nan, dtype=np.float64)
    count = np.zeros(values.shape[1], dtype=np.int64)
    output = np.full(values.shape, np.nan, dtype=np.float64)
    for position, current in enumerate(values):
        finite = np.isfinite(current)
        state = np.where(
            finite,
            np.where(np.isfinite(state), alpha * current + (1.0 - alpha) * state, current),
            np.nan,
        )
        count = np.where(finite, count + 1, 0)
        output[position] = np.where(count >= span, state, np.nan)
    return cast(FloatArray, output)


def _trend_r2(close: FloatArray, window: int) -> FloatArray:
    """Frozen trend candidate arithmetic; not the distinct base Formula kernel."""
    log_close = np.where(np.isfinite(close) & (close > 0.0), np.log(close), np.nan)
    windows, finite = _rolling_windows(cast(FloatArray, log_close), window)
    x: FloatArray = np.arange(window, dtype=np.float64)
    centered_x = x - np.mean(x)
    denominator_x = float(np.sum(centered_x**2))
    safe = np.where(np.isfinite(windows), windows, 0.0)
    centered_y = safe - np.mean(safe, axis=-1, keepdims=True)
    covariance = np.sum(centered_y * centered_x, axis=-1)
    denominator_y = np.sum(centered_y**2, axis=-1)
    measured = np.divide(
        covariance**2,
        denominator_x * denominator_y,
        out=np.full(covariance.shape, np.nan, dtype=np.float64),
        where=finite & (denominator_y > 0.0),
    )
    output = np.full(close.shape, np.nan, dtype=np.float64)
    output[window - 1 :] = measured
    return cast(FloatArray, output)


def _sign_consistency(returns: FloatArray, window: int) -> FloatArray:
    windows, finite = _rolling_windows(returns, window)
    cumulative = np.sum(np.where(np.isfinite(windows), windows, 0.0), axis=-1)
    sign = np.where(cumulative >= 0.0, 1.0, -1.0)
    measured = np.mean(np.sign(windows) == sign[..., None], axis=-1)
    output = np.full(returns.shape, np.nan, dtype=np.float64)
    output[window - 1 :] = np.where(finite, measured, np.nan)
    return cast(FloatArray, output)


def _fast_micro_raw_features(*, ohlcv: FrozenPriceVolumeInputs) -> Mapping[str, FloatArray]:
    open_values = np.asarray(ohlcv.open, dtype=np.float64)
    high = np.asarray(ohlcv.high, dtype=np.float64)
    low = np.asarray(ohlcv.low, dtype=np.float64)
    close = np.asarray(ohlcv.close, dtype=np.float64)
    volume = np.asarray(ohlcv.volume, dtype=np.float64)
    prior_close = _shift_rows(cast(FloatArray, close), 1)
    spread = cast(FloatArray, high - low)
    upper_shadow = cast(FloatArray, high - np.maximum(open_values, close))
    lower_shadow = cast(FloatArray, np.minimum(open_values, close) - low)
    down_gap = cast(FloatArray, np.maximum(0.0, prior_close - open_values))
    bearish_body = cast(FloatArray, np.maximum(0.0, open_values - close))
    path_span = cast(FloatArray, np.abs(open_values - prior_close) + spread)
    true_range = cast(
        FloatArray,
        np.maximum.reduce((spread, np.abs(high - prior_close), np.abs(low - prior_close))),
    )

    clv = _ratio(cast(FloatArray, 2.0 * close - high - low), spread)
    body_ratio = _ratio(cast(FloatArray, close - open_values), spread)
    upper_shadow_share = _ratio(upper_shadow, spread)
    lower_shadow_share = _ratio(lower_shadow, spread)
    down_gap_ratio = _ratio(down_gap, prior_close)
    bearish_body_share = _ratio(bearish_body, spread)
    path_span_ratio = _ratio(path_span, prior_close)
    path_span_3 = _rolling_stat(path_span_ratio, 3, "sum")

    prior_volume = _shift_rows(cast(FloatArray, volume), 1)
    volume_baseline = _rolling_stat(prior_volume, 5, "median")
    relative_volume_ratio = _ratio(cast(FloatArray, volume), volume_baseline)
    relative_volume_5 = np.full(volume.shape, np.nan, dtype=np.float64)
    valid_relative_volume = np.isfinite(relative_volume_ratio) & (relative_volume_ratio >= 0.0)
    relative_volume_5[valid_relative_volume] = np.log1p(
        relative_volume_ratio[valid_relative_volume]
    ) / np.log(2.0)

    true_range_3 = _rolling_stat(true_range, 3, "sum")
    prior_close_3 = _shift_rows(cast(FloatArray, close), 3)
    price_efficiency_3 = _ratio(cast(FloatArray, close - prior_close_3), true_range_3)

    log_close = np.full(close.shape, np.nan, dtype=np.float64)
    finite_close = np.isfinite(close) & (close > 0.0)
    log_close[finite_close] = np.log(close[finite_close])
    log_close_2 = _shift_rows(cast(FloatArray, log_close), 2)
    log_close_5 = _shift_rows(cast(FloatArray, log_close), 5)
    fast_slope_2 = cast(FloatArray, (log_close - log_close_2) / 2.0)
    prior_slope_3 = cast(FloatArray, (log_close_2 - log_close_5) / 3.0)
    natr_5 = _rolling_stat(_ratio(true_range, prior_close), 5, "mean")

    log_returns = _observation_log_returns(cast(FloatArray, close))
    simple_returns = np.full(close.shape, np.nan, dtype=np.float64)
    finite_returns = np.isfinite(log_returns)
    simple_returns[finite_returns] = np.expm1(log_returns[finite_returns])
    dollar_volume = cast(FloatArray, close * volume)
    amihud = _ratio(cast(FloatArray, np.abs(simple_returns)), dollar_volume)
    valid_impact = np.isfinite(amihud) & finite_returns
    up_impact = np.where(valid_impact, np.where(simple_returns > 0.0, amihud, 0.0), np.nan)
    down_impact = np.where(valid_impact, np.where(simple_returns < 0.0, amihud, 0.0), np.nan)
    up_amihud_3 = _rolling_stat(cast(FloatArray, up_impact), 3, "sum")
    down_amihud_3 = _rolling_stat(cast(FloatArray, down_impact), 3, "sum")
    direction_balance_3 = _rolling_stat(
        cast(FloatArray, np.where(finite_returns, np.sign(simple_returns), np.nan)),
        3,
        "mean",
    )

    iie_3 = cast(FloatArray, price_efficiency_3 * relative_volume_5)
    tcd_vol = cast(FloatArray, (0.7 * clv + 0.3 * body_ratio) * relative_volume_5)
    fdfr_numerator = _rolling_stat(
        cast(FloatArray, down_gap + bearish_body + 0.5 * upper_shadow), 3, "sum"
    )
    fdfr_denominator = _rolling_stat(path_span, 3, "sum")
    fdfr_3 = _ratio(fdfr_numerator, fdfr_denominator)
    rca_5 = _ratio(cast(FloatArray, fast_slope_2 - prior_slope_3), natr_5)
    dmi_3 = _ratio(
        cast(FloatArray, up_amihud_3 - down_amihud_3),
        cast(FloatArray, up_amihud_3 + down_amihud_3),
    )

    bounded = {
        "clv": clv,
        "body_ratio": body_ratio,
        "upper_shadow_share": upper_shadow_share,
        "lower_shadow_share": lower_shadow_share,
        "fdfr_3": fdfr_3,
        "dmi_3": dmi_3,
    }
    for feature_id, measured in bounded.items():
        finite = measured[np.isfinite(measured)]
        lower_bound = (
            0.0 if feature_id in {"upper_shadow_share", "lower_shadow_share", "fdfr_3"} else -1.0
        )
        if bool(np.any(finite < lower_bound - 1.0e-12) or np.any(finite > 1.0 + 1.0e-12)):
            raise PanelFeatureBoundaryError(
                f"alpha_research.fast_micro_feature_out_of_bounds:{feature_id}"
            )

    values = {
        "price_efficiency_3": price_efficiency_3,
        "relative_volume_5": cast(FloatArray, relative_volume_5),
        "clv": clv,
        "body_ratio": body_ratio,
        "upper_shadow_share": upper_shadow_share,
        "lower_shadow_share": lower_shadow_share,
        "down_gap_ratio": down_gap_ratio,
        "bearish_body_share": bearish_body_share,
        "path_span_3": path_span_3,
        "fast_slope_2": fast_slope_2,
        "prior_slope_3": prior_slope_3,
        "natr_5": natr_5,
        "up_amihud_3": up_amihud_3,
        "down_amihud_3": down_amihud_3,
        "direction_balance_3": direction_balance_3,
        "iie_3": iie_3,
        "tcd_vol": tcd_vol,
        "fdfr_3": fdfr_3,
        "rca_5": rca_5,
        "dmi_3": dmi_3,
    }
    return MappingProxyType(values)


def _sector_observation_mean(returns: FloatArray) -> FloatArray:
    block = np.expm1(returns)
    finite = np.isfinite(block)
    count = np.sum(finite, axis=1)
    return np.divide(
        np.sum(np.where(finite, block, 0.0), axis=1),
        count,
        out=np.full(len(block), np.nan, dtype=np.float64),
        where=count >= 5,
    )


def _prefix_sectors(
    source: FrozenPriceVolumeInputs, size: int
) -> tuple[tuple[slice, dict[str, str]], ...]:
    """The Sector map each of the first `size` sessions reads, run by run (V346)."""
    return tuple((rows, dict(mapping)) for rows, mapping in _runs(source, size))


def _runs(
    source: FrozenPriceVolumeInputs, count: int
) -> tuple[tuple[slice, Mapping[str, str]], ...]:
    """The first `count` sessions' runs: a history's, or one of a plain map (V346)."""
    if isinstance(source.sector_by_listing_id, SectorHistory):
        return sector_slices(source.sector_by_listing_id, source.formation_sessions[:count])
    return ((slice(0, count), source.sector_by_listing_id),)


def _sector_recovery(source: FrozenPriceVolumeInputs, returns: FloatArray) -> FloatArray:
    sectors = source.ordered_sector_ids
    means: FloatArray = np.full((len(returns), len(sectors)), np.nan, dtype=np.float64)
    # Each run of sessions averages the Sectors in force there (V346); one run while no
    # reclassification falls inside the window.
    for rows, mapping in _runs(source, len(returns)):
        first = rows.start or 0
        for column, sector in enumerate(sectors):
            positions = [
                i
                for i, listing in enumerate(source.ordered_listing_ids)
                if mapping[listing] == sector
            ]
            if source.reference_eligible is None:
                means[rows, column] = _sector_observation_mean(returns[rows][:, positions])
                continue
            masks, inverse = np.unique(
                source.reference_eligible[rows][:, positions], axis=0, return_inverse=True
            )
            for index, mask in enumerate(masks):
                days: IntArray = first + np.flatnonzero(inverse == index)
                members = np.asarray(positions)[mask]
                # Compact C rows keep historical reductions independent of future
                # coverage width and the number of dates in a membership group.
                means[days, column] = _sector_observation_mean(
                    np.ascontiguousarray(returns[np.ix_(days, members)])
                )
    logs: FloatArray = np.full(means.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(means) & (means > -1.0)
    logs[valid] = np.log1p(means[valid])
    wealth: FloatArray = np.full(logs.shape, np.nan, dtype=np.float64)
    for column in range(logs.shape[1]):
        state = 0.0
        for row in range(len(logs)):
            if np.isfinite(logs[row, column]):
                state += logs[row, column]
                wealth[row, column] = np.exp(state)
            else:
                state = 0.0
    count = len(wealth)
    codes = sector_codes(
        source.sector_by_listing_id,
        source.formation_sessions[:count],
        source.ordered_listing_ids,
        sectors,
    )
    rows_index: IntArray = np.arange(count, dtype=np.int64)[:, None]
    mapped = _recovery(wealth, 21)[rows_index, codes][:, :, None]
    return scale_mapped_sector_context_by_listing_session(
        mapped, reference_eligible=source.reference_eligible
    ).values


def prepare_frozen_price_volume_features(
    source: FrozenPriceVolumeInputs,
    *,
    ordered_feature_ids: tuple[str, ...],
    formation_session: date,
) -> FloatArray:
    """Calculate one formation's shared columns before vintage-specific scaling."""
    return prepare_frozen_price_volume_history(
        source, ordered_feature_ids=ordered_feature_ids, through=formation_session
    )[-1]


def prepare_frozen_price_volume_history(
    source: FrozenPriceVolumeInputs,
    *,
    ordered_feature_ids: tuple[str, ...],
    through: date,
) -> FloatArray:
    """One causal pass for training and inference, preserving every source row."""
    try:
        end = source.formation_sessions.index(through) + 1
    except ValueError as error:
        raise PanelFeatureBoundaryError("alpha_research.frozen_formation_unavailable") from error
    # Bound every calculation by formation, even if the source carries later rows.
    source = FrozenPriceVolumeInputs(
        formation_sessions=source.formation_sessions[:end],
        ordered_listing_ids=source.ordered_listing_ids,
        sector_by_listing_id=source.sector_by_listing_id,
        **{
            name: getattr(source, name)[:end]
            for name in (
                "open",
                "high",
                "low",
                "close",
                "volume",
                "market_context_values",
                "sector_trend_values",
            )
        },
        source_binding_hash=source.source_binding_hash,
        formula_values={name: values[:end] for name, values in source.formula_values.items()},
        reference_eligible=(
            source.reference_eligible[:end] if source.reference_eligible is not None else None
        ),
        nominal_member_count=(
            source.nominal_member_count[:end] if source.nominal_member_count is not None else None
        ),
    )
    raw = dict(_fast_micro_raw_features(ohlcv=source))
    returns = _observation_log_returns(source.close)
    raw["mean_return_acceleration_5_21"] = _rolling_stat(returns, 5, "mean") - _rolling_stat(
        returns, 21, "mean"
    )
    raw["recovery_from_21d_low"] = _recovery(source.close, 21)
    ppo = _ratio(_ema(source.close, 12), _ema(source.close, 26)) - 1.0
    raw["ppo_12_26"] = ppo
    raw["ppo_signal_gap_9"] = ppo - _ema(ppo, 9)
    requested = {value.rsplit("::", 1)[-1] for value in ordered_feature_ids}
    for window in (21, 63):
        if f"stock_sharpe_{window}" in requested:
            raw[f"stock_sharpe_{window}"] = _ratio(
                _rolling_stat(returns, window, "mean"), _rolling_stat(returns, window, "std")
            ) * np.sqrt(252.0)
        if f"momentum_sign_consistency_{window}" in requested:
            raw[f"momentum_sign_consistency_{window}"] = _sign_consistency(returns, window)
        if f"trend_r2_{window}" in requested:
            raw[f"trend_r2_{window}"] = _trend_r2(source.close, window)
    columns: dict[str, FloatArray] = {}
    for role in ("RELATIVE_STOCK_CROSS_SECTION", "NON_NEUTRAL_STOCK_CROSS_SECTION"):
        selected = [name for name in ordered_feature_ids if name.startswith(role + "::")]
        if not selected:
            continue
        try:
            formula = np.stack(
                [source.formula_values[name.split("::")[1]] for name in selected], axis=-1
            )
        except KeyError as error:
            raise PanelFeatureBoundaryError(
                "alpha_research.frozen_formula_not_installed"
            ) from error
        scaled = scale_alpha_stock_cross_section(
            formula,
            source=source,
            sector_neutral=role == "RELATIVE_STOCK_CROSS_SECTION",
            reference_eligible=source.reference_eligible,
            sessions=source.formation_sessions,
        )
        columns.update({name: scaled[:, :, i] for i, name in enumerate(selected)})
    stock_ids = [
        value
        for value in ordered_feature_ids
        if value.startswith("ALPHA_DEVELOPMENT_CANDIDATE::") and "::C::" not in value
    ]
    try:
        cube = np.stack([raw[value.rsplit("::", 1)[-1]] for value in stock_ids], axis=-1)
    except KeyError as error:
        raise PanelFeatureBoundaryError("alpha_research.frozen_feature_not_installed") from error
    stock = scale_alpha_stock_cross_section(
        cube, source=source, sector_neutral=False, reference_eligible=source.reference_eligible
    )
    columns.update({name: stock[:, :, i] for i, name in enumerate(stock_ids)})
    market = np.array(source.market_context_values, copy=True)
    market[:, 0] = _shift_rows(market[:, :1], 1)[:, 0]
    for i, name in enumerate(
        (
            "MARKET_CONTEXT::market_drawdown_252::lag1",
            "MARKET_CONTEXT::observed_breadth_positive_share::current",
            "MARKET_CONTEXT::observed_new_high_low_share::current",
        )
    ):
        columns[name] = np.broadcast_to(market[:, i, None], (end, len(source.ordered_listing_ids)))
    # Each session's Sector of each listing (V346).
    trend_codes = sector_codes(
        source.sector_by_listing_id,
        source.formation_sessions,
        source.ordered_listing_ids,
        source.ordered_sector_ids,
    )
    trend_rows: IntArray = np.arange(len(source.formation_sessions), dtype=np.int64)[:, None]
    trend = _shift_rows(source.sector_trend_values, 1)[trend_rows, trend_codes][:, :, None]
    columns["SECTOR_CONTEXT::sector_trend_20::lag1"] = (
        scale_mapped_sector_context_by_listing_session(
            trend, reference_eligible=source.reference_eligible
        ).values[:, :, 0]
    )
    columns["ALPHA_DEVELOPMENT_CANDIDATE::C::sector_recovery_from_low_21"] = _sector_recovery(
        source, returns
    )[:, :, 0]
    try:
        values = np.ascontiguousarray(
            np.stack([columns[name] for name in ordered_feature_ids], axis=-1), dtype=np.float64
        )
    except KeyError as error:
        raise PanelFeatureBoundaryError("alpha_research.frozen_feature_not_installed") from error
    values.setflags(write=False)
    return values


def apply_frozen_price_volume_scale(
    values: FloatArray,
    *,
    ordered_feature_ids: tuple[str, ...],
    market_center: npt.ArrayLike,
    market_scale: npt.ArrayLike,
) -> FloatArray:
    """Only the three Market columns depend on vintage; never redo stock work."""
    positions = [
        index
        for index, name in enumerate(ordered_feature_ids)
        if name.startswith("MARKET_CONTEXT::")
    ]
    result = np.array(values, copy=True, dtype=np.float64)
    if positions:
        result[:, positions] = apply_frozen_temporal_scale(
            values[:, positions], center=market_center, scale=market_scale
        )
    result.setflags(write=False)
    return result
