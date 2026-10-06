"""Frozen component target materialization over causal execution observations."""

from __future__ import annotations

from datetime import date

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from alphalattice.investment.alpha_research.inputs.panel_feature_materialization import (
    ListingSectorAxis,
    scale_alpha_stock_cross_section,
)

from .execution_outcome import AlphaTargetBoundaryError
from .total_return import (
    build_total_return_alpha_target_recipe,
    compile_total_return_alpha_target_surface,
)

type FloatArray = npt.NDArray[np.float64]


def compile_frozen_component_training_targets(
    *,
    source: ListingSectorAxis,
    formation_sessions: tuple[date, ...],
    holding_end_sessions: tuple[date | None, ...],
    raw_log_returns: FloatArray,
    simple_returns: FloatArray,
    target_method_id: str,
    reference_eligible: npt.NDArray[np.bool_] | None = None,
    nominal_member_count: npt.NDArray[np.int64] | None = None,
) -> tuple[FloatArray, tuple[date | None, ...]]:
    """Retain each frozen target's exact arithmetic over matured h1 observations.

    G2's h3 target compounds three consecutive h1 log returns before the frozen
    stock transform. It is not a new Portfolio execution-return method. G6's
    canonical target uses the pre-existing total-return owner, whose summation
    order is deliberately not replaced by the stock transform.
    """
    shape = (len(formation_sessions), len(source.ordered_listing_ids))
    if (
        not formation_sessions
        or formation_sessions != tuple(sorted(set(formation_sessions)))
        or raw_log_returns.shape != shape
        or simple_returns.shape != shape
        or len(holding_end_sessions) != shape[0]
        or any(
            end is not None and end <= day
            for day, end in zip(formation_sessions, holding_end_sessions, strict=True)
        )
        or any(
            end is not None and i + 2 < shape[0] and end != formation_sessions[i + 2]
            for i, end in enumerate(holding_end_sessions)
        )
        or np.isinf(raw_log_returns).any()
        or np.isinf(simple_returns).any()
        or (
            reference_eligible is not None
            and (reference_eligible.shape != shape or reference_eligible.dtype != np.bool_)
        )
        or (
            nominal_member_count is not None
            and (
                reference_eligible is None
                or nominal_member_count.shape != (shape[0],)
                or nominal_member_count.dtype != np.int64
                or np.any(nominal_member_count < reference_eligible.sum(axis=1))
                or np.any(nominal_member_count > shape[1])
                or np.any(nominal_member_count < 1)
            )
        )
    ):
        raise AlphaTargetBoundaryError("alpha_research.component_target_axis_invalid")
    if target_method_id == "EXACT_FROZEN_G0_H1_WHOLE_UNIVERSE_TARGET":
        positions = [i for i, end in enumerate(holding_end_sessions) if end is not None]
        target: FloatArray = np.full(shape, np.nan, dtype=np.float64)
        if positions:
            table = pa.table(
                {
                    "formation_session": [
                        formation_sessions[i] for i in positions for _ in source.ordered_listing_ids
                    ],
                    "holding_end_session": [
                        holding_end_sessions[i]
                        for i in positions
                        for _ in source.ordered_listing_ids
                    ],
                    "listing_id": list(source.ordered_listing_ids) * len(positions),
                    "fit_target": raw_log_returns[positions].reshape(-1),
                    "simple_economic_return": simple_returns[positions].reshape(-1),
                }
            )
            surface = compile_total_return_alpha_target_surface(
                source_table=table,
                recipe=build_total_return_alpha_target_recipe(
                    execution_outcome_recipe_id="NEXT_OPEN_TO_OPEN_ONE_SESSION",
                    reference_coverage=reference_eligible is not None,
                ),
                reference_eligible=(
                    reference_eligible[positions] if reference_eligible is not None else None
                ),
                nominal_member_count=(
                    nominal_member_count[positions] if nominal_member_count is not None else None
                ),
            )
            if surface.ordered_listing_ids != source.ordered_listing_ids:
                raise AlphaTargetBoundaryError(
                    "alpha_research.component_target_listing_axis_invalid"
                )
            target[positions] = (
                surface.targets["fit_target"].to_numpy().reshape(len(positions), shape[1])
            )
        ends = holding_end_sessions
    elif target_method_id == "T1_H3_PURE_TOTAL_RETURN_Z":
        raw: FloatArray = np.full(shape, np.nan, dtype=np.float64)
        ends = tuple(
            holding_end_sessions[i + 2] if i + 2 < shape[0] else None for i in range(shape[0])
        )
        if shape[0] >= 3:
            windows = np.lib.stride_tricks.sliding_window_view(raw_log_returns, 3, axis=0)
            raw[: len(windows)] = np.where(
                np.isfinite(windows).all(axis=-1),
                np.sum(np.where(np.isfinite(windows), windows, 0.0), axis=-1),
                np.nan,
            )
        target = np.array(
            scale_alpha_stock_cross_section(
                raw[:, :, None],
                source=source,
                sector_neutral=False,
                reference_eligible=reference_eligible,
            )[:, :, 0],
            copy=True,
        )
    else:
        raise AlphaTargetBoundaryError("alpha_research.component_target_method_not_installed")
    for i, end in enumerate(ends):
        if end is None:
            target[i] = np.nan
    target.setflags(write=False)
    return target, ends
