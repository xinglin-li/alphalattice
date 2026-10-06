"""Sector-residual reversal candidate over a Host-resolved daily Sector child."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import numpy as np
import pandas as pd

from alphalattice.foundation.feature_engine.catalog.observation_clock import (
    NO_ECONOMIC_SKIP,
)
from alphalattice.kernel.quant.factor_contracts import FactorFamily, FactorSpec, FactorTrack

from .series_math import grouped_pair_rolling_apply, log_return, ordered_frame

RESIDUAL_REVERSAL_METHOD_FAMILY: Final = "RESIDUAL_REVERSAL"
RESIDUAL_REVERSAL_VOL_SCALED_5_ID: Final = (
    "factor.desktop.experimental.residual_reversal_vol_scaled_5.v1"
)
RESIDUAL_REVERSAL_REQUIRED_FIELDS: Final = (
    "provider_adjusted_close",
    "sector_return_log",
)
RESIDUAL_REVERSAL_DECLARATION: Final[Mapping[str, str]] = {
    "implementation_id": RESIDUAL_REVERSAL_VOL_SCALED_5_ID,
    "algorithm": "-sum(residual[t-4:t])/(sample_std(residual[t-62:t])*sqrt(5))",
    "sector_child": "Host-resolved current-membership daily equal-weight log return",
}


def residual_reversal_factor_specs() -> tuple[FactorSpec, ...]:
    """Describe volatility-scaled five-session stock-minus-sector reversal.

    Returns:
        The residual-reversal specification, with its declared source interval. Recipes retain their
        declared source fields, economic skips,
        minimum ordered observations, tolerances, and implementation references.
    """
    return (
        FactorSpec(
            factor_id="residual_reversal_vol_scaled_5",
            family=FactorFamily.REVERSAL,
            formula_ref=RESIDUAL_REVERSAL_VOL_SCALED_5_ID,
            formula="-sum_5(stock_log_return-sector_log_return)/(std_63*sqrt(5))",
            window_sessions=63,
            # No economic skip: the residual window ends at the observation
            # session itself. The session it used to give up was execution safety
            # the entry recipe already provides.
            lag_sessions=NO_ECONOMIC_SKIP,
            return_convention="provider_adjusted_sector_residual_log_return",
            required_fields=RESIDUAL_REVERSAL_REQUIRED_FIELDS,
            literature_sources=("https://doi.org/10.1111/j.1540-6261.1990.tb05106.x",),
            minimum_observations=64,
            absolute_tolerance=1e-10,
            relative_tolerance=1e-10,
            track=FactorTrack.MODEL,
            core_anchor=False,
        ),
    )


def residual_reversal_vol_scaled_5(source: pd.DataFrame, specification: FactorSpec) -> pd.Series:
    """Scale the negated last five stock-minus-sector returns by trailing residual volatility.

    Args:
        source: Listing/session rows with provider-adjusted closes and sector log returns.
        specification: Recipe whose window and economic skip select complete finite source pairs.

    Returns:
        Negative five-residual sum over sample window deviation times sqrt(5).
        The recipe's window and economic skip determine the complete finite source interval. Results
        follow the input row axis; unavailable windows remain missing.

    Raises:
        KeyError: A required listing, session, or source column is absent.
    """
    ordered = ordered_frame(source)
    groups = ordered["listing_id"]
    stock = log_return(ordered["provider_adjusted_close"], groups)
    sector = pd.to_numeric(ordered["sector_return_log"], errors="coerce").astype(float)
    residual = stock - sector

    def evaluate(values: np.ndarray, _unused: np.ndarray) -> float:
        scale = float(np.std(values, ddof=1))
        return -float(np.sum(values[-5:])) / (scale * np.sqrt(5.0)) if scale > 0 else np.nan

    values = grouped_pair_rolling_apply(
        residual,
        residual,
        groups,
        window=specification.window_sessions,
        skip=specification.lag_sessions,
        function=evaluate,
    )
    ordered["_value"] = values.to_numpy(dtype=float)
    restored = ordered.sort_values("_source_position", kind="mergesort")
    return pd.Series(restored["_value"].to_numpy(dtype=float), index=source.index)


__all__ = [
    "RESIDUAL_REVERSAL_DECLARATION",
    "RESIDUAL_REVERSAL_METHOD_FAMILY",
    "RESIDUAL_REVERSAL_REQUIRED_FIELDS",
    "RESIDUAL_REVERSAL_VOL_SCALED_5_ID",
    "residual_reversal_factor_specs",
    "residual_reversal_vol_scaled_5",
]
