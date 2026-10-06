"""The inert observation-session stock liquidity Formula used by overlays."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from alphalattice.foundation.feature_engine.catalog.observation_clock import NO_ECONOMIC_SKIP
from alphalattice.kernel.quant.factor_contracts import FactorFamily, FactorSpec, FactorTrack

SESSION_LIQUIDITY_METHOD_FAMILY: Final = "SESSION_LIQUIDITY"
SESSION_DOLLAR_VOLUME_ID: Final = "factor.desktop.experimental.session_dollar_volume.v1"
SESSION_DOLLAR_VOLUME_REQUIRED_FIELDS: Final = ("close_raw", "volume_raw")
SESSION_DOLLAR_VOLUME_DECLARATION: Final[Mapping[str, str]] = {
    "algorithm": "raw_close[t]*raw_volume[t]",
    "numeric_domain": "float64",
    "observation_clock": "close(t)",
}


def raw_session_dollar_volume(
    raw_close: npt.NDArray[np.float64], raw_volume: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Compute raw close times raw volume on one unchanged observation axis."""
    if raw_close.shape != raw_volume.shape:
        raise ValueError("feature_engine.session_dollar_volume_axis_mismatch")
    return np.multiply(raw_close, raw_volume, dtype=np.float64)


def session_dollar_volume(source: pd.DataFrame, _specification: FactorSpec) -> pd.Series:
    """Stock-level raw dollar volume at the observation session."""
    raw_close = pd.to_numeric(source["close_raw"], errors="coerce").to_numpy(dtype=np.float64)
    raw_volume = pd.to_numeric(source["volume_raw"], errors="coerce").to_numpy(dtype=np.float64)
    return pd.Series(raw_session_dollar_volume(raw_close, raw_volume), index=source.index)


def session_liquidity_factor_specs() -> tuple[FactorSpec, ...]:
    """Return the single registered liquidity Formula; registration is not activation."""
    return (
        FactorSpec(
            factor_id="session_dollar_volume",
            family=FactorFamily.LIQUIDITY,
            formula_ref=SESSION_DOLLAR_VOLUME_ID,
            formula="raw close[t] * raw volume[t]",
            window_sessions=1,
            lag_sessions=NO_ECONOMIC_SKIP,
            return_convention="raw-dollar-volume",
            required_fields=SESSION_DOLLAR_VOLUME_REQUIRED_FIELDS,
            literature_sources=("urn:alphalattice:research:observation-session-stock-liquidity",),
            minimum_observations=1,
            absolute_tolerance=1e-10,
            relative_tolerance=1e-10,
            track=FactorTrack.MODEL,
            core_anchor=False,
        ),
    )


__all__ = [
    "SESSION_DOLLAR_VOLUME_DECLARATION",
    "SESSION_DOLLAR_VOLUME_ID",
    "SESSION_DOLLAR_VOLUME_REQUIRED_FIELDS",
    "SESSION_LIQUIDITY_METHOD_FAMILY",
    "raw_session_dollar_volume",
    "session_dollar_volume",
    "session_liquidity_factor_specs",
]
