"""Plugin entry for the existing observation-session dollar volume computation."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from pydantic import JsonValue

from alphalattice.foundation.feature_engine.producers.factors.session_liquidity import (
    raw_session_dollar_volume,
)


def run(params: Mapping[str, JsonValue]) -> dict[str, list[float]]:
    """Compute dollar volume from the two arrays checked against the action schema."""
    values = raw_session_dollar_volume(
        np.asarray(params["close_raw"], dtype=np.float64),
        np.asarray(params["volume_raw"], dtype=np.float64),
    )
    return {"values": values.tolist()}
