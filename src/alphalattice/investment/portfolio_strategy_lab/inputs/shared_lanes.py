"""The positional lanes a public Portfolio walk reads, built from the owners' published tables.

The realized returns matrix and the sector exposures, read by the public path's resolver and
by the frozen-recipe preparations alike, so both build a lane one way (O3).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from typing import cast

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc

from alphalattice.kernel.quant.sector_history import exposure_sector_ids, sector_exposure

type FloatArray = npt.NDArray[np.float64]


class PortfolioResearchCompositionError(ValueError):
    """Stable refusal for the public path's resolution or Program compilation."""


def outcome_returns(
    table: pa.Table,
    *,
    sessions: tuple[date, ...],
    listings: tuple[str, ...],
) -> FloatArray:
    """The realized simple returns of each session and listing, in the axis's order.

    Read from an execution-outcome table, one row a session and a listing; a table that does
    not hold the axis exactly, or holds an infinite return or a loss of the whole position or
    more, is refused rather than repaired.

    Args:
        table: The outcome rows (`formation_session`, `listing_id`, `simple_return`).
        sessions: The formation sessions, in order.
        listings: The listings, in order.

    Returns:
        A read-only matrix, one row a session and one column a listing.

    Raises:
        PortfolioResearchCompositionError: The table does not hold the axis, or a value is
            invalid.
    """
    filtered = (
        table.filter(
            pc.is_in(pc.field("listing_id"), value_set=pa.array(listings, type=pa.string()))
        )
        .combine_chunks()
        .sort_by([("formation_session", "ascending"), ("listing_id", "ascending")])
    )
    expected_sessions = tuple(session for session in sessions for _ in listings)
    expected_listings = listings * len(sessions)
    if (
        tuple(cast(list[date], filtered["formation_session"].to_pylist())) != expected_sessions
        or tuple(str(value) for value in filtered["listing_id"].to_pylist()) != expected_listings
    ):
        raise PortfolioResearchCompositionError(
            "portfolio_application.execution_outcome_axis_invalid"
        )
    values: FloatArray = np.asarray(
        filtered["simple_return"].to_numpy(zero_copy_only=False),
        dtype=np.float64,
    ).reshape(len(sessions), len(listings))
    if np.isinf(values).any() or bool(np.any(np.isfinite(values) & (values <= -1.0))):
        raise PortfolioResearchCompositionError(
            "portfolio_application.execution_outcome_values_invalid"
        )
    values.setflags(write=False)
    return values


def sector_exposure_lanes(
    classification: Mapping[str, str],
    *,
    listings: tuple[str, ...],
    sessions: Sequence[date] = (),
) -> tuple[FloatArray, FloatArray]:
    """One positional projection for frozen and authored research inputs.

    Args:
        classification: Each listing's sector: a map, or a Sector history (V346).
        listings: The listings, in order.
        sessions: The formation axis; a history whose reclassification falls inside it gives
            one matrix and anchor per session, over every sector some session reads.

    Returns:
        The exposure matrix, one row a sector in sorted order and one column a listing (one per
        session, stacked, while a reclassification falls inside `sessions`), and the
        equal-weight book's exposure to each sector; both read-only.
    """
    sectors = exposure_sector_ids(classification, sessions, listings)
    matrix = sector_exposure(classification, sessions, listings, sectors)
    anchor: FloatArray = matrix @ np.full(len(listings), 1.0 / len(listings), dtype=np.float64)
    anchor.setflags(write=False)
    return matrix, anchor


__all__ = [
    "PortfolioResearchCompositionError",
    "outcome_returns",
    "sector_exposure_lanes",
]
