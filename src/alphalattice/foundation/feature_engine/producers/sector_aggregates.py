"""The Sector's daily equal-weight log return each member reads: `sector_return_log`.

The Sector child the residual reversal declares ("Host-resolved current-membership daily
equal-weight log return", `observation_clock.SECTOR_AGGREGATE_AUTHORITY`): at each session, the
mean of the one-session log returns of that session's members in each Sector, each member in the
Sector it reads that day (`kernel/quant/sector_history`), and every member given its Sector's mean.
A Sector none of whose members has a finite return that session gives its members none, and a
listing outside a session's members has none. A formula reads `sector_return_log`, and its study's
temporal statement describes the Sector treatment.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date

import numpy as np
import numpy.typing as npt

from alphalattice.kernel.quant.sector_history import sector_positions

type FloatArray = npt.NDArray[np.float64]
type BoolArray = npt.NDArray[np.bool_]


def sector_return_log(
    adjusted_close: FloatArray,
    *,
    sessions: Sequence[date],
    listing_ids: Sequence[str],
    sectors: Mapping[str, str],
    members: BoolArray | None = None,
) -> FloatArray:
    """Each member's Sector's equal-weight one-session log return, with the day's Sectors.

    Args:
        adjusted_close: The provider's adjusted closes, one row a session and one column a
            listing, in the axes' orders.
        sessions: The session axis, ascending and without repeats.
        listing_ids: The listing axis.
        sectors: A Sector history, or a map every session reads.
        members: Each session's members, the same shape; every listing when omitted.

    Returns:
        The Sector returns, the closes' shape: the first session's none, a non-member's none.

    Raises:
        ValueError: `feature_engine.sector_return_axis_invalid` when the shapes disagree.
    """
    closes = np.asarray(adjusted_close, dtype=np.float64)
    shape = (len(sessions), len(listing_ids))
    if closes.shape != shape or (members is not None and members.shape != shape):
        raise ValueError("feature_engine.sector_return_axis_invalid")
    held: BoolArray = np.ones(shape, dtype=np.bool_) if members is None else members
    returns: FloatArray = np.full(shape, np.nan, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        returns[1:] = np.log(closes[1:] / closes[:-1])
    returns[~np.isfinite(returns)] = np.nan
    result: FloatArray = np.full(shape, np.nan, dtype=np.float64)
    for rows, _sector_ids, groups in sector_positions(sectors, sessions, listing_ids):
        block = returns[rows]
        block_members = held[rows]
        for group in groups:
            counted = np.isfinite(block[:, group]) & block_members[:, group]
            counts = counted.sum(axis=1)
            totals = np.where(counted, block[:, group], 0.0).sum(axis=1)
            means = np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)
            values = np.where(block_members[:, group], means[:, None], np.nan)
            result[rows, group] = values
    result.setflags(write=False)
    return result


__all__ = ["sector_return_log"]
