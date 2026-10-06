"""The robust cross-section mathematics: MAD winsor, Sector demean and robust z-score.

The median/MAD winsor, the equal-weight Sector demean and the median/MAD z-score, with the
constants that parameterize them. Shared numerics under their own name (UC): the Panel's
preprocessing, Alpha's targets and features and Sector's surface compute with them, and each
of those identities binds this module, never the Feature engine's Panel materialization
around it.
"""

from __future__ import annotations

import warnings
from typing import cast

import numpy as np

MAD_SCALE = 1.4826
WINSOR_MULTIPLIER = 5.0
MIN_COVERAGE = 0.98
MIN_SECTOR_SAMPLE = 5


def median_mad_winsor(
    values: np.ndarray,
    *,
    multiplier: float = WINSOR_MULTIPLIER,
    mad_scale: float = MAD_SCALE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply the Panel's row-wise robust clipping policy to a numeric matrix."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(values, axis=1)
        mad = np.nanmedian(np.abs(values - median[:, None]), axis=1)
    lower = median - multiplier * mad_scale * mad
    upper = median + multiplier * mad_scale * mad
    return np.clip(values, lower[:, None], upper[:, None]), median, mad, lower, upper


def equal_sector_demean(
    values: np.ndarray,
    *,
    sector_positions: tuple[np.ndarray, ...],
    session_local: bool = False,
) -> np.ndarray:
    """Subtract the equal-weight center under the recorded reduction contract.

    The historical numpy reduction depends on a block's memory layout: a
    one-session block can use pairwise summation while a taller one does not.
    Current execution sums in member order independently for each session.
    The historical branch remains for artifacts bound to the earlier policy.
    """
    residual = np.full_like(values, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for positions in sector_positions:
            block = values[:, positions]
            if session_local:
                count = np.count_nonzero(~np.isnan(block), axis=1)
                total: np.ndarray = np.zeros(len(block), dtype=np.float64)
                for column in block.T:
                    total += np.where(np.isnan(column), 0.0, column)
                center = np.divide(
                    total,
                    count,
                    out=np.full(len(block), np.nan),
                    where=count != 0,
                )
            else:
                center = np.nanmean(block, axis=1)
            residual[:, positions] = block - center[:, None]
    return cast(np.ndarray, residual)


def robust_zscore(
    values: np.ndarray, *, mad_scale: float = MAD_SCALE
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the Panel's row-wise median/MAD standardization."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(values, axis=1)
        mad = np.nanmedian(np.abs(values - median[:, None]), axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        score = (values - median[:, None]) / (mad_scale * mad[:, None])
    return score, median, mad


__all__ = [
    "MAD_SCALE",
    "MIN_COVERAGE",
    "MIN_SECTOR_SAMPLE",
    "WINSOR_MULTIPLIER",
    "equal_sector_demean",
    "median_mad_winsor",
    "robust_zscore",
]
