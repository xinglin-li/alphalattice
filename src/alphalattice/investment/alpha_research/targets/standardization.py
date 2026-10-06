"""The installed Alpha target lane standardizations."""

from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt
from scipy.special import ndtri  # type: ignore[import-untyped]

from alphalattice.kernel.quant.cross_section import (
    robust_zscore,
)
from alphalattice.kernel.validation.screening_statistics import average_ranks

from .contracts import StandardizedTargetLane

type FloatArray = npt.NDArray[np.float64]

RANK_GAUSS_STANDARDIZATION_ID = "RANK_GAUSS"
ROBUST_Z_STANDARDIZATION_ID = "ROBUST_Z"
CROSS_SECTIONAL_STD_Z_STANDARDIZATION_ID = "CROSS_SECTIONAL_STD_Z"

CROSS_SECTIONAL_STD_Z_DDOF = 1
"""Sample standard deviation, not population.

The cross-section at one formation is a sample of the investable universe, and
the canonical target definition fixes ``ddof=1`` as part of the method. The two
historical lanes use different scales -- ``RANK_GAUSS`` reports a population
``nanstd`` and ``ROBUST_Z`` reports a raw MAD -- so this constant is what stops
the new method being read as a respelling of either.
"""


def rank_gauss(values: FloatArray) -> FloatArray:
    """Map each cross-section to Gaussian quantiles of its average ranks."""
    result = np.full(values.shape, np.nan, dtype=np.float64)
    for index, row in enumerate(values):
        finite = np.isfinite(row)
        count = int(finite.sum())
        if count < 2:
            continue
        ranks = average_ranks(cast(FloatArray, row[finite]))
        result[index, finite] = ndtri((ranks - 0.5) / count)
    return result


def cross_sectional_std_z(
    values: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Divide each cross-section by its own ``ddof=1`` standard deviation.

    Returns the standardized lane and the scale that produced it, because the
    scale is not a diagnostic here: it is the quantity that converts a predicted
    ``z`` back into return units downstream, so it has to survive as evidence
    rather than be recomputed later from something similar.

    A row with fewer than two finite observations has no sample deviation and is
    left as ``NaN`` rather than filled -- the same convention ``rank_gauss``
    already uses, and the reason the caller can treat ``NaN`` scale as a typed
    availability failure instead of a number.
    """
    dispersion = np.full(values.shape[0], np.nan, dtype=np.float64)
    result = np.full(values.shape, np.nan, dtype=np.float64)
    for index, row in enumerate(values):
        finite = np.isfinite(row)
        if int(finite.sum()) < 2:
            continue
        scale = float(np.std(row[finite], ddof=CROSS_SECTIONAL_STD_Z_DDOF))
        dispersion[index] = scale
        if not np.isfinite(scale) or scale == 0.0:
            continue
        result[index, finite] = row[finite] / scale
    return cast(FloatArray, result), cast(FloatArray, dispersion)


class CrossSectionalStdZStandardization:
    """Ordinary cross-sectional standardization of an already-residual lane.

    Deliberately not the MAD-based ``ROBUST_Z`` adapter under a new name. That
    one divides by a scaled median absolute deviation; this divides by the
    sample standard deviation, and on any real cross-section the two differ.
    Installing it separately is what keeps the distinction auditable.
    """

    standardization_id = CROSS_SECTIONAL_STD_Z_STANDARDIZATION_ID

    def standardize(self, values: FloatArray, *, mad_scale: float) -> StandardizedTargetLane:
        del mad_scale  # this method has no robust-scale parameter
        lane_values, dispersion = cross_sectional_std_z(values)
        return StandardizedTargetLane(values=lane_values, dispersion=dispersion)


class RankGaussStandardization:
    """Own formation-wise rank-Gaussian target standardization."""

    standardization_id = RANK_GAUSS_STANDARDIZATION_ID

    def standardize(self, values: FloatArray, *, mad_scale: float) -> StandardizedTargetLane:
        """Rank-Gaussian transform target values and retain formation population dispersion.

        Args:
            values: Formation-by-listing target values.
            mad_scale: Interface parameter unused by this rank transform.

        Returns:
            Rank-Gaussian values and per-formation nan-aware population standard deviation.
        """
        del mad_scale
        lane_values = rank_gauss(values)
        return StandardizedTargetLane(
            values=lane_values,
            dispersion=np.nanstd(lane_values, axis=1),
        )


class RobustZStandardization:
    """Own formation-wise median/MAD robust target standardization."""

    standardization_id = ROBUST_Z_STANDARDIZATION_ID

    def standardize(self, values: FloatArray, *, mad_scale: float) -> StandardizedTargetLane:
        """Robust-z transform target values with the declared MAD scale.

        Args:
            values: Formation-by-listing target values.
            mad_scale: Declared scale used by the robust z-score owner.

        Returns:
            Robust-z values and the corresponding per-formation MAD dispersion.
        """
        lane_values, _residual_median, lane_mad = robust_zscore(values, mad_scale=mad_scale)
        return StandardizedTargetLane(values=lane_values, dispersion=lane_mad)


__all__ = [
    "RANK_GAUSS_STANDARDIZATION_ID",
    "ROBUST_Z_STANDARDIZATION_ID",
    "RankGaussStandardization",
    "RobustZStandardization",
    "rank_gauss",
]
