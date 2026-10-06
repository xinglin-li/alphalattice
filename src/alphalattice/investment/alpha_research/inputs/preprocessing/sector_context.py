"""Alpha's fold-fitted Sector context transformer.

Stayed in Alpha when the Sector state surface moved to Sector Research, and the
split runs exactly along the ownership line. Sector Research owns what a Sector
*was*; this owns what Alpha does to that state before fitting, which is a
training-fold decision and belongs to the Desk making it.

It consumes an authoritative surface and never recomputes Sector state. The
resulting edge -- Alpha depending on Sector Research -- is one-directional: the
surface, its contracts and its storage import nothing from Alpha, and keeping
this transformer behind is what preserves that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from alphalattice.investment.sector_research.inputs.surface import (
    SectorContextBoundaryError,
)

type FloatArray = npt.NDArray[np.float64]
type BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class SectorListingSessionScaleResult:
    """Session-local listing-axis scale and its typed numerical states."""

    values: FloatArray
    source_missing_session_feature: BoolArray
    constant_session_feature: BoolArray
    nondegenerate_session_feature: BoolArray
    listing_axis_mean_max_abs: float
    listing_axis_population_std_max_abs_error: float


def scale_mapped_sector_context_by_listing_session(
    values: FloatArray,
    *,
    reference_eligible: BoolArray | None = None,
) -> SectorListingSessionScaleResult:
    """Scale mapped Sector columns on their actual listing consumer axis.

    A missing listing invalidates that whole session/feature cell rather than
    silently changing its cross-sectional population.  A complete constant
    cell is legal and projects deterministically to zero.  This is a
    contemporaneous cross-sectional final scale, not a fold-fitted temporal
    scaler and not a second owner of the upstream Sector state.
    """
    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 3 or source.shape[1] < 2 or source.shape[2] < 1:
        raise SectorContextBoundaryError("alpha_research.sector_context_listing_scale_axis_invalid")
    if reference_eligible is None:
        return _scale_sector_reference(source)
    if reference_eligible.shape != source.shape[:2] or reference_eligible.dtype != np.bool_:
        raise SectorContextBoundaryError("alpha_research.sector_context_reference_axis_invalid")
    scaled = np.full(source.shape, np.nan, dtype=np.float64)
    missing = np.ones((source.shape[0], source.shape[2]), dtype=np.bool_)
    constant = np.zeros_like(missing)
    nondegenerate = np.zeros_like(missing)
    mean_error = std_error = 0.0
    masks, inverse = np.unique(reference_eligible, axis=0, return_inverse=True)
    for index, mask in enumerate(masks):
        days, members = np.flatnonzero(inverse == index), np.flatnonzero(mask)
        if len(members) < 2:
            continue
        covered = source[days]
        result = _scale_sector_reference(
            np.ascontiguousarray(covered[:, members]), coverage=covered
        )
        scaled[days] = result.values
        missing[days] = result.source_missing_session_feature
        constant[days] = result.constant_session_feature
        nondegenerate[days] = result.nondegenerate_session_feature
        mean_error = max(mean_error, result.listing_axis_mean_max_abs)
        std_error = max(std_error, result.listing_axis_population_std_max_abs_error)
    for array in (scaled, missing, constant, nondegenerate):
        array.setflags(write=False)
    return SectorListingSessionScaleResult(
        scaled, missing, constant, nondegenerate, mean_error, std_error
    )


def _scale_sector_reference(
    source: FloatArray, *, coverage: FloatArray | None = None
) -> SectorListingSessionScaleResult:
    """Fit once; diagnostics describe the reference, not the wider query set."""
    source_missing = ~np.isfinite(source).all(axis=1)
    complete = ~source_missing
    safe = np.where(np.isfinite(source), source, 0.0)
    means = np.mean(safe, axis=1)
    centered = source - means[:, None, :]
    population_std = np.sqrt(np.mean(np.square(centered), axis=1))
    constant = complete & (population_std == 0.0)
    nondegenerate = complete & (population_std > 0.0) & np.isfinite(population_std)
    invalid_complete = complete & ~(constant | nondegenerate)
    if bool(np.any(invalid_complete)):
        raise SectorContextBoundaryError("alpha_research.sector_context_listing_scale_nonfinite")

    scaled = np.full(source.shape, np.nan, dtype=np.float64)
    scaled = np.where(constant[:, None, :], 0.0, scaled)
    scaled = np.where(
        nondegenerate[:, None, :],
        centered / np.where(population_std[:, None, :] > 0.0, population_std[:, None, :], 1.0),
        scaled,
    )
    scaled_means = np.mean(np.where(np.isfinite(scaled), scaled, 0.0), axis=1)
    scaled_stds = np.sqrt(
        np.mean(
            np.square(np.where(np.isfinite(scaled), scaled, 0.0) - scaled_means[:, None, :]),
            axis=1,
        )
    )
    mean_error = (
        float(np.max(np.abs(scaled_means[nondegenerate]))) if bool(np.any(nondegenerate)) else 0.0
    )
    std_error = (
        float(np.max(np.abs(scaled_stds[nondegenerate] - 1.0)))
        if bool(np.any(nondegenerate))
        else 0.0
    )
    if coverage is not None:
        query = coverage - means[:, None, :]
        # A zero reference variance does not define a scale for a different
        # query value. Only the reference constant itself retains zero.
        scaled = np.where((query == 0.0) & constant[:, None, :], 0.0, np.nan)
        scaled = np.where(
            np.isfinite(query) & nondegenerate[:, None, :],
            query / np.where(population_std[:, None, :] > 0.0, population_std[:, None, :], 1.0),
            scaled,
        )
    for array in (scaled, source_missing, constant, nondegenerate):
        array.setflags(write=False)
    return SectorListingSessionScaleResult(
        values=scaled,
        source_missing_session_feature=source_missing,
        constant_session_feature=constant,
        nondegenerate_session_feature=nondegenerate,
        listing_axis_mean_max_abs=mean_error,
        listing_axis_population_std_max_abs_error=std_error,
    )


__all__ = [
    "SectorListingSessionScaleResult",
    "scale_mapped_sector_context_by_listing_session",
]
