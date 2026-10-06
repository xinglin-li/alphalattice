"""Sector safety bands and the stratified candidate pool that can satisfy them.

A global Top-K and a per-Sector floor are two different admission questions, and
answering the first while enforcing the second is how a floor becomes infeasible
for reasons no error message explains: the top 50 names by score can easily
contain nobody from a Sector whose lower bound is two percent.

So the pool is built *inside* each Sector, with hysteresis, and the bands are
derived from the current-Universe equal-weight reference rather than from
anything the Alpha or Sector forecast says. The bands are a static safety limit;
letting a forecast set them would encode the same view twice -- once in the
objective and once in the constraint -- and the second copy would be invisible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

type FloatArray = npt.NDArray[np.float64]
type BoolArray = npt.NDArray[np.bool_]
type IntArray = npt.NDArray[np.int64]

MINIMUM_SECTOR_ENTER = 3
"""``K_min`` from the canonical plan: a Sector never admits fewer than three."""


class SectorAdmissionError(ValueError):
    """Stable fail-closed boundary for band and pool construction."""


@dataclass(frozen=True, slots=True)
class SectorSafetyBands:
    """The static outer bands one trial is solved under."""

    absolute_deviation: float
    relative_deviation: float
    reference: FloatArray
    lower: FloatArray
    upper: FloatArray

    @property
    def identity(self) -> dict[str, object]:
        """Project exact declared sector deviation controls and numerical lower/upper bands.

        Returns:
            JSON-compatible mapping of absolute/relative tolerances and ordered bounds.
        """
        return {
            "absolute_deviation": self.absolute_deviation,
            "relative_deviation": self.relative_deviation,
            "lower": [float(value) for value in self.lower],
            "upper": [float(value) for value in self.upper],
        }


def derive_sector_safety_bands(
    *,
    equal_weight_sector_exposure: FloatArray,
    absolute_deviation: float,
    relative_deviation: float,
) -> SectorSafetyBands:
    """``d_g = d_abs + rho * b0_g``; ``L_g = max(0, b0-d)``; ``U_g = min(1, b0+d)``."""
    reference = np.asarray(equal_weight_sector_exposure, dtype=np.float64)
    if (
        reference.ndim != 1
        or reference.size < 1
        or not np.isfinite(reference).all()
        or bool(np.any(reference < 0.0))
        or absolute_deviation < 0.0
        or relative_deviation < 0.0
    ):
        raise SectorAdmissionError("portfolio_strategy_lab.band_reference_invalid")
    deviation = absolute_deviation + relative_deviation * reference
    lower = np.maximum(0.0, reference - deviation)
    upper = np.minimum(1.0, reference + deviation)
    lower.setflags(write=False)
    upper.setflags(write=False)
    return SectorSafetyBands(
        absolute_deviation=absolute_deviation,
        relative_deviation=relative_deviation,
        reference=reference,
        lower=lower,
        upper=upper,
    )


def sector_enter_exit_counts(
    *, upper: FloatArray, maximum_weight: float
) -> tuple[IntArray, IntArray]:
    """``K_enter = max(3, ceil(U_g / w_max) + 2)``; ``K_exit = max(K+2, ceil(1.5K))``."""
    if maximum_weight <= 0.0:
        raise SectorAdmissionError("portfolio_strategy_lab.band_cap_invalid")
    enter = np.maximum(
        MINIMUM_SECTOR_ENTER,
        np.ceil(np.asarray(upper, dtype=np.float64) / maximum_weight).astype(np.int64) + 2,
    ).astype(np.int64)
    exit_counts = np.maximum(enter + 2, np.ceil(1.5 * enter).astype(np.int64)).astype(np.int64)
    enter.setflags(write=False)
    exit_counts.setflags(write=False)
    return enter, exit_counts


def stratified_admitted_pool(
    *,
    scores: FloatArray,
    decision_eligible: BoolArray,
    reference_weights: FloatArray,
    sector_by_asset: IntArray,
    enter_counts: IntArray,
    exit_counts: IntArray,
    holding_tolerance: float = 1e-8,
) -> BoolArray:
    """Rank within each Sector, admit inside ``K_enter``, retain until ``K_exit``.

    Hysteresis is the whole point of the two counts. Admitting and dropping on
    one threshold makes a name at the boundary trade every formation it crosses
    it, and that turnover is paid in real costs for no change of view.

    An untradable existing holding is admitted regardless of rank, because it
    cannot be sold: excluding it would not remove the position, it would only
    hide the position from the Sector feasibility arithmetic that has to account
    for it.
    """
    asset_count = scores.size
    if (
        decision_eligible.shape != (asset_count,)
        or reference_weights.shape != (asset_count,)
        or sector_by_asset.shape != (asset_count,)
        or enter_counts.shape != exit_counts.shape
    ):
        raise SectorAdmissionError("portfolio_strategy_lab.pool_axis_invalid")
    held = reference_weights > holding_tolerance
    admitted: BoolArray = np.zeros(asset_count, dtype=np.bool_)
    for sector in range(int(enter_counts.size)):
        members = np.flatnonzero(sector_by_asset == sector)
        if members.size == 0:
            continue
        ranked = members[
            np.lexsort((members, -np.where(np.isfinite(scores[members]), scores[members], -np.inf)))
        ]
        rank = {int(value): index for index, value in enumerate(ranked)}
        enter = int(enter_counts[sector])
        keep = int(exit_counts[sector])
        for asset in members:
            position = rank[int(asset)]
            eligible = bool(decision_eligible[asset]) and bool(np.isfinite(scores[asset]))
            if not bool(held[asset]):
                admitted[asset] = eligible and position < enter
                continue
            if not eligible:
                # Frozen: it is in the book and cannot be traded out of it.
                admitted[asset] = True
                continue
            admitted[asset] = position < keep
    admitted.setflags(write=False)
    return admitted


__all__ = [
    "MINIMUM_SECTOR_ENTER",
    "SectorAdmissionError",
    "SectorSafetyBands",
    "derive_sector_safety_bands",
    "sector_enter_exit_counts",
    "stratified_admitted_pool",
]
