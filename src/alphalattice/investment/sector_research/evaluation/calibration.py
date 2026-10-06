"""``NONNEGATIVE_SECTOR_SHRINK_TO_ZERO_CALIBRATION``, owned by this Desk.

Before a Sector forecast may reach Portfolio it has to answer a different
question from "is the ranking any good": *what scale, if any, should this
forecast carry against the money lane*. That question is settled here rather
than by the Portfolio composer, because a consumer that picks its own scale for
someone else's forecast is refitting the method.

The calibration is a slope through the origin from a method's forecast onto the
raw Sector economic-return lane -- the simple-return sleeve the target evidence
already publishes as its diagnostic, not the log-return modelling centre -- and
it is constrained to ``[0, 1]``::

    slope = clip( sum(forecast * economic) / sum(forecast * forecast), 0, 1 )

No intercept: an intercept would let the calibration contribute a Sector return
level that the forecast never predicted, which is a second method hiding inside
a scale. The upper bound at one says a calibration may shrink a forecast but
never lever it up; the lower bound at zero says the honest answer to "no daily
predictive scale" is to close the contribution, and a slope that lands there is
an admissible scientific result rather than an operational failure.

Cross-fitted, in contiguous time folds. A slope fitted on the same pairs it then
scales would report the in-sample fit of a one-parameter regression and call it
evidence; every calibrated value here is produced by a slope fitted on the other
folds. Folds are contiguous and time-ordered rather than random, because
neighbouring sessions of the same sector are not independent draws and a shuffled
split would leak the local level across the boundary.

The identity control -- slope fixed at one -- is carried alongside so the
comparison can state what the calibration was worth, which is the pre-registered
question the plan asks this Desk to answer.
"""

from __future__ import annotations

from typing import Literal, Self, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.kernel.shared_kernel.identity import canonical_hash

from ..contracts import FloatArray, SectorResearchError, grid_to_matrix
from ..experiments.development_artifacts import SectorForecastSurface
from ..targets.execution import SectorTargetEvidence

SECTOR_SHRINK_CALIBRATION_METHOD_ID = "NONNEGATIVE_SECTOR_SHRINK_TO_ZERO_CALIBRATION"
SECTOR_IDENTITY_SCALE_CONTROL_ID = "IDENTITY_SCALE_CONTROL"


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class SectorShrinkCalibrationFold(_Contract):
    """One held-out block, the slope fitted without it, and what that scored."""

    fold_index: int = Field(ge=0)
    fitted_pair_count: int = Field(ge=0)
    evaluated_pair_count: int = Field(ge=0)
    slope: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    calibrated_squared_error_sum: float = Field(ge=0.0, allow_inf_nan=False)
    identity_squared_error_sum: float = Field(ge=0.0, allow_inf_nan=False)
    zero_squared_error_sum: float = Field(ge=0.0, allow_inf_nan=False)


class SectorShrinkCalibrationEvidence(_Contract):
    """The published calibration for one method's forecast surface.

    ``mean_calibrated_economic_squared_error`` is the number the Campaign
    compares across methods: every value in it was scaled by a slope fitted
    without the fold it scored.
    """

    kind: Literal["SectorShrinkCalibrationEvidence"] = "SectorShrinkCalibrationEvidence"
    calibration_method_id: Literal["NONNEGATIVE_SECTOR_SHRINK_TO_ZERO_CALIBRATION"] = (
        "NONNEGATIVE_SECTOR_SHRINK_TO_ZERO_CALIBRATION"
    )
    surface_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_evidence_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    method_id: str = Field(min_length=1, max_length=96)
    economic_lane_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    fold_count: int = Field(ge=2)
    folds: tuple[SectorShrinkCalibrationFold, ...] = Field(min_length=2)
    evaluated_pair_count: int = Field(ge=0)
    pooled_slope: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    """Fitted on every pair. Reported for interpretation only -- never used to
    scale a value, because nothing would be held out of it."""

    mean_cross_fitted_slope: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    mean_calibrated_economic_squared_error: float | None = Field(
        default=None, ge=0.0, allow_inf_nan=False
    )
    mean_identity_economic_squared_error: float | None = Field(
        default=None, ge=0.0, allow_inf_nan=False
    )
    mean_zero_economic_squared_error: float | None = Field(
        default=None, ge=0.0, allow_inf_nan=False
    )
    calibration_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require aligned fold counts, score availability and exact calibration identity.

        Returns:
            This contract after checking its declared consistency rules.

        Raises:
            SectorResearchError: Fold indices/counts, evaluated-pair totals, empty-score cells or
                calibration_hash are inconsistent.
        """
        if tuple(value.fold_index for value in self.folds) != tuple(range(len(self.folds))):
            raise SectorResearchError("sector_research.calibration_fold_axis_invalid")
        if self.fold_count != len(self.folds):
            raise SectorResearchError("sector_research.calibration_fold_count_invalid")
        if self.evaluated_pair_count != sum(value.evaluated_pair_count for value in self.folds):
            raise SectorResearchError("sector_research.calibration_pair_count_invalid")
        empty = self.evaluated_pair_count == 0
        if empty != (self.mean_calibrated_economic_squared_error is None) or empty != (
            self.mean_zero_economic_squared_error is None
        ):
            raise SectorResearchError("sector_research.calibration_score_cell_invalid")
        if self.calibration_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"calibration_hash"})
        ):
            raise SectorResearchError("sector_research.calibration_identity_invalid")
        return self


def _origin_slope(forecast: FloatArray, economic: FloatArray) -> float:
    """The ``[0, 1]``-constrained through-origin slope, zero when there is no scale.

    A denominator at zero is a forecast that is identically zero wherever it is
    evaluated -- the ZERO control, or a method that refused everywhere. It has
    no scale to estimate, and zero is both the correct answer and the one the
    bound would force anyway.
    """

    denominator = float(np.dot(forecast, forecast))
    if denominator <= 0.0:
        return 0.0
    return float(min(1.0, max(0.0, float(np.dot(forecast, economic)) / denominator)))


def _contiguous_folds(row_count: int, *, fold_count: int) -> tuple[tuple[int, int], ...]:
    """Half-open row ranges, contiguous and time-ordered, sizes differing by at most one."""

    if fold_count < 2 or row_count <= 0:
        raise SectorResearchError("sector_research.calibration_fold_plan_invalid")
    base, remainder = divmod(row_count, fold_count)
    bounds: list[tuple[int, int]] = []
    start = 0
    for index in range(fold_count):
        stop = start + base + (1 if index < remainder else 0)
        bounds.append((start, stop))
        start = stop
    return tuple(bounds)


def calibrate_sector_forecast_surface(
    *,
    evidence: SectorTargetEvidence,
    surface: SectorForecastSurface,
    fold_count: int,
) -> SectorShrinkCalibrationEvidence:
    """Derive the calibration from the two artifacts it relates, and nothing else.

    Pure and adapter-free like the evaluation beside it, so the verifier can
    recompute it byte-for-byte from the published children.
    """
    if surface.target_evidence_hash != evidence.evidence_hash:
        raise SectorResearchError("sector_research.calibration_target_mismatch")
    if surface.ordered_sectors != evidence.ordered_sectors:
        raise SectorResearchError("sector_research.calibration_sector_axis_mismatch")

    session_index = {value: index for index, value in enumerate(evidence.formation_sessions)}
    economic_matrix = grid_to_matrix(evidence.economic_diagnostic_values)
    forecast_matrix = grid_to_matrix(surface.values)

    # Pairs stay grouped by forecast row so a fold is a block of time rather
    # than a block of pairs: rows differ in how many sectors were available.
    row_pairs: list[tuple[FloatArray, FloatArray]] = []
    for row, formation in enumerate(surface.forecast_formation_sessions):
        target_row = session_index.get(formation)
        if target_row is None:
            raise SectorResearchError("sector_research.calibration_session_not_in_target")
        forecasts = forecast_matrix[row, :]
        economics = economic_matrix[target_row, :]
        usable = np.isfinite(forecasts) & np.isfinite(economics)
        row_pairs.append((forecasts[usable], economics[usable]))

    bounds = _contiguous_folds(len(row_pairs), fold_count=fold_count)
    folds: list[SectorShrinkCalibrationFold] = []
    for index, (start, stop) in enumerate(bounds):
        held_out = row_pairs[start:stop]
        fitted = [*row_pairs[:start], *row_pairs[stop:]]
        fit_forecast = (
            np.concatenate([value for value, _ in fitted])
            if fitted
            else np.zeros(0, dtype=np.float64)
        )
        fit_economic = (
            np.concatenate([value for _, value in fitted])
            if fitted
            else np.zeros(0, dtype=np.float64)
        )
        slope = _origin_slope(fit_forecast, fit_economic)
        out_forecast = (
            np.concatenate([value for value, _ in held_out])
            if held_out
            else np.zeros(0, dtype=np.float64)
        )
        out_economic = (
            np.concatenate([value for _, value in held_out])
            if held_out
            else np.zeros(0, dtype=np.float64)
        )
        folds.append(
            SectorShrinkCalibrationFold(
                fold_index=index,
                fitted_pair_count=int(fit_forecast.size),
                evaluated_pair_count=int(out_forecast.size),
                slope=slope,
                calibrated_squared_error_sum=float(
                    np.sum(np.square(slope * out_forecast - out_economic))
                ),
                identity_squared_error_sum=float(np.sum(np.square(out_forecast - out_economic))),
                zero_squared_error_sum=float(np.sum(np.square(out_economic))),
            )
        )

    evaluated = sum(value.evaluated_pair_count for value in folds)
    all_forecast = np.concatenate([value for value, _ in row_pairs]) if row_pairs else None
    all_economic = np.concatenate([value for _, value in row_pairs]) if row_pairs else None
    values: dict[str, object] = {
        "kind": "SectorShrinkCalibrationEvidence",
        "calibration_method_id": SECTOR_SHRINK_CALIBRATION_METHOD_ID,
        "surface_hash": surface.surface_hash,
        "target_evidence_hash": evidence.evidence_hash,
        "method_id": surface.recipe.method_id,
        "economic_lane_identity": evidence.economic_diagnostic_identity,
        "fold_count": len(folds),
        "folds": [value.model_dump(mode="json") for value in folds],
        "evaluated_pair_count": evaluated,
        "pooled_slope": (
            _origin_slope(all_forecast, all_economic)
            if all_forecast is not None and all_economic is not None
            else 0.0
        ),
        # The mean over folds rather than over pairs: each fold contributes one
        # slope, and weighting them by held-out size would let the largest block
        # speak for the method.
        "mean_cross_fitted_slope": float(
            np.mean(np.asarray([value.slope for value in folds], dtype=np.float64))
        ),
        "mean_calibrated_economic_squared_error": (
            sum(value.calibrated_squared_error_sum for value in folds) / evaluated
            if evaluated
            else None
        ),
        "mean_identity_economic_squared_error": (
            sum(value.identity_squared_error_sum for value in folds) / evaluated
            if evaluated
            else None
        ),
        "mean_zero_economic_squared_error": (
            sum(value.zero_squared_error_sum for value in folds) / evaluated if evaluated else None
        ),
    }
    return cast(
        SectorShrinkCalibrationEvidence,
        SectorShrinkCalibrationEvidence.model_validate(
            {**values, "calibration_hash": canonical_hash(values)}
        ),
    )


__all__ = [
    "SECTOR_IDENTITY_SCALE_CONTROL_ID",
    "SECTOR_SHRINK_CALIBRATION_METHOD_ID",
    "SectorShrinkCalibrationEvidence",
    "SectorShrinkCalibrationFold",
    "calibrate_sector_forecast_surface",
]
