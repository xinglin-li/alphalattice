"""Training-only stock-return calibration with dimensionless shrinkage."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Literal, Self

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.kernel.shared_kernel.identity import canonical_hash

type FloatArray = npt.NDArray[np.float64]


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class StockCalibrationRecipe(_Contract):
    """Seal zero-intercept normalized-moment stock-return calibration and its slope bounds.

    The recipe declares positive prior strength and nonnegative slope clipped to the admitted
    minimum/maximum. This is a calibration declaration, not a fit result.
    """

    kind: Literal["StockCalibrationRecipe"] = "StockCalibrationRecipe"
    method_id: Literal["NORMALIZED_MOMENT_NONNEGATIVE_SLOPE"] = (
        "NORMALIZED_MOMENT_NONNEGATIVE_SLOPE"
    )
    intercept_policy: Literal["ZERO"] = "ZERO"
    prior_strength_observations: float = Field(default=20.0, gt=0.0)
    minimum_slope: float = Field(default=0.0, ge=0.0)
    maximum_slope: float = Field(default=2.0, gt=0.0)
    recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, *, prior_strength_observations: float = 20.0) -> Self:
        """Seal the fixed stock-return calibration method with an explicit prior strength.

        Args:
            prior_strength_observations: Positive shrinkage prior strength; defaults to 20
                observations.

        Returns:
            Validated zero-intercept recipe with slope bounds 0 and 2 and canonical recipe_hash.

        Raises:
            pydantic.ValidationError: Prior strength or recipe consistency violates the declared
                contract.
        """
        values = {
            "kind": "StockCalibrationRecipe",
            "method_id": "NORMALIZED_MOMENT_NONNEGATIVE_SLOPE",
            "intercept_policy": "ZERO",
            "prior_strength_observations": prior_strength_observations,
            "minimum_slope": 0.0,
            "maximum_slope": 2.0,
        }
        return cls(**values, recipe_hash=canonical_hash(values))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require ordered slope bounds and exact stock-calibration recipe identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: The maximum slope is not above the minimum or recipe_hash is inconsistent.
        """
        if self.maximum_slope <= self.minimum_slope or self.recipe_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"recipe_hash"})
        ):
            raise ValueError("ALPHA_STOCK_CALIBRATION_RECIPE_INVALID")
        return self


class StockCalibrationFoldState(_Contract):
    """Record fitted stock-calibration moments and disjoint training/output session axes.

    RMS scales, normalized cross moment, shrinkage and nonnegative slope describe the retained fit.
    The enclosing evidence checks that training precedes output.
    """

    fold_index: int = Field(ge=0)
    training_sessions: tuple[date, ...] = Field(min_length=1)
    output_sessions: tuple[date, ...] = Field(min_length=1)
    training_observation_count: int = Field(ge=2)
    score_rms: float = Field(gt=0.0, allow_inf_nan=False)
    target_rms: float = Field(gt=0.0, allow_inf_nan=False)
    normalized_cross_moment: float = Field(allow_inf_nan=False)
    shrinkage_ratio: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    slope: float = Field(ge=0.0, allow_inf_nan=False)


class StockCalibrationEvidence(_Contract):
    """Seal causal fold calibration, source identities and transformed score values.

    The record retains exact hexadecimal transformed values or explicit unavailable cells. Its
    evidence identity binds the recipe, horizon, target/score lineage and fold states.
    """

    kind: Literal["StockCalibrationEvidence"] = "StockCalibrationEvidence"
    recipe: StockCalibrationRecipe
    horizon_sessions: int = Field(ge=1)
    target_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    score_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    ordered_formation_sessions: tuple[date, ...] = Field(min_length=1)
    fold_states: tuple[StockCalibrationFoldState, ...] = Field(min_length=1)
    transformed_values_hex: tuple[str | None, ...] = Field(min_length=1)
    transformed_value_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require aligned causal folds and exact transformed/calibration identities.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Session/value axes differ, output folds overlap, training leaks into output,
                transformed_value_hash is inconsistent or evidence_hash differs from the payload.
        """
        if len(self.ordered_formation_sessions) != len(self.transformed_values_hex):
            raise ValueError("ALPHA_STOCK_CALIBRATION_AXIS_INVALID")
        output = tuple(session for fold in self.fold_states for session in fold.output_sessions)
        if len(set(output)) != len(output):
            raise ValueError("ALPHA_STOCK_CALIBRATION_OUTPUT_OVERLAP")
        if any(
            set(fold.training_sessions) & set(fold.output_sessions)
            or max(fold.training_sessions) >= min(fold.output_sessions)
            for fold in self.fold_states
        ):
            raise ValueError("ALPHA_STOCK_CALIBRATION_FUTURE_FIT_LEAKAGE")
        if self.transformed_value_hash != canonical_hash(self.transformed_values_hex):
            raise ValueError("ALPHA_STOCK_CALIBRATION_VALUES_INVALID")
        if self.evidence_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"evidence_hash"})
        ):
            raise ValueError("ALPHA_STOCK_CALIBRATION_EVIDENCE_INVALID")
        return self


def anchored_expanding_calibration_folds(
    ordered_fold_row_counts: Sequence[int],
) -> tuple[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]], ...]:
    """The cross-fitting split this calibration is applied under, in one place.

    Outer fold ``k`` is calibrated on every earlier fold's rows and applied only
    to its own, so the slope reaching a row was fitted strictly before it. The
    first outer fold therefore has no calibrated output at all -- there is
    nothing before it -- which is why the row axis carries one more outer fold
    than there are calibration folds.

    Written here rather than inside a campaign runner because two consumers need
    the same split: the run that fits it and the successor that re-fits the same
    rows in return units. A rule restated in the second place is a rule that can
    drift, and the drift would be a leak.
    """
    counts = [int(value) for value in ordered_fold_row_counts]
    if len(counts) < 2 or any(value < 1 for value in counts):
        raise ValueError("ALPHA_STOCK_CALIBRATION_FOLD_ROW_COUNTS_INVALID")
    ranges: list[npt.NDArray[np.int64]] = []
    cursor = 0
    for count in counts:
        ranges.append(np.arange(cursor, cursor + count, dtype=np.int64))
        cursor += count
    return tuple(
        (np.concatenate(ranges[:index]).astype(np.int64), ranges[index])
        for index in range(1, len(ranges))
    )


def calibrate_stock_returns_cross_fitted(
    *,
    recipe: StockCalibrationRecipe,
    horizon_sessions: int,
    target_identity: str,
    score_identity: str,
    ordered_formation_sessions: tuple[date, ...],
    scores: FloatArray,
    raw_economic_returns: FloatArray,
    folds: Sequence[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]],
) -> tuple[FloatArray, StockCalibrationEvidence]:
    """Fit normalized moments on each training fold and apply only out of fold."""
    if (
        horizon_sessions < 1
        or scores.shape != raw_economic_returns.shape
        or scores.ndim != 1
        or len(scores) != len(ordered_formation_sessions)
    ):
        raise ValueError("ALPHA_STOCK_CALIBRATION_INPUT_INVALID")
    transformed: FloatArray = np.full(len(scores), np.nan, dtype=np.float64)
    states: list[StockCalibrationFoldState] = []
    for fold_index, (training, output) in enumerate(folds):
        if len(training) < 2 or len(output) < 1 or np.intersect1d(training, output).size:
            raise ValueError("ALPHA_STOCK_CALIBRATION_FOLD_INVALID")
        x = scores[training]
        y = raw_economic_returns[training]
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("ALPHA_STOCK_CALIBRATION_TRAINING_VALUES_INVALID")
        score_rms = float(np.sqrt(np.mean(np.square(x))))
        target_rms = float(np.sqrt(np.mean(np.square(y))))
        if score_rms <= 0.0 or target_rms <= 0.0:
            raise ValueError("ALPHA_STOCK_CALIBRATION_SCALE_INVALID")
        normalized_cross_moment = float(np.mean((x / score_rms) * (y / target_rms)))
        shrinkage_ratio = float(len(x) / (len(x) + recipe.prior_strength_observations))
        slope = shrinkage_ratio * normalized_cross_moment * target_rms / score_rms
        slope = float(np.clip(slope, recipe.minimum_slope, recipe.maximum_slope))
        transformed[output] = slope * scores[output]
        states.append(
            StockCalibrationFoldState(
                fold_index=fold_index,
                training_sessions=tuple(
                    dict.fromkeys(ordered_formation_sessions[int(i)] for i in training)
                ),
                output_sessions=tuple(
                    dict.fromkeys(ordered_formation_sessions[int(i)] for i in output)
                ),
                training_observation_count=len(training),
                score_rms=score_rms,
                target_rms=target_rms,
                normalized_cross_moment=normalized_cross_moment,
                shrinkage_ratio=shrinkage_ratio,
                slope=slope,
            )
        )
    encoded = tuple(None if not np.isfinite(value) else float(value).hex() for value in transformed)
    values = {
        "kind": "StockCalibrationEvidence",
        "recipe": recipe.model_dump(mode="json"),
        "horizon_sessions": horizon_sessions,
        "target_identity": target_identity,
        "score_identity": score_identity,
        "ordered_formation_sessions": [value.isoformat() for value in ordered_formation_sessions],
        "fold_states": [value.model_dump(mode="json") for value in states],
        "transformed_values_hex": list(encoded),
        "transformed_value_hash": canonical_hash(encoded),
    }
    evidence = StockCalibrationEvidence(
        recipe=recipe,
        horizon_sessions=horizon_sessions,
        target_identity=target_identity,
        score_identity=score_identity,
        ordered_formation_sessions=ordered_formation_sessions,
        fold_states=tuple(states),
        transformed_values_hex=encoded,
        transformed_value_hash=canonical_hash(encoded),
        evidence_hash=canonical_hash(values),
    )
    transformed.setflags(write=False)
    return transformed, evidence


__all__ = [
    "StockCalibrationEvidence",
    "StockCalibrationFoldState",
    "StockCalibrationRecipe",
    "anchored_expanding_calibration_folds",
    "calibrate_stock_returns_cross_fitted",
]
