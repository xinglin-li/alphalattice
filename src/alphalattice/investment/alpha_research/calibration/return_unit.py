"""The Alpha-owned return-unit calibration successor, and why the first one was wrong.

Stage 3 fitted ``y = slope * z`` on the fold-selected predicted residual Z and
the canonical raw economic return, and published a slope in *return per unit
predicted Z*. That is a real number and it was applied correctly, but it makes a
claim the method never justified: that one constant converts Z to return across
every formation.

It does not. Z is a residual standardized by the canonical target's own
cross-sectional dispersion, and that dispersion is a per-session quantity which
moves by a factor of three between quiet and stressed formations. The inverse of
the standardization is ``sigma_XS * z``, formation by formation, so the constant
that belongs in front of it is **dimensionless** -- return per return -- and the
session-varying part belongs to the scale surface the target compiler already
publishes.

The composition is therefore::

    x = cross_sectional_dispersion(formation) * predicted_z(formation, listing)
    expected_simple_return = slope * x,   slope dimensionless, bounded, >= 0

and the earlier evidence is left readable and marked
``SUPERSEDED_FOR_RETURN_UNIT_COMPOSITION``. It is not deleted and not corrected
in place: it is what a published decision consumed, and a downstream reader must
be able to see both the number that was used and the number that replaced it.

Two things this deliberately does **not** do. It does not refit a model -- the
predictions are Stage 3's own, read from its published surfaces. And it does not
invent a fitting rule: it calls the installed bounded nonnegative calibration
policy with a different ``x``, so what changed is the quantity being calibrated
rather than how calibration works.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from hashlib import sha256
from typing import Literal, Self, cast

import numpy as np
import numpy.typing as npt
import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.investment.alpha_research.experiments.selected_scores import (
    AlphaSelectedRowAxis,
    AlphaSelectedScoreProjection,
)
from alphalattice.investment.alpha_research.targets.materialization import (
    CanonicalTargetMaterialization,
)
from alphalattice.kernel.shared_kernel.identity import (
    canonical_hash,
    successor_identity_payload,
)

from .stock_returns import (
    StockCalibrationEvidence,
    StockCalibrationRecipe,
    anchored_expanding_calibration_folds,
    calibrate_stock_returns_cross_fitted,
)

type FloatArray = npt.NDArray[np.float64]
type IntArray = npt.NDArray[np.int64]

_HASH = r"^[0-9a-f]{64}$"

RETURN_UNIT_CALIBRATION_METHOD_ID = "DISPERSION_SCALED_NORMALIZED_MOMENT_NONNEGATIVE_SLOPE"
RETURN_UNIT_COMPOSITION = (
    "expected_simple_return = slope * cross_sectional_dispersion * predicted_z"
)
SUPERSEDED_DISPOSITION = "SUPERSEDED_FOR_RETURN_UNIT_COMPOSITION"
RETURN_UNIT_CALIBRATION_CATEGORY = "return-unit-calibrations"


class AlphaReturnUnitCalibrationError(ValueError):
    """Stable fail-closed boundary raised before any slope is fitted or published."""


def _value_identity(values: FloatArray) -> str:
    """Byte identity of one lane, in the canonical compiler's own construction.

    The same shape/dtype/content digest the target compiler seals lane
    identities with, so a rebuilt lane can be held against the published digest
    rather than against a second convention that happens to agree today.
    """

    contiguous = np.ascontiguousarray(values, dtype=np.float64)
    return str(
        canonical_hash(
            {
                "dtype": str(contiguous.dtype),
                "shape": [int(value) for value in contiguous.shape],
                "content": sha256(contiguous.tobytes()).hexdigest(),
            }
        )
    )


_CALIBRATION_SUCCESSOR_FIELDS = ("capability_binding_hash", "implementation_closure_hash")
"""Fields a pre-authority calibration does not carry. See ``successor_identity_payload``."""


class AlphaReturnUnitCalibrationEvidence(BaseModel):  # type: ignore[misc]
    """One dimensionless calibration, bound to everything that produced it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["AlphaReturnUnitCalibrationEvidence"] = "AlphaReturnUnitCalibrationEvidence"
    identity_class: Literal["DEVELOPMENT_EVIDENCE_ONLY_NEVER_PUBLISHED"] = (
        "DEVELOPMENT_EVIDENCE_ONLY_NEVER_PUBLISHED"
    )
    schema_version: Literal["alpha-return-unit-calibration@1"] = "alpha-return-unit-calibration@1"
    method_id: Literal["DISPERSION_SCALED_NORMALIZED_MOMENT_NONNEGATIVE_SLOPE"] = (
        "DISPERSION_SCALED_NORMALIZED_MOMENT_NONNEGATIVE_SLOPE"
    )
    composition: Literal[
        "expected_simple_return = slope * cross_sectional_dispersion * predicted_z"
    ] = "expected_simple_return = slope * cross_sectional_dispersion * predicted_z"
    slope_units: Literal["DIMENSIONLESS_RETURN_PER_RETURN"] = "DIMENSIONLESS_RETURN_PER_RETURN"
    """Named, because the defect this succeeds was exactly a unit that nobody named."""

    # --- the Alpha decision this consumes ----------------------------------
    dossier_hash: str = Field(pattern=_HASH)
    decision_receipt_hash: str = Field(pattern=_HASH)
    program_hash: str = Field(pattern=_HASH)
    selected_score_projection_hash: str = Field(pattern=_HASH)
    horizon_sessions: int = Field(ge=1)

    # --- the canonical target lineage --------------------------------------
    target_evidence_hash: str = Field(pattern=_HASH)
    target_recipe_binding_hash: str = Field(pattern=_HASH)
    outcome_method_binding_hash: str = Field(pattern=_HASH)
    maturity_lag_sessions: int = Field(ge=2)
    canonical_lane_identity_hash: str = Field(pattern=_HASH)
    simple_economic_return_lane_identity: str = Field(pattern=_HASH)
    cross_sectional_dispersion_lane_identity: str = Field(pattern=_HASH)

    # --- the exact row axis -------------------------------------------------
    row_axis_hash: str = Field(pattern=_HASH)
    row_count: int = Field(ge=2)
    ordered_fold_row_counts: tuple[int, ...] = Field(min_length=2)
    row_session_axis_hash: str = Field(pattern=_HASH)
    """Digest of the per-row formation session list.

    The same list the superseded evidence carries, and checked against it before
    fitting. Two calibrations of the same decision that disagree about which rows
    they ran on are not comparable, and nothing else in either artifact would
    show it.
    """

    # --- what was actually fitted ------------------------------------------
    scaled_score_identity: str = Field(pattern=_HASH)
    economic_return_identity: str = Field(pattern=_HASH)
    calibration: StockCalibrationEvidence

    # --- what it replaces ---------------------------------------------------
    superseded_calibration_evidence_hash: str = Field(pattern=_HASH)
    superseded_disposition: Literal["SUPERSEDED_FOR_RETURN_UNIT_COMPOSITION"] = (
        "SUPERSEDED_FOR_RETURN_UNIT_COMPOSITION"
    )
    superseded_slope_units: Literal["RETURN_PER_UNIT_PREDICTED_Z"] = "RETURN_PER_UNIT_PREDICTED_Z"
    superseded_slope_by_fold: tuple[float, ...] = Field(min_length=1)
    """Kept beside the successor's own slopes. Both readable, neither corrected."""

    numerical_environment_hash: str = Field(pattern=_HASH)

    # --- the installed authority that produced it ---------------------------
    capability_binding_hash: str | None = None
    """Identity of the installed capability the sealer resolved before fitting.

    ``None`` only on an artifact published before this was durable. Such a
    document says which numbers it holds and nothing about what was allowed to
    produce them, so a verifier can read it and can never admit it: re-deriving
    it under today's method and calling the result the same artifact would be an
    upgrade nobody performed.

    Written by the Alpha authority owner, never by the numerical function below.
    A primitive that filled this in would be asserting its own authority, which
    is the shape of the defect rather than a fix for it.
    """

    implementation_closure_hash: str | None = None
    """The exact implementation bytes, carried beside the capability hash.

    Redundant with it by construction and worth the field anyway: when a
    capability hash stops matching, this is what says *whether the code moved* --
    a source change and an environment change are different events with different
    answers, and one opaque hash cannot tell a reader which happened.
    """

    evidence_hash: str = Field(pattern=_HASH)

    @property
    def slope_by_fold(self) -> tuple[float, ...]:
        """Read each retained calibration fold slope in fold-state order.

        Returns:
            Float slopes of the sealed calibration fold states.
        """
        return tuple(float(value.slope) for value in self.calibration.fold_states)

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal return-unit calibration using the declared successor-compatible payload.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical evidence_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        draft = dict(values)
        draft.pop("evidence_hash", None)
        provisional = cls.model_construct(**draft, evidence_hash="0" * 64)
        identity = successor_identity_payload(
            provisional.model_dump(mode="json", exclude={"evidence_hash"}),
            _CALIBRATION_SUCCESSOR_FIELDS,
        )
        return cls(**draft, evidence_hash=str(canonical_hash(identity)))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require aligned fold rows, exact source authority and compatible calibration identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            AlphaReturnUnitCalibrationError: Row/fold/session counts differ, score/target identities
                disagree, capability/closure authority is partial or evidence_hash differs from the
                successor-compatible payload.
        """
        if sum(self.ordered_fold_row_counts) != self.row_count:
            raise AlphaReturnUnitCalibrationError(
                "alpha_research.return_unit_calibration_row_counts_invalid"
            )
        if len(self.calibration.fold_states) != len(self.ordered_fold_row_counts) - 1:
            raise AlphaReturnUnitCalibrationError(
                "alpha_research.return_unit_calibration_fold_count_invalid"
            )
        if len(self.calibration.ordered_formation_sessions) != self.row_count:
            raise AlphaReturnUnitCalibrationError(
                "alpha_research.return_unit_calibration_axis_mismatch"
            )
        if self.calibration.target_identity != self.target_evidence_hash:
            raise AlphaReturnUnitCalibrationError(
                "alpha_research.return_unit_calibration_target_not_bound"
            )
        if self.calibration.score_identity != self.scaled_score_identity:
            raise AlphaReturnUnitCalibrationError(
                "alpha_research.return_unit_calibration_score_not_bound"
            )
        if (self.capability_binding_hash is None) != (self.implementation_closure_hash is None):
            # Half an authority is not a weaker authority, it is an unreadable
            # one: a document naming a capability without the code it ran, or the
            # reverse, cannot be placed on either side of the legacy boundary.
            raise AlphaReturnUnitCalibrationError(
                "alpha_research.return_unit_calibration_authority_partial"
            )
        if self.evidence_hash != canonical_hash(
            successor_identity_payload(
                self.model_dump(mode="json", exclude={"evidence_hash"}),
                _CALIBRATION_SUCCESSOR_FIELDS,
            )
        ):
            raise AlphaReturnUnitCalibrationError(
                "alpha_research.return_unit_calibration_identity_invalid"
            )
        return self


@dataclass(frozen=True, slots=True)
class AlphaReturnUnitCalibration:
    """The sealed successor, and the applied values it produced, on the row axis."""

    evidence: AlphaReturnUnitCalibrationEvidence
    scaled_scores: FloatArray
    economic_returns: FloatArray
    applied_expected_returns: FloatArray
    """``slope * x`` out of fold; NaN on the first outer fold, which has no fit."""


def _lane(table: pa.Table, column: str, rows: int, columns: int) -> FloatArray:
    values = np.asarray(
        table[column].combine_chunks().to_numpy(zero_copy_only=False), dtype=np.float64
    )
    if values.size != rows * columns:
        raise AlphaReturnUnitCalibrationError("alpha_research.return_unit_calibration_lane_shape")
    return cast(FloatArray, values.reshape(rows, columns))


def calibrate_alpha_return_unit_signal(
    *,
    projection: AlphaSelectedScoreProjection,
    row_axis: AlphaSelectedRowAxis,
    materialization: CanonicalTargetMaterialization,
    superseded: StockCalibrationEvidence,
    numerical_environment_hash: str,
    recipe: StockCalibrationRecipe | None = None,
) -> AlphaReturnUnitCalibration:
    """Fit the dimensionless slope on Stage 3's own rows, training regions only.

    Every refusal below happens before the first arithmetic. A wrong lane, a
    wrong session, a wrong listing, a missing row and a duplicated row all
    produce arrays of exactly the right shape and dtype, and a fit would succeed
    on any of them; the only thing that separates the right one is that the
    identities it was built from reconcile.
    """
    evidence = materialization.evidence
    if projection.target_evidence_hash != evidence.evidence_hash:
        raise AlphaReturnUnitCalibrationError(
            "alpha_research.return_unit_calibration_target_not_this_projection"
        )
    if projection.target_recipe_binding_hash != materialization.recipe_binding.binding_hash:
        raise AlphaReturnUnitCalibrationError(
            "alpha_research.return_unit_calibration_binding_not_this_projection"
        )
    if projection.outcome_method_binding_hash != materialization.outcome_method_binding_hash:
        raise AlphaReturnUnitCalibrationError(
            "alpha_research.return_unit_calibration_outcome_not_this_projection"
        )
    if superseded.target_identity != evidence.evidence_hash:
        raise AlphaReturnUnitCalibrationError(
            "alpha_research.return_unit_calibration_superseded_target_mismatch"
        )
    if superseded.horizon_sessions != projection.horizon_sessions:
        raise AlphaReturnUnitCalibrationError(
            "alpha_research.return_unit_calibration_superseded_horizon_mismatch"
        )
    # The exact-axis check, against durable evidence rather than a count: the
    # superseded calibration carries one formation session per row, so an axis
    # that is the right length but the wrong rows fails here.
    if tuple(superseded.ordered_formation_sessions) != row_axis.row_sessions:
        raise AlphaReturnUnitCalibrationError(
            "alpha_research.return_unit_calibration_row_axis_not_stage_three"
        )

    surface = materialization.surface
    sessions = tuple(surface.formation_sessions)
    listings = tuple(surface.ordered_listing_ids)
    economic = _lane(surface.targets, "simple_economic_return", len(sessions), len(listings))
    # The lane is taken by name and then held against the identity the
    # reconciliation established, so a surface carrying a substituted column of
    # the same shape is refused rather than fitted.
    if _value_identity(economic) != materialization.simple_economic_return_identity:
        raise AlphaReturnUnitCalibrationError(
            "alpha_research.return_unit_calibration_economic_lane_substituted"
        )
    dispersion = np.asarray(
        surface.dispersion["cross_sectional_dispersion"]
        .combine_chunks()
        .to_numpy(zero_copy_only=False),
        dtype=np.float64,
    )
    if dispersion.size != len(sessions):
        raise AlphaReturnUnitCalibrationError(
            "alpha_research.return_unit_calibration_dispersion_axis_invalid"
        )

    session_position = {value: index for index, value in enumerate(sessions)}
    listing_position = {value: index for index, value in enumerate(listings)}
    try:
        rows: IntArray = np.asarray(
            [session_position[value] for value in row_axis.row_sessions], dtype=np.int64
        )
        columns: IntArray = np.asarray(
            [listing_position[value] for value in row_axis.row_listing_ids], dtype=np.int64
        )
    except KeyError as error:
        # A row naming a session or listing the canonical surface does not carry.
        raise AlphaReturnUnitCalibrationError(
            "alpha_research.return_unit_calibration_row_outside_target_axis"
        ) from error

    scale = dispersion[rows]
    if not bool(np.isfinite(scale).all()) or bool(np.any(scale <= 0.0)):
        # An unavailable formation carries no scale. Substituting one would be
        # inventing the conversion this calibration exists to measure.
        raise AlphaReturnUnitCalibrationError(
            "alpha_research.return_unit_calibration_scale_unavailable"
        )
    scaled: FloatArray = np.ascontiguousarray(scale * row_axis.predicted_z, dtype=np.float64)
    economic_rows: FloatArray = np.ascontiguousarray(economic[rows, columns], dtype=np.float64)
    if not bool(np.isfinite(economic_rows).all()):
        raise AlphaReturnUnitCalibrationError(
            "alpha_research.return_unit_calibration_economic_return_unavailable"
        )

    folds = anchored_expanding_calibration_folds(row_axis.ordered_fold_row_counts)
    applied, calibration = calibrate_stock_returns_cross_fitted(
        recipe=recipe or StockCalibrationRecipe.create(),
        horizon_sessions=projection.horizon_sessions,
        target_identity=evidence.evidence_hash,
        score_identity=_value_identity(scaled),
        ordered_formation_sessions=row_axis.row_sessions,
        scores=scaled,
        raw_economic_returns=economic_rows,
        folds=folds,
    )
    sealed = AlphaReturnUnitCalibrationEvidence.create(
        dossier_hash=projection.dossier_hash,
        decision_receipt_hash=projection.decision_receipt_hash,
        program_hash=projection.program_hash,
        selected_score_projection_hash=projection.projection_hash,
        horizon_sessions=projection.horizon_sessions,
        target_evidence_hash=evidence.evidence_hash,
        target_recipe_binding_hash=materialization.recipe_binding.binding_hash,
        outcome_method_binding_hash=materialization.outcome_method_binding_hash,
        maturity_lag_sessions=materialization.maturity_lag_sessions,
        canonical_lane_identity_hash=evidence.lane_identity_hash,
        simple_economic_return_lane_identity=materialization.simple_economic_return_identity,
        cross_sectional_dispersion_lane_identity=evidence.cross_sectional_dispersion_identity,
        row_axis_hash=row_axis.row_axis_hash,
        row_count=row_axis.row_count,
        ordered_fold_row_counts=row_axis.ordered_fold_row_counts,
        row_session_axis_hash=str(
            canonical_hash([value.isoformat() for value in row_axis.row_sessions])
        ),
        scaled_score_identity=_value_identity(scaled),
        economic_return_identity=_value_identity(economic_rows),
        calibration=calibration,
        superseded_calibration_evidence_hash=superseded.evidence_hash,
        superseded_slope_by_fold=tuple(float(value.slope) for value in superseded.fold_states),
        numerical_environment_hash=numerical_environment_hash,
    )
    scaled.setflags(write=False)
    economic_rows.setflags(write=False)
    return AlphaReturnUnitCalibration(
        evidence=sealed,
        scaled_scores=scaled,
        economic_returns=economic_rows,
        applied_expected_returns=applied,
    )


def applied_expected_return_matrix(
    *,
    calibration_values: FloatArray,
    row_axis: AlphaSelectedRowAxis,
    formation_sessions: tuple[date, ...],
    ordered_listing_ids: tuple[str, ...],
) -> FloatArray:
    """Project the out-of-fold applied signal onto one formation-by-listing axis.

    NaN where the row axis has nothing to say -- the first outer fold, whose rows
    were never calibrated, and any cell outside the requested axis. A consumer
    that needs coverage checks it; nothing here fills a hole.
    """
    session_position = {value: index for index, value in enumerate(formation_sessions)}
    listing_position = {value: index for index, value in enumerate(ordered_listing_ids)}
    matrix: FloatArray = np.full(
        (len(formation_sessions), len(ordered_listing_ids)), np.nan, dtype=np.float64
    )
    applied = np.asarray(calibration_values, dtype=np.float64)
    if applied.shape != (row_axis.row_count,):
        raise AlphaReturnUnitCalibrationError("alpha_research.return_unit_applied_axis_mismatch")
    for index, (session, listing) in enumerate(
        zip(row_axis.row_sessions, row_axis.row_listing_ids, strict=True)
    ):
        row = session_position.get(session)
        column = listing_position.get(listing)
        if row is None or column is None:
            continue
        matrix[row, column] = applied[index]
    matrix = np.ascontiguousarray(matrix, dtype=np.float64)
    matrix.setflags(write=False)
    return matrix


__all__ = [
    "RETURN_UNIT_CALIBRATION_CATEGORY",
    "RETURN_UNIT_CALIBRATION_METHOD_ID",
    "RETURN_UNIT_COMPOSITION",
    "SUPERSEDED_DISPOSITION",
    "AlphaReturnUnitCalibration",
    "AlphaReturnUnitCalibrationError",
    "AlphaReturnUnitCalibrationEvidence",
    "applied_expected_return_matrix",
    "calibrate_alpha_return_unit_signal",
]
