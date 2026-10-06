"""Development Alpha evidence contracts independent of Agent and publication."""

from __future__ import annotations

import math
from datetime import date
from enum import StrEnum
from hashlib import sha256
from itertools import pairwise
from typing import Any, Literal, Self

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.sealing import seal_model

from ..evaluation.contracts import AlphaCandidateMetrics, AlphaFoldMetrics
from ..targets.authority import InstalledAlphaTargetMethodBinding
from ..targets.development import (
    AlphaDevelopmentTargetMaterializationBinding,
)
from .contracts import (
    AlphaCandidateFailure,
    AlphaCandidateRole,
    AlphaCandidateStatus,
    AlphaDevelopmentSplitPolicy,
    AlphaFitLedgerEntry,
)

type _ScoreArray = npt.NDArray[np.float64]


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


def seal_current_contract[ContractT: BaseModel](
    model: type[ContractT],
    values: dict[str, Any],
    identity_field: str,
) -> ContractT:
    return seal_model(model, values, field=identity_field)


class AlphaSelectionAction(StrEnum):
    SELECT_REGISTERED_ALPHA = "SELECT_REGISTERED_ALPHA"
    STOP_WITH_EVIDENCE = "STOP_WITH_EVIDENCE"
    REQUEST_HUMAN_REVIEW = "REQUEST_HUMAN_REVIEW"


class AlphaDevelopmentSurfaceBinding(_Contract):
    kind: Literal["AlphaDevelopmentSurfaceBinding"] = "AlphaDevelopmentSurfaceBinding"
    foundation_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    logical_panel_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    logical_semantic_index_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    causal_outcome_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    listing_set_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    ordered_listing_ids_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    ordered_factor_ids_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    split_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    fold_commitment_hashes: tuple[str, ...] = Field(min_length=1)
    binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> AlphaDevelopmentSurfaceBinding:
        if self.binding_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"binding_hash"})
        ):
            raise ValueError("Alpha development-surface binding hash is invalid")
        return self


class AlphaCandidateExecutionBinding(_Contract):
    kind: Literal["AlphaCandidateExecutionBinding"] = "AlphaCandidateExecutionBinding"
    development_surface_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_id: str
    candidate_card_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    fold_commitment_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    algorithm_package_identity: str
    execution_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> AlphaCandidateExecutionBinding:
        if self.execution_binding_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"execution_binding_hash"})
        ):
            raise ValueError("Alpha candidate execution binding hash is invalid")
        return self


class AlphaParentRequestBinding(_Contract):
    kind: Literal["AlphaParentRequestBinding"] = "AlphaParentRequestBinding"
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    pm_plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    effort_plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    inventory_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    development_surface_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    ordered_candidate_ids: tuple[str, ...] = Field(min_length=1)
    ordered_candidate_card_hashes: tuple[str, ...] = Field(min_length=1)
    binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> AlphaParentRequestBinding:
        if len(self.ordered_candidate_ids) != len(self.ordered_candidate_card_hashes):
            raise ValueError("Alpha parent request candidate axis is incomplete")
        if self.binding_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"binding_hash"})
        ):
            raise ValueError("Alpha parent request binding hash is invalid")
        return self


class AlphaDevelopmentSurfaceManifest(_Contract):
    kind: Literal["AlphaDevelopmentSurfaceManifest"] = "AlphaDevelopmentSurfaceManifest"
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    foundation_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    logical_panel_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    logical_semantic_index_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    causal_outcome_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    listing_set_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    ordered_listing_ids_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    ordered_factor_ids_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_ids: tuple[str, ...] = Field(min_length=1)
    candidate_card_hashes: tuple[str, ...] = Field(min_length=1)
    fold_commitment_hashes: tuple[str, ...] = Field(min_length=1)
    split_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    surface_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> AlphaDevelopmentSurfaceManifest:
        if self.candidate_ids != tuple(dict.fromkeys(self.candidate_ids)):
            raise ValueError("Alpha development candidate axis is not unique")
        if len(self.candidate_ids) != len(self.candidate_card_hashes):
            raise ValueError("Alpha development candidate cards do not reconcile")
        if self.surface_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"surface_hash"})
        ):
            raise ValueError("Alpha development surface hash is invalid")
        return self


class AlphaSessionScoreStatistics(_Contract):
    formation_session: date
    listing_count: int = Field(ge=1)
    scored_count: int = Field(ge=0)
    score_mean: float = Field(allow_inf_nan=False)
    score_std: float = Field(ge=0, allow_inf_nan=False)
    score_coverage: float = Field(ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_counts(self) -> AlphaSessionScoreStatistics:
        if self.scored_count > self.listing_count or not math.isclose(
            self.score_coverage,
            self.scored_count / self.listing_count,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("Alpha session score statistics do not reconcile")
        return self


class AlphaMatrixDiagnostic(_Contract):
    scope: Literal["DEVELOPMENT_FOLD", "CURRENT_REFIT"]
    fold_index: int | None = Field(default=None, ge=0)
    first_session: date
    last_session: date
    row_count: int = Field(ge=1)
    complete_row_count: int = Field(ge=1)
    factor_count: int = Field(ge=1)
    effective_rank: int = Field(ge=0)
    condition_number: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    largest_singular_value: float = Field(ge=0, allow_inf_nan=False)
    smallest_retained_singular_value: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    label_mean: float = Field(allow_inf_nan=False)
    label_std: float = Field(ge=0, allow_inf_nan=False)
    training_mse: float = Field(ge=0, allow_inf_nan=False)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_scope(self) -> AlphaMatrixDiagnostic:
        if (self.scope == "DEVELOPMENT_FOLD") != (self.fold_index is not None):
            raise ValueError("Alpha matrix diagnostic fold identity is inconsistent")
        if self.first_session > self.last_session or self.complete_row_count > self.row_count:
            raise ValueError("Alpha matrix diagnostic boundaries are inconsistent")
        return self


class AlphaFactorContributionDiagnostic(_Contract):
    factor_id: str = Field(min_length=1)
    development_median_coefficient: float = Field(allow_inf_nan=False)
    current_coefficient: float = Field(allow_inf_nan=False)
    current_feature_mean: float = Field(allow_inf_nan=False)
    current_mean_contribution: float = Field(allow_inf_nan=False)


class AlphaCurrentRefitDiagnosticReport(_Contract):
    kind: Literal["AlphaCurrentRefitDiagnosticReport"] = "AlphaCurrentRefitDiagnosticReport"
    prior_runtime_marker_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    foundation_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    selected_candidate_id: str
    development: tuple[AlphaMatrixDiagnostic, ...] = Field(min_length=1)
    current: AlphaMatrixDiagnostic
    factor_contributions: tuple[AlphaFactorContributionDiagnostic, ...] = Field(min_length=1)
    failed_stability_check_ids: tuple[str, ...]
    attribution: Literal[
        "DATA_OR_AXIS_ERROR",
        "IMPLEMENTATION_ERROR",
        "OBSERVED_MODEL_INSTABILITY",
    ]
    downstream_expansion_permitted: bool
    report_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> AlphaCurrentRefitDiagnosticReport:
        if self.downstream_expansion_permitted != (
            self.attribution == "OBSERVED_MODEL_INSTABILITY"
        ):
            raise ValueError("Alpha diagnostic disposition is inconsistent")
        if self.report_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"report_hash"})
        ):
            raise ValueError("Alpha current-refit diagnostic hash is invalid")
        return self


class AlphaEstimatorState(_Contract):
    """One fitted estimator's state in a qualification or the current refit.

    An installed family's state carries its kind's diagnostics (LINEAR coefficients, TREE
    gains). An agent's model is bound to its adapter's projection instead (V342): its adapter,
    numerical binding, fit evidence and projected state, its family the model's own, and a kind
    other than LINEAR or TREE carries no diagnostics. The projection fields are absent from an
    installed family's state, so its identity is what it was.
    """

    kind: Literal["AlphaEstimatorState"] = "AlphaEstimatorState"
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_id: str
    scope: Literal["DEVELOPMENT_FOLD", "CURRENT_REFIT"]
    family_id: str = Field(default="ridge", min_length=1, max_length=96)
    state_kind: str = Field(default="LINEAR", min_length=1, max_length=64)
    adapter_id: str | None = Field(
        default=None, min_length=1, max_length=96, exclude_if=lambda v: v is None
    )
    numerical_binding_hash: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$", exclude_if=lambda v: v is None
    )
    fit_evidence_hash: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$", exclude_if=lambda v: v is None
    )
    state_projection_hash: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$", exclude_if=lambda v: v is None
    )
    state_schema_id: str | None = Field(
        default=None, min_length=1, max_length=128, exclude_if=lambda v: v is None
    )
    projection_payload: dict[str, object] | None = Field(
        default=None, exclude_if=lambda v: v is None
    )
    fold_index: int | None = Field(default=None, ge=0)
    ordered_factor_ids: tuple[str, ...] = Field(min_length=1)
    coefficient_hex: tuple[str, ...] = ()
    intercept_hex: str = "0x0.0p+0"
    coefficient_l2_norm: float = Field(ge=0, allow_inf_nan=False)
    coefficient_max_abs: float = Field(ge=0, allow_inf_nan=False)
    nonzero_support_count: int | None = Field(default=None, ge=0)
    model_text_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    best_iteration: int | None = Field(default=None, ge=1)
    feature_gain_hex: tuple[str, ...] = ()
    top_feature_gain_share: float | None = Field(default=None, ge=0, le=1)
    training_mse: float = Field(ge=0, allow_inf_nan=False)
    validation_score_mean: float = Field(allow_inf_nan=False)
    validation_score_std: float = Field(ge=0, allow_inf_nan=False)
    validation_score_coverage: float = Field(ge=0, le=1, allow_inf_nan=False)
    validation_session_statistics: tuple[AlphaSessionScoreStatistics, ...] = Field(min_length=1)
    state_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> AlphaEstimatorState:
        if self.ordered_factor_ids != tuple(dict.fromkeys(self.ordered_factor_ids)):
            raise ValueError("Alpha estimator factor axis is not unique")
        projection = (
            self.adapter_id,
            self.numerical_binding_hash,
            self.fit_evidence_hash,
            self.state_projection_hash,
            self.state_schema_id,
            self.projection_payload,
        )
        bound = all(value is not None for value in projection)
        if not bound and any(value is not None for value in projection):
            raise ValueError("Alpha estimator projection binding is incomplete")
        if bound and self.state_projection_hash != canonical_hash(
            {
                "adapter_id": self.adapter_id,
                "model_family_id": self.family_id,
                "state_kind": self.state_kind,
                "state_schema_id": self.state_schema_id,
                "payload": self.projection_payload,
            }
        ):
            raise ValueError("Alpha estimator state projection identity changed")
        if self.state_kind not in {"LINEAR", "TREE"}:
            if not bound or (
                self.coefficient_hex
                or self.coefficient_l2_norm != 0.0
                or self.coefficient_max_abs != 0.0
                or self.nonzero_support_count is not None
                or self.model_text_hash is not None
                or self.best_iteration is not None
                or self.feature_gain_hex
                or self.top_feature_gain_share is not None
            ):
                raise ValueError("Alpha declared estimator state is incomplete")
        elif self.state_kind == "LINEAR":
            if (
                (not bound and self.family_id not in {"ridge", "lasso", "elastic_net"})
                or len(self.ordered_factor_ids) != len(self.coefficient_hex)
                or self.model_text_hash is not None
                or self.best_iteration is not None
                or self.feature_gain_hex
                or self.top_feature_gain_share is not None
            ):
                raise ValueError("Alpha linear estimator state is incomplete")
            try:
                coefficients = tuple(float.fromhex(value) for value in self.coefficient_hex)
                intercept = float.fromhex(self.intercept_hex)
            except ValueError as error:
                raise ValueError("Alpha estimator float identity is invalid") from error
            expected_l2 = math.sqrt(sum(value * value for value in coefficients))
            expected_max = max(abs(value) for value in coefficients)
            expected_support = sum(value != 0.0 for value in coefficients)
            if (
                not all(math.isfinite(value) for value in (*coefficients, intercept))
                or not math.isclose(
                    self.coefficient_l2_norm, expected_l2, rel_tol=0.0, abs_tol=1e-12
                )
                or not math.isclose(
                    self.coefficient_max_abs, expected_max, rel_tol=0.0, abs_tol=0.0
                )
                or (
                    self.nonzero_support_count is not None
                    and self.nonzero_support_count != expected_support
                )
            ):
                raise ValueError("Alpha estimator coefficient diagnostics changed")
        else:
            if (
                (not bound and self.family_id != "lightgbm")
                or self.coefficient_hex
                or self.nonzero_support_count is not None
                or self.model_text_hash is None
                or self.best_iteration is None
                or len(self.feature_gain_hex) != len(self.ordered_factor_ids)
                or self.top_feature_gain_share is None
                or self.coefficient_l2_norm != 0.0
                or self.coefficient_max_abs != 0.0
            ):
                raise ValueError("Alpha tree estimator state is incomplete")
            gains = tuple(float.fromhex(value) for value in self.feature_gain_hex)
            total = sum(gains)
            expected_share = max(gains) / total if total > 0.0 else 0.0
            if not all(
                math.isfinite(value) and value >= 0.0 for value in gains
            ) or not math.isclose(
                self.top_feature_gain_share,
                expected_share,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("Alpha tree gain diagnostics changed")
        if (self.scope == "DEVELOPMENT_FOLD") != (self.fold_index is not None):
            raise ValueError("Alpha estimator fold identity is inconsistent")
        sessions = tuple(value.formation_session for value in self.validation_session_statistics)
        if sessions != tuple(sorted(set(sessions))):
            raise ValueError("Alpha estimator session statistics are not canonical")
        identity = self.model_dump(mode="json", exclude={"state_hash"})
        accepted = {canonical_hash(identity)}
        if self.family_id == "ridge" and self.state_kind == "LINEAR":
            legacy = dict(identity)
            for field in (
                "family_id",
                "state_kind",
                "nonzero_support_count",
                "model_text_hash",
                "best_iteration",
                "feature_gain_hex",
                "top_feature_gain_share",
            ):
                legacy.pop(field, None)
            accepted.add(canonical_hash(legacy))
        if self.state_hash not in accepted:
            raise ValueError("Alpha estimator state hash is invalid")
        return self


class AlphaDevelopmentEstimatorState(_Contract):
    """Estimator state whose identity is independent of the parent Alpha request."""

    kind: Literal["AlphaDevelopmentEstimatorState"] = "AlphaDevelopmentEstimatorState"
    execution_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    development_surface_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_id: str
    candidate_card_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    fold_index: int = Field(ge=0)
    fold_commitment_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    adapter_id: str | None = Field(default=None, min_length=1, max_length=96)
    numerical_binding_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    fit_evidence_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    state_projection_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    state_schema_id: str | None = Field(default=None, min_length=1, max_length=128)
    projection_payload: dict[str, object] | None = None
    family_id: str = Field(min_length=1, max_length=96)
    state_kind: str = Field(min_length=1, max_length=64)
    ordered_factor_ids: tuple[str, ...] = Field(min_length=1)
    coefficient_hex: tuple[str, ...] = ()
    intercept_hex: str = "0x0.0p+0"
    coefficient_l2_norm: float = Field(ge=0, allow_inf_nan=False)
    coefficient_max_abs: float = Field(ge=0, allow_inf_nan=False)
    nonzero_support_count: int | None = Field(default=None, ge=0)
    model_text_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    best_iteration: int | None = Field(default=None, ge=1)
    feature_gain_hex: tuple[str, ...] = ()
    top_feature_gain_share: float | None = Field(default=None, ge=0, le=1)
    training_mse: float = Field(ge=0, allow_inf_nan=False)
    validation_score_mean: float = Field(allow_inf_nan=False)
    validation_score_std: float = Field(ge=0, allow_inf_nan=False)
    validation_score_coverage: float = Field(ge=0, le=1, allow_inf_nan=False)
    validation_session_statistics: tuple[AlphaSessionScoreStatistics, ...] = Field(min_length=1)
    state_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> AlphaDevelopmentEstimatorState:
        if self.ordered_factor_ids != tuple(dict.fromkeys(self.ordered_factor_ids)):
            raise ValueError("Alpha development estimator factor axis is not unique")
        projection_fields = (
            self.adapter_id,
            self.state_projection_hash,
            self.state_schema_id,
            self.projection_payload,
        )
        if any(value is not None for value in projection_fields):
            if any(value is None for value in projection_fields):
                raise ValueError("Alpha development state projection binding is incomplete")
            projection_identity = {
                "adapter_id": self.adapter_id,
                "model_family_id": self.family_id,
                "state_kind": self.state_kind,
                "state_schema_id": self.state_schema_id,
                "payload": self.projection_payload,
            }
            if self.state_projection_hash != canonical_hash(projection_identity):
                raise ValueError("Alpha development state projection identity changed")
        if self.state_kind == "LINEAR":
            if (
                len(self.ordered_factor_ids) != len(self.coefficient_hex)
                or self.model_text_hash is not None
                or self.best_iteration is not None
                or self.feature_gain_hex
                or self.top_feature_gain_share is not None
            ):
                raise ValueError("Alpha development linear estimator state is incomplete")
            try:
                coefficients = tuple(float.fromhex(value) for value in self.coefficient_hex)
                intercept = float.fromhex(self.intercept_hex)
            except ValueError as error:
                raise ValueError("Alpha development estimator float identity is invalid") from error
            expected_l2 = math.sqrt(sum(value * value for value in coefficients))
            expected_max = max(abs(value) for value in coefficients)
            expected_support = sum(value != 0.0 for value in coefficients)
            if (
                not all(math.isfinite(value) for value in (*coefficients, intercept))
                or not math.isclose(
                    self.coefficient_l2_norm, expected_l2, rel_tol=0.0, abs_tol=1e-12
                )
                or not math.isclose(
                    self.coefficient_max_abs, expected_max, rel_tol=0.0, abs_tol=0.0
                )
                or (
                    self.nonzero_support_count is not None
                    and self.nonzero_support_count != expected_support
                )
            ):
                raise ValueError("Alpha development estimator diagnostics changed")
        elif self.state_kind == "TREE":
            if (
                self.coefficient_hex
                or self.nonzero_support_count is not None
                or self.model_text_hash is None
                or self.best_iteration is None
                or len(self.feature_gain_hex) != len(self.ordered_factor_ids)
                or self.top_feature_gain_share is None
                or self.coefficient_l2_norm != 0.0
                or self.coefficient_max_abs != 0.0
            ):
                raise ValueError("Alpha development tree estimator state is incomplete")
            gains = tuple(float.fromhex(value) for value in self.feature_gain_hex)
            total = sum(gains)
            expected_share = max(gains) / total if total > 0.0 else 0.0
            if not all(
                math.isfinite(value) and value >= 0.0 for value in gains
            ) or not math.isclose(
                self.top_feature_gain_share,
                expected_share,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("Alpha development tree gain diagnostics changed")
        elif any(
            value is None
            for value in (
                self.adapter_id,
                self.numerical_binding_hash,
                self.fit_evidence_hash,
                self.state_projection_hash,
                self.state_schema_id,
                self.projection_payload,
            )
        ):
            raise ValueError("Alpha development generic estimator binding is incomplete")
        sessions = tuple(value.formation_session for value in self.validation_session_statistics)
        if sessions != tuple(sorted(set(sessions))):
            raise ValueError("Alpha development estimator sessions are not canonical")
        identity = self.model_dump(mode="json", exclude={"state_hash"})
        accepted = {canonical_hash(identity)}
        legacy = dict(identity)
        for field in (
            "adapter_id",
            "numerical_binding_hash",
            "fit_evidence_hash",
            "state_projection_hash",
            "state_schema_id",
            "projection_payload",
        ):
            if legacy.get(field) is None:
                legacy.pop(field, None)
        accepted.add(canonical_hash(legacy))
        if self.state_hash not in accepted:
            raise ValueError("Alpha development estimator state hash is invalid")
        return self


class AlphaDevelopmentValidationChunkRef(_Contract):
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    development_surface_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    fold_index: int = Field(ge=0)
    fold_commitment_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    row_count: int = Field(ge=1)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    metadata_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    uri: str


class AlphaCandidateDevelopmentScoreChunkRef(_Contract):
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    development_surface_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_id: str
    candidate_card_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    fold_index: int = Field(ge=0)
    fold_commitment_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    row_count: int = Field(ge=1)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    metadata_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    uri: str


class AlphaNumericalDevelopmentScoreChunkRef(_Contract):
    """Parent-independent score payload owned by one candidate/fold execution."""

    execution_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    development_surface_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_id: str
    candidate_card_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    fold_index: int = Field(ge=0)
    fold_commitment_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    row_count: int = Field(ge=1)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    metadata_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    uri: str


class AlphaDevelopmentFoldSurface(_Contract):
    kind: Literal["AlphaDevelopmentFoldSurface"] = "AlphaDevelopmentFoldSurface"
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    development_surface_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    fold_index: int = Field(ge=0)
    fold_commitment_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    validation_chunk: AlphaDevelopmentValidationChunkRef
    surface_fold_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> AlphaDevelopmentFoldSurface:
        if self.validation_chunk.fold_index != self.fold_index:
            raise ValueError("Alpha validation surface fold identity differs")
        if self.surface_fold_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"surface_fold_hash"})
        ):
            raise ValueError("Alpha validation surface hash is invalid")
        return self


class AlphaCandidateFoldEvidence(_Contract):
    """Thin parent-owned binding to one reusable numerical fold result."""

    kind: Literal["AlphaCandidateFoldEvidence"] = "AlphaCandidateFoldEvidence"
    execution_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    numerical_result_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    development_surface_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    surface_fold_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_id: str
    candidate_card_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    fold_index: int = Field(ge=0)
    fold_commitment_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    fold_evidence_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> AlphaCandidateFoldEvidence:
        if self.fold_evidence_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"fold_evidence_hash"})
        ):
            raise ValueError("Alpha candidate fold evidence hash is invalid")
        return self


class LegacyAlphaCandidateFoldEvidence(_Contract):
    """Read-only compatibility for parent artifacts published before thin references."""

    kind: Literal["AlphaCandidateFoldEvidence"] = "AlphaCandidateFoldEvidence"
    execution_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    numerical_result_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    development_surface_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    surface_fold_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_id: str
    candidate_card_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    role: AlphaCandidateRole
    fold_index: int = Field(ge=0)
    fold_commitment_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: AlphaCandidateStatus
    score_chunk: (
        AlphaCandidateDevelopmentScoreChunkRef | AlphaNumericalDevelopmentScoreChunkRef | None
    ) = None
    metrics: AlphaFoldMetrics | None = None
    session_rank_ics: tuple[float, ...]
    session_spreads: tuple[float, ...]
    fit_ledger: AlphaFitLedgerEntry | None = None
    estimator_state_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    failure: AlphaCandidateFailure | None = None
    fold_evidence_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> LegacyAlphaCandidateFoldEvidence:
        succeeded = self.status is AlphaCandidateStatus.SUCCEEDED
        if succeeded != (
            self.score_chunk is not None
            and self.metrics is not None
            and self.fit_ledger is not None
            and self.failure is None
        ):
            raise ValueError("Legacy Alpha candidate fold evidence is incomplete")
        if self.score_chunk is not None and (
            self.score_chunk.candidate_id != self.candidate_id
            or self.score_chunk.fold_index != self.fold_index
        ):
            raise ValueError("Legacy Alpha candidate score chunk binding differs")
        identity = self.model_dump(mode="json", exclude={"fold_evidence_hash"})
        accepted = {canonical_hash(identity)}
        without_numerical_reference = dict(identity)
        without_numerical_reference.pop("numerical_result_hash", None)
        accepted.add(canonical_hash(without_numerical_reference))
        if self.fold_evidence_hash not in accepted:
            raise ValueError("Legacy Alpha candidate fold evidence hash is invalid")
        return self

    def restates(self, numerical: AlphaCandidateNumericalFoldResult) -> bool:
        """Whether the numerical facts this evidence repeats are the numerical result's.

        A parent of this era wrote the fold's role, status, metrics, session
        series, fit ledger, failure and references beside the numerical result
        that owns them. Two files each carrying a valid hash prove nothing about
        each other; this is the equality that does. An evidence naming its
        numerical result must repeat every fact and reference of that result
        exactly. One that names none predates that reference, not necessarily
        the numerical chunk: a numerical score chunk is one execution's
        payload and must be this result's whether or not the result is named,
        and an estimator reference must be this result's when present. A
        chunk of the older shape is another payload identity, so its content
        hash is never the numerical chunk's; it is held to the bindings it
        carries -- the request and surface this evidence itself binds, the
        candidate, card, fold and commitment, and the row axis of the
        numerical result's chunk -- as the current publication reader holds
        it against the marker, surface and fold surface.
        """

        if (
            self.role is not numerical.role
            or self.status is not numerical.status
            or self.metrics != numerical.metrics
            or self.session_rank_ics != numerical.session_rank_ics
            or self.session_spreads != numerical.session_spreads
            or self.fit_ledger != numerical.fit_ledger
            or self.failure != numerical.failure
        ):
            return False
        if self.numerical_result_hash is not None:
            return (
                self.score_chunk == numerical.score_chunk
                and self.estimator_state_hash == numerical.estimator_state_hash
            )
        if self.estimator_state_hash not in (None, numerical.estimator_state_hash):
            return False
        if not isinstance(self.score_chunk, AlphaCandidateDevelopmentScoreChunkRef):
            return self.score_chunk == numerical.score_chunk
        return (
            numerical.score_chunk is not None
            and self.score_chunk.request_hash == self.request_hash
            and self.score_chunk.development_surface_hash == self.development_surface_hash
            and self.score_chunk.candidate_card_hash == self.candidate_card_hash
            and self.score_chunk.fold_commitment_hash == self.fold_commitment_hash
            and self.score_chunk.row_count == numerical.score_chunk.row_count
        )


class AlphaCandidateNumericalFoldResult(_Contract):
    """Reusable numerical result; a parent request only references this authority."""

    kind: Literal["AlphaCandidateNumericalFoldResult"] = "AlphaCandidateNumericalFoldResult"
    execution_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    development_surface_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_id: str
    candidate_card_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    role: AlphaCandidateRole
    fold_index: int = Field(ge=0)
    fold_commitment_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: AlphaCandidateStatus
    score_chunk: AlphaNumericalDevelopmentScoreChunkRef | None = None
    metrics: AlphaFoldMetrics | None = None
    session_rank_ics: tuple[float, ...]
    session_spreads: tuple[float, ...]
    fit_ledger: AlphaFitLedgerEntry | None = None
    estimator_state_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    failure: AlphaCandidateFailure | None = None
    numerical_result_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> AlphaCandidateNumericalFoldResult:
        succeeded = self.status is AlphaCandidateStatus.SUCCEEDED
        if succeeded != (
            self.score_chunk is not None
            and self.metrics is not None
            and self.fit_ledger is not None
            and self.failure is None
        ):
            raise ValueError("Alpha numerical fold result is incomplete")
        if self.score_chunk is not None and (
            self.score_chunk.execution_binding_hash != self.execution_binding_hash
            or self.score_chunk.development_surface_binding_hash
            != self.development_surface_binding_hash
            or self.score_chunk.candidate_id != self.candidate_id
            or self.score_chunk.fold_index != self.fold_index
        ):
            raise ValueError("Alpha numerical score chunk binding differs")
        if self.numerical_result_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"numerical_result_hash"})
        ):
            raise ValueError("Alpha numerical fold result hash is invalid")
        return self


class AlphaCandidateDevelopmentReport(_Contract):
    kind: Literal["AlphaCandidateDevelopmentReport"] = "AlphaCandidateDevelopmentReport"
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    development_surface_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    foundation_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_id: str
    candidate_card_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    role: AlphaCandidateRole
    status: AlphaCandidateStatus
    admitted_fold_count: int = Field(ge=1)
    fold_evidence_hashes: tuple[str, ...]
    metrics: AlphaCandidateMetrics | None = None
    estimator_state_hashes: tuple[str, ...]
    failure_codes: tuple[str, ...]
    report_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> AlphaCandidateDevelopmentReport:
        succeeded = self.status is AlphaCandidateStatus.SUCCEEDED
        if succeeded != (
            len(self.fold_evidence_hashes) == self.admitted_fold_count
            and self.metrics is not None
            and not self.failure_codes
        ):
            raise ValueError("Alpha candidate development report is incomplete")
        if self.report_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"report_hash"})
        ):
            raise ValueError("Alpha candidate development report hash is invalid")
        return self


class AlphaCandidateInferenceEvidence(_Contract):
    kind: Literal["AlphaCandidateInferenceEvidence"] = "AlphaCandidateInferenceEvidence"
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_id: str
    report_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    score_chunk_hashes: tuple[str, ...]
    successful_fold_count: int = Field(ge=0)
    admitted_fold_count: int = Field(ge=1)
    scored_row_count: int = Field(ge=0)
    common_surface_row_count: int = Field(ge=0)
    estimator_state_hashes: tuple[str, ...]
    evidence_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> AlphaCandidateInferenceEvidence:
        if (
            self.successful_fold_count > self.admitted_fold_count
            or len(self.score_chunk_hashes) != self.successful_fold_count
            or self.estimator_state_hashes != tuple(dict.fromkeys(self.estimator_state_hashes))
        ):
            raise ValueError("Alpha successful folds exceed admitted folds")
        if self.evidence_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"evidence_hash"})
        ):
            raise ValueError("Alpha candidate inference evidence hash is invalid")
        return self


class AlphaCandidateViability(_Contract):
    candidate_id: str
    benchmark_candidate_id: Literal["benchmark.historical-mean"] = "benchmark.historical-mean"
    dm_statistic: float | None = Field(default=None, allow_inf_nan=False)
    directional_raw_p_value: float | None = Field(default=None, ge=0, le=1)
    holm_adjusted_p_value: float | None = Field(default=None, ge=0, le=1)
    hac_lag: int | None = Field(default=None, ge=0)
    complete_folds: bool
    common_surface_complete: bool
    finite_metrics: bool
    positive_oos_r2: bool
    positive_rank_ic: bool
    positive_gross_spread: bool
    positive_bootstrap_lower: bool
    admitted: bool
    failure_codes: tuple[str, ...]
    candidate_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> AlphaCandidateViability:
        if self.admitted and (
            self.holm_adjusted_p_value is None
            or self.holm_adjusted_p_value > 0.05
            or self.failure_codes
            or not all(
                (
                    self.complete_folds,
                    self.common_surface_complete,
                    self.finite_metrics,
                    self.positive_oos_r2,
                    self.positive_rank_ic,
                    self.positive_gross_spread,
                    self.positive_bootstrap_lower,
                )
            )
        ):
            raise ValueError("Alpha candidate was admitted without every formal gate")
        if self.candidate_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"candidate_hash"})
        ):
            raise ValueError("Alpha candidate viability hash is invalid")
        return self


class AlphaModelViabilityAssessment(_Contract):
    kind: Literal["AlphaModelViabilityAssessment"] = "AlphaModelViabilityAssessment"
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    surface_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    benchmark_candidate_id: Literal["benchmark.historical-mean"] = "benchmark.historical-mean"
    familywise_alpha: float = Field(default=0.05, ge=0, le=1, allow_inf_nan=False)
    hypothesis_count: int = Field(ge=0)
    candidates: tuple[AlphaCandidateViability, ...]
    admissible_candidate_ids: tuple[str, ...]
    disposition: Literal["ADMISSIBLE_MODELS_AVAILABLE", "NO_ADMISSIBLE_ALPHA_MODEL"]
    assessment_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> AlphaModelViabilityAssessment:
        ids = tuple(value.candidate_id for value in self.candidates)
        if self.familywise_alpha != 0.05:
            raise ValueError("Alpha viability familywise alpha is not the registered policy")
        if len(ids) != self.hypothesis_count or ids != tuple(dict.fromkeys(ids)):
            raise ValueError("Alpha viability hypothesis family is inconsistent")
        admitted = tuple(value.candidate_id for value in self.candidates if value.admitted)
        if admitted != self.admissible_candidate_ids:
            raise ValueError("Alpha viability admitted candidates do not reconcile")
        expected = "ADMISSIBLE_MODELS_AVAILABLE" if admitted else "NO_ADMISSIBLE_ALPHA_MODEL"
        if self.disposition != expected:
            raise ValueError("Alpha viability disposition is inconsistent")
        if self.assessment_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"assessment_hash"})
        ):
            raise ValueError("Alpha viability assessment hash is invalid")
        return self


class AlphaModelSelectionProposal(_Contract):
    kind: Literal["AlphaModelSelectionProposal"] = "AlphaModelSelectionProposal"
    invocation_token: str
    viability_assessment_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    action: AlphaSelectionAction
    selected_candidate_id: str | None = None
    reason_code: str
    observed_candidate_ids: tuple[str, ...]
    proposal_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> AlphaModelSelectionProposal:
        if (self.action is AlphaSelectionAction.SELECT_REGISTERED_ALPHA) != (
            self.selected_candidate_id is not None
        ):
            raise ValueError("Alpha selection proposal shape is invalid")
        if self.proposal_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"proposal_hash"})
        ):
            raise ValueError("Alpha selection proposal hash is invalid")
        return self


class AlphaModelSelectionDecision(_Contract):
    kind: Literal["AlphaModelSelectionDecision"] = "AlphaModelSelectionDecision"
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    viability_assessment_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    authoritative_proposal_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    action: AlphaSelectionAction
    selected_candidate_id: str | None = None
    reason_code: str
    decision_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> AlphaModelSelectionDecision:
        if (self.action is AlphaSelectionAction.SELECT_REGISTERED_ALPHA) != (
            self.selected_candidate_id is not None
        ):
            raise ValueError("Alpha selection decision shape is invalid")
        if self.decision_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"decision_hash"})
        ):
            raise ValueError("Alpha selection decision hash is invalid")
        return self


class AlphaDecisionSample(_Contract):
    sample_kind: Literal["AUTHORITATIVE", "SHADOW"]
    sample_index: int = Field(ge=1, le=3)
    proposal_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    decision_surface_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    status: Literal["SUCCEEDED", "PROTOCOL_DEFERRED", "FAILED"]
    model_calls: int = Field(ge=0)
    tool_calls: int = Field(ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    latency_seconds: float = Field(ge=0, allow_inf_nan=False)
    failure_code: str | None = None

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_sample(self) -> AlphaDecisionSample:
        succeeded = self.status == "SUCCEEDED"
        if succeeded != (self.proposal_hash is not None and self.decision_surface_hash is not None):
            raise ValueError("Alpha decision sample success identity is incomplete")
        return self


class AlphaDecisionReproducibilityReport(_Contract):
    kind: Literal["AlphaDecisionReproducibilityReport"] = "AlphaDecisionReproducibilityReport"
    decision_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    viability_assessment_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    sampling_protocol: Literal["AUTHORITATIVE_PLUS_TWO_SHADOWS"] = "AUTHORITATIVE_PLUS_TWO_SHADOWS"
    samples: tuple[AlphaDecisionSample, AlphaDecisionSample, AlphaDecisionSample]
    status: Literal[
        "OBSERVED_EXACT_STABILITY_3_OF_3",
        "OBSERVED_DIVERGENCE",
        "PROBE_INCOMPLETE",
    ]
    shadow_divergence_observed: bool
    report_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> AlphaDecisionReproducibilityReport:
        if tuple(value.sample_index for value in self.samples) != (1, 2, 3):
            raise ValueError("Alpha decision sample order is invalid")
        if tuple(value.sample_kind for value in self.samples) != (
            "AUTHORITATIVE",
            "SHADOW",
            "SHADOW",
        ):
            raise ValueError("Alpha authoritative/shadow roles are invalid")
        if self.report_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"report_hash"})
        ):
            raise ValueError("Alpha reproducibility report hash is invalid")
        return self


class AlphaDevelopmentChildLineage(_Contract):
    """One fitted child, bound to the declared target method that produced it.

    The fold's ``AlphaTrainingInputBinding`` records the *lane* policy, and the
    lane's ``standardization_id`` is a property of a closed enum -- so it says
    rank-gauss whatever the recipe selected. That contract is frozen and shared
    with the current path, so it cannot be told the truth here.

    This is where the truth lives instead. Each entry names exactly one
    ``(candidate_id, fold_index)`` child and carries, beside its published
    identities, the recipe binding it was fitted under and the materialization
    binding its arrays came from. Parallel tuples of estimator hashes and fit
    hashes would assert "robust-z values" and "these children" side by side while
    proving nothing joins them, and would lose the candidate/fold correspondence
    entirely. A lineage entry is that join.

    ``training_input_binding_hash`` is named as legacy on purpose: it is a real
    published identity a reader must be able to follow, and it is the lane-scoped
    one, so it is not evidence about the standardization.
    """

    kind: Literal["AlphaDevelopmentChildLineage"] = "AlphaDevelopmentChildLineage"
    candidate_id: str = Field(min_length=1, max_length=128)
    fold_index: int = Field(ge=0)
    fold_commitment_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    numerical_result_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    estimator_state_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    fit_evidence_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_input_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    estimator_content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    fit_provenance_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    score_evidence_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    ordered_factor_ids: tuple[str, ...] = Field(min_length=1)
    """The axis the estimator itself recorded, so the child's columns are checked
    against the declaration rather than assumed from the plan."""

    target_recipe_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_materialization_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    lineage_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> AlphaDevelopmentChildLineage:
        if self.lineage_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"lineage_hash"})
        ):
            raise ValueError("alpha_research.development_child_lineage_invalid")
        return self


class AlphaDevelopmentExecutionReceipt(_Contract):
    """The development authority parent for one Alpha model execution.

    Both target bindings used to be temporary locals folded into one opaque
    ``desk_input_binding_hash`` and then discarded, so nothing durable recorded
    which target method a run declared, which values it materialized, or which
    fitted children came from them. A reader holding the batch result could
    recover none of it, and the child artifacts describe only the lane.

    Everything here is read back from published artifacts rather than restated
    from the executor's locals. That is what makes it a statement about what was
    written rather than about what the executor believed it was writing.
    """

    kind: Literal["AlphaDevelopmentExecutionReceipt"] = "AlphaDevelopmentExecutionReceipt"
    program_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    desk_program_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    method_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    desk_input_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_recipe_binding: InstalledAlphaTargetMethodBinding
    target_materialization_binding: AlphaDevelopmentTargetMaterializationBinding
    ordered_base_feature_ids: tuple[str, ...] = Field(min_length=1)
    research_recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_adapter_id: str = Field(min_length=1, max_length=96)
    model_adapter_recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    """The adapter's own recipe identity, which is what the fit evidence records.

    Not the same value as ``research_recipe_hash``: that one covers the admitted
    proposal including its search domain and target lane, and the children were
    fitted against this one.
    """

    development_program_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    metric_policy_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The executed metric policy, independently recorded for saved comparisons.

    Older receipts remain readable with ``None``; a descriptive comparison refuses
    to treat that absent evidence as proof of a matching policy.
    """
    split_policy: AlphaDevelopmentSplitPolicy
    """The split geometry this run actually used, including its embargo.

    On the receipt so a verifier can re-derive the embargo the outcome method's
    maturity clock implies and compare. Recorded inside the development Program
    too, but a verifier holding only a receipt would otherwise have to resolve
    the Program to learn what it should have been."""

    model_method_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    """Which model was bound to which target, so the receipt names the whole
    method rather than an estimator that could have been fitted to anything."""

    @property
    def split_policy_hash(self) -> str:
        """The sealed geometry's identity, without a second field carrying it.

        A field would have to be kept in step with the policy beside it, and two
        places stating one fact is how they come to disagree.
        """

        return str(self.split_policy.policy_hash)

    batch_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    batch_result_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    development_surface_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    development_surface_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    fold_surface_hashes: tuple[str, ...] = Field(min_length=1)
    child_lineage: tuple[AlphaDevelopmentChildLineage, ...] = Field(min_length=1)
    receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(
        cls,
        *,
        program_hash: str,
        desk_program_hash: str,
        method_binding_hash: str,
        desk_input_binding_hash: str,
        target_recipe_binding: InstalledAlphaTargetMethodBinding,
        target_materialization_binding: AlphaDevelopmentTargetMaterializationBinding,
        ordered_base_feature_ids: tuple[str, ...],
        research_recipe_hash: str,
        model_adapter_id: str,
        model_adapter_recipe_hash: str,
        development_program_hash: str,
        split_policy: AlphaDevelopmentSplitPolicy,
        model_method_binding_hash: str,
        batch_hash: str,
        batch_result_hash: str,
        development_surface_hash: str,
        development_surface_binding_hash: str,
        fold_surface_hashes: tuple[str, ...],
        child_lineage: tuple[AlphaDevelopmentChildLineage, ...],
        metric_policy_hash: str | None = None,
    ) -> AlphaDevelopmentExecutionReceipt:
        # Built explicitly rather than through ``model_construct``, because the
        # nested bindings and lineage entries are models: an unvalidated
        # provisional dump can serialize a container shape the validated model
        # never has, and the identity would then be computed over that shape.
        identity = {
            "kind": "AlphaDevelopmentExecutionReceipt",
            "program_hash": program_hash,
            "desk_program_hash": desk_program_hash,
            "method_binding_hash": method_binding_hash,
            "desk_input_binding_hash": desk_input_binding_hash,
            "target_recipe_binding": target_recipe_binding.model_dump(mode="json"),
            "target_materialization_binding": target_materialization_binding.model_dump(
                mode="json"
            ),
            "ordered_base_feature_ids": list(ordered_base_feature_ids),
            "research_recipe_hash": research_recipe_hash,
            "model_adapter_id": model_adapter_id,
            "model_adapter_recipe_hash": model_adapter_recipe_hash,
            "development_program_hash": development_program_hash,
            "metric_policy_hash": metric_policy_hash,
            "split_policy": split_policy.model_dump(mode="json"),
            "model_method_binding_hash": model_method_binding_hash,
            "batch_hash": batch_hash,
            "batch_result_hash": batch_result_hash,
            "development_surface_hash": development_surface_hash,
            "development_surface_binding_hash": development_surface_binding_hash,
            "fold_surface_hashes": list(fold_surface_hashes),
            "child_lineage": [value.model_dump(mode="json") for value in child_lineage],
        }
        return cls(
            program_hash=program_hash,
            desk_program_hash=desk_program_hash,
            method_binding_hash=method_binding_hash,
            desk_input_binding_hash=desk_input_binding_hash,
            target_recipe_binding=target_recipe_binding,
            target_materialization_binding=target_materialization_binding,
            ordered_base_feature_ids=tuple(ordered_base_feature_ids),
            research_recipe_hash=research_recipe_hash,
            model_adapter_id=model_adapter_id,
            model_adapter_recipe_hash=model_adapter_recipe_hash,
            development_program_hash=development_program_hash,
            metric_policy_hash=metric_policy_hash,
            split_policy=split_policy,
            model_method_binding_hash=model_method_binding_hash,
            batch_hash=batch_hash,
            batch_result_hash=batch_result_hash,
            development_surface_hash=development_surface_hash,
            development_surface_binding_hash=development_surface_binding_hash,
            fold_surface_hashes=tuple(fold_surface_hashes),
            child_lineage=tuple(child_lineage),
            receipt_hash=str(canonical_hash(identity)),
        )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> AlphaDevelopmentExecutionReceipt:
        materialization = self.target_materialization_binding
        recipe = self.target_recipe_binding
        if materialization.recipe_binding_hash != recipe.binding_hash:
            raise ValueError("alpha_research.development_receipt_target_layers_disagree")
        if materialization.standardization_id != recipe.standardization_id:
            # The two layers must name one method. ``create`` copies this field
            # from the recipe binding, but the contract can be built directly, and
            # a materialization naming the robust-z recipe while declaring
            # rank-gauss standardization validates every hash it owns. That is the
            # exact defect this gate exists to prevent, one level up.
            raise ValueError("alpha_research.development_receipt_standardization_disagrees")
        if tuple(materialization.ordered_base_feature_ids) != self.ordered_base_feature_ids:
            raise ValueError("alpha_research.development_receipt_feature_axis_disagrees")
        records = materialization.fold_records
        if materialization.training_row_count != sum(
            int(value.training_shape[0]) for value in records
        ):
            # A consequence of the records rather than a field beside them.
            raise ValueError("alpha_research.development_receipt_training_rows_disagree")
        seen: set[tuple[str, int]] = set()
        for entry in self.child_lineage:
            key = (entry.candidate_id, entry.fold_index)
            if key in seen:
                raise ValueError("alpha_research.development_receipt_child_duplicated")
            seen.add(key)
            # Positional, not set membership. The records are validated as
            # ``fold_index == range(n)``, so a child claiming one fold's
            # commitment under another fold's index is refused here rather than
            # passing because that commitment appears somewhere in the split.
            if entry.fold_index >= len(records):
                raise ValueError("alpha_research.development_receipt_child_fold_unbound")
            if records[entry.fold_index].fold_commitment_hash != entry.fold_commitment_hash:
                raise ValueError("alpha_research.development_receipt_child_fold_unbound")
            if (
                entry.target_recipe_binding_hash != self.target_recipe_binding.binding_hash
                or entry.target_materialization_binding_hash != materialization.binding_hash
            ):
                raise ValueError("alpha_research.development_receipt_child_target_unbound")
            if entry.ordered_factor_ids != self.ordered_base_feature_ids:
                raise ValueError("alpha_research.development_receipt_child_axis_mismatch")
        identity = self.model_dump(mode="json", exclude={"receipt_hash"})
        legacy_identity = dict(identity)
        legacy_identity.pop("metric_policy_hash", None)
        accepted = {canonical_hash(identity)}
        if self.metric_policy_hash is None:
            accepted.add(canonical_hash(legacy_identity))
        if self.receipt_hash not in accepted:
            raise ValueError("alpha_research.development_receipt_identity_invalid")
        return self


def canonical_score_value_identity(values: _ScoreArray) -> str:
    """dtype, shape and content as one inseparable claim, as everywhere else here."""

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


class CanonicalScoreSurfaceError(ValueError):
    """Stable refusal raised before any canonical score claim is written."""


class CanonicalAlphaScoreFoldRef(_Contract):
    """One fold's contribution to the surface, joined to the child that holds it."""

    kind: Literal["CanonicalAlphaScoreFoldRef"] = "CanonicalAlphaScoreFoldRef"
    fold_index: int = Field(ge=0)
    fold_commitment_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    fold_surface_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    """The fold surface whose validation chunk gives these scores their axes.

    Carried so the surface is self-sufficient: a reader assembling the matrix
    needs the formation and listing of every scored row, and without this it
    would have to resolve the receipt again to find out where to look."""

    numerical_result_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    score_chunk_content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    row_count: int = Field(ge=1)
    first_formation_session: date
    last_formation_session: date

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_span(self) -> Self:
        if self.first_formation_session > self.last_formation_session:
            raise CanonicalScoreSurfaceError("alpha_research.canonical_score_fold_span_invalid")
        return self


class CanonicalAlphaScoreSurface(_Contract):
    """One candidate's predictions over the whole development axis, by reference.

    ``ordered_fold_refs`` is in fold order, and fold order is validation-window
    order, so the surface's formation axis is monotone by construction rather
    than by a sort applied afterwards. A sort would have hidden a fold published
    out of sequence.
    """

    kind: Literal["CanonicalAlphaScoreSurface"] = "CanonicalAlphaScoreSurface"
    development_receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_id: str = Field(min_length=1, max_length=128)
    target_recipe_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_evidence_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    ordered_fold_refs: tuple[CanonicalAlphaScoreFoldRef, ...] = Field(min_length=1)
    ordered_listing_ids: tuple[str, ...] = Field(min_length=1)
    formation_sessions: tuple[date, ...] = Field(min_length=1)
    score_dtype: Literal["float64"] = "float64"
    score_row_count: int = Field(ge=1)
    score_value_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    surface_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        indices = tuple(ref.fold_index for ref in self.ordered_fold_refs)
        if indices != tuple(range(len(indices))):
            # Folds complete and in order. A gap would mean part of the axis has
            # no author, and a permutation would mean the formation axis below
            # describes a different assembly than the one that produced it.
            raise CanonicalScoreSurfaceError("alpha_research.canonical_score_folds_incomplete")
        spans = tuple(
            (ref.first_formation_session, ref.last_formation_session)
            for ref in self.ordered_fold_refs
        )
        for earlier, later in pairwise(spans):
            if earlier[1] >= later[0]:
                # Overlapping validation windows would score a formation twice
                # and leave the surface's own axis ambiguous.
                raise CanonicalScoreSurfaceError("alpha_research.canonical_score_folds_overlap")
        if self.formation_sessions != tuple(sorted(set(self.formation_sessions))):
            raise CanonicalScoreSurfaceError("alpha_research.canonical_score_sessions_invalid")
        if self.ordered_listing_ids != tuple(sorted(set(self.ordered_listing_ids))):
            raise CanonicalScoreSurfaceError("alpha_research.canonical_score_listings_invalid")
        if self.score_row_count != sum(ref.row_count for ref in self.ordered_fold_refs):
            raise CanonicalScoreSurfaceError("alpha_research.canonical_score_row_count_invalid")
        if self.surface_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"surface_hash"})
        ):
            raise CanonicalScoreSurfaceError(
                "alpha_research.canonical_score_surface_identity_invalid"
            )
        return self


__all__ = [
    "AlphaCandidateDevelopmentReport",
    "AlphaCandidateDevelopmentScoreChunkRef",
    "AlphaCandidateExecutionBinding",
    "AlphaCandidateFoldEvidence",
    "AlphaCandidateInferenceEvidence",
    "AlphaCandidateNumericalFoldResult",
    "AlphaCandidateViability",
    "AlphaCurrentRefitDiagnosticReport",
    "AlphaDecisionReproducibilityReport",
    "AlphaDecisionSample",
    "AlphaDevelopmentChildLineage",
    "AlphaDevelopmentEstimatorState",
    "AlphaDevelopmentExecutionReceipt",
    "AlphaDevelopmentFoldSurface",
    "AlphaDevelopmentSurfaceBinding",
    "AlphaDevelopmentSurfaceManifest",
    "AlphaDevelopmentValidationChunkRef",
    "AlphaEstimatorState",
    "AlphaFactorContributionDiagnostic",
    "AlphaMatrixDiagnostic",
    "AlphaModelSelectionDecision",
    "AlphaModelSelectionProposal",
    "AlphaModelViabilityAssessment",
    "AlphaNumericalDevelopmentScoreChunkRef",
    "AlphaParentRequestBinding",
    "AlphaSelectionAction",
    "AlphaSessionScoreStatistics",
    "LegacyAlphaCandidateFoldEvidence",
    "seal_current_contract",
]
