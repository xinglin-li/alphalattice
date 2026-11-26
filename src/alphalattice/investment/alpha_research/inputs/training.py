"""Host-owned identity and admission firewall for Alpha model arrays."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Literal, Self, cast

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.capabilities.alpha_modeling.contracts import (
    BoundAlphaPredictionInput,
    BoundAlphaTrainingInput,
    alpha_model_array_content_hash,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash

from ..targets.execution_outcome import AlphaTargetPolicy

type FloatArray = npt.NDArray[np.float64]
type BoolArray = npt.NDArray[np.bool_]
type AlphaTrainingScope = Literal[
    "DEVELOPMENT_FOLD",
    "CURRENT_REFIT",
    "PORTFOLIO_DEVELOPMENT",
    "PORTFOLIO_POLICY_OOS",
    "PORTFOLIO_POLICY_HOLDOUT",
]


class AlphaTrainingInputAuthorityError(ValueError):
    """Fail-closed invalid-experiment result raised before numerical model work."""

    experiment_disposition = "INVALID_EXPERIMENT"
    fit_call_count = 0
    scientific_admission_effect = "NONE"

    def __init__(self, failure_class: str) -> None:
        """Retain a classified invalid-experiment authority failure.

        Args:
            failure_class: Stable target, feature-axis, row-axis or time authority mismatch class.
        """
        self.failure_class = failure_class
        super().__init__(f"INVALID_EXPERIMENT:{failure_class}")


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class AlphaTrainingInputBinding(_Contract):
    """Content and authority identity that travels with one materialized fit surface."""

    scope: AlphaTrainingScope
    foundation_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_panel_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    causal_outcome_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_surface_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    economic_return_surface_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_catalog_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_surface_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_feature_values_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    prediction_feature_values_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_target_values_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    prediction_target_values_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    ordered_feature_ids: tuple[str, ...] = Field(min_length=1)
    training_row_axis_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    prediction_row_axis_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_mask_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    prediction_mask_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_cutoff: date
    outcome_maturity_session: date
    prediction_anchor: date
    fold_commitment_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    feature_context_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require a unique feature axis, causal training cutoff and scope-bound fold authority.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Features repeat, training cutoff follows maturity/prediction anchor,
                binding_hash is inconsistent or fold commitment presence disagrees with
                development/current scope.
        """
        if (
            self.ordered_feature_ids != tuple(dict.fromkeys(self.ordered_feature_ids))
            or self.training_cutoff > self.outcome_maturity_session
            or self.training_cutoff > self.prediction_anchor
            or self.binding_hash
            != canonical_hash(self.model_dump(mode="json", exclude={"binding_hash"}))
        ):
            raise ValueError("ALPHA_TRAINING_INPUT_BINDING_INVALID")
        if (self.scope == "DEVELOPMENT_FOLD") != (self.fold_commitment_hash is not None):
            raise ValueError("ALPHA_TRAINING_INPUT_SCOPE_INVALID")
        return self


def alpha_target_policy_hash(
    target_policy: object | None, *, causal_outcome_snapshot_hash: str
) -> str:
    """Resolve current target policy identity or the legacy source-bound simple-return identity.

    Args:
        target_policy: Optional declared current target policy.
        causal_outcome_snapshot_hash: Source identity used by the legacy target fallback.

    Returns:
        Validated current policy_hash or canonical legacy target/source identity.

    Raises:
        pydantic.ValidationError: A supplied current target policy violates its model.
    """
    if target_policy is None:
        return cast(
            str,
            canonical_hash(
                {
                    "kind": "LegacySimpleExecutionReturnTarget",
                    "causal_outcome_snapshot_hash": causal_outcome_snapshot_hash,
                }
            ),
        )
    return cast(str, AlphaTargetPolicy.model_validate(target_policy).policy_hash)


def alpha_feature_catalog_binding_hash(
    *,
    foundation_hash: str,
    feature_panel_snapshot_hash: str,
    ordered_feature_ids: tuple[str, ...],
    feature_context_hash: str | None,
) -> str:
    """Hash the exact Foundation, Panel, ordered feature axis and optional context.

    Args:
        foundation_hash: Admitted Foundation identity.
        feature_panel_snapshot_hash: Source Panel identity.
        ordered_feature_ids: Exact ordered feature axis.
        feature_context_hash: Optional appended context identity.

    Returns:
        Canonical declared feature-catalog binding identity.
    """
    return cast(
        str,
        canonical_hash(
            {
                "foundation_hash": foundation_hash,
                "feature_panel_snapshot_hash": feature_panel_snapshot_hash,
                "ordered_feature_ids": ordered_feature_ids,
                "feature_context_hash": feature_context_hash,
            }
        ),
    )


def _array_hash(value: npt.NDArray[np.generic]) -> str:
    return alpha_model_array_content_hash(value)


def _row_axis_hash(sessions: Sequence[date], listings: Sequence[str]) -> str:
    if len(sessions) != len(listings):
        raise AlphaTrainingInputAuthorityError("ROW_AXIS_AUTHORITY_MISMATCH")
    return cast(str, canonical_hash(tuple(zip(sessions, listings, strict=True))))


def _surface_hash(
    *,
    kind: str,
    source_snapshot_hash: str,
    policy_hash: str | None,
    row_axis_hashes: tuple[str, str],
    row_provenance: tuple[tuple[str | None, ...], tuple[str | None, ...]],
    value_hash: str,
) -> str:
    return cast(
        str,
        canonical_hash(
            {
                "kind": kind,
                "source_snapshot_hash": source_snapshot_hash,
                "policy_hash": policy_hash,
                "row_axis_hashes": row_axis_hashes,
                "row_provenance": row_provenance,
                "value_hash": value_hash,
            }
        ),
    )


def build_alpha_training_input_binding(
    *,
    scope: AlphaTrainingScope,
    foundation_hash: str,
    feature_panel_snapshot_hash: str,
    causal_outcome_snapshot_hash: str,
    target_policy: object | None,
    ordered_feature_ids: tuple[str, ...],
    feature_context_hash: str | None,
    training_row_sessions: tuple[date, ...],
    training_row_listing_ids: tuple[str, ...],
    prediction_row_sessions: tuple[date, ...],
    prediction_row_listing_ids: tuple[str, ...],
    training_features: FloatArray,
    training_targets: FloatArray,
    training_mask: BoolArray,
    prediction_features: FloatArray,
    prediction_targets: FloatArray,
    prediction_mask: BoolArray,
    training_feature_row_hashes: tuple[str | None, ...],
    training_outcome_row_hashes: tuple[str | None, ...],
    prediction_feature_row_hashes: tuple[str | None, ...],
    prediction_outcome_row_hashes: tuple[str | None, ...],
    training_economic_returns: FloatArray,
    prediction_economic_returns: FloatArray,
    training_cutoff: date,
    outcome_maturity_session: date,
    prediction_anchor: date,
    fold_commitment_hash: str | None,
) -> AlphaTrainingInputBinding:
    """Seal exact row, feature, target, economic-return, mask and causal-clock input authority.

    The binding derives array and row-provenance identities before model execution. Target and
    economic-return surfaces remain distinct; a context revision rebinds feature authority without
    changing target meaning.

    Args:
        scope: Development-fold or current-refit training scope.
        foundation_hash: Admitted Foundation identity.
        feature_panel_snapshot_hash: Exact source Panel identity.
        causal_outcome_snapshot_hash: Exact causal-outcome source identity.
        target_policy: Optional current target policy; None selects the legacy source-bound target.
        ordered_feature_ids: Exact feature-column order.
        feature_context_hash: Optional appended feature-context identity.
        training_row_sessions: Formation session for each training row.
        training_row_listing_ids: Listing identifier for each training row.
        prediction_row_sessions: Formation session for each prediction row.
        prediction_row_listing_ids: Listing identifier for each prediction row.
        training_features: Declared training feature matrix.
        training_targets: Declared model-fit target vector.
        training_mask: Training admission mask.
        prediction_features: Declared prediction feature matrix.
        prediction_targets: Declared prediction target vector.
        prediction_mask: Prediction admission mask.
        training_feature_row_hashes: Source feature provenance for training rows.
        training_outcome_row_hashes: Source outcome provenance for training rows.
        prediction_feature_row_hashes: Source feature provenance for prediction rows.
        prediction_outcome_row_hashes: Source outcome provenance for prediction rows.
        training_economic_returns: Economic-return lane separate from the fit target.
        prediction_economic_returns: Economic-return lane separate from prediction targets.
        training_cutoff: Last admitted training session.
        outcome_maturity_session: Session through which training outcomes are available.
        prediction_anchor: Formation/validation anchor bounding admitted training.
        fold_commitment_hash: Exact commitment for development folds, absent for current refit.

    Returns:
        Validated training input binding with canonical identities for all declared lanes.

    Raises:
        ValueError: Row axes or declared binding consistency cannot be admitted.
    """
    training_axis_hash = _row_axis_hash(training_row_sessions, training_row_listing_ids)
    prediction_axis_hash = _row_axis_hash(prediction_row_sessions, prediction_row_listing_ids)
    policy_hash = alpha_target_policy_hash(
        target_policy,
        causal_outcome_snapshot_hash=causal_outcome_snapshot_hash,
    )
    feature_catalog_hash = alpha_feature_catalog_binding_hash(
        foundation_hash=foundation_hash,
        feature_panel_snapshot_hash=feature_panel_snapshot_hash,
        ordered_feature_ids=ordered_feature_ids,
        feature_context_hash=feature_context_hash,
    )
    training_feature_values_hash = _array_hash(training_features)
    prediction_feature_values_hash = _array_hash(prediction_features)
    training_target_values_hash = _array_hash(training_targets)
    prediction_target_values_hash = _array_hash(prediction_targets)
    target_value_hash = canonical_hash((training_target_values_hash, prediction_target_values_hash))
    values = {
        "scope": scope,
        "foundation_hash": foundation_hash,
        "feature_panel_snapshot_hash": feature_panel_snapshot_hash,
        "causal_outcome_snapshot_hash": causal_outcome_snapshot_hash,
        "target_policy_hash": policy_hash,
        "target_surface_hash": _surface_hash(
            kind="MODEL_TARGET",
            source_snapshot_hash=causal_outcome_snapshot_hash,
            policy_hash=policy_hash,
            row_axis_hashes=(training_axis_hash, prediction_axis_hash),
            row_provenance=(training_outcome_row_hashes, prediction_outcome_row_hashes),
            value_hash=target_value_hash,
        ),
        "economic_return_surface_hash": _surface_hash(
            kind="SIMPLE_ECONOMIC_RETURN",
            source_snapshot_hash=causal_outcome_snapshot_hash,
            policy_hash=None,
            row_axis_hashes=(training_axis_hash, prediction_axis_hash),
            row_provenance=(training_outcome_row_hashes, prediction_outcome_row_hashes),
            value_hash=canonical_hash(
                (
                    _array_hash(training_economic_returns),
                    _array_hash(prediction_economic_returns),
                )
            ),
        ),
        "feature_catalog_binding_hash": feature_catalog_hash,
        "feature_surface_hash": _surface_hash(
            kind="FEATURE_MATRIX",
            source_snapshot_hash=feature_panel_snapshot_hash,
            policy_hash=feature_catalog_hash,
            row_axis_hashes=(training_axis_hash, prediction_axis_hash),
            row_provenance=(training_feature_row_hashes, prediction_feature_row_hashes),
            value_hash=canonical_hash(
                (training_feature_values_hash, prediction_feature_values_hash)
            ),
        ),
        "training_feature_values_hash": training_feature_values_hash,
        "prediction_feature_values_hash": prediction_feature_values_hash,
        "training_target_values_hash": training_target_values_hash,
        "prediction_target_values_hash": prediction_target_values_hash,
        "ordered_feature_ids": ordered_feature_ids,
        "training_row_axis_hash": training_axis_hash,
        "prediction_row_axis_hash": prediction_axis_hash,
        "training_mask_hash": _array_hash(training_mask),
        "prediction_mask_hash": _array_hash(prediction_mask),
        "training_cutoff": training_cutoff,
        "outcome_maturity_session": outcome_maturity_session,
        "prediction_anchor": prediction_anchor,
        "fold_commitment_hash": fold_commitment_hash,
        "feature_context_hash": feature_context_hash,
    }
    return AlphaTrainingInputBinding(
        **values,
        binding_hash=canonical_hash(values),
    )


def assert_alpha_training_authority(
    binding: AlphaTrainingInputBinding | None,
    *,
    scope: AlphaTrainingScope,
    foundation_hash: str,
    feature_panel_snapshot_hash: str,
    causal_outcome_snapshot_hash: str,
    target_policy: object | None,
    ordered_feature_ids: tuple[str, ...],
    feature_context_hash: str | None,
    training_cutoff: date,
    outcome_maturity_session: date,
    prediction_anchor: date,
    fold_commitment_hash: str | None,
) -> AlphaTrainingInputBinding:
    """Require exact expected source, feature, row-scope and causal-clock authority.

    Args:
        binding: Existing binding to verify; absence is an invalid experiment.
        scope: Development-fold or current-refit training scope.
        foundation_hash: Admitted Foundation identity.
        feature_panel_snapshot_hash: Exact source Panel identity.
        causal_outcome_snapshot_hash: Exact causal-outcome source identity.
        target_policy: Optional current target policy; None selects the legacy source-bound target.
        ordered_feature_ids: Exact feature-column order.
        feature_context_hash: Optional appended feature-context identity.
        training_cutoff: Last admitted training session.
        outcome_maturity_session: Session through which training outcomes are available.
        prediction_anchor: Formation/validation anchor bounding admitted training.
        fold_commitment_hash: Exact commitment for development folds, absent for current refit.

    Returns:
        The supplied binding after all expected authority fields reconcile.

    Raises:
        AlphaTrainingInputAuthorityError: Target, feature-axis, row-scope or time authority differs
            from the expected declaration.
    """
    if binding is None:
        raise AlphaTrainingInputAuthorityError("TARGET_AUTHORITY_MISMATCH")
    expected_target = alpha_target_policy_hash(
        target_policy,
        causal_outcome_snapshot_hash=causal_outcome_snapshot_hash,
    )
    if (
        binding.foundation_hash != foundation_hash
        or binding.causal_outcome_snapshot_hash != causal_outcome_snapshot_hash
        or binding.target_policy_hash != expected_target
    ):
        raise AlphaTrainingInputAuthorityError("TARGET_AUTHORITY_MISMATCH")
    expected_catalog = alpha_feature_catalog_binding_hash(
        foundation_hash=foundation_hash,
        feature_panel_snapshot_hash=feature_panel_snapshot_hash,
        ordered_feature_ids=ordered_feature_ids,
        feature_context_hash=feature_context_hash,
    )
    if (
        binding.feature_panel_snapshot_hash != feature_panel_snapshot_hash
        or binding.ordered_feature_ids != ordered_feature_ids
        or binding.feature_catalog_binding_hash != expected_catalog
        or binding.feature_context_hash != feature_context_hash
    ):
        raise AlphaTrainingInputAuthorityError("FEATURE_AXIS_AUTHORITY_MISMATCH")
    if binding.scope != scope or binding.fold_commitment_hash != fold_commitment_hash:
        raise AlphaTrainingInputAuthorityError("ROW_AXIS_AUTHORITY_MISMATCH")
    if (
        binding.training_cutoff != training_cutoff
        or binding.outcome_maturity_session != outcome_maturity_session
        or binding.prediction_anchor != prediction_anchor
    ):
        raise AlphaTrainingInputAuthorityError("TIME_AUTHORITY_MISMATCH")
    return binding


def bind_alpha_model_inputs(
    *,
    binding: AlphaTrainingInputBinding | None,
    ordered_feature_ids: tuple[str, ...],
    training_features: FloatArray,
    training_targets: FloatArray,
    training_mask: BoolArray,
    prediction_features: FloatArray,
    prediction_mask: BoolArray,
) -> tuple[BoundAlphaTrainingInput, BoundAlphaPredictionInput]:
    """Verify exact input value/mask identities and expose only admitted read-only model rows.

    Args:
        binding: Required sealed training-input authority.
        ordered_feature_ids: Exact feature-column axis.
        training_features: Full declared training feature matrix.
        training_targets: Full declared training target vector.
        training_mask: Training admission mask.
        prediction_features: Full declared prediction feature matrix.
        prediction_mask: Prediction admission mask.

    Returns:
        Bound non-writeable training and prediction arrays selected by the verified masks.

    Raises:
        AlphaTrainingInputAuthorityError: Binding, feature/value/target identities or mask
            identities disagree.
    """
    if binding is None:
        raise AlphaTrainingInputAuthorityError("TARGET_AUTHORITY_MISMATCH")
    if binding.ordered_feature_ids != ordered_feature_ids:
        raise AlphaTrainingInputAuthorityError("FEATURE_AXIS_AUTHORITY_MISMATCH")
    if binding.training_feature_values_hash != _array_hash(
        training_features
    ) or binding.prediction_feature_values_hash != _array_hash(prediction_features):
        raise AlphaTrainingInputAuthorityError("FEATURE_AXIS_AUTHORITY_MISMATCH")
    if binding.training_target_values_hash != _array_hash(training_targets):
        raise AlphaTrainingInputAuthorityError("TARGET_AUTHORITY_MISMATCH")
    if binding.training_mask_hash != _array_hash(
        training_mask
    ) or binding.prediction_mask_hash != _array_hash(prediction_mask):
        raise AlphaTrainingInputAuthorityError("ROW_AXIS_AUTHORITY_MISMATCH")
    admitted_training = np.asarray(training_features[training_mask], dtype=np.float64)
    admitted_targets = np.asarray(training_targets[training_mask], dtype=np.float64)
    admitted_prediction = np.asarray(prediction_features[prediction_mask], dtype=np.float64)
    for value in (admitted_training, admitted_targets, admitted_prediction):
        value.setflags(write=False)
    return (
        BoundAlphaTrainingInput(
            training_binding_hash=binding.binding_hash,
            ordered_feature_ids=ordered_feature_ids,
            features=admitted_training,
            targets=admitted_targets,
        ),
        BoundAlphaPredictionInput(
            training_binding_hash=binding.binding_hash,
            ordered_feature_ids=ordered_feature_ids,
            features=admitted_prediction,
        ),
    )


__all__ = [
    "AlphaTrainingInputAuthorityError",
    "AlphaTrainingInputBinding",
    "AlphaTrainingScope",
    "alpha_feature_catalog_binding_hash",
    "alpha_target_policy_hash",
    "assert_alpha_training_authority",
    "bind_alpha_model_inputs",
    "build_alpha_training_input_binding",
]
