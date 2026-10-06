"""Small typed boundary for deterministic Alpha model implementations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Literal, Protocol, Self

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.kernel.shared_kernel.identity import canonical_hash

type FloatArray = npt.NDArray[np.float64]
type AlphaModelFitProtocol = Literal["DIRECT_FIT", "NESTED_EARLY_STOPPING_REFIT"]
NESTED_FIT_RUNTIME_CAPABILITY = "alpha-model-fit:nested-early-stopping-refit"


def alpha_model_array_content_hash(value: npt.NDArray[np.generic]) -> str:
    """Hash one numerical array with the existing Alpha training-value algorithm.

    Args:
        value: Numerical array whose dtype, shape and contiguous value bytes are to be bound.

    Returns:
        SHA-256 identity over dtype spelling, shape representation and contiguous values.
    """
    contiguous = np.ascontiguousarray(value)
    digest = sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(repr(contiguous.shape).encode("ascii"))
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class AlphaModelRecipeEnvelope(_Contract):
    """Host-sealed recipe routed to one explicitly installed adapter."""

    adapter_id: str = Field(min_length=1, max_length=96)
    recipe_schema_id: str = Field(min_length=1, max_length=128)
    parameters: dict[str, Any]
    recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(
        cls,
        *,
        adapter_id: str,
        recipe_schema_id: str,
        parameters: Mapping[str, object],
    ) -> Self:
        """Seal a recipe for an explicitly installed adapter and recipe schema.

        Args:
            adapter_id: Stable adapter handle named by the recipe.
            recipe_schema_id: Qualified schema through which that adapter interprets parameters.
            parameters: Declared recipe fields to copy into the immutable envelope.

        Returns:
            Immutable record with its fields validated and canonical content identity assigned.

        Raises:
            ValueError: A declared field or canonical contract invariant is invalid.
        """
        values = {
            "adapter_id": adapter_id,
            "recipe_schema_id": recipe_schema_id,
            "parameters": dict(parameters),
        }
        return cls(**values, recipe_hash=canonical_hash(values))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Verify the record's canonical contents and declared identity.

        Returns:
            This validated immutable record.

        Raises:
            ValueError: The recorded identity differs from the canonical fields.
        """
        if self.recipe_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"recipe_hash"})
        ):
            raise ValueError("ALPHA_MODEL_RECIPE_IDENTITY_INVALID")
        return self


class AlphaModelSearchDomainEnvelope(_Contract):
    """Host-sealed, adapter-validated recipe domain admitted by a research Mandate."""

    adapter_id: str = Field(min_length=1, max_length=96)
    recipe_schema_id: str = Field(min_length=1, max_length=128)
    search_domain_schema_id: str = Field(min_length=1, max_length=128)
    constraints: dict[str, Any]
    search_domain_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(
        cls,
        *,
        adapter_id: str,
        recipe_schema_id: str,
        search_domain_schema_id: str,
        constraints: Mapping[str, object],
    ) -> Self:
        """Seal the recipe constraints admitted by a declared model search domain.

        Args:
            adapter_id: Stable adapter owning the search domain.
            recipe_schema_id: Recipe schema constrained by the domain.
            search_domain_schema_id: Qualified constraint schema for that adapter.
            constraints: Declared constraints to copy into the immutable envelope.

        Returns:
            Immutable record with its fields validated and canonical content identity assigned.

        Raises:
            ValueError: A declared field or canonical contract invariant is invalid.
        """
        values = {
            "adapter_id": adapter_id,
            "recipe_schema_id": recipe_schema_id,
            "search_domain_schema_id": search_domain_schema_id,
            "constraints": dict(constraints),
        }
        return cls(**values, search_domain_hash=canonical_hash(values))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Verify the record's canonical contents and declared identity.

        Returns:
            This validated immutable record.

        Raises:
            ValueError: The recorded identity differs from the canonical fields.
        """
        if self.search_domain_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"search_domain_hash"})
        ):
            raise ValueError("ALPHA_MODEL_SEARCH_DOMAIN_IDENTITY_INVALID")
        return self


class AlphaModelNumericalBinding(_Contract):
    """Content identity of one adapter's deterministic numerical behavior."""

    adapter_id: str = Field(min_length=1, max_length=96)
    estimator_content_format_id: str = Field(min_length=1, max_length=128)
    implementation_owners: tuple[str, ...] = Field(min_length=1)
    deterministic_policy: dict[str, Any]
    required_runtime_capabilities: tuple[str, ...]
    numerical_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(
        cls,
        *,
        adapter_id: str,
        estimator_content_format_id: str,
        implementation_owners: tuple[str, ...],
        deterministic_policy: Mapping[str, object],
        required_runtime_capabilities: tuple[str, ...],
    ) -> Self:
        """Seal an adapter's declared numerical policy and runtime capability requirements.

        Args:
            adapter_id: Stable handle owning the numerical behavior.
            estimator_content_format_id: Format used to persist learned estimator content.
            implementation_owners: Unique ordered implementation-owner handles.
            deterministic_policy: Declared numerical choices and reproducibility rules.
            required_runtime_capabilities: Unique ordered capabilities required by the adapter.

        Returns:
            Immutable record with its fields validated and canonical content identity assigned.

        Raises:
            ValueError: A declared field or canonical contract invariant is invalid.
        """
        values = {
            "adapter_id": adapter_id,
            "estimator_content_format_id": estimator_content_format_id,
            "implementation_owners": implementation_owners,
            "deterministic_policy": dict(deterministic_policy),
            "required_runtime_capabilities": required_runtime_capabilities,
        }
        return cls(**values, numerical_binding_hash=canonical_hash(values))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Verify the record's canonical contents and declared identity.

        Returns:
            This validated immutable record.

        Raises:
            ValueError: Implementation/capability handles repeat or the canonical binding identity
                is invalid.
        """
        if (
            self.implementation_owners != tuple(dict.fromkeys(self.implementation_owners))
            or self.required_runtime_capabilities
            != tuple(dict.fromkeys(self.required_runtime_capabilities))
            or self.numerical_binding_hash
            != canonical_hash(self.model_dump(mode="json", exclude={"numerical_binding_hash"}))
        ):
            raise ValueError("ALPHA_MODEL_NUMERICAL_BINDING_INVALID")
        return self


class AlphaModelStateProjection(_Contract):
    """Adapter-owned, model-neutral projection used by development evidence."""

    adapter_id: str = Field(min_length=1, max_length=96)
    model_family_id: str = Field(min_length=1, max_length=96)
    state_kind: str = Field(min_length=1, max_length=64)
    state_schema_id: str = Field(min_length=1, max_length=128)
    payload: dict[str, Any]
    projection_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(
        cls,
        *,
        adapter_id: str,
        model_family_id: str,
        state_kind: str,
        state_schema_id: str,
        payload: Mapping[str, object],
    ) -> Self:
        """Seal a model-neutral projection of adapter-owned learned state.

        Args:
            adapter_id: Adapter which produced the state.
            model_family_id: Family interpreted by development evidence.
            state_kind: Declared state category.
            state_schema_id: Qualified schema used to interpret the payload.
            payload: Adapter-owned state observations copied into the immutable projection.

        Returns:
            Immutable record with its fields validated and canonical content identity assigned.

        Raises:
            ValueError: A declared field or canonical contract invariant is invalid.
        """
        values = {
            "adapter_id": adapter_id,
            "model_family_id": model_family_id,
            "state_kind": state_kind,
            "state_schema_id": state_schema_id,
            "payload": dict(payload),
        }
        return cls(**values, projection_hash=canonical_hash(values))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Verify the record's canonical contents and declared identity.

        Returns:
            This validated immutable record.

        Raises:
            ValueError: The recorded identity differs from the canonical fields.
        """
        if self.projection_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"projection_hash"})
        ):
            raise ValueError("ALPHA_MODEL_STATE_PROJECTION_INVALID")
        return self


class AlphaModelSelectionDiagnostic(_Contract):
    """Model-neutral inner-selection evidence emitted by one nested fit."""

    recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    estimator_content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    metric_id: str = Field(min_length=1, max_length=96)
    direction: Literal["MINIMIZE", "MAXIMIZE"]
    value: float = Field(allow_inf_nan=False)
    selected_iteration: int = Field(ge=1)
    fit_plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    diagnostic_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(
        cls,
        *,
        recipe_hash: str,
        estimator_content_hash: str,
        training_binding_hash: str,
        metric_id: str,
        direction: Literal["MINIMIZE", "MAXIMIZE"],
        value: float,
        selected_iteration: int,
        fit_plan_hash: str,
    ) -> Self:
        """Seal the selection evidence emitted by one nested model fit.

        Args:
            recipe_hash: Recipe selected and refitted.
            estimator_content_hash: Learned estimator content returned by the refit.
            training_binding_hash: Parent training-surface identity.
            metric_id: Declared inner-validation selection metric.
            direction: Whether lower or higher metric values are preferred.
            value: Finite selected metric observation.
            selected_iteration: Positive iteration selected from the tuning run.
            fit_plan_hash: Host-bound nested plan under which selection occurred.

        Returns:
            Immutable record with its fields validated and canonical content identity assigned.

        Raises:
            ValueError: A declared field or canonical contract invariant is invalid.
        """
        values = {
            "recipe_hash": recipe_hash,
            "estimator_content_hash": estimator_content_hash,
            "training_binding_hash": training_binding_hash,
            "metric_id": metric_id,
            "direction": direction,
            "value": value,
            "selected_iteration": selected_iteration,
            "fit_plan_hash": fit_plan_hash,
        }
        return cls(**values, diagnostic_hash=canonical_hash(values))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Verify the record's canonical contents and declared identity.

        Returns:
            This validated immutable record.

        Raises:
            ValueError: The recorded identity differs from the canonical fields.
        """
        if self.diagnostic_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"diagnostic_hash"})
        ):
            raise ValueError("ALPHA_MODEL_SELECTION_DIAGNOSTIC_INVALID")
        return self


@dataclass(frozen=True, slots=True)
class BoundAlphaTrainingInput:
    """Read-only model values admitted by the Alpha Research Host."""

    training_binding_hash: str
    ordered_feature_ids: tuple[str, ...]
    features: FloatArray
    targets: FloatArray

    def __post_init__(self) -> None:
        """Verify immutable finite training values against the declared feature axis.

        Raises:
            ValueError: Binding length, axis presence, matrix/target shape, immutability or finite
                values is invalid.
        """
        if (
            len(self.training_binding_hash) != 64
            or not self.ordered_feature_ids
            or self.features.ndim != 2
            or self.targets.ndim != 1
            or self.features.shape != (len(self.targets), len(self.ordered_feature_ids))
            or self.features.flags.writeable
            or self.targets.flags.writeable
            or not np.isfinite(self.features).all()
            or not np.isfinite(self.targets).all()
        ):
            raise ValueError("ALPHA_BOUND_TRAINING_INPUT_INVALID")


@dataclass(frozen=True, slots=True)
class BoundAlphaPredictionInput:
    """Read-only prediction matrix with the exact fitted feature axis."""

    training_binding_hash: str
    ordered_feature_ids: tuple[str, ...]
    features: FloatArray

    def __post_init__(self) -> None:
        """Verify immutable finite prediction values against the fitted feature axis.

        Raises:
            ValueError: Binding length, axis presence, matrix shape, immutability or finite values
                is invalid.
        """
        if (
            len(self.training_binding_hash) != 64
            or not self.ordered_feature_ids
            or self.features.ndim != 2
            or self.features.shape[1] != len(self.ordered_feature_ids)
            or self.features.flags.writeable
            or not np.isfinite(self.features).all()
        ):
            raise ValueError("ALPHA_BOUND_PREDICTION_INPUT_INVALID")


@dataclass(frozen=True, slots=True)
class BoundAlphaModelFitInput:
    """Host-bound fit plan and optional tuning partitions supplied to an adapter."""

    fit_plan_hash: str
    protocol_id: AlphaModelFitProtocol
    parent_training_binding_hash: str
    ordered_feature_ids: tuple[str, ...]
    tuning_training_row_axis_hash: str | None = None
    purge_row_axis_hash: str | None = None
    tuning_validation_row_axis_hash: str | None = None
    tuning_training_feature_values_hash: str | None = None
    tuning_training_target_values_hash: str | None = None
    tuning_validation_feature_values_hash: str | None = None
    tuning_validation_target_values_hash: str | None = None
    tuning_training_features: FloatArray | None = None
    tuning_training_targets: FloatArray | None = None
    tuning_validation_features: FloatArray | None = None
    tuning_validation_targets: FloatArray | None = None
    tuning_validation_session_codes: npt.NDArray[np.int64] | None = None
    tuning_validation_session_axis_hash: str | None = None
    selection_metric_id: str | None = None
    maximum_iterations: int | None = None
    early_stopping_rounds: int | None = None

    def __post_init__(self) -> None:
        """Verify direct or nested fit-plan authority and exact numerical value bindings.

        Raises:
            ValueError: Plan/parent handles or feature axes are invalid; a direct plan carries
                tuning data;
                nested metadata, shapes, iteration bounds, immutable values, optional session axis
                or measured array hashes do not agree.
        """
        if (
            len(self.fit_plan_hash) != 64
            or len(self.parent_training_binding_hash) != 64
            or not self.ordered_feature_ids
            or self.ordered_feature_ids != tuple(dict.fromkeys(self.ordered_feature_ids))
        ):
            raise ValueError("ALPHA_MODEL_FIT_PLAN_BINDING_INVALID")
        nested_metadata = (
            self.tuning_training_row_axis_hash,
            self.purge_row_axis_hash,
            self.tuning_validation_row_axis_hash,
            self.tuning_training_feature_values_hash,
            self.tuning_training_target_values_hash,
            self.tuning_validation_feature_values_hash,
            self.tuning_validation_target_values_hash,
            self.selection_metric_id,
            self.maximum_iterations,
            self.early_stopping_rounds,
        )
        nested_arrays = (
            self.tuning_training_features,
            self.tuning_training_targets,
            self.tuning_validation_features,
            self.tuning_validation_targets,
        )
        if self.protocol_id == "DIRECT_FIT":
            if any(
                value is not None
                for value in (
                    *nested_metadata,
                    *nested_arrays,
                    self.tuning_validation_session_codes,
                    self.tuning_validation_session_axis_hash,
                )
            ):
                raise ValueError("ALPHA_MODEL_DIRECT_FIT_PLAN_INVALID")
            return
        if any(value is None for value in (*nested_metadata, *nested_arrays)):
            raise ValueError("ALPHA_MODEL_NESTED_FIT_PLAN_INCOMPLETE")
        assert self.tuning_training_features is not None
        assert self.tuning_training_targets is not None
        assert self.tuning_validation_features is not None
        assert self.tuning_validation_targets is not None
        if (
            self.maximum_iterations is None
            or self.maximum_iterations < 1
            or self.early_stopping_rounds is None
            or self.early_stopping_rounds < 1
            or self.early_stopping_rounds > self.maximum_iterations
            or self.tuning_training_features.ndim != 2
            or self.tuning_validation_features.ndim != 2
            or self.tuning_training_targets.ndim != 1
            or self.tuning_validation_targets.ndim != 1
            or self.tuning_training_features.shape
            != (len(self.tuning_training_targets), len(self.ordered_feature_ids))
            or self.tuning_validation_features.shape
            != (len(self.tuning_validation_targets), len(self.ordered_feature_ids))
        ):
            raise ValueError("ALPHA_MODEL_NESTED_FIT_PLAN_INVALID")
        for value in nested_arrays:
            assert value is not None
            if value.flags.writeable or not np.isfinite(value).all():
                raise ValueError("ALPHA_MODEL_NESTED_FIT_VALUES_INVALID")
        if self.tuning_validation_session_codes is not None:
            codes = self.tuning_validation_session_codes
            if (
                self.tuning_validation_session_axis_hash is None
                or codes.dtype != np.dtype(np.int64)
                or codes.ndim != 1
                or codes.shape != self.tuning_validation_targets.shape
                or codes.flags.writeable
                or self.tuning_validation_session_axis_hash
                != canonical_hash([int(value) for value in codes])
            ):
                raise ValueError("ALPHA_MODEL_NESTED_SESSION_AXIS_INVALID")
        elif self.tuning_validation_session_axis_hash is not None:
            raise ValueError("ALPHA_MODEL_NESTED_SESSION_AXIS_INVALID")
        expected_hashes = (
            alpha_model_array_content_hash(self.tuning_training_features),
            alpha_model_array_content_hash(self.tuning_training_targets),
            alpha_model_array_content_hash(self.tuning_validation_features),
            alpha_model_array_content_hash(self.tuning_validation_targets),
        )
        if expected_hashes != (
            self.tuning_training_feature_values_hash,
            self.tuning_training_target_values_hash,
            self.tuning_validation_feature_values_hash,
            self.tuning_validation_target_values_hash,
        ):
            raise ValueError("ALPHA_MODEL_NESTED_FIT_VALUES_HASH_MISMATCH")


class AlphaEstimatorContent(_Contract):
    """Learned model content independent of where and why it was fitted."""

    adapter_id: str = Field(min_length=1, max_length=96)
    content_format_id: str = Field(min_length=1, max_length=128)
    ordered_feature_ids: tuple[str, ...] = Field(min_length=1)
    payload: dict[str, Any]
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(
        cls,
        *,
        adapter_id: str,
        content_format_id: str,
        ordered_feature_ids: tuple[str, ...],
        payload: Mapping[str, object],
    ) -> Self:
        """Seal learned estimator content independently of its fit-event provenance.

        Args:
            adapter_id: Adapter owning the learned content.
            content_format_id: Qualified estimator persistence format.
            ordered_feature_ids: Unique ordered feature axis used when fitting.
            payload: Learned parameters or model content copied into the immutable record.

        Returns:
            Immutable record with its fields validated and canonical content identity assigned.

        Raises:
            ValueError: A declared field or canonical contract invariant is invalid.
        """
        values = {
            "adapter_id": adapter_id,
            "content_format_id": content_format_id,
            "ordered_feature_ids": ordered_feature_ids,
            "payload": dict(payload),
        }
        return cls(**values, content_hash=canonical_hash(values))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Verify the record's canonical contents and declared identity.

        Returns:
            This validated immutable record.

        Raises:
            ValueError: The feature axis repeats or the learned-content hash is invalid.
        """
        if self.ordered_feature_ids != tuple(
            dict.fromkeys(self.ordered_feature_ids)
        ) or self.content_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"content_hash"})
        ):
            raise ValueError("ALPHA_ESTIMATOR_CONTENT_IDENTITY_INVALID")
        return self


class AlphaFitProvenanceReceipt(_Contract):
    """Identity of one fit event, separate from learned estimator content."""

    recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    estimator_content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    package_identity_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    fit_plan_hash: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
        exclude_if=lambda value: value is None,
    )
    numerical_environment_hash: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
        exclude_if=lambda value: value is None,
    )
    fit_call_count: int = Field(ge=0)
    predict_call_count: int = Field(ge=0)
    provenance_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(
        cls,
        *,
        recipe_hash: str,
        training_binding_hash: str,
        estimator_content_hash: str,
        package_identity_hash: str,
        fit_call_count: int,
        predict_call_count: int,
        fit_plan_hash: str | None = None,
        numerical_environment_hash: str | None = None,
    ) -> Self:
        """Seal one fit event without changing the learned estimator's content identity.

        Args:
            recipe_hash: Qualified numerical recipe used by the fit.
            training_binding_hash: Admitted training-surface identity.
            estimator_content_hash: Learned content returned by the fit.
            package_identity_hash: Owner-qualified implementation/package identity.
            fit_call_count: Observed number of numerical fit calls.
            predict_call_count: Observed number of numerical prediction calls.
            fit_plan_hash: Optional qualified Host plan; absent values are omitted from identity.
            numerical_environment_hash: Optional measured runtime environment; absent values are
                omitted.

        Returns:
            Immutable record with its fields validated and canonical content identity assigned.

        Raises:
            ValueError: A declared field or canonical contract invariant is invalid.
        """
        values = {
            "recipe_hash": recipe_hash,
            "training_binding_hash": training_binding_hash,
            "estimator_content_hash": estimator_content_hash,
            "package_identity_hash": package_identity_hash,
            "fit_call_count": fit_call_count,
            "predict_call_count": predict_call_count,
        }
        if fit_plan_hash is not None:
            values["fit_plan_hash"] = fit_plan_hash
        if numerical_environment_hash is not None:
            values["numerical_environment_hash"] = numerical_environment_hash
        return cls(**values, provenance_hash=canonical_hash(values))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Verify the record's canonical contents and declared identity.

        Returns:
            This validated immutable record.

        Raises:
            ValueError: Neither supported representation of absent plan/environment fields matches
                the receipt hash.
        """
        identity = self.model_dump(mode="json", exclude={"provenance_hash"})
        accepted = {canonical_hash(identity)}
        legacy = dict(identity)
        for field in ("fit_plan_hash", "numerical_environment_hash"):
            if legacy.get(field) is None:
                legacy.pop(field, None)
        accepted.add(canonical_hash(legacy))
        if self.provenance_hash not in accepted:
            raise ValueError("ALPHA_FIT_PROVENANCE_IDENTITY_INVALID")
        return self


class AlphaModelFitSidecar(_Contract):
    """Readback link from one Host operation to content and fit provenance."""

    operation_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    estimator_content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    fit_provenance_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    sidecar_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(
        cls,
        *,
        operation_binding_hash: str,
        estimator_content_hash: str,
        fit_provenance_hash: str,
    ) -> Self:
        """Seal the readback link from a Host operation to learned content and fit provenance.

        Args:
            operation_binding_hash: Exact Host operation whose readback this sidecar serves.
            estimator_content_hash: Learned content published by that operation.
            fit_provenance_hash: Qualified fit-event receipt retained beside the content.

        Returns:
            Immutable record with its fields validated and canonical content identity assigned.

        Raises:
            ValueError: A declared field or canonical contract invariant is invalid.
        """
        values = {
            "operation_binding_hash": operation_binding_hash,
            "estimator_content_hash": estimator_content_hash,
            "fit_provenance_hash": fit_provenance_hash,
        }
        return cls(**values, sidecar_hash=canonical_hash(values))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Verify the record's canonical contents and declared identity.

        Returns:
            This validated immutable record.

        Raises:
            ValueError: The recorded identity differs from the canonical fields.
        """
        if self.sidecar_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"sidecar_hash"})
        ):
            raise ValueError("ALPHA_MODEL_FIT_SIDECAR_IDENTITY_INVALID")
        return self


@dataclass(frozen=True, slots=True)
class AlphaModelFitResult:
    """Carry learned content, state and numerical diagnostics from an admitted fit.

    Attributes:
        estimator_content: Immutable learned content, independent of the fit-event receipt.
        state_projection: Adapter-owned model-neutral development evidence.
        training_mse: Mean squared error on the admitted training values.
        iteration_count: Solver/boosting iterations, or None for a closed-form fit.
        fit_call_count: Number of numerical fitting calls performed by the adapter.
        predict_call_count: Number of numerical prediction calls performed during the fit.
        selection_diagnostic: Inner-validation evidence when nested selection was used.
    """

    estimator_content: AlphaEstimatorContent
    state_projection: AlphaModelStateProjection
    training_mse: float
    iteration_count: int | None
    fit_call_count: int = 1
    predict_call_count: int = 1
    selection_diagnostic: AlphaModelSelectionDiagnostic | None = None


@dataclass(frozen=True, slots=True)
class AlphaModelPredictionResult:
    """Carry a finite read-only prediction vector and its numerical call count.

    Attributes:
        predictions: Row-aligned finite output values, frozen against mutation.
        predict_call_count: Exactly one numerical prediction call.
    """

    predictions: FloatArray
    predict_call_count: int = 1

    def __post_init__(self) -> None:
        """Verify a finite immutable prediction vector from exactly one prediction call.

        Raises:
            ValueError: Predictions are not a finite read-only vector or the call count differs from
                one.
        """
        if (
            self.predictions.ndim != 1
            or self.predictions.flags.writeable
            or not np.isfinite(self.predictions).all()
            or self.predict_call_count != 1
        ):
            raise ValueError("ALPHA_MODEL_PREDICTION_RESULT_INVALID")


class AlphaModelAdapter(Protocol):
    """Desk-specific deterministic model seam; the Host retains all authority."""

    adapter_id: str
    recipe_schema_id: str
    search_domain_schema_id: str

    def describe_numerical_binding(self) -> AlphaModelNumericalBinding:
        """Describe the adapter's declared numerical behavior and runtime requirements.

        Returns:
            Qualified numerical policy, estimator format, implementation owners and required
            capabilities.
        """
        ...

    def fit_protocols(self, recipe: AlphaModelRecipeEnvelope) -> tuple[AlphaModelFitProtocol, ...]:
        """The fit-plan protocols this recipe accepts; a Host states the first.

        An adapter that fits a tuning partition needs a nested plan; one that
        fits its whole training surface at a stated iteration count reads no
        partition and states a direct plan. Which of the two a recipe needs is
        the adapter's knowledge -- the numerical binding names the adapter's
        widest capability, not what one recipe consumes -- so the runtime asks
        here before it hands a plan over, and the development planner asks here
        before it seals one.
        """
        ...

    def validate_recipe(self, recipe: AlphaModelRecipeEnvelope) -> object:
        """Validate a numerical recipe under this adapter's qualified schema.

        Args:
            recipe: Host-sealed envelope naming this adapter and its recipe schema.

        Returns:
            Adapter-owned parsed numerical parameters.

        Raises:
            ValueError: The route, schema or numerical parameters are not admitted.
        """
        ...

    def validate_search_domain(self, domain: AlphaModelSearchDomainEnvelope) -> object:
        """Validate the recipe domain admitted for this adapter.

        Args:
            domain: Host-sealed constraint envelope for the declared adapter and schema.

        Returns:
            Adapter-owned validated domain constraints.

        Raises:
            ValueError: The domain route, schema or constraints are not admitted.
        """
        ...

    def validate_recipe_for_domain(
        self,
        *,
        recipe: AlphaModelRecipeEnvelope,
        domain: AlphaModelSearchDomainEnvelope,
    ) -> object:
        """Require a qualified recipe to belong to the admitted search domain.

        Args:
            recipe: Numerical configuration proposed for execution.
            domain: Qualified bounds or profiles admitted by the Host.

        Returns:
            Validated adapter-owned parameters within the admitted domain.

        Raises:
            ValueError: Route, schema, parameter bounds or profiles are not admitted.
        """
        ...

    def fit(
        self,
        *,
        recipe: AlphaModelRecipeEnvelope,
        inputs: BoundAlphaTrainingInput,
        fit_plan: BoundAlphaModelFitInput | None = None,
    ) -> AlphaModelFitResult:
        """Fit the admitted recipe against immutable Host-bound numerical inputs.

        Args:
            recipe: Qualified numerical configuration.
            inputs: Read-only training surface on the admitted feature axis.
            fit_plan: Qualified direct/nested protocol and optional tuning partitions.

        Returns:
            Learned content, model-neutral state, training diagnostics and numerical call counts.

        Raises:
            ValueError: The recipe, fit authority or resulting content violates the adapter
                boundary.
        """
        ...

    def predict(
        self,
        *,
        estimator: AlphaEstimatorContent,
        inputs: BoundAlphaPredictionInput,
    ) -> AlphaModelPredictionResult:
        """Predict from learned content using the exact fitted feature axis.

        Args:
            estimator: Qualified immutable learned parameters/model content.
            inputs: Read-only prediction matrix on the fitted feature axis.

        Returns:
            Finite read-only predictions aligned with the supplied row axis.

        Raises:
            ValueError: Estimator binding, feature axis or numerical output is invalid.
        """
        ...


__all__ = [
    "NESTED_FIT_RUNTIME_CAPABILITY",
    "AlphaEstimatorContent",
    "AlphaFitProvenanceReceipt",
    "AlphaModelAdapter",
    "AlphaModelFitProtocol",
    "AlphaModelFitResult",
    "AlphaModelFitSidecar",
    "AlphaModelNumericalBinding",
    "AlphaModelPredictionResult",
    "AlphaModelRecipeEnvelope",
    "AlphaModelSearchDomainEnvelope",
    "AlphaModelSelectionDiagnostic",
    "AlphaModelStateProjection",
    "BoundAlphaModelFitInput",
    "BoundAlphaPredictionInput",
    "BoundAlphaTrainingInput",
    "alpha_model_array_content_hash",
]
