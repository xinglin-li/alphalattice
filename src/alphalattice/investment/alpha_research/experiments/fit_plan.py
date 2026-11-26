"""Host-owned direct and nested fit-plan authority for Alpha experiments."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal, Self, cast

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.capabilities.alpha_modeling.contracts import (
    NESTED_FIT_RUNTIME_CAPABILITY,
    AlphaModelAdapter,
    AlphaModelFitProtocol,
    AlphaModelNumericalBinding,
    AlphaModelRecipeEnvelope,
    AlphaModelSearchDomainEnvelope,
    BoundAlphaModelFitInput,
    BoundAlphaTrainingInput,
    alpha_model_array_content_hash,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash

from ..inputs.folds import AlphaFoldArrays

type BoolArray = npt.NDArray[np.bool_]
type FloatArray = npt.NDArray[np.float64]


class AlphaFitPlanAuthorityError(ValueError):
    """Fail-closed fit-plan admission error raised before adapter execution."""

    failure_class = "AUTHORITY_FAILURE"

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class AlphaModelFitPlan(_Contract):
    """Durable Host authority for one model/fold fit protocol."""

    kind: Literal["AlphaModelFitPlan"] = "AlphaModelFitPlan"
    protocol_id: AlphaModelFitProtocol
    parent_training_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    fold_commitment_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    search_domain_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    numerical_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    ordered_feature_ids: tuple[str, ...] = Field(min_length=1)
    outer_training_sessions: tuple[date, ...] = Field(min_length=1)
    outer_training_sessions_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    outer_training_session_count: int = Field(ge=1)
    tuning_training_sessions: tuple[date, ...] | None = None
    tuning_training_sessions_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    tuning_training_session_count: int | None = Field(default=None, ge=1)
    tuning_training_row_axis_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    purge_sessions_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    purge_sessions: tuple[date, ...] | None = None
    purge_session_count: int | None = Field(default=None, ge=1)
    purge_row_axis_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    tuning_validation_sessions_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    tuning_validation_sessions: tuple[date, ...] | None = None
    tuning_validation_session_count: int | None = Field(default=None, ge=1)
    tuning_validation_row_axis_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    tuning_training_feature_values_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    tuning_training_target_values_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    tuning_validation_feature_values_hash: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    tuning_validation_target_values_hash: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    selection_metric_id: str | None = Field(default=None, min_length=1, max_length=64)
    maximum_iterations: int | None = Field(default=None, ge=1)
    early_stopping_rounds: int | None = Field(default=None, ge=1)
    fit_plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: Any) -> Self:
        identity = {"kind": "AlphaModelFitPlan", **values}
        normalized = cls.model_construct(
            **identity,
            fit_plan_hash="0" * 64,
        ).model_dump(mode="json", exclude={"fit_plan_hash"})
        return cls(**normalized, fit_plan_hash=canonical_hash(normalized))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        if (
            self.ordered_feature_ids != tuple(dict.fromkeys(self.ordered_feature_ids))
            or self.outer_training_sessions != tuple(sorted(set(self.outer_training_sessions)))
            or self.outer_training_session_count != len(self.outer_training_sessions)
            or self.outer_training_sessions_hash != canonical_hash(self.outer_training_sessions)
            or self.fit_plan_hash
            != canonical_hash(self.model_dump(mode="json", exclude={"fit_plan_hash"}))
        ):
            raise ValueError("ALPHA_MODEL_FIT_PLAN_IDENTITY_INVALID")
        nested = (
            self.tuning_training_sessions,
            self.tuning_training_sessions_hash,
            self.tuning_training_session_count,
            self.tuning_training_row_axis_hash,
            self.purge_sessions,
            self.purge_sessions_hash,
            self.purge_session_count,
            self.purge_row_axis_hash,
            self.tuning_validation_sessions,
            self.tuning_validation_sessions_hash,
            self.tuning_validation_session_count,
            self.tuning_validation_row_axis_hash,
            self.tuning_training_feature_values_hash,
            self.tuning_training_target_values_hash,
            self.tuning_validation_feature_values_hash,
            self.tuning_validation_target_values_hash,
            self.selection_metric_id,
            self.maximum_iterations,
            self.early_stopping_rounds,
        )
        if self.protocol_id == "DIRECT_FIT":
            if any(value is not None for value in nested):
                raise ValueError("ALPHA_MODEL_DIRECT_FIT_PLAN_INVALID")
            return self
        if any(value is None for value in nested):
            raise ValueError("ALPHA_MODEL_NESTED_FIT_PLAN_INCOMPLETE")
        assert self.tuning_training_session_count is not None
        assert self.tuning_training_sessions is not None
        assert self.purge_session_count is not None
        assert self.purge_sessions is not None
        assert self.tuning_validation_session_count is not None
        assert self.tuning_validation_sessions is not None
        assert self.maximum_iterations is not None
        assert self.early_stopping_rounds is not None
        if (
            self.tuning_training_sessions + self.purge_sessions + self.tuning_validation_sessions
            != self.outer_training_sessions
            or self.tuning_training_session_count != len(self.tuning_training_sessions)
            or self.purge_session_count != len(self.purge_sessions)
            or self.tuning_validation_session_count != len(self.tuning_validation_sessions)
            or self.tuning_training_sessions_hash != canonical_hash(self.tuning_training_sessions)
            or self.purge_sessions_hash != canonical_hash(self.purge_sessions)
            or self.tuning_validation_sessions_hash
            != canonical_hash(self.tuning_validation_sessions)
            or self.early_stopping_rounds > self.maximum_iterations
        ):
            raise ValueError("ALPHA_MODEL_NESTED_FIT_PLAN_INVALID")
        return self


def _row_axis_hash(sessions: tuple[date, ...], listings: tuple[str, ...]) -> str:
    if len(sessions) != len(listings):
        raise AlphaFitPlanAuthorityError("ALPHA_MODEL_FIT_PLAN_ROW_AXIS_INVALID")
    return cast(str, canonical_hash(tuple(zip(sessions, listings, strict=True))))


def _readonly(value: npt.NDArray[np.generic]) -> FloatArray:
    result: FloatArray = np.asarray(value, dtype=np.float64)
    result.setflags(write=False)
    return result


def _nested_counts(domain: AlphaModelSearchDomainEnvelope) -> tuple[int, int, int, str, int, int]:
    constraints = domain.constraints
    try:
        values = (
            int(constraints["inner_training_sessions"]),
            int(constraints["purge_sessions"]),
            int(constraints["inner_validation_sessions"]),
            str(constraints["selection_metric_id"]),
            int(constraints["maximum_iterations"]),
            int(constraints["early_stopping_rounds"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise AlphaFitPlanAuthorityError("ALPHA_MODEL_NESTED_FIT_POLICY_INVALID") from error
    inner, purge, validation, metric, maximum, stopping = values
    if min(inner, purge, validation, maximum, stopping) < 1 or not metric or stopping > maximum:
        raise AlphaFitPlanAuthorityError("ALPHA_MODEL_NESTED_FIT_POLICY_INVALID")
    return values


def resolve_alpha_model_fit_protocol(
    *,
    adapter: AlphaModelAdapter,
    recipe: AlphaModelRecipeEnvelope,
    domain: AlphaModelSearchDomainEnvelope,
) -> AlphaModelFitProtocol:
    """The protocol a development fit plan states for this recipe, proved plannable.

    The adapter says which protocols the recipe accepts and the first is the
    one stated. A nested plan is plannable only where the capability's domain
    states the tuning partition (``inner_training_sessions``, ``purge_sessions``,
    ``inner_validation_sessions``, the selection metric and the iteration
    bounds); a recipe that needs one from a domain stating none is refused here,
    at planning, rather than inside a fold.
    """

    protocol = adapter.fit_protocols(recipe)[0]
    if protocol == "NESTED_EARLY_STOPPING_REFIT":
        _nested_counts(domain)
    return protocol


def build_alpha_model_fit_plan(
    *,
    fold: AlphaFoldArrays,
    training_input: BoundAlphaTrainingInput,
    domain: AlphaModelSearchDomainEnvelope,
    numerical_binding: AlphaModelNumericalBinding,
    protocol: AlphaModelFitProtocol,
) -> tuple[AlphaModelFitPlan, BoundAlphaModelFitInput]:
    """Compile one adapter-neutral fit plan from Host-owned fold/session authority.

    ``protocol`` is what ``resolve_alpha_model_fit_protocol`` answered for the
    recipe: a direct plan fits the whole training surface; a nested plan carries
    the tuning partition the domain states.
    """

    if (
        fold.training_input_binding is None
        or fold.training_input_binding.binding_hash != training_input.training_binding_hash
        or training_input.ordered_feature_ids != fold.ordered_factor_ids
        or domain.adapter_id != numerical_binding.adapter_id
        or fold.commitment.train_session_count != len(fold.training_sessions)
        or fold.commitment.train_sessions_hash != canonical_hash(fold.training_sessions)
        or fold.training_input_binding.training_row_axis_hash
        != _row_axis_hash(fold.training_row_sessions, fold.training_listing_ids)
        or (
            protocol == "NESTED_EARLY_STOPPING_REFIT"
            and NESTED_FIT_RUNTIME_CAPABILITY not in numerical_binding.required_runtime_capabilities
        )
    ):
        raise AlphaFitPlanAuthorityError("ALPHA_MODEL_FIT_PLAN_AUTHORITY_MISMATCH")
    nested_required = protocol == "NESTED_EARLY_STOPPING_REFIT"
    common = {
        "parent_training_binding_hash": training_input.training_binding_hash,
        "fold_commitment_hash": fold.commitment.commitment_hash,
        "search_domain_hash": domain.search_domain_hash,
        "numerical_binding_hash": numerical_binding.numerical_binding_hash,
        "ordered_feature_ids": fold.ordered_factor_ids,
        "outer_training_sessions": fold.training_sessions,
        "outer_training_sessions_hash": canonical_hash(fold.training_sessions),
        "outer_training_session_count": len(fold.training_sessions),
    }
    if not nested_required:
        plan = AlphaModelFitPlan.create(protocol_id="DIRECT_FIT", **common)
        return plan, BoundAlphaModelFitInput(
            fit_plan_hash=plan.fit_plan_hash,
            protocol_id=plan.protocol_id,
            parent_training_binding_hash=plan.parent_training_binding_hash,
            ordered_feature_ids=plan.ordered_feature_ids,
        )

    inner_count, purge_count, validation_count, metric, maximum, stopping = _nested_counts(domain)
    if len(fold.training_sessions) != inner_count + purge_count + validation_count:
        raise AlphaFitPlanAuthorityError("ALPHA_MODEL_NESTED_FIT_WINDOW_MISMATCH")
    if len(fold.training_row_sessions) != len(fold.training_listing_ids):
        raise AlphaFitPlanAuthorityError("ALPHA_MODEL_FIT_PLAN_ROW_AXIS_INVALID")
    inner_sessions = fold.training_sessions[:inner_count]
    purge_sessions = fold.training_sessions[inner_count : inner_count + purge_count]
    validation_sessions = fold.training_sessions[inner_count + purge_count :]
    if (
        inner_sessions + purge_sessions + validation_sessions != fold.training_sessions
        or set(inner_sessions) & set(purge_sessions)
        or set(inner_sessions) & set(validation_sessions)
        or set(purge_sessions) & set(validation_sessions)
    ):
        raise AlphaFitPlanAuthorityError("ALPHA_MODEL_NESTED_FIT_WINDOW_MISMATCH")
    row_sessions = fold.training_row_sessions
    inner_session_set = set(inner_sessions)
    purge_session_set = set(purge_sessions)
    validation_session_set = set(validation_sessions)
    inner_mask: BoolArray = (
        np.asarray([value in inner_session_set for value in row_sessions], dtype=np.bool_)
        & fold.training_model_mask
    )
    purge_mask: BoolArray = np.asarray(
        [value in purge_session_set for value in row_sessions], dtype=np.bool_
    )
    validation_mask: BoolArray = (
        np.asarray([value in validation_session_set for value in row_sessions], dtype=np.bool_)
        & fold.training_model_mask
    )
    if not bool(inner_mask.any()) or not bool(validation_mask.any()):
        raise AlphaFitPlanAuthorityError("ALPHA_MODEL_NESTED_FIT_SURFACE_EMPTY")
    inner_features = _readonly(fold.training_features[inner_mask])
    inner_targets = _readonly(fold.training_targets[inner_mask])
    validation_features = _readonly(fold.training_features[validation_mask])
    validation_targets = _readonly(fold.training_targets[validation_mask])
    inner_row_sessions = tuple(
        value for value, used in zip(row_sessions, inner_mask, strict=True) if used
    )
    inner_row_listings = tuple(
        value for value, used in zip(fold.training_listing_ids, inner_mask, strict=True) if used
    )
    purge_row_sessions = tuple(
        value for value, used in zip(row_sessions, purge_mask, strict=True) if used
    )
    purge_row_listings = tuple(
        value for value, used in zip(fold.training_listing_ids, purge_mask, strict=True) if used
    )
    validation_row_sessions = tuple(
        value for value, used in zip(row_sessions, validation_mask, strict=True) if used
    )
    validation_row_listings = tuple(
        value
        for value, used in zip(fold.training_listing_ids, validation_mask, strict=True)
        if used
    )
    plan = AlphaModelFitPlan.create(
        protocol_id="NESTED_EARLY_STOPPING_REFIT",
        **common,
        tuning_training_sessions=inner_sessions,
        tuning_training_sessions_hash=canonical_hash(inner_sessions),
        tuning_training_session_count=len(inner_sessions),
        tuning_training_row_axis_hash=_row_axis_hash(inner_row_sessions, inner_row_listings),
        purge_sessions=purge_sessions,
        purge_sessions_hash=canonical_hash(purge_sessions),
        purge_session_count=len(purge_sessions),
        purge_row_axis_hash=_row_axis_hash(purge_row_sessions, purge_row_listings),
        tuning_validation_sessions=validation_sessions,
        tuning_validation_sessions_hash=canonical_hash(validation_sessions),
        tuning_validation_session_count=len(validation_sessions),
        tuning_validation_row_axis_hash=_row_axis_hash(
            validation_row_sessions, validation_row_listings
        ),
        tuning_training_feature_values_hash=alpha_model_array_content_hash(inner_features),
        tuning_training_target_values_hash=alpha_model_array_content_hash(inner_targets),
        tuning_validation_feature_values_hash=alpha_model_array_content_hash(validation_features),
        tuning_validation_target_values_hash=alpha_model_array_content_hash(validation_targets),
        selection_metric_id=metric,
        maximum_iterations=maximum,
        early_stopping_rounds=stopping,
    )
    return plan, BoundAlphaModelFitInput(
        fit_plan_hash=plan.fit_plan_hash,
        protocol_id=plan.protocol_id,
        parent_training_binding_hash=plan.parent_training_binding_hash,
        ordered_feature_ids=plan.ordered_feature_ids,
        tuning_training_row_axis_hash=plan.tuning_training_row_axis_hash,
        purge_row_axis_hash=plan.purge_row_axis_hash,
        tuning_validation_row_axis_hash=plan.tuning_validation_row_axis_hash,
        tuning_training_feature_values_hash=plan.tuning_training_feature_values_hash,
        tuning_training_target_values_hash=plan.tuning_training_target_values_hash,
        tuning_validation_feature_values_hash=plan.tuning_validation_feature_values_hash,
        tuning_validation_target_values_hash=plan.tuning_validation_target_values_hash,
        tuning_training_features=inner_features,
        tuning_training_targets=inner_targets,
        tuning_validation_features=validation_features,
        tuning_validation_targets=validation_targets,
        selection_metric_id=plan.selection_metric_id,
        maximum_iterations=plan.maximum_iterations,
        early_stopping_rounds=plan.early_stopping_rounds,
    )


__all__ = [
    "NESTED_FIT_RUNTIME_CAPABILITY",
    "AlphaFitPlanAuthorityError",
    "AlphaModelFitPlan",
    "build_alpha_model_fit_plan",
]
