"""Historical Dynamic Panel model schemas and durable identity helpers.

The rejected Campaign executor and its cost-based Alpha selection surface are
retired. These contracts remain only because the artifact owner has real
read-only consumers for frozen development evidence.
"""

from __future__ import annotations

from typing import Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.capabilities.alpha_modeling.catalog import (
    AlphaModelCatalogBinding,
)
from alphalattice.capabilities.alpha_modeling.contracts import (
    AlphaModelRecipeEnvelope,
    AlphaModelSearchDomainEnvelope,
)
from alphalattice.investment.alpha_research.inputs.dynamic_panel import (
    DynamicPanelViewId,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash

type DynamicPanelModelMethodId = Literal[
    "BEST_SINGLE_FACTOR_CONTROL",
    "FIXED_SIGN_EQUAL_WEIGHT_CONTROL",
    "RESIDUAL_RIDGE_CONTROL",
    "RIDGE",
    "ELASTIC_NET",
    "LIGHTGBM_REGULARIZED",
]
type DynamicPanelTargetLane = Literal[
    "TOTAL_RETURN_TARGET_Z",
    "RESIDUAL_TARGET_Z",
]

_HASH = r"^[0-9a-f]{64}$"
_COST_STRESS_BPS: Final = (2, 5, 10, 20)


class DynamicPanelCampaignError(ValueError):
    """Stable refusal for Dynamic Panel model Campaign authority."""


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


def _contract[T: _Contract](model: type[T], values: dict[str, object], field: str) -> T:
    provisional = model.model_construct(**values, **{field: "0" * 64})
    identity = provisional.model_dump(mode="json", exclude={field})
    return model(**values, **{field: str(canonical_hash(identity))})


class DynamicPanelModelViewBinding(_Contract):
    view_id: DynamicPanelViewId
    view_recipe_hash: str = Field(pattern=_HASH)
    ordered_feature_ids: tuple[str, ...] = Field(min_length=1, max_length=512)
    ordered_feature_axis_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        payload = dict(values)
        payload["ordered_feature_axis_hash"] = str(canonical_hash(payload["ordered_feature_ids"]))
        return cls(**payload)


class DynamicPanelTrialPlan(_Contract):
    trial_index: int = Field(ge=0)
    view_id: DynamicPanelViewId
    method_id: DynamicPanelModelMethodId
    target_lane: DynamicPanelTargetLane
    ordered_feature_ids: tuple[str, ...] = Field(min_length=1, max_length=512)
    recipe: AlphaModelRecipeEnvelope
    search_domain: AlphaModelSearchDomainEnvelope
    plan_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        return _contract(cls, dict(values), "plan_hash")


class DynamicPanelModelProgram(_Contract):
    kind: Literal["DynamicPanelModelProgram"] = "DynamicPanelModelProgram"
    scope: Literal["DEVELOPMENT_ONLY_NO_HOLDOUT"] = "DEVELOPMENT_ONLY_NO_HOLDOUT"
    dynamic_panel_surface_hash: str = Field(pattern=_HASH)
    dynamic_panel_catalog_hash: str = Field(pattern=_HASH)
    dynamic_panel_implementation_hash: str = Field(pattern=_HASH)
    target_evidence_hash: str = Field(pattern=_HASH)
    outcome_method_binding_hash: str = Field(pattern=_HASH)
    split_policy_hash: str = Field(pattern=_HASH)
    fold_context_manifest_hashes: tuple[str, ...] = Field(min_length=1)
    model_catalog_binding: AlphaModelCatalogBinding
    installed_implementation_source_closure_hash: str = Field(pattern=_HASH)
    ordered_views: tuple[DynamicPanelModelViewBinding, ...] = Field(min_length=4, max_length=4)
    ordered_trials: tuple[DynamicPanelTrialPlan, ...] = Field(min_length=361)
    inner_selection_policy: dict[str, object]
    cost_stress_bps: tuple[int, ...]
    score_aggregation_policy: dict[str, object]
    portfolio_policy: dict[str, object]
    program_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        return _contract(cls, dict(values), "program_hash")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        counts: dict[tuple[str, str], int] = {}
        for plan in self.ordered_trials:
            key = (plan.view_id, plan.method_id)
            counts[key] = counts.get(key, 0) + 1
        expected: dict[tuple[str, str], int] = {
            ("RELATIVE_FACTOR_CONTROL", "BEST_SINGLE_FACTOR_CONTROL"): 1,
            (
                "RELATIVE_FACTOR_CONTROL",
                "FIXED_SIGN_EQUAL_WEIGHT_CONTROL",
            ): 4,
            ("RELATIVE_FACTOR_CONTROL", "RESIDUAL_RIDGE_CONTROL"): 4,
        }
        for view in (
            "RELATIVE_FACTOR_CONTROL",
            "DYNAMIC_RELATIVE_PANEL",
            "DYNAMIC_CONTEXT_PANEL",
            "DYNAMIC_JOINT_PANEL",
        ):
            expected[(view, "RIDGE")] = 4
            expected[(view, "ELASTIC_NET")] = 12
            expected[(view, "LIGHTGBM_REGULARIZED")] = 72
        if (
            counts != expected
            or tuple(value.trial_index for value in self.ordered_trials)
            != tuple(range(len(self.ordered_trials)))
            or tuple(value.view_id for value in self.ordered_views)
            != (
                "RELATIVE_FACTOR_CONTROL",
                "DYNAMIC_RELATIVE_PANEL",
                "DYNAMIC_CONTEXT_PANEL",
                "DYNAMIC_JOINT_PANEL",
            )
            or self.cost_stress_bps != _COST_STRESS_BPS
            or self.program_hash
            != canonical_hash(self.model_dump(mode="json", exclude={"program_hash"}))
        ):
            raise DynamicPanelCampaignError("alpha_research.dynamic_panel_model_program_invalid")
        return self


class DynamicPanelCandidateSummary(_Contract):
    candidate_id: str
    fold_count: int = Field(ge=1)
    mean_mse: float = Field(ge=0.0)
    mean_rank_ic: float | None = None
    mean_gross_decile_spread: float | None = None
    mean_turnover: float | None = Field(default=None, ge=0.0)
    mean_net_2bps: float | None = None
    mean_net_5bps: float | None = None
    mean_net_10bps: float | None = None
    mean_net_20bps: float | None = None
    fold_net_5bps: tuple[float | None, ...]
    same_direction_early_late_fold_count: int = Field(ge=0)


class DynamicPanelModelCampaignDossier(_Contract):
    kind: Literal["DynamicPanelModelCampaignDossier"] = "DynamicPanelModelCampaignDossier"
    program_hash: str = Field(pattern=_HASH)
    inner_trial_evidence_hashes: tuple[str, ...] = Field(min_length=1)
    inner_selection_hashes: tuple[str, ...] = Field(min_length=1)
    fold_model_evidence_hashes: tuple[str, ...] = Field(min_length=1)
    candidate_summaries: tuple[DynamicPanelCandidateSummary, ...] = Field(min_length=1)
    causal_dynamic_candidate_by_fold: tuple[tuple[int, str], ...] = Field(min_length=1)
    causal_dynamic_evidence_hash_by_fold: tuple[tuple[int, str], ...] = Field(min_length=1)
    fit_call_count: int = Field(ge=1)
    predict_call_count: int = Field(ge=1)
    metric_call_count: int = Field(ge=1)
    network_call_count: Literal[0] = 0
    holdout_read_count: Literal[0] = 0
    pointer_mutation_count: Literal[0] = 0
    source_write_count: Literal[0] = 0
    dossier_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        return _contract(cls, dict(values), "dossier_hash")


__all__ = [
    "DynamicPanelCampaignError",
    "DynamicPanelCandidateSummary",
    "DynamicPanelModelCampaignDossier",
    "DynamicPanelModelProgram",
    "DynamicPanelModelViewBinding",
    "DynamicPanelTrialPlan",
]
