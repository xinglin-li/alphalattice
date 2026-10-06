"""Autonomous-style consumer that depends only on the public Alpha research seam."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from alphalattice.capabilities.alpha_modeling.catalog import AlphaModelCatalog
from alphalattice.capabilities.alpha_modeling.contracts import (
    AlphaEstimatorContent,
    AlphaModelFitProtocol,
    AlphaModelFitResult,
    AlphaModelNumericalBinding,
    AlphaModelPredictionResult,
    AlphaModelRecipeEnvelope,
    AlphaModelSearchDomainEnvelope,
    AlphaModelStateProjection,
    BoundAlphaModelFitInput,
    BoundAlphaPredictionInput,
    BoundAlphaTrainingInput,
)
from alphalattice.investment.alpha_research.experiments.bindings import (
    build_alpha_development_program,
)
from alphalattice.investment.alpha_research.experiments.contracts import (
    AlphaDevelopmentProgram,
    AlphaExperimentBatch,
    AlphaExperimentBatchResult,
    seal_contract,
)
from alphalattice.investment.alpha_research.experiments.development_artifacts import (
    AlphaDevelopmentArtifactStore,
)
from alphalattice.investment.alpha_research.experiments.execution import execute_alpha_model_batch
from alphalattice.investment.alpha_research.experiments.mandate import (
    AlphaModelCapabilityMandate,
    AlphaModelRecipeProposal,
)
from alphalattice.investment.alpha_research.inputs.folds import (
    AlphaArrayWorkspaceLike,
    AlphaFoldArrayPlan,
)

ADAPTER_ID = "consumer_owned_mean"
RECIPE_SCHEMA_ID = "autonomous-consumer.mean"
DOMAIN_SCHEMA_ID = "autonomous-consumer.mean.search-domain"
CONTENT_FORMAT_ID = "autonomous-consumer.mean.scalar"
STATE_SCHEMA_ID = "autonomous-consumer.mean.development-state"


class ConsumerOwnedMeanAdapter:
    """Small deterministic adapter proving that the consumer owns model mechanics."""

    adapter_id = ADAPTER_ID
    recipe_schema_id = RECIPE_SCHEMA_ID
    search_domain_schema_id = DOMAIN_SCHEMA_ID
    observed_fit_calls = 0

    def describe_numerical_binding(self) -> AlphaModelNumericalBinding:
        return AlphaModelNumericalBinding.create(
            adapter_id=self.adapter_id,
            estimator_content_format_id=CONTENT_FORMAT_ID,
            implementation_owners=(
                "case-study.autonomous-alpha-consumer.ConsumerOwnedMeanAdapter",
            ),
            deterministic_policy={"statistic": "scaled_training_mean"},
            required_runtime_capabilities=("numpy",),
        )

    def fit_protocols(self, recipe: AlphaModelRecipeEnvelope) -> tuple[AlphaModelFitProtocol, ...]:
        del recipe
        return ("DIRECT_FIT",)

    def validate_recipe(self, recipe: AlphaModelRecipeEnvelope) -> float:
        if recipe.adapter_id != self.adapter_id or recipe.recipe_schema_id != self.recipe_schema_id:
            raise ValueError("CONSUMER_MEAN_RECIPE_ROUTE_INVALID")
        scale = float(recipe.parameters["scale"])
        if set(recipe.parameters) != {"scale"} or not np.isfinite(scale):
            raise ValueError("CONSUMER_MEAN_RECIPE_INVALID")
        return scale

    def validate_search_domain(self, domain: AlphaModelSearchDomainEnvelope) -> tuple[float, float]:
        if (
            domain.adapter_id != self.adapter_id
            or domain.recipe_schema_id != self.recipe_schema_id
            or domain.search_domain_schema_id != self.search_domain_schema_id
            or set(domain.constraints) != {"scale_min", "scale_max"}
        ):
            raise ValueError("CONSUMER_MEAN_DOMAIN_INVALID")
        return float(domain.constraints["scale_min"]), float(domain.constraints["scale_max"])

    def validate_recipe_for_domain(
        self,
        *,
        recipe: AlphaModelRecipeEnvelope,
        domain: AlphaModelSearchDomainEnvelope,
    ) -> float:
        scale = self.validate_recipe(recipe)
        lower, upper = self.validate_search_domain(domain)
        if not lower <= scale <= upper:
            raise ValueError("ALPHA_MODEL_RECIPE_OUTSIDE_SEARCH_DOMAIN")
        return scale

    def fit(
        self,
        *,
        recipe: AlphaModelRecipeEnvelope,
        inputs: BoundAlphaTrainingInput,
        fit_plan: BoundAlphaModelFitInput | None = None,
    ) -> AlphaModelFitResult:
        if fit_plan is None or (
            fit_plan.protocol_id != "DIRECT_FIT"
            or fit_plan.parent_training_binding_hash != inputs.training_binding_hash
            or fit_plan.ordered_feature_ids != inputs.ordered_feature_ids
        ):
            raise ValueError("CONSUMER_MEAN_FIT_PLAN_INVALID")
        type(self).observed_fit_calls += 1
        scale = self.validate_recipe(recipe)
        fitted_mean = float(np.mean(inputs.targets)) * scale
        slope = 0.1 * scale
        training_predictions = fitted_mean + slope * inputs.features[:, 0]
        content = AlphaEstimatorContent.create(
            adapter_id=self.adapter_id,
            content_format_id=CONTENT_FORMAT_ID,
            ordered_feature_ids=inputs.ordered_feature_ids,
            payload={"fitted_mean_hex": fitted_mean.hex(), "slope_hex": slope.hex()},
        )
        projection = AlphaModelStateProjection.create(
            adapter_id=self.adapter_id,
            model_family_id="consumer_owned_mean",
            state_kind="EXTERNAL_DEVELOPMENT",
            state_schema_id=STATE_SCHEMA_ID,
            payload={"fitted_mean_hex": fitted_mean.hex(), "slope_hex": slope.hex()},
        )
        return AlphaModelFitResult(
            estimator_content=content,
            state_projection=projection,
            training_mse=float(np.mean(np.square(inputs.targets - training_predictions))),
            iteration_count=None,
            fit_call_count=1,
            predict_call_count=1,
        )

    def predict(
        self,
        *,
        estimator: AlphaEstimatorContent,
        inputs: BoundAlphaPredictionInput,
    ) -> AlphaModelPredictionResult:
        if (
            estimator.adapter_id != self.adapter_id
            or estimator.content_format_id != CONTENT_FORMAT_ID
            or estimator.ordered_feature_ids != inputs.ordered_feature_ids
        ):
            raise ValueError("CONSUMER_MEAN_ESTIMATOR_INVALID")
        fitted_mean = float.fromhex(str(estimator.payload["fitted_mean_hex"]))
        slope = float.fromhex(str(estimator.payload["slope_hex"]))
        predictions = np.asarray(fitted_mean + slope * inputs.features[:, 0], dtype=np.float64)
        predictions.setflags(write=False)
        return AlphaModelPredictionResult(predictions=predictions)


def build_consumer_domain() -> AlphaModelSearchDomainEnvelope:
    return AlphaModelSearchDomainEnvelope.create(
        adapter_id=ADAPTER_ID,
        recipe_schema_id=RECIPE_SCHEMA_ID,
        search_domain_schema_id=DOMAIN_SCHEMA_ID,
        constraints={"scale_min": 0.5, "scale_max": 1.5},
    )


def build_external_development_authority(
    fold_plan: AlphaFoldArrayPlan,
) -> tuple[
    AlphaModelCatalog,
    AlphaModelCapabilityMandate,
    AlphaDevelopmentProgram,
    AlphaExperimentBatch,
]:
    """Seal the catalog, mandate, program, and batch without Goal governance."""

    adapter = ConsumerOwnedMeanAdapter()
    catalog = AlphaModelCatalog((adapter,))
    mandate = AlphaModelCapabilityMandate.create(
        catalog_binding=catalog.binding,
        ordered_search_domains=(build_consumer_domain(),),
    )
    target_lane = fold_plan.target_policy.lane if fold_plan.target_policy is not None else None
    recipe = mandate.admit_proposal(
        proposal=AlphaModelRecipeProposal(
            capability_handle="capability-1",
            parameters={"scale": 1.0},
            target_lane=target_lane,
        ),
        catalog=catalog,
        admitted_target_lanes=(target_lane,) if target_lane is not None else None,
    )
    program = build_alpha_development_program(
        fold_plan=fold_plan,
        model_mandate=mandate,
    )
    batch = seal_contract(
        AlphaExperimentBatch,
        {
            "program_hash": program.program_hash,
            "batch_index": 1,
            "specs": (recipe,),
            "predecessor_batch_hash": None,
        },
        "batch_hash",
    )
    return catalog, mandate, program, batch


def run_external_development(
    *,
    fold_plan: AlphaFoldArrayPlan,
    array_workspace: AlphaArrayWorkspaceLike,
    artifact_root: Path,
) -> tuple[AlphaDevelopmentProgram, AlphaExperimentBatchResult]:
    """Run one consumer-owned model through the product development boundary."""

    catalog, mandate, program, batch = build_external_development_authority(fold_plan)
    result = execute_alpha_model_batch(
        program=program,
        batch=batch,
        fold_plan=fold_plan,
        store=AlphaDevelopmentArtifactStore(artifact_root),
        array_workspace=array_workspace,
        model_catalog=catalog,
        model_mandate=mandate,
    )
    return program, result


__all__ = [
    "ConsumerOwnedMeanAdapter",
    "build_consumer_domain",
    "build_external_development_authority",
    "run_external_development",
]
