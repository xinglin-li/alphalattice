"""Deterministic Host dispatch across the explicit Alpha model catalog."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

from alphalattice.capabilities.alpha_modeling.catalog import AlphaModelCatalog
from alphalattice.capabilities.alpha_modeling.contracts import (
    NESTED_FIT_RUNTIME_CAPABILITY,
    AlphaEstimatorContent,
    AlphaFitProvenanceReceipt,
    AlphaModelAdapter,
    AlphaModelFitResult,
    AlphaModelNumericalBinding,
    AlphaModelPredictionResult,
    AlphaModelRecipeEnvelope,
    AlphaModelSearchDomainEnvelope,
    BoundAlphaModelFitInput,
    BoundAlphaPredictionInput,
    BoundAlphaTrainingInput,
)

from .numerical_environment import (
    AlphaModelNumericalEnvironment,
    alpha_model_numerical_scope,
)


class AlphaModelRuntimeAuthorityError(ValueError):
    """A runtime request or adapter result violates admitted model authority."""

    failure_class = "AUTHORITY_FAILURE"


class _RidgeBatchAdapter(Protocol):
    adapter_id: str

    def fit_ridge_batch(
        self,
        *,
        recipes: tuple[AlphaModelRecipeEnvelope, ...],
        inputs: BoundAlphaTrainingInput,
        fit_plan: BoundAlphaModelFitInput,
    ) -> tuple[AlphaModelFitResult, ...]: ...


@dataclass(frozen=True, slots=True)
class AlphaModelExecutionResult:
    """One admitted fit/predict operation before owner-scoped publication."""

    fit: AlphaModelFitResult
    prediction: AlphaModelPredictionResult
    provenance: AlphaFitProvenanceReceipt
    numerical_binding: AlphaModelNumericalBinding
    numerical_environment: AlphaModelNumericalEnvironment
    fit_plan_hash: str


@dataclass(frozen=True, slots=True)
class AlphaModelFitExecutionResult:
    """One admitted fit that has not consumed an outer prediction surface."""

    fit: AlphaModelFitResult
    provenance: AlphaFitProvenanceReceipt
    numerical_binding: AlphaModelNumericalBinding
    numerical_environment: AlphaModelNumericalEnvironment
    fit_plan_hash: str


@dataclass(frozen=True, slots=True)
class AlphaModelPredictionExecutionResult:
    """One explicitly authorized prediction from already-fitted estimator content."""

    prediction: AlphaModelPredictionResult
    numerical_environment: AlphaModelNumericalEnvironment


class AlphaModelRuntimeService:
    """Validate and execute one adapter without research or publication authority."""

    def __init__(self, catalog: AlphaModelCatalog) -> None:
        """Bind the immutable installed model catalog and its declared numerical capabilities.

        Args:
            catalog: Explicit qualified adapters and schema/numerical capability identities.
        """
        self._catalog = catalog
        self._capability_identities = {
            value.adapter_id: value for value in catalog.binding.ordered_capabilities
        }

    def _admit(
        self,
        *,
        recipe: AlphaModelRecipeEnvelope,
        domain: AlphaModelSearchDomainEnvelope,
    ) -> tuple[AlphaModelAdapter, AlphaModelNumericalBinding]:
        try:
            adapter = self._catalog.admit_recipe(recipe=recipe, domain=domain)
            numerical_binding = adapter.describe_numerical_binding()
            installed_identity = self._capability_identities[adapter.adapter_id]
        except (KeyError, ValueError) as error:
            raise AlphaModelRuntimeAuthorityError(
                "ALPHA_MODEL_RUNTIME_CAPABILITY_AUTHORITY_INVALID"
            ) from error
        if installed_identity.numerical_binding_hash != numerical_binding.numerical_binding_hash:
            raise AlphaModelRuntimeAuthorityError("ALPHA_MODEL_RUNTIME_NUMERICAL_BINDING_INVALID")
        return adapter, numerical_binding

    def resolve_numerical_binding(
        self,
        *,
        recipe: AlphaModelRecipeEnvelope,
        domain: AlphaModelSearchDomainEnvelope,
    ) -> AlphaModelNumericalBinding:
        """Resolve the installed numerical capability before the Host builds a fit plan."""
        _adapter, numerical_binding = self._admit(recipe=recipe, domain=domain)
        return numerical_binding

    def fit(
        self,
        *,
        recipe: AlphaModelRecipeEnvelope,
        domain: AlphaModelSearchDomainEnvelope,
        fit_plan: BoundAlphaModelFitInput,
        training_input: BoundAlphaTrainingInput,
        package_identity_hash: str,
    ) -> AlphaModelFitExecutionResult:
        """Admit a recipe and Host plan, then execute one fit in the measured numerical scope.

        Args:
            recipe: Numerical configuration admitted under the supplied domain.
            domain: Host-qualified search constraints for that adapter.
            fit_plan: Qualified direct/nested protocol bound to the training identity and Feature
                axis.
            training_input: Finite immutable training values admitted by the Host.
            package_identity_hash: Qualified owner package identity retained in fit provenance.

        Returns:
            Learned content/state with fit provenance, numerical binding, measured environment and
            plan identity.

        Raises:
            AlphaModelRuntimeAuthorityError: Installed capability, accepted protocol, parent inputs
                or returned fit binding disagrees.
            ValueError: Adapter numerical validation or fitting cannot be admitted.
        """
        adapter, numerical_binding = self._admit(recipe=recipe, domain=domain)
        # The protocols this recipe accepts, stated by the adapter that reads
        # the plan: an adapter without the nested capability accepts only a
        # direct plan, and a nested-capable adapter says per recipe whether a
        # fixed-iteration fit may take a direct one.
        accepted = adapter.fit_protocols(recipe)
        capabilities = numerical_binding.required_runtime_capabilities
        if NESTED_FIT_RUNTIME_CAPABILITY not in capabilities and accepted != ("DIRECT_FIT",):
            raise AlphaModelRuntimeAuthorityError(
                "ALPHA_MODEL_RUNTIME_CAPABILITY_AUTHORITY_INVALID"
            )
        if (
            numerical_binding.adapter_id != adapter.adapter_id
            or fit_plan.protocol_id not in accepted
            or fit_plan.parent_training_binding_hash != training_input.training_binding_hash
            or fit_plan.ordered_feature_ids != training_input.ordered_feature_ids
        ):
            raise AlphaModelRuntimeAuthorityError("ALPHA_MODEL_RUNTIME_INPUT_BINDING_INVALID")
        with alpha_model_numerical_scope(numerical_binding) as numerical_environment:
            fit = adapter.fit(recipe=recipe, inputs=training_input, fit_plan=fit_plan)
            if (
                fit.estimator_content.adapter_id != adapter.adapter_id
                or fit.estimator_content.ordered_feature_ids != training_input.ordered_feature_ids
                or fit.state_projection.adapter_id != adapter.adapter_id
            ):
                raise AlphaModelRuntimeAuthorityError("ALPHA_MODEL_RUNTIME_FIT_BINDING_INVALID")
        provenance = AlphaFitProvenanceReceipt.create(
            recipe_hash=recipe.recipe_hash,
            training_binding_hash=training_input.training_binding_hash,
            estimator_content_hash=fit.estimator_content.content_hash,
            package_identity_hash=package_identity_hash,
            fit_plan_hash=fit_plan.fit_plan_hash,
            numerical_environment_hash=numerical_environment.environment_hash,
            fit_call_count=fit.fit_call_count,
            predict_call_count=fit.predict_call_count,
        )
        return AlphaModelFitExecutionResult(
            fit=fit,
            provenance=provenance,
            numerical_binding=numerical_binding,
            numerical_environment=numerical_environment,
            fit_plan_hash=fit_plan.fit_plan_hash,
        )

    def fit_ridge_batch(
        self,
        *,
        recipes: tuple[AlphaModelRecipeEnvelope, ...],
        domain: AlphaModelSearchDomainEnvelope,
        fit_plan: BoundAlphaModelFitInput,
        training_input: BoundAlphaTrainingInput,
        package_identity_hash: str,
    ) -> tuple[AlphaModelFitExecutionResult, ...]:
        """Admit one installed Ridge fit over one frozen matrix, at one or more alphas.

        One recipe is admitted. The lower bound of two read "a batch of one is not
        a batch", which was true while the alpha grid was always four; a caller
        that states a single configuration rather than searching now presents
        exactly one, and the shared decomposition this path exists to reuse is
        formed once regardless of how many alphas consume it.
        """
        if not recipes:
            raise AlphaModelRuntimeAuthorityError("ALPHA_MODEL_RUNTIME_BATCH_INPUT_INVALID")
        admitted = tuple(self._admit(recipe=recipe, domain=domain) for recipe in recipes)
        adapters = tuple(value[0] for value in admitted)
        bindings = tuple(value[1] for value in admitted)
        adapter = adapters[0]
        numerical_binding = bindings[0]
        batch_fit = getattr(adapter, "fit_ridge_batch", None)
        if (
            batch_fit is None
            or any(value.adapter_id != adapter.adapter_id for value in adapters)
            or any(value != numerical_binding for value in bindings)
            or NESTED_FIT_RUNTIME_CAPABILITY in numerical_binding.required_runtime_capabilities
            or fit_plan.protocol_id != "DIRECT_FIT"
            or fit_plan.parent_training_binding_hash != training_input.training_binding_hash
            or fit_plan.ordered_feature_ids != training_input.ordered_feature_ids
        ):
            raise AlphaModelRuntimeAuthorityError("ALPHA_MODEL_RUNTIME_BATCH_BINDING_INVALID")
        batch_adapter = cast(_RidgeBatchAdapter, adapter)
        with alpha_model_numerical_scope(numerical_binding) as numerical_environment:
            fits = batch_adapter.fit_ridge_batch(
                recipes=recipes,
                inputs=training_input,
                fit_plan=fit_plan,
            )
        if len(fits) != len(recipes) or any(
            fit.estimator_content.adapter_id != adapter.adapter_id
            or fit.estimator_content.ordered_feature_ids != training_input.ordered_feature_ids
            or fit.state_projection.adapter_id != adapter.adapter_id
            for fit in fits
        ):
            raise AlphaModelRuntimeAuthorityError("ALPHA_MODEL_RUNTIME_BATCH_RESULT_INVALID")
        return tuple(
            AlphaModelFitExecutionResult(
                fit=fit,
                provenance=AlphaFitProvenanceReceipt.create(
                    recipe_hash=recipe.recipe_hash,
                    training_binding_hash=training_input.training_binding_hash,
                    estimator_content_hash=fit.estimator_content.content_hash,
                    package_identity_hash=package_identity_hash,
                    fit_plan_hash=fit_plan.fit_plan_hash,
                    numerical_environment_hash=numerical_environment.environment_hash,
                    fit_call_count=fit.fit_call_count,
                    predict_call_count=fit.predict_call_count,
                ),
                numerical_binding=numerical_binding,
                numerical_environment=numerical_environment,
                fit_plan_hash=fit_plan.fit_plan_hash,
            )
            for recipe, fit in zip(recipes, fits, strict=True)
        )

    def predict(
        self,
        *,
        recipe: AlphaModelRecipeEnvelope,
        domain: AlphaModelSearchDomainEnvelope,
        estimator: AlphaEstimatorContent,
        prediction_input: BoundAlphaPredictionInput,
    ) -> AlphaModelPredictionExecutionResult:
        """Admit the installed recipe and predict from already fitted estimator content.

        Args:
            recipe: Qualified numerical configuration naming the installed adapter.
            domain: Host-admitted recipe constraints.
            estimator: Immutable learned content on the fitted Feature axis.
            prediction_input: Finite read-only values on that same axis.

        Returns:
            Prediction values and the measured numerical environment under which they were produced.

        Raises:
            AlphaModelRuntimeAuthorityError: Capability, estimator/Feature binding or output row
                axis disagrees.
            ValueError: The adapter rejects numerical content or prediction values.
        """
        adapter, numerical_binding = self._admit(recipe=recipe, domain=domain)
        if (
            estimator.adapter_id != adapter.adapter_id
            or estimator.ordered_feature_ids != prediction_input.ordered_feature_ids
        ):
            raise AlphaModelRuntimeAuthorityError("ALPHA_MODEL_RUNTIME_PREDICTION_BINDING_INVALID")
        with alpha_model_numerical_scope(numerical_binding) as numerical_environment:
            prediction = adapter.predict(estimator=estimator, inputs=prediction_input)
        if len(prediction.predictions) != len(prediction_input.features):
            raise AlphaModelRuntimeAuthorityError("ALPHA_MODEL_RUNTIME_PREDICTION_AXIS_INVALID")
        return AlphaModelPredictionExecutionResult(
            prediction=prediction,
            numerical_environment=numerical_environment,
        )

    def execute(
        self,
        *,
        recipe: AlphaModelRecipeEnvelope,
        domain: AlphaModelSearchDomainEnvelope,
        fit_plan: BoundAlphaModelFitInput,
        training_input: BoundAlphaTrainingInput,
        prediction_input: BoundAlphaPredictionInput,
        package_identity_hash: str,
    ) -> AlphaModelExecutionResult:
        """Execute a qualified fit and prediction under one consistent numerical environment.

        Args:
            recipe: Qualified numerical recipe admitted by the installed adapter.
            domain: Host-admitted numerical search constraints.
            fit_plan: Plan whose protocol and tuning partitions belong to the admitted training
                surface.
            training_input: Finite immutable parent training values.
            prediction_input: Read-only outer prediction surface with the same training identity and
                Feature axis.
            package_identity_hash: Qualified owner implementation identity retained in provenance.

        Returns:
            Fit, prediction and combined numerical call provenance with the measured
            capability/environment.

        Raises:
            AlphaModelRuntimeAuthorityError: Input binding, fit/prediction authority or measured
                environments disagree.
            ValueError: An adapter rejects recipe, plan or numerical content.
        """
        if (
            training_input.training_binding_hash != prediction_input.training_binding_hash
            or training_input.ordered_feature_ids != prediction_input.ordered_feature_ids
        ):
            raise AlphaModelRuntimeAuthorityError("ALPHA_MODEL_RUNTIME_INPUT_BINDING_INVALID")
        fitted = self.fit(
            recipe=recipe,
            domain=domain,
            fit_plan=fit_plan,
            training_input=training_input,
            package_identity_hash=package_identity_hash,
        )
        predicted = self.predict(
            recipe=recipe,
            domain=domain,
            estimator=fitted.fit.estimator_content,
            prediction_input=prediction_input,
        )
        if (
            predicted.numerical_environment.environment_hash
            != fitted.numerical_environment.environment_hash
        ):
            raise AlphaModelRuntimeAuthorityError("ALPHA_MODEL_RUNTIME_ENVIRONMENT_CHANGED")
        provenance = AlphaFitProvenanceReceipt.create(
            recipe_hash=recipe.recipe_hash,
            training_binding_hash=training_input.training_binding_hash,
            estimator_content_hash=fitted.fit.estimator_content.content_hash,
            package_identity_hash=package_identity_hash,
            fit_plan_hash=fit_plan.fit_plan_hash,
            numerical_environment_hash=fitted.numerical_environment.environment_hash,
            fit_call_count=fitted.fit.fit_call_count,
            predict_call_count=(
                fitted.fit.predict_call_count + predicted.prediction.predict_call_count
            ),
        )
        return AlphaModelExecutionResult(
            fit=fitted.fit,
            prediction=predicted.prediction,
            provenance=provenance,
            numerical_binding=fitted.numerical_binding,
            numerical_environment=fitted.numerical_environment,
            fit_plan_hash=fitted.fit_plan_hash,
        )


__all__ = [
    "AlphaModelExecutionResult",
    "AlphaModelFitExecutionResult",
    "AlphaModelPredictionExecutionResult",
    "AlphaModelRuntimeAuthorityError",
    "AlphaModelRuntimeService",
]
