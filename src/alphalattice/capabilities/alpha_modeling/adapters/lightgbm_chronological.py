"""Chronological multi-seed LightGBM development capability."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Final, cast

import numpy as np
import numpy.typing as npt

from alphalattice.kernel.shared_kernel.identity import canonical_hash

from ..contracts import (
    NESTED_FIT_RUNTIME_CAPABILITY,
    AlphaEstimatorContent,
    AlphaModelFitProtocol,
    AlphaModelFitResult,
    AlphaModelNumericalBinding,
    AlphaModelPredictionResult,
    AlphaModelRecipeEnvelope,
    AlphaModelSearchDomainEnvelope,
    AlphaModelSelectionDiagnostic,
    AlphaModelStateProjection,
    BoundAlphaModelFitInput,
    BoundAlphaPredictionInput,
    BoundAlphaTrainingInput,
    alpha_model_array_content_hash,
)
from ..runtime.lightgbm_threads import lightgbm_fit_threads, model_trees, sequential_l2
from ..runtime.numerical_environment import SINGLE_THREAD_RUNTIME_CAPABILITY
from .lightgbm import _load_lightgbm

CHRONOLOGICAL_LIGHTGBM_ADAPTER_ID = "lightgbm_chronological_multi_seed"
CHRONOLOGICAL_LIGHTGBM_RECIPE_SCHEMA_ID = "alpha-model.lightgbm-chronological-multi-seed"
CHRONOLOGICAL_LIGHTGBM_SEARCH_DOMAIN_SCHEMA_ID = (
    "alpha-model.lightgbm-chronological-multi-seed.search-domain"
)
CHRONOLOGICAL_LIGHTGBM_CONTENT_FORMAT_ID = "alpha-model.lightgbm-chronological.model-text"
REGULARIZED_LIGHTGBM_ADAPTER_ID = "lightgbm_chronological_regularized"
REGULARIZED_LIGHTGBM_RECIPE_SCHEMA_ID = "alpha-model.lightgbm-chronological-regularized"
REGULARIZED_LIGHTGBM_SEARCH_DOMAIN_SCHEMA_ID = (
    "alpha-model.lightgbm-chronological-regularized.search-domain"
)

_DATASET_CACHE: ContextVar[dict[str, Any] | None] = ContextVar(
    "chronological_lightgbm_dataset_cache", default=None
)
"""Fold-local, process-local binned-Dataset reuse; absent unless a scope opens it.

A ``ContextVar`` rather than a module global so the scope is explicit and each
worker process carries its own. Outside a scope the adapter builds every
Dataset fresh, which is exactly what it did before this existed.
"""


def load_lightgbm_runtime() -> Any:
    """Expose the shared optional runtime loader to installed sibling adapters."""
    return _load_lightgbm()


@contextmanager
def chronological_lightgbm_dataset_cache() -> Iterator[None]:
    """Reuse binned Datasets across the plans of one outer fold, then release.

    LightGBM spends most of a fit binning its input, and the plans of one fold
    that share a seed and a leaf minimum bin byte-identical data. Holding those
    Datasets for the fold turns sixty constructions into six; releasing them at
    the fold boundary keeps the memory bounded to one fold's surfaces.

    Deliberately not a service, a manager or a persistent store: the scope is a
    fold, the lifetime is a ``with`` block, and nothing survives it.
    """
    token = _DATASET_CACHE.set({})
    try:
        yield
    finally:
        cache = _DATASET_CACHE.get()
        if cache is not None:
            cache.clear()
        _DATASET_CACHE.reset(token)


_DATASET_CONSTRUCTION_DEFAULTS: Final[dict[str, object]] = {
    "max_bin": 255,
    "max_bin_by_feature": None,
    "min_data_in_bin": 3,
    "bin_construct_sample_cnt": 200_000,
    "use_missing": True,
    "zero_as_missing": False,
    "feature_pre_filter": True,
    "enable_bundle": True,
    "is_enable_sparse": True,
    "pre_partition": False,
    "two_round": False,
    "linear_tree": False,
    "precise_float_parser": False,
    "categorical_feature": "",
    "forcedbins_filename": "",
    "device_type": "cpu",
}
"""Every installed value that can change how a Dataset is binned.

Spelled out rather than left implicit. These are LightGBM's own defaults, and
the recipe surface does not vary them today -- but a cache key that only bound
the values somebody happened to override would silently start reusing across a
real difference the first time the recipe grew.
"""

_DATASET_KEY_ONLY: Final[frozenset[str]] = frozenset({"categorical_feature", "max_bin_by_feature"})
"""Construction facts that belong in the key but not in LightGBM's params dict.

``categorical_feature`` is a constructor argument -- passing it as a parameter
makes LightGBM warn and ignore it -- and ``max_bin_by_feature`` carries no
value here. Both still decide how an input bins, so the key keeps them.
"""

_DATASET_EXECUTION_ONLY: Final[frozenset[str]] = frozenset({"num_threads", "verbosity"})
"""Operator pacing and logging, passed to LightGBM but excluded from Dataset reuse identity."""


@dataclass(frozen=True, slots=True)
class ChronologicalLightGBMParameters:
    """Native LightGBM shape shared by both installed grids.

    ``num_leaves`` admits ``8`` because the regularized grid spells the cap a
    depth-three tree already has rather than relying on LightGBM to clamp a
    larger number silently. Which values a given experiment may actually use is
    decided by its search domain, not here -- ``allowed_num_leaves`` on the
    bounded domain still refuses anything but fifteen and thirty-one.
    """

    seed: int
    num_leaves: int
    learning_rate: float
    max_depth: int
    min_child_samples: int

    def __post_init__(self) -> None:
        """Require the declared seed, capacity and learning fields to use registered values.

        Raises:
            ValueError: A seed, leaf count, learning rate, depth or child minimum lies outside the
                registered set.
        """
        if (
            self.seed not in {1729, 2718, 31415}
            or self.num_leaves not in {8, 15, 31}
            or self.learning_rate not in {0.03, 0.05, 0.1}
            or self.max_depth not in {3, 5}
            or self.min_child_samples not in {20, 50}
        ):
            raise ValueError("ALPHA_CHRONOLOGICAL_LIGHTGBM_PARAMETER_INVALID")


@dataclass(frozen=True, slots=True)
class RegularizedChronologicalLightGBMParameters(ChronologicalLightGBMParameters):
    """Declare registered capacity, regularization and sampling profiles for chronological LightGBM.

    Capacity, penalty and bagging choices are admitted as paired profiles. Seed,
    learning rate, leaf minimum and feature fraction stay within the installed plan.
    """

    lambda_l1: float
    lambda_l2: float
    min_gain_to_split: float = 0.0
    feature_fraction: float = 1.0
    bagging_fraction: float = 1.0
    bagging_freq: int = 0

    def __post_init__(self) -> None:
        """Require registered capacity, penalty, feature and bagging profiles.

        Raises:
            ValueError: Scalar choices or paired capacity/regularization/bagging profiles are not
                registered.
        """
        if (
            self.seed not in {1729, 2718, 31415}
            or (self.max_depth, self.num_leaves) not in {(3, 8), (5, 31)}
            or self.learning_rate != 0.03
            or self.min_child_samples != 50
            or (self.lambda_l1, self.lambda_l2) not in {(0.0, 0.0), (0.0, 1000.0), (0.0, 10000.0)}
            or self.min_gain_to_split != 0.0
            or self.feature_fraction not in {0.5, 1.0}
            or (self.bagging_fraction, self.bagging_freq) not in {(1.0, 0), (0.5, 1)}
        ):
            raise ValueError("ALPHA_REGULARIZED_LIGHTGBM_PARAMETER_INVALID")


def build_chronological_lightgbm_recipe(
    parameters: ChronologicalLightGBMParameters,
) -> AlphaModelRecipeEnvelope:
    """Seal one chronological LightGBM recipe without research or target authority.

    Args:
        parameters: Validated registered numerical fields for this adapter.

    Returns:
        Host recipe envelope bound to this adapter, its schema and the supplied parameters.
    """
    return AlphaModelRecipeEnvelope.create(
        adapter_id=CHRONOLOGICAL_LIGHTGBM_ADAPTER_ID,
        recipe_schema_id=CHRONOLOGICAL_LIGHTGBM_RECIPE_SCHEMA_ID,
        parameters={
            "seed": parameters.seed,
            "num_leaves": parameters.num_leaves,
            "learning_rate": parameters.learning_rate,
            "max_depth": parameters.max_depth,
            "min_child_samples": parameters.min_child_samples,
        },
    )


def build_chronological_lightgbm_search_domain() -> AlphaModelSearchDomainEnvelope:
    """Seal the installed chronological LightGBM search constraints.

    Returns:
        Qualified domain envelope preserving registered parameter bounds/profiles and numerical
        policy.
    """
    return AlphaModelSearchDomainEnvelope.create(
        adapter_id=CHRONOLOGICAL_LIGHTGBM_ADAPTER_ID,
        recipe_schema_id=CHRONOLOGICAL_LIGHTGBM_RECIPE_SCHEMA_ID,
        search_domain_schema_id=CHRONOLOGICAL_LIGHTGBM_SEARCH_DOMAIN_SCHEMA_ID,
        constraints={
            "allowed_seeds": [1729, 2718, 31415],
            "allowed_num_leaves": [15, 31],
            "allowed_learning_rates": [0.03, 0.05, 0.1],
            "allowed_max_depth": [3, 5],
            "allowed_min_child_samples": [20, 50],
            "maximum_iterations": 500,
            "early_stopping_rounds": 50,
            "selection_metric_id": "l2",
            "configured_thread_count": 1,
            "search_configuration_count": 20,
            "seeds_per_configuration": 3,
            "selection_rule": "ONE_STANDARD_ERROR_THEN_COMPLEXITY",
        },
    )


def build_regularized_chronological_lightgbm_recipe(
    parameters: RegularizedChronologicalLightGBMParameters,
) -> AlphaModelRecipeEnvelope:
    """Seal one regularized chronological LightGBM recipe without research or target authority.

    Args:
        parameters: Validated registered numerical fields for this adapter.

    Returns:
        Host recipe envelope bound to this adapter, its schema and the supplied parameters.
    """
    return AlphaModelRecipeEnvelope.create(
        adapter_id=REGULARIZED_LIGHTGBM_ADAPTER_ID,
        recipe_schema_id=REGULARIZED_LIGHTGBM_RECIPE_SCHEMA_ID,
        parameters={
            "seed": parameters.seed,
            "num_leaves": parameters.num_leaves,
            "learning_rate": parameters.learning_rate,
            "max_depth": parameters.max_depth,
            "min_child_samples": parameters.min_child_samples,
            "lambda_l1": parameters.lambda_l1,
            "lambda_l2": parameters.lambda_l2,
            "min_gain_to_split": parameters.min_gain_to_split,
            "feature_fraction": parameters.feature_fraction,
            "bagging_fraction": parameters.bagging_fraction,
            "bagging_freq": parameters.bagging_freq,
        },
    )


def build_regularized_chronological_lightgbm_search_domain() -> AlphaModelSearchDomainEnvelope:
    """Seal the installed regularized chronological LightGBM search constraints.

    Returns:
        Qualified domain envelope preserving registered parameter bounds/profiles and numerical
        policy.
    """
    return AlphaModelSearchDomainEnvelope.create(
        adapter_id=REGULARIZED_LIGHTGBM_ADAPTER_ID,
        recipe_schema_id=REGULARIZED_LIGHTGBM_RECIPE_SCHEMA_ID,
        search_domain_schema_id=REGULARIZED_LIGHTGBM_SEARCH_DOMAIN_SCHEMA_ID,
        constraints={
            "allowed_seeds": [1729, 2718, 31415],
            "allowed_capacity_profiles": [[3, 8], [5, 31]],
            "allowed_learning_rates": [0.03],
            "allowed_min_child_samples": [50],
            "allowed_lambda_profiles": [
                [0.0, 0.0],
                [0.0, 1000.0],
                [0.0, 10000.0],
            ],
            "allowed_min_gain_to_split": [0.0],
            "allowed_feature_fraction": [0.5, 1.0],
            "allowed_bagging_profiles": [[1.0, 0], [0.5, 1]],
            "maximum_iterations": 500,
            "early_stopping_rounds": 50,
            "selection_metric_id": "l2",
            "configured_thread_count": 1,
            "search_configuration_count": 24,
            "seeds_per_configuration": 3,
            "selection_rule": "ONE_STANDARD_ERROR_THEN_SIMPLER_AND_STRONGER_REGULARIZATION",
        },
    )


class ChronologicalLightGBMAdapter:
    """Select boosting iterations on a chronological inner split, then refit the parent surface.

    The Host supplies immutable tuning partitions and the qualified fit policy.
    Dataset reuse is limited to identical fold-local construction; learned tree
    content remains separate from fit provenance and inner-selection evidence.
    """

    adapter_id = CHRONOLOGICAL_LIGHTGBM_ADAPTER_ID
    recipe_schema_id = CHRONOLOGICAL_LIGHTGBM_RECIPE_SCHEMA_ID
    search_domain_schema_id = CHRONOLOGICAL_LIGHTGBM_SEARCH_DOMAIN_SCHEMA_ID

    def describe_numerical_binding(self) -> AlphaModelNumericalBinding:
        """Seal declared chronological inner selection and parent refit and runtime requirements.

        Returns:
            Adapter/content-format handles, implementation owners, deterministic policy and required
            capabilities.
        """
        return AlphaModelNumericalBinding.create(
            adapter_id=self.adapter_id,
            estimator_content_format_id=CHRONOLOGICAL_LIGHTGBM_CONTENT_FORMAT_ID,
            implementation_owners=("lightgbm.Booster", "lightgbm.Dataset", "lightgbm.train"),
            deterministic_policy={
                "objective": "l2",
                "metric": "l2",
                "deterministic": True,
                "force_col_wise": True,
                "sampling": "none",
                "admitted_seeds": (1729, 2718, 31415),
                "thread_count": 1,
                "selection_rule": "ONE_STANDARD_ERROR_THEN_COMPLEXITY",
                "dataset_reuse": "FOLD_LOCAL_IDENTICAL_CONSTRUCTION_ONLY",
            },
            required_runtime_capabilities=(
                "numpy",
                "lightgbm==4.7.0",
                NESTED_FIT_RUNTIME_CAPABILITY,
                SINGLE_THREAD_RUNTIME_CAPABILITY,
            ),
        )

    def fit_protocols(self, recipe: AlphaModelRecipeEnvelope) -> tuple[AlphaModelFitProtocol, ...]:
        """Declare the fit protocol consumed by this installed adapter.

        Args:
            recipe: Protocol argument; this adapter has no per-recipe protocol variation.

        Returns:
            The nested early-stopping/refit protocol.
        """
        del recipe
        return ("NESTED_EARLY_STOPPING_REFIT",)

    def validate_recipe(self, recipe: AlphaModelRecipeEnvelope) -> ChronologicalLightGBMParameters:
        """Validate this adapter's route and registered numerical recipe fields.

        Args:
            recipe: Qualified envelope naming this adapter and recipe schema.

        Returns:
            Parsed family-specific numerical parameters.

        Raises:
            ValueError: Recipe route or registered numerical fields are invalid.
        """
        if recipe.adapter_id != self.adapter_id or recipe.recipe_schema_id != self.recipe_schema_id:
            raise ValueError("ALPHA_CHRONOLOGICAL_LIGHTGBM_RECIPE_ROUTE_INVALID")
        return ChronologicalLightGBMParameters(**recipe.parameters)

    def validate_search_domain(self, domain: AlphaModelSearchDomainEnvelope) -> dict[str, object]:
        """Validate the qualified search-domain route and constraints.

        Args:
            domain: Sealed constraints proposed for this adapter.

        Returns:
            Copied constraints equal to the installed domain.

        Raises:
            ValueError: The domain differs from the installed declaration.
        """
        expected = build_chronological_lightgbm_search_domain()
        if domain != expected:
            raise ValueError("ALPHA_CHRONOLOGICAL_LIGHTGBM_SEARCH_DOMAIN_INVALID")
        return dict(domain.constraints)

    def validate_recipe_for_domain(
        self, *, recipe: AlphaModelRecipeEnvelope, domain: AlphaModelSearchDomainEnvelope
    ) -> ChronologicalLightGBMParameters:
        """Require the numerical recipe to belong to its admitted bounds or profiles.

        Args:
            recipe: Qualified numerical configuration.
            domain: Host-admitted constraints for this adapter.

        Returns:
            Parsed parameters within every admitted bound/profile.

        Raises:
            ValueError: Recipe/domain routes, registered fields, bounds or paired profiles are
                invalid.
        """
        parameters = self.validate_recipe(recipe)
        constraints = self.validate_search_domain(domain)
        for value, key in (
            (parameters.seed, "allowed_seeds"),
            (parameters.num_leaves, "allowed_num_leaves"),
            (parameters.learning_rate, "allowed_learning_rates"),
            (parameters.max_depth, "allowed_max_depth"),
            (parameters.min_child_samples, "allowed_min_child_samples"),
        ):
            if value not in constraints[key]:  # type: ignore[operator]
                raise ValueError("ALPHA_MODEL_RECIPE_OUTSIDE_SEARCH_DOMAIN")
        return parameters

    @staticmethod
    def _dataset_params(parameters: ChronologicalLightGBMParameters) -> dict[str, object]:
        """The complete set of values that decide how this input is binned.

        ``min_data_in_leaf`` belongs here because ``feature_pre_filter`` bakes it
        into the Dataset: LightGBM itself refuses to retrain a constructed
        Dataset under a different leaf minimum, and two plans that differ in it
        genuinely bin differently.
        """
        return {
            **_DATASET_CONSTRUCTION_DEFAULTS,
            "min_data_in_leaf": parameters.min_child_samples,
            "data_random_seed": parameters.seed,
            "seed": parameters.seed,
            "deterministic": True,
            "force_col_wise": True,
            "num_threads": lightgbm_fit_threads(),
            "verbosity": -1,
        }

    @staticmethod
    def _native_params(parameters: ChronologicalLightGBMParameters) -> dict[str, object]:
        """The training parameters, in LightGBM's own names.

        Previously these travelled through ``LGBMRegressor``, which translated
        the scikit-learn spellings and filled its own defaults before handing
        them to LightGBM. Training now goes through ``lgb.train`` so a
        pre-binned Dataset can be reused, which means this must reproduce that
        translation exactly -- including the wrapper defaults that were never
        written down here, all of which happen to equal LightGBM's own.
        """
        return {
            "objective": "l2",
            "metric": "l2",
            "num_leaves": parameters.num_leaves,
            "learning_rate": parameters.learning_rate,
            "max_depth": parameters.max_depth,
            "min_data_in_leaf": parameters.min_child_samples,
            # scikit-learn wrapper defaults, spelled in LightGBM's names.
            "min_sum_hessian_in_leaf": 1e-3,
            "min_gain_to_split": 0.0,
            "lambda_l1": 0.0,
            "lambda_l2": 0.0,
            "bagging_fraction": 1.0,
            "bagging_freq": 0,
            "feature_fraction": 1.0,
            "bin_construct_sample_cnt": 200_000,
            "deterministic": True,
            "force_col_wise": True,
            # The operator's, behind the sealed canary (runtime.lightgbm_threads): the
            # trees are the same at any count.
            "num_threads": lightgbm_fit_threads(),
            "seed": parameters.seed,
            "bagging_seed": parameters.seed,
            "feature_fraction_seed": parameters.seed,
            "data_random_seed": parameters.seed,
            "drop_seed": parameters.seed,
            "extra_seed": parameters.seed,
            "verbosity": -1,
        }

    @staticmethod
    def _dataset(
        lightgbm: Any,
        *,
        features: npt.NDArray[np.float64],
        targets: npt.NDArray[np.float64],
        ordered_feature_ids: tuple[str, ...],
        dataset_params: dict[str, object],
        reference_key: str | None = None,
        reference: Any = None,
    ) -> tuple[Any, str]:
        """Construct or reuse the binned Dataset for exactly this input.

        The key binds the array contents, the ordered feature axis, the dtype
        and shape, every construction parameter above, and -- for a validation
        set -- the training Dataset it borrows its bins from. Anything that
        could bin differently changes the key, so a hit is the same Dataset by
        construction rather than by assumption.
        """
        key = str(
            canonical_hash(
                {
                    "feature_values_hash": alpha_model_array_content_hash(features),
                    "target_values_hash": alpha_model_array_content_hash(targets),
                    "ordered_feature_ids": ordered_feature_ids,
                    "shape": list(features.shape),
                    "dtype": features.dtype.str,
                    "target_dtype": targets.dtype.str,
                    "dataset_params": {
                        name: dataset_params[name]
                        for name in sorted(dataset_params)
                        if name not in _DATASET_EXECUTION_ONLY
                    },
                    "reference_key": reference_key,
                }
            )
        )
        cache = _DATASET_CACHE.get()
        if cache is not None and key in cache:
            return cache[key], key
        dataset = lightgbm.Dataset(
            features,
            label=targets,
            params={
                name: value
                for name, value in dataset_params.items()
                if name not in _DATASET_KEY_ONLY
            },
            reference=reference,
        ).construct()
        if cache is not None:
            cache[key] = dataset
        return dataset, key

    selection_metric_id = "l2"
    maximum_iterations = 500
    early_stopping_rounds = 50

    def _fit_policy(self, parameters: ChronologicalLightGBMParameters) -> tuple[str, int, int]:
        del parameters
        return self.selection_metric_id, self.maximum_iterations, self.early_stopping_rounds

    def _evaluation(
        self,
        *,
        fit_plan: BoundAlphaModelFitInput,
        parameters: ChronologicalLightGBMParameters,
    ) -> tuple[object, Callable[..., object] | None]:
        """Return LightGBM's metric parameter and optional installed evaluator.

        The l2 is computed here in LightGBM's one-thread order: its built-in metric
        sums in parallel, and the last bits of the curve early stopping reads moved
        with the thread count.
        """
        del fit_plan, parameters
        return "None", sequential_l2

    def fit(
        self,
        *,
        recipe: AlphaModelRecipeEnvelope,
        inputs: BoundAlphaTrainingInput,
        fit_plan: BoundAlphaModelFitInput | None = None,
    ) -> AlphaModelFitResult:
        """Select inner boosting iterations and refit the admitted parent training surface.

        Args:
            recipe: Qualified adapter recipe whose numerical fields are validated.
            inputs: Finite read-only parent training matrix and target values.
            fit_plan: Required bound nested tuning partitions and policy.

        Returns:
            Learned tree content, model-neutral state, training MSE and selected-iteration evidence;
            tuning and parent refit are recorded as two fit calls.

        Raises:
            ValueError: Recipe/plan authority, convergence or resulting numerical content is
                invalid.
        """
        parameters = self.validate_recipe(recipe)
        selection_metric_id, maximum_iterations, early_stopping_rounds = self._fit_policy(
            parameters
        )
        if (
            fit_plan is None
            or fit_plan.protocol_id != "NESTED_EARLY_STOPPING_REFIT"
            or fit_plan.parent_training_binding_hash != inputs.training_binding_hash
            or fit_plan.ordered_feature_ids != inputs.ordered_feature_ids
            or fit_plan.selection_metric_id != selection_metric_id
            or fit_plan.maximum_iterations != maximum_iterations
            or fit_plan.early_stopping_rounds != early_stopping_rounds
            or fit_plan.tuning_training_features is None
            or fit_plan.tuning_training_targets is None
            or fit_plan.tuning_validation_features is None
            or fit_plan.tuning_validation_targets is None
        ):
            raise ValueError("ALPHA_CHRONOLOGICAL_LIGHTGBM_FIT_PLAN_INVALID")
        lightgbm = _load_lightgbm()
        native = self._native_params(parameters)
        dataset_params = self._dataset_params(parameters)
        tuning_set, tuning_key = self._dataset(
            lightgbm,
            features=fit_plan.tuning_training_features,
            targets=fit_plan.tuning_training_targets,
            ordered_feature_ids=inputs.ordered_feature_ids,
            dataset_params=dataset_params,
        )
        validation_set, _ = self._dataset(
            lightgbm,
            features=fit_plan.tuning_validation_features,
            targets=fit_plan.tuning_validation_targets,
            ordered_feature_ids=inputs.ordered_feature_ids,
            dataset_params=dataset_params,
            reference_key=tuning_key,
            reference=tuning_set,
        )
        metric, feval = self._evaluation(fit_plan=fit_plan, parameters=parameters)
        tuning_native = {**native, "metric": metric}
        evaluation_history: dict[str, dict[str, list[float]]] = {}
        tuning = lightgbm.train(
            tuning_native,
            tuning_set,
            num_boost_round=maximum_iterations,
            valid_sets=[validation_set],
            feval=feval,
            callbacks=[
                lightgbm.record_evaluation(evaluation_history),
                lightgbm.early_stopping(
                    early_stopping_rounds, first_metric_only=True, verbose=False
                ),
            ],
        )
        best_iteration = int(tuning.best_iteration)
        selection_value = float(tuning.best_score["valid_0"][selection_metric_id])
        if selection_metric_id == "session_rank_ic":
            observed = tuple(evaluation_history["valid_0"][selection_metric_id])
            best_value = max(observed)
            # Frozen rank-policy tie: the first (smaller) iteration within
            # 1e-12 of the observed maximum owns selection.
            best_iteration = next(
                index + 1 for index, value in enumerate(observed) if best_value - value <= 1e-12
            )
            selection_value = float(observed[best_iteration - 1])
        training_set, _ = self._dataset(
            lightgbm,
            features=inputs.features,
            targets=inputs.targets,
            ordered_feature_ids=inputs.ordered_feature_ids,
            dataset_params=dataset_params,
        )
        booster = lightgbm.train(native, training_set, num_boost_round=best_iteration)
        predictions = np.asarray(booster.predict(inputs.features), dtype=np.float64)
        gains = np.asarray(booster.feature_importance(importance_type="gain"), dtype=np.float64)
        model_text = model_trees(str(booster.model_to_string(num_iteration=best_iteration)))
        content = AlphaEstimatorContent.create(
            adapter_id=self.adapter_id,
            content_format_id=CHRONOLOGICAL_LIGHTGBM_CONTENT_FORMAT_ID,
            ordered_feature_ids=inputs.ordered_feature_ids,
            payload={
                "model_text": model_text,
                "best_iteration": best_iteration,
                "feature_gain_hex": tuple(float(value).hex() for value in gains),
            },
        )
        projection = AlphaModelStateProjection.create(
            adapter_id=self.adapter_id,
            model_family_id="lightgbm",
            state_kind="GRADIENT_BOOSTED_TREE",
            state_schema_id="alpha-model.lightgbm-chronological.development-state",
            payload={
                "model_text_hash": canonical_hash(model_text),
                "best_iteration": best_iteration,
                "seed": parameters.seed,
            },
        )
        return AlphaModelFitResult(
            estimator_content=content,
            state_projection=projection,
            training_mse=float(np.mean(np.square(inputs.targets - predictions))),
            iteration_count=best_iteration,
            fit_call_count=2,
            predict_call_count=2,
            selection_diagnostic=AlphaModelSelectionDiagnostic.create(
                recipe_hash=recipe.recipe_hash,
                estimator_content_hash=content.content_hash,
                training_binding_hash=inputs.training_binding_hash,
                metric_id=selection_metric_id,
                direction=("MINIMIZE" if selection_metric_id == "l2" else "MAXIMIZE"),
                value=selection_value,
                selected_iteration=best_iteration,
                fit_plan_hash=fit_plan.fit_plan_hash,
            ),
        )

    def predict(
        self, *, estimator: AlphaEstimatorContent, inputs: BoundAlphaPredictionInput
    ) -> AlphaModelPredictionResult:
        """Reconstruct admitted learned content and predict on its exact Feature axis.

        Args:
            estimator: Qualified immutable learned content for this adapter and format.
            inputs: Finite read-only prediction matrix on the fitted feature axis.

        Returns:
            Finite frozen predictions aligned with the input rows, from one prediction call.

        Raises:
            ValueError: Estimator route, content/feature binding or prediction output is invalid.
        """
        if (
            estimator.adapter_id != self.adapter_id
            or estimator.content_format_id != CHRONOLOGICAL_LIGHTGBM_CONTENT_FORMAT_ID
            or estimator.ordered_feature_ids != inputs.ordered_feature_ids
        ):
            raise ValueError("ALPHA_CHRONOLOGICAL_LIGHTGBM_ESTIMATOR_BINDING_INVALID")
        model_text = str(estimator.payload["model_text"])
        best_iteration = int(estimator.payload["best_iteration"])
        booster = _load_lightgbm().Booster(model_str=model_text)
        predictions = np.asarray(
            booster.predict(inputs.features, num_iteration=best_iteration), dtype=np.float64
        )
        predictions.setflags(write=False)
        return AlphaModelPredictionResult(predictions=predictions)


class RegularizedChronologicalLightGBMAdapter(ChronologicalLightGBMAdapter):
    """Installed successor exposing LightGBM's regularization without duplicating numerics."""

    adapter_id = REGULARIZED_LIGHTGBM_ADAPTER_ID
    recipe_schema_id = REGULARIZED_LIGHTGBM_RECIPE_SCHEMA_ID
    search_domain_schema_id = REGULARIZED_LIGHTGBM_SEARCH_DOMAIN_SCHEMA_ID

    def describe_numerical_binding(self) -> AlphaModelNumericalBinding:
        """Seal registered chronological regularization, sampling and runtime requirements.

        Returns:
            Adapter/content-format handles, implementation owners, deterministic policy and required
            capabilities.
        """
        return AlphaModelNumericalBinding.create(
            adapter_id=self.adapter_id,
            estimator_content_format_id=CHRONOLOGICAL_LIGHTGBM_CONTENT_FORMAT_ID,
            implementation_owners=("lightgbm.Booster", "lightgbm.Dataset", "lightgbm.train"),
            deterministic_policy={
                "objective": "l2",
                "metric": "l2",
                "deterministic": True,
                "force_col_wise": True,
                "sampling": "REGISTERED_FEATURE_AND_BAGGING_PROFILES_SEEDED_PER_PLAN",
                "admitted_seeds": (1729, 2718, 31415),
                "thread_count": 1,
                "regularization": "LEAF_HESSIAN_SCALED_L2_REGISTERED_PROFILES",
                "selection_rule": "ONE_STANDARD_ERROR_THEN_STRONGER_REGULARIZATION",
                "dataset_reuse": "FOLD_LOCAL_IDENTICAL_CONSTRUCTION_ONLY",
            },
            required_runtime_capabilities=(
                "numpy",
                "lightgbm==4.7.0",
                NESTED_FIT_RUNTIME_CAPABILITY,
                SINGLE_THREAD_RUNTIME_CAPABILITY,
            ),
        )

    def validate_recipe(
        self, recipe: AlphaModelRecipeEnvelope
    ) -> RegularizedChronologicalLightGBMParameters:
        """Validate this adapter's route and registered numerical recipe fields.

        Args:
            recipe: Qualified envelope naming this adapter and recipe schema.

        Returns:
            Parsed family-specific numerical parameters.

        Raises:
            ValueError: Recipe route or registered numerical fields are invalid.
        """
        if recipe.adapter_id != self.adapter_id or recipe.recipe_schema_id != self.recipe_schema_id:
            raise ValueError("ALPHA_REGULARIZED_LIGHTGBM_RECIPE_ROUTE_INVALID")
        return RegularizedChronologicalLightGBMParameters(**recipe.parameters)

    def validate_search_domain(self, domain: AlphaModelSearchDomainEnvelope) -> dict[str, object]:
        """Validate the qualified search-domain route and constraints.

        Args:
            domain: Sealed constraints proposed for this adapter.

        Returns:
            Copied constraints equal to the installed domain.

        Raises:
            ValueError: The domain differs from the installed declaration.
        """
        expected = build_regularized_chronological_lightgbm_search_domain()
        if domain != expected:
            raise ValueError("ALPHA_REGULARIZED_LIGHTGBM_SEARCH_DOMAIN_INVALID")
        return dict(domain.constraints)

    def validate_recipe_for_domain(
        self, *, recipe: AlphaModelRecipeEnvelope, domain: AlphaModelSearchDomainEnvelope
    ) -> RegularizedChronologicalLightGBMParameters:
        """Require the numerical recipe to belong to its admitted bounds or profiles.

        Args:
            recipe: Qualified numerical configuration.
            domain: Host-admitted constraints for this adapter.

        Returns:
            Parsed parameters within every admitted bound/profile.

        Raises:
            ValueError: Recipe/domain routes, registered fields, bounds or paired profiles are
                invalid.
        """
        parameters = self.validate_recipe(recipe)
        constraints = self.validate_search_domain(domain)
        scalar_checks = (
            (parameters.seed, "allowed_seeds"),
            (parameters.learning_rate, "allowed_learning_rates"),
            (parameters.min_child_samples, "allowed_min_child_samples"),
            (parameters.min_gain_to_split, "allowed_min_gain_to_split"),
            (parameters.feature_fraction, "allowed_feature_fraction"),
        )
        if any(value not in constraints[key] for value, key in scalar_checks):  # type: ignore[operator]
            raise ValueError("ALPHA_MODEL_RECIPE_OUTSIDE_SEARCH_DOMAIN")
        paired_checks = (
            ((parameters.max_depth, parameters.num_leaves), "allowed_capacity_profiles"),
            ((parameters.lambda_l1, parameters.lambda_l2), "allowed_lambda_profiles"),
            ((parameters.bagging_fraction, parameters.bagging_freq), "allowed_bagging_profiles"),
        )
        for value, key in paired_checks:
            admitted = {tuple(profile) for profile in cast(list[list[float]], constraints[key])}
            if value not in admitted:
                raise ValueError("ALPHA_MODEL_RECIPE_OUTSIDE_SEARCH_DOMAIN")
        return parameters

    @staticmethod
    def _native_params(parameters: ChronologicalLightGBMParameters) -> dict[str, object]:
        if not isinstance(parameters, RegularizedChronologicalLightGBMParameters):
            raise ValueError("ALPHA_REGULARIZED_LIGHTGBM_PARAMETER_INVALID")
        native = ChronologicalLightGBMAdapter._native_params(
            ChronologicalLightGBMParameters(
                seed=parameters.seed,
                num_leaves=parameters.num_leaves,
                learning_rate=parameters.learning_rate,
                max_depth=parameters.max_depth,
                min_child_samples=parameters.min_child_samples,
            )
        )
        native.update(
            {
                "lambda_l1": parameters.lambda_l1,
                "lambda_l2": parameters.lambda_l2,
                "min_gain_to_split": parameters.min_gain_to_split,
                "feature_fraction": parameters.feature_fraction,
                "bagging_fraction": parameters.bagging_fraction,
                "bagging_freq": parameters.bagging_freq,
            }
        )
        return native


__all__ = [
    "CHRONOLOGICAL_LIGHTGBM_ADAPTER_ID",
    "ChronologicalLightGBMAdapter",
    "ChronologicalLightGBMParameters",
    "RegularizedChronologicalLightGBMAdapter",
    "RegularizedChronologicalLightGBMParameters",
    "build_chronological_lightgbm_recipe",
    "build_chronological_lightgbm_search_domain",
    "build_regularized_chronological_lightgbm_recipe",
    "build_regularized_chronological_lightgbm_search_domain",
    "chronological_lightgbm_dataset_cache",
    "load_lightgbm_runtime",
]
