"""Regularized LightGBM successor for role-normalized Dynamic Panel inputs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Literal, Protocol, cast

import numpy as np
import numpy.typing as npt

from alphalattice.capabilities.alpha_modeling.contracts import (
    NESTED_FIT_RUNTIME_CAPABILITY,
    AlphaEstimatorContent,
    AlphaModelFitProtocol,
    AlphaModelFitResult,
    AlphaModelNumericalBinding,
    AlphaModelRecipeEnvelope,
    AlphaModelSearchDomainEnvelope,
    AlphaModelSelectionDiagnostic,
    AlphaModelStateProjection,
    BoundAlphaModelFitInput,
    BoundAlphaTrainingInput,
)
from alphalattice.capabilities.alpha_modeling.runtime.numerical_environment import (
    SINGLE_THREAD_RUNTIME_CAPABILITY,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash

from ..runtime.lightgbm_threads import lightgbm_fit_threads, model_trees, sequential_l2
from .lightgbm_chronological import (
    CHRONOLOGICAL_LIGHTGBM_CONTENT_FORMAT_ID,
    ChronologicalLightGBMAdapter,
    ChronologicalLightGBMParameters,
    load_lightgbm_runtime,
)

type FloatArray = npt.NDArray[np.float64]
type DynamicPanelLightGBMTrainingPolicy = Literal[
    "L2_EARLY_STOPPING",
    "SESSION_RANK_IC_EARLY_STOPPING",
    "FIXED_ITERATION",
]


class _LightGBMDataset(Protocol):
    def get_label(self) -> npt.ArrayLike: ...


DYNAMIC_PANEL_LIGHTGBM_ADAPTER_ID = "dynamic_panel_lightgbm"
DYNAMIC_PANEL_LIGHTGBM_RECIPE_SCHEMA_ID = "alpha-model.dynamic-panel-lightgbm"
DYNAMIC_PANEL_LIGHTGBM_SEARCH_DOMAIN_SCHEMA_ID = "alpha-model.dynamic-panel-lightgbm.search-domain"

DYNAMIC_PANEL_CAPACITY_PROFILES = ((3, 8, 200), (5, 31, 50))
DYNAMIC_PANEL_LEARNING_RATES = (0.03, 0.05)
DYNAMIC_PANEL_SAMPLING_PROFILES = ((1.0, 1.0, 0), (0.7, 0.7, 1))
DYNAMIC_PANEL_REGULARIZATION_PROFILES = (
    (1.0, 1.0, 0.0),
    (10.0, 10.0, 0.001),
    (0.0, 1000.0, 0.0),
    (50.0, 1000.0, 0.01),
    (100.0, 10000.0, 0.05),
)
DYNAMIC_PANEL_LIGHTGBM_SEEDS = (1729, 2718, 31415)
DYNAMIC_PANEL_FIXED_ITERATIONS = (50, 100, 200, 400)
DYNAMIC_PANEL_LIGHTGBM_MAXIMUM_ITERATIONS: Final = 500

# The admissible parameter space, as bounds rather than as an enumeration.
#
# The tuples above enumerate one *search*: the grid `dynamic_panel_lightgbm_parameters`
# walks when a request is asking which hyperparameters win. They were also doing
# duty as the admissible space, which made the two indistinguishable and had a
# concrete cost -- a researcher who wanted to fit one configuration that the grid
# happens not to visit had to edit this file. The sibling regularized-linear
# adapter already separates the two exactly this way (`ridge_alpha_min` /
# `ridge_alpha_max` against a domain, with no enumerated alphas anywhere), so
# this is the convention the domain had already chosen.
DYNAMIC_PANEL_MAX_DEPTH_RANGE: Final = (2, 12)
DYNAMIC_PANEL_NUM_LEAVES_RANGE: Final = (4, 256)
DYNAMIC_PANEL_MIN_CHILD_SAMPLES_RANGE: Final = (5, 2000)
DYNAMIC_PANEL_LEARNING_RATE_RANGE: Final = (0.001, 0.5)
DYNAMIC_PANEL_FRACTION_RANGE: Final = (0.1, 1.0)
DYNAMIC_PANEL_BAGGING_FREQ_RANGE: Final = (0, 50)
DYNAMIC_PANEL_LAMBDA_RANGE: Final = (0.0, 100000.0)
DYNAMIC_PANEL_MIN_GAIN_RANGE: Final = (0.0, 1.0)
DYNAMIC_PANEL_ITERATION_RANGE: Final = (1, DYNAMIC_PANEL_LIGHTGBM_MAXIMUM_ITERATIONS)
DYNAMIC_PANEL_SEED_RANGE: Final = (0, 2**31 - 1)
DYNAMIC_PANEL_LIGHTGBM_PATIENCE: Final = 50
DYNAMIC_PANEL_RANK_IC_METRIC_ID: Final = "session_rank_ic"
DYNAMIC_PANEL_TRAINING_POLICIES: Final[tuple[DynamicPanelLightGBMTrainingPolicy, ...]] = (
    "L2_EARLY_STOPPING",
    "SESSION_RANK_IC_EARLY_STOPPING",
    "FIXED_ITERATION",
)
_DYNAMIC_PANEL_TRAINING_POLICIES: Final[frozenset[str]] = frozenset(DYNAMIC_PANEL_TRAINING_POLICIES)
# The recipe's parameter names in recipe order; ``fixed_iterations`` is the one
# an author may leave out (it is ``None`` for the early-stopping policies).
_DYNAMIC_PANEL_PARAMETER_NAMES: Final = (
    "seed",
    "num_leaves",
    "learning_rate",
    "max_depth",
    "min_child_samples",
    "lambda_l1",
    "lambda_l2",
    "min_gain_to_split",
    "feature_fraction",
    "bagging_fraction",
    "bagging_freq",
    "training_policy",
    "fixed_iterations",
)


def _is_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_real(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


@dataclass(frozen=True, slots=True)
class DynamicPanelLightGBMParameters(ChronologicalLightGBMParameters):
    """Bounded tree, sampling and regularization parameters with a training policy.

    The inherited tree parameters and these sampling/penalty fields describe one
    admitted point. Fixed iteration training declares its iteration count;
    early stopping policies retain their installed tuning/refit protocols.
    """

    lambda_l1: float
    lambda_l2: float
    min_gain_to_split: float
    feature_fraction: float
    bagging_fraction: float
    bagging_freq: int
    training_policy: DynamicPanelLightGBMTrainingPolicy
    fixed_iterations: int | None = None

    def __post_init__(self) -> None:
        """Admit any configuration inside the declared bounds.

        The base class still enforces its own narrower shape for the recipes it
        owns; what changes here is that a Dynamic Panel configuration is checked
        against bounds rather than against membership of the search grid, so
        stating a configuration the grid does not visit is a configuration
        change and not a source change.
        """
        bounded = (
            (self.seed, DYNAMIC_PANEL_SEED_RANGE),
            (self.max_depth, DYNAMIC_PANEL_MAX_DEPTH_RANGE),
            (self.num_leaves, DYNAMIC_PANEL_NUM_LEAVES_RANGE),
            (self.min_child_samples, DYNAMIC_PANEL_MIN_CHILD_SAMPLES_RANGE),
            (self.learning_rate, DYNAMIC_PANEL_LEARNING_RATE_RANGE),
            (self.feature_fraction, DYNAMIC_PANEL_FRACTION_RANGE),
            (self.bagging_fraction, DYNAMIC_PANEL_FRACTION_RANGE),
            (self.bagging_freq, DYNAMIC_PANEL_BAGGING_FREQ_RANGE),
            (self.lambda_l1, DYNAMIC_PANEL_LAMBDA_RANGE),
            (self.lambda_l2, DYNAMIC_PANEL_LAMBDA_RANGE),
            (self.min_gain_to_split, DYNAMIC_PANEL_MIN_GAIN_RANGE),
        )
        if (
            any(not lower <= value <= upper for value, (lower, upper) in bounded)
            or self.num_leaves > 2**self.max_depth
            or (
                self.training_policy == "FIXED_ITERATION"
                and (
                    self.fixed_iterations is None
                    or not DYNAMIC_PANEL_ITERATION_RANGE[0]
                    <= self.fixed_iterations
                    <= DYNAMIC_PANEL_ITERATION_RANGE[1]
                )
            )
            or (self.training_policy != "FIXED_ITERATION" and self.fixed_iterations is not None)
        ):
            raise ValueError("ALPHA_DYNAMIC_PANEL_LIGHTGBM_PARAMETER_INVALID")


def build_dynamic_panel_lightgbm_recipe(
    parameters: DynamicPanelLightGBMParameters,
) -> AlphaModelRecipeEnvelope:
    """Seal an admitted parameter point in the adapter's recipe envelope.

    Args:
        parameters: Validated dynamic-panel LightGBM parameter point.

    Returns:
        Recipe envelope containing the exact adapter/schema route and parameters.
    """
    return AlphaModelRecipeEnvelope.create(
        adapter_id=DYNAMIC_PANEL_LIGHTGBM_ADAPTER_ID,
        recipe_schema_id=DYNAMIC_PANEL_LIGHTGBM_RECIPE_SCHEMA_ID,
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
            "training_policy": parameters.training_policy,
            "fixed_iterations": parameters.fixed_iterations,
        },
    )


def build_dynamic_panel_lightgbm_search_domain() -> AlphaModelSearchDomainEnvelope:
    """The admissible configuration space, as bounds.

    It states what a Dynamic Panel LightGBM configuration may be, not how many of
    them some particular request intends to fit. The predecessor also carried
    `search_configuration_count`, `seeds_per_configuration`, `training_policy_count`
    and `trial_count_per_view`, which described one enumeration rather than the
    space -- so a run that stated a single configuration was sealed against a
    domain claiming seven hundred and twenty trials and a selection rule, and the
    sealed program described a search the run never performed. How many trials a
    request actually fits is reported by preflight and by the program, which
    measure it.
    """
    return AlphaModelSearchDomainEnvelope.create(
        adapter_id=DYNAMIC_PANEL_LIGHTGBM_ADAPTER_ID,
        recipe_schema_id=DYNAMIC_PANEL_LIGHTGBM_RECIPE_SCHEMA_ID,
        search_domain_schema_id=DYNAMIC_PANEL_LIGHTGBM_SEARCH_DOMAIN_SCHEMA_ID,
        constraints={
            "seed_min": DYNAMIC_PANEL_SEED_RANGE[0],
            "seed_max": DYNAMIC_PANEL_SEED_RANGE[1],
            "max_depth_min": DYNAMIC_PANEL_MAX_DEPTH_RANGE[0],
            "max_depth_max": DYNAMIC_PANEL_MAX_DEPTH_RANGE[1],
            "num_leaves_min": DYNAMIC_PANEL_NUM_LEAVES_RANGE[0],
            "num_leaves_max": DYNAMIC_PANEL_NUM_LEAVES_RANGE[1],
            "min_child_samples_min": DYNAMIC_PANEL_MIN_CHILD_SAMPLES_RANGE[0],
            "min_child_samples_max": DYNAMIC_PANEL_MIN_CHILD_SAMPLES_RANGE[1],
            "learning_rate_min": DYNAMIC_PANEL_LEARNING_RATE_RANGE[0],
            "learning_rate_max": DYNAMIC_PANEL_LEARNING_RATE_RANGE[1],
            "fraction_min": DYNAMIC_PANEL_FRACTION_RANGE[0],
            "fraction_max": DYNAMIC_PANEL_FRACTION_RANGE[1],
            "bagging_freq_min": DYNAMIC_PANEL_BAGGING_FREQ_RANGE[0],
            "bagging_freq_max": DYNAMIC_PANEL_BAGGING_FREQ_RANGE[1],
            "lambda_min": DYNAMIC_PANEL_LAMBDA_RANGE[0],
            "lambda_max": DYNAMIC_PANEL_LAMBDA_RANGE[1],
            "min_gain_to_split_min": DYNAMIC_PANEL_MIN_GAIN_RANGE[0],
            "min_gain_to_split_max": DYNAMIC_PANEL_MIN_GAIN_RANGE[1],
            "fixed_iterations_min": DYNAMIC_PANEL_ITERATION_RANGE[0],
            "fixed_iterations_max": DYNAMIC_PANEL_ITERATION_RANGE[1],
            "allowed_training_policies": list(DYNAMIC_PANEL_TRAINING_POLICIES),
            "maximum_iterations": DYNAMIC_PANEL_LIGHTGBM_MAXIMUM_ITERATIONS,
            "early_stopping_rounds": DYNAMIC_PANEL_LIGHTGBM_PATIENCE,
            "selection_metric_ids": ["l2", DYNAMIC_PANEL_RANK_IC_METRIC_ID, "fixed_iteration"],
            "configured_thread_count": 1,
            "selection_rule": "MATURE_INNER_SESSION_RANK_IC_PAIRED_NO_COST_OR_TURNOVER",
        },
    )


def dynamic_panel_lightgbm_parameters() -> tuple[DynamicPanelLightGBMParameters, ...]:
    """Complete installed policy/regularization surface for a selected view."""
    return tuple(
        DynamicPanelLightGBMParameters(
            seed=seed,
            max_depth=max_depth,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            learning_rate=learning_rate,
            feature_fraction=feature_fraction,
            bagging_fraction=bagging_fraction,
            bagging_freq=bagging_freq,
            lambda_l1=lambda_l1,
            lambda_l2=lambda_l2,
            min_gain_to_split=min_gain_to_split,
            training_policy=cast(DynamicPanelLightGBMTrainingPolicy, training_policy),
            fixed_iterations=(fixed_iterations if training_policy == "FIXED_ITERATION" else None),
        )
        for max_depth, num_leaves, min_child_samples in (DYNAMIC_PANEL_CAPACITY_PROFILES)
        for learning_rate in DYNAMIC_PANEL_LEARNING_RATES
        for feature_fraction, bagging_fraction, bagging_freq in (DYNAMIC_PANEL_SAMPLING_PROFILES)
        for lambda_l1, lambda_l2, min_gain_to_split in (DYNAMIC_PANEL_REGULARIZATION_PROFILES)
        for seed in DYNAMIC_PANEL_LIGHTGBM_SEEDS
        for training_policy, fixed_iterations in (
            ("L2_EARLY_STOPPING", None),
            ("SESSION_RANK_IC_EARLY_STOPPING", None),
            *(("FIXED_ITERATION", value) for value in DYNAMIC_PANEL_FIXED_ITERATIONS),
        )
    )


def _average_ranks(values: FloatArray) -> FloatArray:
    """Stable average ranks with deterministic tie handling."""

    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
    stops = np.r_[starts[1:], values.size]
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.repeat((starts + stops - 1) / 2.0, stops - starts)
    ranks.setflags(write=False)
    return ranks


def session_grouped_rank_ic(
    predictions: FloatArray,
    targets: FloatArray,
    session_codes: npt.NDArray[np.int64],
    *,
    minimum_paired_rows: int = 20,
) -> tuple[float, int, int]:
    """Equal-session-weighted Spearman with explicit exclusion evidence.

    Inputs have already crossed the common-axis finite firewall; seeing a
    nonfinite value here is therefore an authority failure, never permission to
    drop a row inside the metric.
    """
    scores = np.asarray(predictions, dtype=np.float64)
    lane = np.asarray(targets, dtype=np.float64)
    codes = np.asarray(session_codes, dtype=np.int64)
    if (
        scores.ndim != 1
        or lane.shape != scores.shape
        or codes.shape != scores.shape
        or not np.isfinite(scores).all()
        or not np.isfinite(lane).all()
    ):
        raise ValueError("ALPHA_SESSION_RANK_IC_FINITE_AXIS_INVALID")
    values: list[float] = []
    insufficient = 0
    constant = 0
    order = np.argsort(codes, kind="stable")
    ordered_codes = codes[order]
    starts = np.r_[0, np.flatnonzero(ordered_codes[1:] != ordered_codes[:-1]) + 1]
    stops = np.r_[starts[1:], ordered_codes.size]
    for start, stop in zip(starts, stops, strict=True):
        positions = order[int(start) : int(stop)]
        left = scores[positions]
        right = lane[positions]
        if left.size < minimum_paired_rows:
            insufficient += 1
            continue
        if np.unique(left).size < 2 or np.unique(right).size < 2:
            constant += 1
            continue
        left_rank = _average_ranks(left)
        right_rank = _average_ranks(right)
        left_centered = left_rank - float(np.mean(left_rank))
        right_centered = right_rank - float(np.mean(right_rank))
        denominator = float(
            np.sqrt(np.dot(left_centered, left_centered) * np.dot(right_centered, right_centered))
        )
        if denominator <= 0.0 or not np.isfinite(denominator):
            constant += 1
            continue
        values.append(float(np.dot(left_centered, right_centered) / denominator))
    return (float(np.mean(values)) if values else 0.0, insufficient, constant)


class DynamicPanelLightGBMAdapter(ChronologicalLightGBMAdapter):
    """Installed successor; inherited numerics preserve Dataset reuse."""

    adapter_id = DYNAMIC_PANEL_LIGHTGBM_ADAPTER_ID
    recipe_schema_id = DYNAMIC_PANEL_LIGHTGBM_RECIPE_SCHEMA_ID
    search_domain_schema_id = DYNAMIC_PANEL_LIGHTGBM_SEARCH_DOMAIN_SCHEMA_ID

    def _fit_policy(self, parameters: ChronologicalLightGBMParameters) -> tuple[str, int, int]:
        if not isinstance(parameters, DynamicPanelLightGBMParameters):
            raise ValueError("ALPHA_DYNAMIC_PANEL_LIGHTGBM_PARAMETER_INVALID")
        if parameters.training_policy == "L2_EARLY_STOPPING":
            return (
                "l2",
                DYNAMIC_PANEL_LIGHTGBM_MAXIMUM_ITERATIONS,
                DYNAMIC_PANEL_LIGHTGBM_PATIENCE,
            )
        if parameters.training_policy == "SESSION_RANK_IC_EARLY_STOPPING":
            return (
                DYNAMIC_PANEL_RANK_IC_METRIC_ID,
                DYNAMIC_PANEL_LIGHTGBM_MAXIMUM_ITERATIONS,
                DYNAMIC_PANEL_LIGHTGBM_PATIENCE,
            )
        # The fixed path never asks the parent for a nested policy.
        raise ValueError("ALPHA_DYNAMIC_PANEL_LIGHTGBM_FIXED_POLICY_NOT_NESTED")

    def _evaluation(
        self,
        *,
        fit_plan: BoundAlphaModelFitInput,
        parameters: ChronologicalLightGBMParameters,
    ) -> tuple[object, Callable[..., object] | None]:
        if not isinstance(parameters, DynamicPanelLightGBMParameters):
            raise ValueError("ALPHA_DYNAMIC_PANEL_LIGHTGBM_PARAMETER_INVALID")
        if parameters.training_policy == "L2_EARLY_STOPPING":
            return "None", sequential_l2
        codes = fit_plan.tuning_validation_session_codes
        if codes is None:
            raise ValueError("ALPHA_DYNAMIC_PANEL_LIGHTGBM_SESSION_CODES_REQUIRED")

        def evaluate(predictions: FloatArray, dataset: object) -> tuple[str, float, bool]:
            targets: FloatArray = np.asarray(
                cast(_LightGBMDataset, dataset).get_label(),
                dtype=np.float64,
            )
            value, _insufficient, _constant = session_grouped_rank_ic(
                np.asarray(predictions, dtype=np.float64), targets, codes
            )
            return DYNAMIC_PANEL_RANK_IC_METRIC_ID, value, True

        return "None", evaluate

    def fit(
        self,
        *,
        recipe: AlphaModelRecipeEnvelope,
        inputs: BoundAlphaTrainingInput,
        fit_plan: BoundAlphaModelFitInput | None = None,
    ) -> AlphaModelFitResult:
        """Fit the declared policy using the bound training inputs and fit plan.

        Early stopping delegates to the chronological owner's tuning/refit path.
        Fixed iteration training reads the training arrays and the declared iteration
        count, trains once, and seals estimator content, state and diagnostics.

        Args:
            recipe: Exact recipe admitted by this adapter.
            inputs: Bound training arrays with their ordered Feature axis.
            fit_plan: Plan declaring the protocol, parent binding and Feature axis.

        Returns:
            Sealed estimator content, state, training error and selection diagnostics.

        Raises:
            ValueError: The recipe or fixed iteration plan violates the admitted
                route, protocol, training binding, Feature axis or iteration bounds.
        """
        parameters = self.validate_recipe(recipe)
        if parameters.training_policy != "FIXED_ITERATION":
            return super().fit(recipe=recipe, inputs=inputs, fit_plan=fit_plan)
        if (
            fit_plan is None
            or fit_plan.protocol_id not in self.fit_protocols(recipe)
            or fit_plan.parent_training_binding_hash != inputs.training_binding_hash
            or fit_plan.ordered_feature_ids != inputs.ordered_feature_ids
            or parameters.fixed_iterations is None
            or not DYNAMIC_PANEL_ITERATION_RANGE[0]
            <= parameters.fixed_iterations
            <= DYNAMIC_PANEL_ITERATION_RANGE[1]
        ):
            raise ValueError("ALPHA_DYNAMIC_PANEL_LIGHTGBM_FIXED_FIT_PLAN_INVALID")
        iterations = int(parameters.fixed_iterations)
        lightgbm = load_lightgbm_runtime()
        native = self._native_params(parameters)
        dataset, _ = self._dataset(
            lightgbm,
            features=inputs.features,
            targets=inputs.targets,
            ordered_feature_ids=inputs.ordered_feature_ids,
            dataset_params=self._dataset_params(parameters),
        )
        booster = lightgbm.train(native, dataset, num_boost_round=iterations)
        # The training error only a caller that reads one asks for (PERF-1): a lifecycle
        # child's evidence stores none, so its fit takes no prediction over the panel.
        predictions = (
            None
            if fit_plan.training_error is None
            else np.asarray(
                booster.predict(inputs.features, num_threads=lightgbm_fit_threads()),
                dtype=np.float64,
            )
        )
        gains = np.asarray(booster.feature_importance(importance_type="gain"), dtype=np.float64)
        model_text = model_trees(str(booster.model_to_string(num_iteration=iterations)))
        content = AlphaEstimatorContent.create(
            adapter_id=self.adapter_id,
            content_format_id=CHRONOLOGICAL_LIGHTGBM_CONTENT_FORMAT_ID,
            ordered_feature_ids=inputs.ordered_feature_ids,
            payload={
                "model_text": model_text,
                "best_iteration": iterations,
                "feature_gain_hex": tuple(float(value).hex() for value in gains),
            },
        )
        projection = AlphaModelStateProjection.create(
            adapter_id=self.adapter_id,
            model_family_id="lightgbm",
            state_kind="GRADIENT_BOOSTED_TREE",
            state_schema_id="alpha-model.lightgbm-dynamic-panel.development-state",
            payload={
                "model_text_hash": canonical_hash(model_text),
                "best_iteration": iterations,
                "seed": parameters.seed,
                "training_policy": parameters.training_policy,
            },
        )
        return AlphaModelFitResult(
            estimator_content=content,
            state_projection=projection,
            training_mse=None
            if predictions is None
            else float(np.mean(np.square(inputs.targets - predictions))),
            iteration_count=iterations,
            fit_call_count=1,
            predict_call_count=0 if predictions is None else 1,
            selection_diagnostic=AlphaModelSelectionDiagnostic.create(
                recipe_hash=recipe.recipe_hash,
                estimator_content_hash=content.content_hash,
                training_binding_hash=inputs.training_binding_hash,
                metric_id="fixed_iteration",
                direction="MINIMIZE",
                value=float(iterations),
                selected_iteration=iterations,
                fit_plan_hash=fit_plan.fit_plan_hash,
            ),
        )

    def describe_numerical_binding(self) -> AlphaModelNumericalBinding:
        """Describe the numerical owners, deterministic policy and runtime capabilities.

        Returns:
            Binding declaration for the installed LightGBM implementations and policies.
        """
        return AlphaModelNumericalBinding.create(
            adapter_id=self.adapter_id,
            estimator_content_format_id=CHRONOLOGICAL_LIGHTGBM_CONTENT_FORMAT_ID,
            implementation_owners=(
                "lightgbm.Booster",
                "lightgbm.Dataset",
                "lightgbm.train",
            ),
            deterministic_policy={
                "objective": "l2",
                "metric": "INSTALLED_L2_OR_SESSION_RANK_IC_OR_FIXED_ITERATION",
                "maximum_iterations": DYNAMIC_PANEL_LIGHTGBM_MAXIMUM_ITERATIONS,
                "early_stopping_patience": DYNAMIC_PANEL_LIGHTGBM_PATIENCE,
                "fixed_iterations": DYNAMIC_PANEL_ITERATION_RANGE,
                "deterministic": True,
                "force_col_wise": True,
                "sampling": "REGISTERED_SEEDED_FEATURE_AND_ROW_PROFILES",
                "admitted_seeds": DYNAMIC_PANEL_LIGHTGBM_SEEDS,
                "thread_count": 1,
                "regularization": ("REGISTERED_LIGHT_AND_HEAVY_L2_NONZERO_L1_PROFILES"),
                "adapter_preprocessing": "NONE",
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
        """A fixed-iteration recipe reads no tuning partition; the others need one.

        The direct plan is what a fixed-iteration fit states. The nested shape
        stays accepted for that policy because two owners bind the nested shape -- the
        product's model renewal and the Panel methodology, whose sealed fit-plan
        identities carry it -- and the fixed path reads none of its partition.
        """
        if self.validate_recipe(recipe).training_policy == "FIXED_ITERATION":
            return ("DIRECT_FIT", "NESTED_EARLY_STOPPING_REFIT")
        return ("NESTED_EARLY_STOPPING_REFIT",)

    def validate_recipe(self, recipe: AlphaModelRecipeEnvelope) -> DynamicPanelLightGBMParameters:
        """Validate a recipe's exact route, parameter names, kinds and bounded values.

        Args:
            recipe: Authored envelope to admit for this adapter.

        Returns:
            Validated dynamic-panel parameter point.

        Raises:
            ValueError: The route, fields, value kinds or parameter bounds are invalid.
        """
        if recipe.adapter_id != self.adapter_id or recipe.recipe_schema_id != self.recipe_schema_id:
            raise ValueError("ALPHA_DYNAMIC_PANEL_LIGHTGBM_RECIPE_ROUTE_INVALID")
        # An authored parameter set reaches here as it was written. A field the
        # schema has no name for, a missing one, or a value of another kind (a
        # quoted number, a boolean where a count belongs, a list or mapping
        # where the policy name belongs) is a recipe outside the schema,
        # refused as one rather than as a construction or hashing error: the
        # kind is checked before the value is looked up anywhere.
        values = dict(recipe.parameters)
        policy = values.get("training_policy")
        counts = ("seed", "num_leaves", "max_depth", "min_child_samples", "bagging_freq")
        reals = (
            "learning_rate",
            "lambda_l1",
            "lambda_l2",
            "min_gain_to_split",
            "feature_fraction",
            "bagging_fraction",
        )
        if (
            set(values) - set(_DYNAMIC_PANEL_PARAMETER_NAMES)
            or any(name not in values for name in _DYNAMIC_PANEL_PARAMETER_NAMES[:-1])
            or any(not _is_count(values[name]) for name in counts)
            or any(not _is_real(values[name]) for name in reals)
            or not isinstance(policy, str)
            or policy not in _DYNAMIC_PANEL_TRAINING_POLICIES
            or not (values.get("fixed_iterations") is None or _is_count(values["fixed_iterations"]))
        ):
            raise ValueError("ALPHA_DYNAMIC_PANEL_LIGHTGBM_RECIPE_PARAMETERS_INVALID")
        return DynamicPanelLightGBMParameters(**values)

    def validate_search_domain(self, domain: AlphaModelSearchDomainEnvelope) -> dict[str, object]:
        """Require the exact installed search-domain declaration.

        Args:
            domain: Domain envelope to compare with the installed owner.

        Returns:
            Admitted constraints copied from that domain.

        Raises:
            ValueError: The domain differs from the installed declaration.
        """
        if domain != build_dynamic_panel_lightgbm_search_domain():
            raise ValueError("ALPHA_DYNAMIC_PANEL_LIGHTGBM_DOMAIN_INVALID")
        return dict(domain.constraints)

    def validate_recipe_for_domain(
        self,
        *,
        recipe: AlphaModelRecipeEnvelope,
        domain: AlphaModelSearchDomainEnvelope,
    ) -> DynamicPanelLightGBMParameters:
        """Check the recipe against every admitted domain bound and training policy.

        Args:
            recipe: Authored model recipe to validate.
            domain: Exact installed search-domain declaration.

        Returns:
            Validated parameters contained by the supplied admitted domain.

        Raises:
            ValueError: The recipe or domain is invalid, or a parameter or policy
                lies outside the admitted domain.
        """
        parameters = self.validate_recipe(recipe)
        constraints = self.validate_search_domain(domain)

        def bounded(value: float, prefix: str) -> bool:
            lower = float(cast(int | float, constraints[f"{prefix}_min"]))
            upper = float(cast(int | float, constraints[f"{prefix}_max"]))
            return lower <= value <= upper

        checks = (
            (float(parameters.seed), "seed"),
            (float(parameters.max_depth), "max_depth"),
            (float(parameters.num_leaves), "num_leaves"),
            (float(parameters.min_child_samples), "min_child_samples"),
            (parameters.learning_rate, "learning_rate"),
            (parameters.feature_fraction, "fraction"),
            (parameters.bagging_fraction, "fraction"),
            (float(parameters.bagging_freq), "bagging_freq"),
            (parameters.lambda_l1, "lambda"),
            (parameters.lambda_l2, "lambda"),
            (parameters.min_gain_to_split, "min_gain_to_split"),
        )
        policies = cast(list[object], constraints["allowed_training_policies"])
        if (
            any(not bounded(value, prefix) for value, prefix in checks)
            or parameters.training_policy not in policies
            or (
                parameters.fixed_iterations is not None
                and not bounded(float(parameters.fixed_iterations), "fixed_iterations")
            )
        ):
            raise ValueError("ALPHA_MODEL_RECIPE_OUTSIDE_SEARCH_DOMAIN")
        return parameters

    @staticmethod
    def _native_params(
        parameters: ChronologicalLightGBMParameters,
    ) -> dict[str, object]:
        if not isinstance(parameters, DynamicPanelLightGBMParameters):
            raise ValueError("ALPHA_DYNAMIC_PANEL_LIGHTGBM_PARAMETER_INVALID")
        # This successor owns its broader min-child domain.  The inherited
        # native mapping reads the shared fields directly; rebuilding the old
        # parameter contract here would incorrectly re-admit against its
        # narrower {20, 50} domain before this installed successor can run.
        base = ChronologicalLightGBMAdapter._native_params(parameters)
        base.update(
            {
                "lambda_l1": parameters.lambda_l1,
                "lambda_l2": parameters.lambda_l2,
                "min_gain_to_split": parameters.min_gain_to_split,
                "feature_fraction": parameters.feature_fraction,
                "bagging_fraction": parameters.bagging_fraction,
                "bagging_freq": parameters.bagging_freq,
            }
        )
        return base


__all__ = [
    "DYNAMIC_PANEL_CAPACITY_PROFILES",
    "DYNAMIC_PANEL_LEARNING_RATES",
    "DYNAMIC_PANEL_LIGHTGBM_ADAPTER_ID",
    "DYNAMIC_PANEL_LIGHTGBM_SEEDS",
    "DYNAMIC_PANEL_REGULARIZATION_PROFILES",
    "DYNAMIC_PANEL_SAMPLING_PROFILES",
    "DynamicPanelLightGBMAdapter",
    "DynamicPanelLightGBMParameters",
    "build_dynamic_panel_lightgbm_recipe",
    "build_dynamic_panel_lightgbm_search_domain",
    "dynamic_panel_lightgbm_parameters",
]
