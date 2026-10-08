"""Requirement tests for the Alpha model seam and training-input firewall."""

from __future__ import annotations

import json
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from alphalattice.capabilities.alpha_modeling.adapters.lightgbm_dynamic_panel import (
    DynamicPanelLightGBMAdapter,
)
from alphalattice.capabilities.alpha_modeling.adapters.regularized_linear import (
    RegularizedLinearAdapter,
    RegularizedLinearParameters,
    build_regularized_linear_recipe,
    fit_regularized_linear,
)
from alphalattice.capabilities.alpha_modeling.catalog import (
    AlphaModelCatalog,
    build_installed_alpha_model_catalog,
)
from alphalattice.capabilities.alpha_modeling.contracts import (
    NESTED_FIT_RUNTIME_CAPABILITY,
    AlphaEstimatorContent,
    AlphaFitProvenanceReceipt,
    AlphaModelFitProtocol,
    AlphaModelFitResult,
    AlphaModelFitSidecar,
    AlphaModelNumericalBinding,
    AlphaModelPredictionResult,
    AlphaModelRecipeEnvelope,
    AlphaModelSearchDomainEnvelope,
    AlphaModelStateProjection,
    BoundAlphaPredictionInput,
    BoundAlphaTrainingInput,
)
from alphalattice.foundation.research_foundation.contracts import (
    ResearchDeskExecutionOutcomeRef,
    ResearchFoundationBinding,
)
from alphalattice.investment.alpha_research.candidates.contracts import (
    AlphaExperimentBatch,
    AlphaResearchProgram,
    seal_contract,
)
from alphalattice.investment.alpha_research.experiments.development_artifacts import (
    AlphaDevelopmentArtifactReadbackError,
)
from alphalattice.investment.alpha_research.experiments.development_contracts import (
    AlphaDevelopmentEstimatorState,
    AlphaEstimatorState,
    AlphaSessionScoreStatistics,
    seal_current_contract,
)
from alphalattice.investment.alpha_research.experiments.execution import execute_alpha_model_batch
from alphalattice.investment.alpha_research.experiments.fit_plan import (
    AlphaFitPlanAuthorityError,
    AlphaModelFitPlan,
    build_alpha_model_fit_plan,
)
from alphalattice.investment.alpha_research.experiments.mandate import (
    AlphaModelRecipeProposal,
    AlphaResearchModelMandate,
    build_current_alpha_research_model_mandate,
)
from alphalattice.investment.alpha_research.inputs.folds import (
    AlphaCurrentRefitArrays,
    AlphaFoldArrayPlan,
    AlphaFoldArrays,
)
from alphalattice.investment.alpha_research.inputs.training import (
    AlphaTrainingInputAuthorityError,
    AlphaTrainingInputBinding,
    bind_alpha_model_inputs,
    build_alpha_training_input_binding,
)
from alphalattice.investment.alpha_research.publication.artifacts import AlphaCurrentArtifactStore
from alphalattice.investment.alpha_research.scores.refit import (
    fit_and_assess_current_model,
    fit_and_assess_current_regularized_linear,
)
from alphalattice.kernel.shared_kernel.domain.enums import DataValidityClass
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.validation.enums import RebalanceFrequency
from tests.alpha_research.fixtures import (
    FIXTURE_FACTOR_IDS,
    synthetic_prepared_arrays,
    synthetic_request,
)


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value)
    result.setflags(write=False)
    return result


def _foundation() -> ResearchFoundationBinding:
    execution = ResearchDeskExecutionOutcomeRef.model_construct(
        research_cadence=RebalanceFrequency.DAILY,
        snapshot_hash="d" * 64,
        schedule_hash="e" * 64,
        development_content_hash="f" * 64,
        sealed_holdout_content_hash="0" * 64,
        marker_hash="1" * 64,
        market_as_of="2026-07-31",
        data_validity_class=DataValidityClass.CURRENT_UNIVERSE_RESEARCH_ONLY,
    )
    return ResearchFoundationBinding.model_construct(
        research_cadence=RebalanceFrequency.DAILY,
        feature_panel_snapshot_hash="2" * 64,
        logical_panel_hash="a" * 64,
        logical_semantic_index_hash="c" * 64,
        factor_training_outcome_snapshot_hash="3" * 64,
        factor_screening_result_hash="4" * 64,
        factor_candidate_slate_hash="5" * 64,
        research_desk_factor_input_hash="b" * 64,
        execution_outcome=execution,
        ordered_factor_ids=FIXTURE_FACTOR_IDS,
        downstream_factor_research_forbidden=True,
        secondary_feature_preprocessing_forbidden=True,
        foundation_hash="9" * 64,
    )


def _bound_fold(fold: AlphaFoldArrays, foundation: ResearchFoundationBinding) -> AlphaFoldArrays:
    training_row_sessions = tuple(
        session
        for session in fold.training_sessions
        for _listing in sorted(set(fold.training_listing_ids))
    )
    training_feature_hashes = tuple(
        canonical_hash(("feature", session, listing))
        for session, listing in zip(training_row_sessions, fold.training_listing_ids, strict=True)
    )
    training_outcome_hashes = tuple(
        canonical_hash(("outcome", session, listing))
        for session, listing in zip(training_row_sessions, fold.training_listing_ids, strict=True)
    )
    training_economic = _readonly(fold.training_targets.copy())
    validation_economic = _readonly(fold.validation_targets.copy())
    binding = build_alpha_training_input_binding(
        scope="DEVELOPMENT_FOLD",
        foundation_hash=foundation.foundation_hash,
        feature_panel_snapshot_hash=foundation.feature_panel_snapshot_hash,
        causal_outcome_snapshot_hash=foundation.execution_outcome.snapshot_hash,
        target_policy=None,
        ordered_feature_ids=fold.ordered_factor_ids,
        feature_context_hash=None,
        training_row_sessions=training_row_sessions,
        training_row_listing_ids=fold.training_listing_ids,
        prediction_row_sessions=fold.validation_row_sessions,
        prediction_row_listing_ids=fold.validation_listing_ids,
        training_features=fold.training_features,
        training_targets=fold.training_targets,
        training_mask=fold.training_model_mask,
        prediction_features=fold.validation_features,
        prediction_targets=fold.validation_targets,
        prediction_mask=fold.validation_feature_complete,
        training_feature_row_hashes=training_feature_hashes,
        training_outcome_row_hashes=training_outcome_hashes,
        prediction_feature_row_hashes=fold.validation_feature_row_hashes,
        prediction_outcome_row_hashes=fold.validation_outcome_row_hashes,
        training_economic_returns=training_economic,
        prediction_economic_returns=validation_economic,
        training_cutoff=fold.training_sessions[-1],
        outcome_maturity_session=fold.training_sessions[-1],
        prediction_anchor=fold.validation_sessions[0],
        fold_commitment_hash=fold.commitment.commitment_hash,
    )
    return replace(
        fold,
        training_economic_returns=training_economic,
        validation_economic_returns=validation_economic,
        training_row_sessions=training_row_sessions,
        training_feature_row_hashes=training_feature_hashes,
        training_outcome_row_hashes=training_outcome_hashes,
        training_input_binding=binding,
    )


class _ArrayWorkspace:
    def __init__(self, folds: tuple[AlphaFoldArrays, ...]) -> None:
        self.folds = folds

    @contextmanager
    def fold_lease(self, fold_index: int):
        yield self.folds[fold_index]

    def load_fold(self, fold_index: int) -> AlphaFoldArrays:
        return self.folds[fold_index]

    def prepare_current_refit(self):
        raise AssertionError("current refit is outside this fixture")


def _program(
    foundation: ResearchFoundationBinding,
    plan: AlphaFoldArrayPlan,
    model_mandate: AlphaResearchModelMandate | None = None,
) -> AlphaResearchProgram:
    admitted_mandate = model_mandate or build_current_alpha_research_model_mandate()
    return seal_contract(
        AlphaResearchProgram,
        {
            "foundation_hash": foundation.foundation_hash,
            "logical_panel_hash": foundation.logical_panel_hash,
            "logical_semantic_index_hash": foundation.logical_semantic_index_hash,
            "causal_outcome_snapshot_hash": foundation.execution_outcome.snapshot_hash,
            "pm_plan_hash": "5" * 64,
            "ordered_listing_ids_hash": canonical_hash(plan.ordered_listing_ids),
            "ordered_factor_ids_hash": canonical_hash(plan.foundation.ordered_factor_ids),
            "split_policy_hash": "8" * 64,
            "metric_policy_hash": "9" * 64,
            "package_identity_hash": "0" * 64,
            "stability_policy_hash": "a" * 64,
            "goal_criteria_hash": "b" * 64,
            "user_authorization_hash": "c" * 64,
            "research_goal_hash": "d" * 64,
            "model_mandate_hash": admitted_mandate.mandate_hash,
            "model_catalog_hash": admitted_mandate.catalog_binding.catalog_hash,
        },
        "program_hash",
    )


def _installed_catalog_counting(adapter: RegularizedLinearAdapter) -> AlphaModelCatalog:
    """The installed inventory with the linear adapter replaced by a counting one.

    The current mandate binds the whole installed catalog, so a test that wants
    to count linear fits under it has to present the same inventory.
    """

    return AlphaModelCatalog((adapter, DynamicPanelLightGBMAdapter()))


def _batch(program: AlphaResearchProgram) -> AlphaExperimentBatch:
    catalog = build_installed_alpha_model_catalog()
    mandate = build_current_alpha_research_model_mandate(catalog=catalog)
    recipes = tuple(
        mandate.admit_proposal(
            proposal=AlphaModelRecipeProposal(
                capability_handle="capability-1",
                parameters={"family": "ridge", "alpha": float(value)},
            ),
            catalog=catalog,
            admitted_target_lanes=None,
        )
        for value in range(1, 7)
    )
    return seal_contract(
        AlphaExperimentBatch,
        {
            "program_hash": program.program_hash,
            "batch_index": 1,
            "specs": recipes,
            "predecessor_batch_hash": None,
        },
        "batch_hash",
    )


def _plan_and_workspace() -> tuple[AlphaFoldArrayPlan, _ArrayWorkspace]:
    request, _inventory = synthetic_request()
    prepared = synthetic_prepared_arrays()
    foundation = _foundation()
    plan = AlphaFoldArrayPlan(
        foundation=foundation,
        ordered_listing_ids=request.ordered_listing_ids,
        feature_reader=object(),  # type: ignore[arg-type]
        outcome_reader=object(),  # type: ignore[arg-type]
        feature_panel_manifest_ref="playpen://fixture/panel",
        causal_outcome_manifest_ref="playpen://fixture/outcome",
        split_plan=prepared.split_plan,
    )
    folds = tuple(_bound_fold(value, foundation) for value in prepared.folds)
    return plan, _ArrayWorkspace(folds)


@pytest.mark.parametrize(
    "parameters",
    (
        RegularizedLinearParameters(family="ridge", alpha=1.0),
        RegularizedLinearParameters(family="lasso", alpha_max_multiplier=0.1),
        RegularizedLinearParameters(family="elastic_net", alpha_max_multiplier=0.1, l1_ratio=0.5),
    ),
)
def test_regularized_linear_adapter_preserves_numerical_owner(
    parameters: RegularizedLinearParameters,
) -> None:
    rng = np.random.default_rng(1729)
    training = _readonly(rng.normal(size=(200, 7)))
    targets = _readonly(rng.normal(size=200))
    prediction = _readonly(rng.normal(size=(50, 7)))
    before = (training.copy(), targets.copy(), prediction.copy())
    old = fit_regularized_linear(
        parameters=parameters,
        training_features=training,
        training_targets=targets,
        prediction_features=prediction,
    )
    recipe = build_regularized_linear_recipe(parameters)
    adapter = build_installed_alpha_model_catalog().resolve(recipe)
    fitted = adapter.fit(
        recipe=recipe,
        inputs=BoundAlphaTrainingInput(
            training_binding_hash="1" * 64,
            ordered_feature_ids=tuple(f"factor-{value}" for value in range(7)),
            features=training,
            targets=targets,
        ),
    )
    predicted = adapter.predict(
        estimator=fitted.estimator_content,
        inputs=BoundAlphaPredictionInput(
            training_binding_hash="1" * 64,
            ordered_feature_ids=tuple(f"factor-{value}" for value in range(7)),
            features=prediction,
        ),
    )
    np.testing.assert_array_equal(predicted.predictions, old.predictions)
    assert fitted.training_mse == old.training_mse
    assert fitted.estimator_content.payload["coefficient_hex"] == tuple(
        float(value).hex() for value in old.coefficients
    )
    for observed, frozen in zip((training, targets, prediction), before, strict=True):
        np.testing.assert_array_equal(observed, frozen)
        assert not observed.flags.writeable


def test_estimator_content_is_independent_of_fit_provenance() -> None:
    parameters = RegularizedLinearParameters(family="ridge", alpha=1.0)
    recipe = build_regularized_linear_recipe(parameters)
    features = _readonly(np.arange(30, dtype=np.float64).reshape(10, 3))
    targets = _readonly(np.linspace(-1.0, 1.0, 10))
    adapter = build_installed_alpha_model_catalog().resolve(recipe)
    fitted = adapter.fit(
        recipe=recipe,
        inputs=BoundAlphaTrainingInput(
            training_binding_hash="1" * 64,
            ordered_feature_ids=("a", "b", "c"),
            features=features,
            targets=targets,
        ),
    )
    first = AlphaFitProvenanceReceipt.create(
        recipe_hash=recipe.recipe_hash,
        training_binding_hash="1" * 64,
        estimator_content_hash=fitted.estimator_content.content_hash,
        package_identity_hash="2" * 64,
        fit_call_count=1,
        predict_call_count=1,
    )
    second = AlphaFitProvenanceReceipt.create(
        recipe_hash=recipe.recipe_hash,
        training_binding_hash="3" * 64,
        estimator_content_hash=fitted.estimator_content.content_hash,
        package_identity_hash="2" * 64,
        fit_call_count=1,
        predict_call_count=1,
    )
    assert first.estimator_content_hash == second.estimator_content_hash
    assert first.provenance_hash != second.provenance_hash


def test_explicit_catalog_rejects_unknown_adapter_and_schema() -> None:
    catalog = build_installed_alpha_model_catalog()
    with pytest.raises(ValueError, match="ADAPTER_NOT_INSTALLED"):
        catalog.resolve(
            AlphaModelRecipeEnvelope.create(
                adapter_id="unknown",
                recipe_schema_id="alpha-model.unknown",
                parameters={},
            )
        )
    with pytest.raises(ValueError, match="SCHEMA_NOT_INSTALLED"):
        catalog.resolve(
            AlphaModelRecipeEnvelope.create(
                adapter_id="regularized_linear",
                recipe_schema_id="alpha-model.wrong-schema",
                parameters={},
            )
        )


@pytest.mark.parametrize(
    "parameters",
    [
        {"family": "ridge", "alpha": 1.0, "seed": 7},
        {"family": "ridge", "alpha": "1.0"},
        {"alpha": 1.0},
    ],
)
def test_regularized_linear_adapter_refuses_parameters_outside_its_schema(
    parameters: dict[str, object],
) -> None:
    """regression: a declared parameter the schema has no field for was a crash.

    ``{"family": "ridge", "alpha": 1.0, "seed": 7}`` reached the dataclass
    constructor and surfaced as a ``TypeError`` -- an HTTP 500 at the Local Web
    PLAN route instead of the typed ``authoring_model_recipe_not_admissible``
    refusal every other inadmissible recipe gets.
    """

    adapter = RegularizedLinearAdapter()
    recipe = AlphaModelRecipeEnvelope.create(
        adapter_id=adapter.adapter_id,
        recipe_schema_id=adapter.recipe_schema_id,
        parameters=parameters,
    )
    with pytest.raises(ValueError, match="ALPHA_REGULARIZED_LINEAR_RECIPE_PARAMETERS_INVALID"):
        adapter.validate_recipe(recipe)


def test_parity_cli_rejects_shared_baseline_and_output_workspace(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[2] / "scripts" / "run_alpha_model_adapter_parity.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--baseline-workspace",
            str(tmp_path),
            "--output-workspace",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "alpha_research.parity_workspace_overlap" in result.stderr


class _CountingRegularizedLinearAdapter(RegularizedLinearAdapter):
    def __init__(self) -> None:
        self.fit_calls = 0

    def fit(self, *, recipe, inputs, fit_plan=None):  # type: ignore[no-untyped-def]
        self.fit_calls += 1
        return super().fit(recipe=recipe, inputs=inputs, fit_plan=fit_plan)


class _SyntheticMeanAdapter:
    """Second model type proving development execution has no family branch."""

    adapter_id = "synthetic_mean"
    recipe_schema_id = "alpha-model.synthetic-mean"
    search_domain_schema_id = "alpha-model.synthetic-mean.search-domain"

    def __init__(self) -> None:
        self.fit_calls = 0

    def describe_numerical_binding(self) -> AlphaModelNumericalBinding:
        return AlphaModelNumericalBinding.create(
            adapter_id=self.adapter_id,
            estimator_content_format_id="alpha-model.synthetic-mean.scalar",
            implementation_owners=("test.synthetic_mean",),
            deterministic_policy={"aggregation": "arithmetic_mean"},
            required_runtime_capabilities=("numpy",),
        )

    def fit_protocols(self, recipe: AlphaModelRecipeEnvelope) -> tuple[AlphaModelFitProtocol, ...]:
        del recipe
        return ("DIRECT_FIT",)

    def validate_recipe(self, recipe: AlphaModelRecipeEnvelope) -> float:
        if (
            recipe.adapter_id != self.adapter_id
            or recipe.recipe_schema_id != self.recipe_schema_id
            or set(recipe.parameters) != {"offset"}
        ):
            raise ValueError("SYNTHETIC_MEAN_RECIPE_INVALID")
        offset = float(recipe.parameters["offset"])
        if not np.isfinite(offset):
            raise ValueError("SYNTHETIC_MEAN_RECIPE_INVALID")
        return offset

    def validate_search_domain(self, domain: AlphaModelSearchDomainEnvelope) -> tuple[float, float]:
        if (
            domain.adapter_id != self.adapter_id
            or domain.recipe_schema_id != self.recipe_schema_id
            or domain.search_domain_schema_id != self.search_domain_schema_id
            or set(domain.constraints) != {"offset_min", "offset_max"}
        ):
            raise ValueError("SYNTHETIC_MEAN_DOMAIN_INVALID")
        lower = float(domain.constraints["offset_min"])
        upper = float(domain.constraints["offset_max"])
        if not np.isfinite((lower, upper)).all() or lower > upper:
            raise ValueError("SYNTHETIC_MEAN_DOMAIN_INVALID")
        return lower, upper

    def validate_recipe_for_domain(
        self,
        *,
        recipe: AlphaModelRecipeEnvelope,
        domain: AlphaModelSearchDomainEnvelope,
    ) -> float:
        offset = self.validate_recipe(recipe)
        lower, upper = self.validate_search_domain(domain)
        if not lower <= offset <= upper:
            raise ValueError("ALPHA_MODEL_RECIPE_OUTSIDE_SEARCH_DOMAIN")
        return offset

    def fit(
        self,
        *,
        recipe: AlphaModelRecipeEnvelope,
        inputs: BoundAlphaTrainingInput,
        fit_plan=None,  # type: ignore[no-untyped-def]
    ) -> AlphaModelFitResult:
        self.fit_calls += 1
        mean = float(np.mean(inputs.targets)) + self.validate_recipe(recipe)
        residuals = inputs.targets - mean
        content = AlphaEstimatorContent.create(
            adapter_id=self.adapter_id,
            content_format_id="alpha-model.synthetic-mean.scalar",
            ordered_feature_ids=inputs.ordered_feature_ids,
            payload={"mean_hex": mean.hex()},
        )
        projection = AlphaModelStateProjection.create(
            adapter_id=self.adapter_id,
            model_family_id="synthetic_mean",
            state_kind="SYNTHETIC_SCALAR",
            state_schema_id="alpha-model.synthetic-mean.development-state",
            payload={"mean_hex": mean.hex()},
        )
        return AlphaModelFitResult(
            estimator_content=content,
            state_projection=projection,
            training_mse=float(np.mean(np.square(residuals))),
            iteration_count=None,
        )

    def predict(
        self,
        *,
        estimator: AlphaEstimatorContent,
        inputs: BoundAlphaPredictionInput,
    ) -> AlphaModelPredictionResult:
        if (
            estimator.adapter_id != self.adapter_id
            or estimator.content_format_id != "alpha-model.synthetic-mean.scalar"
            or estimator.ordered_feature_ids != inputs.ordered_feature_ids
        ):
            raise ValueError("SYNTHETIC_MEAN_ESTIMATOR_BINDING_INVALID")
        predictions = np.full(
            len(inputs.features),
            float.fromhex(str(estimator.payload["mean_hex"])),
            dtype=np.float64,
        )
        predictions.setflags(write=False)
        return AlphaModelPredictionResult(predictions=predictions)


def test_synthetic_adapter_completes_generic_development_execution(tmp_path: Path) -> None:
    plan, workspace = _plan_and_workspace()
    adapter = _SyntheticMeanAdapter()
    catalog = AlphaModelCatalog((adapter,))
    domain = AlphaModelSearchDomainEnvelope.create(
        adapter_id=adapter.adapter_id,
        recipe_schema_id=adapter.recipe_schema_id,
        search_domain_schema_id=adapter.search_domain_schema_id,
        constraints={"offset_min": -1.0, "offset_max": 1.0},
    )
    mandate = AlphaResearchModelMandate.create(
        catalog_binding=catalog.binding,
        ordered_search_domains=(domain,),
        target_current_qualified_candidates=1,
        initial_batch_size=1,
        refinement_batch_max_size=1,
        max_batch_count=1,
        max_unique_new_recipes=1,
    )
    recipe = mandate.admit_proposal(
        proposal=AlphaModelRecipeProposal(
            capability_handle="capability-1",
            parameters={"offset": 0.25},
        ),
        catalog=catalog,
        admitted_target_lanes=None,
    )
    program = _program(plan.foundation, plan, mandate)
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
    store = AlphaCurrentArtifactStore(tmp_path)

    result = execute_alpha_model_batch(
        program=program,
        batch=batch,
        fold_plan=plan,
        store=store,
        array_workspace=workspace,
        model_catalog=catalog,
        model_mandate=mandate,
    )

    assert result.fit_call_count == plan.fold_count
    assert result.predict_call_count == plan.fold_count * 2
    assert result.metric_call_count == plan.fold_count
    assert adapter.fit_calls == plan.fold_count
    candidate = result.candidates[0]
    state = store.load_development_estimator_state(candidate.estimator_state_hashes[0])
    assert state.adapter_id == adapter.adapter_id
    assert state.state_kind == "SYNTHETIC_SCALAR"
    assert state.fit_evidence_hash is not None
    evidence = store.load_development_fit_evidence(state.fit_evidence_hash)
    assert evidence.adapter_id == adapter.adapter_id
    assert evidence.recipe_hash == recipe.recipe.recipe_hash
    assert evidence.training_binding_hash == workspace.folds[0].training_input_binding.binding_hash
    assert evidence.fit_plan_hash is not None
    fit_plan = store.load_model_fit_plan(evidence.fit_plan_hash)
    assert fit_plan.protocol_id == "DIRECT_FIT"
    assert fit_plan.parent_training_binding_hash == evidence.training_binding_hash
    assert evidence.estimator_content_hash
    assert evidence.fit_provenance_hash
    assert evidence.score_evidence_hash


def _nested_fit_plan_inputs():  # type: ignore[no-untyped-def]
    _plan, workspace = _plan_and_workspace()
    fold = workspace.folds[0]
    assert fold.training_input_binding is not None
    training_input, _prediction_input = bind_alpha_model_inputs(
        binding=fold.training_input_binding,
        ordered_feature_ids=fold.ordered_factor_ids,
        training_features=fold.training_features,
        training_targets=fold.training_targets,
        training_mask=fold.training_model_mask,
        prediction_features=fold.validation_features,
        prediction_mask=fold.validation_feature_complete,
    )
    session_count = len(fold.training_sessions)
    domain = AlphaModelSearchDomainEnvelope.create(
        adapter_id="synthetic_nested",
        recipe_schema_id="alpha-model.synthetic-nested",
        search_domain_schema_id="alpha-model.synthetic-nested.search-domain",
        constraints={
            "inner_training_sessions": session_count - 2,
            "purge_sessions": 1,
            "inner_validation_sessions": 1,
            "selection_metric_id": "l2",
            "maximum_iterations": 10,
            "early_stopping_rounds": 2,
        },
    )
    numerical_binding = AlphaModelNumericalBinding.create(
        adapter_id="synthetic_nested",
        estimator_content_format_id="alpha-model.synthetic-nested.content",
        implementation_owners=("test.synthetic_nested",),
        deterministic_policy={"seed": 1729},
        required_runtime_capabilities=(NESTED_FIT_RUNTIME_CAPABILITY,),
    )
    return fold, training_input, domain, numerical_binding


def test_every_fit_plan_call_site_states_its_protocol() -> None:
    """Every `build_alpha_model_fit_plan(...)` call in the source tree passes ``protocol``.

    The parameter is keyword-only and required, so an untold caller fails at
    run time, in the fold, after the arrays were leased -- and the callers are
    not all under strict mypy. Read the source instead.
    """

    import ast

    root = Path(__file__).resolve().parents[2] / "src"
    untold: list[str] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            callee = node.func
            name = callee.id if isinstance(callee, ast.Name) else getattr(callee, "attr", None)
            if name != "build_alpha_model_fit_plan":
                continue
            if not any(keyword.arg == "protocol" for keyword in node.keywords):
                untold.append(f"{path.relative_to(root)}:{node.lineno}")
    assert untold == [], untold


def test_nested_fit_plan_binds_exact_partitions_and_readonly_values() -> None:
    fold, training_input, domain, numerical_binding = _nested_fit_plan_inputs()

    plan, bound = build_alpha_model_fit_plan(
        fold=fold,
        training_input=training_input,
        domain=domain,
        numerical_binding=numerical_binding,
        protocol="NESTED_EARLY_STOPPING_REFIT",
    )

    assert AlphaModelFitPlan.model_validate(plan.model_dump()) == plan
    assert plan.protocol_id == "NESTED_EARLY_STOPPING_REFIT"
    assert plan.tuning_training_sessions is not None
    assert plan.purge_sessions is not None
    assert plan.tuning_validation_sessions is not None
    assert (
        plan.tuning_training_sessions + plan.purge_sessions + plan.tuning_validation_sessions
        == fold.training_sessions
    )
    assert bound.fit_plan_hash == plan.fit_plan_hash
    assert bound.tuning_training_features is not None
    assert bound.tuning_validation_features is not None
    assert not bound.tuning_training_features.flags.writeable
    assert not bound.tuning_validation_features.flags.writeable


@pytest.mark.parametrize("mutation", ("parent", "row_order", "window"))
def test_nested_fit_plan_rejects_authority_mismatch_before_fit(mutation: str) -> None:
    fold, training_input, domain, numerical_binding = _nested_fit_plan_inputs()
    if mutation == "parent":
        training_input = replace(training_input, training_binding_hash="f" * 64)
    elif mutation == "row_order":
        fold = replace(fold, training_row_sessions=tuple(reversed(fold.training_row_sessions)))
    else:
        constraints = dict(domain.constraints)
        constraints["inner_training_sessions"] = int(constraints["inner_training_sessions"]) - 1
        domain = AlphaModelSearchDomainEnvelope.create(
            adapter_id=domain.adapter_id,
            recipe_schema_id=domain.recipe_schema_id,
            search_domain_schema_id=domain.search_domain_schema_id,
            constraints=constraints,
        )

    with pytest.raises(AlphaFitPlanAuthorityError):
        build_alpha_model_fit_plan(
            fold=fold,
            training_input=training_input,
            domain=domain,
            numerical_binding=numerical_binding,
            protocol="NESTED_EARLY_STOPPING_REFIT",
        )


def test_nested_fit_plan_rejects_overlapping_partition_tamper() -> None:
    fold, training_input, domain, numerical_binding = _nested_fit_plan_inputs()
    plan, _bound = build_alpha_model_fit_plan(
        fold=fold,
        training_input=training_input,
        domain=domain,
        numerical_binding=numerical_binding,
        protocol="NESTED_EARLY_STOPPING_REFIT",
    )
    payload = plan.model_dump(mode="json", exclude={"fit_plan_hash"})
    assert plan.tuning_training_sessions is not None
    payload["purge_sessions"] = [plan.tuning_training_sessions[-1].isoformat()]
    payload["purge_sessions_hash"] = canonical_hash((plan.tuning_training_sessions[-1],))
    payload["fit_plan_hash"] = canonical_hash(payload)

    with pytest.raises(ValueError, match="ALPHA_MODEL_NESTED_FIT_PLAN_INVALID"):
        AlphaModelFitPlan.model_validate(payload)


def test_same_shape_wrong_target_fails_before_adapter_fit(tmp_path: Path) -> None:
    plan, workspace = _plan_and_workspace()
    original = workspace.folds[0]
    assert original.training_input_binding is not None
    values = original.training_input_binding.model_dump(mode="python", exclude={"binding_hash"})
    values["target_policy_hash"] = "f" * 64
    values["target_surface_hash"] = "e" * 64
    wrong_binding = AlphaTrainingInputBinding(
        **values,
        binding_hash=canonical_hash(values),
    )
    workspace.folds = (
        replace(original, training_input_binding=wrong_binding),
        *workspace.folds[1:],
    )
    program = _program(plan.foundation, plan)
    adapter = _CountingRegularizedLinearAdapter()
    with pytest.raises(AlphaTrainingInputAuthorityError) as raised:
        execute_alpha_model_batch(
            program=program,
            batch=_batch(program),
            fold_plan=plan,
            store=AlphaCurrentArtifactStore(tmp_path),
            array_workspace=workspace,
            model_catalog=_installed_catalog_counting(adapter),
            model_mandate=build_current_alpha_research_model_mandate(),
        )
    assert raised.value.experiment_disposition == "INVALID_EXPERIMENT"
    assert raised.value.failure_class == "TARGET_AUTHORITY_MISMATCH"
    assert raised.value.fit_call_count == 0
    assert raised.value.scientific_admission_effect == "NONE"
    assert adapter.fit_calls == 0


def test_reordered_feature_axis_fails_before_adapter_fit(tmp_path: Path) -> None:
    plan, workspace = _plan_and_workspace()
    original = workspace.folds[0]
    reversed_ids = tuple(reversed(original.ordered_factor_ids))
    reordered = replace(
        original,
        ordered_factor_ids=reversed_ids,
        training_features=_readonly(original.training_features[:, ::-1].copy()),
        validation_features=_readonly(original.validation_features[:, ::-1].copy()),
        training_input_binding=None,
    )
    workspace.folds = (
        _bound_fold(reordered, plan.foundation),
        *workspace.folds[1:],
    )
    program = _program(plan.foundation, plan)
    adapter = _CountingRegularizedLinearAdapter()
    with pytest.raises(AlphaTrainingInputAuthorityError) as raised:
        execute_alpha_model_batch(
            program=program,
            batch=_batch(program),
            fold_plan=plan,
            store=AlphaCurrentArtifactStore(tmp_path),
            array_workspace=workspace,
            model_catalog=_installed_catalog_counting(adapter),
            model_mandate=build_current_alpha_research_model_mandate(),
        )
    assert raised.value.failure_class == "FEATURE_AXIS_AUTHORITY_MISMATCH"
    assert raised.value.fit_call_count == 0
    assert adapter.fit_calls == 0


def test_goal_development_adapter_matches_frozen_linear_state(tmp_path: Path) -> None:
    plan, workspace = _plan_and_workspace()
    program = _program(plan.foundation, plan)
    batch = _batch(program)
    result = execute_alpha_model_batch(
        program=program,
        batch=batch,
        fold_plan=plan,
        store=AlphaCurrentArtifactStore(tmp_path),
        array_workspace=workspace,
        model_mandate=build_current_alpha_research_model_mandate(),
    )
    first = result.candidates[0]
    store = AlphaCurrentArtifactStore(tmp_path)
    state = store.load_development_estimator_state(first.estimator_state_hashes[0])
    numerical = store.load_candidate_numerical_fold_result(first.numerical_result_hashes[0])
    sidecar, estimator, provenance = store.load_model_fit_sidecar(numerical.execution_binding_hash)
    fold = workspace.folds[0]
    golden = fit_regularized_linear(
        parameters=RegularizedLinearParameters(family="ridge", alpha=1.0),
        training_features=fold.training_features[fold.training_model_mask],
        training_targets=fold.training_targets[fold.training_model_mask],
        prediction_features=fold.validation_features[fold.validation_feature_complete],
    )
    assert state.coefficient_hex == tuple(float(value).hex() for value in golden.coefficients)
    assert state.intercept_hex == float(golden.intercept).hex()
    assert state.training_mse == golden.training_mse
    assert sidecar.estimator_content_hash == estimator.content_hash
    assert sidecar.fit_provenance_hash == provenance.provenance_hash
    assert provenance.training_binding_hash == fold.training_input_binding.binding_hash
    assert estimator.content_hash != provenance.provenance_hash


def test_a_formation_scope_verifies_each_sidecar_once_and_copies_the_estimator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """requirement (PERF-1 lever 5): one formation verifies a sidecar once, not once a use."""
    from alphalattice.investment.alpha_research.experiments.development_artifacts import (
        model_fit_sidecar_scope,
    )

    plan, workspace = _plan_and_workspace()
    program = _program(plan.foundation, plan)
    result = execute_alpha_model_batch(
        program=program,
        batch=_batch(program),
        fold_plan=plan,
        store=AlphaCurrentArtifactStore(tmp_path),
        array_workspace=workspace,
        model_mandate=build_current_alpha_research_model_mandate(),
    )
    store = AlphaCurrentArtifactStore(tmp_path)
    numerical = store.load_candidate_numerical_fold_result(
        result.candidates[0].numerical_result_hashes[0]
    )
    operation = numerical.execution_binding_hash
    fresh = store.load_model_fit_sidecar(operation)
    reads = 0
    read_text = Path.read_text

    def counted(path, *args, **kwargs):
        nonlocal reads
        if path.name == f"{operation}.json" and path.parent.name.startswith("model-fit-sidecar"):
            reads += 1
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counted)
    with model_fit_sidecar_scope():
        loads = [store.load_model_fit_sidecar(operation) for _ in range(3)]
        loads[0][1].payload["mutated"] = True
        after = store.load_model_fit_sidecar(operation)
    assert reads == 1
    assert all(load[0] == fresh[0] and load[2] == fresh[2] for load in loads)
    assert loads[1][1] == fresh[1] and loads[1][1] is not loads[2][1]
    assert "mutated" not in after[1].payload
    store.load_model_fit_sidecar(operation)
    assert reads == 2  # nothing survives the scope


@pytest.mark.parametrize("mutation", ("missing", "operation_mismatch"))
def test_goal_development_reuse_requires_complete_model_sidecar(
    tmp_path: Path,
    mutation: str,
) -> None:
    plan, workspace = _plan_and_workspace()
    program = _program(plan.foundation, plan)
    batch = _batch(program)
    mandate = build_current_alpha_research_model_mandate()
    adapter = _CountingRegularizedLinearAdapter()
    catalog = _installed_catalog_counting(adapter)
    store = AlphaCurrentArtifactStore(tmp_path)
    result = execute_alpha_model_batch(
        program=program,
        batch=batch,
        fold_plan=plan,
        store=store,
        array_workspace=workspace,
        model_catalog=catalog,
        model_mandate=mandate,
    )
    numerical = store.load_candidate_numerical_fold_result(
        result.candidates[0].numerical_result_hashes[0]
    )
    pointer = (
        store.root
        / "current"
        / "model-fit-sidecar-by-operation"
        / f"{numerical.execution_binding_hash}.json"
    )
    fit_calls = adapter.fit_calls
    if mutation == "missing":
        pointer.unlink()
        expected = "model fit sidecar is unavailable"
    else:
        payload = json.loads(pointer.read_text(encoding="utf-8"))
        replacement = AlphaModelFitSidecar.create(
            operation_binding_hash="f" * 64,
            estimator_content_hash=str(payload["estimator_content_hash"]),
            fit_provenance_hash=str(payload["fit_provenance_hash"]),
        )
        pointer.write_text(
            json.dumps(replacement.model_dump(mode="json"), sort_keys=True),
            encoding="utf-8",
        )
        expected = "model fit sidecar operation changed"

    with pytest.raises(AlphaDevelopmentArtifactReadbackError, match=expected):
        execute_alpha_model_batch(
            program=program,
            batch=batch,
            fold_plan=plan,
            store=store,
            array_workspace=workspace,
            model_catalog=catalog,
            model_mandate=mandate,
        )
    assert adapter.fit_calls == fit_calls


def _development_state(
    *,
    candidate_id: str,
    fold_index: int,
    factor_ids: tuple[str, ...],
    coefficients: np.ndarray,
    intercept: float,
    training_mse: float,
    score_mean: float,
    score_std: float,
) -> AlphaEstimatorState:
    return seal_current_contract(
        AlphaEstimatorState,
        {
            "kind": "AlphaEstimatorState",
            "request_hash": "b" * 64,
            "candidate_id": candidate_id,
            "scope": "DEVELOPMENT_FOLD",
            "family_id": "ridge",
            "state_kind": "LINEAR",
            "fold_index": fold_index,
            "ordered_factor_ids": factor_ids,
            "coefficient_hex": tuple(float(value).hex() for value in coefficients),
            "intercept_hex": float(intercept).hex(),
            "coefficient_l2_norm": float(np.linalg.norm(coefficients)),
            "coefficient_max_abs": float(np.max(np.abs(coefficients))),
            "nonzero_support_count": int(np.count_nonzero(coefficients)),
            "model_text_hash": None,
            "best_iteration": None,
            "feature_gain_hex": (),
            "top_feature_gain_share": None,
            "training_mse": training_mse,
            "validation_score_mean": score_mean,
            "validation_score_std": score_std,
            "validation_score_coverage": 1.0,
            "validation_session_statistics": (
                AlphaSessionScoreStatistics(
                    formation_session=date(2025, 1, fold_index + 1),
                    listing_count=100,
                    scored_count=100,
                    score_mean=score_mean,
                    score_std=score_std,
                    score_coverage=1.0,
                ),
            ),
        },
        "state_hash",
    )


def _current_refit_fixture() -> tuple[AlphaFoldArrays, AlphaCurrentRefitArrays]:
    """A fold of the synthetic plan and the current window its validation rows open."""

    plan, workspace = _plan_and_workspace()
    fold = workspace.folds[0]
    assert fold.training_input_binding is not None
    listing_count = len(set(fold.validation_listing_ids))
    current_features = fold.validation_features[:listing_count]
    current_complete = fold.validation_feature_complete[:listing_count]
    current_listings = fold.validation_listing_ids[:listing_count]
    current_sessions = fold.validation_row_sessions[:listing_count]
    current_feature_hashes = fold.validation_feature_row_hashes[:listing_count]
    current_targets = _readonly(np.full(listing_count, np.nan, dtype=np.float64))
    current_economic = _readonly(np.full(listing_count, np.nan, dtype=np.float64))
    assert fold.training_economic_returns is not None
    binding = build_alpha_training_input_binding(
        scope="CURRENT_REFIT",
        foundation_hash=plan.foundation.foundation_hash,
        feature_panel_snapshot_hash=plan.foundation.feature_panel_snapshot_hash,
        causal_outcome_snapshot_hash=plan.foundation.execution_outcome.snapshot_hash,
        target_policy=None,
        ordered_feature_ids=fold.ordered_factor_ids,
        feature_context_hash=None,
        training_row_sessions=fold.training_row_sessions,
        training_row_listing_ids=fold.training_listing_ids,
        prediction_row_sessions=current_sessions,
        prediction_row_listing_ids=current_listings,
        training_features=fold.training_features,
        training_targets=fold.training_targets,
        training_mask=fold.training_model_mask,
        prediction_features=current_features,
        prediction_targets=current_targets,
        prediction_mask=current_complete,
        training_feature_row_hashes=fold.training_feature_row_hashes,
        training_outcome_row_hashes=fold.training_outcome_row_hashes,
        prediction_feature_row_hashes=current_feature_hashes,
        prediction_outcome_row_hashes=(None,) * listing_count,
        training_economic_returns=fold.training_economic_returns,
        prediction_economic_returns=current_economic,
        training_cutoff=fold.training_sessions[-1],
        outcome_maturity_session=fold.validation_sessions[0],
        prediction_anchor=fold.validation_sessions[0],
        fold_commitment_hash=None,
    )
    arrays = AlphaCurrentRefitArrays(
        ordered_factor_ids=fold.ordered_factor_ids,
        ordered_listing_ids=current_listings,
        training_sessions=fold.training_sessions,
        training_cutoff=fold.training_sessions[-1],
        formation_session=fold.validation_sessions[0],
        training_features=fold.training_features,
        training_targets=fold.training_targets,
        training_mask=fold.training_model_mask,
        current_features=current_features,
        current_feature_complete=current_complete,
        training_row_sessions=fold.training_row_sessions,
        training_feature_row_hashes=fold.training_feature_row_hashes,
        training_outcome_row_hashes=fold.training_outcome_row_hashes,
        current_feature_row_hashes=current_feature_hashes,
        training_economic_returns=fold.training_economic_returns,
        training_input_binding=binding,
    )
    return fold, arrays


def test_goal_current_refit_uses_same_adapter_without_numerical_drift() -> None:
    fold, arrays = _current_refit_fixture()
    parameters = RegularizedLinearParameters(family="ridge", alpha=1.0)
    golden = fit_regularized_linear(
        parameters=parameters,
        training_features=fold.training_features[fold.training_model_mask],
        training_targets=fold.training_targets[fold.training_model_mask],
        prediction_features=arrays.current_features[arrays.current_feature_complete],
    )
    candidate_id = "agent-linear-current-fixture"
    development_states = tuple(
        _development_state(
            candidate_id=candidate_id,
            fold_index=index,
            factor_ids=fold.ordered_factor_ids,
            coefficients=golden.coefficients * scale,
            intercept=intercept,
            training_mse=mse,
            score_mean=mean,
            score_std=std,
        )
        for index, scale, intercept, mse, mean, std in (
            (0, 0.5, -1.0, 0.0, -100.0, 0.001),
            (1, 1.0, 0.0, 0.5, 0.0, 50.0),
            (2, 2.0, 1.0, 1.0, 100.0, 100.0),
        )
    )
    result = fit_and_assess_current_regularized_linear(
        request_hash="b" * 64,
        candidate_id=candidate_id,
        parameters=parameters,
        arrays=arrays,
        development_states=development_states,
        package_identity_hash="0" * 64,
        retain_diagnostic_score=True,
    )
    assert result.estimator_state.coefficient_hex == tuple(
        float(value).hex() for value in golden.coefficients
    )
    assert result.estimator_state.intercept_hex == float(golden.intercept).hex()
    assert result.estimator_state.training_mse == golden.training_mse
    assert result.score_table is not None
    np.testing.assert_array_equal(
        np.asarray(result.score_table["score"].to_numpy()), golden.predictions
    )
    assert result.estimator_content_hash is not None
    assert result.fit_provenance_hash is not None
    assert result.fit_operation_hash is not None
    assert result.estimator_content is not None
    assert result.fit_provenance is not None
    assert result.fit_provenance.estimator_content_hash == result.estimator_content.content_hash


class _AgentDeclaredModel:
    """An agent's model whose state is a kind the installed families do not seal (V342)."""

    adapter_id = "agent_declared_mean"
    content_format_id = "alpha-model.agent-declared-mean.estimator"
    recipe_schema_id = "alpha-model.agent-declared-mean"
    search_domain_schema_id = "alpha-model.agent-declared-mean.search-domain"

    def __init__(self, state_kind: str = "DECLARED") -> None:
        self.state_kind = state_kind

    def describe_numerical_binding(self) -> AlphaModelNumericalBinding:
        return AlphaModelNumericalBinding.create(
            adapter_id=self.adapter_id,
            estimator_content_format_id=self.content_format_id,
            implementation_owners=("tests.agent_declared_mean",),
            deterministic_policy={"mode": "test"},
            required_runtime_capabilities=(),
        )

    def fit_protocols(self, recipe: AlphaModelRecipeEnvelope) -> tuple[AlphaModelFitProtocol, ...]:
        del recipe
        return ("DIRECT_FIT",)

    def validate_recipe(self, recipe: AlphaModelRecipeEnvelope) -> None:
        if recipe.adapter_id != self.adapter_id:
            raise ValueError("AGENT_DECLARED_RECIPE_INVALID")

    def fit(
        self,
        *,
        recipe: AlphaModelRecipeEnvelope,
        inputs: BoundAlphaTrainingInput,
        fit_plan: object = None,
    ) -> AlphaModelFitResult:
        del recipe, fit_plan
        x, y = inputs.features, inputs.targets
        coefficients = np.linalg.solve(x.T @ x + np.eye(x.shape[1]), x.T @ y)
        content = AlphaEstimatorContent.create(
            adapter_id=self.adapter_id,
            content_format_id=self.content_format_id,
            ordered_feature_ids=inputs.ordered_feature_ids,
            payload={"coefficient_hex": [float(v).hex() for v in coefficients]},
        )
        projection = AlphaModelStateProjection.create(
            adapter_id=self.adapter_id,
            model_family_id="agent_mean",
            state_kind=self.state_kind,
            state_schema_id=self.content_format_id,
            payload={"weight_count": len(coefficients)},
        )
        mse = float(np.mean(np.square(x @ coefficients - y)))
        return AlphaModelFitResult(
            estimator_content=content,
            state_projection=projection,
            training_mse=mse,
            iteration_count=None,
        )

    def predict(
        self, *, estimator: AlphaEstimatorContent, inputs: BoundAlphaPredictionInput
    ) -> AlphaModelPredictionResult:
        coefficients = np.array([float.fromhex(v) for v in estimator.payload["coefficient_hex"]])
        values = np.asarray(inputs.features @ coefficients, dtype=np.float64)
        values.setflags(write=False)
        return AlphaModelPredictionResult(predictions=values)


def _declared_development_state(
    *, candidate_id: str, fold_index: int, factor_ids: tuple[str, ...], mean: float, std: float
) -> AlphaDevelopmentEstimatorState:
    adapter = _AgentDeclaredModel()
    payload: dict[str, object] = {"weight_count": len(factor_ids)}
    projection_hash = canonical_hash(
        {
            "adapter_id": adapter.adapter_id,
            "model_family_id": "agent_mean",
            "state_kind": "DECLARED",
            "state_schema_id": adapter.content_format_id,
            "payload": payload,
        }
    )
    return seal_current_contract(
        AlphaDevelopmentEstimatorState,
        {
            "kind": "AlphaDevelopmentEstimatorState",
            "execution_binding_hash": "1" * 64,
            "development_surface_binding_hash": "2" * 64,
            "candidate_id": candidate_id,
            "candidate_card_hash": "3" * 64,
            "fold_index": fold_index,
            "fold_commitment_hash": "4" * 64,
            "adapter_id": adapter.adapter_id,
            "numerical_binding_hash": adapter.describe_numerical_binding().numerical_binding_hash,
            "fit_evidence_hash": "5" * 64,
            "state_projection_hash": projection_hash,
            "state_schema_id": adapter.content_format_id,
            "projection_payload": payload,
            "family_id": "agent_mean",
            "state_kind": "DECLARED",
            "ordered_factor_ids": factor_ids,
            "coefficient_hex": (),
            "intercept_hex": "0x0.0p+0",
            "coefficient_l2_norm": 0.0,
            "coefficient_max_abs": 0.0,
            "nonzero_support_count": None,
            "model_text_hash": None,
            "best_iteration": None,
            "feature_gain_hex": (),
            "top_feature_gain_share": None,
            "training_mse": 0.5,
            "validation_score_mean": mean,
            "validation_score_std": std,
            "validation_score_coverage": 1.0,
            "validation_session_statistics": (
                AlphaSessionScoreStatistics(
                    formation_session=date(2025, 1, fold_index + 1),
                    listing_count=100,
                    scored_count=100,
                    score_mean=mean,
                    score_std=std,
                    score_coverage=1.0,
                ),
            ),
        },
        "state_hash",
    )


def test_an_agents_model_of_its_own_kind_is_refitted_through_its_projection() -> None:
    """requirement (V342): qualification and the current refit sealed the installed families'
    states alone, so an agent's model could finish a study and never be qualified; its current
    refit seals the kind its adapter projects, bound to the adapter, its numerical binding and
    the projection, and the stability policy judges it by its model-agnostic statistics; a
    declared state carrying a kind's diagnostics, and a tree an agent projects, are refused."""

    fold, arrays = _current_refit_fixture()
    candidate_id = "agent-declared-current-fixture"
    states = tuple(
        _declared_development_state(
            candidate_id=candidate_id,
            fold_index=index,
            factor_ids=fold.ordered_factor_ids,
            mean=mean,
            std=std,
        )
        for index, mean, std in ((0, -100.0, 0.001), (1, 0.0, 50.0), (2, 100.0, 100.0))
    )
    adapter = _AgentDeclaredModel()
    recipe = AlphaModelRecipeEnvelope.create(
        adapter_id=adapter.adapter_id, recipe_schema_id=adapter.recipe_schema_id, parameters={}
    )

    def refit(model: _AgentDeclaredModel):  # type: ignore[no-untyped-def]
        return fit_and_assess_current_model(
            request_hash="b" * 64,
            candidate_id=candidate_id,
            recipe=recipe,
            arrays=arrays,
            development_states=states,
            package_identity_hash="0" * 64,
            model_catalog=AlphaModelCatalog((model,)),  # type: ignore[arg-type]
            retain_diagnostic_score=True,
        )

    result = refit(adapter)
    state = result.estimator_state
    assert (state.scope, state.family_id, state.state_kind) == (
        "CURRENT_REFIT",
        "agent_mean",
        "DECLARED",
    )
    assert state.adapter_id == adapter.adapter_id
    assert state.numerical_binding_hash == (
        adapter.describe_numerical_binding().numerical_binding_hash
    )
    assert state.fit_evidence_hash == result.fit_provenance_hash
    assert state.projection_payload == {"weight_count": len(fold.ordered_factor_ids)}
    assert not state.coefficient_hex and state.model_text_hash is None
    assert {check.check_id for check in result.assessment.checks} == {
        "TRAINING_MSE",
        "CURRENT_SCORE_MEAN",
        "CURRENT_SCORE_STD",
        "CURRENT_SCORE_COVERAGE",
    }
    assert result.score_table is not None
    with pytest.raises(ValueError, match="declared estimator state is incomplete"):
        AlphaEstimatorState.model_validate(
            {**state.model_dump(mode="json"), "coefficient_hex": ["0x1.0p+0"]}
        )
    with pytest.raises(ValueError, match="ALPHA_CURRENT_REFIT_AGENT_TREE_UNSUPPORTED"):
        refit(_AgentDeclaredModel(state_kind="TREE"))
