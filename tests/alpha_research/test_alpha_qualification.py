"""What stays of the Alpha goal loop: the qualification, its registry and the committer (GR3).

Moved from the loop's test files when the loop retired; the qualification Task over a family
of development studies is proved by its own tests and acceptance."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import pytest
from pydantic import ValidationError

import alphalattice.investment.alpha_research.candidates.qualification as qualification_module
from alphalattice.capabilities.alpha_modeling.catalog import (
    AlphaModelCatalog,
    build_installed_alpha_model_catalog,
)
from alphalattice.capabilities.alpha_modeling.contracts import (
    AlphaModelFitProtocol,
    AlphaModelNumericalBinding,
    AlphaModelRecipeEnvelope,
    AlphaModelSearchDomainEnvelope,
)
from alphalattice.control.product_host.composition.plain_refusals import refused
from alphalattice.investment.alpha_research.candidates.artifacts import (
    AlphaGoalResearchArtifactStore,
)
from alphalattice.investment.alpha_research.candidates.committer import AlphaGoalResultCommitter
from alphalattice.investment.alpha_research.candidates.contracts import (
    AlphaCandidateRecord,
    AlphaCandidateRegistrySnapshot,
    AlphaCandidateStatus,
    AlphaExperimentBatch,
    AlphaExperimentBatchResult,
    AlphaExperimentCandidateResult,
    AlphaGoalDisposition,
    AlphaGoalProgress,
    AlphaModelResearchGoalCriteria,
    AlphaQualificationSnapshot,
    AlphaResearchProgram,
    ElasticNetModelSpec,
    LassoModelSpec,
    RidgeModelSpec,
    resolve_model_spec,
    seal_contract,
)
from alphalattice.investment.alpha_research.candidates.control import (
    build_goal_criteria,
    family_registry,
    resolve_goal_progress,
)
from alphalattice.investment.alpha_research.candidates.qualification import (
    _carry_forward_prior_current_rejection,
    _index_attempted_family,
    qualify_alpha_model_candidates,
)
from alphalattice.investment.alpha_research.candidates.qualification_task import (
    AlphaFamilyQualification,
)
from alphalattice.investment.alpha_research.experiments.contracts import (
    AlphaDevelopmentProgram,
    candidate_id_for_spec,
)
from alphalattice.investment.alpha_research.experiments.family_qualification import (
    METHOD,
    AlphaStudy,
    alpha_family,
    alpha_question_hash,
    planned_calls,
)
from alphalattice.investment.alpha_research.experiments.mandate import (
    AlphaModelRecipeProposal,
    AlphaResearchModelMandate,
    AlphaResearchModelRecipe,
    build_current_alpha_research_model_mandate,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.protocols.research_authoring.contracts import AuthoringError

HASH = "a" * 64


def _model_mandate() -> AlphaResearchModelMandate:
    return build_current_alpha_research_model_mandate()


def _criteria() -> AlphaModelResearchGoalCriteria:
    return build_goal_criteria(_model_mandate())


def _proposals(values: tuple[float, ...]) -> tuple[AlphaModelRecipeProposal, ...]:
    return tuple(
        AlphaModelRecipeProposal(
            capability_handle="capability-1",
            parameters={"family": "ridge", "alpha": value},
        )
        for value in values
    )


def _research_recipes(values: tuple[float, ...]) -> tuple[AlphaResearchModelRecipe, ...]:
    catalog = build_installed_alpha_model_catalog()
    mandate = build_current_alpha_research_model_mandate(catalog=catalog)
    return tuple(
        mandate.admit_proposal(
            proposal=value,
            catalog=catalog,
            admitted_target_lanes=None,
        )
        for value in _proposals(values)
    )


@pytest.mark.parametrize("adapter_id", ("synthetic_mean", "lightgbm"))
def test_adapter_without_current_evaluator_is_scientific_non_admission(
    adapter_id: str,
) -> None:
    recipe = AlphaResearchModelRecipe.create(
        search_domain_hash="1" * 64,
        recipe=AlphaModelRecipeEnvelope.create(
            adapter_id=adapter_id,
            recipe_schema_id="alpha-model.synthetic-mean",
            parameters={"offset": 0.0},
        ),
        target_lane=None,
    )

    with pytest.raises(ValueError, match="SCIENTIFIC_NON_ADMISSION"):
        qualification_module._parameters(recipe)


class _DevelopmentOnlyAdapter:
    adapter_id = "development_only"
    recipe_schema_id = "alpha-model.development-only"
    search_domain_schema_id = "alpha-model.development-only.search-domain"

    def describe_numerical_binding(self) -> AlphaModelNumericalBinding:
        return AlphaModelNumericalBinding.create(
            adapter_id=self.adapter_id,
            estimator_content_format_id="alpha-model.development-only.content",
            implementation_owners=("test.development_only",),
            deterministic_policy={"mode": "test"},
            required_runtime_capabilities=(),
        )

    def fit_protocols(self, recipe: AlphaModelRecipeEnvelope) -> tuple[AlphaModelFitProtocol, ...]:
        del recipe
        return ("DIRECT_FIT",)

    def validate_recipe(self, recipe: AlphaModelRecipeEnvelope) -> None:
        if recipe.adapter_id != self.adapter_id or recipe.parameters != {"value": 1}:
            raise ValueError("DEVELOPMENT_ONLY_RECIPE_INVALID")

    def validate_search_domain(self, domain: AlphaModelSearchDomainEnvelope) -> None:
        if domain.adapter_id != self.adapter_id or domain.constraints != {"values": [1]}:
            raise ValueError("DEVELOPMENT_ONLY_DOMAIN_INVALID")

    def validate_recipe_for_domain(
        self,
        *,
        recipe: AlphaModelRecipeEnvelope,
        domain: AlphaModelSearchDomainEnvelope,
    ) -> None:
        self.validate_recipe(recipe)
        self.validate_search_domain(domain)


def test_model_specs_are_discriminated_bounded_and_content_addressed() -> None:
    ridge = resolve_model_spec({"family": "ridge", "alpha": 0.5})
    lasso = resolve_model_spec({"family": "lasso", "alpha_max_multiplier": 0.05})
    elastic = resolve_model_spec(
        {"family": "elastic_net", "alpha_max_multiplier": 0.2, "l1_ratio": 0.4}
    )

    assert ridge == resolve_model_spec(RidgeModelSpec(alpha=0.5))
    assert lasso == resolve_model_spec(LassoModelSpec(alpha_max_multiplier=0.05))
    assert elastic == resolve_model_spec(
        ElasticNetModelSpec(alpha_max_multiplier=0.2, l1_ratio=0.4)
    )
    assert len({ridge.spec_hash, lasso.spec_hash, elastic.spec_hash}) == 3

    for invalid in (
        {"family": "ridge", "alpha": 0.01},
        {"family": "lasso", "alpha_max_multiplier": math.nan},
        {"family": "elastic_net", "alpha_max_multiplier": 0.2, "l1_ratio": 0.9},
        {"family": "ridge", "alpha": 1.0, "unknown": 1},
    ):
        with pytest.raises(ValidationError):
            resolve_model_spec(invalid)


def test_legacy_batch_codec_preserves_variable_arity_and_rejects_duplicates() -> None:
    specs = tuple(resolve_model_spec({"family": "ridge", "alpha": value}) for value in range(1, 7))
    first = seal_contract(
        AlphaExperimentBatch,
        {
            "program_hash": HASH,
            "batch_index": 1,
            "specs": specs,
            "predecessor_batch_hash": None,
        },
        "batch_hash",
    )
    assert len(first.specs) == 6

    legacy_short = seal_contract(
        AlphaExperimentBatch,
        {
            "program_hash": HASH,
            "batch_index": 1,
            "specs": specs[:5],
            "predecessor_batch_hash": None,
        },
        "batch_hash",
    )
    assert len(legacy_short.specs) == 5
    with pytest.raises(ValidationError, match="repeats"):
        seal_contract(
            AlphaExperimentBatch,
            {
                "program_hash": HASH,
                "batch_index": 2,
                "specs": (specs[0], specs[0]),
                "predecessor_batch_hash": first.batch_hash,
            },
            "batch_hash",
        )


def test_qualification_family_is_complete_independent_of_batch_readback_order() -> None:
    def batch_result(candidate_digit: str, batch_digit: str) -> AlphaExperimentBatchResult:
        candidate = seal_contract(
            AlphaExperimentCandidateResult,
            {
                "candidate_id": f"agent-linear-{candidate_digit * 16}",
                "spec_hash": candidate_digit * 64,
                "development_report_hash": "c" * 64,
                "inference_evidence_hash": "d" * 64,
                "numerical_result_hashes": ("e" * 64,),
                "estimator_state_hashes": ("f" * 64,),
                "pooled_oos_r2": 0.01,
                "mean_rank_ic": 0.02,
                "mean_gross_decile_spread": 0.03,
                "fold_coverage_mean": 1.0,
                "status": "DEVELOPMENT_EVALUATED",
                "failure_codes": (),
            },
            "result_hash",
        )
        return seal_contract(
            AlphaExperimentBatchResult,
            {
                "program_hash": "1" * 64,
                "batch_hash": batch_digit * 64,
                "development_surface_hash": "2" * 64,
                "development_surface_binding_hash": "3" * 64,
                "fold_surface_hashes": ("4" * 64,),
                "candidates": (candidate,),
                "fit_call_count": 1,
                "predict_call_count": 2,
                "metric_call_count": 1,
            },
            "result_hash",
        )

    initial = batch_result("a", "5")
    refinement = batch_result("b", "6")
    attempted = (initial.candidates[0].candidate_id, refinement.candidates[0].candidate_id)

    indexed = _index_attempted_family(
        attempted_ids=attempted,
        batch_results=(refinement, initial),
    )

    assert set(indexed) == set(attempted)
    with pytest.raises(ValueError, match="qualification_attempted_family_incomplete"):
        _index_attempted_family(attempted_ids=attempted, batch_results=(initial, initial))


def test_later_holm_family_preserves_prior_current_stability_rejection() -> None:
    spec = resolve_model_spec(RidgeModelSpec(alpha=3.0))
    record = seal_contract(
        AlphaCandidateRecord,
        {
            "spec": spec,
            "candidate_id": f"agent-linear-{spec.spec_hash[:16]}",
            "batch_hash": "1" * 64,
            "status": AlphaCandidateStatus.CURRENT_STABILITY_REJECTED,
            "development_result_ref": "2" * 64,
            "development_candidate_hash": "3" * 64,
            "current_state_hash": "4" * 64,
            "stability_assessment_hash": "5" * 64,
            "current_score_child_hash": None,
            "holm_adjusted_p_value": 0.01,
            "stability_failed_check_ids": ("TRAINING_MSE",),
            "current_score_mean": 0.0,
            "current_score_std": 1.0,
            "current_score_coverage": 1.0,
            "failure_codes": ("alpha_research.current_refit_insufficient",),
        },
        "record_hash",
    )

    carried = _carry_forward_prior_current_rejection(
        record,
        development_result_ref="6" * 64,
        development_candidate_hash="7" * 64,
        holm_adjusted_p_value=0.02,
    )

    assert carried is not None
    assert carried.status is AlphaCandidateStatus.CURRENT_STABILITY_REJECTED
    assert carried.current_state_hash == record.current_state_hash
    assert carried.stability_assessment_hash == record.stability_assessment_hash
    assert carried.stability_failed_check_ids == ("TRAINING_MSE",)
    assert carried.development_result_ref == "6" * 64
    assert carried.holm_adjusted_p_value == 0.02


def test_renominated_prior_rejection_reuses_current_children(monkeypatch, tmp_path: Path) -> None:
    program = _program()
    spec = _research_recipes((3.0,))[0]
    candidate_id = f"alpha-candidate-{spec.spec_hash[:16]}"
    record = seal_contract(
        AlphaCandidateRecord,
        {
            "spec": spec,
            "candidate_id": candidate_id,
            "batch_hash": "1" * 64,
            "status": AlphaCandidateStatus.CURRENT_STABILITY_REJECTED,
            "development_result_ref": "2" * 64,
            "development_candidate_hash": "3" * 64,
            "current_state_hash": "4" * 64,
            "stability_assessment_hash": "5" * 64,
            "holm_adjusted_p_value": 0.01,
            "stability_failed_check_ids": ("TRAINING_MSE",),
            "current_score_mean": 0.0,
            "current_score_std": 1.0,
            "current_score_coverage": 1.0,
            "failure_codes": ("alpha_research.current_refit_insufficient",),
        },
        "record_hash",
    )
    registry = seal_contract(
        AlphaCandidateRegistrySnapshot,
        {
            "program_hash": program.program_hash,
            "revision": 1,
            "predecessor_registry_hash": "6" * 64,
            "candidates": (record,),
            "latest_hypothesis_family_hash": "7" * 64,
        },
        "registry_hash",
    )
    candidate_result = SimpleNamespace(
        result_hash="8" * 64,
        development_report_hash="b" * 64,
        numerical_result_hashes=(),
        estimator_state_hashes=(),
    )
    candidate_viability = SimpleNamespace(
        candidate_id=candidate_id,
        admitted=True,
        candidate_hash="9" * 64,
        holm_adjusted_p_value=0.02,
        failure_codes=(),
    )
    viability = SimpleNamespace(
        candidates=(candidate_viability,),
        admissible_candidate_ids=(candidate_id,),
        assessment_hash="a" * 64,
    )

    class Store:
        def load_candidate_report(self, _value):
            return SimpleNamespace()

        def publish_viability(self, _value):
            return None

    monkeypatch.setattr(
        qualification_module,
        "_index_attempted_family",
        lambda **_kwargs: {candidate_id: (SimpleNamespace(), candidate_result)},
    )
    monkeypatch.setattr(
        qualification_module,
        "_candidate_rows",
        lambda **_kwargs: (SimpleNamespace(), SimpleNamespace()),
    )
    monkeypatch.setattr(
        qualification_module,
        "_historical_mean_benchmark",
        lambda **_kwargs: (SimpleNamespace(), SimpleNamespace()),
    )
    monkeypatch.setattr(
        qualification_module,
        "assess_alpha_model_viability",
        lambda **_kwargs: viability,
    )
    monkeypatch.setattr(
        qualification_module,
        "fit_and_assess_current_regularized_linear",
        lambda **_kwargs: pytest.fail("verified current rejection was refit"),
    )

    result = qualify_alpha_model_candidates(
        program=program,
        registry=registry,
        batch_results=(SimpleNamespace(result_hash="c" * 64),),
        nominated_candidate_ids=(candidate_id,),
        fold_plan=SimpleNamespace(fold_count=5),
        train_session_count=756,
        store=Store(),
        model_catalog=build_installed_alpha_model_catalog(),
        model_mandate=_model_mandate(),
    )

    assert result.fit_call_count == 0
    assert result.predict_call_count == 0
    assert result.registry.candidates[0].status is AlphaCandidateStatus.CURRENT_STABILITY_REJECTED
    assert result.registry.candidates[0].current_state_hash == record.current_state_hash


def test_registry_and_goal_progress_enforce_target_and_budget() -> None:
    spec = resolve_model_spec({"family": "ridge", "alpha": 0.5})
    record = seal_contract(
        AlphaCandidateRecord,
        {
            "spec": spec,
            "candidate_id": f"agent-linear-{spec.spec_hash[:16]}",
            "batch_hash": HASH,
            "status": AlphaCandidateStatus.PROPOSED,
            "failure_codes": (),
        },
        "record_hash",
    )
    registry = seal_contract(
        AlphaCandidateRegistrySnapshot,
        {
            "program_hash": HASH,
            "revision": 0,
            "predecessor_registry_hash": None,
            "candidates": (record,),
            "exhausted_legacy_specs": (),
            "latest_hypothesis_family_hash": None,
        },
        "registry_hash",
    )
    criteria = _criteria()
    progress = seal_contract(
        AlphaGoalProgress,
        {
            "criteria_hash": criteria.criteria_hash,
            "registry_hash": registry.registry_hash,
            "qualification_hash": None,
            "proposed_count": 1,
            "development_admissible_count": 0,
            "development_rejected_count": 0,
            "current_qualified_count": 0,
            "current_stability_rejected_count": 0,
            "selected_count": 0,
            "completed_batch_count": 1,
            "remaining_batch_count": 1,
            "remaining_spec_count": 3,
            "unresolved_failure_codes": (),
            "disposition": AlphaGoalDisposition.CONTINUE,
        },
        "progress_hash",
    )
    assert progress.disposition is AlphaGoalDisposition.CONTINUE

    with pytest.raises(ValidationError, match="cannot be satisfied"):
        seal_contract(
            AlphaGoalProgress,
            {
                **progress.model_dump(mode="python", exclude={"progress_hash", "disposition"}),
                "disposition": AlphaGoalDisposition.SATISFIED,
            },
            "progress_hash",
        )


def _program(model_mandate: AlphaResearchModelMandate | None = None) -> AlphaResearchProgram:
    mandate = model_mandate or _model_mandate()
    return seal_contract(
        AlphaResearchProgram,
        {
            "foundation_hash": "1" * 64,
            "logical_panel_hash": "2" * 64,
            "logical_semantic_index_hash": "3" * 64,
            "causal_outcome_snapshot_hash": "4" * 64,
            "pm_plan_hash": "5" * 64,
            "ordered_listing_ids_hash": "6" * 64,
            "ordered_factor_ids_hash": "7" * 64,
            "split_policy_hash": "8" * 64,
            "metric_policy_hash": "9" * 64,
            "package_identity_hash": "0" * 64,
            "stability_policy_hash": "a" * 64,
            "goal_criteria_hash": build_goal_criteria(mandate).criteria_hash,
            "user_authorization_hash": "c" * 64,
            "research_goal_hash": "d" * 64,
            "model_mandate_hash": mandate.mandate_hash,
            "model_catalog_hash": mandate.catalog_binding.catalog_hash,
        },
        "program_hash",
    )


def test_development_only_adapter_is_recorded_as_scientific_non_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _DevelopmentOnlyAdapter()
    catalog = AlphaModelCatalog((adapter,))  # type: ignore[arg-type]
    domain = AlphaModelSearchDomainEnvelope.create(
        adapter_id=adapter.adapter_id,
        recipe_schema_id=adapter.recipe_schema_id,
        search_domain_schema_id=adapter.search_domain_schema_id,
        constraints={"values": [1]},
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
            parameters={"value": 1},
        ),
        catalog=catalog,
        admitted_target_lanes=None,
    )
    program = _program(mandate)
    candidate_id = f"alpha-candidate-{recipe.spec_hash[:16]}"
    record = seal_contract(
        AlphaCandidateRecord,
        {
            "spec": recipe,
            "candidate_id": candidate_id,
            "batch_hash": "1" * 64,
            "status": AlphaCandidateStatus.DEVELOPMENT_EVALUATED,
            "development_result_ref": "2" * 64,
            "failure_codes": (),
        },
        "record_hash",
    )
    registry = seal_contract(
        AlphaCandidateRegistrySnapshot,
        {
            "program_hash": program.program_hash,
            "revision": 0,
            "predecessor_registry_hash": None,
            "candidates": (record,),
            "latest_hypothesis_family_hash": None,
        },
        "registry_hash",
    )
    candidate_result = SimpleNamespace(
        result_hash="3" * 64,
        development_report_hash="4" * 64,
        numerical_result_hashes=(),
        estimator_state_hashes=(),
    )
    candidate_viability = SimpleNamespace(
        candidate_id=candidate_id,
        admitted=True,
        candidate_hash="5" * 64,
        holm_adjusted_p_value=0.01,
        failure_codes=(),
    )
    viability = SimpleNamespace(
        candidates=(candidate_viability,),
        admissible_candidate_ids=(candidate_id,),
        assessment_hash="6" * 64,
    )

    class Store:
        def load_candidate_report(self, _value):
            return SimpleNamespace()

        def publish_viability(self, _value):
            return None

    monkeypatch.setattr(
        qualification_module,
        "_index_attempted_family",
        lambda **_kwargs: {candidate_id: (SimpleNamespace(), candidate_result)},
    )
    monkeypatch.setattr(
        qualification_module,
        "_candidate_rows",
        lambda **_kwargs: (SimpleNamespace(), SimpleNamespace()),
    )
    monkeypatch.setattr(
        qualification_module,
        "_historical_mean_benchmark",
        lambda **_kwargs: (SimpleNamespace(), SimpleNamespace()),
    )
    monkeypatch.setattr(
        qualification_module,
        "assess_alpha_model_viability",
        lambda **_kwargs: viability,
    )
    monkeypatch.setattr(
        qualification_module,
        "fit_and_assess_current_regularized_linear",
        lambda **_kwargs: pytest.fail("development-only adapter entered current refit"),
    )

    result = qualify_alpha_model_candidates(
        program=program,
        registry=registry,
        batch_results=(SimpleNamespace(result_hash="7" * 64),),
        nominated_candidate_ids=(candidate_id,),
        fold_plan=SimpleNamespace(fold_count=1, target_policy=None),
        train_session_count=1,
        store=Store(),  # type: ignore[arg-type]
        model_catalog=catalog,
        model_mandate=mandate,
    )

    rejected = result.registry.candidates[0]
    assert result.fit_call_count == result.predict_call_count == 0
    assert rejected.status is AlphaCandidateStatus.CURRENT_STABILITY_REJECTED
    assert rejected.failure_codes == ("SCIENTIFIC_NON_ADMISSION",)
    assert rejected.current_state_hash is None


def _committer(store: AlphaGoalResearchArtifactStore) -> AlphaGoalResultCommitter:
    return AlphaGoalResultCommitter(store, model_mandate=_model_mandate())


def _proposal(alpha: float) -> AlphaModelRecipeProposal:
    return AlphaModelRecipeProposal(
        capability_handle="capability-1",
        parameters={"family": "ridge", "alpha": alpha},
    )


def _qualified_authority(
    program: AlphaResearchProgram,
) -> tuple[AlphaCandidateRegistrySnapshot, AlphaQualificationSnapshot]:
    records = []
    catalog = build_installed_alpha_model_catalog()
    mandate = build_current_alpha_research_model_mandate(catalog=catalog)
    for index, alpha in enumerate((0.2, 0.5, 2.0, 5.0, 20.0, 50.0)):
        spec = mandate.admit_proposal(
            proposal=_proposal(alpha),
            catalog=catalog,
            admitted_target_lanes=None,
        )
        records.append(
            seal_contract(
                AlphaCandidateRecord,
                {
                    "spec": spec,
                    "candidate_id": f"alpha-candidate-{spec.spec_hash[:16]}",
                    "batch_hash": "e" * 64,
                    "status": AlphaCandidateStatus.CURRENT_QUALIFIED,
                    "development_result_ref": canonical_hash((index, "development")),
                    "development_candidate_hash": canonical_hash((index, "viability")),
                    "current_state_hash": canonical_hash((index, "state")),
                    "stability_assessment_hash": canonical_hash((index, "stability")),
                    "current_score_child_hash": canonical_hash((index, "scores")),
                    "failure_codes": (),
                },
                "record_hash",
            )
        )
    family_hash = canonical_hash(tuple(value.candidate_id for value in records))
    registry = seal_contract(
        AlphaCandidateRegistrySnapshot,
        {
            "program_hash": program.program_hash,
            "revision": 1,
            "predecessor_registry_hash": "f" * 64,
            "candidates": tuple(records),
            "exhausted_legacy_specs": (),
            "historical_current_stability_rejections": (),
            "latest_hypothesis_family_hash": family_hash,
        },
        "registry_hash",
    )
    ids = tuple(value.candidate_id for value in records)
    qualification = seal_contract(
        AlphaQualificationSnapshot,
        {
            "program_hash": program.program_hash,
            "registry_hash": registry.registry_hash,
            "hypothesis_family_hash": family_hash,
            "attempted_candidate_ids": ids,
            "nominated_candidate_ids": ids,
            "development_admissible_ids": ids,
            "current_qualified_ids": ids,
            "current_stability_rejected_ids": (),
            "viability_assessment_hash": canonical_hash("viability"),
        },
        "qualification_hash",
    )
    return registry, qualification


def test_terminal_selection_rotates_registry_and_marker_children(tmp_path: Path) -> None:
    program = _program()
    criteria = _criteria()
    registry, qualification = _qualified_authority(program)
    selected = tuple(value.candidate_id for value in registry.candidates[:3])
    progress = resolve_goal_progress(
        criteria=criteria,
        registry=registry,
        qualification=qualification,
        completed_batch_count=1,
    )
    store = AlphaGoalResearchArtifactStore(tmp_path)
    for value, publish in (
        (program, store.publish_program),
        (registry, store.publish_registry),
        (qualification, store.publish_qualification),
        (progress, store.publish_goal_progress),
    ):
        publish(value)  # type: ignore[arg-type]

    committed = _committer(store).commit(
        program=program,
        selected_candidate_ids=selected,
        registry=registry,
        qualification=qualification,
        progress=progress,
        limitations=("Risk remains not admitted.",),
        science_policy_hash=program.stability_policy_hash,
        completed_batch_count=1,
        criteria=criteria,
    )

    final_registry = store.load_registry(committed.marker.registry_hash)
    statuses = {value.candidate_id: value.status for value in final_registry.candidates}
    assert all(
        statuses[value] is AlphaCandidateStatus.SELECTED_FOR_CANDIDATE_SET for value in selected
    )
    assert all(
        value.status is AlphaCandidateStatus.NOT_SELECTED for value in final_registry.candidates[3:]
    )
    assert committed.candidate_set is not None
    assert committed.candidate_set.ordered_candidate_ids == selected
    assert committed.marker.registry_hash == final_registry.registry_hash
    assert store.load_active_marker() == committed.marker


def test_new_program_cas_supersedes_active_pointer_without_deleting_prior_artifacts(
    tmp_path: Path,
) -> None:
    first_program = _program()
    second_program = seal_contract(
        AlphaResearchProgram,
        {
            **first_program.model_dump(mode="python", exclude={"program_hash"}),
            "stability_policy_hash": "e" * 64,
        },
        "program_hash",
    )
    criteria = _criteria()
    store = AlphaGoalResearchArtifactStore(tmp_path)
    committer = _committer(store)
    committed = []
    for program in (first_program, second_program):
        registry, qualification = _qualified_authority(program)
        selected = tuple(value.candidate_id for value in registry.candidates[:3])
        progress = resolve_goal_progress(
            criteria=criteria,
            registry=registry,
            qualification=qualification,
            completed_batch_count=1,
        )
        for value, publish in (
            (program, store.publish_program),
            (registry, store.publish_registry),
            (qualification, store.publish_qualification),
            (progress, store.publish_goal_progress),
        ):
            publish(value)  # type: ignore[arg-type]
        assert committer.load_exact_replay(program.program_hash) is None
        committed.append(
            committer.commit(
                program=program,
                selected_candidate_ids=selected,
                registry=registry,
                qualification=qualification,
                progress=progress,
                limitations=("Risk remains not admitted.",),
                science_policy_hash=program.stability_policy_hash,
                completed_batch_count=1,
                criteria=criteria,
            )
        )

    assert store.load_active_marker() == committed[1].marker
    assert (
        store.load_projection_for_marker(committed[0].marker.marker_hash) == committed[0].projection
    )
    assert committer.load_exact_replay(first_program.program_hash) is None
    assert committer.load_exact_replay(second_program.program_hash) == committed[1]


def test_terminal_pointer_rolls_back_when_authoritative_readback_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    program = _program()
    criteria = _criteria()
    registry, qualification = _qualified_authority(program)
    selected = tuple(value.candidate_id for value in registry.candidates[:3])
    progress = resolve_goal_progress(
        criteria=criteria,
        registry=registry,
        qualification=qualification,
        completed_batch_count=1,
    )
    store = AlphaGoalResearchArtifactStore(tmp_path)
    store.publish_program(program)
    store.publish_registry(registry)
    store.publish_qualification(qualification)
    store.publish_goal_progress(progress)

    def fail_terminal_readback():
        raise ValueError("simulated terminal readback failure")

    monkeypatch.setattr(store, "load_active_marker", fail_terminal_readback)
    with pytest.raises(ValueError, match="simulated terminal readback failure"):
        _committer(store).commit(
            program=program,
            selected_candidate_ids=selected,
            registry=registry,
            qualification=qualification,
            progress=progress,
            limitations=("Risk remains not admitted.",),
            science_policy_hash=program.stability_policy_hash,
            completed_batch_count=1,
            criteria=criteria,
        )
    assert not (store.root / "active.json").exists()


def test_terminal_publication_reuses_projection_after_pre_marker_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    program = _program()
    criteria = _criteria()
    registry, qualification = _qualified_authority(program)
    selected = tuple(value.candidate_id for value in registry.candidates[:3])
    progress = resolve_goal_progress(
        criteria=criteria,
        registry=registry,
        qualification=qualification,
        completed_batch_count=1,
    )
    store = AlphaGoalResearchArtifactStore(tmp_path)
    store.publish_program(program)
    store.publish_registry(registry)
    store.publish_qualification(qualification)
    store.publish_goal_progress(progress)
    real_publish_marker = store.publish_terminal_marker

    def interrupt_before_marker(*_args: object, **_kwargs: object) -> str:
        raise RuntimeError("simulated crash after projection publication")

    monkeypatch.setattr(store, "publish_terminal_marker", interrupt_before_marker)
    first_clock = datetime(2026, 8, 9, 12, tzinfo=UTC)
    with pytest.raises(RuntimeError, match="simulated crash"):
        _committer(store).commit(
            program=program,
            selected_candidate_ids=selected,
            registry=registry,
            qualification=qualification,
            progress=progress,
            limitations=("Risk remains not admitted.",),
            science_policy_hash=program.stability_policy_hash,
            completed_batch_count=1,
            criteria=criteria,
            clock=first_clock,
        )
    projection_pointers = tuple((store.root / "projection-by-marker").glob("*.json"))
    assert len(projection_pointers) == 1
    marker_hash = projection_pointers[0].stem
    stranded_projection = store.load_projection_for_marker(marker_hash)
    assert stranded_projection.updated_at == first_clock
    assert store.find_active_marker() is None

    monkeypatch.setattr(store, "publish_terminal_marker", real_publish_marker)
    recovered = _committer(store).commit(
        program=program,
        selected_candidate_ids=selected,
        registry=registry,
        qualification=qualification,
        progress=progress,
        limitations=("Risk remains not admitted.",),
        science_policy_hash=program.stability_policy_hash,
        completed_batch_count=1,
        criteria=criteria,
        clock=first_clock + timedelta(hours=1),
    )

    assert recovered.marker.marker_hash == marker_hash
    assert recovered.projection == stranded_projection
    assert store.load_active_marker() == recovered.marker


# the qualification Task (GR3)

OPENED = datetime(2026, 9, 28, 9, tzinfo=UTC)
TASK = "00000000-0000-0000-0000-00000000000{}"


def _study(
    task_id: str,
    minutes: int,
    recipe: AlphaResearchModelRecipe,
    *,
    lifecycle: str = "SUCCEEDED",
    question: str = "a" * 64,
) -> AlphaStudy:
    def sealed() -> tuple[str, str, str]:
        return f"out/{task_id}", "1" * 64, "2" * 64

    return AlphaStudy(
        task_id=task_id,
        admitted_at=OPENED + timedelta(minutes=minutes),
        lifecycle=lifecycle,
        question_hash=question,
        recipe=recipe,
        sealed=sealed if lifecycle == "SUCCEEDED" else None,
    )


def test_the_family_is_every_study_on_the_question_from_the_goals_opening() -> None:
    """requirement (GR3): every study on the question admitted from the goal's opening on
    is a member, attributed or not; a cancelled one is named without evidence, a repeated
    recipe counts once, an unsettled one refuses, and an empty family refuses."""

    first, second, third = _research_recipes((1.0, 3.0, 10.0))
    family = alpha_family(
        studies=(
            _study(TASK.format(1), -5, first),  # before the opening
            _study(TASK.format(2), 1, first),
            _study(TASK.format(3), 2, second, lifecycle="CANCELLED"),
            _study(TASK.format(4), 3, third, question="b" * 64),  # another question
            _study(TASK.format(5), 4, first),  # the same recipe again
            _study(TASK.format(6), 5, second),
        ),
        question_hash="a" * 64,
        goal_id=str(UUID(int=7)),
        opened_at=OPENED,
    )
    assert [m.task_id for m in family.members] == [TASK.format(2), TASK.format(6)]
    assert family.unfinished_task_ids == (TASK.format(3),)
    assert family.candidate_ids == tuple(candidate_id_for_spec(r) for r in (first, second))
    with pytest.raises(AuthoringError, match="family_member_unsettled"):
        alpha_family(
            studies=(_study(TASK.format(8), 1, first, lifecycle="RUNNING"),),
            question_hash="a" * 64,
            goal_id=str(UUID(int=7)),
            opened_at=OPENED,
        )
    with pytest.raises(AuthoringError, match="family_empty"):
        alpha_family(studies=(), question_hash="a" * 64, goal_id=str(UUID(int=7)), opened_at=OPENED)


def test_the_question_sets_the_model_aside() -> None:
    """Regression: a ridge and a LightGBM study of one target on one Foundation ask one
    question. Their Programs differ in the mandate and, since narrowed each to its model,
    in the catalog binding too; both enter the Holm family."""

    ridge = AlphaDevelopmentProgram.model_construct(
        foundation_hash="1" * 64,
        model_mandate_hash="2" * 64,
        model_catalog_hash="7" * 64,
        program_hash="3" * 64,
    )
    lightgbm = AlphaDevelopmentProgram.model_construct(
        foundation_hash="1" * 64,
        model_mandate_hash="4" * 64,
        model_catalog_hash="8" * 64,
        program_hash="5" * 64,
    )
    other = AlphaDevelopmentProgram.model_construct(
        foundation_hash="6" * 64,
        model_mandate_hash="2" * 64,
        model_catalog_hash="7" * 64,
        program_hash="3" * 64,
    )
    assert alpha_question_hash(ridge) == alpha_question_hash(lightgbm)
    assert alpha_question_hash(ridge) != alpha_question_hash(other)


def test_the_family_registry_keeps_each_study_standing() -> None:
    first, second = _research_recipes((1.0, 3.0))

    def result(recipe: AlphaResearchModelRecipe, batch: str, status: str) -> Any:
        candidate = seal_contract(
            AlphaExperimentCandidateResult,
            {
                "candidate_id": candidate_id_for_spec(recipe),
                "spec_hash": recipe.spec_hash,
                "development_report_hash": "c" * 64,
                "inference_evidence_hash": "d" * 64,
                "numerical_result_hashes": ("e" * 64,),
                "estimator_state_hashes": ("f" * 64,),
                "status": status,
                "failure_codes": () if status == "DEVELOPMENT_EVALUATED" else ("x.failed",),
            },
            "result_hash",
        )
        return seal_contract(
            AlphaExperimentBatchResult,
            {
                "program_hash": "1" * 64,
                "batch_hash": batch * 64,
                "development_surface_hash": "2" * 64,
                "development_surface_binding_hash": "3" * 64,
                "fold_surface_hashes": ("4" * 64,),
                "candidates": (candidate,),
                "fit_call_count": 1,
                "predict_call_count": 1,
                "metric_call_count": 1,
            },
            "result_hash",
        )

    results = (
        result(first, "5", "DEVELOPMENT_EVALUATED"),
        result(second, "6", "DEVELOPMENT_FAILED"),
    )
    recipes = {
        candidate_id_for_spec(first): (first, "5" * 64),
        candidate_id_for_spec(second): (second, "6" * 64),
    }
    registry = family_registry(program_hash="7" * 64, recipes=recipes, batch_results=results)
    assert [value.status for value in registry.candidates] == [
        AlphaCandidateStatus.DEVELOPMENT_EVALUATED,
        AlphaCandidateStatus.DEVELOPMENT_REJECTED,
    ]
    assert registry.program_hash == "7" * 64
    with pytest.raises(ValueError, match="recipe_mismatch"):
        family_registry(
            program_hash="7" * 64,
            recipes={**recipes, candidate_id_for_spec(first): (first, "9" * 64)},
            batch_results=results,
        )
    with pytest.raises(ValueError, match="repeats_a_candidate"):
        family_registry(program_hash="7" * 64, recipes=recipes, batch_results=results * 2)


def test_a_qualification_binds_only_the_models_its_family_ran() -> None:
    """Requirement: a model installed beside the family's moves no qualification,
    and a family whose model the mandate no longer holds is refused before any work."""
    first, second = _research_recipes((1.0, 3.0))
    family = alpha_family(
        studies=(_study(TASK.format(1), 1, first), _study(TASK.format(2), 2, second)),
        question_hash="a" * 64,
        goal_id=str(UUID(int=7)),
        opened_at=OPENED,
    )

    def unprepared() -> Any:
        pytest.fail("PLAN prepares no arrays")

    def method(mandate: AlphaResearchModelMandate) -> AlphaFamilyQualification:
        return AlphaFamilyQualification(
            family=family,
            prepare_question=unprepared,
            workspace=Path("."),
            model_mandate=mandate,
            model_catalog=build_installed_alpha_model_catalog(),
        )

    installed = _model_mandate()
    narrowed = installed.admitting(first.search_domain_hash)
    assert method(installed).model_mandate == narrowed
    assert method(installed).method_binding_hash() == method(narrowed).method_binding_hash()
    elsewhere = next(
        value
        for value in installed.ordered_search_domains
        if value.search_domain_hash != first.search_domain_hash
    )
    with pytest.raises(AuthoringError, match="qualification_model_not_installed"):
        method(installed.admitting(elsewhere.search_domain_hash))


def test_a_nomination_outside_the_family_names_what_may_be_nominated() -> None:
    """requirement (OP12): the refusal names the field's rule, and `expected` gives the
    family's candidates, so an agent corrects the declaration without guessing."""
    first, second = _research_recipes((1.0, 3.0))
    family = alpha_family(
        studies=(_study(TASK.format(1), 1, first), _study(TASK.format(2), 2, second)),
        question_hash="a" * 64,
        goal_id=str(UUID(int=7)),
        opened_at=OPENED,
    )

    def unprepared() -> Any:
        pytest.fail("PLAN prepares no arrays")

    method = AlphaFamilyQualification(
        family=family,
        prepare_question=unprepared,
        workspace=Path("."),
        model_mandate=_model_mandate(),
        model_catalog=build_installed_alpha_model_catalog(),
    )
    section = {
        "methodology_id": METHOD,
        "goal_id": str(UUID(int=7)),
        "question_task_id": TASK.format(1),
        "nominated_candidate_ids": ["alpha-candidate-" + "0" * 16],
    }
    with pytest.raises(
        AuthoringError, match="qualification_nomination_invalid:not_in_family"
    ) as caught:
        method.validate_declaration({"alpha": section})
    assert caught.value.expected == {"qualification.nominated_candidate_ids": family.candidate_ids}


def test_a_declaration_with_a_study_s_fields_is_refused_by_the_section_it_breaks() -> None:
    """regression (OP12): agent copied a development study's fields into a
    qualification's `alpha` section and met a bare code; the refusal names the section and
    `expected` lists its four fields, and a declaration of another method names the method."""
    first, second = _research_recipes((1.0, 3.0))
    family = alpha_family(
        studies=(_study(TASK.format(1), 1, first), _study(TASK.format(2), 2, second)),
        question_hash="a" * 64,
        goal_id=str(UUID(int=7)),
        opened_at=OPENED,
    )

    def unprepared() -> Any:
        pytest.fail("PLAN prepares no arrays")

    method = AlphaFamilyQualification(
        family=family,
        prepare_question=unprepared,
        workspace=Path("."),
        model_mandate=_model_mandate(),
        model_catalog=build_installed_alpha_model_catalog(),
    )
    section = {
        "methodology_id": METHOD,
        "goal_id": str(UUID(int=7)),
        "question_task_id": TASK.format(1),
        "nominated_candidate_ids": list(family.candidate_ids[:1]),
        "model_parameters": {"family": "ridge", "alpha": 1.0},
    }
    with pytest.raises(AuthoringError, match=r"qualification_declaration_invalid:alpha$") as caught:
        method.validate_declaration({"alpha": section})
    assert caught.value.expected == {
        "alpha": ["goal_id", "methodology_id", "nominated_candidate_ids", "question_task_id"]
    }
    with pytest.raises(
        AuthoringError, match=r"qualification_declaration_invalid:methodology_id$"
    ) as caught:
        method.validate_declaration({"alpha": {**section, "methodology_id": "OTHER"}})
    assert caught.value.expected == {"methodology_id": METHOD}
    explained = refused("alpha_research.qualification_declaration_invalid:alpha")
    assert explained["next_action"] == "EDIT_DECLARATION_AND_PLAN"
    assert "`alpha` section" in explained["detail"]


def test_a_qualification_names_its_goal_and_nominates_from_its_family() -> None:
    first, second = _research_recipes((1.0, 3.0))
    family = alpha_family(
        studies=(_study(TASK.format(1), 1, first), _study(TASK.format(2), 2, second)),
        question_hash="a" * 64,
        goal_id=str(UUID(int=7)),
        opened_at=OPENED,
    )

    def unprepared() -> Any:
        pytest.fail("PLAN prepares no arrays")

    method = AlphaFamilyQualification(
        family=family,
        prepare_question=unprepared,
        workspace=Path("."),
        model_mandate=_model_mandate(),
        model_catalog=build_installed_alpha_model_catalog(),
    )
    section = {
        "methodology_id": METHOD,
        "goal_id": str(UUID(int=7)),
        "question_task_id": TASK.format(1),
        "nominated_candidate_ids": [candidate_id_for_spec(first)],
    }
    envelope: Any = SimpleNamespace(budget=SimpleNamespace(maximum_numerical_calls=5))
    authority: Any = SimpleNamespace(authority_hash="8" * 64)
    compiled = method.compile_desk_program(
        envelope=envelope, document={"alpha": section}, authority=authority
    )
    assert compiled.method_binding_hash == method.method_binding_hash()
    assert planned_calls(family, (candidate_id_for_spec(first),)) == 5
    for change, code in (
        ({"goal_id": str(UUID(int=8))}, "family_goal_mismatch"),
        ({"nominated_candidate_ids": ["alpha-candidate-" + "0" * 16]}, "nomination_invalid"),
        ({"nominated_candidate_ids": []}, "nomination_invalid"),
        ({"extra": 1}, "declaration_invalid"),
    ):
        with pytest.raises(AuthoringError, match=code):
            method.validate_declaration({"alpha": {**section, **change}})
    tight: Any = SimpleNamespace(budget=SimpleNamespace(maximum_numerical_calls=4))
    with pytest.raises(AuthoringError, match="budget_exceeded"):
        method.compile_desk_program(
            envelope=tight, document={"alpha": section}, authority=authority
        )
