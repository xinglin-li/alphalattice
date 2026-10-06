"""Authority tests for the Host catalog and Alpha Research model Mandate."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from alphalattice.capabilities.alpha_modeling.adapters.regularized_linear import (
    build_regularized_linear_search_domain,
)
from alphalattice.capabilities.alpha_modeling.catalog import (
    AlphaModelCatalog,
    AlphaModelCatalogBinding,
    build_installed_alpha_model_catalog,
)
from alphalattice.capabilities.alpha_modeling.contracts import (
    AlphaModelFitProtocol,
    AlphaModelNumericalBinding,
    AlphaModelRecipeEnvelope,
    AlphaModelSearchDomainEnvelope,
)
from alphalattice.investment.alpha_research.candidates.contracts import (
    AlphaCandidateRegistrySnapshot,
    AlphaExperimentBatch,
    AlphaGoalProgress,
    AlphaGoalResearchMarker,
    AlphaGoalResearchSafeProjection,
    AlphaResearchProgram,
    AlphaResearchScientificStop,
    CurrentAlphaCandidateSetSnapshot,
    resolve_model_spec,
    seal_contract,
)
from alphalattice.investment.alpha_research.experiments.mandate import (
    AlphaModelRecipeProposal,
    AlphaResearchModelMandate,
    assert_research_recipe_target_authority,
    build_current_alpha_research_model_mandate,
)
from alphalattice.investment.alpha_research.inputs.training import AlphaTrainingInputAuthorityError
from alphalattice.investment.alpha_research.targets.execution_outcome import (
    AlphaTargetLane,
    build_alpha_target_policy,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash


@dataclass(frozen=True, slots=True)
class _SyntheticAdapter:
    adapter_id: str = "synthetic_threshold"
    recipe_schema_id: str = "alpha-model.synthetic-threshold"
    search_domain_schema_id: str = "alpha-model.synthetic-threshold.search-domain"

    def describe_numerical_binding(self) -> AlphaModelNumericalBinding:
        return AlphaModelNumericalBinding.create(
            adapter_id=self.adapter_id,
            estimator_content_format_id="alpha-model.synthetic-threshold.content",
            implementation_owners=("case-study.synthetic-threshold",),
            deterministic_policy={"threshold_rule": "fixed"},
            required_runtime_capabilities=(),
        )

    def fit_protocols(self, recipe: AlphaModelRecipeEnvelope) -> tuple[AlphaModelFitProtocol, ...]:
        del recipe
        return ("DIRECT_FIT",)

    def validate_recipe(self, recipe: AlphaModelRecipeEnvelope) -> int:
        if recipe.adapter_id != self.adapter_id or recipe.recipe_schema_id != self.recipe_schema_id:
            raise ValueError("SYNTHETIC_RECIPE_ROUTE_INVALID")
        threshold = recipe.parameters.get("threshold")
        if not isinstance(threshold, int):
            raise ValueError("SYNTHETIC_RECIPE_INVALID")
        return threshold

    def validate_search_domain(self, domain: AlphaModelSearchDomainEnvelope) -> tuple[int, int]:
        if (
            domain.adapter_id != self.adapter_id
            or domain.recipe_schema_id != self.recipe_schema_id
            or domain.search_domain_schema_id != self.search_domain_schema_id
        ):
            raise ValueError("SYNTHETIC_SEARCH_DOMAIN_ROUTE_INVALID")
        minimum = domain.constraints.get("threshold_min")
        maximum = domain.constraints.get("threshold_max")
        if not isinstance(minimum, int) or not isinstance(maximum, int) or minimum > maximum:
            raise ValueError("SYNTHETIC_SEARCH_DOMAIN_INVALID")
        return minimum, maximum

    def validate_recipe_for_domain(
        self,
        *,
        recipe: AlphaModelRecipeEnvelope,
        domain: AlphaModelSearchDomainEnvelope,
    ) -> int:
        threshold = self.validate_recipe(recipe)
        minimum, maximum = self.validate_search_domain(domain)
        if not minimum <= threshold <= maximum:
            raise ValueError("SYNTHETIC_RECIPE_OUTSIDE_SEARCH_DOMAIN")
        return threshold


def _synthetic_domain() -> AlphaModelSearchDomainEnvelope:
    adapter = _SyntheticAdapter()
    return AlphaModelSearchDomainEnvelope.create(
        adapter_id=adapter.adapter_id,
        recipe_schema_id=adapter.recipe_schema_id,
        search_domain_schema_id=adapter.search_domain_schema_id,
        constraints={"threshold_min": 2, "threshold_max": 8},
    )


def _multi_capability_mandate() -> tuple[AlphaModelCatalog, AlphaResearchModelMandate]:
    catalog = AlphaModelCatalog(
        (
            build_installed_alpha_model_catalog().resolve_search_domain(
                build_regularized_linear_search_domain()
            ),
            _SyntheticAdapter(),
        )
    )
    mandate = AlphaResearchModelMandate.create(
        catalog_binding=catalog.binding,
        ordered_search_domains=(build_regularized_linear_search_domain(), _synthetic_domain()),
        target_current_qualified_candidates=4,
        initial_batch_size=5,
        refinement_batch_max_size=4,
        max_batch_count=3,
        max_unique_new_recipes=12,
    )
    return catalog, mandate


def test_a_mandate_narrows_to_the_domains_a_family_chose() -> None:
    """requirement (V299, V118): narrowed to a family's domains, a mandate keeps them in its
    own order with their models alone, so a model installed beside them moves nothing; one
    domain narrows as a development Program's does, and a domain it lacks is refused."""

    catalog, mandate = _multi_capability_mandate()
    linear, synthetic = mandate.ordered_search_domains
    assert mandate.admitting_domains((linear.search_domain_hash,)) == mandate.admitting(
        linear.search_domain_hash
    )
    both = mandate.admitting_domains((synthetic.search_domain_hash, linear.search_domain_hash))
    assert both.ordered_search_domains == (linear, synthetic)
    assert both.catalog_binding == catalog.binding
    alone = AlphaResearchModelMandate.create(
        catalog_binding=AlphaModelCatalog(
            (build_installed_alpha_model_catalog().resolve_search_domain(linear),)
        ).binding,
        ordered_search_domains=(linear,),
        target_current_qualified_candidates=4,
        initial_batch_size=5,
        refinement_batch_max_size=4,
        max_batch_count=3,
        max_unique_new_recipes=12,
    )
    assert mandate.admitting_domains((linear.search_domain_hash,)) == alone
    with pytest.raises(ValueError, match="ALPHA_MODEL_SEARCH_DOMAIN_NOT_MANDATED"):
        alone.admitting_domains((synthetic.search_domain_hash,))


def test_catalog_mandate_and_domain_identities_round_trip_and_reject_tamper() -> None:
    catalog, mandate = _multi_capability_mandate()

    assert AlphaResearchModelMandate.model_validate(mandate.model_dump(mode="json")) == mandate
    assert catalog.binding.catalog_hash == canonical_hash(
        catalog.binding.model_dump(mode="json", exclude={"catalog_hash"})
    )

    tampered = mandate.model_dump(mode="json")
    tampered["ordered_search_domains"] = list(reversed(tampered["ordered_search_domains"]))
    with pytest.raises(ValidationError, match="ALPHA_RESEARCH_MODEL_MANDATE_INVALID"):
        AlphaResearchModelMandate.model_validate(tampered)

    domain = _synthetic_domain().model_dump(mode="json")
    domain["search_domain_schema_id"] = "alpha-model.wrong-domain"
    domain["search_domain_hash"] = canonical_hash(
        {key: value for key, value in domain.items() if key != "search_domain_hash"}
    )
    with pytest.raises(ValueError, match="SEARCH_DOMAIN_SCHEMA_NOT_INSTALLED"):
        catalog.resolve_search_domain(AlphaModelSearchDomainEnvelope.model_validate(domain))

    duplicate_route = AlphaModelSearchDomainEnvelope.create(
        adapter_id="synthetic_threshold",
        recipe_schema_id="alpha-model.synthetic-threshold",
        search_domain_schema_id="alpha-model.synthetic-threshold.search-domain",
        constraints={"threshold_min": 3, "threshold_max": 7},
    )
    with pytest.raises(ValidationError, match="ALPHA_RESEARCH_MODEL_MANDATE_INVALID"):
        AlphaResearchModelMandate.create(
            catalog_binding=catalog.binding,
            ordered_search_domains=(_synthetic_domain(), duplicate_route),
            target_current_qualified_candidates=2,
            initial_batch_size=2,
            refinement_batch_max_size=1,
            max_batch_count=2,
            max_unique_new_recipes=3,
        )

    tampered_binding = catalog.binding.model_dump(mode="json")
    tampered_binding["ordered_capabilities"][0]["numerical_binding_hash"] = "0" * 64
    tampered_binding["catalog_hash"] = canonical_hash(
        {key: value for key, value in tampered_binding.items() if key != "catalog_hash"}
    )
    tampered_mandate = AlphaResearchModelMandate.create(
        catalog_binding=AlphaModelCatalogBinding.model_validate(tampered_binding),
        ordered_search_domains=mandate.ordered_search_domains,
        target_current_qualified_candidates=4,
        initial_batch_size=5,
        refinement_batch_max_size=4,
        max_batch_count=3,
        max_unique_new_recipes=12,
    )
    with pytest.raises(ValueError, match="ALPHA_MODEL_CATALOG_MANDATE_MISMATCH"):
        tampered_mandate.admit_proposal(
            proposal=AlphaModelRecipeProposal(
                capability_handle="capability-1",
                parameters={"family": "ridge", "alpha": 1.0},
            ),
            catalog=catalog,
            admitted_target_lanes=None,
        )


def test_pre_numerical_binding_catalog_and_mandate_hashes_remain_readable() -> None:
    domain = build_regularized_linear_search_domain()
    legacy_capability = {
        "adapter_id": "regularized_linear",
        "recipe_schema_id": "alpha-model.regularized-linear",
        "search_domain_schema_id": "alpha-model.regularized-linear.search-domain",
    }
    legacy_catalog_identity = {"ordered_capabilities": (legacy_capability,)}
    catalog_payload = {
        **legacy_catalog_identity,
        "catalog_hash": canonical_hash(legacy_catalog_identity),
    }
    catalog_binding = AlphaModelCatalogBinding.model_validate(catalog_payload)
    assert catalog_binding.ordered_capabilities[0].numerical_binding_hash is None
    mandate_values = {
        "kind": "AlphaResearchModelMandate",
        "catalog_binding": catalog_payload,
        "ordered_search_domains": (domain.model_dump(mode="json"),),
        "target_current_qualified_candidates": 3,
        "initial_batch_size": 6,
        "refinement_batch_max_size": 3,
        "max_batch_count": 2,
        "max_unique_new_recipes": 9,
    }
    mandate_hash = canonical_hash(mandate_values)

    mandate = AlphaResearchModelMandate.model_validate(
        {**mandate_values, "mandate_hash": mandate_hash}
    )

    assert mandate.catalog_binding.catalog_hash == catalog_payload["catalog_hash"]
    assert mandate.catalog_binding.ordered_capabilities[0].numerical_binding_hash is None
    assert mandate.mandate_hash == mandate_hash


def test_second_adapter_is_admitted_without_a_goal_research_model_union() -> None:
    catalog, mandate = _multi_capability_mandate()
    admitted = mandate.admit_proposal(
        proposal=AlphaModelRecipeProposal(
            capability_handle="capability-2",
            parameters={"threshold": 5},
        ),
        catalog=catalog,
        admitted_target_lanes=None,
    )

    assert admitted.recipe.adapter_id == "synthetic_threshold"
    assert admitted.search_domain_hash == _synthetic_domain().search_domain_hash
    assert admitted.research_recipe_hash == canonical_hash(
        admitted.model_dump(
            mode="json",
            exclude={"research_recipe_hash", "target_method_binding_hash"},
        )
    )

    with pytest.raises(ValueError, match="SYNTHETIC_RECIPE_OUTSIDE_SEARCH_DOMAIN"):
        mandate.admit_proposal(
            proposal=AlphaModelRecipeProposal(
                capability_handle="capability-2",
                parameters={"threshold": 99},
            ),
            catalog=catalog,
            admitted_target_lanes=None,
        )


def test_a_model_installed_beside_the_admitted_ones_moves_no_mandate_and_refuses_nothing() -> None:
    """requirement (V118, LAWS.md ID3): a development Program binds the one model its mandate
    admits for it, and a goal's mandate is compared with the installed catalog on the models
    it admits; installing another model moves neither and refuses no proposal."""

    from alphalattice.investment.alpha_research.experiments.mandate import (
        admitted_capabilities_installed,
    )

    linear = build_regularized_linear_search_domain()
    installed = build_installed_alpha_model_catalog()
    alone = AlphaResearchModelMandate.create(
        catalog_binding=installed.binding,
        ordered_search_domains=(linear,),
        target_current_qualified_candidates=1,
        initial_batch_size=1,
        refinement_batch_max_size=1,
        max_batch_count=1,
        max_unique_new_recipes=1,
    )
    grown_catalog = AlphaModelCatalog((*installed.adapters, _SyntheticAdapter()))
    grown = AlphaResearchModelMandate.create(
        catalog_binding=grown_catalog.binding,
        ordered_search_domains=(linear, _synthetic_domain()),
        target_current_qualified_candidates=1,
        initial_batch_size=1,
        refinement_batch_max_size=1,
        max_batch_count=1,
        max_unique_new_recipes=1,
    )
    assert grown_catalog.binding.catalog_hash != installed.binding.catalog_hash
    assert (
        grown.admitting(linear.search_domain_hash).mandate_hash
        == alone.admitting(linear.search_domain_hash).mandate_hash
    )

    assert admitted_capabilities_installed(alone, grown_catalog)
    admitted = alone.admit_proposal(
        proposal=AlphaModelRecipeProposal(
            capability_handle="capability-1", parameters={"family": "ridge", "alpha": 1.0}
        ),
        catalog=grown_catalog,
        admitted_target_lanes=None,
    )
    assert admitted.search_domain_hash == linear.search_domain_hash
    # The admitted model itself is still compared: removed, it refuses.
    assert not admitted_capabilities_installed(grown, installed)
    with pytest.raises(ValueError, match="ALPHA_MODEL_SEARCH_DOMAIN_NOT_MANDATED"):
        alone.admitting(_synthetic_domain().search_domain_hash)


def test_catalog_and_mandate_fail_closed_before_adapter_fit() -> None:
    installed = build_installed_alpha_model_catalog()
    synthetic = _synthetic_domain()
    with pytest.raises(ValueError, match="ALPHA_MODEL_ADAPTER_NOT_INSTALLED"):
        installed.resolve_search_domain(synthetic)

    with pytest.raises(ValueError, match="ALPHA_MODEL_CATALOG_INVALID"):
        AlphaModelCatalog((_SyntheticAdapter(), _SyntheticAdapter()))

    mandate = build_current_alpha_research_model_mandate(catalog=installed)
    with pytest.raises(ValueError, match="ALPHA_MODEL_RECIPE_OUTSIDE_SEARCH_DOMAIN"):
        mandate.admit_proposal(
            proposal=AlphaModelRecipeProposal(
                capability_handle="capability-1",
                parameters={"family": "ridge", "alpha": 1000.0},
            ),
            catalog=installed,
            admitted_target_lanes=None,
        )

    # The second installed capability is the Dynamic Panel LightGBM adapter:
    # linear parameters routed to it are refused by its own schema, and a
    # handle past the inventory resolves to nothing.
    assert mandate.resolve_capability_handle("capability-2").adapter_id == "dynamic_panel_lightgbm"
    with pytest.raises(ValueError, match="ALPHA_DYNAMIC_PANEL_LIGHTGBM_RECIPE_PARAMETERS_INVALID"):
        mandate.admit_proposal(
            proposal=AlphaModelRecipeProposal(
                capability_handle="capability-2",
                parameters={"family": "ridge", "alpha": 1.0},
            ),
            catalog=installed,
            admitted_target_lanes=None,
        )
    with pytest.raises(ValueError, match="ALPHA_MODEL_CAPABILITY_HANDLE_INVALID"):
        mandate.admit_proposal(
            proposal=AlphaModelRecipeProposal(
                capability_handle="capability-3",
                parameters={"family": "ridge", "alpha": 1.0},
            ),
            catalog=installed,
            admitted_target_lanes=None,
        )
    with pytest.raises(ValueError, match="ALPHA_MODEL_CAPABILITY_HANDLE_INVALID"):
        mandate.resolve_capability_handle("capability-0")

    with pytest.raises(ValueError, match="ALPHA_MODEL_TARGET_LANE_NOT_ADMITTED"):
        mandate.admit_proposal(
            proposal=AlphaModelRecipeProposal(
                capability_handle="capability-1",
                parameters={"family": "ridge", "alpha": 1.0},
                target_lane=AlphaTargetLane.SECTOR_RESIDUAL_RANK_GAUSS,
            ),
            catalog=installed,
            admitted_target_lanes=None,
        )

    multi_catalog, multi_mandate = _multi_capability_mandate()
    with pytest.raises(ValueError, match="ALPHA_MODEL_CATALOG_MANDATE_MISMATCH"):
        multi_mandate.admit_proposal(
            proposal=AlphaModelRecipeProposal(
                capability_handle="capability-1",
                parameters={"family": "ridge", "alpha": 1.0},
            ),
            catalog=installed,
            admitted_target_lanes=None,
        )
    assert multi_catalog.binding != installed.binding


def test_candidate_and_projection_axes_follow_mandate_policy_not_three() -> None:
    _catalog, mandate = _multi_capability_mandate()
    candidate_ids = tuple(f"alpha-candidate-{value:016x}" for value in range(4))
    snapshot = seal_contract(
        CurrentAlphaCandidateSetSnapshot,
        {
            "program_hash": "1" * 64,
            "foundation_hash": "2" * 64,
            "registry_hash": "3" * 64,
            "qualification_hash": "4" * 64,
            "model_mandate_hash": mandate.mandate_hash,
            "target_candidate_count": mandate.target_current_qualified_candidates,
            "ordered_candidate_ids": candidate_ids,
            "current_score_child_hashes": tuple(str(value) * 64 for value in range(5, 9)),
            "estimator_state_hashes": tuple(str(value) * 64 for value in range(1, 5)),
            "stability_assessment_hashes": tuple(str(value) * 64 for value in range(5, 9)),
            "limitations": ("Risk remains not admitted.",),
        },
        "snapshot_hash",
    )
    projection = seal_contract(
        AlphaGoalResearchSafeProjection,
        {
            "program_hash": snapshot.program_hash,
            "marker_hash": "9" * 64,
            "goal_target": mandate.target_current_qualified_candidates,
            "current_qualified_count": 4,
            "attempted_spec_count": 7,
            "completed_batch_count": 2,
            "status": "CURRENT_ALPHA_CANDIDATE_SET_READY",
            "next_action": "READY_TO_DESIGN_MULTI_MODEL_DOWNSTREAM_CONSUMPTION",
            "selected_candidate_ids": candidate_ids,
            "limitations": snapshot.limitations,
            "updated_at": datetime(2026, 8, 13, 12, tzinfo=UTC),
        },
        "projection_hash",
    )

    assert len(snapshot.ordered_candidate_ids) == 4
    assert len(projection.selected_candidate_ids) == projection.goal_target == 4


def test_recipe_target_policy_must_belong_to_the_program_before_fit() -> None:
    policy = build_alpha_target_policy(
        lane=AlphaTargetLane.SECTOR_RESIDUAL_RANK_GAUSS,
        sector_revision="a" * 64,
    )
    with pytest.raises(AlphaTrainingInputAuthorityError) as raised:
        assert_research_recipe_target_authority(
            program_target_policy_hashes=("f" * 64,),
            recipe_target_lanes=(AlphaTargetLane.SECTOR_RESIDUAL_RANK_GAUSS,),
            default_target_policy=None,
            lane_target_policies={AlphaTargetLane.SECTOR_RESIDUAL_RANK_GAUSS: policy},
        )

    assert raised.value.experiment_disposition == "INVALID_EXPERIMENT"
    assert raised.value.failure_class == "TARGET_AUTHORITY_MISMATCH"
    assert raised.value.fit_call_count == 0
    assert raised.value.scientific_admission_effect == "NONE"


def test_frozen_goal_artifacts_keep_their_original_hashes() -> None:
    program_values = {
        "kind": "AlphaResearchProgram",
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
        "goal_criteria_hash": "b" * 64,
        "user_authorization_hash": "c" * 64,
        "research_goal_hash": "d" * 64,
    }
    program_hash = canonical_hash(program_values)
    program = AlphaResearchProgram.model_validate({**program_values, "program_hash": program_hash})
    assert program.program_hash == program_hash
    assert program.model_mandate_hash is None

    legacy_spec = resolve_model_spec({"family": "ridge", "alpha": 1.0})
    batch_values = {
        "kind": "AlphaExperimentBatch",
        "program_hash": program_hash,
        "batch_index": 1,
        "specs": (legacy_spec.model_dump(mode="json"),),
        "predecessor_batch_hash": None,
    }
    batch_hash = canonical_hash(batch_values)
    batch = AlphaExperimentBatch.model_validate({**batch_values, "batch_hash": batch_hash})
    assert batch.batch_hash == batch_hash

    registry_values = {
        "kind": "AlphaCandidateRegistrySnapshot",
        "program_hash": program_hash,
        "revision": 0,
        "predecessor_registry_hash": None,
        "candidates": (),
        "exhausted_legacy_specs": (legacy_spec.model_dump(mode="json"),),
        "historical_current_stability_rejections": (),
        "latest_hypothesis_family_hash": None,
    }
    registry_hash = canonical_hash(registry_values)
    registry = AlphaCandidateRegistrySnapshot.model_validate(
        {**registry_values, "registry_hash": registry_hash}
    )
    assert registry.registry_hash == registry_hash

    progress_values = {
        "kind": "AlphaGoalProgress",
        "criteria_hash": "b" * 64,
        "registry_hash": registry_hash,
        "qualification_hash": None,
        "proposed_count": 0,
        "development_admissible_count": 0,
        "development_rejected_count": 0,
        "current_qualified_count": 0,
        "current_stability_rejected_count": 0,
        "selected_count": 0,
        "completed_batch_count": 0,
        "remaining_batch_count": 0,
        "remaining_spec_count": 0,
        "unresolved_failure_codes": ("alpha_research.legacy_stop",),
        "disposition": "EXHAUSTED_STOP",
    }
    progress_hash = canonical_hash(progress_values)
    progress = AlphaGoalProgress.model_validate({**progress_values, "progress_hash": progress_hash})
    assert progress.progress_hash == progress_hash

    candidate_ids = tuple(f"agent-linear-{value:016x}" for value in range(3))
    candidate_set_values = {
        "kind": "CurrentAlphaCandidateSetSnapshot",
        "program_hash": program_hash,
        "foundation_hash": "1" * 64,
        "registry_hash": "2" * 64,
        "qualification_hash": "3" * 64,
        "ordered_candidate_ids": candidate_ids,
        "current_score_child_hashes": ("4" * 64, "5" * 64, "6" * 64),
        "estimator_state_hashes": ("7" * 64, "8" * 64, "9" * 64),
        "stability_assessment_hashes": ("a" * 64, "b" * 64, "c" * 64),
        "downstream_score_modes": (),
        "limitations": ("Frozen legacy readback.",),
    }
    snapshot_hash = canonical_hash(candidate_set_values)
    candidate_set = CurrentAlphaCandidateSetSnapshot.model_validate(
        {**candidate_set_values, "snapshot_hash": snapshot_hash}
    )
    assert candidate_set.snapshot_hash == snapshot_hash
    assert candidate_set.target_candidate_count is None

    stop_values = {
        "kind": "AlphaResearchScientificStop",
        "program_hash": program_hash,
        "registry_hash": "2" * 64,
        "qualification_hash": "3" * 64,
        "goal_progress_hash": "4" * 64,
        "attempted_spec_hashes": ("5" * 64, "6" * 64),
        "current_qualified_count": 1,
        "failure_codes": ("alpha_research.legacy_scientific_stop",),
        "next_research_actions": ("READBACK_ONLY",),
        "science_policy_hash": "7" * 64,
    }
    stop_hash = canonical_hash(stop_values)
    stop = AlphaResearchScientificStop.model_validate({**stop_values, "stop_hash": stop_hash})
    assert stop.stop_hash == stop_hash
    assert stop.attempted_recipe_hashes is None

    marker_values = {
        "kind": "AlphaGoalResearchMarker",
        "program_hash": program_hash,
        "goal_progress_hash": progress_hash,
        "registry_hash": registry_hash,
        "qualification_hash": "3" * 64,
        "board_hash": "4" * 64,
        "candidate_set_snapshot_hash": None,
        "scientific_stop_hash": stop_hash,
        "disposition": "NO_STABLE_CURRENT_ALPHA_MODEL",
        "risk_admitted": False,
    }
    marker_hash = canonical_hash(marker_values)
    marker = AlphaGoalResearchMarker.model_validate({**marker_values, "marker_hash": marker_hash})
    assert marker.marker_hash == marker_hash
