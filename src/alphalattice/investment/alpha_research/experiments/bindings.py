"""Numerical-child and parent-request identity owners for Alpha Research."""

from __future__ import annotations

from typing import TYPE_CHECKING

from alphalattice.foundation.research_foundation.contracts import ResearchFoundationBinding
from alphalattice.kernel.shared_kernel.identity import canonical_hash

from ..evaluation.contracts import AlphaMetricPolicy
from ..inputs.folds import AlphaFoldArrayPlan, AlphaFoldArrays, build_alpha_fold_commitment
from ..inputs.training import build_alpha_training_input_binding
from .contracts import (
    AlphaDevelopmentProgram,
    AlphaDevelopmentProgramAuthority,
    AlphaDevelopmentSplitPolicy,
    AlphaExperimentCard,
    AlphaExperimentRequest,
    AlphaPackageIdentity,
    seal_contract,
)
from .development_contracts import (
    AlphaCandidateExecutionBinding,
    AlphaDevelopmentSurfaceBinding,
    seal_current_contract,
)
from .mandate import AlphaModelCapabilityAuthority
from .policies import (
    load_alpha_metric_policy,
    load_alpha_package_identity,
    load_alpha_split_policy,
)

if TYPE_CHECKING:
    # A type only: the inventory reaches this module as a value, so the PM's effort module is
    # no entry of Alpha's closures (ID8).
    from alphalattice.investment.portfolio_management.mandates.research_effort import (
        ResearchDeskCandidateInventory,
    )


class AlphaExperimentAdmissionError(ValueError):
    """An experiment request is stale, malformed, or outside Host authority."""


class AlphaDevelopmentProgramAuthorityError(ValueError):
    """Standalone development authority differs from its materialized plan."""

    failure_class = "AUTHORITY_FAILURE"
    fit_call_count = 0


def build_alpha_development_program(
    *,
    fold_plan: AlphaFoldArrayPlan,
    model_mandate: AlphaModelCapabilityAuthority,
    metric_policy: AlphaMetricPolicy | None = None,
    package_identity: AlphaPackageIdentity | None = None,
) -> AlphaDevelopmentProgram:
    """Seal one standalone target/fold program from Host-owned identities."""

    foundation = fold_plan.foundation
    if foundation.logical_panel_hash is None or foundation.logical_semantic_index_hash is None:
        raise AlphaDevelopmentProgramAuthorityError("ALPHA_DEVELOPMENT_LOGICAL_FOUNDATION_REQUIRED")
    # The plan's own base axis, which is the foundation's whole axis unless a
    # development run declared a subset. Read from the plan rather than the
    # foundation so the Program's feature identity is the axis the arrays were
    # actually built over -- the two disagreeing is precisely the defect this
    # closes, and it disagreed silently because nothing compared them.
    ordered_feature_ids = (
        *fold_plan.base_feature_ids,
        *fold_plan.additional_factor_ids,
    )
    if ordered_feature_ids != tuple(dict.fromkeys(ordered_feature_ids)):
        raise AlphaDevelopmentProgramAuthorityError("ALPHA_DEVELOPMENT_FEATURE_AXIS_INVALID")
    commitments = tuple(
        build_alpha_fold_commitment(window).commitment_hash
        for window in fold_plan.split_plan.windows
    )
    split_policy = _seal_development_split_policy(fold_plan)
    return seal_contract(
        AlphaDevelopmentProgram,
        {
            "kind": "AlphaDevelopmentProgram",
            "foundation_hash": foundation.foundation_hash,
            "logical_panel_hash": foundation.logical_panel_hash,
            "logical_semantic_index_hash": foundation.logical_semantic_index_hash,
            "causal_outcome_snapshot_hash": foundation.execution_outcome.snapshot_hash,
            "ordered_listing_ids_hash": canonical_hash(fold_plan.ordered_listing_ids),
            "ordered_feature_ids_hash": canonical_hash(ordered_feature_ids),
            "split_policy": split_policy,
            "split_hash": fold_plan.split_plan.split_hash,
            "fold_commitment_hashes": commitments,
            "target_policy": fold_plan.target_policy,
            "target_method_hash": (
                None
                if fold_plan.target_method is None
                else fold_plan.target_method.target_method_hash
            ),
            "metric_policy": metric_policy or load_alpha_metric_policy(),
            "package_identity": package_identity or load_alpha_package_identity(),
            "model_mandate_hash": model_mandate.mandate_hash,
            "model_catalog_hash": model_mandate.catalog_binding.catalog_hash,
        },
        "program_hash",
    )


def _seal_development_split_policy(
    fold_plan: AlphaFoldArrayPlan,
) -> AlphaDevelopmentSplitPolicy:
    spec = fold_plan.split_plan.spec
    return seal_contract(
        AlphaDevelopmentSplitPolicy,
        {
            "mode": spec.mode.value,
            "train_sessions": spec.train_sessions,
            "purge_sessions": spec.purge_sessions,
            "validation_sessions": spec.validation_sessions,
            "step_sessions": spec.step_sessions,
            "embargo_sessions": spec.embargo_sessions,
            "sealed_holdout_sessions": spec.holdout_sessions,
            "minimum_folds": spec.minimum_folds,
            "expected_complete_folds": len(fold_plan.split_plan.windows),
        },
        "policy_hash",
    )


def assert_alpha_development_program_authority(
    *,
    program: AlphaDevelopmentProgramAuthority,
    fold_plan: AlphaFoldArrayPlan,
    model_mandate: AlphaModelCapabilityAuthority,
) -> None:
    """Fail before fit when a standalone program no longer matches its plan."""

    if not isinstance(program, AlphaDevelopmentProgram):
        return
    foundation = fold_plan.foundation
    ordered_feature_ids = (
        *fold_plan.base_feature_ids,
        *fold_plan.additional_factor_ids,
    )
    commitments = tuple(
        build_alpha_fold_commitment(window).commitment_hash
        for window in fold_plan.split_plan.windows
    )
    actual = (
        foundation.foundation_hash,
        foundation.logical_panel_hash,
        foundation.logical_semantic_index_hash,
        foundation.execution_outcome.snapshot_hash,
        canonical_hash(fold_plan.ordered_listing_ids),
        canonical_hash(ordered_feature_ids),
        _seal_development_split_policy(fold_plan),
        fold_plan.split_plan.split_hash,
        commitments,
        fold_plan.target_policy,
        None if fold_plan.target_method is None else fold_plan.target_method.target_method_hash,
        load_alpha_metric_policy(),
        load_alpha_package_identity(),
        model_mandate.mandate_hash,
        model_mandate.catalog_binding.catalog_hash,
    )
    expected = (
        program.foundation_hash,
        program.logical_panel_hash,
        program.logical_semantic_index_hash,
        program.causal_outcome_snapshot_hash,
        program.ordered_listing_ids_hash,
        program.ordered_feature_ids_hash,
        program.split_policy,
        program.split_hash,
        program.fold_commitment_hashes,
        program.target_policy,
        program.target_method_hash,
        program.metric_policy,
        program.package_identity,
        program.model_mandate_hash,
        program.model_catalog_hash,
    )
    if actual != expected:
        raise AlphaDevelopmentProgramAuthorityError("ALPHA_DEVELOPMENT_PROGRAM_AUTHORITY_MISMATCH")


def assert_alpha_development_fold_materialization(
    *,
    program: AlphaDevelopmentProgramAuthority,
    fold_plan: AlphaFoldArrayPlan,
    fold: AlphaFoldArrays,
) -> None:
    """Verify transformed target and raw economic values before adapter fit."""

    if not isinstance(program, AlphaDevelopmentProgram):
        return
    binding = fold.training_input_binding
    if binding is None:
        raise AlphaDevelopmentProgramAuthorityError("ALPHA_DEVELOPMENT_TRAINING_BINDING_MISSING")
    training_economic = (
        fold.training_targets
        if fold.training_economic_returns is None
        else fold.training_economic_returns
    )
    validation_economic = (
        fold.validation_targets
        if fold.validation_economic_returns is None
        else fold.validation_economic_returns
    )
    expected = build_alpha_training_input_binding(
        scope="DEVELOPMENT_FOLD",
        foundation_hash=fold_plan.foundation.foundation_hash,
        feature_panel_snapshot_hash=fold_plan.foundation.feature_panel_snapshot_hash,
        causal_outcome_snapshot_hash=(fold_plan.foundation.execution_outcome.snapshot_hash),
        target_policy=fold_plan.target_policy,
        ordered_feature_ids=fold.ordered_factor_ids,
        feature_context_hash=fold_plan.feature_context_hash,
        training_row_sessions=fold.training_row_sessions,
        training_row_listing_ids=fold.training_listing_ids,
        prediction_row_sessions=fold.validation_row_sessions,
        prediction_row_listing_ids=fold.validation_listing_ids,
        training_features=fold.training_features,
        training_targets=fold.training_targets,
        training_mask=fold.training_model_mask,
        prediction_features=fold.validation_features,
        prediction_targets=fold.validation_targets,
        prediction_mask=fold.validation_feature_complete,
        training_feature_row_hashes=fold.training_feature_row_hashes,
        training_outcome_row_hashes=fold.training_outcome_row_hashes,
        prediction_feature_row_hashes=fold.validation_feature_row_hashes,
        prediction_outcome_row_hashes=fold.validation_outcome_row_hashes,
        training_economic_returns=training_economic,
        prediction_economic_returns=validation_economic,
        training_cutoff=fold.commitment.train_last,
        outcome_maturity_session=fold.commitment.train_last,
        prediction_anchor=fold.commitment.validation_first,
        fold_commitment_hash=fold.commitment.commitment_hash,
    )
    actual_identity = (
        binding.training_row_axis_hash,
        binding.prediction_row_axis_hash,
        binding.training_target_values_hash,
        binding.prediction_target_values_hash,
        binding.target_surface_hash,
        binding.economic_return_surface_hash,
    )
    expected_identity = (
        expected.training_row_axis_hash,
        expected.prediction_row_axis_hash,
        expected.training_target_values_hash,
        expected.prediction_target_values_hash,
        expected.target_surface_hash,
        expected.economic_return_surface_hash,
    )
    if actual_identity != expected_identity:
        raise AlphaDevelopmentProgramAuthorityError("ALPHA_DEVELOPMENT_EVALUATION_SURFACE_MISMATCH")


def build_alpha_experiment_request(
    *,
    foundation: ResearchFoundationBinding,
    inventory: ResearchDeskCandidateInventory,
    cards: tuple[AlphaExperimentCard, ...],
    listing_set_hash: str,
    ordered_listing_ids: tuple[str, ...],
) -> AlphaExperimentRequest:
    """Seal the reusable numerical identity without Agent or publication state."""

    if ordered_listing_ids != tuple(
        sorted(set(ordered_listing_ids))
    ) or listing_set_hash != canonical_hash(ordered_listing_ids):
        raise AlphaExperimentAdmissionError("ALPHA_LISTING_AUTHORITY_INVALID")
    try:
        cards = tuple(
            AlphaExperimentCard.model_validate(value.model_dump(mode="python")) for value in cards
        )
    except Exception as error:
        raise AlphaExperimentAdmissionError("ALPHA_CARD_CONTRACT_INVALID") from error
    if tuple(value.candidate_id for value in cards) != tuple(
        value.candidate_id for value in inventory.alpha_models
    ):
        raise AlphaExperimentAdmissionError("ALPHA_CARD_ORDER_MISMATCH")
    if tuple(value.source_card_hash for value in cards) != tuple(
        value.card_hash for value in inventory.alpha_models
    ):
        raise AlphaExperimentAdmissionError("ALPHA_SOURCE_CARD_HASH_MISMATCH")
    if foundation.logical_panel_hash is None or foundation.logical_semantic_index_hash is None:
        raise AlphaExperimentAdmissionError("ALPHA_LOGICAL_FOUNDATION_IDENTITY_REQUIRED")
    return seal_contract(
        AlphaExperimentRequest,
        {
            "kind": "AlphaExperimentRequest",
            "foundation_hash": foundation.foundation_hash,
            "feature_panel_snapshot_hash": foundation.feature_panel_snapshot_hash,
            "logical_panel_hash": foundation.logical_panel_hash,
            "logical_semantic_index_hash": foundation.logical_semantic_index_hash,
            "causal_outcome_snapshot_hash": foundation.execution_outcome.snapshot_hash,
            "listing_set_hash": listing_set_hash,
            "ordered_listing_ids": ordered_listing_ids,
            "ordered_factor_ids": foundation.ordered_factor_ids,
            "inventory_hash": inventory.inventory_hash,
            "candidate_ids": tuple(value.candidate_id for value in cards),
            "candidate_card_hashes": tuple(value.card_hash for value in cards),
            "split_policy": load_alpha_split_policy(),
            "metric_policy": load_alpha_metric_policy(),
            "package_identity": load_alpha_package_identity(),
        },
        "request_hash",
    )


def build_development_surface_binding(
    *,
    request: AlphaExperimentRequest,
    split_hash: str,
    fold_commitment_hashes: tuple[str, ...],
) -> AlphaDevelopmentSurfaceBinding:
    return seal_current_contract(
        AlphaDevelopmentSurfaceBinding,
        {
            "kind": "AlphaDevelopmentSurfaceBinding",
            "foundation_hash": request.foundation_hash,
            "logical_panel_hash": request.logical_panel_hash,
            "logical_semantic_index_hash": request.logical_semantic_index_hash,
            "causal_outcome_snapshot_hash": request.causal_outcome_snapshot_hash,
            "listing_set_hash": request.listing_set_hash,
            "ordered_listing_ids_hash": canonical_hash(request.ordered_listing_ids),
            "ordered_factor_ids_hash": canonical_hash(request.ordered_factor_ids),
            "split_hash": split_hash,
            "fold_commitment_hashes": fold_commitment_hashes,
        },
        "binding_hash",
    )


def build_candidate_execution_binding(
    *,
    surface: AlphaDevelopmentSurfaceBinding,
    card: AlphaExperimentCard,
    fold_commitment_hash: str,
) -> AlphaCandidateExecutionBinding:
    return seal_current_contract(
        AlphaCandidateExecutionBinding,
        {
            "kind": "AlphaCandidateExecutionBinding",
            "development_surface_binding_hash": surface.binding_hash,
            "candidate_id": card.candidate_id,
            "candidate_card_hash": card.card_hash,
            "fold_commitment_hash": fold_commitment_hash,
            "algorithm_package_identity": card.package_identity,
        },
        "execution_binding_hash",
    )


__all__ = [
    "AlphaDevelopmentProgramAuthorityError",
    "AlphaExperimentAdmissionError",
    "assert_alpha_development_fold_materialization",
    "assert_alpha_development_program_authority",
    "build_alpha_development_program",
    "build_alpha_experiment_request",
    "build_candidate_execution_binding",
    "build_development_surface_binding",
]
