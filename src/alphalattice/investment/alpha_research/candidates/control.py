"""Deterministic registry and candidate-set control for an Alpha qualification."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from ..experiments.mandate import AlphaResearchModelMandate, AlphaResearchModelRecipe
from .contracts import (
    AlphaCandidateRecord,
    AlphaCandidateRegistrySnapshot,
    AlphaCandidateStatus,
    AlphaExperimentBatchResult,
    AlphaGoalDisposition,
    AlphaGoalProgress,
    AlphaHistoricalCandidateEvidence,
    AlphaModelResearchGoalCriteria,
    AlphaQualificationSnapshot,
    CurrentAlphaCandidateSetSnapshot,
    ResolvedRegularizedLinearSpec,
    seal_contract,
)


def build_goal_criteria(
    mandate: AlphaResearchModelMandate | None = None,
) -> AlphaModelResearchGoalCriteria:
    """Seal goal criteria from current model-mandate authority or fixed legacy defaults.

    Args:
        mandate: Optional current mandate supplying target, batch/recipe bounds and exclusive model
            authority.

    Returns:
        Sealed goal criteria; current criteria bind mandate_hash and leave allowed_families unset.
    """
    if mandate is None:
        return seal_contract(AlphaModelResearchGoalCriteria, {}, "criteria_hash")
    return seal_contract(
        AlphaModelResearchGoalCriteria,
        {
            "target_current_qualified_candidates": (mandate.target_current_qualified_candidates),
            "initial_batch_size": mandate.initial_batch_size,
            "refinement_batch_max_size": mandate.refinement_batch_max_size,
            "max_batch_count": mandate.max_batch_count,
            "max_unique_new_specs": mandate.max_unique_new_recipes,
            "allowed_families": None,
            "model_mandate_hash": mandate.mandate_hash,
        },
        "criteria_hash",
    )


def initialize_registry(
    *,
    program_hash: str,
    exhausted_legacy_specs: Iterable[ResolvedRegularizedLinearSpec],
    historical_current_stability_rejections: tuple[AlphaHistoricalCandidateEvidence, ...] = (),
) -> AlphaCandidateRegistrySnapshot:
    """Seal an empty initial candidate registry with deduplicated exhausted legacy recipes.

    Args:
        program_hash: Research-program identity owning this registry.
        exhausted_legacy_specs: Historical exhausted recipes deduplicated by spec_hash in encounter
            order.
        historical_current_stability_rejections: Historical rejection records retained with the
            initial registry.

    Returns:
        Revision-zero registry without predecessor or latest hypothesis family.
    """
    return seal_contract(
        AlphaCandidateRegistrySnapshot,
        {
            "program_hash": program_hash,
            "revision": 0,
            "predecessor_registry_hash": None,
            "candidates": (),
            "exhausted_legacy_specs": tuple(
                {value.spec_hash: value for value in exhausted_legacy_specs}.values()
            ),
            "historical_current_stability_rejections": (historical_current_stability_rejections),
            "latest_hypothesis_family_hash": None,
        },
        "registry_hash",
    )


def family_registry(
    *,
    program_hash: str,
    recipes: Mapping[str, tuple[AlphaResearchModelRecipe, str]],
    batch_results: tuple[AlphaExperimentBatchResult, ...],
) -> AlphaCandidateRegistrySnapshot:
    """The registry a qualification reads: every candidate its family attempted (GR3).

    Each candidate keeps the standing its own development study sealed for it.

    Args:
        program_hash: The qualification Program the registry belongs to.
        recipes: Each candidate's recipe and the batch its study sealed it in.
        batch_results: The studies' sealed results, one candidate each or more.

    Returns:
        The registry, its candidates in the results' order.
    """
    records: list[AlphaCandidateRecord] = []
    for batch_result in batch_results:
        for result in batch_result.candidates:
            try:
                spec, batch_hash = recipes[result.candidate_id]
            except KeyError as error:
                raise ValueError("alpha_research.qualification_family_recipe_missing") from error
            if spec.spec_hash != result.spec_hash or batch_hash != batch_result.batch_hash:
                raise ValueError("alpha_research.qualification_family_recipe_mismatch")
            records.append(
                seal_contract(
                    AlphaCandidateRecord,
                    {
                        "spec": spec,
                        "candidate_id": result.candidate_id,
                        "batch_hash": batch_hash,
                        "status": (
                            AlphaCandidateStatus.DEVELOPMENT_REJECTED
                            if result.status == "DEVELOPMENT_FAILED"
                            else AlphaCandidateStatus.DEVELOPMENT_EVALUATED
                        ),
                        "development_result_ref": result.result_hash,
                        "failure_codes": result.failure_codes,
                    },
                    "record_hash",
                )
            )
    if len({value.candidate_id for value in records}) != len(records):
        raise ValueError("alpha_research.qualification_family_repeats_a_candidate")
    return _replace_registry(
        initialize_registry(program_hash=program_hash, exhausted_legacy_specs=()),
        tuple(records),
        None,
    )


def update_registry_candidates(
    registry: AlphaCandidateRegistrySnapshot,
    candidates: tuple[AlphaCandidateRecord, ...],
    *,
    hypothesis_family_hash: str | None,
) -> AlphaCandidateRegistrySnapshot:
    """Replace candidate evidence/status while preserving the exact recipe and batch axis.

    Args:
        registry: Previous registry snapshot.
        candidates: Same ordered candidate identifiers, recipes and batch identities with revised
            evidence/status.
        hypothesis_family_hash: Latest tested family identity retained in the revised snapshot.

    Returns:
        Sealed successor registry with the unchanged ordered candidate axis.

    Raises:
        ValueError: Ordered candidate identifiers, recipes or batch hashes differ from the previous
            registry.
    """
    if tuple(value.candidate_id for value in candidates) != tuple(
        value.candidate_id for value in registry.candidates
    ):
        raise ValueError("alpha_research.candidate_registry_axis_mismatch")
    for previous, current in zip(registry.candidates, candidates, strict=True):
        if previous.spec != current.spec or previous.batch_hash != current.batch_hash:
            raise ValueError("alpha_research.candidate_registry_identity_changed")
    return _replace_registry(registry, candidates, hypothesis_family_hash)


def resolve_goal_progress(
    *,
    criteria: AlphaModelResearchGoalCriteria,
    registry: AlphaCandidateRegistrySnapshot,
    qualification: AlphaQualificationSnapshot | None,
    completed_batch_count: int,
    blocked_failure_codes: tuple[str, ...] = (),
) -> AlphaGoalProgress:
    """Derive bounded candidate progress and disposition from exact current registry evidence.

    Args:
        criteria: Target and batch/recipe limits admitted for this program.
        registry: Current candidate records whose status/failure counts are summarized.
        qualification: Optional assessment matching registry, candidate axis and latest hypothesis
            family.
        completed_batch_count: Number of completed batches subtracted from the admitted limit.
        blocked_failure_codes: Explicit blockers take precedence over satisfaction or remaining
            work.

    Returns:
        Sealed progress with counts, remaining bounds, deduplicated failures and goal disposition.

    Raises:
        ValueError: Supplied qualification is stale or the derived counts violate admitted bounds.
    """
    if qualification is not None:
        candidate_ids = tuple(value.candidate_id for value in registry.candidates)
        if (
            qualification.registry_hash != registry.registry_hash
            or qualification.attempted_candidate_ids != candidate_ids
            or qualification.hypothesis_family_hash != registry.latest_hypothesis_family_hash
        ):
            raise ValueError("alpha_research.goal_qualification_stale")
    counts = {status: 0 for status in AlphaCandidateStatus}
    failures: list[str] = []
    for candidate in registry.candidates:
        counts[candidate.status] += 1
        failures.extend(candidate.failure_codes)
    current_qualified = (
        counts[AlphaCandidateStatus.CURRENT_QUALIFIED]
        + counts[AlphaCandidateStatus.SELECTED_FOR_CANDIDATE_SET]
        + counts[AlphaCandidateStatus.NOT_SELECTED]
    )
    development_admissible = (
        counts[AlphaCandidateStatus.DEVELOPMENT_ADMISSIBLE]
        + counts[AlphaCandidateStatus.CURRENT_QUALIFIED]
        + counts[AlphaCandidateStatus.CURRENT_STABILITY_REJECTED]
        + counts[AlphaCandidateStatus.SELECTED_FOR_CANDIDATE_SET]
        + counts[AlphaCandidateStatus.NOT_SELECTED]
    )
    remaining_batches = criteria.max_batch_count - completed_batch_count
    remaining_specs = criteria.max_unique_new_specs - len(registry.candidates)
    if blocked_failure_codes:
        disposition = AlphaGoalDisposition.BLOCKED
    elif current_qualified >= criteria.target_current_qualified_candidates:
        disposition = AlphaGoalDisposition.SATISFIED
    elif remaining_batches > 0 and remaining_specs > 0:
        disposition = AlphaGoalDisposition.CONTINUE
    else:
        disposition = AlphaGoalDisposition.EXHAUSTED_STOP
    return seal_contract(
        AlphaGoalProgress,
        {
            "criteria_hash": criteria.criteria_hash,
            "registry_hash": registry.registry_hash,
            "qualification_hash": qualification.qualification_hash if qualification else None,
            "proposed_count": len(registry.candidates),
            "development_admissible_count": development_admissible,
            "development_rejected_count": counts[AlphaCandidateStatus.DEVELOPMENT_REJECTED],
            "current_qualified_count": current_qualified,
            "current_stability_rejected_count": counts[
                AlphaCandidateStatus.CURRENT_STABILITY_REJECTED
            ],
            "selected_count": counts[AlphaCandidateStatus.SELECTED_FOR_CANDIDATE_SET],
            "completed_batch_count": completed_batch_count,
            "remaining_batch_count": remaining_batches,
            "remaining_spec_count": remaining_specs,
            "target_candidate_count": criteria.target_current_qualified_candidates,
            "max_batch_count": criteria.max_batch_count,
            "unresolved_failure_codes": tuple(dict.fromkeys((*failures, *blocked_failure_codes))),
            "disposition": disposition,
        },
        "progress_hash",
    )


def build_candidate_set_snapshot(
    *,
    program_hash: str,
    foundation_hash: str,
    registry: AlphaCandidateRegistrySnapshot,
    qualification: AlphaQualificationSnapshot,
    ordered_candidate_ids: tuple[str, ...],
    model_mandate: AlphaResearchModelMandate,
    limitations: tuple[str, ...],
    sector_ema_manifest_hash: str | None = None,
) -> CurrentAlphaCandidateSetSnapshot:
    """Seal the exact mandate-sized qualified peer selection and complete numerical evidence.

    Args:
        program_hash: Research-program identity.
        foundation_hash: Admitted Foundation identity.
        registry: Registry containing the selected candidate records.
        qualification: Qualification bound to that exact registry.
        ordered_candidate_ids: Unique mandate-sized selection drawn from current-qualified
            identifiers.
        model_mandate: Authority deciding target count.
        limitations: Declared downstream limitations.
        sector_ema_manifest_hash: Optional Sector component identity admitting the paired score
            modes.

    Returns:
        Sealed ordered candidate set with score/state/stability peer evidence.

    Raises:
        ValueError: Qualification is stale, selection is not mandate-sized/current-qualified or
            selected evidence is incomplete.
    """
    by_id = {value.candidate_id: value for value in registry.candidates}
    if qualification.registry_hash != registry.registry_hash:
        raise ValueError("alpha_research.candidate_set_qualification_stale")
    target = model_mandate.target_current_qualified_candidates
    if (
        len(ordered_candidate_ids) != target
        or len(set(ordered_candidate_ids)) != target
        or not set(ordered_candidate_ids) <= set(qualification.current_qualified_ids)
    ):
        raise ValueError("alpha_research.candidate_set_not_current_qualified")
    selected = tuple(by_id[value] for value in ordered_candidate_ids)
    if any(
        value.current_score_child_hash is None
        or value.current_state_hash is None
        or value.stability_assessment_hash is None
        for value in selected
    ):
        raise ValueError("alpha_research.candidate_set_readback_incomplete")
    return seal_contract(
        CurrentAlphaCandidateSetSnapshot,
        {
            "program_hash": program_hash,
            "foundation_hash": foundation_hash,
            "registry_hash": registry.registry_hash,
            "qualification_hash": qualification.qualification_hash,
            "model_mandate_hash": model_mandate.mandate_hash,
            "target_candidate_count": target,
            "ordered_candidate_ids": ordered_candidate_ids,
            "current_score_child_hashes": tuple(
                value.current_score_child_hash for value in selected
            ),
            "estimator_state_hashes": tuple(value.current_state_hash for value in selected),
            "stability_assessment_hashes": tuple(
                value.stability_assessment_hash for value in selected
            ),
            "sector_ema_manifest_hash": sector_ema_manifest_hash,
            "downstream_score_modes": (
                ("STOCK_ONLY", "STOCK_PLUS_SECTOR_COMPONENT")
                if sector_ema_manifest_hash is not None
                else ()
            ),
            "limitations": limitations,
        },
        "snapshot_hash",
    )


def finalize_candidate_selection(
    *,
    registry: AlphaCandidateRegistrySnapshot,
    qualification: AlphaQualificationSnapshot,
    selected_candidate_ids: tuple[str, ...],
    model_mandate: AlphaResearchModelMandate,
) -> tuple[AlphaCandidateRegistrySnapshot, AlphaQualificationSnapshot]:
    """Mark qualified peer selection and rebind qualification to the successor registry.

    Args:
        registry: Current candidate registry.
        qualification: Exact qualification for that registry.
        selected_candidate_ids: Unique mandate-sized subset of current-qualified identifiers.
        model_mandate: Authority supplying the required peer target.

    Returns:
        Revised registry and qualification rebound to its exact identity, retaining correlation
        evidence.

    Raises:
        ValueError: Registry lineage, selected qualification membership or target count disagrees
            with authority.
    """
    target = model_mandate.target_current_qualified_candidates
    if (
        qualification.registry_hash != registry.registry_hash
        or not set(selected_candidate_ids) <= set(qualification.current_qualified_ids)
        or len(selected_candidate_ids) != target
        or len(set(selected_candidate_ids)) != target
    ):
        raise ValueError("alpha_research.candidate_selection_authority_mismatch")
    selected = set(selected_candidate_ids)
    records = tuple(
        seal_contract(
            AlphaCandidateRecord,
            {
                "spec": record.spec,
                "candidate_id": record.candidate_id,
                "batch_hash": record.batch_hash,
                "status": (
                    AlphaCandidateStatus.SELECTED_FOR_CANDIDATE_SET
                    if record.candidate_id in selected
                    else (
                        AlphaCandidateStatus.NOT_SELECTED
                        if record.candidate_id in qualification.current_qualified_ids
                        else record.status
                    )
                ),
                "development_result_ref": record.development_result_ref,
                "development_candidate_hash": record.development_candidate_hash,
                "current_state_hash": record.current_state_hash,
                "stability_assessment_hash": record.stability_assessment_hash,
                "current_score_child_hash": record.current_score_child_hash,
                "holm_adjusted_p_value": record.holm_adjusted_p_value,
                "oos_evidence_classification": record.oos_evidence_classification,
                "mean_rank_ic": record.mean_rank_ic,
                "mean_gross_decile_spread": record.mean_gross_decile_spread,
                "stability_failed_check_ids": record.stability_failed_check_ids,
                "current_score_mean": record.current_score_mean,
                "current_score_std": record.current_score_std,
                "current_score_coverage": record.current_score_coverage,
                "failure_codes": record.failure_codes,
            },
            "record_hash",
        )
        for record in registry.candidates
    )
    revised = update_registry_candidates(
        registry,
        records,
        hypothesis_family_hash=registry.latest_hypothesis_family_hash,
    )
    rebound = seal_contract(
        AlphaQualificationSnapshot,
        {
            **qualification.model_dump(
                mode="python",
                exclude={"qualification_hash", "registry_hash", "score_correlations"},
            ),
            "registry_hash": revised.registry_hash,
            "score_correlations": qualification.score_correlations,
        },
        "qualification_hash",
    )
    return revised, rebound


def _replace_registry(
    registry: AlphaCandidateRegistrySnapshot,
    candidates: tuple[AlphaCandidateRecord, ...],
    hypothesis_family_hash: str | None,
) -> AlphaCandidateRegistrySnapshot:
    return seal_contract(
        AlphaCandidateRegistrySnapshot,
        {
            "program_hash": registry.program_hash,
            "revision": registry.revision + 1,
            "predecessor_registry_hash": registry.registry_hash,
            "candidates": candidates,
            "exhausted_legacy_specs": registry.exhausted_legacy_specs,
            "historical_current_stability_rejections": (
                registry.historical_current_stability_rejections
            ),
            "latest_hypothesis_family_hash": hypothesis_family_hash,
        },
        "registry_hash",
    )


__all__ = [
    "build_candidate_set_snapshot",
    "build_goal_criteria",
    "family_registry",
    "finalize_candidate_selection",
    "initialize_registry",
    "resolve_goal_progress",
    "update_registry_candidates",
]
