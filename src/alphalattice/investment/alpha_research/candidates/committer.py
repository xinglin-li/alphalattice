"""Marker-last terminal publication of an Alpha qualification (GR3)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from ..experiments.mandate import AlphaResearchModelMandate, AlphaResearchModelRecipe
from ..publication.artifacts import AlphaCurrentArtifactStore
from .artifacts import AlphaGoalResearchArtifactStore
from .contracts import (
    AlphaCandidateRegistrySnapshot,
    AlphaGoalDisposition,
    AlphaGoalProgress,
    AlphaGoalResearchMarker,
    AlphaGoalResearchSafeProjection,
    AlphaModelResearchGoalCriteria,
    AlphaQualificationSnapshot,
    AlphaResearchProgram,
    AlphaResearchScientificStop,
    CurrentAlphaCandidateSetSnapshot,
    seal_contract,
)
from .control import (
    build_candidate_set_snapshot,
    build_goal_criteria,
    finalize_candidate_selection,
    resolve_goal_progress,
)


@dataclass(frozen=True, slots=True)
class CommittedAlphaGoalResult:
    """Retain the terminal marker, safe projection and candidate-set or scientific-stop result."""

    marker: AlphaGoalResearchMarker
    projection: AlphaGoalResearchSafeProjection
    candidate_set: CurrentAlphaCandidateSetSnapshot | None
    scientific_stop: AlphaResearchScientificStop | None


class AlphaGoalResultCommitter:
    """Commit terminal Alpha results against admitted model authority."""

    def __init__(
        self,
        store: AlphaGoalResearchArtifactStore,
        numerical_store: AlphaCurrentArtifactStore | None = None,
        sector_ema_manifest_hash: str | None = None,
        model_mandate: AlphaResearchModelMandate | None = None,
    ) -> None:
        """Bind terminal publication to goal storage and optional current numerical evidence.

        Args:
            store: Goal-driven artifact publication and active-marker owner.
            numerical_store: Optional numerical store used for candidate evidence readback.
            sector_ema_manifest_hash: Optional admitted Sector component identity.
            model_mandate: Exact model mandate that terminal commit must reconcile.
        """
        self.store = store
        self.numerical_store = numerical_store
        self.sector_ema_manifest_hash = sector_ema_manifest_hash
        self.model_mandate = model_mandate

    def commit(
        self,
        *,
        program: AlphaResearchProgram,
        selected_candidate_ids: tuple[str, ...],
        registry: AlphaCandidateRegistrySnapshot,
        qualification: AlphaQualificationSnapshot,
        progress: AlphaGoalProgress,
        limitations: tuple[str, ...],
        science_policy_hash: str,
        completed_batch_count: int,
        criteria: AlphaModelResearchGoalCriteria,
        clock: datetime | None = None,
    ) -> CommittedAlphaGoalResult:
        """Seal a candidate set or scientific stop after authority reconciliation.

        Exact active replay for the same program is read back without republishing. New candidate
        selection updates registry/qualification lineage before projection and compare-and-swap
        marker publication; nonterminal progress is refused.

        Args:
            program: Research program bound to the installed model mandate/catalog.
            selected_candidate_ids: Exact mandate-sized qualified selection, or empty
                scientific-stop selection.
            registry: Current candidate registry containing admitted model recipes.
            qualification: Current qualification evidence for the registry.
            progress: Satisfied or exhausted bounded goal progress.
            limitations: Limitations retained in the safe projection and candidate set.
            science_policy_hash: Policy identity retained by an exhausted scientific stop.
            completed_batch_count: Completed batch count retained in publication.
            criteria: Exact criteria derived from the model mandate.
            clock: Optional timezone-aware projection clock; existing exact projection clock is
                preserved.

        Returns:
            Committed marker/projection and the candidate set or scientific stop.

        Raises:
            ValueError: Model/target authority, recipe kind, terminal disposition, candidate
                selection or evidence readback is inconsistent.
        """
        if (
            self.model_mandate is None
            or program.model_mandate_hash != self.model_mandate.mandate_hash
            or program.model_catalog_hash != self.model_mandate.catalog_binding.catalog_hash
            or criteria.model_mandate_hash != self.model_mandate.mandate_hash
            or criteria != build_goal_criteria(self.model_mandate)
            or program.goal_criteria_hash != criteria.criteria_hash
        ):
            raise ValueError("alpha_research.commit_model_authority_mismatch")
        if not all(
            isinstance(value.spec, AlphaResearchModelRecipe) for value in registry.candidates
        ):
            raise ValueError("alpha_research.legacy_model_spec_not_active")
        targeted = tuple(
            value.spec.target_lane
            for value in registry.candidates
            if isinstance(value.spec, AlphaResearchModelRecipe)
        )
        if (
            program.target_policy_hashes is None and any(value is not None for value in targeted)
        ) or (
            program.target_policy_hashes is not None
            and (not targeted or any(value is None for value in targeted))
        ):
            raise ValueError("alpha_research.commit_target_authority_mismatch")
        target = self.model_mandate.target_current_qualified_candidates
        active = self.store.find_active_marker()
        if active is not None and active.program_hash == program.program_hash:
            return self._load(active)
        expected_active_marker_hash = active.marker_hash if active is not None else None
        selected_ids = selected_candidate_ids
        candidate_set = None
        stop = None
        if progress.disposition is AlphaGoalDisposition.SATISFIED:
            if len(selected_ids) != target:
                raise ValueError("alpha_research.candidate_set_selection_incomplete")
            registry, qualification = finalize_candidate_selection(
                registry=registry,
                qualification=qualification,
                selected_candidate_ids=selected_ids,
                model_mandate=self.model_mandate,
            )
            self.store.publish_registry(registry)
            self.store.publish_qualification(qualification)
            progress = resolve_goal_progress(
                criteria=criteria,
                registry=registry,
                qualification=qualification,
                completed_batch_count=completed_batch_count,
            )
            self.store.publish_goal_progress(progress)
            candidate_set = build_candidate_set_snapshot(
                program_hash=program.program_hash,
                foundation_hash=program.foundation_hash,
                registry=registry,
                qualification=qualification,
                ordered_candidate_ids=selected_ids,
                model_mandate=self.model_mandate,
                limitations=limitations,
                sector_ema_manifest_hash=self.sector_ema_manifest_hash,
            )
            self._readback_candidate_set(candidate_set)
            self.store.publish_candidate_set(candidate_set)
            disposition = "CURRENT_ALPHA_CANDIDATE_SET_READY"
            next_action = "READY_TO_DESIGN_MULTI_MODEL_DOWNSTREAM_CONSUMPTION"
        elif progress.disposition is AlphaGoalDisposition.EXHAUSTED_STOP:
            stop = seal_contract(
                AlphaResearchScientificStop,
                {
                    "program_hash": program.program_hash,
                    "registry_hash": registry.registry_hash,
                    "qualification_hash": qualification.qualification_hash,
                    "goal_progress_hash": progress.progress_hash,
                    "attempted_recipe_hashes": tuple(
                        value.spec.spec_hash for value in registry.candidates
                    ),
                    "current_qualified_count": progress.current_qualified_count,
                    "model_mandate_hash": self.model_mandate.mandate_hash,
                    "target_candidate_count": target,
                    "failure_codes": progress.unresolved_failure_codes
                    or ("alpha_research.goal_exhausted_without_mandate_candidates",),
                    "next_research_actions": (
                        "VERSIONED_FAMILY_OR_STABILITY_POLICY_REVIEW_REQUIRES_NEW_AUTHORITY",
                    ),
                    "science_policy_hash": science_policy_hash,
                },
                "stop_hash",
            )
            self.store.publish_scientific_stop(stop)
            disposition = "NO_STABLE_CURRENT_ALPHA_MODEL"
            next_action = "STOP_WITH_EVIDENCE"
        else:
            raise ValueError("alpha_research.goal_not_terminal")
        marker = seal_contract(
            AlphaGoalResearchMarker,
            {
                "program_hash": program.program_hash,
                "goal_progress_hash": progress.progress_hash,
                "registry_hash": registry.registry_hash,
                "qualification_hash": qualification.qualification_hash,
                "candidate_set_snapshot_hash": (
                    candidate_set.snapshot_hash if candidate_set is not None else None
                ),
                "scientific_stop_hash": stop.stop_hash if stop is not None else None,
                "disposition": disposition,
                "risk_admitted": False,
            },
            "marker_hash",
        )
        existing_projection = self.store.find_projection_for_marker(marker.marker_hash)
        projection = seal_contract(
            AlphaGoalResearchSafeProjection,
            {
                "program_hash": program.program_hash,
                "marker_hash": marker.marker_hash,
                "goal_target": target,
                "current_qualified_count": progress.current_qualified_count,
                "attempted_spec_count": len(registry.candidates),
                "completed_batch_count": completed_batch_count,
                "status": disposition,
                "next_action": next_action,
                "selected_candidate_ids": selected_ids,
                "limitations": limitations,
                "updated_at": (
                    existing_projection.updated_at
                    if existing_projection is not None
                    else clock or datetime.now(UTC)
                ),
            },
            "projection_hash",
        )
        self.store.publish_projection(projection)
        self.store.publish_terminal_marker(
            marker,
            expected_active_marker_hash=expected_active_marker_hash,
        )
        return CommittedAlphaGoalResult(marker, projection, candidate_set, stop)

    def _load(self, marker: AlphaGoalResearchMarker) -> CommittedAlphaGoalResult:
        candidate_set = (
            self.store.load_candidate_set(marker.candidate_set_snapshot_hash)
            if marker.candidate_set_snapshot_hash is not None
            else None
        )
        stop = (
            self.store.load_scientific_stop(marker.scientific_stop_hash)
            if marker.scientific_stop_hash is not None
            else None
        )
        projection = self.store.load_projection_for_marker(marker.marker_hash)
        if candidate_set is not None:
            self._readback_candidate_set(candidate_set)
        return CommittedAlphaGoalResult(marker, projection, candidate_set, stop)

    def _readback_candidate_set(self, value: CurrentAlphaCandidateSetSnapshot) -> None:
        if value.sector_ema_manifest_hash is not None:
            self.store.load_sector_ema_manifest(value.sector_ema_manifest_hash)
        if self.numerical_store is None:
            return
        for candidate_id, child_hash, state_hash, assessment_hash in zip(
            value.ordered_candidate_ids,
            value.current_score_child_hashes,
            value.estimator_state_hashes,
            value.stability_assessment_hashes,
            strict=True,
        ):
            child = self.numerical_store.load_current_candidate_score(child_hash)
            state = self.numerical_store.load_estimator_state(state_hash)
            assessment = self.numerical_store.load_refit_assessment(assessment_hash)
            if (
                child.candidate_id != candidate_id
                or child.estimator_state_hash != state.state_hash
                or child.stability_assessment_hash != assessment.assessment_hash
                or assessment.selected_candidate_id != candidate_id
                or (value.sector_ema_manifest_hash is None and not assessment.passed)
            ):
                raise ValueError("alpha_research.candidate_set_readback_incomplete")

    def load_exact_replay(self, program_hash: str) -> CommittedAlphaGoalResult | None:
        """Read back the active terminal result only for the exact requested program.

        Args:
            program_hash: Requested immutable research-program identity.

        Returns:
            Verified committed result, or None when no active marker or a different program is
            active.
        """
        marker = self.store.find_active_marker()
        if marker is None:
            return None
        if marker.program_hash != program_hash:
            return None
        return self._load(marker)


__all__ = ["AlphaGoalResultCommitter", "CommittedAlphaGoalResult"]
