"""Admit a selected Factor development result as a Research Foundation, and read it back."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.foundation.causal_outcomes.execution.readers import (
    CausalExecutionOutcomeDevelopmentReader,
)
from alphalattice.foundation.factor_research.experiments.development_evidence import (
    FactorDevelopmentReceipt,
)
from alphalattice.foundation.factor_research.programs.program import (
    FactorResearchDeterministicEvidence,
)
from alphalattice.foundation.factor_research.publication.artifacts import (
    FactorResearchArtifactStore,
)
from alphalattice.foundation.factor_research.research_loop.contracts import (
    FactorResearchReviewDecisionReceipt,
)
from alphalattice.foundation.factor_research.research_loop.decisions import (
    verify_factor_research_curation,
)
from alphalattice.foundation.research_foundation.contracts import ResearchFoundationAdmission
from alphalattice.foundation.research_foundation.mandate.foundation import (
    ResearchFoundationService,
    build_current_research_foundation_binding,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.protocols.actor_execution.contracts import ActorSubmissionBinding


def admit_selected_factor_foundation(
    *,
    receipt: FactorDevelopmentReceipt,
    checkpoint: FactorResearchDeterministicEvidence,
    decision: FactorResearchReviewDecisionReceipt,
    panel_manifest: dict[str, Any],
    artifact_root: Path,
    logical_panel_hash: str,
    logical_semantic_index_hash: str,
    factor_task_id: str,
    input_id: str,
    input_binding_hash: str,
    factor_input_binding_hash: str,
    recorded_admission: ResearchFoundationAdmission | None = None,
) -> ResearchFoundationAdmission:
    """Bind verified development evidence under the existing one-session policy.

    Host reopens the receipt and its Program/Panel before this call. The Factor
    owner, not an actor or a current pointer, re-derives the selected input.
    A supplied recorded admission permits only exact reconstruction of that
    historical value, not a new admission under an unavailable curation policy.
    """
    verify_factor_research_curation(
        decision=decision,
        checkpoint=checkpoint,
        review_binding_hash=receipt.method_binding_hash,
        recorded_readback=recorded_admission is not None,
    )
    selected = decision.submission.research_input
    if not set((*selected.core_factor_ids, *selected.conditional_factor_ids)).issubset(
        receipt.selected_factor_ids
    ):
        raise ValueError("research_foundation.selection_outside_report")
    outcome_reader = CausalExecutionOutcomeDevelopmentReader(artifact_root)
    outcome = outcome_reader.load_manifest(checkpoint.program.causal_outcome_snapshot_hash)
    seal = outcome_reader.resolve_method_seal(outcome.snapshot_hash)
    if seal.disposition != "METHOD_BOUND" or seal.outcome_marker is None:
        raise ValueError("research_foundation.outcome_method_unbound")
    foundation = build_current_research_foundation_binding(
        panel_manifest=panel_manifest,
        factor_program=checkpoint.program,
        factor_input=selected,
        evidence_report_hash=checkpoint.evidence_report.report_hash,
        redundancy_structure_hash=checkpoint.redundancy_structure.structure_hash,
        execution_outcome=outcome,
        execution_marker_hash=seal.outcome_marker.marker_hash,
        logical_panel_hash=logical_panel_hash,
        logical_semantic_index_hash=logical_semantic_index_hash,
    )
    values = dict(
        kind="ResearchFoundationAdmission",
        foundation=foundation,
        factor_task_id=factor_task_id,
        factor_receipt_hash=receipt.receipt_hash,
        curation_receipt_hash=decision.receipt_hash,
        input_id=input_id,
        input_binding_hash=input_binding_hash,
        factor_input_binding_hash=factor_input_binding_hash,
    )
    draft = ResearchFoundationAdmission.model_construct(**values, admission_hash="")
    result = ResearchFoundationAdmission(
        **values,
        admission_hash=canonical_hash(draft.model_dump(mode="json", exclude={"admission_hash"})),
    )
    if recorded_admission is not None and result != recorded_admission:
        raise ValueError("research_foundation.recorded_source_mismatch")
    return result


def read_foundation_admission(
    artifact_root: Path, admission_hash: str
) -> ResearchFoundationAdmission:
    store = FactorResearchArtifactStore(artifact_root)
    admission = ResearchFoundationAdmission.model_validate(
        store.load_foundation_admission(admission_hash)
    )
    binding = store.load_research_foundation(
        store.uri("research-desk/foundations", admission.foundation.foundation_hash)
    )
    if binding != admission.foundation.model_dump(mode="json", exclude_none=True):
        raise ValueError("research_foundation.admission_binding_mismatch")
    return admission


def publish_foundation_admission(
    artifact_root: Path,
    admission: ResearchFoundationAdmission,
    *,
    published_at: datetime,
    actor: ActorSubmissionBinding,
) -> str:
    store = FactorResearchArtifactStore(artifact_root)
    if actor.submission_hash != admission.admission_hash:
        raise ValueError("research_foundation.confirmation_mismatch")
    existed = (
        store.root / "research-desk/foundation-admissions" / f"{admission.admission_hash}.json"
    ).exists()
    ResearchFoundationService(artifact_store=store, mutation_gate=WorkspaceMutationGate()).publish(
        binding=admission.foundation,
        execution_outcome_marker_hash=admission.foundation.execution_outcome.marker_hash,
        updated_at=published_at,
        publish_projection=False,
    )
    store.record_foundation_confirmation(actor.model_dump(mode="json"))
    store.publish_foundation_admission(
        payload=admission.model_dump(mode="json"), admission_hash=admission.admission_hash
    )
    if read_foundation_admission(artifact_root, admission.admission_hash) != admission:
        raise ValueError("research_foundation.admission_readback_mismatch")
    return "REUSED_EXACT" if existed else "FOUNDATION_SEALED"


__all__ = []
