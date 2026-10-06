"""Curating one Factor development checkpoint, for any actor, through the decision owner.

The actor names the receipt and supplies a proposal; the dossier, the research input, the
limitations, the decision policy and the submission identity are derived from the checkpoint the
receipt points at, the Factor decision owner admits or refuses, and what it admits is persisted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alphalattice.foundation.factor_research.experiments.development_evidence import (
    FactorDevelopmentCurationReader,
    FactorDevelopmentReceipt,
    FactorDevelopmentReceiptReader,
    publish_factor_development_curation,
)
from alphalattice.foundation.factor_research.inputs.research_input import (
    FactorResearchCandidateRole,
    FactorResearchProposal,
    FactorResearchProposalChoice,
    build_factor_research_proposal,
)
from alphalattice.foundation.factor_research.programs.program import (
    FactorResearchDeterministicEvidence,
)
from alphalattice.foundation.factor_research.research_loop.contracts import (
    FactorResearchReviewDecisionReceipt,
)
from alphalattice.foundation.factor_research.research_loop.decisions import (
    FACTOR_REQUIRED_RESEARCH_LIMITATIONS,
    admit_factor_research_curation,
    compile_factor_research_curation_submission,
    verify_factor_research_curation,
)
from alphalattice.protocols.actor_execution.contracts import ActorKind, AgentExecutionBinding


@dataclass(frozen=True, slots=True)
class FactorDevelopmentCuration:
    """One admitted decision, the checkpoint it judges, and where it was written."""

    artifact_uri: str
    decision: FactorResearchReviewDecisionReceipt
    checkpoint: FactorResearchDeterministicEvidence
    receipt: FactorDevelopmentReceipt


def submit_factor_development_curation(
    *,
    evidence_root: Path,
    receipt_handle: str,
    proposal: FactorResearchProposal,
    actor_kind: ActorKind,
    actor_id: str,
    agent_execution: AgentExecutionBinding | None = None,
) -> FactorDevelopmentCuration:
    """Curate one development checkpoint, through Host authority, for any actor.

    The actor names the receipt and supplies a proposal. It does not supply the
    dossier, the research input, the limitations, the decision policy or the
    submission identity -- all of those are derived from the checkpoint the
    receipt points at, so a Human, an Installed Agent and an external automation
    submitting the same proposal produce the same domain identity and differ only
    in provenance.

    The review binding comes from the development receipt's own
    ``method_binding_hash``: that is the Desk method the run was sealed under,
    and therefore the binding the dossier a decision answers was built from. The
    evidence verifier resolves it the same way during replay, so a decision is
    persisted exactly when it would also verify.
    """
    root = Path(evidence_root)
    receipt, checkpoint = FactorDevelopmentReceiptReader(root).load(receipt_handle)
    decision = admit_factor_research_curation(
        checkpoint=checkpoint,
        review_binding_hash=receipt.method_binding_hash,
        proposal=proposal,
        actor_kind=actor_kind,
        actor_id=actor_id,
        agent_execution=agent_execution,
    )
    uri = publish_factor_development_curation(
        root,
        checkpoint=checkpoint,
        review_binding_hash=receipt.method_binding_hash,
        receipt=decision,
    )
    return FactorDevelopmentCuration(
        artifact_uri=uri,
        decision=decision,
        checkpoint=checkpoint,
        receipt=receipt,
    )


def curation_readback(
    root: Path, receipt: FactorDevelopmentReceipt, checkpoint: FactorResearchDeterministicEvidence
) -> dict[str, Any]:
    """Read back a checkpoint's curation decisions, each verified against the checkpoint.

    Args:
        root: The evidence root the decisions were written under.
        receipt: The development receipt the checkpoint was produced under.
        checkpoint: The development checkpoint the decisions judge.

    Returns:
        The decisions with each factor's evidence and redundancy cluster.
    """
    decisions = FactorDevelopmentCurationReader(root).submissions(checkpoint.checkpoint_hash)
    for decision in decisions:
        verify_factor_research_curation(
            decision=decision,
            checkpoint=checkpoint,
            review_binding_hash=receipt.method_binding_hash,
        )
    items = {v.factor_id: v for v in checkpoint.evidence_report.items}
    clusters = {
        factor: cluster
        for cluster in checkpoint.redundancy_structure.clusters
        for factor in cluster.member_factor_ids
    }
    options = []
    for factor in receipt.selected_factor_ids:
        item, cluster = items[factor], clusters[factor]
        allowed = []
        # Ask the actual decision owner; the UI does not duplicate its classifier.
        for role in FactorResearchCandidateRole:
            proposal = build_factor_research_proposal(
                evidence_report_hash=checkpoint.evidence_report.report_hash,
                redundancy_structure_hash=checkpoint.redundancy_structure.structure_hash,
                choices=(
                    FactorResearchProposalChoice(
                        factor_id=factor,
                        role=role,
                        evidence_hash=item.evidence_hash,
                        cluster_id=cluster.cluster_id,
                        rationale="Preview admissible role.",
                    ),
                ),
                limitations_acknowledged=FACTOR_REQUIRED_RESEARCH_LIMITATIONS,
            )
            try:
                compile_factor_research_curation_submission(
                    checkpoint=checkpoint,
                    review_binding_hash=receipt.method_binding_hash,
                    proposal=proposal,
                )
            except ValueError:
                continue
            allowed.append(role.value)
        options.append(
            {
                "factor_id": factor,
                "classification": item.classification.value,
                "cluster_id": cluster.cluster_id,
                "roles": allowed,
                "reason_codes": item.reason_codes,
            }
        )
    return {
        "status": "AVAILABLE",
        "receipt_hash": receipt.receipt_hash,
        "choices": options,
        "limitations": FACTOR_REQUIRED_RESEARCH_LIMITATIONS,
        "decisions": [v.model_dump(mode="json") for v in decisions],
    }
