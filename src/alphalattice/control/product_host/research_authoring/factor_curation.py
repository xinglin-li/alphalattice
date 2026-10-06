"""Host composition for curating one Factor development checkpoint.

The development route had no product path at all. A case study read a receipt,
built a dossier, sealed a decision and wrote it, which meant the only thing that
had ever exercised Factor curation outside the current runtime was the test that
also defined it. A gap like that does not show up as a failure; it shows up as a
green suite proving that a case study can call four functions in order.

This is the missing composition and nothing more. It resolves the checkpoint the
handle names, hands the actor's proposal to the Factor decision owner, and
persists what that owner admits. It owns no policy: which limitations are
required, which research input the proposal implies and which policy identity a
decision is judged under are all decided in
``factor_research.research_loop.decisions``, and this module could not override
them if it wanted to.

Factor-specific on purpose. A cross-Desk curation manager would have to learn
what every Desk's checkpoint, dossier and research input are, which is the
accumulation the Host composition boundary exists to prevent.
"""

from __future__ import annotations

from pathlib import Path

from alphalattice.foundation.factor_research.experiments.development_evidence import (
    FactorDevelopmentReceipt,
)
from alphalattice.foundation.factor_research.inputs.research_input import (
    FactorResearchCandidateRole,
    FactorResearchProposalChoice,
    build_factor_research_proposal,
)
from alphalattice.foundation.factor_research.programs.program import (
    FactorResearchDeterministicEvidence,
)
from alphalattice.foundation.factor_research.research_loop.development_curation import (
    FactorDevelopmentCuration,
    submit_factor_development_curation,
)
from alphalattice.interface.local_application.portfolio_research import FactorCurationRequest
from alphalattice.protocols.actor_execution.contracts import ActorKind, AgentExecutionBinding
from alphalattice.protocols.research_authoring.contracts import AuthoringError


def submit_report_curation(
    *,
    root: Path,
    receipt: FactorDevelopmentReceipt,
    checkpoint: FactorResearchDeterministicEvidence,
    request: FactorCurationRequest,
    actor_kind: ActorKind,
    actor_id: str,
    agent_execution: AgentExecutionBinding | None = None,
) -> FactorDevelopmentCuration:
    """Bind explicit actor choices to the exact report and redundancy evidence before curation.

    Args:
        root: Exact retained Factor evidence root.
        receipt: Exact Factor development receipt.
        checkpoint: Verified deterministic report/redundancy evidence.
        request: Explicit choices with expected report receipt.
        actor_kind: Declared admitted actor kind.
        actor_id: Exact actor provenance identifier.
        agent_execution: Optional exact installed-agent execution binding.

    Returns:
        Deterministic immutable Factor development curation.

    Raises:
        AuthoringError: Report receipt is stale or selected factors lie outside the report axis.
    """
    if request.expected_receipt_hash != receipt.receipt_hash:
        raise AuthoringError("factor_research.curation_report_stale")
    if any(v.factor_id not in receipt.selected_factor_ids for v in request.choices):
        raise AuthoringError("factor_research.curation_outside_report_axis")
    items = {v.factor_id: v for v in checkpoint.evidence_report.items}
    clusters = {
        factor: cluster
        for cluster in checkpoint.redundancy_structure.clusters
        for factor in cluster.member_factor_ids
    }
    proposal = build_factor_research_proposal(
        evidence_report_hash=checkpoint.evidence_report.report_hash,
        redundancy_structure_hash=checkpoint.redundancy_structure.structure_hash,
        choices=tuple(
            FactorResearchProposalChoice(
                factor_id=v.factor_id,
                role=FactorResearchCandidateRole(v.role),
                rationale=v.rationale,
                evidence_hash=items[v.factor_id].evidence_hash,
                cluster_id=clusters[v.factor_id].cluster_id,
            )
            for v in request.choices
        ),
        limitations_acknowledged=request.limitations_acknowledged,
    )
    return submit_factor_development_curation(
        evidence_root=root,
        receipt_handle=receipt.receipt_hash,
        proposal=proposal,
        actor_kind=actor_kind,
        actor_id=actor_id,
        agent_execution=agent_execution,
    )


__all__ = [
    "submit_report_curation",
]
