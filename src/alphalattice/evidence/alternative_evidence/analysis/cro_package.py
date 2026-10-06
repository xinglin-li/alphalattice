"""Deterministically compile verified Alternative Evidence for the review consumer.

Beside verifying citation, lineage and cutoff authority, the package derives
how each finding is *structured*: how many distinct documents support it, how
many contradict it, and the resulting state. The reviewer's interpretation is
later capped by that structure, so a claim with one source cannot be presented
as corroborated and a contradicted claim cannot be presented as settled.
"""

from __future__ import annotations

from alphalattice.protocols.actor_execution.answers import AnswerProblem

from ..contracts import (
    AlternativeEvidenceRequest,
    AlternativeEvidenceSnapshot,
    seal_contract,
)
from ..documents.contracts import AlternativeEvidenceDocumentSet
from ..retrieval.contracts import (
    AlternativeEvidenceResolvedSpan,
    CurrentRetrievalGeneration,
)
from .contracts import (
    AlternativeEvidenceAnalystBrief,
    AlternativeEvidenceBriefFinding,
    AlternativeEvidenceFindingStructure,
    AlternativeEvidenceRetrievalAccessReceipt,
    CROAlternativeEvidencePackage,
    EvidenceStructureState,
)


def finding_structure(
    finding: AlternativeEvidenceBriefFinding,
    spans: dict[str, AlternativeEvidenceResolvedSpan],
) -> AlternativeEvidenceFindingStructure:
    """Count distinct documents, not spans: two spans of one filing are one source."""
    supporting = {spans[handle].document_handle for handle in finding.supporting_span_handles}
    contradicting = {spans[handle].document_handle for handle in finding.contradicting_span_handles}
    if contradicting:
        state = EvidenceStructureState.CONTESTED
    elif not supporting:
        state = EvidenceStructureState.UNSUPPORTED
    elif len(supporting) >= 2:
        state = EvidenceStructureState.SUPPORTED
    else:
        state = EvidenceStructureState.SINGLE_SOURCE
    cited = (*finding.supporting_span_handles, *finding.contradicting_span_handles)
    return AlternativeEvidenceFindingStructure(
        finding_handle=finding.finding_handle,
        supporting_document_count=len(supporting),
        contradicting_document_count=len(contradicting),
        latest_available_at=max(spans[handle].available_at for handle in cited),
        state=state,
    )


def compile_cro_alternative_evidence_package(
    *,
    request: AlternativeEvidenceRequest,
    snapshot: AlternativeEvidenceSnapshot,
    document_set: AlternativeEvidenceDocumentSet,
    generation: CurrentRetrievalGeneration,
    access_receipt: AlternativeEvidenceRetrievalAccessReceipt,
    brief: AlternativeEvidenceAnalystBrief,
    resolved_spans: tuple[AlternativeEvidenceResolvedSpan, ...],
    dropped: tuple[AnswerProblem, ...] = (),
) -> CROAlternativeEvidencePackage:
    """Verify citation/time authority and derive structure; claim no entailment.

    `dropped` is what the Host could not admit from the Analyst's last answer
    after two corrections: an item the Analyst reported and nobody can read,
    so it leads the package's gaps rather than vanishing.
    """
    if (
        snapshot.request_hash != request.request_hash
        or document_set.request_hash != request.request_hash
        or document_set.source_snapshot_hash != snapshot.snapshot_hash
        or generation.document_set_hash != document_set.document_set_hash
        or access_receipt.request_hash != request.request_hash
        or access_receipt.document_set_hash != document_set.document_set_hash
        or access_receipt.retrieval_generation_hash != generation.generation_hash
        or brief.request_hash != request.request_hash
        or brief.source_snapshot_hash != snapshot.snapshot_hash
        or brief.document_set_hash != document_set.document_set_hash
        or brief.retrieval_generation_hash != generation.generation_hash
        or brief.access_receipt_hash != access_receipt.receipt_hash
    ):
        raise ValueError("alternative_evidence.cro_package_lineage_invalid")
    span_by_handle = {value.span_handle: value for value in resolved_spans}
    cited = tuple(
        dict.fromkeys(
            handle
            for finding in brief.findings
            for handle in (
                *finding.supporting_span_handles,
                *finding.contradicting_span_handles,
            )
        )
    )
    if not set(cited).issubset(access_receipt.delivered_span_handles) or not set(cited).issubset(
        span_by_handle
    ):
        raise ValueError("alternative_evidence.cro_package_citation_invalid")
    verified = tuple(span_by_handle[value] for value in cited)
    if any(value.available_at > request.evidence_as_of for value in verified):
        raise ValueError("alternative_evidence.cro_package_cutoff_invalid")
    structures = tuple(finding_structure(value, span_by_handle) for value in brief.findings)
    documented = {value.entity_id for value in document_set.documents}
    # The machine coverage ledger reaches the package the review stands on:
    # an integrated receipt's cell gaps -- unread unit needs, undelivered
    # tables, source and representation gaps, budget-skipped questions --
    # travel as compact lines beside the document gaps, and no actor's
    # reported check erases them.
    routing = access_receipt.routing
    missing = tuple(
        dict.fromkeys(
            (
                *(
                    f"The Analyst's answer lost item {value.item} after two corrections: "
                    f"{value.text}"
                    for value in dropped
                ),
                *(
                    f"Rejected {value.semantic_handle}: {value.code}"
                    for value in document_set.rejections
                ),
                *(
                    f"No admitted document for {entity_id} before the cutoff"
                    for entity_id in request.ordered_entity_ids
                    if entity_id not in documented
                ),
                *(() if routing is None else routing.gap_lines()),
            )
        )
    )
    limitations = tuple(
        dict.fromkeys(
            (
                *snapshot.limitations,
                *(value for finding in brief.findings for value in finding.limitations),
                "Host verification proves citation and authority, not textual entailment.",
                "Alternative Evidence may only constrain or de-risk; it cannot create Alpha.",
            )
        )
    )
    return seal_contract(
        CROAlternativeEvidencePackage,
        "package_hash",
        request_hash=request.request_hash,
        source_snapshot_hash=snapshot.snapshot_hash,
        document_set_hash=document_set.document_set_hash,
        retrieval_generation_hash=generation.generation_hash,
        analyst_brief_hash=brief.brief_hash,
        access_receipt_hash=access_receipt.receipt_hash,
        obligation_hash=brief.obligation_hash,
        source_status=snapshot.status,
        admitted_document_count=len(document_set.documents),
        rejected_document_count=len(document_set.rejections),
        verified_spans=verified,
        finding_structures=structures,
        missing_evidence=missing,
        expires_at=snapshot.expires_at,
        limitations=limitations,
    )


__all__ = ["compile_cro_alternative_evidence_package", "finding_structure"]
