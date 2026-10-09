"""Host validation for actor-neutral Alternative Evidence answers and briefs.

An actor answers with judgment only (`AlternativeEvidenceAnalystAnswer`): the
Host screens each finding on its own against the packet it was given, maps
the view's excerpt aliases back to span handles, assigns the handles, writes
the brief's texts from delivery facts and seals it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.kernel.shared_kernel.domain.serialization import canonical_json_bytes
from alphalattice.kernel.shared_kernel.identity import canonical_hash, schema_structure
from alphalattice.kernel.shared_kernel.source_identity import source_rule_closure_hash
from alphalattice.protocols.actor_execution import (
    ActorKind,
    ActorSubmissionBinding,
    AgentExecutionBinding,
    seal_actor_submission,
)
from alphalattice.protocols.actor_execution.answers import (
    AnswerProblem,
    ScreenedAnswer,
    screen_answer,
)

from ..contracts import (
    AlternativeEvidenceRequest,
    AlternativeEvidenceReviewStatus,
    AlternativeEvidenceSnapshot,
    seal_contract,
)
from ..documents.contracts import AlternativeEvidenceDocumentSet
from ..retrieval.contracts import (
    AlternativeEvidenceResolvedSpan,
    AlternativeEvidenceRetrievalGeneration,
    CurrentRetrievalGeneration,
    WholeFilingsGeneration,
)
from .contracts import (
    ANALYST_ANSWER_TEXT_FIELDS,
    COMPLETION_SCHEMA_V3,
    AlternativeEvidenceAnalystAnswer,
    AlternativeEvidenceAnalystBrief,
    AlternativeEvidenceAnalystBriefReceipt,
    AlternativeEvidenceAnalystBriefSubmission,
    AlternativeEvidenceAnswerFinding,
    AlternativeEvidenceBriefFinding,
    AlternativeEvidenceBriefFindingSubmission,
    AlternativeEvidenceResearchObligation,
    AlternativeEvidenceRetrievalAccessReceipt,
    AnalysisCompletion,
    IssuerCheckOutcome,
    IssuerReviewState,
)
from .packet import span_aliases


class AlternativeEvidenceBriefAuthorityError(ValueError):
    """An evidence brief failed its frozen Host lineage checks."""


ALTERNATIVE_ANALYSIS_POLICY_OWNER = "alternative_evidence.analysis"
ALTERNATIVE_DECISION_POLICY_OWNER = "alternative_evidence.runtime.document_intelligence"
ALTERNATIVE_POLICY_VERSION = "2026-09-02"
ALTERNATIVE_BRIEF_RECEIPT_CATEGORY = "analyst-brief-receipts"


class AlternativeEvidenceHostPolicyBinding(BaseModel):  # type: ignore[misc]
    """One Host policy, defined and built beside the sealer that applies it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["AlternativeEvidenceHostPolicyBinding"] = "AlternativeEvidenceHostPolicyBinding"
    owner_id: str = Field(min_length=1, max_length=128)
    policy_version: str = Field(min_length=1, max_length=32)
    contract_schema_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    validation_source_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    semantics_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_binding(self) -> Self:
        """Verify the sealed Host policy binding hash."""
        if self.binding_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"binding_hash"})
        ):
            raise ValueError("alternative_evidence.host_policy_binding_invalid")
        return self


def _host_policy(
    *,
    owner_id: str,
    contract_schemas: dict[str, object],
    tracked_paths: tuple[str, ...],
    semantics: dict[str, object],
    playpen_root: Path,
) -> AlternativeEvidenceHostPolicyBinding:
    values = {
        "kind": "AlternativeEvidenceHostPolicyBinding",
        "owner_id": owner_id,
        "policy_version": ALTERNATIVE_POLICY_VERSION,
        "contract_schema_hash": str(canonical_hash(contract_schemas)),
        "validation_source_hash": source_rule_closure_hash(
            root=playpen_root.resolve(),
            tracked_paths=tracked_paths,
            semantic_owner="alternative_evidence",
            numerical_role="EVIDENCE_HOST_POLICY",
        ),
        "semantics_hash": str(canonical_hash(semantics)),
    }
    return AlternativeEvidenceHostPolicyBinding(
        **values,
        binding_hash=str(canonical_hash(values)),
    )


def build_alternative_analysis_policy_binding(
    playpen_root: Path,
) -> AlternativeEvidenceHostPolicyBinding:
    """What the Host requires of one analyst brief, whoever wrote it.

    The query program is part of this policy: which spans an analyst may see is
    as much the Host's analysis rule as which citations it accepts, so
    `packet.py` sits inside the source closure beside this sealer. So are the
    typed disclosure rules (`disclosures.py`), the litigation matter inventory
    (`matters.py`), the corporate-event inventory (`events.py`), the
    financing inventory (`financing.py`), the operations inventory the packet
    shows (`operations.py`), the topic routing and the temporal
    comparison of the integrated selection (`routing.py`, `comparison.py`),
    the table views (`tables.py`) and the section reader they locate with
    (`structure.py`): a rule edit rotates this policy, never a sealed
    receipt's recorded rule identity.
    """
    return _host_policy(
        owner_id=ALTERNATIVE_ANALYSIS_POLICY_OWNER,
        contract_schemas={
            "answer": schema_structure(AlternativeEvidenceAnalystAnswer),
            "submission": schema_structure(AlternativeEvidenceAnalystBriefSubmission),
            "finding": schema_structure(AlternativeEvidenceBriefFinding),
            "brief": schema_structure(AlternativeEvidenceAnalystBrief),
            "resolved_span": schema_structure(AlternativeEvidenceResolvedSpan),
            "access_receipt": schema_structure(AlternativeEvidenceRetrievalAccessReceipt),
        },
        tracked_paths=(
            "src/alphalattice/evidence/alternative_evidence/analysis/disclosures.py",
            "src/alphalattice/evidence/alternative_evidence/analysis/events.py",
            "src/alphalattice/evidence/alternative_evidence/analysis/financing.py",
            "src/alphalattice/evidence/alternative_evidence/analysis/matters.py",
            "src/alphalattice/evidence/alternative_evidence/analysis/operations.py",
            "src/alphalattice/evidence/alternative_evidence/analysis/packet.py",
            "src/alphalattice/evidence/alternative_evidence/analysis/routing.py",
            "src/alphalattice/evidence/alternative_evidence/analysis/submissions.py",
            "src/alphalattice/evidence/alternative_evidence/documents/comparison.py",
            "src/alphalattice/evidence/alternative_evidence/documents/structure.py",
            "src/alphalattice/evidence/alternative_evidence/documents/tables.py",
            "src/alphalattice/evidence/alternative_evidence/retrieval/session.py",
        ),
        semantics={
            "answer": (
                "the analyst answers with judgment fields only; each finding is checked on "
                "its own, a subset is an answer and an empty list is an answer with no findings"
            ),
            "aliases": (
                "a view names each delivered excerpt S1.. in reading order and the Host maps "
                "the aliases back; the Host assigns every handle"
            ),
            "span_citation": "every finding cites resolved spans admitted by the access receipt",
            "span_selection": (
                "the Host runs the installed query program; the analyst does not search"
            ),
            "typed_disclosures": (
                "the Host reads three closed disclosure families from the filings' own "
                "sections under frozen rule identities, as source assertions beside the "
                "program's spans; the analyst neither extracts nor certifies them"
            ),
            "litigation_matters": (
                "the Host inventories the filings' Legal Proceedings items and contingencies "
                "notes into provisional matters and reads their windows whole under a "
                "declared per-session allowance, reporting pending windows as unread; the "
                "analyst reads the windows and states nothing the source does not"
            ),
            "freshness": "brief completion may not follow source snapshot expiry",
            "coverage": (
                "issuer review states and the brief's coverage text are written by the Host "
                "from delivery facts; no actor reports a check or asks for review"
            ),
        },
        playpen_root=playpen_root,
    )


def build_alternative_evidence_decision_policy_binding(
    playpen_root: Path,
) -> AlternativeEvidenceHostPolicyBinding:
    """How the Host seals and publishes one brief decision, independent of analysis."""
    return _host_policy(
        owner_id=ALTERNATIVE_DECISION_POLICY_OWNER,
        contract_schemas={
            "receipt": schema_structure(AlternativeEvidenceAnalystBriefReceipt),
            "actor_submission": schema_structure(ActorSubmissionBinding),
        },
        tracked_paths=("src/alphalattice/evidence/alternative_evidence/analysis/submissions.py",),
        semantics={
            "category": ALTERNATIVE_BRIEF_RECEIPT_CATEGORY,
            "identity_field": "receipt_hash",
            "actor_provenance": "recorded beside the decision, never inside it",
        },
        playpen_root=playpen_root,
    )


MAXIMUM_ANSWER_FINDINGS = 32
"""The findings one answer holds, as the answer model bounds them."""

MAXIMUM_SUBMISSION_BYTES = 64 * 1024
"""The whole serialized brief submission. Every field is bounded on its own;
this bounds their sum, so a submission is one delivery the consumer can hold."""


def validate_submission_bytes(submission: BaseModel) -> None:
    size = len(canonical_json_bytes(submission.model_dump(mode="json")))
    if size > MAXIMUM_SUBMISSION_BYTES:
        raise AlternativeEvidenceBriefAuthorityError(
            f"alternative_evidence.submission_bytes_exceeded:{size}>{MAXIMUM_SUBMISSION_BYTES}"
        )


def analysis_completion(
    *,
    obligation: AlternativeEvidenceResearchObligation,
    document_set: AlternativeEvidenceDocumentSet,
    resolved_spans: tuple[AlternativeEvidenceResolvedSpan, ...],
    findings: tuple[AlternativeEvidenceBriefFinding, ...],
) -> AnalysisCompletion:
    """Per-issuer states from delivery facts alone (`issuer-delivery-facts-v3`).

    An issuer with no admitted document could not be read; one with documents
    and no delivered excerpt was not read; one whose excerpts were delivered
    was read and answered -- with findings or with none, which is a clean
    record of *reported* findings and never proof of no risk. No actor
    reports a check, so none is listed as executed.
    """
    outcomes = []
    for entity_id in obligation.ordered_entity_ids:
        documents = sum(1 for value in document_set.documents if value.entity_id == entity_id)
        spans = sum(1 for value in resolved_spans if value.entity_id == entity_id)
        own = [value for value in findings if entity_id in value.affected_entities]
        contradicting = sum(1 for value in own if value.contradicting_span_handles)
        if documents == 0:
            state = IssuerReviewState.SOURCE_MISSING
        elif spans == 0:
            state = IssuerReviewState.NO_SPANS_DELIVERED
        elif own:
            state = IssuerReviewState.EXECUTED_WITH_FINDINGS
        else:
            state = IssuerReviewState.EXECUTED_NO_FINDINGS
        outcomes.append(
            IssuerCheckOutcome(
                entity_id=entity_id,
                state=state,
                documents_admitted=documents,
                spans_delivered=spans,
                findings=len(own),
                contradicting_findings=contradicting,
                checks_executed=(),
            )
        )
    return AnalysisCompletion(
        completion_schema=COMPLETION_SCHEMA_V3,
        required_checks=obligation.required_checks,
        issuers=tuple(outcomes),
    )


def screen_analyst_answer(
    raw: object,
    *,
    entity_ids: Sequence[str],
    aliases: Mapping[str, AlternativeEvidenceResolvedSpan],
) -> ScreenedAnswer[AlternativeEvidenceAnswerFinding]:
    """Check each written finding on its own against the packet it answers.

    A finding is acceptable when it names an issuer of the packet and every
    alias it cites is an excerpt of the view, of that issuer, cited on one
    side only. Nothing omitted is a problem. Every problem is stated in plain
    words for the agent that wrote it.
    """
    issuers = {value.casefold(): value for value in entity_ids}
    listed = ", ".join(entity_ids)

    def check(item: AlternativeEvidenceAnswerFinding) -> list[str]:
        problems: list[str] = []
        entity = issuers.get(item.issuer.strip().casefold())
        if entity is None:
            problems.append(
                f"issuer '{item.issuer}' is not an issuer of this bundle; use one of {listed}."
            )
        cite = _aliases(item.cite)
        contrary = _aliases(item.contrary)
        for alias in dict.fromkeys((*cite, *contrary)):
            span = aliases.get(alias)
            if span is None:
                problems.append(f"{alias} is not an excerpt of this bundle.")
            elif entity is not None and span.entity_id != entity:
                problems.append(f"{alias} is an excerpt of {span.entity_id}, not of {entity}.")
        for alias in sorted(set(cite) & set(contrary)):
            problems.append(f"{alias} is cited both for and against; keep it in one list.")
        return problems

    return screen_answer(
        raw,
        items_field="findings",
        item_model=AlternativeEvidenceAnswerFinding,
        maximum_items=MAXIMUM_ANSWER_FINDINGS,
        check_item=check,
        text_fields=ANALYST_ANSWER_TEXT_FIELDS,
    )


def accepted_analyst_answer(
    screened: ScreenedAnswer[AlternativeEvidenceAnswerFinding],
    *,
    entity_ids: Sequence[str],
) -> AlternativeEvidenceAnalystAnswer:
    """The screen's accepted part, spelled canonically.

    The screen's accepted part, spelled canonically: the issuer as the
    packet names it, each alias once and upper-cased.
    """
    issuers = {value.casefold(): value for value in entity_ids}
    return AlternativeEvidenceAnalystAnswer(
        findings=tuple(
            item.model_copy(
                update={
                    "issuer": issuers[item.issuer.strip().casefold()],
                    "cite": _aliases(item.cite),
                    "contrary": _aliases(item.contrary),
                }
            )
            for _, item in screened.items
        ),
        notes=screened.texts.get("notes", ""),
    )


def normalize_analyst_answer(
    answer: AlternativeEvidenceAnalystAnswer,
    *,
    obligation: AlternativeEvidenceResearchObligation,
    document_set: AlternativeEvidenceDocumentSet,
    resolved_spans: tuple[AlternativeEvidenceResolvedSpan, ...],
) -> AlternativeEvidenceAnalystBriefSubmission:
    """The brief's actor-side content, written by the Host from an answer.

    Each finding gets the handle of its place in the answer (`FIND-001`..)
    and its aliases become span handles; the summary and coverage texts are
    stated from delivery facts; nothing asks for review and no check is
    reported. Raises when the answer is not the screen's accepted part.
    """
    aliases = span_aliases(resolved_spans, obligation.ordered_entity_ids)
    screened = screen_analyst_answer(
        answer.model_dump(mode="json"), entity_ids=obligation.ordered_entity_ids, aliases=aliases
    )
    if not screened.clean:
        first = screened.problems[0]
        raise AlternativeEvidenceBriefAuthorityError(
            f"alternative_evidence.answer_invalid:item_{first.item or 0}"
        )
    findings = tuple(
        AlternativeEvidenceBriefFindingSubmission(
            finding_handle=f"FIND-{index:03d}",
            affected_entities=(item.issuer,),
            topic=item.topic,
            lifecycle=item.lifecycle,
            direction=item.direction,
            summary=item.summary,
            supporting_span_handles=tuple(aliases[value].span_handle for value in item.cite),
            contradicting_span_handles=tuple(aliases[value].span_handle for value in item.contrary),
            limitations=(),
        )
        for index, item in enumerate(answer.findings, start=1)
    )
    issuers = obligation.ordered_entity_ids
    reported = {item.issuer for item in answer.findings}
    read = [value for value in issuers if any(s.entity_id == value for s in resolved_spans)]
    unsourced = [
        value
        for value in issuers
        if not any(document.entity_id == value for document in document_set.documents)
    ]
    summary = answer.notes or (
        f"{_count(len(findings), 'finding')} reported on {len(reported)} of "
        f"{_count(len(issuers), 'issuer')}."
    )
    coverage = (
        f"{_count(len(document_set.documents), 'admitted document')}; "
        f"{_count(len(resolved_spans), 'excerpt')} delivered for {len(read)} of "
        f"{_count(len(issuers), 'issuer')}."
    )
    if unsourced:
        coverage += " No admitted document for: " + ", ".join(unsourced) + "."
    return AlternativeEvidenceAnalystBriefSubmission(
        executive_summary=summary[:2000],
        findings=findings,
        unresolved_questions=(),
        source_coverage_assessment=coverage[:1000],
        requires_human_review=False,
        limitations_acknowledged=True,
    )


def _aliases(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value.strip().upper() for value in values))


def _count(value: int, noun: str) -> str:
    return f"{value} {noun}{'' if value == 1 else 's'}"


def seal_alternative_evidence_analyst_brief(
    *,
    request: AlternativeEvidenceRequest,
    obligation: AlternativeEvidenceResearchObligation,
    snapshot: AlternativeEvidenceSnapshot,
    document_set: AlternativeEvidenceDocumentSet,
    generation: CurrentRetrievalGeneration,
    access_receipt: AlternativeEvidenceRetrievalAccessReceipt,
    resolved_spans: tuple[AlternativeEvidenceResolvedSpan, ...],
    analysis_policy: AlternativeEvidenceHostPolicyBinding,
    decision_policy: AlternativeEvidenceHostPolicyBinding,
    playpen_root: Path,
    answer: AlternativeEvidenceAnalystAnswer,
    completed_at: datetime,
    actor_kind: ActorKind,
    actor_id: str,
    dropped: tuple[AnswerProblem, ...] = (),
    agent_execution: AgentExecutionBinding | None = None,
    model_call_count: int = 0,
    protocol_repair_count: int = 0,
) -> AlternativeEvidenceAnalystBriefReceipt:
    """Seal one answer's brief after exact document, span and freshness checks.

    Both policies are re-derived from ``playpen_root`` and compared field for
    field, so an actor cannot hand over bindings it built itself and have the
    brief record them as Host authority. `answer` is the accepted part of
    what the actor wrote; `dropped` names what was not accepted and why.
    """
    analysis_policy = AlternativeEvidenceHostPolicyBinding.model_validate(analysis_policy)
    decision_policy = AlternativeEvidenceHostPolicyBinding.model_validate(decision_policy)
    if analysis_policy != build_alternative_analysis_policy_binding(
        playpen_root
    ) or decision_policy != build_alternative_evidence_decision_policy_binding(playpen_root):
        raise AlternativeEvidenceBriefAuthorityError(
            "alternative_evidence.host_policy_unauthorized"
        )
    request = AlternativeEvidenceRequest.model_validate(request)
    obligation = AlternativeEvidenceResearchObligation.model_validate(obligation)
    snapshot = AlternativeEvidenceSnapshot.model_validate(snapshot)
    document_set = AlternativeEvidenceDocumentSet.model_validate(document_set)
    if not isinstance(generation, AlternativeEvidenceRetrievalGeneration | WholeFilingsGeneration):
        # A new brief stands on a current format: an index or a unit
        # delivered whole (W4); the retired v3 format is history.
        raise AlternativeEvidenceBriefAuthorityError("alternative_evidence.brief_lineage_mismatch")
    access_receipt = AlternativeEvidenceRetrievalAccessReceipt.model_validate(access_receipt)
    answer = AlternativeEvidenceAnalystAnswer.model_validate(answer)
    if completed_at.tzinfo is None or completed_at.utcoffset() is None:
        raise AlternativeEvidenceBriefAuthorityError("alternative_evidence.brief_clock_invalid")
    if completed_at > snapshot.expires_at:
        raise AlternativeEvidenceBriefAuthorityError("alternative_evidence.brief_source_stale")
    if (
        tuple(obligation.ordered_entity_ids) != tuple(request.ordered_entity_ids)
        or obligation.evidence_as_of != request.evidence_as_of
    ):
        raise AlternativeEvidenceBriefAuthorityError(
            "alternative_evidence.brief_obligation_mismatch"
        )
    if (
        snapshot.request_hash != request.request_hash
        or document_set.request_hash != request.request_hash
        or document_set.source_snapshot_hash != snapshot.snapshot_hash
        or generation.document_set_hash != document_set.document_set_hash
        or access_receipt.request_hash != request.request_hash
        or access_receipt.document_set_hash != document_set.document_set_hash
        or access_receipt.retrieval_generation_hash != generation.generation_hash
    ):
        raise AlternativeEvidenceBriefAuthorityError("alternative_evidence.brief_lineage_mismatch")
    spans = {value.span_handle: value for value in resolved_spans}
    delivered = set(access_receipt.delivered_span_handles)
    if len(spans) != len(resolved_spans) or set(spans) != delivered:
        raise AlternativeEvidenceBriefAuthorityError(
            "alternative_evidence.brief_span_receipt_mismatch"
        )
    document_handles = {value.semantic_handle for value in document_set.documents}
    if any(value.document_handle not in document_handles for value in spans.values()):
        raise AlternativeEvidenceBriefAuthorityError(
            "alternative_evidence.brief_document_lineage_invalid"
        )
    validate_submission_bytes(answer)
    submission = normalize_analyst_answer(
        answer, obligation=obligation, document_set=document_set, resolved_spans=resolved_spans
    )
    findings = tuple(
        AlternativeEvidenceBriefFinding.model_validate(finding.model_dump())
        for finding in submission.findings
    )
    completion = analysis_completion(
        obligation=obligation,
        document_set=document_set,
        resolved_spans=tuple(spans.values()),
        findings=findings,
    )
    brief = seal_contract(
        AlternativeEvidenceAnalystBrief,
        "brief_hash",
        request_hash=request.request_hash,
        source_snapshot_hash=snapshot.snapshot_hash,
        document_set_hash=document_set.document_set_hash,
        retrieval_generation_hash=generation.generation_hash,
        access_receipt_hash=access_receipt.receipt_hash,
        review_binding_hash=analysis_policy.binding_hash,
        obligation_hash=obligation.obligation_hash,
        # An answer is complete as written: an empty one is an answer with no
        # findings, not an incomplete analysis.
        review_status=AlternativeEvidenceReviewStatus.COMPLETE,
        executive_summary=submission.executive_summary,
        findings=findings,
        unresolved_questions=submission.unresolved_questions,
        source_coverage_assessment=submission.source_coverage_assessment,
        requires_human_review=submission.requires_human_review,
        completed_at=completed_at,
        completion=completion,
    )
    actor_submission = seal_actor_submission(
        actor_kind=actor_kind,
        actor_id=actor_id,
        submission_hash=canonical_hash(answer.model_dump(mode="json")),
        agent_execution=agent_execution,
    )
    return seal_contract(
        AlternativeEvidenceAnalystBriefReceipt,
        "receipt_hash",
        submission=submission,
        answer=answer,
        dropped=dropped,
        actor_submission=actor_submission,
        brief=brief,
        decision_policy_hash=decision_policy.binding_hash,
        model_call_count=model_call_count,
        protocol_repair_count=protocol_repair_count,
    )


__all__ = [
    "ALTERNATIVE_ANALYSIS_POLICY_OWNER",
    "ALTERNATIVE_BRIEF_RECEIPT_CATEGORY",
    "ALTERNATIVE_DECISION_POLICY_OWNER",
    "ALTERNATIVE_POLICY_VERSION",
    "AlternativeEvidenceBriefAuthorityError",
    "AlternativeEvidenceHostPolicyBinding",
    "accepted_analyst_answer",
    "analysis_completion",
    "build_alternative_analysis_policy_binding",
    "build_alternative_evidence_decision_policy_binding",
    "normalize_analyst_answer",
    "screen_analyst_answer",
    "seal_alternative_evidence_analyst_brief",
]
