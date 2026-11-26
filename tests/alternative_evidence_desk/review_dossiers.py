"""Issue, submission, finding, issuer, dossier and seal builders of the Portfolio
review decision suites.

The synthetic review inputs that the casebook and both evidence review route
suites drove through the decision test module. Test support beside its
owner; nothing here is product authority or evidence.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from alphalattice.evidence.alternative_evidence.analysis.contracts import (
    EvidenceDirection,
    EvidenceLifecycle,
    EvidenceStructureState,
    EvidenceTopic,
)
from alphalattice.evidence.alternative_evidence.contracts import (
    seal_contract,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.oversight.chief_risk_officer.decision.book_evidence import (
    REVIEW_CLAIM_LIMITS,
)
from alphalattice.oversight.chief_risk_officer.decision.portfolio_review import (
    CROEvidenceInterpretation,
    CROEvidenceRelevance,
    CROPortfolioMitigation,
    CROPositionImpactDirection,
    CRORiskConfidence,
    CRORiskSeverity,
    CROSeverityIfTrue,
    PortfolioReviewAnswer,
    PortfolioReviewAnswerRisk,
    PortfolioReviewAssessmentSubmission,
    PortfolioReviewCoverage,
    PortfolioReviewDossier,
    PortfolioReviewDossierCitation,
    PortfolioReviewDossierFinding,
    PortfolioReviewDossierIssuer,
    PortfolioReviewMaterialIssue,
    PortfolioReviewReceipt,
    PortfolioReviewRecommendation,
)
from alphalattice.oversight.chief_risk_officer.decision.submissions import (
    build_portfolio_review_policy_binding,
    seal_portfolio_review,
)
from alphalattice.oversight.chief_risk_officer.portfolio_evidence.contracts import (
    BookAuthority,
    ExposureBand,
)
from alphalattice.oversight.chief_risk_officer.runtime.portfolio_review_task import (
    PortfolioReviewActorResult,
    SubmittedPortfolioReviewActor,
)
from alphalattice.protocols.actor_execution import ActorKind, AgentExecutionBinding
from alphalattice.protocols.actor_execution.answers import AnswerProblem

REPO_ROOT = Path(__file__).resolve().parents[2]

_NOW = datetime(2026, 8, 12, 16, tzinfo=UTC)


def _issue(
    finding: str,
    entity: str,
    *,
    entities: tuple[str, ...] | None = None,
    relevance: CROEvidenceRelevance = CROEvidenceRelevance.DIRECT,
    direction: CROPositionImpactDirection = CROPositionImpactDirection.ADVERSE,
    severity: CROSeverityIfTrue = CROSeverityIfTrue.HIGH,
    interpretation: CROEvidenceInterpretation = CROEvidenceInterpretation.SUPPORTED,
    mitigation: CROPortfolioMitigation = CROPortfolioMitigation.NONE,
    handle: str = "ISSUE-1",
    causal_channel: str = "A named supply constraint reduces deliverable units.",
) -> PortfolioReviewMaterialIssue:
    return PortfolioReviewMaterialIssue(
        issue_handle=handle,
        cited_finding_handles=(finding,),
        affected_entities=entities if entities is not None else (entity,),
        relevance=relevance,
        position_impact_direction=direction,
        causal_channel=causal_channel,
        severity_if_true=severity,
        evidence_interpretation=interpretation,
        portfolio_mitigation=mitigation,
        uncertainty="The duration of the constraint is not stated.",
        what_would_change_the_conclusion="A qualified second source with dated capacity.",
    )


def _submission(
    *issues: PortfolioReviewMaterialIssue,
    requires_human_review: bool = False,
) -> PortfolioReviewAssessmentSubmission:
    return PortfolioReviewAssessmentSubmission(
        material_issues=issues,
        overall_rationale="One bounded reading of the admitted spans.",
        unresolved_questions=(),
        requires_human_review=requires_human_review,
        limitations_acknowledged=True,
    )


FINDING = "FIND-AAPL-OPERATIONS-SUPPLY"

ENTITY = "AAPL"

_CITATIONS: tuple[tuple[str, str], ...] = (
    ("SPAN-S01-R01", "DOC-AAPL-Q3"),
    ("SPAN-S02-R01", "DOC-AAPL-8K"),
    ("SPAN-S03-R01", "DOC-AAPL-PR"),
)

_STRUCTURE_COUNTS: dict[EvidenceStructureState, tuple[int, int]] = {
    EvidenceStructureState.SUPPORTED: (2, 0),
    EvidenceStructureState.SINGLE_SOURCE: (1, 0),
    EvidenceStructureState.CONTESTED: (1, 1),
    EvidenceStructureState.UNSUPPORTED: (0, 0),
}


def _finding(
    *,
    handle: str = FINDING,
    entity: str = ENTITY,
    entities: tuple[str, ...] | None = None,
    structure: EvidenceStructureState = EvidenceStructureState.SUPPORTED,
    summary: str = "Supplier interruption reduced available component capacity.",
) -> PortfolioReviewDossierFinding:
    supporting, contradicting = _STRUCTURE_COUNTS[structure]
    return PortfolioReviewDossierFinding(
        finding_handle=handle,
        affected_entities=entities if entities is not None else (entity,),
        topic=EvidenceTopic.OPERATIONS_SUPPLY,
        lifecycle=EvidenceLifecycle.ONGOING,
        direction=EvidenceDirection.ADVERSE,
        summary=summary,
        supporting_span_handles=tuple(span for span, _ in _CITATIONS[:2])[:supporting],
        contradicting_span_handles=(_CITATIONS[2][0],)[:contradicting],
        limitations=("Recorded official fixture only.",),
        supporting_document_count=supporting,
        contradicting_document_count=contradicting,
        structure=structure,
    )


def _issuer(
    entity: str = ENTITY,
    *,
    band: ExposureBand = ExposureBand.HIGH,
    weight_rank: int = 4,
) -> PortfolioReviewDossierIssuer:
    return PortfolioReviewDossierIssuer(
        entity_id=entity,
        tickers=(entity,),
        ending_weight=0.06,
        signed_change=0.01,
        transition="INCREASED",
        weight_rank=weight_rank,
        exposure_band=band,
        selection_reason="OPENED_OR_INCREASED_BY_POSITIVE_CHANGE",
    )


def _dossier(
    *,
    band: ExposureBand = ExposureBand.HIGH,
    structure: EvidenceStructureState = EvidenceStructureState.SUPPORTED,
    missing_evidence: tuple[str, ...] = (),
    analyst_requires_human_review: bool = False,
    reviewed_ending_weight_coverage: float = 1.0,
    mapping_failure_count: int = 0,
    unavailable_reasons: tuple[str, ...] = (),
    book_authority: BookAuthority = BookAuthority.DEVELOPMENT_RESULT,
    issuers: tuple[PortfolioReviewDossierIssuer, ...] | None = None,
    findings: tuple[PortfolioReviewDossierFinding, ...] | None = None,
    nothing_filed_ending_weight_coverage: float | None = None,
    unreached_ending_weight_coverage: float | None = None,
) -> PortfolioReviewDossier:
    """One dossier, sealed directly, so the route function can be exercised alone."""

    handoff = book_authority is BookAuthority.VALIDATED_HANDOFF
    return seal_contract(
        PortfolioReviewDossier,
        "dossier_hash",
        book_authority=book_authority,
        report_hash="c" * 64,
        result_hash=None if handoff else "a" * 64,
        candidate_hash=None if book_authority is BookAuthority.DEVELOPMENT_RESULT else "b" * 64,
        handoff_hash="f" * 64 if handoff else None,
        issuer_scope_hash="d" * 64,
        exposure_projection_hash="e" * 64,
        registry_hash="1" * 64,
        analysis_publication_hash="0" * 64,
        analyst_brief_hash="2" * 64,
        cro_package_hash="3" * 64,
        obligation_hash="4" * 64,
        evidence_as_of=_NOW,
        evidence_expires_at=_NOW + timedelta(days=1),
        issuers=issuers if issuers is not None else (_issuer(band=band),),
        findings=findings if findings is not None else (_finding(structure=structure),),
        citations=tuple(
            PortfolioReviewDossierCitation(
                span_handle=span,
                document_handle=document,
                entity_id=ENTITY,
                available_at=_NOW - timedelta(days=1),
            )
            for span, document in _CITATIONS
        ),
        coverage=PortfolioReviewCoverage(
            reviewed_ending_weight_coverage=reviewed_ending_weight_coverage,
            reviewed_absolute_change_coverage=1.0,
            mapping_coverage=1.0,
            selected_issuer_coverage=1.0,
            missing_evidence=missing_evidence,
            unavailable_reasons=unavailable_reasons,
            nothing_filed_ending_weight_coverage=nothing_filed_ending_weight_coverage,
            unreached_ending_weight_coverage=unreached_ending_weight_coverage,
            nothing_filed_window_days=(
                None if nothing_filed_ending_weight_coverage is None else 30
            ),
        ),
        analyst_requires_human_review=analyst_requires_human_review,
        mapping_failure_count=mapping_failure_count,
        held_count=5,
        window_end_effective_n=4.2,
        claim_limits=REVIEW_CLAIM_LIMITS,
        limitations=("Host verification proves citation and authority.",),
        portfolio_report_link="/report?result_hash=aaaa",
    )


@dataclass(frozen=True, slots=True)
class ControlledRisk:
    """A reviewer's risk in a test: the issuer whose findings it cites, and
    the judgment written about it."""

    entity: str
    severity: CRORiskSeverity = CRORiskSeverity.HIGH
    confidence: CRORiskConfidence = CRORiskConfidence.SUPPORTED
    why: str = "A named supply constraint reduces deliverable units."
    recommendation: str = "Keep the position under review until the constraint is resolved."


def _findings_of(
    dossier: PortfolioReviewDossier | Mapping[str, Any],
) -> list[tuple[str, tuple[str, ...]]]:
    if isinstance(dossier, PortfolioReviewDossier):
        return [(value.finding_handle, value.affected_entities) for value in dossier.findings]
    return [
        (str(value["finding_handle"]), tuple(value["affected_entities"]))
        for value in dossier["findings"]
    ]


def controlled_answer(
    dossier: PortfolioReviewDossier | Mapping[str, Any],
    *risks: ControlledRisk,
    summary: str = "",
) -> PortfolioReviewAnswer:
    """The answer a controlled reviewer writes: for each risk, every finding
    of its issuer cited by alias (`F1`.. in dossier order). A risk whose
    issuer has no finding in the dossier names nothing and is left out."""

    findings = _findings_of(dossier)
    written = []
    for risk in risks:
        cited = tuple(
            f"F{index}"
            for index, (_, entities) in enumerate(findings, start=1)
            if risk.entity in entities
        )[:8]
        if cited:
            written.append(
                PortfolioReviewAnswerRisk(
                    findings=cited,
                    why=risk.why,
                    severity=risk.severity,
                    confidence=risk.confidence,
                    recommendation=risk.recommendation,
                )
            )
    return PortfolioReviewAnswer(risks=tuple(written), summary=summary)


def issuer_finding(dossier: PortfolioReviewDossier | Mapping[str, Any], entity: str) -> str:
    """The handle of the one finding the Host sealed for `entity`."""

    handles = [handle for handle, entities in _findings_of(dossier) if entity in entities]
    assert len(handles) == 1, (entity, handles)
    return handles[0]


class CitingReviewActor(SubmittedPortfolioReviewActor):
    """A controlled reviewer: handed a dossier, it answers the given risks
    by the aliases of that dossier's findings."""

    def __init__(self, *, risks: tuple[ControlledRisk, ...] = (), **values: Any) -> None:
        super().__init__(answer=PortfolioReviewAnswer(), **values)
        self.risks = risks

    @property
    def process_binding_hash(self) -> str:
        """Its risks are its identity: two controlled reviewers answering
        differently are two reviews, never an exact reuse."""

        return str(
            canonical_hash(
                {
                    "submitted": super().process_binding_hash,
                    "risks": [
                        [r.entity, str(r.severity), str(r.confidence), r.why, r.recommendation]
                        for r in self.risks
                    ],
                }
            )
        )

    def __call__(
        self, *, dossier: PortfolioReviewDossier, deadline_seconds: float
    ) -> PortfolioReviewActorResult:
        result = super().__call__(dossier=dossier, deadline_seconds=deadline_seconds)
        return replace(result, answer=controlled_answer(dossier, *self.risks))


_SEVERITY_WORDS = {
    CROSeverityIfTrue.HIGH: CRORiskSeverity.HIGH,
    CROSeverityIfTrue.MODERATE: CRORiskSeverity.MEDIUM,
    CROSeverityIfTrue.LOW: CRORiskSeverity.LOW,
}


def answer_from_issues(
    dossier: PortfolioReviewDossier | Mapping[str, Any],
    submission: PortfolioReviewAssessmentSubmission,
) -> PortfolioReviewAnswer:
    """What a reviewer answering in the judgment-only format writes for the
    same issues: each adverse, relevant issue as a risk citing the same
    findings by alias. An issue the answer cannot state (not relevant,
    favourable) is simply not named; mitigation is not asked for."""

    alias_of = {handle: f"F{index}" for index, (handle, _) in enumerate(_findings_of(dossier), 1)}
    risks = []
    for issue in submission.material_issues:
        if (
            issue.relevance is CROEvidenceRelevance.NOT_RELEVANT
            or issue.position_impact_direction
            in {
                CROPositionImpactDirection.FAVOURABLE,
                CROPositionImpactDirection.NONE,
            }
        ):
            continue
        risks.append(
            PortfolioReviewAnswerRisk(
                findings=tuple(
                    alias_of.get(handle, "F9999") for handle in issue.cited_finding_handles
                ),
                why=issue.causal_channel,
                severity=_SEVERITY_WORDS[issue.severity_if_true],
                confidence=CRORiskConfidence(issue.evidence_interpretation.value)
                if issue.evidence_interpretation is not CROEvidenceInterpretation.INSUFFICIENT
                else CRORiskConfidence.LIMITED,
                recommendation=issue.what_would_change_the_conclusion or "Review the position.",
            )
        )
    return PortfolioReviewAnswer(risks=tuple(risks), summary=submission.overall_rationale)


def answer_body(dossier: Mapping[str, Any], submission: Mapping[str, Any]) -> dict[str, Any]:
    """The request body of the answer stating `submission`'s issues against an
    exported dossier: what a controlled reviewer posts."""

    return answer_from_issues(
        dossier, PortfolioReviewAssessmentSubmission.model_validate(submission)
    ).model_dump(mode="json")


def _seal(
    dossier: PortfolioReviewDossier,
    written: PortfolioReviewAnswer | PortfolioReviewAssessmentSubmission,
    *,
    evidence_is_current: bool = True,
    actor_kind: ActorKind = ActorKind.HUMAN,
    agent_execution: AgentExecutionBinding | None = None,
    dropped: tuple[AnswerProblem, ...] = (),
) -> tuple[PortfolioReviewReceipt, PortfolioReviewRecommendation]:
    """Seal through the real sealer. An answer is sealed as written; issues
    built with `_issue` are first written as the answer that states them."""

    answer = (
        written
        if isinstance(written, PortfolioReviewAnswer)
        else answer_from_issues(dossier, written)
    )
    return seal_portfolio_review(
        dossier=dossier,
        answer=answer,
        evidence_is_current=evidence_is_current,
        decision_policy=build_portfolio_review_policy_binding(REPO_ROOT),
        playpen_root=REPO_ROOT,
        actor_kind=actor_kind,
        actor_id="gate-9c5",
        agent_execution=agent_execution,
        dropped=dropped,
    )
