"""Host validation and sealing for the actor-neutral Portfolio evidence review.

A reviewer answers with judgment only (`PortfolioReviewAnswer`): the Host
screens each risk on its own against the dossier it was given, maps the
view's finding aliases back to handles, derives what the answer does not say
and seals the normalized assessment with the route.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from alphalattice.kernel.shared_kernel.domain.serialization import canonical_json_bytes
from alphalattice.kernel.shared_kernel.identity import canonical_hash, schema_structure
from alphalattice.kernel.shared_kernel.source_identity import source_rule_closure_hash
from alphalattice.protocols.actor_execution import (
    ActorKind,
    AgentExecutionBinding,
    seal_actor_submission,
)
from alphalattice.protocols.actor_execution.answers import (
    AnswerProblem,
    ScreenedAnswer,
    screen_answer,
)

from .portfolio_review import (
    PORTFOLIO_REVIEW_ROUTE_POLICY,
    REVIEW_ANSWER_TEXT_FIELDS,
    CROEvidenceInterpretation,
    CROEvidenceRelevance,
    CROPositionImpactDirection,
    CRORiskSeverity,
    CROSeverityIfTrue,
    PortfolioReviewAnswer,
    PortfolioReviewAnswerResolution,
    PortfolioReviewAnswerRisk,
    PortfolioReviewAssessmentSubmission,
    PortfolioReviewDossier,
    PortfolioReviewIssueResolution,
    PortfolioReviewMaterialIssue,
    PortfolioReviewReceipt,
    PortfolioReviewRecommendation,
    compile_portfolio_review_recommendation,
    finding_aliases,
    open_issue_aliases,
    review_basis,
    route_portfolio_review,
)


class CRODecisionAuthorityError(ValueError):
    """A CRO submission exceeded the frozen Host dossier authority."""


PORTFOLIO_REVIEW_POLICY_ROLE = "chief_risk_officer.portfolio_review_policy"
"""The identity role of the review policy's binding (`config/identity-roles.json`): a
recorded `decision_policy_hash` names the installed policy when recorded moves lead to it."""


class CROHostPolicyBinding(BaseModel):  # type: ignore[misc]
    """Host-owned policy rebuilt beside the sealer that enforces it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["CROHostPolicyBinding"] = "CROHostPolicyBinding"
    owner_id: str = Field(min_length=1, max_length=128)
    policy_version: Literal["2026-09-02"] = "2026-09-02"
    contract_schema_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    validation_source_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    semantics_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_binding(self) -> Self:
        """Reject a policy binding whose hash does not seal its fields."""
        if self.binding_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"binding_hash"})
        ):
            raise ValueError("chief_risk_officer.host_policy_binding_invalid")
        return self


def build_portfolio_review_policy_binding(playpen_root: Path) -> CROHostPolicyBinding:
    """The Host policy for the Portfolio evidence review, rebuilt beside every seal."""
    values = {
        "kind": "CROHostPolicyBinding",
        "owner_id": "chief_risk_officer.portfolio_review",
        "policy_version": "2026-09-02",
        "contract_schema_hash": canonical_hash(
            {
                "dossier": schema_structure(PortfolioReviewDossier),
                "answer": schema_structure(PortfolioReviewAnswer),
                "submission": schema_structure(PortfolioReviewAssessmentSubmission),
                "receipt": schema_structure(PortfolioReviewReceipt),
                "recommendation": schema_structure(PortfolioReviewRecommendation),
            }
        ),
        "validation_source_hash": source_rule_closure_hash(
            root=playpen_root.resolve(),
            tracked_paths=(
                "src/alphalattice/oversight/chief_risk_officer/decision/portfolio_review.py",
                "src/alphalattice/oversight/chief_risk_officer/decision/submissions.py",
            ),
            semantic_owner="chief_risk_officer",
            numerical_role="CRO_HOST_POLICY",
        ),
        "semantics_hash": canonical_hash(
            {
                "authority": "interpret cited evidence; the Host alone selects the route",
                "answer": (
                    "the reviewer names risks with judgment fields only; the Host derives the "
                    "issuers, exposure, relevance and direction, assigns the handles and asks "
                    "for no disposition; a subset is an answer"
                ),
                "route_policy": PORTFOLIO_REVIEW_ROUTE_POLICY,
                "structure": "the finding's citation structure caps the stated interpretation",
                "absence": (
                    "missing evidence is a stated limitation of the review; it never asserts "
                    "an adverse fact and never gates the route"
                ),
                "route": (
                    "high severity on an exposed holding objects when supported and asks a "
                    "person (advisory) when contested or limited; other material risks are "
                    "accepted with limits; expired evidence is refused, never routed"
                ),
                "activation": "every route is a recommendation and never an action",
            }
        ),
    }
    return CROHostPolicyBinding(**values, binding_hash=canonical_hash(values))


MAXIMUM_REVIEW_SUBMISSION_BYTES = 256 * 1024
"""The whole serialized answer; one past this is refused by name."""

MAXIMUM_ANSWER_RISKS = 16
"""The risks one answer holds, as the answer model bounds them."""

MAXIMUM_ANSWER_RESOLUTIONS = next(
    bound.max_length
    for bound in PortfolioReviewAnswer.model_fields["resolved"].metadata
    if isinstance(getattr(bound, "max_length", None), int)
)
"""The open issues one answer resolves, the answer model's own bound, which the screen reads
so the one past it is corrected by its author, never refused as the Host's limit."""

_SEVERITY: dict[CRORiskSeverity, CROSeverityIfTrue] = {
    CRORiskSeverity.HIGH: CROSeverityIfTrue.HIGH,
    CRORiskSeverity.MEDIUM: CROSeverityIfTrue.MODERATE,
    CRORiskSeverity.LOW: CROSeverityIfTrue.LOW,
}


@dataclass(frozen=True, slots=True)
class ScreenedReviewAnswer(ScreenedAnswer[PortfolioReviewAnswerRisk]):
    """Record screened risks, texts, and open-issue resolutions.

    Each resolution is accepted on its own (W3).
    """

    resolved: tuple[PortfolioReviewAnswerResolution, ...] = ()


def screen_review_answer(raw: object, *, dossier: PortfolioReviewDossier) -> ScreenedReviewAnswer:
    """Check each written risk and resolution on its own against the dossier.

    A risk is acceptable when every finding alias it cites is one of the
    view's, its findings name at most eight issuers, and the open issue it
    states again, if any, is one of the view's and no other risk's. A
    resolution is acceptable when its open issue is one of the view's, no
    risk states it again, and every finding it cites is one of the view's.
    Nothing omitted is a problem -- an open issue the reviewer does not
    address stands as last assessed; every problem is stated in plain words.
    """
    aliases = finding_aliases(dossier)
    opened = open_issue_aliases(dossier)
    body = raw.model_dump(mode="json") if isinstance(raw, BaseModel) else raw
    written_resolutions: object = ()
    if isinstance(body, dict) and "resolved" in body:
        body = dict(body)
        written_resolutions = body.pop("resolved")
    carried = [
        str(value.get("carries", "")).strip().upper()
        for value in (body.get("risks") or [] if isinstance(body, dict) else [])
        if isinstance(value, dict) and value.get("carries")
    ]

    def check(item: PortfolioReviewAnswerRisk) -> list[str]:
        problems: list[str] = []
        entities: list[str] = []
        for alias in _aliases(item.findings):
            handle = aliases.get(alias)
            if handle is None:
                problems.append(f"{alias} is not a finding of this bundle.")
            else:
                entities.extend(dossier.finding(handle).affected_entities)
        if len(dict.fromkeys(entities)) > 8:
            problems.append(
                "the cited findings name more than eight issuers; split the risk so each "
                "names at most eight."
            )
        if item.carries:
            if item.carries not in opened:
                problems.append(f"{item.carries} is not an open issue of this bundle.")
            elif carried.count(item.carries) > 1:
                problems.append(f"{item.carries} is stated again by more than one risk.")
        return problems

    screened = screen_answer(
        body,
        items_field="risks",
        item_model=PortfolioReviewAnswerRisk,
        maximum_items=MAXIMUM_ANSWER_RISKS,
        check_item=check,
        text_fields=REVIEW_ANSWER_TEXT_FIELDS,
    )
    problems = list(screened.problems)
    resolved: list[PortfolioReviewAnswerResolution] = []
    if not isinstance(written_resolutions, (list, tuple)):
        problems.append(AnswerProblem(text="'resolved' must be a list."))
        written_resolutions = ()
    if len(written_resolutions) > MAXIMUM_ANSWER_RESOLUTIONS:
        # As a list of risks past its bound: the first are read and the author corrects.
        problems.append(
            AnswerProblem(
                text=(
                    f"The answer has {len(written_resolutions)} resolutions; only the first "
                    f"{MAXIMUM_ANSWER_RESOLUTIONS} are read."
                )
            )
        )
        written_resolutions = written_resolutions[:MAXIMUM_ANSWER_RESOLUTIONS]
    for number, value in enumerate(written_resolutions, start=1):
        try:
            resolution = PortfolioReviewAnswerResolution.model_validate(value)
        except ValidationError:
            problems.append(
                AnswerProblem(
                    text=f"resolution {number}: write an open issue (O1..), the findings that "
                    "resolve it and why, at most 600 characters."
                )
            )
            continue
        wrong = [alias for alias in _aliases(resolution.findings) if alias not in aliases] + (
            [] if resolution.issue in opened else [resolution.issue]
        )
        if wrong or resolution.issue in carried:
            problems.append(
                AnswerProblem(
                    text=f"resolution {number}: "
                    + (
                        f"{', '.join(wrong)} is not in this bundle."
                        if wrong
                        else f"{resolution.issue} is also stated again as a risk; keep one."
                    )
                )
            )
            continue
        resolved.append(resolution.model_copy(update={"findings": _aliases(resolution.findings)}))
    return ScreenedReviewAnswer(
        items=screened.items,
        texts=screened.texts,
        problems=tuple(problems),
        resolved=tuple(resolved),
    )


def accepted_review_answer(
    screened: ScreenedAnswer[PortfolioReviewAnswerRisk],
) -> PortfolioReviewAnswer:
    """The screen's accepted part, each alias once and upper-cased."""
    return PortfolioReviewAnswer(
        risks=tuple(
            item.model_copy(update={"findings": _aliases(item.findings)})
            for _, item in screened.items
        ),
        summary=screened.texts.get("summary", ""),
        resolved=getattr(screened, "resolved", ()),
    )


def normalize_review_answer(
    answer: PortfolioReviewAnswer, *, dossier: PortfolioReviewDossier
) -> PortfolioReviewAssessmentSubmission:
    """The sealed assessment, written by the Host from an answer.

    Each risk becomes the material issue of its place (`ISSUE-001`..): the
    cited aliases become handles, the affected issuers are the cited
    findings' own, a cited finding of a book position is direct, a risk is
    adverse, the reason is the causal channel, `MEDIUM` is the policy's
    `MODERATE`, the stated confidence is the stated interpretation (the
    route caps it by structure) and the advice is kept in words. No
    disposition, mitigation or review flag is asked for or invented.
    """
    screened = screen_review_answer(answer.model_dump(mode="json"), dossier=dossier)
    if not screened.clean:
        first = screened.problems[0]
        raise CRODecisionAuthorityError(f"chief_risk_officer.answer_invalid:item_{first.item or 0}")
    aliases = finding_aliases(dossier)
    opened = open_issue_aliases(dossier)
    issues = []
    for index, risk in enumerate(answer.risks, start=1):
        handles = tuple(aliases[value] for value in risk.findings)
        entities = tuple(
            dict.fromkeys(
                entity for handle in handles for entity in dossier.finding(handle).affected_entities
            )
        )
        issues.append(
            PortfolioReviewMaterialIssue(
                issue_handle=f"ISSUE-{index:03d}",
                cited_finding_handles=handles,
                affected_entities=entities,
                relevance=CROEvidenceRelevance.DIRECT,
                position_impact_direction=CROPositionImpactDirection.ADVERSE,
                causal_channel=risk.why,
                severity_if_true=_SEVERITY[risk.severity],
                evidence_interpretation=CROEvidenceInterpretation(risk.confidence.value),
                recommendation=risk.recommendation,
                open_issue_handle=opened[risk.carries] if risk.carries else None,
            )
        )
    resolutions = tuple(
        PortfolioReviewIssueResolution(
            open_issue_handle=opened[value.issue],
            resolving_finding_handles=tuple(aliases[alias] for alias in value.findings),
            why=value.why,
        )
        for value in answer.resolved
    )
    # An open issue the reviewer neither states again nor resolves stands as
    # last assessed: carried, never dropped (W3).
    addressed = {value.open_issue_handle for value in issues} | {
        value.open_issue_handle for value in resolutions
    }
    for value in dossier.open_issues:
        if value.open_issue_handle in addressed:
            continue
        issues.append(
            PortfolioReviewMaterialIssue(
                issue_handle=f"ISSUE-{len(issues) + 1:03d}",
                cited_finding_handles=value.cited_finding_handles,
                affected_entities=value.affected_entities,
                relevance=CROEvidenceRelevance.DIRECT,
                position_impact_direction=CROPositionImpactDirection.ADVERSE,
                causal_channel=value.causal_channel,
                severity_if_true=value.severity_if_true,
                evidence_interpretation=value.evidence_interpretation,
                recommendation=value.recommendation,
                open_issue_handle=value.open_issue_handle,
                carried_on=value.assessed_on,
            )
        )
    count = len(issues) - sum(1 for value in issues if value.carried_on is not None)
    rationale = answer.summary or (
        f"{count} risk{'' if count == 1 else 's'} named in the evidence read."
        if count
        else "No major negative was named in the evidence read."
    )
    return PortfolioReviewAssessmentSubmission(
        material_issues=tuple(issues),
        overall_rationale=rationale,
        unresolved_questions=(),
        requires_human_review=False,
        limitations_acknowledged=True,
        resolved_issues=resolutions,
    )


def _aliases(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value.strip().upper() for value in values))


def seal_portfolio_review(
    *,
    dossier: PortfolioReviewDossier,
    answer: PortfolioReviewAnswer,
    evidence_is_current: bool,
    decision_policy: CROHostPolicyBinding,
    playpen_root: Path,
    actor_kind: ActorKind,
    actor_id: str,
    dropped: tuple[AnswerProblem, ...] = (),
    agent_execution: AgentExecutionBinding | None = None,
    model_call_count: int = 0,
    protocol_repair_count: int = 0,
    carried: PortfolioReviewAssessmentSubmission | None = None,
) -> tuple[PortfolioReviewReceipt, PortfolioReviewRecommendation]:
    """Normalize the answer against its dossier, then apply the one route function.

    `answer` is the accepted part of what the reviewer wrote; one citing a
    finding this dossier does not contain is refused rather than routed.
    `dropped` names what was not accepted and why. The receipt records the
    basis the assessment rests on, so a later dossier on the same basis finds
    it (W3); `carried` is that earlier assessment when it carries
    forward, routed on this dossier as it was sealed (in its handles).
    """
    decision_policy = CROHostPolicyBinding.model_validate(decision_policy)
    if decision_policy != build_portfolio_review_policy_binding(playpen_root):
        raise CRODecisionAuthorityError("chief_risk_officer.host_policy_unauthorized")
    dossier = PortfolioReviewDossier.model_validate(dossier)
    answer = PortfolioReviewAnswer.model_validate(answer)
    size = len(canonical_json_bytes(answer.model_dump(mode="json")))
    if size > MAXIMUM_REVIEW_SUBMISSION_BYTES:
        raise CRODecisionAuthorityError(
            f"chief_risk_officer.submission_bytes_exceeded:{size}>{MAXIMUM_REVIEW_SUBMISSION_BYTES}"
        )
    submission = carried or normalize_review_answer(answer, dossier=dossier)

    outcome = route_portfolio_review(
        dossier=dossier,
        submission=submission,
        evidence_is_current=evidence_is_current,
        dropped=dropped,
    )
    actor_submission = seal_actor_submission(
        actor_kind=actor_kind,
        actor_id=actor_id,
        submission_hash=canonical_hash(answer.model_dump(mode="json")),
        agent_execution=agent_execution,
    )
    from alphalattice.evidence.alternative_evidence.contracts import seal_contract

    receipt = seal_contract(
        PortfolioReviewReceipt,
        "receipt_hash",
        dossier_hash=dossier.dossier_hash,
        submission=submission,
        answer=answer,
        dropped=dropped,
        actor_submission=actor_submission,
        outcome=outcome,
        decision_policy_hash=decision_policy.binding_hash,
        model_call_count=model_call_count,
        protocol_repair_count=protocol_repair_count,
        review_basis=review_basis(dossier, decision_policy_hash=decision_policy.binding_hash),
    )
    return receipt, compile_portfolio_review_recommendation(dossier=dossier, receipt=receipt)


__all__ = [
    "MAXIMUM_REVIEW_SUBMISSION_BYTES",
    "PORTFOLIO_REVIEW_POLICY_ROLE",
    "CRODecisionAuthorityError",
    "CROHostPolicyBinding",
    "ScreenedReviewAnswer",
    "accepted_review_answer",
    "build_portfolio_review_policy_binding",
    "normalize_review_answer",
    "screen_review_answer",
    "seal_portfolio_review",
]
