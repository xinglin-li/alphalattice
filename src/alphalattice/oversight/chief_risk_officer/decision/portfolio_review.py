"""The Portfolio evidence review: one bounded judgment, one deterministic route.

Three things are deliberately separated here:

- **What the actor may say.** `PortfolioReviewAnswer` names risks -- which
  findings, why they matter to the book, how severe if true, how well supported,
  advice in words -- and nothing else. It has no numeric field, so no prompt and
  no model can smuggle a probability, an expected loss or a target weight
  through it. The Host normalizes it into `PortfolioReviewAssessmentSubmission`
  (issuers, relevance and direction derived, handles assigned), the form every
  sealed receipt holds.
- **What the Host decides.** `route_portfolio_review` is a pure versioned
  function over those primitives plus deterministic structure and exposure
  facts. It judges first: the route comes from the named risks and the book's
  exposure, and coverage, gaps and findings nobody named are limitations the
  program states beside it, never a gate. The actor never selects the route,
  and its stated confidence is capped by how the finding is actually held up
  by its citations.
- **What the product publishes.** A pointer-free recommendation that separates
  the disposition, how complete the review was, and what must happen next. The
  required action depends on which kind of sealed book was reviewed: an
  objection to a pre-freeze book asks Portfolio to reconsider the candidate; an
  objection to a validated handoff declines activation. Neither changes a weight.

Missing evidence is a stated limitation: it never manufactures an adverse
fact and never stops the review. Expired evidence is refused, never routed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.evidence.alternative_evidence.analysis.contracts import (
    EvidenceDirection,
    EvidenceLifecycle,
    EvidenceStructureState,
    EvidenceTopic,
)
from alphalattice.evidence.alternative_evidence.contracts import AlternativeEvidenceContract
from alphalattice.oversight.chief_risk_officer.portfolio_evidence.contracts import (
    BookAuthority,
    ExposureBand,
    PortfolioExperimentReviewSubject,
    PortfolioUpdateReviewSubject,
    validate_review_book_authority,
)
from alphalattice.protocols.actor_execution import ActorKind, ActorSubmissionBinding
from alphalattice.protocols.actor_execution.answers import AnswerProblem

_HASH = r"^[0-9a-f]{64}$"

PORTFOLIO_REVIEW_ROUTE_POLICY: str = "chief_risk_officer.portfolio-evidence-review"
"""Stable policy identity; the receipt's decision_policy_hash binds its implementation.

The serialized field remains named policy_version for historical compatibility;
new behavior is not named by incrementing an implementation suffix.
"""

READABLE_PORTFOLIO_REVIEW_ROUTE_POLICIES: tuple[str, ...] = (PORTFOLIO_REVIEW_ROUTE_POLICY,)
"""The name a sealed review may carry. The numbered names before it
(`...portfolio-review-route.v2`, `.v3`) sealed only reviews the QA copies hold,
which are history (D2, 2026-09-23); the stable name, in use since 2026-09-09,
does not grant reuse across a changed decision_policy_hash."""

MINIMUM_REVIEWED_ENDING_WEIGHT_COVERAGE: float = 0.60
"""The reviewed share of the book's ending weight a review is designed for:
below it the program says so among the limitations; it no longer gates."""


class PortfolioReviewRoute(StrEnum):
    """The whole public route grammar. There is no sixth answer."""

    NO_MATERIAL_OBJECTION = "NO_MATERIAL_OBJECTION"
    ACCEPT_WITH_LIMITS = "ACCEPT_WITH_LIMITS"
    REQUEST_EVIDENCE_REFRESH = "REQUEST_EVIDENCE_REFRESH"
    HUMAN_REVIEW_REQUIRED = "HUMAN_REVIEW_REQUIRED"
    MATERIAL_OBJECTION = "MATERIAL_OBJECTION"


_ROUTE_PRIORITY: dict[PortfolioReviewRoute, int] = {
    PortfolioReviewRoute.NO_MATERIAL_OBJECTION: 0,
    PortfolioReviewRoute.ACCEPT_WITH_LIMITS: 1,
    PortfolioReviewRoute.REQUEST_EVIDENCE_REFRESH: 2,
    PortfolioReviewRoute.HUMAN_REVIEW_REQUIRED: 3,
    PortfolioReviewRoute.MATERIAL_OBJECTION: 4,
}


class ReviewState(StrEnum):
    """How complete the review was. Separate from what it concluded."""

    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    UNAVAILABLE = "UNAVAILABLE"


class RequiredActionKind(StrEnum):
    """Follow-up action a Portfolio evidence review may require."""

    NONE = "NONE"
    REFRESH_EVIDENCE = "REFRESH_EVIDENCE"
    RESOLVE_ISSUER_MAPPING = "RESOLVE_ISSUER_MAPPING"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    RECONSIDER_CANDIDATE = "RECONSIDER_CANDIDATE"
    DO_NOT_ACTIVATE = "DO_NOT_ACTIVATE"


class RequiredAction(BaseModel):  # type: ignore[misc]
    """Record a required follow-up and whether it blocks the reviewed decision.

    Attributes:
        action: Kind of follow-up required by the review.
        entity_id: Issuer the action concerns, when it has issuer scope.
        reason: Evidence-based explanation for the action.
        blocking: Whether the action must be resolved before proceeding.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: RequiredActionKind
    entity_id: str | None = Field(default=None, min_length=1, max_length=32)
    reason: str = Field(min_length=1, max_length=400)
    blocking: bool


class CROEvidenceRelevance(StrEnum):
    """Relationship of an evidence finding to a reviewed issuer or position."""

    DIRECT = "DIRECT"
    INDIRECT = "INDIRECT"
    NOT_RELEVANT = "NOT_RELEVANT"


class CROPositionImpactDirection(StrEnum):
    """Direction of an evidence finding's impact on a reviewed position."""

    ADVERSE = "ADVERSE"
    FAVOURABLE = "FAVOURABLE"
    AMBIGUOUS = "AMBIGUOUS"
    NONE = "NONE"


class CROSeverityIfTrue(StrEnum):
    """Severity *if the cited claim is true*, which is not a probability."""

    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"


class CROEvidenceInterpretation(StrEnum):
    """Support state assigned to the evidence behind an issuer issue."""

    SUPPORTED = "SUPPORTED"
    LIMITED = "LIMITED"
    CONTESTED = "CONTESTED"
    INSUFFICIENT = "INSUFFICIENT"


class CROPortfolioMitigation(StrEnum):
    """Reported strength of Portfolio mitigation for an issuer issue."""

    NONE = "NONE"
    PARTIAL = "PARTIAL"
    STRONG = "STRONG"
    UNKNOWN = "UNKNOWN"


class CRORiskSeverity(StrEnum):
    """How severe a risk would be for the book if its cited findings are true."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class CRORiskConfidence(StrEnum):
    """State how well the cited findings support a risk.

    The reviewer chooses the confidence; the Host caps it by the findings'
    evidence structure.
    """

    SUPPORTED = "SUPPORTED"
    CONTESTED = "CONTESTED"
    LIMITED = "LIMITED"


class PortfolioReviewAnswerRisk(BaseModel):  # type: ignore[misc]
    """One risk as the reviewer writes it: judgment only.

    `findings` are the aliases of the cited findings; `recommendation` is
    advice in words, never a weight, an order or a route.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    findings: tuple[str, ...] = Field(min_length=1, max_length=8)
    why: str = Field(min_length=1, max_length=600)
    severity: CRORiskSeverity
    confidence: CRORiskConfidence
    recommendation: str = Field(min_length=1, max_length=600)
    carries: str = Field(
        default="", pattern=r"^(O[0-9]{1,3})?$", exclude_if=lambda value: value == ""
    )
    """The open issue (`O1`, `O2`..) this risk states again, as the reviewer now
    assesses it (W3); empty -- and absent -- for a new risk, so the answer's
    schema still holds no number and no null."""


class PortfolioReviewAnswerResolution(BaseModel):  # type: ignore[misc]
    """An open issue the reviewer closes, with the findings that resolve it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    issue: str = Field(pattern=r"^O[0-9]{1,3}$")
    findings: tuple[str, ...] = Field(min_length=1, max_length=8)
    why: str = Field(min_length=1, max_length=600)


class PortfolioReviewAnswer(BaseModel):  # type: ignore[misc]
    """Report real major negatives in Analyst findings for the reviewed book.

    The answer carries no hash, handle, route, or disposition. The Host derives
    the affected issuers, exposure, relevance and direction, assigns handles,
    caps confidence, and routes. A subset is an answer; an empty list says no
    major negative was found in the evidence read.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    risks: tuple[PortfolioReviewAnswerRisk, ...] = Field(default=(), max_length=16)
    summary: str = Field(default="", max_length=2400, exclude_if=lambda v: v == "")
    resolved: tuple[PortfolioReviewAnswerResolution, ...] = Field(
        default=(), max_length=16, exclude_if=lambda value: value == ()
    )
    """Open issues the reviewer closes, each citing what resolves it."""


REVIEW_ANSWER_TEXT_FIELDS: dict[str, int] = {"summary": 2400}
"""The answer's optional texts and their bounds, beside its `risks`."""


OPEN_ISSUE_PATTERN = r"^OPEN-[0-9A-F]{12}$"

MAXIMUM_AFFECTED_ISSUERS = 8
"""The held issuers one finding, open issue or material issue names."""

MAXIMUM_REVIEWER_ISSUES = 16
"""What one reviewer states in one review: an answer's risks, an assessment's own issues."""

MAXIMUM_OPEN_ISSUES = 256
"""The issuers' register a dossier carries, and so what an assessment carries as last
assessed beside its own issues (W3)."""


class PortfolioReviewOpenIssue(BaseModel):  # type: ignore[misc]
    """Keep a CRO issue open until a review resolves it with evidence.

    The issue was raised on a held issuer and no review has resolved it (W3).
    It carries the CRO's last statement and does not age out with its filing.
    Every book holding the issuer reads it until resolution; a review silent
    about it counts as the last assessment of the retained issue.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    open_issue_handle: str = Field(pattern=OPEN_ISSUE_PATTERN)
    affected_entities: tuple[str, ...] = Field(min_length=1, max_length=MAXIMUM_AFFECTED_ISSUERS)
    """The book's holdings among the issuers it names."""
    raised_on: date
    assessed_on: date
    """The cutoff of the review that last stated it."""
    cited_finding_handles: tuple[str, ...] = Field(min_length=1, max_length=8)
    """The findings it rests on, by this dossier's handles."""
    causal_channel: str = Field(min_length=1, max_length=600)
    severity_if_true: CROSeverityIfTrue
    evidence_interpretation: CROEvidenceInterpretation
    recommendation: str = Field(default="", max_length=600)


class PortfolioReviewDossierIssuer(BaseModel):  # type: ignore[misc]
    """One reviewed issuer, exactly as the issuer scope described it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    entity_id: str = Field(min_length=1, max_length=32)
    tickers: tuple[str, ...] = Field(min_length=1)
    ending_weight: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    signed_change: float = Field(allow_inf_nan=False)
    transition: Literal["OPENED", "INCREASED", "HELD", "REDUCED", "EXITED"]
    weight_rank: int = Field(ge=1)
    exposure_band: ExposureBand
    selection_reason: str = Field(min_length=1, max_length=64)
    review_state: str = Field(
        default="UNSTATED", max_length=32, exclude_if=lambda v: v == "UNSTATED"
    )
    """`issuer-review-state-v1`: `EXECUTED_WITH_FINDINGS`, `EXECUTED_NO_FINDINGS`,
    `NO_SPANS_DELIVERED` or `SOURCE_MISSING` from the analysis's completion;
    `NOTHING_FILED` for a holding whose filing index at the cutoff held nothing
    in the window, which no analysis read and no child carries; `UNSTATED`
    (absent) for dossiers compiled from briefs sealed before it."""


class PortfolioReviewUnresolvedQuestion(BaseModel):  # type: ignore[misc]
    """One question the Analyst left open, carried with the unit that asked it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    unit_id: str = Field(pattern=r"^u[0-9]{2,3}$")
    text: str = Field(min_length=1, max_length=600)


class PortfolioReviewFindingDisposition(BaseModel):  # type: ignore[misc]
    """Record a finding's explicit or derived review disposition.

    Before typed answers, every finding needed a disposition when any were
    given, and a deferred finding was not reviewed. For an answer, the program
    derives MATERIAL_ISSUE for a finding cited by a risk and NOT_ADDRESSED for
    one read but not named.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    finding_handle: str = Field(min_length=1, max_length=80)
    disposition: Literal[
        "MATERIAL_ISSUE", "NOT_MATERIAL", "RESOLVED_WITH_EVIDENCE", "DEFERRED", "NOT_ADDRESSED"
    ]
    rationale: str = Field(min_length=1, max_length=600)
    evidence_span_handles: tuple[str, ...] = Field(default=(), max_length=8)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_disposition(self) -> Self:
        """Reject a finding disposition with inconsistent evidence or judgment fields."""
        if self.disposition == "RESOLVED_WITH_EVIDENCE" and not self.evidence_span_handles:
            raise ValueError("chief_risk_officer.disposition_evidence_required")
        if len(set(self.evidence_span_handles)) != len(self.evidence_span_handles):
            raise ValueError("chief_risk_officer.disposition_span_duplicate")
        return self


class PortfolioReviewDossierFinding(BaseModel):  # type: ignore[misc]
    """One source-grounded finding with the structure the Host derived for it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    finding_handle: str = Field(pattern=r"^FIND-[A-Z0-9-]{1,80}$")
    """An analyst's handle, or -- in a dossier that aggregates more than one
    evidence publication -- that handle qualified by its publication
    (`FIND-P<8 hex>-...`), so two units' local names never collide."""
    affected_entities: tuple[str, ...] = Field(min_length=1, max_length=MAXIMUM_AFFECTED_ISSUERS)
    topic: EvidenceTopic
    lifecycle: EvidenceLifecycle | None = Field(default=None, exclude_if=lambda v: v is None)
    """None when the Analyst's answer stated none; absent then, so earlier
    dossiers keep their hashes."""
    direction: EvidenceDirection
    summary: str = Field(min_length=1, max_length=1200)
    supporting_span_handles: tuple[str, ...] = Field(max_length=8)
    contradicting_span_handles: tuple[str, ...] = Field(max_length=8)
    limitations: tuple[str, ...]
    supporting_document_count: int = Field(ge=0, le=8)
    contradicting_document_count: int = Field(ge=0, le=8)
    structure: EvidenceStructureState


class PortfolioReviewDossierCitation(BaseModel):  # type: ignore[misc]
    """Locate an issuer citation and record when its evidence became available.

    Attributes:
        span_handle: Evidence span referenced by a dossier finding.
        document_handle: Document containing that span.
        entity_id: Issuer whose evidence the citation supports.
        available_at: Time the cited evidence became available.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    span_handle: str = Field(min_length=1, max_length=160)
    document_handle: str = Field(min_length=1, max_length=160)
    entity_id: str = Field(min_length=1, max_length=32)
    available_at: datetime


class PortfolioReviewEvidenceChild(BaseModel):  # type: ignore[misc]
    """One unit's evidence publication inside a whole-book dossier.

    A book wider than one evidence request is prepared and analysed as units
    of at most eight issuers; the dossier names every unit's exact publication
    so the review, the export and the currency check reach each child by its
    own handle, and no child is lost behind an aggregate.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    unit_id: str = Field(pattern=r"^[uc][0-9]{2,3}$")
    """`u..`: a unit read at the dossier's cutoff; `c..`: an earlier reading
    carried because nothing new was filed (W3)."""
    ordered_entity_ids: tuple[str, ...] = Field(max_length=8)
    obligation_hash: str = Field(pattern=_HASH)
    analysis_publication_hash: str = Field(pattern=_HASH)
    analyst_brief_hash: str = Field(pattern=_HASH)
    cro_package_hash: str = Field(pattern=_HASH)
    evidence_as_of: datetime
    evidence_expires_at: datetime
    """A unit read at the cutoff: its analysis's own expiry. A carried reading
    is current while its filings stay in the window, whatever its analysis's
    expiry: the day its earliest carried filing leaves the window, or the
    window from the cutoff when it is carried only for an open issue (Z1). The
    dossier's expiry is the earliest of its children's."""
    read_as_of: datetime | None = Field(default=None, exclude_if=lambda value: value is None)
    """When a carried reading was made: its analysis's cutoff, before the
    dossier's; absent on a unit read at the cutoff, so every earlier dossier
    keeps its hash."""

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_child(self) -> Self:
        """Reject a child citation that exceeds the dossier's evidence bounds."""
        carried = self.unit_id.startswith("c")
        if carried != (self.read_as_of is not None) or (
            not carried and not self.ordered_entity_ids
        ):
            raise ValueError("chief_risk_officer.dossier_children_invalid")
        if self.read_as_of is not None and (
            self.read_as_of > self.evidence_as_of or self.evidence_expires_at < self.evidence_as_of
        ):
            raise ValueError("chief_risk_officer.dossier_children_invalid")
        return self


class PortfolioReviewCoverage(BaseModel):  # type: ignore[misc]
    """Four coverage components and their reasons. Never one score."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reviewed_ending_weight_coverage: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    reviewed_absolute_change_coverage: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    mapping_coverage: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    selected_issuer_coverage: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    missing_evidence: tuple[str, ...] = ()
    unavailable_reasons: tuple[str, ...] = ()
    nothing_filed_ending_weight_coverage: float | None = Field(
        default=None, ge=0.0, le=1.0, allow_inf_nan=False, exclude_if=lambda v: v is None
    )
    """The share of the book's ending weight in holdings that filed nothing in
    the window: read as nothing filed, never as no risk. Absent, with the two
    below, from dossiers that name no such holding."""
    unreached_ending_weight_coverage: float | None = Field(
        default=None, ge=0.0, le=1.0, allow_inf_nan=False, exclude_if=lambda v: v is None
    )
    """The share neither read nor filed nothing: exactly 0.0 when every holding is
    accounted for, which is what a complete review asks of the weight."""
    nothing_filed_window_days: int | None = Field(
        default=None, ge=1, exclude_if=lambda v: v is None
    )


class PortfolioReviewDossier(AlternativeEvidenceContract):
    """Everything the review is allowed to stand on, assembled by the Host.

    It *links* to the sole Portfolio report rather than restating it. The only
    Portfolio facts here are the book's own holdings, changes, ranks and bands
    plus two window-level concentration facts; no return, Risk, turnover, cost
    or Validation metric is copied.
    """

    kind: Literal["PortfolioReviewDossier"] = "PortfolioReviewDossier"
    decision_semantics: Literal["PORTFOLIO_EVIDENCE_REVIEW"] = "PORTFOLIO_EVIDENCE_REVIEW"
    book_authority: BookAuthority
    report_hash: str | None = Field(pattern=_HASH)
    update_subject: PortfolioUpdateReviewSubject | None = Field(
        default=None, exclude_if=lambda v: v is None
    )
    experiment_subject: PortfolioExperimentReviewSubject | None = Field(
        default=None, exclude_if=lambda v: v is None
    )
    result_hash: str | None = Field(default=None, pattern=_HASH)
    candidate_hash: str | None = Field(default=None, pattern=_HASH)
    handoff_hash: str | None = Field(default=None, pattern=_HASH)
    issuer_scope_hash: str = Field(pattern=_HASH)
    exposure_projection_hash: str = Field(pattern=_HASH)
    registry_hash: str = Field(pattern=_HASH)
    analysis_publication_hash: str = Field(pattern=_HASH)
    analyst_brief_hash: str = Field(pattern=_HASH)
    cro_package_hash: str = Field(pattern=_HASH)
    obligation_hash: str = Field(pattern=_HASH)
    evidence_as_of: datetime
    evidence_expires_at: datetime
    unresolved_questions: tuple[PortfolioReviewUnresolvedQuestion, ...] = Field(
        default=(), max_length=512, exclude_if=lambda value: value == ()
    )
    """The Analysts' open questions, each with the unit that asked it; carried
    explicitly rather than folded into limitations."""
    completion_schema: str = Field(default="", max_length=40, exclude_if=lambda v: v == "")
    """`issuer-review-state-v1` when every issuer carries a `review_state` from
    its analysis's completion and the coverage counts only executed issuers;
    absent for dossiers compiled from briefs sealed before it (legacy rule:
    every issuer in the request counted as reviewed)."""
    evidence_children: tuple[PortfolioReviewEvidenceChild, ...] = Field(
        default=(), max_length=64, exclude_if=lambda v: v == ()
    )
    """The units' publications when the book was analysed as more than one
    unit; empty -- and absent from the identity -- for a single publication,
    whose handles and hashes the top-level fields carry as they always did.
    For an aggregate the top-level `analysis_publication_hash`,
    `analyst_brief_hash` and `cro_package_hash` are the aggregate's own
    identities (`aggregate_hash` of the children), never a child's."""
    issuers: tuple[PortfolioReviewDossierIssuer, ...] = Field(max_length=512)
    findings: tuple[PortfolioReviewDossierFinding, ...] = Field(max_length=2048)
    citations: tuple[PortfolioReviewDossierCitation, ...] = Field(max_length=4096)
    coverage: PortfolioReviewCoverage
    analyst_requires_human_review: bool = False
    mapping_failure_count: int = Field(ge=0)
    held_count: int = Field(ge=0)
    window_end_effective_n: float = Field(ge=0.0, allow_inf_nan=False)
    claim_limits: tuple[str, ...] = Field(min_length=1)
    limitations: tuple[str, ...] = Field(min_length=1)
    open_issues: tuple[PortfolioReviewOpenIssue, ...] = Field(
        default=(), max_length=MAXIMUM_OPEN_ISSUES, exclude_if=lambda value: value == ()
    )
    """The issuers' register's open issues on the book's holdings, in the order
    the reviewer's aliases name them (`O1`, `O2`..); absent when none is open."""
    portfolio_report_link: str = Field(min_length=1, max_length=200)
    allowed_routes: tuple[PortfolioReviewRoute, ...] = tuple(PortfolioReviewRoute)
    dossier_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_dossier(self) -> Self:
        """Reject inconsistent holdings, findings, citations or coverage in the dossier."""
        if any(
            value.tzinfo is None or value.utcoffset() is None
            for value in (self.evidence_as_of, self.evidence_expires_at)
        ):
            raise ValueError("chief_risk_officer.dossier_clock_invalid")
        entities = tuple(value.entity_id for value in self.issuers)
        handles = tuple(value.finding_handle for value in self.findings)
        spans = tuple(value.span_handle for value in self.citations)
        if (
            len(entities) != len(set(entities))
            or len(handles) != len(set(handles))
            or len(spans) != len(set(spans))
            or self.allowed_routes != tuple(PortfolioReviewRoute)
        ):
            raise ValueError("chief_risk_officer.dossier_invalid")
        try:
            validate_review_book_authority(
                authority=self.book_authority,
                report_hash=self.report_hash,
                result_hash=self.result_hash,
                candidate_hash=self.candidate_hash,
                handoff_hash=self.handoff_hash,
                update_subject=self.update_subject,
                experiment_subject=self.experiment_subject,
            )
        except ValueError as error:
            raise ValueError("chief_risk_officer.dossier_authority_invalid") from error
        if any(value.available_at > self.evidence_as_of for value in self.citations):
            raise ValueError("chief_risk_officer.dossier_cutoff_invalid")
        if self.evidence_children:
            units = tuple(value.unit_id for value in self.evidence_children)
            children_entities = [
                entity for value in self.evidence_children for entity in value.ordered_entity_ids
            ]
            # A holding that filed nothing in the window is in the dossier with
            # its weight and in no child: nothing was there to analyse.
            analysed = {
                value.entity_id
                for value in self.issuers
                if value.review_state not in {"NOTHING_FILED", "UNREVIEWED"}
            }
            if (
                len(units) != len(set(units))
                or len(children_entities) != len(set(children_entities))
                or set(children_entities) != analysed
                or self.evidence_as_of != min(v.evidence_as_of for v in self.evidence_children)
                or self.evidence_expires_at
                != min(v.evidence_expires_at for v in self.evidence_children)
            ):
                raise ValueError("chief_risk_officer.dossier_children_invalid")
        scoped = set(entities)
        if any(not set(value.affected_entities) <= scoped for value in self.findings):
            raise ValueError("chief_risk_officer.dossier_entity_invalid")
        available = {value.span_handle for value in self.citations}
        for finding in self.findings:
            cited = {*finding.supporting_span_handles, *finding.contradicting_span_handles}
            if not cited <= available:
                raise ValueError("chief_risk_officer.dossier_citation_invalid")
        opened = [value.open_issue_handle for value in self.open_issues]
        if len(opened) != len(set(opened)) or any(
            not set(value.affected_entities) <= scoped
            or not set(value.cited_finding_handles) <= set(handles)
            for value in self.open_issues
        ):
            raise ValueError("chief_risk_officer.dossier_open_issue_invalid")
        _identity(self, "dossier_hash")
        return self

    @property
    def finding_handles(self) -> frozenset[str]:
        """Return the handles of all findings in this dossier."""
        return frozenset(value.finding_handle for value in self.findings)

    @property
    def evidence_publication_hashes(self) -> tuple[str, ...]:
        """Return the evidence publications on which this dossier rests.

        Use child publications for a composite dossier or its own publication for a
        single-unit dossier.
        """
        if self.evidence_children:
            return tuple(value.analysis_publication_hash for value in self.evidence_children)
        return (self.analysis_publication_hash,)

    def finding(self, handle: str) -> PortfolioReviewDossierFinding:
        """Return the finding named by its sealed handle."""
        for value in self.findings:
            if value.finding_handle == handle:
                return value
        raise KeyError(handle)

    def issuer(self, entity_id: str) -> PortfolioReviewDossierIssuer:
        """Return the issuer named by its stable entity identifier."""
        for value in self.issuers:
            if value.entity_id == entity_id:
                return value
        raise KeyError(entity_id)


NOT_ADDRESSED_RATIONALE = "Read by the reviewer and not named as a risk."


def review_dispositions(
    submission: PortfolioReviewAssessmentSubmission,
    *,
    answered: bool,
    finding_handles: Sequence[str],
) -> tuple[PortfolioReviewFindingDisposition, ...]:
    """Return every finding's disposition in one review.

    For a pre-answer assessment, return the sealed disposition. For a later
    answer, derive a material issue for a cited finding and a read-but-not-named
    disposition for every other finding. No reviewer states either route.
    """
    if not answered:
        return submission.finding_dispositions
    derived: dict[str, PortfolioReviewFindingDisposition] = {}
    for issue in submission.material_issues:
        for handle in issue.cited_finding_handles:
            derived.setdefault(
                handle,
                PortfolioReviewFindingDisposition(
                    finding_handle=handle,
                    disposition="MATERIAL_ISSUE",
                    rationale=issue.causal_channel,
                ),
            )
    for handle in finding_handles:
        derived.setdefault(
            handle,
            PortfolioReviewFindingDisposition(
                finding_handle=handle,
                disposition="NOT_ADDRESSED",
                rationale=NOT_ADDRESSED_RATIONALE,
            ),
        )
    return tuple(derived.values())


def finding_aliases(dossier: PortfolioReviewDossier) -> dict[str, str]:
    """Map each view alias to its dossier finding handle.

    Aliases follow dossier order: `F1`, `F2` and so on. The Host maps them back
    to sealed handles before submission.
    """
    return {
        f"F{index}": value.finding_handle for index, value in enumerate(dossier.findings, start=1)
    }


def translate_submission(
    submission: PortfolioReviewAssessmentSubmission,
    *,
    source: PortfolioReviewDossier,
    target: PortfolioReviewDossier,
) -> PortfolioReviewAssessmentSubmission | None:
    """Translate an assessment into another dossier's finding handles.

    Match each cited finding by its analysis and original handle. Return nothing
    when the other dossier lacks one; carried reviews retain their sealed route.
    """
    theirs = finding_sources(source)
    ours = {value[:2]: handle for handle, value in finding_sources(target).items()}

    def moved(handles: tuple[str, ...]) -> tuple[str, ...] | None:
        mapped = [ours.get(theirs[handle][:2]) if handle in theirs else None for handle in handles]
        return None if None in mapped else tuple(value for value in mapped if value is not None)

    issues = []
    for issue in submission.material_issues:
        cited = moved(issue.cited_finding_handles)
        if cited is None:
            return None
        issues.append(issue.model_copy(update={"cited_finding_handles": cited}))
    resolved = []
    for value in submission.resolved_issues:
        cited = moved(value.resolving_finding_handles)
        if cited is None:
            return None
        resolved.append(value.model_copy(update={"resolving_finding_handles": cited}))
    if submission.finding_dispositions:
        return None
    translated: PortfolioReviewAssessmentSubmission = submission.model_copy(
        update={"material_issues": tuple(issues), "resolved_issues": tuple(resolved)}
    )
    return translated


def raised_issue_handle(receipt_hash: str, issue_handle: str) -> str:
    """Name a raised issue for the issuers' register.

    Combine the receipt identity with the issue's position so a later review can
    refer to the same issue.
    """
    from alphalattice.kernel.shared_kernel.identity import canonical_hash

    digest = str(canonical_hash({"receipt": receipt_hash, "issue": issue_handle}))
    return "OPEN-" + digest[:12].upper()


def open_issue_aliases(dossier: PortfolioReviewDossier) -> dict[str, str]:
    """`O1`, `O2`.. in dossier order, and the open issue each names."""
    return {
        f"O{index}": value.open_issue_handle
        for index, value in enumerate(dossier.open_issues, start=1)
    }


def finding_sources(dossier: PortfolioReviewDossier) -> dict[str, tuple[str, str, date]]:
    """Return each finding's analysis, source handle and reading day.

    Child findings qualify their handles with their publication hash. A
    single-unit dossier uses bare handles read at its cutoff.
    """
    if not dossier.evidence_children:
        found = dossier.evidence_as_of.date()
        return {
            value.finding_handle: (dossier.analysis_publication_hash, value.finding_handle, found)
            for value in dossier.findings
        }
    sources: dict[str, tuple[str, str, date]] = {}
    for child in dossier.evidence_children:
        prefix = f"FIND-P{child.analysis_publication_hash[:8].upper()}-"
        found = (child.read_as_of or child.evidence_as_of).date()
        for value in dossier.findings:
            if value.finding_handle.startswith(prefix):
                sources[value.finding_handle] = (
                    child.analysis_publication_hash,
                    "FIND-" + value.finding_handle.removeprefix(prefix),
                    found,
                )
    return sources


def review_basis(dossier: PortfolioReviewDossier, *, decision_policy_hash: str) -> str:
    """Bind the review question to its policy, cited findings and exposed book.

    The basis includes each named holding's exposure band and review state, and
    the findings in alias order with their source analyses. A later dossier with
    the same basis and issuers' register carries the sealed review forward without
    a model call. A new finding, exposure band or review state makes a new
    question. A holding with no finding or open issue cannot change the answer.
    """
    from alphalattice.kernel.shared_kernel.identity import canonical_hash

    sources = finding_sources(dossier)
    named = {
        *(entity for value in dossier.findings for entity in value.affected_entities),
        *(entity for value in dossier.open_issues for entity in value.affected_entities),
    }
    return str(
        canonical_hash(
            {
                "decision_policy_hash": decision_policy_hash,
                "issuers": sorted(
                    [value.entity_id, str(value.exposure_band), value.review_state]
                    for value in dossier.issuers
                    if value.entity_id in named
                ),
                "findings": [list(sources[value.finding_handle][:2]) for value in dossier.findings],
            }
        )
    )


class PortfolioReviewMaterialIssue(BaseModel):  # type: ignore[misc]
    """One judgment about one or more cited findings. Every field is a bounded primitive.

    Sealed receipts hold it as an actor wrote it, or -- for an answer -- as
    the Host normalized one risk: issuers from the cited findings, relevance
    direct, direction adverse, the reason as the causal channel, the advice
    as `recommendation`, and no mitigation, uncertainty or counterfactual
    (none was asked).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    issue_handle: str = Field(pattern=r"^ISSUE-[A-Z0-9-]{1,48}$")
    cited_finding_handles: tuple[str, ...] = Field(min_length=1, max_length=8)
    affected_entities: tuple[str, ...] = Field(min_length=1, max_length=MAXIMUM_AFFECTED_ISSUERS)
    relevance: CROEvidenceRelevance
    position_impact_direction: CROPositionImpactDirection
    causal_channel: str = Field(min_length=1, max_length=600)
    severity_if_true: CROSeverityIfTrue
    evidence_interpretation: CROEvidenceInterpretation
    portfolio_mitigation: CROPortfolioMitigation | None = Field(
        default=None, exclude_if=lambda v: v is None
    )
    """None for a normalized answer: the book's mitigation is not stated."""
    uncertainty: str = Field(default="", max_length=600, exclude_if=lambda v: v == "")
    what_would_change_the_conclusion: str = Field(
        default="", max_length=600, exclude_if=lambda v: v == ""
    )
    recommendation: str = Field(default="", max_length=600, exclude_if=lambda v: v == "")
    """The reviewer's advice in words; absent from issues sealed before it."""
    open_issue_handle: str | None = Field(
        default=None, pattern=OPEN_ISSUE_PATTERN, exclude_if=lambda value: value is None
    )
    """The open issue this states again (W3); absent for a new one."""
    carried_on: date | None = Field(default=None, exclude_if=lambda value: value is None)
    """Set when the reviewer said nothing of the open issue: the Host counts it
    as last assessed, on this day."""

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_issue(self) -> Self:
        """Reject a material issue with inconsistent cited findings or judgment."""
        if len(set(self.cited_finding_handles)) != len(self.cited_finding_handles):
            raise ValueError("chief_risk_officer.issue_citation_duplicate")
        if self.relevance is CROEvidenceRelevance.NOT_RELEVANT and (
            self.position_impact_direction is not CROPositionImpactDirection.NONE
        ):
            raise ValueError("chief_risk_officer.issue_relevance_invalid")
        return self


class PortfolioReviewIssueResolution(BaseModel):  # type: ignore[misc]
    """An open issue a review closed, with the findings that resolve it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    open_issue_handle: str = Field(pattern=OPEN_ISSUE_PATTERN)
    resolving_finding_handles: tuple[str, ...] = Field(min_length=1, max_length=8)
    why: str = Field(min_length=1, max_length=600)


class PortfolioReviewAssessmentSubmission(BaseModel):  # type: ignore[misc]
    """The one thing the actor returns. It does not select the route."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    material_issues: tuple[PortfolioReviewMaterialIssue, ...] = Field(
        max_length=MAXIMUM_REVIEWER_ISSUES + MAXIMUM_OPEN_ISSUES
    )
    """The reviewer's own issues, at most sixteen, and every open issue the reviewer
    neither stated again nor resolved, carried as last assessed (`carried_on`, W3). A
    book's register may hold more open issues than one answer can address, so the carried
    ones are outside the reviewer's bound and none is dropped (CS)."""
    overall_rationale: str = Field(min_length=1, max_length=2400)
    unresolved_questions: tuple[str, ...] = Field(max_length=8)
    requires_human_review: bool
    limitations_acknowledged: bool
    finding_dispositions: tuple[PortfolioReviewFindingDisposition, ...] = Field(
        default=(), max_length=2048, exclude_if=lambda value: value == ()
    )
    """Versioned (`finding-dispositions-v1`): one per finding of the dossier,
    validated as a complete set by the sealer. Absent on submissions made
    before it existed; those keep their historical reading, `UNSTATED`."""
    resolved_issues: tuple[PortfolioReviewIssueResolution, ...] = Field(
        default=(), max_length=256, exclude_if=lambda value: value == ()
    )
    """Open issues this review closed (W3)."""

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_submission(self) -> Self:
        """Reject a submission that violates the Host's normalized review contract."""
        handles = tuple(value.issue_handle for value in self.material_issues)
        if len(handles) != len(set(handles)):
            raise ValueError("chief_risk_officer.submission_duplicate_issue")
        own = sum(1 for value in self.material_issues if value.carried_on is None)
        if own > MAXIMUM_REVIEWER_ISSUES:
            raise ValueError("chief_risk_officer.submission_own_issues_exceeded")
        opened = [
            *(value.open_issue_handle for value in self.material_issues if value.open_issue_handle),
            *(value.open_issue_handle for value in self.resolved_issues),
        ]
        if len(opened) != len(set(opened)):
            raise ValueError("chief_risk_officer.submission_open_issue_duplicate")
        if not self.limitations_acknowledged:
            raise ValueError("chief_risk_officer.submission_limitations_unacknowledged")
        dispositions = tuple(value.finding_handle for value in self.finding_dispositions)
        if len(dispositions) != len(set(dispositions)):
            raise ValueError("chief_risk_officer.submission_duplicate_disposition")
        return self

    @property
    def disposition_completeness(self) -> Literal["UNSTATED", "COMPLETE", "PARTIAL"]:
        """Report whether the submission states every finding's disposition."""
        if not self.finding_dispositions:
            return "UNSTATED"
        if any(value.disposition == "DEFERRED" for value in self.finding_dispositions):
            return "PARTIAL"
        return "COMPLETE"

    @property
    def deferred_finding_handles(self) -> frozenset[str]:
        """Return the finding handles deferred by this assessment."""
        return frozenset(
            value.finding_handle
            for value in self.finding_dispositions
            if value.disposition == "DEFERRED"
        )


class IssueEvaluation(BaseModel):  # type: ignore[misc]
    """What the matrix did with one issue *for one issuer*, and why.

    An issue may name several affected issuers, and they do not share an
    exposure band. Each affected issuer therefore earns its own evaluation at
    its own band, so a small position cannot inherit a large one's exposure and
    a named issuer cannot fall out of the conclusions.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    issue_handle: str = Field(pattern=r"^ISSUE-[A-Z0-9-]{1,48}$")
    entity_id: str = Field(min_length=1, max_length=32)
    exposure_band: ExposureBand
    stated_interpretation: CROEvidenceInterpretation
    effective_interpretation: CROEvidenceInterpretation
    structure: EvidenceStructureState
    route: PortfolioReviewRoute
    rule_id: str = Field(min_length=1, max_length=40)


IssuerConclusionKind = Literal[
    "NOTHING_FILED",
    "NO_FINDING_IN_SCOPE",
    "CONCERNS_NOT_ADJUDICATED",
    "NO_ADVERSE_ISSUE",
    "ISSUE_NOTED_WITHIN_LIMITS",
    "EVIDENCE_GAP",
    "HUMAN_REVIEW_REQUIRED",
    "ISSUE_STANDS",
]


class IssuerConclusion(BaseModel):  # type: ignore[misc]
    """Summarize an issuer's assessed issues within the reviewed book.

    Attributes:
        entity_id: Issuer the conclusion concerns.
        exposure_band: Issuer exposure band in the reviewed book.
        finding_count: Number of findings considered for the issuer.
        adverse_issue_count: Number of adverse issues derived from those findings.
        conclusion: Deterministic conclusion assigned by the review compiler.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    entity_id: str = Field(min_length=1, max_length=32)
    exposure_band: ExposureBand
    finding_count: int = Field(ge=0)
    adverse_issue_count: int = Field(ge=0)
    conclusion: IssuerConclusionKind


class PortfolioReviewOutcome(BaseModel):  # type: ignore[misc]
    """What the pure route function decided, and the rules that decided it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    route: PortfolioReviewRoute
    review_state: ReviewState
    disposition_completeness: str = Field(
        default="UNSTATED", max_length=16, exclude_if=lambda v: v == "UNSTATED"
    )
    """`COMPLETE` when every qualified finding received a non-deferred disposition,
    `PARTIAL` when some were deferred, `UNSTATED` (absent) for submissions that
    carried none: those keep their historical reading."""
    policy_version: str = Field(min_length=1, max_length=120)
    reasons: tuple[str, ...] = Field(min_length=1, max_length=8)
    rule_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    required_actions: tuple[RequiredAction, ...] = Field(min_length=1, max_length=16)
    issue_evaluations: tuple[IssueEvaluation, ...] = Field(
        max_length=(MAXIMUM_REVIEWER_ISSUES + MAXIMUM_OPEN_ISSUES) * MAXIMUM_AFFECTED_ISSUERS
    )
    """Every issue the assessment holds, its own and the carried ones, at eight affected
    issuers each: one evaluation per pair."""
    issuer_conclusions: tuple[IssuerConclusion, ...] = Field(max_length=512)
    limitations: tuple[str, ...] = Field(
        default=(), max_length=16, exclude_if=lambda value: value == ()
    )
    """What the review did not read, written by the program beside the route:
    coverage, issuers not reviewed, the analysis's gaps, findings read and
    not named. Absent from outcomes sealed before; there a guard stood."""


def effective_interpretation(
    stated: CROEvidenceInterpretation, structure: EvidenceStructureState
) -> CROEvidenceInterpretation:
    """Cap what the actor said by how the finding is actually held up.

    A finding with no supporting span cannot be more than `INSUFFICIENT`; one
    with a contradicting span is `CONTESTED` unless the actor already found it
    insufficient; one with a single document behind it cannot be `SUPPORTED`.
    The actor may always be *more* cautious than the structure.
    """
    if structure is EvidenceStructureState.UNSUPPORTED:
        return CROEvidenceInterpretation.INSUFFICIENT
    if structure is EvidenceStructureState.CONTESTED:
        if stated is CROEvidenceInterpretation.INSUFFICIENT:
            return stated
        return CROEvidenceInterpretation.CONTESTED
    if structure is EvidenceStructureState.SINGLE_SOURCE and (
        stated is CROEvidenceInterpretation.SUPPORTED
    ):
        return CROEvidenceInterpretation.LIMITED
    return stated


def _risk_cell(
    *,
    severity: CROSeverityIfTrue,
    band: ExposureBand,
    interpretation: CROEvidenceInterpretation,
) -> tuple[PortfolioReviewRoute, str]:
    """Route one risk against one issuer's exposure.

    A supported high-severity risk on an exposed holding objects; limited or
    contested support asks a person to weigh it. Other material risks are accepted
    with limits, while low risks are listed without objection.
    """
    exposed = band in {ExposureBand.HIGH, ExposureBand.CRITICAL}
    if severity is CROSeverityIfTrue.HIGH and exposed:
        if interpretation is CROEvidenceInterpretation.SUPPORTED:
            return PortfolioReviewRoute.MATERIAL_OBJECTION, "R-OBJECTION"
        return PortfolioReviewRoute.HUMAN_REVIEW_REQUIRED, "R-HIGH-UNSETTLED"
    if severity is not CROSeverityIfTrue.LOW:
        return PortfolioReviewRoute.ACCEPT_WITH_LIMITS, "R-LIMITS"
    return PortfolioReviewRoute.NO_MATERIAL_OBJECTION, "R-LOW"


NO_MAJOR_NEGATIVE = "No major negative was found in the evidence read."
"""The program's wording when no material risk was named."""


def route_portfolio_review(
    *,
    dossier: PortfolioReviewDossier,
    submission: PortfolioReviewAssessmentSubmission,
    evidence_is_current: bool,
    dropped: tuple[AnswerProblem, ...] = (),
) -> PortfolioReviewOutcome:
    """The one deterministic route: judge first, state the gaps. Pure and versioned.

    Every named risk is evaluated at each affected issuer's exposure, its
    stated confidence capped by the structure of the findings it cites; the
    book's route is the most severe. What the review could not read --
    coverage, issuers not reviewed, the analysis's gaps, findings nobody named
    -- is written beside the route as limitations and never stops it. Nothing
    here produces an adverse claim the reviewer did not ground in a citation.
    Expired evidence is refused by name; no route asks anyone to refresh.

    `dropped` is what the Host could not admit from the reviewer's last answer
    after two corrections: unfinished work, never a clean review. The review
    is partial, the drop is its first limitation, and a route below a
    person's reading is raised to one -- the dropped items may have been the
    major negatives, so no major negative is claimed.
    """
    if not evidence_is_current:
        raise ValueError("chief_risk_officer.review_evidence_expired")
    gaps = _mapping_actions(dossier)
    state = ReviewState.COMPLETE
    if not dossier.issuers:
        state = ReviewState.UNAVAILABLE
    elif (
        dossier.mapping_failure_count
        or dossier.coverage.unavailable_reasons
        or _weight_unaccounted(dossier.coverage)
        or dossier.coverage.missing_evidence
        # A deferred disposition (sealed before answers) is a finding not reviewed.
        or submission.disposition_completeness == "PARTIAL"
        or dropped
    ):
        state = ReviewState.PARTIAL

    evaluations: list[IssueEvaluation] = []
    for issue in submission.material_issues:
        if issue.relevance is CROEvidenceRelevance.NOT_RELEVANT or (
            issue.position_impact_direction
            in {CROPositionImpactDirection.FAVOURABLE, CROPositionImpactDirection.NONE}
        ):
            continue
        structures = tuple(
            dossier.finding(handle).structure for handle in issue.cited_finding_handles
        )
        structure = _weakest(structures)
        effective = effective_interpretation(issue.evidence_interpretation, structure)
        # One evaluation per affected issuer, at that issuer's own exposure
        # band. Sorted so the order the actor happened to list its issuers in
        # cannot change the outcome.
        for entity_id in sorted(set(issue.affected_entities)):
            band = dossier.issuer(entity_id).exposure_band
            route, rule_id = _risk_cell(
                severity=issue.severity_if_true, band=band, interpretation=effective
            )
            evaluations.append(
                IssueEvaluation(
                    issue_handle=issue.issue_handle,
                    entity_id=entity_id,
                    exposure_band=band,
                    stated_interpretation=issue.evidence_interpretation,
                    effective_interpretation=effective,
                    structure=structure,
                    route=route,
                    rule_id=rule_id,
                )
            )

    route = PortfolioReviewRoute.NO_MATERIAL_OBJECTION
    for value in evaluations:
        if _ROUTE_PRIORITY[value.route] > _ROUTE_PRIORITY[route]:
            route = value.route
    incomplete = bool(dropped) and (
        _ROUTE_PRIORITY[route] < _ROUTE_PRIORITY[PortfolioReviewRoute.HUMAN_REVIEW_REQUIRED]
    )
    if incomplete:
        route = PortfolioReviewRoute.HUMAN_REVIEW_REQUIRED
    triggering = tuple(value for value in evaluations if value.route is route)
    # One representative per issuer, before anything is limited for
    # presentation. Reasons and actions are per issuer, not per issue: a book
    # whose first issuer collected eight issues would otherwise fill the
    # eight-item presentation bound and silently drop the ninth issuer, which
    # the conclusions still report as standing.
    spokesmen = _per_issuer(triggering)
    reasons: tuple[str, ...]
    rule_ids: tuple[str, ...]
    actions: tuple[RequiredAction, ...]
    if incomplete:
        reasons = (
            f"The reviewer's answer had {len(dropped)} item(s) the Host could not admit "
            "after two corrections; they are not part of this review, so a person should "
            "read the answer as written.",
        )
        rule_ids = ("R-ANSWER-INCOMPLETE",)
        actions = (
            RequiredAction(
                action=RequiredActionKind.HUMAN_REVIEW,
                reason="Advisory: read the reviewer's answer as written before relying on "
                "this book.",
                blocking=False,
            ),
        )
    elif route is PortfolioReviewRoute.MATERIAL_OBJECTION:
        reasons = tuple(
            f"{value.issue_handle} ({value.entity_id}): a high-severity risk its findings "
            "support stands on an exposed holding."
            for value in spokesmen
        )[:8]
        rule_ids = tuple(sorted({value.rule_id for value in triggering}))[:8]
        kind = (
            RequiredActionKind.DO_NOT_ACTIVATE
            if dossier.book_authority is BookAuthority.VALIDATED_HANDOFF
            else RequiredActionKind.RECONSIDER_CANDIDATE
        )
        actions = tuple(
            RequiredAction(
                action=kind,
                entity_id=value.entity_id,
                reason=(
                    "Do not act on this book while the cited risk stands."
                    if kind is RequiredActionKind.DO_NOT_ACTIVATE
                    else "Reconsider this research recommendation under a new identity; "
                    "it is not an instruction to trade."
                    if dossier.update_subject is not None or dossier.experiment_subject is not None
                    else "Reconsider the candidate before freezing; a new candidate returns "
                    "through development under its own identity."
                ),
                blocking=True,
            )
            for value in spokesmen
        )[:8]
    elif route is PortfolioReviewRoute.HUMAN_REVIEW_REQUIRED:
        reasons = tuple(
            f"{value.issue_handle} ({value.entity_id}): a high-severity risk on an exposed "
            "holding whose findings are contested or limited; a person should weigh them."
            for value in spokesmen
        )[:8]
        rule_ids = tuple(sorted({value.rule_id for value in triggering}))[:8]
        actions = tuple(
            RequiredAction(
                action=RequiredActionKind.HUMAN_REVIEW,
                entity_id=value.entity_id,
                reason="Advisory: weigh the cited sources before relying on this book.",
                blocking=False,
            )
            for value in spokesmen
        )[:8]
    elif route is PortfolioReviewRoute.ACCEPT_WITH_LIMITS:
        reasons = tuple(
            f"{value.issue_handle} ({value.entity_id}): a material risk stands within the "
            "stated limits."
            for value in spokesmen
        )[:8]
        rule_ids = tuple(sorted({value.rule_id for value in triggering}))[:8]
        actions = (
            RequiredAction(
                action=RequiredActionKind.NONE,
                reason="Carry the stated risks and limits with any use of this book.",
                blocking=False,
            ),
        )
    else:
        low = sum(1 for value in evaluations if value.rule_id == "R-LOW")
        reasons = (
            f"{NO_MAJOR_NEGATIVE} This is not a statement that the Portfolio or any issuer "
            "is safe.",
            *(
                ()
                if not low
                else (f"{low} low-severity risk evaluation(s) are listed without objection.",)
            ),
        )
        rule_ids = ("R-LOW",) if low else ("R-NONE",)
        actions = (
            RequiredAction(
                action=RequiredActionKind.NONE,
                reason="None. The stated limits and claim limits still apply.",
                blocking=False,
            ),
        )
    return _outcome(
        dossier,
        route=route,
        state=state,
        reasons=reasons,
        rule_ids=rule_ids,
        actions=(*actions, *gaps),
        evaluations=tuple(evaluations),
        deferred=submission.deferred_finding_handles,
        disposition_completeness=submission.disposition_completeness,
        limitations=_limitations(dossier, submission, dropped),
    )


def _weight_unaccounted(coverage: PortfolioReviewCoverage) -> bool:
    """Report whether any book weight lacks a reading or a no-finding record.

    Use the dossier's exact unread share when present; otherwise treat any share
    short of complete reading as unaccounted for.
    """
    if coverage.unreached_ending_weight_coverage is not None:
        return coverage.unreached_ending_weight_coverage > 0.0
    return coverage.reviewed_ending_weight_coverage < 1.0


def _limitations(
    dossier: PortfolioReviewDossier,
    submission: PortfolioReviewAssessmentSubmission,
    dropped: tuple[AnswerProblem, ...] = (),
) -> tuple[str, ...]:
    """Describe evidence the review did not read without gating its route.

    Put text the Host dropped from the answer first so a bound cannot hide it.
    """
    coverage = dossier.coverage
    lines: list[str] = []
    if dropped:
        lines.append(
            f"The reviewer's last answer had {len(dropped)} item(s) the Host could not "
            "admit after two corrections; they are not part of this review. "
            + " ".join(f"Item {value.item}: {value.text}" for value in dropped[:3])
            + (f" And {len(dropped) - 3} more." if len(dropped) > 3 else "")
        )
    if not dossier.issuers:
        lines.append("No issuer in this book could be reviewed; nothing was read.")
    elif coverage.nothing_filed_ending_weight_coverage is not None:
        # Weight three ways: read, nothing filed, not read.
        share = coverage.reviewed_ending_weight_coverage
        lines.append(
            f"The review read holdings carrying {share:.2%} of the book's ending weight; "
            f"{coverage.nothing_filed_ending_weight_coverage:.2%} is in holdings that filed "
            f"nothing with the SEC in the last {coverage.nothing_filed_window_days} days, which "
            "is not a finding of no risk"
            + (
                f"; {coverage.unreached_ending_weight_coverage:.2%} was not read."
                if coverage.unreached_ending_weight_coverage
                else "."
            )
        )
    elif coverage.reviewed_ending_weight_coverage < 1.0:
        share = coverage.reviewed_ending_weight_coverage
        lines.append(
            f"The review read issuers carrying {share:.2%} of the book's ending weight"
            + (
                f", below the {MINIMUM_REVIEWED_ENDING_WEIGHT_COVERAGE:.0%} it is designed for"
                if share < MINIMUM_REVIEWED_ENDING_WEIGHT_COVERAGE
                else ""
            )
            + "; the rest was not read."
        )
    if dossier.mapping_failure_count:
        lines.append(
            f"{dossier.mapping_failure_count} holding(s) did not map to an admitted issuer "
            "and were not reviewed."
        )
    if coverage.unavailable_reasons:
        # Each entry names what was not reviewed -- an issuer, or a unit with its
        # issuers -- and says so in its own words, so the line joins the entries
        # and counts entries, never issuers it did not count.
        reasons = coverage.unavailable_reasons
        lines.append(
            "; ".join(reasons[:3])
            + (f"; and {len(reasons) - 3} more." if len(reasons) > 3 else ".")
        )
    if coverage.missing_evidence:
        lines.append(
            f"The analysis records {len(coverage.missing_evidence)} gap(s) in what could be "
            "read -- topics without a source of their shape, needs or candidates left unread; "
            "they are limits of this review, not findings."
        )
    if dossier.analyst_requires_human_review:
        lines.append("The analysis asked for a person to read its sources.")
    cited = {
        handle for issue in submission.material_issues for handle in issue.cited_finding_handles
    }
    unnamed = [value for value in dossier.findings if value.finding_handle not in cited]
    if unnamed:
        lines.append(
            f"{len(unnamed)} of {len(dossier.findings)} finding(s) were read by the reviewer "
            "and not named as a risk."
        )
    return tuple(value[:600] for value in lines)[:16]


def _per_issuer(triggering: tuple[IssueEvaluation, ...]) -> tuple[IssueEvaluation, ...]:
    """One evaluation per issuer, chosen without reference to submission order.

    The lowest issue handle speaks for its issuer, and the issuers are ordered
    by id, so the same issues in any order produce the same actions.
    """
    spokesman: dict[str, IssueEvaluation] = {}
    for value in triggering:
        held = spokesman.get(value.entity_id)
        if held is None or value.issue_handle < held.issue_handle:
            spokesman[value.entity_id] = value
    return tuple(spokesman[entity_id] for entity_id in sorted(spokesman))


def _mapping_actions(dossier: PortfolioReviewDossier) -> tuple[RequiredAction, ...]:
    if not dossier.mapping_failure_count:
        return ()
    return (
        RequiredAction(
            action=RequiredActionKind.RESOLVE_ISSUER_MAPPING,
            reason=(
                f"{dossier.mapping_failure_count} holding(s) did not map to an admitted issuer "
                "and were not reviewed."
            ),
            blocking=False,
        ),
    )


def _outcome(
    dossier: PortfolioReviewDossier,
    *,
    route: PortfolioReviewRoute,
    state: ReviewState,
    reasons: tuple[str, ...],
    rule_ids: tuple[str, ...],
    actions: tuple[RequiredAction, ...],
    evaluations: tuple[IssueEvaluation, ...] = (),
    deferred: frozenset[str] = frozenset(),
    disposition_completeness: str = "UNSTATED",
    limitations: tuple[str, ...] = (),
) -> PortfolioReviewOutcome:
    """Seal a route with issuer conclusions and stated limitations.

    Record disposition completeness as a fact about the submitted answer;
    `UNSTATED` is valid for an answer that was not asked for dispositions.
    """
    conclusions = tuple(
        _issuer_conclusion(dossier, issuer, evaluations, deferred=deferred)
        for issuer in dossier.issuers
    )
    return PortfolioReviewOutcome(
        route=route,
        review_state=state,
        disposition_completeness=disposition_completeness,
        policy_version=PORTFOLIO_REVIEW_ROUTE_POLICY,
        reasons=reasons,
        rule_ids=rule_ids,
        required_actions=tuple(dict.fromkeys(actions))[:16],
        issue_evaluations=evaluations,
        issuer_conclusions=conclusions,
        limitations=limitations,
    )


def _issuer_conclusion(
    dossier: PortfolioReviewDossier,
    issuer: PortfolioReviewDossierIssuer,
    evaluations: tuple[IssueEvaluation, ...],
    *,
    deferred: frozenset[str] = frozenset(),
) -> IssuerConclusion:
    findings = sum(issuer.entity_id in value.affected_entities for value in dossier.findings)
    own = tuple(value for value in evaluations if value.entity_id == issuer.entity_id)
    conclusion: IssuerConclusionKind
    routes = {value.route for value in own}
    has_deferred = any(
        issuer.entity_id in value.affected_entities and value.finding_handle in deferred
        for value in dossier.findings
    )
    if issuer.review_state == "NOTHING_FILED":
        # Nothing in the window to read: said as such, never as no risk.
        conclusion = "NOTHING_FILED"
    elif issuer.review_state not in {
        "UNSTATED",
        "EXECUTED_NO_FINDINGS",
        "EXECUTED_WITH_FINDINGS",
    }:
        # The analysis did not execute its checks for this issuer -- no
        # source, nothing delivered, nothing reported or checks deferred --
        # so nothing here is a clean result; a finding it did record is a
        # concern that was not reviewed to the end.
        conclusion = "EVIDENCE_GAP" if findings == 0 else "CONCERNS_NOT_ADJUDICATED"
    elif has_deferred and PortfolioReviewRoute.MATERIAL_OBJECTION not in routes:
        # A finding the reviewer deferred was not reviewed for this issuer.
        conclusion = "CONCERNS_NOT_ADJUDICATED"
    elif PortfolioReviewRoute.MATERIAL_OBJECTION in routes:
        conclusion = "ISSUE_STANDS"
    elif PortfolioReviewRoute.HUMAN_REVIEW_REQUIRED in routes:
        conclusion = "HUMAN_REVIEW_REQUIRED"
    elif PortfolioReviewRoute.REQUEST_EVIDENCE_REFRESH in routes:
        conclusion = "EVIDENCE_GAP"
    elif PortfolioReviewRoute.ACCEPT_WITH_LIMITS in routes:
        conclusion = "ISSUE_NOTED_WITHIN_LIMITS"
    elif findings == 0:
        conclusion = "NO_FINDING_IN_SCOPE"
    else:
        conclusion = "NO_ADVERSE_ISSUE"
    return IssuerConclusion(
        entity_id=issuer.entity_id,
        exposure_band=issuer.exposure_band,
        finding_count=findings,
        adverse_issue_count=len(own),
        conclusion=conclusion,
    )


def _weakest(structures: tuple[EvidenceStructureState, ...]) -> EvidenceStructureState:
    order = (
        EvidenceStructureState.UNSUPPORTED,
        EvidenceStructureState.CONTESTED,
        EvidenceStructureState.SINGLE_SOURCE,
        EvidenceStructureState.SUPPORTED,
    )
    return min(structures, key=order.index)


@dataclass(frozen=True, slots=True)
class OpenIssueState:
    """Represent an issuer-register issue still open at an assessment date.

    A review raised it and none has resolved it since (W3). It retains the
    CRO's last statement and the findings from the analyses it rests on.
    """

    open_issue_handle: str
    raised_on: date
    assessed_on: date
    issue: PortfolioReviewMaterialIssue
    findings: tuple[tuple[str, str, date], ...]
    """(analysis publication, finding handle as its brief states it, day read)."""


class PortfolioReviewReceipt(AlternativeEvidenceContract):
    """Host provenance for one actor-neutral assessment and the route it produced."""

    kind: Literal["PortfolioReviewReceipt"] = "PortfolioReviewReceipt"
    dossier_hash: str = Field(pattern=_HASH)
    submission: PortfolioReviewAssessmentSubmission
    answer: PortfolioReviewAnswer | None = Field(default=None, exclude_if=lambda v: v is None)
    """The reviewer's answer -- its accepted risks -- when it answered in the
    judgment-only format; `submission` is then the Host's normalization and
    the actor binding names the answer. Absent from receipts sealed before."""
    dropped: tuple[AnswerProblem, ...] = Field(
        default=(), max_length=256, exclude_if=lambda v: v == ()
    )
    """Risks of the answer the Host did not accept, each with its problem."""
    actor_submission: ActorSubmissionBinding
    outcome: PortfolioReviewOutcome
    decision_policy_hash: str = Field(pattern=_HASH)
    model_call_count: int = Field(ge=0, le=3)
    """Calls started: the first answer and at most two corrections."""
    protocol_repair_count: int = Field(ge=0, le=2)
    review_basis: str | None = Field(
        default=None, pattern=_HASH, exclude_if=lambda value: value is None
    )
    """What the assessment rests on (`review_basis`), so a later dossier on the
    same basis finds it (W3); absent from receipts sealed before."""
    receipt_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_receipt(self) -> Self:
        """Reject a receipt with inconsistent review or routing fields."""
        from alphalattice.kernel.shared_kernel.identity import canonical_hash

        if self.outcome.policy_version not in READABLE_PORTFOLIO_REVIEW_ROUTE_POLICIES:
            raise ValueError("chief_risk_officer.receipt_policy_invalid")
        if self.answer is not None and self.actor_submission.submission_hash != canonical_hash(
            self.answer.model_dump(mode="json")
        ):
            raise ValueError("chief_risk_officer.receipt_answer_mismatch")
        if self.dropped and self.answer is None:
            raise ValueError("chief_risk_officer.receipt_answer_drops_invalid")
        _identity(self, "receipt_hash")
        return self

    @property
    def carried_forward(self) -> bool:
        """Report whether the Host carried the review without a model call.

        This is never a reviewer's own new assessment.
        """
        return self.actor_submission.actor_kind is ActorKind.HOST_FALLBACK


class PortfolioReviewRecommendation(AlternativeEvidenceContract):
    """The rendered recommendation. Pointer-free, and never an activation."""

    kind: Literal["PortfolioReviewRecommendation"] = "PortfolioReviewRecommendation"
    dossier_hash: str = Field(pattern=_HASH)
    decision_receipt_hash: str = Field(pattern=_HASH)
    book_authority: BookAuthority
    report_hash: str | None = Field(pattern=_HASH)
    update_subject: PortfolioUpdateReviewSubject | None = Field(
        default=None, exclude_if=lambda v: v is None
    )
    experiment_subject: PortfolioExperimentReviewSubject | None = Field(
        default=None, exclude_if=lambda v: v is None
    )
    result_hash: str | None = Field(default=None, pattern=_HASH)
    route: PortfolioReviewRoute
    review_state: ReviewState
    policy_version: str = Field(min_length=1, max_length=120)
    actor_kind: ActorKind
    reviewed_entity_ids: tuple[str, ...] = Field(max_length=512)
    evidence_as_of: datetime
    evidence_expires_at: datetime
    reasons: tuple[str, ...] = Field(min_length=1, max_length=8)
    required_actions: tuple[RequiredAction, ...] = Field(min_length=1, max_length=16)
    issuer_conclusions: tuple[IssuerConclusion, ...] = Field(max_length=512)
    claim_limits: tuple[str, ...] = Field(min_length=1)
    limitations: tuple[str, ...] = Field(
        default=(), max_length=16, exclude_if=lambda value: value == ()
    )
    """The outcome's program-written limitations; absent before them."""
    portfolio_report_link: str = Field(min_length=1, max_length=200)
    action_activation: Literal["NOT_AUTHORIZED"] = "NOT_AUTHORIZED"
    recommendation_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_recommendation(self) -> Self:
        """Reject an inconsistent pointer-free review recommendation."""
        if any(
            value.tzinfo is None or value.utcoffset() is None
            for value in (self.evidence_as_of, self.evidence_expires_at)
        ):
            raise ValueError("chief_risk_officer.recommendation_clock_invalid")
        if self.policy_version not in READABLE_PORTFOLIO_REVIEW_ROUTE_POLICIES:
            raise ValueError("chief_risk_officer.recommendation_policy_invalid")
        _identity(self, "recommendation_hash")
        return self


class PortfolioReviewPublication(AlternativeEvidenceContract):
    """One immutable publication per review key. No pointer is ever written."""

    kind: Literal["PortfolioReviewPublication"] = "PortfolioReviewPublication"
    review_key: str = Field(pattern=_HASH)
    book_authority: BookAuthority
    report_hash: str | None = Field(pattern=_HASH)
    update_subject: PortfolioUpdateReviewSubject | None = Field(
        default=None, exclude_if=lambda v: v is None
    )
    experiment_subject: PortfolioExperimentReviewSubject | None = Field(
        default=None, exclude_if=lambda v: v is None
    )
    result_hash: str | None = Field(default=None, pattern=_HASH)
    issuer_scope_hash: str = Field(pattern=_HASH)
    analysis_publication_hash: str = Field(pattern=_HASH)
    dossier_hash: str = Field(pattern=_HASH)
    decision_receipt_hash: str = Field(pattern=_HASH)
    recommendation_hash: str = Field(pattern=_HASH)
    decision_policy_hash: str = Field(pattern=_HASH)
    published_at: datetime
    publication_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_publication(self) -> Self:
        """Reject a publication whose review records do not agree."""
        if self.published_at.tzinfo is None or self.published_at.utcoffset() is None:
            raise ValueError("chief_risk_officer.publication_clock_invalid")
        _identity(self, "publication_hash")
        return self


def compile_portfolio_review_recommendation(
    *,
    dossier: PortfolioReviewDossier,
    receipt: PortfolioReviewReceipt,
) -> PortfolioReviewRecommendation:
    """A pure function of two sealed artifacts, rebuildable after a restart."""
    from alphalattice.evidence.alternative_evidence.contracts import seal_contract

    if receipt.dossier_hash != dossier.dossier_hash:
        raise ValueError("chief_risk_officer.recommendation_dossier_mismatch")
    return seal_contract(
        PortfolioReviewRecommendation,
        "recommendation_hash",
        dossier_hash=dossier.dossier_hash,
        decision_receipt_hash=receipt.receipt_hash,
        book_authority=dossier.book_authority,
        report_hash=dossier.report_hash,
        update_subject=dossier.update_subject,
        experiment_subject=dossier.experiment_subject,
        result_hash=dossier.result_hash,
        route=receipt.outcome.route,
        review_state=receipt.outcome.review_state,
        policy_version=receipt.outcome.policy_version,
        actor_kind=receipt.actor_submission.actor_kind,
        reviewed_entity_ids=tuple(value.entity_id for value in dossier.issuers),
        evidence_as_of=dossier.evidence_as_of,
        evidence_expires_at=dossier.evidence_expires_at,
        reasons=receipt.outcome.reasons,
        required_actions=receipt.outcome.required_actions,
        issuer_conclusions=receipt.outcome.issuer_conclusions,
        claim_limits=dossier.claim_limits,
        limitations=receipt.outcome.limitations,
        portfolio_report_link=dossier.portfolio_report_link,
    )


def _identity(value: AlternativeEvidenceContract, field: str) -> None:
    from alphalattice.kernel.shared_kernel.identity import canonical_hash

    expected = canonical_hash(value.model_dump(mode="json", exclude={field}))
    if getattr(value, field) != expected:
        raise ValueError("chief_risk_officer.identity_invalid")


__all__ = [
    "MINIMUM_REVIEWED_ENDING_WEIGHT_COVERAGE",
    "NO_MAJOR_NEGATIVE",
    "OPEN_ISSUE_PATTERN",
    "PORTFOLIO_REVIEW_ROUTE_POLICY",
    "REVIEW_ANSWER_TEXT_FIELDS",
    "CROEvidenceInterpretation",
    "CROEvidenceRelevance",
    "CROPortfolioMitigation",
    "CROPositionImpactDirection",
    "CRORiskConfidence",
    "CRORiskSeverity",
    "CROSeverityIfTrue",
    "IssueEvaluation",
    "IssuerConclusion",
    "IssuerConclusionKind",
    "OpenIssueState",
    "PortfolioReviewAnswer",
    "PortfolioReviewAnswerResolution",
    "PortfolioReviewAnswerRisk",
    "PortfolioReviewAssessmentSubmission",
    "PortfolioReviewCoverage",
    "PortfolioReviewDossier",
    "PortfolioReviewDossierCitation",
    "PortfolioReviewDossierFinding",
    "PortfolioReviewDossierIssuer",
    "PortfolioReviewEvidenceChild",
    "PortfolioReviewFindingDisposition",
    "PortfolioReviewIssueResolution",
    "PortfolioReviewMaterialIssue",
    "PortfolioReviewOpenIssue",
    "PortfolioReviewOutcome",
    "PortfolioReviewPublication",
    "PortfolioReviewReceipt",
    "PortfolioReviewRecommendation",
    "PortfolioReviewRoute",
    "PortfolioReviewUnresolvedQuestion",
    "RequiredAction",
    "RequiredActionKind",
    "ReviewState",
    "compile_portfolio_review_recommendation",
    "effective_interpretation",
    "finding_aliases",
    "finding_sources",
    "open_issue_aliases",
    "raised_issue_handle",
    "review_basis",
    "review_dispositions",
    "route_portfolio_review",
    "translate_submission",
]
