"""The Evidence & CRO projection: one small section beside the Portfolio result.

It names every intermediate state rather than collapsing them into an empty
panel: no book selected, no admitted evidence authority, awaiting or refreshing
evidence, a prepared analyst packet, evidence ready for review, expired or
ambiguous evidence, and one published review. Every value is a typed fact read
from a sealed artifact; nothing is computed here, and the browser is handed
strings it can only display. The next step out of each state is a request the
product composed, with its bindings, never one the page infers. An agent answers
the Analyst's packet and the CRO's dossier through its bundle; the product runs no
model of its own (AG2).
"""

from __future__ import annotations

import html
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Protocol

EvidenceCroState = Literal[
    "REVIEW_INPUT_INCOMPLETE",
    "NO_BOOK_TO_REVIEW",
    "EVIDENCE_AUTHORITY_NOT_ADMITTED",
    "EVIDENCE_REFRESH_IN_PROGRESS",
    "AWAITING_ALTERNATIVE_EVIDENCE",
    "ANALYST_PACKET_PREPARED",
    "ALTERNATIVE_EVIDENCE_READY_FOR_REVIEW",
    "ALTERNATIVE_EVIDENCE_EXPIRED",
    "ALTERNATIVE_EVIDENCE_SUPERSEDED",
    "EVIDENCE_SELECTION_AMBIGUOUS",
    "REVIEW_PUBLISHED",
]
"""What this section can be. Eleven states, and none of them is a guess."""

NextRequests = dict[str, dict[str, str]]
"""The requests that lead out of a state, keyed by step, each carrying the exact
book selector, Task or publication identity it is bound to. Composed by the
Host from what it verified; a page presents them and never derives one."""

EvidenceSelectionLabel = Literal[
    "UNIQUE_CURRENT",
    "EXPLICIT_CURRENT_SELECTION",
    "EXPLICIT_OLDER_CURRENT_SELECTION",
]

SELECTION_EXPLANATIONS: dict[str, str] = {
    "UNIQUE_CURRENT": "One current analysis answers this issuer scope.",
    "EXPLICIT_CURRENT_SELECTION": (
        "Several current analyses answer this issuer scope; the most recent one was selected."
    ),
    "EXPLICIT_OLDER_CURRENT_SELECTION": (
        "Several current analyses answer this issuer scope; an earlier one -- "
        "still within its validity window -- was selected in preference to the "
        "most recent."
    ),
}

AMBIGUOUS_EXPLANATION = (
    "More than one current Alternative Evidence analysis answers this issuer "
    "scope, and none was selected. A review cannot proceed until one is chosen, "
    "because which analysis was read is part of what a recommendation means."
)

BOOK_EXPLANATIONS: dict[str, str] = {
    "CONDITIONAL_RESEARCH_PROPOSAL": (
        "A close-marked conditional estimate, not an execution target. "
        "Review cannot change it or authorize a trade."
    ),
    "OBSERVED_RESEARCH_ENTRY": (
        "An observed daily-bar research entry, not verified venue execution. "
        "Review cannot change it or authorize a trade."
    ),
    "DEVELOPMENT_RESULT": (
        "A development result. A review here is a pre-freeze challenge: an objection "
        "asks Portfolio to reconsider the candidate before it is frozen."
    ),
    "FROZEN_CANDIDATE": (
        "A frozen candidate. A review here is a pre-Validation challenge: an objection "
        "asks Portfolio to reconsider the candidate under a new identity."
    ),
    "VALIDATED_HANDOFF": (
        "A validated handoff. A review here may explain risk, narrow claims or decline "
        "activation; it cannot change the validated book."
    ),
}


@dataclass(frozen=True, slots=True)
class EvidenceCroBook:
    """Identify the sealed book described by this section.

    The browser receives the book identity from the results list.
    """

    authority: str
    explanation: str
    result_hash: str | None
    formation_session: str | None = None
    held_count: int | None = None
    update_subject: dict[str, object] | None = None
    experiment_subject: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class EvidenceCroIssuerRow:
    """Summarize one reviewed issuer and its portfolio exposure."""

    entity_id: str
    tickers: tuple[str, ...]
    ending_weight: str
    signed_change: str
    transition: str
    exposure_band: str
    finding_count: int
    adverse_issue_count: int
    conclusion: str
    findings_summary: str
    """What became of this issuer's concerns, said rather than counted."""
    findings_summary_parts: tuple[tuple[str, str], ...] = ()
    """Generated words paired with exact authored text; never parse their flattened delimiters."""


@dataclass(frozen=True, slots=True)
class EvidenceCroRequiredAction:
    """Describe an action required before the review can advance."""

    action: str
    entity_id: str | None
    reason: str
    blocking: bool


@dataclass(frozen=True, slots=True)
class EvidenceCroIssueCard:
    """One material issue, with its four kinds of statement kept apart."""

    issue_handle: str
    affected_entities: tuple[str, ...]
    observed_source_text: tuple[str, ...]
    cro_inference: str
    portfolio_mitigation: str | None
    """None when the reviewer's answer stated none (the answer format asks for none)."""
    unknowns: tuple[str, ...]
    severity_if_true: str
    stated_interpretation: str
    effective_interpretation: str
    evidence_structure: str
    position_impact_direction: str
    exposure_band: str
    rule_id: str
    cited_finding_handles: tuple[str, ...]
    recommendation: str = ""
    """The reviewer's advice in words; empty for reviews sealed before it."""
    found_on: str | None = None
    """When the earliest finding it cites was found (ISO date): a reading carried
    from an earlier day says so (W3)."""
    open_since: str | None = None
    """The day an earlier review raised it, when it is an open issue of the
    issuers' register (W3); with `carried_on` when the reviewer left it as last
    assessed."""
    carried_on: str | None = None


@dataclass(frozen=True, slots=True)
class EvidenceCroCitation:
    """Identify a cited source span and when it became available."""

    span_handle: str
    document_handle: str
    entity_id: str
    available_at: str


@dataclass(frozen=True, slots=True)
class EvidenceCroEvidenceVersion:
    """One analysis a person may choose for this book's review."""

    analysis_publication_hash: str
    evidence_as_of: str
    evidence_expires_at: str
    issuer_count: int
    is_selected: bool


@dataclass(frozen=True, slots=True)
class EvidenceCroUnmappedPosition:
    """One position no admitted issuer could be found for."""

    listing_id: str
    ticker: str | None
    ending_weight: str
    absolute_change: str
    status: str
    reason: str


@dataclass(frozen=True, slots=True)
class EvidenceCroScopeCoverage:
    """How much of the book this review speaks for, in positions and in weight.

    Two denominators, kept apart because they answer different questions: the
    whole book, which is what a reader cares about, and the selected issuer
    scope, which is what the review actually examined. A review can be complete
    within its scope and still speak for very little of the book.
    """

    book_positions: int
    held_positions: int
    exited_positions: int
    reviewed_issuers: int
    reviewed_positions: int
    unmapped_positions: int
    whole_book_reviewed_weight: str
    whole_book_attainable_weight: str
    within_scope_reviewed_weight: str
    minimum_required_weight: str
    attainable_below_minimum: bool
    unmapped: tuple[EvidenceCroUnmappedPosition, ...]


@dataclass(frozen=True, slots=True)
class EvidenceCroStageWork:
    """Report progress in the stage owner's unit.

    The Host holds this telemetry while the task runs; it is not unit state.
    """

    stage: str
    """`acquire_source_evidence`, `canonicalize_documents`, `build_retrieval_generation`
    or `select_evidence_spans`."""
    unit_name: str
    """`filings`, `documents`, `chunks` or `units`."""
    completed: int
    total: int
    running: bool
    updated_at: str | None


@dataclass(frozen=True, slots=True)
class EvidenceCroUnitProgress:
    """One execution unit of the book's coverage run: which issuers, how far."""

    unit_id: str
    entity_ids: tuple[str, ...]
    listing_count: int
    ending_weight: str
    state: str
    """`PREPARED`, `PENDING`, `FAILED`, `ANALYZED`, `REVIEWED` or `NOT_STARTED`."""
    stages_done: int
    stages_expected: int
    failure_code: str | None
    packet_task_id: str | None
    packet_unit_id: str | None
    analysis_publication_hash: str | None
    evidence_as_of: str | None
    work: EvidenceCroStageWork | None = None
    """While the unit's run continues: its current stage's count (section 10.10)."""
    detail: str | None = None
    """A failed unit's words: what stopped it and what to do (V541)."""
    next_action: str | None = None
    issuers_without_source: tuple[str, ...] | None = None
    """A unit failed for its sources: its issuers without a source document; None where the
    failure names none, as a failure sealed before V541 does not."""


@dataclass(frozen=True, slots=True)
class EvidenceCroCoverageProgress:
    """Every held listing accounted for, as units of bounded work.

    Three states kept apart -- prepared, analyzed, reviewed -- because a
    prepared packet is not an analysis and an analysis is not a CRO decision.
    The denominators are the whole book: listings and issuers that mapped to
    nothing, and units that failed, stay in them with their weight.
    """

    run_hash: str | None
    unit_limit: int
    book_listings: int
    mapped_issuers: int
    unmapped_listings: int
    units_total: int
    units_prepared: int
    units_failed: int
    units_pending: int
    units_analyzed: int
    units_reviewed: int
    issuers_prepared: int
    issuers_failed: int
    issuers_analyzed: int
    prepared_weight: str
    analyzed_weight: str
    complete: bool
    """Every unit prepared: the book's preparation speaks for every mapped issuer."""
    units: tuple[EvidenceCroUnitProgress, ...]
    issuers_nothing_filed: int = 0
    """Holdings whose filing index at the cutoff held nothing in the window:
    named apart from the units, not prepared and never a finding of no risk."""
    nothing_filed_weight: str | None = None
    issuers_carried: int = 0
    """Holdings with no new filing in the window, every filing read earlier:
    not prepared again, reviewed by what those readings found (W3)."""
    carried_weight: str | None = None
    preparation: tuple[EvidenceCroStageWork, ...] = ()
    """While the book's run continues: filings fetched, documents canonicalized,
    chunks embedded and units selected, each of its own total (section 10.10)."""


@dataclass(frozen=True, slots=True)
class EvidenceCroCoverage:
    """Components, never a traffic light."""

    reviewed_ending_weight_coverage: str
    reviewed_absolute_change_coverage: str
    mapping_coverage: str
    selected_issuer_coverage: str
    nothing_filed_ending_weight_coverage: str | None = None
    unreached_ending_weight_coverage: str | None = None
    nothing_filed_window_days: int | None = None
    accounted_ending_weight_coverage: str | None = None
    """Read or officially filed nothing, only when the owner records both parts.

    None preserves a legacy review's absence of whole-book accounting proof.
    A no-filing record accounts for review work and is never a finding of no risk.
    """


@dataclass(frozen=True, slots=True)
class EvidenceCroProjection:
    """Everything the section renders. Safe by construction: no path, no store."""

    state: EvidenceCroState
    explanation: str
    review_publication_hash: str | None = None
    book: EvidenceCroBook | None = None
    portfolio_report_link: str | None = None
    disposition: str | None = None
    review_state: str | None = None
    review_attribution: tuple[str, ...] = ()
    required_actions: tuple[EvidenceCroRequiredAction, ...] = ()
    reviewed_entity_ids: tuple[str, ...] = ()
    evidence_as_of: str | None = None
    evidence_expires_at: str | None = None
    coverage: EvidenceCroCoverage | None = None
    gaps: tuple[str, ...] = ()
    scope_coverage: EvidenceCroScopeCoverage | None = None
    eligible_versions: tuple[EvidenceCroEvidenceVersion, ...] = ()
    issuer_rows: tuple[EvidenceCroIssuerRow, ...] = ()
    issue_cards: tuple[EvidenceCroIssueCard, ...] = ()
    citations: tuple[EvidenceCroCitation, ...] = ()
    published_reading: dict[str, object] | None = None
    """A requested read-only view of the published review's exact delivered evidence;
    the owner applies its filters, preserving boundary, unknown-time and historical context.
    Separate from the immutable export, whose bytes and hash remain unchanged."""
    claim_limits: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    """What the review did not read, written by the program (empty for a
    review sealed before the judgment-only answers)."""
    not_addressed_findings: tuple[str, ...] = ()
    not_addressed_issuers: tuple[str, ...] = ()
    """For an answered review: the findings the reviewer read and cited in no
    risk, and the issuers whose findings none of its risks names."""
    rule_ids: tuple[str, ...] = ()
    policy_version: str | None = None
    evidence_selection: EvidenceSelectionLabel | None = None
    evidence_selection_detail: str | None = None
    task_id: str | None = None
    task_lifecycle: str | None = None
    coverage_progress: EvidenceCroCoverageProgress | None = None
    """Per-unit progress of the book's coverage run, whenever a scope exists."""
    source_ways: dict[str, object] | None = None
    """Where units failed for too few sources under the recorded package: the person's official
    acquisition, and the package covering the heaviest such unit (V541)."""
    available_actions: tuple[str, ...] = field(default=("REFRESH_EVIDENCE", "REVIEW_WITH_CRO"))
    """The managed (Provider-backed) operations this workspace admits now; empty
    without an admitted credential. Native preparation, packet, dossier and
    submission entries are `next_requests`, not actions."""
    next_requests: NextRequests = field(default_factory=dict)


class MaterialIssuePort(Protocol):
    issue_handle: str
    cited_finding_handles: tuple[str, ...]
    affected_entities: tuple[str, ...]
    causal_channel: str
    uncertainty: str
    what_would_change_the_conclusion: str
    recommendation: str

    @property
    def severity_if_true(self) -> object: ...
    @property
    def evidence_interpretation(self) -> object: ...
    @property
    def portfolio_mitigation(self) -> object: ...
    @property
    def position_impact_direction(self) -> object: ...


class SubmissionPort(Protocol):
    material_issues: tuple[MaterialIssuePort, ...]


class IssueEvaluationPort(Protocol):
    issue_handle: str
    rule_id: str

    @property
    def exposure_band(self) -> object: ...
    @property
    def effective_interpretation(self) -> object: ...
    @property
    def structure(self) -> object: ...


class IssuerConclusionPort(Protocol):
    entity_id: str
    finding_count: int
    adverse_issue_count: int
    conclusion: str

    @property
    def exposure_band(self) -> object: ...


class RequiredActionPort(Protocol):
    entity_id: str | None
    reason: str
    blocking: bool

    @property
    def action(self) -> object: ...


class OutcomePort(Protocol):
    rule_ids: tuple[str, ...]
    issue_evaluations: tuple[IssueEvaluationPort, ...]
    issuer_conclusions: tuple[IssuerConclusionPort, ...]


class ReviewerPort(Protocol):
    actor_id: str

    @property
    def actor_kind(self) -> object: ...


class ReceiptPort(Protocol):
    """Expose the sealed review submission, outcome, and reviewer."""

    submission: SubmissionPort
    outcome: OutcomePort
    model_call_count: int

    @property
    def actor_submission(self) -> ReviewerPort:
        """Return the actor attributed to the submission."""
        ...

    @property
    def answer(self) -> object | None:
        """Return the submitted answer when one was recorded."""
        ...


class FindingPort(Protocol):
    finding_handle: str
    summary: str


class CitationPort(Protocol):
    span_handle: str
    document_handle: str
    entity_id: str

    @property
    def available_at(self) -> datetime: ...


class CoveragePort(Protocol):
    reviewed_ending_weight_coverage: float
    reviewed_absolute_change_coverage: float
    mapping_coverage: float
    selected_issuer_coverage: float
    missing_evidence: tuple[str, ...]
    unavailable_reasons: tuple[str, ...]
    nothing_filed_ending_weight_coverage: float | None
    unreached_ending_weight_coverage: float | None
    nothing_filed_window_days: int | None


class DossierIssuerPort(Protocol):
    entity_id: str
    tickers: tuple[str, ...]
    ending_weight: float
    signed_change: float
    transition: str

    @property
    def exposure_band(self) -> object: ...


class DossierPort(Protocol):
    """Structural, so the interface layer never imports a domain contract."""

    portfolio_report_link: str
    result_hash: str | None
    held_count: int
    findings: tuple[FindingPort, ...]
    citations: tuple[CitationPort, ...]
    issuers: tuple[DossierIssuerPort, ...]

    @property
    def book_authority(self) -> object:
        """Return the authority under which this book was sealed."""
        ...

    @property
    def coverage(self) -> CoveragePort:
        """Return the dossier's measured issuer coverage."""
        ...


class RecommendationPort(Protocol):
    """Expose the bounded recommendation and its evidence validity."""

    reviewed_entity_ids: tuple[str, ...]
    claim_limits: tuple[str, ...]
    reasons: tuple[str, ...]
    limitations: tuple[str, ...]
    policy_version: str
    required_actions: tuple[RequiredActionPort, ...]

    @property
    def route(self) -> object:
        """Return the recommendation's decision route."""
        ...

    @property
    def review_state(self) -> object:
        """Return the state of the published review."""
        ...

    @property
    def evidence_as_of(self) -> datetime:
        """Return the evidence cutoff for this recommendation."""
        ...

    @property
    def evidence_expires_at(self) -> datetime:
        """Return the end of the evidence validity window."""
        ...


NO_BOOK_EXPLANATION = (
    "No sealed book is selected. Complete a run in Explore, or select a completed "
    "result, and the review can begin. Nothing is invented here."
)

AUTHORITY_NOT_ADMITTED_EXPLANATION = (
    "A sealed book exists, but no issuer registry and listing authority are admitted "
    "for this workspace, so this section cannot name an issuer. Admit them to enable "
    "evidence preparation and review."
)

SOURCE_AUTHORITY_NOT_ADMITTED_EXPLANATION = (
    "This book's issuers are named, but no local source package and pinned retrieval "
    "runtime are admitted for this workspace, so no evidence can be prepared or read. "
    "Admit them (scripts/materialize_evidence_cro_authority.py) to enable source "
    "preparation."
)

REFRESH_EXPLANATION = (
    "An Alternative Evidence refresh is running. It is a separate background task, "
    "and this workspace runs one task at a time, so a review admitted now would wait "
    "behind it."
)

AWAITING_EVIDENCE_EXPLANATION = (
    "No admitted Alternative Evidence analysis covers this book's issuers yet. "
    "Prepare the admitted sources into an analyst packet and submit an analysis "
    "against it; a review without evidence would have nothing to read."
)

PACKET_PREPARED_EXPLANATION = (
    "An analyst packet is prepared for this issuer scope and no analysis has been "
    "submitted against it. Read the packet and submit a structured analysis brief; "
    "the product verifies the packet and its bindings when it is read."
)

READY_EXPLANATION = (
    "A current Alternative Evidence analysis covers this issuer scope. It is ready "
    "for the bounded CRO review: read the dossier and submit an assessment. No "
    "recommendation has been published yet."
)

EXPIRED_EXPLANATION = (
    "The matching Alternative Evidence analysis expired. It remains historical "
    "readback, but it cannot support a current CRO review. Prepare evidence again "
    "as of a new cutoff."
)

SUPERSEDED_EXPLANATION = (
    "The matching Alternative Evidence analysis was sealed under a retrieval "
    "contract this workspace has since superseded. It remains exact historical "
    "readback by its handle, but it cannot support a current CRO review. Prepare "
    "evidence again under the current contract."
)

MANAGED_WORK_NOTE = (
    "Managed (Provider-backed) analysis and review need an admitted credential; "
    "none is admitted here, so only the native entries above are offered."
)


def book_projection(
    *,
    authority: object,
    result_hash: str | None,
    formation_session: str | None = None,
    held_count: int | None = None,
    update_subject: dict[str, object] | None = None,
    experiment_subject: dict[str, object] | None = None,
) -> EvidenceCroBook:
    """Describe a sealed book without changing its authority or identity."""
    key = str(authority)
    return EvidenceCroBook(
        authority=key,
        explanation=(
            "An authored development replay at the selected holdings date, not current "
            "advice or venue execution. Review does not change this experiment."
            if experiment_subject is not None
            else BOOK_EXPLANATIONS.get(key, key)
        ),
        result_hash=result_hash,
        formation_session=formation_session,
        held_count=held_count,
        update_subject=update_subject,
        experiment_subject=experiment_subject,
    )


def no_book_to_review() -> EvidenceCroProjection:
    """Explain why review is unavailable before a book is selected."""
    return EvidenceCroProjection(
        state="NO_BOOK_TO_REVIEW", explanation=NO_BOOK_EXPLANATION, available_actions=()
    )


def evidence_authority_not_admitted() -> EvidenceCroProjection:
    """Explain that issuer and listing authority are absent."""
    return EvidenceCroProjection(
        state="EVIDENCE_AUTHORITY_NOT_ADMITTED",
        explanation=AUTHORITY_NOT_ADMITTED_EXPLANATION,
        available_actions=(),
    )


def source_authority_not_admitted(*, book: EvidenceCroBook) -> EvidenceCroProjection:
    """Issuers are named; the source package or retrieval runtime is what is missing."""
    return EvidenceCroProjection(
        state="EVIDENCE_AUTHORITY_NOT_ADMITTED",
        explanation=SOURCE_AUTHORITY_NOT_ADMITTED_EXPLANATION,
        book=book,
        available_actions=(),
    )


def evidence_refresh_in_progress(
    *, book: EvidenceCroBook, task_id: str | None = None, lifecycle: str | None = None
) -> EvidenceCroProjection:
    """Show the current evidence refresh and its task state."""
    return EvidenceCroProjection(
        state="EVIDENCE_REFRESH_IN_PROGRESS",
        explanation=REFRESH_EXPLANATION,
        book=book,
        task_id=task_id,
        task_lifecycle=lifecycle,
    )


def evidence_selection_ambiguous(
    *,
    book: EvidenceCroBook,
    versions: tuple[EvidenceCroEvidenceVersion, ...] = (),
) -> EvidenceCroProjection:
    """Require an explicit choice among current evidence versions."""
    return EvidenceCroProjection(
        state="EVIDENCE_SELECTION_AMBIGUOUS",
        explanation=AMBIGUOUS_EXPLANATION,
        book=book,
        eligible_versions=versions,
        required_actions=(
            EvidenceCroRequiredAction(
                action="SELECT_ANALYSIS",
                entity_id=None,
                reason=(
                    "The workspace authority must select the exact current analysis. "
                    "Another refresh does not resolve this ambiguity."
                ),
                blocking=True,
            ),
        ),
    )


def awaiting_alternative_evidence(
    *, book: EvidenceCroBook, detail: str | None = None
) -> EvidenceCroProjection:
    """Explain that no admitted analysis covers this book yet."""
    return EvidenceCroProjection(
        state="AWAITING_ALTERNATIVE_EVIDENCE",
        explanation=AWAITING_EVIDENCE_EXPLANATION
        if detail is None
        else (AWAITING_EVIDENCE_EXPLANATION + " " + detail),
        book=book,
    )


def analyst_packet_prepared(
    *, book: EvidenceCroBook, task_id: str, evidence_as_of: str, evidence_expires_at: str
) -> EvidenceCroProjection:
    """A packet the product prepared and can still read; the analysis is the next step."""
    return EvidenceCroProjection(
        state="ANALYST_PACKET_PREPARED",
        explanation=PACKET_PREPARED_EXPLANATION,
        book=book,
        task_id=task_id,
        task_lifecycle="SUCCEEDED",
        evidence_as_of=evidence_as_of,
        evidence_expires_at=evidence_expires_at,
    )


def alternative_evidence_ready_for_review(
    *,
    book: EvidenceCroBook,
    evidence_as_of: str | None = None,
    evidence_expires_at: str | None = None,
) -> EvidenceCroProjection:
    """Show the admitted analysis and its validity window for review."""
    return EvidenceCroProjection(
        state="ALTERNATIVE_EVIDENCE_READY_FOR_REVIEW",
        explanation=READY_EXPLANATION,
        book=book,
        evidence_as_of=evidence_as_of,
        evidence_expires_at=evidence_expires_at,
    )


def alternative_evidence_expired(
    *,
    book: EvidenceCroBook,
    evidence_as_of: str | None = None,
    evidence_expires_at: str | None = None,
) -> EvidenceCroProjection:
    """Require new evidence after the selected analysis expires."""
    return EvidenceCroProjection(
        state="ALTERNATIVE_EVIDENCE_EXPIRED",
        explanation=EXPIRED_EXPLANATION
        if evidence_expires_at is None
        else EXPIRED_EXPLANATION.replace("expired.", f"expired at {evidence_expires_at}."),
        book=book,
        evidence_as_of=evidence_as_of,
        evidence_expires_at=evidence_expires_at,
        required_actions=(
            EvidenceCroRequiredAction(
                action="PREPARE_EVIDENCE",
                entity_id=None,
                reason=(
                    "Prepare Alternative Evidence as of a new cutoff and submit an analysis "
                    "before requesting a current CRO review."
                ),
                blocking=True,
            ),
        ),
    )


def alternative_evidence_superseded(
    *,
    book: EvidenceCroBook,
    evidence_as_of: str | None = None,
    evidence_expires_at: str | None = None,
) -> EvidenceCroProjection:
    """Sealed under a supported historical contract: readable, never current."""
    return EvidenceCroProjection(
        state="ALTERNATIVE_EVIDENCE_SUPERSEDED",
        explanation=SUPERSEDED_EXPLANATION,
        book=book,
        evidence_as_of=evidence_as_of,
        evidence_expires_at=evidence_expires_at,
        required_actions=(
            EvidenceCroRequiredAction(
                action="PREPARE_EVIDENCE",
                entity_id=None,
                reason=(
                    "Prepare Alternative Evidence under the current retrieval contract and "
                    "submit an analysis before requesting a current CRO review."
                ),
                blocking=True,
            ),
        ),
    )


def project_published_review(
    *,
    recommendation: RecommendationPort,
    dossier: DossierPort,
    receipt: ReceiptPort,
    percent: Callable[[float], str],
    change: Callable[[float], str],
    book: EvidenceCroBook,
    selection: EvidenceSelectionLabel | None = None,
    scope: object | None = None,
    exposure: object | None = None,
) -> EvidenceCroProjection:
    """Project one published review. Every figure arrives already written."""
    evaluations = {value.issue_handle: value for value in receipt.outcome.issue_evaluations}
    conclusions = {value.entity_id: value for value in receipt.outcome.issuer_conclusions}
    children = [
        (
            str(child.unit_id),
            tuple(child.ordered_entity_ids),
            str(child.analysis_publication_hash),
            (child.read_as_of or child.evidence_as_of).date().isoformat(),
        )
        for child in getattr(dossier, "evidence_children", ())
    ]
    found = found_dates(
        [(publication, date) for _unit, _entities, publication, date in children],
        default=recommendation.evidence_as_of.date().isoformat(),
        handles=[value.finding_handle for value in dossier.findings],
    )
    carried = {
        entity: date
        for unit_id, entities, _publication, date in children
        if unit_id.startswith("c")
        for entity in entities
    }
    raised = {
        str(value.open_issue_handle): value.raised_on.isoformat()
        for value in getattr(dossier, "open_issues", ())
    }
    open_on: dict[str, list[str]] = {}
    open_parts: dict[str, list[tuple[str, str]]] = {}
    for issue in receipt.submission.material_issues:
        handle = getattr(issue, "open_issue_handle", None)
        if handle in raised:
            for entity in issue.affected_entities:
                open_on.setdefault(entity, []).append(
                    f"open issue from {raised[handle]}: {issue.causal_channel[:160]}"
                )
                open_parts.setdefault(entity, []).append(
                    (f"open issue from {raised[handle]}", issue.causal_channel[:160])
                )
    # An answered review asks for no disposition: what it left unnamed is
    # written here, from the risks it named.
    answered = receipt.answer is not None
    cited = {
        handle
        for value in receipt.submission.material_issues
        for handle in value.cited_finding_handles
    }
    issues = []
    for issue in receipt.submission.material_issues:
        evaluation = evaluations.get(issue.issue_handle)
        issues.append(
            EvidenceCroIssueCard(
                issue_handle=issue.issue_handle,
                affected_entities=tuple(issue.affected_entities),
                observed_source_text=tuple(
                    _finding_summary(dossier, handle) for handle in issue.cited_finding_handles
                ),
                cro_inference=issue.causal_channel,
                portfolio_mitigation=None
                if issue.portfolio_mitigation is None
                else str(issue.portfolio_mitigation),
                unknowns=tuple(
                    value
                    for value in (issue.uncertainty, issue.what_would_change_the_conclusion)
                    if value
                ),
                severity_if_true=str(issue.severity_if_true),
                stated_interpretation=str(issue.evidence_interpretation),
                effective_interpretation=(
                    str(issue.evidence_interpretation)
                    if evaluation is None
                    else str(evaluation.effective_interpretation)
                ),
                evidence_structure="NOT_EVALUATED"
                if evaluation is None
                else str(evaluation.structure),
                position_impact_direction=str(issue.position_impact_direction),
                exposure_band="NOT_EVALUATED"
                if evaluation is None
                else str(evaluation.exposure_band),
                rule_id="NOT_EVALUATED" if evaluation is None else evaluation.rule_id,
                cited_finding_handles=tuple(issue.cited_finding_handles),
                recommendation=issue.recommendation,
                found_on=min(
                    (found[handle] for handle in issue.cited_finding_handles if handle in found),
                    default=None,
                ),
                open_since=raised.get(str(getattr(issue, "open_issue_handle", None))),
                carried_on=(
                    None
                    if (carried_day := getattr(issue, "carried_on", None)) is None
                    else carried_day.isoformat()
                ),
            )
        )
    weight_of = {issuer.entity_id: issuer.ending_weight for issuer in dossier.issuers}
    rows = []
    not_addressed_issuers = []
    for issuer in dossier.issuers:
        conclusion = conclusions.get(issuer.entity_id)
        findings = 0 if conclusion is None else conclusion.finding_count
        adverse = 0 if conclusion is None else conclusion.adverse_issue_count
        kind = "NOT_EVALUATED" if conclusion is None else conclusion.conclusion
        own = [
            value.finding_handle
            for value in dossier.findings
            if issuer.entity_id in getattr(value, "affected_entities", ())
        ]
        named = sum(handle in cited for handle in own)
        if answered and own and not named:
            not_addressed_issuers.append(issuer.entity_id)
        summary = (
            _answered_findings_summary(kind, findings=len(own), named=named)
            if answered
            else _findings_summary(kind, findings=findings, adverse=adverse)
        )
        summary_parts = [*open_parts.get(issuer.entity_id, ())]
        if issuer.entity_id in carried:
            summary_parts.append((f"no new filing; read {carried[issuer.entity_id]}", ""))
        summary_parts.append((summary, ""))
        rows.append(
            EvidenceCroIssuerRow(
                entity_id=issuer.entity_id,
                tickers=tuple(issuer.tickers),
                ending_weight=percent(issuer.ending_weight),
                signed_change=change(issuer.signed_change),
                transition=issuer.transition,
                exposure_band=str(issuer.exposure_band),
                finding_count=findings,
                adverse_issue_count=adverse,
                conclusion=kind,
                findings_summary=(
                    "".join(f"{value}; " for value in open_on.get(issuer.entity_id, ()))
                    + (
                        ""
                        if issuer.entity_id not in carried
                        else f"no new filing; read {carried[issuer.entity_id]}: "
                    )
                )
                + summary,
                findings_summary_parts=tuple(summary_parts),
            )
        )
    coverage = dossier.coverage
    nothing_filed = getattr(coverage, "nothing_filed_ending_weight_coverage", None)
    unreached = getattr(coverage, "unreached_ending_weight_coverage", None)
    return EvidenceCroProjection(
        scope_coverage=_scope_coverage(scope, exposure, percent=percent, change=change),
        state="REVIEW_PUBLISHED",
        explanation=(
            "One published review of the issuers named below, against the approved "
            "sources at the stated cutoff. It is not an activation and changes no weight."
        ),
        book=book,
        portfolio_report_link=dossier.portfolio_report_link,
        disposition=str(recommendation.route),
        review_state=str(recommendation.review_state),
        review_attribution=_review_attribution(
            str(receipt.actor_submission.actor_kind),
            receipt.actor_submission.actor_id,
            receipt.model_call_count,
        ),
        required_actions=tuple(
            EvidenceCroRequiredAction(
                action=str(value.action),
                entity_id=value.entity_id,
                reason=value.reason,
                blocking=value.blocking,
            )
            for value in recommendation.required_actions
        ),
        reviewed_entity_ids=tuple(recommendation.reviewed_entity_ids),
        evidence_as_of=recommendation.evidence_as_of.isoformat(),
        evidence_expires_at=recommendation.evidence_expires_at.isoformat(),
        coverage=EvidenceCroCoverage(
            reviewed_ending_weight_coverage=percent(coverage.reviewed_ending_weight_coverage),
            reviewed_absolute_change_coverage=percent(coverage.reviewed_absolute_change_coverage),
            mapping_coverage=percent(coverage.mapping_coverage),
            selected_issuer_coverage=percent(coverage.selected_issuer_coverage),
            nothing_filed_ending_weight_coverage=(
                None if nothing_filed is None else percent(nothing_filed)
            ),
            unreached_ending_weight_coverage=None if unreached is None else percent(unreached),
            nothing_filed_window_days=getattr(coverage, "nothing_filed_window_days", None),
            accounted_ending_weight_coverage=(
                None
                if nothing_filed is None or unreached is None
                else percent(coverage.reviewed_ending_weight_coverage + nothing_filed)
            ),
        ),
        gaps=tuple((*coverage.missing_evidence, *coverage.unavailable_reasons)),
        issuer_rows=tuple(rows),
        # The cards in the order the report shows its risks: by the heaviest
        # holding each names (a display order; the sealed order is the answer's).
        issue_cards=tuple(
            sorted(
                issues,
                key=lambda card: (
                    -max(
                        (weight_of.get(entity, 0.0) for entity in card.affected_entities),
                        default=0.0,
                    ),
                    card.issue_handle,
                ),
            )
        ),
        citations=tuple(
            EvidenceCroCitation(
                span_handle=value.span_handle,
                document_handle=value.document_handle,
                entity_id=value.entity_id,
                available_at=value.available_at.isoformat(),
            )
            for value in dossier.citations
        ),
        claim_limits=tuple(recommendation.claim_limits),
        reasons=tuple(recommendation.reasons),
        limitations=tuple(recommendation.limitations),
        not_addressed_findings=tuple(
            value.finding_handle for value in dossier.findings if value.finding_handle not in cited
        )
        if answered
        else (),
        not_addressed_issuers=tuple(not_addressed_issuers),
        rule_ids=tuple(receipt.outcome.rule_ids),
        policy_version=recommendation.policy_version,
        evidence_selection=selection,
        evidence_selection_detail=(
            None if selection is None else SELECTION_EXPLANATIONS[selection]
        ),
    )


def _scope_coverage(
    scope: object | None,
    exposure: object | None,
    *,
    percent: Callable[[float], str],
    change: Callable[[float], str],
) -> EvidenceCroScopeCoverage | None:
    """Read the admitted scope owner. Nothing here is inferred from a name."""
    if scope is None:
        return None
    from alphalattice.oversight.chief_risk_officer.decision.portfolio_review import (
        MINIMUM_REVIEWED_ENDING_WEIGHT_COVERAGE,
    )

    failures = tuple(scope.mapping_failures)  # type: ignore[attr-defined]
    selected = tuple(scope.selected_issuers)  # type: ignore[attr-defined]
    positions = () if exposure is None else tuple(exposure.positions)  # type: ignore[attr-defined]
    held = sum(1 for value in positions if value.ending_weight > 0.0)
    reviewed_positions = sum(len(value.listing_ids) for value in selected)
    attainable = float(scope.mapping_coverage)  # type: ignore[attr-defined]
    return EvidenceCroScopeCoverage(
        book_positions=len(positions),
        held_positions=held,
        exited_positions=len(positions) - held,
        reviewed_issuers=len(selected),
        reviewed_positions=reviewed_positions,
        unmapped_positions=len(failures),
        whole_book_reviewed_weight=percent(
            float(scope.reviewed_ending_weight_coverage)  # type: ignore[attr-defined]
        ),
        whole_book_attainable_weight=percent(attainable),
        within_scope_reviewed_weight=percent(
            float(scope.selected_issuer_coverage)  # type: ignore[attr-defined]
        ),
        minimum_required_weight=percent(MINIMUM_REVIEWED_ENDING_WEIGHT_COVERAGE),
        attainable_below_minimum=attainable < MINIMUM_REVIEWED_ENDING_WEIGHT_COVERAGE,
        unmapped=tuple(
            EvidenceCroUnmappedPosition(
                listing_id=value.listing_id,
                ticker=value.ticker,
                ending_weight=percent(value.ending_weight),
                absolute_change=change(value.absolute_change),
                status="HELD" if value.ending_weight > 0.0 else "EXITED",
                reason=str(value.reason),
            )
            for value in failures
        ),
    )


def _findings_summary(conclusion: str, *, findings: int, adverse: int) -> str:
    """One phrase for the issuer row, chosen by what the review actually did.

    A count of zero adverse issues is only reassuring when something adjudicated
    them. When a guard stopped the review first the same zero means nobody
    looked, and printing it beside the concern count reads as a clean bill.
    """
    concerns = "1 concern" if findings == 1 else f"{findings} concerns"
    if conclusion == "NOTHING_FILED":
        return "nothing filed in the window"
    if conclusion == "CONCERNS_NOT_ADJUDICATED":
        return f"{concerns} raised, none adjudicated"
    if conclusion == "NOT_EVALUATED":
        return f"{concerns} raised, not evaluated"
    if findings == 0:
        return "no concern raised"
    adjudicated = "1 adverse" if adverse == 1 else f"{adverse} adverse"
    return f"{concerns}, {adjudicated}"


def _answered_findings_summary(conclusion: str, *, findings: int, named: int) -> str:
    """Summarize the issuer's findings in an answered review.

    Count findings read and cited by a named risk, never a clean bill.
    """
    if conclusion == "NOTHING_FILED":
        return "nothing filed in the window"
    if findings == 0:
        return (
            "no evidence could be read" if conclusion == "EVIDENCE_GAP" else "no finding reported"
        )
    read = "1 finding read" if findings == 1 else f"{findings} findings read"
    return f"{read}, none named as a risk" if named == 0 else f"{read}, {named} named in a risk"


def found_dates(
    children: list[tuple[str, str]], *, default: str, handles: list[str]
) -> dict[str, str]:
    """Return the ISO date on which each finding was found.

    Use the child publication's date for a carried reading, or the review cutoff
    for a single-unit dossier's bare handles.
    """
    prefixes = [(f"FIND-P{publication[:8].upper()}-", date) for publication, date in children]
    return {
        handle: next((date for prefix, date in prefixes if handle.startswith(prefix)), default)
        for handle in handles
    }


def _finding_summary(dossier: DossierPort, handle: str) -> str:
    for finding in dossier.findings:
        if finding.finding_handle == handle:
            return str(finding.summary)
    return handle


def _review_attribution(kind: str, actor_id: str, model_calls: int) -> tuple[str, ...]:
    if kind == "HOST_FALLBACK" and actor_id == "carried-forward":
        return (
            "Carried forward by the Host: nothing the CRO read has changed since its last "
            "review, which stands as of this cutoff.",
            f"Product-managed model calls: {model_calls}.",
        )
    return (
        f"Assessed by {kind}: {actor_id}.",
        f"Product-managed model calls: {model_calls}.",
        *(
            (
                "External host/model is not attested; its token usage and cost "
                "are unavailable to the product.",
            )
            if kind == "EXTERNAL_AUTOMATION"
            else ()
        ),
    )


def _review_evidence_content(review: dict[str, Any], evidence: object) -> str:
    """Human-readable findings/citations from the already verified export, not new judgments."""
    dossier = review["dossier"]
    recommendation = review["recommendation"]
    cited_handles = {citation["span_handle"] for citation in dossier.get("citations", [])}
    spans = (
        {span["span_handle"]: span for span in evidence.get("verified_spans", [])}
        if isinstance(evidence, dict)
        else {}
    )
    receipt = review.get("receipt", {})
    outcome = receipt.get("outcome", {}) if isinstance(receipt, dict) else {}
    completeness = str(outcome.get("disposition_completeness", "UNSTATED"))
    answered = _answered_findings(review)
    pieces = [
        "<h3>Review completeness: " + html.escape(str(recommendation["review_state"])) + "</h3>",
        (
            "<p>Finding dispositions: "
            + html.escape(completeness)
            + (
                " (the reviewer answered every qualified finding)"
                if completeness == "COMPLETE"
                else " (some findings were deferred; a deferred finding is not reviewed)"
                if completeness == "PARTIAL"
                else " (this assessment carried no per-finding dispositions -- a historical "
                "reading sealed before dispositions were required; an empty issue list is "
                "not evidence that every finding was reviewed)"
            )
            + "</p>"
        )
        if answered is None
        else "<p>Findings: "
        + html.escape(_answered_words(answered))
        + ". The reviewer answers with the major negatives it finds; no disposition is asked "
        "for the rest.</p>",
        "<p>Findings below are analyst concerns, not automatically adjudicated risks. "
        "Selected-issuer coverage is not whole-book coverage. This readback renders the "
        "sealed summaries, findings and citations; the reviewer's evidence-aware judgment "
        "is recorded in the dispositions and issues, not re-derived here.</p>",
        "<h3>Coverage limits</h3><ul>",
    ]
    for gap in (
        *dossier["coverage"].get("unavailable_reasons", []),
        *dossier["coverage"].get("missing_evidence", []),
        *dossier.get("limitations", []),
    ):
        pieces.append("<li>" + html.escape(str(gap)) + "</li>")
    pieces.append("</ul>")
    questions = dossier.get("unresolved_questions", [])
    if questions:
        pieces.append("<h3>Analyst unresolved questions (open, not adjudicated)</h3><ul>")
        for question in questions:
            pieces.append(
                "<li>"
                + html.escape(str(question.get("unit_id", "")))
                + ": "
                + html.escape(str(question.get("text", "")))
                + "</li>"
            )
        pieces.append("</ul>")
    submission = receipt.get("submission", {}) if isinstance(receipt, dict) else {}
    dispositions = (
        submission.get("finding_dispositions", []) if isinstance(submission, dict) else []
    )
    if dispositions:
        pieces.append("<h3>Finding dispositions</h3><ul>")
        for value in dispositions:
            pieces.append(
                "<li>"
                + html.escape(str(value.get("finding_handle", "")))
                + ": "
                + html.escape(str(value.get("disposition", "")))
                + " &mdash; "
                + html.escape(str(value.get("rationale", "")))
                + "</li>"
            )
        pieces.append("</ul>")
    pieces.append("<h3>Recorded issuer conclusions</h3><ul>")
    for item in recommendation.get("issuer_conclusions", []):
        pieces.append(
            "<li>"
            + html.escape(str(item["entity_id"]))
            + ": "
            + html.escape(str(item["conclusion"]))
            + " (findings: "
            + html.escape(str(item["finding_count"]))
            + (
                "; adjudication not completed"
                if item["conclusion"] == "CONCERNS_NOT_ADJUDICATED"
                else "; adverse issues recorded: " + html.escape(str(item["adverse_issue_count"]))
            )
            + ")</li>"
        )
    pieces.append("</ul><h3>Analyst findings</h3>")
    for finding in dossier.get("findings", []):
        pieces.append(
            "<article><h4>"
            + html.escape(str(finding["finding_handle"]))
            + "</h4><p>"
            + html.escape(str(finding["summary"]))
            + "</p><p>"
            + html.escape(str(finding["topic"]))
            + " / "
            + html.escape(str(finding["direction"]))
            + " / "
            + html.escape(str(finding["structure"]))
            + "</p>"
        )
        for label, key in (
            ("Supporting", "supporting_span_handles"),
            ("Contrary", "contradicting_span_handles"),
        ):
            handles = finding.get(key, [])
            pieces.append(
                "<p>"
                + label
                + " citations: "
                + (
                    ", ".join(
                        (
                            '<a href="#citation-'
                            + html.escape(handle, quote=True)
                            + '">'
                            + html.escape(handle)
                            + "</a>"
                            if handle in cited_handles
                            else html.escape(handle) + " (citation record unavailable)"
                        )
                        for handle in handles
                    )
                    if handles
                    else "None recorded; not proof of absence"
                )
                + "</p>"
            )
        pieces.append(
            "<p>" + html.escape("; ".join(finding.get("limitations", []))) + "</p></article>"
        )
    pieces.append("<h3>Citations and source excerpts</h3>")
    for citation in dossier.get("citations", []):
        handle = citation["span_handle"]
        span = spans.get(handle)
        pieces.append(
            '<details id="citation-'
            + html.escape(handle, quote=True)
            + '"><summary>'
            + html.escape(
                " / ".join(
                    str(citation[k])
                    for k in ("span_handle", "document_handle", "entity_id", "available_at")
                )
            )
            + "</summary>"
        )
        if span is None:
            pieces.append("<p>Source excerpt unavailable in this export.</p>")
        else:
            time = span.get("time")
            if isinstance(time, dict):
                pieces.append(
                    "<p>Source time: published "
                    + html.escape(str(time.get("published_at") or "not stated"))
                    + " ("
                    + html.escape(str(time.get("published_precision")))
                    + " precision); accepted "
                    + html.escape(str(time.get("accepted_at") or "unknown"))
                    + "; available "
                    + html.escape(str(time.get("available_at")))
                    + " by "
                    + html.escape(str(time.get("availability_basis")))
                    + "; report period end "
                    + html.escape(str(time.get("report_period_end") or "not stated"))
                    + ". A midnight under DATE precision is a date, not an instant; "
                    "none of these is when an event happened.</p>"
                )
            pieces.append(
                "<p>Source characters "
                + html.escape(str(span.get("character_start", "Not recorded")))
                + " to "
                + html.escape(str(span.get("character_end", "Not recorded")))
                + "; lines "
                + html.escape(str(span["start_line"]))
                + " to "
                + html.escape(str(span["end_line"]))
                + "</p><blockquote>"
                + html.escape(span["excerpt"])
                + "</blockquote>"
            )
            unit = span.get("unit_declaration")
            if isinstance(unit, dict):
                pieces.append(
                    "<p>Source scale statement, characters "
                    + html.escape(str(unit.get("character_start")))
                    + " to "
                    + html.escape(str(unit.get("character_end")))
                    + ": "
                    + html.escape(str(unit.get("text")))
                    + " (the nearest before this passage; an amount naming its own "
                    "scale is in that scale)</p>"
                )
        pieces.append("</details>")
    return "".join(pieces)


_ROUTE_IN_WORDS = {
    "NO_MATERIAL_OBJECTION": "no material objection was recorded against the reviewed scope",
    "REQUEST_EVIDENCE_REFRESH": "the review asks for the evidence to be refreshed before it "
    "can conclude",
    "REQUIRE_HUMAN_REVIEW": "the review requires a human reviewer before any conclusion",
    "REDUCE_EXPOSURE": "the review records material objections that a human must weigh; "
    "no weight changes here",
}


_ANSWERED_ROUTE_IN_WORDS = {
    "NO_MATERIAL_OBJECTION": "no major negative was found in the evidence read",
    "ACCEPT_WITH_LIMITS": "the reviewer named material risks that do not rise to an "
    "objection; they are listed with the limits of what was read",
    "HUMAN_REVIEW_REQUIRED": "a high-severity risk on an exposed holding rests on contested "
    "or limited findings; a person should read it (advisory, not blocking)",
    "MATERIAL_OBJECTION": "a supported high-severity risk on an exposed holding is a major "
    "negative for the book; a human must weigh it and no weight changes here",
}
"""The route in words for a review answered in the judgment-only format; a
review sealed before reads in the words it was published with."""


def _route_in_words(route: str, *, answered: bool = False) -> str:
    if answered and route in _ANSWERED_ROUTE_IN_WORDS:
        return _ANSWERED_ROUTE_IN_WORDS[route]
    return _ROUTE_IN_WORDS.get(route, route.replace("_", " ").lower())


def _dispositions_of(review: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return each finding's sealed or derived disposition.

    For judgment-only answers, a cited finding is a material issue; all others
    were not addressed.
    """
    receipt = review.get("receipt", {})
    submission = receipt.get("submission", {}) if isinstance(receipt, dict) else {}
    if isinstance(receipt, dict) and "answer" in receipt and isinstance(submission, dict):
        from alphalattice.oversight.chief_risk_officer.decision.portfolio_review import (
            PortfolioReviewAssessmentSubmission,
            review_dispositions,
        )

        return {
            value.finding_handle: value.model_dump(mode="json")
            for value in review_dispositions(
                PortfolioReviewAssessmentSubmission.model_validate(submission),
                answered=True,
                finding_handles=tuple(
                    str(finding["finding_handle"])
                    for finding in review.get("dossier", {}).get("findings", [])
                ),
            )
        }
    values = submission.get("finding_dispositions", []) if isinstance(submission, dict) else []
    return {str(value.get("finding_handle")): value for value in values}


def _answered_findings(review: dict[str, Any]) -> tuple[int, int] | None:
    """Count cited and unnamed findings in a judgment-only answer.

    Return None for a review sealed before answers; its dispositions are read
    from the seal.
    """
    receipt = review.get("receipt", {})
    if not isinstance(receipt, dict) or "answer" not in receipt:
        return None
    derived = _dispositions_of(review)
    cited = sum(1 for value in derived.values() if value["disposition"] == "MATERIAL_ISSUE")
    return cited, len(derived) - cited


def _answered_words(answered: tuple[int, int]) -> str:
    cited, rest = answered
    return (
        f"{cited} cited by a named risk, {rest} read and not named as a risk"
        if cited or rest
        else "none in the dossier"
    )


def _issue_register_text(issue: dict[str, Any]) -> str:
    """Render one material issue's register entry.

    Use the sealed words, or the reason, severity, support, and advice from an
    answered risk.
    """
    head = (
        html.escape(str(issue.get("issue_handle", "")))
        + " ("
        + html.escape(str(issue.get("relevance", "")))
        + ", "
        + html.escape(str(issue.get("position_impact_direction", "")))
        + "): "
        + html.escape(str(issue.get("causal_channel", "")))
        + " Severity if true: "
        + html.escape(str(issue.get("severity_if_true", "")))
        + "; evidence: "
        + html.escape(str(issue.get("evidence_interpretation", "")))
    )
    if "portfolio_mitigation" not in issue:
        return head + ". Recommendation: " + html.escape(str(issue.get("recommendation", "")))
    return (
        head
        + "; mitigation: "
        + html.escape(str(issue.get("portfolio_mitigation", "")))
        + ". Uncertainty: "
        + html.escape(str(issue.get("uncertainty", "")))
        + " What would change the conclusion: "
        + html.escape(str(issue.get("what_would_change_the_conclusion", "")))
    )


def _issues_of(review: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """The reviewer's material issues, by each finding they cite."""
    receipt = review.get("receipt", {})
    submission = receipt.get("submission", {}) if isinstance(receipt, dict) else {}
    values = submission.get("material_issues", []) if isinstance(submission, dict) else []
    by_finding: dict[str, list[dict[str, Any]]] = {}
    for value in values:
        for handle in value.get("cited_finding_handles", []):
            by_finding.setdefault(str(handle), []).append(value)
    return by_finding


def _conclusion_section(review: dict[str, Any]) -> str:
    """Render the conclusion section with route and review scope.

    Include the sealed recommendation and its limitations without creating a
    new judgment or trading authority.
    """
    dossier = review["dossier"]
    recommendation = review["recommendation"]
    receipt = review.get("receipt", {})
    outcome = receipt.get("outcome", {}) if isinstance(receipt, dict) else {}
    completeness = str(outcome.get("disposition_completeness", "UNSTATED"))
    actor = receipt["actor_submission"] if isinstance(receipt, dict) else {}
    answered = _answered_findings(review)
    pieces = ["<h3>1. Conclusion</h3>"]
    pieces.append(
        "<p><strong>"
        + html.escape(str(recommendation["route"]))
        + "</strong>: "
        + html.escape(_route_in_words(str(recommendation["route"]), answered=answered is not None))
        + ". Review completeness: "
        + html.escape(str(recommendation["review_state"]))
        + (
            "; finding dispositions: "
            + html.escape(completeness)
            + (
                " (every qualified finding answered)"
                if completeness == "COMPLETE"
                else " (some findings deferred; a deferred finding is not reviewed)"
                if completeness == "PARTIAL"
                else " (no per-finding dispositions were carried by this assessment)"
            )
            if answered is None
            else "; findings: " + html.escape(_answered_words(answered))
        )
        + ".</p>"
    )
    pieces.append(
        "<p>Scope: "
        + html.escape(str(len(dossier.get("issuers", []))))
        + " reviewed issuers of the book; evidence as of "
        + html.escape(str(dossier.get("evidence_as_of")))
        + ", expiring "
        + html.escape(str(dossier.get("evidence_expires_at")))
        + ".</p>"
    )
    pieces.append(
        "<ul>"
        + "".join(f"<li>{html.escape(str(v))}</li>" for v in recommendation.get("reasons", []))
        + "</ul>"
    )
    unread = recommendation.get("limitations", [])
    if unread:
        pieces.append(
            "<p>What this review did not read (written by the program):</p><ul>"
            + "".join(f"<li>{html.escape(str(v))}</li>" for v in unread)
            + "</ul>"
        )
    if actor:
        pieces.extend(
            f"<p>{html.escape(line)}</p>"
            for line in _review_attribution(
                actor["actor_kind"], actor["actor_id"], receipt["model_call_count"]
            )
        )
    limits = [*dossier.get("claim_limits", []), *dossier.get("limitations", [])]
    pieces.append(
        "<p>Principal limitations:</p><ul>"
        + "".join(f"<li>{html.escape(str(v))}</li>" for v in limits)
        + "</ul>"
    )
    return "".join(pieces)


def _coverage_section(review: dict[str, Any]) -> str:
    """Render the coverage section for every known held listing.

    Keep mapping and each completion stage separate from count and weight
    coverage.
    """
    dossier = review["dossier"]
    recommendation = review["recommendation"]
    coverage_labels = {
        "mapping_coverage": "Whole-book mapped ending weight",
        "reviewed_ending_weight_coverage": "Whole-book reviewed ending weight",
        "reviewed_absolute_change_coverage": "Whole-book reviewed absolute change",
        "selected_issuer_coverage": "Selected-issuer scope reviewed coverage",
        "nothing_filed_ending_weight_coverage": (
            "Whole-book ending weight with nothing filed in the window (not a finding of no risk)"
        ),
        "unreached_ending_weight_coverage": "Whole-book ending weight not read",
    }
    dispositions = _dispositions_of(review)
    conclusions = {
        str(item["entity_id"]): item for item in recommendation.get("issuer_conclusions", [])
    }
    findings_by_entity: dict[str, list[dict[str, Any]]] = {}
    for finding in dossier.get("findings", []):
        for entity in finding.get("affected_entities", []):
            findings_by_entity.setdefault(str(entity), []).append(finding)
    pieces = ["<h3>2. Coverage and work state</h3>"]
    pieces.append(
        "<dl>"
        + "".join(
            f"<dt>{html.escape(coverage_labels.get(key, key.replace('_', ' ')))}</dt>"
            f"<dd>{float(value):.3%}</dd>"
            for key, value in dossier["coverage"].items()
            if key.endswith("coverage")
        )
        + f"<dt>Reviewed issuers</dt><dd>{len(dossier.get('issuers', []))}</dd>"
        + "<dt>Mapping failures</dt><dd>"
        + html.escape(str(dossier.get("mapping_failure_count")))
        + "</dd><dt>Held positions</dt><dd>"
        + html.escape(str(dossier.get("held_count")))
        + "</dd>"
        + "</dl>"
    )
    pieces.append(
        "<table><thead><tr><th>Issuer</th><th>Tickers</th><th>Ending weight</th>"
        "<th>Transition</th><th>Analysis state</th><th>Findings</th><th>Dispositions</th>"
        "<th>Conclusion</th></tr></thead><tbody>"
    )
    for issuer in dossier.get("issuers", []):
        entity = str(issuer["entity_id"])
        entity_findings = findings_by_entity.get(entity, [])
        answered = sum(1 for f in entity_findings if str(f["finding_handle"]) in dispositions)
        conclusion = conclusions.get(entity, {})
        pieces.append(
            "<tr><td>"
            + html.escape(entity)
            + "</td><td>"
            + html.escape(", ".join(issuer.get("tickers", [])))
            + "</td><td>"
            + f"{float(issuer.get('ending_weight', 0.0)):.3%}"
            + "</td><td>"
            + html.escape(str(issuer.get("transition")))
            + "</td><td>"
            + html.escape(str(issuer.get("review_state", "UNSTATED")))
            + "</td><td>"
            + str(len(entity_findings))
            + "</td><td>"
            + f"{answered} of {len(entity_findings)}"
            + "</td><td>"
            + html.escape(str(conclusion.get("conclusion", "UNSTATED")))
            + "</td></tr>"
        )
    pieces.append("</tbody></table>")
    if _answered_findings(review) is not None:
        cited = {
            handle
            for handle, value in dispositions.items()
            if value["disposition"] == "MATERIAL_ISSUE"
        }
        unnamed = [
            str(issuer["entity_id"])
            for issuer in dossier.get("issuers", [])
            if findings_by_entity.get(str(issuer["entity_id"]))
            and not any(
                str(f["finding_handle"]) in cited
                for f in findings_by_entity[str(issuer["entity_id"])]
            )
        ]
        if unnamed:
            pieces.append(
                "<p>Issuers whose findings no risk names (written by the program): "
                + html.escape(", ".join(unnamed))
                + ".</p>"
            )
    gaps = [
        *dossier["coverage"].get("unavailable_reasons", []),
        *dossier["coverage"].get("missing_evidence", []),
    ]
    if gaps:
        pieces.append(
            "<p>Coverage limits:</p><ul>"
            + "".join(f"<li>{html.escape(str(v))}</li>" for v in gaps)
            + "</ul>"
        )
    return "".join(pieces)


def _finding_row(
    finding: dict[str, Any], disposition: dict[str, Any] | None, cited_handles: set[str]
) -> str:
    handles = [
        *finding.get("supporting_span_handles", []),
        *finding.get("contradicting_span_handles", []),
    ]
    return (
        "<tr><td>"
        + html.escape(str(finding["finding_handle"]))
        + "</td><td>"
        + html.escape(", ".join(finding.get("affected_entities", [])))
        + "</td><td>"
        + html.escape(str(finding["summary"]))
        + "</td><td>"
        + html.escape(
            f"{finding['topic']} / "
            f"{finding.get('lifecycle') or 'lifecycle not recorded'} / "
            f"{finding['direction']}"
        )
        + "</td><td>"
        + (
            "None recorded"
            if disposition is None
            else html.escape(str(disposition.get("rationale", "")))
        )
        + "</td><td>"
        + html.escape(
            f"{len(finding.get('supporting_span_handles', []))} for / "
            f"{len(finding.get('contradicting_span_handles', []))} against"
        )
        + " &mdash; "
        + ", ".join(
            '<a href="#citation-' + html.escape(h, quote=True) + '">' + html.escape(h) + "</a>"
            if h in cited_handles
            else html.escape(h) + " (citation record unavailable)"
            for h in handles
        )
        + "</td></tr>"
    )


def _registers_section(review: dict[str, Any]) -> str:
    """Render material and other finding registers.

    Account for each finding, including resolved, deferred, and unanswered ones.
    """
    dossier = review["dossier"]
    cited_handles = {str(citation["span_handle"]) for citation in dossier.get("citations", [])}
    dispositions = _dispositions_of(review)
    issues = _issues_of(review)
    groups: dict[str, list[str]] = {
        "MATERIAL_ISSUE": [],
        "NOT_MATERIAL": [],
        "RESOLVED_WITH_EVIDENCE": [],
        "DEFERRED": [],
        "UNANSWERED": [],
        "NOT_ADDRESSED": [],
    }
    for finding in dossier.get("findings", []):
        handle = str(finding["finding_handle"])
        disposition = dispositions.get(handle)
        kind = "UNANSWERED" if disposition is None else str(disposition["disposition"])
        row = _finding_row(finding, disposition, cited_handles)
        if kind == "MATERIAL_ISSUE":
            for issue in issues.get(handle, []):
                row += '<tr><td></td><td colspan="5">' + _issue_register_text(issue) + "</td></tr>"
        groups.setdefault(kind, []).append(row)
    header = (
        "<table><thead><tr><th>Finding</th><th>Issuer(s)</th><th>Atomic issue</th>"
        "<th>Topic / state / direction</th><th>Reviewer's basis</th>"
        "<th>Support / contrary</th></tr></thead><tbody>"
    )
    pieces = ["<h3>3. Material issue register</h3>"]
    pieces.append(
        header + "".join(groups["MATERIAL_ISSUE"]) + "</tbody></table>"
        if groups["MATERIAL_ISSUE"]
        else "<p>No finding was dispositioned as a material issue.</p>"
    )
    pieces.append("<h3>4. Non-material, resolved and deferred register</h3>")
    for kind, title in (
        ("NOT_MATERIAL", "Not material"),
        ("RESOLVED_WITH_EVIDENCE", "Resolved with evidence"),
        ("DEFERRED", "Deferred (not reviewed; a remaining obligation)"),
        ("UNANSWERED", "Not answered by this assessment (a remaining obligation)"),
        ("NOT_ADDRESSED", "Read and not named as a risk by the reviewer"),
    ):
        if groups[kind]:
            pieces.append(
                f"<h4>{html.escape(title)}</h4>"
                + header
                + "".join(groups[kind])
                + "</tbody></table>"
            )
    if not any(
        groups[k]
        for k in (
            "NOT_MATERIAL",
            "RESOLVED_WITH_EVIDENCE",
            "DEFERRED",
            "UNANSWERED",
            "NOT_ADDRESSED",
        )
    ):
        pieces.append("<p>Every finding is in the material issue register.</p>")
    return "".join(pieces)


def _actions_section(review: dict[str, Any]) -> str:
    """5. The existing typed actions and the analyst's open questions; nothing invented."""
    dossier = review["dossier"]
    recommendation = review["recommendation"]
    pieces = ["<h3>5. Follow-up actions</h3>"]
    actions = recommendation.get("required_actions", [])
    pieces.append(
        "<ul>"
        + "".join(
            "<li>"
            + html.escape(str(a["action"]))
            + (" (blocking)" if a.get("blocking") else "")
            + ": "
            + html.escape(str(a["reason"]))
            + "</li>"
            for a in actions
        )
        + "</ul>"
        if actions
        else "<p>No typed action is required by this route.</p>"
    )
    questions = dossier.get("unresolved_questions", [])
    if questions:
        pieces.append("<p>Analyst unresolved questions (open, not adjudicated):</p><ul>")
        pieces.extend(
            "<li>"
            + html.escape(str(q.get("unit_id", "")))
            + ": "
            + html.escape(str(q.get("text", "")))
            + "</li>"
            for q in questions
        )
        pieces.append("</ul>")
    return "".join(pieces)


def _changes_section(changes: object) -> str:
    """6. Changes since an explicitly selected prior report, or why none are shown."""
    pieces = ["<h3>6. Changes since the selected prior report</h3>"]
    if not isinstance(changes, dict):
        pieces.append("<p>No prior report was selected; nothing is compared.</p>")
        return "".join(pieces)
    if changes.get("status") != "REVIEW_CHANGES":
        pieces.append(
            "<p>Unavailable: "
            + html.escape(str(changes.get("reason", "the reviews are not comparable")))
            + ".</p>"
        )
        return "".join(pieces)
    pieces.append(
        "<p>Prior review "
        + html.escape(str(changes["prior_publication_hash"]))
        + " (published "
        + html.escape(str(changes["prior_published_at"]))
        + ", evidence as of "
        + html.escape(str(changes["prior_evidence_as_of"]))
        + "); route "
        + html.escape(str(changes["route"]["prior"]))
        + " &rarr; "
        + html.escape(str(changes["route"]["current"]))
        + ".</p>"
    )
    findings = changes.get("findings", {})
    for key, title in (
        ("new", "New findings"),
        ("changed", "Changed findings"),
        ("no_longer_found", "Findings no longer found (not thereby resolved)"),
    ):
        rows = findings.get(key, [])
        pieces.append(f"<h4>{title}: {len(rows)}</h4>")
        if rows:
            pieces.append(
                "<ul>"
                + "".join(
                    "<li>"
                    + html.escape(", ".join(r.get("affected_entities", [])))
                    + " / "
                    + html.escape(str(r.get("topic")))
                    + ": "
                    + html.escape(str(r.get("summary")))
                    + (
                        " (was: " + html.escape(str(r["prior_summary"])) + ")"
                        if r.get("prior_summary")
                        else ""
                    )
                    + "</li>"
                    for r in rows
                )
                + "</ul>"
            )
    moved = changes.get("dispositions_moved", [])
    pieces.append(f"<h4>Dispositions that moved: {len(moved)}</h4>")
    if moved:
        pieces.append(
            "<ul>"
            + "".join(
                "<li>"
                + html.escape(str(r["finding_handle"]))
                + ": "
                + html.escape(str(r.get("prior_disposition")))
                + " &rarr; "
                + html.escape(str(r.get("disposition")))
                + "</li>"
                for r in moved
            )
            + "</ul>"
        )
    coverage = changes.get("coverage_changes", {})
    positions = changes.get("position_changes", [])
    pieces.append(f"<h4>Coverage changes: {len(coverage)}; position changes: {len(positions)}</h4>")
    if coverage:
        pieces.append(
            "<ul>"
            + "".join(
                f"<li>{html.escape(k)}: {float(v['prior']):.3%} &rarr; "
                f"{float(v['current']):.3%}</li>"
                for k, v in coverage.items()
            )
            + "</ul>"
        )
    if positions:
        pieces.append(
            "<ul>"
            + "".join(
                "<li>"
                + html.escape(str(p["entity_id"]))
                + ": "
                + html.escape(str(p["change"]))
                + " "
                + html.escape(str(p.get("prior_ending_weight")))
                + " &rarr; "
                + html.escape(str(p.get("ending_weight")))
                + "</li>"
                for p in positions
            )
            + "</ul>"
        )
    pieces.append(_typed_changes(changes.get("typed_disclosure_changes")))
    pieces.append("<p>" + html.escape(str(changes.get("claim", ""))) + "</p>")
    return "".join(pieces)


def _typed_changes(typed: object) -> str:
    """Render typed disclosure changes between two reviews.

    Report each scope and changed field as supplied by the comparison owner,
    including any unavailable reason.
    """
    if not isinstance(typed, dict):
        return "<h4>Typed disclosure changes: not compared</h4>"
    if typed.get("status") != "TYPED_DISCLOSURE_CHANGES":
        return (
            "<h4>Typed disclosure changes: unavailable</h4><p>"
            + html.escape(str(typed.get("reason", "")))
            + "</p>"
        )
    rows = typed.get("changes", [])
    kinds = sorted({str(r.get("kind")) for r in rows})
    summary = f" ({', '.join(kinds)})" if kinds else ""
    pieces = [f"<h4>Typed disclosure changes: {len(rows)}{html.escape(summary)}</h4>"]
    moved = [r for r in rows if r.get("kind") != "CONTINUING_ASSERTION"]
    if moved:
        pieces.append(
            "<ul>"
            + "".join(
                "<li>"
                + html.escape(str(r.get("scope")))
                + ": "
                + html.escape(str(r.get("kind")))
                + " &mdash; "
                + html.escape(str(r.get("detail")))
                + (
                    "<ul>"
                    + "".join(
                        "<li>" + html.escape(str(field)) + "</li>"
                        for field in r.get("changed_fields", [])
                    )
                    + "</ul>"
                    if r.get("changed_fields")
                    else ""
                )
                + "</li>"
                for r in moved
            )
            + "</ul>"
        )
    return "".join(pieces)


def _appendix_section(review: dict[str, Any], evidence: object) -> str:
    """Render the sealed evidence appendix.

    Include readable excerpts, verified citations, producing publications, and
    method versions.
    """
    dossier = review["dossier"]
    pieces = ["<h3>7. Evidence appendix</h3>"]
    spans = evidence.get("verified_spans", []) if isinstance(evidence, dict) else []
    children = dossier.get("evidence_children", [])
    pieces.append(
        "<p>The reviewer received the dossier's findings and citation handles; this export "
        "carries the "
        + str(len(spans))
        + " verified excerpts those handles resolve to, from "
        + str(len(children) or 1)
        + " evidence publication(s). A summary is not an original-evidence review; the "
        "excerpts below are the delivered set.</p>"
    )
    if children:
        pieces.append(
            "<ul>"
            + "".join(
                "<li>"
                + html.escape(str(c["unit_id"]))
                + ": "
                + html.escape(", ".join(c.get("ordered_entity_ids", [])))
                + " &mdash; publication "
                + html.escape(str(c["analysis_publication_hash"]))
                + ", evidence as of "
                + html.escape(str(c["evidence_as_of"]))
                + "</li>"
                for c in children
            )
            + "</ul>"
        )
    publication = review.get("publication", {})
    pieces.append(
        "<p>Method versions: "
        + "; ".join(
            html.escape(f"{k} {publication[k]}")
            for k in ("decision_policy_hash", "dossier_hash", "publication_hash")
            if k in publication
        )
        + ".</p>"
    )
    pieces.append(_review_evidence_content(review, evidence))
    return "".join(pieces)


def _weight_of(dossier: dict[str, Any]) -> dict[str, float]:
    return {
        str(issuer.get("entity_id")): float(issuer.get("ending_weight", 0.0))
        for issuer in dossier.get("issuers", [])
        if isinstance(issuer, dict)
    }


def _by_weight(issues: list[dict[str, Any]], weights: dict[str, float]) -> list[dict[str, Any]]:
    """Risks by the heaviest holding each names, then by handle: a display order."""
    return sorted(
        issues,
        key=lambda issue: (
            -max(
                (weights.get(str(e), 0.0) for e in issue.get("affected_entities", [])), default=0.0
            ),
            str(issue.get("issue_handle")),
        ),
    )


def _glance_section(review: dict[str, Any]) -> str:
    """Render the first-screen summary of the CRO review.

    Order risks by holding weight and show the review limits.
    """
    dossier = review["dossier"]
    recommendation = review["recommendation"]
    receipt = review.get("receipt", {})
    submission = receipt.get("submission", {}) if isinstance(receipt, dict) else {}
    issues = submission.get("material_issues", []) if isinstance(submission, dict) else []
    answered = _answered_findings(review)
    route = str(recommendation["route"])
    weights = _weight_of(dossier)
    opened = {
        str(value.get("open_issue_handle")): str(value.get("raised_on", ""))
        for value in dossier.get("open_issues", [])
        if isinstance(value, dict)
    }
    found = found_dates(
        [
            (
                str(child.get("analysis_publication_hash", "")),
                str(child.get("read_as_of") or child.get("evidence_as_of", ""))[:10],
            )
            for child in dossier.get("evidence_children", [])
            if isinstance(child, dict)
        ],
        default=str(recommendation.get("evidence_as_of", ""))[:10],
        handles=[
            str(value.get("finding_handle"))
            for value in dossier.get("findings", [])
            if isinstance(value, dict)
        ],
    )
    pieces = [
        '<section id="review-at-a-glance"><h3>At a glance</h3>',
        "<p><strong>"
        + html.escape(route)
        + "</strong>: "
        + html.escape(_route_in_words(route, answered=answered is not None))
        + ". Review completeness: "
        + html.escape(str(recommendation["review_state"]))
        + ".</p>",
    ]
    rationale = submission.get("overall_rationale") if isinstance(submission, dict) else None
    if rationale:
        pieces.append("<p>" + html.escape(str(rationale)) + "</p>")
    if issues:
        rows = []
        for issue in _by_weight(list(issues), weights):
            entities = [str(e) for e in issue.get("affected_entities", [])]
            heaviest = max((weights.get(e, 0.0) for e in entities), default=0.0)
            first = min(
                (
                    found[str(handle)]
                    for handle in issue.get("cited_finding_handles", [])
                    if str(handle) in found
                ),
                default="",
            )
            if issue.get("open_issue_handle") in opened:
                first = f"open since {opened[issue['open_issue_handle']]}" + (
                    f", as assessed {issue['carried_on']}" if issue.get("carried_on") else ""
                )
            rows.append(
                "<tr><td>"
                + html.escape(", ".join(entities))
                + f"</td><td>{heaviest:.3%}</td><td>"
                + html.escape(first)
                + "</td><td>"
                + html.escape(str(issue.get("severity_if_true", "")))
                + "</td><td>"
                + html.escape(str(issue.get("evidence_interpretation", "")))
                + "</td><td>"
                + html.escape(str(issue.get("causal_channel", "")))
                + "</td><td>"
                + html.escape(str(issue.get("recommendation", "")))
                + "</td></tr>"
            )
        pieces.append(
            "<table><thead><tr><th>Holding</th><th>Ending weight</th><th>Found</th>"
            "<th>Severity if true</th>"
            "<th>Confidence</th><th>Why it matters</th><th>Recommendation</th></tr></thead>"
            "<tbody>" + "".join(rows) + "</tbody></table>"
        )
    unread = recommendation.get("limitations", [])
    if unread:
        pieces.append(
            f"<p>{len(unread)} limitation(s) of what this review read are listed in the full "
            "review below; nothing unread is a finding of no risk.</p>"
        )
    pieces.append("</section>")
    return "".join(pieces)


def _provenance_section(snapshot: dict[str, object]) -> str:
    """Render the identities bound by this export.

    The structured objects, coverage facts, and citations remain in the export's
    JSON rather than being embedded again here.
    """
    review = snapshot.get("review")
    rows: list[tuple[str, object]] = []
    if isinstance(review, dict):
        for label, part, key in (
            ("Review publication", "publication", "publication_hash"),
            ("Dossier", "dossier", "dossier_hash"),
            ("Recommendation", "recommendation", "recommendation_hash"),
            ("Receipt", "receipt", "receipt_hash"),
        ):
            value = review.get(part)
            if isinstance(value, dict) and value.get(key):
                rows.append((label, value[key]))
    return (
        "<details><summary>Exact provenance</summary><dl>"
        + "".join(
            f"<dt>{html.escape(label)}</dt><dd><code>{html.escape(str(value))}</code></dd>"
            for label, value in rows
        )
        + "</dl><p>The complete structured record is this export's JSON.</p></details>"
    )


def render_review_export(snapshot: dict[str, object], base_html: str | None) -> str:
    """Append exact review facts to the existing Portfolio HTML, without I/O."""
    review = snapshot["review"]
    pieces = ['<section id="linked-evidence-review"><h2>Evidence &amp; CRO</h2>']
    pieces.append(f"<p>{html.escape(str(snapshot['claim']))}</p>")
    subject = snapshot.get("experiment_subject") or snapshot.get("update_subject")
    if isinstance(subject, dict):
        pieces.append(
            "<dl>"
            + "".join(
                f"<dt>{html.escape(key.replace('_', ' '))}</dt>"
                f"<dd>{html.escape(str(subject[key]))}</dd>"
                for key in (
                    (
                        "experiment_task_id",
                        "portfolio_session",
                        "preceding_session",
                        "research_as_of_session",
                        "research_as_of_phase",
                        "position_basis",
                    )
                    if snapshot.get("experiment_subject") is not None
                    else (
                        "strategy_package_id",
                        "position_basis",
                        "formation_session",
                        "entry_session",
                        "observed_through",
                    )
                )
            )
            + "</dl>"
        )
    if isinstance(review, dict):
        # The first screen: the route, the CRO's words and its risks by weight;
        # the seven sections are the whole review, on demand.
        pieces.append(_glance_section(review))
        pieces.append(
            "<details><summary>The full review: conclusion, coverage, registers, "
            "actions, changes and evidence</summary>"
        )
        pieces.append(_conclusion_section(review))
        pieces.append(_coverage_section(review))
        pieces.append(_registers_section(review))
        pieces.append(_actions_section(review))
        pieces.append(_changes_section(snapshot.get("changes_since_prior_review")))
        pieces.append(_appendix_section(review, snapshot.get("evidence")))
        pieces.append("</details>")
    else:
        pieces.append("<p>Not reviewed. No review was selected or created by this export.</p>")
    pieces.append(_provenance_section(snapshot) + "</section>")
    section = "".join(pieces)
    if base_html is not None:
        summary = (
            "No review selected"
            if not isinstance(review, dict)
            else str(review["recommendation"]["route"])
        )
        notice = (
            "<aside><h2>Linked recommendation review</h2><p>"
            + html.escape(summary)
            + "</p><p>The Portfolio report below is preserved as issued. Its statement "
            "that CRO was not evaluated describes publication time; the separate "
            "review in this export does not rewrite that history.</p>"
            '<p><a href="#linked-evidence-review">Read the bound Evidence &amp; CRO '
            "assessment, coverage and citations</a></p></aside>"
        )
        if "<main>" in base_html:
            base_html = base_html.replace("<main>", "<main>" + notice, 1)
        elif "<body>" in base_html:
            base_html = base_html.replace("<body>", "<body>" + notice, 1)
        elif "<h1" in base_html:
            base_html = base_html.replace("<h1", notice + "<h1", 1)
        # The existing CU renderer uses HTML's implicit body; do not silently
        # drop the review because no explicit </body> was written.
        for closing in ("</main>", "</body>", "</html>"):
            if closing in base_html:
                return base_html.replace(closing, section + closing, 1)
        raise ValueError("local_application.review_export_html_invalid")
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>Portfolio Evidence Review</title><body>" + section + "</body></html>"
    )


def evidence_cro_body(projection: EvidenceCroProjection) -> dict[str, object]:
    """One JSON shape for every state, so a client has no branch to get wrong."""
    coverage = projection.coverage
    book = projection.book
    return {
        "state": projection.state,
        "explanation": projection.explanation,
        "review_attribution": list(projection.review_attribution),
        "book": (
            None
            if book is None
            else {
                "authority": book.authority,
                "explanation": book.explanation,
                "result_hash": book.result_hash,
                "formation_session": book.formation_session,
                "held_count": book.held_count,
                **(
                    {"update_subject": book.update_subject}
                    if book.update_subject is not None
                    else {}
                ),
                **(
                    {"experiment_subject": book.experiment_subject}
                    if book.experiment_subject is not None
                    else {}
                ),
            }
        ),
        "portfolio_report_link": projection.portfolio_report_link,
        # The publication identity the owner already resolved, for every subject
        # kind: it is the public locator history and the export route take, so a
        # client never has to infer it from history metadata. None when there is
        # no published review to name.
        "review_publication_hash": projection.review_publication_hash,
        "disposition": projection.disposition,
        "review_state": projection.review_state,
        "required_actions": [
            {
                "action": value.action,
                "entity_id": value.entity_id,
                "reason": value.reason,
                "blocking": value.blocking,
            }
            for value in projection.required_actions
        ],
        "evidence_selection": projection.evidence_selection,
        "evidence_selection_detail": projection.evidence_selection_detail,
        "task_id": projection.task_id,
        "task_lifecycle": projection.task_lifecycle,
        "reviewed_entity_ids": list(projection.reviewed_entity_ids),
        "evidence_as_of": projection.evidence_as_of,
        "evidence_expires_at": projection.evidence_expires_at,
        "coverage": (
            None
            if coverage is None
            else {
                "reviewed_ending_weight": coverage.reviewed_ending_weight_coverage,
                "reviewed_absolute_change": coverage.reviewed_absolute_change_coverage,
                "mapping": coverage.mapping_coverage,
                "selected_issuers": coverage.selected_issuer_coverage,
                "nothing_filed_ending_weight": coverage.nothing_filed_ending_weight_coverage,
                "unreached_ending_weight": coverage.unreached_ending_weight_coverage,
                "nothing_filed_window_days": coverage.nothing_filed_window_days,
                "accounted_ending_weight": coverage.accounted_ending_weight_coverage,
            }
        ),
        "gaps": list(projection.gaps),
        "eligible_versions": [
            {
                "analysis_publication_hash": value.analysis_publication_hash,
                "evidence_as_of": value.evidence_as_of,
                "evidence_expires_at": value.evidence_expires_at,
                "issuer_count": value.issuer_count,
                "is_selected": value.is_selected,
            }
            for value in projection.eligible_versions
        ],
        "scope_coverage": None
        if projection.scope_coverage is None
        else {
            "book_positions": projection.scope_coverage.book_positions,
            "held_positions": projection.scope_coverage.held_positions,
            "exited_positions": projection.scope_coverage.exited_positions,
            "reviewed_issuers": projection.scope_coverage.reviewed_issuers,
            "reviewed_positions": projection.scope_coverage.reviewed_positions,
            "unmapped_positions": projection.scope_coverage.unmapped_positions,
            "whole_book_reviewed_weight": projection.scope_coverage.whole_book_reviewed_weight,
            "whole_book_attainable_weight": projection.scope_coverage.whole_book_attainable_weight,
            "within_scope_reviewed_weight": projection.scope_coverage.within_scope_reviewed_weight,
            "minimum_required_weight": projection.scope_coverage.minimum_required_weight,
            "attainable_below_minimum": projection.scope_coverage.attainable_below_minimum,
            "unmapped": [
                {
                    "listing_id": row.listing_id,
                    "ticker": row.ticker,
                    "ending_weight": row.ending_weight,
                    "absolute_change": row.absolute_change,
                    "status": row.status,
                    "reason": row.reason,
                }
                for row in projection.scope_coverage.unmapped
            ],
        },
        "issuer_rows": [
            {
                "entity_id": row.entity_id,
                "tickers": list(row.tickers),
                "ending_weight": row.ending_weight,
                "signed_change": row.signed_change,
                "transition": row.transition,
                "exposure_band": row.exposure_band,
                "finding_count": row.finding_count,
                "adverse_issue_count": row.adverse_issue_count,
                "conclusion": row.conclusion,
                "findings_summary": row.findings_summary,
                "findings_summary_parts": [list(part) for part in row.findings_summary_parts],
            }
            for row in projection.issuer_rows
        ],
        "issue_cards": [
            {
                "issue_handle": card.issue_handle,
                "affected_entities": list(card.affected_entities),
                "observed_source_text": list(card.observed_source_text),
                "cro_inference": card.cro_inference,
                "portfolio_mitigation": card.portfolio_mitigation,
                "unknowns": list(card.unknowns),
                "severity_if_true": card.severity_if_true,
                "stated_interpretation": card.stated_interpretation,
                "effective_interpretation": card.effective_interpretation,
                "evidence_structure": card.evidence_structure,
                "position_impact_direction": card.position_impact_direction,
                "exposure_band": card.exposure_band,
                "rule_id": card.rule_id,
                "cited_finding_handles": list(card.cited_finding_handles),
                "recommendation": card.recommendation,
                "found_on": card.found_on,
                "open_since": card.open_since,
                "carried_on": card.carried_on,
            }
            for card in projection.issue_cards
        ],
        "citations": [
            {
                "span_handle": value.span_handle,
                "document_handle": value.document_handle,
                "entity_id": value.entity_id,
                "available_at": value.available_at,
            }
            for value in projection.citations
        ],
        "published_reading": projection.published_reading,
        "coverage_progress": None
        if projection.coverage_progress is None
        else {
            "run_hash": projection.coverage_progress.run_hash,
            "unit_limit": projection.coverage_progress.unit_limit,
            "book_listings": projection.coverage_progress.book_listings,
            "mapped_issuers": projection.coverage_progress.mapped_issuers,
            "unmapped_listings": projection.coverage_progress.unmapped_listings,
            "units_total": projection.coverage_progress.units_total,
            "units_prepared": projection.coverage_progress.units_prepared,
            "units_failed": projection.coverage_progress.units_failed,
            "units_pending": projection.coverage_progress.units_pending,
            "units_analyzed": projection.coverage_progress.units_analyzed,
            "units_reviewed": projection.coverage_progress.units_reviewed,
            "issuers_prepared": projection.coverage_progress.issuers_prepared,
            "issuers_failed": projection.coverage_progress.issuers_failed,
            "issuers_analyzed": projection.coverage_progress.issuers_analyzed,
            "prepared_weight": projection.coverage_progress.prepared_weight,
            "analyzed_weight": projection.coverage_progress.analyzed_weight,
            "complete": projection.coverage_progress.complete,
            "issuers_nothing_filed": projection.coverage_progress.issuers_nothing_filed,
            "nothing_filed_weight": projection.coverage_progress.nothing_filed_weight,
            "issuers_carried": projection.coverage_progress.issuers_carried,
            "carried_weight": projection.coverage_progress.carried_weight,
            **(
                {
                    "preparation": [
                        _stage_work(value) for value in projection.coverage_progress.preparation
                    ]
                }
                if projection.coverage_progress.preparation
                else {}
            ),
            "units": [
                {
                    "unit_id": unit.unit_id,
                    "entity_ids": list(unit.entity_ids),
                    "listing_count": unit.listing_count,
                    "ending_weight": unit.ending_weight,
                    "state": unit.state,
                    "stages_done": unit.stages_done,
                    "stages_expected": unit.stages_expected,
                    "failure_code": unit.failure_code,
                    "packet_task_id": unit.packet_task_id,
                    "packet_unit_id": unit.packet_unit_id,
                    "analysis_publication_hash": unit.analysis_publication_hash,
                    "evidence_as_of": unit.evidence_as_of,
                    **({} if unit.work is None else {"work": _stage_work(unit.work)}),
                    **(
                        {}
                        if unit.detail is None
                        else {
                            "detail": unit.detail,
                            "next_action": unit.next_action,
                            "issuers_without_source": None
                            if unit.issuers_without_source is None
                            else list(unit.issuers_without_source),
                        }
                    ),
                }
                for unit in projection.coverage_progress.units
            ],
        },
        "source_ways": projection.source_ways,
        "claim_limits": list(projection.claim_limits),
        "reasons": list(projection.reasons),
        "limitations": list(projection.limitations),
        "not_addressed_findings": list(projection.not_addressed_findings),
        "not_addressed_issuers": list(projection.not_addressed_issuers),
        "rule_ids": list(projection.rule_ids),
        "policy_version": projection.policy_version,
        "available_actions": list(projection.available_actions),
        "next_requests": {key: dict(value) for key, value in projection.next_requests.items()},
    }


def _stage_work(work: EvidenceCroStageWork) -> dict[str, object]:
    return {
        "stage": work.stage,
        "unit_name": work.unit_name,
        "completed": work.completed,
        "total": work.total,
        "running": work.running,
        "updated_at": work.updated_at,
    }


__all__ = [
    "AMBIGUOUS_EXPLANATION",
    "AWAITING_EVIDENCE_EXPLANATION",
    "BOOK_EXPLANATIONS",
    "EXPIRED_EXPLANATION",
    "MANAGED_WORK_NOTE",
    "NO_BOOK_EXPLANATION",
    "PACKET_PREPARED_EXPLANATION",
    "READY_EXPLANATION",
    "REFRESH_EXPLANATION",
    "SELECTION_EXPLANATIONS",
    "SOURCE_AUTHORITY_NOT_ADMITTED_EXPLANATION",
    "SUPERSEDED_EXPLANATION",
    "DossierPort",
    "EvidenceCroBook",
    "EvidenceCroCitation",
    "EvidenceCroCoverage",
    "EvidenceCroCoverageProgress",
    "EvidenceCroEvidenceVersion",
    "EvidenceCroIssueCard",
    "EvidenceCroIssuerRow",
    "EvidenceCroProjection",
    "EvidenceCroRequiredAction",
    "EvidenceCroScopeCoverage",
    "EvidenceCroState",
    "EvidenceCroUnitProgress",
    "EvidenceCroUnmappedPosition",
    "EvidenceSelectionLabel",
    "NextRequests",
    "ReceiptPort",
    "RecommendationPort",
    "alternative_evidence_expired",
    "alternative_evidence_ready_for_review",
    "alternative_evidence_superseded",
    "analyst_packet_prepared",
    "awaiting_alternative_evidence",
    "book_projection",
    "evidence_authority_not_admitted",
    "evidence_cro_body",
    "evidence_refresh_in_progress",
    "evidence_selection_ambiguous",
    "no_book_to_review",
    "project_published_review",
    "source_authority_not_admitted",
]
