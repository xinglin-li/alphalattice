"""Typed evidence-extraction contracts: obligation, packet receipt, brief, package.

The analyst -- installed Agent, Human or external automation -- receives one
Host-selected packet of exact source spans and returns one answer of atomic
findings. Each finding says what a document states, classified on a small
taxonomy, and cites the excerpts that carry it by the alias its view shows.
The Host maps the aliases back, assigns the handles, derives how the findings
are structured (how many distinct documents support or contradict each one)
and seals the brief; it never asks the analyst for a severity, a Portfolio
effect or a probability: those belong to the review that consumes the package.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from enum import StrEnum
from typing import ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.protocols.actor_execution import ActorSubmissionBinding
from alphalattice.protocols.actor_execution.answers import AnswerProblem

from ..contracts import (
    AlternativeEvidenceContract,
    AlternativeEvidenceReviewStatus,
    validate_contract_identity,
)
from ..retrieval.contracts import SPAN_HANDLE_PATTERN, AlternativeEvidenceResolvedSpan

_HASH = r"^[0-9a-f]{64}$"


class AlternativeEvidenceResearchObligation(AlternativeEvidenceContract):
    """The one bounded question an analysis answers, and nothing about the book.

    An issuer axis, a cutoff, the source families that may be read and the
    checks that must be performed. A weight-only change to a Portfolio cannot
    rotate this identity, and an analyst reading it cannot be steered by how
    large a position is.
    """

    question: str = Field(min_length=1, max_length=1200)
    ordered_entity_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence_as_of: datetime
    approved_source_families: tuple[str, ...] = Field(min_length=1, max_length=8)
    required_checks: tuple[str, ...] = Field(min_length=1, max_length=8)
    obligation_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_obligation(self) -> Self:
        """Validate issuer normalization, distinct axes, an aware cutoff and identity.

        Returns:
            This validated contract.

        Raises:
            ValueError: Issuer normalization, clocks, distinct axes or obligation identity are
                invalid.
        """
        normalized = tuple(value.strip().upper() for value in self.ordered_entity_ids)
        if normalized != self.ordered_entity_ids or len(set(normalized)) != len(normalized):
            raise ValueError("alternative_evidence.scoped_obligation_axis_invalid")
        if self.evidence_as_of.tzinfo is None or self.evidence_as_of.utcoffset() is None:
            raise ValueError("alternative_evidence.scoped_obligation_clock_invalid")
        if len(set(self.approved_source_families)) != len(self.approved_source_families):
            raise ValueError("alternative_evidence.scoped_obligation_sources_invalid")
        if len(self.required_checks) != len(set(self.required_checks)):
            raise ValueError("alternative_evidence.analysis_obligation_invalid")
        validate_contract_identity(self, "obligation_hash")
        return self


def seal_research_obligation(**values: object) -> AlternativeEvidenceResearchObligation:
    """Seal and validate the host-owned research obligation.

    Args:
        values: Obligation fields supplied by the research owner.

    Returns:
        The validated obligation with its canonical identity.
    """
    from ..contracts import seal_contract

    return seal_contract(AlternativeEvidenceResearchObligation, "obligation_hash", **values)


class EvidenceTopic(StrEnum):
    """Eight issuer-level topics a filing can state something about."""

    LIQUIDITY_GOING_CONCERN = "LIQUIDITY_GOING_CONCERN"
    CAPITAL_DILUTION = "CAPITAL_DILUTION"
    LEGAL_REGULATORY = "LEGAL_REGULATORY"
    OPERATIONS_SUPPLY = "OPERATIONS_SUPPLY"
    PRODUCT_SAFETY_CYBER = "PRODUCT_SAFETY_CYBER"
    GOVERNANCE_CONTROLS = "GOVERNANCE_CONTROLS"
    COMMERCIAL_COUNTERPARTY = "COMMERCIAL_COUNTERPARTY"
    CORPORATE_ACTION_LISTING = "CORPORATE_ACTION_LISTING"


class EvidenceLifecycle(StrEnum):
    """Record whether a cited matter is announced, ongoing, resolved or disputed."""

    ANNOUNCED = "ANNOUNCED"
    ONGOING = "ONGOING"
    RESOLVED = "RESOLVED"
    DISPUTED = "DISPUTED"


class EvidenceDirection(StrEnum):
    """Direction at the issuer level. Position impact is the reviewer's question."""

    ADVERSE = "ADVERSE"
    MITIGATING = "MITIGATING"
    MIXED = "MIXED"


class EvidenceQueryRecord(AlternativeEvidenceContract):
    """One query the Host ran and the span handles it returned, in rank order."""

    query_id: str = Field(pattern=r"^Q-[A-Z0-9-]{1,48}$")
    topic: EvidenceTopic
    text: str = Field(min_length=1, max_length=512)
    hit_span_handles: tuple[str, ...] = Field(max_length=180)
    """Up to twenty per issuer of the request (eight issuers and a trailing
    group of documents outside every issuer), in the reranked order."""
    kind: str = Field(
        default="ADVERSE", pattern=r"^(ADVERSE|STATE)$", exclude_if=lambda v: v == "ADVERSE"
    )
    """Which check the question serves: `ADVERSE` asks for the topic's
    downside events (supporting evidence), `STATE` for its compliance,
    resolution and routine-event families (where contradicting and
    current-state evidence is found). Absent from the identity at the
    default, so receipts sealed before the state questions keep their hash."""
    disposition: str = Field(
        default="RUN",
        pattern=r"^(RUN|SKIPPED_NO_SCOPE|SKIPPED_BUDGET)$",
        exclude_if=lambda v: v == "RUN",
    )
    """Whether the question was asked: `RUN`; `SKIPPED_NO_SCOPE` when the
    routing left it no residual scope (every routed range of its topic is an
    inventoried region, read as units); `SKIPPED_BUDGET` when the session's
    residual rerank-pair budget was spent before it. A skipped question has
    no hits and is not a search. Absent at the default, so every earlier
    receipt keeps its hash."""
    scope_windows: int | None = Field(default=None, ge=0, exclude_if=lambda v: v is None)
    """How many proved windows the question's scope held when it ran scoped
    (the residual search); None for the unconditional program."""
    reranked_pairs: int | None = Field(default=None, ge=0, exclude_if=lambda v: v is None)
    """How many (question, passage) pairs the cross-encoder scored for this
    question, as the kernel's trace reported; None when not recorded."""


class EvidenceSpanGroup(AlternativeEvidenceContract):
    """Group same-filing candidates whose previews match the delivered representative.

    Candidates of one filing whose previews are the same text as the
    representative's, grouped under it rather than deleted: a filing repeats
    its facts, and the packet reads each once, but which passages it stood
    for is recorded and each keeps its own identity.
    """

    representative_span_handle: str = Field(pattern=SPAN_HANDLE_PATTERN)
    member_span_handles: tuple[str, ...] = Field(min_length=1, max_length=64)


class SelectedWindowFacet(AlternativeEvidenceContract):
    """Record the structural family and heading that justified a selected window.

    Why a structurally discovered span was read: the family and heading
    path that governed it, the coverage facet it filled, the utility the
    frozen features gave it, the gaps its closure could not fill, and the
    other spans it is possibly related to. Selection provenance, not a
    judgment of the passage.
    """

    span_handle: str = Field(pattern=SPAN_HANDLE_PATTERN)
    family: str = Field(min_length=1, max_length=48)
    path: tuple[str, ...] = Field(default=(), max_length=6)
    matter: str = Field(min_length=1, max_length=120)
    role: str = Field(min_length=1, max_length=24)
    utility: float = Field(ge=0.0, le=1.0)
    selection_pass: str = Field(min_length=1, max_length=16)
    gaps: tuple[str, ...] = Field(default=(), max_length=8)
    related_span_handles: tuple[str, ...] = Field(default=(), max_length=8)


WHOLE_FILINGS_RULES_ID: Literal["alternative-evidence.whole-filings.v1"] = (
    "alternative-evidence.whole-filings.v1"
)


class WholeFilingsRecord(AlternativeEvidenceContract):
    """Record delivery of every filing in a whole-filing unit.

    A unit delivered whole (W4): every filing of it, in pieces at line
    bounds, under the bound its source policy names; no question, index or
    rerank chose them.
    """

    rules_id: Literal["alternative-evidence.whole-filings.v1"] = WHOLE_FILINGS_RULES_ID
    bound_bytes: int = Field(ge=1)
    document_count: int = Field(ge=1, le=24)
    delivered_bytes: int = Field(ge=1)


class StructuralScanRecord(AlternativeEvidenceContract):
    """Record structural inspection and the resulting coverage selection.

    What a structural scan inspected and what the coverage selector did
    with it: the rules and policy it ran under, the windows it walked, the
    frontier it admitted, every rejection by reason, the model pairs it
    spent, and the facet of each span it read.
    """

    structure_rules_id: str = Field(min_length=1, max_length=80)
    selection_policy_id: str = Field(min_length=1, max_length=80)
    selection_policy_hash: str = Field(pattern=_HASH)
    windows_inspected: int = Field(ge=0)
    frontier_size: int = Field(ge=0)
    families_with_opportunity: int = Field(ge=0)
    families_without_candidate: tuple[str, ...] = Field(default=(), max_length=256)
    """`<issuer>:<family>` buckets that surfaced no admissible window: a
    reported gap, never proof that the family holds no material fact."""
    rejections: tuple[tuple[str, int], ...] = Field(default=(), max_length=32)
    ranking_pairs: int = Field(ge=0)
    ranking_context: str = Field(default="NONE", min_length=1, max_length=48)
    selected_bytes: int = Field(ge=0)
    byte_budget: int = Field(ge=0)
    facets: tuple[SelectedWindowFacet, ...] = Field(default=(), max_length=128)


class TypedDisclosureField(AlternativeEvidenceContract):
    """Bind one parsed disclosure field to its source and extraction rule.

    One bound field of a typed assertion: the name, the value the rule read
    (absent when the source does not state it), the source words that bind
    it and their exact range in the document, and where a derived value came
    from ("the period covered by this report: ...").
    """

    name: str = Field(min_length=1, max_length=48)
    value: str | None = Field(default=None, max_length=240)
    text: str = Field(min_length=1, max_length=400)
    character_start: int = Field(ge=0)
    character_end: int = Field(ge=1)
    basis: str = Field(default="", max_length=240)


class TypedDisclosureInstance(AlternativeEvidenceContract):
    """Record one source assertion parsed under a typed family definition.

    One source assertion parsed under a family definition: its subject,
    action, polarity and period as the source states them, its fields, the
    qualifiers kept verbatim, the fields the source leaves unknown, and the
    verified span that carries it (with its exact range). Not a judgment.
    """

    instance_id: str = Field(pattern=r"^T-[A-Z0-9-]{1,64}$")
    span_handle: str | None = Field(default=None, pattern=SPAN_HANDLE_PATTERN)
    """None only when the session's typed span budget was exhausted; the
    assertion, its range and its text stay in the record, never dropped."""
    character_start: int = Field(ge=0)
    character_end: int = Field(ge=1)
    subject: str = Field(min_length=1, max_length=200)
    action: str = Field(min_length=1, max_length=40)
    polarity: str = Field(min_length=1, max_length=32)
    period_end: str | None = Field(default=None, max_length=10)
    period_text: str = Field(default="", max_length=200)
    period_basis: str = Field(default="", max_length=240)
    statement_text: str = Field(min_length=1, max_length=2400)
    fields: tuple[TypedDisclosureField, ...] = Field(default=(), max_length=16)
    qualifiers: tuple[str, ...] = Field(default=(), max_length=8)
    unknown_fields: tuple[str, ...] = Field(default=(), max_length=8)


class TypedDisclosureObservation(AlternativeEvidenceContract):
    """Record what a family's rules observed in an admitted filing.

    What one family's rules observed on one admitted filing: the state
    (`EXTRACTED`, `EXPLICIT_NONE`, `NOT_FOUND`, `REFERENCE_REQUIRED`,
    `AMBIGUOUS`, `SOURCE_UNAVAILABLE`, `NOT_APPLICABLE`), its reason, the
    section inspected, the report period, every instance, the relevant
    sentences no rule read, the documents it refers to, and a scope span
    a reader can verify the section or the reason against.
    """

    observation_id: str = Field(pattern=r"^TO-[A-Z0-9-]{1,80}$")
    entity_id: str = Field(min_length=1, max_length=32)
    document_handle: str = Field(pattern=r"^DOC-[A-Z0-9-]{1,48}$")
    document_type: str = Field(min_length=1, max_length=40)
    revision_label: str = Field(min_length=1, max_length=120)
    family: str = Field(min_length=1, max_length=48)
    rule_id: str = Field(min_length=1, max_length=80)
    state: str = Field(min_length=1, max_length=24)
    reason: str = Field(min_length=1, max_length=600)
    part: str | None = Field(default=None, max_length=16)
    item: str | None = Field(default=None, max_length=8)
    region_title: str = Field(default="", max_length=160)
    report_period_end: str | None = Field(default=None, max_length=10)
    report_period_status: str = Field(min_length=1, max_length=16)
    report_period_basis: str = Field(default="", max_length=200)
    instances: tuple[TypedDisclosureInstance, ...] = Field(default=(), max_length=32)
    unrecognized: tuple[str, ...] = Field(default=(), max_length=8)
    references: tuple[str, ...] = Field(default=(), max_length=4)
    context: tuple[str, ...] = Field(default=(), max_length=6)
    scope_span_handle: str | None = Field(default=None, pattern=SPAN_HANDLE_PATTERN)
    inspected_ranges: tuple[tuple[int, int], ...] = Field(default=(), max_length=4)
    undelivered_instance_count: int = Field(default=0, ge=0)


class TypedDisclosureRecord(AlternativeEvidenceContract):
    """Retain typed-family rules and observations beside question-program evidence.

    The typed families read beside the question program: the rules and
    definitions they ran under, every observation, the typed spans issued
    and read under their own budget, and the issuers with no periodic
    filing admitted. An explicit accounting entry, never a share of the
    generic top-k.
    """

    rules_id: str = Field(min_length=1, max_length=80)
    definitions_hash: str = Field(pattern=_HASH)
    documents_inspected: int = Field(ge=0)
    documents_skipped_by_form: int = Field(ge=0)
    observations: tuple[TypedDisclosureObservation, ...] = Field(default=(), max_length=96)
    span_handles: tuple[str, ...] = Field(default=(), max_length=64)
    read_call_count: int = Field(ge=0, le=16)
    issuers_without_periodic_filing: tuple[str, ...] = Field(default=(), max_length=8)


class LitigationRegionRecord(AlternativeEvidenceContract):
    """Identify a litigation disclosure region and its source boundaries.

    One Legal Proceedings item or contingencies/litigation note body of a
    filing, with what bounded it: the next governing heading, or -- when
    another part of the report opened before the note's own end was seen --
    an uncertain end the reader is told about.
    """

    kind: str = Field(min_length=1, max_length=8)
    heading: str = Field(min_length=1, max_length=160)
    character_start: int = Field(ge=0)
    character_end: int = Field(ge=1)
    end_basis: str = Field(min_length=1, max_length=240)
    end_uncertain: bool
    family: str = Field(
        default="LITIGATION",
        pattern=r"^(LITIGATION|CORPORATE_EVENT|FINANCING|OPERATIONS)$",
        exclude_if=lambda value: value == "LITIGATION",
    )
    """Which inventory opened the region: the litigation regions (absent
    at the default, so every earlier record keeps its hash), the
    corporate-event regions -- an event note or a current report's item --
    the financing regions -- a debt, borrowings or financing note -- or
    the operations regions -- a restructuring, impairment or operating-
    charge note, or a narrative item's paragraph holding a dated
    operations statement."""


class ProvisionalMatterRecord(AlternativeEvidenceContract):
    """Identify a provisional litigation matter from source text and signposts.

    One provisional litigation matter: where its text is, what signpost
    opened it, what the source calls it and the references it states.
    Navigation, local to one document: a handle links nothing across
    filings, and two matters that share a party or a court stay two.
    """

    handle: str = Field(pattern=r"^M-[A-Z0-9-]{1,48}-[0-9]{3}$")
    title: str = Field(min_length=1, max_length=160)
    basis: str = Field(min_length=1, max_length=32)
    named: bool
    """Whether the source names it as a proceeding (a caption, a filing, a
    court or a case number) rather than a topical sub-heading group."""
    region_heading: str = Field(min_length=1, max_length=160)
    group: str = Field(default="", max_length=160)
    character_start: int = Field(ge=0)
    character_end: int = Field(ge=1)
    source_bytes: int = Field(ge=1)
    aliases: tuple[str, ...] = Field(default=(), max_length=12)
    case_numbers: tuple[str, ...] = Field(default=(), max_length=12)
    courts: tuple[str, ...] = Field(default=(), max_length=8)
    planned_windows: int = Field(ge=1)
    read_windows: int = Field(ge=0)
    family: str = Field(
        default="LITIGATION",
        pattern=r"^(LITIGATION|CORPORATE_EVENT|FINANCING|OPERATIONS)$",
        exclude_if=lambda value: value == "LITIGATION",
    )
    """Which inventory opened the unit: a litigation matter (absent at the
    default); a corporate event -- a dated action of an event note, or one
    item of a current report -- whose handle carries `-E-`; or a financing
    unit -- a paragraph of a debt or borrowings note naming an instrument,
    program or facility -- whose handle carries `-F-`; or an operations
    unit -- a paragraph of a restructuring, impairment or operating-charge
    note, or a dated operations statement of a narrative item -- whose
    handle carries `-O-`."""


class UnassignedRangeRecord(AlternativeEvidenceContract):
    """Retain visible litigation-region text not claimed by a matter signpost.

    Text inside a litigation region that no signpost claimed: visible,
    readable on demand, never counted as inspected.
    """

    character_start: int = Field(ge=0)
    character_end: int = Field(ge=1)
    reason: str = Field(min_length=1, max_length=80)


class MatterAttributionRecord(AlternativeEvidenceContract):
    """Attribute exact delivered excerpt ranges to litigation matters.

    What one delivered excerpt carries, range by range: the exact
    intersection of the excerpt with a provisional matter's text or with a
    region-level statement. The excerpt is transport (the planned window
    plus the sentences the reader grew it by); the attribution is the
    claim, at its own offsets in the source. An excerpt can carry two
    matters, each under its own handle at its own range (a co-occurrence),
    never one matter's statement under another's handle. `scope` MATTER
    binds the range to `matter_handle`; REGION is a region statement the
    source says applies to the region's matters ('described above', 'these
    matters'); UNRESOLVED is a region statement whose scope the source
    leaves open -- visible at the region, associated with no matter.
    """

    matter_handle: str | None = Field(
        default=None,
        pattern=r"^M-[A-Z0-9-]{1,48}-[0-9]{3}$",
        exclude_if=lambda value: value is None,
    )
    role: str = Field(pattern=r"^(CORE|CONTINUATION|REGION_STATEMENT)$")
    """`CORE`: the range holds the matter's opening (its lead);
    `CONTINUATION`: a later part of the matter's text; `REGION_STATEMENT`:
    a paragraph of the region no matter claims."""
    scope: str = Field(pattern=r"^(MATTER|REGION|UNRESOLVED)$")
    basis: str = Field(min_length=1, max_length=160)
    """`planned window` when the range lies inside the window that was
    issued for it, `reader expansion` when only the reader's growth to
    sentence boundaries delivered it; for a region statement, its position
    and the reference that binds it to the matters, if any."""
    region_heading: str = Field(min_length=1, max_length=160)
    character_start: int = Field(ge=0)
    character_end: int = Field(ge=1)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_attribution(self) -> Self:
        if self.character_end <= self.character_start:
            raise ValueError("alternative_evidence.matter_attribution_invalid")
        bound = self.scope == "MATTER"
        if bound != (self.matter_handle is not None) or bound != (self.role != "REGION_STATEMENT"):
            raise ValueError("alternative_evidence.matter_attribution_invalid")
        return self


class MatterWindowRecord(AlternativeEvidenceContract):
    """Record one planned reading window and its delivery state.

    One exact window of the reading plan. `READ` means issued and
    returned by the verified reader (`span_handle` names the span);
    `PENDING` means planned and not read -- a pointer to unread source,
    which is not evidence. `delivered_whole` is false when the reader had
    to bound the excerpt, which the plan's byte sizing is meant to make
    impossible and the packet still reports.
    """

    document_handle: str = Field(pattern=r"^DOC-[A-Z0-9-]{1,48}$")
    matter_handle: str = Field(min_length=1, max_length=64)
    """A provisional matter's handle, or `UNASSIGNED`."""
    part: int = Field(ge=1)
    part_count: int = Field(ge=1)
    character_start: int = Field(ge=0)
    character_end: int = Field(ge=1)
    source_bytes: int = Field(ge=1)
    status: str = Field(pattern=r"^(READ|PENDING|PRIOR)$")
    """`PRIOR`: proved read by an earlier session of the same plan (before
    this record's `plan_offset`), carried with the handle that session
    issued so a citation of it stays unambiguous; `READ`: issued and
    returned whole or bounded by this session's verified reader;
    `PENDING`: planned, never issued, a pointer to unread source."""
    span_handle: str | None = Field(default=None, pattern=SPAN_HANDLE_PATTERN)
    delivered_whole: bool = False
    attributions: tuple[MatterAttributionRecord, ...] = Field(
        default=(), max_length=64, exclude_if=lambda value: value == ()
    )
    """What the delivered excerpt carries, by exact range (the candidate
    selection records them; the production plan records none, and a
    record without them keeps its identity)."""


class LitigationMatterDocument(AlternativeEvidenceContract):
    """Record matter inventory coverage for one admitted filing.

    What the matter inventory found and read in one admitted filing, and
    how far the reading got: `inspection` is `COMPLETE` only when every
    planned window was read, `PARTIAL` when the allowance left some
    pending, `NONE` when none was read, `NO_LITIGATION_REGION` when the
    filing has no such region. Exhaustion is never completion.
    """

    document_handle: str = Field(pattern=r"^DOC-[A-Z0-9-]{1,48}$")
    entity_id: str = Field(min_length=1, max_length=32)
    document_type: str = Field(min_length=1, max_length=40)
    revision_label: str = Field(min_length=1, max_length=120)
    report_period_end: str | None = Field(default=None, max_length=10)
    regions: tuple[LitigationRegionRecord, ...] = Field(default=(), max_length=8)
    matters: tuple[ProvisionalMatterRecord, ...] = Field(default=(), max_length=160)
    unassigned: tuple[UnassignedRangeRecord, ...] = Field(default=(), max_length=64)
    inspection: str = Field(pattern=r"^(COMPLETE|PARTIAL|NONE|NO_LITIGATION_REGION)$")
    """Coverage of the declared reading plan: COMPLETE only when every planned
    window is a proved whole read (this session's or an earlier session's);
    the qualifications below stay disclosed whatever the coverage."""
    qualifications: tuple[str, ...] = Field(
        default=(), max_length=8, exclude_if=lambda value: value == ()
    )
    """What a complete plan does not settle: a region whose end the structure
    could not see, text no signpost claimed, a window the reader had to
    bound."""
    planned_windows: int = Field(ge=0)
    read_windows: int = Field(ge=0)
    pending_windows: int = Field(ge=0)
    region_source_bytes: int = Field(ge=0)
    """UTF-8 bytes of the regions' text without surrounding whitespace: zero
    means the region heading opened no readable text."""
    read_source_bytes: int = Field(ge=0)
    pending_source_bytes: int = Field(ge=0)


class EvidenceNeedRecord(AlternativeEvidenceContract):
    """Record one requirement for complete reading of a filing's litigation regions.

    One thing a complete reading of a filing's litigation regions needs,
    and whether this delivery provided it: a matter's opening (`LEAD`) and
    the rest of its text (`CONTINUATION`); a region-level statement
    (`REGION_STATEMENT`, no matter handle: the accrual position, the
    materiality closing, the whole disclosure of a filing that names no
    matter); a region statement a matter depends on because the source says
    it applies to the region's matters (`QUALIFICATION`); an explicit
    reference the region makes to another location (`REFERENCE`).
    `PROVIDED` names the span that delivers the range; `PROVIDED_SHARED`,
    a span that delivers it beside another need's range (one excerpt, no
    duplicated bytes); `PENDING_ALLOWANCE`, that the allowance or the
    budget left it unread; `UNRESOLVED`, that a reference's target is not
    held in the document set or was not located. What is not provided is
    disclosed here, never implied read.
    """

    document_handle: str = Field(pattern=r"^DOC-[A-Z0-9-]{1,48}$")
    matter_handle: str | None = Field(
        default=None,
        pattern=r"^M-[A-Z0-9-]{1,48}-[0-9]{3}$",
        exclude_if=lambda value: value is None,
    )
    kind: str = Field(pattern=r"^(LEAD|CONTINUATION|REGION_STATEMENT|QUALIFICATION|REFERENCE)$")
    status: str = Field(pattern=r"^(PROVIDED|PROVIDED_SHARED|PENDING_ALLOWANCE|UNRESOLVED)$")
    character_start: int | None = Field(default=None, ge=0, exclude_if=lambda value: value is None)
    character_end: int | None = Field(default=None, ge=1, exclude_if=lambda value: value is None)
    span_handle: str | None = Field(
        default=None, pattern=SPAN_HANDLE_PATTERN, exclude_if=lambda value: value is None
    )
    detail: str = Field(default="", max_length=240, exclude_if=lambda value: value == "")
    target_state: str | None = Field(
        default=None,
        pattern=r"^(LOCATED|PARTIALLY_READ|DELIVERED|UNRESOLVED)$",
        exclude_if=lambda value: value is None,
    )
    """For a `REFERENCE`: what its target got. `LOCATED`: the target is
    identified and none of its required evidence was delivered;
    `PARTIALLY_READ`: some of it was; `DELIVERED`: all of it was (the only
    state a provided reference may carry); `UNRESOLVED`: the target is not
    held or was not located. A touched region, a filing with some window
    read or a search hit in the same filing is location, not delivery.

    A reference sealed before target states were recorded (candidate
    records of `486cea54`, whose `PROVIDED_SHARED` meant any overlap with
    the target) carries none: it reads under its own contract with its
    bytes and hash unchanged, its target's completeness is *unspecified* --
    never `DELIVERED`, never backfilled -- and no current path continues
    it. A current writer records a state for every reference
    (`packet._need_record` refuses one without)."""

    @property
    def target_completeness_unspecified(self) -> bool:
        """Identify a historical reference whose target completeness was not recorded.

        A historical reference: no target state was recorded when it was
        sealed, so whether its target's evidence was delivered is unknown.
        """
        return self.kind == "REFERENCE" and self.target_state is None

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_need(self) -> Self:
        provided = self.status in {"PROVIDED", "PROVIDED_SHARED"}
        if provided != (self.span_handle is not None):
            raise ValueError("alternative_evidence.evidence_need_invalid")
        if self.kind != "REFERENCE" and self.target_state is not None:
            raise ValueError("alternative_evidence.evidence_need_invalid")
        if self.target_state is not None:
            expected = {
                "PROVIDED": "DELIVERED",
                "PROVIDED_SHARED": "DELIVERED",
                "PENDING_ALLOWANCE": None,
                "UNRESOLVED": "UNRESOLVED",
            }[self.status]
            if expected is None:
                if self.target_state not in {"LOCATED", "PARTIALLY_READ"}:
                    raise ValueError("alternative_evidence.evidence_need_invalid")
            elif self.target_state != expected:
                raise ValueError("alternative_evidence.evidence_need_invalid")
        if (self.character_start is None) != (self.character_end is None) or (
            self.character_end is not None
            and self.character_start is not None
            and self.character_end <= self.character_start
        ):
            raise ValueError("alternative_evidence.evidence_need_invalid")
        if self.kind in {"LEAD", "CONTINUATION", "QUALIFICATION"} and self.matter_handle is None:
            raise ValueError("alternative_evidence.evidence_need_invalid")
        if self.kind == "REGION_STATEMENT" and self.matter_handle is not None:
            raise ValueError("alternative_evidence.evidence_need_invalid")
        return self


def _earlier_series(handle: str, session_index: int) -> bool:
    """Recognize a matter handle from an earlier session of the same chain.

    Whether a matter handle was issued by an earlier session of the same
    chain: series `M01` .. `M<session_index - 1>`.
    """
    series = handle.split("-")[1]
    return series[0] == "M" and 1 <= int(series[1:]) < session_index


class LitigationMatterRecord(AlternativeEvidenceContract):
    """Retain litigation matter inventory beside question and typed-disclosure evidence.

    The litigation matter inventory read beside the program and the typed
    families under its own allowance: every admitted filing's regions,
    provisional matters and unassigned text, every planned window with its
    status, the windows read (as spans of the packet) and the ones left
    pending with the offset a continuation would resume at. An explicit
    accounting entry: what was read, what was not, and that a budget's end
    is not an inspection's end.
    """

    rules_id: str = Field(min_length=1, max_length=80)
    allocation_rules_id: str | None = Field(
        default=None, max_length=80, exclude_if=lambda v: v is None
    )
    """How the windows were chosen when not as the prefix of the merged
    reading plan: the candidate's unmet-needs allocation, named. None for
    the production plan, whose records keep their identity."""
    families: tuple[str, ...] = Field(
        default=("LITIGATION",),
        min_length=1,
        max_length=4,
        exclude_if=lambda value: value == ("LITIGATION",),
    )
    """The disclosure-unit families the candidate inventoried, in the
    order `packet.MATTER_FAMILIES` names them: the litigation regions
    alone at the default (absent, so every earlier record keeps its
    hash), or with the corporate-event regions of the periodic filings
    and the items of the current reports. A continuation must name the same."""
    plan_hash: str | None = Field(default=None, pattern=_HASH, exclude_if=lambda v: v is None)
    """The identity of the reading plan: rules, ceiling, allowance, the
    generation (hence the document set and every revision) and every planned
    window's range. A continuation is valid only against the same plan; a
    record sealed before plans carried an identity has none and cannot be
    continued. The continuation fields are absent from the identity at their
    defaults, so every earlier sealed receipt keeps its hash."""
    plan_offset: int = Field(ge=0)
    """How many windows of the merged plan earlier sessions proved read: the
    prior record's offset plus its read windows, never a requested position.
    Zero for a first reading."""
    continued_from: str | None = Field(default=None, pattern=_HASH, exclude_if=lambda v: v is None)
    """The access receipt this session continued; None for a first reading."""
    session_index: int = Field(default=1, ge=1, le=99, exclude_if=lambda v: v == 1)
    """This session's place in its chain; its windows carry the series
    `M<session_index>`, so no handle is reused across sessions."""
    cumulative_read_windows: int | None = Field(default=None, ge=0, exclude_if=lambda v: v is None)
    """Windows proved read across the chain including this session; None on
    a record sealed before chains were accounted."""
    session_limit: int | None = Field(default=None, ge=1, exclude_if=lambda v: v is None)
    window_limit: int | None = Field(default=None, ge=1, exclude_if=lambda v: v is None)
    """The cumulative allowance the continuation was requested under, bound
    into the chain; None for a first reading, which has no continuation
    allowance of its own."""
    window_allowance: int = Field(ge=1)
    read_allowance: int = Field(ge=1)
    window_byte_ceiling: int = Field(ge=1)
    documents_inspected: int = Field(ge=0)
    documents_skipped_by_form: int = Field(ge=0)
    documents: tuple[LitigationMatterDocument, ...] = Field(default=(), max_length=64)
    windows: tuple[MatterWindowRecord, ...] = Field(default=(), max_length=1024)
    """Every planned window in reading order, read ones first by
    construction of the order (the allowance is a prefix of the plan)."""
    span_handles: tuple[str, ...] = Field(default=(), max_length=64)
    read_call_count: int = Field(ge=0, le=16)
    planned_windows: int = Field(ge=0)
    read_windows: int = Field(ge=0)
    pending_windows: int = Field(ge=0)
    prior_windows: int = Field(ge=0)
    planned_source_bytes: int = Field(ge=0)
    read_source_bytes: int = Field(ge=0)
    pending_source_bytes: int = Field(ge=0)
    needs: tuple[EvidenceNeedRecord, ...] = Field(
        default=(), max_length=1024, exclude_if=lambda value: value == ()
    )
    """Every need the candidate allocation identified, with its status: what
    was provided, shared, left by the allowance, or could not be resolved.
    Empty on the production plan."""

    @property
    def series(self) -> str:
        """Derive the current matter-session handle series.

        Returns:
            The M-prefixed session index padded to at least two digits.
        """
        return f"M{self.session_index:02d}"

    @property
    def historical_reference_needs(self) -> int:
        """Count reference needs recorded without a target completeness state.

        How many of this record's references were sealed without a target
        state: zero for every current record; a record with any is a
        historical candidate that reads under its own contract and is
        continued by no current path.
        """
        return sum(1 for need in self.needs if need.target_completeness_unspecified)

    @property
    def delivered_matter_handles(self) -> tuple[str, ...]:
        """Return current and prior matter-window handles delivered by this record.

        Every matter window a packet under this record delivers: this
        session's reads and the proved reads of the sessions it continued.
        """
        return tuple(
            window.span_handle
            for window in self.windows
            if window.status in {"READ", "PRIOR"} and window.span_handle is not None
        )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_record(self) -> Self:
        """Validate matter-plan counts, session lineage, window ordering and delivered handles.

        Returns:
            This validated contract.

        Raises:
            ValueError: Counts, prior-session linkage, handle partitions, window order or completion
                coverage disagree.
        """
        by_status = {"READ": 0, "PENDING": 0, "PRIOR": 0}
        for window in self.windows:
            by_status[window.status] += 1
        if (by_status["READ"], by_status["PENDING"], by_status["PRIOR"]) != (
            self.read_windows,
            self.pending_windows,
            self.prior_windows,
        ) or self.planned_windows != len(self.windows):
            raise ValueError("alternative_evidence.litigation_window_accounting_invalid")
        if self.prior_windows != self.plan_offset:
            raise ValueError("alternative_evidence.litigation_window_accounting_invalid")
        if (self.continued_from is None) != (self.session_index == 1):
            raise ValueError("alternative_evidence.litigation_continuation_chain_invalid")
        if (
            self.cumulative_read_windows is not None
            and self.cumulative_read_windows != self.plan_offset + self.read_windows
        ):
            raise ValueError("alternative_evidence.litigation_window_accounting_invalid")
        if self.session_index > 1 and (
            self.plan_hash is None or self.cumulative_read_windows is None
        ):
            raise ValueError("alternative_evidence.litigation_continuation_chain_invalid")
        series = f"SPAN-{self.series}-"
        read = [window.span_handle for window in self.windows if window.status == "READ"]
        if (
            any(handle is None or not handle.startswith(series) for handle in read)
            or set(read) != set(self.span_handles)
            or len(read) != len(set(read))
        ):
            raise ValueError("alternative_evidence.litigation_window_unread")
        # A first reading has nothing prior; a continuation's PRIOR windows
        # carry handles of earlier sessions of this chain only.
        prior = [window.span_handle for window in self.windows if window.status == "PRIOR"]
        if (
            (self.session_index == 1 and prior)
            or any(
                handle is None or not _earlier_series(handle, self.session_index)
                for handle in prior
            )
            or set(prior) & set(read)
            or len(prior) != len(set(prior))
        ):
            raise ValueError("alternative_evidence.litigation_prior_window_unproved")
        if any(
            window.span_handle is not None or window.attributions
            for window in self.windows
            if window.status == "PENDING"
        ):
            raise ValueError("alternative_evidence.litigation_window_accounting_invalid")
        # A need is provided by a delivered window of this record, or by a
        # search read the receipt accounts for; never by a pending one.
        delivered = set(self.delivered_matter_handles)
        if any(
            need.span_handle.startswith("SPAN-M") and need.span_handle not in delivered
            for need in self.needs
            if need.span_handle is not None
        ):
            raise ValueError("alternative_evidence.evidence_need_invalid")
        # The plan is read as a prefix: no PENDING window precedes a read one,
        # and no PRIOR window follows one this session read.
        seen_read = seen_pending = False
        for window in self.windows:
            if window.status == "PRIOR" and (seen_read or seen_pending):
                raise ValueError("alternative_evidence.litigation_prior_window_unproved")
            if window.status == "READ":
                if seen_pending:
                    raise ValueError("alternative_evidence.litigation_window_accounting_invalid")
                seen_read = True
            if window.status == "PENDING":
                seen_pending = True
        for document in self.documents:
            # COMPLETE is proved coverage of the plan: nothing pending, and a
            # region with source text has windows planned over it.
            if document.inspection == "COMPLETE" and (
                document.pending_windows
                or (document.planned_windows == 0 and document.region_source_bytes > 0)
            ):
                raise ValueError("alternative_evidence.litigation_inspection_unproved")
        return self


class IssuerTopicCellRecord(AlternativeEvidenceContract):
    """Seal source coverage and routing for one issuer/topic cell.

    One (issuer, topic) cell of the routing as sealed: what sources the
    issuer holds, what regions the structure routes to the topic and by
    which basis, what each method examined, what the residual search was
    scoped to, and the gaps named. Machine states of sources, representation,
    discovery and delivery -- never an actor's check.
    """

    entity_id: str = Field(min_length=1, max_length=32)
    topic: EvidenceTopic
    forms_held: tuple[str, ...] = Field(default=(), max_length=8)
    regions: int = Field(ge=0)
    regions_by_basis: tuple[tuple[str, int], ...] = Field(default=(), max_length=8)
    unit_needs: int = Field(ge=0)
    """Window needs of the inventories whose units serve the topic."""
    unit_needs_delivered: int = Field(default=0, ge=0)
    """Of those, the needs a delivered excerpt provides (this session's reads
    or a proved earlier read of the same chain, or an exact repeat's read)."""
    typed_observations: int = Field(ge=0)
    tables: int = Field(ge=0)
    """Tables inside the routed regions with a retained original."""
    tables_delivered: int = Field(default=0, ge=0)
    """Tables a page of the chain has reached -- dealt a view, whole or in
    part. Completion is not this count: the sealed pages prove it
    (`table_progress_of`), and the read model's cell states say COMPLETE
    only when every table's declared rows are delivered."""
    tables_without_original: int = Field(ge=0)
    tables_unrenderable: int = Field(default=0, ge=0)
    """Routed tables whose original is retained but whose view the boundary
    refused by name (no heading row, or a correspondence the original does
    not prove): representation gaps, each named in the routing record."""
    residual: str = Field(pattern=r"^(QUEUED|COVERED|BROADER|NO_SOURCE)$")
    residual_windows: int = Field(default=0, ge=0)
    """Proved windows of the residual scope, as the session's chunk index
    counted them."""
    residual_hits: int = Field(default=0, ge=0)
    """Passages of this cell the residual questions delivered (read spans
    of the issuer whose scope the topic's questions ran over)."""
    candidates_pending: int = Field(default=0, ge=0, exclude_if=lambda v: v == 0)
    """Returned candidates of this cell the packet sealed as pending and no
    session of the chain has read yet."""
    candidates_read: int = Field(default=0, ge=0, exclude_if=lambda v: v == 0)
    """Sealed candidates of this cell a later session of the chain read."""
    candidates_covered: int = Field(default=0, ge=0, exclude_if=lambda v: v == 0)
    """Returned candidates of this cell whose range a delivered span of the
    same filing already holds: in the packet under that span, never read twice."""
    gaps: tuple[str, ...] = Field(default=(), max_length=8)


class UnitCorrespondenceRecord(AlternativeEvidenceContract):
    """Compare a later filing unit with its prior same-family counterpart.

    One unit of a later filing against the previous filing of the same
    issuer that inventoried the same family, as the comparison rules state
    it: an exact repeat, a changed aligned unit with its changed character
    count, a unit first observed, a unit of the earlier filing absent from
    the later one (absence, not resolution), or unresolved under the bound.
    Both sources are named; their times are the references'.
    """

    entity_id: str = Field(min_length=1, max_length=32)
    family: str = Field(min_length=1, max_length=32)
    later_document_handle: str = Field(pattern=r"^DOC-[A-Z0-9-]{1,48}$")
    later_unit_handle: str | None = Field(
        default=None, max_length=64, exclude_if=lambda v: v is None
    )
    earlier_document_handle: str | None = Field(
        default=None, pattern=r"^DOC-[A-Z0-9-]{1,48}$", exclude_if=lambda v: v is None
    )
    earlier_unit_handle: str | None = Field(
        default=None, max_length=64, exclude_if=lambda v: v is None
    )
    state: str = Field(
        pattern=r"^(EXACT_REPEAT|CHANGED_ALIGNED|FIRST_OBSERVED|ABSENT_LATER|UNRESOLVED)$"
    )
    ratio: float | None = Field(default=None, ge=0.0, le=1.0, exclude_if=lambda v: v is None)
    changed_characters: int = Field(default=0, ge=0)
    basis: str = Field(default="", max_length=200, exclude_if=lambda v: v == "")


class TableViewRecord(AlternativeEvidenceContract):
    """Record one delivered table page and its source/routing coverage.

    One page of a table view the integrated selection delivered: which
    table of which filing, under which topics, from which routed region.
    """

    span_handle: str = Field(pattern=SPAN_HANDLE_PATTERN)
    document_handle: str = Field(pattern=r"^DOC-[A-Z0-9-]{1,48}$")
    entity_id: str = Field(min_length=1, max_length=32)
    topics: tuple[EvidenceTopic, ...] = Field(min_length=1, max_length=8)
    region_heading: str = Field(default="", max_length=160)
    table_ordinal: int = Field(ge=1)
    rows_total: int = Field(ge=0)
    rows_from: int = Field(default=0, ge=0, exclude_if=lambda v: v == 0)
    """The first row this page holds (1-based); 0 on a record sealed before
    pages were resumable, whose progress is unknown and is not resumed."""
    rows_to: int = Field(default=0, ge=0, exclude_if=lambda v: v == 0)
    remaining_rows: int = Field(default=0, ge=0, exclude_if=lambda v: v == 0)
    """Rows after this page the chain has not delivered: a later session
    resumes the table at `rows_to + 1` with its headings."""
    rows_clipped: int = Field(default=0, ge=0, exclude_if=lambda v: v == 0)
    """Rows of this page bounded at a word (one row alone exceeded the
    reader's ceiling): the page is delivered in part and the table is never
    complete; absent at 0 so every earlier record keeps its hash."""
    session_index: int = Field(default=1, ge=1, exclude_if=lambda v: v == 1)


class TableProgress(AlternativeEvidenceContract):
    """Describe table-row progress proven by sealed delivered pages.

    What the sealed pages prove of one table: the rows delivered
    contiguously from row 1 against the rendered total the pages declare,
    and the state that follows -- `COMPLETE` (every declared row delivered,
    no gap, no clipped row), `PARTIAL` (rows remain, a gap, or a clipped
    row that no page can resume), `UNKNOWN_PROGRESS` (a page sealed before
    pages carried row progress: nothing is proved and nothing is resumed).
    Never inferred across a gap, a duplicate page or a clipped cell; a
    maximum row number alone proves nothing.
    """

    document_handle: str
    table_ordinal: int
    entity_id: str
    topics: tuple[EvidenceTopic, ...]
    state: Literal["COMPLETE", "PARTIAL", "UNKNOWN_PROGRESS"]
    rows_declared: int
    """The rendered rows the pages' bindings declare; 0 when unknown."""
    rows_delivered: int
    """Rows proved delivered contiguously from row 1 (a gap stops the count)."""
    next_row: int | None
    """The first row a continuation would read; None when complete, when a
    clipped row leaves nothing resumable, or when the progress is unknown."""
    pages: int
    gap: bool
    rows_clipped: int

    @property
    def resumable(self) -> bool:
        return self.state == "PARTIAL" and self.next_row is not None


def table_progress_of(
    views: tuple[TableViewRecord, ...],
) -> dict[tuple[str, int], TableProgress]:
    """Derive per-table progress from sealed page identities.

    The progress of every table the pages name, by page identity
    (filing handle, table ordinal). One owner of the completion rule: the
    routing record's gap lines, the read model's cells, the continuation
    scope and the page planner all read it.
    """
    grouped: dict[tuple[str, int], list[TableViewRecord]] = {}
    for view in views:
        grouped.setdefault((view.document_handle, view.table_ordinal), []).append(view)
    out: dict[tuple[str, int], TableProgress] = {}
    for identity, pages in grouped.items():
        first = pages[0]
        topics = tuple(dict.fromkeys(topic for page in pages for topic in page.topics))
        clipped = sum(page.rows_clipped for page in pages)
        if any(page.rows_from == 0 for page in pages):
            out[identity] = TableProgress(
                document_handle=first.document_handle,
                table_ordinal=first.table_ordinal,
                entity_id=first.entity_id,
                topics=topics,
                state="UNKNOWN_PROGRESS",
                rows_declared=0,
                rows_delivered=0,
                next_row=None,
                pages=len(pages),
                gap=False,
                rows_clipped=clipped,
            )
            continue
        declared = max(page.rows_to + page.remaining_rows for page in pages)
        covered_to = 0
        gap = False
        for start, end in sorted((page.rows_from, page.rows_to) for page in pages):
            if start > covered_to + 1:
                gap = True
                break
            covered_to = max(covered_to, end)
        complete = not gap and covered_to >= declared and clipped == 0
        if complete:
            next_row: int | None = None
        elif clipped and covered_to >= declared:
            next_row = None
        else:
            next_row = covered_to + 1
        out[identity] = TableProgress(
            document_handle=first.document_handle,
            table_ordinal=first.table_ordinal,
            entity_id=first.entity_id,
            topics=topics,
            state="COMPLETE" if complete else "PARTIAL",
            rows_declared=declared,
            rows_delivered=min(covered_to, declared),
            next_row=next_row,
            pages=len(pages),
            gap=gap,
            rows_clipped=clipped,
        )
    return out


def cell_table_states(
    progress: Mapping[tuple[str, int], TableProgress], entity_id: str, topic: EvidenceTopic
) -> dict[str, int]:
    """The tables of one cell by state, from the pages' own identities."""
    counts = {"COMPLETE": 0, "PARTIAL": 0, "UNKNOWN_PROGRESS": 0}
    for value in progress.values():
        if value.entity_id == entity_id and topic in value.topics:
            counts[value.state] += 1
    return counts


class PendingCandidateRecord(AlternativeEvidenceContract):
    """Retain one residual search candidate left unread by the batch.

    One residual candidate a search returned and the batch did not read,
    sealed so a later session of the chain reads it at its exact range
    without a search: the cell it belongs to, the source, the range, the
    passage hash the reading must reproduce, every question that returned
    it with the best question's ranks, its order in the cell, and -- once
    read -- the handle and session that read it. A candidate whose range
    another channel of the chain already delivered (a unit window, a typed
    statement, a table page of the same filing) is `COVERED` by that span:
    the evidence is in the packet under the covering handle, the questions
    that returned it travel to that handle, and no read is spent on it. A
    pending record is what the chain still owes; it is never counted as
    delivered.
    """

    entity_id: str = Field(min_length=1, max_length=32)
    topic: EvidenceTopic
    document_handle: str = Field(pattern=r"^DOC-[A-Z0-9-]{1,48}$")
    character_start: int = Field(ge=0)
    character_end: int = Field(gt=0)
    passage_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    found_by: tuple[str, ...] = Field(min_length=1, max_length=8)
    """The query ids that returned this passage, best first."""
    best_global_rank: int = Field(ge=1)
    best_local_rank: int = Field(ge=1)
    """The best question's rank of this passage across the unit's issuers,
    and among the issuer's own returned passages of that question."""
    order: int = Field(ge=1)
    """The candidate's place in its cell's pending order (1 = next)."""
    state: str = Field(default="PENDING", pattern=r"^(PENDING|READ|COVERED)$")
    span_handle: str | None = Field(
        default=None, pattern=SPAN_HANDLE_PATTERN, exclude_if=lambda v: v is None
    )
    session_index: int | None = Field(default=None, ge=1, exclude_if=lambda v: v is None)
    """The session that read the candidate, or that found it covered."""
    covered_by: str | None = Field(
        default=None, pattern=SPAN_HANDLE_PATTERN, exclude_if=lambda v: v is None
    )
    """The delivered span of the same filing holding the candidate's first
    character (`COVERED`): where the evidence is read; with `covered_with`,
    the exact union of delivered ranges that holds the candidate whole."""
    covered_with: tuple[str, ...] = Field(default=(), max_length=16, exclude_if=lambda v: not v)
    """The further delivered spans, in range order, whose ranges join
    `covered_by` to hold the candidate's whole range; empty when one
    span holds it."""

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_candidate(self) -> Self:
        if self.character_end <= self.character_start:
            raise ValueError("alternative_evidence.pending_candidate_invalid")
        if self.covered_with and (
            self.state != "COVERED"
            or any(not re.fullmatch(SPAN_HANDLE_PATTERN, h) for h in self.covered_with)
            or len(set(self.covered_with)) != len(self.covered_with)
            or self.covered_by in self.covered_with
        ):
            raise ValueError("alternative_evidence.pending_candidate_invalid")
        if (self.state == "READ") != (self.span_handle is not None):
            raise ValueError("alternative_evidence.pending_candidate_invalid")
        if self.span_handle is not None and not self.span_handle.startswith("SPAN-C"):
            raise ValueError("alternative_evidence.pending_candidate_invalid")
        if (self.state == "COVERED") != (self.covered_by is not None):
            raise ValueError("alternative_evidence.pending_candidate_invalid")
        if (self.state == "PENDING") != (self.session_index is None):
            raise ValueError("alternative_evidence.pending_candidate_invalid")
        if self.state == "READ" and (self.session_index or 0) < 2:
            raise ValueError("alternative_evidence.pending_candidate_invalid")
        return self


CANDIDATE_FRONTIER_LIMIT = 4096
"""The most candidates one frontier seals: twenty questions returning
twenty per issuer over eight issuers is 3,200 before any repeat is
collapsed; what a search returns beyond this is counted, never named."""


class CandidateFrontier(AlternativeEvidenceContract):
    """Retain the complete frontier of unread residual search candidates.

    Every residual candidate the questions returned and the batch did not
    read, as the receipt seals it: the same records as
    `PendingCandidateRecord`, in compact columns -- the issuers, filings
    and questions interned once, each record an index into them -- so
    a whole search's return is kept at about a third of the inline
    record's bytes and nothing paid for is dropped uncounted. The records
    stay in cell order (issuer, then the topics' declared order, then
    the cell's own order) and are read back as records.
    """

    entities: tuple[str, ...] = Field(default=(), max_length=8)
    documents: tuple[str, ...] = Field(default=(), max_length=64)
    queries: tuple[str, ...] = Field(default=(), max_length=24)
    entity: tuple[int, ...] = Field(default=(), max_length=CANDIDATE_FRONTIER_LIMIT)
    topic: tuple[int, ...] = Field(default=(), max_length=CANDIDATE_FRONTIER_LIMIT)
    document: tuple[int, ...] = Field(default=(), max_length=CANDIDATE_FRONTIER_LIMIT)
    character_start: tuple[int, ...] = Field(default=(), max_length=CANDIDATE_FRONTIER_LIMIT)
    character_end: tuple[int, ...] = Field(default=(), max_length=CANDIDATE_FRONTIER_LIMIT)
    passage_hash: tuple[str, ...] = Field(default=(), max_length=CANDIDATE_FRONTIER_LIMIT)
    found_by: tuple[tuple[int, ...], ...] = Field(default=(), max_length=CANDIDATE_FRONTIER_LIMIT)
    best_global_rank: tuple[int, ...] = Field(default=(), max_length=CANDIDATE_FRONTIER_LIMIT)
    best_local_rank: tuple[int, ...] = Field(default=(), max_length=CANDIDATE_FRONTIER_LIMIT)
    order: tuple[int, ...] = Field(default=(), max_length=CANDIDATE_FRONTIER_LIMIT)
    state: tuple[str, ...] = Field(default=(), max_length=CANDIDATE_FRONTIER_LIMIT)
    span_handle: tuple[str | None, ...] = Field(default=(), max_length=CANDIDATE_FRONTIER_LIMIT)
    session_index: tuple[int | None, ...] = Field(default=(), max_length=CANDIDATE_FRONTIER_LIMIT)
    covered_by: tuple[str | None, ...] = Field(default=(), max_length=CANDIDATE_FRONTIER_LIMIT)
    covered_with: tuple[tuple[str, ...], ...] = Field(
        default=(), max_length=CANDIDATE_FRONTIER_LIMIT, exclude_if=lambda v: not v
    )
    """Per record, the further spans of the union that covers it; absent
    from the identity when no record has one, and absent on a frontier
    sealed before the exact-union rule, which read every cover as one
    span, so every earlier receipt keeps its hash."""

    _COLUMNS: ClassVar[tuple[str, ...]] = (
        "entity",
        "topic",
        "document",
        "character_start",
        "character_end",
        "passage_hash",
        "found_by",
        "best_global_rank",
        "best_local_rank",
        "order",
        "state",
        "span_handle",
        "session_index",
        "covered_by",
    )

    def __len__(self) -> int:
        return len(self.entity)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_frontier(self) -> Self:
        count = len(self.entity)
        if any(len(getattr(self, column)) != count for column in self._COLUMNS) or (
            len(self.covered_with) not in (0, count)
        ):
            raise ValueError("alternative_evidence.candidate_frontier_invalid")
        topics = tuple(EvidenceTopic)
        if (
            any(not 0 <= index < len(self.entities) for index in self.entity)
            or any(not 0 <= index < len(topics) for index in self.topic)
            or any(not 0 <= index < len(self.documents) for index in self.document)
            or any(
                not found or any(not 0 <= index < len(self.queries) for index in found)
                for found in self.found_by
            )
        ):
            raise ValueError("alternative_evidence.candidate_frontier_invalid")
        # Every record reads back as a valid pending record.
        for index in range(count):
            self.record(index)
        return self

    def record(self, index: int) -> PendingCandidateRecord:
        return PendingCandidateRecord(
            entity_id=self.entities[self.entity[index]],
            topic=tuple(EvidenceTopic)[self.topic[index]],
            document_handle=self.documents[self.document[index]],
            character_start=self.character_start[index],
            character_end=self.character_end[index],
            passage_hash=self.passage_hash[index],
            found_by=tuple(self.queries[q] for q in self.found_by[index]),
            best_global_rank=self.best_global_rank[index],
            best_local_rank=self.best_local_rank[index],
            order=self.order[index],
            state=self.state[index],
            span_handle=self.span_handle[index],
            session_index=self.session_index[index],
            covered_by=self.covered_by[index],
            covered_with=self.covered_with[index] if self.covered_with else (),
        )

    def records(self) -> tuple[PendingCandidateRecord, ...]:
        return tuple(self.record(index) for index in range(len(self.entity)))

    @classmethod
    def from_records(cls, records: Sequence[PendingCandidateRecord]) -> CandidateFrontier:
        entities = tuple(dict.fromkeys(r.entity_id for r in records))
        documents = tuple(dict.fromkeys(r.document_handle for r in records))
        queries = tuple(dict.fromkeys(q for r in records for q in r.found_by))
        topics = tuple(EvidenceTopic)
        return cls(
            entities=entities,
            documents=documents,
            queries=queries,
            entity=tuple(entities.index(r.entity_id) for r in records),
            topic=tuple(topics.index(r.topic) for r in records),
            document=tuple(documents.index(r.document_handle) for r in records),
            character_start=tuple(r.character_start for r in records),
            character_end=tuple(r.character_end for r in records),
            passage_hash=tuple(r.passage_hash for r in records),
            found_by=tuple(tuple(queries.index(q) for q in r.found_by) for r in records),
            best_global_rank=tuple(r.best_global_rank for r in records),
            best_local_rank=tuple(r.best_local_rank for r in records),
            order=tuple(r.order for r in records),
            state=tuple(r.state for r in records),
            span_handle=tuple(r.span_handle for r in records),
            session_index=tuple(r.session_index for r in records),
            covered_by=tuple(r.covered_by for r in records),
            covered_with=(
                tuple(r.covered_with for r in records)
                if any(r.covered_with for r in records)
                else ()
            ),
        )


class ComparedUnitRecord(AlternativeEvidenceContract):
    """Retain one unit correspondence inside a sealed filing comparison.

    One unit correspondence as a sealed comparison keeps it: the later
    and earlier unit handles (each None for a unit the other filing does
    not align with), the state and the measure.
    """

    later_handle: str | None = Field(default=None, max_length=64, exclude_if=lambda v: v is None)
    earlier_handle: str | None = Field(default=None, max_length=64, exclude_if=lambda v: v is None)
    state: str = Field(
        pattern=r"^(EXACT_REPEAT|CHANGED_ALIGNED|FIRST_OBSERVED|ABSENT_LATER|UNRESOLVED)$"
    )
    ratio: float | None = Field(default=None, ge=0.0, le=1.0, exclude_if=lambda v: v is None)
    changed_characters: int = Field(default=0, ge=0)
    basis: str = Field(max_length=200)


class FilingComparisonRecord(AlternativeEvidenceContract):
    """Seal a temporal filing-pair comparison for one issuer and family.

    The temporal comparison of one filing pair of one issuer and family,
    sealed once and reused by a routing whose chain binds it. Two identities,
    kept apart: `comparison_hash` is the closure of the inputs -- the two
    canonical texts by digest, each filing's unit inventory (handle, family
    and text of every unit, so the same texts under another arrangement of
    units are another closure), the structure, inventory and comparison
    rules, the issuer and the family -- and is what a lookup is keyed on;
    `record_hash` is the identity of the committed output, the whole record,
    and is what the store names the record by. A routing resolves a closure
    to a record only through the binding (closure -> record hash) a sealed
    receipt of its chain carries: a payload rewritten under the same
    closure is another record hash the binding does not name, and a record
    whose bytes no longer hash to the bound identity is refused, never
    recomputed silently. `results_hash` seals the units alone.
    """

    entity_id: str = Field(min_length=1, max_length=32)
    family: str = Field(min_length=1, max_length=32)
    later_content_sha256: str = Field(pattern=_HASH)
    earlier_content_sha256: str = Field(pattern=_HASH)
    later_units_hash: str = Field(pattern=_HASH)
    earlier_units_hash: str = Field(pattern=_HASH)
    comparison_rules_id: str = Field(min_length=1, max_length=80)
    structure_rules_id: str = Field(min_length=1, max_length=80)
    unit_rules: tuple[str, ...] = Field(min_length=1, max_length=16)
    units: tuple[ComparedUnitRecord, ...] = Field(default=(), max_length=4096)
    results_hash: str = Field(pattern=_HASH)
    comparison_hash: str = Field(pattern=_HASH)
    record_hash: str = Field(pattern=_HASH)

    @staticmethod
    def units_hash_of(units: Sequence[tuple[str, str, str]]) -> str:
        """Hash the ordered unit inventory recorded for one side of a comparison.

        The identity of one filing's unit inventory as the comparison
        aligns it: (handle, family, text digest) per unit, in order.
        """
        digest: str = canonical_hash([list(unit) for unit in units])
        return digest

    @staticmethod
    def closure_hash(
        *,
        entity_id: str,
        family: str,
        later_content_sha256: str,
        earlier_content_sha256: str,
        later_units_hash: str,
        earlier_units_hash: str,
        comparison_rules_id: str,
        structure_rules_id: str,
        unit_rules: tuple[str, ...],
    ) -> str:
        """The identity of the inputs: what a lookup is keyed on."""
        closure: str = canonical_hash(
            {
                "entity_id": entity_id,
                "family": family,
                "later_content_sha256": later_content_sha256,
                "earlier_content_sha256": earlier_content_sha256,
                "later_units_hash": later_units_hash,
                "earlier_units_hash": earlier_units_hash,
                "comparison_rules_id": comparison_rules_id,
                "structure_rules_id": structure_rules_id,
                "unit_rules": list(unit_rules),
            }
        )
        return closure

    @staticmethod
    def record_hash_of(content: Mapping[str, object]) -> str:
        """The identity of the committed output: every field but itself."""
        digest: str = canonical_hash({k: v for k, v in content.items() if k != "record_hash"})
        return digest

    @staticmethod
    def results_hash_of(units: Sequence[ComparedUnitRecord]) -> str:
        """The seal over what a comparison computed."""
        digest: str = canonical_hash([unit.model_dump(mode="json") for unit in units])
        return digest

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_comparison(self) -> Self:
        expected = self.closure_hash(
            entity_id=self.entity_id,
            family=self.family,
            later_content_sha256=self.later_content_sha256,
            earlier_content_sha256=self.earlier_content_sha256,
            later_units_hash=self.later_units_hash,
            earlier_units_hash=self.earlier_units_hash,
            comparison_rules_id=self.comparison_rules_id,
            structure_rules_id=self.structure_rules_id,
            unit_rules=self.unit_rules,
        )
        results = self.results_hash_of(self.units)
        if (
            self.comparison_hash != expected
            or self.results_hash != results
            or self.record_hash != self.record_hash_of(self.model_dump(mode="json"))
        ):
            raise ValueError("alternative_evidence.identity_invalid")
        return self


class ComparisonBindingRecord(AlternativeEvidenceContract):
    """Bind a sealed filing-pair comparison closure to its receipt record.

    One filing pair's comparison as a sealed receipt binds it: the
    closure the routing looked up and the record hash that answered it
    -- what a continuation of the chain may resolve the pair through.
    """

    closure_hash: str = Field(pattern=_HASH)
    record_hash: str = Field(pattern=_HASH)


class TableViewRefusalRecord(AlternativeEvidenceContract):
    """Record a routed table that reached a delivery boundary.

    One routed table the session's share reached and the boundary
    refused by name -- a representation gap, never a fabricated view.
    """

    document_handle: str = Field(pattern=r"^DOC-[A-Z0-9-]{1,48}$")
    entity_id: str = Field(min_length=1, max_length=32)
    table_ordinal: int = Field(ge=1)
    code: str = Field(min_length=1, max_length=80)


CONTEXT_COMPLETE_SELECTION_RULES_ID = (
    "alternative-evidence.residual-selection.context-complete-bundles.v1"
)
"""A retired rule's identity, kept for the receipts that name it
(`TopicRoutingRecord.residual_selection_rules_id`): the context-complete
allocation of section V's Gate A, retired in record section X. A receipt
dealt under it reads back unchanged and is never reused as a default
selection; a request that asks for it is refused by name
(`alternative_evidence.matter_selection_policy_retired`)."""


class TopicRoutingRecord(AlternativeEvidenceContract):
    """Seal integrated-selection routing beside the other evidence records.

    The integrated selection's routing as sealed beside the program's,
    the typed and the matter entries: the rules the cells were routed and
    the units compared under, every (issuer, topic) cell with its states,
    the unit correspondences across filings, the table views delivered
    under the matter window allowance (series `X`), the residual search's
    pair budget and what it spent, and the questions run or skipped. What
    was not delivered is named here, never implied read.
    """

    rules_id: str = Field(min_length=1, max_length=80)
    comparison_rules_id: str = Field(min_length=1, max_length=80)
    allocation_rules_id: str = Field(min_length=1, max_length=80)
    cells: tuple[IssuerTopicCellRecord, ...] = Field(default=(), max_length=64)
    correspondences: tuple[UnitCorrespondenceRecord, ...] = Field(default=(), max_length=512)
    comparisons: tuple[ComparisonBindingRecord, ...] = Field(
        default=(), max_length=256, exclude_if=lambda v: not v
    )
    """The sealed comparison each filing pair resolved to (closure ->
    record), the binding a continuation reuses them through; absent on
    receipts sealed before comparisons were bound."""
    exact_repeats_collapsed: int = Field(ge=0)
    """Earlier filings' units an exact repeat in a later filing stands for:
    served by the later reading, each with its own source and time kept."""
    table_view_span_handles: tuple[str, ...] = Field(default=(), max_length=64)
    """Pages of table views this session read (series `X`), in reading order."""
    table_views: tuple[TableViewRecord, ...] = Field(default=(), max_length=64)
    """Each page's table, filing, topics and region, in the same order."""
    table_view_refusals: tuple[TableViewRefusalRecord, ...] = Field(default=(), max_length=64)
    """The routed tables the share reached that the boundary refused."""
    table_views_pending: int = Field(ge=0)
    """Table needs the session's table share left undelivered."""
    residual_selection_rules_id: str = Field(
        default="", max_length=80, exclude_if=lambda v: v == ""
    )
    """The rule the residual candidates were selected and sealed under;
    empty on a record sealed before the issuer-topic cell rule;
    `CONTEXT_COMPLETE_SELECTION_RULES_ID` on a record dealt under the
    retired context-complete allocation, which reads back and is never
    reused as the default's selection."""
    pending_candidates: tuple[PendingCandidateRecord, ...] = Field(
        default=(), max_length=512, exclude_if=lambda v: v == ()
    )
    """The sealed pending plan of a record sealed before the candidate
    frontier (six per cell, inline); a current record seals its plan as
    `candidate_frontier` and leaves this empty. Read both through `candidates()`."""
    candidate_frontier: CandidateFrontier | None = Field(
        default=None, exclude_if=lambda v: v is None
    )
    """The sealed pending plan of the residual channel: every returned
    candidate the batch did not read, in cell order, with the reads later
    sessions of the chain made of them and the spans that cover them
    (`CandidateFrontier`)."""
    candidates_beyond_plan: int = Field(default=0, ge=0, exclude_if=lambda v: v == 0)
    """Returned candidates beyond what the frontier can seal
    (`CANDIDATE_FRONTIER_LIMIT`; six per cell on a record sealed before it):
    counted, reachable only by a new preparation."""
    candidate_span_handles: tuple[str, ...] = Field(
        default=(), max_length=512, exclude_if=lambda v: v == ()
    )
    """Sealed candidates the chain read (series `C`), in reading order."""
    residual_rerank_pair_budget: int = Field(ge=0)
    residual_reranked_pairs: int = Field(ge=0)
    questions_run: int = Field(ge=0)
    questions_skipped_no_scope: int = Field(ge=0)
    questions_skipped_budget: int = Field(ge=0)
    unrouted_windows: int = Field(ge=0)
    """Proved windows of the evidence families no topic's scope reached:
    retained as the count of what no route read, never discarded."""

    MAXIMUM_GAP_LINES: ClassVar[int] = 64

    def candidates(self) -> tuple[PendingCandidateRecord, ...]:
        """Expose pending candidates from the sealed plan representation.

        The sealed pending plan's records, whichever form sealed them:
        the frontier of a current record, the inline records of an earlier
        one.
        """
        if self.candidate_frontier is not None:
            return self.candidate_frontier.records()
        return self.pending_candidates

    def comparison_bindings(self) -> dict[str, str]:
        """Map sealed comparison closure hashes to their record hashes.

        Closure -> record hash of every sealed comparison this receipt
        stands on: what a continuation may resolve a pair through. Empty
        on a receipt sealed before comparisons were bound, which then
        computes its comparisons again.
        """
        return {b.closure_hash: b.record_hash for b in self.comparisons}

    def table_progress(self) -> dict[tuple[str, int], TableProgress]:
        """The progress of every table the sealed pages name."""
        return table_progress_of(self.table_views)

    def gap_lines(self) -> tuple[str, ...]:
        """Render machine-proven routing gaps as compact explanation lines.

        The machine-proved gaps of this routing as compact lines a
        readiness consumer carries beside the actor's report: every cell
        with a source or representation gap, unread unit needs, tables not
        dealt, delivered in part or of unknown progress, unread candidates
        or a queued residual scope, in cell order; the questions the pair
        budget skipped; and a closing count when the lines are cut at
        `MAXIMUM_GAP_LINES`. An actor's check erases none of them.
        """
        lines: list[str] = []
        progress = self.table_progress()
        for cell in self.cells:
            parts: list[str] = []
            pending = cell.unit_needs - cell.unit_needs_delivered
            if pending > 0:
                parts.append(f"{pending} of {cell.unit_needs} unit need(s) unread")
            tables_pending = cell.tables - cell.tables_delivered - cell.tables_unrenderable
            if tables_pending > 0:
                parts.append(f"{tables_pending} table(s) not dealt a view")
            states = cell_table_states(progress, cell.entity_id, cell.topic)
            if states["PARTIAL"]:
                parts.append(f"{states['PARTIAL']} table(s) delivered in part")
            if states["UNKNOWN_PROGRESS"]:
                parts.append(
                    f"{states['UNKNOWN_PROGRESS']} table(s) whose page progress is unknown"
                )
            if cell.tables_unrenderable:
                parts.append(
                    f"{cell.tables_unrenderable} table(s) unrenderable (REPRESENTATION_GAP)"
                )
            if cell.candidates_pending:
                parts.append(f"{cell.candidates_pending} returned candidate(s) unread")
            for gap in cell.gaps:
                parts.append(gap.split(":", 1)[0] if ":" in gap else gap)
            if parts:
                lines.append(f"{cell.entity_id} {cell.topic}: " + "; ".join(parts))
        if self.questions_skipped_budget:
            lines.append(
                f"{self.questions_skipped_budget} residual question(s) skipped by the "
                f"pair budget ({self.residual_reranked_pairs} of "
                f"{self.residual_rerank_pair_budget} pairs spent)"
            )
        if len(lines) > self.MAXIMUM_GAP_LINES:
            kept = lines[: self.MAXIMUM_GAP_LINES - 1]
            kept.append(f"{len(lines) - len(kept)} more cell gap line(s) in the routing record")
            return tuple(kept)
        return tuple(lines)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_routing(self) -> Self:
        """Validate distinct routing cells and delivered table/candidate handle agreement.

        Returns:
            This validated contract.

        Raises:
            ValueError: Routing cells or handles repeat, frontier representations coexist, or
                recorded handles disagree.
        """
        handles = self.table_view_span_handles
        if len(set(handles)) != len(handles) or any(
            not handle.startswith("SPAN-X") for handle in handles
        ):
            raise ValueError("alternative_evidence.routing_table_view_invalid")
        if self.candidate_frontier is not None and self.pending_candidates:
            raise ValueError("alternative_evidence.routing_candidates_invalid")
        seen = {(cell.entity_id, cell.topic) for cell in self.cells}
        if len(seen) != len(self.cells):
            raise ValueError("alternative_evidence.routing_cell_duplicate")
        if tuple(view.span_handle for view in self.table_views) != handles:
            raise ValueError("alternative_evidence.routing_table_view_invalid")
        read = tuple(
            c.span_handle
            for c in self.candidates()
            if c.state == "READ" and c.span_handle is not None
        )
        if (
            len(set(self.candidate_span_handles)) != len(self.candidate_span_handles)
            or any(not h.startswith("SPAN-C") for h in self.candidate_span_handles)
            or set(read) != set(self.candidate_span_handles)
        ):
            raise ValueError("alternative_evidence.routing_candidate_read_invalid")
        return self


class AlternativeEvidenceRetrievalAccessReceipt(AlternativeEvidenceContract):
    """Exactly what the Host read on the analyst's behalf, and by which program.

    The queries are the installed program, not the analyst's choice: which
    spans reached the analyst is therefore a deterministic function of the
    generation and the program, and two runs over the same generation read the
    same spans.
    """

    request_hash: str = Field(pattern=_HASH)
    document_set_hash: str = Field(pattern=_HASH)
    retrieval_generation_hash: str = Field(pattern=_HASH)
    query_program_hash: str = Field(pattern=_HASH)
    queries: tuple[EvidenceQueryRecord, ...] = Field(max_length=26)
    """The questions the Host ran or skipped by name -- the program's twenty
    and up to six region-scoped searches of the gap-directed residual
    search (`Q-<TOPIC>-GAP<n>`); the searches run stay bounded by
    `search_call_count`. Empty only when a structural scan stands in their
    place (`structural_scan`), never both absent."""
    # 128 spans and thirty-two reads: the packet is one evidence batch per
    # issuer rather than one global cut -- sixteen for twenty questions -- and
    # the read budget is what bounds it (`packet._allocate_by_cell` deals the
    # integrated selection's residual reads under the same bounds).
    read_span_handles: tuple[str, ...] = Field(max_length=128)
    search_call_count: int = Field(ge=0, le=24)
    span_read_call_count: int = Field(ge=0, le=32)
    span_groups: tuple[EvidenceSpanGroup, ...] = Field(
        default=(), max_length=128, exclude_if=lambda value: value == ()
    )
    """Same-filing candidates a read span stood for (see `EvidenceSpanGroup`);
    absent from the identity when empty, so earlier receipts keep theirs."""
    structural_scan: StructuralScanRecord | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    """The scan and coverage selection that read the spans, when the packet
    was selected from structure rather than from the question program;
    absent from the identity otherwise, so earlier receipts keep theirs."""
    typed_disclosures: TypedDisclosureRecord | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    """The typed disclosure families read beside the program, with their own
    span accounting (`TypedDisclosureRecord.span_handles`); absent from the
    identity when the families were not run, so earlier receipts keep theirs."""
    litigation_matters: LitigationMatterRecord | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    """The litigation matter inventory read beside the program and the typed
    families under its own allowance (`LitigationMatterRecord.span_handles`);
    absent from the identity when it was not run, so earlier receipts keep
    theirs."""
    routing: TopicRoutingRecord | None = Field(default=None, exclude_if=lambda value: value is None)
    """The integrated selection's routing (`TopicRoutingRecord`): the cells,
    the correspondences, the table views (its own span accounting) and the
    residual search's budget; absent from the identity when the selection
    was not the integrated one, so earlier receipts keep theirs."""
    selection_policy_hash: str | None = Field(
        default=None, pattern=_HASH, exclude_if=lambda value: value is None
    )
    """The analysis policy -- the program, the typed rules, the matter rules
    -- this selection ran under: the commitment a later request over the
    same generation content reuses it on. Absent from the identity on
    receipts sealed before it was recorded, which are never reused."""
    reused_from_receipt_hash: str | None = Field(
        default=None, pattern=_HASH, exclude_if=lambda value: value is None
    )
    """The receipt whose selection this one carries unchanged: sealed for
    another request over the same document references, generation content,
    program and policy, without a session -- its queries, reads and typed
    and matter entries are that receipt's; its request, document set and
    generation are this one's. Absent on a receipt sealed by its own
    session."""
    span_set_hash: str | None = Field(
        default=None, pattern=_HASH, exclude_if=lambda value: value is None
    )
    """The resolved span set this receipt delivered, sealed before the
    receipt and named by it: the one commitment a reuse of this selection
    stands on. A span set that merely carries the same handles under the
    same request and generation is not it. Absent from the identity on
    receipts sealed before it was recorded, which are never reused."""
    pair_score_commitment_hash: str | None = Field(
        default=None, pattern=_HASH, exclude_if=lambda value: value is None
    )
    """The commitment (`PairScoreCommitmentRecord`) naming the cross-encoder
    pair-score blocks this session sealed, sealed before the receipt and
    named by it: the authority a later reader serves those blocks under.
    Absent when the session sealed none (a continuation, a reuse, a refused
    admission) and on receipts sealed before scores were committed."""
    whole_filings: WholeFilingsRecord | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    """Present when the unit was delivered whole (W4): its pieces are the read
    spans, and no question ran; absent otherwise, so earlier receipts keep
    their identities."""
    receipt_hash: str = Field(pattern=_HASH)

    @property
    def delivered_span_handles(self) -> tuple[str, ...]:
        """Return all span handles delivered under the sealed access receipt.

        Every handle the packet delivered: the program's reads, the typed
        families' spans, the matter windows read, the sealed candidates a
        later session read, and the table views, in packet order.
        """
        typed = () if self.typed_disclosures is None else self.typed_disclosures.span_handles
        matters = (
            ()
            if self.litigation_matters is None
            else self.litigation_matters.delivered_matter_handles
        )
        tables = () if self.routing is None else self.routing.table_view_span_handles
        candidates = () if self.routing is None else self.routing.candidate_span_handles
        return (*self.read_span_handles, *typed, *matters, *candidates, *tables)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_receipt(self) -> Self:
        # A receipt names its program's questions, a structural scan, or the
        # integrated routing (whose questions may all have been skipped for
        # want of residual scope: a legitimate zero-query selection, with the
        # routing record saying why); never none of the three.
        """Validate handle partitions, query IDs, recorded access and receipt identity.

        Returns:
            This validated contract.

        Raises:
            ValueError: Access is absent, handles overlap/repeat or lack delivery, query IDs repeat,
                or identity differs.
        """
        if (
            not self.queries
            and self.structural_scan is None
            and self.routing is None
            and self.whole_filings is None
        ):
            raise ValueError("alternative_evidence.access_program_missing")
        if len(self.read_span_handles) != len(set(self.read_span_handles)):
            raise ValueError("alternative_evidence.access_span_duplicate")
        typed = self.typed_disclosures
        if typed is not None:
            typed_handles = set(typed.span_handles)
            if len(typed_handles) != len(typed.span_handles) or typed_handles & set(
                self.read_span_handles
            ):
                raise ValueError("alternative_evidence.access_span_duplicate")
            cited = {
                handle
                for observation in typed.observations
                for handle in (
                    observation.scope_span_handle,
                    *(instance.span_handle for instance in observation.instances),
                )
                if handle is not None
            }
            if not cited.issubset(typed_handles):
                raise ValueError("alternative_evidence.typed_disclosure_span_unread")
        matters = self.litigation_matters
        if matters is not None:
            delivered = matters.delivered_matter_handles
            matter_handles = set(delivered)
            others = set(self.read_span_handles) | (
                set() if typed is None else set(typed.span_handles)
            )
            if len(matter_handles) != len(delivered) or matter_handles & others:
                raise ValueError("alternative_evidence.access_span_duplicate")
        if self.routing is not None:
            tables = set(self.routing.table_view_span_handles)
            others = set(self.read_span_handles) | (
                set() if typed is None else set(typed.span_handles)
            )
            if matters is not None:
                others |= set(matters.delivered_matter_handles)
            if tables & others:
                raise ValueError("alternative_evidence.access_span_duplicate")
        read = set(self.read_span_handles)
        members = [handle for group in self.span_groups for handle in group.member_span_handles]
        if any(group.representative_span_handle not in read for group in self.span_groups):
            raise ValueError("alternative_evidence.access_span_group_representative_unread")
        if len(members) != len(set(members)) or read & set(members):
            raise ValueError("alternative_evidence.access_span_group_invalid")
        ids = tuple(value.query_id for value in self.queries)
        if len(ids) != len(set(ids)):
            raise ValueError("alternative_evidence.access_query_duplicate")
        validate_contract_identity(self, "receipt_hash")
        return self


class AlternativeEvidenceBriefFindingSubmission(BaseModel):  # type: ignore[misc]
    """One atomic, source-grounded statement. What the document says, classified.

    Sealed receipts hold it as the actor wrote it, or -- for an answer -- as
    the Host normalized the answer's finding (Host handle, span handles).
    """

    model_config = ConfigDict(extra="forbid")

    finding_handle: str = Field(pattern=r"^FIND-[A-Z0-9-]{1,48}$")
    affected_entities: tuple[str, ...] = Field(min_length=1, max_length=8)
    topic: EvidenceTopic
    lifecycle: EvidenceLifecycle | None = Field(default=None, exclude_if=lambda v: v is None)
    """None when the answer stated no lifecycle; absent from the serialization
    then, so every earlier submission keeps its hash."""
    direction: EvidenceDirection
    summary: str = Field(min_length=1, max_length=1200)
    supporting_span_handles: tuple[str, ...] = Field(max_length=8)
    contradicting_span_handles: tuple[str, ...] = Field(max_length=8)
    limitations: tuple[str, ...] = Field(max_length=8)


class CheckOutcomeKind(StrEnum):
    """What the actor says it did with one required check for one issuer."""

    COMPLETED = "COMPLETED"
    DEFERRED = "DEFERRED"


class AlternativeEvidenceCheckOutcomeSubmission(BaseModel):  # type: ignore[misc]
    """The actor's own account of one required check for one issuer.

    A check is completed only when the actor says so here; delivered spans,
    an existing packet or the absence of findings never complete one. A
    deferral says why, so the reviewer sees what was not done.
    """

    model_config = ConfigDict(extra="forbid")

    entity_id: str = Field(min_length=1, max_length=32)
    check: str = Field(min_length=1, max_length=80)
    outcome: CheckOutcomeKind
    note: str = Field(default="", max_length=600)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_outcome(self) -> Self:
        """Require a nonblank explanation for a deferred check.

        Returns:
            This validated contract.

        Raises:
            ValueError: A deferred outcome has no nonblank note.
        """
        if self.outcome is CheckOutcomeKind.DEFERRED and not self.note.strip():
            raise ValueError("alternative_evidence.check_outcome_deferral_unexplained")
        return self


class AlternativeEvidenceAnalystBriefSubmission(BaseModel):  # type: ignore[misc]
    """The sealed brief's actor-side content.

    Until the answer format existed an actor wrote this whole document --
    handles, summaries, a coverage assessment, a review flag and per-check
    reports -- and sealed receipts hold it that way; it is read back, never
    admitted as a new submission. For an answer the Host writes it: the
    answer's findings normalized, the texts written from delivery facts,
    no check report and no review flag.
    """

    model_config = ConfigDict(extra="forbid")

    executive_summary: str = Field(min_length=1, max_length=2000)
    findings: tuple[AlternativeEvidenceBriefFindingSubmission, ...] = Field(max_length=32)
    unresolved_questions: tuple[str, ...] = Field(max_length=8)
    source_coverage_assessment: str = Field(min_length=1, max_length=1000)
    requires_human_review: bool
    limitations_acknowledged: bool
    check_outcomes: tuple[AlternativeEvidenceCheckOutcomeSubmission, ...] = Field(
        default=(), max_length=64, exclude_if=lambda value: value == ()
    )
    """Per issuer and required check, what the actor completed or deferred.
    Absent from the serialization when empty, so submissions sealed before
    it existed keep their hashes; an issuer with delivered spans and no
    outcome here reads as `NOT_REPORTED`, never as reviewed."""


class AlternativeEvidenceAnswerFinding(BaseModel):  # type: ignore[misc]
    """One finding as the Analyst writes it: what the cited excerpts state, judged.

    `issuer` is one of the bundle's issuers as its view names it; `cite` and
    `contrary` are the aliases of the excerpts that state and that contradict
    it. The Host assigns the handle and maps each alias to its excerpt.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    issuer: str = Field(min_length=1, max_length=32)
    topic: EvidenceTopic
    direction: EvidenceDirection
    summary: str = Field(min_length=1, max_length=1200)
    cite: tuple[str, ...] = Field(min_length=1, max_length=8)
    contrary: tuple[str, ...] = Field(default=(), max_length=8, exclude_if=lambda v: v == ())
    lifecycle: EvidenceLifecycle | None = Field(default=None, exclude_if=lambda v: v is None)


class AlternativeEvidenceAnalystAnswer(BaseModel):  # type: ignore[misc]
    """The Analyst's answer: judgment fields only.

    No hash, no handle, no review flag, no coverage account and no per-check
    report: those are the Host's. A subset of what the material holds is an
    answer, and an empty list is an answer with no findings.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    findings: tuple[AlternativeEvidenceAnswerFinding, ...] = Field(default=(), max_length=32)
    notes: str = Field(default="", max_length=2000, exclude_if=lambda v: v == "")


ANALYST_ANSWER_TEXT_FIELDS: dict[str, int] = {"notes": 2000}
"""The answer's optional texts and their bounds, beside its `findings`."""


class AlternativeEvidenceBriefFinding(AlternativeEvidenceContract):
    """Bind one bounded analyst finding to affected issuers and cited evidence.

    Attributes:
        finding_handle: Semantic FIND handle.
        affected_entities: Bounded nonempty affected issuer axis.
        topic: Declared finding topic.
        lifecycle: Optional matter lifecycle when recorded.
        direction: Declared finding direction.
        summary: Bounded analyst interpretation.
        supporting_span_handles: Up to eight supporting evidence handles.
        contradicting_span_handles: Up to eight contrary evidence handles.
        limitations: Bounded qualifications retained with the finding.
    """

    finding_handle: str = Field(pattern=r"^FIND-[A-Z0-9-]{1,48}$")
    affected_entities: tuple[str, ...] = Field(min_length=1, max_length=8)
    topic: EvidenceTopic
    lifecycle: EvidenceLifecycle | None = Field(default=None, exclude_if=lambda v: v is None)
    """None when the answer stated none; absent then, so earlier briefs keep
    their hashes."""
    direction: EvidenceDirection
    summary: str = Field(min_length=1, max_length=1200)
    supporting_span_handles: tuple[str, ...] = Field(max_length=8)
    contradicting_span_handles: tuple[str, ...] = Field(max_length=8)
    limitations: tuple[str, ...] = Field(max_length=8)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_finding(self) -> Self:
        """Require at least one cited handle and disjoint unique supporting/contradicting citations.

        Returns:
            This validated contract.

        Raises:
            ValueError: The finding has no citations or repeats a handle across its citation lists.
        """
        handles = (*self.supporting_span_handles, *self.contradicting_span_handles)
        if not handles or len(handles) != len(set(handles)):
            raise ValueError("alternative_evidence.brief_finding_citations_invalid")
        return self


class IssuerReviewState(StrEnum):
    """What the analysis established for one issuer, apart from what it found.

    Derived by the sealer from what the packet held and what the brief
    returned -- never from the brief's opinion of itself: an issuer with no
    admitted document could not be checked; one with documents but no
    delivered span was not read; one that was read and yielded no finding
    was checked and is clean of *reported* findings, which is not proof of
    no risk; one with findings was checked and found something.
    """

    SOURCE_MISSING = "SOURCE_MISSING"
    NO_SPANS_DELIVERED = "NO_SPANS_DELIVERED"
    NOT_REPORTED = "NOT_REPORTED"
    """(v2) Spans were delivered and the actor reported no check outcome for
    the issuer: delivered, not reviewed."""
    CHECKS_INCOMPLETE = "CHECKS_INCOMPLETE"
    """(v2) The actor deferred or left unreported at least one required check."""
    EXECUTED_NO_FINDINGS = "EXECUTED_NO_FINDINGS"
    """Read and answered with no finding: clean of *reported* findings, never
    proof of no risk. Under v3 a delivery fact: excerpts were delivered and
    the answer named nothing for the issuer."""
    EXECUTED_WITH_FINDINGS = "EXECUTED_WITH_FINDINGS"


EXECUTED_STATES: frozenset[IssuerReviewState] = frozenset(
    {IssuerReviewState.EXECUTED_NO_FINDINGS, IssuerReviewState.EXECUTED_WITH_FINDINGS}
)


COMPLETION_SCHEMA_V3 = "issuer-delivery-facts-v3"
"""Completions of answers: every state is a delivery fact -- documents
admitted, excerpts delivered, findings answered -- and no actor reports a
check, so `checks_executed` is empty. Earlier completions read as sealed."""


class IssuerCheckDeferral(AlternativeEvidenceContract):
    check: str = Field(min_length=1, max_length=80)
    note: str = Field(min_length=1, max_length=600)


class IssuerCheckOutcome(AlternativeEvidenceContract):
    """One issuer's outcome under the obligation's required checks."""

    entity_id: str = Field(min_length=1, max_length=32)
    state: IssuerReviewState
    documents_admitted: int = Field(ge=0)
    spans_delivered: int = Field(ge=0)
    findings: int = Field(ge=0)
    contradicting_findings: int = Field(ge=0)
    checks_executed: tuple[str, ...] = Field(max_length=8)
    """Under v2, the required checks the actor reported completed for this
    issuer; under v1, every required check where spans were delivered; under
    v3, empty -- no check is self-reported."""
    checks_deferred: tuple[IssuerCheckDeferral, ...] = Field(
        default=(), max_length=8, exclude_if=lambda value: value == ()
    )
    checks_unreported: tuple[str, ...] = Field(
        default=(), max_length=8, exclude_if=lambda value: value == ()
    )
    """Required checks the actor neither completed nor deferred (v2)."""


class AnalysisCompletion(AlternativeEvidenceContract):
    """Record versioned per-issuer review completion and check outcomes.

    The brief's completion, versioned: per-issuer outcomes under the
    required checks. `review_status` keeps its historical, findings-dependent
    meaning; this is the explicit one readers should branch on.
    """

    completion_schema: Literal[
        "issuer-check-outcomes-v1", "issuer-check-outcomes-v2", "issuer-delivery-facts-v3"
    ] = "issuer-check-outcomes-v2"
    required_checks: tuple[str, ...] = Field(min_length=1, max_length=8)
    issuers: tuple[IssuerCheckOutcome, ...] = Field(min_length=1, max_length=8)

    @property
    def executed_entity_ids(self) -> tuple[str, ...]:
        return tuple(value.entity_id for value in self.issuers if value.state in EXECUTED_STATES)

    def state_of(self, entity_id: str) -> IssuerReviewState | None:
        for value in self.issuers:
            if value.entity_id == entity_id:
                return value.state
        return None


class AlternativeEvidenceAnalystBrief(AlternativeEvidenceContract):
    """Seal analyst interpretations against the admitted evidence and review lineage.

    Attributes:
        kind: Brief discriminator.
        request_hash: Research request identity.
        source_snapshot_hash: Admitted source snapshot identity.
        document_set_hash: Canonical document-set identity.
        retrieval_generation_hash: Retrieval generation identity.
        access_receipt_hash: Recorded access commitment.
        review_binding_hash: Installed review implementation binding.
        obligation_hash: Host research obligation identity.
        review_status: Complete or analysis-incomplete review state.
        executive_summary: Bounded analyst summary.
        findings: Uniquely handled findings, empty for incomplete analysis.
        unresolved_questions: Bounded outstanding questions.
        source_coverage_assessment: Declared review coverage.
        requires_human_review: Required for an incomplete analysis.
        completed_at: Aware completion clock.
        completion: Optional structured completion record.
        brief_hash: Canonical brief identity.
    """

    kind: Literal["AlternativeEvidenceAnalystBrief"] = "AlternativeEvidenceAnalystBrief"
    request_hash: str = Field(pattern=_HASH)
    source_snapshot_hash: str = Field(pattern=_HASH)
    document_set_hash: str = Field(pattern=_HASH)
    retrieval_generation_hash: str = Field(pattern=_HASH)
    access_receipt_hash: str = Field(pattern=_HASH)
    review_binding_hash: str = Field(pattern=_HASH)
    obligation_hash: str = Field(pattern=_HASH)
    review_status: AlternativeEvidenceReviewStatus
    executive_summary: str = Field(min_length=1, max_length=2000)
    findings: tuple[AlternativeEvidenceBriefFinding, ...] = Field(max_length=32)
    unresolved_questions: tuple[str, ...] = Field(max_length=8)
    source_coverage_assessment: str = Field(min_length=1, max_length=1000)
    requires_human_review: bool
    completed_at: datetime
    completion: AnalysisCompletion | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    """Absent from the identity of briefs sealed before it existed."""
    brief_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_brief(self) -> Self:
        """Validate completion, incomplete-review safeguards, finding handles and identity.

        Returns:
            This validated contract.

        Raises:
            ValueError: The clock is naive, incomplete safeguards fail, handles repeat, or canonical
                identity differs.
        """
        if self.completed_at.tzinfo is None or self.completed_at.utcoffset() is None:
            raise ValueError("alternative_evidence.analyst_brief_clock_invalid")
        if self.review_status is AlternativeEvidenceReviewStatus.ANALYSIS_INCOMPLETE and (
            self.findings or not self.requires_human_review
        ):
            raise ValueError("alternative_evidence.incomplete_brief_invalid")
        handles = tuple(value.finding_handle for value in self.findings)
        if len(handles) != len(set(handles)):
            raise ValueError("alternative_evidence.finding_handle_duplicate")
        validate_contract_identity(self, "brief_hash")
        return self


class AlternativeEvidenceAnalystBriefReceipt(AlternativeEvidenceContract):
    """Host-sealed provenance for one actor-neutral evidence extraction."""

    kind: Literal["AlternativeEvidenceAnalystBriefReceipt"] = (
        "AlternativeEvidenceAnalystBriefReceipt"
    )
    submission: AlternativeEvidenceAnalystBriefSubmission
    answer: AlternativeEvidenceAnalystAnswer | None = Field(
        default=None, exclude_if=lambda v: v is None
    )
    """The actor's answer -- its accepted items -- when it answered in the
    judgment-only format; `submission` is then the Host's normalization and
    the actor binding names the answer. Absent from receipts sealed before."""
    dropped: tuple[AnswerProblem, ...] = Field(
        default=(), max_length=256, exclude_if=lambda v: v == ()
    )
    """Items of the answer the Host did not accept, each with its problem."""
    actor_submission: ActorSubmissionBinding
    brief: AlternativeEvidenceAnalystBrief
    decision_policy_hash: str = Field(pattern=_HASH)
    model_call_count: int = Field(ge=0, le=3)
    """Calls started: the first answer and at most two corrections."""
    protocol_repair_count: int = Field(ge=0, le=2)
    receipt_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_actor_receipt(self) -> Self:
        """Verify actor submission bytes, retained answer semantics and receipt identity.

        Returns:
            This validated contract.

        Raises:
            ValueError: Actor submission identity, dropped-answer presence, retained brief fields or
                receipt identity disagree.
        """
        written = self.submission if self.answer is None else self.answer
        if self.actor_submission.submission_hash != canonical_hash(written.model_dump(mode="json")):
            raise ValueError("alternative_evidence.actor_submission_mismatch")
        if self.dropped and self.answer is None:
            raise ValueError("alternative_evidence.actor_answer_drops_invalid")
        if (
            self.submission.executive_summary != self.brief.executive_summary
            or self.submission.unresolved_questions != self.brief.unresolved_questions
            or self.submission.source_coverage_assessment != self.brief.source_coverage_assessment
            or self.submission.requires_human_review != self.brief.requires_human_review
        ):
            raise ValueError("alternative_evidence.actor_brief_content_mismatch")
        validate_contract_identity(self, "receipt_hash")
        return self


class EvidenceStructureState(StrEnum):
    """How a finding is held up by its own citations. Derived, never asserted.

    `SUPPORTED` needs two or more distinct documents behind it and no span
    against it; one document is `SINGLE_SOURCE`; any contradicting span makes
    it `CONTESTED` whatever the count; no supporting span at all is
    `UNSUPPORTED`.
    """

    SUPPORTED = "SUPPORTED"
    SINGLE_SOURCE = "SINGLE_SOURCE"
    CONTESTED = "CONTESTED"
    UNSUPPORTED = "UNSUPPORTED"


class AlternativeEvidenceFindingStructure(AlternativeEvidenceContract):
    """Describe evidence structure for one finding without deciding its investment meaning.

    Attributes:
        finding_handle: Finding whose evidence structure is described.
        supporting_document_count: Bounded distinct supporting-document count.
        contradicting_document_count: Bounded distinct contrary-document count.
        latest_available_at: Latest availability clock retained for this structure.
        state: Declared evidence structure state.
    """

    finding_handle: str = Field(pattern=r"^FIND-[A-Z0-9-]{1,48}$")
    supporting_document_count: int = Field(ge=0, le=8)
    contradicting_document_count: int = Field(ge=0, le=8)
    latest_available_at: datetime
    state: EvidenceStructureState


class CROAlternativeEvidencePackage(AlternativeEvidenceContract):
    """Seal verified evidence structure for CRO review without a Portfolio decision.

    Attributes:
        kind: CRO package discriminator.
        request_hash: Research request identity.
        source_snapshot_hash: Source snapshot commitment.
        document_set_hash: Canonical document-set identity.
        retrieval_generation_hash: Retrieval generation identity.
        analyst_brief_hash: Analyst brief identity.
        access_receipt_hash: Recorded evidence-access commitment.
        obligation_hash: Research obligation identity.
        source_status: Declared source coverage state.
        admitted_document_count: Number of admitted documents.
        rejected_document_count: Number of rejected documents.
        verified_spans: Unique verified source spans.
        finding_structures: Unique bounded finding structures.
        missing_evidence: Explicit evidence gaps.
        expires_at: Aware review-package expiry.
        limitations: Review qualifications.
        package_hash: Canonical package identity.
    """

    kind: Literal["CROAlternativeEvidencePackage"] = "CROAlternativeEvidencePackage"
    request_hash: str = Field(pattern=_HASH)
    source_snapshot_hash: str = Field(pattern=_HASH)
    document_set_hash: str = Field(pattern=_HASH)
    retrieval_generation_hash: str = Field(pattern=_HASH)
    analyst_brief_hash: str = Field(pattern=_HASH)
    access_receipt_hash: str = Field(pattern=_HASH)
    obligation_hash: str = Field(pattern=_HASH)
    source_status: str
    admitted_document_count: int = Field(ge=0, le=24)
    rejected_document_count: int = Field(ge=0, le=24)
    verified_spans: tuple[AlternativeEvidenceResolvedSpan, ...]
    finding_structures: tuple[AlternativeEvidenceFindingStructure, ...] = Field(max_length=32)
    missing_evidence: tuple[str, ...]
    expires_at: datetime
    limitations: tuple[str, ...]
    package_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_package(self) -> Self:
        """Require aware expiry, unique verified/finding handles and canonical package identity.

        Returns:
            This validated contract.

        Raises:
            ValueError: The expiry is naive, handles repeat, or canonical package identity differs.
        """
        if self.expires_at.tzinfo is None or self.expires_at.utcoffset() is None:
            raise ValueError("alternative_evidence.cro_package_clock_invalid")
        handles = tuple(value.span_handle for value in self.verified_spans)
        if len(handles) != len(set(handles)):
            raise ValueError("alternative_evidence.cro_package_span_duplicate")
        structures = tuple(value.finding_handle for value in self.finding_structures)
        if len(structures) != len(set(structures)):
            raise ValueError("alternative_evidence.cro_package_structure_duplicate")
        validate_contract_identity(self, "package_hash")
        return self

    def structure(self, finding_handle: str) -> AlternativeEvidenceFindingStructure:
        """Find the recorded evidence structure for a finding.

        Args:
            finding_handle: Semantic finding handle to look up.

        Returns:
            The matching finding structure.

        Raises:
            KeyError: No structure has the requested handle.
        """
        for value in self.finding_structures:
            if value.finding_handle == finding_handle:
                return value
        raise KeyError(finding_handle)


__all__ = [
    "ANALYST_ANSWER_TEXT_FIELDS",
    "COMPLETION_SCHEMA_V3",
    "CONTEXT_COMPLETE_SELECTION_RULES_ID",
    "WHOLE_FILINGS_RULES_ID",
    "AlternativeEvidenceAnalystAnswer",
    "AlternativeEvidenceAnalystBrief",
    "AlternativeEvidenceAnalystBriefReceipt",
    "AlternativeEvidenceAnalystBriefSubmission",
    "AlternativeEvidenceAnswerFinding",
    "AlternativeEvidenceBriefFinding",
    "AlternativeEvidenceBriefFindingSubmission",
    "AlternativeEvidenceCheckOutcomeSubmission",
    "AlternativeEvidenceFindingStructure",
    "AlternativeEvidenceResearchObligation",
    "AlternativeEvidenceRetrievalAccessReceipt",
    "CROAlternativeEvidencePackage",
    "CheckOutcomeKind",
    "EvidenceDirection",
    "EvidenceLifecycle",
    "EvidenceQueryRecord",
    "EvidenceSpanGroup",
    "EvidenceStructureState",
    "EvidenceTopic",
    "IssuerTopicCellRecord",
    "LitigationMatterDocument",
    "LitigationMatterRecord",
    "LitigationRegionRecord",
    "MatterWindowRecord",
    "ProvisionalMatterRecord",
    "SelectedWindowFacet",
    "StructuralScanRecord",
    "TableViewRecord",
    "TableViewRefusalRecord",
    "TopicRoutingRecord",
    "TypedDisclosureField",
    "TypedDisclosureInstance",
    "TypedDisclosureObservation",
    "TypedDisclosureRecord",
    "UnassignedRangeRecord",
    "UnitCorrespondenceRecord",
    "WholeFilingsRecord",
    "seal_research_obligation",
]
