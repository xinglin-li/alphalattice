"""Typed authority, source and snapshot contracts for Alternative Evidence.

Everything here is Host-owned. A request names an issuer axis, a cutoff and a
source policy; an admission records the permissions the Host granted for it; a
snapshot freezes what the sources actually yielded before the cutoff. None of
these carry a Portfolio weight, a Portfolio effect or a current pointer: the
active publication is pointer-free and the Portfolio binding lives in the
review that consumes it.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.sealing import seal_model_validated

_HASH = r"^[0-9a-f]{64}$"


class AlternativeEvidenceContract(BaseModel):  # type: ignore[misc]
    """Freeze evidence contract values and reject undeclared fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class AlternativeEvidenceMode(StrEnum):
    """Select recorded sources or admitted live official acquisition.

    The mode declares acquisition intent. Network consent and permission to perform
    live acquisition are carried separately by the host-owned admission.
    """

    RECORDED = "RECORDED"
    LIVE_OFFICIAL = "LIVE_OFFICIAL"


class AlternativeEvidenceReadingDepth(StrEnum):
    """Choose material sections or the full filing for document admission."""

    MATERIAL_SECTIONS = "MATERIAL_SECTIONS"
    FULL_FILING = "FULL_FILING"


class AlternativeEvidenceClass(StrEnum):
    """Identify SEC filings, SEC company facts or recorded issuer evidence."""

    SEC_FILING = "SEC_FILING"
    SEC_COMPANYFACTS = "SEC_COMPANYFACTS"
    ISSUER_OFFICIAL_RECORDED = "ISSUER_OFFICIAL_RECORDED"


SOURCE_FAMILY_BY_EVIDENCE_CLASS: dict[AlternativeEvidenceClass, str] = {
    AlternativeEvidenceClass.SEC_FILING: "SEC_EDGAR_OFFICIAL",
    AlternativeEvidenceClass.SEC_COMPANYFACTS: "SEC_EDGAR_OFFICIAL",
    AlternativeEvidenceClass.ISSUER_OFFICIAL_RECORDED: "SEC_EDGAR_OFFICIAL",
}
"""Which approved source family may serve each admitted evidence class.

Declared once, as a table, because the alternative is inferring the relationship
from the two names -- and a rule that reads `SEC_FILING` and `SEC_EDGAR_OFFICIAL`
as related by their shared prefix would silently accept or reject the next class
someone adds depending on what they called it.
"""


class AlternativeEvidenceSnapshotStatus(StrEnum):
    """Classify source coverage from available, missing and failed source counts.

    Complete coverage has every expected source. Partial coverage has at least one
    available source; without one, failed sources make the snapshot unavailable and
    an absence without failures makes it empty.
    """

    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    EMPTY = "EMPTY"
    UNAVAILABLE = "UNAVAILABLE"


class AlternativeEvidenceReviewStatus(StrEnum):
    """Distinguish completed review from an explicitly incomplete analysis."""

    COMPLETE = "COMPLETE"
    ANALYSIS_INCOMPLETE = "ANALYSIS_INCOMPLETE"


ADMITTED_DOCUMENT_CAPACITY: int = 24
"""The most documents one canonicalized document set admits. One request's
issuers share it by their own selections: each issuer's selection is
planned under its policy budget alone, the unit is packed from those
selections (`runtime.coverage.coverage_units`), and a unit whose selections
still exceed the set defers event filings by name (`sec_edgar.apply_unit_capacity`,
`recorded.RecordedEvidenceSource.select_documents`) -- so a document the
canonicalizer would reject for capacity is never acquired, and adding an
issuer to a batch never removes another issuer's filing from its plan."""

WHOLE_FILING_BYTES: int = 100 * 1024
"""The bundle's file bound (`protocols.actor_execution.bundles.BUNDLE_FILE_BYTES`): a unit
whose filings together hold no more is delivered whole, with no retrieval index (W4)."""

MAXIMUM_READ_FILINGS: int = 512
"""The most earlier-read filings one request names: eight issuers, each with
every filing a 30-day window can hold read over the days before."""


MAXIMUM_SOURCE_DOCUMENT_BYTES: int = 10_000_000
"""The most one source document may hold, anywhere: the ceiling a request's
source policy can admit at acquisition and the bound canonicalization
applies to retained bytes are the same number, stated once."""

DEFAULT_SOURCE_DOCUMENT_BYTES: int = 10_000_000
"""The per-document cap a request carries when none is declared -- the one
default the source policy, the product's admitted policy and the
materializer's flag all read. Set to the ceiling on 2026-09-19: of the 69
real bodies the exploration retained, 25 exceeded the earlier default of
2,000,000 bytes (the largest 7,481,927), so a normal user obtained no
supported annual filing without a flag. A smaller cap declared explicitly
stays enforced; a sealed request keeps the cap it was sealed with."""

PublishedPrecision = Literal["DATE", "INSTANT"]
"""How exact a document's `published_at` is. `DATE`: the source states a
date only (an SEC filing date), and the stored value is midnight UTC of that
date -- not a publication instant. `INSTANT`: the source states the instant.
Absent: unknown, as for documents acquired before precision was recorded; a
consumer must not read such a midnight as an exact time."""


class AlternativeEvidenceSourcePolicy(AlternativeEvidenceContract):
    """Bound official-source lookback, issuer scope and acquisition bytes.

    The 30-day policy does not select fixed latest-filing baselines. The historical
    90-day policy retains both baseline flags for readback and is marked retired.

    Attributes:
        sec_recent_8k_days: Supported current or historical event lookback in days.
        include_latest_10q: Historical baseline-selection flag tied to the 90-day policy.
        include_latest_10k: Historical baseline-selection flag tied to the 90-day policy.
        include_companyfacts_summary: Whether official company facts belong to the request.
        maximum_issuers: Issuer admission capacity, between one and eight.
        maximum_documents_per_issuer: Per-issuer document budget, between three and twenty.
        maximum_document_bytes: Bound on the bytes of one source document.
        whole_filing_bytes: Optional whole-filing byte budget, omitted from serialization when zero.
    """

    sec_recent_8k_days: Literal[30, 90] = 30
    """The filing window, in days before the cutoff. 30 since W1 (2026-09-24):
    what each issuer filed recently, and nothing older is kept as a baseline.
    90, with the latest 10-K and 10-Q as fixed baselines, is the retired
    policy: a request sealed under it still reads back, and is prepared again
    under this one rather than executed."""
    include_latest_10q: bool = False
    include_latest_10k: bool = False
    include_companyfacts_summary: Literal[True] = True
    maximum_issuers: int = Field(default=8, ge=1, le=8)
    maximum_documents_per_issuer: int = Field(default=12, ge=3, le=20)
    maximum_document_bytes: int = Field(
        default=DEFAULT_SOURCE_DOCUMENT_BYTES, ge=1_000, le=MAXIMUM_SOURCE_DOCUMENT_BYTES
    )
    whole_filing_bytes: int = Field(
        default=0, ge=0, le=WHOLE_FILING_BYTES, exclude_if=lambda value: value == 0
    )
    """A unit whose filings together hold at most this many bytes is delivered
    whole: no retrieval index, no residual questions, no rerank (W4). The live
    policy sets the bundle's file bound; 0 -- the recorded policy's, and absent
    from the identity -- delivers every unit through the selection."""

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_source_policy(self) -> Self:
        # The baselines belong to the retired window and to nothing else.
        """Require baseline flags to agree with the declared lookback policy.

        Returns:
            This validated contract.

        Raises:
            ValueError: Either baseline flag disagrees with the historical 90-day selection mode.
        """
        if (self.include_latest_10q, self.include_latest_10k) != (self.retired, self.retired):
            raise ValueError("alternative_evidence.source_policy_invalid")
        return self

    @property
    def retired(self) -> bool:
        """The 90-day policy with fixed baselines, retired by W1."""
        return self.sec_recent_8k_days == 90


MATTER_SELECTION_PRODUCTION = "PRODUCTION_PLAN_PREFIX"
MATTER_SELECTION_CANDIDATE = "CANDIDATE_UNMET_NEEDS"
MATTER_SELECTION_INTEGRATED = "INTEGRATED_TOPIC_ROUTING"
MATTER_FAMILY_LITIGATION = "LITIGATION"
MATTER_FAMILY_CORPORATE_EVENT = "CORPORATE_EVENT"
MATTER_FAMILY_FINANCING = "FINANCING"
INTEGRATED_FAMILY_SPELLING = (
    MATTER_FAMILY_LITIGATION,
    MATTER_FAMILY_CORPORATE_EVENT,
    MATTER_FAMILY_FINANCING,
)
"""The one spelling of the integrated method's families: the three the
workspace manifests bind into their authority hash and every sealed
request carries. The method's inventories are `packet.MATTER_FAMILIES`
(the operations inventory joined them under the eight-topic completeness
assignment, 2026-09-20); a sealed receipt records which it ran, and the
analysis policy hash tells one version of the method from another. The
spelling moves only when the request contract's next identity move is
authorized, with every manifest re-sealed."""


class MatterSelectionPolicy(AlternativeEvidenceContract):
    """Declare the versioned matter-selection method, families and residual policy.

    Which matter selection a request asks the Host to run beside the
    query program and the typed families, bound into the request's identity
    and so into every Task, receipt and continuation that names it:
    `PRODUCTION_PLAN_PREFIX`, the production reading plan over the litigation
    regions of the periodic filings (the default, absent from the request at
    that default so every earlier request keeps its hash), or
    `CANDIDATE_UNMET_NEEDS`, the candidate needs allocation over the named
    families -- the litigation regions; with `CORPORATE_EVENT`, the event
    notes of the periodic filings and the items of the current reports;
    with `FINANCING`, the debt, borrowings and financing notes -- under the
    same one per-session allowance, the families written in that one order
    so a family set has one identity; or `INTEGRATED_TOPIC_ROUTING`, the
    integrated pipeline -- shared discovery of every unit family of
    `packet.MATTER_FAMILIES` once, the eight topics routed by source shape,
    the units dealt in issuer-topic lanes latest filing first with exact
    repeats collapsed, tables inside routed regions delivered as verified
    views, and the question bank run only over the residual scope under a
    pair budget -- which is written with `INTEGRATED_FAMILY_SPELLING`, the
    method's one identity. A QA opt-in the admitted policy carries; never a
    global default and never a wider ceiling.
    """

    method: Literal["PRODUCTION_PLAN_PREFIX", "CANDIDATE_UNMET_NEEDS", "INTEGRATED_TOPIC_ROUTING"]
    families: tuple[str, ...] = Field(default=("LITIGATION",), min_length=1, max_length=4)
    allocation: Literal["CELL_ROUNDS", "CONTEXT_COMPLETE"] = Field(
        default="CELL_ROUNDS", exclude_if=lambda v: v == "CELL_ROUNDS"
    )
    """How the integrated selection deals its residual reads: `CELL_ROUNDS`,
    the issuer-topic cells in rounds, one candidate a turn -- the one
    policy the runtime deals. `CONTEXT_COMPLETE` (the same rounds dealing
    paragraph-complete bundles; section V's Gate A) is retired: the value
    stays on the contract so every request and receipt sealed under it
    reads back with its identity, and a request that carries it is
    refused by name (`retired`). Absent from the identity at the default."""
    residual_search: Literal["QUESTION_BANK", "GAP_DIRECTED"] = Field(
        default="QUESTION_BANK", exclude_if=lambda v: v == "QUESTION_BANK"
    )
    """The residual search: the question bank over each topic's residual
    scope (`QUESTION_BANK`), the one policy the runtime deals.
    `GAP_DIRECTED` (that bank followed by bounded region-scoped searches;
    section V's Gate C) is retired on the same terms as `CONTEXT_COMPLETE`.
    Absent at the default."""

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_policy(self) -> Self:
        """Require canonical families and the selection method allowed for each residual policy.

        Returns:
            This validated contract.

        Raises:
            ValueError: Families repeat, are unknown or out of order, or method and residual policy
                disagree.
        """
        if (
            self.allocation != "CELL_ROUNDS" or self.residual_search != "QUESTION_BANK"
        ) and self.method != MATTER_SELECTION_INTEGRATED:
            raise ValueError("alternative_evidence.matter_selection_policy_invalid")
        known = (
            MATTER_FAMILY_LITIGATION,
            MATTER_FAMILY_CORPORATE_EVENT,
            MATTER_FAMILY_FINANCING,
        )
        if len(set(self.families)) != len(self.families) or any(
            family not in known for family in self.families
        ):
            raise ValueError("alternative_evidence.matter_families_unknown")
        # One spelling per family set: the identity a request, a sealed
        # selection and a published analysis are compared by is the families'
        # text, so the set is written in the families' one order.
        if tuple(family for family in known if family in self.families) != self.families:
            raise ValueError("alternative_evidence.matter_families_unordered")
        if self.method == MATTER_SELECTION_PRODUCTION and self.families != (
            MATTER_FAMILY_LITIGATION,
        ):
            raise ValueError("alternative_evidence.matter_selection_families_invalid")
        if (
            self.method == MATTER_SELECTION_INTEGRATED
            and self.families != INTEGRATED_FAMILY_SPELLING
        ):
            raise ValueError("alternative_evidence.matter_selection_families_invalid")
        return self

    @property
    def candidate(self) -> bool:
        """Report whether candidate selection was declared.

        Returns:
            Whether the method is CANDIDATE.
        """
        return self.method == MATTER_SELECTION_CANDIDATE

    @property
    def integrated(self) -> bool:
        """Report whether integrated selection was declared.

        Returns:
            Whether the method is INTEGRATED.
        """
        return self.method == MATTER_SELECTION_INTEGRATED

    @property
    def retired(self) -> bool:
        """Report whether the residual policy is retained only for historical readback.

        Whether this policy asks for a residual policy the runtime no
        longer deals (`allocation=CONTEXT_COMPLETE`,
        `residual_search=GAP_DIRECTED`: section V's opt-ins, retired in
        record section X without a supported consumer). A retired policy
        keeps its identity for readback and is refused by name at
        execution (`alternative_evidence.matter_selection_policy_retired`),
        never reinterpreted as the default.
        """
        return self.allocation != "CELL_ROUNDS" or self.residual_search != "QUESTION_BANK"

    @property
    def selection_id(self) -> str:
        """Derive the declared selection identity from method, families and residual policy.

        The selection as one comparable name: the method and its families,
        and every non-default residual policy after a `#`.
        """
        base = f"{self.method}:{','.join(self.families)}"
        extras = [
            *([f"allocation={self.allocation}"] if self.allocation != "CELL_ROUNDS" else []),
            *(
                [f"residual_search={self.residual_search}"]
                if self.residual_search != "QUESTION_BANK"
                else []
            ),
        ]
        return base if not extras else base + "#" + ";".join(extras)


def matter_selection_identity(selection: MatterSelectionPolicy | None) -> str:
    """Return the declared selection identity or the historical default identity.

    The one name a matter selection is compared by everywhere -- the
    request, the admitted policy, the sealed selection index and the
    current-analysis match: the production plan over the litigation family
    when none is named, else the policy's own `selection_id`.
    """
    if selection is None:
        return f"{MATTER_SELECTION_PRODUCTION}:{MATTER_FAMILY_LITIGATION}"
    return selection.selection_id


def matter_selection_retired(selection: MatterSelectionPolicy | None) -> bool:
    """Report whether selection is outside the current integrated policy.

    Whether new work under this selection is refused by name
    (`alternative_evidence.matter_selection_policy_retired`): every
    selection but the integrated one at its current residual policies.
    The production plan (the omitted field) and the candidate needs
    allocation were retired by the first-release integration, the residual
    opt-ins in record section X. What was sealed under any of them reads
    back with its identity; a new first reading or a continuation under
    one is refused before anything is found or opened, and never dealt
    under another method.
    """
    return selection is None or not selection.integrated or selection.retired


class AlternativeEvidenceReadFiling(BaseModel):  # type: ignore[misc]
    """Bind one prior issuer filing read to the publication that recorded it.

    One filing in the window at the cutoff that an earlier current analysis
    read: the filing is the unit of reuse, so it is not read again while it
    stays in the window, and its findings carry with the analysis named here.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    entity_id: str = Field(min_length=1, max_length=32)
    accession: str = Field(min_length=1, max_length=120)
    publication_hash: str = Field(pattern=_HASH)


class AlternativeEvidenceRequest(AlternativeEvidenceContract):
    """Seal issuer scope, evidence clocks, source policy and prior filing reads.

    Attributes:
        kind: Request discriminator.
        ordered_entity_ids: Nonempty normalized unique issuer axis.
        evidence_as_of: Aware cutoff for admissible evidence.
        acquisition_deadline: Aware deadline strictly after the cutoff.
        evidence_classes: Distinct requested evidence classes.
        source_policy: Official-source lookback and acquisition bounds.
        ttl_seconds: Lifetime used to derive publication expiry.
        mode: Recorded or admitted live official acquisition intent.
        matter_selection: Optional sealed matter-selection policy.
        read_filings: Distinct prior filing reads belonging to the requested issuer axis.
        request_hash: Canonical identity excluding this hash.
    """

    kind: Literal["AlternativeEvidenceRequest"] = "AlternativeEvidenceRequest"
    ordered_entity_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence_as_of: datetime
    acquisition_deadline: datetime
    evidence_classes: tuple[AlternativeEvidenceClass, ...] = Field(min_length=1)
    source_policy: AlternativeEvidenceSourcePolicy
    ttl_seconds: int = Field(ge=300, le=604_800)
    mode: AlternativeEvidenceMode
    matter_selection: MatterSelectionPolicy | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    """The matter selection this request asks for; absent (the production
    plan) at the default, so every earlier request keeps its hash."""
    read_filings: tuple[AlternativeEvidenceReadFiling, ...] = Field(
        default=(), max_length=MAXIMUM_READ_FILINGS, exclude_if=lambda value: value == ()
    )
    """The filings of these issuers in the window that an earlier current
    analysis read, by issuer: the selection passes over them and the Analyst
    is told what was found in them, so only what is new is read. Absent from
    the identity when empty, so every earlier request keeps its hash."""
    request_hash: str = Field(pattern=_HASH)

    @property
    def matter_selection_id(self) -> str:
        """Expose the request's declared or historical-default selection identity.

        The selection this request runs, the production plan when none
        is named -- what a sealed selection must match to be reused.
        """
        return matter_selection_identity(self.matter_selection)

    def read_accessions(self, entity_id: str) -> frozenset[str]:
        """The filings of this issuer an earlier analysis read."""
        return frozenset(
            value.accession for value in self.read_filings if value.entity_id == entity_id
        )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_request(self) -> Self:
        """Validate request clocks, issuer/class axes, prior read keys and canonical identity.

        Returns:
            This validated contract.

        Raises:
            ValueError: Clocks, normalized axes, read references or request identity are invalid.
        """
        _aware(self.evidence_as_of, "alternative_evidence.evidence_as_of_invalid")
        _aware(self.acquisition_deadline, "alternative_evidence.acquisition_deadline_invalid")
        if self.acquisition_deadline <= self.evidence_as_of:
            raise ValueError("alternative_evidence.acquisition_window_invalid")
        normalized = tuple(value.strip().upper() for value in self.ordered_entity_ids)
        if normalized != self.ordered_entity_ids or len(set(normalized)) != len(normalized):
            raise ValueError("alternative_evidence.entity_axis_invalid")
        if len(set(self.evidence_classes)) != len(self.evidence_classes):
            raise ValueError("alternative_evidence.evidence_classes_invalid")
        read = [(value.entity_id, value.accession) for value in self.read_filings]
        if len(read) != len(set(read)) or any(
            entity not in self.ordered_entity_ids for entity, _accession in read
        ):
            raise ValueError("alternative_evidence.read_filings_invalid")
        _identity(self, "request_hash")
        return self


class AlternativeEvidenceAdmission(AlternativeEvidenceContract):
    """Host-only permission. It is intentionally absent from model views."""

    kind: Literal["AlternativeEvidenceAdmission"] = "AlternativeEvidenceAdmission"
    request_hash: str = Field(pattern=_HASH)
    network_consent: bool
    admit_live_official: bool
    admit_model_review: bool
    admitted_at: datetime
    admission_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_admission(self) -> Self:
        """Require network consent for admitted live acquisition and verify admission identity.

        Returns:
            This validated contract.

        Raises:
            ValueError: Live official acquisition lacks consent, the admission clock is naive, or
                identity differs.
        """
        _aware(self.admitted_at, "alternative_evidence.admission_clock_invalid")
        if self.admit_live_official and not self.network_consent:
            raise ValueError("alternative_evidence.live_consent_missing")
        _identity(self, "admission_hash")
        return self


class SecIssuerRegistryEntry(AlternativeEvidenceContract):
    """Map one normalized entity and ticker to an official SEC issuer.

    Attributes:
        entity_id: Uppercase research entity identifier.
        ticker: Uppercase listing ticker.
        cik: Ten-digit SEC issuer identifier.
        legal_name: Official issuer legal name.
    """

    entity_id: str = Field(min_length=1, max_length=32)
    ticker: str = Field(min_length=1, max_length=16)
    cik: str = Field(pattern=r"^[0-9]{10}$")
    legal_name: str = Field(min_length=1, max_length=300)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def normalize_entry(self) -> Self:
        """Require entity and ticker values to already be uppercase.

        Returns:
            This validated contract.

        Raises:
            ValueError: The supplied entity identifier or ticker is not normalized.
        """
        if self.entity_id != self.entity_id.upper() or self.ticker != self.ticker.upper():
            raise ValueError("alternative_evidence.registry_entry_not_normalized")
        return self


class SecIssuerRegistrySnapshot(AlternativeEvidenceContract):
    """Seal an official company-ticker registry capture and its source commitment.

    Attributes:
        source_name: Official SEC company-ticker source discriminator.
        captured_at: Aware registry capture clock.
        entries: Nonempty issuer registry entries.
        source_content_hash: Commitment to the acquired source content.
        registry_hash: Canonical snapshot identity excluding this hash.
    """

    kind: Literal["SecIssuerRegistrySnapshot"] = "SecIssuerRegistrySnapshot"
    source_name: Literal["SEC_OFFICIAL_COMPANY_TICKERS"] = "SEC_OFFICIAL_COMPANY_TICKERS"
    captured_at: datetime
    entries: tuple[SecIssuerRegistryEntry, ...] = Field(min_length=1)
    source_content_hash: str = Field(pattern=_HASH)
    registry_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_registry(self) -> Self:
        """Require an aware capture clock, unique ticker/entity keys and registry identity.

        Returns:
            This validated contract.

        Raises:
            ValueError: The clock is naive, registry keys repeat, or canonical identity differs.
        """
        _aware(self.captured_at, "alternative_evidence.registry_clock_invalid")
        tickers = tuple(value.ticker for value in self.entries)
        entities = tuple(value.entity_id for value in self.entries)
        if any(len(set(axis)) != len(axis) for axis in (tickers, entities)):
            raise ValueError("alternative_evidence.registry_mapping_ambiguous")
        _identity(self, "registry_hash")
        return self


class SecCompanyFactPoint(AlternativeEvidenceContract):
    """Carry one bounded SEC company-facts observation with filing provenance.

    Attributes:
        taxonomy: US GAAP taxonomy discriminator.
        concept: Supported financial concept.
        unit: Source-reported unit.
        value: Numeric source observation.
        period_end: Source-reported period-end string.
        filed_on: Source filing-date string.
        form: 10-K or 10-Q filing form.
        accession: SEC filing accession identifier.
    """

    taxonomy: Literal["us-gaap"] = "us-gaap"
    concept: Literal[
        "Assets",
        "Liabilities",
        "Revenues",
        "NetIncomeLoss",
        "CashAndCashEquivalentsAtCarryingValue",
    ]
    unit: str = Field(min_length=1, max_length=20)
    value: float
    period_end: str = Field(min_length=10, max_length=10)
    filed_on: str = Field(min_length=10, max_length=10)
    form: Literal["10-K", "10-Q"]
    accession: str = Field(min_length=18, max_length=24)


class SecCompanyFactsSnapshot(AlternativeEvidenceContract):
    """Seal a bounded company-facts capture for one SEC issuer.

    Attributes:
        entity_id: Research issuer identifier.
        cik: Ten-digit SEC issuer identifier.
        captured_at: Aware acquisition clock.
        facts: Observations with distinct concept/unit keys.
        source_content_hash: Commitment to acquired company-facts content.
        companyfacts_hash: Canonical snapshot identity excluding this hash.
    """

    kind: Literal["SecCompanyFactsSnapshot"] = "SecCompanyFactsSnapshot"
    entity_id: str = Field(min_length=1, max_length=32)
    cik: str = Field(pattern=r"^[0-9]{10}$")
    captured_at: datetime
    facts: tuple[SecCompanyFactPoint, ...]
    source_content_hash: str = Field(pattern=_HASH)
    companyfacts_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_companyfacts(self) -> Self:
        """Require an aware capture, unique concept/unit keys and company-facts identity.

        Returns:
            This validated contract.

        Raises:
            ValueError: The capture clock is naive, fact keys repeat, or identity differs.
        """
        _aware(self.captured_at, "alternative_evidence.companyfacts_clock_invalid")
        keys = tuple((value.concept, value.unit) for value in self.facts)
        if len(keys) != len(set(keys)):
            raise ValueError("alternative_evidence.companyfacts_duplicate")
        _identity(self, "companyfacts_hash")
        return self


class AlternativeEvidenceCitation(AlternativeEvidenceContract):
    """Host evidence identity plus a semantic handle safe for model references."""

    semantic_handle: str = Field(pattern=r"^CIT-[A-Z0-9-]{1,48}$")
    entity_id: str = Field(min_length=1, max_length=32)
    source_name: Literal["SEC_EDGAR", "ISSUER_RECORDED"]
    source_right: Literal["SEC_PUBLIC_OFFICIAL_ACCESS", "USER_PROVIDED_FOR_LOCAL_RESEARCH"]
    evidence_class: AlternativeEvidenceClass
    document_type: str = Field(min_length=1, max_length=40)
    revision: str = Field(min_length=1, max_length=120)
    published_at: datetime | None = None
    accepted_at: datetime | None = None
    available_at: datetime
    excerpt: str = Field(min_length=1, max_length=1600)
    limitations: tuple[str, ...] = ()
    immutable_source: bool
    source_content_hash: str = Field(pattern=_HASH)
    excerpt_binding_hash: str | None = Field(
        default=None, pattern=_HASH, exclude_if=lambda value: value is None
    )
    """The canonicalization binding the excerpt was extracted under: the
    commitment a later request reuses this excerpt on when the same bytes
    (`source_content_hash`) are cited again. Absent from the identity on
    citations sealed before it was recorded, which are never reused."""
    citation_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_citation(self) -> Self:
        """Validate citation clocks, required SEC acceptance and canonical citation identity.

        Returns:
            This validated contract.

        Raises:
            ValueError: A clock is naive, SEC acceptance is absent, or citation identity differs.
        """
        _aware(self.available_at, "alternative_evidence.citation_clock_invalid")
        for value in (self.published_at, self.accepted_at):
            if value is not None:
                _aware(value, "alternative_evidence.citation_clock_invalid")
        if self.source_name == "SEC_EDGAR" and self.accepted_at is None:
            raise ValueError("alternative_evidence.sec_accepted_at_missing")
        _identity(self, "citation_hash")
        return self

    def model_view(self) -> dict[str, object]:
        """Project the citation, provenance, clocks, excerpt and limits for model review.

        Returns:
            Model-facing citation fields without storage or excerpt-binding commitments.
        """
        return {
            "citation_handle": self.semantic_handle,
            "entity_id": self.entity_id,
            "source_name": self.source_name,
            "source_right": self.source_right,
            "evidence_class": self.evidence_class,
            "document_type": self.document_type,
            "revision": self.revision,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "accepted_at": self.accepted_at.isoformat() if self.accepted_at else None,
            "available_at": self.available_at.isoformat(),
            "excerpt": self.excerpt,
            "limitations": self.limitations,
        }


AcquisitionOutcome = Literal["REUSED_LOCAL", "FETCHED", "DEFERRED", "FAILED"]


class AlternativeEvidenceDocumentAcquisition(AlternativeEvidenceContract):
    """What happened to one selected resource of a live acquisition.

    The resource is `(source, issuer CIK, accession, document name)`; the
    outcome says whether its bytes were reused from a verified local object
    without a request, fetched from the official locator, deferred (oversize,
    outside the cap or the policy) or failed, with the reason named. Nothing
    here is a judgement about the filing.
    """

    entity_id: str = Field(min_length=1, max_length=32)
    cik: str = Field(pattern=r"^[0-9]{10}$")
    accession: str = Field(min_length=18, max_length=24)
    document_name: str = Field(min_length=1, max_length=300)
    form: str = Field(min_length=1, max_length=40)
    outcome: AcquisitionOutcome
    detail: str = Field(default="", max_length=400)
    content_bytes: int = Field(default=0, ge=0)


class AlternativeEvidenceAcquisitionAccounting(AlternativeEvidenceContract):
    """Record source traffic and document acquisition accounting separately.

    A live acquisition's traffic and outcomes, metadata apart from bodies,
    logical references apart from physical bytes.
    """

    inventory_request_count: int = Field(ge=0)
    """Official issuer inventories read, the older history shards among them."""
    history_shard_request_count: int = Field(default=0, ge=0)
    body_request_count: int = Field(ge=0)
    """Filing bodies requested, whether or not the request succeeded."""
    fetched_count: int = Field(ge=0)
    reused_local_count: int = Field(ge=0)
    deferred_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    fetched_bytes: int = Field(ge=0)
    reused_bytes: int = Field(ge=0)
    documents: tuple[AlternativeEvidenceDocumentAcquisition, ...]


class AlternativeEvidenceSnapshot(AlternativeEvidenceContract):
    """Seal source coverage, admitted citations and the publication expiry.

    Attributes:
        request_hash: Request identity whose scope this snapshot answers.
        registry_hash: Issuer registry commitment.
        acquisition_binding_hash: Installed acquisition implementation binding.
        status: Coverage state derived from the source counts.
        expected_source_count: Positive expected source count.
        available_source_count: Count of available sources.
        missing_source_count: Count of absent sources.
        failed_source_count: Count of failed acquisitions.
        citations: Uniquely handled admitted citations.
        limitations: Explicit coverage qualifications.
        published_at: Aware publication clock.
        expires_at: Aware expiry strictly after publication.
        acquisition: Optional acquisition traffic accounting, omitted when absent.
        snapshot_hash: Canonical snapshot identity excluding this hash.
    """

    kind: Literal["AlternativeEvidenceSnapshot"] = "AlternativeEvidenceSnapshot"
    request_hash: str = Field(pattern=_HASH)
    registry_hash: str = Field(pattern=_HASH)
    acquisition_binding_hash: str = Field(pattern=_HASH)
    status: AlternativeEvidenceSnapshotStatus
    expected_source_count: int = Field(ge=1)
    available_source_count: int = Field(ge=0)
    missing_source_count: int = Field(ge=0)
    failed_source_count: int = Field(ge=0)
    citations: tuple[AlternativeEvidenceCitation, ...]
    limitations: tuple[str, ...]
    published_at: datetime
    expires_at: datetime
    acquisition: AlternativeEvidenceAcquisitionAccounting | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    """The live acquisition's per-resource outcomes and traffic; absent for a
    recorded snapshot and for snapshots sealed before it was recorded."""
    snapshot_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_snapshot(self) -> Self:
        """Validate source counts/status, citation handles, publication clocks and identity.

        Returns:
            This validated contract.

        Raises:
            ValueError: Clocks, source totals, status, citation handles or canonical identity are
                invalid.
        """
        _aware(self.published_at, "alternative_evidence.snapshot_clock_invalid")
        _aware(self.expires_at, "alternative_evidence.snapshot_clock_invalid")
        if self.expires_at <= self.published_at:
            raise ValueError("alternative_evidence.snapshot_ttl_invalid")
        if self.available_source_count + self.missing_source_count + self.failed_source_count != (
            self.expected_source_count
        ):
            raise ValueError("alternative_evidence.snapshot_counts_invalid")
        handles = tuple(value.semantic_handle for value in self.citations)
        if len(handles) != len(set(handles)):
            raise ValueError("alternative_evidence.citation_handle_duplicate")
        if self.available_source_count == self.expected_source_count:
            expected = AlternativeEvidenceSnapshotStatus.COMPLETE
        elif self.available_source_count > 0:
            expected = AlternativeEvidenceSnapshotStatus.PARTIAL
        elif self.failed_source_count > 0:
            expected = AlternativeEvidenceSnapshotStatus.UNAVAILABLE
        else:
            expected = AlternativeEvidenceSnapshotStatus.EMPTY
        if self.status is not expected:
            raise ValueError("alternative_evidence.snapshot_status_invalid")
        _identity(self, "snapshot_hash")
        return self


def seal_contract[ContractT: AlternativeEvidenceContract](
    model: type[ContractT], identity_field: str, /, **values: object
) -> ContractT:
    """Compute a contract identity and construct the validated immutable model.

    Args:
        model: Contract model class to validate.
        identity_field: Field receiving the canonical identity.
        values: Contract field values supplied to the sealing owner.

    Returns:
        The sealed and validated contract.
    """
    return seal_model_validated(model, identity_field, **values)


def validate_contract_identity(value: AlternativeEvidenceContract, field: str) -> None:
    """Validate one content-addressed contract without exposing its storage identity."""
    _identity(value, field)


def snapshot_expiry(request: AlternativeEvidenceRequest, published_at: datetime) -> datetime:
    """Add the requested lifetime to the supplied publication clock.

    Args:
        request: Request declaring the lifetime in seconds.
        published_at: Publication clock to advance.

    Returns:
        Publication clock plus the request TTL.
    """
    return published_at + timedelta(seconds=request.ttl_seconds)


def _aware(value: datetime, code: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(code)


def _identity(value: AlternativeEvidenceContract, field: str) -> None:
    expected = canonical_hash(value.model_dump(mode="json", exclude={field}))
    if getattr(value, field) != expected:
        raise ValueError("alternative_evidence.identity_invalid")


__all__ = [
    "DEFAULT_SOURCE_DOCUMENT_BYTES",
    "MATTER_FAMILY_CORPORATE_EVENT",
    "MATTER_FAMILY_FINANCING",
    "MATTER_FAMILY_LITIGATION",
    "MATTER_SELECTION_CANDIDATE",
    "MATTER_SELECTION_INTEGRATED",
    "MATTER_SELECTION_PRODUCTION",
    "MAXIMUM_READ_FILINGS",
    "MAXIMUM_SOURCE_DOCUMENT_BYTES",
    "SOURCE_FAMILY_BY_EVIDENCE_CLASS",
    "WHOLE_FILING_BYTES",
    "AcquisitionOutcome",
    "AlternativeEvidenceAcquisitionAccounting",
    "AlternativeEvidenceAdmission",
    "AlternativeEvidenceCitation",
    "AlternativeEvidenceClass",
    "AlternativeEvidenceContract",
    "AlternativeEvidenceDocumentAcquisition",
    "AlternativeEvidenceMode",
    "AlternativeEvidenceReadFiling",
    "AlternativeEvidenceReadingDepth",
    "AlternativeEvidenceRequest",
    "AlternativeEvidenceReviewStatus",
    "AlternativeEvidenceSnapshot",
    "AlternativeEvidenceSnapshotStatus",
    "AlternativeEvidenceSourcePolicy",
    "MatterSelectionPolicy",
    "PublishedPrecision",
    "SecCompanyFactPoint",
    "SecCompanyFactsSnapshot",
    "SecIssuerRegistryEntry",
    "SecIssuerRegistrySnapshot",
    "matter_selection_identity",
    "matter_selection_retired",
    "seal_contract",
    "snapshot_expiry",
    "validate_contract_identity",
]
