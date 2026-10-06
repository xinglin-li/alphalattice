"""Typed source payloads before canonical document admission."""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.kernel.shared_kernel.identity import canonical_hash

from ..contracts import (
    MAXIMUM_SOURCE_DOCUMENT_BYTES,
    AlternativeEvidenceClass,
    AlternativeEvidenceContract,
    PublishedPrecision,
    validate_contract_identity,
)

_HASH = r"^[0-9a-f]{64}$"


def _validate_temporal_fields(
    *,
    published_at: datetime | None,
    published_precision: str | None,
    report_period_end: date | None,
    retrieved_at: datetime | None,
    code: str,
) -> None:
    if published_precision is not None and published_at is None:
        raise ValueError(f"{code}_precision_without_publication")
    if retrieved_at is not None and (
        retrieved_at.tzinfo is None or retrieved_at.utcoffset() is None
    ):
        raise ValueError(f"{code}_clock_invalid")
    if (
        published_at is not None
        and published_precision == "DATE"
        and (
            published_at.hour
            or published_at.minute
            or published_at.second
            or published_at.microsecond
        )
    ):
        raise ValueError(f"{code}_date_precision_not_midnight")
    del report_period_end


class AcquiredEvidenceDocument(BaseModel):  # type: ignore[misc]
    """Bounded source bytes and provenance; never exposed to the model."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        ser_json_bytes="base64",
        val_json_bytes="base64",
    )

    semantic_handle: str = Field(pattern=r"^DOC-[A-Z0-9-]{1,48}$")
    entity_id: str = Field(min_length=1, max_length=32)
    source_name: Literal["SEC_EDGAR", "ISSUER_RECORDED"]
    source_right: Literal[
        "SEC_PUBLIC_OFFICIAL_ACCESS",
        "USER_PROVIDED_FOR_LOCAL_RESEARCH",
    ]
    evidence_class: AlternativeEvidenceClass
    document_type: str = Field(min_length=1, max_length=40)
    revision: str = Field(min_length=1, max_length=120)
    title: str = Field(min_length=1, max_length=300)
    published_at: datetime | None = None
    accepted_at: datetime | None = None
    available_at: datetime
    media_type: Literal["text/html", "text/plain", "text/markdown"]
    content: bytes = Field(min_length=1)
    immutable_source: bool
    limitations: tuple[str, ...] = ()
    source_content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    # Time is evidence metadata: each meaning is kept apart and absent from
    # the identity when unknown, so documents acquired before these were
    # recorded keep their references' hashes.
    published_precision: PublishedPrecision | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    report_period_end: date | None = Field(default=None, exclude_if=lambda value: value is None)
    """The period the filing reports on, as the official submissions metadata
    states it (SEC `reportDate`); distinct from the filing date, the acceptance
    instant and any date the text states."""
    retrieved_at: datetime | None = Field(default=None, exclude_if=lambda value: value is None)
    """When this system obtained the bytes -- the response's own clock. Never a
    disclosure time: a later retrieval of an old filing does not change its age."""
    source_cik: str | None = Field(default=None, exclude_if=lambda value: value is None)
    source_document_name: str | None = Field(
        default=None, max_length=300, exclude_if=lambda value: value is None
    )
    """The resource identity the bytes were acquired under -- the issuer's
    CIK and the document name inside the accession -- so a later scope can
    find the retained bytes before asking the source again. Absent on
    documents acquired before it was recorded."""

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_document(self) -> Self:
        """Validate source clocks, SEC acceptance, resource metadata and acquired-content identity.

        Returns:
            This validated contract.

        Raises:
            ValueError: CIK, temporal metadata, required SEC acceptance or source-content commitment
                is invalid.
        """
        if self.source_cik is not None and not re.fullmatch(r"[0-9]{10}", self.source_cik):
            raise ValueError("alternative_evidence.source_cik_invalid")
        for value in (self.published_at, self.accepted_at, self.available_at):
            if value is not None and (value.tzinfo is None or value.utcoffset() is None):
                raise ValueError("alternative_evidence.source_document_clock_invalid")
        if self.source_name == "SEC_EDGAR" and self.accepted_at is None:
            raise ValueError("alternative_evidence.sec_accepted_at_missing")
        _validate_temporal_fields(
            published_at=self.published_at,
            published_precision=self.published_precision,
            report_period_end=self.report_period_end,
            retrieved_at=self.retrieved_at,
            code="alternative_evidence.source_document",
        )
        expected = canonical_hash(
            {
                "source_name": self.source_name,
                "revision": self.revision,
                "media_type": self.media_type,
                "content_hex": self.content.hex(),
            }
        )
        if self.source_content_hash != expected:
            raise ValueError("alternative_evidence.source_document_hash_invalid")
        return self


class AcquiredEvidenceDocumentSet(AlternativeEvidenceContract):
    """The retained inline form: every request's set carried its documents' bytes.

    Read for the Tasks and analyses sealed under it; no new set is written this
    way. `AcquiredEvidenceSourceReferenceSet` is the active writer.
    """

    kind: Literal["AcquiredEvidenceDocumentSet"] = "AcquiredEvidenceDocumentSet"
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    documents: tuple[AcquiredEvidenceDocument, ...]
    acquired_at: datetime
    source_set_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_set(self) -> Self:
        """Require unique acquired document handles, an aware acquisition clock and set identity.

        Returns:
            This validated contract.

        Raises:
            ValueError: Document handles repeat, the acquisition clock is naive, or source-set
                identity differs.
        """
        handles = tuple(value.semantic_handle for value in self.documents)
        if len(handles) != len(set(handles)):
            raise ValueError("alternative_evidence.source_document_handle_duplicate")
        if self.acquired_at.tzinfo is None or self.acquired_at.utcoffset() is None:
            raise ValueError("alternative_evidence.source_document_set_clock_invalid")
        validate_contract_identity(self, "source_set_hash")
        return self


SOURCE_OBJECT_ROOT = ".system/source-objects"
"""Where the evidence Workspace keeps original source bytes, one object per
distinct content, addressed by the SHA-256 of the bytes."""


def source_object_relative(content_sha256: str) -> str:
    return f"{SOURCE_OBJECT_ROOT}/{content_sha256}"


class AcquiredEvidenceDocumentReference(BaseModel):  # type: ignore[misc]
    """One acquired document's provenance, naming its original bytes by address.

    Everything `AcquiredEvidenceDocument` records except the bytes themselves,
    which live once in the evidence Workspace's source-object store; a set
    that references a filing already held adds its provenance only. The
    `source_content_hash` is the one the acquisition computed over the bytes
    and is verified again when the bytes are resolved.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    semantic_handle: str = Field(pattern=r"^DOC-[A-Z0-9-]{1,48}$")
    entity_id: str = Field(min_length=1, max_length=32)
    source_name: Literal["SEC_EDGAR", "ISSUER_RECORDED"]
    source_right: Literal[
        "SEC_PUBLIC_OFFICIAL_ACCESS",
        "USER_PROVIDED_FOR_LOCAL_RESEARCH",
    ]
    evidence_class: AlternativeEvidenceClass
    document_type: str = Field(min_length=1, max_length=40)
    revision: str = Field(min_length=1, max_length=120)
    title: str = Field(min_length=1, max_length=300)
    published_at: datetime | None = None
    accepted_at: datetime | None = None
    available_at: datetime
    media_type: Literal["text/html", "text/plain", "text/markdown"]
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_bytes: int = Field(gt=0)
    content_object_path: str = Field(min_length=1)
    immutable_source: bool
    limitations: tuple[str, ...] = ()
    source_content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    published_precision: PublishedPrecision | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    report_period_end: date | None = Field(default=None, exclude_if=lambda value: value is None)
    retrieved_at: datetime | None = Field(default=None, exclude_if=lambda value: value is None)
    source_cik: str | None = Field(default=None, exclude_if=lambda value: value is None)
    source_document_name: str | None = Field(
        default=None, max_length=300, exclude_if=lambda value: value is None
    )

    @property
    def resource_key(self) -> tuple[str, str | None, str, str | None]:
        """Return the source resource key retained with the acquired document reference.

        `(source, CIK, accession, document name)`; the CIK and the name are
        None on references sealed before they were recorded.
        """
        return (self.source_name, self.source_cik, self.revision, self.source_document_name)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_reference(self) -> Self:
        if self.source_cik is not None and not re.fullmatch(r"[0-9]{10}", self.source_cik):
            raise ValueError("alternative_evidence.source_cik_invalid")
        for value in (self.published_at, self.accepted_at, self.available_at):
            if value is not None and (value.tzinfo is None or value.utcoffset() is None):
                raise ValueError("alternative_evidence.source_document_clock_invalid")
        if self.source_name == "SEC_EDGAR" and self.accepted_at is None:
            raise ValueError("alternative_evidence.sec_accepted_at_missing")
        _validate_temporal_fields(
            published_at=self.published_at,
            published_precision=self.published_precision,
            report_period_end=self.report_period_end,
            retrieved_at=self.retrieved_at,
            code="alternative_evidence.source_document",
        )
        if self.content_object_path != source_object_relative(self.content_sha256):
            raise ValueError("alternative_evidence.source_object_path_not_content_addressed")
        return self

    @classmethod
    def of(cls, document: AcquiredEvidenceDocument, *, content_sha256: str) -> Self:
        return cls(
            **document.model_dump(mode="python", exclude={"content"}),
            content_sha256=content_sha256,
            content_bytes=len(document.content),
            content_object_path=source_object_relative(content_sha256),
        )

    def resolve(self, content: bytes) -> AcquiredEvidenceDocument:
        """The document with its bytes; the acquisition's own hash proves them."""
        return AcquiredEvidenceDocument(
            **self.model_dump(
                mode="python",
                exclude={"content_sha256", "content_bytes", "content_object_path"},
            ),
            content=content,
        )


class AcquiredEvidenceSourceReferenceSet(AlternativeEvidenceContract):
    """One request's acquired documents, by reference to their retained bytes."""

    kind: Literal["AcquiredEvidenceSourceReferenceSet"] = "AcquiredEvidenceSourceReferenceSet"
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    documents: tuple[AcquiredEvidenceDocumentReference, ...]
    acquired_at: datetime
    source_set_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_set(self) -> Self:
        handles = tuple(value.semantic_handle for value in self.documents)
        if len(handles) != len(set(handles)):
            raise ValueError("alternative_evidence.source_document_handle_duplicate")
        if self.acquired_at.tzinfo is None or self.acquired_at.utcoffset() is None:
            raise ValueError("alternative_evidence.source_document_set_clock_invalid")
        validate_contract_identity(self, "source_set_hash")
        return self


SOURCE_COMMIT_CATEGORY = "source-document-commits"
"""The artifact category one body's commit is sealed under."""

SOURCE_DEFERRAL_CATEGORY = "source-document-deferrals"
"""The artifact category a resource's bounded deferral is sealed under."""


class AcquiredEvidenceSourceDeferral(AlternativeEvidenceContract):
    """Seal a bounded source-resource deferral and its recheck conditions.

    One resource's bounded deferral, sealed on its own: the exact source
    identity (issuer CIK, accession, document name), the inventory metadata
    it was selected under, the reason observed, the cap it was observed
    under, how much of it was read before the transfer was abandoned, and
    when. A later request under a cap no larger than the observed one does
    not transfer the resource again -- a larger cap or an explicit accession
    scope does, once; the resource is reported deferred by this record's
    name. Rebuildable into the lookup
    with the sets and commits; it asserts nothing about the filing's
    content and is never a substitute for it.
    """

    kind: Literal["AcquiredEvidenceSourceDeferral"] = "AcquiredEvidenceSourceDeferral"
    source_name: Literal["SEC_EDGAR"] = "SEC_EDGAR"
    source_cik: str = Field(pattern=r"^[0-9]{10}$")
    accession: str = Field(min_length=18, max_length=24)
    document_name: str = Field(min_length=1, max_length=300)
    form: str = Field(min_length=1, max_length=40)
    accepted_at: datetime
    inventory_size_bytes: int | None = Field(default=None, ge=0)
    reason: Literal["OVERSIZE"] = "OVERSIZE"
    admitted_cap_bytes: int = Field(ge=1_000)
    observed_bytes: int = Field(ge=0)
    """The bytes the source declared or the transfer read before it was
    abandoned at the cap: a lower bound of the body's size, never its size."""
    observed_at: datetime
    recheck_after: datetime | None = Field(default=None, exclude_if=lambda value: value is None)
    """The time-based recheck of records sealed before W6 (2026-09-24), read
    back as sealed: under the 30-day window the filing left it before the
    recheck was due, so no request is admitted by time; absent since."""
    deferral_hash: str = Field(pattern=_HASH)

    @property
    def resource_key(self) -> tuple[str, str, str, str]:
        """Return the exact resource key retained by this deferral.

        Returns:
            Source name, CIK, accession and document name in that order.
        """
        return (self.source_name, self.source_cik, self.accession, self.document_name)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_deferral(self) -> Self:
        """Require aware deferral clocks, a later recheck when present, and canonical identity.

        Returns:
            This validated contract.

        Raises:
            ValueError: A clock is naive, recheck is not later than observation, or deferral
                identity differs.
        """
        for value in (self.accepted_at, self.observed_at, self.recheck_after):
            if value is not None and (value.tzinfo is None or value.utcoffset() is None):
                raise ValueError("alternative_evidence.source_deferral_clock_invalid")
        if self.recheck_after is not None and self.recheck_after <= self.observed_at:
            raise ValueError("alternative_evidence.source_deferral_recheck_invalid")
        validate_contract_identity(self, "deferral_hash")
        return self


class AcquiredEvidenceSourceCommit(AlternativeEvidenceContract):
    """Seal the durable commit of one acquired source body.

    One acquired body's durable commit, sealed on its own before the
    request's set exists: the reference to the retained bytes by content
    address with its provenance and resource identity, so a failure after it
    -- the next file, the next issuer, a cancellation -- discards nothing
    already obtained, and a later scope finds the body without asking the
    source again. Rebuildable into the lookup with the source sets; it
    confers no authority the bytes' own verification does not.
    """

    kind: Literal["AcquiredEvidenceSourceCommit"] = "AcquiredEvidenceSourceCommit"
    reference: AcquiredEvidenceDocumentReference
    committed_at: datetime
    commit_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_commit(self) -> Self:
        """Require an aware durable-commit clock and canonical source-commit identity.

        Returns:
            This validated contract.

        Raises:
            ValueError: The commit clock is naive or canonical commit identity differs.
        """
        if self.committed_at.tzinfo is None or self.committed_at.utcoffset() is None:
            raise ValueError("alternative_evidence.source_commit_clock_invalid")
        validate_contract_identity(self, "commit_hash")
        return self


AcquiredEvidenceSourceSet = AcquiredEvidenceDocumentSet | AcquiredEvidenceSourceReferenceSet
"""Either durable form of a request's acquired documents."""


def parse_source_set(payload: dict[str, object]) -> AcquiredEvidenceSourceSet:
    """Read a source set in either supported durable form, by its own `kind`."""
    kind = payload.get("kind")
    if kind == "AcquiredEvidenceSourceReferenceSet":
        return cast(
            AcquiredEvidenceSourceReferenceSet,
            AcquiredEvidenceSourceReferenceSet.model_validate(payload),
        )
    if kind == "AcquiredEvidenceDocumentSet":
        return cast(
            AcquiredEvidenceDocumentSet, AcquiredEvidenceDocumentSet.model_validate(payload)
        )
    raise ValueError("alternative_evidence.source_set_kind_unknown")


SecMaterialForm = Literal[
    "8-K",
    "8-K/A",
    "10-Q",
    "10-Q/A",
    "10-K",
    "10-K/A",
    "NT 10-K",
    "NT 10-K/A",
    "NT 10-Q",
    "NT 10-Q/A",
]
"""The forms the selection reads: the baselines, the current reports, and the
late-filing notices (Form 12b-25), which are events beside the current reports."""


class SecFilingInventoryEntry(AlternativeEvidenceContract):
    """One cutoff-valid material filing from the official submissions endpoint."""

    entity_id: str = Field(min_length=1, max_length=32)
    cik: str = Field(pattern=r"^[0-9]{10}$")
    accession: str = Field(min_length=18, max_length=24)
    form: SecMaterialForm
    report_date: date | None = None
    filed_on: date
    accepted_at: datetime
    primary_document: str = Field(min_length=1, max_length=300)
    size_bytes: int | None = Field(default=None, ge=0, exclude_if=lambda value: value is None)
    """The official inventory's stated size of the filing, when it states one:
    a capacity estimate before any body is fetched, never proof of the bytes."""
    amendment_of_accession: str | None = Field(default=None, min_length=18, max_length=24)
    amendment_relation: Literal["NOT_APPLICABLE", "LINKED", "ORIGINAL_NOT_IN_RECENT_SUBMISSIONS"]
    items: tuple[str, ...] = Field(default=(), max_length=16, exclude_if=lambda value: value == ())
    """A current report's item numbers as the official index states them (`2.02`,
    `9.01`); empty for every other form, and kept out of the sealed form when
    empty, so a plan sealed before items were recorded reads back as sealed."""

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_entry(self) -> Self:
        """Validate aware acceptance, 8-K item labels and accession amendment status.

        Returns:
            This validated contract.

        Raises:
            ValueError: The acceptance clock, item labels or amendment/parent status is invalid.
        """
        if self.accepted_at.tzinfo is None or self.accepted_at.utcoffset() is None:
            raise ValueError("alternative_evidence.inventory_clock_invalid")
        if self.items and (
            self.form.removesuffix("/A") != "8-K"
            or any(re.fullmatch(r"[1-9]\.[0-9]{2}", item) is None for item in self.items)
        ):
            raise ValueError("alternative_evidence.inventory_items_invalid")
        is_amendment = self.form.endswith("/A")
        expected = (
            "LINKED"
            if is_amendment and self.amendment_of_accession is not None
            else "ORIGINAL_NOT_IN_RECENT_SUBMISSIONS"
            if is_amendment
            else "NOT_APPLICABLE"
        )
        if self.amendment_relation != expected:
            raise ValueError("alternative_evidence.inventory_amendment_binding_invalid")
        return self


class SecFilingSelectionDeferral(BaseModel):  # type: ignore[misc]
    """One discovered filing this plan did not select, with the reason it names."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    accession: str = Field(min_length=18, max_length=24)
    form: SecMaterialForm
    accepted_at: datetime
    reason: Literal[
        "BEYOND_CAPACITY",
        "BEYOND_UNIT_CAPACITY",
        "OUTSIDE_EVENT_WINDOW",
        "SUPERSEDED_BASELINE",
        "OUTSIDE_ACCESSION_SCOPE",
        "READ_EARLIER",
    ]
    """`BEYOND_CAPACITY`: beyond the issuer's own selection capacity (the
    policy budget); `BEYOND_UNIT_CAPACITY`: selected by the issuer's own
    plan and deferred because the unit's selections together exceed the
    admitted document set -- a packing outcome, named as such (2026-09-22),
    never the issuer's plan silently cut; `READ_EARLIER`: in the window and
    read by an earlier current analysis, whose findings carry (W3)."""


class SecFilingSelectionPlan(AlternativeEvidenceContract):
    """What one issuer's cutoff-valid SEC inventory holds and what will be fetched.

    Planned before any filing is acquired: the filings accepted inside the
    window -- major negatives first, then periodic reports, then the other
    current reports, each newest first -- as far as the capacity reaches.
    (Plans sealed under the retired 90-day policy took the latest 10-K and 10-Q
    families first as required baselines, and read as sealed.) The capacity is the smaller of
    the policy's per-issuer budget and the admitted document set's share for
    this request's issuers, so nothing is fetched that canonicalization would
    reject for capacity. Every discovered filing is either selected or listed
    as deferred with its reason; superseded baselines and events outside the
    window are counted. A capacity that cannot hold the required baselines is
    refused by name at planning, never met by dropping a baseline.

    Under an accession scope -- named originals to re-acquire, as for a
    historical repair -- the plan selects exactly the scoped accessions the
    issuer's own official submissions index holds under the cutoff, in their
    accepted order, defers every other discovered filing as outside the
    scope, and names a scoped accession the index does not hold: ownership,
    form and primary document come from the index, never from the accession
    prefix.
    """

    kind: Literal["SecFilingSelectionPlan"] = "SecFilingSelectionPlan"
    entity_id: str = Field(min_length=1, max_length=32)
    cik: str = Field(pattern=r"^[0-9]{10}$")
    evidence_as_of: datetime
    accession_scope: tuple[str, ...] = Field(
        default=(), max_length=4, exclude_if=lambda value: value == ()
    )
    """Absent from the identity when the plan is the selector's own, so plans
    sealed before scopes existed keep their hashes."""
    event_window_days: int = Field(ge=1)
    policy_budget: int = Field(ge=1)
    """The source policy's `maximum_documents_per_issuer`."""
    unit_capacity: int = Field(ge=1)
    """The admitted document set's capacity this issuer's plan is bounded by.
    Since 2026-09-22 the whole set (`ADMITTED_DOCUMENT_CAPACITY`): an
    issuer's selection is its own, and the unit's issuers share the set by
    their selections, never by division (`apply_unit_capacity`); plans
    sealed before then carry the share `24 // issuers` they were planned
    under and read as sealed."""
    capacity: int = Field(ge=1)
    discovered_count: int = Field(ge=0)
    required: tuple[SecFilingInventoryEntry, ...] = Field(max_length=4)
    events: tuple[SecFilingInventoryEntry, ...]
    deferred: tuple[SecFilingSelectionDeferral, ...]
    superseded_baseline_count: int = Field(ge=0)
    outside_window_event_count: int = Field(ge=0)
    limitations: tuple[str, ...]
    plan_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_plan(self) -> Self:
        """Validate capacity, inventory accounting, accession scope and plan identity.

        Returns:
            This validated contract.

        Raises:
            ValueError: Clock, capacity, accession/resource scope, discovery accounting or canonical
                identity disagrees.
        """
        if self.evidence_as_of.tzinfo is None or self.evidence_as_of.utcoffset() is None:
            raise ValueError("alternative_evidence.selection_plan_clock_invalid")
        if self.capacity != min(self.policy_budget, self.unit_capacity):
            raise ValueError("alternative_evidence.selection_plan_capacity_invalid")
        if len(self.required) + len(self.events) > self.capacity:
            raise ValueError("alternative_evidence.selection_plan_over_capacity")
        accessions = [value.accession for value in (*self.required, *self.events)]
        accessions.extend(value.accession for value in self.deferred)
        if len(accessions) != len(set(accessions)):
            raise ValueError("alternative_evidence.selection_plan_accession_duplicate")
        accounted = len(accessions) + self.superseded_baseline_count
        accounted += self.outside_window_event_count
        if accounted != self.discovered_count:
            raise ValueError("alternative_evidence.selection_plan_accounting_invalid")
        if any(
            value.entity_id != self.entity_id or value.cik != self.cik
            for value in (*self.required, *self.events)
        ):
            raise ValueError("alternative_evidence.selection_plan_issuer_invalid")
        if self.accession_scope:
            if len(set(self.accession_scope)) != len(self.accession_scope):
                raise ValueError("alternative_evidence.selection_plan_scope_duplicate")
            if self.events or any(
                value.accession not in self.accession_scope for value in self.required
            ):
                raise ValueError("alternative_evidence.selection_plan_outside_scope")
        validate_contract_identity(self, "plan_hash")
        return self

    @property
    def selected(self) -> tuple[SecFilingInventoryEntry, ...]:
        """Return selected filings in their declared acquisition order.

        The accessions to fetch, in acquisition order: baselines oldest first,
        then the events newest first.
        """
        return (*self.required, *self.events)

    @property
    def nothing_filed(self) -> bool:
        """Nothing accepted in the window: the issuer's own plan selected and deferred no filing.

        The quiet holding V541 names NOTHING_FILED, which never counts against a coverage
        floor (V587). A plan scoped to named accessions reads those, not the window, so it
        never says so.
        """
        return not self.accession_scope and not self.selected and not self.deferred


def inventory_read_key(
    *,
    entity_id: str,
    evidence_as_of: datetime,
    event_window_days: int,
    policy_budget: int,
    unit_capacity: int,
) -> str:
    """Derive the issuer/cutoff/policy identity answered by an inventory read.

    What an inventory read answers, as its identity: one issuer at one cutoff
    under one window, budget and capacity -- never what the index held.
    """
    return str(
        canonical_hash(
            {
                "kind": "SecFilingInventoryRead",
                "entity_id": entity_id,
                "evidence_as_of": evidence_as_of.isoformat(),
                "event_window_days": event_window_days,
                "policy_budget": policy_budget,
                "unit_capacity": unit_capacity,
            }
        )
    )


class SecFilingInventoryRead(AlternativeEvidenceContract):
    """One issuer's filing index, read at a cutoff before its unit is packed.

    The plan it holds is the issuer's own selection under the window and the
    budget: the packing counts it (an empty plan is an issuer that filed
    nothing in the window), and the unit's acquisition takes it, so the index
    is read once per cutoff and what was packed is what is fetched. Found by
    what it answers (`inventory_read_key`), not by what the index held.
    """

    kind: Literal["SecFilingInventoryRead"] = "SecFilingInventoryRead"
    plan: SecFilingSelectionPlan
    read_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_read(self) -> Self:
        """Require a pure inventory read to use its exact issuer/cutoff/policy key as identity.

        Returns:
            This validated contract.

        Raises:
            ValueError: The read contains accession selections or its inventory key/hash differs.
        """
        plan = self.plan
        expected = inventory_read_key(
            entity_id=plan.entity_id,
            evidence_as_of=plan.evidence_as_of,
            event_window_days=plan.event_window_days,
            policy_budget=plan.policy_budget,
            unit_capacity=plan.unit_capacity,
        )
        if plan.accession_scope or plan.required or self.read_hash != expected:
            raise ValueError("alternative_evidence.inventory_read_invalid")
        return self

    @property
    def nothing_filed(self) -> bool:
        """Nothing accepted in the window: its plan's own `nothing_filed`."""
        return self.plan.nothing_filed


__all__ = [
    "MAXIMUM_SOURCE_DOCUMENT_BYTES",
    "SOURCE_COMMIT_CATEGORY",
    "SOURCE_DEFERRAL_CATEGORY",
    "AcquiredEvidenceDocument",
    "AcquiredEvidenceDocumentSet",
    "AcquiredEvidenceSourceCommit",
    "AcquiredEvidenceSourceDeferral",
    "PublishedPrecision",
    "SecFilingInventoryEntry",
    "SecFilingInventoryRead",
    "SecFilingSelectionDeferral",
    "SecFilingSelectionPlan",
    "inventory_read_key",
]
