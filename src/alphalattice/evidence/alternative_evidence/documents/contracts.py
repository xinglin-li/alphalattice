"""Canonical document and Workspace publication contracts."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, model_validator

from ..contracts import (
    AlternativeEvidenceClass,
    AlternativeEvidenceContract,
    AlternativeEvidenceReadingDepth,
    PublishedPrecision,
    seal_contract,
    validate_contract_identity,
)


class DocumentRejectionCode(StrEnum):
    """Identify the deterministic reason a source document was not admitted.

    Codes distinguish content, decoding, media type, source identity, causal cutoff,
    rights, size and extraction failures, including unavailable section extraction.
    """

    EMPTY = "EMPTY"
    INVALID_ENCODING = "INVALID_ENCODING"
    INVALID_MEDIA_TYPE = "INVALID_MEDIA_TYPE"
    BOILERPLATE_ONLY = "BOILERPLATE_ONLY"
    SOURCE_IDENTITY_INVALID = "SOURCE_IDENTITY_INVALID"
    CUTOFF_VIOLATION = "CUTOFF_VIOLATION"
    RIGHTS_UNAVAILABLE = "RIGHTS_UNAVAILABLE"
    SIZE_LIMIT_EXCEEDED = "SIZE_LIMIT_EXCEEDED"
    EXTRACTION_FAILED = "EXTRACTION_FAILED"
    SECTION_EXTRACTION_UNAVAILABLE = "SECTION_EXTRACTION_UNAVAILABLE"


class AlternativeEvidenceCanonicalDocument(AlternativeEvidenceContract):
    """Transient admitted Markdown passed directly to the Workspace owner."""

    semantic_handle: str = Field(pattern=r"^DOC-[A-Z0-9-]{1,48}$")
    entity_id: str = Field(min_length=1, max_length=32)
    source_name: str = Field(min_length=1, max_length=40)
    source_right: str = Field(min_length=1, max_length=80)
    evidence_class: AlternativeEvidenceClass
    document_type: str = Field(min_length=1, max_length=40)
    revision: str = Field(min_length=1, max_length=120)
    title: str = Field(min_length=1, max_length=300)
    published_at: datetime | None = None
    accepted_at: datetime | None = None
    available_at: datetime
    immutable_source: bool
    limitations: tuple[str, ...] = ()
    reading_depth: AlternativeEvidenceReadingDepth = AlternativeEvidenceReadingDepth.FULL_FILING
    section_labels: tuple[str, ...] = ()
    published_precision: PublishedPrecision | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    report_period_end: date | None = Field(default=None, exclude_if=lambda value: value is None)
    canonical_markdown: bytes = Field(min_length=1)
    canonical_content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    character_count: int = Field(ge=1)
    byte_count: int = Field(ge=1)
    document_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_document(self) -> Self:
        """Verify nonblank UTF-8 content, exact text/byte counts and document identity.

        Returns:
            This validated contract.

        Raises:
            ValueError: Content cannot decode, is blank, has inconsistent counts, or has a different
                identity.
        """
        if self.byte_count != len(self.canonical_markdown):
            raise ValueError("alternative_evidence.canonical_byte_count_invalid")
        try:
            text = self.canonical_markdown.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("alternative_evidence.canonical_encoding_invalid") from error
        if self.character_count != len(text) or not text.strip():
            raise ValueError("alternative_evidence.canonical_character_count_invalid")
        validate_contract_identity(self, "document_hash")
        return self


class AlternativeEvidenceDocumentRejection(AlternativeEvidenceContract):
    """Retain a bounded admission failure for one semantic document handle.

    Attributes:
        semantic_handle: Rejected DOC handle.
        entity_id: Issuer associated with the source.
        code: Deterministic rejection reason.
        summary: Bounded explanatory admission summary.
    """

    semantic_handle: str = Field(pattern=r"^DOC-[A-Z0-9-]{1,48}$")
    entity_id: str = Field(min_length=1, max_length=32)
    code: DocumentRejectionCode
    summary: str = Field(min_length=1, max_length=300)


class AlternativeEvidenceDocumentReference(AlternativeEvidenceContract):
    """Carry admitted document provenance and an immutable workspace revision reference.

    Attributes:
        semantic_handle: Admitted DOC handle.
        entity_id: Research issuer identifier.
        source_name: SEC or recorded issuer source.
        source_right: Declared source access right.
        evidence_class: Requested evidence class.
        document_type: Source document type.
        revision_label: Source revision label.
        title: Source document title.
        published_at: Optional aware publication clock.
        accepted_at: Optional aware SEC acceptance clock.
        available_at: Aware evidence availability clock.
        immutable_source: Whether source provenance is immutable.
        reading_depth: Full-filing or material-section admission depth.
        section_labels: Admitted section labels.
        workspace_document_id: Immutable workspace document identifier.
        workspace_revision: Positive immutable workspace revision.
        character_count: Positive admitted text length.
        byte_count: Positive admitted byte length.
        published_precision: Optional precision basis for the publication clock.
        report_period_end: Optional reported period end, separate from disclosure clocks.
    """

    semantic_handle: str = Field(pattern=r"^DOC-[A-Z0-9-]{1,48}$")
    entity_id: str = Field(min_length=1, max_length=32)
    source_name: str = Field(min_length=1, max_length=40)
    source_right: str = Field(min_length=1, max_length=80)
    evidence_class: AlternativeEvidenceClass
    document_type: str = Field(min_length=1, max_length=40)
    revision_label: str = Field(min_length=1, max_length=120)
    title: str = Field(min_length=1, max_length=300)
    published_at: datetime | None = None
    accepted_at: datetime | None = None
    available_at: datetime
    immutable_source: bool
    reading_depth: AlternativeEvidenceReadingDepth = AlternativeEvidenceReadingDepth.FULL_FILING
    section_labels: tuple[str, ...] = ()
    workspace_document_id: str = Field(pattern=r"^[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?$")
    workspace_revision: int = Field(ge=1)
    character_count: int = Field(ge=1)
    byte_count: int = Field(ge=1)
    published_precision: PublishedPrecision | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    report_period_end: date | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_reference(self) -> Self:
        """Require every present disclosure clock to be timezone-aware.

        Returns:
            This validated contract.

        Raises:
            ValueError: A present publication, acceptance or availability clock is naive.
        """
        for value in (self.published_at, self.accepted_at, self.available_at):
            if value is not None and (value.tzinfo is None or value.utcoffset() is None):
                raise ValueError("alternative_evidence.document_reference_clock_invalid")
        return self

    def temporal_view(self) -> dict[str, object]:
        """Expose disclosure clocks together with their recorded precision basis.

        Each meaning of this document's time, named, with the basis it rests
        on -- the compact view every consumer of a span carries or resolves
        through the document handle. Publication is the source's date at its
        stated precision (a midnight under DATE precision is a date, not an
        instant; UNKNOWN precision was never recorded); acceptance is the
        official source's own timestamp (UTC); availability is what the
        admission owner bounded it by -- the acceptance itself, or a recorded
        import's capture, which is later; the report period is the official
        metadata's. Nothing here is when an event happened, and a document
        retrieved later is as old as its acceptance says.
        """
        accepted = self.accepted_at
        if accepted is not None and self.available_at == accepted:
            basis = "OFFICIAL_ACCEPTANCE"
        elif self.source_name == "ISSUER_RECORDED":
            basis = (
                "RECORDED_IMPORT_CAPTURE"
                if accepted is not None
                else "RECORDED_IMPORT_CAPTURE_ACCEPTANCE_UNKNOWN"
            )
        else:
            basis = "SOURCE_STATED"
        return {
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "published_precision": self.published_precision or "UNKNOWN",
            "accepted_at": accepted.isoformat() if accepted else None,
            "available_at": self.available_at.isoformat(),
            "availability_basis": basis,
            "report_period_end": (
                None if self.report_period_end is None else self.report_period_end.isoformat()
            ),
        }

    def model_view(self) -> dict[str, object]:
        """Project a packet index entry with auditable source-clock meaning.

        The index entry a packet carries per document; its time is the one
        temporal view, stated once with each meaning's precision and basis.
        """
        return {
            "document_handle": self.semantic_handle,
            "entity_id": self.entity_id,
            "source_name": self.source_name,
            "source_right": self.source_right,
            "evidence_class": self.evidence_class,
            "document_type": self.document_type,
            "title": self.title,
            "character_count": self.character_count,
            "time": self.temporal_view(),
        }


class AlternativeEvidenceDocumentSet(AlternativeEvidenceContract):
    """Seal admitted and rejected document handles against one workspace snapshot.

    Attributes:
        request_hash: Research request identity.
        source_snapshot_hash: Admitted source snapshot identity.
        canonicalization_binding_hash: Installed canonicalization implementation binding.
        reading_depth: Declared admission reading depth.
        workspace_snapshot_id: Workspace snapshot UUID.
        workspace_snapshot_hash: Immutable workspace snapshot commitment.
        documents: Unique admitted document references.
        rejections: Document rejection records.
        published_at: Aware publication clock.
        document_set_hash: Canonical document-set identity.
    """

    kind: Literal["AlternativeEvidenceDocumentSet"] = "AlternativeEvidenceDocumentSet"
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonicalization_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    reading_depth: AlternativeEvidenceReadingDepth
    workspace_snapshot_id: str = Field(
        pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
    )
    workspace_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    documents: tuple[AlternativeEvidenceDocumentReference, ...]
    rejections: tuple[AlternativeEvidenceDocumentRejection, ...]
    published_at: datetime
    document_set_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_document_set(self) -> Self:
        """Validate admitted/rejected handles, publication awareness and set identity.

        Returns:
            This validated contract.

        Raises:
            ValueError: Admitted handles repeat or are rejected, publication is naive, or canonical
                identity differs.
        """
        handles = tuple(item.semantic_handle for item in self.documents)
        rejected = tuple(item.semantic_handle for item in self.rejections)
        if len(handles) != len(set(handles)) or set(handles) & set(rejected):
            raise ValueError("alternative_evidence.document_set_handles_invalid")
        if self.published_at.tzinfo is None or self.published_at.utcoffset() is None:
            raise ValueError("alternative_evidence.document_set_clock_invalid")
        validate_contract_identity(self, "document_set_hash")
        return self


def seal_canonical_document(**values: object) -> AlternativeEvidenceCanonicalDocument:
    """Seal and validate the canonical document from supplied field values.

    Args:
        values: Canonical document fields supplied to the sealing owner.

    Returns:
        The validated canonical document with its computed identity.
    """
    return seal_contract(AlternativeEvidenceCanonicalDocument, "document_hash", **values)


__all__ = [
    "AlternativeEvidenceCanonicalDocument",
    "AlternativeEvidenceDocumentReference",
    "AlternativeEvidenceDocumentRejection",
    "AlternativeEvidenceDocumentSet",
    "DocumentRejectionCode",
    "seal_canonical_document",
]
