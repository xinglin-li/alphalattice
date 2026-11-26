"""Canonicalize admitted source bytes with the tracked extraction owner."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import cast

from alphalattice.kernel.live_evidence.errors import LiveEvidenceError
from alphalattice.kernel.live_evidence.online_sources import extract_canonical_markdown
from alphalattice.kernel.shared_kernel.domain.serialization import sha256_hex

from ..contracts import ADMITTED_DOCUMENT_CAPACITY, AlternativeEvidenceReadingDepth
from ..sources.contracts import AcquiredEvidenceDocument
from .contracts import (
    AlternativeEvidenceCanonicalDocument,
    AlternativeEvidenceDocumentRejection,
    DocumentRejectionCode,
    seal_canonical_document,
)
from .quality import (
    MAXIMUM_SOURCE_BYTES,
    AlternativeEvidenceDocumentQualityError,
    validate_canonical_text,
    validate_source_document,
)
from .sec_sections import (
    SecSectionDiscoveryPort,
    resolve_sec_section_discovery,
    section_discovery_is_available,
)
from .structure import heading_kind

TABLE_CARRY_RULES_ID = "alternative-evidence.table-carry.v1"
"""v1 (2026-09-18): tables are carried into the canonical text of an SEC
filing in the items the desk reads for disclosures -- a 10-Q's Part II
Items 1 to 5 and its Part I Item 4; a 10-K's Items 1 to 5 and 9 to 9C in
Parts I and II; every item of an 8-K -- and nowhere before the first Part
or Item heading. The financial statements and their notes, MD&A, the
proxy items, the exhibit indexes and the signature pages keep their
tables in the original; the canonical text says where each stood."""

_DISCLOSURE_ITEMS_10K = frozenset(
    {"1", "1A", "1B", "1C", "2", "3", "4", "5", "9", "9A", "9B", "9C"}
)
_DISCLOSURE_ITEMS_10Q_PART_II = frozenset({"1", "1A", "2", "3", "4", "5"})


class SecTableCarry:
    """Track SEC Part and Item context for table admission.

    The `TableCarryPolicy` of an SEC filing: a running Part and Item read
    from the short blocks the extractor passes in source order, and the
    admission above. It is a running heading for one decision, not the
    document's structure -- that is read from the canonical text.
    """

    def __init__(self, document_type: str) -> None:
        """Track the current SEC form, Part, and Item for table admission."""
        self.family_form = document_type.upper().removesuffix("/A")
        self.part: str | None = None
        self.item: str | None = None

    def observe(self, text: str) -> None:
        """Update the current Part or Item from a source heading block."""
        kind = heading_kind(text, document_type=self.family_form)
        if kind is None:
            return
        label, value = kind
        if label == "PART":
            self.part = value
            self.item = None
        else:
            self.item = value

    def carries(self) -> bool:
        """Decide whether the current SEC item carries tables into canonical text."""
        if self.family_form == "8-K":
            return self.item is not None
        if self.family_form == "10-Q":
            if self.part == "II":
                return self.item is None or self.item in _DISCLOSURE_ITEMS_10Q_PART_II
            return self.part == "I" and self.item == "4"
        if self.family_form == "10-K":
            # The financial statements set after Part IV re-use "Item 1".
            return self.part in {None, "I", "II"} and self.item in _DISCLOSURE_ITEMS_10K
        return False


def canonicalize_source_documents(
    documents: Iterable[AcquiredEvidenceDocument],
    *,
    reading_depth: AlternativeEvidenceReadingDepth = AlternativeEvidenceReadingDepth.FULL_FILING,
    sec_section_discovery: SecSectionDiscoveryPort | None = None,
    known_canonical: Callable[[AcquiredEvidenceDocument], bytes | None] | None = None,
) -> tuple[
    tuple[AlternativeEvidenceCanonicalDocument, ...],
    tuple[AlternativeEvidenceDocumentRejection, ...],
]:
    """Admit each source document as canonical Markdown, or reject it by name.

    A caller that holds the canonical text a sealed commitment proves for a
    document's exact bytes under this binding and depth supplies it through
    `known_canonical`; the text is then not extracted again, and the
    document is sealed exactly as an extraction would seal it -- the same
    validations, the same labels and limitations, the same hash.
    """
    admitted: list[AlternativeEvidenceCanonicalDocument] = []
    rejected: list[AlternativeEvidenceDocumentRejection] = []
    seen_revisions: set[tuple[str, str, str]] = set()
    for document in documents:
        if len(admitted) >= ADMITTED_DOCUMENT_CAPACITY:
            rejected.append(
                AlternativeEvidenceDocumentRejection(
                    semantic_handle=document.semantic_handle,
                    entity_id=document.entity_id,
                    code=DocumentRejectionCode.SIZE_LIMIT_EXCEEDED,
                    summary=(
                        f"document-set capacity of {ADMITTED_DOCUMENT_CAPACITY} admitted "
                        "documents was reached"
                    ),
                )
            )
            continue
        identity = (document.entity_id, document.source_name, document.revision)
        if identity in seen_revisions:
            rejected.append(
                AlternativeEvidenceDocumentRejection(
                    semantic_handle=document.semantic_handle,
                    entity_id=document.entity_id,
                    code=DocumentRejectionCode.SOURCE_IDENTITY_INVALID,
                    summary="duplicate source revision",
                )
            )
            continue
        seen_revisions.add(identity)
        try:
            validate_source_document(document)
            known = None if known_canonical is None else known_canonical(document)
            if known is not None and _full_narrative_applies(document, reading_depth):
                payload, section_labels, scope_limitations = known, *_FULL_NARRATIVE
            else:
                payload, section_labels, scope_limitations = _canonical_markdown(
                    document,
                    reading_depth=reading_depth,
                    sec_section_discovery=sec_section_discovery,
                )
            text = payload.decode("utf-8")
            validate_canonical_text(text)
            admitted.append(
                seal_canonical_document(
                    semantic_handle=document.semantic_handle,
                    entity_id=document.entity_id,
                    source_name=document.source_name,
                    source_right=document.source_right,
                    evidence_class=document.evidence_class,
                    document_type=document.document_type,
                    revision=document.revision,
                    title=document.title,
                    published_at=document.published_at,
                    accepted_at=document.accepted_at,
                    available_at=document.available_at,
                    immutable_source=document.immutable_source,
                    limitations=(*document.limitations, *scope_limitations),
                    reading_depth=reading_depth,
                    section_labels=section_labels,
                    published_precision=document.published_precision,
                    report_period_end=document.report_period_end,
                    canonical_markdown=payload,
                    canonical_content_hash=sha256_hex(payload),
                    character_count=len(text),
                    byte_count=len(payload),
                )
            )
        except AlternativeEvidenceDocumentQualityError as error:
            rejected.append(
                AlternativeEvidenceDocumentRejection(
                    semantic_handle=document.semantic_handle,
                    entity_id=document.entity_id,
                    code=error.code,
                    summary=str(error),
                )
            )
        except (LiveEvidenceError, UnicodeDecodeError) as error:
            rejected.append(
                AlternativeEvidenceDocumentRejection(
                    semantic_handle=document.semantic_handle,
                    entity_id=document.entity_id,
                    code=DocumentRejectionCode.EXTRACTION_FAILED,
                    summary=str(error),
                )
            )
    return tuple(admitted), tuple(rejected)


def _canonical_markdown(
    document: AcquiredEvidenceDocument,
    *,
    reading_depth: AlternativeEvidenceReadingDepth,
    sec_section_discovery: SecSectionDiscoveryPort | None,
) -> tuple[bytes, tuple[str, ...], tuple[str, ...]]:
    if document.media_type == "text/html":
        if (
            reading_depth is AlternativeEvidenceReadingDepth.MATERIAL_SECTIONS
            and document.source_name == "SEC_EDGAR"
        ):
            discovery = (sec_section_discovery or resolve_sec_section_discovery())(
                document.content,
                form=document.document_type,
                source_revision=document.revision,
                input_cap_bytes=MAXIMUM_SOURCE_BYTES,
            )
            if not section_discovery_is_available(discovery):
                raise AlternativeEvidenceDocumentQualityError(
                    DocumentRejectionCode.SECTION_EXTRACTION_UNAVAILABLE,
                    "SEC material sections were unavailable: "
                    f"{discovery.unavailable_reason or 'UNKNOWN'}",
                )
            selected = tuple(
                section
                for section in discovery.sections
                if _material_section(document.document_type, section.item_label)
            )
            if not selected:
                raise AlternativeEvidenceDocumentQualityError(
                    DocumentRejectionCode.SECTION_EXTRACTION_UNAVAILABLE,
                    "SEC filing exposed no pre-registered material sections",
                )
            labels = tuple(section.item_label for section in selected)
            payload = b"\n".join(section.canonical_markdown.rstrip() for section in selected)
            return (
                payload.rstrip() + b"\n",
                labels,
                ("Default reading scope is limited to pre-registered material SEC sections.",),
            )
        return (
            cast(
                bytes,
                extract_canonical_markdown(
                    document.content,
                    input_cap_bytes=MAXIMUM_SOURCE_BYTES,
                    tables=(
                        SecTableCarry(document.document_type)
                        if document.source_name == "SEC_EDGAR"
                        else None
                    ),
                ),
            ),
            *_FULL_NARRATIVE,
        )
    try:
        text = document.content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise AlternativeEvidenceDocumentQualityError(
            DocumentRejectionCode.INVALID_ENCODING,
            "source document is not strict UTF-8",
        ) from error
    canonical = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    return (canonical + "\n").encode("utf-8"), ("ALL_NARRATIVE",), ()


_FULL_NARRATIVE: tuple[tuple[str, ...], tuple[str, ...]] = (
    ("ALL_NARRATIVE",),
    ("Full filing narrative was admitted for deep reading.",),
)
"""The labels and scope limitation every full-filing HTML extraction carries."""


def _full_narrative_applies(
    document: AcquiredEvidenceDocument, reading_depth: AlternativeEvidenceReadingDepth
) -> bool:
    """Decide whether a sealed full-filing narrative may be reused.

    Whether `_canonical_markdown` would take the full-filing HTML path for
    this document -- the one path a sealed canonical text may stand in for.
    """
    return (
        document.media_type == "text/html"
        and reading_depth is AlternativeEvidenceReadingDepth.FULL_FILING
    )


def _material_section(form: str, label: str) -> bool:
    family = form.upper().removesuffix("/A")
    if family == "8-K":
        return label == "ALL_NARRATIVE"
    if label.startswith("SUBSEQUENT_EVENTS_"):
        return True
    if family == "10-K":
        return label in {"ITEM_1A", "ITEM_3", "ITEM_7", "ITEM_7A"}
    if family == "10-Q":
        return label in {
            "PART_I_ITEM_2",
            "PART_I_ITEM_3",
            "PART_II_ITEM_1",
            "PART_II_ITEM_1A",
        }
    return False


__all__ = ["TABLE_CARRY_RULES_ID", "SecTableCarry", "canonicalize_source_documents"]
