"""Deterministic quality gates for already acquired evidence bytes."""

from __future__ import annotations

import re

from ..sources.contracts import MAXIMUM_SOURCE_DOCUMENT_BYTES, AcquiredEvidenceDocument
from .contracts import DocumentRejectionCode

_ALLOWED_MEDIA_TYPES = frozenset({"text/html", "text/plain", "text/markdown"})
MAXIMUM_SOURCE_BYTES: int = MAXIMUM_SOURCE_DOCUMENT_BYTES
"""The most an original may hold to be canonicalized: the same ceiling the
source policy can admit at acquisition (10,000,000 bytes). The two bounds were
5,000,000 here against 10,000,000 there, so a filing the acquisition admitted
-- a 7.5 MB inline-XBRL 10-K -- was fetched, retained and then refused at
parse; one ceiling, stated by the root contracts, keeps them one."""


class AlternativeEvidenceDocumentQualityError(ValueError):
    """Name a document quality rejection with its stable code."""

    def __init__(self, code: DocumentRejectionCode, message: str) -> None:
        """Carry the rejection code and safe explanation."""
        super().__init__(message)
        self.code = code


def validate_source_document(document: AcquiredEvidenceDocument) -> None:
    """Reject unsupported, oversized, empty, or invalid source bytes."""
    if document.media_type not in _ALLOWED_MEDIA_TYPES:
        raise AlternativeEvidenceDocumentQualityError(
            DocumentRejectionCode.INVALID_MEDIA_TYPE,
            "source media type is not admitted",
        )
    if len(document.content) > MAXIMUM_SOURCE_BYTES:
        raise AlternativeEvidenceDocumentQualityError(
            DocumentRejectionCode.SIZE_LIMIT_EXCEEDED,
            f"source document of {len(document.content)} bytes exceeds the canonicalization "
            f"bound of {MAXIMUM_SOURCE_BYTES} bytes",
        )
    if not document.content.strip():
        raise AlternativeEvidenceDocumentQualityError(
            DocumentRejectionCode.EMPTY,
            "source document is empty",
        )
    if b"\x00" in document.content:
        raise AlternativeEvidenceDocumentQualityError(
            DocumentRejectionCode.INVALID_ENCODING,
            "source document contains a NUL byte",
        )


def validate_canonical_text(text: str) -> None:
    """Reject empty or boilerplate-only canonical narrative."""
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        raise AlternativeEvidenceDocumentQualityError(
            DocumentRejectionCode.EMPTY,
            "canonical document is empty",
        )
    if not any(character.isalpha() for character in compact):
        raise AlternativeEvidenceDocumentQualityError(
            DocumentRejectionCode.BOILERPLATE_ONLY,
            "canonical document has no narrative text",
        )
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    if len(lines) >= 10 and len(set(lines)) * 5 <= len(lines):
        raise AlternativeEvidenceDocumentQualityError(
            DocumentRejectionCode.BOILERPLATE_ONLY,
            "canonical document is dominated by repeated boilerplate",
        )


__all__ = [
    "AlternativeEvidenceDocumentQualityError",
    "validate_canonical_text",
    "validate_source_document",
]
