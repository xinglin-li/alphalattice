"""Strict contracts for Workspace-owned local knowledge retrieval."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Self
from uuid import UUID

from pydantic import Field, StringConstraints, field_validator, model_validator

from alphalattice.kernel.shared_kernel.domain.base import (
    DomainModel,
    NonEmptyString,
    Sha256Hex,
    ShortString,
    UtcDatetime,
    Uuid4,
)
from alphalattice.kernel.shared_kernel.domain.serialization import canonical_json_bytes, sha256_hex

StableKnowledgeId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        pattern=r"^[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?$",
    ),
]
PositiveInt = Annotated[int, Field(ge=1)]
NonNegativeInt = Annotated[int, Field(ge=0)]
TopK = Annotated[int, Field(ge=1, le=20)]
FiniteScore = Annotated[float, Field(allow_inf_nan=False)]


def _logical_hash(value: DomainModel, *, exclude: set[str]) -> str:
    return sha256_hex(canonical_json_bytes(value.model_dump(mode="python", exclude=exclude)))


def _require_sorted_unique(values: tuple[object, ...], label: str) -> tuple[object, ...]:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique")
    if values != tuple(sorted(values, key=str)):
        raise ValueError(f"{label} must be sorted")
    return values


class KnowledgeNamespace(StrEnum):
    """Distinguish installed system references from user-owned references."""

    SYSTEM_REFERENCE = "SYSTEM_REFERENCE"
    USER_REFERENCE = "USER_REFERENCE"


class KnowledgeAccessClass(StrEnum):
    """Declare system access or private user access for a knowledge source."""

    SYSTEM = "SYSTEM"
    USER_PRIVATE = "USER_PRIVATE"


class KnowledgeMediaType(StrEnum):
    """Identify admitted UTF-8 Markdown and strict JSON knowledge content."""

    MARKDOWN = "text/markdown"
    JSON = "application/json"


class RetrievalStatus(StrEnum):
    """Distinguish resolved ranked hits from an empty retrieval result."""

    RESOLVED = "RESOLVED"
    EMPTY = "EMPTY"


class RetrievalChannel(StrEnum):
    """Identify the lexical, CJK, dense and trigram channels supporting retrieval."""

    IDENTIFIER_EXACT = "IDENTIFIER_EXACT"
    TERM_EXACT = "TERM_EXACT"
    TERM_ALL = "TERM_ALL"
    TERM_ANY = "TERM_ANY"
    CJK_BIGRAM = "CJK_BIGRAM"
    DENSE = "DENSE"
    TRIGRAM = "TRIGRAM"


class RetrievalMatchKind(StrEnum):
    """Describe identifier, exact-term, fuzzy or trigram-only hit matching."""

    IDENTIFIER_EXACT = "IDENTIFIER_EXACT"
    TERM_EXACT = "TERM_EXACT"
    FUZZY = "FUZZY"
    TRIGRAM_ONLY = "TRIGRAM_ONLY"


class KnowledgeSourceCommitment(DomainModel):
    """Bind a source revision, license, access class and availability interval.

    Any expiry must follow available_at. source_logical_hash identifies the exact source revision;
    this record does not itself publish the content.
    """

    source_id: StableKnowledgeId
    source_revision: PositiveInt
    source_logical_hash: Sha256Hex
    license_id: ShortString
    access_class: KnowledgeAccessClass
    available_at: UtcDatetime
    expires_at: UtcDatetime | None = None

    @model_validator(mode="after")
    def validate_expiry(self) -> Self:
        """Require any expiry to follow the source availability time.

        Returns:
            This contract after validating the declared consistency rules.

        Raises:
            ValueError: expires_at is equal to or earlier than available_at.
        """
        if self.expires_at is not None and self.expires_at <= self.available_at:
            raise ValueError("expires_at must follow available_at")
        return self


class WorkspaceKnowledgeDocument(DomainModel):
    """Seal a document identifier, namespace, title and media type.

    logical_hash binds document metadata, independently of its later content revisions.
    """

    document_id: StableKnowledgeId
    namespace: KnowledgeNamespace
    title: ShortString
    media_type: KnowledgeMediaType
    logical_hash: Sha256Hex

    @model_validator(mode="after")
    def validate_logical_hash(self) -> Self:
        """Require the exact canonical document identity.

        Returns:
            This contract after validating the declared consistency rules.

        Raises:
            ValueError: logical_hash differs from the document content excluding that hash.
        """
        if self.logical_hash != _logical_hash(self, exclude={"logical_hash"}):
            raise ValueError("document logical_hash is inconsistent")
        return self

    @classmethod
    def create(
        cls,
        *,
        document_id: str,
        namespace: KnowledgeNamespace,
        title: str,
        media_type: KnowledgeMediaType,
    ) -> WorkspaceKnowledgeDocument:
        """Seal document metadata with its canonical logical identity.

        Args:
            document_id: Stable document identifier.
            namespace: System-reference or user-reference namespace.
            title: Nonempty human-readable document title.
            media_type: Admitted Markdown or JSON content type.

        Returns:
            Validated document with logical_hash derived from its canonical metadata.

        Raises:
            pydantic.ValidationError: Document metadata violates the contract.
        """
        payload = {
            "schema_version": "1",
            "document_id": document_id,
            "namespace": namespace,
            "title": title,
            "media_type": media_type,
        }
        return cls(
            document_id=document_id,
            namespace=namespace,
            title=title,
            media_type=media_type,
            logical_hash=sha256_hex(canonical_json_bytes(payload)),
        )


class WorkspaceKnowledgeRevision(DomainModel):
    """Seal document/source metadata and exact content-addressed bytes.

    The revision commits SHA-256, byte length and knowledge/blobs path. Namespace and source access
    must agree; logical_hash binds metadata and byte commitments.
    """

    document: WorkspaceKnowledgeDocument
    revision: PositiveInt
    source: KnowledgeSourceCommitment
    created_at: UtcDatetime
    content_sha256: Sha256Hex
    content_size_bytes: NonNegativeInt
    blob_path: NonEmptyString
    logical_hash: Sha256Hex

    @model_validator(mode="after")
    def validate_revision(self) -> Self:
        """Require content-addressed storage, namespace-compatible access and revision hash.

        Returns:
            This contract after validating the declared consistency rules.

        Raises:
            ValueError: The blob path, source access class or canonical revision identity is
                invalid.
        """
        expected_blob = f"knowledge/blobs/{self.content_sha256}"
        if self.blob_path != expected_blob:
            raise ValueError("revision blob_path is not content-addressed")
        if self.document.namespace is KnowledgeNamespace.SYSTEM_REFERENCE:
            if self.source.access_class is not KnowledgeAccessClass.SYSTEM:
                raise ValueError("system reference requires SYSTEM access")
        elif self.source.access_class is not KnowledgeAccessClass.USER_PRIVATE:
            raise ValueError("user reference requires USER_PRIVATE access")
        if self.logical_hash != _logical_hash(self, exclude={"logical_hash"}):
            raise ValueError("revision logical_hash is inconsistent")
        return self

    @classmethod
    def create(
        cls,
        *,
        document: WorkspaceKnowledgeDocument,
        revision: int,
        source: KnowledgeSourceCommitment,
        created_at: datetime,
        content: bytes,
    ) -> WorkspaceKnowledgeRevision:
        """Seal revision metadata and content commitments without publishing files.

        Args:
            document: Sealed document metadata.
            revision: Positive revision number.
            source: Exact source/access/availability commitment.
            created_at: UTC revision creation metadata.
            content: Bytes whose SHA-256, size and blob path are committed.

        Returns:
            Validated revision with content-addressed blob_path and canonical logical_hash.

        Raises:
            pydantic.ValidationError: Revision metadata or namespace/access consistency is invalid.
        """
        content_hash = sha256_hex(content)
        payload = {
            "schema_version": "1",
            "document": document,
            "revision": revision,
            "source": source,
            "created_at": created_at,
            "content_sha256": content_hash,
            "content_size_bytes": len(content),
            "blob_path": f"knowledge/blobs/{content_hash}",
        }
        return cls(
            document=document,
            revision=revision,
            source=source,
            created_at=created_at,
            content_sha256=content_hash,
            content_size_bytes=len(content),
            blob_path=f"knowledge/blobs/{content_hash}",
            logical_hash=sha256_hex(canonical_json_bytes(payload)),
        )


class WorkspaceKnowledgeSnapshot(DomainModel):
    """Seal one canonical set of immutable document revisions.

    A snapshot is nonempty and holds exactly one revision per document, ordered by
    namespace/document/revision. Its UUID and creation metadata participate in logical_hash.
    """

    snapshot_id: Uuid4
    created_at: UtcDatetime
    revisions: tuple[WorkspaceKnowledgeRevision, ...]
    logical_hash: Sha256Hex

    @field_validator("revisions")
    @classmethod
    def validate_revisions(
        cls,
        values: tuple[WorkspaceKnowledgeRevision, ...],
    ) -> tuple[WorkspaceKnowledgeRevision, ...]:
        """Require a nonempty canonical revision list with one revision per document.

        Args:
            values: Revisions ordered by namespace, document identifier and revision.

        Returns:
            The unchanged revision tuple.

        Raises:
            ValueError: Revisions are empty/unordered or a document appears more than once.
        """
        if not values:
            raise ValueError("snapshot requires at least one revision")
        identities = tuple(
            (
                revision.document.namespace.value,
                revision.document.document_id,
                revision.revision,
            )
            for revision in values
        )
        if identities != tuple(sorted(identities)):
            raise ValueError("snapshot revisions must be canonically sorted")
        documents = tuple(
            (revision.document.namespace.value, revision.document.document_id)
            for revision in values
        )
        if len(documents) != len(set(documents)):
            raise ValueError("snapshot contains more than one revision for a document")
        return values

    @model_validator(mode="after")
    def validate_logical_hash(self) -> Self:
        """Require the exact canonical snapshot identity.

        Returns:
            This contract after validating the declared consistency rules.

        Raises:
            ValueError: logical_hash differs from the snapshot content excluding that hash.
        """
        if self.logical_hash != _logical_hash(self, exclude={"logical_hash"}):
            raise ValueError("snapshot logical_hash is inconsistent")
        return self

    @classmethod
    def create(
        cls,
        *,
        snapshot_id: UUID,
        created_at: datetime,
        revisions: tuple[WorkspaceKnowledgeRevision, ...],
    ) -> WorkspaceKnowledgeSnapshot:
        """Seal a canonical revision set without publishing a snapshot manifest.

        Args:
            snapshot_id: UUID for the immutable snapshot.
            created_at: UTC snapshot creation metadata.
            revisions: Nonempty canonical tuple, with one revision per document.

        Returns:
            Validated snapshot with logical_hash derived from its complete declared content.

        Raises:
            pydantic.ValidationError: Snapshot metadata or revision ordering violates the contract.
        """
        payload = {
            "schema_version": "1",
            "snapshot_id": snapshot_id,
            "created_at": created_at,
            "revisions": revisions,
        }
        return cls(
            snapshot_id=snapshot_id,
            created_at=created_at,
            revisions=revisions,
            logical_hash=sha256_hex(canonical_json_bytes(payload)),
        )


class KnowledgePublicationReceipt(DomainModel):
    """Bind a published revision to its canonical immutable manifest path."""

    revision: WorkspaceKnowledgeRevision
    manifest_path: NonEmptyString

    @model_validator(mode="after")
    def validate_manifest_path(self) -> Self:
        """Require the canonical document/revision manifest path.

        Returns:
            This contract after validating the declared consistency rules.

        Raises:
            ValueError: manifest_path differs from the path implied by the published revision.
        """
        expected = (
            f"knowledge/documents/{self.revision.document.document_id}/"
            f"{self.revision.revision}.json"
        )
        if self.manifest_path != expected:
            raise ValueError("receipt manifest_path is not canonical")
        return self


class LexicalIndexSpec(DomainModel):
    """Seal the installed fixed-v1 lexical chunking, query and fusion recipe.

    Chunk size/overlap, query/top-k limits, RRF constant and channel weights must equal the
    installed fixed recipe. logical_hash binds those values; this contract admits no arbitrary
    tuning.
    """

    policy_id: Annotated[str, StringConstraints(pattern=r"^fixed-v1$")]
    chunk_size_codepoints: Annotated[int, Field(ge=1)]
    chunk_overlap_codepoints: NonNegativeInt
    maximum_query_codepoints: Annotated[int, Field(ge=1)]
    maximum_query_terms: Annotated[int, Field(ge=1)]
    maximum_top_k: Annotated[int, Field(ge=1)]
    rrf_constant: Annotated[int, Field(ge=1)]
    identifier_bm25_weight: FiniteScore
    title_bm25_weight: FiniteScore
    heading_bm25_weight: FiniteScore
    body_bm25_weight: FiniteScore
    term_all_weight: FiniteScore
    term_any_weight: FiniteScore
    cjk_bigram_weight: FiniteScore
    trigram_weight: FiniteScore
    logical_hash: Sha256Hex

    @model_validator(mode="after")
    def validate_fixed_v1(self) -> Self:
        """Require the exact fixed-v1 lexical recipe and its canonical hash.

        Returns:
            This contract after validating the declared consistency rules.

        Raises:
            ValueError: Recipe values, chunk overlap or logical_hash violate the fixed-v1 contract.
        """
        expected = {
            "policy_id": "fixed-v1",
            "chunk_size_codepoints": 512,
            "chunk_overlap_codepoints": 64,
            "maximum_query_codepoints": 512,
            "maximum_query_terms": 64,
            "maximum_top_k": 20,
            "rrf_constant": 60,
            "identifier_bm25_weight": 12.0,
            "title_bm25_weight": 8.0,
            "heading_bm25_weight": 5.0,
            "body_bm25_weight": 1.0,
            "term_all_weight": 1.25,
            "term_any_weight": 1.0,
            "cjk_bigram_weight": 0.9,
            "trigram_weight": 0.5,
        }
        actual = self.model_dump(mode="python", exclude={"schema_version", "logical_hash"})
        if actual != expected:
            raise ValueError("LexicalIndexSpec must match fixed-v1")
        if self.chunk_overlap_codepoints >= self.chunk_size_codepoints:
            raise ValueError("chunk overlap must be smaller than chunk size")
        if self.logical_hash != _logical_hash(self, exclude={"logical_hash"}):
            raise ValueError("index spec logical_hash is inconsistent")
        return self

    @classmethod
    def fixed_v1(cls) -> LexicalIndexSpec:
        """Construct the installed fixed-v1 lexical recipe.

        Returns:
            Exact lexical limits, chunking and RRF/channel weights with a canonical logical_hash.
        """
        payload = {
            "schema_version": "1",
            "policy_id": "fixed-v1",
            "chunk_size_codepoints": 512,
            "chunk_overlap_codepoints": 64,
            "maximum_query_codepoints": 512,
            "maximum_query_terms": 64,
            "maximum_top_k": 20,
            "rrf_constant": 60,
            "identifier_bm25_weight": 12.0,
            "title_bm25_weight": 8.0,
            "heading_bm25_weight": 5.0,
            "body_bm25_weight": 1.0,
            "term_all_weight": 1.25,
            "term_any_weight": 1.0,
            "cjk_bigram_weight": 0.9,
            "trigram_weight": 0.5,
        }
        return cls(
            policy_id="fixed-v1",
            chunk_size_codepoints=512,
            chunk_overlap_codepoints=64,
            maximum_query_codepoints=512,
            maximum_query_terms=64,
            maximum_top_k=20,
            rrf_constant=60,
            identifier_bm25_weight=12.0,
            title_bm25_weight=8.0,
            heading_bm25_weight=5.0,
            body_bm25_weight=1.0,
            term_all_weight=1.25,
            term_any_weight=1.0,
            cjk_bigram_weight=0.9,
            trigram_weight=0.5,
            logical_hash=sha256_hex(canonical_json_bytes(payload)),
        )


class KnowledgeCitation(DomainModel):
    """Bind a retrieved chunk to exact snapshot, revision, content and source spans.

    Line bounds are inclusive; character and UTF-8 byte ranges must be nonempty and ordered.
    Document/chunk identifiers and hashes support authoritative citation readback.
    """

    document_id: StableKnowledgeId
    revision: PositiveInt
    snapshot_id: Uuid4
    snapshot_logical_hash: Sha256Hex
    content_sha256: Sha256Hex
    heading_path: tuple[ShortString, ...] = ()
    start_line: PositiveInt
    end_line: PositiveInt
    character_start: NonNegativeInt
    character_end: PositiveInt
    utf8_byte_start: NonNegativeInt
    utf8_byte_end: PositiveInt
    """The source range covering the matched chunk.

    Source-exact, and covering rather than equal: a chunk built by joining two
    paragraphs carries a synthetic separator where the source had other
    whitespace, so the range is the smallest run of real source that contains
    the chunk. An excerpt sliced from it is always sealed source bytes; it is
    not always byte-identical to the embedded body.

    Lines say where a reader should look; they do not say what was matched. A
    document whose sections span the whole file gives every one of its chunks
    the same line range, so a line-addressed read returns the head of the file
    for all of them. The character range is what the retriever actually
    matched, and the byte range lets a reader verify the slice against the
    sealed bytes without decoding the whole document.
    """

    chunk_id: Sha256Hex
    chunk_hash: Sha256Hex

    @model_validator(mode="after")
    def validate_span(self) -> Self:
        """Require ordered lines and nonempty character and UTF-8 byte spans.

        Returns:
            This contract after validating the declared consistency rules.

        Raises:
            ValueError: Lines are crossed, either span is empty/crossed or its byte extent is
                shorter than its character extent.
        """
        if self.end_line < self.start_line:
            raise ValueError("citation end_line precedes start_line")
        if self.character_end <= self.character_start:
            raise ValueError("citation character range is empty or crossed")
        if self.utf8_byte_end <= self.utf8_byte_start:
            raise ValueError("citation byte range is empty or crossed")
        if self.utf8_byte_end - self.utf8_byte_start < self.character_end - self.character_start:
            raise ValueError("citation byte range is shorter than its character range")
        return self


class KnowledgeRetrievalHit(DomainModel):
    """Describe one ranked preview, contributing channels and exact source citation.

    rrf_score is finite ranking evidence. Channels are nonempty, sorted and unique; the citation
    carries document/revision/snapshot and content commitments.
    """

    rank: PositiveInt
    title: ShortString
    preview: NonEmptyString
    match_kind: RetrievalMatchKind
    channels: tuple[RetrievalChannel, ...]
    rrf_score: FiniteScore
    citation: KnowledgeCitation

    @field_validator("channels")
    @classmethod
    def validate_channels(
        cls,
        values: tuple[RetrievalChannel, ...],
    ) -> tuple[RetrievalChannel, ...]:
        """Require at least one sorted unique contributing retrieval channel.

        Args:
            values: Channels supporting this hit.

        Returns:
            The unchanged channel tuple.

        Raises:
            ValueError: Channels are empty, unordered or duplicated.
        """
        if not values:
            raise ValueError("hit requires at least one channel")
        return _require_sorted_unique(values, "hit channels")  # type: ignore[return-value]


__all__ = [
    "KnowledgeAccessClass",
    "KnowledgeCitation",
    "KnowledgeMediaType",
    "KnowledgeNamespace",
    "KnowledgePublicationReceipt",
    "KnowledgeRetrievalHit",
    "KnowledgeSourceCommitment",
    "LexicalIndexSpec",
    "RetrievalChannel",
    "RetrievalMatchKind",
    "RetrievalStatus",
    "WorkspaceKnowledgeDocument",
    "WorkspaceKnowledgeRevision",
    "WorkspaceKnowledgeSnapshot",
]
