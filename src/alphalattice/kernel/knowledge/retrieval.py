"""Workspace-owned immutable local knowledge and lexical retrieval."""

from __future__ import annotations

import re
import sqlite3
import unicodedata
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from itertools import accumulate
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID

from pydantic import ValidationError

from alphalattice.control.workspace_runtime.paths import (
    make_confined_parents,
    resolve_confined,
)
from alphalattice.kernel.knowledge.retrieval_contracts import (
    KnowledgeAccessClass,
    KnowledgeCitation,
    KnowledgeMediaType,
    KnowledgeNamespace,
    KnowledgePublicationReceipt,
    KnowledgeSourceCommitment,
    LexicalIndexSpec,
    WorkspaceKnowledgeDocument,
    WorkspaceKnowledgeRevision,
    WorkspaceKnowledgeSnapshot,
)
from alphalattice.kernel.knowledge.retrieval_errors import KnowledgeRetrievalError
from alphalattice.kernel.shared_kernel.domain.errors import (
    DomainValidationError,
)
from alphalattice.kernel.shared_kernel.domain.serialization import (
    canonical_json_bytes,
    parse_model,
    sha256_hex,
    strict_json_loads,
)
from alphalattice.kernel.shared_kernel.persistence import write_new

if TYPE_CHECKING:
    from alphalattice.control.workspace_runtime.core import Workspace

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_TERM = re.compile(r"[^\W_]+(?:[._-][^\W_]+)*", re.UNICODE)
_SPACE = re.compile(r"\s+")


def _retrieval_error(
    message: str,
    *,
    code: str,
    retryable: bool = False,
    cause: Exception | None = None,
) -> KnowledgeRetrievalError:
    error = KnowledgeRetrievalError(message, code=code, retryable=retryable)
    if cause is not None:
        error.__cause__ = cause
    return error


def _canonical_hash(payload: Any) -> str:
    return sha256_hex(canonical_json_bytes(payload))


def _utc_text(value: datetime) -> str:
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _normalize_text(value: str) -> str:
    return _SPACE.sub(" ", unicodedata.normalize("NFKC", value).casefold()).strip()


_CJK_RUN = re.compile("[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002fa1f]+")
"""A maximal run of the code points `_is_cjk` admits, found in one scan: the
per-character Python loop it replaces cost 2.2 million calls on a fifty-issuer
corpus that holds no CJK text at all."""


def _cjk_terms(value: str) -> tuple[str, ...]:
    terms: set[str] = set()
    for match in _CJK_RUN.finditer(unicodedata.normalize("NFKC", value)):
        run = match.group(0)
        terms.update(run)
        terms.update(run[index : index + 2] for index in range(len(run) - 1))
    return tuple(sorted(terms))


def _literal_fts_term(value: str) -> str:
    return f'"{value.replace(chr(34), chr(34) * 2)}"'


def _compiled_terms(value: str) -> tuple[str, ...]:
    return tuple(sorted({_normalize_text(match.group(0)) for match in _TERM.finditer(value)}))


def _validate_source_content(media_type: KnowledgeMediaType, content: bytes) -> None:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _retrieval_error(
            "knowledge source is not strict UTF-8",
            code="retrieval.source_integrity",
            cause=exc,
        ) from exc
    if media_type is KnowledgeMediaType.JSON:
        try:
            strict_json_loads(text)
        except DomainValidationError as exc:
            raise _retrieval_error(
                "knowledge JSON source is invalid",
                code="retrieval.source_integrity",
                cause=exc,
            ) from exc


class WorkspaceKnowledgeLibrary:
    """Authoritative immutable publication and readback for local knowledge."""

    def __init__(self, workspace: Workspace) -> None:
        """Bind immutable knowledge publication/readback to one workspace.

        Args:
            workspace: Workspace owning confined knowledge blobs, revision manifests and snapshots.
        """
        self.workspace = workspace

    @staticmethod
    def _manifest_relative(document_id: str, revision: int) -> str:
        return f"knowledge/documents/{document_id}/{revision}.json"

    @staticmethod
    def _snapshot_relative(snapshot_id: UUID) -> str:
        return f"knowledge/snapshots/{snapshot_id}.json"

    def publish_revision(
        self,
        *,
        document: WorkspaceKnowledgeDocument,
        revision: int,
        source: KnowledgeSourceCommitment,
        created_at: datetime,
        content: bytes,
    ) -> KnowledgePublicationReceipt:
        """Publish an immutable revision under the workspace lock, with its manifest last.

        Args:
            document: Sealed document identity, namespace and media type.
            revision: Positive document revision number.
            source: Exact source identity, license, access and availability commitment.
            created_at: UTC revision creation metadata.
            content: Exact UTF-8 Markdown or strict JSON bytes to commit.

        Returns:
            Receipt binding the sealed revision and canonical manifest path. Exact replay is
            accepted.

        Raises:
            KnowledgeRetrievalError: Content is invalid or an existing immutable target has
                conflicting bytes.
            pydantic.ValidationError: Revision metadata violates the publication contract.
        """
        _validate_source_content(document.media_type, content)
        record = WorkspaceKnowledgeRevision.create(
            document=document,
            revision=revision,
            source=source,
            created_at=created_at,
            content=content,
        )
        manifest_relative = self._manifest_relative(document.document_id, revision)
        manifest_bytes = canonical_json_bytes(record)
        with self.workspace.lock():
            blob = resolve_confined(self.workspace.root, record.blob_path)
            manifest = make_confined_parents(self.workspace.root, manifest_relative)
            manifest = resolve_confined(self.workspace.root, manifest_relative)
            if manifest.exists():
                self._publish_or_replay(
                    manifest,
                    manifest_bytes,
                    conflict_code="retrieval.source_integrity",
                )
                self._publish_or_replay(
                    blob,
                    content,
                    conflict_code="retrieval.source_integrity",
                )
                return KnowledgePublicationReceipt(
                    revision=record,
                    manifest_path=manifest_relative,
                )
            self._publish_or_replay(
                blob,
                content,
                conflict_code="retrieval.source_integrity",
            )
            self._publish_or_replay(
                manifest,
                manifest_bytes,
                conflict_code="retrieval.source_integrity",
            )
        return KnowledgePublicationReceipt(
            revision=record,
            manifest_path=manifest_relative,
        )

    @staticmethod
    def _publish_or_replay(path: Path, content: bytes, *, conflict_code: str) -> None:
        if path.exists():
            if not path.is_file() or path.read_bytes() != content:
                raise _retrieval_error(
                    f"immutable knowledge target conflicts: {path.name}",
                    code=conflict_code,
                )
            return
        write_new(path, content)

    def read_revision(
        self,
        document_id: str,
        revision: int,
    ) -> tuple[WorkspaceKnowledgeRevision, bytes]:
        """Verify a canonical revision manifest and its committed source bytes.

        Args:
            document_id: Exact document identifier.
            revision: Exact positive revision number.

        Returns:
            Validated revision and bytes after path, encoding, size, digest and content checks.

        Raises:
            KnowledgeRetrievalError: The manifest/blob is absent, malformed or differs from its
                commitment.
        """
        relative = self._manifest_relative(document_id, revision)
        path = resolve_confined(self.workspace.root, relative)
        if not path.is_file():
            raise _retrieval_error(
                f"knowledge revision does not exist: {document_id}@{revision}",
                code="retrieval.source_integrity",
            )
        raw = path.read_bytes()
        try:
            record = parse_model(raw, WorkspaceKnowledgeRevision)
        except (DomainValidationError, ValidationError) as exc:
            raise _retrieval_error(
                f"knowledge revision manifest is invalid: {document_id}@{revision}",
                code="retrieval.source_integrity",
                cause=exc,
            ) from exc
        if (
            raw != canonical_json_bytes(record)
            or record.document.document_id != document_id
            or record.revision != revision
        ):
            raise _retrieval_error(
                f"knowledge revision identity or encoding is invalid: {document_id}@{revision}",
                code="retrieval.source_integrity",
            )
        blob = resolve_confined(self.workspace.root, record.blob_path)
        if not blob.is_file():
            raise _retrieval_error(
                f"knowledge source blob is missing: {record.content_sha256}",
                code="retrieval.source_integrity",
            )
        content = blob.read_bytes()
        if (
            len(content) != record.content_size_bytes
            or sha256_hex(content) != record.content_sha256
        ):
            raise _retrieval_error(
                f"knowledge source blob differs from revision: {document_id}@{revision}",
                code="retrieval.source_integrity",
            )
        _validate_source_content(record.document.media_type, content)
        return record, content

    def freeze_snapshot(
        self,
        *,
        snapshot_id: UUID,
        created_at: datetime,
        revisions: tuple[WorkspaceKnowledgeRevision, ...],
    ) -> WorkspaceKnowledgeSnapshot:
        """Publish a sealed snapshot after rechecking every committed revision under the lock.

        Args:
            snapshot_id: Caller-owned UUID identifying the immutable snapshot.
            created_at: UTC snapshot creation metadata.
            revisions: Nonempty canonical revision tuple, with one revision per document.

        Returns:
            Sealed snapshot; exact manifest replay is accepted.

        Raises:
            KnowledgeRetrievalError: A revision differs from published source or the snapshot target
                conflicts.
            pydantic.ValidationError: Snapshot metadata or revision ordering violates the contract.
        """
        snapshot = WorkspaceKnowledgeSnapshot.create(
            snapshot_id=snapshot_id,
            created_at=created_at,
            revisions=revisions,
        )
        content = canonical_json_bytes(snapshot)
        relative = self._snapshot_relative(snapshot_id)
        with self.workspace.lock():
            for revision in revisions:
                published, _ = self.read_revision(
                    revision.document.document_id,
                    revision.revision,
                )
                if published != revision:
                    raise _retrieval_error(
                        "snapshot revision differs from authoritative publication",
                        code="retrieval.source_integrity",
                    )
            target = make_confined_parents(self.workspace.root, relative)
            target = resolve_confined(self.workspace.root, relative)
            self._publish_or_replay(
                target,
                content,
                conflict_code="retrieval.source_integrity",
            )
        return snapshot

    def read_snapshot(self, snapshot_id: UUID) -> WorkspaceKnowledgeSnapshot:
        """Verify a canonical snapshot and every revision committed by it.

        Args:
            snapshot_id: Exact immutable snapshot UUID.

        Returns:
            Validated snapshot whose child revisions match authoritative published records.

        Raises:
            KnowledgeRetrievalError: The snapshot or a child is absent, malformed or inconsistent.
        """
        relative = self._snapshot_relative(snapshot_id)
        path = resolve_confined(self.workspace.root, relative)
        if not path.is_file():
            raise _retrieval_error(
                f"knowledge snapshot does not exist: {snapshot_id}",
                code="retrieval.source_integrity",
            )
        raw = path.read_bytes()
        try:
            snapshot = parse_model(raw, WorkspaceKnowledgeSnapshot)
        except (DomainValidationError, ValidationError) as exc:
            raise _retrieval_error(
                f"knowledge snapshot is invalid: {snapshot_id}",
                code="retrieval.source_integrity",
                cause=exc,
            ) from exc
        if raw != canonical_json_bytes(snapshot) or snapshot.snapshot_id != snapshot_id:
            raise _retrieval_error(
                f"knowledge snapshot identity or encoding is invalid: {snapshot_id}",
                code="retrieval.source_integrity",
            )
        for expected in snapshot.revisions:
            actual, _ = self.read_revision(
                expected.document.document_id,
                expected.revision,
            )
            if actual != expected:
                raise _retrieval_error(
                    "snapshot child differs from its committed revision",
                    code="retrieval.source_integrity",
                )
        return snapshot


@dataclass(frozen=True, slots=True)
class _Section:
    heading_path: tuple[str, ...]
    heading: str
    body: str
    start_line: int
    end_line: int
    body_char_start: int
    body_char_end: int


@dataclass(frozen=True, slots=True)
class _Chunk:
    document_id: str
    revision: int
    ordinal: int
    title: str
    heading: str
    heading_path: tuple[str, ...]
    body: str
    start_line: int
    end_line: int
    character_start: int
    character_end: int
    utf8_byte_start: int
    utf8_byte_end: int
    chunk_id: str
    chunk_hash: str
    citation: KnowledgeCitation
    namespace: KnowledgeNamespace
    access_class: KnowledgeAccessClass
    available_at: datetime
    expires_at: datetime | None


def _markdown_sections(text: str, title: str) -> tuple[_Section, ...]:
    lines = text.splitlines()
    line_starts = [0]
    line_starts.extend(index + 1 for index, character in enumerate(text) if character == "\n")
    headings: list[str] = []
    current_heading = ""
    current_start = 1
    body_start_line = 1
    body: list[str] = []
    sections: list[_Section] = []

    def flush(end_line: int) -> None:
        nonlocal body
        joined = "\n".join(body)
        raw = joined.strip()
        if raw or current_heading:
            if raw:
                leading = len(joined) - len(joined.lstrip())
                body_char_start = line_starts[body_start_line - 1] + leading
                body_char_end = body_char_start + len(raw)
            else:
                heading_line = lines[current_start - 1]
                heading_offset = heading_line.find(current_heading)
                if heading_offset < 0:
                    raise AssertionError("Markdown heading is absent from its source line")
                body_char_start = line_starts[current_start - 1] + heading_offset
                body_char_end = body_char_start + len(current_heading)
            sections.append(
                _Section(
                    heading_path=tuple(headings),
                    heading=current_heading,
                    body=raw or current_heading or title,
                    start_line=current_start,
                    end_line=max(current_start, end_line),
                    body_char_start=body_char_start,
                    body_char_end=body_char_end,
                )
            )
        body = []

    for number, line in enumerate(lines, start=1):
        match = _HEADING.match(line)
        if match is None:
            body.append(line)
            continue
        flush(number - 1)
        level = len(match.group(1))
        heading = match.group(2).strip()
        headings = headings[: level - 1]
        headings.append(heading)
        current_heading = heading
        current_start = number
        body_start_line = number + 1
    flush(len(lines) or 1)
    if not sections:
        sections.append(
            _Section(
                heading_path=(),
                heading="",
                body=text.strip() or title,
                start_line=1,
                end_line=max(1, len(lines)),
                body_char_start=0,
                body_char_end=len(text),
            )
        )
    return tuple(sections)


def _paragraph_chunks(text: str, *, size: int, overlap: int) -> tuple[str, ...]:
    paragraphs = tuple(item.strip() for item in re.split(r"\n\s*\n", text) if item.strip())
    if not paragraphs:
        return ()
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(candidate) <= size:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = current[-overlap:] if overlap else ""
        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        while len(candidate) > size:
            chunks.append(candidate[:size])
            candidate = candidate[size - overlap :] if overlap else candidate[size:]
        current = candidate
    if current:
        chunks.append(current)
    return tuple(chunks)


@dataclass(frozen=True, slots=True)
class _ParagraphChunkSlice:
    body: str
    char_start: int
    char_end: int


def _paragraph_spans(text: str) -> tuple[tuple[str, int, int], ...]:
    spans: list[tuple[str, int, int]] = []
    raw_start = 0
    for separator in re.finditer(r"\n\s*\n", text):
        raw_end = separator.start()
        raw = text[raw_start:raw_end]
        paragraph = raw.strip()
        if paragraph:
            char_start = raw_start + len(raw) - len(raw.lstrip())
            char_end = raw_start + len(raw.rstrip())
            spans.append((paragraph, char_start, char_end))
        raw_start = separator.end()
    raw = text[raw_start:]
    paragraph = raw.strip()
    if paragraph:
        char_start = raw_start + len(raw) - len(raw.lstrip())
        char_end = raw_start + len(raw.rstrip())
        spans.append((paragraph, char_start, char_end))
    return tuple(spans)


def _paragraph_chunk_slices(
    text: str,
    *,
    size: int,
    overlap: int,
) -> tuple[_ParagraphChunkSlice, ...]:
    """Legacy-identical paragraph chunks, each covering an exact source range.

    The bodies are byte-for-byte what ``_paragraph_chunks`` produces, because
    they are what gets embedded and their hashes are load-bearing. What is new
    is that every chunk also carries the source range it came from.

    A body is not always a literal slice: joining two paragraphs inserts a
    synthetic ``\n\n`` where the source may have had different whitespace. The
    range is therefore the *covering* range, from the first contributing source
    position to the last. For a long paragraph cut into fixed-size pieces --
    which is what an inline-XBRL filing is, and so the overwhelming majority of
    this corpus -- the cover is the slice exactly.
    """

    paragraphs = _paragraph_spans(text)
    if not paragraphs:
        return ()
    slices: list[_ParagraphChunkSlice] = []
    current = ""
    current_start: int | None = None
    current_end: int | None = None

    def emit(body: str, char_start: int, char_end: int) -> None:
        if not (0 <= char_start < char_end <= len(text)):
            raise ValueError("paragraph chunk range escapes its canonical source")
        slices.append(_ParagraphChunkSlice(body=body, char_start=char_start, char_end=char_end))

    for paragraph, paragraph_start, paragraph_end in paragraphs:
        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        candidate_start = paragraph_start if current_start is None else current_start
        if len(candidate) <= size:
            current = candidate
            current_start = candidate_start
            current_end = paragraph_end
            continue
        if current:
            assert current_start is not None and current_end is not None
            emit(current, current_start, current_end)
            if overlap:
                current = current[-overlap:]
                # The tail is the end of what was just emitted, so its source
                # start is bounded below by that chunk's own start.
                current_start = max(current_start, current_end - len(current))
            else:
                current = ""
                current_start = None
        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        candidate_start = paragraph_start if current_start is None else current_start
        # How much of `candidate` is carried-over tail rather than this
        # paragraph, so a cut inside the paragraph maps back to a source offset.
        # `carried` is how many leading characters of `candidate` came from the
        # previous chunk's tail rather than from this paragraph, and
        # `consumed` how far into this paragraph the cursor has already moved.
        # Together they turn a cut inside `candidate` into a source position.
        carried = len(candidate) - len(paragraph)
        consumed = 0
        step = size - overlap if overlap else size
        while len(candidate) > size:
            if carried:
                reached = max(0, size - carried)
                end_position = min(
                    paragraph_end,
                    max(candidate_start + 1, paragraph_start + consumed + reached),
                )
            else:
                # Wholly inside this paragraph, so the cover is the slice exactly.
                end_position = min(paragraph_end, candidate_start + size)
            emit(candidate[:size], candidate_start, end_position)
            candidate = candidate[step:]
            if step <= carried:
                candidate_start += step
                carried -= step
            else:
                consumed += step - carried
                carried = 0
                candidate_start = min(
                    paragraph_start + consumed, max(paragraph_start, paragraph_end - 1)
                )
        current = candidate
        current_start = candidate_start
        current_end = paragraph_end
    if current:
        assert current_start is not None and current_end is not None
        emit(current, current_start, current_end)
    if tuple(item.body for item in slices) != _paragraph_chunks(text, size=size, overlap=overlap):
        raise AssertionError("position-aware chunking differs from fixed-v1")
    return tuple(slices)


def _source_sections(
    revision: WorkspaceKnowledgeRevision,
    content: bytes,
) -> tuple[_Section, ...]:
    text = content.decode("utf-8")
    if revision.document.media_type is KnowledgeMediaType.MARKDOWN:
        return _markdown_sections(text, revision.document.title)
    parsed = strict_json_loads(text)
    canonical = canonical_json_bytes(parsed).decode("utf-8")
    return (
        _Section(
            heading_path=("$",),
            heading="$",
            body=canonical,
            start_line=1,
            end_line=max(1, len(text.splitlines())),
            body_char_start=0,
            body_char_end=len(canonical),
        ),
    )


def _utf8_byte_prefix(text: str) -> tuple[int, ...] | None:
    """Cumulative UTF-8 byte offset per character position, or None for ASCII.

    Built once per document, because encoding a prefix per chunk would be
    quadratic and this corpus reaches 9,725 chunks over 4.2 million characters.
    Pure ASCII needs no table at all: a character offset is already the byte
    offset, and skipping it keeps the per-query chunk rebuild cheap.
    """

    if text.isascii():
        return None
    return (0, *accumulate(len(character.encode("utf-8")) for character in text))


def _build_chunks(
    snapshot: WorkspaceKnowledgeSnapshot,
    library: WorkspaceKnowledgeLibrary,
    spec: LexicalIndexSpec,
) -> tuple[_Chunk, ...]:
    chunks: list[_Chunk] = []
    for revision in snapshot.revisions:
        authoritative, content = library.read_revision(
            revision.document.document_id,
            revision.revision,
        )
        if authoritative != revision:
            raise _retrieval_error(
                "snapshot revision differs during chunk construction",
                code="retrieval.source_integrity",
            )
        source_text = content.decode("utf-8")
        byte_prefix = _utf8_byte_prefix(source_text)
        ordinal = 0
        for section in _source_sections(revision, content):
            addressable = source_text[section.body_char_start : section.body_char_end] == (
                section.body
            )
            for slice_ in _paragraph_chunk_slices(
                section.body,
                size=spec.chunk_size_codepoints,
                overlap=spec.chunk_overlap_codepoints,
            ):
                normalized = _normalize_text(slice_.body)
                if not normalized:
                    continue
                ordinal += 1
                chunk_hash = sha256_hex(normalized.encode("utf-8"))
                if addressable:
                    character_start = section.body_char_start + slice_.char_start
                    character_end = section.body_char_start + slice_.char_end
                    start_line = source_text.count("\n", 0, character_start) + 1
                    end_line = start_line + source_text.count("\n", character_start, character_end)
                else:
                    # A canonicalized section body is not a slice of the sealed
                    # bytes, so the whole section stays the addressable range.
                    character_start = section.body_char_start
                    character_end = section.body_char_end
                    start_line = section.start_line
                    end_line = section.end_line
                if byte_prefix is None:
                    byte_start, byte_end = character_start, character_end
                else:
                    byte_start = byte_prefix[character_start]
                    byte_end = byte_prefix[character_end]
                identity_payload = {
                    "document_id": revision.document.document_id,
                    "revision": revision.revision,
                    "content_sha256": revision.content_sha256,
                    "heading_path": section.heading_path,
                    "ordinal": ordinal,
                    "start_line": start_line,
                    "end_line": end_line,
                    "character_start": character_start,
                    "character_end": character_end,
                    "chunk_hash": chunk_hash,
                }
                chunk_id = _canonical_hash(identity_payload)
                citation = KnowledgeCitation(
                    document_id=revision.document.document_id,
                    revision=revision.revision,
                    snapshot_id=snapshot.snapshot_id,
                    snapshot_logical_hash=snapshot.logical_hash,
                    content_sha256=revision.content_sha256,
                    heading_path=section.heading_path,
                    start_line=start_line,
                    end_line=end_line,
                    character_start=character_start,
                    character_end=character_end,
                    utf8_byte_start=byte_start,
                    utf8_byte_end=byte_end,
                    chunk_id=chunk_id,
                    chunk_hash=chunk_hash,
                )
                chunks.append(
                    _Chunk(
                        document_id=revision.document.document_id,
                        revision=revision.revision,
                        ordinal=ordinal,
                        title=revision.document.title,
                        heading=section.heading,
                        heading_path=section.heading_path,
                        body=slice_.body,
                        start_line=start_line,
                        end_line=end_line,
                        character_start=character_start,
                        character_end=character_end,
                        utf8_byte_start=byte_start,
                        utf8_byte_end=byte_end,
                        chunk_id=chunk_id,
                        chunk_hash=chunk_hash,
                        citation=citation,
                        namespace=revision.document.namespace,
                        access_class=revision.source.access_class,
                        available_at=revision.source.available_at,
                        expires_at=revision.source.expires_at,
                    )
                )
    return tuple(chunks)


def _projection_row(chunk: _Chunk) -> dict[str, Any]:
    normalized_title = _normalize_text(chunk.title)
    normalized_heading = _normalize_text(chunk.heading)
    normalized_body = _normalize_text(chunk.body)
    identifier_terms = " ".join(_compiled_terms(chunk.document_id))
    title_terms = " ".join(_compiled_terms(normalized_title))
    heading_terms = " ".join(_compiled_terms(normalized_heading))
    body_terms = " ".join(_compiled_terms(normalized_body))
    cjk = " ".join(
        sorted(set(_cjk_terms(f"{chunk.document_id} {chunk.title} {chunk.heading} {chunk.body}")))
    )
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "revision": chunk.revision,
        "ordinal": chunk.ordinal,
        "namespace": chunk.namespace.value,
        "access_class": chunk.access_class.value,
        "available_at": _utc_text(chunk.available_at),
        "expires_at": _utc_text(chunk.expires_at) if chunk.expires_at else None,
        "title_raw": chunk.title,
        "heading_raw": chunk.heading,
        "body_raw": chunk.body,
        "identifier_norm": _normalize_text(chunk.document_id),
        "title_norm": normalized_title,
        "heading_norm": normalized_heading,
        "identifier_terms": identifier_terms,
        "title_terms": title_terms,
        "heading_terms": heading_terms,
        "body_terms": body_terms,
        "cjk_terms": cjk,
        "citation_json": canonical_json_bytes(chunk.citation).decode("utf-8"),
        "chunk_hash": chunk.chunk_hash,
    }


def _execute(
    connection: sqlite3.Connection,
    statement: str,
    parameters: Sequence[Any] = (),
) -> None:
    with closing(connection.execute(statement, parameters)):
        pass


__all__ = [
    "WorkspaceKnowledgeLibrary",
]
