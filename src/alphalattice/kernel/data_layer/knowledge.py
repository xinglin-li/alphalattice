"""Knowledge's shared SQLite cache, written through the serving Host's log.

Vectors are encoded outside the log writer and committed in batches. The cache maps a
chunk digest, chunking recipe and model to its immutable batch and position; a second
workspace or rebuild reads that same vector. Each int8 batch has one scale per dimension.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
import zipfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from threading import RLock
from typing import Any, Final, Literal, Protocol, Self

import numpy as np
from numpy.typing import NDArray

from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer.chain import canonical, digest
from alphalattice.kernel.data_layer.encoders import DEADLINE_SECONDS, SHORTLIST
from alphalattice.kernel.data_layer.log import HostLog, uuid7
from alphalattice.kernel.resources import carry_resource_lease, cpu_width, worker_width

type Precision = Literal["float16", "int8"]
type Matrix = NDArray[np.float32]
type EvidenceVersions = frozenset[tuple[str, str]]

_HAN = re.compile(r"([\u4e00-\u9fff])")
_TERMS = re.compile(r"[\u4e00-\u9fff]+|[A-Za-z0-9_]+")
CHUNK_SPLITTER: Final = ("paragraph-sentence-word", "1")
_PARAGRAPH = re.compile(r"\r?\n(?:[^\S\r\n]*\r?\n)*")
_SENTENCE = re.compile(r"(?<=[.!?])(?=\s|$)|(?<=[\u3002\uff01\uff1f])")
_SPACE = re.compile(r"\s+")
# From this many indexed chunks the lexical query leaves out the words at least half of them
# hold. Matching costs about 1 microsecond a chunk (98,000 matched rows took 72-105 ms), so a
# smaller index scores them all in under 10 ms and keeps every word.
COMMON_WORDS_FROM = 10_000
# The source kind a sealed selection reads: evidence review's references (read through
# `recall.search_references`), whose hit set its reference spec binds to its own inputs. Such a
# read keeps every word and, in English, its cross-encoder rerank; recall's reads may leave out
# common words and answer in the fused order.
SEALED_KIND = "reference"


def _spaced(text: str) -> str:
    """Make Han characters visible to unicode61 without changing the source text."""
    return _HAN.sub(r" \1 ", text)


@dataclass(frozen=True)
class EmbeddingRead:
    """Requested vectors in input order, with actual cache-work counts."""

    vectors: Matrix
    embedded: int
    reused: int


@dataclass(frozen=True)
class SourceOrigin:
    """The body's authority, unchanged across workspace membership and cache rebuilds."""

    source_id: str
    version: str
    kind: Literal["reference", "evidence", "experience"]
    known_us: int


def _chunk_slices(
    content: str, capacity: int, overlap: int, *, trim: bool
) -> Iterator[tuple[int, str]]:
    """Cut bounded character windows and return each slice's original start offset."""
    if (
        type(capacity) is not int
        or type(overlap) is not int
        or not 0 <= overlap < capacity
        or type(trim) is not bool
    ):
        raise DataError(
            "the chunk recipe needs a positive character capacity, smaller nonnegative overlap"
            " and a boolean trim setting",
            code="data_layer.knowledge_recipe_invalid",
        )
    start = 0
    while start < len(content):
        if trim and (leading := _SPACE.match(content, start)):
            start = leading.end()
        end = min(len(content), start + capacity)
        if end < len(content):
            # A longer run of newlines separates larger units. Fill the window through
            # the last separator at the highest level that fits and advances the slice.
            paragraph = max(
                (
                    match
                    for match in _PARAGRAPH.finditer(content, start, end)
                    if match.end() > start + overlap
                ),
                key=lambda match: (match[0].count("\n"), match.end()),
                default=None,
            )
            if paragraph is not None:
                end = paragraph.end()
            else:
                for boundary in (_SENTENCE, _SPACE):
                    cut = max(
                        (match.end() for match in boundary.finditer(content, start, end)), default=0
                    )
                    if cut > start + overlap:
                        end = cut
                        break
        text = content[start:end]
        if trim:
            text = text.rstrip()
        if text:
            yield start, text
        if end == len(content):
            break
        start = end - overlap


@dataclass(frozen=True)
class PublishedRevision:
    """Publication metadata and origin; only a reference's body belongs to this log."""

    metadata: dict[str, Any]
    content: str | None
    origin: SourceOrigin


@dataclass(frozen=True)
class Chunk:
    """A source-bound slice; offsets address the original Unicode text."""

    chunk_id: str
    start: int
    end: int
    digest: str


@dataclass(frozen=True)
class SearchHit:
    """A match with the source authority and original slice."""

    source_id: str
    version: str
    ordinal: int
    chunk: Chunk
    score: float
    rowid: int
    channels: tuple[str, ...] = ()
    """The channels that found the chunk: "lexical", "vector" or both, in that order."""


@dataclass(frozen=True)
class RetrievedHit:
    """A scored source range with text verified against the source at this read."""

    hit: SearchHit
    text: str


@dataclass(frozen=True)
class Retrieved:
    """One search's hits and the meaning of their shared score."""

    hits: tuple[RetrievedHit, ...]
    score_type: Literal["LINEAR_FUSION", "CROSS_ENCODER_LOGIT"]
    reranked: int = 0
    """Distinct (query, passage) pairs actually scored, including completed pairs on timeout."""

    rerank_fallback: Literal["DEADLINE_EXCEEDED"] | None = None


def fuse(
    lexical: Sequence[SearchHit], vector: Sequence[SearchHit], *, limit: int = 50
) -> tuple[SearchHit, ...]:
    """Fuse the calibrated top-50 channels, with source row order breaking equal scores."""

    def normalized(hits: Sequence[SearchHit]) -> dict[str, float]:
        if not hits:
            return {}
        low, high = min(hit.score for hit in hits), max(hit.score for hit in hits)
        return {
            hit.chunk.chunk_id: (hit.score - low) / (high - low) if high > low else 1.0
            for hit in hits
        }

    lexical_scores, vector_scores = normalized(lexical), normalized(vector)
    hits = {hit.chunk.chunk_id: hit for hit in (*lexical, *vector)}
    ranked = (
        replace(
            hit,
            score=0.5 * lexical_scores.get(key, 0.0) + 0.5 * vector_scores.get(key, 0.0),
            channels=("lexical",) * (key in lexical_scores) + ("vector",) * (key in vector_scores),
        )
        for key, hit in hits.items()
    )
    return tuple(sorted(ranked, key=lambda hit: (-hit.score, hit.rowid))[:limit])


@dataclass(frozen=True)
class Scope:
    """What one search may return: a workspace's sources known by `cutoff_us`, of `kind` (any when
    None), current (`superseded` False), retired experiences only (True) or both (None), and the
    world evidence versions resolved once at that cutoff for both channels."""

    workspace: str
    cutoff_us: int
    kind: str | None = None
    superseded: bool | None = False
    evidence: EvidenceVersions = frozenset()
    ranges: tuple[tuple[str, int, int], ...] = ()
    """When given, only chunks inside one of these (source id, first ordinal, last ordinal)
    ranges: a residual search within the passages a review has already inspected."""


class VectorPort(Protocol):
    """The Host's replaceable vector search over its admitted source versions."""

    recipe_id: str

    def search(
        self, query: Matrix, scope: Scope, *, limit: int = 50, per_source: bool = False
    ) -> tuple[SearchHit, ...]:
        """The top `limit` chunks in `scope` by similarity to `query` (each source's top `limit`
        when `per_source`), ties in source row order."""
        ...

    def close(self) -> None:
        """Release what the index holds."""
        ...


class WordCounts:
    """How many indexed chunks hold each word a query has asked for: a word at least half of them
    hold is common.

    FTS5's bm25 clamps a common word's idf to almost nothing, so it ranks no chunk above another,
    yet matching it makes the query score nearly every chunk: measured at 100,000 chunks, an OR
    of such words scored ~98,000 rows for 72-105 ms. Each word's count is read once and kept
    until a chunk is added; an index smaller than `COMMON_WORDS_FROM` has no common word."""

    def __init__(self) -> None:
        """Start with no counts: the first query reads the index's chunk count."""
        self.at: int | None = None
        """The last chunk row the counts were taken at."""
        self.documents = 0
        self.holding: dict[str, int] = {}

    def common(
        self,
        last: int,
        documents: Callable[[], int],
        holding: Callable[[str], int],
        words: list[str],
    ) -> frozenset[str]:
        """The common `words`, given the index's last chunk row and readers of its chunk count
        and of the chunks holding one word."""
        if last != self.at:
            self.documents, self.holding, self.at = documents(), {}, last
        if self.documents < COMMON_WORDS_FROM:
            return frozenset()
        for word in words:
            if word not in self.holding:
                self.holding[word] = holding(word)
        return frozenset(word for word in words if 2 * self.holding[word] >= self.documents)


class KnowledgeStore:
    """The Host's shared source and embedding store; never opens another SQLite writer."""

    def __init__(
        self, log: HostLog, *, evidence_asof: Callable[[int], EvidenceVersions] | None = None
    ) -> None:
        """Use the serving Host's writer and its shared cache-build ownership."""
        self.log = log
        self.evidence_asof = evidence_asof
        self._words = WordCounts()

    def scope(
        self,
        *,
        workspace: str,
        cutoff_us: int,
        kind: str | None = None,
        superseded: bool | None = False,
    ) -> Scope:
        """A search's scope, its world evidence resolved once before either channel selects."""
        return Scope(
            workspace,
            cutoff_us,
            kind,
            superseded,
            self._visible_evidence(workspace, cutoff_us, kind, superseded),
        )

    def _visible_evidence(
        self, workspace: str, cutoff_us: int, kind: str | None, superseded: bool | None
    ) -> EvidenceVersions:
        if superseded is True or kind in ("experience", "reference"):
            return frozenset()
        if self.evidence_asof is not None:
            return self.evidence_asof(cutoff_us)
        with self.log.reading() as con:
            # from the sources: the narrow source table's kinds, then only an evidence source's
            # versions in the workspace (2.8 ms at 1,000,000 chunks, where walking the
            # workspace's every version took 30 ms)
            found = con.execute(
                "SELECT 1 FROM knowledge_source s"
                " CROSS JOIN knowledge_source_version v ON v.source_id = s.source_id"
                " CROSS JOIN knowledge_workspace_source w"
                " ON w.version_id = v.version_id AND w.workspace_id = ?"
                " WHERE s.kind = 'evidence' LIMIT 1",
                (workspace,),
            ).fetchone()
        if found:
            raise DataError(
                "world is unavailable for evidence retrieval",
                code="data_layer.knowledge_source_missing",
            )
        return frozenset()

    def publish_experience(
        self, *, workspace_id: str, experience_id: str, version: int, blobs: Path
    ) -> str:
        """Admit the log's experience identity and its existing, verified rendered body."""
        content, known_us = self._experience_body(
            workspace_id=workspace_id, experience_id=experience_id, version=version, blobs=blobs
        )
        return self.publish_revision(
            workspace_id=workspace_id,
            metadata={"document": {"document_id": experience_id}, "revision": version},
            content=content,
            origin=SourceOrigin(experience_id, str(version), "experience", known_us),
        )

    def experience_body(
        self, *, workspace_id: str, experience_id: str, version: int, blobs: Path, cutoff_us: int
    ) -> str:
        """Read an experience's verified original body only when it was known at the cutoff."""
        content, _ = self._experience_body(
            workspace_id=workspace_id,
            experience_id=experience_id,
            version=version,
            blobs=blobs,
            cutoff_us=cutoff_us,
        )
        return content

    def _experience_body(
        self,
        *,
        workspace_id: str,
        experience_id: str,
        version: int,
        blobs: Path,
        cutoff_us: int | None = None,
    ) -> tuple[str, int]:
        with self.log.reading() as con:
            row = con.execute(
                "SELECT v.body, v.size, e.recorded_us FROM experience_version v"
                " JOIN event e ON e.seq = v.recorded_seq"
                " WHERE v.experience_id = ? AND v.version = ? AND e.workspace_id = ?"
                " AND (? IS NULL OR e.recorded_us <= ?)",
                (experience_id, version, workspace_id, cutoff_us, cutoff_us),
            ).fetchone()
        if row is None or re.fullmatch(r"blake3:[0-9a-f]{64}", row[0]) is None:
            raise DataError(
                f"experience {experience_id}@{version} is not held in workspace {workspace_id}",
                code="data_layer.knowledge_source_missing",
            )
        try:
            body = (blobs / row[0].removeprefix("blake3:")).read_bytes()
            content = body.decode("utf-8")
        except (OSError, UnicodeError) as error:
            raise DataError(
                f"experience {experience_id}@{version} has no readable body",
                code="data_layer.knowledge_source_missing",
            ) from error
        if len(body) != row[1] or digest(body) != row[0]:
            raise DataError(
                f"experience {experience_id}@{version} differs from its body",
                code="data_layer.knowledge_source_conflict",
            )
        return content, int(row[2])

    def publish_revision(
        self,
        *,
        workspace_id: str,
        metadata: Mapping[str, Any],
        content: str,
        origin: SourceOrigin | None = None,
    ) -> str:
        """Publish one immutable source version and its workspace edge through the log.

        External bodies are used only to check their digest; they never enter event text.
        An exact replay keeps the ids it first received; conflicting publication is refused.
        """
        document, revision = str(metadata["document"]["document_id"]), int(metadata["revision"])
        body_digest = digest(content.encode())
        serialized = canonical(dict(metadata)).decode()
        with self.log.unit() as con:
            held = con.execute(
                "SELECT v.version_id, v.body_digest, w.metadata, v.source_id, v.version,"
                " s.kind, v.known_us FROM knowledge_workspace_source w"
                " JOIN knowledge_source_version v USING (version_id)"
                " JOIN knowledge_source s USING (source_id)"
                " WHERE w.workspace_id = ? AND w.document_id = ? AND w.revision = ?",
                (workspace_id, document, revision),
            ).fetchone()
            if held:
                actual = SourceOrigin(held[3], held[4], held[5], held[6])
                if held[1] != body_digest or held[2] != serialized or (origin and origin != actual):
                    raise DataError(
                        f"knowledge revision conflicts: {document}@{revision}",
                        code="data_layer.knowledge_publication_conflict",
                    )
                return str(held[0])
            if origin is None:
                prior = con.execute(
                    "SELECT v.source_id FROM knowledge_workspace_source w"
                    " JOIN knowledge_source_version v USING (version_id)"
                    " WHERE w.workspace_id = ? AND w.document_id = ? LIMIT 1",
                    (workspace_id, document),
                ).fetchone()
                origin = SourceOrigin(
                    str(prior[0]) if prior else uuid7(),
                    str(revision),
                    "reference",
                    self.log.clock(),
                )
            if (
                not origin.source_id
                or not origin.version
                or origin.kind not in ("reference", "evidence", "experience")
            ):
                raise DataError(
                    "a source requires its authority's id, version and kind",
                    code="data_layer.knowledge_source_missing",
                )
            source = con.execute(
                "SELECT kind FROM knowledge_source WHERE source_id = ?", (origin.source_id,)
            ).fetchone()
            if source is not None and source[0] != origin.kind:
                raise DataError(
                    "source kind conflicts with its authority",
                    code="data_layer.knowledge_publication_conflict",
                )
            simultaneous = con.execute(
                "SELECT 1 FROM knowledge_source_version WHERE source_id = ? AND known_us = ?"
                " AND version != ?",
                (origin.source_id, origin.known_us, origin.version),
            ).fetchone()
            if simultaneous:
                raise DataError(
                    "source versions share a known time",
                    code="data_layer.knowledge_publication_conflict",
                )
            existing = con.execute(
                "SELECT v.version_id, v.known_us, v.body_digest, s.kind"
                " FROM knowledge_source_version v JOIN knowledge_source s USING (source_id)"
                " WHERE v.source_id = ? AND v.version = ?",
                (origin.source_id, origin.version),
            ).fetchone()
            if existing and tuple(existing)[1:] != (
                origin.known_us,
                body_digest,
                origin.kind,
            ):
                raise DataError(
                    f"source version conflicts: {origin.source_id}@{origin.version}",
                    code="data_layer.knowledge_publication_conflict",
                )
            version_id = str(existing[0]) if existing else uuid7()
            attrs = {
                "source_id": origin.source_id,
                "kind": origin.kind,
                "version_id": version_id,
                "version": origin.version,
                "known_us": origin.known_us,
                "body_digest": body_digest,
                "membership_id": uuid7(),
                "document_id": document,
                "revision": revision,
            }
            payload: dict[str, Any] = {"metadata": dict(metadata)}
            if origin.kind == "reference":
                payload["content"] = content
            self.log.record(
                "knowledge.revision_published",
                ("knowledge_source", origin.source_id),
                attrs,
                text=canonical(payload).decode(),
                workspace=workspace_id,
            )
            return version_id

    def read_revision(self, workspace: str, document: str, revision: int) -> PublishedRevision:
        """Read one workspace's immutable metadata and the source authority it names."""
        with self.log.reading() as con:
            row = con.execute(
                "SELECT w.metadata, t.text, v.source_id, v.version, s.kind, v.known_us"
                " FROM knowledge_workspace_source w"
                " JOIN knowledge_source_version v USING (version_id)"
                " JOIN knowledge_source s USING (source_id)"
                " LEFT JOIN event_text t ON t.seq = v.body_seq"
                " WHERE w.workspace_id = ? AND w.document_id = ? AND w.revision = ?",
                (workspace, document, revision),
            ).fetchone()
        if row is None:
            raise DataError(
                f"no knowledge revision {document}@{revision} in workspace {workspace}",
                code="data_layer.knowledge_revision_missing",
            )
        content = None if row[1] is None else json.loads(row[1])["content"]
        return PublishedRevision(json.loads(row[0]), content, SourceOrigin(*row[2:]))

    def retire(self, source_id: str, *, known_us: int) -> int:
        """Record a source's retirement at the authority's known time, once."""
        with self.log.unit() as con:
            if (
                con.execute(
                    "SELECT 1 FROM knowledge_source WHERE source_id = ?", (source_id,)
                ).fetchone()
                is None
            ):
                raise DataError(
                    f"no knowledge source {source_id}", code="data_layer.knowledge_source_missing"
                )
            held = con.execute(
                "SELECT seq FROM event WHERE type = 'knowledge.source_retired' AND subject_id = ?"
                " AND json_extract(attrs, '$.known_us') = ?",
                (source_id, known_us),
            ).fetchone()
            if held is not None:
                return int(held[0])
            return self.log.record(
                "knowledge.source_retired", ("knowledge_source", source_id), {"known_us": known_us}
            )

    def publish_snapshot(self, *, workspace_id: str, metadata: Mapping[str, Any]) -> str:
        """Seal a snapshot only after every revision matches this workspace's publication."""
        reference = str(metadata["snapshot_id"])
        with self.log.unit() as con:
            held = con.execute(
                "SELECT s.snapshot_id, t.text FROM knowledge_snapshot s"
                " JOIN event_text t ON t.seq = s.metadata_seq"
                " WHERE s.workspace_id = ? AND s.reference_id = ?",
                (workspace_id, reference),
            ).fetchone()
            if held:
                if json.loads(held[1])["metadata"] != dict(metadata):
                    raise DataError(
                        f"knowledge snapshot conflicts: {reference}",
                        code="data_layer.knowledge_publication_conflict",
                    )
                return str(held[0])
            count, matches, links = con.execute(
                "SELECT count(v.version_id), min(w.metadata = json(j.value)),"
                " json_group_array(json_object('membership_id', uuid7(),"
                " 'version_id', v.version_id))"
                " FROM json_each(?) j LEFT JOIN knowledge_workspace_source w"
                " ON w.workspace_id = ?"
                " AND w.document_id = json_extract(j.value, '$.document.document_id')"
                " AND w.revision = json_extract(j.value, '$.revision')"
                " LEFT JOIN knowledge_source_version v USING (version_id)",
                (canonical(metadata["revisions"]).decode(), workspace_id),
            ).fetchone()
            if not count or count != len(metadata["revisions"]) or not matches:
                raise DataError(
                    "snapshot revisions differ from their workspace's publications",
                    code="data_layer.knowledge_publication_conflict",
                )
            members = json.loads(links)
            snapshot = uuid7()
            self.log.record(
                "knowledge.snapshot_published",
                ("knowledge_snapshot", snapshot),
                {"reference_id": reference},
                workspace=workspace_id,
                text=canonical({"metadata": dict(metadata), "members": members}).decode(),
            )
            return snapshot

    def read_snapshot(self, workspace: str, reference: str) -> dict[str, Any]:
        """Read the snapshot through its workspace boundary, never a workspace file."""
        with self.log.reading() as con:
            row = con.execute(
                "SELECT t.text FROM knowledge_snapshot s"
                " JOIN event_text t ON t.seq = s.metadata_seq"
                " WHERE s.workspace_id = ? AND s.reference_id = ?",
                (workspace, reference),
            ).fetchone()
        if row is None:
            raise DataError(
                f"no knowledge snapshot {reference} in workspace {workspace}",
                code="data_layer.knowledge_snapshot_missing",
            )
        return dict(json.loads(row[0])["metadata"])

    def recipe(self, spec: Mapping[str, Any]) -> str:
        """Register complete pinned chunking settings once and return their Host-minted id."""
        text = canonical(dict(spec)).decode("utf-8")
        with self.log.unit() as con:
            con.execute(
                "INSERT INTO knowledge_recipe VALUES (?, ?) ON CONFLICT (spec) DO NOTHING",
                (uuid7(), text),
            )
            return str(
                con.execute(
                    "SELECT recipe_id FROM knowledge_recipe WHERE spec = ?", (text,)
                ).fetchone()[0]
            )

    def index_chunks(self, *, version_id: str, recipe_id: str, content: str) -> tuple[Chunk, ...]:
        """Index a verified source version once, with the recipe's original character offsets."""
        with self.log.reading() as con:
            source = con.execute(
                "SELECT body_digest FROM knowledge_source_version WHERE version_id = ?",
                (version_id,),
            ).fetchone()
            if source is None:
                raise DataError(
                    "a chunk requires an admitted source version",
                    code="data_layer.knowledge_source_missing",
                )
            if source[0] != digest(content.encode()):
                raise DataError(
                    "the source body differs from its version",
                    code="data_layer.knowledge_source_conflict",
                )
            recipe = con.execute(
                "SELECT spec FROM knowledge_recipe WHERE recipe_id = ?", (recipe_id,)
            ).fetchone()
            if recipe is None:
                raise DataError(
                    "the chunk recipe is missing", code="data_layer.knowledge_recipe_missing"
                )
            spec = json.loads(recipe[0])
            if (spec.get("splitter"), spec.get("version")) != CHUNK_SPLITTER:
                raise DataError(
                    "the chunk recipe differs from the current splitter; index under a new recipe",
                    code="data_layer.knowledge_recipe_invalid",
                )
        # Splitting and hashing may be large; the Host's one SQLite writer stays free while
        # immutable source and recipe inputs are transformed.
        slices = list(
            _chunk_slices(content, spec.get("capacity"), spec.get("overlap"), trim=spec.get("trim"))
        )
        expected = [(start, start + len(text), digest(text.encode())) for start, text in slices]
        tokens = [_spaced(text) for _, text in slices]
        with self.log.unit() as con:
            rows = con.execute(
                "SELECT chunk_id, char_start, char_end, chunk_digest FROM knowledge_chunk"
                " WHERE version_id = ? AND recipe_id = ? ORDER BY ordinal",
                (version_id, recipe_id),
            ).fetchall()
            if rows and [tuple(row[1:]) for row in rows] != expected:
                raise DataError(
                    "the indexed source chunks differ", code="data_layer.knowledge_source_conflict"
                )
            if not rows:
                rows = [(uuid7(), *item) for item in expected]
                con.executemany(
                    "INSERT INTO knowledge_chunk VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [
                        (
                            row[0],
                            version_id,
                            recipe_id,
                            ordinal,
                            row[1],
                            row[2],
                            row[3],
                        )
                        for ordinal, row in enumerate(rows)
                    ],
                )
                rowids = con.execute(
                    "SELECT rowid FROM knowledge_chunk WHERE version_id = ? AND recipe_id = ?"
                    " ORDER BY ordinal",
                    (version_id, recipe_id),
                ).fetchall()
                con.executemany(
                    "INSERT INTO knowledge_fts(rowid, fts_text) VALUES (?, ?)",
                    [(rowid[0], text) for rowid, text in zip(rowids, tokens, strict=True)],
                )
            return tuple(Chunk(*row) for row in rows)

    def optimize(self) -> None:
        """Compact the FTS segments after a bulk indexing build."""
        with self.log.unit() as con:
            con.execute("INSERT INTO knowledge_fts(knowledge_fts) VALUES ('optimize')")

    def lexical(
        self,
        query: str,
        scope: Scope,
        *,
        recipe_id: str | None = None,
        limit: int = 50,
        per_source: bool = False,
    ) -> tuple[SearchHit, ...]:
        """Search the FTS index within an admitted workspace and source-time boundary: the best
        `limit` matches, or each source's best `limit` when `per_source`. Outside the precise mode
        (`per_source`) and a sealed read (`SEALED_KIND`), the words most chunks hold are left out
        (`WordCounts`); those two keep every word, as their inputs name no index-wide count."""
        terms = _TERMS.findall(query)
        if not terms or limit < 1:
            return ()
        common: frozenset[str] = frozenset()
        if not per_source and scope.kind != SEALED_KIND:
            with self.log.reading() as con:
                common = self._words.common(
                    con.execute("SELECT coalesce(max(rowid), 0) FROM knowledge_chunk").fetchone()[
                        0
                    ],
                    lambda: con.execute("SELECT count(*) FROM knowledge_fts_docsize").fetchone()[0],
                    lambda word: con.execute(
                        "SELECT count(*) FROM knowledge_fts WHERE knowledge_fts MATCH ?",
                        (f'"{word}"',),
                    ).fetchone()[0],
                    [*(t.lower() for t in terms if not _HAN.search(t)), *_HAN.findall(query)],
                )
        terms = [term for term in terms if term.lower() not in common]
        if not terms:  # every word is one most chunks hold: the vector channel ranks alone
            return ()
        cutoff_us, kind, superseded = scope.cutoff_us, scope.kind, scope.superseded
        world = json.dumps([f"{document}/{version}" for document, version in scope.evidence])
        expression = " OR ".join('"' + _spaced(term).strip() + '"' for term in terms)
        sql = (
            "SELECT s.source_id, v.version, c.ordinal, c.chunk_id, c.char_start, c.char_end,"
            " c.chunk_digest, bm25(knowledge_fts, 1.0) AS rank, c.rowid"
            " FROM knowledge_fts JOIN knowledge_chunk c ON c.rowid = knowledge_fts.rowid"
            " JOIN knowledge_source_version v ON v.version_id = c.version_id"
            " JOIN knowledge_source s ON s.source_id = v.source_id"
            " LEFT JOIN experience_made m ON s.kind = 'experience'"
            " AND m.experience_id = s.source_id"
            " LEFT JOIN event retired ON retired.seq = m.replaced_seq"
            " WHERE knowledge_fts MATCH ? AND v.known_us <= ?"
            " AND (? IS NULL OR ? = (coalesce(v.superseded_us <= ?, 0)"
            " OR coalesce(retired.recorded_us <= ?, 0)))"
            " AND (? IS NOT 1 OR s.kind = 'experience')"
            " AND (s.kind = 'experience' OR v.superseded_us IS NULL OR v.superseded_us > ?)"
            " AND (s.kind != 'evidence' OR s.source_id || '/' || v.version"
            " IN (SELECT value FROM json_each(?)))"
            " AND (? IS NULL OR s.kind = ?)"
            " AND (? IS NULL OR c.recipe_id = ?)"
            " AND EXISTS (SELECT 1 FROM knowledge_workspace_source w"
            " WHERE w.workspace_id = ? AND w.version_id = v.version_id)"
        )
        range_parameters: tuple[Any, ...] = ()
        if scope.ranges:
            inside = (
                " FROM json_each(?) r"
                " JOIN knowledge_source_version version ON version.source_id = r.value ->> 0"
                " JOIN knowledge_chunk chunk ON chunk.version_id = version.version_id"
                " AND chunk.ordinal BETWEEN r.value ->> 1 AND r.value ->> 2"
            )
            ranges = json.dumps(scope.ranges)
            first, last = self.log.read(
                f"SELECT min(chunk.rowid), max(chunk.rowid){inside}", (ranges,)
            )[0]
            if first is None:  # the ranges hold no chunk
                return ()
            # FTS5 walks only the rows between the ranges' first and last chunk, in one cursor:
            # bm25 reads its index-wide statistics once a cursor (22 ms a query at 1,000,000
            # chunks), so the ranges are never queried one by one. Within that span the rowid is
            # tested against one set; unary + keeps SQLite from restarting MATCH for each rowid.
            sql += " AND knowledge_fts.rowid BETWEEN ? AND ?"
            sql += f" AND +knowledge_fts.rowid IN (SELECT chunk.rowid{inside})"
            range_parameters = (first, last, ranges)
        sql = (
            "SELECT * FROM (SELECT *, row_number() OVER (PARTITION BY source_id"
            f" ORDER BY rank, rowid) AS place FROM ({sql})) WHERE place <= ?"
            " ORDER BY rank, rowid"
            if per_source
            else sql + " ORDER BY rank, c.rowid LIMIT ?"
        )
        filters = (
            cutoff_us,
            superseded,
            superseded,
            cutoff_us,
            cutoff_us,
            superseded,
            cutoff_us,
            world,
            kind,
            kind,
            recipe_id,
            recipe_id,
            scope.workspace,
            *range_parameters,
            limit,
        )
        with self.log.reading() as con:
            rows = con.execute(sql, (expression, *filters)).fetchall()
            if not rows and any(_HAN.search(term) for term in terms):
                # A phrase no chunk holds is matched by its characters, less the common ones: an
                # AND of characters most chunks hold matches most chunks and ranks none.
                parts = [
                    "(" + " AND ".join(f'"{char}"' for char in term if char not in common) + ")"
                    if _HAN.search(term)
                    else f'"{term}"'
                    for term in terms
                    if not _HAN.search(term) or set(term) - common
                ]
                if parts:
                    rows = con.execute(sql, (" OR ".join(parts), *filters)).fetchall()
        return tuple(
            SearchHit(
                row[0], row[1], row[2], Chunk(*row[3:7]), -float(row[7]), row[8], ("lexical",)
            )
            for row in rows
        )

    def vector_index(self, *, recipe_id: str, model_id: str, directory: Path) -> NumpyVectorIndex:
        """Build the Host's read-only matrix from its durable embedding cache."""
        return NumpyVectorIndex(self, recipe_id=recipe_id, model_id=model_id, directory=directory)

    def model(
        self, manifest: Mapping[str, Any], *, dimension: int, precision: Precision = "int8"
    ) -> str:
        """Register the consumed pack manifest, vector dimension and admitted storage precision."""
        if dimension < 1 or precision not in ("float16", "int8"):
            raise DataError(
                "a model needs a positive dimension and float16 or int8 storage",
                code="data_layer.knowledge_model_invalid",
            )
        text = canonical(dict(manifest)).decode("utf-8")
        with self.log.unit() as con:
            con.execute(
                "INSERT INTO knowledge_model VALUES (?, ?, ?, ?)"
                " ON CONFLICT (manifest, dimension, precision) DO NOTHING",
                (uuid7(), text, dimension, precision),
            )
            return str(
                con.execute(
                    "SELECT model_id FROM knowledge_model"
                    " WHERE manifest = ? AND dimension = ? AND precision = ?",
                    (text, dimension, precision),
                ).fetchone()[0]
            )

    def embeddings(
        self,
        digests: Sequence[str],
        *,
        recipe_id: str,
        model_id: str,
        encode: Callable[[tuple[str, ...]], Matrix],
    ) -> EmbeddingRead:
        """Encode only missing cache keys and read the held vectors in the caller's order.

        Inference holds the Host's cache claim, never the SQLite writer. A failed encoder
        writes nothing. Duplicate inputs share one cache entry, including within a build.
        """
        keys = tuple(digests)
        wanted = json.dumps(keys, ensure_ascii=False)
        with self.log.knowledge_cache_lock:
            with self.log.reading() as con:
                model = con.execute(
                    "SELECT dimension, precision FROM knowledge_model WHERE model_id = ?",
                    (model_id,),
                ).fetchone()
                if model is None:
                    raise DataError(
                        f"no registered embedding model {model_id}",
                        code="data_layer.knowledge_model_missing",
                    )
                if (
                    con.execute(
                        "SELECT 1 FROM knowledge_recipe WHERE recipe_id = ?", (recipe_id,)
                    ).fetchone()
                    is None
                ):
                    raise DataError(
                        f"no registered chunking recipe {recipe_id}",
                        code="data_layer.knowledge_recipe_missing",
                    )
                raw = con.execute(
                    "SELECT json_group_array(chunk_digest) FROM ("
                    " SELECT j.value AS chunk_digest FROM json_each(?) j"
                    " LEFT JOIN knowledge_embedding e ON e.chunk_digest = j.value"
                    " AND e.recipe_id = ? AND e.model_id = ? WHERE e.chunk_digest IS NULL"
                    " GROUP BY j.value ORDER BY min(CAST(j.key AS INTEGER)))",
                    (wanted, recipe_id, model_id),
                ).fetchone()[0]
            missing = tuple(json.loads(raw))
            dimension, precision = int(model[0]), str(model[1])
            if missing:
                vectors = np.asarray(encode(missing), dtype=np.float32)
                if vectors.shape != (len(missing), dimension) or not np.isfinite(vectors).all():
                    raise DataError(
                        "the encoder returned nonfinite vectors or a different shape",
                        code="data_layer.knowledge_embedding_invalid",
                    )
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                if not np.isfinite(norms).all() or np.any(norms == 0):
                    raise DataError(
                        "the encoder returned a zero or unnormalizable vector",
                        code="data_layer.knowledge_embedding_invalid",
                    )
                vectors = vectors / norms
                self._keep(missing, vectors, recipe_id, model_id, precision)
            return EmbeddingRead(
                self._read(wanted, len(keys), dimension, recipe_id, model_id, precision),
                len(missing),
                len(keys) - len(missing),
            )

    def _keep(
        self, digests: tuple[str, ...], vectors: Matrix, recipe: str, model: str, precision: str
    ) -> None:
        """Commit a whole numeric batch and its key-to-position mapping in one writer unit."""
        scales = None
        if precision == "int8":
            scale = np.max(np.abs(vectors), axis=0).astype("<f4") / 127
            scale[scale == 0] = 1
            encoded = np.clip(np.rint(vectors / scale), -127, 127).astype("i1")
            scales = scale.tobytes()
        else:
            encoded = vectors.astype("<f2")
        batch = uuid7()
        with self.log.unit() as con:
            con.execute(
                "INSERT INTO knowledge_embedding_batch VALUES (?, ?, ?, ?, ?)",
                (batch, model, encoded.tobytes(), scales, len(digests)),
            )
            con.execute(
                "INSERT INTO knowledge_embedding"
                " SELECT uuid7(), value, ?, ?, ?, CAST(key AS INTEGER) FROM json_each(?)",
                (recipe, model, batch, json.dumps(digests)),
            )

    def _read(
        self, wanted: str, count: int, dimension: int, recipe: str, model: str, precision: str
    ) -> Matrix:
        """Decode once per batch in NumPy, then gather the requested rows with array indexing.
        The rows are grouped by batch before each batch's vectors are joined, so a batch's blob is
        read once rather than carried by every row through the grouping (8x faster a 1,024-row
        read at 1,000,000 chunks)."""
        result = np.empty((count, dimension), dtype=np.float32)
        with self.log.reading() as con:
            batches = tuple(
                con.execute(
                    "SELECT b.vectors, b.scales, b.count, g.positions, g.ordinals FROM"
                    " (SELECT e.batch_id, json_group_array(CAST(j.key AS INTEGER)) AS positions,"
                    " json_group_array(e.ordinal) AS ordinals FROM json_each(?) j"
                    " JOIN knowledge_embedding e ON e.chunk_digest = j.value"
                    " AND e.recipe_id = ? AND e.model_id = ? GROUP BY e.batch_id) g"
                    " JOIN knowledge_embedding_batch b ON b.batch_id = g.batch_id",
                    (wanted, recipe, model),
                )
            )
        for blob, scales, rows, positions, ordinals in batches:
            result[np.asarray(json.loads(positions), dtype=np.intp)] = _decoded(
                blob, scales, rows, dimension, precision, np.asarray(json.loads(ordinals))
            )
        return result


def decode_batches(
    log: HostLog, rows: Sequence[Any], into: Matrix, at: int, dimension: int, precision: str
) -> None:
    """Write the vectors of `rows` (`_ROWS`: each carries its batch and ordinal) into `into`
    from row `at`, reading and decoding each encoded batch of `log` once, in at most `DECODED`
    rows a step: a whole corpus is often one batch, which a slice-by-slice read decoded again
    for every slice (40 s to open 100,000 chunks)."""
    held: dict[str, tuple[list[int], list[int]]] = {}
    for position, row in enumerate(rows):
        positions, ordinals = held.setdefault(str(row[7]), ([], []))
        positions.append(at + position)
        ordinals.append(int(row[8]))
    for batch, (positions, ordinals) in held.items():
        with log.reading() as con:
            blob, scales, count = con.execute(
                "SELECT vectors, scales, count FROM knowledge_embedding_batch WHERE batch_id = ?",
                (batch,),
            ).fetchone()
        for start in range(0, len(positions), DECODED):
            into[np.asarray(positions[start : start + DECODED], dtype=np.intp)] = _decoded(
                blob,
                scales,
                count,
                dimension,
                precision,
                np.asarray(ordinals[start : start + DECODED]),
            )


def _decoded(
    blob: bytes, scales: bytes | None, count: int, dimension: int, precision: str, rows: Any
) -> Matrix:
    """The float32 vectors at `rows` of one encoded batch: only those rows are converted, and
    each is the value a whole-batch decode gives it (int8 times its dimension's scale)."""
    encoded = np.frombuffer(blob, dtype="i1" if precision == "int8" else "<f2")
    vectors = encoded.reshape(count, dimension)[rows.astype(np.intp)].astype(np.float32)
    if precision == "int8":
        vectors *= np.frombuffer(scales or b"", dtype="<f4")
    return vectors


INDEX_FILE: Final = "knowledge-index"
"""The vector index's derived files in its folder, rebuilt from the log and never read for a fact:
`.npz`, each row's chunk, version, source and ordinal with the stamp of the log they were read
from, and `.f32`, the float32 matrix, row for row. The stamp names the format, the recipe, the
model and the last chunk held, by its rowid and id: a chunk row is never deleted, so while the log
holds that chunk at that rowid, the rows up to it are the same."""
INDEX_FORMAT: Final = 1
"""The derived files' layout; files of another format are rebuilt."""
DELTA: Final = 1024
"""Chunks a serving index takes after its matrix in memory; past them the Host opens it again,
which writes them into new derived files."""
DECODED: Final = 65_536
"""Vectors decoded at once while derived files are written: 128 MiB of float32 at 512 wide."""
SCOPE_EVENTS: Final = (
    "knowledge.revision_published",
    "knowledge.source_retired",
    "experience.recorded",
)
"""The events whose projections are all a scope reads: a source version's workspace, kind and
known time, its supersession or retirement, and an experience's replacement."""
_ROWS: Final = (
    "SELECT c.chunk_id, c.chunk_digest, c.version_id, e.embedding_id, v.source_id, c.ordinal,"
    " c.rowid, e.batch_id, e.ordinal FROM knowledge_chunk c"
    " JOIN knowledge_source_version v ON v.version_id = c.version_id"
    " LEFT JOIN knowledge_embedding e"
    " ON e.chunk_digest = c.chunk_digest AND e.recipe_id = c.recipe_id"
    " AND e.model_id = ? WHERE c.recipe_id = ? AND c.rowid > ? ORDER BY c.rowid"
)
"""A recipe's chunks after a rowid, in order, each with its model's vector if the log holds one."""


class NumpyVectorIndex:
    """A read-only float32 mapped matrix of the log's encoded vectors, kept as derived files."""

    def __init__(self, store: KnowledgeStore, *, recipe_id: str, model_id: str, directory: Path):
        """Map `model_id`'s vectors for `recipe_id`: the derived files in `directory` while their
        stamp holds (`INDEX_FILE`), with the log's chunks after them written into new files, so an
        open holds every row in one matrix and scores them in one product, as a full build does;
        a model the log does not hold is refused."""
        self.store, self.recipe_id, self.model_id = store, recipe_id, model_id
        self._guard = RLock()
        self._stem = directory / INDEX_FILE
        with store.log.reading() as con:
            model = con.execute(
                "SELECT dimension, precision FROM knowledge_model WHERE model_id = ?", (model_id,)
            ).fetchone()
            if model is None:
                raise DataError(
                    "the vector model is missing", code="data_layer.knowledge_model_missing"
                )
            self._dimension, self._precision = int(model[0]), str(model[1])
            self._load(con)
        self._delta = np.empty((DELTA, self._dimension), dtype=np.float32)
        self._delta_count = 0
        self._rebuild_required = False
        try:
            rows = self._after()
            # Any chunk after the files goes into them: a BLAS product's float32 scores depend on
            # the rows it is given, so a tail scored apart from the matrix could seal other bits.
            if rows:
                self._write(rows)
            else:
                self._take(rows)
            self.refresh_scopes()
        except BaseException:
            self.close()
            raise

    def _load(self, con: sqlite3.Connection) -> None:
        """Take the derived files' rows and map their matrix when their stamp still names this
        log's chunk at its rowid for this recipe and model; else hold no row."""
        self._ids: list[str] = []
        self._versions: list[str] = []
        self._places: list[tuple[str, int]] = []
        self._version_codes: dict[str, int] = {}
        self._source_codes: dict[str, int] = {}
        self._codes = self._row_sources = self._ordinals = np.empty(0, dtype=np.int32)
        self._last = 0
        self.matrix: Matrix = self._none()
        try:
            with np.load(self._stem.with_suffix(".npz")) as held:
                stamp = held["stamp"]
                if stamp.shape != (5,) or stamp[:3].tolist() != [
                    str(INDEX_FORMAT),
                    self.recipe_id,
                    self.model_id,
                ]:
                    return
                last = int(stamp[3])
                ids, versions = held["ids"], held["versions"]
                sources, ordinals = held["sources"], held["ordinals"]
                if (
                    ids.ndim != 1
                    or not ids.size
                    or any(part.shape != ids.shape for part in (versions, sources, ordinals))
                    or ids.dtype.kind != "U"
                    or versions.dtype.kind != "U"
                    or sources.dtype.kind != "U"
                    or ordinals.dtype.kind not in {"i", "u"}
                    or ids[-1] != stamp[4]
                ):
                    return
                matrix = self._stem.with_suffix(".f32")
                size = matrix.stat().st_size
                if size != ids.size * self._dimension * np.dtype(np.float32).itemsize:
                    return
                found = con.execute(
                    "SELECT chunk_id FROM knowledge_chunk WHERE rowid = ?", (last,)
                ).fetchone()
                if found is None or found[0] != stamp[4]:
                    return
                places = list(zip(sources.tolist(), ordinals.tolist(), strict=True))
                mapped = np.memmap(
                    matrix, mode="r", dtype=np.float32, shape=(ids.size, self._dimension)
                )
        except (OSError, KeyError, ValueError, OverflowError, EOFError, zipfile.BadZipFile):
            return
        self.matrix = mapped
        self._ids, self._versions = ids.tolist(), versions.tolist()
        self._places = places
        self._last = last

    def _after(self) -> list[Any]:
        """The recipe's chunks after the last one held; a chunk without its vector is refused."""
        with self.store.log.reading() as con:
            rows = con.execute(_ROWS, (self.model_id, self.recipe_id, self._last)).fetchall()
        if any(row[3] is None for row in rows):
            raise DataError(
                "a chunk has no cached vector", code="data_layer.knowledge_embedding_missing"
            )
        return rows

    def _vectors(self, rows: Sequence[Any]) -> Matrix:
        return self.store._read(
            json.dumps([str(row[1]) for row in rows]),
            len(rows),
            self._dimension,
            self.recipe_id,
            self.model_id,
            self._precision,
        )

    def _take(self, rows: Sequence[Any]) -> None:
        """Hold `rows` after the rows held, and index their versions and places."""
        self._ids.extend(str(row[0]) for row in rows)
        self._versions.extend(str(row[2]) for row in rows)
        self._places.extend((str(row[4]), int(row[5])) for row in rows)
        if rows:
            self._last = int(rows[-1][6])
        self._index_places()

    def _add(self, rows: Sequence[Any]) -> None:
        """Take `rows` into the delta after the matrix."""
        if rows:
            self._delta[self._delta_count : self._delta_count + len(rows)] = self._vectors(rows)
            self._delta_count += len(rows)
        self._take(rows)

    def _write(self, rows: Sequence[Any]) -> None:
        """Write the matrix and `rows` after it into new derived files, then map them. The old
        files go first, so a crash leaves no stamp naming another matrix."""
        held = len(self._ids)
        matrix, staged = self._stem.with_suffix(".f32"), self._stem.with_suffix(".f32.staging")
        writable = np.memmap(
            staged, mode="w+", dtype=np.float32, shape=(held + len(rows), self._dimension)
        )
        writable[:held] = self.matrix
        decode_batches(self.store.log, rows, writable, held, self._dimension, self._precision)
        writable.flush()
        writable._mmap.close()
        del writable
        self._close_matrix()
        self._take(rows)
        self._stem.with_suffix(".npz").unlink(missing_ok=True)
        os.replace(staged, matrix)
        names = self._stem.with_suffix(".npz.staging")
        with names.open("wb") as handle:
            np.savez(
                handle,
                stamp=np.array(
                    [
                        str(INDEX_FORMAT),
                        self.recipe_id,
                        self.model_id,
                        str(self._last),
                        self._ids[-1],
                    ]
                ),
                ids=np.array(self._ids),
                versions=np.array(self._versions),
                sources=np.array([source for source, _ in self._places]),
                ordinals=np.array([ordinal for _, ordinal in self._places], dtype=np.int64),
            )
        os.replace(names, self._stem.with_suffix(".npz"))
        self.matrix = np.memmap(
            matrix, mode="r", dtype=np.float32, shape=(len(self._ids), self._dimension)
        )

    def refresh_scopes(self) -> None:
        """Project workspace, kind and validity into arrays off the query path, as of the log's
        last event: every write they read is a projection of a `SCOPE_EVENTS` event."""
        with self.store.log.reading() as con:
            self._seq = con.execute("SELECT coalesce(max(seq), 0) FROM event").fetchone()[0]
            rows = con.execute(
                "SELECT DISTINCT w.workspace_id, s.kind, v.version_id, v.known_us,"
                " CASE WHEN retired.recorded_us IS NULL THEN v.superseded_us"
                " WHEN v.superseded_us IS NULL THEN retired.recorded_us"
                " ELSE min(v.superseded_us, retired.recorded_us) END, s.source_id, v.version"
                " FROM knowledge_workspace_source w"
                " JOIN knowledge_source_version v USING (version_id)"
                " JOIN knowledge_source s USING (source_id)"
                " LEFT JOIN experience_made m ON s.kind = 'experience'"
                " AND m.experience_id = s.source_id"
                " LEFT JOIN event retired ON retired.seq = m.replaced_seq"
            ).fetchall()
        known = np.zeros(len(self._version_codes), dtype=np.int64)
        end = np.full(len(self._version_codes), np.iinfo(np.int64).max, dtype=np.int64)
        experience = np.zeros(len(self._version_codes), dtype=np.bool_)
        evidence: dict[int, tuple[str, str]] = {}
        codes: dict[tuple[str, str | None], set[int]] = {}
        for workspace, kind, version_id, start, superseded, source_id, version_id_external in rows:
            code = self._version_codes.get(version_id)
            if code is None:
                continue
            known[code] = start
            if superseded is not None:
                end[code] = superseded
            experience[code] = kind == "experience"
            if kind == "evidence":
                evidence[code] = (source_id, version_id_external)
            codes.setdefault((workspace, kind), set()).add(code)
            codes.setdefault((workspace, None), set()).add(code)
        self._known, self._end = known, end
        self._experience, self._evidence = experience, evidence
        members, scopes = {}, {}
        for scope, values in codes.items():
            member = np.zeros(len(self._version_codes), dtype=np.bool_)
            member[list(values)] = True
            members[scope], scopes[scope] = member, np.flatnonzero(member[self._codes])
        self._members, self._scopes = members, scopes

    def _scopes_moved(self) -> bool:
        """Whether an event since the scopes were projected changed what they admit: one seek
        of the log's events after that point (`SCOPE_EVENTS`), none on an unchanged log."""
        head, moved = self.store.log.read(
            "SELECT max(seq), coalesce(max(type IN (?, ?, ?)), 0) FROM event WHERE seq > ?",
            (*SCOPE_EVENTS, self._seq),
        )[0]
        if head is not None and not moved:
            self._seq = head
        return bool(moved)

    def refresh(self) -> bool:
        """Add the chunks encoded since to the delta, and project the scopes again when chunks
        were added or a `SCOPE_EVENTS` event was recorded since; on an unchanged log it costs
        two indexed seeks. False asks the Host for a rebuild, which writes the chunks into new
        derived files."""
        with self._guard:
            rows = self._after()
            if self._delta_count + len(rows) > DELTA:
                self._rebuild_required = True
                return False
            if rows:
                self._add(rows)
            if rows or self._scopes_moved():
                self.refresh_scopes()
            self._rebuild_required = False
            return True

    def search(
        self, query: Matrix, scope: Scope, *, limit: int = 50, per_source: bool = False
    ) -> tuple[SearchHit, ...]:
        """Score one matrix with BLAS, then select the eligible top-k (each source's top-k when
        `per_source`) with stable ties."""
        with self._guard:
            return self._hits(*self._search(query, scope, limit, per_source))

    def _with_lexical(
        self,
        query: Matrix,
        scope: Scope,
        limit: int,
        per_source: bool,
        lexical: Callable[[], tuple[SearchHit, ...]],
    ) -> tuple[tuple[SearchHit, ...], tuple[SearchHit, ...]]:
        """Overlap FTS with matrix ranking while the caller owns both SQLite reads."""
        with self._guard:
            with ThreadPoolExecutor(max_workers=worker_width(1)) as pool:
                selected = pool.submit(
                    carry_resource_lease(partial(self._search, query, scope, limit, per_source))
                )
                words = lexical()
            return words, self._hits(*selected.result(timeout=0))

    def _search(
        self, query: Matrix, scope: Scope, limit: int, per_source: bool
    ) -> tuple[list[str], list[float]]:
        """Rank the in-memory index while its caller holds the index guard."""
        if self._rebuild_required:
            raise DataError(
                "the vector index requires its background rebuild",
                code="data_layer.knowledge_rebuild_required",
            )
        query = np.asarray(query, dtype=np.float32)
        if query.shape != (self._dimension,) or not np.isfinite(query).all():
            raise DataError(
                "the query vector is invalid", code="data_layer.knowledge_embedding_invalid"
            )
        norm = np.linalg.norm(query)
        if not np.isfinite(norm) or norm == 0:
            raise DataError(
                "the query vector is zero or unnormalizable",
                code="data_layer.knowledge_embedding_invalid",
            )
        query = query / norm
        if limit < 1 or not len(self._ids):
            return [], []
        scoped = self._scopes.get((scope.workspace, scope.kind))
        if scoped is None:
            return [], []
        if scope.ranges:  # the rows inside the ranges first, then those the scope holds
            inside = self._in_ranges(scope.ranges)
            scoped = inside[self._members[scope.workspace, scope.kind][self._codes[inside]]]
        codes = self._codes[scoped]
        eligible_mask = self._known[codes] <= scope.cutoff_us
        retired = scope.cutoff_us >= self._end[codes]
        if scope.superseded is True:
            eligible_mask &= retired & self._experience[codes]
        elif scope.superseded is False:
            eligible_mask &= ~retired
        else:
            eligible_mask &= ~retired | self._experience[codes]
        if self._evidence:
            allowed = np.ones(len(self._version_codes), dtype=np.bool_)
            for code, key in self._evidence.items():
                allowed[code] = key in scope.evidence
            eligible_mask &= allowed[codes]
        eligible = scoped[eligible_mask]
        if not len(eligible):
            return [], []
        scores = np.concatenate((self.matrix @ query, self._delta[: self._delta_count] @ query))
        values = scores[eligible]
        if per_source:  # each source's best `limit`, in one global order
            order = np.lexsort((eligible, -values))
            sources = self._row_sources[eligible[order]]
            grouped = np.argsort(sources, kind="stable")
            starts = np.r_[0, np.flatnonzero(np.diff(sources[grouped])) + 1]
            place = np.empty(len(order), dtype=np.int64)
            place[grouped] = np.arange(len(order)) - np.repeat(
                starts, np.diff(np.r_[starts, len(order)])
            )
            order = order[place < limit]
            positions = eligible[order]
            scores = [float(values[index]) for index in order]
        else:
            count = min(limit, len(eligible))
            candidate = np.argpartition(-values, count - 1)[:count]
            boundary = values[candidate].min()
            candidate = np.flatnonzero(values >= boundary)
            order = np.lexsort((eligible[candidate], -values[candidate]))[:count]
            positions = eligible[candidate[order]]
            scores = [float(values[candidate[index]]) for index in order]
        return [self._ids[int(position)] for position in positions], scores

    def _hits(self, selected: list[str], scores: list[float]) -> tuple[SearchHit, ...]:
        """Read the selected chunks on the caller's log connection in rank order."""
        if not selected:
            return ()
        with self.store.log.reading() as con:
            rows = con.execute(
                "SELECT s.source_id, v.version, c.ordinal, c.chunk_id, c.char_start, c.char_end,"
                " c.chunk_digest, c.rowid FROM json_each(?) j"
                " JOIN knowledge_chunk c ON c.chunk_id = j.value"
                " JOIN knowledge_source_version v USING (version_id)"
                " JOIN knowledge_source s USING (source_id) ORDER BY CAST(j.key AS INTEGER)",
                (json.dumps(selected),),
            ).fetchall()
        return tuple(
            SearchHit(row[0], row[1], row[2], Chunk(*row[3:7]), score, row[7], ("vector",))
            for row, score in zip(rows, scores, strict=True)
        )

    def _index_places(self) -> None:
        """Code the rows not yet coded: each row's version, for scopes, and its source and chunk
        ordinal, for range scopes. Rows are only appended, so the rows already coded keep their
        codes and only the new ones are read."""
        start = len(self._codes)
        versions, places = self._versions[start:], self._places[start:]
        codes, sources = self._version_codes, self._source_codes
        self._codes = np.concatenate(
            (
                self._codes,
                np.fromiter(
                    (codes.setdefault(value, len(codes)) for value in versions),
                    dtype=np.int32,
                    count=len(versions),
                ),
            )
        )
        self._row_sources = np.concatenate(
            (
                self._row_sources,
                np.fromiter(
                    (sources.setdefault(source, len(sources)) for source, _ in places),
                    dtype=np.int32,
                    count=len(places),
                ),
            )
        )
        self._ordinals = np.concatenate(
            (
                self._ordinals,
                np.fromiter((ordinal for _, ordinal in places), dtype=np.int32, count=len(places)),
            )
        )
        self._source_rows = np.argsort(self._row_sources, kind="stable")
        self._source_offsets = np.r_[0, np.cumsum(np.bincount(self._row_sources))]

    def _in_ranges(self, ranges: tuple[tuple[str, int, int], ...]) -> np.ndarray:
        """The rows inside one of `ranges` (source id, first ordinal, last ordinal), ascending:
        read from each source's own rows, never a pass over every row."""
        inside = []
        for source, first, last in ranges:
            code = self._source_codes.get(source)
            if code is not None:
                rows = self._source_rows[
                    self._source_offsets[code] : self._source_offsets[code + 1]
                ]
                ordinals = self._ordinals[rows]
                inside.append(rows[(ordinals >= first) & (ordinals <= last)])
        return np.unique(np.concatenate(inside)) if inside else np.empty(0, dtype=np.intp)

    def _none(self) -> Matrix:
        """A read-only matrix of no row, as the mapped one is read-only."""
        matrix = np.empty((0, self._dimension), dtype=np.float32)
        matrix.flags.writeable = False
        return matrix

    def _close_matrix(self) -> None:
        if isinstance(self.matrix, np.memmap):
            self.matrix._mmap.close()
        self.matrix = self._none()

    def close(self) -> None:
        """Release the mapping of the derived matrix; its files stay for the next open."""
        self._close_matrix()

    def __enter__(self) -> Self:
        """The index itself; leaving the block closes it."""
        return self

    def __exit__(self, *_: object) -> None:
        """Close the index whether or not the block raised."""
        self.close()


WINDOW: Final = 32
"""Candidates each source supplies per channel in a precise search."""
FUSED_CANDIDATES_PER_SOURCE: Final = 12
"""A filing's candidate quota before the search's single shortlist is chosen."""


def _no_span(_query: str) -> AbstractContextManager[None]:
    return nullcontext()


RETRIEVAL_SPAN: Callable[[str], AbstractContextManager[None]] = _no_span
"""The span one retrieval runs in, given its query: the Host's `gen_ai.retrieval` step, which
history installs when it loads (history imports this module, so this one never imports it)."""


class HybridSearch:
    """The Host's FTS and vector retrieval, fused, with source verification."""

    def __init__(
        self,
        store: KnowledgeStore,
        vector: VectorPort,
        encoder: Any,
        reranker: Any,
        source: Callable[[SearchHit, int], str],
    ) -> None:
        """Search `store`'s FTS and `vector`'s index, embed a query with `encoder`, rerank with
        `reranker`, and verify each hit's text through `source`."""
        self.store, self.vector, self.encoder, self.reranker, self.source = (
            store,
            vector,
            encoder,
            reranker,
            source,
        )

    def retrieve(
        self,
        query: str,
        *,
        language: Literal["en", "zh"],
        workspace: str,
        cutoff_us: int,
        kind: str | None = None,
        superseded: bool | None = False,
        limit: int = 5,
        ranges: tuple[tuple[str, int, int], ...] = (),
        groups: tuple[tuple[str, ...], ...] = (),
        depth: int | None = None,
    ) -> Retrieved:
        """Fuse both scoped channels; English is reranked by the cross-encoder in the precise mode
        (a `depth`) and in a sealed read (`SEALED_KIND`, its best `SHORTLIST` without a depth), and
        any other answer is the fused order.

        The ordinal ranges restrict both channels. In precise mode each source contributes to
        one fused shortlist, whose English pairs are scored once under the search deadline.
        The fused order stands on timeout. Each group takes its own limit of the final order.
        """
        if language not in ("en", "zh"):
            raise DataError(
                "the query language is unsupported", code="data_layer.knowledge_language_invalid"
            )
        if (
            not query.strip()
            or len(query) > 512
            or not 1 <= limit <= 20
            or (depth is not None and not 1 <= depth <= WINDOW)
        ):
            raise DataError(
                "the query or result limit is invalid", code="data_layer.knowledge_query_invalid"
            )
        deadline = time.monotonic() + DEADLINE_SECONDS
        with RETRIEVAL_SPAN(query):
            query_vector = self.encoder.embed_query(query)
            scope = replace(
                self.store.scope(
                    workspace=workspace, cutoff_us=cutoff_us, kind=kind, superseded=superseded
                ),
                ranges=ranges,
            )
            window, per_source = (WINDOW, True) if depth else (50, False)
            read_lexical = partial(
                self.store.lexical,
                query,
                scope,
                recipe_id=self.vector.recipe_id,
                limit=window,
                per_source=per_source,
            )
            if language == "zh" and isinstance(self.vector, NumpyVectorIndex) and cpu_width() > 1:
                lexical, dense = self.vector._with_lexical(
                    query_vector, scope, window, per_source, read_lexical
                )
            else:
                lexical = read_lexical()
                dense = self.vector.search(query_vector, scope, limit=window, per_source=per_source)
            ranked = list(fuse(lexical, dense, limit=len(lexical) + len(dense)))
            if depth is None:
                ranked = ranked[:50]
            else:
                taken: dict[str, int] = {}
                candidates = []
                for hit in ranked:
                    count = taken.get(hit.source_id, 0)
                    if count < FUSED_CANDIDATES_PER_SOURCE:
                        taken[hit.source_id] = count + 1
                        candidates.append(hit)
                ranked = candidates
            score_type: Literal["LINEAR_FUSION", "CROSS_ENCODER_LOGIT"] = "LINEAR_FUSION"
            bodies: dict[tuple[str, str], str] = {}
            texts: dict[str, str] = {}

            def read(hit: SearchHit) -> str:
                key = (hit.source_id, hit.version)
                if key not in bodies:
                    bodies[key] = self.source(hit, cutoff_us)
                text = bodies[key][hit.chunk.start : hit.chunk.end]
                if digest(text.encode()) != hit.chunk.digest:
                    raise DataError(
                        "the source chunk differs from its admission",
                        code="data_layer.knowledge_source_conflict",
                    )
                texts[hit.chunk.chunk_id] = text
                return text

            reranked = 0
            logits: dict[int, float] = {}
            fallback: Literal["DEADLINE_EXCEEDED"] | None = None
            if language == "en" and ranked and (depth or kind == SEALED_KIND):
                first = ranked[: min(depth or SHORTLIST, SHORTLIST)]
                passages = [read(hit) for hit in first]
                unique = tuple(dict.fromkeys(passages))
                result = self.reranker.score(query, unique, deadline=deadline)
                reranked = len(result.scores)
                if result.timed_out:
                    fallback = "DEADLINE_EXCEEDED"
                else:
                    if reranked != len(unique) or not np.isfinite(result.scores).all():
                        raise DataError(
                            "the reranker returned invalid scores",
                            code="data_layer.knowledge_embedding_invalid",
                        )
                    scores = dict(zip(unique, result.scores, strict=True))
                    logits = {
                        hit.rowid: scores[passage]
                        for hit, passage in zip(first, passages, strict=True)
                    }
                    ranked[: len(first)] = sorted(
                        first, key=lambda hit: (-logits[hit.rowid], hit.rowid)
                    )
            selected = _cut(ranked, limit, groups)
            # The fused tail keeps each issuer's candidates available. A result set
            # exposes one score scale, even when its shortlist was reordered.
            if logits and all(hit.rowid in logits for hit in selected):
                selected = [replace(hit, score=logits[hit.rowid]) for hit in selected]
                score_type = "CROSS_ENCODER_LOGIT"
            return Retrieved(
                tuple(
                    RetrievedHit(hit, texts.get(hit.chunk.chunk_id) or read(hit))
                    for hit in selected
                ),
                score_type,
                reranked,
                fallback,
            )


def _cut(
    ranked: Sequence[SearchHit], limit: int, groups: tuple[tuple[str, ...], ...]
) -> list[SearchHit]:
    """The final cut: one global `limit`, or `limit` for each group of source ids, sources in no
    group sharing one trailing group; the order, and so each hit's rank, is unchanged."""
    if not groups:
        return list(ranked[:limit])
    group_of = {source: index for index, group in enumerate(groups) for source in group}
    taken: dict[int, int] = {}
    selected = []
    for hit in ranked:
        group = group_of.get(hit.source_id, len(groups))
        if taken.get(group, 0) < limit:
            taken[group] = taken.get(group, 0) + 1
            selected.append(hit)
    return selected
