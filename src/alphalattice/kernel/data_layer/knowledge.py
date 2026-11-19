"""Knowledge's shared SQLite cache, written through the serving Host's log.

Vectors are encoded outside the log writer and committed in batches. The cache maps a
chunk digest, chunking recipe and model to its immutable batch and position; a second
workspace or rebuild reads that same vector. Each int8 batch has one scale per dimension.
"""

from __future__ import annotations

import json
import re
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, ExitStack, contextmanager, nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from threading import RLock
from typing import Any, Literal, Protocol, Self

import numpy as np
from numpy.typing import NDArray

from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer.chain import canonical, digest
from alphalattice.kernel.data_layer.log import HostLog, uuid7
from alphalattice.kernel.shared_kernel.environment import package_versions

type Precision = Literal["float16", "int8"]
type Matrix = NDArray[np.float32]
type EvidenceVersions = frozenset[tuple[str, str]]

_HAN = re.compile(r"([\u4e00-\u9fff])")
_TERMS = re.compile(r"[\u4e00-\u9fff]+|[A-Za-z0-9_]+")


def _spaced(text: str) -> str:
    """Make Han characters visible to unicode61 without changing the source text."""
    return _HAN.sub(r" \1 ", text)


@contextmanager
def _mapping_file(directory: Path) -> Iterator[Any]:
    """Hold the matrix file until its owning index closes."""
    with tempfile.TemporaryFile(dir=directory) as handle:
        yield handle


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


def _splitter_version(name: str) -> str:
    """The installed splitter's version, read through the environment helper."""
    return dict(package_versions((name,)))[name]


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
        replace(hit, score=0.5 * lexical_scores.get(key, 0.0) + 0.5 * vector_scores.get(key, 0.0))
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


class VectorPort(Protocol):
    """The Host's replaceable vector search over its admitted source versions."""

    recipe_id: str

    def search(self, query: Matrix, scope: Scope, *, limit: int = 50) -> tuple[SearchHit, ...]:
        """The top `limit` chunks in `scope` by similarity to `query`, ties in source row order."""
        ...

    def close(self) -> None:
        """Release what the index holds."""
        ...


class KnowledgeStore:
    """The Host's shared source and embedding store; never opens another SQLite writer."""

    def __init__(
        self, log: HostLog, *, evidence_asof: Callable[[int], EvidenceVersions] | None = None
    ) -> None:
        """Use the serving Host's writer and its shared cache-build ownership."""
        self.log = log
        self.evidence_asof = evidence_asof

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
            found = con.execute(
                "SELECT 1 FROM knowledge_workspace_source w"
                " JOIN knowledge_source_version v USING (version_id)"
                " JOIN knowledge_source s USING (source_id)"
                " WHERE w.workspace_id = ? AND s.kind = 'evidence' LIMIT 1",
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
        from semantic_text_splitter import TextSplitter  # lazy: the optional semantic extra

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
            if spec.get("splitter") != "semantic-text-splitter" or spec.get(
                "version"
            ) != _splitter_version("semantic-text-splitter"):
                raise DataError(
                    "the chunk recipe differs from the installed splitter",
                    code="data_layer.knowledge_recipe_invalid",
                )
        # Splitting and hashing may be large; the Host's one SQLite writer stays free while
        # immutable source and recipe inputs are transformed.
        splitter = TextSplitter(spec["capacity"], overlap=spec["overlap"], trim=spec["trim"])
        slices = splitter.chunk_indices(content)
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
        self, query: str, scope: Scope, *, recipe_id: str | None = None, limit: int = 50
    ) -> tuple[SearchHit, ...]:
        """Search the FTS index within an admitted workspace and source-time boundary."""
        terms = _TERMS.findall(query)
        if not terms or limit < 1:
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
            " ORDER BY rank, c.rowid LIMIT ?"
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
            limit,
        )
        with self.log.reading() as con:
            rows = con.execute(sql, (expression, *filters)).fetchall()
            if not rows and any(_HAN.search(term) for term in terms):
                loose = " OR ".join(
                    "(" + " AND ".join('"' + char + '"' for char in term) + ")"
                    if _HAN.search(term)
                    else '"' + term + '"'
                    for term in terms
                )
                rows = con.execute(sql, (loose, *filters)).fetchall()
        return tuple(
            SearchHit(row[0], row[1], row[2], Chunk(*row[3:7]), -float(row[7]), row[8])
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
        """Decode once per batch in NumPy, then gather the requested rows with array indexing."""
        result = np.empty((count, dimension), dtype=np.float32)
        with self.log.reading() as con:
            batches = tuple(
                con.execute(
                    "SELECT b.vectors, b.scales, b.count, json_group_array(CAST(j.key AS INTEGER)),"
                    " json_group_array(e.ordinal) FROM json_each(?) j"
                    " JOIN knowledge_embedding e ON e.chunk_digest = j.value"
                    " AND e.recipe_id = ? AND e.model_id = ?"
                    " JOIN knowledge_embedding_batch b ON b.batch_id = e.batch_id"
                    " GROUP BY b.batch_id",
                    (wanted, recipe, model),
                )
            )
        for blob, scales, rows, positions, ordinals in batches:
            batch = np.frombuffer(blob, dtype="i1" if precision == "int8" else "<f2")
            batch = batch.reshape(rows, dimension).astype(np.float32)
            if precision == "int8":
                batch *= np.frombuffer(scales, dtype="<f4")
            result[np.asarray(json.loads(positions), dtype=np.intp)] = batch[
                np.asarray(json.loads(ordinals), dtype=np.intp)
            ]
        return result


class NumpyVectorIndex:
    """A read-only float32 mapped matrix rebuilt from the log's encoded vectors."""

    def __init__(self, store: KnowledgeStore, *, recipe_id: str, model_id: str, directory: Path):
        """Map `model_id`'s vectors for `recipe_id` into a float32 file under `directory`; a model
        the log does not hold is refused."""
        self.store, self.recipe_id, self.model_id = store, recipe_id, model_id
        self._guard = RLock()
        with store.log.reading() as con:
            model = con.execute(
                "SELECT dimension, precision FROM knowledge_model WHERE model_id = ?", (model_id,)
            ).fetchone()
            if model is None:
                raise DataError(
                    "the vector model is missing", code="data_layer.knowledge_model_missing"
                )
            rows = con.execute(
                "SELECT c.chunk_id, c.chunk_digest, c.version_id, e.embedding_id"
                " FROM knowledge_chunk c LEFT JOIN knowledge_embedding e"
                " ON e.chunk_digest = c.chunk_digest AND e.recipe_id = c.recipe_id"
                " AND e.model_id = ? WHERE c.recipe_id = ? ORDER BY c.rowid",
                (model_id, recipe_id),
            ).fetchall()
        if any(row[3] is None for row in rows):
            raise DataError(
                "a chunk has no cached vector", code="data_layer.knowledge_embedding_missing"
            )
        self._ids = [str(row[0]) for row in rows]
        digests = tuple(str(row[1]) for row in rows)
        versions = [str(row[2]) for row in rows]
        self._versions = versions
        self._current = set(self._ids)
        self._version_codes = {value: index for index, value in enumerate(dict.fromkeys(versions))}
        self._codes = np.fromiter(
            (self._version_codes[value] for value in versions), dtype=np.int32
        )
        self._dimension = int(model[0])
        self._precision = str(model[1])
        self._delta = np.empty((1024, self._dimension), dtype=np.float32)
        self._delta_count = 0
        self._rebuild_required = False
        self.refresh_scopes()
        self._stack = ExitStack()
        self._file = self._stack.enter_context(_mapping_file(directory))
        try:
            if not rows:
                self.matrix = np.empty((0, self._dimension), dtype=np.float32)
                self.matrix.flags.writeable = False
                return
            writable = np.memmap(
                self._file, mode="w+", dtype=np.float32, shape=(len(rows), self._dimension)
            )
            for start in range(0, len(rows), 1024):
                batch = digests[start : start + 1024]
                writable[start : start + len(batch)] = store._read(
                    json.dumps(batch),
                    len(batch),
                    self._dimension,
                    recipe_id,
                    model_id,
                    str(model[1]),
                )
            writable.flush()
            del writable
            self.matrix = np.memmap(
                self._file, mode="r", dtype=np.float32, shape=(len(rows), self._dimension)
            )
        except BaseException:
            self._stack.close()
            raise

    def refresh_scopes(self) -> None:
        """Project workspace, kind and validity into arrays off the query path."""
        with self.store.log.reading() as con:
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
        active = np.fromiter((key in self._current for key in self._ids), dtype=np.bool_)
        self._scopes = {
            scope: np.flatnonzero(active & np.isin(self._codes, list(values)))
            for scope, values in codes.items()
        }

    def refresh(self) -> bool:
        """Add newly encoded chunks to the delta; False asks the Host for an off-path rebuild."""
        with self._guard:
            with self.store.log.reading() as con:
                rows = con.execute(
                    "SELECT c.chunk_id, c.chunk_digest, c.version_id, e.embedding_id"
                    " FROM knowledge_chunk c LEFT JOIN knowledge_embedding e"
                    " ON e.chunk_digest = c.chunk_digest AND e.recipe_id = c.recipe_id"
                    " AND e.model_id = ? WHERE c.recipe_id = ? ORDER BY c.rowid",
                    (self.model_id, self.recipe_id),
                ).fetchall()
            if any(row[3] is None for row in rows):
                raise DataError(
                    "a chunk has no cached vector", code="data_layer.knowledge_embedding_missing"
                )
            held = set(self._ids)
            new = [row for row in rows if row[0] not in held]
            if self._delta_count + len(new) > len(self._delta):
                self._rebuild_required = True
                return False
            if new:
                digests = tuple(str(row[1]) for row in new)
                vectors = self.store._read(
                    json.dumps(digests),
                    len(new),
                    self._dimension,
                    self.recipe_id,
                    self.model_id,
                    self._precision,
                )
                self._delta[self._delta_count : self._delta_count + len(new)] = vectors
                self._delta_count += len(new)
                self._ids.extend(str(row[0]) for row in new)
                self._versions.extend(str(row[2]) for row in new)
                self._version_codes = {
                    value: index for index, value in enumerate(dict.fromkeys(self._versions))
                }
                self._codes = np.fromiter(
                    (self._version_codes[value] for value in self._versions), dtype=np.int32
                )
            self._current = {str(row[0]) for row in rows}
            self.refresh_scopes()
            self._rebuild_required = False
            return True

    def search(self, query: Matrix, scope: Scope, *, limit: int = 50) -> tuple[SearchHit, ...]:
        """Score one matrix with BLAS, then select the eligible top-k with stable ties."""
        with self._guard:
            return self._search(query, scope, limit)

    def _search(self, query: Matrix, scope: Scope, limit: int) -> tuple[SearchHit, ...]:
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
            return ()
        scoped = self._scopes.get((scope.workspace, scope.kind), np.empty(0, dtype=np.intp))
        codes = self._codes[scoped]
        eligible_mask = self._known[codes] <= scope.cutoff_us
        retired = scope.cutoff_us >= self._end[codes]
        if scope.superseded is True:
            eligible_mask &= retired & self._experience[codes]
        elif scope.superseded is False:
            eligible_mask &= ~retired
        else:
            eligible_mask &= ~retired | self._experience[codes]
        allowed = np.ones(len(self._version_codes), dtype=np.bool_)
        for code, key in self._evidence.items():
            allowed[code] = key in scope.evidence
        eligible_mask &= allowed[codes]
        eligible = scoped[eligible_mask]
        if not len(eligible):
            return ()
        scores = np.concatenate((self.matrix @ query, self._delta[: self._delta_count] @ query))
        values = scores[eligible]
        count = min(limit, len(eligible))
        candidate = np.argpartition(-values, count - 1)[:count]
        boundary = values[candidate].min()
        candidate = np.flatnonzero(values >= boundary)
        order = np.lexsort((eligible[candidate], -values[candidate]))[:count]
        positions = eligible[candidate[order]]
        selected = [self._ids[int(position)] for position in positions]
        scores = [float(values[candidate[index]]) for index in order]
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
            SearchHit(row[0], row[1], row[2], Chunk(*row[3:7]), score, row[7])
            for row, score in zip(rows, scores, strict=True)
        )

    def close(self) -> None:
        """Release the mapping and the temporary file owned by this Host index."""
        if isinstance(self.matrix, np.memmap):
            self.matrix._mmap.close()
        self._stack.close()

    def __enter__(self) -> Self:
        """The index itself; leaving the block closes it."""
        return self

    def __exit__(self, *_: object) -> None:
        """Close the index whether or not the block raised."""
        self.close()


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
    ) -> Retrieved:
        """Fuse both scoped channels and rerank English only within the model's deadline."""
        if language not in ("en", "zh"):
            raise DataError(
                "the query language is unsupported", code="data_layer.knowledge_language_invalid"
            )
        if not query.strip() or len(query) > 512 or not 1 <= limit <= 20:
            raise DataError(
                "the query or result limit is invalid", code="data_layer.knowledge_query_invalid"
            )
        with RETRIEVAL_SPAN(query):
            query_vector = self.encoder.embed_query(query)
            scope = self.store.scope(
                workspace=workspace, cutoff_us=cutoff_us, kind=kind, superseded=superseded
            )
            lexical = self.store.lexical(query, scope, recipe_id=self.vector.recipe_id)
            dense = self.vector.search(query_vector, scope)
            ranked = list(fuse(lexical, dense))
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

            if language == "en" and ranked:
                first = ranked[:20]
                scores = self.reranker.score(query, tuple(read(hit) for hit in first))
                if scores is not None:
                    if len(scores) != len(first) or not np.isfinite(scores).all():
                        raise DataError(
                            "the reranker returned invalid scores",
                            code="data_layer.knowledge_embedding_invalid",
                        )
                    rescored = [
                        replace(hit, score=float(score))
                        for hit, score in zip(first, scores, strict=True)
                    ]
                    ranked = sorted(rescored, key=lambda hit: (-hit.score, hit.rowid))
                    score_type = "CROSS_ENCODER_LOGIT"
            return Retrieved(
                tuple(
                    RetrievedHit(hit, texts.get(hit.chunk.chunk_id) or read(hit))
                    for hit in ranked[:limit]
                ),
                score_type,
            )
