"""Workspace-owned, source-authoritative local Hybrid v2 retrieval.

A generation is keyed by the corpus it embeds and the spec that embedded it,
never by the snapshot that asked for it: two snapshots whose sorted revision
commitments are the same resolve to one physical generation, and the passage
vectors the pinned model produced are committed as a content-addressed payload
beside the index, so a cold open or a rebuild verifies those bytes instead of
running the model over the corpus again. The `knowledge-hybrid-v3` generations
built before that commitment are history (2026-09-23): their manifests are still
read for storage accounting and eviction, and a request to open one is refused.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import secrets
import sqlite3
from collections.abc import Callable, Sequence
from contextlib import closing, suppress
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

from pydantic import ValidationError

from alphalattice.control.workspace_runtime.paths import make_confined_parents, resolve_confined
from alphalattice.kernel.knowledge.hybrid_contracts import (
    ENCODER_RUNTIME_TORCH_CUDA,
    HybridCapabilityReport,
    HybridGenerationAnchor,
    HybridIndexEvictionMarker,
    HybridIndexSpec,
    HybridKnowledgeIndexManifest,
    HybridKnowledgeRetrievalRequest,
    HybridKnowledgeRetrievalResult,
    HybridKnowledgeRetrievalTrace,
    HybridV3KnowledgeIndexManifest,
    HybridVectorBlock,
    HybridVectorBlockSidecar,
    HybridVectorCommitment,
    KnowledgeRetrievalMode,
    PairScoreAdmission,
    RetrievalScoreSemantics,
    SealedPairScoreBlock,
    SemanticPackStatus,
    embedding_context_hash,
    vector_asset_key,
)
from alphalattice.kernel.knowledge.hybrid_contracts import (
    HYBRID_INDEX_SCHEMA_VERSION as _INDEX_SCHEMA_VERSION,
)
from alphalattice.kernel.knowledge.hybrid_contracts import (
    LEGACY_HYBRID_INDEX_SCHEMA_VERSION as _LEGACY_SCHEMA_VERSION,
)
from alphalattice.kernel.knowledge.hybrid_contracts import (
    VECTOR_BLOCK_SIDECAR_SUFFIX as _SIDECAR_SUFFIX,
)
from alphalattice.kernel.knowledge.hybrid_contracts import (
    VECTOR_PAYLOAD_ROOT as _VECTOR_PAYLOAD_ROOT,
)
from alphalattice.kernel.knowledge.retrieval import (
    WorkspaceKnowledgeLibrary,
    _build_chunks,
    _canonical_hash,
    _cjk_terms,
    _compiled_terms,
    _execute,
    _literal_fts_term,
    _normalize_text,
    _projection_row,
    _retrieval_error,
    _utc_text,
)
from alphalattice.kernel.knowledge.retrieval_contracts import (
    KnowledgeCitation,
    KnowledgeRetrievalHit,
    RetrievalChannel,
    RetrievalMatchKind,
    RetrievalStatus,
    WorkspaceKnowledgeSnapshot,
)
from alphalattice.kernel.knowledge.retrieval_errors import KnowledgeRetrievalError
from alphalattice.kernel.shared_kernel.domain.errors import (
    DomainValidationError,
    WorkspaceConflictError,
)
from alphalattice.kernel.shared_kernel.domain.serialization import (
    canonical_json_bytes,
    parse_model,
    sha256_hex,
)
from alphalattice.kernel.shared_kernel.persistence import replace_with_retry

if TYPE_CHECKING:
    from alphalattice.control.workspace_runtime.core import Workspace
    from alphalattice.kernel.knowledge.retrieval import _Chunk

_SEMANTIC_MODULE = "alphalattice.kernel.knowledge._embeddings"


def _semantic_module() -> Any:
    try:
        return importlib.import_module(_SEMANTIC_MODULE)
    except (ImportError, OSError) as exc:
        raise _retrieval_error(
            "semantic runtime is unavailable",
            code="retrieval.semantic_pack_unavailable",
            retryable=True,
            cause=exc,
        ) from exc


def _torch_module() -> Any:
    """The one GPU runtime's loader, imported only for a recipe that names it."""

    try:
        return importlib.import_module("alphalattice.kernel.knowledge._torch")
    except (ImportError, OSError) as exc:
        raise _retrieval_error(
            "GPU retrieval runtime is unavailable",
            code="retrieval.semantic_pack_unavailable",
            retryable=True,
            cause=exc,
        ) from exc


def _encoder_on_torch(spec: HybridIndexSpec) -> bool:
    return bool(spec.encoder_runtime == ENCODER_RUNTIME_TORCH_CUDA)


def _encoder_pack_paths(model_root: Path, spec: HybridIndexSpec) -> tuple[Path, ...]:
    semantic = _semantic_module()
    if _encoder_on_torch(spec):
        return cast(
            tuple[Path, ...],
            _torch_module().torch_encoder_pack_paths(semantic.encoder_root(model_root, spec)),
        )
    return cast(tuple[Path, ...], semantic.semantic_pack_paths(model_root, spec))


def _prove_capability(model_root: Path, spec: HybridIndexSpec) -> HybridCapabilityReport:
    probe = (
        _torch_module().probe_torch_encoder
        if _encoder_on_torch(spec)
        else _semantic_module().probe_semantic_pack
    )
    report = cast(HybridCapabilityReport, probe(model_root, spec))
    # The reranker pack is part of readiness: a generation is only useful if
    # the model that orders its hits is the one the spec names.
    details = tuple(
        sorted((*report.details, *_reranking_module().verify_reranker_pack(model_root, spec)))
    )
    payload = {**report.model_dump(mode="python", exclude={"logical_hash"}), "details": details}
    return HybridCapabilityReport.model_validate(
        {**payload, "logical_hash": sha256_hex(canonical_json_bytes(payload))}
    )


def _default_capability_probe(
    model_root: Path,
    spec: HybridIndexSpec,
) -> HybridCapabilityReport:
    """The packs proven by content once per process, reused while unchanged on disk."""

    semantic = _semantic_module()
    report, reused = semantic.VERIFIED_PACKS.resolve(
        ("capability", model_root.resolve().as_posix(), spec.logical_hash),
        consumed=(
            *_encoder_pack_paths(model_root, spec),
            *_reranking_module().reranker_pack_paths(model_root, spec),
        ),
        load=lambda: _prove_capability(model_root, spec),
    )
    if reused:
        semantic.MODEL_WORK.pack_probe_reuses += 1
    return cast(HybridCapabilityReport, report)


def _default_adapter_factory(model_root: Path, spec: HybridIndexSpec) -> Any:
    semantic = _semantic_module()
    if _encoder_on_torch(spec):
        torch_module = _torch_module()
        root = semantic.encoder_root(model_root, spec)
        return semantic.shared_embedding_adapter(
            model_root,
            spec,
            consumed=torch_module.torch_encoder_pack_paths(root),
            load=lambda: torch_module.TorchEmbeddingAdapter(root, spec),
        )
    return semantic.shared_embedding_adapter(model_root, spec)


def _reranking_module() -> Any:
    return importlib.import_module("alphalattice.kernel.knowledge._reranking")


def _default_reranker_factory(model_root: Path, spec: HybridIndexSpec) -> Any:
    return _reranking_module().shared_reranker_adapter(model_root, spec)


def _load_vector_extension(connection: sqlite3.Connection) -> str:
    return str(_semantic_module().load_vector_extension(connection))


def _serialize_vector(vector: Sequence[float]) -> bytes:
    return bytes(_semantic_module().serialize_vector(vector))


def _deserialize_vector(blob: bytes, *, dimension: int) -> tuple[float, ...]:
    return tuple(
        float(value) for value in _semantic_module().deserialize_vector(blob, dimension=dimension)
    )


def _semantic_unavailable_capability(
    spec: HybridIndexSpec, cause: str | None = None
) -> HybridCapabilityReport:
    """The pack, runtime or device is not usable; `cause` names which, as the
    probe stated it, so the refusal a person reads is the reason and not
    only the fact."""

    details = ["semantic-pack:unavailable"]
    if cause:
        details.append("cause:" + " ".join(cause.split())[:240])
    payload = {
        "schema_version": "1",
        "mode": KnowledgeRetrievalMode.STANDARD_HYBRID,
        "status": SemanticPackStatus.SEMANTIC_PACK_UNAVAILABLE,
        "model_id": spec.model_id,
        "model_revision": spec.model_revision,
        "model_artifact_sha256": spec.model_artifact_sha256,
        "tokenizer_sha256": spec.tokenizer_sha256,
        "onnx_runtime_version": spec.onnx_runtime_version,
        "sentencepiece_version": spec.sentencepiece_version,
        "vector_extension_version": spec.vector_extension_version,
        "cpu_execution_provider_only": False,
        "details": tuple(sorted(details)),
    }
    return HybridCapabilityReport.model_validate(
        {**payload, "logical_hash": sha256_hex(canonical_json_bytes(payload))}
    )


def probe_hybrid_retrieval_capabilities(
    model_root: Path,
    spec: HybridIndexSpec,
) -> HybridCapabilityReport:
    """Return explicit Standard semantic readiness without falling back."""
    try:
        return _default_capability_probe(model_root, spec)
    except KnowledgeRetrievalError:
        return _semantic_unavailable_capability(spec)


def _require_capability(report: HybridCapabilityReport) -> None:
    if report.status is not SemanticPackStatus.READY:
        raise _retrieval_error(
            "local semantic search pack is unavailable",
            code="retrieval.semantic_pack_unavailable",
            retryable=True,
        )


def corpus_logical_hash(snapshot: WorkspaceKnowledgeSnapshot) -> str:
    """The content of a snapshot without the snapshot.

    A snapshot's own identity carries the moment it was frozen and the id it
    was frozen under; what a generation embeds is its revisions -- each one a
    sealed commitment to a document, its source rights, its availability and
    its bytes -- in their canonical order. Two snapshots with the same
    revisions are one corpus, whatever asked for them and whenever.
    """
    return _canonical_hash(
        {
            "kind": "workspace-knowledge-corpus",
            "revisions": tuple(revision.logical_hash for revision in snapshot.revisions),
        }
    )


def _index_id(corpus_hash: str, spec: HybridIndexSpec) -> str:
    return _index_id_of(corpus_hash, spec.logical_hash)


def _index_id_of(corpus_hash: str, spec_logical_hash: str) -> str:
    return _canonical_hash(
        {
            "schema": _INDEX_SCHEMA_VERSION,
            "corpus_logical_hash": corpus_hash,
            "index_spec_logical_hash": spec_logical_hash,
        }
    )


def index_identity_of_database(database_relative: str) -> str | None:
    """The generation identity a `knowledge-hybrid-v4` database path names.

    The path is `<corpus>-<spec>.db` under the v4 root; the identity is a
    function of those two hashes and the schema, so an index file whose
    record is not (yet) sealed can still be matched against the leases this
    process holds. Anything else -- a legacy path, a foreign name -- is None.
    """
    prefix = f".system/knowledge-indexes/{_INDEX_SCHEMA_VERSION}/"
    if not database_relative.startswith(prefix) or not database_relative.endswith(".db"):
        return None
    stem = database_relative[len(prefix) : -3]
    corpus_hash, separator, spec_hash = stem.partition("-")
    if not separator or len(corpus_hash) != 64 or len(spec_hash) != 64:
        return None
    return _index_id_of(corpus_hash, spec_hash)


@dataclass(frozen=True, slots=True)
class KnowledgeInventoryWindow:
    """Hold authoritative source-order chunks and citations without query scores.

    One authoritative chunk of an open generation, as a structural scan
    reads it: the kernel's citation (the same identity a search hit carries),
    its position in the document and its normalised body. No score: nothing
    was queried, and the citation is what a later verified read proves.
    """

    citation: KnowledgeCitation
    ordinal: int
    body: str
    chunk_hash: str


def _cut_per_group(
    ranked: list[tuple[int, float, str, _Candidate]],
    request: HybridKnowledgeRetrievalRequest,
) -> list[tuple[int, float, str, _Candidate]]:
    """The final cut: one global `top_k`, or `top_k` for each document group.

    Groups give every issuer's filings their own places in what a query
    returns. A global cut of twenty over eight issuers' filings is decided by
    whichever filings say the most on the topic; an issuer whose best passage
    is the twenty-first is not returned and no later allocation can recover
    it. With groups the reranked order is walked once and each group takes
    its first `top_k`; documents in no group share one trailing group. The
    order of the hits is unchanged -- rank still means position in the
    reranked order -- only how far down each group it reaches.
    """

    if not request.document_groups:
        return ranked[: request.top_k]
    group_of = {
        str(document_id): index
        for index, group in enumerate(request.document_groups)
        for document_id in group
    }
    trailing = len(request.document_groups)
    taken: dict[int, int] = {}
    selected: list[tuple[int, float, str, _Candidate]] = []
    for item in ranked:
        group = group_of.get(str(item[3].row["document_id"]), trailing)
        if taken.get(group, 0) < request.top_k:
            taken[group] = taken.get(group, 0) + 1
            selected.append(item)
    return selected


def _request_index_id(request: HybridKnowledgeRetrievalRequest) -> str:
    if request.index_schema == _INDEX_SCHEMA_VERSION:
        return _canonical_hash(
            {
                "schema": _INDEX_SCHEMA_VERSION,
                "corpus_logical_hash": request.corpus_logical_hash,
                "index_spec_logical_hash": request.index_spec_logical_hash,
            }
        )
    return _canonical_hash(
        {
            "schema": _LEGACY_SCHEMA_VERSION,
            "snapshot_id": request.snapshot_id,
            "snapshot_logical_hash": request.snapshot_logical_hash,
            "index_spec_logical_hash": request.index_spec_logical_hash,
        }
    )


def _trigram_applies(normalized_query: str, spec: HybridIndexSpec) -> bool:
    """Whether the trigram channel can serve this query, by the query's shape.

    Trigram exists for substring and typo matching on one contiguous token: it
    asks the index for that exact run of characters. Asking it for a whole
    multi-word query asks for a phrase that a keyword query never is. Measured
    on the admitted 9,725-chunk corpus, the eight installed product queries are
    62 to 79 characters and returned **zero rows for 190.7 seconds combined**,
    while a single short term costs 0.88 seconds and returns its full depth.

    The rule is the query's shape and the index's own chunk size, never a clock
    or a machine speed: a probe longer than a chunk cannot be contained in one,
    and a probe shorter than a trigram cannot be tokenized at all.
    """

    if " " in normalized_query or "\t" in normalized_query or "\n" in normalized_query:
        return False
    return 3 <= len(normalized_query) <= spec.lexical_spec.chunk_size_codepoints


def _database_path(corpus_hash: str, spec: HybridIndexSpec) -> str:
    return committed_database_relative(corpus_hash, spec.logical_hash)


def committed_index_held(root: Path, database_relative: str) -> bool:
    """Inspect committed generation presence without treating disk presence as proof.

    Whether a committed generation's index is on disk under its identity
    and no approved eviction removed it: presence, never proof. A reader
    proves the projection when it opens; a caller that will not open it (a
    selection reused whole from a sealed receipt, record section Y) learns
    only that the normal build would find the index in place rather than
    restore it. An invalid marker refuses as it would at open.

    Args:
        root: Workspace root.
        database_relative: Confined committed-generation database path.

    Returns:
        True only when the database exists and no approved eviction marker removed it.

    Raises:
        KnowledgeRetrievalError: An eviction marker is invalid.
    """
    return (
        resolve_confined(root, database_relative).is_file()
        and _read_eviction_marker(root, database_relative) is None
    )


def committed_database_relative(corpus_hash: str, spec_logical_hash: str) -> str:
    """Where a `knowledge-hybrid-v4` generation's index lives, by its identity."""
    return f".system/knowledge-indexes/{_INDEX_SCHEMA_VERSION}/{corpus_hash}-{spec_logical_hash}.db"


def _payload_relative(payload_sha256: str) -> str:
    return f"{_VECTOR_PAYLOAD_ROOT}/{payload_sha256}.f32"


def _eviction_marker_relative(database_relative: str) -> str:
    return f"{database_relative}.evicted.json"


def _read_eviction_marker(root: Path, database_relative: str) -> dict[str, Any] | None:
    """The approved eviction that removed this index, if one did."""

    path = resolve_confined(root, _eviction_marker_relative(database_relative))
    if not path.is_file():
        return None
    try:
        marker = parse_model(path.read_bytes(), HybridIndexEvictionMarker)
    except (DomainValidationError, ValidationError) as exc:
        raise _retrieval_error(
            "hybrid index eviction marker is invalid",
            code="retrieval.index_corrupt",
            cause=exc,
        ) from exc
    return marker.model_dump(mode="python")


def _refuse_evicted(marker: dict[str, Any]) -> KnowledgeRetrievalError:
    return _retrieval_error(
        "hybrid knowledge index was evicted by an approved cleanup "
        f"(plan {str(marker['eviction_plan_hash'])[:12]}); rebuild it explicitly",
        code="retrieval.index_evicted",
    )


_FUNCTION_WORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "by",
        "during",
        "for",
        "from",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "our",
        "over",
        "that",
        "the",
        "these",
        "this",
        "those",
        "to",
        "under",
        "us",
        "was",
        "were",
        "we",
        "with",
        "about",
        "after",
        "before",
        "will",
        "may",
        "could",
        "would",
        "might",
        "have",
        "has",
        "had",
        "not",
        "no",
        "if",
        "their",
        "there",
        "them",
        "they",
        "other",
        "such",
        "any",
        "all",
        "more",
        "most",
        "than",
        "then",
        "when",
        "which",
        "who",
        "whom",
        "whose",
        "while",
    ]
)


def _content_terms(terms: tuple[str, ...]) -> tuple[str, ...]:
    """The terms a lexical channel should match on: the ones that carry content.

    A query written as a sentence carries function words, and matching on them
    is close to matching on nothing: measured on the admitted corpus, the legal
    authority query matched 8,757 of 9,725 chunks with its function words and
    995 without, because almost every passage in a filing contains "we", "our"
    and "the". That is paid twice -- the window has to rank nine tenths of the
    corpus, and bm25 spends its mass on words that do not discriminate.

    Dropped only for the lexical channels. The dense channel embeds the query as
    written, where the same words carry the grammar that makes it a sentence.
    """

    content = tuple(term for term in terms if term not in _FUNCTION_WORDS)
    return content or terms


def _embedding_input_hash(chunk: _Chunk, spec: HybridIndexSpec) -> str:
    if spec.embedding_input_policy.value == "E5_ASYMMETRIC_V1":
        value = f"passage: {chunk.body}"
    else:
        value = chunk.body
    return sha256_hex(value.encode("utf-8"))


def _corpus_citation_json(chunk: _Chunk) -> str:
    """The citation a generation stores: the source range without the snapshot.

    A chunk's citation names the snapshot it was derived under, which is what
    a hit must carry back to its caller. Stored rows are shared by every
    snapshot that resolves to the corpus, so they hold the citation minus the
    snapshot and a hit restores the requesting snapshot when it is returned.
    """

    payload = chunk.citation.model_dump(
        mode="python", exclude={"snapshot_id", "snapshot_logical_hash"}
    )
    return canonical_json_bytes(payload).decode("utf-8")


def _hybrid_rows(
    chunks: tuple[_Chunk, ...],
    spec: HybridIndexSpec,
    *,
    corpus_citations: bool = True,
) -> tuple[dict[str, Any], ...]:
    rows = []
    for chunk in chunks:
        row = {
            **_projection_row(chunk),
            "embedding_input_hash": _embedding_input_hash(chunk, spec),
        }
        if corpus_citations:
            row["citation_json"] = _corpus_citation_json(chunk)
        rows.append(row)
    return tuple(rows)


def _vector_payload(vectors: Sequence[Sequence[float]]) -> bytes:
    return b"".join(_serialize_vector(vector) for vector in vectors)


def _block_contents(composition: _Composition) -> tuple[bytes, ...]:
    """Each block's bytes, cut from the composed payload by its length."""

    contents = []
    offset = 0
    for block in composition.blocks:
        contents.append(composition.payload[offset : offset + block.byte_length])
        offset += block.byte_length
    return tuple(contents)


def _estimated_index_bytes(rows: Sequence[dict[str, Any]], payload_bytes: int) -> int:
    """An upper bound on the index a set of rows builds, before it is built.

    The projection stores each chunk's text in the content table and three
    FTS5 tables (term, CJK, trigram -- the trigram postings alone were 23 MB
    of a 71 MB retained index whose text was 15 MB), its citation, and its
    vector twice (vec0 and the byte-for-byte copy). Measured on the retained
    generations the index is 3.5-4.7 x its text plus its vectors; the bound
    used for admission is 5 x text + 2 x vectors + 1 MiB, exceeded by no
    retained index, so a build refused for capacity is refused before it
    stages anything.
    """

    text = sum(
        len(str(row.get(key, "")).encode("utf-8"))
        for row in rows
        for key in ("body_raw", "title_raw", "heading_raw", "citation_json")
    )
    return 5 * text + 2 * payload_bytes + 1024 * 1024


def _payload_blobs(payload: bytes, *, count: int, dimension: int) -> tuple[bytes, ...]:
    width = dimension * 4
    if len(payload) != count * width:
        raise _retrieval_error(
            "vector payload byte length differs from its commitment",
            code="retrieval.vector_commitment_mismatch",
        )
    return tuple(payload[index * width : (index + 1) * width] for index in range(count))


def _commitment_for(
    spec: HybridIndexSpec, *, count: int, payload_sha256: str
) -> HybridVectorCommitment:
    """The commitment a digest names, for reading a payload back by its anchor."""

    return HybridVectorCommitment(
        embedding_dimension=spec.embedding_dimension,
        dtype="float32",
        byte_order="little",
        vector_normalization=spec.vector_normalization,
        vector_count=count,
        payload_bytes=count * spec.embedding_dimension * 4,
        payload_sha256=payload_sha256,
        payload_path=_payload_relative(payload_sha256),
    )


def _commitment(spec: HybridIndexSpec, *, count: int, payload: bytes) -> HybridVectorCommitment:
    digest = sha256_hex(payload)
    return HybridVectorCommitment(
        embedding_dimension=spec.embedding_dimension,
        dtype="float32",
        byte_order="little",
        vector_normalization=spec.vector_normalization,
        vector_count=count,
        payload_bytes=len(payload),
        payload_sha256=digest,
        payload_path=_payload_relative(digest),
    )


def _read_vector_payload(
    root: Path, commitment: HybridVectorCommitment, *, manifest_logical_hash: str
) -> bytes:
    """The committed vectors, or a refusal that names what is missing or wrong.

    A payload is one file at its address, or the blocks the generation's
    sidecar names concatenated in order. Either way the bytes must hash to
    the commitment; a block-built payload also checks each block against
    the digest the sidecar records for it, so the refusal names the block.
    """

    path = resolve_confined(root, commitment.payload_path)
    if path.is_file():
        payload = path.read_bytes()
    else:
        sidecar = _read_block_sidecar(
            root, manifest_logical_hash, payload_sha256=commitment.payload_sha256
        )
        if sidecar is None:
            raise _retrieval_error(
                "committed vector payload is unavailable; rebuild or re-prepare explicitly",
                code="retrieval.vector_payload_unavailable",
            )
        payload = b"".join(_proved_blocks(root, sidecar, payload_sha256=commitment.payload_sha256))
    if len(payload) != commitment.payload_bytes or sha256_hex(payload) != commitment.payload_sha256:
        raise _retrieval_error(
            "vector payload differs from its commitment",
            code="retrieval.vector_commitment_mismatch",
        )
    return payload


READER_PAGE_CACHE_KIB = 65_536
"""The page cache an open reader's connection may hold, in KiB (64 MiB):
bounded per connection, allocated only for pages the reader touches,
released when the reader closes. Sized to hold a whole retained
generation's projection (30-200 MB on the QA copies) or the hot part of a
larger one; a build's verification connection that is not kept open
keeps SQLite's default."""

_REPAIRABLE = frozenset(
    {"retrieval.vector_payload_unavailable", "retrieval.vector_commitment_mismatch"}
)
"""What an explicit rebuild may replace: vector objects that are not held or
do not verify against the sealed commitment. A manifest or projection that
differs is another matter and refuses as before."""


def _sidecar_relative(manifest_logical_hash: str) -> str:
    return f"{_VECTOR_PAYLOAD_ROOT}/{manifest_logical_hash}{_SIDECAR_SUFFIX}"


def _read_block_sidecar(
    root: Path, manifest_logical_hash: str, *, payload_sha256: str
) -> HybridVectorBlockSidecar | None:
    path = resolve_confined(root, _sidecar_relative(manifest_logical_hash))
    if not path.is_file():
        return None
    try:
        sidecar = parse_model(path.read_bytes(), HybridVectorBlockSidecar)
    except (DomainValidationError, ValidationError) as exc:
        raise _retrieval_error(
            "vector block sidecar is invalid",
            code="retrieval.vector_commitment_mismatch",
            cause=exc,
        ) from exc
    if (
        sidecar.manifest_logical_hash != manifest_logical_hash
        or sidecar.payload_sha256 != payload_sha256
    ):
        raise _retrieval_error(
            "vector block sidecar names another generation",
            code="retrieval.vector_commitment_mismatch",
        )
    return sidecar


def _read_block(root: Path, block: HybridVectorBlock) -> bytes:
    path = resolve_confined(root, block.block_path)
    if not path.is_file():
        raise _retrieval_error(
            "committed vector block is unavailable; rebuild or re-prepare explicitly",
            code="retrieval.vector_payload_unavailable",
        )
    content = path.read_bytes()
    if len(content) != block.byte_length or sha256_hex(content) != block.block_sha256:
        raise _retrieval_error(
            "vector block differs from its commitment",
            code="retrieval.vector_commitment_mismatch",
        )
    return content


def _proved_blocks(
    root: Path, sidecar: HybridVectorBlockSidecar, *, payload_sha256: str
) -> tuple[bytes, ...]:
    """The blocks a sidecar names, read, each checked, and hashed as a whole to
    the committed digest; a sidecar that names other bytes cannot pass."""

    contents = tuple(_read_block(root, block) for block in sidecar.blocks)
    if sha256_hex(b"".join(contents)) != payload_sha256:
        raise _retrieval_error(
            "vector blocks do not compose the committed payload",
            code="retrieval.vector_commitment_mismatch",
        )
    return contents


def committed_payload_references(
    root: Path, *, manifest_logical_hash: str, payload_sha256: str
) -> tuple[str, ...]:
    """The files that hold one generation's committed vectors, proved.

    The payload file at the digest's address when it is there, and the
    sidecar with the blocks it names when the generation was composed (a
    one-revision composition is both: its one block is the payload) -- in
    every case the bytes are read and hashed to the committed digest before
    the files count as this generation's. Refuses by name when nothing
    verifiable holds the payload (`retrieval.vector_payload_unavailable`) or
    what is held does not compose it (`retrieval.vector_commitment_mismatch`).
    The set is never guessed: an inventory that cannot prove a generation's
    references does not know which vector objects that generation needs, and
    must treat every one as possibly required.
    """
    references: list[str] = []
    single = _payload_relative(payload_sha256)
    path = resolve_confined(root, single)
    if path.is_file():
        if sha256_hex(path.read_bytes()) != payload_sha256:
            raise _retrieval_error(
                "vector payload address holds other bytes than its commitment",
                code="retrieval.vector_commitment_mismatch",
            )
        references.append(single)
    sidecar = _read_block_sidecar(root, manifest_logical_hash, payload_sha256=payload_sha256)
    if sidecar is not None:
        _proved_blocks(root, sidecar, payload_sha256=payload_sha256)
        references.append(_sidecar_relative(manifest_logical_hash))
        references.extend(block.block_path for block in sidecar.blocks)
    if not references:
        raise _retrieval_error(
            "committed vector payload is unavailable; rebuild or re-prepare explicitly",
            code="retrieval.vector_payload_unavailable",
        )
    return tuple(dict.fromkeys(references))


def verified_block_map(
    root: Path,
    *,
    database_relative: str,
    manifest_logical_hash: str,
    payload_sha256: str,
) -> dict[str, bytes] | None:
    """Verify committed blocks and map exact encoder inputs to reusable vector assets.

    Every block of a committed generation by the asset key its committed
    manifest derives, proved as a whole.

    None when the generation has no sidecar (a payload written as one
    file; the slice path serves it). Nothing in the sidecar is trusted: the blocks
    it names are read and hashed to the committed digest, and each block's
    key is derived from the manifest the sealed record commits to -- the
    encoder context of its spec and the ordered encoder inputs at the block's
    position and count -- so a sidecar cannot map a key to other vectors, to
    other inputs or to another context. A sidecar whose recorded keys,
    context or dimension differ from that derivation, or that names two
    blocks under one key with different bytes, is refused by name and serves
    nothing. A generation whose committed manifest is not on disk (an evicted
    or lost index) proves no mapping: it is refused by name, its blocks serve
    no other corpus until the index is restored by an explicit rebuild, and
    its own cold readback (the whole payload against its digest) is unchanged.

    A manifest sealed before the final close-out (F1) names the threads its
    sessions ran on: its sidecar is proved against that context, and its blocks
    serve under the context without them -- the one a build asks with since F1,
    whose vectors the retrieval canary proves are the same at every thread count.

    Args:
        root: Workspace root.
        database_relative: Exact generation database path.
        manifest_logical_hash: Caller-held sealed manifest identity.
        payload_sha256: Caller-held sealed vector-payload digest.

    Returns:
        Verified vector bytes by exact asset key; None for a generation without a block sidecar.

    Raises:
        KnowledgeRetrievalError: The committed manifest, sidecar or vector blocks cannot be
            verified.
    """
    target = resolve_confined(root, database_relative)
    if not target.is_file():
        raise _retrieval_error(
            "the generation's committed manifest is not on disk; its blocks serve no "
            "other corpus until the index is rebuilt explicitly",
            code="retrieval.vector_block_mapping_unproved",
        )
    manifest = _read_manifest(target, HybridKnowledgeIndexManifest)
    if manifest.logical_hash != manifest_logical_hash:
        raise _retrieval_error(
            "hybrid index differs from the generation its sealed record commits to",
            code="retrieval.snapshot_mismatch",
        )
    if manifest.vector_commitment.payload_sha256 != payload_sha256:
        raise _retrieval_error(
            "hybrid manifest names a payload its sealed record does not",
            code="retrieval.index_corrupt",
        )
    sidecar = _read_block_sidecar(root, manifest_logical_hash, payload_sha256=payload_sha256)
    if sidecar is None:
        return None
    contents = _proved_blocks(root, sidecar, payload_sha256=payload_sha256)
    context = embedding_context_hash(manifest.index_spec)
    if (
        sidecar.embedding_context_hash != context
        or sidecar.embedding_dimension != manifest.index_spec.embedding_dimension
    ):
        raise _retrieval_error(
            "vector block sidecar names another encoder context than the committed manifest",
            code="retrieval.vector_commitment_mismatch",
        )
    inputs = manifest.ordered_embedding_input_hashes
    served = embedding_context_hash(manifest.index_spec.without_execution())
    block_map: dict[str, bytes] = {}
    offset = 0
    for block, content in zip(sidecar.blocks, contents, strict=True):
        placed = tuple(inputs[offset : offset + block.vector_count])
        key = vector_asset_key(context, placed)
        offset += block.vector_count
        if block.asset_key != key:
            raise _retrieval_error(
                "vector block sidecar maps a block to inputs the committed manifest does "
                "not place there",
                code="retrieval.vector_commitment_mismatch",
            )
        key = vector_asset_key(served, placed)
        held = block_map.get(key)
        if held is not None and held != content:
            raise _retrieval_error(
                "two vector blocks claim one asset key with different vectors",
                code="retrieval.vector_commitment_mismatch",
            )
        block_map[key] = content
    return block_map


def committed_payload_slice(
    root: Path,
    *,
    database_relative: str,
    manifest_logical_hash: str,
    payload_sha256: str,
    context_hash: str,
    embedding_input_hashes: tuple[str, ...],
) -> bytes | None:
    """One revision's vectors cut from a payload written as a single file.

    The generation's manifest must be the one the caller's sealed record
    commits to (it gives the chunk order and each chunk's encoder input), the
    payload must hash to the committed digest, and the requested inputs must
    appear as one contiguous run under the same encoder context (a manifest
    sealed before F1 under its context without threads, as `verified_block_map`
    serves one). The slice is then exactly those chunks' committed vectors.
    None when the index is not on disk, the inputs are not there, or the
    context differs.
    """
    target = resolve_confined(root, database_relative)
    if not target.is_file():
        return None
    manifest = _read_manifest(target, HybridKnowledgeIndexManifest)
    if manifest.logical_hash != manifest_logical_hash:
        raise _retrieval_error(
            "hybrid index differs from the generation its sealed record commits to",
            code="retrieval.snapshot_mismatch",
        )
    if (
        manifest.vector_commitment.payload_sha256 != payload_sha256
        or embedding_context_hash(manifest.index_spec.without_execution()) != context_hash
    ):
        return None
    inputs = manifest.ordered_embedding_input_hashes
    count = len(embedding_input_hashes)
    if count == 0 or count > len(inputs):
        return None
    start = -1
    for candidate in range(len(inputs) - count + 1):
        if inputs[candidate : candidate + count] == embedding_input_hashes:
            start = candidate
            break
    if start < 0:
        return None
    payload = _read_vector_payload(
        root, manifest.vector_commitment, manifest_logical_hash=manifest.logical_hash
    )
    width = manifest.index_spec.embedding_dimension * 4
    return payload[start * width : (start + count) * width]


@dataclass(frozen=True, slots=True)
class _Composition:
    """One corpus's vectors, composed from committed blocks and fresh passes."""

    payload: bytes
    vectors: tuple[tuple[float, ...], ...]
    blocks: tuple[HybridVectorBlock, ...]
    embedded_chunks: int
    reused_chunks: int
    model_calls: int


def _primed(
    manifest: HybridKnowledgeIndexManifest,
    chunks: tuple[_Chunk, ...],
    vectors: tuple[tuple[float, ...], ...],
    composition: _Composition | None,
) -> HybridIndexPriming:
    """A priming that carries what its composition cost, or nothing when reused."""

    if composition is None:
        return HybridIndexPriming(manifest=manifest, chunks=chunks, vectors=vectors, embedded=False)
    return HybridIndexPriming(
        manifest=manifest,
        chunks=chunks,
        vectors=vectors,
        embedded=composition.model_calls > 0,
        embedded_chunks=composition.embedded_chunks,
        reused_chunks=composition.reused_chunks,
        model_calls=composition.model_calls,
    )


def _revision_ranges(
    snapshot: WorkspaceKnowledgeSnapshot, chunks: tuple[_Chunk, ...]
) -> tuple[tuple[str, int, int], ...]:
    """Each revision's contiguous chunk range, in corpus order."""

    ranges: list[tuple[str, int, int]] = []
    position = 0
    for revision in snapshot.revisions:
        start = position
        while (
            position < len(chunks)
            and chunks[position].document_id == revision.document.document_id
            and chunks[position].revision == revision.revision
        ):
            position += 1
        if position > start:
            ranges.append((revision.logical_hash, start, position))
    if position != len(chunks):
        raise _retrieval_error(
            "chunks are not in snapshot revision order",
            code="retrieval.source_integrity",
        )
    return tuple(ranges)


def _manifest(
    *,
    corpus_hash: str,
    document_count: int,
    spec: HybridIndexSpec,
    rows: tuple[dict[str, Any], ...],
    commitment: HybridVectorCommitment,
) -> HybridKnowledgeIndexManifest:
    commitments = _source_commitments(
        corpus_hash=corpus_hash, document_count=document_count, spec=spec, rows=rows
    )
    payload = {
        "schema_version": "1",
        **{key: value for key, value in commitments.items() if key != "index_spec_logical_hash"},
        "database_path": _database_path(corpus_hash, spec),
        "index_spec": spec,
        "vector_commitment": commitment,
    }
    return HybridKnowledgeIndexManifest.model_validate(
        {**payload, "logical_hash": _canonical_hash(payload)}
    )


def _source_commitments(
    *,
    corpus_hash: str,
    document_count: int,
    spec: HybridIndexSpec,
    rows: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    """What a source re-derivation reproduces; compared against a stored manifest."""

    return {
        "index_id": _index_id(corpus_hash, spec),
        "corpus_logical_hash": corpus_hash,
        "index_spec_logical_hash": spec.logical_hash,
        "document_count": document_count,
        "chunk_count": len(rows),
        "ordered_chunk_ids": tuple(row["chunk_id"] for row in rows),
        "ordered_citation_hashes": tuple(
            sha256_hex(str(row["citation_json"]).encode("utf-8")) for row in rows
        ),
        "ordered_embedding_input_hashes": tuple(row["embedding_input_hash"] for row in rows),
        "projection_logical_hash": _canonical_hash(rows),
    }


AnyHybridManifest = HybridKnowledgeIndexManifest | HybridV3KnowledgeIndexManifest


def _manifest_of[ManifestT: AnyHybridManifest](
    connection: sqlite3.Connection, model: type[ManifestT]
) -> ManifestT:
    """The manifest row of an open generation, parsed and required canonical."""

    with closing(connection.execute("SELECT value FROM metadata WHERE key = 'manifest'")) as cursor:
        row = cursor.fetchone()
    if row is None:
        raise sqlite3.DatabaseError("hybrid manifest is missing")
    raw = str(row[0]).encode("utf-8")
    manifest = parse_model(raw, model)
    if raw != canonical_json_bytes(manifest):
        raise sqlite3.DatabaseError("hybrid manifest encoding is non-canonical")
    return manifest


def _read_manifest[ManifestT: AnyHybridManifest](path: Path, model: type[ManifestT]) -> ManifestT:
    try:
        _reject_sidecars(path)
        with closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)) as connection:
            manifest = _manifest_of(connection, model)
            with closing(connection.execute("PRAGMA integrity_check")) as cursor:
                integrity = cursor.fetchone()
            if integrity is None or integrity[0] != "ok":
                raise sqlite3.DatabaseError("hybrid index integrity_check failed")
            return manifest
    except (sqlite3.Error, DomainValidationError, ValidationError) as exc:
        raise _retrieval_error(
            "hybrid knowledge index is invalid",
            code="retrieval.index_corrupt",
            cause=exc,
        ) from exc


def _reject_sidecars(path: Path, *, publication: bool = False) -> None:
    sidecars = tuple(
        candidate
        for candidate in (
            path.with_name(f"{path.name}-wal"),
            path.with_name(f"{path.name}-shm"),
            path.with_name(f"{path.name}-journal"),
        )
        if candidate.exists()
    )
    if sidecars:
        raise _retrieval_error(
            (
                "hybrid index publication retained SQLite sidecar files"
                if publication
                else "active hybrid index has SQLite sidecar files"
            ),
            code=("retrieval.index_publish_blocked" if publication else "retrieval.index_corrupt"),
            retryable=publication,
        )


def _vector_matches(
    expected: Sequence[float],
    actual: Sequence[float],
    *,
    tolerance: float,
) -> bool:
    return len(expected) == len(actual) and all(
        abs(left - right) <= tolerance for left, right in zip(expected, actual, strict=True)
    )


def _projection_keys() -> tuple[str, ...]:
    return (
        "chunk_id",
        "document_id",
        "revision",
        "ordinal",
        "namespace",
        "access_class",
        "available_at",
        "expires_at",
        "title_raw",
        "heading_raw",
        "body_raw",
        "identifier_norm",
        "title_norm",
        "heading_norm",
        "identifier_terms",
        "title_terms",
        "heading_terms",
        "body_terms",
        "cjk_terms",
        "citation_json",
        "chunk_hash",
        "embedding_input_hash",
    )


def _verify_projection(
    path: Path,
    manifest: AnyHybridManifest,
    *,
    expected_blobs: Sequence[bytes] | None = None,
    expected_vectors: Sequence[Sequence[float]] | None = None,
    keep_open: bool = False,
) -> sqlite3.Connection | None:
    """Prove the published rows against the manifest, the vectors against
    their commitment, and the trigram postings against the content they index.

    Committed generations compare every stored vector byte for byte with the
    payload the manifest commits to; legacy generations compare with vectors
    the caller embedded, within the spec's tolerance.
    """

    if (expected_blobs is None) == (expected_vectors is None):
        raise ValueError("exactly one vector expectation is required")
    expected_count = len(expected_blobs if expected_blobs is not None else expected_vectors)  # type: ignore[arg-type]
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        if keep_open:
            # The reader's page cache, bounded and connection-local: SQLite's
            # default of 2 MB re-reads a generation's pages from the operating
            # system on every query. Measured on the development copy's KO
            # unit (a 78 MB projection): one search read 240-450 MB in about
            # 62,000 ReadFile calls -- the dense channel's vec0 lookup a
            # scoped row, the per-hit verification's vec0 lookup a hit, and
            # the exact channels' index scans -- pages this cache holds
            # after the first query. Nothing about a query, its order or its
            # verification changes; the cache is released with the reader.
            connection.execute(f"PRAGMA cache_size = -{READER_PAGE_CACHE_KIB}")
        loaded_version = _load_vector_extension(connection)
        if loaded_version != manifest.index_spec.vector_extension_version:
            raise sqlite3.DatabaseError("sqlite-vec version differs from manifest")
        with closing(
            connection.execute(
                """
                SELECT
                    c.*,
                    t.identifier_terms,
                    t.title_terms,
                    t.heading_terms,
                    t.body_terms,
                    j.cjk_terms,
                    g.identifier_raw AS trigram_identifier_raw,
                    g.title_raw AS trigram_title_raw,
                    g.heading_raw AS trigram_heading_raw,
                    g.body_raw AS trigram_body_raw,
                    n.chunk_id AS dense_chunk_id,
                    n.embedding_input_hash,
                    n.vector_hash,
                    v.embedding
                FROM content AS c
                JOIN term_fts AS t ON t.rowid = c.rowid
                JOIN cjk_fts AS j ON j.rowid = c.rowid
                JOIN trigram_fts AS g ON g.rowid = c.rowid
                JOIN dense_nodes AS n ON n.rowid = c.rowid
                JOIN vec_chunks AS v ON v.rowid = c.rowid
                ORDER BY c.rowid
                """
            )
        ) as cursor:
            raw_rows = cursor.fetchall()
        for table in (
            "content",
            "term_fts",
            "cjk_fts",
            "trigram_fts",
            "dense_nodes",
            "vec_chunks",
        ):
            with closing(connection.execute(f"SELECT COUNT(*) FROM {table}")) as cursor:
                count_row = cursor.fetchone()
            if count_row is None or int(count_row[0]) != manifest.chunk_count:
                raise sqlite3.DatabaseError(
                    f"hybrid projection table has unknown or missing rows: {table}"
                )
        rows = tuple({key: row[key] for key in _projection_keys()} for row in raw_rows)
        if len(raw_rows) != expected_count:
            raise sqlite3.DatabaseError("dense row count differs from expected vectors")
        for index, row in enumerate(raw_rows):
            blob = bytes(row["embedding"])
            if expected_blobs is not None:
                vector_ok = blob == expected_blobs[index]
            else:
                assert expected_vectors is not None
                vector_ok = _vector_matches(
                    expected_vectors[index],
                    _deserialize_vector(blob, dimension=manifest.index_spec.embedding_dimension),
                    tolerance=manifest.index_spec.vector_tolerance,
                )
            if not vector_ok:
                raise _retrieval_error(
                    "stored vector differs from its commitment",
                    code=(
                        "retrieval.vector_commitment_mismatch"
                        if expected_blobs is not None
                        else "retrieval.index_corrupt"
                    ),
                )
            if (
                row["dense_chunk_id"] != row["chunk_id"]
                or str(row["vector_hash"]) != sha256_hex(blob)
                or row["trigram_identifier_raw"] != row["document_id"]
                or row["trigram_title_raw"] != row["title_raw"]
                or row["trigram_heading_raw"] != row["heading_raw"]
                or row["trigram_body_raw"] != row["body_raw"]
            ):
                raise sqlite3.DatabaseError("hybrid projection differs from source")
        if (
            len(rows) != manifest.chunk_count
            or tuple(row["chunk_id"] for row in rows) != manifest.ordered_chunk_ids
            or tuple(sha256_hex(str(row["citation_json"]).encode("utf-8")) for row in rows)
            != manifest.ordered_citation_hashes
            or tuple(row["embedding_input_hash"] for row in rows)
            != manifest.ordered_embedding_input_hashes
            or _canonical_hash(rows) != manifest.projection_logical_hash
        ):
            raise sqlite3.DatabaseError("hybrid projection differs from manifest")
        _verify_trigram_postings(connection)
        if keep_open:
            return connection
        connection.close()
        return None
    except KnowledgeRetrievalError:
        if connection is not None:
            connection.close()
        raise
    except (sqlite3.Error, TypeError, ValueError) as exc:
        if connection is not None:
            connection.close()
        raise _retrieval_error(
            "hybrid knowledge projection is invalid",
            code="retrieval.index_corrupt",
            cause=exc,
        ) from exc


def _verify_trigram_postings(connection: sqlite3.Connection) -> None:
    """Prove the trigram postings against the content they index.

    The trigram table is an FTS5 external-content table: it stores no text of
    its own, so a JOIN or a COUNT reads the content table through it and
    proves nothing about its postings, and `PRAGMA integrity_check` verifies
    the FTS5 structure without comparing it to external content. Measured on
    the locked SQLite 3.53.1: a row's postings removed with the FTS5 'delete'
    command leave the pragma, the JOIN and the COUNT passing while a MATCH no
    longer finds the row. The term and CJK tables keep their own content,
    which that pragma does compare. FTS5's own integrity-check command with
    its second argument set compares an external-content index with its
    content, but it is an INSERT and a read-only connection refuses it -- so
    it runs against a backup of the index in memory: the retained file is
    never written by a read, and nothing on disk changes. About 0.1 s on the
    4,912-chunk retained index (a 27 ms copy, a 70 ms check).
    """

    with closing(sqlite3.connect(":memory:")) as copy:
        connection.backup(copy)
        try:
            copy.execute("INSERT INTO trigram_fts(trigram_fts, rank) VALUES ('integrity-check', 1)")
        except sqlite3.DatabaseError as exc:
            raise sqlite3.DatabaseError(
                "trigram postings do not correspond to the content they index"
            ) from exc


@dataclass(frozen=True, slots=True)
class HybridIndexPriming:
    """What a build already proved, carried to the first query of that generation.

    Embedding the corpus is the expensive half of retrieval. A build embeds it
    to publish the projection, and the first query used to embed it a second
    time only to verify the projection it had just written. This carries the
    exact chunks and vectors of that one pass instead.

    Chunks and vectors, never a connection. A SQLite handle belongs to the
    thread that opened it, and the thread that builds a generation is not
    always the thread that queries it, so the connection stays lazy and is
    opened on the querying thread from these vectors. A build's proof of
    the published projection is not carried either: it stands for the
    bytes that build read, not for the file as it is when the reader
    opens -- rows deleted between the two never become hits, so no per-hit
    check could notice them (record section Y) -- and the reader proves
    the projection on its own connection at first open.
    """

    manifest: HybridKnowledgeIndexManifest
    chunks: tuple[_Chunk, ...]
    vectors: tuple[tuple[float, ...], ...]
    embedded: bool = True
    """Whether this build ran the passage model, or reused a generation whose
    committed vectors it verified instead."""
    embedded_chunks: int = 0
    """Chunks the passage model embedded in this build."""
    reused_chunks: int = 0
    """Chunks whose vectors came from committed blocks of other generations."""
    model_calls: int = 0
    """Passage-model calls this build made (one per embedded revision)."""
    derivation_proof: str | None = None
    """Set by this process's own build: the chunks were derived from the
    authoritative library for exactly this manifest and snapshot (a digest
    keyed by a process-private nonce, `_derivation_proof`). A priming
    without it, or with a proof that does not match, is a claim the primed
    open re-derives the corpus to test. The proof binds the chunks'
    content (`_chunks_digest`), so a body altered, swapped or reordered
    beside a genuine proof does not match it."""


_DERIVATION_NONCE = secrets.token_bytes(32)
"""Process-private: a derivation proof is worth nothing outside the process
whose build derived the chunks."""


def _derivation_proof(
    manifest: HybridKnowledgeIndexManifest,
    snapshot_logical_hash: str,
    chunks: Sequence[_Chunk],
) -> str:
    """The proof a build attaches to its priming: this process derived
    exactly these chunks from the snapshot named, under the manifest's
    spec, and proved the manifest against them. Recomputed at the primed
    open over the chunks the priming carries and compared, so the open
    need not derive the rows again to know the chunks are the manifest's."""

    return sha256_hex(
        _DERIVATION_NONCE
        + manifest.logical_hash.encode("utf-8")
        + snapshot_logical_hash.encode("utf-8")
        + manifest.corpus_logical_hash.encode("utf-8")
        + manifest.index_spec.logical_hash.encode("utf-8")
        + _chunks_digest(chunks).encode("utf-8")
    )


def _chunks_digest(chunks: Sequence[_Chunk]) -> str:
    """The chunks' own content, in order: what the derivation proof binds
    beside the manifest. Measured on the KO unit, re-deriving the rows and
    the source commitments from the primed chunks to prove the same thing
    cost 3.7 s an open; hashing the content costs milliseconds."""

    digest = hashlib.sha256()
    for chunk in chunks:
        for part in (
            chunk.chunk_id,
            chunk.chunk_hash,
            chunk.document_id,
            str(chunk.revision),
            str(chunk.ordinal),
            chunk.title,
            chunk.heading,
            "\x1f".join(chunk.heading_path),
            chunk.body,
            _corpus_citation_json(chunk),
            chunk.namespace.value,
            chunk.access_class.value,
            _utc_text(chunk.available_at),
            "" if chunk.expires_at is None else _utc_text(chunk.expires_at),
            str(chunk.character_start),
            str(chunk.character_end),
        ):
            digest.update(part.encode("utf-8"))
            digest.update(b"\x1e")
        digest.update(b"\x1d")
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class HybridPrimingFacts:
    """The cost of the build behind a primed reader, without its vectors."""

    embedded: bool = False
    embedded_chunks: int = 0
    reused_chunks: int = 0
    model_calls: int = 0

    @classmethod
    def of(cls, priming: HybridIndexPriming | None) -> HybridPrimingFacts:
        if priming is None:
            return cls()
        return cls(
            embedded=priming.embedded,
            embedded_chunks=priming.embedded_chunks,
            reused_chunks=priming.reused_chunks,
            model_calls=priming.model_calls,
        )


VectorBlockSource = Callable[[str, tuple[str, ...]], bytes | None]
"""`(embedding_context_hash, embedding_input_hashes) -> verified block bytes`:
the caller's way of producing one revision's committed vectors from a
generation it sealed, or None when it holds none for those inputs."""


class WorkspaceHybridKnowledgeIndex:
    """Build one immutable hybrid generation from one verified WP70A snapshot."""

    def __init__(
        self,
        workspace: Workspace,
        *,
        model_root: Path,
        capability_probe: Callable[
            [Path, HybridIndexSpec], HybridCapabilityReport
        ] = _default_capability_probe,
        adapter_factory: Callable[[Path, HybridIndexSpec], Any] = _default_adapter_factory,
        reranker_factory: Callable[[Path, HybridIndexSpec], Any] = _default_reranker_factory,
        replacer: Callable[[Path, Path], None] = replace_with_retry,
    ) -> None:
        """Configure local hybrid-index dependencies without building a generation.

        Args:
            workspace: Workspace owning authoritative knowledge and index artifacts.
            model_root: Local root containing the installed encoder and reranker packs.
            capability_probe: Capability checker invoked before semantic work.
            adapter_factory: Factory for the recipe-bound passage/query encoder.
            reranker_factory: Factory for the recipe-bound cross encoder.
            replacer: Publication replacement operation used for index files.
        """
        self.workspace = workspace
        self.model_root = model_root
        self.library = WorkspaceKnowledgeLibrary(workspace)
        self._capability_probe = capability_probe
        self._adapter_factory = adapter_factory
        self._reranker_factory = reranker_factory
        self._replacer = replacer

    def rebuild(
        self,
        snapshot: WorkspaceKnowledgeSnapshot,
        index_spec: HybridIndexSpec,
        *,
        cancelled: Callable[[], bool] | None = None,
        anchor_for: Callable[[str], HybridGenerationAnchor | None] | None = None,
        admit: Callable[[int], None] | None = None,
        block_for: VectorBlockSource | None = None,
    ) -> HybridKnowledgeIndexManifest:
        """Publish one generation. The caller reopens it cold and pays a second pass."""
        return self._rebuild(
            snapshot,
            index_spec,
            cancelled=cancelled,
            anchor_for=anchor_for,
            admit=admit,
            block_for=block_for,
        ).manifest

    def rebuild_and_open(
        self,
        snapshot: WorkspaceKnowledgeSnapshot,
        index_spec: HybridIndexSpec,
        *,
        cancelled: Callable[[], bool] | None = None,
        anchor_for: Callable[[str], HybridGenerationAnchor | None] | None = None,
        admit: Callable[[int], None] | None = None,
        block_for: VectorBlockSource | None = None,
    ) -> WorkspaceHybridKnowledgeRetriever:
        """Publish one generation and hand its verified vectors to a reader.

        The reader is not open yet: it holds the chunks and vectors this build
        proved, and opens its connection on the thread that first queries it.
        `anchor_for(index_id)` is the caller's durable commitment for the
        generation this corpus and spec identify, or None when the caller has
        sealed none; `admit(additional_bytes)` is the caller's capacity
        admission, consulted before anything is staged and again, exactly,
        before anything is placed.
        """
        priming = self._rebuild(
            snapshot,
            index_spec,
            cancelled=cancelled,
            anchor_for=anchor_for,
            admit=admit,
            block_for=block_for,
        )
        # This build derived the chunks from the authoritative library and
        # proved the manifest against them: the reader it primes need not
        # derive the same corpus again to know that.
        priming = replace(
            priming,
            derivation_proof=_derivation_proof(
                priming.manifest, snapshot.logical_hash, priming.chunks
            ),
        )
        return WorkspaceHybridKnowledgeRetriever(
            self.workspace,
            model_root=self.model_root,
            index_spec=index_spec,
            capability_probe=self._capability_probe,
            adapter_factory=self._adapter_factory,
            reranker_factory=self._reranker_factory,
            priming=priming,
        )

    def _rebuild(
        self,
        snapshot: WorkspaceKnowledgeSnapshot,
        index_spec: HybridIndexSpec,
        *,
        cancelled: Callable[[], bool] | None = None,
        anchor_for: Callable[[str], HybridGenerationAnchor | None] | None = None,
        admit: Callable[[int], None] | None = None,
        block_for: VectorBlockSource | None = None,
    ) -> HybridIndexPriming:
        """Materialize the generation the caller's commitment names, or embed one.

        The corpus is derived from the authoritative snapshot first. When the
        caller holds a sealed commitment for this corpus and spec, the
        generation is materialized against it: an index on disk is reused only
        if its manifest is the committed one and its rows hold the committed
        payload byte for byte; an evicted or missing index is rebuilt from the
        committed payload with no model, or, with the payload lost, from one
        model pass whose output must reproduce the committed digest. Nothing
        found on disk is its own authority. Without a commitment the corpus is
        embedded under the admitted build; an index already at its identity is
        kept only if this pass reproduces it exactly, and refused otherwise.
        """

        _require_capability(self._capability_probe(self.model_root, index_spec))
        authoritative = self.library.read_snapshot(snapshot.snapshot_id)
        if authoritative != snapshot:
            raise _retrieval_error(
                "snapshot differs from authoritative publication",
                code="retrieval.snapshot_mismatch",
            )
        corpus_hash = corpus_logical_hash(snapshot)
        chunks = _build_chunks(snapshot, self.library, index_spec.lexical_spec)
        rows = _hybrid_rows(chunks, index_spec)
        document_count = len(snapshot.revisions)
        index_id = _index_id(corpus_hash, index_spec)
        anchor = anchor_for(index_id) if anchor_for is not None else None
        if anchor is not None:
            with self.workspace.lock():
                return self._materialize_committed(
                    anchor,
                    corpus_hash=corpus_hash,
                    document_count=document_count,
                    spec=index_spec,
                    rows=rows,
                    chunks=chunks,
                    snapshot=snapshot,
                    cancelled=cancelled,
                    admit=admit,
                    block_for=block_for,
                    replace_divergent=False,
                )

        # No sealed commitment names this generation: whatever the store holds
        # for it (an index a build placed before its record was sealed, a
        # marker, a payload) is not trusted. The corpus is composed now: each
        # revision's block from a generation the caller sealed when it holds
        # one, embedded otherwise.
        composition = self._compose_vectors(
            snapshot, chunks, rows, index_spec, block_for=block_for, cancelled=cancelled
        )
        payload, vectors = composition.payload, composition.vectors
        commitment = _commitment(index_spec, count=len(vectors), payload=payload)
        manifest = _manifest(
            corpus_hash=corpus_hash,
            document_count=document_count,
            spec=index_spec,
            rows=rows,
            commitment=commitment,
        )
        blobs = _payload_blobs(
            payload, count=len(vectors), dimension=index_spec.embedding_dimension
        )
        target_relative = _database_path(corpus_hash, index_spec)
        with self.workspace.lock():
            later = anchor_for(index_id) if anchor_for is not None else None
            if later is not None:
                # Another writer sealed this generation while the model ran:
                # its commitment wins; this composition is verified against it
                # and dropped if it does not reproduce it.
                return self._materialize_committed(
                    later,
                    corpus_hash=corpus_hash,
                    document_count=document_count,
                    spec=index_spec,
                    rows=rows,
                    chunks=chunks,
                    snapshot=snapshot,
                    cancelled=cancelled,
                    admit=admit,
                    block_for=block_for,
                    replace_divergent=False,
                    computed=composition,
                )
            target = resolve_confined(self.workspace.root, target_relative)
            if target.is_file():
                _reject_sidecars(target, publication=True)
                existing = _read_manifest(target, HybridKnowledgeIndexManifest)
                if existing != manifest:
                    raise _retrieval_error(
                        "an index no sealed record names holds other vectors at this "
                        "generation's identity; release it through cleanup",
                        code="retrieval.index_publish_conflict",
                    )
                # The orphan is exactly what this composition produced: proved
                # by the recomputation, not by its own consistency.
                _verify_projection(target, existing, expected_blobs=blobs)
                if admit is not None:
                    admit(self._new_block_bytes(composition, index_spec, existing.logical_hash))
                self._publish_blocks(composition, index_spec, existing.logical_hash)
                return _primed(existing, chunks, vectors, composition)
            if admit is not None:
                admit(
                    self._new_block_bytes(composition, index_spec, manifest.logical_hash)
                    + _estimated_index_bytes(rows, len(payload))
                )
            published = self._publish_index(
                target_relative,
                rows,
                blobs,
                manifest,
                before_place=lambda staged: self._place_blocks(
                    composition,
                    index_spec,
                    manifest.logical_hash,
                    admit=admit,
                    staged_bytes=staged,
                ),
            )
            self._clear_marker(target_relative)
            return _primed(published, chunks, vectors, composition)

    def _compose_vectors(
        self,
        snapshot: WorkspaceKnowledgeSnapshot,
        chunks: tuple[_Chunk, ...],
        rows: tuple[dict[str, Any], ...],
        spec: HybridIndexSpec,
        *,
        block_for: VectorBlockSource | None,
        cancelled: Callable[[], bool] | None,
    ) -> _Composition:
        """The corpus's vectors, one block per revision.

        A revision whose block the caller can produce from a sealed generation
        -- the same encoder inputs under the same encoder context -- reuses
        it; every other revision is embedded now, one model call per
        revision. The pinned encoder gives each chunk the same vector whatever
        else is in its batch (measured bit-for-bit on the retained filings at
        478 and 4,912 chunks), so a corpus composed this way is the corpus
        embedded in one pass.
        """

        context = embedding_context_hash(spec)
        width = spec.embedding_dimension * 4
        blocks: list[HybridVectorBlock] = []
        parts: list[bytes] = []
        embedded = reused = calls = 0
        adapter: Any = None
        try:
            for revision_hash, start, end in _revision_ranges(snapshot, chunks):
                inputs = tuple(str(row["embedding_input_hash"]) for row in rows[start:end])
                key = vector_asset_key(context, inputs)
                content = block_for(context, inputs) if block_for is not None else None
                if content is not None and len(content) != (end - start) * width:
                    raise _retrieval_error(
                        "reused vector block does not match its revision's chunks",
                        code="retrieval.vector_commitment_mismatch",
                    )
                if content is None:
                    if adapter is None:
                        adapter = self._adapter_factory(self.model_root, spec)
                    vectors = self._embed_with(
                        adapter, chunks[start:end], spec, cancelled=cancelled
                    )
                    content = _vector_payload(vectors)
                    calls += 1
                    embedded += end - start
                else:
                    reused += end - start
                blocks.append(
                    HybridVectorBlock(
                        revision_logical_hash=revision_hash,
                        asset_key=key,
                        block_sha256=sha256_hex(content),
                        vector_count=end - start,
                        byte_length=len(content),
                    )
                )
                parts.append(content)
        finally:
            if adapter is not None:
                adapter.close()
        payload = b"".join(parts)
        vectors_out = tuple(
            _deserialize_vector(blob, dimension=spec.embedding_dimension)
            for blob in _payload_blobs(
                payload, count=len(chunks), dimension=spec.embedding_dimension
            )
        )
        return _Composition(
            payload=payload,
            vectors=vectors_out,
            blocks=tuple(blocks),
            embedded_chunks=embedded,
            reused_chunks=reused,
            model_calls=calls,
        )

    @staticmethod
    def _embed_with(
        adapter: Any,
        chunks: tuple[_Chunk, ...],
        index_spec: HybridIndexSpec,
        *,
        cancelled: Callable[[], bool] | None,
    ) -> tuple[tuple[float, ...], ...]:
        embedded = adapter.embed_passages(
            tuple(chunk.body for chunk in chunks),
            cancelled=cancelled,
        )
        vectors = tuple(tuple(float(value) for value in vector) for vector in embedded)
        if len(vectors) != len(chunks) or any(
            len(vector) != index_spec.embedding_dimension for vector in vectors
        ):
            raise _retrieval_error(
                "semantic runtime returned an inconsistent vector count",
                code="retrieval.semantic_pack_unavailable",
                retryable=True,
            )
        return vectors

    def _new_block_bytes(
        self, composition: _Composition, spec: HybridIndexSpec, manifest_logical_hash: str
    ) -> int:
        """The bytes publishing these blocks adds: blocks not yet held, and the sidecar."""

        added = 0
        for block in composition.blocks:
            if not resolve_confined(self.workspace.root, block.block_path).is_file():
                added += block.byte_length
        return added + len(
            canonical_json_bytes(self._sidecar(composition, spec, manifest_logical_hash))
        )

    @staticmethod
    def _sidecar(
        composition: _Composition, spec: HybridIndexSpec, manifest_logical_hash: str
    ) -> HybridVectorBlockSidecar:
        return HybridVectorBlockSidecar(
            manifest_logical_hash=manifest_logical_hash,
            payload_sha256=sha256_hex(composition.payload),
            embedding_context_hash=embedding_context_hash(spec),
            embedding_dimension=spec.embedding_dimension,
            blocks=composition.blocks,
        )

    def _publish_blocks(
        self,
        composition: _Composition,
        spec: HybridIndexSpec,
        manifest_logical_hash: str,
        *,
        repair: bool = False,
    ) -> None:
        """Place each block at its content address, then the sidecar, once, durably.

        A block already held must hold these exact bytes. The sidecar is the
        last write: a crash before it leaves content-addressed blocks a later
        build re-verifies or re-derives, never a payload nobody can compose.
        Under an explicit rebuild (`repair`) an object at one of these
        addresses holding other bytes is replaced: the composition reproduced
        the committed digest first, so what is written is the committed
        generation's own metadata and blocks, and a damaged sidecar or block
        is the reason the rebuild was asked for.
        """

        for block, content in zip(composition.blocks, _block_contents(composition), strict=True):
            self._place_immutable(block.block_path, content, block.block_sha256, repair=repair)
        sidecar = self._sidecar(composition, spec, manifest_logical_hash)
        self._place_immutable(
            _sidecar_relative(manifest_logical_hash),
            canonical_json_bytes(sidecar),
            None,
            repair=repair,
        )

    def _place_immutable(
        self, relative: str, content: bytes, digest: str | None, *, repair: bool = False
    ) -> None:
        target = make_confined_parents(self.workspace.root, relative)
        target = resolve_confined(self.workspace.root, relative)
        if target.is_file():
            if target.read_bytes() == content:
                return
            if not repair:
                raise _retrieval_error(
                    "vector object address holds other bytes",
                    code="retrieval.vector_commitment_mismatch",
                )
        staging_relative = f".system/staging/knowledge-vectors.{uuid4()}.tmp"
        staging = make_confined_parents(self.workspace.root, staging_relative)
        staging = resolve_confined(self.workspace.root, staging_relative)
        try:
            with staging.open("wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            written = staging.read_bytes()
            if written != content or (digest is not None and sha256_hex(written) != digest):
                raise _retrieval_error(
                    "vector object did not read back as written",
                    code="retrieval.vector_commitment_mismatch",
                )
            self._replacer(staging, target)
        finally:
            with suppress(OSError):
                staging.unlink(missing_ok=True)

    def _place_blocks(
        self,
        composition: _Composition,
        spec: HybridIndexSpec,
        manifest_logical_hash: str,
        *,
        admit: Callable[[int], None] | None,
        staged_bytes: int,
        repair: bool = False,
    ) -> None:
        """Admit the exact bytes about to be placed, then place the immutable children."""

        if admit is not None:
            admit(self._new_block_bytes(composition, spec, manifest_logical_hash) + staged_bytes)
        self._publish_blocks(composition, spec, manifest_logical_hash, repair=repair)

    def _held_payload(
        self,
        anchor: HybridGenerationAnchor,
        *,
        count: int,
        spec: HybridIndexSpec,
        computed: _Composition | None,
        repair: bool,
    ) -> bytes | None:
        """The committed payload from the blocks its sidecar names, or None when a
        block is not held (then it is composed again); other bytes refuse --
        except under an explicit rebuild, where a sidecar or block that does
        not verify is what the rebuild replaces, so it is composed again."""

        if computed is not None:
            return None
        try:
            if (
                _read_block_sidecar(
                    self.workspace.root,
                    anchor.manifest_logical_hash,
                    payload_sha256=anchor.payload_sha256,
                )
                is None
            ):
                return None
            return _read_vector_payload(
                self.workspace.root,
                _commitment_for(spec, count=count, payload_sha256=anchor.payload_sha256),
                manifest_logical_hash=anchor.manifest_logical_hash,
            )
        except KnowledgeRetrievalError as exc:
            if exc.failure.code == "retrieval.vector_payload_unavailable" or (
                repair and exc.failure.code == "retrieval.vector_commitment_mismatch"
            ):
                return None
            raise

    def _clear_marker(self, target_relative: str) -> None:
        marker_path = resolve_confined(
            self.workspace.root, _eviction_marker_relative(target_relative)
        )
        with suppress(OSError):
            marker_path.unlink(missing_ok=True)

    def _materialize_committed(
        self,
        anchor: HybridGenerationAnchor,
        *,
        corpus_hash: str,
        document_count: int,
        spec: HybridIndexSpec,
        rows: tuple[dict[str, Any], ...],
        chunks: tuple[_Chunk, ...],
        snapshot: WorkspaceKnowledgeSnapshot,
        cancelled: Callable[[], bool] | None,
        admit: Callable[[int], None] | None,
        block_for: VectorBlockSource | None,
        replace_divergent: bool,
        computed: _Composition | None = None,
    ) -> HybridIndexPriming:
        """The generation the anchor commits to, on disk and verified. Under the lock.

        An index whose manifest is the committed one is verified against the
        committed payload and reused. One whose manifest is not -- rewritten,
        corrupt, or another build's -- is refused for ordinary work and
        replaced only by an explicit rebuild. A missing index (evicted, or
        never placed) is rebuilt from the payload at the committed address;
        a missing payload from one model pass that must reproduce the
        committed digest. The manifest that results must be the committed one
        before anything is placed.
        """

        target_relative = _database_path(corpus_hash, spec)
        target = resolve_confined(self.workspace.root, target_relative)
        if target.is_file():
            _reject_sidecars(target, publication=True)
            existing: HybridKnowledgeIndexManifest | None
            try:
                existing = _read_manifest(target, HybridKnowledgeIndexManifest)
            except KnowledgeRetrievalError as exc:
                if exc.failure.code == "retrieval.index_publish_blocked" or not replace_divergent:
                    raise
                existing = None
            if existing is not None and existing.logical_hash == anchor.manifest_logical_hash:
                if existing.vector_commitment.payload_sha256 != anchor.payload_sha256:
                    raise _retrieval_error(
                        "hybrid manifest names a payload its sealed record does not",
                        code="retrieval.index_corrupt",
                    )
                repaired: _Composition | None = None
                try:
                    payload = _read_vector_payload(
                        self.workspace.root,
                        existing.vector_commitment,
                        manifest_logical_hash=existing.logical_hash,
                    )
                except KnowledgeRetrievalError as exc:
                    if not replace_divergent or exc.failure.code not in _REPAIRABLE:
                        raise
                    # The index is the committed one, but its vector objects
                    # are not held or do not verify (a lost block, a damaged
                    # sidecar): the explicit rebuild composes the payload
                    # again, requires the committed digest, proves the index
                    # against it and replaces the damaged objects.
                    repaired = computed or self._compose_vectors(
                        snapshot, chunks, rows, spec, block_for=block_for, cancelled=cancelled
                    )
                    payload = repaired.payload
                    if sha256_hex(payload) != anchor.payload_sha256:
                        raise _retrieval_error(
                            "rebuilt vectors do not reproduce the committed payload",
                            code="retrieval.vector_commitment_mismatch",
                        ) from exc
                blobs = _payload_blobs(
                    payload, count=existing.chunk_count, dimension=spec.embedding_dimension
                )
                _verify_projection(target, existing, expected_blobs=blobs)
                if repaired is not None:
                    if admit is not None:
                        admit(self._new_block_bytes(repaired, spec, existing.logical_hash))
                    self._publish_blocks(repaired, spec, existing.logical_hash, repair=True)
                return _primed(
                    existing,
                    chunks,
                    tuple(
                        _deserialize_vector(blob, dimension=spec.embedding_dimension)
                        for blob in blobs
                    ),
                    repaired,
                )
            if not replace_divergent:
                raise _retrieval_error(
                    "hybrid index differs from the generation its sealed record commits "
                    "to; rebuild it explicitly",
                    code="retrieval.snapshot_mismatch",
                )
        payload_path = resolve_confined(
            self.workspace.root, _payload_relative(anchor.payload_sha256)
        )
        composition: _Composition | None = None
        if payload_path.is_file():
            payload = payload_path.read_bytes()
            if sha256_hex(payload) != anchor.payload_sha256:
                raise _retrieval_error(
                    "vector payload address holds other bytes than its commitment",
                    code="retrieval.vector_commitment_mismatch",
                )
        elif (
            held := self._held_payload(
                anchor, count=len(rows), spec=spec, computed=computed, repair=replace_divergent
            )
        ) is not None:
            payload = held
        else:
            # The committed bytes are not held: compose them again -- blocks
            # from generations the caller sealed where it has them, the
            # model for the rest -- and require the composition to reproduce
            # the committed digest before anything is placed.
            composition = computed or self._compose_vectors(
                snapshot, chunks, rows, spec, block_for=block_for, cancelled=cancelled
            )
            payload = composition.payload
        if sha256_hex(payload) != anchor.payload_sha256:
            raise _retrieval_error(
                "rebuilt vectors do not reproduce the committed payload",
                code="retrieval.vector_commitment_mismatch",
            )
        commitment = _commitment(spec, count=len(rows), payload=payload)
        manifest = _manifest(
            corpus_hash=corpus_hash,
            document_count=document_count,
            spec=spec,
            rows=rows,
            commitment=commitment,
        )
        if manifest.logical_hash != anchor.manifest_logical_hash:
            raise _retrieval_error(
                "today's derivation does not reproduce the committed manifest",
                code="retrieval.snapshot_mismatch",
            )
        blobs = _payload_blobs(payload, count=len(rows), dimension=spec.embedding_dimension)
        placed = composition

        def before_place(staged: int) -> None:
            if placed is not None:
                self._place_blocks(
                    placed,
                    spec,
                    manifest.logical_hash,
                    admit=admit,
                    staged_bytes=staged,
                    repair=replace_divergent,
                )
            elif admit is not None:
                admit(staged)

        if admit is not None:
            admit(
                (
                    self._new_block_bytes(placed, spec, manifest.logical_hash)
                    if placed is not None
                    else 0
                )
                + _estimated_index_bytes(rows, len(payload))
            )
        published = self._publish_index(
            target_relative, rows, blobs, manifest, before_place=before_place
        )
        self._clear_marker(target_relative)
        return _primed(
            published,
            chunks,
            tuple(_deserialize_vector(blob, dimension=spec.embedding_dimension) for blob in blobs),
            composition,
        )

    def _publish_index(
        self,
        target_relative: str,
        rows: tuple[dict[str, Any], ...],
        blobs: tuple[bytes, ...],
        manifest: HybridKnowledgeIndexManifest,
        *,
        before_place: Callable[[int], None] | None = None,
    ) -> HybridKnowledgeIndexManifest:
        """Build the index in staging, verify it against the payload, place it,
        re-read and verify it. Called under the workspace lock.

        `before_place(staged_bytes)` runs after the staged index is verified
        and before anything is placed: it is where the caller admits the exact
        bytes and places the immutable payload, so a refusal leaves nothing
        but the staging file this method removes.
        """

        target = make_confined_parents(self.workspace.root, target_relative)
        target = resolve_confined(self.workspace.root, target_relative)
        staging_relative = f".system/staging/knowledge-hybrid-index.{uuid4()}.db"
        staging = make_confined_parents(self.workspace.root, staging_relative)
        staging = resolve_confined(self.workspace.root, staging_relative)
        try:
            self._build_database(staging, rows, blobs, manifest)
            _reject_sidecars(staging, publication=True)
            _verify_projection(staging, manifest, expected_blobs=blobs)
            if before_place is not None:
                before_place(staging.stat().st_size)
            try:
                self._replacer(staging, target)
            except WorkspaceConflictError as exc:
                if exc.failure.code != "catalog.replace_blocked":
                    raise
                raise _retrieval_error(
                    "hybrid index replacement is blocked by an open handle",
                    code="retrieval.index_publish_blocked",
                    retryable=True,
                    cause=exc,
                ) from exc
            published = _read_manifest(target, HybridKnowledgeIndexManifest)
            if published != manifest:
                raise _retrieval_error(
                    "published hybrid manifest differs from staging",
                    code="retrieval.index_corrupt",
                )
            _verify_projection(target, manifest, expected_blobs=blobs)
            return published
        finally:
            for candidate in (
                staging,
                staging.with_name(staging.name + "-wal"),
                staging.with_name(staging.name + "-shm"),
                staging.with_name(staging.name + "-journal"),
            ):
                with suppress(OSError):
                    candidate.unlink(missing_ok=True)

    def evict(self, database_relative: str, *, plan_hash: str, evicted_at: datetime) -> int:
        """Remove one published index under an approved plan, leaving a marker.

        The marker records the generation's identity, the manifest that was
        removed and (for a committed generation) the payload it committed
        to, so a later open refuses by name and an explicit rebuild -- or the
        next build of that corpus -- restores the same generation from the
        retained payload without the model. The payload is not touched.
        Returns the bytes released.
        """
        with self.workspace.lock():
            target = resolve_confined(self.workspace.root, database_relative)
            if not target.is_file():
                raise _retrieval_error(
                    "hybrid knowledge index to evict is not present",
                    code="retrieval.index_unavailable",
                )
            _reject_sidecars(target, publication=True)
            if database_relative.startswith(f".system/knowledge-indexes/{_INDEX_SCHEMA_VERSION}/"):
                committed = _read_manifest(target, HybridKnowledgeIndexManifest)
                payload: dict[str, Any] = {
                    "index_schema": _INDEX_SCHEMA_VERSION,
                    "index_id": committed.index_id,
                    "manifest_logical_hash": committed.logical_hash,
                    "vector_commitment": committed.vector_commitment,
                }
            elif database_relative.startswith(
                f".system/knowledge-indexes/{_LEGACY_SCHEMA_VERSION}/"
            ):
                legacy = _read_manifest(target, HybridV3KnowledgeIndexManifest)
                payload = {
                    "index_schema": _LEGACY_SCHEMA_VERSION,
                    "index_id": legacy.index_id,
                    "manifest_logical_hash": legacy.logical_hash,
                    "vector_commitment": None,
                }
            else:
                raise _retrieval_error(
                    "path is not a hybrid knowledge index",
                    code="retrieval.invalid_input",
                )
            marker = HybridIndexEvictionMarker.model_validate(
                {
                    "schema_version": "1",
                    **payload,
                    "database_path": database_relative,
                    "eviction_plan_hash": plan_hash,
                    "evicted_at": evicted_at,
                }
            )
            marker_relative = _eviction_marker_relative(database_relative)
            marker_path = make_confined_parents(self.workspace.root, marker_relative)
            marker_path = resolve_confined(self.workspace.root, marker_relative)
            released = target.stat().st_size
            marker_path.write_bytes(canonical_json_bytes(marker))
            target.unlink()
            return released

    def release_vector_object(self, relative: str) -> int:
        """Release a caller-admitted unreferenced vector object under the workspace lock.

        Remove one vector object (a block, a payload file or a sidecar) under the
        Workspace lock. The caller has established that no sealed generation
        composes it; the object is content-addressed, so a later build that
        needs the same bytes writes them again. Returns the bytes released.

        Args:
            relative: Confined vector-object path already admitted for release by its owner.

        Returns:
            Number of bytes released; zero when the object was absent.
        """
        if not relative.startswith(f"{_VECTOR_PAYLOAD_ROOT}/") or not relative.endswith(
            (".f32", _SIDECAR_SUFFIX)
        ):
            raise _retrieval_error("path is not a vector object", code="retrieval.invalid_input")
        with self.workspace.lock():
            target = resolve_confined(self.workspace.root, relative)
            if not target.is_file():
                raise _retrieval_error(
                    "vector object to release is not present",
                    code="retrieval.index_unavailable",
                )
            released = target.stat().st_size
            target.unlink()
            return released

    def rebuild_from_commitment(
        self,
        snapshot: WorkspaceKnowledgeSnapshot,
        index_spec: HybridIndexSpec,
        *,
        manifest_logical_hash: str,
        payload_sha256: str,
        cancelled: Callable[[], bool] | None = None,
        admit: Callable[[int], None] | None = None,
        block_for: VectorBlockSource | None = None,
    ) -> HybridIndexPriming:
        """Restore or repair one generation explicitly, from its sealed commitments.

        The caller names the generation it wants by the manifest hash and
        payload digest its own sealed record carries. The corpus is re-derived
        from the authoritative snapshot; an index on disk that is the committed
        one is verified and kept; one that is not is replaced; with the payload
        retained the index is rebuilt with no model run; with the payload lost
        the pinned model is run once, and its output must reproduce the
        committed digest exactly or the rebuild refuses rather than re-seal a
        different generation.
        """
        _require_capability(self._capability_probe(self.model_root, index_spec))
        authoritative = self.library.read_snapshot(snapshot.snapshot_id)
        if authoritative != snapshot:
            raise _retrieval_error(
                "snapshot differs from authoritative publication",
                code="retrieval.snapshot_mismatch",
            )
        corpus_hash = corpus_logical_hash(snapshot)
        chunks = _build_chunks(snapshot, self.library, index_spec.lexical_spec)
        rows = _hybrid_rows(chunks, index_spec)
        anchor = HybridGenerationAnchor(
            manifest_logical_hash=manifest_logical_hash, payload_sha256=payload_sha256
        )
        with self.workspace.lock():
            return self._materialize_committed(
                anchor,
                corpus_hash=corpus_hash,
                document_count=len(snapshot.revisions),
                spec=index_spec,
                rows=rows,
                chunks=chunks,
                snapshot=snapshot,
                cancelled=cancelled,
                admit=admit,
                block_for=block_for,
                replace_divergent=True,
            )

    @staticmethod
    def _build_database(
        path: Path,
        rows: tuple[dict[str, Any], ...],
        blobs: Sequence[bytes],
        manifest: HybridKnowledgeIndexManifest,
    ) -> None:
        try:
            with closing(sqlite3.connect(path)) as connection:
                loaded_version = _load_vector_extension(connection)
                if loaded_version != manifest.index_spec.vector_extension_version:
                    raise sqlite3.DatabaseError("sqlite-vec version differs from spec")
                _execute(connection, "PRAGMA journal_mode=DELETE")
                _execute(connection, "PRAGMA synchronous=FULL")
                with connection:
                    with closing(
                        connection.executescript(
                            f"""
                            CREATE TABLE metadata (
                                key TEXT PRIMARY KEY,
                                value TEXT NOT NULL
                            );
                            CREATE TABLE content (
                                rowid INTEGER PRIMARY KEY,
                                chunk_id TEXT NOT NULL UNIQUE,
                                document_id TEXT NOT NULL,
                                revision INTEGER NOT NULL,
                                ordinal INTEGER NOT NULL,
                                namespace TEXT NOT NULL,
                                access_class TEXT NOT NULL,
                                available_at TEXT NOT NULL,
                                expires_at TEXT,
                                title_raw TEXT NOT NULL,
                                heading_raw TEXT NOT NULL,
                                body_raw TEXT NOT NULL,
                                identifier_norm TEXT NOT NULL,
                                title_norm TEXT NOT NULL,
                                heading_norm TEXT NOT NULL,
                                citation_json TEXT NOT NULL,
                                chunk_hash TEXT NOT NULL
                            );
                            CREATE INDEX content_identifier ON content(identifier_norm);
                            CREATE INDEX content_title ON content(title_norm);
                            CREATE INDEX content_heading ON content(heading_norm);
                            CREATE INDEX content_filter
                                ON content(
                                    namespace,
                                    access_class,
                                    available_at,
                                    expires_at
                                );
                            CREATE VIRTUAL TABLE term_fts USING fts5(
                                identifier_terms,
                                title_terms,
                                heading_terms,
                                body_terms,
                                tokenize='unicode61 remove_diacritics 2'
                            );
                            CREATE VIRTUAL TABLE cjk_fts USING fts5(
                                cjk_terms,
                                tokenize='unicode61 remove_diacritics 2'
                            );
                            CREATE VIEW trigram_source AS
                                SELECT
                                    rowid,
                                    document_id AS identifier_raw,
                                    title_raw,
                                    heading_raw,
                                    body_raw
                                FROM content;
                            CREATE VIRTUAL TABLE trigram_fts USING fts5(
                                identifier_raw,
                                title_raw,
                                heading_raw,
                                body_raw,
                                tokenize='trigram',
                                content='trigram_source',
                                content_rowid='rowid'
                            );
                            -- The trigram index reads its text from the content
                            -- table through the view above instead of keeping a
                            -- copy: measured on the 4,912-chunk retained corpus
                            -- that copy was 2.54 MB of a 34.7 MB index (7.3 %),
                            -- the largest duplicate a layout change can remove
                            -- without touching a query channel, a score or a
                            -- verification. The term index keeps its own copy
                            -- because it stores derived term strings, and the
                            -- vectors stay in vec0 because the dense channel
                            -- queries them there.
                            CREATE TABLE dense_nodes (
                                rowid INTEGER PRIMARY KEY,
                                chunk_id TEXT NOT NULL UNIQUE,
                                embedding_input_hash TEXT NOT NULL,
                                vector_hash TEXT NOT NULL
                            );
                            CREATE VIRTUAL TABLE vec_chunks USING vec0(
                                embedding float[{manifest.index_spec.embedding_dimension}]
                            );
                            """
                        )
                    ):
                        pass
                    _execute(
                        connection,
                        "INSERT INTO metadata(key, value) VALUES ('manifest', ?)",
                        (canonical_json_bytes(manifest).decode("utf-8"),),
                    )
                    for rowid, (row, blob) in enumerate(
                        zip(rows, blobs, strict=True),
                        start=1,
                    ):
                        _insert_projection_row(connection, rowid, row, blob)
                with closing(connection.execute("PRAGMA integrity_check")) as cursor:
                    integrity = cursor.fetchone()
                if integrity is None or integrity[0] != "ok":
                    raise sqlite3.DatabaseError("hybrid staging integrity_check failed")
        except KnowledgeRetrievalError:
            raise
        except sqlite3.Error as exc:
            raise _retrieval_error(
                "hybrid staging database is invalid",
                code="retrieval.index_corrupt",
                cause=exc,
            ) from exc


def _insert_projection_row(
    connection: sqlite3.Connection,
    rowid: int,
    row: dict[str, Any],
    vector_blob: bytes,
) -> None:
    _execute(
        connection,
        """
        INSERT INTO content VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        (
            rowid,
            row["chunk_id"],
            row["document_id"],
            row["revision"],
            row["ordinal"],
            row["namespace"],
            row["access_class"],
            row["available_at"],
            row["expires_at"],
            row["title_raw"],
            row["heading_raw"],
            row["body_raw"],
            row["identifier_norm"],
            row["title_norm"],
            row["heading_norm"],
            row["citation_json"],
            row["chunk_hash"],
        ),
    )
    _execute(
        connection,
        "INSERT INTO term_fts(rowid, identifier_terms, title_terms, "
        "heading_terms, body_terms) VALUES (?, ?, ?, ?, ?)",
        (
            rowid,
            row["identifier_terms"],
            row["title_terms"],
            row["heading_terms"],
            row["body_terms"],
        ),
    )
    _execute(
        connection,
        "INSERT INTO cjk_fts(rowid, cjk_terms) VALUES (?, ?)",
        (rowid, row["cjk_terms"]),
    )
    _execute(
        connection,
        "INSERT INTO trigram_fts(rowid, identifier_raw, title_raw, "
        "heading_raw, body_raw) VALUES (?, ?, ?, ?, ?)",
        (
            rowid,
            row["document_id"],
            row["title_raw"],
            row["heading_raw"],
            row["body_raw"],
        ),
    )
    _execute(
        connection,
        "INSERT INTO dense_nodes(rowid, chunk_id, embedding_input_hash, vector_hash) "
        "VALUES (?, ?, ?, ?)",
        (
            rowid,
            row["chunk_id"],
            row["embedding_input_hash"],
            sha256_hex(vector_blob),
        ),
    )
    _execute(
        connection,
        "INSERT INTO vec_chunks(rowid, embedding) VALUES (?, ?)",
        (rowid, vector_blob),
    )


@dataclass(slots=True)
class _Candidate:
    row: sqlite3.Row
    channels: set[RetrievalChannel]
    channel_ranks: dict[RetrievalChannel, int]
    tier: int


class WorkspaceHybridKnowledgeRetriever:
    """Own one verified read connection and embedding session."""

    def __init__(
        self,
        workspace: Workspace,
        *,
        model_root: Path,
        index_spec: HybridIndexSpec,
        capability_probe: Callable[
            [Path, HybridIndexSpec], HybridCapabilityReport
        ] = _default_capability_probe,
        adapter_factory: Callable[[Path, HybridIndexSpec], Any] = _default_adapter_factory,
        reranker_factory: Callable[[Path, HybridIndexSpec], Any] = _default_reranker_factory,
        priming: HybridIndexPriming | None = None,
    ) -> None:
        """Configure a lazy hybrid reader for one exact numerical recipe.

        Args:
            workspace: Workspace owning authoritative knowledge and index artifacts.
            model_root: Local root containing the installed encoder and reranker packs.
            capability_probe: Capability checker invoked before semantic work.
            adapter_factory: Factory for the recipe-bound passage/query encoder.
            reranker_factory: Factory for the recipe-bound cross encoder.
            index_spec: Installed recipe identity required by subsequent requests.
            priming: Optional verified build state consumed by the first query only.
        """
        self.workspace = workspace
        self.model_root = model_root
        self.index_spec = index_spec
        self.library = WorkspaceKnowledgeLibrary(workspace)
        self._capability_probe = capability_probe
        self._adapter_factory = adapter_factory
        self._reranker_factory = reranker_factory
        self._adapter: Any | None = None
        self._reranker: Any | None = None
        self._scores: Any | None = None
        """The workspace's sealed pair scores under this reranker context
        (`PairScoreStore`), opened at the first rerank under
        `pair_score_admission`; None without an admission (in-process
        cache only)."""
        self.pair_score_admission: PairScoreAdmission | None = None
        """What this reader may reuse and seal, set by the owner that reads
        the workspace's sealed commitments before the first rerank."""
        self.sealed_pair_score_blocks: tuple[SealedPairScoreBlock, ...] = ()
        """The blocks this reader sealed (`seal_pair_scores`), for the owner
        to commit."""
        self._scores_sampled = False
        """Whether this reader has re-derived a sample of the scores the store
        served it: once a session, the first time the store serves any."""
        self._connection: sqlite3.Connection | None = None
        self._manifest: AnyHybridManifest | None = None
        self._schema: str | None = None
        self._chunks: dict[str, _Chunk] = {}
        self._vectors: dict[str, tuple[float, ...]] = {}
        """The vectors this reader proved when it opened the generation.

        Opening verifies the whole published projection against the model, row
        by row. Every query then used to re-embed its `top_k` bodies to verify
        the hits it was about to return, which re-proves a fact the open
        already established and pays the model for it on every query. Holding
        the verified vectors is the same commitment without the second
        inference: a hit is still checked against the stored row, and a
        database edited after open still fails that check.
        """

        self._priming = priming
        """Consumed by the first query and never reused, so a second generation
        cannot be opened from the vectors of the first."""
        self.primed_by_embedding: bool | None = None if priming is None else priming.embedded
        """For a reader handed back by a build: whether that build ran the passage
        model or reused the corpus's committed generation. Nothing for a cold reader."""
        self.priming_facts = HybridPrimingFacts.of(priming)
        """What the build that primed this reader cost, kept after the priming is
        consumed: chunks embedded and reused, model calls."""

    @property
    def manifest(self) -> AnyHybridManifest:
        """The generation this reader is bound to, before or after its first query."""
        if self._manifest is not None:
            return self._manifest
        if self._priming is not None:
            return self._priming.manifest
        raise _retrieval_error(
            "retriever is not bound to a hybrid generation",
            code="retrieval.index_unavailable",
        )

    def __enter__(self) -> WorkspaceHybridKnowledgeRetriever:
        """Return this reader for context-managed resource cleanup.

        Returns:
            This retriever; model and database resources remain lazy until use.
        """
        return self

    def __exit__(self, *_args: object) -> None:
        """Close this reader on context exit.

        Args:
            _args: Context-manager exception details; exceptions remain unsuppressed.
        """
        self.close()

    def close(self) -> None:
        """Close local resources, seal admitted pair scores and discard generation state.

        Database, encoder and reranker handles are released. Verified chunks/vectors and unconsumed
        priming are cleared; sealed score blocks remain available for the owner to commit.
        """
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        if self._adapter is not None:
            self._adapter.close()
            self._adapter = None
        if self._reranker is not None:
            close = getattr(self._reranker, "close", None)
            if callable(close):
                close()
            self._reranker = None
        if self._scores is not None:
            self.seal_pair_scores()
            self._scores = None
        self._manifest = None
        self._schema = None
        self._chunks = {}
        self._vectors = {}
        self._priming = None

    def capability(self) -> HybridCapabilityReport:
        """Report readiness for this exact configured retriever."""
        try:
            return self._capability_probe(self.model_root, self.index_spec)
        except KnowledgeRetrievalError as exc:
            if exc.failure.code != "retrieval.semantic_pack_unavailable":
                raise
            return _semantic_unavailable_capability(self.index_spec, cause=str(exc))

    def retrieve(
        self,
        request: HybridKnowledgeRetrievalRequest,
    ) -> HybridKnowledgeRetrievalResult:
        """Retrieve cited hybrid hits under an exact request and uncalibrated score policy.

        The reader validates compiled query limits, capability and index binding before using
        verified source/index commitments. Cross-encoder logits are ranking evidence, never
        confidence percentages. Failures during generation use close its local handles.

        Args:
            request: Sealed query, source snapshot, filters, recipe and index-generation
                commitments.

        Returns:
            Canonical result and trace with contiguous ranked citations, or an EMPTY result.

        Raises:
            KnowledgeRetrievalError: Input, capability, snapshot, source or index verification
                fails.
        """
        normalized_query = _normalize_text(request.query)
        terms = _content_terms(_compiled_terms(normalized_query))
        cjk_terms = _cjk_terms(normalized_query)
        compiled = tuple(sorted(set((*terms, *cjk_terms))))
        if (
            not normalized_query
            or not compiled
            or len(compiled) > self.index_spec.lexical_spec.maximum_query_terms
        ):
            raise _retrieval_error(
                "retrieval query cannot be compiled within hybrid-v2 limits",
                code="retrieval.invalid_input",
            )

        if self._manifest is None:
            _require_capability(self._capability_probe(self.model_root, self.index_spec))
        if (
            request.index_spec_logical_hash != self.index_spec.logical_hash
            or request.index_id != _request_index_id(request)
        ):
            raise _retrieval_error(
                "hybrid request does not bind the configured index spec",
                code="retrieval.snapshot_mismatch",
            )
        try:
            self._ensure_open(request)
            self._verify_authoritative_source(request)
            assert self._adapter is not None
            query_vector = self._adapter.embed_query(request.query)
            candidates, queried = self._query(
                request=request,
                normalized_query=normalized_query,
                terms=terms,
                cjk_terms=cjk_terms,
                query_vector=query_vector,
            )
            hits, reranked = self._hits(candidates, request=request)
            semantics = RetrievalScoreSemantics.cross_encoder_logit()
            trace_payload = {
                "schema_version": "1",
                "normalized_query": normalized_query,
                "compiled_terms": compiled,
                "channels_queried": tuple(sorted(queried, key=lambda item: item.value)),
                "candidate_count": len(candidates),
                "score_semantics": semantics,
                "reranked_count": reranked,
            }
            trace = HybridKnowledgeRetrievalTrace.model_validate(
                {**trace_payload, "logical_hash": _canonical_hash(trace_payload)}
            )
            status = RetrievalStatus.RESOLVED if hits else RetrievalStatus.EMPTY
            result_payload = {
                "schema_version": "1",
                "request": request,
                "status": status,
                "hits": hits,
                "trace": trace,
            }
            return HybridKnowledgeRetrievalResult.model_validate(
                {**result_payload, "logical_hash": _canonical_hash(result_payload)}
            )
        except Exception:
            self.close()
            raise

    def inventory(
        self, request: HybridKnowledgeRetrievalRequest
    ) -> tuple[KnowledgeInventoryWindow, ...]:
        """Return an authoritative source-order inventory within the request groups.

        Every proved chunk of the request's document groups, in source
        order, without a query.

        The same binding and source proofs a search performs -- the index spec
        and generation the request names, the snapshot re-proved against the
        library -- and then the authoritative chunks the open proved, filtered
        to the request's groups. A structural scan walks these once; it does
        not search, embed or score, and it issues nothing a verified read
        cannot later prove against the source bytes at the citation's range.

        Args:
            request: Exact admitted source/index binding and optional document groups.

        Returns:
            Verified windows in source order, without query scoring.
        """
        if self._manifest is None:
            _require_capability(self._capability_probe(self.model_root, self.index_spec))
        if (
            request.index_spec_logical_hash != self.index_spec.logical_hash
            or request.index_id != _request_index_id(request)
        ):
            raise _retrieval_error(
                "hybrid request does not bind the configured index spec",
                code="retrieval.snapshot_mismatch",
            )
        try:
            self._ensure_open(request)
            self._verify_authoritative_source(request)
        except Exception:
            self.close()
            raise
        allowed = {str(document_id) for group in request.document_groups for document_id in group}
        windows = [
            KnowledgeInventoryWindow(
                citation=chunk.citation,
                ordinal=int(chunk.ordinal),
                body=str(chunk.body),
                chunk_hash=str(chunk.chunk_hash),
            )
            for chunk in self._chunks.values()
            if not allowed or str(chunk.document_id) in allowed
        ]
        windows.sort(key=lambda value: (str(value.citation.document_id), value.ordinal))
        return tuple(windows)

    def score_in_context(self, context: str, chunk_ids: Sequence[str]) -> tuple[float, ...]:
        """Score proved chunks under one declared cross-encoder context.

        The configured reranker's score of each proved chunk against one
        declared relevance context, in the order given: one pair per chunk,
        answered from the bounded exact-pair cache where the pair is known.
        Raw logits of different contexts are not comparable; the caller
        ranks within the context it declared.

        Args:
            context: Declared relevance text supplied to the configured cross encoder.
            chunk_ids: Proved chunk identifiers in requested score order.

        Returns:
            Cross-encoder scores in the same chunk order, comparable only within this context.

        Raises:
            KnowledgeRetrievalError: The reader is not open or a requested chunk is not
                authoritative.
        """
        if self._manifest is None:
            raise _retrieval_error(
                "hybrid retriever is not open for scoring", code="retrieval.index_unavailable"
            )
        bodies = []
        for chunk_id in chunk_ids:
            chunk = self._chunks.get(str(chunk_id))
            if chunk is None:
                raise _retrieval_error(
                    "scoring request names an unknown authoritative chunk",
                    code="retrieval.index_corrupt",
                )
            bodies.append(str(chunk.body))
        if not bodies:
            return ()
        return self._score_pairs(context, bodies)

    def _score_pairs(self, query: str, passages: Sequence[str]) -> tuple[float, ...]:
        """One score per passage: the workspace's sealed pair scores first, the
        cross-encoder for the rest, what it scored recorded for sealing at
        close. The first time the store serves this reader, a sample of the
        served scores is re-derived by the model before any is used."""

        if self._reranker is None:
            self._reranker = self._reranker_factory(self.model_root, self.index_spec)
        reranker = self._reranker
        if self.pair_score_admission is None:
            return tuple(float(value) for value in reranker.score(query, passages))
        if self._scores is None:
            self._scores = _reranking_module().pair_score_store(
                self.workspace.root, self.index_spec, self.pair_score_admission
            )
        scores, sampled = self._scores.serve(
            query,
            passages,
            lambda batch: reranker.score(query, batch),
            verify_sample=not self._scores_sampled,
        )
        if sampled:
            self._scores_sampled = True
        return tuple(float(value) for value in scores)

    def seal_pair_scores(self) -> tuple[SealedPairScoreBlock, ...]:
        """Seal admitted pair-score blocks for their owner to commit.

        Seal what the model scored in this reader under the admission's
        storage admission; the blocks, for the owner to commit. Idempotent:
        a second call seals nothing more and returns what was sealed.

        Returns:
            Sealed blocks for the owner to commit; repeated sealing is idempotent.
        """
        if self._scores is not None:
            sealed = self._scores.flush()
            if sealed:
                self.sealed_pair_score_blocks = (*self.sealed_pair_score_blocks, *sealed)
        return self.sealed_pair_score_blocks

    def pair_score_facts(self) -> dict[str, int]:
        """Report observed pair-score reuse, sealing and refusal counts.

        The store's counts for the owner's accounting: hits, misses, blocks
        loaded, unanchored files, missing admitted blocks, pairs sealed and
        refused.

        Returns:
            Counts of pair-score hits, misses, blocks, missing/unanchored entries and refusals.
        """
        store = self._scores
        if store is None:
            return {}
        return {
            "hits": int(store.hits),
            "misses": int(store.misses),
            "blocks": int(store.block_count),
            "unanchored": int(store.unanchored),
            "missing": int(store.missing),
            "sealed_pairs": int(store.sealed_pairs),
            "unsealed_pairs": int(store.unsealed),
            "refused_full": int(store.refused_full),
        }

    def _ensure_open(self, request: HybridKnowledgeRetrievalRequest) -> None:
        if self._manifest is not None:
            if self._manifest.index_id != request.index_id:
                raise _retrieval_error(
                    "retriever already owns another hybrid generation",
                    code="retrieval.snapshot_mismatch",
                )
            return
        priming = self._priming
        # Single use, consumed before it is trusted: a refused priming must not
        # be retried against a different request.
        self._priming = None
        if priming is not None:
            self._open_primed(priming, request)
            return
        if request.index_schema == _LEGACY_SCHEMA_VERSION:
            raise _retrieval_error(
                "a knowledge-hybrid-v3 generation is history and is not opened",
                code="retrieval.index_generation_retired",
            )
        self._open_committed(request)

    def _open_committed(self, request: HybridKnowledgeRetrievalRequest) -> None:
        """Open a `knowledge-hybrid-v4` generation cold, embedding nothing.

        The request's snapshot is read from the authoritative library and
        resolved to its corpus; the stored manifest must equal the caller's
        sealed expectation of it (the database is never the authority for
        itself) and must be reproduced by today's derivation of that corpus;
        the committed payload is read and checked against its digest; every
        stored vector is compared byte for byte with the payload; the rows are
        proved against the manifest. Then the reader is open.
        """

        snapshot = self.library.read_snapshot(request.snapshot_id)
        if snapshot.logical_hash != request.snapshot_logical_hash:
            raise _retrieval_error(
                "hybrid request snapshot commitment is stale",
                code="retrieval.snapshot_mismatch",
            )
        corpus_hash = corpus_logical_hash(snapshot)
        if corpus_hash != request.corpus_logical_hash:
            raise _retrieval_error(
                "hybrid request corpus differs from its snapshot",
                code="retrieval.snapshot_mismatch",
            )
        database_relative = _database_path(corpus_hash, self.index_spec)
        path = resolve_confined(self.workspace.root, database_relative)
        if not path.is_file():
            marker = _read_eviction_marker(self.workspace.root, database_relative)
            if marker is not None:
                raise _refuse_evicted(marker)
            raise _retrieval_error(
                "hybrid knowledge index is unavailable; rebuild it explicitly",
                code="retrieval.index_unavailable",
            )
        manifest = _read_manifest(path, HybridKnowledgeIndexManifest)
        if (
            manifest.logical_hash != request.index_manifest_logical_hash
            or manifest.index_id != request.index_id
            or manifest.corpus_logical_hash != corpus_hash
            or manifest.index_spec.logical_hash != request.index_spec_logical_hash
        ):
            raise _retrieval_error(
                "hybrid generation commitments differ from the request",
                code="retrieval.snapshot_mismatch",
            )
        chunks = _build_chunks(snapshot, self.library, self.index_spec.lexical_spec)
        rows = _hybrid_rows(chunks, self.index_spec)
        expected = _source_commitments(
            corpus_hash=corpus_hash,
            document_count=len(snapshot.revisions),
            spec=self.index_spec,
            rows=rows,
        )
        if manifest.source_commitments() != expected:
            raise _retrieval_error(
                "hybrid manifest differs from authoritative reconstruction",
                code="retrieval.index_corrupt",
            )
        payload = _read_vector_payload(
            self.workspace.root,
            manifest.vector_commitment,
            manifest_logical_hash=manifest.logical_hash,
        )
        blobs = _payload_blobs(
            payload, count=manifest.chunk_count, dimension=self.index_spec.embedding_dimension
        )
        connection = _verify_projection(path, manifest, expected_blobs=blobs, keep_open=True)
        assert connection is not None
        try:
            adapter = self._adapter_factory(self.model_root, self.index_spec)
        except Exception:
            connection.close()
            raise
        self._adapter = adapter
        self._connection = connection
        self._manifest = manifest
        self._schema = _INDEX_SCHEMA_VERSION
        self._chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self._vectors = {
            chunk.chunk_id: _deserialize_vector(blob, dimension=self.index_spec.embedding_dimension)
            for chunk, blob in zip(chunks, blobs, strict=True)
        }

    def _verify_authoritative_source(self, request: HybridKnowledgeRetrievalRequest) -> None:
        """Re-prove the source this generation was built from, on every query.

        Opening proved that the chunks derive from this snapshot. What a query
        must still establish is that the snapshot has not moved underneath the
        reader since, and `read_snapshot` does exactly that: it re-reads every
        revision through `read_revision`, which verifies each blob's SHA-256,
        its canonical manifest encoding and its media type, and refuses if a
        child differs from its committed revision.

        So this re-proves the bytes without re-deriving the corpus, which is
        the same commitment for about a fortieth of the work, and it does not
        repeat the loop `read_snapshot` already owns.
        """

        assert self._manifest is not None
        snapshot = self.library.read_snapshot(request.snapshot_id)
        # The request's snapshot must still be the one the reader opened
        # against, and -- for a committed generation shared by every snapshot
        # of its corpus -- must still resolve to that corpus.
        if snapshot.logical_hash != request.snapshot_logical_hash or (
            isinstance(self._manifest, HybridKnowledgeIndexManifest)
            and corpus_logical_hash(snapshot) != self._manifest.corpus_logical_hash
        ):
            raise _retrieval_error(
                "authoritative snapshot changed after generation activation",
                code="retrieval.source_integrity",
            )

    def _open_primed(
        self,
        priming: HybridIndexPriming,
        request: HybridKnowledgeRetrievalRequest,
    ) -> None:
        """Open against vectors this process already proved, without re-embedding.

        Every commitment the cold path checks is checked here except what
        this process's own build already proved and the priming carries: the
        corpus embedding always, and -- for a priming whose derivation proof
        binds these chunks -- the rows' derivation from them. The published
        projection is proved row by row here, on this reader's own
        connection, whatever the build proved: a build's proof stands for
        the bytes it read, and rows removed from the file between that
        proof and this open would never surface as hits for the per-hit
        check to refuse (record section Y).
        """

        manifest = priming.manifest
        spec = manifest.index_spec
        if (
            request.index_schema != _INDEX_SCHEMA_VERSION
            or manifest.index_id != request.index_id
            or manifest.logical_hash != request.index_manifest_logical_hash
            or manifest.corpus_logical_hash != request.corpus_logical_hash
            or spec.logical_hash != request.index_spec_logical_hash
            or spec.logical_hash != self.index_spec.logical_hash
        ):
            raise _retrieval_error(
                "primed hybrid generation differs from the request",
                code="retrieval.snapshot_mismatch",
            )
        if (
            len(priming.vectors) != len(priming.chunks)
            or len(priming.chunks) != manifest.chunk_count
            or any(len(vector) != spec.embedding_dimension for vector in priming.vectors)
        ):
            raise _retrieval_error(
                "primed chunks and vectors do not match the published manifest",
                code="retrieval.index_corrupt",
            )
        path = resolve_confined(self.workspace.root, manifest.database_path)
        if not path.is_file():
            raise _retrieval_error(
                "hybrid knowledge index is unavailable; rebuild it explicitly",
                code="retrieval.index_unavailable",
            )
        # Priming is a claim that these chunks came from the authoritative
        # source, and nothing above tests it: the manifest, the projection and
        # the vectors can all be made coherent with fabricated chunks. This
        # process's own build proved the derivation when it built or verified
        # the manifest from the library, and says so with a proof keyed by a
        # process-private nonce over exactly this manifest, snapshot and
        # chunk content. Any other priming is re-derived once, here, and the
        # manifest it produces must be the one being opened; a body altered,
        # swapped or reordered beside a genuine proof does not match the
        # proof and is re-derived and refused the same way. Every later
        # query then rides on this proof instead of re-deriving the corpus.
        snapshot = self.library.read_snapshot(request.snapshot_id)
        if snapshot.logical_hash != request.snapshot_logical_hash:
            raise _retrieval_error(
                "hybrid request snapshot commitment is stale",
                code="retrieval.snapshot_mismatch",
            )
        published = _read_manifest(path, HybridKnowledgeIndexManifest)
        if published != manifest:
            raise _retrieval_error(
                "published hybrid manifest differs from the primed generation",
                code="retrieval.index_corrupt",
            )
        own_build = priming.derivation_proof is not None and (
            priming.derivation_proof
            == _derivation_proof(manifest, snapshot.logical_hash, priming.chunks)
        )
        if own_build:
            chunks = priming.chunks
        else:
            chunks = _build_chunks(snapshot, self.library, self.index_spec.lexical_spec)
            if tuple(chunk.chunk_id for chunk in chunks) != tuple(
                chunk.chunk_id for chunk in priming.chunks
            ):
                raise _retrieval_error(
                    "primed chunk order differs from the authoritative derivation",
                    code="retrieval.source_integrity",
                )
            if tuple(chunks) != tuple(priming.chunks):
                # The same ids over other bodies: not a derivation of the
                # source, refused rather than served from the authoritative
                # derivation as if it were.
                raise _retrieval_error(
                    "primed chunks differ from the authoritative derivation",
                    code="retrieval.source_integrity",
                )
            expected = _source_commitments(
                corpus_hash=corpus_logical_hash(snapshot),
                document_count=len(snapshot.revisions),
                spec=self.index_spec,
                rows=_hybrid_rows(chunks, self.index_spec),
            )
            if expected != published.source_commitments():
                raise _retrieval_error(
                    "primed chunks do not derive from the authoritative source",
                    code="retrieval.source_integrity",
                )
        # The primed vectors must be the committed payload, not merely
        # coherent with the rows: a build's own bytes are what the
        # commitment names.
        blobs = tuple(_serialize_vector(vector) for vector in priming.vectors)
        commitment = published.vector_commitment
        if sha256_hex(b"".join(blobs)) != commitment.payload_sha256 or (
            _read_vector_payload(
                self.workspace.root, commitment, manifest_logical_hash=published.logical_hash
            )
            != b"".join(blobs)
        ):
            raise _retrieval_error(
                "primed vectors differ from the committed payload",
                code="retrieval.vector_commitment_mismatch",
            )
        connection = _verify_projection(path, published, expected_blobs=blobs, keep_open=True)
        assert connection is not None
        try:
            adapter = self._adapter_factory(self.model_root, self.index_spec)
        except Exception:
            connection.close()
            raise
        self._adapter = adapter
        self._connection = connection
        self._manifest = published
        self._schema = _INDEX_SCHEMA_VERSION
        # The chunks the proof binds, or the authoritative derivation above.
        self._chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self._vectors = {
            chunk.chunk_id: tuple(vector)
            for chunk, vector in zip(chunks, priming.vectors, strict=True)
        }

    @staticmethod
    def _filter_clause(
        request: HybridKnowledgeRetrievalRequest,
        *,
        alias: str = "c",
    ) -> tuple[str, tuple[Any, ...]]:
        namespace_marks = ", ".join("?" for _ in request.allowed_namespaces)
        access_marks = ", ".join("?" for _ in request.allowed_access_classes)
        as_of = _utc_text(request.as_of)
        clause = (
            f"{alias}.namespace IN ({namespace_marks}) "
            f"AND {alias}.access_class IN ({access_marks}) "
            f"AND {alias}.available_at <= ? "
            f"AND ({alias}.expires_at IS NULL OR ? < {alias}.expires_at)"
        )
        parameters: tuple[Any, ...] = (
            *(item.value for item in request.allowed_namespaces),
            *(item.value for item in request.allowed_access_classes),
            as_of,
            as_of,
        )
        if request.scope:
            # The scope is part of the filter every channel applies, so a
            # chunk outside it is never a candidate, never fused, never a
            # pair the cross-encoder reads -- not a cut of the final order.
            ranges = " OR ".join(
                f"({alias}.document_id = ? AND {alias}.ordinal BETWEEN ? AND ?)"
                for _entry in request.scope
            )
            clause += f" AND ({ranges})"
            parameters = (
                *parameters,
                *(
                    value
                    for document_id, first, last in request.scope
                    for value in (str(document_id), first, last)
                ),
            )
        return clause, parameters

    def _query(
        self,
        *,
        request: HybridKnowledgeRetrievalRequest,
        normalized_query: str,
        terms: tuple[str, ...],
        cjk_terms: tuple[str, ...],
        query_vector: Sequence[float],
    ) -> tuple[dict[str, _Candidate], set[RetrievalChannel]]:
        assert self._connection is not None
        assert self._manifest is not None
        connection = self._connection
        candidates: dict[str, _Candidate] = {}
        queried: set[RetrievalChannel] = set()
        clause, filters = self._filter_clause(request)
        # Depth is stated per document and spent across the documents that
        # matched, so admitting an unrelated filing never shrinks anyone
        # else's share. Where fewer documents match, the same window hands
        # each of them correspondingly deeper passages.
        limit = self.index_spec.candidate_depth_per_document * self._manifest.document_count

        def collect(
            channel: RetrievalChannel,
            body: str,
            parameters: tuple[Any, ...],
            *,
            tier: int,
            score: str | None = None,
        ) -> None:
            """Take this channel's best passages, allocated per document.

            A channel used to take a single global best-`n` rows. On a
            multi-issuer corpus that is winner-takes-all: for the legal topic,
            859 chunks match and the two most litigious filings take 56 and 26
            of the 100 places, while three matching documents first appear at
            global ranks 106, 113 and 572 and so never reach fusion at all. An
            issuer whose filing is not the wordiest on a topic cannot be
            reviewed, however clearly it states the fact.

            The window walks the documents that matched in parallel, taking
            each one's best passage, then each one's second, and so on. Depth
            is therefore stated per document rather than for the corpus: a
            document's share does not shrink when an unrelated filing is
            admitted, and capacity the non-matching documents do not use is
            spent on the ones that did match. Nothing about the per-channel
            ordering changes -- only how far down each document it reaches.

            Measured on the admitted corpus, a global depth of 100 over 21
            documents left roughly four passages each, and the passage stating
            the event was the eighth or tenth best of its own filing for two of
            the six probes -- present in the index, absent from every channel.
            """

            queried.add(channel)
            ordering = (
                "channel_score, document_id, revision, ordinal"
                if score
                else "document_id, revision, ordinal"
            )
            # The same total depth as before, spent differently. A channel used
            # to take its global best rows, which on a multi-issuer corpus is
            # winner-takes-all: for the legal topic 859 chunks match and the two
            # most litigious filings take 56 and 26 of the 100 places, while
            # three matching documents first appear at global ranks 106, 113 and
            # 572 and never reach fusion.
            #
            # Taking each document's best in turn fixes that without capping
            # anyone: rows are ordered by their rank *within* their document
            # first, so every matching document is served before any document is
            # served twice, and a document that is alone in matching keeps the
            # whole budget. Unused capacity is refilled by construction, because
            # the only limit is the total.
            #
            # Four layers, because `bm25()` is only valid where FTS5 puts it:
            # compute the score, rank within each document, take the round in
            # document order, then restore the channel's own order so the ranks
            # this channel reports still mean what they meant.
            statement = (
                f"SELECT * FROM (SELECT * FROM (SELECT *, ROW_NUMBER() OVER ("
                f"PARTITION BY document_id ORDER BY {ordering}) AS document_rank"
                f" FROM (SELECT {columns}"
                + (f", {score} AS channel_score" if score else "")
                + f" {body})) ORDER BY document_rank, {ordering} LIMIT ?)"
                f" ORDER BY {ordering}"
            )
            try:
                with closing(connection.execute(statement, parameters)) as cursor:
                    rows = cursor.fetchall()
            except sqlite3.Error as exc:
                raise _retrieval_error(
                    "hybrid knowledge query failed",
                    code="retrieval.index_corrupt",
                    cause=exc,
                ) from exc
            for rank, row in enumerate(rows, start=1):
                chunk_id = str(row["chunk_id"])
                candidate = candidates.get(chunk_id)
                if candidate is None:
                    candidate = _Candidate(
                        row=row,
                        channels=set(),
                        channel_ranks={},
                        tier=tier,
                    )
                    candidates[chunk_id] = candidate
                candidate.channels.add(channel)
                candidate.channel_ranks[channel] = rank
                candidate.tier = min(candidate.tier, tier)

        columns = (
            "c.rowid AS content_rowid, c.chunk_id, c.document_id, c.revision, "
            "c.ordinal, c.title_raw, c.heading_raw, c.body_raw, c.citation_json, "
            "c.chunk_hash"
        )
        collect(
            RetrievalChannel.IDENTIFIER_EXACT,
            f"FROM content AS c WHERE c.identifier_norm = ? AND {clause}",
            (normalized_query, *filters, limit),
            tier=0,
        )
        collect(
            RetrievalChannel.TERM_EXACT,
            f"FROM content AS c WHERE (c.title_norm = ? OR c.heading_norm = ?) AND {clause}",
            (normalized_query, normalized_query, *filters, limit),
            tier=1,
        )
        # `CROSS JOIN`, never `JOIN`: SQLite's planner otherwise takes the
        # content filter index as the outer loop and runs the full-text MATCH
        # once per content row -- measured on the admitted corpus, 1.3 s for
        # a query matching 4,554 chunks and 0.8 s for one matching none,
        # against 0.04 s and 0.000 s with the full-text scan outermost. The
        # rows, their scores and their order are identical; only the loop
        # order is fixed, which is what CROSS JOIN means to SQLite.
        if terms:
            all_query = " AND ".join(_literal_fts_term(term) for term in terms)
            any_query = " OR ".join(_literal_fts_term(term) for term in terms)
            lexical = self.index_spec.lexical_spec
            bm25 = (
                f"bm25(term_fts, {lexical.identifier_bm25_weight}, "
                f"{lexical.title_bm25_weight}, {lexical.heading_bm25_weight}, "
                f"{lexical.body_bm25_weight})"
            )
            for channel, match in (
                (RetrievalChannel.TERM_ALL, all_query),
                (RetrievalChannel.TERM_ANY, any_query),
            ):
                collect(
                    channel,
                    "FROM term_fts CROSS JOIN content AS c ON c.rowid = term_fts.rowid "
                    f"WHERE term_fts MATCH ? AND {clause}",
                    (match, *filters, limit),
                    tier=2,
                    score=bm25,
                )
        if cjk_terms:
            cjk_query = " OR ".join(_literal_fts_term(term) for term in cjk_terms)
            collect(
                RetrievalChannel.CJK_BIGRAM,
                "FROM cjk_fts CROSS JOIN content AS c ON c.rowid = cjk_fts.rowid "
                f"WHERE cjk_fts MATCH ? AND {clause}",
                (cjk_query, *filters, limit),
                tier=2,
                score="bm25(cjk_fts)",
            )
        if _trigram_applies(normalized_query, self.index_spec):
            collect(
                RetrievalChannel.TRIGRAM,
                "FROM trigram_fts CROSS JOIN content AS c ON c.rowid = trigram_fts.rowid "
                f"WHERE trigram_fts MATCH ? AND {clause}",
                (_literal_fts_term(normalized_query), *filters, limit),
                tier=2,
                score="bm25(trigram_fts)",
            )
        vector_blob = _serialize_vector(query_vector)
        decimals = self.index_spec.dense_distance_decimal_places
        # Not `CROSS JOIN` here: the planner's content-filter outer loop asks
        # vec0 for one row at a time, and measured on the development copy's
        # KO unit (4,912 chunks, the whole corpus in scope) that beats one
        # sequential vec0 scan joined by rowid -- 130 ms against 178 ms a
        # search, 17 against 20 MB of page reads under the reader's cache;
        # the rows, distances and order are the same either way.
        collect(
            RetrievalChannel.DENSE,
            "FROM vec_chunks AS v "
            "JOIN dense_nodes AS n ON n.rowid = v.rowid "
            "JOIN content AS c ON c.rowid = n.rowid "
            f"WHERE {clause}",
            (vector_blob, *filters, limit),
            tier=2,
            score=f"ROUND(vec_distance_cosine(v.embedding, ?), {decimals})",
        )
        return candidates, queried

    def _hits(
        self,
        candidates: dict[str, _Candidate],
        *,
        request: HybridKnowledgeRetrievalRequest,
    ) -> tuple[tuple[KnowledgeRetrievalHit, ...], int]:
        """The reranked hits cut per group, and how many pairs the
        cross-encoder scored to order them."""

        weights = {
            RetrievalChannel.TERM_ALL: self.index_spec.term_all_weight,
            RetrievalChannel.TERM_ANY: self.index_spec.term_any_weight,
            RetrievalChannel.CJK_BIGRAM: self.index_spec.cjk_bigram_weight,
            RetrievalChannel.TRIGRAM: self.index_spec.trigram_weight,
            RetrievalChannel.DENSE: self.index_spec.dense_weight,
        }
        fused: list[tuple[int, float, str, _Candidate]] = []
        for chunk_id, candidate in candidates.items():
            score = sum(
                weights[channel] / (self.index_spec.rrf_constant + rank)
                for channel, rank in candidate.channel_ranks.items()
                if channel in weights
            )
            fused.append((candidate.tier, -score, chunk_id, candidate))
        fused.sort(key=lambda item: (item[0], item[1], item[2]))
        # Fusion decides what is considered; the cross-encoder decides the
        # order. It reads the query and the passage together rather than each
        # alone, and on the admitted corpus that ordering carried more of the
        # casebook's facts through this cut than rank fusion did. That is this
        # corpus's measured result, not a property of either family. The score
        # it returns is what the packet compares across queries.
        # Walked per document: each filing's best `reranker_depth_per_document`
        # fused candidates are read, so a filing whose event sits below another
        # filing's boilerplate in the fused order is still read. Measured, the
        # facts a global cut of 150 never reached were every one of them
        # inside their own filing's top eight.
        # The request may read deeper than the spec's six: measured on the
        # admitted corpus, a compliance statement a direct question ranked
        # sixth by the dense channel and twelfth by bm25 fused ninth in its
        # 300-chunk filing and was never scored. The window bounds it.
        depth = min(
            self.index_spec.candidate_depth_per_document,
            request.reranker_depth_per_document or self.index_spec.reranker_depth_per_document,
        )
        taken: dict[str, int] = {}
        considered: list[tuple[int, float, str, _Candidate]] = []
        for item in fused:
            document_id = str(item[3].row["document_id"])
            if taken.get(document_id, 0) < depth:
                taken[document_id] = taken.get(document_id, 0) + 1
                considered.append(item)
        scores = self._score_pairs(
            request.query, [str(item[3].row["body_raw"]) for item in considered]
        )
        ranked = sorted(
            (
                (item[0], -score, item[2], item[3])
                for item, score in zip(considered, scores, strict=True)
            ),
            key=lambda item: (item[0], item[1], item[2]),
        )
        selected = _cut_per_group(ranked, request)
        # No passage inference here. The vectors were proved against the model
        # when this generation was opened, and a query verifies its hits
        # against that commitment rather than paying for it again.
        verified = tuple(self._vectors.get(item[2]) for item in selected)
        if any(vector is None for vector in verified):
            raise _retrieval_error(
                "hybrid hit is outside the verified projection commitment",
                code="retrieval.index_corrupt",
            )
        hits: list[KnowledgeRetrievalHit] = []
        for (_, negative_score, chunk_id, candidate), vector in zip(
            selected,
            verified,
            strict=True,
        ):
            authoritative = self._chunks.get(chunk_id)
            if authoritative is None or vector is None:
                raise _retrieval_error(
                    "hybrid index returned an unknown authoritative chunk",
                    code="retrieval.index_corrupt",
                )
            self._verify_hit(candidate, authoritative, vector)
            channels = tuple(sorted(candidate.channels, key=lambda item: item.value))
            if candidate.tier == 0:
                kind = RetrievalMatchKind.IDENTIFIER_EXACT
            elif candidate.tier == 1:
                kind = RetrievalMatchKind.TERM_EXACT
            elif candidate.channels == {RetrievalChannel.TRIGRAM}:
                kind = RetrievalMatchKind.TRIGRAM_ONLY
            else:
                kind = RetrievalMatchKind.FUZZY
            preview = _normalize_text(authoritative.body)[:240]
            if not preview:
                preview = authoritative.heading or authoritative.title
            hits.append(
                KnowledgeRetrievalHit(
                    rank=len(hits) + 1,
                    title=authoritative.title,
                    preview=preview,
                    match_kind=kind,
                    channels=channels,
                    rrf_score=-negative_score,
                    citation=authoritative.citation,
                )
            )
        return tuple(hits), len(considered)

    def _verify_hit(
        self,
        candidate: _Candidate,
        authoritative: _Chunk,
        vector: Sequence[float],
    ) -> None:
        assert self._connection is not None
        try:
            citation_raw = str(candidate.row["citation_json"]).encode("utf-8")
            if self._schema == _INDEX_SCHEMA_VERSION:
                citation_ok = citation_raw == _corpus_citation_json(authoritative).encode("utf-8")
            else:
                citation = parse_model(citation_raw, KnowledgeCitation)
                citation_ok = (
                    citation_raw == canonical_json_bytes(citation)
                    and citation == authoritative.citation
                )
            rowid = int(candidate.row["content_rowid"])
            with closing(
                self._connection.execute(
                    """
                    SELECT
                        n.chunk_id,
                        n.embedding_input_hash,
                        n.vector_hash,
                        v.embedding
                    FROM dense_nodes AS n
                    JOIN vec_chunks AS v ON v.rowid = n.rowid
                    WHERE n.rowid = ?
                    """,
                    (rowid,),
                )
            ) as cursor:
                dense = cursor.fetchone()
            if dense is None:
                raise sqlite3.DatabaseError("hit dense mapping is missing")
            blob = bytes(dense["embedding"])
            stored_vector = _deserialize_vector(
                blob,
                dimension=self.index_spec.embedding_dimension,
            )
            if (
                not citation_ok
                or str(candidate.row["chunk_hash"]) != authoritative.chunk_hash
                or str(candidate.row["document_id"]) != authoritative.document_id
                or int(candidate.row["revision"]) != authoritative.revision
                or str(candidate.row["title_raw"]) != authoritative.title
                or str(candidate.row["heading_raw"]) != authoritative.heading
                or str(candidate.row["body_raw"]) != authoritative.body
                or str(dense["chunk_id"]) != authoritative.chunk_id
                or str(dense["embedding_input_hash"])
                != _embedding_input_hash(authoritative, self.index_spec)
                or str(dense["vector_hash"]) != sha256_hex(blob)
                or not _vector_matches(
                    vector,
                    stored_vector,
                    tolerance=self.index_spec.vector_tolerance,
                )
            ):
                raise sqlite3.DatabaseError("hybrid hit differs from authoritative source")
        except (
            sqlite3.Error,
            DomainValidationError,
            ValidationError,
            TypeError,
            ValueError,
        ) as exc:
            raise _retrieval_error(
                "hybrid hit verification failed",
                code="retrieval.index_corrupt",
                cause=exc,
            ) from exc


__all__ = [
    "HybridIndexPriming",
    "KnowledgeInventoryWindow",
    "WorkspaceHybridKnowledgeIndex",
    "WorkspaceHybridKnowledgeRetriever",
    "committed_index_held",
    "corpus_logical_hash",
    "probe_hybrid_retrieval_capabilities",
]
