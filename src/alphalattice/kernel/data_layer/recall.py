"""The Host's held search over its retrieval store (`knowledge`), embedded by the store's models
(`encoders`): recall over the experience corpus the store projects from the log, and the search
a review runs over the reference revisions it published.

The corpus is a projection the Host rebuilds from the log: every experience version the store does
not hold yet is published, chunked and embedded before a query reads it, so an experience made
while recall was never asked is found at the first recall. The search, its matrix and its models
are held per held log and dropped when the history closes.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import alphalattice.kernel.data_layer.knowledge as knowledge
from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer.chain import digest
from alphalattice.kernel.data_layer.knowledge import (
    Chunk,
    HybridSearch,
    KnowledgeStore,
    Retrieved,
    SearchHit,
)
from alphalattice.kernel.data_layer.log import HostLog
from alphalattice.kernel.shared_kernel.environment import package_versions

PRECISION: Final = "int8"
"""The cache's storage precision: the retrieval measurement admits int8 (61/96 and 91/100
against float16's 60/96 and 89/100); search runs in float32."""
CAPACITY: Final = 512
"""Characters a chunk holds at most, the splitter's own measure; chunks do not overlap."""
_HAN: Final = re.compile(r"[一-鿿]")
_DUE: Final = (  # ?1 the active recipe, ?2 the active model
    "SELECT v.experience_id, v.version, e.workspace_id, e.recorded_us FROM experience_version v"
    " JOIN event e ON e.seq = v.recorded_seq"
    " LEFT JOIN knowledge_source_version k"
    " ON k.source_id = v.experience_id AND k.version = CAST(v.version AS TEXT)"
    " WHERE k.version_id IS NULL"
    " OR NOT EXISTS (SELECT 1 FROM knowledge_chunk c"
    " WHERE c.version_id = k.version_id AND c.recipe_id = ?1)"
    " OR EXISTS (SELECT 1 FROM knowledge_chunk c LEFT JOIN knowledge_embedding m"
    " ON m.chunk_digest = c.chunk_digest AND m.recipe_id = c.recipe_id AND m.model_id = ?2"
    " WHERE c.version_id = k.version_id AND c.recipe_id = ?1 AND m.embedding_id IS NULL)"
    " ORDER BY v.recorded_seq"
)
"""The experience versions recall cannot yet search under the active recipe and model: not
published, not chunked, or with a chunk lacking its vector. Each step is idempotent, so a catch-up
a stop or a failed encoding interrupted resumes from what the store holds."""
_TRIALS: Final = (
    "SELECT trial_id FROM trial WHERE experience_id = ?1"
    " UNION SELECT trial_id FROM trial_experience WHERE experience_id = ?1 AND role = 'reviewed'"
    " ORDER BY 1"
)
_REFERENCE: Final = (
    "SELECT v.version_id, t.text FROM knowledge_workspace_source w"
    " JOIN knowledge_source_version v USING (version_id)"
    " JOIN knowledge_source s USING (source_id)"
    " LEFT JOIN event_text t ON t.seq = v.body_seq"
    " WHERE w.workspace_id = ?1 AND w.document_id = ?2 AND w.revision = ?3 AND s.kind = 'reference'"
)
_BODY: Final = (
    "SELECT t.text FROM knowledge_source_version v JOIN event_text t ON t.seq = v.body_seq"
    " WHERE v.source_id = ? AND v.version = ?"
)

_guard = threading.Lock()
_held: dict[str, Any] = {}


def _recipe() -> dict[str, Any]:
    return {
        "splitter": "semantic-text-splitter",
        "version": dict(package_versions(("semantic-text-splitter",)))["semantic-text-splitter"],
        "capacity": CAPACITY,
        "overlap": 0,
        "trim": True,
    }


def _manifest(pack: Any) -> dict[str, Any]:
    return {"pack": pack.pack_id, "revision": pack.revision, "files": dict(pack.files)}


def reference_spec() -> dict[str, Any]:
    """What a reference search returns depends on beside the revisions: the chunking recipe, the
    embedding model and its storage, the reranker, and the search's fusion, windows and
    deadline."""
    import alphalattice.kernel.data_layer.encoders as encoders  # lazy: the semantic extra's owner
    from alphalattice.kernel.knowledge import model_store  # lazy: the setup's pack owner

    encoder, reranker = model_store.RETRIEVAL_STORE_PACKS
    return {
        "recipe": _recipe(),
        "model": {"manifest": _manifest(encoder), "dimension": 512, "precision": PRECISION},
        "reranker": _manifest(reranker),
        "search": {
            "fusion": "min-max linear, 0.5 lexical + 0.5 vector",
            "window": knowledge.WINDOW,
            "shortlist": knowledge.SHORTLIST,
            "pair_tokens": encoders.PAIR_TOKENS,
            "deadline_seconds": encoders.DEADLINE_SECONDS,
        },
    }


def _packs() -> tuple[Path, Path, dict[str, Any]]:
    """The installed encoder and reranker pack directories and the manifest the cache keys on;
    a pack not installed whole is refused by name."""
    from alphalattice.kernel.knowledge import model_store  # lazy: the setup's pack owner

    store = model_store.default_store_root()
    directories = []
    for pack in model_store.RETRIEVAL_STORE_PACKS:
        directory = model_store.pack_directory(store, pack)
        if model_store.verify_directory(directory, pack) is not None:
            raise DataError(
                f"the {pack.pack_id} model pack is not installed; install the"
                f" {model_store.RETRIEVAL_STORE} packs with the retrieval pack setup",
                code="data_layer.knowledge_model_unavailable",
            )
        directories.append(directory)
    return directories[0], directories[1], _manifest(model_store.RETRIEVAL_STORE_PACKS[0])


def _catch_up(
    store: KnowledgeStore, recipe_id: str, model_id: str, encoder: Any, blobs: Path
) -> None:
    """Publish, chunk and embed every experience version not yet searchable under the active recipe
    and model (`_DUE`); a vector the cache holds is never encoded again."""
    for experience_id, number, workspace, recorded_us in store.log.read(
        _DUE, (recipe_id, model_id)
    ):
        version_id = store.publish_experience(
            workspace_id=workspace, experience_id=experience_id, version=number, blobs=blobs
        )
        content = store.experience_body(
            workspace_id=workspace,
            experience_id=experience_id,
            version=number,
            blobs=blobs,
            cutoff_us=recorded_us,
        )
        chunks = store.index_chunks(version_id=version_id, recipe_id=recipe_id, content=content)
        texts = {chunk.digest: content[chunk.start : chunk.end] for chunk in chunks}
        store.embeddings(
            tuple(texts),
            recipe_id=recipe_id,
            model_id=model_id,
            encode=lambda missing, texts=texts: encoder.embed_passages([texts[d] for d in missing]),
        )


def _body(store: KnowledgeStore, blobs: Path) -> Any:
    def body(hit: SearchHit, cutoff_us: int) -> str:
        (workspace,) = store.log.read(
            "SELECT e.workspace_id FROM experience_version v JOIN event e ON e.seq = v.recorded_seq"
            " WHERE v.experience_id = ? AND v.version = ?",
            (hit.source_id, int(hit.version)),
        )[0]
        return store.experience_body(
            workspace_id=workspace,
            experience_id=hit.source_id,
            version=int(hit.version),
            blobs=blobs,
            cutoff_us=cutoff_us,
        )

    return body


@dataclass(frozen=True)
class ReferenceIndex:
    """What a set of reference revisions is searched with: the chunking recipe and the model, as
    specified; each source's chunks in ordinal order; and the digest of their vectors in that
    order, with how many were encoded now and how many the cache held."""

    recipe: dict[str, Any]
    model: dict[str, Any]
    chunks: dict[str, tuple[Chunk, ...]]
    vectors: str
    embedded: int
    reused: int


def _hold(log: HostLog) -> dict[str, Any]:
    """The held log's store, recipe, model and model sessions, made at its first search."""
    from alphalattice.kernel.data_layer.encoders import (  # lazy: onnxruntime and tokenizers
        Reranker,
        StaticEncoder,
    )

    if _held.get("log") is not log:
        _drop()
        encoder_pack, reranker_pack, manifest = _packs()
        store = KnowledgeStore(log)
        recipe_id = store.recipe(_recipe())
        model_id = store.model(manifest, dimension=512, precision=PRECISION)
        _held.update(log=log, store=store, recipe_id=recipe_id, model_id=model_id)
        _held["model"] = {"manifest": manifest, "dimension": 512, "precision": PRECISION}
        _held["encoder"], _held["reranker"] = StaticEncoder(encoder_pack), Reranker(reranker_pack)
    return _held


def _index(store: KnowledgeStore, root: Path) -> Any:
    """The held matrix, brought up to the vectors the store holds."""
    index = _held.get("index")
    if index is None or not index.refresh():
        if index is not None:
            index.close()
        index = _held["index"] = store.vector_index(
            recipe_id=_held["recipe_id"], model_id=_held["model_id"], directory=root / "staging"
        )
    return index


def _search(log: HostLog, root: Path) -> HybridSearch:
    """The held log's recall, brought up to the log's experiences at each."""
    held = _hold(log)
    store = held["store"]
    _catch_up(store, held["recipe_id"], held["model_id"], held["encoder"], root / "blobs")
    return HybridSearch(
        store, _index(store, root), held["encoder"], held["reranker"], _body(store, root / "blobs")
    )


def index_references(
    log: HostLog, *, workspace: str, revisions: Sequence[tuple[str, int]]
) -> ReferenceIndex:
    """Chunk and embed each (document, revision) `workspace` published as a reference, once: a
    chunk or vector the store holds is never made again. A revision that is not a published
    reference is refused by name."""
    with _guard:
        held = _hold(log)
        store, recipe_id, model_id = held["store"], held["recipe_id"], held["model_id"]
        chunks: dict[str, tuple[Chunk, ...]] = {}
        texts: dict[str, str] = {}
        for document, revision in revisions:
            rows = log.read(_REFERENCE, (workspace, document, revision))
            if not rows or rows[0][1] is None:
                raise DataError(
                    f"no reference revision {document}@{revision} in workspace {workspace}",
                    code="data_layer.knowledge_revision_missing",
                )
            version_id, text = rows[0]
            content = json.loads(text)["content"]
            found = store.index_chunks(version_id=version_id, recipe_id=recipe_id, content=content)
            chunks[document] = found
            texts.update({chunk.digest: content[chunk.start : chunk.end] for chunk in found})
        ordered = tuple(chunk.digest for found in chunks.values() for chunk in found)
        read = store.embeddings(
            ordered,
            recipe_id=recipe_id,
            model_id=model_id,
            encode=lambda missing: held["encoder"].embed_passages([texts[d] for d in missing]),
        )
        return ReferenceIndex(
            _recipe(),
            dict(held["model"]),
            chunks,
            digest(read.vectors.astype("<f4").tobytes()),
            read.embedded,
            read.reused,
        )


def search_references(
    log: HostLog,
    root: Path,
    query: str,
    *,
    workspace: str,
    cutoff_us: int,
    ranges: tuple[tuple[str, int, int], ...],
    groups: tuple[tuple[str, ...], ...],
    depth: int | None,
    limit: int,
) -> Retrieved:
    """The reference chunks of `workspace` known at `cutoff_us` most like `query`, within `ranges`
    when given, each group of sources taking its own `limit`, each source reranked to `depth`
    (`HybridSearch.retrieve`); English reranked and Chinese not."""
    with _guard:
        held = _hold(log)
        store = held["store"]

        def body(hit: SearchHit, _cutoff_us: int) -> str:
            (text,) = log.read(_BODY, (hit.source_id, hit.version))[0]
            return str(json.loads(text)["content"])

        return HybridSearch(
            store, _index(store, root), held["encoder"], held["reranker"], body
        ).retrieve(
            query,
            language="zh" if _HAN.search(query) else "en",
            workspace=workspace,
            cutoff_us=cutoff_us,
            kind="reference",
            limit=limit,
            ranges=ranges,
            groups=groups,
            depth=depth,
        )


def recall(
    log: HostLog,
    root: Path,
    *,
    query: str,
    workspace: str,
    cutoff_us: int,
    superseded: bool | None = False,
    limit: int = 5,
) -> tuple[Retrieved, dict[str, tuple[str, ...]]]:
    """The experiences most like `query` in `workspace` as known at `cutoff_us`, in the fused order
    (the default path calls no cross-encoder), and each hit's trials (the trial its note is on, or
    each one its review task reviewed)."""
    with _guard:
        found = _search(log, root).retrieve(
            query,
            language="zh" if _HAN.search(query) else "en",
            workspace=workspace,
            cutoff_us=cutoff_us,
            kind="experience",
            superseded=superseded,
            limit=limit,
        )
    trials = {
        hit.hit.source_id: tuple(row[0] for row in log.read(_TRIALS, (hit.hit.source_id,)))
        for hit in found.hits
    }
    return found, trials


def close() -> None:
    """Drop the held search: its matrix file and its model sessions."""
    with _guard:
        _drop()


def _drop() -> None:
    index = _held.pop("index", None)
    if index is not None:
        index.close()
    _held.clear()
