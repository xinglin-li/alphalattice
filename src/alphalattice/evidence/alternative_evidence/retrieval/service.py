"""A review's retrieval over the Host's retrieval store.

The document set's canonical revisions are reference sources of the store (`documents.library`).
A generation chunks and embeds them through the store, once for every request over the same
revisions, and seals what a search of them reads; a session searches them through the store's
hybrid search. Nothing here is an index file: the store's chunks and vectors are the index.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from alphalattice.kernel.data_layer import history, recall
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.identity_successors import is_current

from ..contracts import seal_contract
from ..documents.contracts import AlternativeEvidenceDocumentSet
from ..documents.library import EvidenceSnapshot
from ..documents.workspace import AlternativeEvidenceDocumentPublisher
from ..publication.contracts import RETRIEVAL_BINDING_ROLE
from .contracts import AlternativeEvidenceRetrievalGeneration, RetrievalGenerationRecord
from .session import AlternativeEvidenceRetrievalSession, OriginalReader


@dataclass(frozen=True)
class StoreCapability:
    """Whether the store's encoder and reranker packs are installed whole: `READY`, or
    `UNAVAILABLE` with the cause; `logical_hash` is the spec a generation is sealed under."""

    status: str
    details: tuple[str, ...]
    logical_hash: str


def reference_spec_identity(spec: dict[str, Any] | None = None) -> str:
    """The identity of the recipe, models and search settings every result depends on
    (`recall.reference_spec()` when `spec` is None): a generation's `index_spec_hash` and the
    capability an authority's setup seals."""
    return str(canonical_hash(recall.reference_spec() if spec is None else spec))


def store_capability(store: Path | None = None) -> StoreCapability:
    """The store's packs verified in `store` (the application model store when None), without
    building anything: what an authority's setup seals and a review's workspace checks."""
    from alphalattice.kernel.knowledge import model_store  # lazy: the setup's pack owner

    root = model_store.default_store_root() if store is None else store
    spec_hash = reference_spec_identity()
    for pack in model_store.RETRIEVAL_STORE_PACKS:
        refused = model_store.verify_directory(model_store.pack_directory(root, pack), pack)
        if refused is not None:
            return StoreCapability("UNAVAILABLE", (f"cause:{pack.pack_id} {refused}",), spec_hash)
    return StoreCapability("READY", (), spec_hash)


class AlternativeEvidenceRetrievalService:
    """A workspace's review retrieval: generations sealed over the store, sessions searching it."""

    def __init__(self, *, workspace_root: Path, retrieval_binding_hash: str) -> None:
        """Bind the workspace's canonical documents and the retrieval binding generations seal."""
        self.publisher = AlternativeEvidenceDocumentPublisher(workspace_root)
        self.workspace = self.publisher.workspace
        self.retrieval_binding_hash = retrieval_binding_hash
        self.index_spec = recall.reference_spec()
        """The recipe, models and search settings every result depends on (`index_spec_hash`)."""
        self.index_spec_hash = reference_spec_identity(self.index_spec)
        self.embedded_chunk_count = 0
        """Chunks the store encoded for this service's generations; a chunk it held adds none."""
        self.reused_chunk_count = 0
        """Chunks whose vectors the store already held, over every generation."""
        self.passage_embedding_pass_count = 0
        """Builds that encoded every chunk of their corpus: the store held none of its passages."""
        self.incremental_build_count = 0
        """Builds that encoded some chunks and found the rest in the store."""
        self.generation_reuse_count = 0
        """Builds that encoded nothing: the store held every chunk's vector."""
        self.generation_release_count = 0
        """Selections reused from a sealed receipt, verified without opening a session."""
        self._verified_sets: set[str] = set()
        """Document sets verified against their retained content in this process (`_verify`)."""
        self._indexes: dict[str, recall.ReferenceIndex] = {}
        """Each generation this service built or opened, by its hash, with the store's index read
        for it: a later open in this process carries it rather than reading the index again."""

    def close(self) -> None:
        """Nothing is held between calls: the Host holds the store's search."""

    def _verify(self, document_set: AlternativeEvidenceDocumentSet) -> None:
        """Verify a document set against its retained content once in this process: the set is
        identified by its hash, and its published revisions never change."""
        if document_set.document_set_hash not in self._verified_sets:
            self.publisher.verify(document_set)
            self._verified_sets.add(document_set.document_set_hash)

    def _snapshot(self, document_set: AlternativeEvidenceDocumentSet) -> EvidenceSnapshot:
        self._verify(document_set)
        return self.publisher.library.read_snapshot(UUID(document_set.workspace_snapshot_id))

    def index_id(self, document_set: AlternativeEvidenceDocumentSet) -> str:
        """The corpus under the spec: the same for every request over the same revisions."""
        return _index_id(self._snapshot(document_set), self.index_spec_hash)

    def _indexed(
        self, document_set: AlternativeEvidenceDocumentSet
    ) -> tuple[EvidenceSnapshot, recall.ReferenceIndex]:
        snapshot = self._snapshot(document_set)
        index = history.index_references(
            workspace=self.publisher.library.workspace_id,
            revisions=[(item.document.document_id, item.revision) for item in snapshot.revisions],
        )
        if index is None:
            raise ValueError("alternative_evidence.host_history_unavailable")
        self.embedded_chunk_count += index.embedded
        self.reused_chunk_count += index.reused
        return snapshot, index

    def capability(self) -> StoreCapability:
        """Verify the store's packs without building anything (`store_capability`)."""
        return store_capability()

    def work_accounting(self) -> dict[str, int]:
        """What the store encoded for this service beside what it already held."""
        return {
            "published_passage_embedding_passes": int(self.passage_embedding_pass_count),
            "published_incremental_builds": int(self.incremental_build_count),
            "generation_reuses": int(self.generation_reuse_count),
            "published_embedded_chunks": int(self.embedded_chunk_count),
            "published_reused_chunks": int(self.reused_chunk_count),
        }

    def build(
        self, *, document_set: AlternativeEvidenceDocumentSet, built_at: datetime
    ) -> AlternativeEvidenceRetrievalGeneration:
        """Chunk and embed the verified document set through the store and seal the generation."""
        snapshot, index = self._indexed(document_set)
        if not index.embedded:
            self.generation_reuse_count += 1
        elif index.reused:
            self.incremental_build_count += 1
        else:
            self.passage_embedding_pass_count += 1
        generation = seal_contract(
            AlternativeEvidenceRetrievalGeneration,
            "generation_hash",
            document_set_hash=document_set.document_set_hash,
            workspace_snapshot_id=document_set.workspace_snapshot_id,
            workspace_snapshot_hash=document_set.workspace_snapshot_hash,
            retrieval_binding_hash=self.retrieval_binding_hash,
            reading_depth=document_set.reading_depth,
            index_id=_index_id(snapshot, self.index_spec_hash),
            index_spec_hash=self.index_spec_hash,
            chunk_set_hash=_chunk_set_hash(snapshot, index),
            vector_set_hash=index.vectors,
            document_count=len(snapshot.revisions),
            chunk_count=sum(len(chunks) for chunks in index.chunks.values()),
            built_at=built_at,
        )
        self._indexes[generation.generation_hash] = index
        return generation

    def verify_lineage(
        self,
        document_set: AlternativeEvidenceDocumentSet,
        generation: RetrievalGenerationRecord,
    ) -> None:
        """Verify generation lineage against its document set and the current spec."""
        self._verify(document_set)
        if (
            generation.document_set_hash != document_set.document_set_hash
            or generation.workspace_snapshot_id != document_set.workspace_snapshot_id
            or generation.workspace_snapshot_hash != document_set.workspace_snapshot_hash
            or not is_current(
                RETRIEVAL_BINDING_ROLE,
                generation.retrieval_binding_hash,
                self.retrieval_binding_hash,
            )
            or generation.index_spec_hash != self.index_spec_hash
            or generation.reading_depth is not document_set.reading_depth
        ):
            raise ValueError("alternative_evidence.retrieval_generation_lineage_invalid")

    def release_without_session(
        self,
        *,
        document_set: AlternativeEvidenceDocumentSet,
        generation: RetrievalGenerationRecord,
    ) -> None:
        """Verify what a session would verify, for a selection a sealed receipt already proves."""
        self.verify_lineage(document_set, generation)
        self.generation_release_count += 1

    def open_session(
        self,
        *,
        document_set: AlternativeEvidenceDocumentSet,
        generation: RetrievalGenerationRecord,
        evidence_as_of: datetime,
        original_for: OriginalReader | None = None,
    ) -> AlternativeEvidenceRetrievalSession:
        """Open a read session over a verified generation.

        The store's chunks and vectors for the document set must be the ones the record sealed:
        read once in this process, by the build that sealed it or by the first open, whose index
        later opens carry. A generation sealed before the store is history and is never opened.
        With `original_for`, the session renders table views from the retained originals.
        """
        self.verify_lineage(document_set, generation)
        if not isinstance(generation, AlternativeEvidenceRetrievalGeneration):
            raise ValueError("alternative_evidence.retrieval_generation_retired")
        index = self._indexes.get(generation.generation_hash)
        if index is None:
            snapshot, index = self._indexed(document_set)
            if (_chunk_set_hash(snapshot, index), index.vectors) != (
                generation.chunk_set_hash,
                generation.vector_set_hash,
            ):
                raise ValueError("alternative_evidence.retrieval_generation_divergent")
            self._indexes[generation.generation_hash] = index
        return AlternativeEvidenceRetrievalSession(
            library=self.publisher.library,
            search=_searcher(self.publisher.library.workspace_id),
            chunks={
                document: tuple((chunk.start, chunk.end) for chunk in chunks)
                for document, chunks in index.chunks.items()
            },
            document_set=document_set,
            generation=generation,
            evidence_as_of=evidence_as_of,
            original_for=original_for,
        )


def _index_id(snapshot: EvidenceSnapshot, spec_hash: str) -> str:
    digest: str = canonical_hash(
        {
            "revisions": [
                [item.document.document_id, item.revision, item.content_sha256]
                for item in snapshot.revisions
            ],
            "spec": spec_hash,
        }
    )
    return digest


def _chunk_set_hash(snapshot: EvidenceSnapshot, index: recall.ReferenceIndex) -> str:
    digest: str = canonical_hash(
        [
            [
                item.document.document_id,
                [
                    [chunk.start, chunk.end, chunk.digest]
                    for chunk in index.chunks[item.document.document_id]
                ],
            ]
            for item in snapshot.revisions
        ]
    )
    return digest


def _searcher(workspace: str) -> Callable[..., Any]:
    def search(query: str, **scope: Any) -> Any:
        found = history.search_references(query, workspace=workspace, **scope)
        if found is None:
            raise ValueError("alternative_evidence.host_history_unavailable")
        return found

    return search


__all__ = ["AlternativeEvidenceRetrievalService", "StoreCapability"]
