"""Thin adapter over the tracked Workspace hybrid index and retriever."""

from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from alphalattice.control.workspace_runtime.core import Workspace
from alphalattice.kernel.knowledge.hybrid import (
    VectorBlockSource,
    WorkspaceHybridKnowledgeIndex,
    WorkspaceHybridKnowledgeRetriever,
    committed_database_relative,
    committed_index_held,
    corpus_logical_hash,
    index_identity_of_database,
)
from alphalattice.kernel.knowledge.hybrid_contracts import (
    VECTOR_PAYLOAD_ROOT,
    HybridCapabilityReport,
    HybridGenerationAnchor,
    HybridIndexSpec,
    PairScoreAdmission,
)
from alphalattice.kernel.shared_kernel.identity_successors import is_current

from ..contracts import seal_contract
from ..documents.contracts import AlternativeEvidenceDocumentSet
from ..documents.workspace import AlternativeEvidenceDocumentPublisher
from ..publication.contracts import RETRIEVAL_BINDING_ROLE
from .contracts import (
    AlternativeEvidenceRetrievalGeneration,
    RetrievalGenerationRecord,
)
from .session import AlternativeEvidenceRetrievalSession, OriginalReader

IndexFactory = Callable[[Workspace, Path], Any]
RetrieverFactory = Callable[[Workspace, Path, HybridIndexSpec], Any]
AnchorResolver = Callable[[str], HybridGenerationAnchor | None]
"""`(index_id) -> the sealed commitment for that generation`, or None."""
StorageAdmission = Callable[[int], None]
"""`(additional_bytes)`: the workspace's capacity admission; raises to refuse."""


def _index(workspace: Workspace, model_root: Path) -> WorkspaceHybridKnowledgeIndex:
    return WorkspaceHybridKnowledgeIndex(workspace, model_root=model_root)


def _retriever(
    workspace: Workspace,
    model_root: Path,
    spec: HybridIndexSpec,
) -> WorkspaceHybridKnowledgeRetriever:
    return WorkspaceHybridKnowledgeRetriever(
        workspace,
        model_root=model_root,
        index_spec=spec,
    )


class AlternativeEvidenceRetrievalService:
    """Own domain identity while delegating ranking and index bytes to Workspace.

    One generation record is one request's use of one physical index. The
    Workspace keys that index by the corpus the request's document set
    resolves to, so a second request over the same eligible bytes -- another
    cutoff, another book, another day -- reuses it and embeds nothing; only
    a corpus without an index is embedded. A built or reused index is kept
    open until the session that reads it is opened; a later process reopens
    the generation from the Workspace by the commitments this record carries.
    """

    def __init__(
        self,
        *,
        workspace_root: Path,
        model_root: Path,
        retrieval_binding_hash: str,
        index_factory: IndexFactory | None = None,
        retriever_factory: RetrieverFactory | None = None,
        index_spec: HybridIndexSpec | None = None,
    ) -> None:
        """Bind retrieval storage, model artifacts, and active lease state."""
        self.publisher = AlternativeEvidenceDocumentPublisher(workspace_root)
        self.workspace = self.publisher.workspace
        self.model_root = model_root
        self.retrieval_binding_hash = retrieval_binding_hash
        self.index_spec = index_spec or HybridIndexSpec.fixed_v2()
        self._index_factory = index_factory or _index
        self._retriever_factory = retriever_factory or _retriever
        self._pending_generations: dict[str, Any] = {}
        # One lease registry for every hold this process has on a generation:
        # a build in progress (its identity known before the model runs), a
        # built index waiting for its session, and every open session until
        # it closes. Cleanup consults it under the same lock it is changed
        # under, so a hold taken between a plan and its apply is seen.
        self._lease_lock = threading.Lock()
        self._building: dict[str, int] = {}
        self._open_sessions: dict[str, str] = {}
        self.passage_embedding_pass_count = 0
        """Builds that put every chunk of their corpus through the passage model:
        a corpus none of whose revisions had a committed block. A reused
        generation adds nothing; an incremental build is counted apart."""
        self.incremental_build_count = 0
        """Builds that embedded some revisions and composed the rest from blocks."""
        self.embedded_chunk_count = 0
        """Chunks put through the passage model, over every build."""
        self.reused_chunk_count = 0
        """Chunks whose vectors came from committed blocks, over every build."""
        self.embedding_call_count = 0
        """Passage-model calls, one per embedded revision."""
        self.generation_reuse_count = 0
        """Builds answered by an existing verified generation of the same corpus."""
        self.generation_handoff_count = 0
        self.independent_generation_reopen_count = 0
        self.generation_release_count = 0
        """Built leases closed without a session because a sealed receipt
        already proved the selection (`release_without_session`)."""

    def close(self) -> None:
        """Release all retrieval resources held by this service."""
        for lease in self._pending_generations.values():
            lease.close()
        self._pending_generations.clear()

    def active_index_ids(self) -> frozenset[str]:
        """List the index generations held by this process.

        Every generation this process holds: being built, built and waiting
        for its session, or read by an open session.
        """
        with self._lease_lock:
            return self._active_ids_locked()

    def _active_ids_locked(self) -> frozenset[str]:
        found = set(self._building)
        found.update(self._open_sessions.values())
        for lease in self._pending_generations.values():
            try:
                found.add(str(lease.manifest.index_id))
            except Exception:
                continue
        return frozenset(found)

    def _hold_building(self, index_id: str) -> None:
        with self._lease_lock:
            self._building[index_id] = self._building.get(index_id, 0) + 1

    def _release_building(self, index_id: str) -> None:
        with self._lease_lock:
            count = self._building.get(index_id, 0) - 1
            if count > 0:
                self._building[index_id] = count
            else:
                self._building.pop(index_id, None)

    def discard_pending(self, generation_hash: str) -> None:
        """Release an unused built generation lease.

        Drop a built lease no session will take: the caller refused to
        seal its record, or the selection it was built for is already
        proved by a sealed receipt (`release_without_session`).
        """
        with self._lease_lock:
            lease = self._pending_generations.pop(generation_hash, None)
        if lease is not None:
            lease.close()

    def _release_session(self, token: str) -> None:
        with self._lease_lock:
            self._open_sessions.pop(token, None)

    def evict_generation(
        self, database_relative: str, *, plan_hash: str, evicted_at: datetime
    ) -> int:
        """Release one index under an approved plan; the Workspace leaves the marker.

        Refused while any hold on the generation is alive -- a build, a
        waiting lease or an open session -- checked under the lease lock so
        a session opened after the plan was made cannot be pulled from under
        its reader; the cleanup is left to resume once the hold ends.
        """
        with self._lease_lock:
            if database_relative.startswith(VECTOR_PAYLOAD_ROOT + "/"):
                # A vector object no sealed generation composes: released as a
                # plain file, but never while a build is composing -- a block
                # read a moment ago may be about to be referenced.
                if self._building:
                    raise ValueError("alternative_evidence.retrieval_index_in_use")
                index = self._index_factory(self.workspace, self.model_root)
                return int(index.release_vector_object(database_relative))
            index_id = index_identity_of_database(database_relative)
            if index_id is not None and index_id in self._active_ids_locked():
                raise ValueError("alternative_evidence.retrieval_index_in_use")
            index = self._index_factory(self.workspace, self.model_root)
            return int(index.evict(database_relative, plan_hash=plan_hash, evicted_at=evicted_at))

    def rebuild_generation(
        self,
        *,
        document_set: AlternativeEvidenceDocumentSet,
        generation: AlternativeEvidenceRetrievalGeneration,
        admit: StorageAdmission | None = None,
        block_for: VectorBlockSource | None = None,
    ) -> dict[str, object]:
        """Restore the index a committed generation record names, from its payload.

        No lease is kept: the caller asked for the index to be on disk, not for
        a session. The pinned model runs only when the payload itself is lost,
        and the Workspace then requires its output to reproduce the committed
        digest exactly.
        """
        self.publisher.verify(document_set)
        if (
            generation.document_set_hash != document_set.document_set_hash
            or generation.index_spec_hash != self.index_spec.logical_hash
        ):
            raise ValueError("alternative_evidence.retrieval_generation_lineage_invalid")
        snapshot = self.publisher.library.read_snapshot(
            _snapshot_id(document_set.workspace_snapshot_id)
        )
        index = self._index_factory(self.workspace, self.model_root)
        self._hold_building(generation.index_id)
        try:
            priming = index.rebuild_from_commitment(
                snapshot,
                self.index_spec,
                manifest_logical_hash=generation.index_manifest_hash,
                payload_sha256=generation.vector_payload_sha256,
                admit=admit,
                block_for=block_for,
            )
        finally:
            self._release_building(generation.index_id)
        self._count(priming)
        manifest = priming.manifest
        database = self.workspace.root / Path(*manifest.database_path.split("/"))
        return {
            "index_id": manifest.index_id,
            "generation_format": generation.generation_format,
            "corpus_hash": manifest.corpus_logical_hash,
            "index_manifest_hash": manifest.logical_hash,
            "vector_payload_sha256": manifest.vector_commitment.payload_sha256,
            "chunk_count": manifest.chunk_count,
            "passage_embedding_pass": priming.embedded,
            "embedded_chunks": priming.embedded_chunks,
            "reused_chunks": priming.reused_chunks,
            "model_calls": priming.model_calls,
            "database_bytes": database.stat().st_size,
        }

    def work_accounting(self) -> dict[str, int]:
        """What was published beside what was attempted.

        The owner's own counts move when a build publishes; the model ledger
        moves when a model is asked, so a build that failed or was cancelled
        after embedding still shows the passages it cost, and query embeddings
        and cross-encoder pairs -- which no build count carries -- are here.
        """
        from alphalattice.kernel.knowledge._embeddings import MODEL_WORK

        return {
            "published_passage_embedding_passes": int(self.passage_embedding_pass_count),
            "published_incremental_builds": int(self.incremental_build_count),
            "published_embedded_chunks": int(self.embedded_chunk_count),
            "published_reused_chunks": int(self.reused_chunk_count),
            "published_embedding_calls": int(self.embedding_call_count),
            "generation_reuses": int(self.generation_reuse_count),
            **{f"model_{name}": value for name, value in MODEL_WORK.snapshot().items()},
        }

    def _count(self, priming: Any) -> None:
        """Account for what a build cost: a full pass, an incremental build, or reuse."""
        self.embedded_chunk_count += int(priming.embedded_chunks)
        self.reused_chunk_count += int(priming.reused_chunks)
        self.embedding_call_count += int(priming.model_calls)
        if not priming.embedded:
            return
        if priming.reused_chunks == 0:
            self.passage_embedding_pass_count += 1
        else:
            self.incremental_build_count += 1

    def capability(self) -> HybridCapabilityReport:
        """Verify the exact configured semantic pack without building an index."""
        retriever = self._retriever_factory(self.workspace, self.model_root, self.index_spec)
        try:
            probe = getattr(retriever, "capability", None)
            if not callable(probe):
                raise ValueError("alternative_evidence.retrieval_capability_unavailable")
            return probe()
        finally:
            close = getattr(retriever, "close", None)
            if callable(close):
                close()

    def build(
        self,
        *,
        document_set: AlternativeEvidenceDocumentSet,
        built_at: datetime,
        anchor_for: AnchorResolver | None = None,
        admit: StorageAdmission | None = None,
        block_for: VectorBlockSource | None = None,
    ) -> AlternativeEvidenceRetrievalGeneration:
        """Build a retrieval generation for the verified document set."""
        self.publisher.verify(document_set)
        snapshot = self.publisher.library.read_snapshot(
            _snapshot_id(document_set.workspace_snapshot_id)
        )
        index = self._index_factory(self.workspace, self.model_root)
        # One build, one reader, at most one corpus embedding pass. The owner
        # hands back a reader primed with the vectors it proved -- embedded
        # now, or read from the commitment the caller sealed for the
        # generation this corpus already had -- so the first query verifies
        # the projection instead of embedding the corpus again. The
        # generation's identity is held from before the model runs until the
        # lease is registered, so no cleanup can take its index in between.
        index_id = index_identity_of_database(
            f".system/knowledge-indexes/knowledge-hybrid-v4/"
            f"{corpus_logical_hash(snapshot)}-{self.index_spec.logical_hash}.db"
        )
        assert index_id is not None
        self._hold_building(index_id)
        try:
            lease = index.rebuild_and_open(
                snapshot,
                self.index_spec,
                anchor_for=anchor_for,
                admit=admit,
                block_for=block_for,
            )
            manifest = lease.manifest
            if str(manifest.index_id) != index_id:
                lease.close()
                raise ValueError("alternative_evidence.retrieval_generation_identity_invalid")
            priming = lease.priming_facts
            if priming.embedded or priming.reused_chunks:
                self._count(priming)
            else:
                self.generation_reuse_count += 1
            try:
                generation = seal_contract(
                    AlternativeEvidenceRetrievalGeneration,
                    "generation_hash",
                    document_set_hash=document_set.document_set_hash,
                    workspace_snapshot_id=document_set.workspace_snapshot_id,
                    workspace_snapshot_hash=document_set.workspace_snapshot_hash,
                    retrieval_binding_hash=self.retrieval_binding_hash,
                    reading_depth=document_set.reading_depth,
                    corpus_hash=manifest.corpus_logical_hash,
                    index_id=manifest.index_id,
                    index_spec_hash=self.index_spec.logical_hash,
                    index_manifest_hash=manifest.logical_hash,
                    vector_payload_sha256=manifest.vector_commitment.payload_sha256,
                    document_count=manifest.document_count,
                    chunk_count=manifest.chunk_count,
                    built_at=built_at,
                )
            except Exception:
                lease.close()
                raise
            with self._lease_lock:
                prior = self._pending_generations.pop(generation.generation_hash, None)
                self._pending_generations[generation.generation_hash] = lease
            if prior is not None:
                prior.close()
        finally:
            self._release_building(index_id)
        return generation

    def verify_lineage(
        self,
        document_set: AlternativeEvidenceDocumentSet,
        generation: RetrievalGenerationRecord,
    ) -> None:
        """Verify generation lineage against its document set and index."""
        self.publisher.verify(document_set)
        if (
            generation.document_set_hash != document_set.document_set_hash
            or generation.workspace_snapshot_id != document_set.workspace_snapshot_id
            or generation.workspace_snapshot_hash != document_set.workspace_snapshot_hash
            or not is_current(
                RETRIEVAL_BINDING_ROLE,
                generation.retrieval_binding_hash,
                self.retrieval_binding_hash,
            )
            or generation.index_spec_hash != self.index_spec.logical_hash
            or generation.reading_depth is not document_set.reading_depth
        ):
            raise ValueError("alternative_evidence.retrieval_generation_lineage_invalid")

    def committed_index(self, document_set: AlternativeEvidenceDocumentSet) -> tuple[str, bool]:
        """Resolve the index identity committed for a document set.

        The identity of the index this document set's corpus resolves to
        under this spec, and whether that index is held on disk under it
        (present and not evicted). Presence, never proof: a reader proves the
        projection when it opens, and a caller that opens nothing (a
        selection reused whole from a sealed receipt) learns only that the
        build would find the index in place rather than restore it.
        """
        snapshot = self.publisher.library.read_snapshot(
            _snapshot_id(document_set.workspace_snapshot_id)
        )
        relative = committed_database_relative(
            corpus_logical_hash(snapshot), self.index_spec.logical_hash
        )
        index_id = index_identity_of_database(relative)
        assert index_id is not None
        return index_id, committed_index_held(self.workspace.root, relative)

    def release_without_session(
        self,
        *,
        document_set: AlternativeEvidenceDocumentSet,
        generation: RetrievalGenerationRecord,
    ) -> None:
        """Verify lineage and release a build without opening a session.

        Verify what a session would verify -- the document set's revisions
        and the generation's lineage -- and close the build's lease without
        opening one: the selection this generation was built for is proved
        by a sealed receipt, so no session will take the lease, and an index
        held by a lease nobody takes could never be evicted.
        """
        self.verify_lineage(document_set, generation)
        self.discard_pending(generation.generation_hash)
        self.generation_release_count += 1

    def open_session(
        self,
        *,
        document_set: AlternativeEvidenceDocumentSet,
        generation: RetrievalGenerationRecord,
        evidence_as_of: datetime,
        original_for: OriginalReader | None = None,
        pair_scores: PairScoreAdmission | None = None,
    ) -> AlternativeEvidenceRetrievalSession:
        """Open a leased read session over a verified generation.

        One session over a generation; with `original_for`, a session
        that can render table views from the retained originals; with
        `pair_scores`, a reader that serves the committed pair-score blocks
        the admission names and seals new ones under its storage
        admission (without it, the reader keeps its in-process cache only).
        """
        self.verify_lineage(document_set, generation)
        token = str(uuid4())
        with self._lease_lock:
            retriever = self._pending_generations.pop(generation.generation_hash, None)
            # The session's hold replaces the build's before the lock is
            # released: the generation is never unheld between the two.
            self._open_sessions[token] = str(generation.index_id)
        if retriever is not None:
            self.generation_handoff_count += 1
        else:
            self.independent_generation_reopen_count += 1
            retriever = self._retriever_factory(self.workspace, self.model_root, self.index_spec)
        if pair_scores is not None and hasattr(retriever, "pair_score_admission"):
            retriever.pair_score_admission = pair_scores
        return AlternativeEvidenceRetrievalSession(
            library=self.publisher.library,
            retriever=retriever,
            document_set=document_set,
            generation=generation,
            index_spec=self.index_spec,
            evidence_as_of=evidence_as_of,
            on_close=lambda: self._release_session(token),
            original_for=original_for,
        )


def _snapshot_id(value: str) -> Any:
    from uuid import UUID

    return UUID(value)


__all__ = ["AlternativeEvidenceRetrievalService"]
