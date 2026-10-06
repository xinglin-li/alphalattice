"""An inventory of the Alternative Evidence store's bytes and the references its records hold.

The store keeps five kinds of bytes, accounted apart: source blobs (evidence), sealed artifacts
and commitments (small), vector payloads (the committed output of the passage model: an index is
rebuilt from them without the model), index payloads (derived, rebuildable) and staging.

Which vector objects are referenced is decided only over a proved reference graph: every
committed generation's payload files are read and hashed to the digest its sealed record commits
to. A generation whose vectors cannot be proved -- its sidecar missing, unreadable or naming the
wrong blocks -- has an unknown reference set, and unknown reachability is not unreachability: no
vector object is reported unreferenced until the graph is proved again, and the refusal names the
generation and the recovery step.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from alphalattice.control.task_control.contracts import TaskLifecycle
from alphalattice.control.task_control.registry import DuckDbTaskControlRegistry
from alphalattice.kernel.knowledge.hybrid import committed_payload_references
from alphalattice.kernel.knowledge.hybrid_contracts import (
    HYBRID_INDEX_SCHEMA_VERSION,
    LEGACY_HYBRID_INDEX_SCHEMA_VERSION,
    VECTOR_PAYLOAD_ROOT,
)
from alphalattice.kernel.knowledge.retrieval_errors import KnowledgeRetrievalError

from ..retrieval.contracts import (
    AlternativeEvidenceRetrievalGeneration,
    RetrievalGenerationRecord,
    WholeFilingsGeneration,
    parse_retrieval_generation,
)
from ..runtime.task_adapter import TASK_KIND
from ..sources.contracts import (
    SOURCE_COMMIT_CATEGORY,
    SOURCE_OBJECT_ROOT,
    AcquiredEvidenceSourceCommit,
    AcquiredEvidenceSourceReferenceSet,
    parse_source_set,
)

INDEX_ROOT = ".system/knowledge-indexes"
VECTOR_REFERENCES_UNPROVED = "storage.evidence_vector_references_unproved"
"""The vector cleanup refusal: a sealed generation's vectors could not be proved, so which vector
objects are required is unknown."""
NO_PAYLOAD_COMMITMENT = "NOT_COMMITTED"
"""A legacy generation commits to no payload: nothing to prove, nothing held."""


@dataclass(frozen=True)
class EvidenceStorageBinding:
    """Where the evidence bytes live and who can act on an index for the owner."""

    knowledge_root: Path
    """The evidence Workspace: `knowledge/blobs`, `.system/knowledge-indexes`,
    `.system/knowledge-vectors`, `.system/staging`."""
    artifact_root: Path
    """The content-addressed store of sealed evidence artifacts."""
    active_index_ids: Callable[[], frozenset[str]] = lambda: frozenset()
    """Generations this process holds open (a built index waiting for its session)."""
    evict: Callable[[str, str, datetime], int] | None = None
    """`(database_relative, plan_hash, evicted_at) -> bytes released`: the index
    owner removes the file and leaves its marker. Without it no index is a target."""
    rebuild: Callable[[str], dict[str, object]] | None = None
    """`(index_id) -> facts`: restore one evicted generation from its commitments."""


def _tree_bytes(root: Path, *, suffixes: tuple[str, ...] | None = None) -> int:
    if not root.is_dir():
        return 0
    total = 0
    for directory, _names, files in os.walk(root):
        for name in files:
            if suffixes is None or name.endswith(suffixes):
                total += os.stat(os.path.join(directory, name)).st_size
    return total


class EvidenceStorageInventory:
    """A fresh inventory of the evidence store's bytes and generation references."""

    def __init__(
        self,
        *,
        workspace: Path,
        binding: EvidenceStorageBinding,
        registry: DuckDbTaskControlRegistry,
        pinned_index_ids: frozenset[str],
    ) -> None:
        """Bind the workspace, the store, the Task registry and the pinned indexes.

        Args:
            workspace: The product workspace the store is reported relative to.
            binding: Where the evidence bytes live.
            registry: The Task registry, read for the Tasks that built each generation.
            pinned_index_ids: The indexes a person pinned.
        """
        self.workspace = workspace.resolve()
        self.binding = binding
        self.registry = registry
        self.pinned = pinned_index_ids
        self._proofs: dict[tuple[str, str], tuple[str, ...] | KnowledgeRetrievalError] = {}

    # ---------------------------------------------------------------- records

    def generation_records(self) -> tuple[RetrievalGenerationRecord, ...]:
        """Return every generation record that names an index.

        A unit delivered whole (W4) built none, so there is nothing of it to account or evict.

        Returns:
            The records, by file name.

        Raises:
            ValueError: `storage.evidence_generation_record_misfiled` for a record stored under
                another hash.
        """
        directory = self.binding.artifact_root / "retrieval-generations"
        if not directory.is_dir():
            return ()
        records = []
        for path in sorted(directory.glob("*.json")):
            value = parse_retrieval_generation(json.loads(path.read_bytes()))
            if value.generation_hash != path.stem:
                raise ValueError("storage.evidence_generation_record_misfiled")
            if not isinstance(value, WholeFilingsGeneration):
                records.append(value)
        return tuple(records)

    @staticmethod
    def database_relative(record: RetrievalGenerationRecord) -> str:
        """Return the index file a generation record names, relative to the evidence Workspace.

        Args:
            record: The generation record.

        Returns:
            The index database's relative path.
        """
        if isinstance(record, AlternativeEvidenceRetrievalGeneration):
            return (
                f"{INDEX_ROOT}/{HYBRID_INDEX_SCHEMA_VERSION}/"
                f"{record.corpus_hash}-{record.index_spec_hash}.db"
            )
        return (
            f"{INDEX_ROOT}/{LEGACY_HYBRID_INDEX_SCHEMA_VERSION}/"
            f"{record.workspace_snapshot_hash}-{record.index_spec_hash}.db"
        )

    # ------------------------------------------------------------ references

    def payload_references(
        self, record: AlternativeEvidenceRetrievalGeneration
    ) -> tuple[str, ...] | KnowledgeRetrievalError:
        """Return the vector objects one committed generation holds, proved, or the refusal.

        Proved once per generation for the life of this inventory, which is one plan or one
        readback.

        Args:
            record: The committed generation.

        Returns:
            The vector objects' paths, relative to the evidence Workspace, or the kernel's
            refusal.
        """
        pair = (record.index_manifest_hash, record.vector_payload_sha256)
        if pair not in self._proofs:
            try:
                self._proofs[pair] = committed_payload_references(
                    self.binding.knowledge_root,
                    manifest_logical_hash=record.index_manifest_hash,
                    payload_sha256=record.vector_payload_sha256,
                )
            except KnowledgeRetrievalError as refused:
                self._proofs[pair] = refused
        return self._proofs[pair]

    def vector_reference_graph(self) -> dict[str, object]:
        """Say whether every committed generation's vectors are proved.

        Returns:
            `PROVED`, or `UNPROVED` with each unproved generation, why, and the step that
            proves it again.
        """
        unproved = []
        seen: set[str] = set()
        for record in self.generation_records():
            if not isinstance(record, AlternativeEvidenceRetrievalGeneration):
                continue
            proof = self.payload_references(record)
            if not isinstance(proof, KnowledgeRetrievalError) or record.index_id in seen:
                continue
            seen.add(record.index_id)
            unproved.append(
                {
                    "index_id": record.index_id,
                    "failure_code": proof.failure.code,
                    "detail": str(proof),
                    "recovery": (
                        "restore the generation's vector objects and sidecar from a "
                        "backup, or rebuild the generation explicitly: the rebuild "
                        "composes the committed payload again (blocks other sealed "
                        "generations hold, the pinned model for the rest), requires "
                        "the committed digest, and replaces the damaged sidecar or "
                        "block; vector cleanup is refused until every sealed "
                        "generation's vectors are proved"
                    ),
                }
            )
        return {"status": "UNPROVED" if unproved else "PROVED", "unproved": unproved}

    def vector_cleanup_refusal(self) -> dict[str, object] | None:
        """Return the typed refusal of vector cleanup while the graph is unproved.

        Returns:
            The refusal, or None when every generation's vectors are proved.
        """
        graph = self.vector_reference_graph()
        if graph["status"] == "PROVED":
            return None
        return {
            "failure_code": VECTOR_REFERENCES_UNPROVED,
            "detail": (
                "a sealed generation's vectors could not be proved, so which vector "
                "objects are required is unknown; no vector object is released"
            ),
            "generations": graph["unproved"],
        }

    # ------------------------------------------------------------ protections

    def generation_readers(self) -> dict[str, tuple[tuple[TaskLifecycle, dict[str, object]], ...]]:
        """Return every evidence Task that built or sealed a generation: the index's readers.

        Returns:
            By generation hash, each Task's lifecycle and its reader facts (task, goal,
            lifecycle, unit).
        """
        found: dict[str, list[tuple[TaskLifecycle, dict[str, object]]]] = {}
        for task in self.registry.tasks():
            if task.task_kind != TASK_KIND:
                continue
            for item in self.registry.work_items(task.task_id):
                # One request's stage, or one unit's of a coverage run
                # (`u03_build_retrieval_generation`): both build a generation
                # the rest of that Task still reads.
                if not item.stage_id.endswith("build_retrieval_generation"):
                    continue
                prefix = item.stage_id.split("_", 1)[0]
                unit = prefix if prefix[:1] == "u" and prefix[1:].isdigit() else None
                for evidence in item.evidence:
                    if evidence.evidence_kind == "alternative_evidence_retrieval_generation":
                        found.setdefault(str(evidence.content_hash), []).append(
                            (
                                task.lifecycle,
                                {
                                    "task_id": str(task.task_id),
                                    "goal_kind": task.goal.goal_kind,
                                    "lifecycle": task.lifecycle.value,
                                    "unit_id": unit,
                                },
                            )
                        )
        return {key: tuple(value) for key, value in found.items()}

    # -------------------------------------------------------------- inventory

    def unreferenced(self) -> dict[str, object]:
        """Report the published index, vector and source objects no sealed record names.

        An index without a record is a build that died between placing the index and sealing
        its record, or a record deleted by hand; a vector object no proved generation composes
        is a build that died before sealing, or a block every generation that composed it has
        stopped naming; a source object no source set references is an acquisition that died
        before its set was sealed. None is a generation or a set; all are reported. While any
        generation's vectors are unproved the vector reference graph is unknown, no vector
        object is reported as unreferenced, and `vector_reference_graph` says which and why.

        Returns:
            The unreferenced objects' paths and bytes by kind, with the vector reference graph.
        """
        indexes = self._unreferenced_indexes()
        payloads = self._unreferenced_vector_objects()
        sources = self._unreferenced_source_objects()
        return {
            "index_paths": [self._workspace_relative(path) for path in indexes],
            "index_bytes": sum(path.stat().st_size for path in indexes),
            "payload_paths": [self._workspace_relative(path) for path in payloads],
            "payload_bytes": sum(path.stat().st_size for path in payloads),
            "source_object_paths": [self._workspace_relative(path) for path in sources],
            "source_object_bytes": sum(path.stat().st_size for path in sources),
            "vector_reference_graph": self.vector_reference_graph(),
        }

    def _unreferenced_source_objects(self) -> list[Path]:
        """Source objects no sealed source set and no per-document commit
        names. A body committed on its own -- kept before the request's set
        existed, or kept by a request that then failed or was cancelled --
        is referenced by its commit: retained for the next attempt, not an
        acquisition that died."""

        root = self.binding.knowledge_root
        objects_root = root / Path(*SOURCE_OBJECT_ROOT.split("/"))
        if not objects_root.is_dir():
            return []
        referenced: set[str] = set()
        directory = self.binding.artifact_root / "source-document-sets"
        if directory.is_dir():
            for path in sorted(directory.glob("*.json")):
                value = parse_source_set(json.loads(path.read_bytes()))
                if isinstance(value, AcquiredEvidenceSourceReferenceSet):
                    referenced.update(document.content_sha256 for document in value.documents)
        commits = self.binding.artifact_root / SOURCE_COMMIT_CATEGORY
        if commits.is_dir():
            for path in sorted(commits.glob("*.json")):
                commit = AcquiredEvidenceSourceCommit.model_validate_json(path.read_bytes())
                referenced.add(commit.reference.content_sha256)
        return [
            path
            for path in sorted(objects_root.iterdir())
            if path.is_file() and path.name not in referenced
        ]

    def _unreferenced_indexes(self) -> list[Path]:
        root = self.binding.knowledge_root
        referenced_databases = {
            root / Path(*self.database_relative(record).split("/"))
            for record in self.generation_records()
        }
        return [
            path
            for path in sorted((root / Path(*INDEX_ROOT.split("/"))).rglob("*.db"))
            if path not in referenced_databases
        ]

    def _unreferenced_vector_objects(self) -> list[Path]:
        """Vector objects no proved generation composes; none while the graph is
        unknown, because an object outside an unknown set cannot be told from
        one inside it."""

        root = self.binding.knowledge_root
        referenced: set[Path] = set()
        for record in self.generation_records():
            if not isinstance(record, AlternativeEvidenceRetrievalGeneration):
                continue
            proof = self.payload_references(record)
            if isinstance(proof, KnowledgeRetrievalError):
                return []
            referenced.update(root / Path(*relative.split("/")) for relative in proof)
        vector_root = root / Path(*VECTOR_PAYLOAD_ROOT.split("/"))
        entries = sorted(vector_root.iterdir()) if vector_root.is_dir() else []
        return [path for path in entries if path.is_file() and path not in referenced]

    def summary(self) -> dict[str, object]:
        """Count the store's bytes by kind, with the workspace's Task state beside them.

        Returns:
            The bytes of source blobs and objects, sealed artifacts, vector payloads (and how
            many objects), index payloads, staging and Task state.
        """
        root = self.binding.knowledge_root
        runtime = self.workspace / "runtime"
        task_state = 0
        if runtime.is_dir():
            task_state = sum(
                path.stat().st_size
                for path in runtime.iterdir()
                if path.is_file() and path.suffix in {".duckdb", ".sqlite"}
            )
        vector_root = root / Path(*VECTOR_PAYLOAD_ROOT.split("/"))
        return {
            "source_blob_bytes": _tree_bytes(root / "knowledge"),
            "source_object_bytes": _tree_bytes(root / Path(*SOURCE_OBJECT_ROOT.split("/"))),
            "sealed_artifact_bytes": _tree_bytes(self.binding.artifact_root),
            "vector_payload_bytes": _tree_bytes(vector_root),
            "vector_object_count": (
                sum(1 for path in vector_root.iterdir() if path.is_file())
                if vector_root.is_dir()
                else 0
            ),
            "index_payload_bytes": _tree_bytes(
                root / Path(*INDEX_ROOT.split("/")), suffixes=(".db",)
            ),
            "staging_bytes": _tree_bytes(root / ".system" / "staging"),
            "task_state_bytes": task_state,
        }

    def inside_workspace(self) -> bool:
        """Say whether the evidence store lies inside the product workspace.

        Returns:
            True when the knowledge root is under the workspace.
        """
        return self.binding.knowledge_root.resolve().is_relative_to(self.workspace)

    def _workspace_relative(self, path: Path) -> str:
        resolved = path.resolve()
        if resolved.is_relative_to(self.workspace):
            return resolved.relative_to(self.workspace).as_posix()
        return resolved.as_posix()


__all__ = [
    "INDEX_ROOT",
    "NO_PAYLOAD_COMMITMENT",
    "VECTOR_REFERENCES_UNPROVED",
    "EvidenceStorageBinding",
    "EvidenceStorageInventory",
]
