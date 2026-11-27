"""An inventory of the Alternative Evidence store's bytes and its unreferenced source objects.

The store keeps its bytes by kind: source blobs and objects (evidence), sealed artifacts and
commitments (small), staging, and the vector and index payloads earlier builds left; the
Host's retrieval store keeps no per-generation files, so those are counted, never planned
for release. A source object no sealed source set names is reported, never released.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from alphalattice.kernel.identity import blake3_filename

from ..sources.contracts import (
    SOURCE_COMMIT_CATEGORY,
    SOURCE_OBJECT_ROOT,
    AcquiredEvidenceSourceCommit,
    AcquiredEvidenceSourceReferenceSet,
    parse_source_set,
)

INDEX_ROOT = ".system/knowledge-indexes"
VECTOR_PAYLOAD_ROOT = ".system/knowledge-vectors"
"""Where earlier builds left their per-generation index and vector files: counted by kind."""


@dataclass(frozen=True)
class EvidenceStorageBinding:
    """Where the evidence bytes live."""

    knowledge_root: Path
    """The evidence Workspace: `knowledge/blobs`, `.system/knowledge-indexes`,
    `.system/knowledge-vectors`, `.system/staging`."""
    artifact_root: Path
    """The content-addressed store of sealed evidence artifacts."""


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
    """A fresh inventory of the evidence store's bytes and unreferenced source objects."""

    def __init__(self, *, workspace: Path, binding: EvidenceStorageBinding) -> None:
        """Bind the workspace the store is reported relative to, and the store.

        Args:
            workspace: The product workspace.
            binding: Where the evidence bytes live.
        """
        self.workspace = workspace.resolve()
        self.binding = binding

    def unreferenced(self) -> dict[str, object]:
        """Report the source objects no sealed source set names: an acquisition that died
        before its set was sealed. They are reported, never released.

        Returns:
            Their paths and bytes.
        """
        sources = self._unreferenced_source_objects()
        return {
            "source_object_paths": [self._workspace_relative(path) for path in sources],
            "source_object_bytes": sum(path.stat().st_size for path in sources),
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
                    referenced.update(
                        blake3_filename(doc.content_digest) for doc in value.documents
                    )
        commits = self.binding.artifact_root / SOURCE_COMMIT_CATEGORY
        if commits.is_dir():
            for path in sorted(commits.glob("*.json")):
                commit = AcquiredEvidenceSourceCommit.model_validate_json(path.read_bytes())
                referenced.add(blake3_filename(commit.reference.content_digest))
        return [
            path
            for path in sorted(objects_root.iterdir())
            if path.is_file() and path.name not in referenced
        ]

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
    "VECTOR_PAYLOAD_ROOT",
    "EvidenceStorageBinding",
    "EvidenceStorageInventory",
]
