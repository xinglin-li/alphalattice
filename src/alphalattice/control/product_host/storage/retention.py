"""Bounded, root-based retention planning for large desktop artifacts."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from hashlib import file_digest
from pathlib import Path
from typing import Any

from alphalattice.control.product_host.storage.contracts import (
    PhysicalAvailability,
    StorageEvictionPlan,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash

_HASH = re.compile(r"^[0-9a-f]{64}$")
EVIDENCE_INDEX_PREFIX = "runtime/evidence-knowledge/.system/knowledge-indexes/"
EVIDENCE_VECTOR_PREFIX = "runtime/evidence-knowledge/.system/knowledge-vectors/"
"""Where the evidence index payloads live, relative to the workspace: the one
evidence root an approved plan may release, through the index owner's own
eviction so a marker stays where the file was."""


def require_available_input(workspace: Path, relative: str) -> None:
    """Distinguish authorised eviction from an unexplained missing input."""
    root = workspace / "artifacts/storage-governance"
    for path in (root / "input-cleanup-started").glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if canonical_hash(payload) != path.stem:
            raise StorageRetentionError(
                "storage.retention_root_mismatch", "Invalid cleanup admission."
            )
        if relative in payload["targets"]:
            complete = root / "input-cleanup-completed" / f"{path.stem}.json"
            if complete.is_file() and complete.read_bytes() == path.read_bytes():
                raise StorageRetentionError(
                    "research_experiment.input_evicted_by_retention",
                    "Large input was released by an approved cleanup; metadata is retained.",
                )
            raise StorageRetentionError(
                "storage.cleanup_recovery_required",
                "Resume the admitted cleanup before using this input.",
            )


def file_content_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def pending_input_cleanups(workspace: Path) -> tuple[str, ...]:
    root = workspace / "artifacts/storage-governance"
    pending = []
    for path in sorted((root / "input-cleanup-started").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if canonical_hash(payload) != path.stem:
            raise StorageRetentionError(
                "storage.retention_root_mismatch", "Cleanup intent changed."
            )
        completed = root / "input-cleanup-completed" / path.name
        if not completed.exists():
            pending.append(path.stem)
        elif completed.read_bytes() != path.read_bytes():
            raise StorageRetentionError(
                "storage.retention_root_mismatch", "Cleanup receipt changed."
            )
    return tuple(pending)


def require_no_pending_cleanup(workspace: Path) -> None:
    if pending_input_cleanups(workspace):
        raise StorageRetentionError(
            "storage.cleanup_recovery_required", "Resume approved cleanup before new mutation."
        )


class StorageRetentionError(RuntimeError):
    """Carry a named public storage retention refusal with explicit detail."""

    def __init__(self, failure_code: str, detail: str) -> None:
        """Retain the explicit storage retention failure code and message.

        Args:
            failure_code: Named owner-supplied public failure code.
            detail: Explicit refusal explanation.
        """
        super().__init__(detail)
        self.failure_code = failure_code


class BoundedStorageRetentionOwner:
    """Resolve current/rollback roots and freeze exact unrooted file candidates."""

    def __init__(self, workspace: Path) -> None:
        """Resolve confined artifact and storage governance roots for one explicit workspace.

        Args:
            workspace: Caller-owned admitted workspace root.
        """
        self.workspace = workspace.resolve()
        self.artifact_root = self.workspace / "artifacts"
        self.governance_root = self.artifact_root / "storage-governance"

    def apply(self, plan: StorageEvictionPlan) -> int:
        """Apply a supplied retention plan to existing confined targets and record availability.

        Args:
            plan: Explicit admitted retention plan.

        Returns:
            Number of existing files removed.

        Raises:
            StorageRetentionError: A resolved eviction target escapes the workspace.
        """
        deleted = 0
        for relative in plan.evict_paths:
            target = (self.workspace / relative).resolve()
            try:
                target.relative_to(self.workspace)
            except ValueError as exc:
                raise StorageRetentionError(
                    "storage.retention_root_mismatch", "Eviction target escaped the workspace."
                ) from exc
            if target.is_file():
                target.unlink()
                deleted += 1
        self.publish_availability(plan)
        return deleted

    def input_plan(self, plan_hash: str) -> dict[str, Any]:
        """A published input plan's payload, proved by its identity."""
        if not _HASH.fullmatch(plan_hash):
            raise StorageRetentionError("storage.plan_invalid", "Invalid plan identity.")
        path = self.governance_root / "input-plans" / f"{plan_hash}.json"
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        if canonical_hash(payload) != plan_hash:
            raise StorageRetentionError("storage.plan_invalid", "Plan bytes changed.")
        return payload

    def publish_input_plan(self, values: dict[str, object]) -> str:
        """Publish an explicit exact input cleanup declaration by canonical identity.

        Args:
            values: Owner-resolved cleanup fields and reference/target bindings.

        Returns:
            Canonical retained input-plan identity.
        """
        identity = canonical_hash(values)
        self._publish("input-plans", identity, values)
        return str(identity)

    def apply_input_plan(
        self,
        plan_hash: str,
        *,
        references_hash: str,
        eligible_targets: dict[str, str],
        evidence_evictor: Callable[[str, str], int] | None = None,
    ) -> int:
        """Journal authorised intent before unlink; recovery repeats this exact plan.

        An evidence index target is released by `evidence_evictor(relative,
        plan_hash)` -- the index owner removes the file and leaves its marker --
        never by a bare unlink; without an evictor such a target is refused.
        A target the journalled plan already released is not asked to be
        eligible again: it is gone, and the resume releases only what remains.
        """
        if not _HASH.fullmatch(plan_hash):
            raise StorageRetentionError("storage.plan_invalid", "Invalid plan identity.")
        path = self.governance_root / "input-plans" / f"{plan_hash}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if canonical_hash(payload) != plan_hash or payload["references_hash"] != references_hash:
            raise StorageRetentionError(
                "storage.cleanup_plan_stale", "Storage references changed; preview again."
            )
        started = self.governance_root / "input-cleanup-started" / f"{plan_hash}.json"
        completed = self.governance_root / "input-cleanup-completed" / f"{plan_hash}.json"
        for name, digest in payload["targets"].items():
            released = started.exists() and not (self.workspace / name).exists()
            if eligible_targets.get(name) != digest and not released:
                raise StorageRetentionError(
                    "storage.cleanup_target_protected",
                    "A target is not eligible under current roots.",
                )
        if completed.is_file():
            if completed.read_bytes() != path.read_bytes():
                raise StorageRetentionError(
                    "storage.retention_root_mismatch", "Cleanup receipt differs."
                )
            return 0
        targets = []
        for relative, expected in payload["targets"].items():
            target = (self.workspace / relative).resolve()
            evidence_index = relative.startswith(EVIDENCE_INDEX_PREFIX) and relative.endswith(".db")
            evidence_vector = relative.startswith(EVIDENCE_VECTOR_PREFIX) and relative.endswith(
                (".f32", ".blocks.json")
            )
            if (evidence_index or evidence_vector) and evidence_evictor is None:
                raise StorageRetentionError(
                    "storage.cleanup_target_protected",
                    "An evidence object can be released only by its owner; none is admitted.",
                )
            allowed = (
                relative.startswith("research-inputs/")
                or (
                    relative.startswith(
                        (
                            "artifacts/feature-panel/chunks/",
                            "artifacts/data-operations/execution-outcomes/development/chunks/",
                        )
                    )
                    and relative.endswith(".parquet")
                )
                or evidence_index
                or evidence_vector
                or (
                    relative.startswith(
                        (
                            "artifacts/alpha-research/current/workspace-observation-history/",
                            "artifacts/data-operations/execution-outcomes/local-qa-preparation/",
                        )
                    )
                    and relative.endswith((".parquet", ".json"))
                )
            )
            if (
                not target.is_relative_to(self.workspace)
                or not allowed
                or target.relative_to(self.workspace).as_posix() != relative
            ):
                raise StorageRetentionError(
                    "storage.retention_root_mismatch", "Cleanup target is outside input storage."
                )
            if not target.exists() and not started.exists():
                raise StorageRetentionError(
                    "storage.cleanup_plan_stale", "Input disappeared before cleanup admission."
                )
            if target.exists() and file_content_hash(target) != expected:
                raise StorageRetentionError("storage.cleanup_plan_stale", "Input bytes changed.")
            targets.append(target)
        self._publish("input-cleanup-started", plan_hash, payload)
        deleted = 0
        for target in targets:
            if not target.exists():
                continue
            relative = target.relative_to(self.workspace).as_posix()
            if relative.startswith((EVIDENCE_INDEX_PREFIX, EVIDENCE_VECTOR_PREFIX)):
                assert evidence_evictor is not None
                evidence_evictor(relative, plan_hash)
            else:
                target.unlink()
            deleted += 1
        self._publish_availability(tuple(payload["targets"]), plan_hash)
        self._publish("input-cleanup-completed", plan_hash, payload)
        return deleted

    def publish_availability(self, plan: StorageEvictionPlan) -> str:
        """Publish physical availability records for an exact eviction plan.

        Args:
            plan: Explicit retention plan whose paths are recorded.

        Returns:
            Retained availability publication identity.
        """
        return self._publish_availability(plan.evict_paths, plan.plan_hash)

    def _publish_availability(self, paths: tuple[str, ...], plan_hash: str) -> str:
        entries = [
            {
                "relative_path": relative,
                "availability": PhysicalAvailability.EVICTED_BY_RETENTION.value,
                "eviction_plan_hash": plan_hash,
            }
            for relative in paths
        ]
        identity = {
            "kind": "StoragePhysicalAvailabilityIndex",
            "eviction_plan_hash": plan_hash,
            "entries": entries,
        }
        index_hash = str(canonical_hash(identity))
        self._publish("availability", index_hash, {**identity, "index_hash": index_hash})
        return index_hash

    def _publish(self, category: str, identity: str, payload: dict[str, object]) -> None:
        target = self.governance_root / category / f"{identity}.json"
        content = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        target.parent.mkdir(parents=True, exist_ok=True)
        staged = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        if target.exists() and target.read_bytes() != content:
            raise StorageRetentionError(
                "storage.retention_root_mismatch", "Storage governance identity was reused."
            )
        if not target.exists():
            staged.write_bytes(content)
            os.replace(staged, target)
        staged.unlink(missing_ok=True)


_DECLARED_ARTIFACT_REFERENCES = {
    "availability_closure_hash",
    "base_closure_hash",
    "chunk_hash",
    "recipe_hash",
    "row_receipt_assignment_hash",
    "sector_map_hash",
}


def _artifact_references(value: object) -> set[str]:
    """Return only declared artifact edges, never arbitrary provenance hashes."""
    found: set[str] = set()
    if isinstance(value, list):
        for item in value:
            found.update(_artifact_references(item))
    elif isinstance(value, dict):
        descriptor_hash = value.get("content_hash") if "uri" in value else None
        if isinstance(descriptor_hash, str) and _HASH.fullmatch(descriptor_hash):
            found.add(descriptor_hash)
        for key, item in value.items():
            if key in _DECLARED_ARTIFACT_REFERENCES:
                candidate = str(item)
                if _HASH.fullmatch(candidate):
                    found.add(candidate)
            if isinstance(item, (dict, list)):
                found.update(_artifact_references(item))
    return found


__all__ = ["BoundedStorageRetentionOwner", "StorageRetentionError"]
