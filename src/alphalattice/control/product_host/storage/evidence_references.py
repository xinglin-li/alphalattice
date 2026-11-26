"""Project the evidence store's generations into the existing retention owner.

The Alternative Evidence store's inventory (its bytes by kind and the references its records hold)
is Evidence's own (`alternative_evidence.storage.inventory`). This owner decides from it which
index a plan may release, by a fresh reference inventory: a generation whose index an unfinished
Task still needs, a reader in this process holds open, or a person pinned is protected; any other
published index is a candidate. Deleting is journalled and marked by the Workspace's own index
owner, so a later open refuses by the plan's name and an explicit rebuild restores the same
generation from its payload. A vector object is a candidate only over a proved reference graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from alphalattice.control.product_host.storage.contracts import PhysicalAvailability
from alphalattice.control.product_host.storage.retention import file_content_hash
from alphalattice.control.task_control.contracts import TaskLifecycle
from alphalattice.evidence.alternative_evidence.retrieval.contracts import (
    AlternativeEvidenceRetrievalGeneration,
    RetrievalGenerationRecord,
)
from alphalattice.evidence.alternative_evidence.storage.inventory import (
    NO_PAYLOAD_COMMITMENT,
    EvidenceStorageInventory,
)
from alphalattice.kernel.knowledge.hybrid import index_identity_of_database
from alphalattice.kernel.knowledge.hybrid_contracts import (
    LEGACY_HYBRID_INDEX_SCHEMA_VERSION,
    VECTOR_PAYLOAD_ROOT,
)
from alphalattice.kernel.knowledge.retrieval_errors import KnowledgeRetrievalError

_IN_FLIGHT = frozenset(
    {
        TaskLifecycle.QUEUED,
        TaskLifecycle.RUNNING,
        TaskLifecycle.DEFERRED,
        TaskLifecycle.REVIEW_PENDING,
        TaskLifecycle.CANCEL_REQUESTED,
        TaskLifecycle.RECOVERY_REQUIRED,
        # Not BLOCKED: Task Control treats a blocked evidence Task as terminal
        # (never resumed, never cancelled), so it owes no generation.
    }
)


@dataclass(frozen=True)
class EvidenceIndexRow:
    """Retain exact index/payload availability, bytes and reader protections.

    Retain index/payload physical availability, verified byte counts and explicit reader
    protections.
    """

    index_id: str
    generation_format: str
    corpus_hash: str | None
    relative_path: str
    """Relative to the product workspace."""
    database_bytes: int
    availability: PhysicalAvailability
    payload_path: str | None
    payload_bytes: int
    payload_available: bool
    payload_proof: str
    """`PROVED` when the committed payload was read and hashed to its digest;
    otherwise the kernel's failure code, or `NOT_COMMITTED` for a legacy record."""
    record_count: int
    protected_by: tuple[str, ...]
    readers: tuple[dict[str, object], ...] = ()
    """The evidence Tasks that built or sealed one of this index's
    generations (task, goal, lifecycle, unit): who a cleanup affects. Shown
    to a person; not part of the references a storage plan binds."""

    def body(self) -> dict[str, object]:
        """Project index availability and eligibility under retained protections.

        Project index metadata and eligibility from physical availability and retained protections.

        Returns:
            JSON-compatible index row with explicit payload proof, readers and eligibility.
        """
        return {
            "index_id": self.index_id,
            "generation_format": self.generation_format,
            "corpus_hash": self.corpus_hash,
            "relative_path": self.relative_path,
            "database_bytes": self.database_bytes,
            "availability": self.availability.value,
            "payload_path": self.payload_path,
            "payload_bytes": self.payload_bytes,
            "payload_available": self.payload_available,
            "payload_proof": self.payload_proof,
            "record_count": self.record_count,
            "protected_by": list(self.protected_by),
            "eligible": self.availability is PhysicalAvailability.AVAILABLE
            and not self.protected_by,
            "readers": list(self.readers),
        }


class EvidenceStorageReferences(EvidenceStorageInventory):
    """A fresh inventory of evidence bytes and generation references for one plan."""

    def rows(self) -> tuple[EvidenceIndexRow, ...]:
        """Resolve retained index generations, proven payload bytes and exact active/pinned readers.

        Shared vector payload objects are sized once within their proved composition. Missing index
        bytes and recorded retention eviction remain distinguishable.

        Returns:
            Ordered physical index rows with availability/refusal proof and active, in-flight or
            user-pin protections.
        """
        records = self.generation_records()
        readers = self.generation_readers()
        in_flight = frozenset(
            generation
            for generation, found in readers.items()
            if any(lifecycle in _IN_FLIGHT for lifecycle, _reader in found)
        )
        active = self.binding.active_index_ids()
        by_index: dict[str, list[RetrievalGenerationRecord]] = {}
        for record in records:
            by_index.setdefault(record.index_id, []).append(record)
        rows = []
        root = self.binding.knowledge_root
        for index_id, group in sorted(by_index.items()):
            record = group[0]
            relative = self.database_relative(record)
            database = root / Path(*relative.split("/"))
            if database.is_file():
                availability = PhysicalAvailability.AVAILABLE
                database_bytes = database.stat().st_size
            elif database.with_name(database.name + ".evicted.json").is_file():
                availability = PhysicalAvailability.EVICTED_BY_RETENTION
                database_bytes = 0
            else:
                availability = PhysicalAvailability.MISSING_OR_TAMPERED
                database_bytes = 0
            payload_path: str | None = None
            payload_bytes = 0
            payload_available = False
            payload_proof = NO_PAYLOAD_COMMITMENT
            corpus_hash: str | None = None
            if isinstance(record, AlternativeEvidenceRetrievalGeneration):
                corpus_hash = record.corpus_hash
                payload_path = f"{VECTOR_PAYLOAD_ROOT}/{record.vector_payload_sha256}.f32"
                # One file at the digest's address, or the sidecar and the
                # blocks of the generation's composition, proved against the
                # committed digest; sized by the proved objects, which count
                # once (two revisions with the same vectors name one block).
                proof = self.payload_references(record)
                if isinstance(proof, KnowledgeRetrievalError):
                    payload_proof = proof.failure.code
                else:
                    payload_proof = "PROVED"
                    payload_available = True
                    payload_bytes = sum(
                        (root / Path(*relative.split("/"))).stat().st_size for relative in proof
                    )
            protected = []
            if any(value.generation_hash in in_flight for value in group):
                protected.append("IN_FLIGHT_RECOVERY")
            if index_id in active:
                protected.append("ACTIVE_LEASE")
            if index_id in self.pinned:
                protected.append("USER_PINNED")
            rows.append(
                EvidenceIndexRow(
                    index_id=index_id,
                    generation_format=(
                        record.generation_format
                        if isinstance(record, AlternativeEvidenceRetrievalGeneration)
                        else LEGACY_HYBRID_INDEX_SCHEMA_VERSION
                    ),
                    corpus_hash=corpus_hash,
                    relative_path=self._workspace_relative(database),
                    database_bytes=database_bytes,
                    availability=availability,
                    payload_path=payload_path,
                    payload_bytes=payload_bytes,
                    payload_available=payload_available,
                    payload_proof=payload_proof,
                    record_count=len(group),
                    protected_by=tuple(protected),
                    readers=tuple(
                        reader
                        for _key, reader in sorted(
                            {
                                (str(reader["task_id"]), str(reader["unit_id"] or "")): reader
                                for value in group
                                for _lifecycle, reader in readers.get(value.generation_hash, ())
                            }.items()
                        )
                    ),
                )
            )
        return tuple(rows)

    def targets(self) -> dict[str, str]:
        """Eligible index files, each bound to the bytes read when it was planned.

        Only an evidence store inside the managed workspace has targets: bytes
        outside it are reported, never released, by this owner. An index no
        record names is a candidate unless this process holds its generation
        -- a build that placed its index and has not sealed its record yet is
        work in progress, not abandoned storage.
        """
        if self.binding.evict is None or not self.inside_workspace():
            return {}
        active = self.binding.active_index_ids()
        targets: dict[str, str] = {}
        for row in self.rows():
            if row.availability is not PhysicalAvailability.AVAILABLE or row.protected_by:
                continue
            path = self.workspace / Path(*row.relative_path.split("/"))
            targets[row.relative_path] = file_content_hash(path)
        root = self.binding.knowledge_root.resolve()
        for path in self._unreferenced_indexes():
            identity = index_identity_of_database(path.resolve().relative_to(root).as_posix())
            if identity is not None and identity in active:
                continue
            targets[self._workspace_relative(path)] = file_content_hash(path)
        # A vector object no proved generation composes is a candidate, but
        # only while nothing is being built -- a build that has read a block
        # and not yet sealed its record would otherwise lose it -- and only
        # over a proved graph: `_unreferenced_vector_objects` names nothing
        # while any generation's vectors are unproved.
        if not active:
            for path in self._unreferenced_vector_objects():
                targets[self._workspace_relative(path)] = file_content_hash(path)
        return targets

    def evidence(self) -> dict[str, object]:
        """What the references hash covers: the references, not the bytes.

        A new record, pin, open reader or unfinished Task changes it and stales
        a plan; the physical state of an index does not, so a journalled plan
        still matches its references while it is being applied and can resume.
        """
        return {
            "rows": [
                {
                    "index_id": row.index_id,
                    "generation_format": row.generation_format,
                    "relative_path": row.relative_path,
                    "record_count": row.record_count,
                    "protected_by": list(row.protected_by),
                }
                for row in self.rows()
            ],
            "pins": sorted(self.pinned),
            "active": sorted(self.binding.active_index_ids()),
        }


__all__ = [
    "EvidenceIndexRow",
    "EvidenceStorageReferences",
]
