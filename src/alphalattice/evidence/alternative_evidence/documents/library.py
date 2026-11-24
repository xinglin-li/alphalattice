"""Canonical evidence revisions and snapshots, held in the Host's retrieval store.

Review searches and cites a filing's canonical markdown, a deterministic derivation of the original
bytes world holds. Each canonical document is a reference source of the store, its body kept in the
Host's log under the document's content-addressed id, so a citation names the same document id the
workspace library gave it; a snapshot is the store's snapshot of those revisions. Nothing here is a
workspace file.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from alphalattice.capabilities.domain_contracts.domain.serialization import sha256_hex
from alphalattice.capabilities.domain_contracts.identity import canonical_hash
from alphalattice.foundation.data_layer.knowledge import KnowledgeStore, SourceOrigin
from alphalattice.foundation.data_layer.log import HostLog


class EvidenceDocumentUnverifiable(ValueError):
    """A canonical revision whose body no longer hashes to what it was published with."""

    code = "alternative_evidence.workspace_document_tampered"

    def __init__(self) -> None:
        """Name the refusal by its code."""
        super().__init__(self.code)


@dataclass(frozen=True)
class EvidenceDocument:
    """A canonical document's identity in its workspace."""

    document_id: str


@dataclass(frozen=True)
class EvidenceRevision:
    """One published revision of a canonical document and the hash of its UTF-8 body."""

    document: EvidenceDocument
    revision: int
    content_sha256: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class EvidenceSnapshot:
    """A frozen set of revisions; `logical_hash` binds its id, time and every revision."""

    snapshot_id: UUID
    logical_hash: str
    revisions: tuple[EvidenceRevision, ...]


def _revision(metadata: dict[str, Any]) -> EvidenceRevision:
    return EvidenceRevision(
        EvidenceDocument(str(metadata["document"]["document_id"])),
        int(metadata["revision"]),
        str(metadata["content_sha256"]),
        metadata,
    )


class EvidenceLibrary:
    """A workspace's canonical evidence in the Host's store, through its one writer."""

    def __init__(self, log: HostLog, workspace_id: str) -> None:
        """Read and publish in `workspace_id` through `log`."""
        self.store, self.workspace_id = KnowledgeStore(log), workspace_id

    def publish_revision(
        self, *, document_id: str, title: str, source: dict[str, Any], content: str
    ) -> EvidenceRevision:
        """Publish one canonical revision once; an exact replay returns the same revision."""
        metadata = {
            "document": {"document_id": document_id, "title": title},
            "revision": 1,
            "source": source,
            "content_sha256": sha256_hex(content.encode()),
        }
        known_us = round(datetime.fromisoformat(str(source["available_at"])).timestamp() * 1e6)
        self.store.publish_revision(
            workspace_id=self.workspace_id,
            metadata=metadata,
            content=content,
            origin=SourceOrigin(document_id, "1", "reference", known_us),
        )
        return _revision(metadata)

    def freeze_snapshot(
        self, *, snapshot_id: UUID, created_at: datetime, revisions: tuple[EvidenceRevision, ...]
    ) -> EvidenceSnapshot:
        """Freeze `revisions` under `snapshot_id`; the same snapshot frozen again is returned."""
        metadata = {
            "snapshot_id": str(snapshot_id),
            "created_at": created_at.isoformat(),
            "revisions": [item.metadata for item in revisions],
        }
        self.store.publish_snapshot(workspace_id=self.workspace_id, metadata=metadata)
        return self.read_snapshot(snapshot_id)

    def read_snapshot(self, snapshot_id: UUID) -> EvidenceSnapshot:
        """The snapshot as frozen; a snapshot this workspace never froze is refused by name."""
        metadata = self.store.read_snapshot(self.workspace_id, str(snapshot_id))
        return EvidenceSnapshot(
            snapshot_id,
            canonical_hash(metadata),
            tuple(_revision(item) for item in metadata["revisions"]),
        )

    def read_revision(self, document_id: str, revision: int) -> tuple[EvidenceRevision, bytes]:
        """The revision and its body, verified against the hash it was published with."""
        published = self.store.read_revision(self.workspace_id, document_id, revision)
        content = (published.content or "").encode()
        held = _revision(published.metadata)
        if published.content is None or sha256_hex(content) != held.content_sha256:
            raise EvidenceDocumentUnverifiable
        return held, content
