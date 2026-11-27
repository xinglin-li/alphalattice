"""Publish canonical Alternative Evidence through the tracked Workspace owner."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from uuid import UUID

from alphalattice.capabilities.domain_contracts.identity import canonical_hash
from alphalattice.capabilities.domain_contracts.persistence import fsync_directory, write_new
from alphalattice.control.workspace_runtime.core import Workspace
from alphalattice.control.workspace_runtime.paths import make_confined_parents, resolve_confined
from alphalattice.foundation.data_layer import history
from alphalattice.kernel.identity import blake3_digest, blake3_hex

from ..contracts import AlternativeEvidenceReadingDepth, seal_contract
from ..sources.contracts import (
    AcquiredEvidenceDocument,
    AcquiredEvidenceDocumentReference,
)
from ..storage import world, world_reads
from .contracts import (
    AlternativeEvidenceCanonicalDocument,
    AlternativeEvidenceDocumentReference,
    AlternativeEvidenceDocumentRejection,
    AlternativeEvidenceDocumentSet,
)
from .library import EvidenceLibrary


class AlternativeEvidenceDocumentPublisher:
    """Thin domain adapter; Workspace owns bytes, revisions, and snapshots."""

    def __init__(self, workspace_root: Path) -> None:
        """Bind document publication to one confined workspace."""
        self.workspace = Workspace.create(workspace_root)
        self._library: EvidenceLibrary | None = None

    @property
    def library(self) -> EvidenceLibrary:
        """The canonical documents in the Host's store, held from the first use; refused by name
        outside a Host whose history records, which alone holds the store's writer."""
        if self._library is None:
            opened = history.opened_log()
            if opened is None:
                raise ValueError("alternative_evidence.host_history_unavailable")
            self._library = EvidenceLibrary(*opened)
        return self._library

    # ------------------------------------------------------- source objects

    def publish_source_object(
        self,
        document: AcquiredEvidenceDocument,
        *,
        admit: Callable[[int], None] | None = None,
        world_root: Path | None = None,
    ) -> AcquiredEvidenceDocumentReference:
        """Keep one document's original bytes once, by content address.

        The object is immutable: an address already held must hold these
        exact bytes (`retrieval.source_integrity` otherwise) and adds nothing;
        a new address is admitted by the caller's budget, written to a staging
        sibling and renamed into place, so an interrupted write leaves a
        staging remnant and never a partial object at a content address
        that a later attempt would have to read as a tamper. Original bytes
        and canonical text are different assets -- a normalization is not
        reversible -- so both are kept, each once.
        """
        reference = self.source_reference_of(document)
        relative = reference.content_object_path
        with self.workspace.lock():
            target = make_confined_parents(self.workspace.root, relative)
            target = resolve_confined(self.workspace.root, relative)
            if target.exists():
                try:
                    if not target.is_file():
                        raise ValueError("not a file")
                    world_reads.workspace_body(target, reference, root=world_root)
                except ValueError as error:
                    raise ValueError("alternative_evidence.source_object_integrity") from error
            else:
                if admit is not None:
                    admit(len(document.content))
                staged = target.with_name(f".{target.name}.{os.getpid()}.staging")
                staged.unlink(missing_ok=True)
                stored = (
                    document.content
                    if world_root is None
                    else world.keep_original(world_root, document.content)
                )
                write_new(staged, stored)
                os.replace(staged, target)
                fsync_directory(target.parent)
        return reference

    def source_reference_of(
        self, document: AcquiredEvidenceDocument
    ) -> AcquiredEvidenceDocumentReference:
        """Compute the source reference without publishing its bytes.

        The reference `publish_source_object` would return for this document,
        without keeping anything: the content address is the bytes' own.
        """
        return AcquiredEvidenceDocumentReference.of(
            document, content_digest=blake3_digest(document.content)
        )

    def holds_source_object(
        self, reference: AcquiredEvidenceDocumentReference, *, as_of: datetime | None
    ) -> bool:
        """Whether the reference's bytes are held: as world knew its document at `as_of` when world
        serves the evidence reads (`world_reads.serving`) and holds that body, else at this
        workspace's content address. `as_of` None asks this workspace's own store, as a writer
        sizing what it adds does."""
        port = None if as_of is None else world_reads.serving()
        if port is not None and as_of is not None:
            served = world_reads.body(port, reference, as_of)
            if served.held:
                return served.content is not None
        path = resolve_confined(self.workspace.root, reference.content_object_path)
        if port is None or as_of is None or not path.is_file():
            return bool(path.is_file())
        content = world_reads.workspace_body(path, reference, root=port.root)
        return world_reads.known(port, content, as_of) is not False

    def require_known(self, content: bytes, *, as_of: datetime | None) -> None:
        """Refuse as missing a body world holds but did not know at `as_of`, however it is read:
        from this workspace's store, under another source's reference, or inline. Nothing when
        world does not serve the evidence reads, `as_of` is None, or world does not hold it."""
        port = None if as_of is None else world_reads.serving()
        knew = None if port is None or as_of is None else world_reads.known(port, content, as_of)
        if knew is False:
            raise ValueError("alternative_evidence.source_object_missing")

    def source_versions(
        self, references: Sequence[AcquiredEvidenceDocumentReference], *, as_of: datetime | None
    ) -> tuple[world_reads.Versions, ...] | None:
        """The references' candidate world versions at `as_of`, read in one statement, for
        `resolve_source_object` to take each from; None when world does not serve the reads."""
        port = None if as_of is None else world_reads.serving()
        return (
            None if port is None or as_of is None else world_reads.versions(port, references, as_of)
        )

    def resolve_source_object(
        self,
        reference: AcquiredEvidenceDocumentReference,
        *,
        as_of: datetime | None,
        versions: world_reads.Versions | None = None,
    ) -> AcquiredEvidenceDocument:
        """The document with its retained bytes, or a refusal that names what is wrong: read through
        world's `evidence_doc_asof` at `as_of` when world serves the evidence reads, a body world
        did not know then refused as missing; else from this workspace's store, as `as_of` None
        always reads, for a store being imported. `versions` is its `source_versions` entry."""
        port = None if as_of is None else world_reads.serving()
        if port is not None and as_of is not None:
            served = world_reads.body(port, reference, as_of, versions)
            if served.held:
                if served.content is None:
                    raise ValueError("alternative_evidence.source_object_missing")
                return reference.resolve(served.content)
        path = resolve_confined(self.workspace.root, reference.content_object_path)
        if not path.is_file():
            raise ValueError("alternative_evidence.source_object_missing")
        content = world_reads.workspace_body(
            path, reference, root=None if port is None else port.root
        )
        self.require_known(content, as_of=as_of)
        try:
            return reference.resolve(content)
        except ValueError as error:
            raise ValueError("alternative_evidence.source_object_tampered") from error

    def publish(
        self,
        *,
        request_hash: str,
        source_snapshot_hash: str,
        canonicalization_binding_hash: str,
        reading_depth: AlternativeEvidenceReadingDepth,
        documents: tuple[AlternativeEvidenceCanonicalDocument, ...],
        rejections: tuple[AlternativeEvidenceDocumentRejection, ...],
        published_at: datetime,
        admit: Callable[[int], None] | None = None,
    ) -> AlternativeEvidenceDocumentSet:
        """Publish a canonical document set and its source references."""
        if not documents:
            raise ValueError("alternative_evidence.document_set_empty")
        if admit is not None:
            # The bytes this publication may add to the Host's log: each canonical body and one
            # small manifest per revision (a revision the store holds already adds nothing).
            admit(sum(value.byte_count for value in documents) + 4096 * len(documents))
        revisions = []
        references = []
        for value in sorted(documents, key=lambda item: item.semantic_handle):
            document_id = _workspace_document_id(value)
            receipt = self.library.publish_revision(
                document_id=document_id,
                title=value.title[:160],
                # Dated by the evidence, not by the task that happened to read it: the document id
                # is content-addressed, so a second book asking about a filing already published
                # republishes the same revision, byte for byte.
                source={
                    "source_id": f"ae.{value.source_name.casefold().replace('_', '-')}",
                    "source_logical_hash": value.canonical_content_hash,
                    "license_id": (
                        "SEC-PUBLIC-FILING"
                        if value.source_name == "SEC_EDGAR"
                        else "USER-PROVIDED-LOCAL-RESEARCH"
                    ),
                    "available_at": value.available_at.isoformat(),
                },
                content=value.canonical_markdown.decode("utf-8"),
            )
            revisions.append(receipt)
            references.append(
                AlternativeEvidenceDocumentReference(
                    semantic_handle=value.semantic_handle,
                    entity_id=value.entity_id,
                    source_name=value.source_name,
                    source_right=value.source_right,
                    evidence_class=value.evidence_class,
                    document_type=value.document_type,
                    revision_label=value.revision,
                    title=value.title,
                    published_at=value.published_at,
                    accepted_at=value.accepted_at,
                    available_at=value.available_at,
                    immutable_source=value.immutable_source,
                    reading_depth=value.reading_depth,
                    section_labels=value.section_labels,
                    workspace_document_id=document_id,
                    workspace_revision=1,
                    character_count=value.character_count,
                    byte_count=value.byte_count,
                    published_precision=value.published_precision,
                    report_period_end=value.report_period_end,
                )
            )
        ordered_revisions = tuple(
            sorted(revisions, key=lambda item: (item.document.document_id, item.revision))
        )
        snapshot_id = _stable_uuid(
            canonical_hash(
                {
                    "request_hash": request_hash,
                    "source_snapshot_hash": source_snapshot_hash,
                    "reading_depth": reading_depth,
                    "document_hashes": tuple(value.document_hash for value in documents),
                }
            )
        )
        snapshot = self.library.freeze_snapshot(
            snapshot_id=snapshot_id,
            created_at=published_at,
            revisions=ordered_revisions,
        )
        return seal_contract(
            AlternativeEvidenceDocumentSet,
            "document_set_hash",
            request_hash=request_hash,
            source_snapshot_hash=source_snapshot_hash,
            canonicalization_binding_hash=canonicalization_binding_hash,
            reading_depth=reading_depth,
            workspace_snapshot_id=str(snapshot.snapshot_id),
            workspace_snapshot_hash=snapshot.logical_hash,
            documents=tuple(sorted(references, key=lambda item: item.semantic_handle)),
            rejections=tuple(sorted(rejections, key=lambda item: item.semantic_handle)),
            published_at=published_at,
        )

    def verify(self, document_set: AlternativeEvidenceDocumentSet) -> None:
        """Verify a published document set and its retained content."""
        snapshot = self.library.read_snapshot(UUID(document_set.workspace_snapshot_id))
        if snapshot.logical_hash != document_set.workspace_snapshot_hash:
            raise ValueError("alternative_evidence.workspace_snapshot_tampered")
        revisions = {
            (item.document.document_id, item.revision): item for item in snapshot.revisions
        }
        expected = {
            (item.workspace_document_id, item.workspace_revision) for item in document_set.documents
        }
        if set(revisions) != expected or any(
            item.reading_depth is not document_set.reading_depth for item in document_set.documents
        ):
            raise ValueError("alternative_evidence.workspace_document_lineage_invalid")
        for reference in document_set.documents:
            revision, content = self.library.read_revision(
                reference.workspace_document_id,
                reference.workspace_revision,
            )
            if (
                revision
                != revisions[(reference.workspace_document_id, reference.workspace_revision)]
                or len(content) != reference.byte_count
                or len(content.decode("utf-8")) != reference.character_count
            ):
                raise ValueError("alternative_evidence.workspace_document_tampered")


def _workspace_document_id(document: AlternativeEvidenceCanonicalDocument) -> str:
    entity = document.entity_id.casefold().replace("_", "-")
    return f"ae.{entity}.{blake3_hex(document.document_hash)[:24]}"


def _stable_uuid(identity: str) -> UUID:
    return UUID(bytes=bytes.fromhex(blake3_hex(identity)[:32]), version=4)


__all__ = ["AlternativeEvidenceDocumentPublisher"]
