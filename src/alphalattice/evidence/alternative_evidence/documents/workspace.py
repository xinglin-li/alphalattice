"""Publish canonical Alternative Evidence through the tracked Workspace owner."""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from uuid import UUID

from alphalattice.control.workspace_runtime.core import Workspace
from alphalattice.control.workspace_runtime.paths import make_confined_parents, resolve_confined
from alphalattice.kernel.knowledge.retrieval import WorkspaceKnowledgeLibrary
from alphalattice.kernel.knowledge.retrieval_contracts import (
    KnowledgeAccessClass,
    KnowledgeMediaType,
    KnowledgeNamespace,
    KnowledgeSourceCommitment,
    WorkspaceKnowledgeDocument,
)
from alphalattice.kernel.shared_kernel.domain.serialization import sha256_hex
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.persistence import fsync_directory, write_new

from ..contracts import AlternativeEvidenceReadingDepth, seal_contract
from ..sources.contracts import (
    AcquiredEvidenceDocument,
    AcquiredEvidenceDocumentReference,
    source_object_relative,
)
from .contracts import (
    AlternativeEvidenceCanonicalDocument,
    AlternativeEvidenceDocumentReference,
    AlternativeEvidenceDocumentRejection,
    AlternativeEvidenceDocumentSet,
)


class AlternativeEvidenceDocumentPublisher:
    """Thin domain adapter; Workspace owns bytes, revisions, and snapshots."""

    def __init__(self, workspace_root: Path) -> None:
        """Bind document publication to one confined workspace."""
        self.workspace = Workspace.create(workspace_root)
        self.library = WorkspaceKnowledgeLibrary(self.workspace)

    # ------------------------------------------------------- source objects

    def publish_source_object(
        self, document: AcquiredEvidenceDocument, *, admit: Callable[[int], None] | None = None
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
        content_sha256 = sha256_hex(document.content)
        relative = source_object_relative(content_sha256)
        with self.workspace.lock():
            target = make_confined_parents(self.workspace.root, relative)
            target = resolve_confined(self.workspace.root, relative)
            if target.exists():
                if not target.is_file() or target.read_bytes() != document.content:
                    raise ValueError("alternative_evidence.source_object_integrity")
            else:
                if admit is not None:
                    admit(len(document.content))
                staged = target.with_name(f".{target.name}.{os.getpid()}.staging")
                staged.unlink(missing_ok=True)
                write_new(staged, document.content)
                os.replace(staged, target)
                fsync_directory(target.parent)
        return AcquiredEvidenceDocumentReference.of(document, content_sha256=content_sha256)

    def source_reference_of(
        self, document: AcquiredEvidenceDocument
    ) -> AcquiredEvidenceDocumentReference:
        """Compute the source reference without publishing its bytes.

        The reference `publish_source_object` would return for this document,
        without keeping anything: the content address is the bytes' own.
        """
        return AcquiredEvidenceDocumentReference.of(
            document, content_sha256=sha256_hex(document.content)
        )

    def holds_source_object(self, content_sha256: str) -> bool:
        """Whether bytes are already kept at this content address."""
        path = resolve_confined(self.workspace.root, source_object_relative(content_sha256))
        return bool(path.is_file())

    def resolve_source_object(
        self, reference: AcquiredEvidenceDocumentReference
    ) -> AcquiredEvidenceDocument:
        """The document with its retained bytes, or a refusal that names what is wrong."""
        path = resolve_confined(self.workspace.root, reference.content_object_path)
        if not path.is_file():
            raise ValueError("alternative_evidence.source_object_missing")
        content = path.read_bytes()
        if (
            len(content) != reference.content_bytes
            or sha256_hex(content) != reference.content_sha256
        ):
            raise ValueError("alternative_evidence.source_object_tampered")
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
            # The bytes this publication adds: canonical blobs the library does
            # not hold yet (content-addressed, so a filing already in the
            # library adds nothing) and one small manifest per revision.
            new_blobs = 0
            for value in documents:
                blob = self.workspace.root / "knowledge" / "blobs" / value.canonical_content_hash
                if not blob.is_file():
                    new_blobs += value.byte_count
            admit(new_blobs + 4096 * len(documents))
        revisions = []
        references = []
        for value in sorted(documents, key=lambda item: item.semantic_handle):
            document_id = _workspace_document_id(value)
            access_class = (
                KnowledgeAccessClass.SYSTEM
                if value.source_name == "SEC_EDGAR"
                else KnowledgeAccessClass.USER_PRIVATE
            )
            namespace = (
                KnowledgeNamespace.SYSTEM_REFERENCE
                if access_class is KnowledgeAccessClass.SYSTEM
                else KnowledgeNamespace.USER_REFERENCE
            )
            document = WorkspaceKnowledgeDocument.create(
                document_id=document_id,
                namespace=namespace,
                title=value.title[:160],
                media_type=KnowledgeMediaType.MARKDOWN,
            )
            source = KnowledgeSourceCommitment(
                source_id=f"ae.{value.source_name.casefold().replace('_', '-')}",
                source_revision=1,
                source_logical_hash=value.canonical_content_hash,
                license_id=(
                    "SEC-PUBLIC-FILING"
                    if value.source_name == "SEC_EDGAR"
                    else "USER-PROVIDED-LOCAL-RESEARCH"
                ),
                access_class=access_class,
                available_at=value.available_at,
            )
            receipt = self.library.publish_revision(
                document=document,
                revision=1,
                source=source,
                # Dated by the evidence, not by the task that happened to read
                # it. The document id is content-addressed, so a second book
                # asking about a filing already in the library republishes the
                # same id; with the task's clock here the manifest differed and
                # the immutable store refused it, so only the first evidence
                # refresh in a workspace could ever succeed. `available_at` is
                # a property of the filing, which makes the republish
                # byte-identical and takes the library's replay path.
                created_at=value.available_at,
                content=value.canonical_markdown,
            )
            revisions.append(receipt.revision)
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
            sorted(
                revisions,
                key=lambda item: (
                    item.document.namespace.value,
                    item.document.document_id,
                    item.revision,
                ),
            )
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
    return f"ae.{entity}.{document.document_hash[:24]}"


def _stable_uuid(identity: str) -> UUID:
    return UUID(bytes=bytes.fromhex(identity[:32]), version=4)


__all__ = ["AlternativeEvidenceDocumentPublisher"]
