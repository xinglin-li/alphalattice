"""Content-addressed artifact store and lineage validation for Alternative Evidence.

The store holds every evidence and review artifact by its own identity. There
is no current pointer: what may be *used now* is decided by the reader that
reopens a publication against a clock, and what a review consumed is named by
the review's own identity.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import cast

from pydantic import BaseModel

from alphalattice.evidence.alternative_evidence.analysis.contracts import (
    AlternativeEvidenceAnalystBrief,
    AlternativeEvidenceRetrievalAccessReceipt,
    CROAlternativeEvidencePackage,
)
from alphalattice.evidence.alternative_evidence.contracts import (
    AlternativeEvidenceRequest,
    AlternativeEvidenceSnapshot,
    SecIssuerRegistrySnapshot,
)
from alphalattice.evidence.alternative_evidence.documents.contracts import (
    AlternativeEvidenceDocumentSet,
)
from alphalattice.evidence.alternative_evidence.retrieval.contracts import (
    RetrievalGenerationRecord,
    parse_retrieval_generation,
)
from alphalattice.evidence.alternative_evidence.sources.contracts import (
    AcquiredEvidenceSourceSet,
    parse_source_set,
)


class AlternativeEvidencePublicationError(ValueError):
    """Stable fail-closed publication boundary error."""


_VERIFIED_RECORDS: ContextVar[dict[tuple[str, ...], BaseModel] | None] = ContextVar(
    "alternative_evidence_verified_records", default=None
)


@contextmanager
def verified_evidence_records() -> Iterator[None]:
    """Verify each stored record once per operation, then release.

    One Evidence/CRO read replays each analysis publication from several places,
    and every replay reopened the publication and its nine lineage records, so a
    section over seven analyses loaded 68 records 388 times. Inside this scope a
    record loaded and verified by its identity is kept for the rest of the
    operation: the store is content-addressed and refuses other bytes under an
    identity, so the record the operation verified is the one its later reads
    ask for. Only frozen records are kept. An operation run inside another joins
    the scope already open; nothing is kept across operations, and a Task's
    execution, which runs outside any operation, verifies in full.
    """
    if _VERIFIED_RECORDS.get() is not None:
        yield
        return
    token = _VERIFIED_RECORDS.set({})
    try:
        yield
    finally:
        _VERIFIED_RECORDS.reset(token)


class AlternativeEvidenceArtifactStore:
    """Own the JSON artifacts of Alternative Evidence and the review that reads them."""

    def __init__(self, artifact_root: Path) -> None:
        """Bind the content-addressed store beneath its artifact root."""
        self.root = artifact_root.resolve() / "alternative-evidence"
        self.read_count = 0
        self.write_count = 0

    def publish(self, category: str, identity: str, model: BaseModel) -> None:
        """Publish one artifact by identity, refusing conflicting bytes.

        Place one artifact under its identity, or verify the one already
        there: an artifact at the identity with other bytes is refused by
        name (`artifact_identity_reused`), never overwritten.
        """
        self._write(self._path(category, identity), _serialized(model))

    def holds(self, category: str, identity: str, model: BaseModel) -> bool:
        """Check whether a stored artifact matches the given record.

        Whether the store already holds exactly this artifact: True when
        the file at the identity carries the bytes this record serializes
        to (the seal's own canonical form, for the single-form categories),
        False when nothing is there. An artifact under the identity with
        other bytes -- damaged, substituted, another record -- is refused by
        name (`artifact_identity_reused`), never taken for the record
        because its path exists (review finding on `_publish_selection`,
        record section X) and never overwritten or repaired.
        """
        target = self._path(category, identity)
        if not target.is_file():
            return False
        if target.read_bytes() != _serialized(model):
            raise AlternativeEvidencePublicationError(
                "alternative_evidence.artifact_identity_reused"
            )
        return True

    def place(
        self,
        members: Sequence[tuple[str, str, BaseModel]],
        *,
        admit: Callable[[int], None] | None,
    ) -> int:
        """Admit and place missing artifacts in dependency order.

        Place the members the store does not hold yet, in the given order
        (a dependency before the record that names it), under one admission
        of their bytes together: every member is first verified through
        `holds` -- a damaged or substituted member refuses by name before
        any admission or write -- then the missing ones are admitted as one
        sum through `admit` (the workspace's storage budget; None admits
        nothing and refuses nothing, a library use without a Host) and
        written in order. Members the store holds cost no admission; a set
        the store holds whole is a valid retry that writes nothing and needs
        no allowance. A refused admission places nothing. Returns how many
        members were placed.
        """
        pending = [
            (self._path(category, identity), _serialized(model))
            for category, identity, model in members
            if not self.holds(category, identity, model)
        ]
        if not pending:
            return 0
        if admit is not None:
            admit(sum(len(content) for _target, content in pending))
        for target, content in pending:
            self._write(target, content)
        return len(pending)

    def exists(self, category: str, identity: str) -> bool:
        """Check whether a category holds a file at the given identity."""
        return self._path(category, identity).is_file()

    def _write(self, target: Path, content: bytes) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file():
            if target.read_bytes() != content:
                raise AlternativeEvidencePublicationError(
                    "alternative_evidence.artifact_identity_reused"
                )
            return
        self._atomic_write(target, content)
        self.write_count += 1

    @staticmethod
    def serialized(model: BaseModel) -> bytes:
        """The bytes a publication of this artifact writes."""
        return _serialized(model)

    def load[ModelT: BaseModel](self, category: str, identity: str, model: type[ModelT]) -> ModelT:
        """Load and verify one content-addressed artifact by identity."""
        target = self._path(category, identity)
        key = (str(self.root), category, identity, model.__qualname__)
        kept = self._kept(key)
        if kept is not None:
            return cast(ModelT, kept)
        if not target.is_file():
            raise FileNotFoundError("alternative_evidence.artifact_missing")
        self.read_count += 1
        value = model.model_validate_json(target.read_bytes())
        if getattr(value, _identity_field(model)) != identity:
            raise AlternativeEvidencePublicationError("alternative_evidence.artifact_tampered")
        return cast(ModelT, self._keep(key, value))

    def load_source_set(self, identity: str) -> AcquiredEvidenceSourceSet:
        """A source set in either supported durable form, verified by identity."""
        target = self._path("source-document-sets", identity)
        key = (str(self.root), "source-document-sets", identity)
        kept = self._kept(key)
        if kept is not None:
            return cast(AcquiredEvidenceSourceSet, kept)
        if not target.is_file():
            raise FileNotFoundError("alternative_evidence.artifact_missing")
        self.read_count += 1
        value = parse_source_set(json.loads(target.read_bytes()))
        if value.source_set_hash != identity:
            raise AlternativeEvidencePublicationError("alternative_evidence.artifact_tampered")
        return cast(AcquiredEvidenceSourceSet, self._keep(key, value))

    def load_retrieval_generation(self, identity: str) -> RetrievalGenerationRecord:
        """A generation record in either supported durable format, verified by identity."""
        target = self._path("retrieval-generations", identity)
        key = (str(self.root), "retrieval-generations", identity)
        kept = self._kept(key)
        if kept is not None:
            return cast(RetrievalGenerationRecord, kept)
        if not target.is_file():
            raise FileNotFoundError("alternative_evidence.artifact_missing")
        self.read_count += 1
        value = parse_retrieval_generation(json.loads(target.read_bytes()))
        if value.generation_hash != identity:
            raise AlternativeEvidencePublicationError("alternative_evidence.artifact_tampered")
        return cast(RetrievalGenerationRecord, self._keep(key, value))

    @staticmethod
    def _kept(key: tuple[str, ...]) -> BaseModel | None:
        """The record this operation already verified under `key`, if any."""
        scope = _VERIFIED_RECORDS.get()
        return None if scope is None else scope.get(key)

    @staticmethod
    def _keep(key: tuple[str, ...], value: BaseModel) -> BaseModel:
        """Keep a verified frozen record for the rest of the operation."""
        scope = _VERIFIED_RECORDS.get()
        if scope is not None and value.model_config.get("frozen"):
            scope[key] = value
        return value

    def values[ModelT: BaseModel](self, category: str, model: type[ModelT]) -> tuple[ModelT, ...]:
        """Load every verified artifact in a category."""
        directory = self.root / category
        if category not in _CATEGORIES or not directory.is_dir():
            return ()
        return tuple(
            self.load(category, path.stem, model) for path in sorted(directory.glob("*.json"))
        )

    def _path(self, category: str, identity: str) -> Path:
        if not _is_hash(identity) or category not in _CATEGORIES:
            raise AlternativeEvidencePublicationError(
                "alternative_evidence.artifact_reference_invalid"
            )
        return self.root / category / f"{identity}.json"

    @staticmethod
    def _atomic_write(target: Path, content: bytes) -> None:
        staged = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        staged.write_bytes(content)
        os.replace(staged, target)
        staged.unlink(missing_ok=True)


def validate_document_lineage(
    *,
    request: AlternativeEvidenceRequest,
    registry: SecIssuerRegistrySnapshot,
    snapshot: AlternativeEvidenceSnapshot,
    document_set: AlternativeEvidenceDocumentSet,
    generation: RetrievalGenerationRecord,
    access_receipt: AlternativeEvidenceRetrievalAccessReceipt,
    brief: AlternativeEvidenceAnalystBrief,
    cro_package: CROAlternativeEvidencePackage,
) -> None:
    """Validate the document and review lineage before publication."""
    registry_entities = {value.entity_id for value in registry.entries}
    document_entities = {value.entity_id for value in document_set.documents}
    verified_handles = tuple(value.span_handle for value in cro_package.verified_spans)
    finding_handles = tuple(
        dict.fromkeys(
            handle
            for finding in brief.findings
            for handle in (
                *finding.supporting_span_handles,
                *finding.contradicting_span_handles,
            )
        )
    )
    if not set(request.ordered_entity_ids).issubset(registry_entities):
        raise AlternativeEvidencePublicationError(
            "alternative_evidence.registry_authority_mismatch"
        )
    if not document_entities.issubset(request.ordered_entity_ids):
        raise AlternativeEvidencePublicationError(
            "alternative_evidence.document_entity_axis_invalid"
        )
    if any(
        value.available_at > request.evidence_as_of
        for value in (*document_set.documents, *cro_package.verified_spans)
    ):
        raise AlternativeEvidencePublicationError("alternative_evidence.causal_cutoff_invalid")
    if (
        snapshot.request_hash != request.request_hash
        or snapshot.registry_hash != registry.registry_hash
        or document_set.request_hash != request.request_hash
        or document_set.source_snapshot_hash != snapshot.snapshot_hash
        or generation.document_set_hash != document_set.document_set_hash
        or generation.workspace_snapshot_id != document_set.workspace_snapshot_id
        or generation.workspace_snapshot_hash != document_set.workspace_snapshot_hash
        or access_receipt.request_hash != request.request_hash
        or access_receipt.document_set_hash != document_set.document_set_hash
        or access_receipt.retrieval_generation_hash != generation.generation_hash
        or brief.request_hash != request.request_hash
        or brief.source_snapshot_hash != snapshot.snapshot_hash
        or brief.document_set_hash != document_set.document_set_hash
        or brief.retrieval_generation_hash != generation.generation_hash
        or brief.access_receipt_hash != access_receipt.receipt_hash
        or cro_package.request_hash != request.request_hash
        or cro_package.source_snapshot_hash != snapshot.snapshot_hash
        or cro_package.document_set_hash != document_set.document_set_hash
        or cro_package.retrieval_generation_hash != generation.generation_hash
        or cro_package.analyst_brief_hash != brief.brief_hash
        or cro_package.access_receipt_hash != access_receipt.receipt_hash
        or cro_package.obligation_hash != brief.obligation_hash
        or not set(finding_handles).issubset(access_receipt.delivered_span_handles)
        or set(verified_handles) != set(finding_handles)
        or tuple(value.finding_handle for value in cro_package.finding_structures)
        != tuple(value.finding_handle for value in brief.findings)
        or cro_package.expires_at != snapshot.expires_at
    ):
        raise AlternativeEvidencePublicationError(
            "alternative_evidence.current_document_authority_mismatch"
        )


def _identity_field(model: type[BaseModel]) -> str:
    matches = [name for name in model.model_fields if name.endswith("_hash")]
    # Every stored model has exactly one self-identity as its final hash field.
    return str(matches[-1])


def _is_hash(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _serialized(model: BaseModel) -> bytes:
    return json.dumps(model.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


_CATEGORIES = {
    "requests",
    "registries",
    "snapshots",
    "companyfacts-snapshots",
    "document-sets",
    "source-document-sets",
    # One body's durable commit, sealed as it arrives; with the source sets,
    # the rebuildable lookup of retained bodies by resource identity.
    "source-document-commits",
    # One resource's bounded deferral, sealed as it is observed oversize.
    "source-document-deferrals",
    "retrieval-generations",
    "retrieval-access-receipts",
    "resolved-span-sets",
    "analyst-briefs",
    "analyst-brief-receipts",
    "cro-packages",
    "analysis-publications",
    # The Portfolio evidence review. Four content-addressed families in the
    # store that already holds every evidence artifact, rather than a second
    # store beside it.
    "cro-review-dossiers",
    "cro-review-receipts",
    "cro-review-recommendations",
    "cro-review-publications",
    # Which analysis a review scope is read against, chosen by a person and
    # recorded append-only so the choice survives a restart.
    "cro-evidence-selections",
    # What each model stage cost, recorded beside the work it paid for.
    "provider-stage-usage",
    # Each answer an external agent gave one bundle, slot by slot, so the
    # bound on corrections survives a restart.
    "agent-answers",
    # What each bundle directory the Host prepared binds: its role and the
    # submission its answer completes, kept here and never beside the files.
    "agent-bundles",
    # A book wider than one request: the sealed run its Task executes, and
    # the typed refusal one unit records so the others can go on.
    "evidence-coverage-runs",
    "evidence-unit-failures",
    # What one issuer's SEC inventory held at the cutoff and what was fetched,
    # deferred or superseded, planned before capacity was spent.
    "sec-selection-plans",
    # One issuer's filing index read at a cutoff before packing, found by what
    # it answers: counted by the packing, taken by the unit's acquisition.
    "sec-inventory-reads",
    # One filing pair's temporal comparison under its complete closure,
    # sealed once and reused by every routing of the pair.
    "filing-comparisons",
    "pair-score-commitments",
}


__all__ = [
    "AlternativeEvidenceArtifactStore",
    "AlternativeEvidencePublicationError",
    "validate_document_lineage",
    "verified_evidence_records",
]
