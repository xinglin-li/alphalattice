"""Recorded official evidence with explicit rights and availability time."""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.kernel.shared_kernel.identity import canonical_hash

from ..contracts import (
    ADMITTED_DOCUMENT_CAPACITY,
    AlternativeEvidenceCitation,
    AlternativeEvidenceClass,
    AlternativeEvidenceRequest,
    seal_contract,
)
from .contracts import AcquiredEvidenceDocument, PublishedPrecision

_BASELINE_TYPES = ("10-K", "10-K/A", "10-Q", "10-Q/A")
"""Recorded document types kept first when an issuer's material exceeds its
capacity: the newest of each, then the rest newest first."""

RECORDED_SELECTION_RULES_ID = "alternative-evidence.recorded-source-selection.v2"
"""v2 (2026-09-22): an issuer's selection is its own -- the cutoff-valid
documents of the requested classes, the newest of each baseline type first
and the rest newest first, as many as the policy's per-issuer budget
admits -- decided before the unit is packed and independent of how many
issuers share the unit. The admitted document set's capacity is then a
bound on the unit as a whole: a unit whose selections exceed it defers
non-baseline documents, round-robin from the issuer holding the most,
oldest first, each named with the reason, and a unit whose baselines alone
exceed it is refused by name. v1 divided the capacity by the issuer count
(`24 // issuers`), so adding an issuer to a batch removed another issuer's
documents from its selection."""


class RecordedEvidenceDocument(BaseModel):  # type: ignore[misc]
    """A user-imported official document; text never crosses the Host boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    entity_id: str = Field(min_length=1, max_length=32)
    source_name: Literal["ISSUER_RECORDED"] = "ISSUER_RECORDED"
    source_right: Literal["USER_PROVIDED_FOR_LOCAL_RESEARCH"]
    evidence_class: AlternativeEvidenceClass
    document_type: str = Field(min_length=1, max_length=40)
    revision: str = Field(min_length=1, max_length=120)
    published_at: datetime | None = None
    captured_at: datetime
    available_at: datetime
    text: str = Field(min_length=1)
    immutable_source: bool
    limitations: tuple[str, ...] = ()
    # The original's time, kept beside the import's: acceptance is when the
    # official source accepted the filing, capture is when this system
    # obtained it, and availability for an offline import is never earlier
    # than its capture. Each is absent from the identity when unknown, so
    # bundles installed before these were recorded keep their hashes.
    accepted_at: datetime | None = Field(default=None, exclude_if=lambda value: value is None)
    published_precision: PublishedPrecision | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    report_period_end: date | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_document(self) -> Self:
        """Validate recorded source clocks and their availability order."""
        for value in (self.captured_at, self.available_at, self.published_at, self.accepted_at):
            if value is not None and (value.tzinfo is None or value.utcoffset() is None):
                raise ValueError("alternative_evidence.recorded_clock_invalid")
        if self.available_at < self.captured_at:
            raise ValueError("alternative_evidence.recorded_availability_invalid")
        if self.accepted_at is not None and self.available_at < self.accepted_at:
            raise ValueError("alternative_evidence.recorded_acceptance_after_availability")
        if self.published_precision is not None and self.published_at is None:
            raise ValueError("alternative_evidence.recorded_precision_without_publication")
        return self


class RecordedEvidenceSource:
    """Resolve recorded imports without discarding their bounded source text."""

    @staticmethod
    def capacity(request: AlternativeEvidenceRequest) -> int:
        """Bound one issuer's recorded document selection.

        Documents one issuer of this request may bring: the policy's own
        per-issuer budget, bounded by one admitted document set. The unit's
        issuers do not divide the set between them; the unit is packed from
        their selections (`RECORDED_SELECTION_RULES_ID`).
        """
        return min(request.source_policy.maximum_documents_per_issuer, ADMITTED_DOCUMENT_CAPACITY)

    @staticmethod
    def logical_selection(
        *,
        entity_id: str,
        documents: tuple[RecordedEvidenceDocument, ...],
        evidence_as_of: datetime,
        evidence_classes: frozenset[AlternativeEvidenceClass],
        policy_budget: int,
    ) -> tuple[
        tuple[tuple[int, RecordedEvidenceDocument], ...],
        tuple[tuple[int, RecordedEvidenceDocument], ...],
    ]:
        """Select one issuer's cutoff-valid recorded documents.

        One issuer's own selection from the recorded library at a cutoff:
        `(selected, deferred beyond the policy budget)`, each with the
        library sequence numbers, the newest of each baseline type first and
        the rest newest first. Independent of any other issuer: what the
        packer counts and what the acquisition takes are the same list.
        """
        capacity = min(policy_budget, ADMITTED_DOCUMENT_CAPACITY)
        items = [
            (sequence, document)
            for sequence, document in enumerate(documents, start=1)
            if document.entity_id == entity_id
            and document.evidence_class in evidence_classes
            and document.available_at <= evidence_as_of
        ]
        if len(items) <= capacity:
            return tuple(items), ()
        newest = sorted(items, key=lambda item: (item[1].available_at, item[0]), reverse=True)
        chosen: list[tuple[int, RecordedEvidenceDocument]] = []
        for document_type in _BASELINE_TYPES:
            for item in newest:
                if item[1].document_type == document_type and item not in chosen:
                    chosen.append(item)
                    break
        chosen = chosen[:capacity]
        for item in newest:
            if len(chosen) >= capacity:
                break
            if item not in chosen:
                chosen.append(item)
        deferred = tuple(item for item in newest if item not in chosen)
        return tuple(sorted(chosen, key=lambda item: item[0])), deferred

    def select_documents(
        self,
        *,
        request: AlternativeEvidenceRequest,
        documents: tuple[RecordedEvidenceDocument, ...],
    ) -> tuple[tuple[tuple[int, RecordedEvidenceDocument], ...], tuple[str, ...]]:
        """Select the unit's documents and name every deferral.

        The cutoff-valid documents of the requested issuers, each issuer's
        own selection (`logical_selection`), with their library sequence
        numbers, and one line per deferral naming what and why: beyond an
        issuer's policy budget, or beyond the admitted document set's
        capacity for the unit as a whole -- in which case non-baseline
        documents are deferred round-robin from the issuer holding the
        most, oldest first, and a unit whose baselines alone exceed the
        capacity is refused by name. A deferred document is named here,
        never silently rejected later by the canonicalizer's capacity.
        """
        # An issuer this book does not hold is skipped, not refused. The
        # recorded package is a library installed once for the workspace,
        # while the entity axis comes from one book's positions, so any
        # book narrower than the library used to block the whole task.
        # `sequence` still counts the whole library, so a document keeps
        # the same handle whichever book asked for it.
        evidence_classes = frozenset(request.evidence_classes)
        budget = request.source_policy.maximum_documents_per_issuer
        chosen: dict[str, list[tuple[int, RecordedEvidenceDocument]]] = {}
        deferrals: list[str] = []
        for entity_id in request.ordered_entity_ids:
            selected, deferred = self.logical_selection(
                entity_id=entity_id,
                documents=documents,
                evidence_as_of=request.evidence_as_of,
                evidence_classes=evidence_classes,
                policy_budget=budget,
            )
            chosen[entity_id] = list(selected)
            if deferred:
                deferrals.append(
                    f"{entity_id}: {len(deferred)} recorded document(s) deferred beyond the "
                    f"policy budget of {min(budget, ADMITTED_DOCUMENT_CAPACITY)} documents per "
                    "issuer: "
                    + ", ".join(f"{item[1].document_type} {item[1].revision}" for item in deferred)
                    + "."
                )
        unit_deferred: dict[str, list[tuple[int, RecordedEvidenceDocument]]] = {}
        while sum(len(items) for items in chosen.values()) > ADMITTED_DOCUMENT_CAPACITY:
            # The issuer holding the most non-baseline documents yields its
            # oldest one; the baselines are never deferred here.
            candidates = [
                (
                    sum(1 for _s, d in items if d.document_type not in _BASELINE_TYPES),
                    entity_id,
                )
                for entity_id, items in chosen.items()
            ]
            count, entity_id = max(candidates, key=lambda pair: (pair[0], pair[1]))
            if count == 0:
                raise ValueError("alternative_evidence.unit_capacity_below_baselines")
            oldest = min(
                (
                    item
                    for item in chosen[entity_id]
                    if item[1].document_type not in _BASELINE_TYPES
                ),
                key=lambda item: (item[1].available_at, item[0]),
            )
            chosen[entity_id].remove(oldest)
            unit_deferred.setdefault(entity_id, []).append(oldest)
        for entity_id, items in unit_deferred.items():
            deferrals.append(
                f"{entity_id}: {len(items)} recorded document(s) deferred beyond the admitted "
                f"document set's capacity of {ADMITTED_DOCUMENT_CAPACITY} for this unit of "
                f"{len(request.ordered_entity_ids)} issuers (pack fewer issuers per unit): "
                + ", ".join(f"{item[1].document_type} {item[1].revision}" for item in items)
                + "."
            )
        selected_all = sorted(
            (item for items in chosen.values() for item in items), key=lambda item: item[0]
        )
        return tuple(selected_all), tuple(deferrals)

    def acquire_documents(
        self,
        *,
        request: AlternativeEvidenceRequest,
        documents: tuple[RecordedEvidenceDocument, ...],
    ) -> tuple[AcquiredEvidenceDocument, ...]:
        """Materialize the selected recorded documents with stable handles."""
        resolved: list[AcquiredEvidenceDocument] = []
        selected, _deferrals = self.select_documents(request=request, documents=documents)
        for sequence, document in selected:
            content = document.text.encode("utf-8")
            resolved.append(
                AcquiredEvidenceDocument(
                    semantic_handle=f"DOC-{document.entity_id}-{sequence:03d}",
                    entity_id=document.entity_id,
                    source_name=document.source_name,
                    source_right=document.source_right,
                    evidence_class=document.evidence_class,
                    document_type=document.document_type,
                    revision=document.revision,
                    title=f"{document.entity_id} {document.document_type}",
                    published_at=document.published_at,
                    published_precision=document.published_precision,
                    accepted_at=document.accepted_at,
                    available_at=document.available_at,
                    report_period_end=document.report_period_end,
                    media_type="text/markdown",
                    content=content,
                    immutable_source=document.immutable_source,
                    limitations=document.limitations,
                    source_content_hash=canonical_hash(
                        {
                            "source_name": document.source_name,
                            "revision": document.revision,
                            "media_type": "text/markdown",
                            "content_hex": content.hex(),
                        }
                    ),
                )
            )
        return tuple(resolved)

    def resolve(
        self,
        *,
        request: AlternativeEvidenceRequest,
        documents: tuple[RecordedEvidenceDocument, ...],
    ) -> tuple[AlternativeEvidenceCitation, ...]:
        """Seal citations for the selected recorded documents."""
        citations: list[AlternativeEvidenceCitation] = []
        acquired = self.acquire_documents(request=request, documents=documents)
        for sequence, source in enumerate(acquired, 1):
            excerpt = _compact_excerpt(source.content.decode("utf-8"))
            citations.append(
                seal_contract(
                    AlternativeEvidenceCitation,
                    "citation_hash",
                    semantic_handle=f"CIT-{source.entity_id}-{sequence:03d}",
                    entity_id=source.entity_id,
                    source_name=source.source_name,
                    source_right=source.source_right,
                    evidence_class=source.evidence_class,
                    document_type=source.document_type,
                    revision=source.revision,
                    published_at=source.published_at,
                    accepted_at=source.accepted_at,
                    available_at=source.available_at,
                    excerpt=excerpt,
                    limitations=source.limitations,
                    immutable_source=source.immutable_source,
                    source_content_hash=source.source_content_hash,
                )
            )
        return tuple(citations)


def _compact_excerpt(text: str) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        raise ValueError("alternative_evidence.recorded_text_empty")
    return compact[:1600]


__all__ = ["RECORDED_SELECTION_RULES_ID", "RecordedEvidenceDocument", "RecordedEvidenceSource"]
