"""Setup shared by the document intelligence suites that were one module.

The helpers two or more of those suites reach, moved here once when the
3,300-line module was split by lifecycle; nothing here is product authority
or evidence.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from alphalattice.evidence.alternative_evidence.analysis.contracts import (
    AlternativeEvidenceResearchObligation,
    AlternativeEvidenceRetrievalAccessReceipt,
    LitigationMatterRecord,
    seal_research_obligation,
)
from alphalattice.evidence.alternative_evidence.contracts import (
    INTEGRATED_FAMILY_SPELLING,
    MATTER_SELECTION_INTEGRATED,
    AlternativeEvidenceAdmission,
    AlternativeEvidenceClass,
    AlternativeEvidenceMode,
    AlternativeEvidenceRequest,
    AlternativeEvidenceSourcePolicy,
    MatterSelectionPolicy,
    SecIssuerRegistryEntry,
    SecIssuerRegistrySnapshot,
    seal_contract,
)
from alphalattice.evidence.alternative_evidence.runtime.service import (
    AlternativeEvidenceDocumentIntelligenceRuntime,
)
from alphalattice.evidence.alternative_evidence.sources.recorded import RecordedEvidenceDocument
from alphalattice.foundation.data_layer import encoders, recall
from tests.alternative_evidence_desk.planted_corpus import (
    _NOW,
    CIKS,
    _PlantedEncoder,
    _recorded_document,
    _runtime,
)

APPROVED_SOURCE_FAMILIES: tuple[str, ...] = ("SEC_EDGAR_OFFICIAL",)

INTEGRATED_SELECTION = MatterSelectionPolicy(
    method=MATTER_SELECTION_INTEGRATED, families=INTEGRATED_FAMILY_SPELLING
)
"""The one current matter selection (first-release integration T5): a new
request names it; the production plan (an omitted selection) and the
candidate allocation are retired and read back only."""


def _request(
    entities: tuple[str, ...] = ("AAPL",),
    *,
    matter_selection: MatterSelectionPolicy | None = INTEGRATED_SELECTION,
) -> AlternativeEvidenceRequest:
    return seal_contract(
        AlternativeEvidenceRequest,
        "request_hash",
        ordered_entity_ids=entities,
        evidence_as_of=_NOW,
        acquisition_deadline=_NOW + timedelta(hours=1),
        evidence_classes=(AlternativeEvidenceClass.ISSUER_OFFICIAL_RECORDED,),
        source_policy=AlternativeEvidenceSourcePolicy(),
        ttl_seconds=86_400,
        mode=AlternativeEvidenceMode.RECORDED,
        **({} if matter_selection is None else {"matter_selection": matter_selection}),
    )


def _request_at(cutoff: Any) -> AlternativeEvidenceRequest:
    return seal_contract(
        AlternativeEvidenceRequest,
        "request_hash",
        ordered_entity_ids=("AAPL",),
        evidence_as_of=cutoff,
        acquisition_deadline=cutoff + timedelta(hours=1),
        evidence_classes=(AlternativeEvidenceClass.ISSUER_OFFICIAL_RECORDED,),
        source_policy=AlternativeEvidenceSourcePolicy(),
        ttl_seconds=86_400,
        mode=AlternativeEvidenceMode.RECORDED,
    )


def _obligation(
    request: AlternativeEvidenceRequest,
    *,
    approved_source_families: tuple[str, ...] = APPROVED_SOURCE_FAMILIES,
) -> AlternativeEvidenceResearchObligation:
    return seal_research_obligation(
        question=(
            "Identify issuer-specific downside, contradiction, uncertainty or de-risk "
            "evidence at or before the cutoff."
        ),
        ordered_entity_ids=request.ordered_entity_ids,
        evidence_as_of=request.evidence_as_of,
        approved_source_families=approved_source_families,
        required_checks=("supporting evidence", "contradicting evidence"),
    )


def _registry(entities: tuple[str, ...] = ("AAPL",)) -> SecIssuerRegistrySnapshot:
    return seal_contract(
        SecIssuerRegistrySnapshot,
        "registry_hash",
        captured_at=_NOW - timedelta(days=1),
        entries=tuple(
            SecIssuerRegistryEntry(
                entity_id=entity,
                ticker=entity,
                cik=CIKS[entity],
                legal_name=f"{entity} Inc.",
            )
            for entity in entities
        ),
        source_content_hash="blake3:" + "1" * 64,
    )


def _document_with_text(
    entity_id: str = "AAPL", *, text: str, form: str = "10-Q", revision: str | None = None
) -> RecordedEvidenceDocument:
    """The planted recorded document with its text, form and revision replaced."""

    base = _recorded_document(entity_id)
    return base.model_copy(
        update={
            "text": text,
            "document_type": form,
            "revision": revision if revision is not None else f"{form}-2026".lower(),
        }
    )


def _open_recorded(
    tmp_path: Path,
    *,
    entities: tuple[str, ...] = ("AAPL",),
    documents: tuple[RecordedEvidenceDocument, ...] | None = None,
) -> tuple[
    AlternativeEvidenceDocumentIntelligenceRuntime, AlternativeEvidenceRequest, Any, Any, Any, Any
]:
    """Acquire, canonicalize and index. Returns the runtime and every sealed input."""

    runtime = _runtime(tmp_path)
    request = _request(entities)
    registry = _registry(entities)
    snapshot, source_set = runtime.acquire_recorded(
        request=request,
        registry=registry,
        documents=documents if documents is not None else (_recorded_document(),),
        published_at=_NOW,
    )
    document_set = runtime.canonicalize(source_set=source_set, published_at=_NOW, as_of=_NOW)
    generation = runtime.build_retrieval(document_set=document_set, built_at=_NOW)
    return runtime, request, registry, snapshot, document_set, generation


def _counted_runtime(
    tmp_path: Path,
    *,
    queries: list[str] | None = None,
) -> tuple[AlternativeEvidenceDocumentIntelligenceRuntime, list[int]]:
    """A runtime on the store whose encoder records the size of every passage batch, and
    every query when `queries` is given.

    Wall time measures the machine. The number of passages put through the model measures
    the design: the store encodes a chunk once, so only a first build of new passages
    encodes any, and a query encodes its query alone. The counting encoder replaces the
    planted one under `review_store`, which restores it.
    """
    passes: list[int] = []

    class Counting(_PlantedEncoder):
        def embed_passages(self, texts: list[str]) -> Any:
            passes.append(len(texts))
            return super().embed_passages(texts)

        def embed_query(self, text: str) -> Any:
            if queries is not None:
                queries.append(text)
            return super().embed_query(text)

    setattr(encoders, "StaticEncoder", Counting)  # noqa: B010 -- restored by review_store
    recall.close()
    runtime = AlternativeEvidenceDocumentIntelligenceRuntime(
        artifact_root=tmp_path / "artifacts", workspace_root=tmp_path / "workspace"
    )
    return runtime, passes


def _corpus_passes(passes: list[int], generation: Any) -> int:
    """How many times this generation's corpus was embedded, in whole passes.

    A build embeds one revision per model call, so the corpus of a
    multi-document generation is several calls; the measure is the chunks
    put through the model divided by the corpus's chunk count. Per-hit
    checks embed queries, never passages, and are not in `passes`.
    """

    return sum(passes) // generation.chunk_count


def _built(
    runtime: AlternativeEvidenceDocumentIntelligenceRuntime,
    *,
    entities: tuple[str, ...] = ("AAPL",),
    documents: tuple[RecordedEvidenceDocument, ...] | None = None,
) -> tuple[Any, Any, Any]:
    """Acquire, canonicalize and build one generation on a given runtime."""

    request = _request(entities)
    _snapshot, source_set = runtime.acquire_recorded(
        request=request,
        registry=_registry(entities),
        documents=documents if documents is not None else (_recorded_document(),),
        published_at=_NOW,
    )
    document_set = runtime.canonicalize(source_set=source_set, published_at=_NOW, as_of=_NOW)
    generation = runtime.build_retrieval(document_set=document_set, built_at=_NOW)
    return request, document_set, generation


def _receipt_with(
    program: AlternativeEvidenceRetrievalAccessReceipt,
    record: LitigationMatterRecord,
    session_counts: tuple[int, int],
) -> AlternativeEvidenceRetrievalAccessReceipt:
    return seal_contract(
        AlternativeEvidenceRetrievalAccessReceipt,
        "receipt_hash",
        request_hash=program.request_hash,
        document_set_hash=program.document_set_hash,
        retrieval_generation_hash=program.retrieval_generation_hash,
        query_program_hash=program.query_program_hash,
        queries=program.queries,
        read_span_handles=program.read_span_handles,
        search_call_count=session_counts[0],
        span_read_call_count=session_counts[1],
        span_groups=program.span_groups,
        litigation_matters=record,
    )


def live_admission(
    request: AlternativeEvidenceRequest, *, admitted_at: datetime
) -> AlternativeEvidenceAdmission:
    return seal_contract(
        AlternativeEvidenceAdmission,
        "admission_hash",
        request_hash=request.request_hash,
        network_consent=True,
        admit_live_official=True,
        admit_model_review=False,
        admitted_at=admitted_at,
    )
