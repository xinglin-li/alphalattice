"""The planted-claims filing corpus and the offline retrieval runtime of the
Alternative Evidence suites.

The issuers' CIKs, the claims planted in their recorded filings, the one-hot
embedding adapter and fake reranker that make retrieval deterministic without
a model pack, the recorded-document writer and the citing actor that drives
them. Three suites drove all of this through the document intelligence test
module. Test support beside its owner; nothing here is product authority or
evidence.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from alphalattice.evidence.alternative_evidence.analysis.contracts import (
    AlternativeEvidenceAnalystAnswer,
    AlternativeEvidenceAnswerFinding,
    EvidenceDirection,
    EvidenceLifecycle,
    EvidenceTopic,
)
from alphalattice.evidence.alternative_evidence.analysis.packet import (
    AlternativeEvidencePacket,
    span_aliases,
)
from alphalattice.evidence.alternative_evidence.contracts import (
    AlternativeEvidenceClass,
)
from alphalattice.evidence.alternative_evidence.runtime.service import (
    AlternativeEvidenceActorSubmission,
    AlternativeEvidenceDocumentIntelligenceRuntime,
)
from alphalattice.evidence.alternative_evidence.sources.recorded import RecordedEvidenceDocument
from alphalattice.kernel.knowledge.hybrid import (
    WorkspaceHybridKnowledgeIndex,
    WorkspaceHybridKnowledgeRetriever,
)
from alphalattice.kernel.knowledge.hybrid_contracts import (
    HybridCapabilityReport,
    HybridIndexSpec,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.protocols.actor_execution import ActorKind

_NOW = datetime(2026, 8, 12, 16, 0, tzinfo=UTC)

PLAYPEN_ROOT = Path(__file__).resolve().parents[2]

CIKS: dict[str, str] = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "NVDA": "0001045810",
    "TSLA": "0001318605",
    "AMZN": "0001018724",
}

PLANTED_CLAIMS: dict[EvidenceTopic, str] = {
    EvidenceTopic.LIQUIDITY_GOING_CONCERN: (
        "Management concluded that substantial doubt exists about the ability to continue "
        "as a going concern."
    ),
    EvidenceTopic.CAPITAL_DILUTION: (
        "The board approved convertible notes that may dilute existing holders by up to 12 percent."
    ),
    EvidenceTopic.LEGAL_REGULATORY: (
        "The company received a subpoena from a federal regulator concerning revenue recognition."
    ),
    EvidenceTopic.OPERATIONS_SUPPLY: (
        "A supplier interruption reduced available component capacity by 18 percent."
    ),
    EvidenceTopic.PRODUCT_SAFETY_CYBER: (
        "A voluntary recall of the flagship device was initiated after a battery safety defect."
    ),
    EvidenceTopic.GOVERNANCE_CONTROLS: (
        "The auditor identified a material weakness in controls over inventory valuation."
    ),
    EvidenceTopic.COMMERCIAL_COUNTERPARTY: (
        "The largest customer served notice of contract termination effective next quarter."
    ),
    EvidenceTopic.CORPORATE_ACTION_LISTING: (
        "The exchange issued a delisting warning for failure to meet the minimum bid price."
    ),
}

_TOPIC_KEYWORDS: tuple[str, ...] = (
    "supplier interruption",
    "going concern",
    "convertible notes",
    "subpoena",
    "recall",
    "material weakness",
    "contract termination",
    "delisting",
)


def _one_hot(position: int) -> tuple[float, ...]:
    return tuple(1.0 if index == position else 0.0 for index in range(384))


def _embedding(text: str) -> tuple[float, ...]:
    """A keyword one-hot: each topic keyword is its own direction, background is another."""

    lowered = text.casefold()
    for position, keyword in enumerate(_TOPIC_KEYWORDS):
        if keyword in lowered:
            return _one_hot(position)
    return _one_hot(len(_TOPIC_KEYWORDS))


class _EmbeddingAdapter:
    def close(self) -> None:
        return None

    def embed_passages(
        self,
        texts: tuple[str, ...],
        *,
        cancelled: Any | None = None,
    ) -> tuple[tuple[float, ...], ...]:
        if cancelled is not None and cancelled():
            raise RuntimeError("cancelled")
        return tuple(_embedding(text) for text in texts)

    def embed_query(self, text: str) -> tuple[float, ...]:
        return _embedding(text)


class _FakeReranker:
    """Scores a passage by how many of the query's words it states.

    Stands in for the cross-encoder the way `_EmbeddingAdapter` stands in for
    the embedder: deterministic, cheap, and ordered the way a real reranker
    would order these synthetic corpora, where the planted claim restates the
    query and the background does not.
    """

    def score(self, query: str, passages: list[str]) -> tuple[float, ...]:
        # Five-letter stems, so "interrupted" credits "interruption" the way a
        # real reranker would.
        wanted = {word.strip("?,.")[:5] for word in query.casefold().split() if len(word) > 4}
        return tuple(
            float(sum(1 for stem in wanted if stem in passage.casefold())) for passage in passages
        )


def _ready(_root: Path, spec: HybridIndexSpec) -> HybridCapabilityReport:
    return HybridCapabilityReport.ready(spec)


def _index_factory(workspace: Any, model_root: Path) -> WorkspaceHybridKnowledgeIndex:
    return WorkspaceHybridKnowledgeIndex(
        workspace,
        model_root=model_root,
        capability_probe=_ready,
        adapter_factory=lambda _root, _spec: _EmbeddingAdapter(),
        reranker_factory=lambda _root, _spec: _FakeReranker(),
    )


def _retriever_factory(
    workspace: Any,
    model_root: Path,
    spec: HybridIndexSpec,
) -> WorkspaceHybridKnowledgeRetriever:
    return WorkspaceHybridKnowledgeRetriever(
        workspace,
        model_root=model_root,
        index_spec=spec,
        capability_probe=_ready,
        adapter_factory=lambda _root, _spec: _EmbeddingAdapter(),
        reranker_factory=lambda _root, _spec: _FakeReranker(),
    )


def _recorded_document(
    entity_id: str = "AAPL",
    *,
    claims: tuple[str, ...] = (PLANTED_CLAIMS[EvidenceTopic.OPERATIONS_SUPPLY],),
    revision: str = "issuer-release-2026-q3",
    document_type: str = "8-K",
) -> RecordedEvidenceDocument:
    """One recorded official release, filed as a current report (the form
    the product acquires and the integrated selection reads; a text with no
    item heading past its cover is read whole by the residual questions):
    routine background with planted sentences."""

    background = "".join(
        f"Background section {index} reports routine operating context.\n" for index in range(60)
    )
    body = f"# Official quarterly update for {entity_id}\n\n" + background
    for index, claim in enumerate(claims, start=1):
        body += f"## Section {index}\n\n{claim}\n" + background
    return RecordedEvidenceDocument(
        entity_id=entity_id,
        source_right="USER_PROVIDED_FOR_LOCAL_RESEARCH",
        evidence_class=AlternativeEvidenceClass.ISSUER_OFFICIAL_RECORDED,
        document_type=document_type,
        revision=revision,
        published_at=_NOW - timedelta(days=2),
        captured_at=_NOW - timedelta(days=1),
        available_at=_NOW - timedelta(days=1),
        text=body,
        immutable_source=True,
    )


def _runtime(tmp_path: Path) -> AlternativeEvidenceDocumentIntelligenceRuntime:
    return AlternativeEvidenceDocumentIntelligenceRuntime(
        artifact_root=tmp_path / "artifacts",
        workspace_root=tmp_path / "workspace",
        model_root=tmp_path / "models",
        index_factory=_index_factory,
        retriever_factory=_retriever_factory,
    )


def packet_aliases(packet: AlternativeEvidencePacket) -> dict[str, str]:
    """Each delivered span's alias, as every view of the packet names it."""

    return {
        span.span_handle: alias
        for alias, span in span_aliases(packet.spans, packet.request.ordered_entity_ids).items()
    }


class _CitingActor:
    """An automation actor that cites, per issuer, every packet span of that issuer.

    One finding per issuer with delivered spans, in the request's issuer
    order, so the Host assigns `FIND-001`.. in that order. It is the
    actor-neutral port in its simplest honest form: it reads the packet it
    was handed and nothing else, and cites by the aliases its view shows.
    """

    def __init__(
        self,
        *,
        topics: dict[str, EvidenceTopic] | None = None,
        actor_id: str = "alternative-evidence.citing-automation",
    ) -> None:
        self.topics = topics or {}
        self.actor_id = actor_id
        self.packets_seen: list[AlternativeEvidencePacket] = []

    @property
    def process_binding_hash(self) -> str:
        return str(canonical_hash({"owner": "test.citing_actor", "version": 2}))

    def __call__(self, *, packet: AlternativeEvidencePacket) -> AlternativeEvidenceActorSubmission:
        self.packets_seen.append(packet)
        alias_of = packet_aliases(packet)
        findings: list[AlternativeEvidenceAnswerFinding] = []
        for entity_id in packet.obligation.ordered_entity_ids:
            cited = tuple(
                alias_of[span.span_handle] for span in packet.spans if span.entity_id == entity_id
            )
            if not cited:
                continue
            topic = self.topics.get(entity_id, EvidenceTopic.OPERATIONS_SUPPLY)
            findings.append(
                AlternativeEvidenceAnswerFinding(
                    issuer=entity_id,
                    topic=topic,
                    lifecycle=EvidenceLifecycle.ONGOING,
                    direction=EvidenceDirection.ADVERSE,
                    summary=(
                        f"{entity_id}: the admitted documents state a "
                        f"{topic.value.casefold().replace('_', ' ')} matter."
                    ),
                    cite=cited[:8],
                )
            )
        return AlternativeEvidenceActorSubmission(
            answer=AlternativeEvidenceAnalystAnswer(findings=tuple(findings)),
            actor_kind=ActorKind.EXTERNAL_AUTOMATION,
            actor_id=self.actor_id,
        )
