"""Synthetic filings and deterministic retrieval/actors; no product authority or evidence."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

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
from alphalattice.kernel.data_layer import encoders, history, recall
from alphalattice.kernel.knowledge import model_store
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


def _embedding(text: str) -> np.ndarray:
    """A keyword one-hot: each topic keyword is its own direction, background is another."""

    lowered = text.casefold()
    position = next(
        (index for index, keyword in enumerate(_TOPIC_KEYWORDS) if keyword in lowered),
        len(_TOPIC_KEYWORDS),
    )
    row = np.zeros(encoders.DIMENSION, dtype=np.float32)
    row[position] = 1.0
    return row


class _PlantedEncoder:
    """The store's encoder for the planted corpora: `_embedding` of each text."""

    def __init__(self, _pack: Path | None = None) -> None:
        self.passages: list[int] = []
        """The size of each passage batch encoded."""

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        self.passages.append(len(texts))
        return np.stack([_embedding(text) for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return _embedding(text)


class _FakeReranker:
    """Scores a passage by how many of the query's words it states.

    Stands in for the cross-encoder the way `_PlantedEncoder` stands in for
    the embedder: deterministic, cheap, and ordered the way a real reranker
    would order these synthetic corpora, where the planted claim restates the
    query and the background does not.
    """

    def __init__(self, _pack: Path | None = None) -> None:
        pass

    def score(self, query: str, passages: list[str]) -> tuple[float, ...]:
        # Five-letter stems, so "interrupted" credits "interruption" the way a
        # real reranker would.
        wanted = {word.strip("?,.")[:5] for word in query.casefold().split() if len(word) > 4}
        return tuple(
            float(sum(1 for stem in wanted if stem in passage.casefold())) for passage in passages
        )


@contextmanager
def review_store(folder: Path) -> Iterator[None]:
    """The Host's history and the store's two models as a review needs them: the planted
    encoder and reranker in place of the packs, and a history opened in `folder` unless this
    process's Host already records one. The held search is dropped when it ends."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(encoders, "StaticEncoder", _PlantedEncoder)
        patch.setattr(encoders, "Reranker", _FakeReranker)
        patch.setattr(model_store, "default_store_root", lambda: folder / "models")
        patch.setattr(model_store, "verify_directory", lambda _directory, _pack: None)
        opened = not history.recording()
        if opened:
            history.open_history(f"test-{uuid4().hex[:12]}", "test", folder)
        recall.close()
        try:
            yield
        finally:
            recall.close()
            if opened:
                history.release_history()


@contextmanager
def tampered_reference(document_id: str) -> Iterator[None]:
    """The canonical text of `document_id` changed in the Host's log, as an edited or damaged
    data root would hold it, and restored when it ends. The log refuses edits by trigger; this
    stands in for a change made outside it, so the trigger is lifted for the write alone."""
    opened = history.opened_log()
    assert opened is not None, "a review's documents live in the Host's log"
    log = opened[0]
    with log.unit() as con:
        ((seq, text),) = con.execute(
            "SELECT v.body_seq, t.text FROM knowledge_source_version v"
            " JOIN event_text t ON t.seq = v.body_seq WHERE v.source_id = ?",
            (document_id,),
        ).fetchall()
    payload = json.loads(text)
    body = payload["content"]
    payload["content"] = body[:-1] + ("?" if body.endswith("!") else "!")

    def write(value: str) -> None:
        with log.unit() as con:
            (trigger,) = con.execute(
                "SELECT sql FROM sqlite_master WHERE name = 'event_text_no_update'"
            ).fetchone()
            con.execute("DROP TRIGGER event_text_no_update")
            con.execute("UPDATE event_text SET text = ? WHERE seq = ?", (value, seq))
            con.execute(trigger)

    write(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    try:
        yield
    finally:
        write(text)


def _recorded_document(
    entity_id: str = "AAPL",
    *,
    claims: tuple[str, ...] = (PLANTED_CLAIMS[EvidenceTopic.OPERATIONS_SUPPLY],),
    revision: str = "issuer-release-2026-q3",
    document_type: str = "8-K",
    now: datetime = _NOW,
) -> RecordedEvidenceDocument:
    """A current report with planted sentences and scenario-relative dates."""

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
        published_at=now - timedelta(days=2),
        captured_at=now - timedelta(days=1),
        available_at=now - timedelta(days=1),
        text=body,
        immutable_source=True,
    )


def _runtime(tmp_path: Path) -> AlternativeEvidenceDocumentIntelligenceRuntime:
    return AlternativeEvidenceDocumentIntelligenceRuntime(
        artifact_root=tmp_path / "artifacts",
        workspace_root=tmp_path / "workspace",
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
