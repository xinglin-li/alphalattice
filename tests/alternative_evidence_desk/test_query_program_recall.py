"""The question bank's recall and ranking over the planted corpus.

Every planted claim is recalled with exact attribution through the
integrated selection's residual reading, and one index ranks a claim above an
earlier background document. (The production plan's packet rule -- a span at
its best rank, repeated occurrences not moving it -- went with the plan:
first-release integration T5.)
"""

from __future__ import annotations

from pathlib import Path

from alphalattice.evidence.alternative_evidence.analysis.contracts import (
    EvidenceTopic,
)
from alphalattice.evidence.alternative_evidence.analysis.packet import (
    EVIDENCE_QUERY_PROGRAM,
    MAXIMUM_PACKET_SPANS,
    query_program_hash,
)
from tests.alternative_evidence_desk.document_intelligence_support import (
    _open_recorded,
)
from tests.alternative_evidence_desk.planted_corpus import (
    PLANTED_CLAIMS,
    _recorded_document,
)

DOCUMENT_TOPICS: dict[str, tuple[EvidenceTopic, ...]] = {
    "AAPL": (
        EvidenceTopic.LIQUIDITY_GOING_CONCERN,
        EvidenceTopic.OPERATIONS_SUPPLY,
        EvidenceTopic.PRODUCT_SAFETY_CYBER,
    ),
    "MSFT": (EvidenceTopic.LEGAL_REGULATORY, EvidenceTopic.GOVERNANCE_CONTROLS),
    "NVDA": (
        EvidenceTopic.CAPITAL_DILUTION,
        EvidenceTopic.COMMERCIAL_COUNTERPARTY,
        EvidenceTopic.CORPORATE_ACTION_LISTING,
    ),
}


def test_the_query_program_recalls_every_planted_claim_with_exact_attribution(
    tmp_path: Path,
) -> None:
    """Which spans reach the analyst is a function of the generation and the program.

    Eight topics planted across three issuers, one program run: every planted
    sentence is in the packet, attributed to the issuer whose document carries
    it, within the packet cap, and identically on a second run. This is the
    offline evaluation the design promised in place of a model's own search.
    """

    documents = tuple(
        _recorded_document(
            entity,
            claims=tuple(PLANTED_CLAIMS[topic] for topic in topics),
            revision=f"{entity.casefold()}-2026-q3",
        )
        for entity, topics in DOCUMENT_TOPICS.items()
    )
    runtime, request, _registry_value, _snapshot, document_set, generation = _open_recorded(
        tmp_path, entities=tuple(DOCUMENT_TOPICS), documents=documents
    )
    receipt, spans = runtime.select_evidence(
        request=request, document_set=document_set, generation=generation
    )

    # Eight topics, still every one of them. The program carries more
    # queries than topics because a topic can hold several unrelated event
    # families, so the count to pin is the topics, not the queries.
    assert len(receipt.queries) == len(EVIDENCE_QUERY_PROGRAM)
    assert len({query.topic for query in EVIDENCE_QUERY_PROGRAM}) == 8
    assert {record.topic for record in receipt.queries} == set(EvidenceTopic)
    assert receipt.query_program_hash == query_program_hash()
    assert 0 < len(spans) <= MAXIMUM_PACKET_SPANS
    assert receipt.read_span_handles == tuple(value.span_handle for value in spans)

    owner = {topic: entity for entity, topics in DOCUMENT_TOPICS.items() for topic in topics}
    found: dict[EvidenceTopic, str] = {}
    for span in spans:
        for topic, claim in PLANTED_CLAIMS.items():
            if claim.casefold() in span.excerpt.casefold():
                assert span.entity_id == owner[topic], (topic, span.entity_id)
                found.setdefault(topic, span.span_handle)
    assert set(found) == set(PLANTED_CLAIMS), sorted(set(PLANTED_CLAIMS) - set(found))

    # The whole receipt -- the program's reads and the typed families' entry
    # beside them -- is the same on a second run of the same owner.
    again_receipt, again_spans = runtime.select_evidence(
        request=request, document_set=document_set, generation=generation
    )
    assert again_receipt.receipt_hash == receipt.receipt_hash
    assert [value.span_handle for value in again_spans] == [value.span_handle for value in spans]
    assert again_receipt.typed_disclosures is not None
    assert again_receipt.typed_disclosures.observations == ()
    assert again_receipt.typed_disclosures.documents_skipped_by_form == len(documents)


def test_one_index_ranks_a_claim_above_an_earlier_background_document(tmp_path: Path) -> None:
    """The regression behind the single index.

    The retired per-document components merged their hits by `(rank, document)`,
    so a background document's best passage sat beside the claim document's best
    passage regardless of score. One index over the whole generation ranks the
    passage that states the claim first, whichever document it came from.
    """

    liquidity = EvidenceTopic.LIQUIDITY_GOING_CONCERN
    documents = (
        _recorded_document("AAPL", claims=(), revision="aapl-background"),
        _recorded_document("MSFT", claims=(PLANTED_CLAIMS[liquidity],), revision="msft-q3"),
    )
    runtime, request, _registry_value, _snapshot, document_set, generation = _open_recorded(
        tmp_path, entities=("AAPL", "MSFT"), documents=documents
    )
    query = next(value for value in EVIDENCE_QUERY_PROGRAM if value.topic is liquidity)
    session = runtime.retrieval.open_session(
        document_set=document_set,
        generation=generation,
        evidence_as_of=request.evidence_as_of,
    )
    try:
        result = session.search(query=query.text, top_k=3)
        spans = session.read_spans(span_handles=(result.hits[0].span_handle,))
    finally:
        session.close()
    assert result.hits[0].entity_id == "MSFT"
    assert PLANTED_CLAIMS[liquidity].casefold() in spans[0].excerpt.casefold()
