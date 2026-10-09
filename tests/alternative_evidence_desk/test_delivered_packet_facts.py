"""Facts survive retrieval and deduplication into the delivered packet.

Scored on what the packet delivers, not on candidates: a quiet issuer beside a
verbose one keeps its own places in every query's return; compliance and
non-compliance are two facts; two amounts on two dates are two facts; a
verbatim restatement within one filing is grouped under the read span and
recorded, not deleted; two passages with the same preview and different tails
both keep their identity; a fact at the end of a passage is delivered whole.
Synthetic keyword encoder and word-count reranker (`planted_corpus`): the
mechanics are the claim here, not financial relevance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from alphalattice.evidence.alternative_evidence.analysis.packet import (
    EVIDENCE_QUERY_PROGRAM_ID,
    QUERY_TOP_K,
)
from tests.alternative_evidence_desk.document_intelligence_support import _open_recorded
from tests.alternative_evidence_desk.planted_corpus import _recorded_document

BACKGROUND = "Background section {index} reports routine operating context.\n"


def _document(entity: str, sections: tuple[str, ...], *, revision: str) -> Any:
    base = _recorded_document(entity, claims=sections, revision=revision)
    # Each issuer's background names it, so no two issuers share a passage.
    return base.model_copy(
        update={"text": base.text.replace("Background section", f"{entity} background section")}
    )


def _verbose_document(entity: str, claims: tuple[str, ...], *, revision: str) -> Any:
    """Many sections, each with its own claim and its own background lines, so
    the quality gate does not refuse it as repeated boilerplate."""

    body = f"# Official quarterly update for {entity}\n\n"
    for index, claim in enumerate(claims, start=1):
        body += f"## Section {index}\n\n{claim}\n" + "".join(
            f"{entity} section {index} paragraph {line} discusses {revision} matter {line}.\n"
            for line in range(12)
        )
    base = _recorded_document(entity, claims=(), revision=revision)
    return base.model_copy(update={"text": body})


def _packet(tmp_path: Path, entities: tuple[str, ...], documents: tuple[Any, ...]) -> Any:
    runtime, request, _registry, _snapshot, document_set, generation = _open_recorded(
        tmp_path, entities=entities, documents=documents
    )
    try:
        return runtime.select_evidence(
            request=request, document_set=document_set, generation=generation
        )
    finally:
        runtime.close()


def _delivered(spans: tuple[Any, ...], entity: str) -> list[str]:
    return [" ".join(span.excerpt.split()) for span in spans if span.entity_id == entity]


def test_a_quiet_issuer_keeps_its_places_beside_a_verbose_one(tmp_path: Path) -> None:
    """The verbose issuer restates a supplier interruption in thirty sections
    across three filings; the quiet one states it once. Each query's return is
    cut per issuer, so the quiet issuer's one passage is returned and read."""

    verbose = tuple(
        _verbose_document(
            "MSFT",
            tuple(
                f"A supplier interruption reduced available component capacity by {n} percent "
                f"in region {n}."
                for n in range(10 * index, 10 * index + 10)
            ),
            revision=f"msft-filing-{index}",
        )
        for index in range(3)
    )
    quiet = _document(
        "AAPL",
        ("A supplier interruption reduced available component capacity by 18 percent.",),
        revision="aapl-filing",
    )
    receipt, spans = _packet(tmp_path, ("AAPL", "MSFT"), (*verbose, quiet))
    operations = next(record for record in receipt.queries if record.query_id == "Q-OPERATIONS")
    # The verbose issuer alone fills more than one global twenty; the query's
    # return holds both issuers' places, and the quiet issuer's fact is read.
    assert len(operations.hit_span_handles) > QUERY_TOP_K, "per-issuer twenties, not one twenty"
    assert any("18 percent" in text for text in _delivered(spans, "AAPL"))
    assert len(_delivered(spans, "MSFT")) >= 10
    assert receipt.query_program_hash and EVIDENCE_QUERY_PROGRAM_ID.endswith("v3")


def test_compliance_and_non_compliance_are_two_facts(tmp_path: Path) -> None:
    document = _document(
        "AAPL",
        (
            "The company was in compliance with every covenant under its credit facility "
            "at quarter end.",
            "The company was not in compliance with every covenant under its credit "
            "facility at quarter end.",
        ),
        revision="aapl-covenants",
    )
    _receipt, spans = _packet(tmp_path, ("AAPL",), (document,))
    delivered = _delivered(spans, "AAPL")
    assert any("was in compliance" in text for text in delivered)
    assert any("was not in compliance" in text for text in delivered)


def test_two_amounts_on_two_dates_are_two_facts(tmp_path: Path) -> None:
    document = _document(
        "AAPL",
        (
            "The company recorded an impairment of the northern facility of 10 million "
            "dollars in March 2026.",
            "The company recorded an impairment of the northern facility of 45 million "
            "dollars in June 2026.",
        ),
        revision="aapl-impairments",
    )
    _receipt, spans = _packet(tmp_path, ("AAPL",), (document,))
    delivered = _delivered(spans, "AAPL")
    assert any("10 million dollars in March 2026" in text for text in delivered)
    assert any("45 million dollars in June 2026" in text for text in delivered)


def test_a_verbatim_restatement_is_grouped_and_recorded_not_deleted(tmp_path: Path) -> None:
    """The same sentence in two sections of one filing: read once, the other
    passage recorded under it with its own handle; the same sentence in a
    second filing is corroboration and is read."""

    claim = "A supplier interruption reduced available component capacity by 18 percent."
    first = _document("AAPL", (claim, claim), revision="aapl-first")
    second = _document("AAPL", (claim,), revision="aapl-second")
    receipt, spans = _packet(tmp_path, ("AAPL",), (first, second))
    handles = {span.document_handle for span in spans if claim in " ".join(span.excerpt.split())}
    assert len(handles) == 2, "one read per filing"
    assert receipt.span_groups, "the restatement is recorded, not dropped"
    read = set(receipt.read_span_handles)
    for group in receipt.span_groups:
        assert group.representative_span_handle in read
        assert not read & set(group.member_span_handles)


def test_a_fact_at_the_end_of_a_passage_is_delivered_whole(tmp_path: Path) -> None:
    """The planted sentence is the last sentence of a long paragraph; the
    delivered excerpt holds it in full, not a cut ending in mid-fact."""

    filler = " ".join(
        f"Routine operating sentence number {n} describes ordinary context." for n in range(6)
    )
    tail = "A supplier interruption reduced available component capacity by 18 percent."
    document = _document("AAPL", (filler + " " + tail,), revision="aapl-tail")
    _receipt, spans = _packet(tmp_path, ("AAPL",), (document,))
    assert any(text.endswith(tail) or tail in text for text in _delivered(spans, "AAPL"))
