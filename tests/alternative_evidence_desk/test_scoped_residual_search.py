"""The residual search's scope is an owner-enforced filter, not a cut of the
final order: a scoped search considers, fuses and scores only the chunks of
the named source ranges, so a passage outside the scope is never a pair the
cross-encoder reads and never a hit; the pairs each search spends are
reported for the budget; and `scope_for` names a region's chunks from exact
character ranges without a query or a model call."""

from __future__ import annotations

from pathlib import Path

from tests.alternative_evidence_desk.document_intelligence_support import (
    _document_with_text,
    _open_recorded,
)
from tests.alternative_evidence_desk.litigation_support import BOILERPLATE, _filing, _matter

_DEBT = (
    "NOTE 9. DEBT\n\nIn March 2025 the Company entered into a revolving credit facility of "
    "$500 million maturing in 2030; no borrowings were outstanding under it at year end. The "
    "facility bears interest at SOFR plus a margin and requires the Company to maintain a "
    "leverage ratio below 3.5 to 1.0, with which the Company was in compliance.\n\n"
)


def _two_filings() -> tuple[str, str]:
    legal = _filing(note_lines=[BOILERPLATE, _matter(1), _matter(2)])
    debt = legal.replace("NOTE 8. SEGMENT INFORMATION", _DEBT + "NOTE 8. SEGMENT INFORMATION")
    return legal, debt


def test_a_scoped_search_scores_and_returns_only_the_scoped_chunks(tmp_path: Path) -> None:
    """requirement (6E): the scope holds before the channel limits and the
    reranker: fewer pairs are scored than an unscoped search of the same
    question, every hit lies inside a scoped range, and a scope over a
    document that never mentions the question returns no hit from it."""

    legal, debt = _two_filings()
    runtime, request, _registry, _snapshot, document_set, generation = _open_recorded(
        tmp_path,
        entities=("AAPL", "MSFT"),
        documents=(
            _document_with_text("AAPL", text=legal, form="10-Q", revision="aapl-10q"),
            _document_with_text("MSFT", text=debt, form="10-Q", revision="msft-10q"),
        ),
    )
    try:
        session = runtime.retrieval.open_session(
            document_set=document_set, generation=generation, evidence_as_of=request.evidence_as_of
        )
        try:
            documents = {d.entity_id: d for d in session.inspect_documents()}
            msft = documents["MSFT"]
            note_start = msft.text.index("NOTE 9. DEBT")
            note_end = msft.text.index("NOTE 8. SEGMENT INFORMATION")
            question = "Was there a revolving credit facility and were borrowings outstanding?"
            everywhere = session.search(query=question, top_k=5, reranker_depth=12)
            scope = session.scope_for({msft.document_id: [(note_start, note_end)]})
            assert scope and all(entry[0] == msft.document_id for entry in scope)
            scoped = session.search(query=question, top_k=5, reranker_depth=12, scope=scope)
            assert everywhere.reranked_pairs > scoped.reranked_pairs > 0
            assert session.reranked_pairs == everywhere.reranked_pairs + scoped.reranked_pairs
            assert scoped.hits and all(hit.entity_id == "MSFT" for hit in scoped.hits)
            read = session.read_spans(span_handles=tuple(h.span_handle for h in scoped.hits[:4]))
            for span in read:
                assert span.character_start < note_end and note_start < span.character_end, (
                    "every scoped hit lies inside the scoped note"
                )
            # The same question scoped to the legal filing, which never
            # mentions a facility: no hit is invented from it.
            aapl = documents["AAPL"]
            legal_scope = session.scope_for({aapl.document_id: [(0, len(aapl.text))]})
            elsewhere = session.search(
                query=question, top_k=5, reranker_depth=12, scope=legal_scope
            )
            assert all(hit.entity_id == "AAPL" for hit in elsewhere.hits)
            assert not any("credit facility" in hit.preview for hit in elsewhere.hits)
        finally:
            session.close()
    finally:
        runtime.close()


def test_scope_for_names_the_chunks_that_overlap_the_ranges(tmp_path: Path) -> None:
    """A range inside one chunk names that chunk; adjacent chunks merge into
    one entry; a document with no overlapping chunk contributes nothing."""

    legal, debt = _two_filings()
    runtime, request, _registry, _snapshot, document_set, generation = _open_recorded(
        tmp_path,
        entities=("AAPL", "MSFT"),
        documents=(
            _document_with_text("AAPL", text=legal, form="10-Q", revision="aapl-10q"),
            _document_with_text("MSFT", text=debt, form="10-Q", revision="msft-10q"),
        ),
    )
    try:
        session = runtime.retrieval.open_session(
            document_set=document_set, generation=generation, evidence_as_of=request.evidence_as_of
        )
        try:
            msft = next(d for d in session.inspect_documents() if d.entity_id == "MSFT")
            whole = session.scope_for({msft.document_id: [(0, len(msft.text))]})
            assert len(whole) == 1, "one merged entry for the whole filing"
            first = whole[0][1]
            assert first < whole[0][2], "the kernel's ordinals, first to last"
            first_chunk = session.scope_for({msft.document_id: [(0, 1)]})
            assert first_chunk == ((msft.document_id, first, first),)
            assert session.scope_for({"unknown-document": [(0, 10)]}) == ()
            assert session.scope_for({msft.document_id: []}) == ()
        finally:
            session.close()
    finally:
        runtime.close()
