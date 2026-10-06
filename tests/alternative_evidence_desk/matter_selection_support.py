"""The fixture book of the matter-selection policy tests: for the book's
first issuer, a 10-Q whose contingencies note holds more matter windows
than one session's allowance (long matters, since the candidate packs
adjacent openings into one window), the corporate-event 10-K, an 8-K, and
-- with the debt note -- one more 10-K holding a financing region. Real
filings' shapes, not their text."""

from __future__ import annotations

from datetime import timedelta

from alphalattice.evidence.alternative_evidence.sources.recorded import RecordedEvidenceDocument
from tests.alternative_evidence_desk.document_intelligence_support import _document_with_text
from tests.alternative_evidence_desk.event_support import _current_report, _event_filing
from tests.alternative_evidence_desk.financing_support import _financing_filing
from tests.alternative_evidence_desk.litigation_support import BOILERPLATE, _filing, _matter
from tests.alternative_evidence_desk.planted_corpus import _NOW


def _long_matter(number: int) -> str:
    """One matter longer than a window: the candidate packs adjacent openings
    into one window, so only long matters outrun a session's allowance."""

    return (
        _matter(number)
        + " "
        + " ".join(
            f"On March {day}, 2025, the court entered scheduling order number {day} in the "
            f"Patent Action {number}, setting the next deadline."
            for day in range(1, 29)
        )
    )


def _filings(entities: tuple[str, ...]) -> tuple[RecordedEvidenceDocument, ...]:
    """For the book's first issuer: a 10-Q whose note holds more matter
    windows than one session's allowance, the event 10-K and an 8-K."""

    first = entities[0]
    text = _filing(note_lines=[BOILERPLATE, *(_long_matter(n) for n in range(1, 60))])
    return (
        _document_with_text(first, text=text, form="10-Q", revision="10-q-2026"),
        _document_with_text(first, text=_event_filing(), form="10-K", revision="10-k-2025"),
        _document_with_text(first, text=_current_report(), form="8-K", revision="8-k-2026"),
    )


def _filings_with_debt(entities: tuple[str, ...]) -> tuple[RecordedEvidenceDocument, ...]:
    """The same book with one more 10-K of the first issuer: a debt note."""

    return (
        *_filings(entities),
        _document_with_text(
            entities[0], text=_financing_filing(), form="10-K", revision="10-k-2025-debt"
        ),
    )


def _repeating_filings() -> tuple[RecordedEvidenceDocument, ...]:
    """The second issuer's two 10-Qs: the same two matters restated word for
    word, the later filing published a quarter after the earlier."""

    text = _filing(note_lines=[BOILERPLATE, _matter(1), _matter(2)])
    earlier = _document_with_text("MSFT", text=text, form="10-Q", revision="10-q-2025-q4")
    later = _document_with_text("MSFT", text=text, form="10-Q", revision="10-q-2026-q1")
    return (
        earlier.model_copy(update={"published_at": _NOW - timedelta(days=100)}),
        later.model_copy(update={"published_at": _NOW - timedelta(days=10)}),
    )
