"""Bounded table-and-text evidence through the verified read boundary: a
placeholder of the canonical text becomes a table view rendered from the
retained original, matched by position and row count, paged by whole rows
with its headings, caption, scale and footnote, bound to the original's
content hash and the parser, under the matter allowance; a placeholder
whose original is not retained, does not verify or does not correspond is
refused by name, never a view; a recorded text has no view."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from alphalattice.evidence.alternative_evidence.documents.canonicalization import SecTableCarry
from alphalattice.evidence.alternative_evidence.documents.structure import DocumentStructure
from alphalattice.evidence.alternative_evidence.documents.tables import (
    TABLE_VIEW_RULES_ID,
    TableViewError,
    render_table_view,
    table_catalogue,
)
from alphalattice.evidence.alternative_evidence.retrieval.contracts import (
    AlternativeEvidenceResolvedSpan,
)
from alphalattice.kernel.live_evidence.online_sources import not_carried_tables, table_lines
from tests.alternative_evidence_desk.document_intelligence_support import (
    _document_with_text,
    _open_recorded,
)
from tests.alternative_evidence_desk.incremental_acquisition_support import (
    AAPL,
    REGISTRY,
    TEN_K,
    _acquire,
    _request,
)
from tests.alternative_evidence_desk.planted_corpus import _runtime
from tests.alternative_evidence_desk.sec_scenario_transport import SecScenarioTransport
from tests.alternative_evidence_desk.table_view_support import (
    INSTRUMENTS,
    canonical_text,
    original_html,
    year_headed_html,
)


def test_the_catalogue_and_the_rendering_agree_with_the_original() -> None:
    """requirement (6B): the placeholder names the table's row count, its
    caption, footnote, heading path, family and scale statement; the same
    walk of the original finds the one uncarried table; the view renders
    every row over its headings with caption, table note, scale, columns
    and footnote; a page under a small ceiling holds whole rows with the
    headings repeated and names what remains; a row count that disagrees or
    a table without headings is refused by name."""

    html = original_html()
    text = canonical_text(html)
    structure = DocumentStructure(text, document_type="10-K")
    catalogue = table_catalogue(text, structure)
    assert len(catalogue) == 1
    placeholder = catalogue[0]
    assert placeholder.ordinal == 1 and placeholder.rows_total == 9
    assert placeholder.path[-1] == "NOTE 9. DEBT" and placeholder.family == "FINANCING_CAPITAL"
    assert placeholder.caption is not None
    assert "consisted of the following" in text[slice(*placeholder.caption)]
    assert len(placeholder.footnotes) == 1
    assert text[slice(*placeholder.footnotes[0])].startswith("(1) The revolving credit facility")
    assert placeholder.unit_declaration is not None
    assert placeholder.unit_declaration[0] == "(In millions, except per share data)"
    tables = not_carried_tables(html, SecTableCarry("10-K"))
    assert [len(rows) for rows in tables] == [9]
    sha = hashlib.sha256(html).hexdigest()
    view = render_table_view(
        html,
        parent_source_content_hash=sha,
        policy=SecTableCarry("10-K"),
        placeholder=placeholder,
        text=text,
        byte_ceiling=3900,
    )
    assert view.rules_id == TABLE_VIEW_RULES_ID and view.rows_total == len(INSTRUMENTS)
    assert view.rows_from == 1 and view.rows_to == len(INSTRUMENTS) and view.remaining_rows == 0
    assert view.headings == (
        "Instrument",
        "December 31, 2025 Principal",
        "December 31, 2024 Principal",
        "Maturity",
    )
    assert view.notes == ("Long-term debt (in millions)",)
    assert view.scale == "(In millions, except per share data)" and view.caption.startswith("The")
    assert view.footnotes[0].startswith("(1) The revolving credit facility")
    assert "Instrument: Term loan facility; December 31, 2025 Principal: $300" in view.text
    assert "Instrument: Revolving credit facility" in view.text
    assert view.text.endswith("[rows 1-6 of 6]")
    # A ceiling that holds two rows: whole rows, headings repeated, the rest named.
    small = render_table_view(
        html,
        parent_source_content_hash=sha,
        policy=SecTableCarry("10-K"),
        placeholder=placeholder,
        text=text,
        byte_ceiling=len("\n".join(view.lines[:4]).encode("utf-8")) + 340,
    )
    assert 1 <= small.rows_to < len(INSTRUMENTS) and small.remaining_rows > 0
    assert "Columns: Instrument" in small.text and small.text.endswith(
        f"; {small.remaining_rows} more row(s) not in this page]"
    )
    next_page = render_table_view(
        html,
        parent_source_content_hash=sha,
        policy=SecTableCarry("10-K"),
        placeholder=placeholder,
        text=text,
        rows_from=small.rows_to + 1,
        byte_ceiling=3900,
    )
    assert next_page.rows_from == small.rows_to + 1 and next_page.rows_to == len(INSTRUMENTS)
    assert "Columns: Instrument" in next_page.text, "the headings repeat on every page"
    # Correspondence: the placeholder must state the table's own row count.
    wrong = replace(placeholder, rows_total=8)
    with pytest.raises(TableViewError, match="table_view_correspondence_unproved"):
        render_table_view(
            html,
            parent_source_content_hash=sha,
            policy=SecTableCarry("10-K"),
            placeholder=wrong,
            text=text,
            byte_ceiling=3900,
        )
    beyond = replace(placeholder, ordinal=2)
    with pytest.raises(TableViewError, match="table_view_correspondence_unproved"):
        render_table_view(
            html,
            parent_source_content_hash=sha,
            policy=SecTableCarry("10-K"),
            placeholder=beyond,
            text=text,
            byte_ceiling=3900,
        )
    with pytest.raises(TableViewError, match="table_view_rows_invalid"):
        render_table_view(
            html,
            parent_source_content_hash=sha,
            policy=SecTableCarry("10-K"),
            placeholder=placeholder,
            text=text,
            rows_from=99,
            byte_ceiling=3900,
        )


def test_a_table_view_is_delivered_through_the_verified_session_read(tmp_path: Path) -> None:
    """requirement (6B): a live-acquired original stands behind its canonical
    document; the session issues a view handle under the matter budget at
    the placeholder's exact range, reads it as any span (the placeholder
    range proved against the canonical bytes), renders the page from the
    verified original and binds the span to the original's content hash
    and the parser; the span set seals and reads back; a tampered original
    is refused by name; a recorded text has no original and no view."""

    html = original_html()
    transport = SecScenarioTransport(
        registry={"AAPL": REGISTRY["AAPL"]},
        filings={AAPL: [TEN_K]},
        bodies={SecScenarioTransport.locator(AAPL, TEN_K): html},
    )
    runtime = _runtime(tmp_path)
    try:
        request = _request(("AAPL",), budget=3)
        _registry, _snapshot, source_set = _acquire(runtime, transport, request)
        document_set = runtime.canonicalize(
            source_set=source_set, published_at=request.evidence_as_of
        )
        assert len(document_set.documents) == 1
        generation = runtime.build_retrieval(
            document_set=document_set, built_at=request.evidence_as_of
        )
        session = runtime.open_session(
            document_set=document_set, generation=generation, evidence_as_of=request.evidence_as_of
        )
        try:
            (document,) = session.inspect_documents()
            assert session.original_available(document.document_id)
            placeholders = session.table_placeholders(document.document_id)
            assert len(placeholders) == 1
            (handle,) = session.issue_table_views(
                document.document_id, ((placeholders[0], 1),), series="X01"
            )
            assert handle == "SPAN-X01-R0001" and session.issued_matter_window_count == 1
            (span,) = session.read_table_views(span_handles=(handle,))
            assert session.matter_read_count == 1
            assert span.table_view is not None
            assert span.table_view.rules_id == TABLE_VIEW_RULES_ID
            assert span.table_view.parent_source_content_hash == hashlib.sha256(html).hexdigest()
            assert (
                span.table_view.rows_total == len(INSTRUMENTS)
                and span.table_view.remaining_rows == 0
            )
            assert (
                span.table_view.headings[0] == "Instrument" and span.table_view.footnote_count == 1
            )
            assert span.character_start == placeholders[0].character_start
            assert span.character_end == placeholders[0].character_end
            assert "Instrument: Revolving credit facility" in span.excerpt
            assert "Footnote: (1) The revolving credit facility" in span.excerpt
            assert span.unit_declaration is not None
            assert span.unit_declaration.text == "(In millions, except per share data)"
            assert any("rendering of the retained original" in value for value in span.limitations)
            # The span seals into a set, is placed under the runtime's own
            # admission (the set and a receipt together; here the set alone,
            # the receipt of this session being none) and reads back as the
            # same object.
            sealed = runtime._seal_span_set(request, generation, (span,))
            runtime.admit_storage(len(runtime.artifacts.serialized(sealed)))
            runtime.artifacts.publish("resolved-span-sets", sealed.span_set_hash, sealed)
            loaded = runtime.artifacts.load(
                "resolved-span-sets", sealed.span_set_hash, type(sealed)
            )
            assert loaded.spans[0] == span
            payload = span.model_dump(mode="json")
            assert AlternativeEvidenceResolvedSpan.model_validate(payload) == span
            with pytest.raises(ValueError, match="resolved_span_view_invalid"):
                AlternativeEvidenceResolvedSpan.model_validate(
                    {**payload, "span_handle": "SPAN-M01-R0001"}
                )
            # The packet reader proves the view again when the packet is
            # reopened -- the placeholder range against the canonical text,
            # the original against the binding's content hash, the excerpt
            # against the page rendered again -- and refuses by name what
            # it cannot prove (the book journey's first coverage read had
            # refused every view as a source mismatch).
            from alphalattice.evidence.alternative_evidence.runtime.task_adapter import (
                _verify_table_view,
            )

            (canonical,) = document_set.documents
            _revision, content = runtime.documents.library.read_revision(
                canonical.workspace_document_id, canonical.workspace_revision
            )
            text = content.decode("utf-8")
            original = runtime.original_reader(document_set)(canonical.workspace_document_id)
            assert original is not None
            assert original[1] == span.table_view.parent_source_content_hash
            _verify_table_view(span, text=text, reference=canonical, original=original)
            with pytest.raises(ValueError, match="brief_span_source_mismatch"):
                _verify_table_view(
                    span.model_copy(update={"excerpt": span.excerpt.replace("Footnote", "Note")}),
                    text=text,
                    reference=canonical,
                    original=original,
                )
            with pytest.raises(ValueError, match="brief_table_view_original_mismatch"):
                _verify_table_view(
                    span,
                    text=text,
                    reference=canonical,
                    original=(original[0] + b" ", original[1], original[2]),
                )
            with pytest.raises(ValueError, match="brief_table_view_original_unavailable"):
                _verify_table_view(span, text=text, reference=canonical, original=None)
            with pytest.raises(ValueError, match="brief_table_view_placeholder_mismatch"):
                _verify_table_view(
                    span.model_copy(
                        update={
                            "table_view": span.table_view.model_copy(update={"table_ordinal": 2})
                        }
                    ),
                    text=text,
                    reference=canonical,
                    original=original,
                )
        finally:
            session.close()
        # A tampered original: the view refuses by name, nothing is rendered.
        reference = next(iter(source_set.documents))
        path = runtime.documents.workspace.root / Path(*reference.content_object_path.split("/"))
        original_bytes = path.read_bytes()
        tampered = original_bytes.replace(b"Term loan facility", b"Term loan facilitY", 1)
        assert tampered != original_bytes
        path.write_bytes(tampered)
        try:
            session = runtime.open_session(
                document_set=document_set,
                generation=generation,
                evidence_as_of=request.evidence_as_of,
            )
            try:
                (document,) = session.inspect_documents()
                placeholders = session.table_placeholders(document.document_id)
                (handle,) = session.issue_table_views(
                    document.document_id, ((placeholders[0], 1),), series="X01"
                )
                with pytest.raises(ValueError, match="source_object"):
                    session.read_table_views(span_handles=(handle,))
            finally:
                session.close()
        finally:
            path.write_bytes(original_bytes)
    finally:
        runtime.close()
    # A recorded text: the placeholder is catalogued, no original stands
    # behind it, and the view is refused by name -- a representation gap.
    text = canonical_text(html)
    runtime, request, _registry, _snapshot, document_set, generation = _open_recorded(
        tmp_path / "recorded",
        entities=("AAPL",),
        documents=(_document_with_text("AAPL", text=text, form="10-K", revision="aapl-10k"),),
    )
    try:
        session = runtime.open_session(
            document_set=document_set, generation=generation, evidence_as_of=request.evidence_as_of
        )
        try:
            (document,) = session.inspect_documents()
            assert not session.original_available(document.document_id)
            placeholders = session.table_placeholders(document.document_id)
            assert len(placeholders) == 1
            (handle,) = session.issue_table_views(
                document.document_id, ((placeholders[0], 1),), series="X01"
            )
            with pytest.raises(TableViewError, match="table_view_original_unavailable"):
                session.read_table_views(span_handles=(handle,))
        finally:
            session.close()
    finally:
        runtime.close()


def test_a_leading_row_of_years_is_the_view_s_heading_row_and_the_canonical_text_stays() -> None:
    """requirement (section 7 of the completeness assignment; view rules
    v2): the one demonstrated general shape of the refused tables -- a
    leading row of years, or of years beside words, read as a body row by
    the numeric test -- is read as the heading row by the table view: the
    view's headings are the source's years, every row reads over them, the
    rules id names v2; the canonical rendering keeps the table uncarried,
    so its placeholder and every canonical byte stay as they were; a first
    row that holds a number that is not a year (a maturity schedule's
    '2026 | 300') stays refused by name."""

    assert TABLE_VIEW_RULES_ID == "alternative-evidence.table-view.v2"
    html = year_headed_html()
    text = canonical_text(html)
    structure = DocumentStructure(text, document_type="10-K")
    catalogue = table_catalogue(text, structure)
    tables = not_carried_tables(html, SecTableCarry("10-K"))
    assert [p.rows_total for p in catalogue] == [3, 4, 3], (
        "every year-headed table is still uncarried by the canonical rendering"
    )
    assert "[Table of 3 rows not carried" in text and "Operating lease cost" not in text
    plain, labelled, schedule = catalogue
    for placeholder in (plain, labelled):
        rows = tables[placeholder.ordinal - 1]
        assert table_lines(rows) is None, "the canonical reading has no heading row here"
        view = render_table_view(
            html,
            parent_source_content_hash=hashlib.sha256(html).hexdigest(),
            policy=SecTableCarry("10-K"),
            placeholder=placeholder,
            text=text,
            byte_ceiling=3900,
        )
        assert (
            view.rules_id == TABLE_VIEW_RULES_ID and view.rows_total == placeholder.rows_total - 1
        )
    view = render_table_view(
        html,
        parent_source_content_hash=hashlib.sha256(html).hexdigest(),
        policy=SecTableCarry("10-K"),
        placeholder=plain,
        text=text,
        byte_ceiling=3900,
    )
    assert view.headings == ("2025", "2024")
    assert "Operating lease cost; 2025: $32; 2024: $34" in view.lines
    view = render_table_view(
        html,
        parent_source_content_hash=hashlib.sha256(html).hexdigest(),
        policy=SecTableCarry("10-K"),
        placeholder=labelled,
        text=text,
        byte_ceiling=3900,
    )
    assert view.headings == ("2025", "2024", "% Change")
    assert any("% Change: 15.3" in line for line in view.lines)
    with pytest.raises(TableViewError, match="table_view_headings_absent"):
        render_table_view(
            html,
            parent_source_content_hash=hashlib.sha256(html).hexdigest(),
            policy=SecTableCarry("10-K"),
            placeholder=schedule,
            text=text,
            byte_ceiling=3900,
        )
