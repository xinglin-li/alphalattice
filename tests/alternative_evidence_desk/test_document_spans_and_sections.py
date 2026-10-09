"""Spans, excerpts, document quality and SEC material sections.

A planted claim is retrievable at its exact byte span, ranges are bounded and
delivered as whole sentences, malformed ranges refuse, revisions are
quality-checked and SEC item boundaries are identity-distinct or fail closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from alphalattice.evidence.alternative_evidence.contracts import (
    AlternativeEvidenceClass,
    AlternativeEvidenceReadingDepth,
)
from alphalattice.evidence.alternative_evidence.documents.canonicalization import (
    canonicalize_source_documents,
)
from alphalattice.evidence.alternative_evidence.documents.contracts import DocumentRejectionCode
from alphalattice.evidence.alternative_evidence.retrieval.session import (
    _readable_source_span,
    _unit_declaration,
)
from alphalattice.evidence.alternative_evidence.sources.contracts import AcquiredEvidenceDocument
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from tests.alternative_evidence_desk.document_intelligence_support import (
    _open_recorded,
)
from tests.alternative_evidence_desk.planted_corpus import (
    _NOW,
)


@dataclass(frozen=True)
class _SecSection:
    item_label: str
    canonical_markdown: bytes


@dataclass(frozen=True)
class _SecDiscovery:
    status: str
    unavailable_reason: str | None
    sections: tuple[_SecSection, ...]


def test_a_planted_claim_beyond_the_excerpt_is_retrievable_with_exact_span(
    tmp_path: Path,
) -> None:
    runtime, request, _registry_value, snapshot, document_set, generation = _open_recorded(tmp_path)
    assert "supplier interruption" not in snapshot.citations[0].excerpt.casefold()
    session = runtime.retrieval.open_session(
        document_set=document_set,
        generation=generation,
        evidence_as_of=request.evidence_as_of,
    )
    try:
        result = session.search(query="supplier interruption capacity", top_k=3)
        spans = session.read_spans(span_handles=(result.hits[0].span_handle,))
    finally:
        session.close()
    assert "supplier interruption" in spans[0].excerpt.casefold()
    assert spans[0].start_line <= spans[0].end_line
    assert spans[0].character_end > spans[0].character_start
    assert spans[0].utf8_byte_end - spans[0].utf8_byte_start == len(
        spans[0].excerpt.encode("utf-8")
    )
    revision, content = runtime.documents.library.read_revision(
        document_set.documents[0].workspace_document_id,
        document_set.documents[0].workspace_revision,
    )
    assert revision.content_sha256
    assert (
        content[spans[0].utf8_byte_start : spans[0].utf8_byte_end].decode("utf-8")
        == spans[0].excerpt
    )
    assert result.hits[0].document_handle == document_set.documents[0].semantic_handle


def test_the_widened_session_budgets_bind_at_their_edge(tmp_path: Path) -> None:
    """The widened session budgets bind at their edge."""

    from alphalattice.evidence.alternative_evidence.retrieval.session import (
        MAXIMUM_SEARCHES,
        MAXIMUM_SPAN_READS,
    )

    runtime, request, _registry_value, _snapshot, document_set, generation = _open_recorded(
        tmp_path
    )
    session = runtime.retrieval.open_session(
        document_set=document_set,
        generation=generation,
        evidence_as_of=request.evidence_as_of,
    )
    try:
        assert MAXIMUM_SEARCHES == 24 and MAXIMUM_SPAN_READS == 32
        result = session.search(query="supplier interruption capacity", top_k=3)
        assert result.remaining_searches == MAXIMUM_SEARCHES - 1
        for _ in range(MAXIMUM_SEARCHES - 1):
            result = session.search(query="supplier interruption capacity", top_k=3)
        assert result.remaining_searches == 0
        with pytest.raises(ValueError, match="search_budget_exhausted"):
            session.search(query="supplier interruption capacity", top_k=3)
        handle = result.hits[0].span_handle
        for _ in range(MAXIMUM_SPAN_READS):
            session.read_spans(span_handles=(handle,))
        with pytest.raises(ValueError, match="span_read_budget_exhausted"):
            session.read_spans(span_handles=(handle,))
    finally:
        session.close()


def test_a_long_source_range_is_bounded_without_losing_exact_byte_identity() -> None:
    body = "material downside 事实 " * 2_000
    text = "header\n" + body + "\ntail\n"
    start = text.index(body)
    excerpt, character_start, character_end, was_bounded = _readable_source_span(
        text,
        character_start=start,
        character_end=start + len(body),
        maximum_bytes=4_096,
        context_budget=320,
    )
    assert was_bounded
    assert character_start == start
    assert len(excerpt.encode("utf-8")) <= 4_096
    assert text[character_start:character_end] == excerpt
    encoded = text.encode("utf-8")
    byte_start = len(text[:character_start].encode("utf-8"))
    byte_end = byte_start + len(excerpt.encode("utf-8"))
    assert encoded[byte_start:byte_end].decode("utf-8") == excerpt


def test_a_matched_range_is_delivered_as_whole_sentences() -> None:
    text = (
        "Item 1. Legal Proceedings\n\n"
        "In May 2019, a putative class action was filed against the Company in Delaware. "
        "In August 2023, the court dismissed all claims except the negligence claim. "
        "In March 2025, the court granted the motion to transfer.\n\n"
        "Item 2. Properties\n"
    )
    opening = text.index("In August")
    # a window that begins and ends inside a word, which is what chunking gives
    start, end = opening + 4, opening + 40
    assert text[start:end].startswith("ugust"), text[start:end]

    excerpt, character_start, character_end, was_bounded = _readable_source_span(
        text, character_start=start, character_end=end, maximum_bytes=4_096, context_budget=320
    )

    assert not was_bounded
    assert excerpt == text[character_start:character_end], "the excerpt stays exact source text"
    assert text[start:end] in excerpt, "the retrieved range is never given up"
    assert excerpt.startswith("In August 2023"), excerpt
    assert excerpt.rstrip().endswith("claim."), excerpt
    assert "In May 2019" not in excerpt, "one sentence of context, not the whole section"

    # The matched range wins when the ceiling cannot hold the context too:
    # the added sentences go first and the retrieved range is delivered whole.
    tight, tight_start, tight_end, tight_bounded = _readable_source_span(
        text, character_start=start, character_end=end, maximum_bytes=40, context_budget=320
    )
    assert not tight_bounded, "a range that fits is never truncated"
    assert tight_start <= start and tight_end == end, "the match is delivered whole"
    assert len(tight.encode("utf-8")) <= 40, "the ceiling still binds"
    assert tight == text[tight_start:tight_end]

    # Only a matched range larger than the ceiling on its own is truncated.
    cut, cut_start, cut_end, cut_bounded = _readable_source_span(
        text, character_start=start, character_end=end, maximum_bytes=12, context_budget=320
    )
    assert cut_bounded and cut_start == start and cut_end < end
    assert cut == text[cut_start:cut_end]

    # A range already sitting on sentence edges is returned unchanged.
    sentence = text.index("In March")
    exact, exact_start, exact_end, _ = _readable_source_span(
        text,
        character_start=sentence,
        character_end=text.index("\n\nItem 2"),
        maximum_bytes=4_096,
        context_budget=320,
    )
    assert exact_start == sentence and exact.endswith("transfer.")
    assert exact == text[exact_start:exact_end]


def test_a_bare_amount_carries_the_source_scale_statement_nearest_before_it() -> None:
    """A bare amount carries the source scale statement nearest before it."""

    note = (
        "Note 5. Debt\n\n(In thousands, except share and per share data)\n\n"
        + ("The following table summarizes our indebtedness. " * 40)
        + "As of June 30, 2025, we had $29,595 outstanding under the Facility. "
        + "Note 6. Leases\n\n(in millions)\n\n"
        + "Total lease liabilities were $1,234.5 at period end, and we paid $1.2 billion "
        "in the year. "
    )
    debt = note.index("As of June 30")
    lease = note.index("Total lease")

    declared = _unit_declaration(note, debt, note[debt:lease])
    assert declared is not None
    assert declared.text == "(In thousands, except share and per share data)"
    assert note[declared.character_start : declared.character_end] == declared.text
    assert declared.character_start < debt, "the statement precedes the passage"

    nearest = _unit_declaration(note, lease, note[lease:])
    assert nearest is not None and nearest.text == "(in millions)", "the nearest, not the first"

    assert _unit_declaration(note, lease, "we paid $1.2 billion in the year.") is None, (
        "an amount naming its own scale needs no declaration"
    )
    assert _unit_declaration(note, lease, "a matter filed in 2025 before the court.") is None
    assert _unit_declaration("no scale is stated in this filing at all. ", 20, "$29,595") is None


def test_a_one_cell_layout_table_keeps_its_amount_in_the_sentence() -> None:
    """A one cell layout table keeps its amount in the sentence."""

    from alphalattice.kernel.live_evidence.online_sources import (
        CANONICAL_EXTRACTION_RULES_ID,
        extract_canonical_markdown,
    )

    assert CANONICAL_EXTRACTION_RULES_ID.endswith(".v6")
    amount = (
        b"<html><body><p>On September 3, 2024, the Company entered into a credit agreement "
        b"which provides for a $<table><tr><td><ix:nonFraction name='us-gaap:X' "
        b"contextRef='c1' unitRef='usd' decimals='-6' scale='6'>2,375</ix:nonFraction>"
        b"</td></tr></table> billion unsecured five-year revolving credit facility.</p>"
        b"<p>In September 2025, the Company redeemed $<table><tr><td>500</td></tr></table>"
        b" million and in December 2025 redeemed $<table><tr><td>750</td></tr></table> "
        b"million of notes and paid the related premiums.</p></body></html>"
    )
    text = extract_canonical_markdown(amount, input_cap_bytes=100_000).decode("utf-8")
    assert "$ 2,375 billion unsecured five-year revolving credit facility" in text
    assert "redeemed $ 500 million and in December 2025 redeemed $ 750 million" in text
    real = (
        b"<html><body><p>The following table summarizes the notes outstanding at year end "
        b"for the periods presented below.</p><table><tr><th>Series</th><th>Amount</th></tr>"
        b"<tr><td>2028 Notes</td><td>$<table><tr><td>500</td></tr></table></td></tr></table>"
        b"<p>Second paragraph with enough words to look like narrative text for the "
        b"extractor to keep going.</p></body></html>"
    )
    text = extract_canonical_markdown(real, input_cap_bytes=100_000).decode("utf-8")
    # Without a table policy a real table is not carried and says so; the
    # layout cell inside it is that table's cell, never narrative.
    assert "[Table of 2 rows not carried into the canonical text; the original retains it.]" in text
    assert "500" not in text and "Second paragraph" in text
    text = extract_canonical_markdown(
        real, input_cap_bytes=100_000, tables=_CarryEverywhere()
    ).decode("utf-8")
    assert "Series: 2028 Notes; Amount: $500" in text


class _CarryEverywhere:
    """A table policy that admits every table: the rendering alone under test."""

    def observe(self, text: str) -> None:
        return None

    def carries(self) -> bool:
        return True


def test_the_sec_table_policy_admits_disclosure_items_only() -> None:
    """The SEC table policy admits disclosure items only."""

    from alphalattice.evidence.alternative_evidence.documents.canonicalization import (
        TABLE_CARRY_RULES_ID,
        SecTableCarry,
    )

    assert TABLE_CARRY_RULES_ID.endswith(".v1")
    quarterly = SecTableCarry("10-Q")
    assert not quarterly.carries(), "front matter"
    quarterly.observe("PART I. FINANCIAL INFORMATION")
    quarterly.observe("Item 1. Financial Statements")
    assert not quarterly.carries(), "the statements and their notes"
    quarterly.observe("Note 7 - Commitments and Contingencies")
    assert not quarterly.carries(), "a note heading changes nothing"
    quarterly.observe("Item 2. Management's Discussion and Analysis")
    assert not quarterly.carries()
    quarterly.observe("ITEM 4. CONTROLS AND PROCEDURES")
    assert quarterly.carries()
    quarterly.observe("PART II. OTHER INFORMATION")
    assert quarterly.carries(), "Part II before its first item"
    for heading in ("ITEM 1. LEGAL PROCEEDINGS.", "Item 2. Unregistered Sales", "Item 5. Other"):
        quarterly.observe(heading)
        assert quarterly.carries(), heading
    quarterly.observe("Item 6. Exhibits")
    assert not quarterly.carries(), "the exhibit index and the signatures after it"
    annual = SecTableCarry("10-K/A")
    assert not annual.carries()
    for heading, carried in (
        ("PART I", False),
        ("Item 1. Business", True),
        ("Item 1C. Cybersecurity", True),
        ("Item 3. Legal Proceedings", True),
        ("Item 5. Market for Registrant's Common Equity", True),
        ("Item 7. Management's Discussion and Analysis", False),
        ("Item 7A. Quantitative and Qualitative Disclosures", False),
        ("Item 8. Financial Statements and Supplementary Data", False),
        ("Item 9B. Other Information", True),
        ("PART III", False),
        ("Item 11. Executive Compensation.", False),
        ("PART IV", False),
        ("Item 15. Exhibits and Financial Statement Schedules", False),
        ("Item 1.Financial Statements.", False),
    ):
        annual.observe(heading)
        assert annual.carries() is carried, heading
    current = SecTableCarry("8-K")
    assert not current.carries()
    current.observe("Item 2.02 Results of Operations and Financial Condition")
    assert current.carries()
    current.observe("A paragraph of the item that is not a heading at all.")
    assert current.carries()
    # A current report's item heading is the item's official title, at its
    # own length: 5.02 runs to 156 characters after the number and 2.03 to
    # 124; measured on the retained book, five of sixty-nine item lines had
    # fallen into the preceding item's region under the general bound.
    from alphalattice.evidence.alternative_evidence.documents.structure import heading_kind

    officers = (
        "Item 5.02 Departure of Directors or Certain Officers; Election of Directors; "
        "Appointment of Certain Officers; Compensatory Arrangements of Certain Officers."
    )
    obligation = (
        "Item 2.03 Creation of a Direct Financial Obligation or an Obligation under an "
        "Off-Balance Sheet Arrangement of a Registrant."
    )
    assert len(officers) > 120 and len(obligation) > 120
    assert heading_kind(officers, document_type="8-K") == ("ITEM", "5.02")
    assert heading_kind(obligation, document_type="8-K") == ("ITEM", "2.03")
    assert heading_kind(f"**{officers}**", document_type="8-K") == ("ITEM", "5.02")
    assert heading_kind(officers + " " + "x" * 40, document_type="8-K") is None, (
        "longer than any official title"
    )
    assert heading_kind("Item 1A. " + "Risk factors " * 12, document_type="10-K") is None, (
        "an annual report's bound is unchanged"
    )
    fresh = SecTableCarry("8-K")
    fresh.observe(officers)
    assert fresh.carries(), "the item's tables are carried under its own heading"
    # The structure's own scan applies the same bounds (it had its own copy
    # of the general bound, so the rule and the scan disagreed on the book
    # journey's delta filing): the item is a heading and opens its region.
    from alphalattice.evidence.alternative_evidence.documents.structure import DocumentStructure

    report = "\n\n".join(
        (
            "UNITED STATES SECURITIES AND EXCHANGE COMMISSION",
            "FORM 8-K",
            obligation,
            "On September 19, 2026, the registrant entered into a term loan credit agreement "
            "providing for a $500 million term loan facility maturing in 2029.",
            officers,
            "On September 19, 2026, the chief financial officer resigned.",
            "Item 9.01 Financial Statements and Exhibits.",
            "None.",
        )
    )
    structure = DocumentStructure(report, document_type="8-K")
    items = [heading.text[:9] for heading in structure.headings if heading.kind == "ITEM"]
    assert items == ["Item 2.03", "Item 5.02", "Item 9.01"]
    annual = DocumentStructure(
        "\n\n".join(("FORM 10-K", "Item 1A. " + "Risk factors " * 12, "prose")),
        document_type="10-K",
    )
    assert [h for h in annual.headings if h.kind == "ITEM"] == [], "the annual bound is unchanged"


def test_the_cover_ends_in_the_front_matter_not_at_a_later_cover_like_line() -> None:
    """The cover ends in the front matter not at a later cover like line."""

    from alphalattice.evidence.alternative_evidence.analysis.matters import litigation_regions
    from alphalattice.evidence.alternative_evidence.documents.structure import (
        STRUCTURE_RULES_ID,
        DocumentStructure,
    )

    assert STRUCTURE_RULES_ID.endswith(".v4")
    text = (
        "UNITED STATES SECURITIES AND EXCHANGE COMMISSION\n\nFORM 10-K\n\n"
        "Commission File Number 1-8841\n\nIndicate by check mark whether the registrant is a "
        "shell company (as defined in Rule 12b-2 of the Act). Yes No\n\n"
        "PART I\n\nItem 1. Business\n\nThe registrants generate electricity in ordinary "
        "course and describe the business here at length for the reader.\n\n"
        "Item 3. Legal Proceedings\n\nSee Note 15 - Legal Proceedings for the matters the "
        "registrants are party to.\n\n"
        "Item 4. Mine Safety Disclosures\n\nNot applicable.\n\n"
        "PART II\n\nItem 8. Financial Statements and Supplementary Data\n\n"
        "15. Commitments and Contingencies\n\nLegal Proceedings - In March 2024, a lawsuit "
        "was filed against FPL in the Circuit Court alleging damages from an outage; FPL "
        "believes the claims are without merit.\n\n"
        "16. Segment Information\n\nThe segments are described here.\n\n"
        "PART IV\n\nItem 15. Exhibits\n\nNo annual report, proxy statement, form of proxy or "
        "other proxy soliciting material has been sent to security holders of FPL during the "
        "period covered by this Annual Report on Form 10-K.\n"
    )
    structure = DocumentStructure(text, document_type="10-K")
    assert structure.cover_end < text.index("PART I")
    regions = litigation_regions(text, structure)
    assert [region.heading for region in regions] == [
        "Item 3. Legal Proceedings",
        "15. Commitments and Contingencies",
    ]
    # A lettered note the word Note names (HRL, MLM) is a note heading; a
    # bare letter is a sub-item, as before.
    lettered = text.replace(
        "15. Commitments and Contingencies", "NOTE K - COMMITMENTS AND CONTINGENCIES"
    ).replace("16. Segment Information", "NOTE L - SEGMENT INFORMATION")
    structure = DocumentStructure(lettered, document_type="10-K")
    regions = litigation_regions(lettered, structure)
    assert [region.heading for region in regions][1] == "NOTE K - COMMITMENTS AND CONTINGENCIES"
    assert regions[1].end_basis.startswith("next NOTE heading: NOTE L")
    bare = lettered.replace("NOTE K - COMMITMENTS", "K. COMMITMENTS")
    structure = DocumentStructure(bare, document_type="10-K")
    assert [h.kind for h in structure.headings if h.text.startswith("K. COMMITMENTS")] != ["NOTE"]


def test_the_report_period_is_the_cover_statement_never_a_year_the_body_names() -> None:
    """The report period is the cover statement never a year the body names."""

    from alphalattice.evidence.alternative_evidence.documents.structure import (
        DocumentStructure,
    )

    body = (
        "PART I\n\nItem 1. Business\n\nThe incentive plan measures performance for the "
        "fiscal year ended December 31, 2028 and each year after it.\n\n"
        "PART II\n\nItem 8. Financial Statements\n\nRevenue for the year ended "
        "December 31, 2024 is compared below.\n"
    )
    cover = "FORM 10-K\n\nFor the fiscal year ended December 31, 2025\n\n"
    annual = DocumentStructure(cover + body, document_type="10-K")
    assert annual.report_period.status == "RESOLVED"
    assert str(annual.report_period.period_end) == "2025-12-31"
    assert annual.report_period.candidates == ("2025-12-31",)
    unstated = DocumentStructure("FORM 10-K\n\nAnnual report\n\n" + body, document_type="10-K")
    assert unstated.report_period.status == "ABSENT"
    assert unstated.report_period.period_end is None
    # A quarterly report's comparisons name the prior year's quarter in the
    # same words: its cover alone decides, and one cover date resolves.
    quarterly = DocumentStructure(
        "FORM 10-Q\n\nFor the quarterly period ended June 30, 2026\n\n"
        "PART I\n\nItem 2. Management's Discussion\n\nSales for the quarterly period "
        "ended June 30, 2025 were lower.\n",
        document_type="10-Q",
    )
    assert quarterly.report_period.status == "RESOLVED"
    assert str(quarterly.report_period.period_end) == "2026-06-30"


def _cell(text: str = "", *, span: int = 1) -> str:
    attribute = f' colspan="{span}"' if span > 1 else ""
    return f"<td{attribute}>{text}</td>"


def test_tables_reach_the_canonical_text_as_self_describing_rows() -> None:
    """Tables reach the canonical text as self describing rows."""

    from alphalattice.evidence.alternative_evidence.documents.canonicalization import (
        SecTableCarry,
    )
    from alphalattice.kernel.live_evidence.online_sources import (
        TABLE_ROW_LIMIT,
        extract_canonical_markdown,
    )

    lead = (
        "<p>The following table sets forth information relating to repurchases of our equity "
        "securities during the three months ended June 30, 2026:</p>"
    )
    statement = (
        "<table><tr><td></td><td>2026</td><td>2025</td></tr>"
        "<tr><td>Revenues</td><td>1,234</td><td>1,100</td></tr>"
        "<tr><td>Net income</td><td>210</td><td>180</td></tr></table>"
    )
    heading = (
        "<tr>"
        + "".join(
            _cell(text, span=3)
            for text in (
                "Period",
                "",
                "Total Number<br/>of Shares (or Units) Purchased",
                "",
                "Average<br/>Price Paid per<br/>Share (or Unit) (1)",
                "",
                "Maximum Number (or Approximate Dollar Value) of Shares that May Yet Be "
                "Purchased Under the Plans or Programs (Dollars in Billions)",
            )
        )
        + "</tr>"
    )
    units = "<tr>" + _cell("(in millions, except share and per share data)", span=21) + "</tr>"

    def row(
        period: str, shares: str, mark: str, price: str, remaining: str, sign: str = "$"
    ) -> str:
        return (
            "<tr>"
            + _cell(period, span=3)
            + _cell("", span=3)
            + _cell(shares, span=2)
            + _cell()
            + _cell(mark, span=3)
            + _cell(sign)
            + _cell(price)
            + _cell()
            + _cell("", span=3)
            + _cell(sign)
            + _cell(remaining)
            + _cell()
            + "</tr>"
        )

    table = (
        "<table>"
        + heading
        + units
        + row("April 1, 2026 \u2013", "7,056,917", "(2)", "178.49", "16.9")
        + row("April 30, 2026", "124", "(3)", "174.04", "N/A", sign="")
        + row("May 1, 2026 - May 31, 2026", "\u2014", "", "\u2014", "10,699,052")
        + row("Total", "7,057,041", "", "", "")
        + "</table>"
    )
    layout = "<table><tr><td>ITEM 1.</td><td>LEGAL PROCEEDINGS.</td></tr></table>"
    large = (
        "<table>"
        + "".join(
            f"<tr><td>Line {index}</td><td>{index * 7:,}</td></tr>"
            for index in range(TABLE_ROW_LIMIT + 5)
        )
        + "</table>"
    )
    contents = (
        "<table>"
        + "".join(
            f"<tr><td>Item {index}.</td><td>Section {index}</td><td>{index + 2}</td></tr>"
            for index in range(1, 7)
        )
        + "</table>"
    )
    html = (
        "<html><body>"
        + contents
        + "<p>PART I. FINANCIAL INFORMATION</p><p>Item 1. Financial Statements</p>"
        + statement
        + "<p>The notes to the statements follow and explain the amounts presented above.</p>"
        + "<p>PART II. OTHER INFORMATION</p>"
        + layout
        + "<p>Litigation is described in Note 7 of the financial statements below.</p>"
        + "<p>Item 2. Unregistered Sales of Equity Securities and Use of Proceeds</p>"
        + lead
        + table
        + "<p>(1) Average price paid per share excludes commissions.</p>"
        + "<p>Item 5. Other Information</p>"
        + large
        + "<p>Closing paragraph with enough words for the extractor to keep as narrative text.</p>"
        "</body></html>"
    ).encode("utf-8")
    text = extract_canonical_markdown(
        html, input_cap_bytes=1_000_000, tables=SecTableCarry("10-Q")
    ).decode("utf-8")
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    assert "ITEM 1. LEGAL PROCEEDINGS." in lines, "a one-row layout table is the line it shows"
    # The statement in Part I Item 1 is not carried: the text says a table
    # stood there and the original retains it.
    assert "Revenues" not in text and "1,234" not in text
    marker = "[Table of 3 rows not carried into the canonical text; the original retains it.]"
    assert lines.index(marker) < lines.index("PART II. OTHER INFORMATION")
    assert "(in millions, except share and per share data)" in lines, "a units row keeps its place"
    rows = [line for line in lines if line.startswith("Period:")]
    assert rows[0] == (
        "Period: April 1, 2026 \u2013; Total Number of Shares (or Units) Purchased: 7,056,917 (2); "
        "Average Price Paid per Share (or Unit) (1): $178.49; Maximum Number (or Approximate "
        "Dollar Value) of Shares that May Yet Be Purchased Under the Plans or Programs (Dollars "
        "in Billions): $16.9"
    )
    assert rows[1].startswith(
        "Period: April 30, 2026; Total Number of Shares (or Units) Purchased: 124 (3); "
        "Average Price Paid per Share (or Unit) (1): 174.04;"
    )
    assert rows[1].endswith("(Dollars in Billions): N/A")
    # A dash stays a dash, a blank cell says nothing, a total row is the
    # source's own and nothing is summed or merged across rows.
    assert rows[2] == (
        "Period: May 1, 2026 - May 31, 2026; Total Number of Shares (or Units) Purchased: \u2014; "
        "Average Price Paid per Share (or Unit) (1): $ \u2014; Maximum Number (or Approximate "
        "Dollar Value) of Shares that May Yet Be Purchased Under the Plans or Programs (Dollars "
        "in Billions): $10,699,052"
    )
    assert rows[3] == "Period: Total; Total Number of Shares (or Units) Purchased: 7,057,041"
    assert len(rows) == 4 and text.index(rows[0]) > text.index("repurchases of our equity")
    assert text.index("(1) Average price paid") > text.index(rows[3]), "source order kept"
    assert "[Table of contents (6 rows) not carried into the canonical text.]" in lines
    assert (
        f"[Table of {TABLE_ROW_LIMIT + 5} rows not carried into the canonical text; "
        "the original retains it.]"
    ) in lines
    assert "Line 3" not in text and "Section 4" not in text
    # A heading row repeated after a page break is not a record.
    repeated = (
        b"<html><body><p>PART II. OTHER INFORMATION</p><p>Item 2. Unregistered Sales</p>"
        b"<table><tr><td>Period</td><td>Shares</td></tr><tr><td>April</td><td>10</td></tr>"
        b"<tr><td>Period</td><td>Shares</td></tr><tr><td>May</td><td>20</td></tr></table>"
        b"<p>Closing paragraph with enough words for the extractor to keep as narrative text.</p>"
        b"</body></html>"
    )
    text = extract_canonical_markdown(
        repeated, input_cap_bytes=1_000_000, tables=SecTableCarry("10-Q")
    ).decode("utf-8")
    rows = [line for line in text.split("\n") if line.startswith("Period:")]
    assert rows == ["Period: April; Shares: 10", "Period: May; Shares: 20"]


FRT_COVENANT_CELL = (
    "on or before the 90th day after the original issuance of the Notes, file a shelf "
    "registration statement (which will be an automatic shelf registration statement if the "
    "Parent is then a well-known seasoned issuer (\u201cWKSI\u201d)) or a resale prospectus "
    "supplement to an effective shelf registration statement with the Securities and Exchange "
    "Commission (the \u201cSEC\u201d) providing for the registration of, and the sale on a "
    "continuous or delayed basis by the holders of the common shares, if any, issuable upon "
    "exchange of the Notes;"
)
"""A 519-character cell of FRT's retained 8-K (a registration-rights covenant);
its qualification "if any" and its object sit past the 400th character. The
retained original sets it in a headingless bullet table that is not carried;
the headed table below is a development shape built from that real cell."""


def test_a_long_cell_is_never_cut_silently() -> None:
    """A long cell is never cut silently."""

    from alphalattice.evidence.alternative_evidence.documents.canonicalization import (
        SecTableCarry,
    )
    from alphalattice.kernel.live_evidence.online_sources import (
        _TABLE_CELL_LIMIT,
        extract_canonical_markdown,
    )

    assert len(FRT_COVENANT_CELL) > _TABLE_CELL_LIMIT
    short_cell = "file the required exhibits with the Commission within four business days."
    html = (
        "<html><body><p>Item 8.01 Other Events</p>"
        "<p>The registration rights agreement requires the Parent to take the following steps:</p>"
        "<table><tr><td>Covenant</td><td>Terms</td></tr>"
        f"<tr><td>Registration</td><td>{FRT_COVENANT_CELL}</td></tr>"
        f"<tr><td>Reporting</td><td>{short_cell}</td></tr></table>"
        "<p>Closing paragraph with enough words for the extractor to keep as narrative text.</p>"
        "</body></html>"
    ).encode()
    text = extract_canonical_markdown(
        html, input_cap_bytes=1_000_000, tables=SecTableCarry("8-K")
    ).decode("utf-8")
    rows = [line for line in text.split("\n") if line.startswith("Covenant:")]
    assert len(rows) == 2
    assert rows[1] == f"Covenant: Reporting; Terms: {short_cell}", "a cell within the size is whole"
    registration = rows[0]
    # Cut at a word before the size; what the original retains is counted.
    kept = FRT_COVENANT_CELL[: FRT_COVENANT_CELL.rfind(" ", 0, _TABLE_CELL_LIMIT)].rstrip()
    omitted = len(FRT_COVENANT_CELL) - len(kept)
    assert registration == (
        f"Covenant: Registration; Terms: {kept} [cell continues: {omitted} more characters "
        "not carried; the original retains them]"
    ), "the cut is said in the cell, never silent"
    assert "if any" not in registration, "the omitted qualification is not pretended"


def test_a_spanning_cell_keeps_its_grid_position() -> None:
    """A spanning cell keeps its grid position."""

    from alphalattice.evidence.alternative_evidence.documents.canonicalization import (
        SecTableCarry,
    )
    from alphalattice.kernel.live_evidence.online_sources import extract_canonical_markdown

    # FRT's dividends table, reduced to its shape: a blank corner cell and the
    # dividend heading span two heading rows; "High" and "Low" sit under
    # "Price Per Share".
    dividends = (
        "<table>"
        '<tr><td rowspan="2" colspan="3"></td><td colspan="9">Price Per Share</td>'
        '<td colspan="3"></td><td rowspan="2" colspan="3">Dividends Declared Per Share</td></tr>'
        '<tr><td colspan="3">High</td><td colspan="3"></td><td colspan="3">Low</td></tr>'
        '<tr><td colspan="3">2025</td></tr>'
        '<tr><td colspan="3">Fourth quarter</td><td>$</td><td>102.81</td><td colspan="4"></td>'
        '<td>$</td><td>90.03</td><td colspan="4"></td><td>$</td><td>1.130</td></tr>'
        '<tr><td colspan="3">Third quarter</td><td>$</td><td>102.94</td><td colspan="4"></td>'
        '<td>$</td><td>89.99</td><td colspan="4"></td><td>$</td><td>1.130</td></tr>'
        "</table>"
    )
    # A body subject spanning two records: a holder with two dated lots.
    lots = (
        "<table><tr><td>Holder</td><td>Date Adopted</td><td>Shares</td></tr>"
        '<tr><td rowspan="2">Jane Q. Rivera</td><td>May 8, 2026</td><td>12,000</td></tr>'
        "<tr><td>June 2, 2026</td><td>4,000</td></tr></table>"
    )
    html = (
        "<html><body><p>PART II</p><p>Item 5. Market for Registrant's Common Equity</p>"
        "<p>The following table sets forth the high and low sales prices and dividends.</p>"
        + dividends
        + "<p>Item 9B. Other Information</p>"
        "<p>The following arrangements were adopted during the quarter:</p>"
        + lots
        + "<p>Closing paragraph with enough words for the extractor to keep as narrative text.</p>"
        "</body></html>"
    ).encode("utf-8")
    text = extract_canonical_markdown(
        html, input_cap_bytes=1_000_000, tables=SecTableCarry("10-K")
    ).decode("utf-8")
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    quarter = next(line for line in lines if "Fourth quarter" in line)
    assert quarter == (
        "Fourth quarter; Price Per Share High: $102.81; Price Per Share Low: $90.03; "
        "Dividends Declared Per Share: $1.130"
    ), quarter
    assert "High: Fourth quarter" not in text, "the quarter is not a high price"
    first = next(line for line in lines if "May 8, 2026" in line)
    second = next(line for line in lines if "June 2, 2026" in line)
    assert first == "Holder: Jane Q. Rivera; Date Adopted: May 8, 2026; Shares: 12,000"
    assert second == "Holder: Jane Q. Rivera; Date Adopted: June 2, 2026; Shares: 4,000", second
    assert "Holder: June 2, 2026" not in text, "a date is not a holder"


def test_inline_text_beside_a_replaced_table_is_kept() -> None:
    """Inline text beside a replaced table is kept."""

    from alphalattice.kernel.live_evidence.online_sources import extract_canonical_markdown

    narrative = "".join(
        f"<p>Paragraph {i} of ordinary narrative text that describes the business of the "
        "registrant in enough words to be kept by the extractor as content.</p>"
        for i in range(12)
    )
    html = (
        "<html><body><p>FORM 10-Q</p>"
        '<div style="text-align:center"><span>Mark one:</span><table><tr><td>\u2612</td>'
        "<td>QUARTERLY REPORT PURSUANT TO SECTION 13</td></tr></table>"
        "<span>For the quarterly period ended June 30, 2026</span></div>"
        "<p>Commission file number: 1-07533</p>" + narrative + "</body></html>"
    ).encode("utf-8")
    text = extract_canonical_markdown(html, input_cap_bytes=1_000_000).decode("utf-8")
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    assert lines[:4] == [
        "FORM 10-Q",
        "Mark one:",
        "\u2612 QUARTERLY REPORT PURSUANT TO SECTION 13",
        "For the quarterly period ended June 30, 2026",
    ]


def test_a_div_paragraph_among_p_paragraphs_is_read() -> None:
    """A div paragraph among p paragraphs is read."""

    from alphalattice.kernel.live_evidence.online_sources import extract_canonical_markdown

    conclusion = (
        "Under the supervision and with the participation of our management, we conducted "
        "an evaluation of our disclosure controls and procedures. Based on this evaluation, "
        "our principal executive officer and our principal financial officer concluded that "
        "our disclosure controls and procedures were effective."
    )
    div_paragraph = (
        '<div style="text-indent:36pt;"><span style="display:inline-block;">(a)</span>'
        "<i>Disclosure Controls and Procedures</i>"
        f'<span style="white-space:pre-wrap;">.  {conclusion}</span></div>'
    )
    narrative = "".join(
        f"<p>Paragraph {i} of ordinary narrative text that describes the business of the "
        "registrant in enough words to be kept by the extractor as content.</p>"
        for i in range(12)
    )
    mixed = (
        "<html><body><p>ITEM 4. CONTROLS AND PROCEDURES.</p>"
        + div_paragraph
        + narrative
        + "</body></html>"
    ).encode("utf-8")
    text = extract_canonical_markdown(mixed, input_cap_bytes=1_000_000).decode("utf-8")
    assert "concluded that our disclosure controls and procedures were effective" in text
    assert text.index("(a)") < text.index("Paragraph 0")
    # A filing of div paragraphs keeps the extractor's div path: nothing is
    # converted, and every paragraph is still read.
    all_divs = (
        "<html><body><div><span>ITEM 4. CONTROLS AND PROCEDURES.</span></div>"
        + div_paragraph
        + narrative.replace("<p>", "<div><span>").replace("</p>", "</span></div>")
        + "</body></html>"
    ).encode("utf-8")
    text = extract_canonical_markdown(all_divs, input_cap_bytes=1_000_000).decode("utf-8")
    assert "concluded that our disclosure controls and procedures were effective" in text
    assert "Paragraph 11 of ordinary narrative" in text


def test_a_linked_paragraph_set_as_a_div_closes_before_the_next_block() -> None:
    """A linked paragraph set as a div closes before the next block."""

    from lxml import etree

    from alphalattice.evidence.alternative_evidence.analysis.financing import financing_regions
    from alphalattice.evidence.alternative_evidence.documents.structure import DocumentStructure
    from alphalattice.kernel.live_evidence.online_sources import (
        _linked_paragraph_divs,
        extract_canonical_markdown,
    )

    def div(text: str, *, bold: bool = False) -> str:
        style = "font-weight:700" if bold else "font-weight:400"
        return f'<div style="margin-top:9pt"><span style="{style}">{text}</span></div>'

    fair_value = (
        "There were no material fair value adjustments to these items during 2025 and 2024. "
        'Please see <span><a href="#note1">Note 1</a></span><span> for additional '
        "information.</span>"
    )
    borrowings = (
        "The Company maintains various short-term bank credit facilities, with a borrowing "
        "capacity of $1,220 and $1,198, in 2025 and 2024. Short-term borrowings outstanding at "
        "the end of 2025 and 2024 were immaterial."
    )
    contents = "".join(
        f'<div><span><a href="#i{n}">Item {n}. Heading {n}</a></span><span> {n * 3}</span></div>'
        for n in range(1, 6)
    )
    filing = (
        "<html><body>"
        + div("PART II")
        + div("ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA", bold=True)
        + div("Note 3—Fair Value Measurement", bold=True)
        + div(fair_value)
        + div("Note 4—Debt", bold=True)
        + div("Short-Term Borrowings")
        + div(borrowings)
        + div("Note 5—Leases", bold=True)
        + div("The Company leases land and buildings under operating leases.")
        + "</body></html>"
    ).encode("utf-8")
    text = extract_canonical_markdown(filing, input_cap_bytes=1_000_000).decode("utf-8")
    lines = text.split("\n")
    assert "Note 4—Debt" in lines, text
    assert "Please see Note 1 for additional information." in text
    assert not any(line.strip() == "Note 1" for line in lines), "the reference is no heading"
    assert lines.index("Note 4—Debt") > lines.index(
        next(line for line in lines if line.startswith("There were no material"))
    )
    assert " ".join(text.split()).count("Note 4—Debt") == 1
    words = " ".join(text.split())
    for sentence in (
        "There were no material fair value adjustments",
        "Please see Note 1 for additional information.",
        "borrowing capacity of $1,220 and $1,198",
        "Short-Term Borrowings",
        "The Company leases land and buildings",
    ):
        assert sentence in words
    structure = DocumentStructure(text, document_type="10-K")
    assert [region.heading for region in financing_regions(text, structure)] == ["Note 4—Debt"]
    # The helper converts the linked prose paragraph only: a block of links
    # (a table of contents set as divs) keeps the extractor's own reading, and
    # a paragraph without a link is left to the extractor's div path.
    tree = etree.fromstring(
        ("<html><body>" + contents + div(fair_value) + div(borrowings) + "</body></html>").encode(),
        etree.HTMLParser(),
    )
    assert _linked_paragraph_divs(tree) == 1
    converted = [element for element in tree.iter("p")]
    assert len(converted) == 1 and "Please see" in "".join(converted[0].itertext())
    assert sum(1 for element in tree.iter("div") if "Heading" in "".join(element.itertext())) == 5


def test_inline_xbrl_is_unwrapped_and_its_hidden_header_is_never_prose() -> None:
    """Inline XBRL is unwrapped and its hidden header is never prose."""

    from alphalattice.kernel.live_evidence.online_sources import extract_canonical_markdown

    def amount(value: str, *, sign: str = "") -> str:
        return (
            f'<ix:nonFraction unitRef="usd" contextRef="c1" decimals="-5" '
            f'format="ixt:num-dot-decimal" name="dg:GiftCardsOutstandingLiability" '
            f'scale="6" id="Narr_{value}"{sign}>{value}</ix:nonFraction>'
        )

    prefix = (
        "The Company recognizes gift card sales revenue at the time of redemption, and the "
        "liability for gift cards is established for the cash value at the time of purchase "
        "of the gift card, which management reviews each period against redemption "
        "experience. "
    ) * 3
    paragraphs = "".join(
        f'<p style="font-family:Times;font-size:10pt;margin:0pt;">{prefix} The liability for '
        f"outstanding gift cards was approximately ${amount(f'1{index}.5')}&#160;million and "
        f"${amount(f'1{index}.4')}&#160;million at January 30, 2026 and January 31, 2025, "
        "respectively, and is recorded in accrued expenses and other liabilities.</p>"
        for index in range(12)
    )
    signed = (
        '<p style="margin:0pt;">Net cash used was $('
        + amount("41.2", sign=' sign="-"')
        + ") million in the period, a decrease from the prior year.</p>"
    )
    hidden = (
        '<div style="display:none"><ix:header><ix:hidden>'
        '<ix:nonNumeric name="dei:AmendmentFlag" contextRef="c1">false</ix:nonNumeric>'
        '<ix:nonNumeric name="dei:DocumentFiscalYearFocus" contextRef="c1">2025</ix:nonNumeric>'
        "</ix:hidden><ix:references><link:schemaRef xlink:href='dg-20260130.xsd'/>"
        "</ix:references><ix:resources><xbrli:context id='c1'><xbrli:entity>"
        "<xbrli:identifier scheme='http://www.sec.gov/CIK'>0000029534</xbrli:identifier>"
        "</xbrli:entity></xbrli:context></ix:resources></ix:header></div>"
    )
    html = (
        '<?xml version="1.0" encoding="ASCII"?><html xmlns="http://www.w3.org/1999/xhtml" '
        'xmlns:ix="http://www.xbrl.org/2013/inlineXBRL"><head><title>t</title></head><body>'
        + hidden
        + paragraphs
        + signed
        + "</body></html>"
    ).encode("utf-8")
    text = extract_canonical_markdown(html, input_cap_bytes=1_000_000).decode("utf-8")
    assert text.count("respectively, and is recorded in accrued expenses") == 12
    assert "approximately $10.5 million and $10.4 million at January 30, 2026" in text
    assert "was $(41.2) million in the period" in text, "the sign as displayed, nothing added"
    assert "AmendmentFlag" not in text and "0000029534" not in text and "false" not in text
    assert "2025" in text and "dg-20260130.xsd" not in text


def test_a_malformed_character_range_refuses() -> None:
    text = "alpha\nbravo\ncharlie\n"
    for character_start, character_end in ((-1, 4), (4, 4), (6, 3), (0, len(text) + 1)):
        with pytest.raises(ValueError, match="span_character_range_invalid"):
            _readable_source_span(
                text,
                character_start=character_start,
                character_end=character_end,
                maximum_bytes=4_096,
                context_budget=320,
            )


def test_document_quality_rejects_empty_and_duplicate_revisions() -> None:
    def source(handle: str, content: bytes, *, revision: str) -> AcquiredEvidenceDocument:
        return AcquiredEvidenceDocument(
            semantic_handle=handle,
            entity_id="AAPL",
            source_name="ISSUER_RECORDED",
            source_right="USER_PROVIDED_FOR_LOCAL_RESEARCH",
            evidence_class=AlternativeEvidenceClass.ISSUER_OFFICIAL_RECORDED,
            document_type="EARNINGS_RELEASE",
            revision=revision,
            title="AAPL release",
            published_at=_NOW - timedelta(days=2),
            accepted_at=None,
            available_at=_NOW - timedelta(days=1),
            media_type="text/plain",
            content=content,
            immutable_source=True,
            source_content_hash=canonical_hash(
                {
                    "source_name": "ISSUER_RECORDED",
                    "revision": revision,
                    "media_type": "text/plain",
                    "content_hex": content.hex(),
                }
            ),
        )

    admitted, rejected = canonicalize_source_documents(
        (
            source("DOC-AAPL-001", b"Narrative evidence.", revision="same-revision"),
            source("DOC-AAPL-002", b"Duplicate evidence.", revision="same-revision"),
            source("DOC-AAPL-003", b" ", revision="empty-revision"),
            source("DOC-AAPL-004", b"\xff\xfe", revision="invalid-encoding"),
        )
    )
    assert len(admitted) == 1
    assert tuple(value.code for value in rejected) == (
        DocumentRejectionCode.SOURCE_IDENTITY_INVALID,
        DocumentRejectionCode.EMPTY,
        DocumentRejectionCode.INVALID_ENCODING,
    )


def _sec_source(handle: str, revision: str, content: bytes, *, document_type: str) -> Any:
    return AcquiredEvidenceDocument(
        semantic_handle=handle,
        entity_id="AAPL",
        source_name="SEC_EDGAR",
        source_right="SEC_PUBLIC_OFFICIAL_ACCESS",
        evidence_class=AlternativeEvidenceClass.SEC_FILING,
        document_type=document_type,
        revision=revision,
        title="AAPL annual filing",
        published_at=_NOW - timedelta(days=2),
        accepted_at=_NOW - timedelta(days=2),
        available_at=_NOW - timedelta(days=2),
        media_type="text/html",
        content=content,
        immutable_source=True,
        source_content_hash=canonical_hash(
            {
                "source_name": "SEC_EDGAR",
                "revision": revision,
                "media_type": "text/html",
                "content_hex": content.hex(),
            }
        ),
    )


def test_sec_material_sections_are_bounded_and_identity_distinct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    background = "Ordinary business background and general description. " * 120
    detail = "Material liquidity, legal, market, and risk evidence. " * 12
    markdown = (
        f"# Item 1. Business\n{background}\n"
        f"# Item 1A. Risk Factors\n{detail}\n"
        f"# Item 2. Properties\n{background}\n"
        f"# Item 3. Legal Proceedings\n{detail}\n"
        f"# Item 7. Management's Discussion and Analysis\n{detail}\n"
        f"# Item 7A. Quantitative and Qualitative Disclosures About Market Risk\n{detail}\n"
        f"# Item 8. Financial Statements\n{background}\n"
        f"## Subsequent Events\n{detail}\n"
    )
    monkeypatch.setattr("trafilatura.extract", lambda *_args, **_kwargs: markdown)
    source = _sec_source(
        "DOC-AAPL-10K",
        "0000320193-26-000001",
        b"<html><body>bounded filing</body></html>",
        document_type="10-K",
    )

    full, full_rejected = canonicalize_source_documents(
        (source,),
        reading_depth=AlternativeEvidenceReadingDepth.FULL_FILING,
    )
    material, material_rejected = canonicalize_source_documents(
        (source,),
        reading_depth=AlternativeEvidenceReadingDepth.MATERIAL_SECTIONS,
        sec_section_discovery=lambda *_args, **_kwargs: _SecDiscovery(
            status="AVAILABLE",
            unavailable_reason=None,
            sections=(
                _SecSection("ITEM_1A", f"# Item 1A. Risk Factors\n{detail}\n".encode()),
                _SecSection("ITEM_3", f"# Item 3. Legal Proceedings\n{detail}\n".encode()),
                _SecSection(
                    "ITEM_7",
                    f"# Item 7. Management's Discussion and Analysis\n{detail}\n".encode(),
                ),
                _SecSection(
                    "ITEM_7A",
                    (
                        "# Item 7A. Quantitative and Qualitative Disclosures About Market Risk\n"
                        f"{detail}\n"
                    ).encode(),
                ),
                _SecSection(
                    "SUBSEQUENT_EVENTS_1",
                    f"## Subsequent Events\n{detail}\n".encode(),
                ),
            ),
        ),
    )

    assert not full_rejected and not material_rejected
    assert material[0].byte_count <= full[0].byte_count * 0.30
    assert material[0].reading_depth is AlternativeEvidenceReadingDepth.MATERIAL_SECTIONS
    assert full[0].reading_depth is AlternativeEvidenceReadingDepth.FULL_FILING
    assert material[0].canonical_content_hash != full[0].canonical_content_hash
    assert "ITEM_1A" in material[0].section_labels
    assert b"Material liquidity" in material[0].canonical_markdown
    assert b"Ordinary business background" not in material[0].canonical_markdown
    assert b"Ordinary business background" in full[0].canonical_markdown


def test_sec_material_sections_fail_closed_when_item_boundaries_are_ambiguous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repeated = "Substantive filing narrative. " * 20
    markdown = (
        f"# Item 1A. Risk Factors\n{repeated}\n"
        f"# Item 2. Properties\n{repeated}\n"
        f"# Item 1A. Risk Factors\n{repeated}\n"
        f"# Item 3. Legal Proceedings\n{repeated}\n"
    )
    monkeypatch.setattr("trafilatura.extract", lambda *_args, **_kwargs: markdown)
    source = _sec_source(
        "DOC-AAPL-AMBIGUOUS",
        "0000320193-26-000002",
        b"<html><body>ambiguous filing</body></html>",
        document_type="10-K/A",
    )

    admitted, rejected = canonicalize_source_documents(
        (source,),
        reading_depth=AlternativeEvidenceReadingDepth.MATERIAL_SECTIONS,
        sec_section_discovery=lambda *_args, **_kwargs: _SecDiscovery(
            status="UNAVAILABLE",
            unavailable_reason="AMBIGUOUS_BOUNDARIES",
            sections=(),
        ),
    )
    assert not admitted
    assert tuple(value.code for value in rejected) == (
        DocumentRejectionCode.SECTION_EXTRACTION_UNAVAILABLE,
    )


def test_a_heading_longer_than_a_heading_can_be_is_carried_as_a_paragraph() -> None:
    """Text exceeding the heading-length bound is carried as a paragraph."""

    from alphalattice.kernel.live_evidence.online_sources import (
        CANONICAL_EXTRACTION_RULES_ID,
        extract_canonical_markdown,
    )

    footnote = (
        "In accordance with Item 601(b)(32)(ii) of Regulation S-K and SEC Release Nos. "
        "33-8238 and 34-47986, Final Rule: Management's Reports on Internal Control Over "
        "Financial Reporting and Certification of Disclosure in Exchange Act Periodic "
        "Reports, the certifications furnished in Exhibits 32.1 and 32.2 hereto are "
        "deemed to accompany this Annual Report on Form 10-K and will not be deemed filed."
    )
    assert len(footnote) > 256
    html = (
        "<html><body><h2>Item 15. Exhibits and Financial Statement Schedules</h2>"
        "<p>The following exhibits are filed as part of this report.</p>"
        f"<h3>{footnote}</h3><p>Ordinary narrative after the footnote.</p></body></html>"
    ).encode()
    text = extract_canonical_markdown(html, input_cap_bytes=1_000_000).decode()
    assert CANONICAL_EXTRACTION_RULES_ID == "live-evidence.canonical-markdown.v6"
    assert "## Item 15. Exhibits and Financial Statement Schedules" in text
    assert f"\n{footnote}\n" in text and f"# {footnote}" not in text
    assert all(len(line.lstrip("# ")) <= 256 for line in text.splitlines() if line.startswith("#"))


def test_an_item_heading_closing_with_a_scale_parenthetical_is_the_item() -> None:
    """An item heading closing with a scale parenthetical is the item."""

    from alphalattice.evidence.alternative_evidence.documents.structure import (
        STRUCTURE_RULES_ID,
        DocumentStructure,
        heading_kind,
    )

    assert STRUCTURE_RULES_ID.endswith(".v4")
    heading = (
        "Item 7\u2014Management's Discussion and Analysis of Financial Condition and Results of "
        "Operations (amounts in millions, except per share, share, percentages and warehouse "
        "count data)"
    )
    assert len(heading) > 160
    text = "\n\n".join(
        (
            "UNITED STATES SECURITIES AND EXCHANGE COMMISSION",
            "FORM 10-K",
            "PART II",
            "Item 6\u2014Reserved",
            heading,
            "Overview",
            "We believe that our cash and investment positions will be sufficient to meet our "
            "liquidity and capital requirements for the foreseeable future.",
            "Item 8\u2014Financial Statements and Supplementary Data",
            "The consolidated financial statements follow.",
        )
    )
    structure = DocumentStructure(text, document_type="10-K")
    items = {region.item: region for region in structure.item_regions()}
    assert set(items) == {"6", "7", "8"}
    assert items["7"].title.startswith("Management's Discussion and Analysis")
    assert items["7"].heading_end - items["7"].heading_start < len(heading), (
        "the heading is the title; the parenthetical is not part of it"
    )
    unit = next(h for h in structure.headings if h.kind == "UNIT")
    assert (
        unit.text.startswith("(amounts in millions")
        and text[unit.character_start : unit.character_end] == unit.text
    )
    statement = text.index("We believe that our cash")
    window = structure.locate(statement, statement + 40)
    assert window.path[1].startswith("Item 7") and window.unit_declaration is not None
    assert window.unit_declaration[0].startswith("(amounts in millions")
    assert heading_kind(heading, document_type="10-K") == ("ITEM", "7")
    long_paragraph = "Our warehouses " + "and depots " * 20 + "(in thousands of square feet)"
    assert len(long_paragraph) > 160 and heading_kind(long_paragraph, document_type="10-K") is None


def test_a_filing_that_labels_no_item_heads_its_items_by_their_official_titles() -> None:
    """A filing that labels no item heads its items by their official titles."""

    from alphalattice.evidence.alternative_evidence.documents.structure import DocumentStructure

    bare = "\n\n".join(
        (
            "UNITED STATES SECURITIES AND EXCHANGE COMMISSION",
            "FORM 10-K",
            "Business",
            "The Company is a consumer financial services company.",
            "Risk Factors",
            "Our business is subject to risks.",
            "Cybersecurity",
            "Risk Management and Strategy",
            "Cybersecurity threats have not materially affected the Company during the past "
            "three fiscal years.",
            "Legal Proceedings",
            "See Note 18. Legal Proceedings and Regulatory Matters.",
            "Controls and Procedures",
            "Evaluation of Disclosure Controls and Procedures",
            "Our Chief Executive Officer and Chief Financial Officer concluded that our "
            "disclosure controls and procedures were effective as of December 31, 2025.",
            "Other Information",
            "Rule 10b5-1 Trading Plans",
            "Certain of our directors adopted trading plans.",
            "Legal Proceedings",
            "A later repeat of a title inside the exhibits.",
        )
    )
    structure = DocumentStructure(bare, document_type="10-K")
    regions = {region.item: region for region in structure.item_regions()}
    assert set(regions) == {"1", "1A", "1C", "3", "9A", "9B"}
    assert regions["9A"].part == "PART II" and regions["1C"].part == "PART I"
    assert regions["9A"].title == "Controls and Procedures"
    promoted = [h for h in structure.headings if h.kind == "ITEM"]
    assert all(h.implied_part is not None for h in promoted)
    assert all(
        bare[h.character_start : h.character_end] == h.text.split(". ", 1)[1] for h in promoted
    ), "the range is the source line; the item number is the rule's label"
    later = bare.rindex("Legal Proceedings")
    assert structure.locate(later, later + 20).path[:2] == (
        "PART II",
        "Item 9B. Other Information",
    ), "the repeat stays inside Other Information, under its implied part"
    conclusion = bare.index("concluded that our disclosure controls")
    assert (
        structure.locate(conclusion, conclusion + 20).path[1] == "Item 9A. Controls and Procedures"
    )
    labelled = bare.replace("Business\n", "Item 1. Business\n", 1)
    with_label = DocumentStructure(labelled, document_type="10-K")
    assert [r.item for r in with_label.item_regions()] == ["1"], (
        "a labelled filing keeps its bare titles as sub-headings"
    )
    few = "\n\n".join(("FORM 10-K", "Cybersecurity", "Text.", "Controls and Procedures", "Text."))
    assert DocumentStructure(few, document_type="10-K").item_regions() == ()


def test_a_headingless_filing_s_cover_ends_before_its_body() -> None:
    """A headingless filing's cover ends before its body."""

    from alphalattice.evidence.alternative_evidence.documents.structure import DocumentStructure

    body = (
        "The Annual Meeting of the Company's Shareholders was held on May 28, 2026. The "
        "following are the final voting results on proposals considered and voted upon by "
        "the shareholders at the meeting, as certified by the inspector of election."
    )
    text = "\n\n".join(
        (
            "UNITED STATES SECURITIES AND EXCHANGE COMMISSION",
            "FORM 8-K",
            "Check the appropriate box below if the Form 8-K filing is intended to "
            "simultaneously satisfy the filing obligation of the registrant under any of the "
            "following provisions.",
            "Emerging growth company",
            body,
            "The appointment of the independent registered public accounting firm was ratified.",
            "(c) Shell company transactions. N/A",
            "SIGNATURE",
        )
    )
    structure = DocumentStructure(text, document_type="8-K")
    assert not any(h.kind in {"PART", "ITEM"} for h in structure.headings)
    assert structure.cover_end < text.index(body)
    assert text.index("was ratified") > structure.cover_end
    # A filing with headings keeps the earlier rule: its cover is every
    # cover-like line before the first Part or Item heading.
    with_item = text.replace(
        body, "Item 5.07 Submission of Matters to a Vote of Security Holders.\n\n" + body
    )
    labelled = DocumentStructure(with_item, document_type="8-K")
    assert labelled.cover_end < with_item.index("Item 5.07")
