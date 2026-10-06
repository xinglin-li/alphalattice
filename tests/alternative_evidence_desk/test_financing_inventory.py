"""The financing-disclosure inventory (`financing.py`) under its frozen
definition: regions from the filing's own headings (a securities note held
as an asset and the MD&A are not regions), units the source's paragraphs
open when they name an instrument, program or facility -- the paragraph
whole, titled by the first instrument as written and listing every other,
a bulleted list continuing its lead, a sub-heading naming the group -- the
explicit negative kept as text and marked, the covenant statement that
only refers to the notes left to the region unscoped, the table placeholder
unassigned as not carried, no amount or balance parsed anywhere; and the
family delivered through the integrated selection beside the others under the one
allowance with `M01:FINANCING` provenance."""

from __future__ import annotations

import re
from pathlib import Path

from alphalattice.evidence.alternative_evidence.analysis import packet as packet_module
from alphalattice.evidence.alternative_evidence.analysis.contracts import ProvisionalMatterRecord
from alphalattice.evidence.alternative_evidence.analysis.financing import (
    FINANCING_CUE,
    FINANCING_FAMILY,
    FINANCING_INVENTORY_RULES_ID,
    REFERS_TO_FINANCING,
    financing_inventory,
    financing_regions,
)
from alphalattice.evidence.alternative_evidence.analysis.matters import (
    document_needs,
    region_statements,
)
from alphalattice.evidence.alternative_evidence.documents.structure import DocumentStructure
from alphalattice.evidence.alternative_evidence.retrieval.session import (
    MAXIMUM_ISSUED_MATTER_WINDOWS,
)
from tests.alternative_evidence_desk.document_intelligence_support import (
    _document_with_text as _document,
)
from tests.alternative_evidence_desk.document_intelligence_support import (
    _open_recorded,
)
from tests.alternative_evidence_desk.event_support import _current_report, _event_filing
from tests.alternative_evidence_desk.financing_support import (
    CAPACITY_BULLET_2,
    CAPACITY_LEAD,
    CAPACITY_NEGATIVE,
    COMMERCIAL_PAPER,
    COVENANT,
    FOOTNOTE,
    LINES_OF_CREDIT,
    LONG_TERM_LEAD,
    MATURED,
    MDA_LIQUIDITY,
    MIXED_POLARITY,
    NO_PAPER,
    PRONOMINAL_ABSENCE,
    PRONOMINAL_DRAW,
    QUALIFIED_NEGATIVE,
    SECURITIES_NOTE,
    TABLE,
    TWO_IN_ONE_CLAUSE,
    TWO_NOTES,
    _financing_filing,
)
from tests.alternative_evidence_desk.litigation_support import _matters_filing

HANDLE = re.compile(r"^M-[A-Z0-9-]{1,48}-[0-9]{3}$")
AMOUNT = re.compile(r"^\s*(?:\$|\d[\d,.]*\s*(?:million|billion)?\s*$)")
"""A bare amount, balance or capacity: what no unit may carry as a field."""


def test_financing_units_are_the_source_paragraphs_that_name_an_instrument() -> None:
    """requirement (4/5): the debt note is the region and the securities note
    and the MD&A are not; each paragraph naming an instrument is one unit
    titled by the first instrument as written with every other listed; the
    stated absence is a unit of its own basis; the covenant statement that
    only refers to the notes is a cued region statement with its scope not
    stated; the table placeholder is unassigned as not carried; the
    sub-heading names the group; the bulleted capacity list continues its
    lead; no amount, rate or balance is parsed into any field."""

    text = _financing_filing()
    structure = DocumentStructure(text, document_type="10-K")
    inventory = financing_inventory(text, structure, document_id="T")
    assert inventory.rules_id == FINANCING_INVENTORY_RULES_ID
    assert [r.heading for r in inventory.regions] == ["NOTE 11: DEBT AND BORROWING ARRANGEMENTS"]
    assert all(r.family == FINANCING_FAMILY and not r.end_uncertain for r in inventory.regions)
    region = inventory.regions[0]
    for outside in (SECURITIES_NOTE, MDA_LIQUIDITY):
        assert not (region.body_start <= text.index(outside) < region.body_end)
    assert (
        financing_regions(
            _current_report(), DocumentStructure(_current_report(), document_type="8-K")
        )
        == ()
    )
    units = {text[u.ranges[0][0] : u.ranges[0][1]][:24]: u for u in inventory.matters}
    assert all(HANDLE.match(u.handle) and "-F-" in u.handle for u in inventory.matters)
    assert all(u.family == FINANCING_FAMILY and u.named for u in inventory.matters)
    paper = units[COMMERCIAL_PAPER[:24]]
    assert paper.title == "commercial paper" and paper.basis == "instrument paragraph"
    assert paper.group == "Loans and Notes Payable"
    assert "commercial paper borrowings" in paper.aliases
    lines = units[LINES_OF_CREDIT[:24]]
    assert lines.title == "unused lines of credit" and lines.basis == "mixed statement"
    assert {"corporate backup lines of credit", "credit facilities"} <= set(lines.aliases)
    assert text.index("There were no borrowings") in range(*lines.ranges[0])
    two = units[TWO_NOTES[:24]]
    assert two.title == "Senior Notes"
    assert "Guaranteed Senior Notes" in two.aliases and "4.9% Senior Notes due 2035" in two.aliases
    assert two.group == "Long-Term Debt"
    absent = units[NO_PAPER[:24]]
    assert absent.basis == "explicit negative" and absent.title == "commercial paper"
    capacity = units[CAPACITY_LEAD[:24]]
    assert capacity.ranges[0][1] == text.index(CAPACITY_BULLET_2) + len(CAPACITY_BULLET_2)
    assert {"securitization financings", "discount window"} <= set(capacity.aliases)
    assert capacity.group == "Additional Sources of Liquidity"
    matured = units[MATURED[:24]]
    assert matured.title == "unsecured revolving credit facility"
    # Nothing parsed: a unit names instruments as the source writes them (a
    # coupon inside a name is the name), never an amount, a balance or a
    # capacity as a field of its own.
    for unit in inventory.matters:
        body = " ".join(text[unit.ranges[0][0] : unit.ranges[0][1]].split())
        assert all(alias in body for alias in unit.aliases)
        assert not any(AMOUNT.match(alias) for alias in unit.aliases)
        assert not unit.case_numbers and not unit.courts
    unassigned = {text[s:e][:20]: why for s, e, why in inventory.unassigned}
    assert unassigned[TABLE[:20]] == "table not carried"
    assert unassigned["Loans and Notes Payab"[:20]] == "sub-heading"
    assert unassigned[COVENANT[:20]] == "no instrument named"
    assert unassigned[LONG_TERM_LEAD[:20]] == "no instrument named"
    assert unassigned[FOOTNOTE[:20]] == "no instrument named"
    statements = region_statements(text, inventory, cue=FINANCING_CUE, refers=REFERS_TO_FINANCING)
    by_start = {s.character_start: s for s in statements}
    covenant = by_start[text.index(COVENANT)]
    assert covenant.cued and not covenant.refers_to_matters, (
        "a reference to the notes alone qualifies no unit"
    )
    needs = document_needs(
        text,
        structure,
        inventory,
        document_key="d",
        issuer="KO",
        window_bytes=3900,
        cue=FINANCING_CUE,
        refers=REFERS_TO_FINANCING,
    )
    leads = [n for n in needs.needs if n.kind == "LEAD"]
    assert [n.matter_handle for n in leads] == [u.handle for u in inventory.matters]
    assert not any(n.kind == "QUALIFICATION" for n in needs.needs)
    statement_priority = {
        n.character_start: n.priority for n in needs.needs if n.kind == "REGION_STATEMENT"
    }
    assert statement_priority[text.index(COVENANT)] == 4
    assert statement_priority[text.index(TABLE)] == 6
    record = ProvisionalMatterRecord(
        handle=two.handle,
        title=two.title,
        basis=two.basis,
        named=two.named,
        region_heading=two.region_heading,
        group=two.group,
        character_start=two.ranges[0][0],
        character_end=two.ranges[0][1],
        source_bytes=two.source_bytes,
        aliases=two.aliases,
        planned_windows=1,
        read_windows=0,
        family=two.family,
    )
    assert record.family == FINANCING_FAMILY


def test_the_financing_family_is_delivered_beside_the_others_under_one_allowance(
    tmp_path: Path,
) -> None:
    """requirement (5/6): with all three families requested the financing
    units are inventoried, dealt through the one 64-window allowance beside
    the litigation matters and the event units, read whole, attributed under
    `-F-` handles with `M01:FINANCING` provenance, and the record names the
    families; no separate allowance exists."""

    runtime, request, _registry, _snapshot, document_set, generation = _open_recorded(
        tmp_path,
        entities=("AAPL",),
        documents=(
            _document("AAPL", text=_matters_filing()),
            _document("AAPL", text=_event_filing(), form="10-K", revision="10-k-2025"),
            _document("AAPL", text=_financing_filing(), form="10-K", revision="10-k-2025-debt"),
            _document("AAPL", text=_current_report(), form="8-K"),
        ),
    )
    try:
        receipt, spans = runtime.select_evidence(
            request=request, document_set=document_set, generation=generation
        )
        record = receipt.litigation_matters
        assert record is not None
        assert record.families == ("LITIGATION", "CORPORATE_EVENT", "FINANCING", "OPERATIONS")
        assert record.window_allowance == MAXIMUM_ISSUED_MATTER_WINDOWS
        financing = {
            m.handle: m for d in record.documents for m in d.matters if m.family == FINANCING_FAMILY
        }
        assert financing and all("-F-" in handle for handle in financing)
        assert {m.family for d in record.documents for m in d.matters} == {
            "LITIGATION",
            "CORPORATE_EVENT",
            FINANCING_FAMILY,
        }
        read = [w for w in record.windows if w.status == "READ"]
        delivered = {
            a.matter_handle
            for w in read
            for a in w.attributions
            if a.matter_handle in financing and a.role == "CORE"
        }
        assert delivered == set(financing), "every financing unit's opening was delivered"
        found_by, _facets = packet_module.span_provenance(receipt)
        financing_windows = [w for w in read if w.matter_handle in financing]
        assert financing_windows
        for window in financing_windows:
            assert f"M01:FINANCING:{window.matter_handle}" in found_by[window.span_handle or ""]
        views = packet_module.matter_views(receipt)
        view = packet_module.matter_view(*views[financing_windows[0].span_handle or ""])
        assert view["family"] == FINANCING_FAMILY and view["aliases"]
        by_handle = {span.span_handle: span for span in spans}
        for window in financing_windows:
            span = by_handle[window.span_handle or ""]
            unit = financing[window.matter_handle]
            assert span.character_start <= unit.character_start
            assert unit.character_end <= span.character_end
        assert packet_module.litigation_matter_summary(receipt)["families"] == list(record.families)
    finally:
        runtime.close()


def test_an_absence_stated_for_one_instrument_negates_no_other(tmp_path: Path) -> None:
    """requirement (3): the negation's scope is the clause that states it. A
    paragraph whose only instrument mention is a stated absence is an
    `explicit negative`; one sentence that states an absence for one
    instrument beside an amount outstanding under another, and a negative
    qualified by an exception, are `mixed statement` units -- visible as
    text, never a unit-wide negative; the multi-sentence paragraph that
    states a capacity and then 'no borrowings under these ... lines' is a
    mixed statement too. The basis reaches the need's detail and the
    delivered unit view unchanged."""

    text = _financing_filing()
    inventory = financing_inventory(
        text, DocumentStructure(text, document_type="10-K"), document_id="T"
    )
    units = {text[u.ranges[0][0] : u.ranges[0][1]][:24]: u for u in inventory.matters}
    assert units[NO_PAPER[:24]].basis == "explicit negative"
    mixed = units[MIXED_POLARITY[:24]]
    assert mixed.basis == "mixed statement", mixed.basis
    assert mixed.title == "commercial paper" and "revolving credit facility" in mixed.aliases
    assert units[QUALIFIED_NEGATIVE[:24]].basis == "mixed statement"
    assert units[LINES_OF_CREDIT[:24]].basis == "mixed statement"
    assert units[COMMERCIAL_PAPER[:24]].basis == "instrument paragraph"
    # one instrument's own capacity beside its absence is that instrument's
    # absence; two instruments and an amount in one clause are not
    assert units[CAPACITY_NEGATIVE[:24]].basis == "explicit negative"
    assert units[TWO_IN_ONE_CLAUSE[:24]].basis == "mixed statement"
    # a continuation that names the instrument by a pronoun is read for the
    # polarity it states: a draw 'under it' makes the paragraph mixed; an
    # absence 'under it' keeps it negative (the lead's second finding)
    assert units[PRONOMINAL_DRAW[:24]].basis == "mixed statement"
    assert units[PRONOMINAL_ABSENCE[:24]].basis == "explicit negative"
    for unit in inventory.matters:
        assert text[unit.ranges[0][0] : unit.ranges[0][1]].strip() in text, "source text kept whole"
    runtime, request, _registry, _snapshot, document_set, generation = _open_recorded(
        tmp_path,
        entities=("AAPL",),
        documents=(_document("AAPL", text=text, form="10-K", revision="10-k-2025-debt"),),
    )
    try:
        receipt, _spans = runtime.select_evidence(
            request=request, document_set=document_set, generation=generation
        )
        record = receipt.litigation_matters
        assert record is not None
        by_handle = {m.handle: m for d in record.documents for m in d.matters}
        handles = {
            key: next(
                m.handle for m in by_handle.values() if m.character_start == units[key].ranges[0][0]
            )
            for key in (NO_PAPER[:24], MIXED_POLARITY[:24], QUALIFIED_NEGATIVE[:24])
        }
        assert by_handle[handles[NO_PAPER[:24]]].basis == "explicit negative"
        assert by_handle[handles[MIXED_POLARITY[:24]]].basis == "mixed statement"
        assert by_handle[handles[QUALIFIED_NEGATIVE[:24]]].basis == "mixed statement"
        details = {n.matter_handle: n.detail for n in record.needs if n.kind == "LEAD"}
        assert details[handles[MIXED_POLARITY[:24]]].startswith("mixed statement: commercial paper")
        # Downstream: the delivered view of a window names its first unit's
        # basis and carries every unit the excerpt holds under its own handle.
        listed = {handle: matter.basis for handle, matter in by_handle.items()}
        views = packet_module.matter_views(receipt)
        carried: dict[str, str] = {}
        for span_handle in views:
            view = packet_module.matter_view(*views[span_handle])
            if view["matter_handle"] in listed:
                assert view["basis"] == listed[view["matter_handle"]]
            for carry in view["carries"]:
                if carry["role"] == "CORE":
                    carried[carry["matter_handle"]] = span_handle
        assert {handles[key] for key in handles} <= set(carried), "every control was delivered"
    finally:
        runtime.close()
