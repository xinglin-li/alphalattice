"""The corporate-event inventory (`events.py`): regions from the filing's
own structure (the notes that name the family, the note whose lead dates
an acquisition, the items of a current report), units opened by the
source's dated actions with a run-in title allowed, the accounting of an
event continuing it, the region-level paragraphs left to the region, and
the needs the litigation owner states from it under the family's own cue
and scope phrasings. Controlled fixtures of real filings' shapes."""

from __future__ import annotations

import re

from alphalattice.evidence.alternative_evidence.analysis.contracts import (
    ProvisionalMatterRecord,
)
from alphalattice.evidence.alternative_evidence.analysis.events import (
    CORPORATE_EVENT_FAMILY,
    EVENT_CUE,
    EVENT_INVENTORY_RULES_ID,
    REFERS_TO_EVENTS,
    event_inventory,
    event_regions,
)
from alphalattice.evidence.alternative_evidence.analysis.matters import (
    document_needs,
    region_statements,
)
from alphalattice.evidence.alternative_evidence.documents.structure import DocumentStructure
from tests.alternative_evidence_desk.event_support import (
    ALLOCATION,
    CATEGORY_LEAD_IN,
    CCEP_SALE,
    DEBT_ISSUE,
    DIVIDEND_DECLARED,
    EVALUATED,
    GROQ_LEAD,
    INDIA_REFRANCHISING,
    LATE_FILING_CHANGE,
    LATE_FILING_NARRATIVE,
    MDA_SALE,
    OTHER_EVENTS_ITEM,
    PETS_BEST,
    PRO_FORMA,
    RESULTS_ITEM,
    TABLE,
    _current_report,
    _event_filing,
    _late_filing_notice,
)
from tests.alternative_evidence_desk.litigation_support import _filing

HANDLE = re.compile(r"^M-[A-Z0-9-]{1,48}-[0-9]{3}$")


def test_event_regions_and_units_come_from_the_source_structure() -> None:
    """requirement (K3): the acquisitions note, the note named after its
    counterparty whose lead dates an acquisition, and the subsequent-events
    note are regions; the policy note, the debt note with a dated issue and
    the MD&A with a dated sale are not. A dated action opens a unit, a
    run-in title before the date included; the allocation and table of that
    event continue it; the category lead-in, the pro forma paragraph and
    the evaluation statement stay the region's; handles fit the record."""

    text = _event_filing()
    structure = DocumentStructure(text, document_type="10-K")
    inventory = event_inventory(text, structure, document_id="T")
    assert inventory.rules_id == EVENT_INVENTORY_RULES_ID
    assert [r.heading for r in inventory.regions] == [
        "NOTE 2. ACQUISITIONS AND DIVESTITURES",
        "NOTE 4. GROQ",
        "NOTE 5. SUBSEQUENT EVENTS",
    ]
    assert all(r.kind == "NOTE" and r.family == CORPORATE_EVENT_FAMILY for r in inventory.regions)
    assert not any(r.end_uncertain for r in inventory.regions)
    covered = [(r.body_start, r.body_end) for r in inventory.regions]
    for outside in (MDA_SALE, DEBT_ISSUE):
        position = text.index(outside)
        assert not any(s <= position < e for s, e in covered), outside[:30]
    units = {text[u.ranges[0][0] : u.ranges[0][1]][:20]: u for u in inventory.matters}
    assert [u.basis for u in inventory.matters] == ["dated event"] * 5
    assert all(u.family == CORPORATE_EVENT_FAMILY and u.named for u in inventory.matters)
    assert all(HANDLE.match(u.handle) for u in inventory.matters)
    assert [u.handle for u in inventory.matters] == [f"M-T-E-{n:03d}" for n in range(1, 6)]
    # Each dated sentence opens its own unit at its own range.
    ccep, india = units[CCEP_SALE[:20]], units[INDIA_REFRANCHISING[:20]]
    assert ccep.ranges == ((text.index(CCEP_SALE), text.index(CCEP_SALE) + len(CCEP_SALE)),)
    assert india.ranges[0][0] == text.index(INDIA_REFRANCHISING)
    assert ccep.aliases == ("March 2025",) and india.aliases == ("May 2025",)
    # The run-in title is part of the unit; the allocation sentence and the
    # table that follow continue it; the pro forma paragraph does not.
    pets = units[PETS_BEST[:20]]
    start, end = pets.ranges[0]
    assert start == text.index(PETS_BEST) and pets.title.startswith("Pets Best In March 2024")
    assert text.index(ALLOCATION) > start and text.index(TABLE) + len(TABLE) <= end
    assert text.index(PRO_FORMA) >= end
    # The note named after its counterparty is one unit; the dividend is one.
    groq = units[GROQ_LEAD[:20]]
    assert (
        groq.region_heading == "NOTE 4. GROQ"
        and "goodwill" in text[groq.ranges[0][0] : groq.ranges[0][1]]
    )
    dividend = units[DIVIDEND_DECLARED[:20]]
    assert dividend.aliases == ("June 24, 2026",)
    assert dividend.ranges[0][1] <= text.index(EVALUATED)
    # What no unit claimed is visible with its reason.
    unassigned = {text[s:e][:20]: why for s, e, why in inventory.unassigned}
    assert unassigned == {
        CATEGORY_LEAD_IN[:20]: "region-level statement",
        PRO_FORMA[:20]: "region-level statement",
        EVALUATED[:20]: "region-level statement",
    }
    # The record contract accepts the unit as the packet carries it.
    record = ProvisionalMatterRecord(
        handle=pets.handle,
        title=pets.title,
        basis=pets.basis,
        named=pets.named,
        region_heading=pets.region_heading,
        group=pets.group,
        character_start=start,
        character_end=end,
        source_bytes=pets.source_bytes,
        aliases=pets.aliases,
        planned_windows=1,
        read_windows=0,
        family=pets.family,
    )
    assert record.family == CORPORATE_EVENT_FAMILY
    assert "family" not in ProvisionalMatterRecord.model_validate(
        {**record.model_dump(mode="json"), "family": "LITIGATION"}
    ).model_dump(mode="json")
    # Region statements under the family's own phrasings: the pro forma
    # paragraph says it applies to the transactions above and qualifies
    # every unit of its note; the evaluation statement qualifies the
    # dividend; the category lead-in names no unit.
    statements = region_statements(text, inventory, cue=EVENT_CUE, refers=REFERS_TO_EVENTS)
    by_start = {s.character_start: s for s in statements}
    lead_in = by_start[text.index(CATEGORY_LEAD_IN)]
    assert lead_in.position == "LEAD_IN" and lead_in.cued and not lead_in.refers_to_matters
    pro_forma = by_start[text.index(PRO_FORMA)]
    assert pro_forma.position == "TRAILING" and pro_forma.refers_to_matters
    assert pro_forma.reference_text == "the transactions described above"
    assert by_start[text.index(EVALUATED)].reference_text == "subsequent events through"
    needs = document_needs(
        text,
        structure,
        inventory,
        document_key="d",
        issuer="KO",
        window_bytes=3900,
        cue=EVENT_CUE,
        refers=REFERS_TO_EVENTS,
    )
    leads = [n for n in needs.needs if n.kind == "LEAD"]
    assert [n.matter_handle for n in leads] == [u.handle for u in inventory.matters]
    qualified = {
        n.matter_handle
        for n in needs.needs
        if n.kind == "QUALIFICATION" and n.character_start == text.index(PRO_FORMA)
    }
    assert qualified == {ccep.handle, india.handle, pets.handle}
    assert {
        n.matter_handle
        for n in needs.needs
        if n.kind == "QUALIFICATION" and n.character_start == text.index(EVALUATED)
    } == {dividend.handle}
    region_level = {
        n.character_start: n.priority for n in needs.needs if n.kind == "REGION_STATEMENT"
    }
    assert region_level[text.index(PRO_FORMA)] == 1 and region_level[text.index(EVALUATED)] == 1
    assert region_level[text.index(CATEGORY_LEAD_IN)] == 4


def test_a_current_report_is_one_unit_per_item_and_a_quarterly_without_events_has_no_need() -> None:
    """requirement (K3): every item of an 8-K other than the exhibits item is
    a region and one unit -- its whole body, dated by the first date it
    states -- whatever the item says; a 10-Q with no event note has no event
    region, no unit and no need, and its litigation note is not an event
    region."""

    text = _current_report()
    structure = DocumentStructure(text, document_type="8-K")
    inventory = event_inventory(text, structure, document_id="C")
    assert [r.heading for r in inventory.regions] == [
        "Item 2.02 Results of Operations and Financial Condition.",
        "Item 8.01 Other Events.",
    ]
    assert all(r.kind == "ITEM" for r in inventory.regions)
    assert [u.basis for u in inventory.matters] == ["8-K item", "8-K item"]
    results, other = inventory.matters
    assert results.title.startswith("Item 2.02") and other.title.startswith("Item 8.01")
    assert text[results.ranges[0][0] : results.ranges[0][1]] == RESULTS_ITEM
    assert text[other.ranges[0][0] : other.ranges[0][1]] == OTHER_EVENTS_ITEM
    assert results.aliases == other.aliases == ("July 7, 2026",)
    assert not inventory.unassigned
    assert not any("9.01" in r.heading for r in event_regions(text, structure))
    quarterly = _filing(note_lines=["On May 1, 2025, a lawsuit was filed against the Company."])
    quarterly_structure = DocumentStructure(quarterly, document_type="10-Q")
    empty = event_inventory(quarterly, quarterly_structure, document_id="Q")
    assert not empty.regions and not empty.matters and not empty.unassigned
    needs = document_needs(
        quarterly,
        quarterly_structure,
        empty,
        document_key="q",
        issuer="AAPL",
        window_bytes=3900,
        cue=EVENT_CUE,
        refers=REFERS_TO_EVENTS,
    )
    assert not needs.needs and not needs.statements


def test_a_late_filing_notice_is_one_unit_per_narrative_part() -> None:
    """requirement (Q3, rules v2): a late-filing notice (Form 12b-25, NT 10-Q)
    is read by the event family: its narrative (Part III) and its other
    information (Part IV) are each a region and one unit, the whole body under
    the part heading; the registrant and rule parts are not events, and the
    signature closes the last part. The same text filed as a quarterly report
    has no event region: the rule is the late-filing notice's own."""

    text = _late_filing_notice()
    structure = DocumentStructure(text, document_type="NT 10-Q")
    inventory = event_inventory(text, structure, document_id="N")
    assert [(r.heading, r.kind) for r in inventory.regions] == [
        ("PART III -- NARRATIVE", "PART"),
        ("PART IV -- OTHER INFORMATION", "PART"),
    ]
    narrative, change = inventory.matters
    assert [u.basis for u in inventory.matters] == ["late-filing notice"] * 2
    assert text[narrative.ranges[0][0] : narrative.ranges[0][1]] == LATE_FILING_NARRATIVE
    assert text[change.ranges[0][0] : change.ranges[0][1]] == LATE_FILING_CHANGE
    assert narrative.family == change.family == CORPORATE_EVENT_FAMILY
    assert not inventory.unassigned
    amended = event_inventory(text, DocumentStructure(text, document_type="NT 10-Q/A"))
    assert len(amended.matters) == 2, "an amended notice is read by its base form"
    quarterly = event_inventory(text, DocumentStructure(text, document_type="10-Q"))
    assert not quarterly.regions and not quarterly.matters
