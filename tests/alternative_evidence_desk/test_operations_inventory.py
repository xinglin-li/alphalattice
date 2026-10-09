"""The operations-disclosure inventory (`operations.py`) under its frozen
definition and the routing that serves it (rules v4): regions from the
filing's own headings -- the restructuring and operating-charge notes, not
the acquisitions note the corporate-event family keeps and not a segment
disclosures note -- and, in the narrative items, the paragraphs holding a
dated operations statement; units the source's paragraphs open, titled by
the dated sentence, grouped by the note's sub-heading, a cross-reference
stub left unassigned; the family served in the operations topic's lane
with `M01:OPERATIONS` provenance; and the cue regions the trace found
unrouted -- an uncertain sub-heading naming a covenant, an operating-charge
note -- opened for their topics."""

from __future__ import annotations

import re
from pathlib import Path

from alphalattice.evidence.alternative_evidence.analysis import packet as packet_module
from alphalattice.evidence.alternative_evidence.analysis.contracts import EvidenceTopic
from alphalattice.evidence.alternative_evidence.analysis.matters import document_needs
from alphalattice.evidence.alternative_evidence.analysis.operations import (
    OPERATIONS_CUE,
    OPERATIONS_FAMILY,
    OPERATIONS_INVENTORY_RULES_ID,
    REFERS_TO_OPERATIONS,
    dated_statement_regions,
    operations_inventory,
    operations_regions,
)
from alphalattice.evidence.alternative_evidence.analysis.routing import (
    ROUTING_RULES_ID,
    TOPIC_ROUTES,
    RoutedDocument,
    route_documents,
    topics_of_unit,
)
from alphalattice.evidence.alternative_evidence.documents.structure import DocumentStructure
from tests.alternative_evidence_desk.document_intelligence_support import (
    _document_with_text as _document,
)
from tests.alternative_evidence_desk.document_intelligence_support import (
    _open_recorded,
)
from tests.alternative_evidence_desk.event_support import _current_report
from tests.alternative_evidence_desk.operations_support import (
    ACQUISITION_LEAD,
    CHARGES_2024,
    CHARGES_2025,
    COVENANT_STATEMENT,
    CREDIT_RATINGS_STATEMENT,
    CROSS_REFERENCE,
    MDA_CLOSURE,
    MDA_OPENINGS,
    NONOPERATING,
    OTHER_CHARGES_HEADING,
    RESTRUCTURING_LEAD,
    RISK_CLOSURE,
    RISK_PLAIN,
    SEGMENT_NOTE,
    _operations_filing,
)

HANDLE = re.compile(r"^M-[A-Z0-9-]{1,48}-O-[0-9]{3}$")


def test_operations_units_are_the_notes_paragraphs_and_the_dated_statements() -> None:
    """Operations units preserve the source note paragraphs and dated statements."""

    text = _operations_filing()
    structure = DocumentStructure(text, document_type="10-K")
    inventory = operations_inventory(text, structure, document_id="T")
    assert inventory.rules_id == OPERATIONS_INVENTORY_RULES_ID
    assert all(r.family == OPERATIONS_FAMILY for r in inventory.regions)
    by_kind = {(r.kind, r.heading[:22]): r for r in inventory.regions}
    assert {k[1] for k in by_kind if k[0] == "NOTE"} == {
        "NOTE 5. RESTRUCTURING ",
        "NOTE 18: SIGNIFICANT O",
    }, "the acquisitions note is the event family's; a segment disclosures note is no closure"
    for outside in (ACQUISITION_LEAD, SEGMENT_NOTE, RISK_PLAIN, MDA_OPENINGS):
        position = text.index(outside)
        assert not any(r.body_start <= position < r.body_end for r in inventory.regions), outside[
            :30
        ]
    dated = dated_statement_regions(text, structure)
    assert [(r.kind, r.heading[:12]) for r in dated] == [
        ("ITEM", "ITEM 1A. RIS"),
        ("ITEM", "ITEM 7. MANA"),
    ]
    assert dated[0].body_start == text.index(RISK_CLOSURE)
    assert dated[0].body_end == text.index(RISK_CLOSURE) + len(RISK_CLOSURE)
    assert dated[0].end_basis == "the paragraph's own end" and not dated[0].end_uncertain
    assert (
        operations_regions(
            _current_report(), DocumentStructure(_current_report(), document_type="8-K")
        )
        == ()
    )
    units = {text[u.ranges[0][0] : u.ranges[0][1]][:24]: u for u in inventory.matters}
    assert all(HANDLE.match(u.handle) for u in inventory.matters)
    assert all(u.family == OPERATIONS_FAMILY and u.named for u in inventory.matters)
    assert all(not u.case_numbers and not u.courts for u in inventory.matters)
    closure = units[RISK_CLOSURE[:24]]
    assert closure.basis == "dated operations statement"
    assert closure.title.startswith("In the first quarter of 2025, we closed 45 pOpshelf stores")
    assert closure.ranges == (
        (text.index(RISK_CLOSURE), text.index(RISK_CLOSURE) + len(RISK_CLOSURE)),
    )
    assert closure.aliases == ("the first quarter of 2025",)
    restaurants = units[MDA_CLOSURE[:24]]
    assert restaurants.basis == "dated operations statement"
    assert restaurants.title.startswith("On February 3, 2026, we announced")
    assert MDA_OPENINGS[:24] not in units
    charges = units[CHARGES_2025[:24]]
    assert charges.basis == "dated operations statement"
    assert (
        charges.title == "In 2025, the Company recorded other operating charges of $1,261 million."
    )
    assert charges.group == OTHER_CHARGES_HEADING
    assert units[CHARGES_2024[:24]].group == OTHER_CHARGES_HEADING
    transformation = units[RESTRUCTURING_LEAD[:24]]
    assert transformation.basis == "operations paragraph"
    assert transformation.group == "NOTE 5. RESTRUCTURING ACTIONS"
    assert CROSS_REFERENCE[:24] not in units and NONOPERATING[:24] not in units
    unassigned = {text[s:e][:20]: why for s, e, why in inventory.unassigned}
    assert unassigned[CROSS_REFERENCE[:20]] == "cross-reference"
    assert unassigned[OTHER_CHARGES_HEADING[:20]] == "sub-heading"
    assert unassigned[NONOPERATING[:20]] == "no operations charge or action stated"
    for unit in inventory.matters:
        assert not any(alias.startswith("$") for alias in unit.aliases)
    needs = document_needs(
        text,
        structure,
        inventory,
        document_key="d",
        issuer="KO",
        window_bytes=3900,
        cue=OPERATIONS_CUE,
        refers=REFERS_TO_OPERATIONS,
        family=OPERATIONS_FAMILY,
    )
    leads = [n for n in needs.needs if n.kind == "LEAD"]
    assert [n.matter_handle for n in leads] == [u.handle for u in inventory.matters]
    assert all(n.family == OPERATIONS_FAMILY for n in needs.needs)
    # A dated statement's region is its paragraph: no region statement is
    # made of the risk factors around it.
    statements = [n for n in needs.needs if n.kind == "REGION_STATEMENT"]
    assert not any(n.region_heading.startswith("ITEM 1A") for n in statements)
    assert any(n.character_start == text.index(CROSS_REFERENCE) for n in statements)


def test_routing_v4_serves_the_operations_family_and_opens_the_uncertain_cue_regions() -> None:
    """Routing serves the operations family and opens uncertain cue regions within their source
    bounds."""

    assert ROUTING_RULES_ID == "alternative-evidence.topic-routing.v6"
    assert topics_of_unit(
        family=OPERATIONS_FAMILY,
        region_heading="ITEM 1A. RISK FACTORS",
        title="In the first quarter of 2025, we closed 45 pOpshelf stores",
        document_type="10-K",
        excerpt="",
    ) == (EvidenceTopic.OPERATIONS_SUPPLY,)
    operations_route = TOPIC_ROUTES[EvidenceTopic.OPERATIONS_SUPPLY]
    assert OPERATIONS_FAMILY in operations_route.unit_families
    assert operations_route.heading_cue.search("SIGNIFICANT OPERATING AND NONOPERATING ITEMS")
    assert TOPIC_ROUTES[EvidenceTopic.LIQUIDITY_GOING_CONCERN].heading_cue.search("Covenants")
    assert TOPIC_ROUTES[EvidenceTopic.COMMERCIAL_COUNTERPARTY].typed_families == (
        "CUSTOMER_CONCENTRATION",
    )
    assert TOPIC_ROUTES[EvidenceTopic.PRODUCT_SAFETY_CYBER].typed_families == (
        "CYBERSECURITY_THREAT_EFFECT",
    )
    text = _operations_filing()
    structure = DocumentStructure(text, document_type="10-K")
    covenant_heading = next(h for h in structure.headings if h.text.strip() == "Covenants")
    assert covenant_heading.kind == "SUB_UNCERTAIN"
    inventory = operations_inventory(text, structure, document_id="T")
    needs = document_needs(
        text,
        structure,
        inventory,
        document_key="doc-10k",
        issuer="KO",
        window_bytes=3900,
        cue=OPERATIONS_CUE,
        refers=REFERS_TO_OPERATIONS,
        family=OPERATIONS_FAMILY,
    )
    document = RoutedDocument(
        document_key="doc-10k",
        document_handle="DOC-KO-001",
        entity_id="KO",
        document_type="10-K",
        text=text,
        structure=structure,
        accepted_at=None,
        published_at=None,
        report_period_end=None,
    )
    routing = route_documents(
        documents=(document,),
        needs_by_document={"doc-10k": needs},
        placeholders={},
        originals={},
        typed_by_issuer={"KO": {"CUSTOMER_CONCENTRATION": 1}},
        entity_ids=("KO",),
    )
    cells = {str(c.topic): c for c in routing.cells}
    operations = cells["OPERATIONS_SUPPLY"]
    assert dict(operations.regions_by_basis)["inventory:OPERATIONS"] == len(inventory.regions)
    assert operations.unit_needs >= len(inventory.matters)
    charges = text.index(CHARGES_2025)
    closure = text.index(RISK_CLOSURE)

    def in_residual(topic: str, position: int) -> bool:
        return any(s <= position < e for _k, s, e in cells[topic].residual_ranges)

    assert not in_residual("OPERATIONS_SUPPLY", charges) and not in_residual(
        "OPERATIONS_SUPPLY", closure
    ), "an inventoried unit is served by its window, not searched for again"
    assert any(
        r.basis == "heading cue:NOTE" and r.heading.startswith("NOTE 18")
        for r in routing.regions
        if str(r.topic) == "OPERATIONS_SUPPLY"
    )
    uncertain = [
        r
        for r in routing.regions
        if str(r.topic) == "LIQUIDITY_GOING_CONCERN" and r.basis == "heading cue:SUB_UNCERTAIN"
    ]
    assert [r.heading for r in uncertain] == ["Liquidity and Capital Resources", "Covenants"]
    assert not any(
        r.basis == "heading cue:SUB_UNCERTAIN" and r.heading == "Other Nonoperating Items"
        for r in routing.regions
    ), "an uncertain sub-heading opens a region inside the MD&A only; the note is its own region"
    covenant = [r for r in uncertain if r.heading == "Covenants"]
    assert covenant[0].character_start <= text.index(COVENANT_STATEMENT)
    assert covenant[0].character_end <= text.index(CREDIT_RATINGS_STATEMENT), (
        "the region closes at the next heading of any rank"
    )
    assert in_residual("LIQUIDITY_GOING_CONCERN", text.index(COVENANT_STATEMENT))
    commercial = cells["COMMERCIAL_COUNTERPARTY"]
    assert commercial.typed_observations == 1


def test_the_operations_family_is_delivered_in_its_own_lane_beside_the_others(
    tmp_path: Path,
) -> None:
    """The operations family is delivered in its own lane beside the others."""

    runtime, request, _registry, _snapshot, document_set, generation = _open_recorded(
        tmp_path,
        entities=("AAPL",),
        documents=(
            _document("AAPL", text=_operations_filing(), form="10-K", revision="10-k-2025"),
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
        operations = {
            m.handle: m
            for d in record.documents
            for m in d.matters
            if m.family == OPERATIONS_FAMILY
        }
        assert operations and all("-O-" in handle for handle in operations)
        read = [w for w in record.windows if w.status == "READ"]
        delivered = {
            a.matter_handle
            for w in read
            for a in w.attributions
            if a.matter_handle in operations and a.role == "CORE"
        }
        assert delivered == set(operations), "every operations unit's opening was delivered"
        found_by, _facets = packet_module.span_provenance(receipt)
        windows = [w for w in read if w.matter_handle in operations]
        for window in windows:
            assert f"M01:OPERATIONS:{window.matter_handle}" in found_by[window.span_handle or ""]
        views = packet_module.matter_views(receipt)
        view = packet_module.matter_view(*views[windows[0].span_handle or ""])
        assert view["family"] == OPERATIONS_FAMILY
        by_handle = {span.span_handle: span for span in spans}
        for window in windows:
            span = by_handle[window.span_handle or ""]
            unit = operations[window.matter_handle]
            assert span.character_start <= unit.character_start
            assert unit.character_end <= span.character_end
    finally:
        runtime.close()
