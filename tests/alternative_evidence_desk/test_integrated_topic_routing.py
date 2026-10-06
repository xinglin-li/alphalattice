"""The integrated selection: one discovery, eight topics routed by source
shape, the units dealt in issuer-topic lanes latest filing first with
exact repeats collapsed, tables inside routed regions delivered as views
of the retained original, and the question bank run only over each
topic's residual scope under a pair budget -- sealed as one receipt whose
routing record accounts for every cell, every skipped question and every
undelivered table, reused across requests over the same content, and
continued as one chain."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from alphalattice.evidence.alternative_evidence.analysis.contracts import (
    CONTEXT_COMPLETE_SELECTION_RULES_ID,
    AlternativeEvidenceRetrievalAccessReceipt,
    CandidateFrontier,
    EvidenceTopic,
    FilingComparisonRecord,
    PendingCandidateRecord,
    TopicRoutingRecord,
)
from alphalattice.evidence.alternative_evidence.analysis.events import CORPORATE_EVENT_FAMILY
from alphalattice.evidence.alternative_evidence.analysis.matters import (
    TOPIC_LANES_ALLOCATION_ID,
)
from alphalattice.evidence.alternative_evidence.analysis.packet import (
    EVIDENCE_QUERY_PROGRAM,
    MATTER_FAMILIES,
    MAXIMUM_PACKET_SPANS,
    RESIDUAL_RERANK_PAIR_BUDGET,
    RESIDUAL_SELECTION_RULES_ID,
    SPANS_PER_ISSUER,
    _allocate_by_cell,
    delivered_table_identities,
    litigation_continuation_scope,
    plan_table_pages,
    query_program_hash,
    routed_documents,
    routed_pending_work,
    shared_inventory,
    span_provenance,
    topic_fair_order,
)
from alphalattice.evidence.alternative_evidence.analysis.read_model import (
    bundles_of,
    topic_coverage_ledger,
)
from alphalattice.evidence.alternative_evidence.analysis.routing import (
    ROUTING_RULES_ID,
    TABLE_VIEWS_PER_SESSION,
    TOPIC_ROUTES,
    TOPICS,
    RoutedDocument,
    TableNeed,
    route_documents,
    topic_fair_views,
    topics_of_need,
    topics_of_unit,
)
from alphalattice.evidence.alternative_evidence.contracts import (
    MATTER_FAMILY_CORPORATE_EVENT,
    MATTER_FAMILY_FINANCING,
    MATTER_FAMILY_LITIGATION,
    MATTER_SELECTION_CANDIDATE,
    MATTER_SELECTION_INTEGRATED,
    AlternativeEvidenceClass,
    AlternativeEvidenceMode,
    AlternativeEvidenceRequest,
    AlternativeEvidenceSourcePolicy,
    MatterSelectionPolicy,
    seal_contract,
)
from alphalattice.evidence.alternative_evidence.documents.comparison import (
    COMPARISON_RULES_ID,
)
from alphalattice.evidence.alternative_evidence.documents.structure import DocumentStructure
from alphalattice.evidence.alternative_evidence.documents.tables import TablePlaceholder
from alphalattice.evidence.alternative_evidence.publication.artifacts import (
    AlternativeEvidencePublicationError,
)
from alphalattice.evidence.alternative_evidence.runtime.history import (
    receipt_matter_selection_id,
)
from alphalattice.kernel.knowledge._embeddings import MODEL_WORK
from tests.alternative_evidence_desk.document_intelligence_support import (
    _document_with_text,
    _registry,
)
from tests.alternative_evidence_desk.event_support import (
    LATE_FILING_NARRATIVE,
    _late_filing_notice,
)
from tests.alternative_evidence_desk.incremental_acquisition_support import (
    AAPL,
    REGISTRY,
    TEN_K,
    _acquire,
)
from tests.alternative_evidence_desk.matter_selection_support import (
    _filings_with_debt,
    _repeating_filings,
)
from tests.alternative_evidence_desk.planted_corpus import _NOW, _runtime
from tests.alternative_evidence_desk.sec_scenario_transport import SecScenarioTransport
from tests.alternative_evidence_desk.table_view_support import original_html

ALL_FAMILIES = (
    MATTER_FAMILY_LITIGATION,
    MATTER_FAMILY_CORPORATE_EVENT,
    MATTER_FAMILY_FINANCING,
)
INTEGRATED = MatterSelectionPolicy(method=MATTER_SELECTION_INTEGRATED, families=ALL_FAMILIES)
CANDIDATE = MatterSelectionPolicy(method=MATTER_SELECTION_CANDIDATE, families=ALL_FAMILIES)


def _request(
    entities: tuple[str, ...],
    *,
    selection: MatterSelectionPolicy | None,
    ttl_seconds: int = 86_400,
) -> AlternativeEvidenceRequest:
    return seal_contract(
        AlternativeEvidenceRequest,
        "request_hash",
        ordered_entity_ids=entities,
        evidence_as_of=_NOW,
        acquisition_deadline=_NOW + timedelta(hours=1),
        evidence_classes=(AlternativeEvidenceClass.ISSUER_OFFICIAL_RECORDED,),
        source_policy=AlternativeEvidenceSourcePolicy(),
        ttl_seconds=ttl_seconds,
        mode=AlternativeEvidenceMode.RECORDED,
        matter_selection=selection,
    )


def _open(tmp_path: Path, request: AlternativeEvidenceRequest) -> tuple[Any, Any, Any]:
    runtime = _runtime(tmp_path)
    entities = request.ordered_entity_ids
    _snapshot, source_set = runtime.acquire_recorded(
        request=request,
        registry=_registry(entities),
        documents=(*_filings_with_debt(entities), *_repeating_filings()),
        published_at=_NOW,
    )
    document_set = runtime.canonicalize(source_set=source_set, published_at=_NOW)
    generation = runtime.build_retrieval(document_set=document_set, built_at=_NOW)
    return runtime, document_set, generation


def test_the_policy_names_all_three_families_and_the_routing_table_covers_every_topic() -> None:
    """requirement (C): the integrated method is one spelling with all three
    families; every one of the eight topics has an executable route -- unit
    families, structure families, a heading cue -- and a named deficit."""

    assert INTEGRATED.integrated and INTEGRATED.selection_id.startswith("INTEGRATED_TOPIC_ROUTING:")
    with pytest.raises(ValueError, match="matter_selection_families_invalid"):
        MatterSelectionPolicy(
            method=MATTER_SELECTION_INTEGRATED, families=(MATTER_FAMILY_LITIGATION,)
        )
    assert tuple(TOPIC_ROUTES) == TOPICS == tuple(EvidenceTopic)
    for topic, route in TOPIC_ROUTES.items():
        assert route.topic is topic
        assert route.unit_families and route.structure_families and route.deficit
        assert route.heading_cue.search("liquidity") is not None or route.heading_cue.pattern


def test_a_delisting_notice_and_a_late_filing_notice_route_by_their_own_rules() -> None:
    """requirement (Q3, rules v5): Item 3.01 -- a delisting notice or a failure
    to meet a listing standard -- serves the corporate-action and listing topic
    (under v4 the capital topic); the unregistered sales and the changes to
    holders' rights of Items 3.02-3.03 still serve the capital topic; a
    late-filing notice's parts, amended or not, serve the governance topic."""

    def topics(heading: str, document_type: str) -> tuple[str, ...]:
        return tuple(
            str(topic)
            for topic in topics_of_unit(
                family=CORPORATE_EVENT_FAMILY,
                region_heading=heading,
                title=heading,
                document_type=document_type,
                excerpt="",
            )
        )

    # The rules v5 introduced, carried by every later version.
    assert int(ROUTING_RULES_ID.rsplit(".v", 1)[1]) >= 5
    delisting = "Item 3.01 Notice of Delisting or Failure to Satisfy a Continued Listing Rule."
    assert topics(delisting, "8-K") == ("CORPORATE_ACTION_LISTING",)
    assert topics("Item 3.02 Unregistered Sales of Equity Securities.", "8-K") == (
        "CAPITAL_DILUTION",
    )
    assert topics("Item 3.03 Material Modification to Rights of Security Holders.", "8-K/A") == (
        "CAPITAL_DILUTION",
    )
    for document_type in ("NT 10-Q", "NT 10-K/A"):
        assert topics("PART III -- NARRATIVE", document_type) == ("GOVERNANCE_CONTROLS",)


def test_a_late_filing_notice_is_delivered_in_the_governance_lane(tmp_path: Path) -> None:
    """requirement (Q3): the packet read a filing's units only for a periodic
    report or a current report, so a late-filing notice beside them had no unit
    and only a residual search could meet it. Now the event family inventories
    the notice: its two parts are the governance cell's unit needs, both
    delivered, and the narrative is among the delivered spans."""

    request = _request(("AAPL",), selection=INTEGRATED)
    runtime = _runtime(tmp_path)
    try:
        notice = _document_with_text(
            "AAPL", text=_late_filing_notice(), form="NT 10-Q", revision="nt-10-q-2026-q2"
        )
        _snapshot, source_set = runtime.acquire_recorded(
            request=request,
            registry=_registry(("AAPL",)),
            documents=(*_filings_with_debt(("AAPL",)), notice),
            published_at=_NOW,
        )
        document_set = runtime.canonicalize(source_set=source_set, published_at=_NOW)
        generation = runtime.build_retrieval(document_set=document_set, built_at=_NOW)
        receipt, spans = runtime.select_evidence(
            request=request, document_set=document_set, generation=generation
        )
        assert receipt.routing is not None
        governance = next(
            cell
            for cell in receipt.routing.cells
            if cell.entity_id == "AAPL" and cell.topic == "GOVERNANCE_CONTROLS"
        )
        assert governance.unit_needs == governance.unit_needs_delivered == 2
        assert dict(governance.regions_by_basis) == {"inventory:CORPORATE_EVENT": 2}
        handle = next(
            document.semantic_handle
            for document in document_set.documents
            if document.document_type == "NT 10-Q"
        )
        assert any(
            span.document_handle == handle and LATE_FILING_NARRATIVE[:60] in span.excerpt
            for span in spans
        )
    finally:
        runtime.close()


def test_an_unstructured_document_is_every_topic_s_residual_scope() -> None:
    """requirement (C, rules v2): a current report whose canonical text
    carries no item heading has no shape to route by; its body joins every
    topic's residual scope as an `unstructured document` region, so the
    questions read it as the unconditional program did, and the cell says
    why. The measured loss under v1: two former held-out cases in 8-K texts
    without item headings, outside every routed range."""

    text = "\n\n".join(
        (
            "UNITED STATES",
            "SECURITIES AND EXCHANGE COMMISSION",
            "FORM 8-K",
            "CURRENT REPORT",
            "Pursuant to Section 13 or 15(d) of the Securities Exchange Act of 1934",
            "Date of Report (Date of earliest event reported): August 10, 2026",
            "On August 10, 2026, the company issued a press release announcing the launch of "
            "cash tender offers for certain of its debt securities.",
            "SIGNATURES",
        )
    )
    structure = DocumentStructure(text, document_type="8-K")
    assert not any(h.kind in {"PART", "ITEM", "NOTE"} for h in structure.headings)
    document = RoutedDocument(
        document_key="doc-8k",
        document_handle="DOC-AAPL-001",
        entity_id="AAPL",
        document_type="8-K",
        text=text,
        structure=structure,
        accepted_at=None,
        published_at=None,
        report_period_end=None,
    )
    routing = route_documents(
        documents=(document,),
        needs_by_document={},
        placeholders={},
        originals={},
        typed_by_issuer={},
        entity_ids=("AAPL",),
    )
    assert len(routing.cells) == len(TOPICS)
    for cell in routing.cells:
        assert dict(cell.regions_by_basis) == {"unstructured document": 1}
        assert cell.residual == "QUEUED"
        assert cell.residual_ranges == (("doc-8k", structure.cover_end, structure.length),)
        assert "ROUTE_GAP" not in " ".join(cell.gaps)
    assert routing.rules_id == ROUTING_RULES_ID == "alternative-evidence.topic-routing.v6"


def test_an_inventoried_region_withdraws_only_the_topics_its_units_serve(tmp_path: Path) -> None:
    """requirement (the shared-gap closeout, boundary A; rules v3): a
    structural inventory replaces the residual search for the requirement
    it supplies -- the topics the region's own units serve -- and for no
    other topic routed to the same region. Four controls at the routing
    owner over one book: (1) a recognized inventory that leaves relevant
    residual prose (the litigation note: the legal topic covered by its
    units, the product topic keeps the note in its residual scope); (2) a
    region serving more than one topic (the debt note's units serve the
    capital and the liquidity topics: withdrawn from both, kept by a topic
    no unit there serves); (3) a valid structured route with pending
    delivery (the legal cell: covered, its needs counted, no search spent);
    (4) a genuinely satisfied requirement searched no second time (the
    results item whose units serve the operations topic is withdrawn from
    it while the topics its units do not serve keep it). Under v2 every
    family's regions were subtracted from every topic's scope: the product
    topic held none of the litigation note."""

    request = _request(("AAPL", "MSFT"), selection=INTEGRATED)
    runtime, document_set, generation = _open(tmp_path, request)
    try:
        session = runtime.open_session(
            document_set=document_set, generation=generation, evidence_as_of=request.evidence_as_of
        )
        try:
            inventory = shared_inventory(session)
            routing = route_documents(
                documents=routed_documents(session, inventory),
                needs_by_document=inventory.needs_by_document,
                placeholders={k: session.table_placeholders(k) for k in inventory.inspected},
                originals={k: session.original_available(k) for k in inventory.inspected},
                typed_by_issuer={},
            )
        finally:
            session.close()
    finally:
        runtime.close()
    assert routing.rules_id == "alternative-evidence.topic-routing.v6"
    cells = {(c.entity_id, str(c.topic)): c for c in routing.cells}
    regions: dict[tuple[str, str], tuple[str, int, int]] = {}
    for key, needs in inventory.needs_by_document.items():
        handle = inventory.inspected[key].document_handle
        for region in needs.inventory.regions:
            label = " ".join(region.heading.split()[:2])
            regions[(handle, label)] = (key, region.body_start, region.body_end)

    def kept(topic: str, handle: str, label: str) -> float:
        key, start, end = regions[(handle, label)]
        overlap = sum(
            max(0, min(e, end) - max(s, start))
            for k, s, e in cells[("AAPL", topic)].residual_ranges
            if k == key
        )
        return overlap / (end - start)

    def routed(topic: str, handle: str, label: str) -> bool:
        key, start, end = regions[(handle, label)]
        return any(
            r.document_key == key and r.character_start < end and start < r.character_end
            for r in routing.regions
            if r.entity_id == "AAPL" and str(r.topic) == topic
        )

    # (1) The 10-K's litigation note is routed to the legal and the product
    # topics; its units serve the legal topic only.
    legal, product = cells[("AAPL", "LEGAL_REGULATORY")], cells[("AAPL", "PRODUCT_SAFETY_CYBER")]
    assert routed("LEGAL_REGULATORY", "DOC-AAPL-001", "NOTE 7.")
    assert routed("PRODUCT_SAFETY_CYBER", "DOC-AAPL-001", "NOTE 7.")
    assert legal.residual == "COVERED" and legal.residual_ranges == () and legal.unit_needs > 0
    assert product.residual == "QUEUED" and product.unit_needs == 0
    assert kept("PRODUCT_SAFETY_CYBER", "DOC-AAPL-001", "NOTE 7.") > 0.99, (
        "the note is prose to the product topic: it stays in the residual scope"
    )
    assert kept("LEGAL_REGULATORY", "DOC-AAPL-001", "NOTE 7.") == 0.0
    # (2) The 10-Q's debt note serves two topics: withdrawn from both, and
    # from neither more than once; the governance topic, which no region
    # routes to and no unit there serves, reads it in its broader pass.
    for topic in ("CAPITAL_DILUTION", "LIQUIDITY_GOING_CONCERN"):
        assert routed(topic, "DOC-AAPL-002", "NOTE 3."), topic
        assert cells[("AAPL", topic)].unit_needs > 0, topic
        assert kept(topic, "DOC-AAPL-002", "NOTE 3.") == 0.0, topic
    governance = cells[("AAPL", "GOVERNANCE_CONTROLS")]
    assert governance.residual == "BROADER" and governance.unit_needs == 0
    assert kept("GOVERNANCE_CONTROLS", "DOC-AAPL-002", "NOTE 3.") == 1.0
    # (3) The legal cell is a valid structured route whose delivery is the
    # allowance's: its needs are counted, its search scope is none.
    assert legal.unit_needs >= 8 and legal.regions_by_basis
    # (4) The 8-K's results item: its units serve the operations topic,
    # which searches it no second time; the product and commercial topics,
    # which no unit there serves, keep it.
    operations = cells[("AAPL", "OPERATIONS_SUPPLY")]
    assert routed("OPERATIONS_SUPPLY", "DOC-AAPL-003", "Item 2.02")
    assert operations.residual == "QUEUED" and operations.unit_needs > 0
    assert kept("OPERATIONS_SUPPLY", "DOC-AAPL-003", "Item 2.02") == 0.0
    for topic in ("PRODUCT_SAFETY_CYBER", "COMMERCIAL_COUNTERPARTY"):
        assert cells[("AAPL", topic)].unit_needs == 0
        assert kept(topic, "DOC-AAPL-003", "Item 2.02") == 1.0, topic
    # The 8-K's other-events item, served by the corporate topic's units
    # only, stays in the liquidity and the operations scopes (v2 withdrew
    # it from both as the family's region).
    assert kept("CORPORATE_ACTION_LISTING", "DOC-AAPL-003", "Item 8.01") == 0.0
    assert kept("LIQUIDITY_GOING_CONCERN", "DOC-AAPL-003", "Item 8.01") == 1.0
    assert kept("OPERATIONS_SUPPLY", "DOC-AAPL-003", "Item 8.01") == 1.0


def test_the_integrated_selection_seals_one_receipt_with_its_routing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """requirement (C, D, E, F): one discovery routes every (issuer, topic)
    cell; the units are dealt under the topic lanes with the earlier
    filing's exact repeats served by the later reading; the bank runs only
    scoped questions, each with its scope and pairs recorded, none beyond
    the pair budget; the receipt validates, reloads, reuses for another
    request over the same content, is not reused by the candidate
    selection, and continues as one chain that keeps the routing."""

    request = _request(("AAPL", "MSFT"), selection=INTEGRATED)
    runtime, document_set, generation = _open(tmp_path, request)
    try:
        receipt, spans = runtime.select_evidence(
            request=request, document_set=document_set, generation=generation
        )
        routing = receipt.routing
        assert routing is not None
        assert routing.rules_id == ROUTING_RULES_ID
        assert routing.comparison_rules_id == COMPARISON_RULES_ID
        assert routing.allocation_rules_id == TOPIC_LANES_ALLOCATION_ID
        matters = receipt.litigation_matters
        assert matters is not None
        assert matters.allocation_rules_id == TOPIC_LANES_ALLOCATION_ID
        # The receipt records the inventories the method ran (v4: the operations
        # family joined them); the request keeps the method's one spelling.
        assert matters.families == MATTER_FAMILIES == (*ALL_FAMILIES, "OPERATIONS")
        assert receipt_matter_selection_id(receipt) == INTEGRATED.selection_id
        # Every cell of the book, once, with a residual state and its scope.
        assert {(c.entity_id, c.topic) for c in routing.cells} == {
            (entity, topic) for entity in ("AAPL", "MSFT") for topic in TOPICS
        }
        assert all(c.residual in {"QUEUED", "COVERED", "BROADER"} for c in routing.cells)
        legal = next(
            c for c in routing.cells if c.entity_id == "AAPL" and c.topic == "LEGAL_REGULATORY"
        )
        assert legal.unit_needs > 0 and legal.regions_by_basis
        assert legal.unit_needs_delivered <= legal.unit_needs
        liquidity = next(
            c
            for c in routing.cells
            if c.entity_id == "AAPL" and c.topic == "LIQUIDITY_GOING_CONCERN"
        )
        assert liquidity.unit_needs > 0, "the financing units serve the liquidity topic"
        assert liquidity.tables_without_original >= 0 and liquidity.tables == 0, (
            "a recorded text has no retained original: no table view, a representation gap"
        )
        # A valid structured route with pending delivery stays pending by
        # count, and its topic's questions are skipped by name where every
        # issuer's cell is covered -- the units are not searched twice.
        assert legal.residual == "COVERED" and legal.unit_needs > legal.unit_needs_delivered
        if all(c.residual == "COVERED" for c in routing.cells if c.topic == "LEGAL_REGULATORY"):
            assert {q.disposition for q in receipt.queries if q.topic == "LEGAL_REGULATORY"} == {
                "SKIPPED_NO_SCOPE"
            }
        # The questions: scoped, budgeted, each recorded with its disposition,
        # run topic-fair -- every topic's adverse question, then every topic's
        # first state question before any topic's second -- so a budget's
        # stop falls on the topics' last questions, never on the last topics.
        assert len(receipt.queries) == 20
        fair = topic_fair_order(EVIDENCE_QUERY_PROGRAM)
        assert [q.query_id for q in receipt.queries] == [q.query_id for q in fair]
        assert {q.query_id for q in fair} == {q.query_id for q in EVIDENCE_QUERY_PROGRAM}
        assert [q.query_id for q in fair[:8]] == [
            q.query_id for q in EVIDENCE_QUERY_PROGRAM if q.kind != "STATE"
        ]
        first_round = [q.topic for q in fair[8:16]]
        assert len(set(first_round)) == 8, "one state question per topic before any second"
        assert [q.query_id for q in fair[16:]] == [
            "Q-LIQUIDITY-CAPACITY",
            "Q-CAPITAL-RETURN",
            "Q-GOVERNANCE-MEETING",
            "Q-CORPORATE-VENTURE",
        ]
        assert query_program_hash(EVIDENCE_QUERY_PROGRAM) == receipt.query_program_hash, (
            "the program's identity is its own order"
        )
        run = [q for q in receipt.queries if q.disposition == "RUN"]
        assert run, "the residual search ran for the topics with residual scope"
        assert all(q.scope_windows is not None and q.scope_windows > 0 for q in run)
        assert all(q.reranked_pairs is not None for q in run)
        assert receipt.search_call_count == len(run) == routing.questions_run
        assert routing.questions_skipped_no_scope == sum(
            1 for q in receipt.queries if q.disposition == "SKIPPED_NO_SCOPE"
        )
        assert routing.residual_rerank_pair_budget == RESIDUAL_RERANK_PAIR_BUDGET
        assert routing.residual_reranked_pairs == sum(q.reranked_pairs or 0 for q in run)
        assert all(not q.hit_span_handles for q in receipt.queries if q.disposition != "RUN"), (
            "a skipped question has no hits"
        )
        # The exact repeats: MSFT's earlier 10-Q restates its two matters
        # word for word; the later filing is read, the earlier's needs are
        # served by that reading and say so, both sources named.
        # The residual batch is dealt by issuer-topic cell (the sealed rule), and
        # what it had no room for is sealed as a pending plan: source, range,
        # passage hash, every question that returned it, its order in the cell.
        assert routing.residual_selection_rules_id == RESIDUAL_SELECTION_RULES_ID
        # The frontier (the shared algorithm and cost initiative, A): every
        # returned candidate the batch did not read is sealed in compact
        # columns and reads back as the same records; nothing returned is
        # dropped uncounted, the inline plan of earlier records stays empty.
        assert routing.pending_candidates == () and routing.candidate_frontier is not None
        sealed_plan = routing.candidates()
        assert sealed_plan, "the fixture returns more than the batch holds"
        assert routing.candidates_beyond_plan == 0
        assert len(sealed_plan) == len(routing.candidate_frontier)
        assert all(c.state in {"PENDING", "COVERED"} and c.span_handle is None for c in sealed_plan)
        assert all(c.found_by and c.best_local_rank >= 1 for c in sealed_plan)
        per_cell: dict[tuple[str, str], list[int]] = {}
        for c in sealed_plan:
            per_cell.setdefault((c.entity_id, str(c.topic)), []).append(c.order)
        assert all(orders == list(range(1, len(orders) + 1)) for orders in per_cell.values())
        assert sum(c.candidates_pending for c in routing.cells) == sum(
            1 for c in sealed_plan if c.state == "PENDING"
        )
        assert sum(c.candidates_covered for c in routing.cells) == sum(
            1 for c in sealed_plan if c.state == "COVERED"
        )
        assert routing.candidate_span_handles == ()
        # A covered candidate names the delivered spans of the same filing
        # whose exact union holds its whole range, is never read, and lends
        # each of them every question that returned it (B: a shared range
        # is charged once); a table page or a residual span is never a cover.
        covered = [c for c in sealed_plan if c.state == "COVERED"]
        assert covered, "the fixture's unit windows hold residual candidates"
        delivered_by_handle = {s.span_handle: s for s in spans}
        found_by, _facets = span_provenance(receipt)
        for c in covered:
            assert c.covered_by is not None and c.session_index == 1
            union = [delivered_by_handle[h] for h in (c.covered_by, *c.covered_with)]
            assert not any(s.span_handle.startswith(("SPAN-S", "SPAN-X")) for s in union)
            assert all(s.document_handle == c.document_handle for s in union)
            reached = c.character_start
            for s in sorted(union, key=lambda s: s.character_start):
                assert s.character_start <= reached
                reached = max(reached, s.character_end)
            assert reached >= c.character_end, "covered means delivered whole"
            assert all(
                any(e.startswith(f"{q}:") for e in found_by[c.covered_by]) for q in c.found_by
            )
            assert any(e.startswith("C01:COVERED_CANDIDATE:") for e in found_by[c.covered_by])
        residual_ranges = {
            (s.document_handle, s.character_start, s.character_end)
            for s in spans
            if s.span_handle.startswith("SPAN-S")
        }
        assert not any(
            (c.document_handle, c.character_start, c.character_end) in residual_ranges
            for c in covered
        ), "a covered candidate is not read as a residual span"
        ledger = topic_coverage_ledger(receipt)
        assert ledger is not None and ledger["candidates"]["pending"] == sum(
            1 for c in sealed_plan if c.state == "PENDING"
        )
        assert ledger["candidates"]["covered"] == len(covered)
        assert ledger["candidates"]["sealed"] == len(sealed_plan)
        assert ledger["candidates"]["read"] == 0
        # A record sealed before the frontier (the inline plan) reads back
        # through the same accessor; both forms on one record are refused.
        inline = routing.model_copy(
            update={"candidate_frontier": None, "pending_candidates": sealed_plan[:5]}
        )
        assert inline.candidates() == sealed_plan[:5]
        with pytest.raises(ValueError, match="routing_candidates_invalid"):
            TopicRoutingRecord.model_validate(
                routing.model_copy(update={"pending_candidates": sealed_plan[:1]}).model_dump()
            )
        # The gap lines name unread candidates only where some are unread: in
        # this fixture every candidate past the batch lies inside a delivered
        # window, so none is owed and no line claims it.
        unread = any(c.state == "PENDING" for c in sealed_plan)
        assert unread == any("returned candidate(s) unread" in line for line in routing.gap_lines())
        assert routing.exact_repeats_collapsed == 2
        repeats = [c for c in routing.correspondences if c.state == "EXACT_REPEAT"]
        assert len(repeats) == 2 and all(c.entity_id == "MSFT" for c in repeats)
        assert all(c.later_document_handle != c.earlier_document_handle for c in repeats)
        served = [
            n
            for n in matters.needs
            if n.kind == "LEAD" and "exact repeat of" in n.detail and n.status == "PROVIDED_SHARED"
        ]
        assert len(served) == 2
        earlier_handle = repeats[0].earlier_document_handle
        assert all(n.document_handle == earlier_handle for n in served)
        assert all(n.span_handle in matters.delivered_matter_handles for n in served)
        read_windows = [w for w in matters.windows if w.status == "READ"]
        assert any(w.document_handle == repeats[0].later_document_handle for w in read_windows)
        assert not any(
            w.document_handle == earlier_handle
            and n.character_start is not None
            and n.character_end is not None
            and w.character_start <= n.character_start
            and n.character_end <= w.character_end
            for w in read_windows
            for n in served
        ), "the repeated units are not read twice (the region statements still are)"
        # The receipt is one sealed object over one span set.
        assert receipt.delivered_span_handles == tuple(s.span_handle for s in spans)
        assert routing.table_view_span_handles == () and routing.table_views_pending == 0
        loaded = runtime.artifacts.load(
            "retrieval-access-receipts", receipt.receipt_hash, type(receipt)
        )
        assert loaded == receipt
        assert (
            AlternativeEvidenceRetrievalAccessReceipt.model_validate(
                receipt.model_dump(mode="json")
            )
            == receipt
        )
        # Reuse: the same content and selection under another request seals
        # the selection again without a session; the candidate does not.
        other = _request(("AAPL", "MSFT"), selection=INTEGRATED, ttl_seconds=3600)
        assert other.request_hash != request.request_hash
        reused, reused_spans = runtime.select_evidence(
            request=other, document_set=document_set, generation=generation
        )
        assert reused.reused_from_receipt_hash == receipt.receipt_hash
        assert reused.routing == routing and reused_spans == spans
        assert runtime.selection_reuse_count == 1
        # Reuse under the other request's own document set and generation
        # record (the same corpus, another cutoff and build), then continue
        # the reused selection: the plan's identity binds the corpus, so the
        # chain goes on where the first reading stopped (the book journey's
        # refresh had reused every unit and its continuation was refused as
        # a changed plan).
        other_set = runtime.canonicalize(
            source_set=runtime.acquire_recorded(
                request=other,
                registry=_registry(("AAPL", "MSFT")),
                documents=(*_filings_with_debt(("AAPL", "MSFT")), *_repeating_filings()),
                published_at=_NOW + timedelta(hours=1),
            )[1],
            published_at=_NOW + timedelta(hours=1),
        )
        other_generation = runtime.build_retrieval(
            document_set=other_set, built_at=_NOW + timedelta(hours=1)
        )
        assert other_generation.generation_hash != generation.generation_hash
        assert other_generation.corpus_hash == generation.corpus_hash
        rotated, rotated_spans = runtime.select_evidence(
            request=other, document_set=other_set, generation=other_generation
        )
        assert rotated.reused_from_receipt_hash == receipt.receipt_hash
        assert rotated.retrieval_generation_hash == other_generation.generation_hash
        assert runtime.selection_reuse_count == 2
        assert rotated.litigation_matters is not None
        assert rotated.litigation_matters.plan_hash == matters.plan_hash
        rotated_chain, _ = runtime.continue_evidence(
            request=other,
            document_set=other_set,
            generation=other_generation,
            prior_receipt=rotated,
            prior_spans=rotated_spans,
            session_limit=3,
            window_limit=192,
        )
        assert rotated_chain.litigation_matters is not None
        assert rotated_chain.litigation_matters.continued_from == rotated.receipt_hash
        assert rotated_chain.litigation_matters.plan_hash == matters.plan_hash
        assert rotated_chain.litigation_matters.session_index == 2
        # A retired selection over the same corpus -- the production plan
        # (an omitted selection) or the candidate allocation -- is refused by
        # name before anything is found or opened, for a first reading and a
        # continuation alike, and never dealt under the integrated selection.
        for retired in (None, CANDIDATE):
            retired_request = _request(("AAPL", "MSFT"), selection=retired)
            with pytest.raises(ValueError, match="matter_selection_policy_retired"):
                runtime.select_evidence(
                    request=retired_request, document_set=document_set, generation=generation
                )
            with pytest.raises(ValueError, match="matter_selection_policy_retired"):
                runtime.continue_evidence(
                    request=retired_request,
                    document_set=document_set,
                    generation=generation,
                    prior_receipt=receipt,
                    prior_spans=spans,
                    session_limit=3,
                    window_limit=192,
                )
        assert runtime.selection_reuse_count == 2
        # Continuation: the chain keeps the routing and the same plan.
        assert matters.pending_windows > 0
        work_before = MODEL_WORK.snapshot()
        successor, successor_spans = runtime.continue_evidence(
            request=request,
            document_set=document_set,
            generation=generation,
            prior_receipt=receipt,
            prior_spans=spans,
            session_limit=3,
            window_limit=192,
        )
        work_after = MODEL_WORK.snapshot()
        chain = successor.litigation_matters
        assert chain is not None and chain.continued_from == receipt.receipt_hash
        assert chain.plan_hash == matters.plan_hash
        assert chain.allocation_rules_id == TOPIC_LANES_ALLOCATION_ID
        assert successor.routing is not None
        assert successor.routing.rules_id == ROUTING_RULES_ID
        assert successor.routing.questions_run == routing.questions_run
        assert successor.queries == receipt.queries, "a continuation spends no search"
        assert successor.delivered_span_handles == tuple(s.span_handle for s in successor_spans)
        # A candidate the batch did not read is COVERED only when the exact
        # union of delivered windows and typed statements holds its whole
        # range (in this fixture most of them: the unit windows hold whole
        # paragraphs); the rest stay pending and the chain reads exactly
        # those, at their sealed ranges, spending no search; the ledger
        # counts the covered, the read and the pending apart, and a partial
        # overlap is reported apart from a cover.
        assert work_after == work_before, "a continuation embeds nothing and scores no pair"
        read = successor.routing.candidate_span_handles
        successor_plan = successor.routing.candidates()
        assert read and all(h.startswith("SPAN-C02-") for h in read)
        first_plan = routing.candidates()
        assert not any(c.state == "PENDING" for c in successor_plan)
        covered = [c for c in successor_plan if c.state == "COVERED"]
        assert covered, "the fixture's windows hold whole candidates"
        # What the continuation's own windows came to hold whole is covered
        # in this session, not read; the rest of the pending is read.
        covered_now = [c for c in covered if c.session_index == 2]
        assert covered_now, "the second session's windows cover more"
        assert len(read) == sum(1 for c in first_plan if c.state == "PENDING") - len(covered_now)
        assert sum(c.candidates_covered for c in successor.routing.cells) == len(covered)
        assert sum(c.candidates_read for c in successor.routing.cells) == len(read)
        by_handle = {s.span_handle: s for s in successor_spans}
        for c in covered:
            assert c.covered_by is not None
            union = [by_handle[h] for h in (c.covered_by, *c.covered_with)]
            assert all(s.document_handle == c.document_handle for s in union)
            reached = c.character_start
            for s in sorted(union, key=lambda s: s.character_start):
                assert s.character_start <= reached
                reached = max(reached, s.character_end)
            assert reached >= c.character_end, "covered means delivered whole"
        bundles = {
            b["span_handle"]: b
            for b in bundles_of(
                successor,
                {d.semantic_handle: d for d in document_set.documents},
                successor_spans,
            )
        }
        for c in covered:
            for handle in (c.covered_by, *c.covered_with):
                cover = bundles[handle]
                assert cover["method"] != "RESIDUAL_SEARCH"
                assert any(entry.startswith(f"{c.found_by[0]}:") for entry in cover["found_by"])
                marker = f"C{c.session_index:02d}:COVERED_CANDIDATE:{c.order}"
                assert marker in cover["found_by"]
        successor_ledger = topic_coverage_ledger(successor, successor_spans)
        assert successor_ledger is not None
        assert successor_ledger["candidates"]["read"] == len(read)
        assert successor_ledger["candidates"]["covered"] == len(covered)
        assert successor_ledger["candidates"]["overlapping"] == 0, (
            "nothing pending, nothing overlaps"
        )
        first_ledger = topic_coverage_ledger(receipt, spans)
        assert first_ledger is not None and "overlapping" in first_ledger["candidates"]
        assert all("candidates_overlapping" in cell for cell in first_ledger["cells"])
        assert "overlapping" not in topic_coverage_ledger(receipt)["candidates"]  # type: ignore[index]
        # Determinism: the same prior continued again seals the same successor.
        again, again_spans = runtime.continue_evidence(
            request=request,
            document_set=document_set,
            generation=generation,
            prior_receipt=receipt,
            prior_spans=spans,
            session_limit=3,
            window_limit=192,
        )
        assert again.receipt_hash == successor.receipt_hash
        assert [s.span_handle for s in again_spans] == [s.span_handle for s in successor_spans]
        # The temporal comparison of a filing pair is sealed once and reused
        # by the chain (the shared algorithm and cost initiative, D, under
        # review finding R3): the first response computed and sealed every
        # pair and bound each closure to the record that answered it in its
        # receipt; the two continuations resolved the pairs through that
        # binding alone, and the correspondences are the same records.
        memo = runtime.sealed_comparisons
        assert memo.computed > 0 and memo.reused >= memo.computed
        assert successor.routing.correspondences == routing.correspondences
        assert again.routing is not None
        assert again.routing.correspondences == routing.correspondences
        bindings = routing.comparison_bindings()
        assert len(bindings) == memo.computed
        assert successor.routing.comparison_bindings() == bindings
        store = runtime.artifacts.root / "filing-comparisons"
        assert {p.stem for p in store.glob("*.json")} == set(bindings.values())
        closure, record_hash = next(iter(bindings.items()))
        bound_file = store / f"{record_hash}.json"
        original = bound_file.read_bytes()
        sealed = FilingComparisonRecord.model_validate_json(original)
        assert sealed.comparison_hash == closure and sealed.record_hash == record_hash

        def continued() -> Any:
            return runtime.continue_evidence(
                request=request,
                document_set=document_set,
                generation=generation,
                prior_receipt=receipt,
                prior_spans=spans,
                session_limit=3,
                window_limit=192,
            )

        # The lead's counterexample: the units rewritten (a changed
        # alignment collapsed to an exact repeat) with every seal recomputed
        # is a valid contract with the same closure -- and another record
        # hash. Under the bound file's name it is refused by the store
        # (its bytes no longer hash to the bound identity); under its own
        # name it is never consulted, and the chain still resolves the pair
        # to the record its receipt binds.
        rewritten_units = tuple(
            u.model_copy(update={"state": "EXACT_REPEAT", "ratio": 1.0, "changed_characters": 0})
            if u.state != "EXACT_REPEAT"
            else u
            for u in sealed.units
        )
        assert rewritten_units != sealed.units, "the fixture's pair has a changed unit to collapse"
        content = {
            **sealed.model_dump(mode="json", exclude={"units", "results_hash", "record_hash"}),
            "units": [u.model_dump(mode="json") for u in rewritten_units],
            "results_hash": FilingComparisonRecord.results_hash_of(rewritten_units),
        }
        rewritten = FilingComparisonRecord(
            **{k: v for k, v in content.items() if k != "units"},
            units=rewritten_units,
            record_hash=FilingComparisonRecord.record_hash_of(content),
        )
        assert rewritten.comparison_hash == closure and rewritten.record_hash != record_hash
        bound_file.write_bytes(rewritten.model_dump_json().encode("utf-8"))
        try:
            with pytest.raises(ValueError, match="artifact_tampered"):
                continued()
        finally:
            bound_file.write_bytes(original)
        (store / f"{rewritten.record_hash}.json").write_bytes(
            rewritten.model_dump_json().encode("utf-8")
        )
        reused_before, computed_before = memo.reused, memo.computed
        beside = continued()[0]
        assert beside.routing is not None
        assert beside.routing.correspondences == routing.correspondences
        assert beside.routing.comparison_bindings() == bindings, "the bound record, not the rewrite"
        assert memo.reused > reused_before and memo.computed == computed_before
        (store / f"{rewritten.record_hash}.json").unlink()
        # The bound record's own seals disagreeing (a unit's basis altered
        # under the bound name) is refused by the contract, never recomputed.
        bound_file.write_bytes(original.replace(b'"basis":"', b'"basis":"altered ', 1))
        try:
            with pytest.raises(ValueError, match="identity_invalid"):
                continued()
        finally:
            bound_file.write_bytes(original)
        # A verified record that answers another closure under the bound name.
        (other_hash,) = [h for h in bindings.values() if h != record_hash][:1] or [None]
        if other_hash is not None:
            bound_file.write_bytes((store / f"{other_hash}.json").read_bytes())
            try:
                with pytest.raises(ValueError, match="artifact_tampered"):
                    continued()
            finally:
                bound_file.write_bytes(original)
        # A missing bound record is a missing proof: the pair is computed
        # and sealed again, the same correspondences, the chain continues.
        bound_file.unlink()
        missing_before, computed_before = memo.missing, memo.computed
        recomputed = continued()[0]
        assert recomputed.routing is not None
        assert recomputed.routing.correspondences == routing.correspondences
        assert memo.missing == missing_before + 1 and memo.computed == computed_before + 1
        assert bound_file.is_file(), "sealed again under the same record hash"
        assert bound_file.read_bytes() == original
        # A record on disk under a computed record's name that is not it is
        # refused by name at the seal -- never staged, never bound, never
        # overwritten (section X); the record itself, held, is its own seal.
        bound_file.write_bytes(original + b"\n")
        try:
            with pytest.raises(
                AlternativeEvidencePublicationError, match="artifact_identity_reused"
            ):
                memo.seal(sealed)
        finally:
            bound_file.write_bytes(original)
        assert memo.seal(sealed) == record_hash and memo.commit() == frozenset()
        assert bound_file.read_bytes() == original
        # A record under the closure's name (the layout before the binding)
        # is never consulted: the chain binds record hashes.
        (store / f"{closure}.json").write_bytes(rewritten.model_dump_json().encode("utf-8"))
        legacy = continued()[0]
        assert legacy.routing is not None
        assert legacy.routing.correspondences == routing.correspondences
        (store / f"{closure}.json").unlink()
        # The same texts under another arrangement of units are another
        # closure: the unit inventories are part of what a lookup is keyed on.
        assert (
            FilingComparisonRecord.closure_hash(
                **{
                    **sealed.model_dump(
                        include={
                            "entity_id",
                            "family",
                            "later_content_sha256",
                            "earlier_content_sha256",
                            "earlier_units_hash",
                            "comparison_rules_id",
                            "structure_rules_id",
                            "unit_rules",
                        }
                    ),
                    "later_units_hash": FilingComparisonRecord.units_hash_of(
                        (("M-KO-099-L-001", "LITIGATION", "0" * 64),)
                    ),
                }
            )
            != closure
        )
        delivered_before = {(c.entity_id, c.topic): c.unit_needs_delivered for c in routing.cells}
        for cell in successor.routing.cells:
            assert cell.unit_needs_delivered >= delivered_before[(cell.entity_id, cell.topic)]
        assert sum(c.unit_needs_delivered for c in successor.routing.cells) > sum(
            delivered_before.values()
        )
        # The records a routing computes are admitted together, once (section
        # W): the first response's pairs cost one admission, the continuations
        # that resolved through the bindings none, the recomputation of the
        # missing record one more; and a refused batch is used for its own
        # routing, unsealed and unbound, so the next chain computes again.
        assert memo.admissions == 2, "the first response, then the missing proof"
        refused_runtime_calls = [0]

        def refuse(_bytes: int) -> None:
            refused_runtime_calls[0] += 1
            raise RuntimeError("storage.managed_capacity_exceeded")

        runtime.storage_admission = refuse
        unsealed_before = memo.unsealed
        # Another request identity over the same filings, its selection
        # computed by a session (the sealed one is not consulted here).
        opted = _request(
            ("AAPL", "MSFT"),
            selection=MatterSelectionPolicy(
                method=MATTER_SELECTION_INTEGRATED, families=ALL_FAMILIES
            ),
            ttl_seconds=5_400,
        )
        monkeypatch.setattr(runtime.sealed_selections, "find", lambda **_kwargs: None)
        for path in store.glob("*.json"):
            path.unlink()
        receipts_before = {
            p.name for p in (runtime.artifacts.root / "retrieval-access-receipts").glob("*.json")
        }
        sets_before = {
            p.name for p in (runtime.artifacts.root / "resolved-span-sets").glob("*.json")
        }
        with pytest.raises(RuntimeError, match="managed_capacity_exceeded"):
            runtime.select_evidence(request=opted, document_set=document_set, generation=generation)
        assert memo.unsealed == unsealed_before + len(bindings), (
            "the routing's batch refused, used for its routing, unbound"
        )
        assert refused_runtime_calls[0] >= 2, (
            "the comparisons' batch, then the selection's own artifacts"
        )
        assert not list(store.glob("*.json")), "nothing placed under a refused admission"
        # The receipt and the span set are durable writes like any other
        # (section W): refused, neither is placed and the selection is not
        # sealed as delivered.
        assert {
            p.name for p in (runtime.artifacts.root / "retrieval-access-receipts").glob("*.json")
        } == receipts_before
        assert {
            p.name for p in (runtime.artifacts.root / "resolved-span-sets").glob("*.json")
        } == sets_before
        runtime.storage_admission = None
        admitted_bytes: list[int] = []
        runtime.storage_admission = admitted_bytes.append
        admitted_receipt, _admitted_spans = runtime.select_evidence(
            request=opted, document_set=document_set, generation=generation
        )
        placed_receipt = (
            runtime.artifacts.root
            / "retrieval-access-receipts"
            / f"{admitted_receipt.receipt_hash}.json"
        )
        placed_set = (
            runtime.artifacts.root / "resolved-span-sets" / f"{admitted_receipt.span_set_hash}.json"
        )
        assert placed_receipt.is_file() and placed_set.is_file()
        assert placed_receipt.stat().st_size + placed_set.stat().st_size in admitted_bytes, (
            "the pair admitted together, at the bytes placed"
        )
        assert (
            len(admitted_bytes) >= 2
            and sum(admitted_bytes)
            >= sum(p.stat().st_size for p in store.glob("*.json"))
            + placed_receipt.stat().st_size
            + placed_set.stat().st_size
        )
        runtime.storage_admission = None
    finally:
        runtime.close()


def test_the_table_share_is_dealt_across_the_topics_that_have_tables() -> None:
    """requirement (the shared-gap closeout, boundary B): the session's table
    share is dealt topic-fair -- the topics in their declared order, each
    taking the best-ranked table of its own queue, a table serving several
    topics taken once and counted for all of them, a topic with no table
    passed over, a table the boundary cannot render passed over by name --
    and no priority follows from which family recognized the region.
    Measured on the retained book under the former basis rank, one topic
    held 2,467 routed tables and was dealt none."""

    def need(document: str, ordinal: int, *topics: str, recency: int = 0) -> TableNeed:
        placeholder = TablePlaceholder(
            ordinal=ordinal,
            character_start=ordinal * 100,
            character_end=ordinal * 100 + 20,
            rows_total=3,
            caption=None,
            footnotes=(),
            path=("NOTE",),
            family="OPERATIONS_RESULTS_OUTLOOK",
            unit_declaration=None,
        )
        return TableNeed(
            document_key=document,
            entity_id="AAPL",
            topics=tuple(EvidenceTopic(t) for t in topics),
            placeholder=placeholder,
            region_heading="NOTE",
            rank=(recency, ordinal),
        )

    liquidity = [need("d1", n, "LIQUIDITY_GOING_CONCERN") for n in range(1, 6)]
    shared = need("d2", 1, "LIQUIDITY_GOING_CONCERN", "OPERATIONS_SUPPLY", recency=-1)
    operations = [need("d3", n, "OPERATIONS_SUPPLY") for n in range(1, 3)]
    governance = need("d4", 1, "GOVERNANCE_CONTROLS")
    unrenderable = need("d5", 1, "COMMERCIAL_COUNTERPARTY")
    needs = [*liquidity, shared, *operations, governance, unrenderable]
    renderable = lambda n: n.document_key != "d5"  # noqa: E731
    views = topic_fair_views(needs, share=6, renderable=renderable)
    dealt = [(v.document_key, v.placeholder.ordinal) for v in views]
    # Round one in the topics' declared order: liquidity takes the shared
    # table (its latest), operations finds it dealt and takes its own,
    # governance its only one, commercial's is refused by name; round two
    # and three go to the topics that still hold tables.
    assert dealt == [("d2", 1), ("d3", 1), ("d4", 1), ("d1", 1), ("d3", 2), ("d1", 2)]
    assert len(set(dealt)) == len(dealt), "no table dealt twice"
    assert ("d5", 1) not in dealt
    assert views == topic_fair_views(needs, share=6, renderable=renderable)
    assert topic_fair_views(needs, share=0, renderable=renderable) == []
    whole = topic_fair_views(needs, share=16, renderable=renderable)
    assert [(v.document_key, v.placeholder.ordinal) for v in whole] == [
        *dealt,
        ("d1", 3),
        ("d1", 4),
        ("d1", 5),
    ], "the share left over is dealt to the topics with tables, the refused one never"
    assert len(topic_fair_views([governance], share=8, renderable=renderable)) == 1


def test_the_residual_batch_is_dealt_by_issuer_topic_cell_and_the_rest_is_pending() -> None:
    """requirement (the completeness assignment, W1.1; `RESIDUAL_SELECTION_RULES_ID`):
    an issuer's sixteen are dealt in rounds over its cells in the topics'
    declared order -- every cell's best before any cell's second, a cell
    ordered by its questions' own ranks in rounds (v2: every question's
    first, then every question's second; no score compared across
    questions or cells) -- under the unchanged sixteen and 128; a
    same-content repeat from the same filing is grouped, never dealt and
    never pending; what the batch has no room for is returned per cell in
    the cell's order; a ninth issuer past the packet cap keeps everything
    pending. Measured on the retained book under the question-round rule:
    ten development units and three cases returned and cut."""

    def hit(
        local: int, order: int, rank: int, issuer: str, doc: str, content: str, query: str
    ) -> Any:
        return (local, order, rank, f"SPAN-S{order:02d}-R{rank:02d}", issuer, doc, content, query)

    cell_of = {
        "Q-LIQUIDITY": EvidenceTopic.LIQUIDITY_GOING_CONCERN,
        "Q-LIQUIDITY-COVENANTS": EvidenceTopic.LIQUIDITY_GOING_CONCERN,
        "Q-DILUTION": EvidenceTopic.CAPITAL_DILUTION,
        "Q-COMMERCIAL": EvidenceTopic.COMMERCIAL_COUNTERPARTY,
    }
    ranked = []
    # Liquidity: fourteen candidates from two questions, seven each, at the
    # questions' local ranks 1..7; the covenants question's scores would
    # run lower, and no score enters the key.
    for i in range(14):
        ranked.append(
            hit(
                i // 2 + 1,
                0 if i % 2 == 0 else 8,
                i + 1,
                "AAPL",
                "DOC-AAPL-001",
                f"liq{i}",
                "Q-LIQUIDITY" if i % 2 == 0 else "Q-LIQUIDITY-COVENANTS",
            )
        )
    # Capital: six at local ranks 1..6; the second is a same-content repeat of the first.
    for i in range(6):
        ranked.append(
            hit(
                i + 1,
                1,
                20 + i,
                "AAPL",
                "DOC-AAPL-002",
                "cap0" if i == 1 else f"cap{i}",
                "Q-DILUTION",
            )
        )
    # Commercial: two, at the deepest global ranks of all.
    for i in range(2):
        ranked.append(hit(i + 1, 6, 40 + i, "AAPL", "DOC-AAPL-001", f"com{i}", "Q-COMMERCIAL"))
    ranked.sort()
    selected, groups, pending = _allocate_by_cell(ranked, cell_of)
    assert len(selected) == SPANS_PER_ISSUER
    by_handle = {item[3]: item for item in ranked}
    chosen = [by_handle[h] for h in selected]
    # Round one holds every cell's best, the deepest-ranked commercial
    # candidate among them: nothing is compared across cells. Inside the
    # liquidity cell the two questions alternate by their own ranks.
    assert [cell_of[c[7]] for c in chosen[:3]] == [
        EvidenceTopic.LIQUIDITY_GOING_CONCERN,
        EvidenceTopic.CAPITAL_DILUTION,
        EvidenceTopic.COMMERCIAL_COUNTERPARTY,
    ]
    assert chosen[0][6] == "liq0" and chosen[1][6] == "cap0" and chosen[2][6] == "com0"
    # The capital repeat is grouped under its first, and the group costs the
    # cell no turn: round two deals the third capital candidate (cap2).
    (group,) = groups
    assert group.representative_span_handle == chosen[1][3]
    assert [by_handle[h][6] for h in group.member_span_handles] == ["cap0"]
    assert chosen[3][6] == "liq1" and chosen[4][6] == "cap2" and chosen[5][6] == "com1"
    counts = {}
    for c in chosen:
        counts[cell_of[c[7]]] = counts.get(cell_of[c[7]], 0) + 1
    assert counts == {
        EvidenceTopic.LIQUIDITY_GOING_CONCERN: 9,
        EvidenceTopic.CAPITAL_DILUTION: 5,
        EvidenceTopic.COMMERCIAL_COUNTERPARTY: 2,
    }, "sixteen: liquidity's tenth waits, capital's five distinct all dealt"
    assert [c[6] for c, _cover in pending[("AAPL", EvidenceTopic.LIQUIDITY_GOING_CONCERN)]] == [
        f"liq{i}" for i in range(9, 14)
    ]
    assert all(
        cover == () for _c, cover in pending[("AAPL", EvidenceTopic.LIQUIDITY_GOING_CONCERN)]
    )
    assert ("AAPL", EvidenceTopic.CAPITAL_DILUTION) not in pending
    assert ("AAPL", EvidenceTopic.COMMERCIAL_COUNTERPARTY) not in pending
    assert _allocate_by_cell(ranked, cell_of) == (selected, groups, pending), "deterministic"
    # A candidate a delivered span covers costs its cell no turn: liq1 is
    # skipped with its cover named, round two deals liq2 in its place, and
    # the covered candidate keeps its place in the cell's sealed order.
    covered_selected, _groups, covered_pending = _allocate_by_cell(
        ranked, cell_of, lambda item: ("SPAN-M01-R0001",) if item[6] == "liq1" else ()
    )
    covered_chosen = [by_handle[h][6] for h in covered_selected]
    assert "liq1" not in covered_chosen and covered_chosen[3] == "liq2"
    assert len(covered_selected) == SPANS_PER_ISSUER
    liquidity_rest = covered_pending[("AAPL", EvidenceTopic.LIQUIDITY_GOING_CONCERN)]
    assert (liquidity_rest[0][0][6], liquidity_rest[0][1]) == ("liq1", ("SPAN-M01-R0001",))
    assert [c[6] for c, _cover in liquidity_rest][1:] == [f"liq{i}" for i in range(10, 14)]
    # The packet cap: nine issuers of sixteen; the ninth in the issuers'
    # order is dealt nothing and keeps every candidate pending.
    crowd = []
    for n in range(9):
        for i in range(16):
            crowd.append(
                hit(i + 1, 0, n * 16 + i + 1, f"I{n}", f"DOC-I{n}-001", f"c{n}-{i}", "Q-LIQUIDITY")
            )
    crowd.sort()
    selected, _groups, pending = _allocate_by_cell(crowd, cell_of)
    assert len(selected) == MAXIMUM_PACKET_SPANS
    assert len(pending[("I8", EvidenceTopic.LIQUIDITY_GOING_CONCERN)]) == 16


def test_delivered_tables_follow_sealed_page_identities_not_a_prefix_of_the_needs() -> None:
    """requirement (W1.2, the audit): after the topic-fair share the first N
    table needs are not the N tables dealt; the chain's delivered tables and
    the next pages are derived from the sealed page records -- a partial
    page resumes at its next row, a page sealed without progress does not,
    a delivered or refused table is never dealt again."""

    from alphalattice.evidence.alternative_evidence.analysis.contracts import (
        TableViewRecord,
        TableViewRefusalRecord,
        TopicRoutingRecord,
    )
    from alphalattice.evidence.alternative_evidence.analysis.routing import (
        TableNeed,
        TopicRouting,
    )
    from alphalattice.evidence.alternative_evidence.documents.structure import DocumentStructure
    from alphalattice.evidence.alternative_evidence.retrieval.session import InspectedDocument

    def need(document: str, ordinal: int, *topics: str, rows: int = 3) -> TableNeed:
        placeholder = TablePlaceholder(
            ordinal=ordinal,
            character_start=ordinal * 100,
            character_end=ordinal * 100 + 20,
            rows_total=rows,
            caption=None,
            footnotes=(),
            path=("NOTE",),
            family="OPERATIONS_RESULTS_OUTLOOK",
            unit_declaration=None,
        )
        return TableNeed(
            document_key=document,
            entity_id="AAPL",
            topics=tuple(EvidenceTopic(t) for t in topics),
            placeholder=placeholder,
            region_heading="NOTE",
            rank=(0, ordinal),
        )

    needs = (
        need("d1", 1, "LIQUIDITY_GOING_CONCERN"),
        need("d1", 2, "LIQUIDITY_GOING_CONCERN"),
        need("d2", 1, "OPERATIONS_SUPPLY", rows=40),
        need("d2", 2, "OPERATIONS_SUPPLY"),
    )
    routing = TopicRouting(
        rules_id="r",
        comparison_rules_id="c",
        cells=(),
        regions=(),
        table_needs=needs,
        correspondences=(),
        repeats={},
    )
    inspected = {
        key: InspectedDocument(
            document_id=key,
            document_handle=f"DOC-AAPL-00{key[-1]}",
            entity_id="AAPL",
            document_type="10-K",
            revision_label="r",
            text="x",
            structure=DocumentStructure("x", document_type="10-K"),
        )
        for key in ("d1", "d2")
    }
    # The first session dealt d1's first and d2's first (topic-fair), d2's
    # first in part; the prefix of the needs would say d1's first and second.
    views = (
        TableViewRecord(
            span_handle="SPAN-X01-R0001",
            document_handle="DOC-AAPL-001",
            entity_id="AAPL",
            topics=(EvidenceTopic.LIQUIDITY_GOING_CONCERN,),
            table_ordinal=1,
            rows_total=3,
            rows_from=1,
            rows_to=3,
            remaining_rows=0,
        ),
        TableViewRecord(
            span_handle="SPAN-X01-R0002",
            document_handle="DOC-AAPL-002",
            entity_id="AAPL",
            topics=(EvidenceTopic.OPERATIONS_SUPPLY,),
            table_ordinal=1,
            rows_total=40,
            rows_from=1,
            rows_to=25,
            remaining_rows=15,
        ),
    )
    assert delivered_table_identities(views, inspected) == (("d1", 1), ("d2", 1))
    assert delivered_table_identities(views, inspected) != tuple(
        (n.document_key, n.placeholder.ordinal) for n in needs[:2]
    )
    prior = TopicRoutingRecord(
        rules_id="r",
        comparison_rules_id="c",
        allocation_rules_id="a",
        exact_repeats_collapsed=0,
        table_view_span_handles=tuple(v.span_handle for v in views),
        table_views=views,
        table_view_refusals=(
            TableViewRefusalRecord(
                document_handle="DOC-AAPL-001",
                entity_id="AAPL",
                table_ordinal=2,
                code="alternative_evidence.table_view_headings_absent",
            ),
        ),
        table_views_pending=2,
        residual_rerank_pair_budget=0,
        residual_reranked_pairs=0,
        questions_run=0,
        questions_skipped_no_scope=0,
        questions_skipped_budget=0,
        unrouted_windows=0,
    )
    pages = plan_table_pages(
        prior=prior, routing=routing, inspected=inspected, renderable=lambda n: True, share=8
    )
    assert [(n.document_key, n.placeholder.ordinal, row) for n, row in pages] == [
        ("d2", 1, 26),
        ("d2", 2, 1),
    ], "the partial page resumes at its next row; the refused and the delivered are not dealt again"
    unknown = views[1].model_copy(update={"rows_from": 0, "rows_to": 0, "remaining_rows": 0})
    stale = prior.model_copy(update={"table_views": (views[0], unknown)})
    pages = plan_table_pages(
        prior=stale, routing=routing, inspected=inspected, renderable=lambda n: True, share=8
    )
    assert [(n.document_key, n.placeholder.ordinal, row) for n, row in pages] == [("d2", 2, 1)], (
        "a page sealed without row progress is not resumed from a guess"
    )


def test_a_table_is_complete_only_when_its_declared_rows_are_proved_delivered() -> None:
    """requirement (the first-release safety closeout, finding B): a table's
    state is derived from its sealed pages and nothing else. Contiguous
    pages from row 1 to the declared total, with no clipped row, are
    COMPLETE; rows after the covered prefix, a gap between pages or a
    clipped row leave it PARTIAL -- resumable at the row after the prefix,
    or not resumable when the clipped row is all that remains; a page
    sealed before pages carried rows is UNKNOWN_PROGRESS, never complete
    and never resumed; a duplicate page adds no rows; a table whose
    rendering declares no rows is complete on its one page. The routing's
    gap lines and the read model's cell states name each kind, and the
    continuation scope is PENDING only while something resumes."""

    from alphalattice.evidence.alternative_evidence.analysis.contracts import (
        IssuerTopicCellRecord,
        TableViewRecord,
        TopicRoutingRecord,
        cell_table_states,
        table_progress_of,
    )

    def page(handle: str, ordinal: int, rows_from: int, rows_to: int, remaining: int, **extra: Any):
        return TableViewRecord(
            span_handle=handle,
            document_handle="DOC-AAPL-001",
            entity_id="AAPL",
            topics=(EvidenceTopic.LIQUIDITY_GOING_CONCERN,),
            table_ordinal=ordinal,
            rows_total=rows_to + remaining,
            rows_from=rows_from,
            rows_to=rows_to,
            remaining_rows=remaining,
            **extra,
        )

    views = (
        page("SPAN-X01-R0001", 1, 1, 25, 38),  # first page of a 63-row table
        page("SPAN-X02-R0001", 1, 26, 63, 0, session_index=2),  # its last page
        page("SPAN-X01-R0002", 2, 1, 25, 38),  # a first page, nothing after
        page("SPAN-X01-R0003", 3, 1, 25, 38),  # a gap: rows 26-39 never delivered
        page("SPAN-X02-R0003", 3, 40, 63, 0, session_index=2),
        page("SPAN-X01-R0004", 4, 0, 0, 0),  # sealed before pages carried rows
        page("SPAN-X01-R0005", 5, 1, 1, 0, rows_clipped=1),  # one row, clipped
        page("SPAN-X01-R0006", 6, 1, 1, 2, rows_clipped=1),  # clipped, rows after
        page("SPAN-X01-R0007", 7, 1, 0, 0),  # a rendering that declares no rows
        page("SPAN-X01-R0008", 8, 1, 25, 38),
        page("SPAN-X02-R0008", 8, 1, 25, 38, session_index=2),  # a duplicate page
    )
    progress = table_progress_of(views)
    by_ordinal = {ordinal: value for (_handle, ordinal), value in progress.items()}
    assert [
        (o, v.state, v.rows_delivered, v.next_row, v.resumable)
        for o, v in sorted(by_ordinal.items())
    ] == [
        (1, "COMPLETE", 63, None, False),
        (2, "PARTIAL", 25, 26, True),
        (3, "PARTIAL", 25, 26, True),
        (4, "UNKNOWN_PROGRESS", 0, None, False),
        (5, "PARTIAL", 1, None, False),
        (6, "PARTIAL", 1, 2, True),
        (7, "COMPLETE", 0, None, False),
        (8, "PARTIAL", 25, 26, True),
    ]
    assert by_ordinal[3].gap and by_ordinal[3].rows_declared == 63, "a gap is never inferred across"
    assert by_ordinal[8].pages == 2 and by_ordinal[8].rows_delivered == 25, (
        "a duplicate adds no rows"
    )
    assert by_ordinal[5].rows_clipped == 1 and by_ordinal[1].rows_clipped == 0
    assert cell_table_states(progress, "AAPL", EvidenceTopic.LIQUIDITY_GOING_CONCERN) == {
        "COMPLETE": 2,
        "PARTIAL": 5,
        "UNKNOWN_PROGRESS": 1,
    }
    assert cell_table_states(progress, "AAPL", EvidenceTopic.OPERATIONS_SUPPLY) == {
        "COMPLETE": 0,
        "PARTIAL": 0,
        "UNKNOWN_PROGRESS": 0,
    }
    cell = IssuerTopicCellRecord(
        entity_id="AAPL",
        topic=EvidenceTopic.LIQUIDITY_GOING_CONCERN,
        regions=1,
        unit_needs=0,
        typed_observations=0,
        tables=9,
        tables_delivered=8,
        tables_without_original=0,
        residual="COVERED",
    )
    routing = TopicRoutingRecord(
        rules_id="r",
        comparison_rules_id="c",
        allocation_rules_id="a",
        cells=(cell,),
        exact_repeats_collapsed=0,
        table_view_span_handles=tuple(v.span_handle for v in views),
        table_views=views,
        table_views_pending=1,
        residual_rerank_pair_budget=0,
        residual_reranked_pairs=0,
        questions_run=0,
        questions_skipped_no_scope=0,
        questions_skipped_budget=0,
        unrouted_windows=0,
    )
    assert routing.gap_lines() == (
        "AAPL LIQUIDITY_GOING_CONCERN: 1 table(s) not dealt a view; 5 table(s) delivered in "
        "part; 1 table(s) whose page progress is unknown",
    )
    from alphalattice.evidence.alternative_evidence.analysis.read_model import _cell_view

    view = _cell_view(cell, progress)
    assert view["delivery_state"] == "PARTIAL"
    assert view["incomplete"] == [
        "TABLES_NOT_DEALT",
        "TABLES_DELIVERED_IN_PART",
        "TABLES_PROGRESS_UNKNOWN",
    ]
    assert (view["tables_complete"], view["tables_partial"], view["tables_progress_unknown"]) == (
        2,
        5,
        1,
    )
    whole = _cell_view(
        cell.model_copy(update={"tables": 1, "tables_delivered": 1}),
        {("DOC-AAPL-001", 1): by_ordinal[1]},
    )
    assert whole["delivery_state"] == "COMPLETE" and whole["incomplete"] == []
    unknown_only = _cell_view(
        cell.model_copy(update={"tables": 1, "tables_delivered": 1}),
        {("DOC-AAPL-001", 4): by_ordinal[4]},
    )
    assert unknown_only["delivery_state"] == "PARTIAL", "unknown progress is never complete"
    assert unknown_only["incomplete"] == ["TABLES_PROGRESS_UNKNOWN"]
    clipped_only = _cell_view(
        cell.model_copy(update={"tables": 1, "tables_delivered": 1}),
        {("DOC-AAPL-001", 5): by_ordinal[5]},
    )
    assert clipped_only["incomplete"] == ["TABLES_DELIVERED_IN_PART"]
    complete_cell = cell.model_copy(update={"tables": 1, "tables_delivered": 1})
    complete_progress = {("DOC-AAPL-001", 1): by_ordinal[1]}
    source_gap = _cell_view(
        complete_cell.model_copy(
            update={"gaps": ("SOURCE_GAP: no periodic filing (10-K or 10-Q) held for the issuer",)}
        ),
        complete_progress,
    )
    assert source_gap["delivery_state"] == "PARTIAL" and source_gap["incomplete"] == ["SOURCE_GAP"]
    route_gap = _cell_view(
        cell.model_copy(
            update={
                "tables": 0,
                "tables_delivered": 0,
                "typed_observations": 1,
                "residual": "BROADER",
                "gaps": ("ROUTE_GAP: no region routes to the topic",),
            }
        ),
        {},
    )
    assert route_gap["delivery_state"] == "PARTIAL" and route_gap["incomplete"] == ["ROUTE_GAP"], (
        "a typed observation is delivered; the broader pass is a search, not a scope"
    )
    no_route = _cell_view(
        cell.model_copy(update={"tables": 0, "tables_delivered": 0, "residual": "BROADER"}),
        {},
    )
    assert no_route["delivery_state"] == "NO_ROUTE"


def test_a_chain_reads_its_pending_candidates_and_table_pages_without_a_search(
    tmp_path: Path,
) -> None:
    """requirement (W1.2, 5.3): a unit whose matter plan the first session
    served whole still continues -- the sealed pending plan's residual
    candidates are read at their ranges (series C) under the program's reads
    with no query embedded and no pair scored, a table the first page left
    in part resumes at its next row with its headings (series X02), the
    delivered tables follow the sealed identities, the chain survives a
    restart from the sealed artifacts, a candidate whose sealed passage the
    source no longer holds is refused by name, and a chain with nothing
    pending anywhere is refused by name."""

    html = original_html(rows=60)
    transport = SecScenarioTransport(
        registry={"AAPL": REGISTRY["AAPL"]},
        filings={AAPL: [TEN_K]},
        bodies={SecScenarioTransport.locator(AAPL, TEN_K): html},
    )
    runtime = _runtime(tmp_path)
    try:
        from tests.alternative_evidence_desk.incremental_acquisition_support import (
            _request as _live_request,
        )

        base = _live_request(("AAPL",), budget=3)
        request = seal_contract(
            AlternativeEvidenceRequest,
            "request_hash",
            **base.model_dump(exclude={"request_hash", "matter_selection"}),
            matter_selection=INTEGRATED,
        )
        _registry_snapshot, _snapshot, source_set = _acquire(runtime, transport, request)
        document_set = runtime.canonicalize(
            source_set=source_set, published_at=request.evidence_as_of
        )
        generation = runtime.build_retrieval(
            document_set=document_set, built_at=request.evidence_as_of
        )
        receipt, spans = runtime.select_evidence(
            request=request, document_set=document_set, generation=generation
        )
        routing = receipt.routing
        matters = receipt.litigation_matters
        assert routing is not None and matters is not None
        assert matters.pending_windows == 0, "the fixture's plan is served in one session"
        (view_handle,) = routing.table_view_span_handles
        (view,) = routing.table_views
        assert view.rows_from == 1 and view.remaining_rows > 0 and view.rows_to < view.rows_total
        # Finding B (the first-release safety closeout): one page of a
        # multi-page table is not the table. Every cell the table serves is
        # PARTIAL and says why; the gap line names the part; the ledger, the
        # bundle and the continuation scope agree; and the continuation is
        # PENDING on the page alone, with no matter window pending.
        identity = (view.document_handle, view.table_ordinal)
        first_progress = routing.table_progress()[identity]
        assert first_progress.state == "PARTIAL" and first_progress.next_row == view.rows_to + 1
        assert first_progress.rows_delivered == view.rows_to
        assert first_progress.rows_declared == view.rows_to + view.remaining_rows
        first_ledger = topic_coverage_ledger(receipt)
        assert first_ledger is not None
        served = [c for c in first_ledger["cells"] if c["tables"]]
        assert served and all(
            c["delivery_state"] == "PARTIAL"
            and c["tables_partial"] == 1
            and c["tables_complete"] == 0
            and "TABLES_DELIVERED_IN_PART" in c["incomplete"]
            for c in served
        ), "a delivered page never completes the table's cell"
        assert all(c["delivery_state"] != "COMPLETE" for c in first_ledger["cells"] if c["tables"])
        assert any("1 table(s) delivered in part" in line for line in routing.gap_lines())
        assert first_ledger["tables"] == {
            "pages_delivered": 1,
            "complete": 0,
            "partial": 1,
            "partial_resumable": 1,
            "progress_unknown": 0,
            "not_dealt": 0,
            "unrenderable": 0,
            "without_original": 0,
        }
        (page_bundle,) = [b for b in bundles_of(receipt, {}, spans) if b["table"] is not None]
        assert page_bundle["delivery"]["whole"] is False
        assert page_bundle["table"]["progress"]["state"] == "PARTIAL"
        assert page_bundle["table"]["progress"]["next_row"] == view.rows_to + 1
        assert routed_pending_work(receipt) == {
            "candidates": sum(1 for c in routing.candidates() if c.state == "PENDING"),
            "partial_table_pages": 1,
            "tables_not_dealt": 0,
        }
        first_scope = litigation_continuation_scope(receipt)
        assert first_scope is not None and first_scope["state"] == "PENDING"
        assert first_scope["pending_windows"] == 0 and first_scope["partial_table_pages"] == 1
        pending_before = sum(1 for c in routing.candidates() if c.state == "PENDING")
        print(
            "DEBUG chain pending_before",
            pending_before,
            "covered",
            sum(1 for c in routing.candidates() if c.state == "COVERED"),
            "sealed",
            len(routing.candidates()),
        )
        work_before = MODEL_WORK.snapshot()
        second, second_spans = runtime.continue_evidence(
            request=request,
            document_set=document_set,
            generation=generation,
            prior_receipt=receipt,
            prior_spans=spans,
            session_limit=4,
            window_limit=192,
        )
        assert MODEL_WORK.snapshot() == work_before, "a read, not a search campaign"
        chain = second.litigation_matters
        assert chain is not None and chain.session_index == 2 and chain.read_windows == 0
        assert second.routing is not None
        assert all(h.startswith("SPAN-C02-") for h in second.routing.candidate_span_handles)
        assert len(second.routing.candidate_span_handles) == min(pending_before, SPANS_PER_ISSUER)
        assert len(second.routing.table_view_span_handles) == 2
        first_page, next_page = second.routing.table_views
        assert first_page == view
        assert next_page.span_handle.startswith("SPAN-X02-")
        assert next_page.rows_from == view.rows_to + 1 and next_page.session_index == 2
        assert next_page.table_ordinal == view.table_ordinal
        by_handle = {s.span_handle: s for s in second_spans}
        page_span = by_handle[next_page.span_handle]
        assert page_span.table_view is not None
        assert page_span.table_view.rows_from == view.rows_to + 1
        assert page_span.table_view.headings == by_handle[view_handle].table_view.headings
        rendered_total = page_span.table_view.rows_total
        assert (
            f"[rows {next_page.rows_from}-{next_page.rows_to} of {rendered_total}"
            in page_span.excerpt
        )
        assert second.delivered_span_handles == tuple(s.span_handle for s in second_spans)
        assert second.delivered_span_handles[-1] == next_page.span_handle
        cells = {(c.entity_id, c.topic): c for c in second.routing.cells}
        assert cells[("AAPL", "LIQUIDITY_GOING_CONCERN")].tables_delivered == 1, (
            "two pages of one table are one table delivered"
        )
        ledger = topic_coverage_ledger(second)
        assert ledger is not None and ledger["tables"]["pages_delivered"] == 2
        second_progress = second.routing.table_progress()[identity]
        assert second_progress.pages == 2 and second_progress.rows_delivered == next_page.rows_to
        if next_page.remaining_rows == 0:
            assert second_progress.state == "COMPLETE" and second_progress.next_row is None
            assert ledger["tables"]["complete"] == 1 and ledger["tables"]["partial"] == 0
            assert all(
                c["tables_complete"] == 1 and "TABLES_DELIVERED_IN_PART" not in c["incomplete"]
                for c in ledger["cells"]
                if c["tables"]
            ), "every declared row delivered from sealed pages: the table is complete"
            assert not any("delivered in part" in line for line in second.routing.gap_lines())
        else:
            assert second_progress.state == "PARTIAL"
            assert second_progress.next_row == next_page.rows_to + 1
            assert ledger["tables"]["complete"] == 0 and ledger["tables"]["partial"] == 1
        assert ledger["candidates"]["read"] == len(second.routing.candidate_span_handles)
        loaded = runtime.artifacts.load(
            "retrieval-access-receipts", second.receipt_hash, type(second)
        )
        assert loaded == second
        # A changed analysis method (the policy hash the chain's plan was
        # sealed under has moved) never continues the prior plan silently:
        # the continuation refuses by name, and nothing of the plan is read.
        sealed_under = runtime.analysis_policy_hash
        runtime.analysis_policy_hash = "0" * 64
        try:
            with pytest.raises(ValueError, match="litigation_continuation_policy_changed"):
                runtime.continue_evidence(
                    request=request,
                    document_set=document_set,
                    generation=generation,
                    prior_receipt=second,
                    prior_spans=second_spans,
                    session_limit=4,
                    window_limit=192,
                )
        finally:
            runtime.analysis_policy_hash = sealed_under
    finally:
        runtime.close()
    # Restart: a fresh runtime over the sealed artifacts holds the chain --
    # the sealed span set is the chain's own -- and the table's last page
    # having been read, a third session has nothing pending anywhere and is
    # refused by name, never a session that reads nothing.
    runtime = _runtime(tmp_path)
    try:
        from alphalattice.evidence.alternative_evidence.retrieval.contracts import (
            AlternativeEvidenceResolvedSpanSet,
        )

        loaded = runtime.artifacts.load(
            "retrieval-access-receipts", second.receipt_hash, type(second)
        )
        span_set = runtime.artifacts.load(
            "resolved-span-sets", loaded.span_set_hash, AlternativeEvidenceResolvedSpanSet
        )
        prior_spans = tuple(span_set.spans)
        assert [s.span_handle for s in prior_spans] == [s.span_handle for s in second_spans]
        assert loaded.routing is not None
        if next_page.remaining_rows == 0 and not any(
            c.state == "PENDING" for c in loaded.routing.candidates()
        ):
            with pytest.raises(ValueError, match="litigation_continuation_nothing_pending"):
                runtime.continue_evidence(
                    request=request,
                    document_set=document_set,
                    generation=generation,
                    prior_receipt=loaded,
                    prior_spans=prior_spans,
                    session_limit=4,
                    window_limit=192,
                )
        else:
            third, third_spans = runtime.continue_evidence(
                request=request,
                document_set=document_set,
                generation=generation,
                prior_receipt=loaded,
                prior_spans=prior_spans,
                session_limit=4,
                window_limit=192,
            )
            assert third.litigation_matters is not None
            assert third.litigation_matters.session_index == 3
            assert third.delivered_span_handles == tuple(s.span_handle for s in third_spans)
            assert third.routing is not None
            last_page = third.routing.table_views[-1]
            assert last_page.span_handle.startswith("SPAN-X03-")
            assert last_page.rows_from == next_page.rows_to + 1 and last_page.session_index == 3
            assert (last_page.document_handle, last_page.table_ordinal) == (
                view.document_handle,
                view.table_ordinal,
            ), "three pages of one table"
            third_cells = {(c.entity_id, c.topic): c for c in third.routing.cells}
            assert third_cells[("AAPL", "LIQUIDITY_GOING_CONCERN")].tables_delivered == 1
    finally:
        runtime.close()


def test_a_routed_table_is_delivered_as_a_view_inside_the_matter_allowance(
    tmp_path: Path,
) -> None:
    """requirement (B, F): a live-acquired 10-K whose debt note holds a table
    the canonical text did not carry: the routing places the table in the
    liquidity (and capital) cells, the integrated selection issues its view
    under the matter allowance's table share (series X) and reads it from
    the retained original, the receipt lists the view among its delivered
    spans, and the unit windows keep the rest of the allowance. The note's
    second table has no heading row: the boundary refuses it by name, the
    refusal is recorded as a representation gap and the share moves on."""

    html = original_html(headless_table=True)
    transport = SecScenarioTransport(
        registry={"AAPL": REGISTRY["AAPL"]},
        filings={AAPL: [TEN_K]},
        bodies={SecScenarioTransport.locator(AAPL, TEN_K): html},
    )
    runtime = _runtime(tmp_path)
    try:
        from tests.alternative_evidence_desk.incremental_acquisition_support import (
            _request as _live_request,
        )

        base = _live_request(("AAPL",), budget=3)
        request = seal_contract(
            AlternativeEvidenceRequest,
            "request_hash",
            **base.model_dump(exclude={"request_hash", "matter_selection"}),
            matter_selection=INTEGRATED,
        )
        _registry_snapshot, _snapshot, source_set = _acquire(runtime, transport, request)
        document_set = runtime.canonicalize(
            source_set=source_set, published_at=request.evidence_as_of
        )
        generation = runtime.build_retrieval(
            document_set=document_set, built_at=request.evidence_as_of
        )
        receipt, spans = runtime.select_evidence(
            request=request, document_set=document_set, generation=generation
        )
        routing = receipt.routing
        assert routing is not None
        (view_handle,) = routing.table_view_span_handles
        assert view_handle.startswith("SPAN-X01-R"), "one counter: the view follows the windows"
        assert routing.table_views_pending == 0
        (refusal,) = routing.table_view_refusals
        assert refusal.code == "alternative_evidence.table_view_headings_absent"
        assert refusal.table_ordinal == 2 and refusal.entity_id == "AAPL"
        cells = {(c.entity_id, c.topic): c for c in routing.cells}
        assert cells[("AAPL", "LIQUIDITY_GOING_CONCERN")].tables == 2
        assert cells[("AAPL", "LIQUIDITY_GOING_CONCERN")].tables_delivered == 1
        assert cells[("AAPL", "LIQUIDITY_GOING_CONCERN")].tables_unrenderable == 1
        assert cells[("AAPL", "LIQUIDITY_GOING_CONCERN")].tables_without_original == 0
        assert any("unrenderable" in line for line in routing.gap_lines())
        view = next(s for s in spans if s.span_handle == view_handle)
        assert view.table_view is not None and view.table_view.rows_total == 6
        assert "Instrument: Term loan facility" in view.excerpt
        assert receipt.delivered_span_handles[-1] == view_handle
        matters = receipt.litigation_matters
        assert matters is not None
        assert matters.window_allowance == 64 - min(TABLE_VIEWS_PER_SESSION, 1)
        assert (
            AlternativeEvidenceRetrievalAccessReceipt.model_validate(
                receipt.model_dump(mode="json")
            )
            == receipt
        )
        # The routing over the shared inventory is deterministic and names
        # the placeholder's region for the liquidity topic.
        session = runtime.open_session(
            document_set=document_set, generation=generation, evidence_as_of=request.evidence_as_of
        )
        try:
            inventory = shared_inventory(session)
            documents = routed_documents(session, inventory)
            (document,) = documents
            assert document.accepted_at is not None or document.published_at is not None
            again = route_documents(
                documents=documents,
                needs_by_document=inventory.needs_by_document,
                placeholders={k: session.table_placeholders(k) for k in inventory.inspected},
                originals={k: session.original_available(k) for k in inventory.inspected},
                typed_by_issuer={},
            )
            need, second = again.table_needs
            assert second.placeholder.ordinal == 2 and need.rank < second.rank
            # An issuer of the unit with no admitted document: eight cells,
            # each a source gap, nothing to search.
            absent = route_documents(
                documents=documents,
                needs_by_document=inventory.needs_by_document,
                placeholders={},
                originals={},
                typed_by_issuer={},
                entity_ids=("AAPL", "ZZZZ"),
            )
            empty = [c for c in absent.cells if c.entity_id == "ZZZZ"]
            assert len(empty) == len(TOPICS)
            assert all(c.residual == "NO_SOURCE" and not c.residual_ranges for c in empty)
            assert all(
                c.gaps == ("SOURCE_GAP: no admitted document held for the issuer",) for c in empty
            )
            assert need.region_heading.startswith("NOTE 9") and "LIQUIDITY_GOING_CONCERN" in {
                str(t) for t in need.topics
            }
            financing_needs = [
                n
                for n in inventory.needs_by_document[document.document_key].needs
                if n.family == MATTER_FAMILY_FINANCING and n.priority > 0
            ]
            assert financing_needs
            topics = topics_of_need(financing_needs[0], document_type="10-K", text=document.text)
            assert topics[0] == EvidenceTopic.LIQUIDITY_GOING_CONCERN
        finally:
            session.close()
    finally:
        runtime.close()


def _liquidity_paragraph(number: int) -> str:
    """One distinct MD&A liquidity paragraph no inventory family reads: a
    statement of cash, working capital and covenant compliance for one
    quarter, different figures each time, so the liquidity questions return
    many candidates and none is a matter, an event or an instrument."""

    return (
        f"During the quarter ended in month {number}, cash and cash equivalents totaled "
        f"${1_000 + number * 37} million and working capital was ${2_500 + number * 53} million. "
        f"Operating activities provided ${300 + number * 11} million of cash, and the Company "
        f"remained in compliance with the financial covenants of its credit agreement, with "
        f"{number + 4} quarters of undrawn capacity available under the revolving facility."
    )


def _liquidity_filing(paragraphs: int) -> str:
    from tests.alternative_evidence_desk.financing_support import ANNUAL_COVER

    lines = [
        *ANNUAL_COVER,
        "ITEM 7. MANAGEMENT\u2019S DISCUSSION AND ANALYSIS",
        "Liquidity and Capital Resources",
        *(_liquidity_paragraph(number) for number in range(1, paragraphs + 1)),
        "ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA",
        "NOTE 12: COMMITMENTS AND CONTINGENCIES",
        "The Company is involved in various legal proceedings.",
        "PART IV",
        "ITEM 15. EXHIBITS AND FINANCIAL STATEMENT SCHEDULES",
        "See the exhibit index.",
        "SIGNATURES",
    ]
    return "\n\n".join(lines) + "\n"


def test_a_continuation_reads_the_frontier_and_skips_what_the_chain_has_covered(
    tmp_path: Path,
) -> None:
    """requirement (the shared algorithm and cost initiative, A and B): a
    10-K whose MD&A holds forty distinct liquidity paragraphs no inventory
    reads makes the liquidity questions return more candidates than the
    batch holds; every one of them is sealed in the frontier (past the six
    a cell the earlier plan kept), a continuation reads them round by round
    at their sealed ranges with no search, and a pending candidate whose
    range the chain has since delivered through another channel is covered
    by that span rather than read again. The chain's ledger counts sealed,
    pending, read and covered apart."""

    from tests.alternative_evidence_desk.document_intelligence_support import _document_with_text

    request = _request(("AAPL",), selection=INTEGRATED)
    runtime = _runtime(tmp_path)
    try:
        _snapshot, source_set = runtime.acquire_recorded(
            request=request,
            registry=_registry(("AAPL",)),
            documents=(
                *_filings_with_debt(("AAPL",)),
                _document_with_text(
                    "AAPL", text=_liquidity_filing(40), form="10-K", revision="10-k-2024-mdna"
                ),
            ),
            published_at=_NOW,
        )
        document_set = runtime.canonicalize(source_set=source_set, published_at=_NOW)
        generation = runtime.build_retrieval(document_set=document_set, built_at=_NOW)
        receipt, spans = runtime.select_evidence(
            request=request, document_set=document_set, generation=generation
        )
        routing = receipt.routing
        assert routing is not None and routing.candidate_frontier is not None
        plan = routing.candidates()
        liquidity = [
            c
            for c in plan
            if c.topic == EvidenceTopic.LIQUIDITY_GOING_CONCERN and c.state == "PENDING"
        ]
        assert len(liquidity) > 6, "the frontier keeps more than the six a cell the old plan sealed"
        assert routing.candidates_beyond_plan == 0
        assert max(c.order for c in liquidity) == len(
            [c for c in plan if c.topic == EvidenceTopic.LIQUIDITY_GOING_CONCERN]
        )
        first_ledger = topic_coverage_ledger(receipt)
        assert first_ledger is not None
        assert first_ledger["candidates"]["sealed"] == len(plan)
        assert first_ledger["candidates"]["pending"] == sum(1 for c in plan if c.state == "PENDING")
        work_before = MODEL_WORK.snapshot()
        second, second_spans = runtime.continue_evidence(
            request=request,
            document_set=document_set,
            generation=generation,
            prior_receipt=receipt,
            prior_spans=spans,
            session_limit=3,
            window_limit=192,
        )
        assert MODEL_WORK.snapshot() == work_before, (
            "a continuation embeds nothing and scores no pair"
        )
        assert second.routing is not None
        read = second.routing.candidate_span_handles
        assert read and all(h.startswith("SPAN-C02-") for h in read)
        second_plan = second.routing.candidates()
        # Candidates the session's own windows covered are not read; the rest
        # are read sixteen an issuer, from the frontier.
        covered_now = [c for c in second_plan if c.state == "COVERED" and c.session_index == 2]
        assert len(read) == min(
            sum(1 for c in plan if c.state == "PENDING") - len(covered_now), SPANS_PER_ISSUER
        ), "sixteen an issuer a session, from the frontier, less what this session covered"
        assert all(
            c.covered_by is not None and c.covered_by.startswith("SPAN-M") for c in covered_now
        )
        assert {c.span_handle for c in second_plan if c.state == "READ"} == set(read)
        # The read candidates are read round by round over the cells in the
        # topics' order: their cell orders form a prefix of each cell's order.
        by_cell: dict[tuple[str, str], list[int]] = {}
        for c in second_plan:
            if c.state == "READ":
                by_cell.setdefault((c.entity_id, str(c.topic)), []).append(c.order)
        for (_entity, topic), orders in by_cell.items():
            cell_orders = sorted(
                c.order for c in second_plan if str(c.topic) == topic and c.state != "COVERED"
            )
            assert sorted(orders) == cell_orders[: len(orders)], topic
        assert sum(c.candidates_read for c in second.routing.cells) == len(read)
        by_handle = {s.span_handle: s for s in second_spans}
        for c in second_plan:
            if c.state == "READ":
                assert c.span_handle is not None and c.session_index == 2
                span = by_handle[c.span_handle]
                assert span.character_start <= c.character_start
                assert c.character_end <= span.character_end
        second_ledger = topic_coverage_ledger(second)
        assert second_ledger is not None and second_ledger["candidates"]["read"] == len(read)
        assert second_ledger["candidates"]["sealed"] == len(plan)
        # A pending candidate whose range the chain has delivered since (a
        # unit window read in the continuation, here: the prior spans as the
        # delivered ranges) is covered, not read: prove the rule at the
        # owner with a delivered range over one pending candidate.
        from alphalattice.evidence.alternative_evidence.analysis.packet import (
            continue_routed_reads,
            covering_span_handles,
            delivered_ranges,
        )

        pending_now = [c for c in plan if c.state == "PENDING"]
        assert pending_now, "the first response's frontier owes candidates"
        # A candidate whose sealed passage the source no longer holds is
        # refused by name before any read; nothing is manufactured from a
        # stale plan.
        stale_index = next(i for i, c in enumerate(plan) if c.state == "PENDING")
        moved = routing.model_copy(
            update={
                "candidate_frontier": CandidateFrontier.from_records(
                    (
                        *plan[:stale_index],
                        plan[stale_index].model_copy(update={"passage_hash": "0" * 64}),
                        *plan[stale_index + 1 :],
                    )
                )
            }
        )
        with pytest.raises(ValueError, match="pending_candidate_source_moved"):
            runtime.continue_evidence(
                request=request,
                document_set=document_set,
                generation=generation,
                prior_receipt=receipt.model_copy(update={"routing": moved}),
                prior_spans=spans,
                session_limit=3,
                window_limit=192,
            )
        target = pending_now[0]
        held = delivered_ranges(spans)
        assert (
            covering_span_handles(
                held, target.document_handle, target.character_start, target.character_end
            )
            == ()
        )
        cover = {
            target.document_handle: [
                (target.character_start, target.character_end, "SPAN-M03-R0001")
            ]
        }
        assert covering_span_handles(
            cover, target.document_handle, target.character_start, target.character_end
        ) == ("SPAN-M03-R0001",)
        session = runtime.retrieval.open_session(
            document_set=document_set, generation=generation, evidence_as_of=request.evidence_as_of
        )
        try:
            from alphalattice.evidence.alternative_evidence.analysis.packet import (
                route_session,
                shared_inventory,
            )

            inventory = shared_inventory(session)
            topic_routing = route_session(
                session, inventory, receipt.typed_disclosures, entity_ids=("AAPL",)
            )
            continued = continue_routed_reads(
                session=session,
                prior=routing,
                inventory=inventory,
                routing=topic_routing,
                session_index=2,
                pages=(),
                refusals=(),
                unrenderable=(),
                delivered=cover,
            )
        finally:
            session.close()
        states = {c.state for c in continued.pending_candidates}
        assert "COVERED" in states
        covered_now = next(
            c
            for c in continued.pending_candidates
            if c.character_start == target.character_start
            and c.document_handle == target.document_handle
            and c.topic == target.topic
        )
        assert covered_now.state == "COVERED" and covered_now.covered_by == "SPAN-M03-R0001"
        assert covered_now.session_index == 2
        assert not any(
            s.character_start == target.character_start for s in continued.candidate_spans
        ), "a covered candidate is not read"
    finally:
        runtime.close()


def _delivered(*items: tuple[str, int, int, str]) -> tuple[Any, ...]:
    """Resolved-span stand-ins for the cover rule: (handle, start, end, document)."""

    from types import SimpleNamespace

    return tuple(
        SimpleNamespace(
            span_handle=handle, character_start=start, character_end=end, document_handle=doc
        )
        for handle, start, end, doc in items
    )


def test_a_candidate_is_covered_only_when_its_whole_range_was_delivered() -> None:
    """requirement (review finding R1, reproduced on `ae9bc093`: a candidate
    [0, 100) was discharged as COVERED by a span holding [0, 80) under an
    80% rule, and a table page's placeholder range counted as a cover):
    COVERED means every character of the candidate's range was delivered
    by the exact union of the same filing's unit windows, typed statements
    and read candidates. A span holding most of the range is a partial
    overlap -- the undelivered tail may hold the qualification or the
    counter-statement -- and leaves the candidate pending, reported apart;
    two spans jointly holding the range cover it and both are named; a gap
    between them does not; a table page covers nothing; another filing's
    span covers nothing; character offsets are the unit on both sides, so
    a multibyte character counts once."""

    from alphalattice.evidence.alternative_evidence.analysis.packet import (
        covering_span_handles,
        delivered_ranges,
        overlapping_span_handles,
    )

    # The lead's counterexample: most of the range, not all of it.
    held = delivered_ranges(_delivered(("SPAN-M01-R0001", 0, 80, "DOC-A")))
    assert covering_span_handles(held, "DOC-A", 0, 100) == ()
    assert overlapping_span_handles(held, "DOC-A", 0, 100) == ("SPAN-M01-R0001",)
    # A decisive qualification in the uncovered tail is exactly what an
    # 80% rule discharged: the candidate stays pending until read.
    tail = delivered_ranges(_delivered(("SPAN-T01-R0001", 0, 99, "DOC-A")))
    assert covering_span_handles(tail, "DOC-A", 0, 100) == ()
    # Truly complete coverage, by one span or by an exact union.
    whole = delivered_ranges(_delivered(("SPAN-M01-R0001", 0, 120, "DOC-A")))
    assert covering_span_handles(whole, "DOC-A", 0, 100) == ("SPAN-M01-R0001",)
    joint = delivered_ranges(
        _delivered(("SPAN-M01-R0002", 50, 100, "DOC-A"), ("SPAN-M01-R0001", 0, 50, "DOC-A"))
    )
    assert covering_span_handles(joint, "DOC-A", 0, 100) == ("SPAN-M01-R0001", "SPAN-M01-R0002")
    assert overlapping_span_handles(joint, "DOC-A", 0, 100) == ()
    overlapping_union = delivered_ranges(
        _delivered(("SPAN-M01-R0001", 0, 60, "DOC-A"), ("SPAN-T01-R0003", 40, 100, "DOC-A"))
    )
    assert covering_span_handles(overlapping_union, "DOC-A", 0, 100) == (
        "SPAN-M01-R0001",
        "SPAN-T01-R0003",
    )
    # Disjoint coverage with a gap between the spans is not a cover.
    gap = delivered_ranges(
        _delivered(("SPAN-M01-R0001", 0, 40, "DOC-A"), ("SPAN-M01-R0002", 60, 100, "DOC-A"))
    )
    assert covering_span_handles(gap, "DOC-A", 0, 100) == ()
    assert overlapping_span_handles(gap, "DOC-A", 0, 100) == ("SPAN-M01-R0001", "SPAN-M01-R0002")
    # A table page's placeholder range never covers a passage; the residual
    # batch's own spans never cover another candidate.
    page = delivered_ranges(
        _delivered(("SPAN-X01-R0001", 0, 200, "DOC-A"), ("SPAN-S01-R0001", 0, 200, "DOC-A"))
    )
    assert page == {}
    assert covering_span_handles(page, "DOC-A", 0, 100) == ()
    # The wrong filing: the same range in another document is not a cover.
    other = delivered_ranges(_delivered(("SPAN-M01-R0001", 0, 200, "DOC-B")))
    assert covering_span_handles(other, "DOC-A", 0, 100) == ()
    # Multibyte text: the cover is decided in characters, as the excerpt's
    # range is; a byte-based reading of these ranges would differ.
    text = "货币资金及现金等价物合计 ¥1,234 百万元。" * 8
    assert len(text.encode("utf-8")) != len(text)
    span_end = len(text) - 3
    multibyte = delivered_ranges(_delivered(("SPAN-M01-R0001", 0, span_end, "DOC-A")))
    assert covering_span_handles(multibyte, "DOC-A", 4, span_end) == ("SPAN-M01-R0001",)
    assert covering_span_handles(multibyte, "DOC-A", 4, len(text)) == ()
    # The sealed record names the union: the span holding the first
    # character as `covered_by`, the rest as `covered_with`; the pair is
    # only valid on a COVERED record and the frontier carries it.
    record = PendingCandidateRecord(
        entity_id="AAPL",
        topic=EvidenceTopic.LIQUIDITY_GOING_CONCERN,
        document_handle="DOC-A",
        character_start=0,
        character_end=100,
        passage_hash="a" * 64,
        found_by=("Q-LIQUIDITY",),
        best_global_rank=1,
        best_local_rank=1,
        order=1,
        state="COVERED",
        session_index=1,
        covered_by="SPAN-M01-R0001",
        covered_with=("SPAN-M01-R0002",),
    )
    frontier = CandidateFrontier.from_records((record,))
    assert frontier.records()[0] == record
    assert CandidateFrontier.model_validate_json(frontier.model_dump_json()).records() == (record,)
    with pytest.raises(ValueError, match="pending_candidate_invalid"):
        PendingCandidateRecord(
            **{
                **record.model_dump(),
                "state": "PENDING",
                "session_index": None,
                "covered_by": None,
            }
        )
    with pytest.raises(ValueError, match="pending_candidate_invalid"):
        PendingCandidateRecord(**{**record.model_dump(), "covered_with": ("SPAN-M01-R0001",)})
    # A frontier sealed before the exact-union rule carries no
    # `covered_with` column and still reads back.
    older = frontier.model_dump()
    older.pop("covered_with")
    assert CandidateFrontier.model_validate(older).records()[0].covered_with == ()


def test_a_retired_residual_policy_is_refused_by_name_and_never_dealt_as_the_default(
    tmp_path: Path,
) -> None:
    """requirement (section X, C2): the two residual opt-ins of section V
    (`allocation=CONTEXT_COMPLETE`, `residual_search=GAP_DIRECTED`) are
    retired -- a request that carries one still seals and reads back with
    its identity, and the runtime refuses it by name before a session
    opens, for a first reading and for a continuation alike, never dealing
    it under the default; a receipt dealt under a retired rule (its routing
    names the retired rules id, or its queries hold a gap search) reads
    back and is never the default's sealed selection."""

    default = _request(
        ("AAPL",),
        selection=MatterSelectionPolicy(method=MATTER_SELECTION_INTEGRATED, families=ALL_FAMILIES),
    )
    retired = _request(
        ("AAPL",),
        selection=MatterSelectionPolicy(
            method=MATTER_SELECTION_INTEGRATED,
            families=ALL_FAMILIES,
            residual_search="GAP_DIRECTED",
        ),
    )
    assert retired.matter_selection is not None and retired.matter_selection.retired
    assert AlternativeEvidenceRequest.model_validate_json(retired.model_dump_json()) == retired
    assert (
        retired.matter_selection_id == default.matter_selection_id + "#residual_search=GAP_DIRECTED"
    )
    runtime, document_set, generation = _open(tmp_path, default)
    try:
        receipt, spans = runtime.select_evidence(
            request=default, document_set=document_set, generation=generation
        )
        assert receipt.routing is not None
        assert receipt.routing.residual_selection_rules_id == RESIDUAL_SELECTION_RULES_ID
        with pytest.raises(ValueError, match="matter_selection_policy_retired"):
            runtime.select_evidence(
                request=retired, document_set=document_set, generation=generation
            )
        with pytest.raises(ValueError, match="matter_selection_policy_retired"):
            runtime.continue_evidence(
                request=retired,
                document_set=document_set,
                generation=generation,
                prior_receipt=receipt,
                prior_spans=spans,
                session_limit=3,
                window_limit=192,
            )
        receipts = runtime.artifacts.values(
            "retrieval-access-receipts", AlternativeEvidenceRetrievalAccessReceipt
        )
        assert [r.receipt_hash for r in receipts] == [receipt.receipt_hash], "nothing else sealed"
        index = runtime.sealed_selections
        assert index._eligible(receipt)
        under_retired_rule = receipt.model_copy(
            update={
                "routing": receipt.routing.model_copy(
                    update={"residual_selection_rules_id": CONTEXT_COMPLETE_SELECTION_RULES_ID}
                )
            }
        )
        assert not index._eligible(under_retired_rule)
        gap = receipt.queries[0].model_copy(
            update={"query_id": receipt.queries[0].query_id + "-GAP1"}
        )
        assert not index._eligible(
            receipt.model_copy(update={"queries": (gap, *receipt.queries[1:])})
        )
    finally:
        runtime.close()
