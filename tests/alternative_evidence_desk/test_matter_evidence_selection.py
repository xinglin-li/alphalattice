"""Litigation matters segmented by the source's own signposts (rules v3)
and the needs a filing states, dealt by the shared allocator's sealed service
order. Controlled fixtures of real filings' shapes: two filings in one
paragraph (Madera, then Howell), a dated filing cut by a page break, a
materiality closing after the last matter, an accrual lead-in that says it
applies to the matters below, a filing that names no matter, an explicit
reference to a filing not held. The candidate delivery that ran these needs
under its own allocation went with the candidate selection (first-release
integration T5); the integrated selection deals them by topic lanes, and the
allocator's lane function is the caller's -- here one lane per family."""

from __future__ import annotations

import pytest

from alphalattice.evidence.alternative_evidence.analysis.matters import (
    MATTER_INVENTORY_RULES_ID,
    MATTER_SEGMENTATION_RULES_ID,
    TOPIC_LANES_ALLOCATION_ID,
    Allocation,
    DocumentNeeds,
    EvidenceNeed,
    allocate_windows,
    cross_references,
    document_needs,
    matter_inventory,
    region_statements,
    resolve_reference,
)
from alphalattice.evidence.alternative_evidence.documents.structure import DocumentStructure
from alphalattice.evidence.alternative_evidence.retrieval.session import (
    MAXIMUM_ISSUED_MATTER_WINDOWS,
)
from tests.alternative_evidence_desk.litigation_support import (
    ACCRUAL_LEAD_IN,
    BETWEEN_MATTERS,
    GENERIC_ONLY,
    TRAILING_CLOSING,
    _filing,
    _matters_filing,
)


def _by_family(need: EvidenceNeed) -> str:
    return need.family


def _allocate(documents: tuple[DocumentNeeds, ...], **options: int) -> Allocation:
    return allocate_windows(
        documents,
        lane_of=_by_family,
        lanes_in_order=("LITIGATION", "CORPORATE_EVENT", "FINANCING", "OPERATIONS"),
        **options,
    )


def test_segmentation_rules_v3_split_by_the_source_and_leave_region_statements_to_the_region() -> (
    None
):
    """requirement (4.1/4.2): a further filing with its own caption opens a
    matter inside a paragraph; a paragraph cut by a page break is read
    whole so its dated filing is seen; a materiality closing after the last
    matter is the region's, never that matter's; the accrual lead-in that
    says 'some matters described below' is associated with the matters and
    a closing that names none stays at the region with its scope open."""

    text = _matters_filing()
    structure = DocumentStructure(text, document_type="10-Q")
    production = matter_inventory(text, structure, document_id="T")
    candidate = matter_inventory(
        text, structure, document_id="T", rules=MATTER_SEGMENTATION_RULES_ID
    )
    assert production.rules_id == MATTER_INVENTORY_RULES_ID
    assert candidate.rules_id == MATTER_SEGMENTATION_RULES_ID
    with pytest.raises(ValueError, match="matter_inventory_rules_unknown"):
        matter_inventory(text, structure, document_id="T", rules="v9")
    howell = text.index("A second class action")
    usvi = text.index("On April 11, 2025")
    closing = text.index("The Company does not believe")

    def owner_of(inventory: object, position: int) -> str | None:
        for matter in inventory.matters:  # type: ignore[attr-defined]
            if any(start <= position < end for start, end in matter.ranges):
                return str(matter.title)
        return None

    # v2: one paragraph, one matter -- Howell's filing sits under Madera's
    # lead; the page-cut filing and the closing are swept into it too.
    assert owner_of(production, howell) == owner_of(production, text.index("In January 2026"))
    assert owner_of(production, usvi) == owner_of(production, howell)
    assert owner_of(production, closing) == owner_of(production, usvi)
    # v3: Howell opens a matter of its own from its sentence; the joined
    # paragraph opens the Virgin Islands matter; the closing is unassigned.
    howell_title = owner_of(candidate, howell)
    assert howell_title is not None and howell_title.startswith("Howell v. Costco Wholesale")
    assert owner_of(candidate, text.index("In January 2026")) != howell_title
    assert owner_of(candidate, usvi) not in {None, owner_of(candidate, howell)}
    assert owner_of(candidate, closing) is None
    assert (
        closing,
        closing + len(TRAILING_CLOSING),
        "region-level statement",
    ) in candidate.unassigned
    titles = [(m.title, m.basis) for m in candidate.matters if m.named]
    assert (howell_title, "second filing") in titles
    assert any(
        basis == "dated filing" and "Virgin Islands" in text[m.ranges[0][0] : m.ranges[0][1]]
        for m in candidate.matters
        for basis in [m.basis]
        if m.named
    )
    # The lead-in under the note's sub-heading and the closing are region
    # statements: the first says it applies to the matters below, the
    # second names none by itself but says 'any pending claim'.
    statements = region_statements(text, candidate)
    lead_in = next(s for s in statements if s.character_start == text.index(ACCRUAL_LEAD_IN))
    assert lead_in.position == "LEAD_IN" and lead_in.cued and lead_in.refers_to_matters
    assert lead_in.scope == "REGION"
    assert "matters described below" in lead_in.reference_text
    trailing = next(s for s in statements if s.character_start == closing)
    assert trailing.position == "TRAILING" and trailing.refers_to_matters
    # References: the item's 'See Note 7' resolves to the note region by
    # structure; the quarterly report it names is another filing.
    references = cross_references(text, candidate)
    kinds = {(r.kind, r.target) for r in references}
    assert ("NOTE", "Note 7") in kinds
    assert any(kind == "FILING" and "March 31, 2026" in target for kind, target in kinds)
    note = resolve_reference(next(r for r in references if r.kind == "NOTE"), candidate, structure)
    assert note.kind == "REGION" and note.label.startswith("NOTE 7")
    filing = resolve_reference(
        next(r for r in references if r.kind == "FILING"), candidate, structure
    )
    assert filing.kind == "FILING"


def test_needs_are_dealt_in_the_sealed_service_order_and_adjacent_needs_share_a_window() -> None:
    """requirement (4.3, service order v2): each lane's region qualifications
    first (a filing that names no matter has only those), the issuers in
    turn; then a lane alternates an opening with an extension of a
    disclosure already opened while both exist, so no opening waits behind
    every other opening; a need that touches a chosen window joins it; what
    the allowance or the budget refuses is listed with the reason, never
    silently dropped."""

    matters_text = _matters_filing(extra=12, long=True)
    generic_text = _filing(note_lines=[GENERIC_ONLY])
    window_bytes = 1800
    prepared = []
    for key, issuer, text in (("d-a", "AAPL", matters_text), ("d-b", "MSFT", generic_text)):
        structure = DocumentStructure(text, document_type="10-Q")
        inventory = matter_inventory(
            text, structure, document_id=key.upper(), rules=MATTER_SEGMENTATION_RULES_ID
        )
        prepared.append(
            document_needs(
                text,
                structure,
                inventory,
                document_key=key,
                issuer=issuer,
                window_bytes=window_bytes,
            )
        )
    needs_a, needs_b = prepared
    assert not needs_b.named_matters and needs_b.statements
    # A region's opening and closing qualifications are class 1; a statement
    # between two matters that refers to them is class 3, read with them;
    # the filing that names no matter has its whole disclosure in class 1.
    by_start = {n.character_start: n for n in needs_a.needs if n.kind == "REGION_STATEMENT"}
    assert by_start[matters_text.index(ACCRUAL_LEAD_IN)].priority == 1
    assert by_start[matters_text.index(TRAILING_CLOSING)].priority == 1
    assert by_start[matters_text.index(BETWEEN_MATTERS)].priority == 3
    assert by_start[matters_text.index(BETWEEN_MATTERS)].refers_to_matters
    generic = {n.character_start: n for n in needs_b.needs if n.kind == "REGION_STATEMENT"}
    assert generic[generic_text.index(GENERIC_ONLY)].priority == 1
    assert needs_b.statements[0].position == "WHOLE"
    kinds_a = [need.kind for need in needs_a.needs]
    assert kinds_a.count("LEAD") == len(needs_a.named_matters) == 18
    assert "QUALIFICATION" in kinds_a and "REFERENCE" in kinds_a and "REGION_STATEMENT" in kinds_a
    qualifications = [n for n in needs_a.needs if n.kind == "QUALIFICATION"]
    # Every named matter depends on the lead-in, the statement between the
    # matters and the closing, which all say they apply to the matters.
    assert len(qualifications) == 3 * len(needs_a.named_matters)
    references = [n for n in needs_a.needs if n.kind == "REFERENCE"]
    assert {n.target.kind for n in references if n.target is not None} == {
        "REGION",
        "FILING",
        "UNLOCATED",
    }

    allocation = _allocate(
        (needs_a, needs_b),
        window_allowance=MAXIMUM_ISSUED_MATTER_WINDOWS,
        window_bytes=window_bytes,
    )
    assert allocation.rules_id == TOPIC_LANES_ALLOCATION_ID and not allocation.refused
    served = [[need.kind for need in window.needs] for window in allocation.windows]
    # The first window of each issuer's lane is an opening: the first matter
    # of the litigation filing, the whole disclosure of the filing that
    # names no matter; the lead-in that qualifies the matters follows its
    # region's first opening and joins its window.
    assert served[0][0] == "LEAD" and allocation.windows[0].document_key == "d-a"
    assert served[1][0] == "REGION_STATEMENT" and allocation.windows[1].document_key == "d-b"
    assert allocation.windows[1].needs[0].priority == 1
    # The declared order: each lane's region qualifications before its
    # openings; then openings and extensions alternate inside the lane: the
    # long matter (the last one here) is extended right after it is opened,
    # before the statement between the matters, and its parts stay in order.
    dealt = [need.kind for need in allocation.admitted]
    assert "CONTINUATION" in dealt
    lane_a = [need for need in allocation.admitted if need.document_key == "d-a"]
    assert lane_a[0].kind == "LEAD" and lane_a[1].priority == 1 and lane_a[1].part == 1, (
        "a region's qualification follows the region's first opening in its lane"
    )
    long_matter = allocation.admitted[dealt.index("CONTINUATION")].disclosure
    opened_at = next(
        i for i, need in enumerate(allocation.admitted) if need.disclosure == long_matter
    )
    assert allocation.admitted[opened_at].kind == "LEAD"
    assert dealt.index("CONTINUATION") == opened_at + 1
    long_parts = [need.part for need in allocation.admitted if need.disclosure == long_matter]
    assert long_parts == sorted(long_parts) and long_parts[0] == 1
    assert all(need.disclosure for need in allocation.admitted if need.priority > 0)
    # Adjacent needs joined one window: the lead-in and the first matter are
    # one excerpt, and no byte is planned twice within a window.
    shared = [w for w in allocation.windows if len(w.needs) > 1]
    assert shared, "adjacent needs share a window"
    lead_in_window = next(
        w
        for w in allocation.windows
        if any(n.kind == "REGION_STATEMENT" and n.part == 1 for n in w.needs)
        and w.document_key == "d-a"
        and w.character_start == matters_text.index(ACCRUAL_LEAD_IN)
    )
    assert {n.kind for n in lead_in_window.needs} >= {"REGION_STATEMENT", "LEAD"}
    assert all(w.source_bytes <= window_bytes for w in allocation.windows)
    for window in allocation.windows:
        for need in window.needs:
            assert need.character_start is not None and need.character_end is not None
            assert window.character_start <= need.character_start
            assert need.character_end <= window.character_end
    # A budget refuses by name and keeps going: a smaller later need can
    # still fit; an allowance refuses once it is spent.
    budgeted = _allocate(
        (needs_a, needs_b),
        window_allowance=MAXIMUM_ISSUED_MATTER_WINDOWS,
        window_bytes=window_bytes,
        byte_budget=6000,
    )
    assert budgeted.source_bytes <= 6000
    assert {why for _need, why in budgeted.refused} <= {"byte budget", "opening pending"}
    assert "byte budget" in {why for _need, why in budgeted.refused}
    assert all(need.part > 1 for need, why in budgeted.refused if why == "opening pending"), (
        "only an extension waits for its opening"
    )
    assert len(budgeted.windows) < len(allocation.windows)
    capped = _allocate((needs_a, needs_b), window_allowance=3, window_bytes=window_bytes)
    assert len(capped.windows) == 3 and {why for _n, why in capped.refused} == {"window allowance"}
    assert len(capped.refused) + sum(len(w.needs) for w in capped.windows) == sum(
        1 for needs in prepared for n in needs.needs if n.priority > 0
    )
