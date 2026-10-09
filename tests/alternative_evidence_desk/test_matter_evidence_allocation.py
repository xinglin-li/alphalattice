"""The shared allocator's sealed service order: one lane per issuer and
lane name, served in turn; inside a lane an opening and an extension
alternate while both exist; the order is a function of the needs alone, so
a continuation resumes it. The lane function is the caller's -- the
integrated selection passes the routing owner's topics -- and these
invariants are stated with one lane per family (the retired candidate
allocation's lanes, first-release integration T5) on controlled fixtures
and stressed: a dense financing note beside sparse legal evidence, the same
source split into more units, duplicate and overlapping needs, one long
disclosure beside many small openings, an absent family, and an allowance
or a byte budget that admits little. Nothing here claims output invariance
where the source obligations differ: what is invariant is the litigation
lane's service under a non-exhausting other lane, and the sealed order
itself."""

from __future__ import annotations

from dataclasses import replace

from alphalattice.evidence.alternative_evidence.analysis.financing import (
    FINANCING_CUE,
    REFERS_TO_FINANCING,
    financing_inventory,
)
from alphalattice.evidence.alternative_evidence.analysis.matters import (
    MATTER_SEGMENTATION_RULES_ID,
    TOPIC_LANES_ALLOCATION_ID,
    Allocation,
    DocumentNeeds,
    EvidenceNeed,
    allocate_windows,
    document_needs,
    matter_inventory,
    service_order,
)
from alphalattice.evidence.alternative_evidence.documents.structure import DocumentStructure
from tests.alternative_evidence_desk.event_support import ANNUAL_COVER
from tests.alternative_evidence_desk.litigation_support import BOILERPLATE, _filing, _matter
from tests.alternative_evidence_desk.matter_selection_support import _long_matter

WINDOW_BYTES = 500
"""Small enough that two instrument paragraphs, or two matters, never share
one window, so a unit is one window and the lanes' turns are visible."""
FAMILY_LANES = ("LITIGATION", "CORPORATE_EVENT", "FINANCING", "OPERATIONS")


def _by_family(need: EvidenceNeed) -> str:
    return need.family


def _order(documents: tuple[DocumentNeeds, ...]) -> tuple[EvidenceNeed, ...]:
    return service_order(documents, lane_of=_by_family, lanes_in_order=FAMILY_LANES)


def _allocate(documents: tuple[DocumentNeeds, ...], **options: int) -> Allocation:
    return allocate_windows(documents, lane_of=_by_family, lanes_in_order=FAMILY_LANES, **options)


INSTRUMENTS = (
    "commercial paper program",
    "revolving credit facility",
    "term loan facility",
    "senior notes due 2030",
    "unsecured notes due 2032",
    "convertible notes due 2028",
    "securitization financings",
    "backup lines of credit",
    "subordinated debentures due 2035",
    "guaranteed senior notes",
)


def _instrument_paragraph(index: int, instruments: tuple[str, ...]) -> str:
    named = " and the ".join(instruments)
    return (
        f"In month {index % 12 + 1} of 2025, the Company amended the {named} under agreement "
        f"number {index}, extending the maturity by one year and keeping the customary "
        "covenants, which the Company complied with at year end; borrowings bear interest at "
        "the rates the agreement states and the lenders may terminate it on notice."
    )


def _debt_note(count: int, *, per_paragraph: int = 1) -> str:
    """A 10-K whose debt note holds `count` instrument paragraphs, each
    naming `per_paragraph` distinct instruments."""

    paragraphs = []
    for index in range(count):
        names = tuple(
            INSTRUMENTS[(index * per_paragraph + k) % len(INSTRUMENTS)] + f" number {index}-{k}"
            for k in range(per_paragraph)
        )
        paragraphs.append(_instrument_paragraph(index, names))
    lines = [
        *ANNUAL_COVER,
        "ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA",
        "NOTE 11: DEBT AND BORROWING ARRANGEMENTS",
        *paragraphs,
        "NOTE 12: SEGMENT INFORMATION",
        "The Company manages its business as one segment.",
        "PART IV",
        "ITEM 15. EXHIBITS AND FINANCIAL STATEMENT SCHEDULES",
        "See the exhibit index.",
        "SIGNATURES",
    ]
    return "\n\n".join(lines) + "\n"


def _litigation_needs(
    issuer: str, key: str, *, matters: int, long_first: bool = False
) -> DocumentNeeds:
    lines = [BOILERPLATE]
    for number in range(1, matters + 1):
        lines.append(_long_matter(number) if long_first and number == 1 else _matter(number))
    text = _filing(note_lines=lines)
    structure = DocumentStructure(text, document_type="10-Q")
    inventory = matter_inventory(
        text, structure, document_id=key.upper(), rules=MATTER_SEGMENTATION_RULES_ID
    )
    return document_needs(
        text, structure, inventory, document_key=key, issuer=issuer, window_bytes=WINDOW_BYTES
    )


def _financing_needs(issuer: str, key: str, *, units: int, per_paragraph: int = 1) -> DocumentNeeds:
    text = _debt_note(units, per_paragraph=per_paragraph)
    structure = DocumentStructure(text, document_type="10-K")
    inventory = financing_inventory(text, structure, document_id=key.upper())
    assert len(inventory.matters) == units, [m.title for m in inventory.matters]
    return document_needs(
        text,
        structure,
        inventory,
        document_key=key,
        issuer=issuer,
        window_bytes=WINDOW_BYTES,
        cue=FINANCING_CUE,
        refers=REFERS_TO_FINANCING,
        family="FINANCING",
    )


def _lane(allocation: Allocation, family: str) -> list[tuple[str, int, str]]:
    return [(n.disclosure, n.part, n.kind) for n in allocation.admitted if n.family == family]


def test_a_family_s_unit_count_grants_it_no_service_over_another_lane() -> None:
    """A family's unit count grants it no service over another lane."""

    litigation = _litigation_needs("AAPL", "d-lit", matters=14, long_first=True)
    allowance = 16
    served: dict[int, list[tuple[str, int, str]]] = {}
    windows: dict[int, int] = {}
    for units in (3, 10, 30):
        financing = _financing_needs("AAPL", "d-debt", units=units)
        allocation = _allocate(
            (litigation, financing), window_allowance=allowance, window_bytes=WINDOW_BYTES
        )
        assert allocation.rules_id == TOPIC_LANES_ALLOCATION_ID
        served[units] = _lane(allocation, "LITIGATION")
        windows[units] = len(allocation.windows)
        assert len(allocation.windows) == allowance, "the allowance is spent while lanes have work"
    assert served[10] == served[30], "density grants no cross-family priority"
    assert served[3][: len(served[30])] == served[30]
    assert len(served[3]) > len(served[30]), "an exhausted lane relinquishes its turns"
    financing_30 = _lane(
        _allocate(
            (litigation, _financing_needs("AAPL", "d-debt", units=30)),
            window_allowance=allowance,
            window_bytes=WINDOW_BYTES,
        ),
        "FINANCING",
    )
    assert len(financing_30) == allowance // 2, "an equal share, not one per unit"


def test_splitting_the_same_source_into_more_units_changes_no_other_lane_s_service() -> None:
    """Splitting the same source into more units changes no other lane's service."""

    litigation = _litigation_needs("AAPL", "d-lit", matters=14, long_first=True)
    coarse = _financing_needs("AAPL", "d-debt", units=8, per_paragraph=2)
    fine = _financing_needs("AAPL", "d-debt", units=16)
    allowance = 16
    with_coarse = _allocate(
        (litigation, coarse), window_allowance=allowance, window_bytes=WINDOW_BYTES
    )
    with_fine = _allocate((litigation, fine), window_allowance=allowance, window_bytes=WINDOW_BYTES)
    assert _lane(with_coarse, "LITIGATION") == _lane(with_fine, "LITIGATION")
    assert len(_lane(with_coarse, "FINANCING")) == len(_lane(with_fine, "FINANCING")) == 8


def test_duplicate_and_overlapping_needs_open_no_second_window() -> None:
    """Duplicate and overlapping needs open no second window."""

    financing = _financing_needs("AAPL", "d-debt", units=4)
    first = next(n for n in financing.needs if n.kind == "LEAD")
    duplicate = replace(first, detail="the same need stated twice")
    assert first.window is not None
    inner = replace(
        first,
        character_start=first.window[0] + 20,
        character_end=first.window[1] - 20,
        source_bytes=first.source_bytes - 40,
        detail="a range inside the first need's",
    )
    stressed = replace(financing, needs=(*financing.needs, duplicate, inner))
    plain = _allocate((financing,), window_allowance=8, window_bytes=WINDOW_BYTES)
    with_duplicates = _allocate((stressed,), window_allowance=8, window_bytes=WINDOW_BYTES)
    assert len(with_duplicates.windows) == len(plain.windows)
    assert with_duplicates.source_bytes == plain.source_bytes
    holding = next(w for w in with_duplicates.windows if first in w.needs)
    assert duplicate in holding.needs and inner in holding.needs
    assert not with_duplicates.refused


def test_one_long_disclosure_cannot_monopolise_its_lane() -> None:
    """One long disclosure cannot monopolise its lane."""

    litigation = _litigation_needs("AAPL", "d-lit", matters=11, long_first=True)
    lane = [
        (n.disclosure, n.part, "OPEN" if n.part == 1 and n.priority > 1 else "COMPLETE")
        for n in _order((litigation,))
        if n.family == "LITIGATION"
    ]
    long_matter = next(disclosure for disclosure, part, _turn in lane if part == 2)
    long_parts = [part for disclosure, part, _turn in lane if disclosure == long_matter]
    assert long_parts == sorted(long_parts) and long_parts[0] == 1 and len(long_parts) >= 3
    first_twelve = lane[:12]
    turns = [turn for _d, _p, turn in first_twelve]
    assert turns[0] == "OPEN" and turns[1] == "COMPLETE"
    assert all(turns[i] != turns[i + 1] for i in range(len(turns) - 1)), (
        "openings and completions alternate while both exist"
    )
    assert sum(1 for d, _p, _t in first_twelve if d == long_matter) <= 12 // 2 + 1, (
        "its opening and at most every other turn after it"
    )


def test_an_absent_family_leaves_no_turn_unused_and_the_order_is_sealed() -> None:
    """An absent family leaves no turn unused and the order is sealed."""

    a = _litigation_needs("AAPL", "d-a", matters=12, long_first=True)
    b = _litigation_needs("MSFT", "d-b", matters=12)
    order = _order((a, b))
    assert order == _order((a, b))
    assert {n.family for n in order} == {"LITIGATION"}
    assert [n.document_key for n in order[:4]] == ["d-a", "d-b", "d-a", "d-b"], "issuers in turn"
    full = _allocate((a, b), window_allowance=8, window_bytes=WINDOW_BYTES)
    assert len(full.windows) == 8 and full.admitted[0] is order[0]
    first = _allocate((a, b), window_allowance=4, window_bytes=WINDOW_BYTES)
    served_first = {id(n) for w in first.windows for n in w.needs}
    second = _allocate(
        (a, b),
        window_allowance=4,
        window_bytes=WINDOW_BYTES,
        delivered=lambda need: id(need) in served_first,
    )
    resumed_at = next(n for n in order if id(n) not in served_first)
    assert second.admitted[0] is resumed_at, "the sealed order resumes where the chain stopped"
    assert not any(id(n) in served_first for n in second.admitted)
    # The windows the two sessions open are the windows the one session
    # opens, in the same order: the chain is the plan, cut by the allowance.
    assert [w.needs[0] for w in (*first.windows, *second.windows)] == [
        w.needs[0] for w in full.windows
    ]


def test_an_allowance_or_a_budget_that_admits_little_refuses_the_rest_by_name() -> None:
    """An allowance or a budget that admits little refuses the rest by name."""

    litigation = _litigation_needs("AAPL", "d-lit", matters=6, long_first=True)
    financing = _financing_needs("AAPL", "d-debt", units=4)
    stated = sum(1 for d in (litigation, financing) for n in d.needs if n.priority > 0)
    capped = _allocate((litigation, financing), window_allowance=2, window_bytes=WINDOW_BYTES)
    assert len(capped.windows) == 2
    assert {why for _n, why in capped.refused} == {"window allowance"}
    assert len(capped.refused) + sum(len(w.needs) for w in capped.windows) == stated
    starved = _allocate(
        (litigation, financing), window_allowance=8, window_bytes=WINDOW_BYTES, byte_budget=10
    )
    assert not starved.windows and not starved.admitted
    assert {why for n, why in starved.refused if n.part == 1} == {"byte budget"}
    assert {why for n, why in starved.refused if n.part > 1} == {"opening pending"}
    assert len(starved.refused) == stated


def _synthetic_need(
    document_key: str,
    *,
    disclosure: str,
    kind: str,
    priority: int,
    part: int,
    part_count: int,
    start: int,
    end: int,
    handle: str | None = None,
) -> EvidenceNeed:
    return EvidenceNeed(
        document_key=document_key,
        matter_handle=handle,
        kind=kind,
        priority=priority,
        character_start=start,
        character_end=end,
        source_bytes=end - start,
        part=part,
        part_count=part_count,
        region_heading="NOTE 7. COMMITMENTS AND CONTINGENCIES",
        detail=f"{kind} {disclosure} part {part}",
        family="LITIGATION",
        disclosure=disclosure,
    )


def test_progress_keys_are_qualified_by_document() -> None:
    """Progress keys are qualified by document."""

    a_base = _litigation_needs("AAA", "doc-a", matters=1)
    b_base = _litigation_needs("BBB", "doc-b", matters=2)
    a = replace(
        a_base,
        needs=(
            _synthetic_need(
                "doc-a",
                disclosure="M-DOC-A-001",
                kind="LEAD",
                priority=2,
                part=1,
                part_count=1,
                start=300,
                end=520,
                handle="M-DOC-A-001",
            ),
            _synthetic_need(
                "doc-a",
                disclosure="S:600",
                kind="REGION_STATEMENT",
                priority=4,
                part=1,
                part_count=2,
                start=600,
                end=900,
            ),
            _synthetic_need(
                "doc-a",
                disclosure="S:600",
                kind="CONTINUATION",
                priority=3,
                part=2,
                part_count=2,
                start=900,
                end=1200,
            ),
        ),
    )
    b = replace(
        b_base,
        needs=(
            _synthetic_need(
                "doc-b",
                disclosure="M-DOC-B-001",
                kind="LEAD",
                priority=2,
                part=1,
                part_count=1,
                start=300,
                end=520,
                handle="M-DOC-B-001",
            ),
            _synthetic_need(
                "doc-b",
                disclosure="M-DOC-B-002",
                kind="LEAD",
                priority=2,
                part=1,
                part_count=1,
                start=1300,
                end=1500,
                handle="M-DOC-B-002",
            ),
            _synthetic_need(
                "doc-b",
                disclosure="S:600",
                kind="REGION_STATEMENT",
                priority=4,
                part=1,
                part_count=2,
                start=600,
                end=900,
            ),
            _synthetic_need(
                "doc-b",
                disclosure="S:600",
                kind="CONTINUATION",
                priority=3,
                part=2,
                part_count=2,
                start=900,
                end=1200,
            ),
        ),
    )
    opened: set[tuple[str, str]] = set()
    for need in _order((a, b)):
        key = (need.document_key, need.disclosure)
        if need.part == 1:
            opened.add(key)
        else:
            assert key in opened, f"{need.document_key} {need.disclosure} part {need.part}"
    allocation = _allocate((a, b), window_allowance=6, window_bytes=WINDOW_BYTES)
    admitted = [(n.document_key, n.disclosure, n.part) for n in allocation.admitted]
    assert ("doc-b", "S:600", 1) in admitted and ("doc-b", "S:600", 2) not in admitted
    assert [(n.document_key, n.disclosure, n.part, why) for n, why in allocation.refused] == [
        ("doc-b", "S:600", 2, "window allowance")
    ]


def test_every_need_names_its_family_and_its_disclosure() -> None:
    """requirement (5E): the lane and the disclosure a need belongs to are
    the need's own, stated by the inventory that produced it."""

    litigation = _litigation_needs("AAPL", "d-lit", matters=3, long_first=True)
    financing = _financing_needs("AAPL", "d-debt", units=2)
    for need in (*litigation.needs, *financing.needs):
        assert isinstance(need, EvidenceNeed)
        if need.priority > 0:
            assert need.disclosure, need
    assert {n.family for n in litigation.needs} == {"LITIGATION"}
    assert {n.family for n in financing.needs} == {"FINANCING"}
    parts = [n for n in litigation.needs if n.kind == "CONTINUATION"]
    assert parts and all(n.disclosure == n.matter_handle for n in parts)
