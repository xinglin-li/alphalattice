"""Inventory dated operations disclosures in filings.

Operations-disclosure units: a provisional inventory of what a filing
states about restructuring, impairment, closure and other operating charges
-- in the notes that carry them and, as dated statements, in the business,
risk-factor and MD&A items.

The eight-topic completeness assignment (section Q of the initiative
record, 2026-09-20) traced the operations topic's development losses to
their boundary: KO's other operating charges sat in "NOTE 18: SIGNIFICANT
OPERATING AND NONOPERATING ITEMS" (10-K) and "NOTE 12" (10-Q), notes no
route named; DRI's restaurant closures and impairments sat in
"IMPAIRMENTS AND DISPOSAL OF ASSETS" notes and in an MD&A paragraph dated
February 3, 2026, both returned by the residual search below its
allowance; DG's 45 pOpshelf closures sat in a dated sentence of its risk
factors, outside every routed range. This owner gives those statements the
litigation inventory's shape (regions from the filing's own headings,
units the source's own paragraphs open, needs, allocation, attribution)
with the family's own signposts: a note whose heading names the family,
and a sentence that opens with a date and states an operations action. It
parses no amount into a field, sums no charges, decides no materiality,
and infers no program from a heading: the unit is the source's paragraph,
delivered whole with its qualifications.

Consumer: `packet.select_matter_evidence` with `families` naming
`OPERATIONS` (the integrated method's `MATTER_FAMILIES` name it); the
routing serves its units in the operations topic's lane. No model, no
query; `retrieval/session.py` issues and proves the ranges.
"""

from __future__ import annotations

import re

from ..documents.structure import DocumentStructure
from .events import _EVENT_NOTE
from .matters import (
    LitigationRegion,
    MatterInventory,
    Paragraph,
    ProvisionalMatter,
    _paragraphs,
    _plain,
    bounded_region,
    first_sentence,
    signposted_sentences,
    source_blocks,
)

OPERATIONS_INVENTORY_RULES_ID = "alternative-evidence.operations-disclosure-inventory.v1"
"""v1 (2026-09-20): in a periodic report, a note whose heading names
restructuring, impairment, other operating charges, significant operating
or nonoperating items, exit or disposal activities, assets held for sale,
write-downs, closures or a productivity program is a region, bounded by
the next note, item or part heading -- unless the corporate-event family
names the same note (an acquisition or divestiture note stays that
family's). Inside a region, every prose paragraph that states an
operations charge or action (`OPERATIONS_CUE`) is one unit: the paragraph
whole, titled by its first sentence; a sub-heading line names the group of
the units under it; a table placeholder is unassigned as a table not
carried; a paragraph stating no such charge or action is a region-level
statement. In the business, risk-factor and MD&A items (10-K Items 1, 1A
and 7; 10-Q Part I Item 2 and Part II Item 1A), a paragraph holding a
sentence that opens with a date and states an operations action -- a
closure of stores, plants or facilities, an impairment, restructuring or
other operating charge recorded or incurred, a restructuring, productivity
or workforce program announced or completed, production or a business
ceased, idled or exited, a workforce reduced -- is a region of its own
and one unit: the paragraph whole, titled by the dated sentence. A
current report has none: its items are the corporate-event family's, and
Items 2.05 and 2.06 already serve the operations topic. Nothing is parsed
into a field; identity stays local to a document and provisional."""

OPERATIONS_FAMILY = "OPERATIONS"

_OPERATIONS_NOTE = re.compile(
    r"\b(?:restructuring|impairments?|other operating charges|significant operating|"
    r"nonoperating items|exit (?:and|or) disposal|disposal activit\w*|held for sale|"
    r"asset write-?(?:downs?|offs?)|store closings?|closures?|productivity)\b",
    re.IGNORECASE,
)
"""A note heading that names the family, word by word: a segment
disclosures note is not a closure."""
_CROSS_REFERENCE_LEAD = re.compile(r"^(?:refer to|see)\s+(?:also\s+)?note\b", re.IGNORECASE)
"""A short paragraph that only points at another note is a reference the
cross-reference reader resolves, not a unit."""
_MONTH = (
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
)
_DATE = (
    rf"(?:{_MONTH}\s+\d{{1,2}},?\s+\d{{4}}|{_MONTH}\s+\d{{4}}|"
    r"the\s+(?:first|second|third|fourth)\s+(?:fiscal\s+)?quarter\s+of\s+(?:fiscal\s+)?\d{4}|"
    r"the\s+(?:first|second|third|fourth)\s+quarter\s+of\s+fiscal\s+(?:year\s+)?\d{4}|"
    r"fiscal\s+(?:year\s+)?\d{4}|\d{4})"
)
_CHARGE = (
    r"(?:impairment|restructuring|other\s+operating|exit|severance|closure|store\s+closing|"
    r"write-?downs?|write-?offs?|asset\s+impairment)"
)
_OPERATIONS_VERB = (
    r"(?:(?:recorded|recognized|recognised|incurred|took)\s+[^.]{0,80}?\b" + _CHARGE + r"\b"
    r"|\bimpaired\b"
    r"|(?:permanently\s+|temporarily\s+)?clos(?:ed|e|ing)\s+(?:\d[\d,]*|all|certain|"
    r"approximately|about|its|our|the|these|those|several|a\s+number\s+of|substantially)\b"
    r"|(?:announced|approved|initiated|launched|commenced|began|implemented|completed)\s+"
    r"[^.]{0,60}?\b(?:restructuring|productivity|transformation|realignment|reorganization|"
    r"cost[-\s](?:savings?|reduction)|workforce|reduction\s+in\s+force|closures?|exit)\b"
    r"|(?:ceased|discontinued|suspended|idled|halted|wound\s+down|exited)\s+[^.]{0,40}?"
    r"\b(?:production|operations|manufacturing|business|product\s+line|brand|facility|"
    r"facilities|plants?|stores?|restaurants?)\b"
    r"|reduced\s+(?:our|its|the)?\s*(?:global\s+)?(?:workforce|headcount)\b"
    r"|eliminated\s+(?:approximately\s+)?\d[\d,]*\s+(?:positions|jobs|roles)\b)"
)
_RUN_IN_TITLE = r"(?:(?:[A-Z][A-Za-z0-9&'\u2019.,-]*|of|and|the|for|&)\s+){1,7}"
_DATED_ACTION = re.compile(
    rf"^(?:{_RUN_IN_TITLE})?(?:On|In|During|Effective|As\s+of|Beginning\s+in|Since)\s+"
    rf"(?i:{_DATE}),?\s+[^.]{{0,160}}?(?i:{_OPERATIONS_VERB})",
    re.DOTALL,
)
"""A sentence that opens with a date and states an operations action: the
source's own signpost for one operations statement. Case-sensitive at the
opening so a run-in title is told from prose; the date and the action are
matched in any case."""
_OPERATIONS_CUE = re.compile(
    r"\b(?:impairments?|impaired|restructuring|other\s+operating\s+charges?|severance|"
    r"closures?|closed|closing|exit(?:ed)?|disposals?|write-?downs?|write-?offs?|"
    r"held\s+for\s+sale|productivity|reinvestment|charges?|idled|ceased|discontinued)\b",
    re.IGNORECASE,
)
"""The words an operations charge or action is described with; a region
paragraph that carries none is not read first, and inside an operations
note a paragraph that carries one is a unit."""
_REFERS_TO_OPERATIONS = re.compile(
    r"\b(?:these|those|such|the)\s+(?:charges|actions|initiatives|programs|activities|"
    r"closures|impairments|restructuring\s+(?:charges|activities|actions))\s*"
    r"(?:described|discussed|referred\s+to)?\s*(?:above|below)?\b"
    r"|\bthe\s+(?:restructuring|productivity|transformation)\s+program\s+described\s+above\b",
    re.IGNORECASE,
)
"""A region-level statement that says it applies to the region's units as a
body ('these charges', 'the restructuring activities described above')."""
_ANY_DATE = re.compile(_DATE)
_NARRATIVE_ITEMS = {"10-K": frozenset({"1", "1A", "7"}), "10-Q": frozenset({"2", "1A"})}
"""The items whose dated operations statements are read, by form: the
business, risk-factor and MD&A items."""


def operations_regions(text: str, structure: DocumentStructure) -> tuple[LitigationRegion, ...]:
    """The regions the operations inventory reads.

    The regions the operations inventory reads: in a periodic report,
    every note whose heading names the family and that the corporate-event
    family does not claim, bounded by the next note, item or part heading,
    and every paragraph of the narrative items that holds a dated
    operations statement (`dated_statement_regions`). A current report has
    none.
    """
    form = structure.family_form.upper()
    if form not in _NARRATIVE_ITEMS:
        return ()
    regions: list[LitigationRegion] = []
    headings = structure.headings
    for index, heading in enumerate(headings):
        if heading.character_start < structure.cover_end or heading.kind != "NOTE":
            continue
        if (
            _OPERATIONS_NOTE.search(heading.text) is None
            or _EVENT_NOTE.search(heading.text) is not None
        ):
            continue
        region = bounded_region(
            text,
            headings,
            index,
            closers={"PART", "ITEM", "NOTE"},
            kind="NOTE",
            family=OPERATIONS_FAMILY,
        )
        if region is not None:
            regions.append(region)
    note_ranges = [(r.body_start, r.body_end) for r in regions]
    for region in dated_statement_regions(text, structure):
        if any(s <= region.body_start < e for s, e in note_ranges):
            continue
        regions.append(region)
    return tuple(sorted(regions, key=lambda r: r.body_start))


def _dated_sentence(paragraph: Paragraph) -> tuple[int, int] | None:
    """Find an opening dated operations action in a paragraph.

    The range of the first sentence of the paragraph that opens with a
    date and states an operations action, or None.
    """
    found = signposted_sentences(paragraph, _DATED_ACTION)
    return found[0] if found else None


def dated_statement_regions(
    text: str, structure: DocumentStructure
) -> tuple[LitigationRegion, ...]:
    """Find dated operations statements in narrative items.

    Every paragraph of the form's narrative items that holds a dated
    operations statement, as a region of its own headed by the item it
    sits in. The region is the paragraph, so the unit it opens leaves no
    unassigned remainder and no region-level statement.
    """
    form = structure.family_form.upper()
    items = _NARRATIVE_ITEMS.get(form, frozenset())
    regions: list[LitigationRegion] = []
    for item in structure.item_regions():
        if item.item.upper() not in items or item.body_end <= item.body_start:
            continue
        heading = _plain(text[item.heading_start : item.heading_end])[:160] or f"Item {item.item}"
        for paragraph in _paragraphs(text, item.body_start, item.body_end, join_cut=True):
            if _dated_sentence(paragraph) is None:
                continue
            regions.append(
                LitigationRegion(
                    kind="ITEM",
                    heading=heading,
                    heading_start=item.heading_start,
                    body_start=paragraph.character_start,
                    body_end=paragraph.character_end,
                    end_basis="the paragraph's own end",
                    family=OPERATIONS_FAMILY,
                )
            )
    return tuple(regions)


def _states_operations(plain: str) -> bool:
    return _OPERATIONS_CUE.search(plain) is not None


def operations_inventory(
    text: str, structure: DocumentStructure, *, document_id: str = "DOC"
) -> MatterInventory:
    """Inventory provisional operations units and unassigned text.

    The provisional operations units of one canonical text, with the
    unassigned remainder of the operations notes. Deterministic; no model.
    Unit handles are `M-<document>-O-<n>`: the same namespace the packet's
    unit records carry, in the family's own series.
    """
    regions = operations_regions(text, structure)
    units: list[ProvisionalMatter] = []
    unassigned: list[tuple[int, int, str]] = []
    counter = 0

    def emit(
        region: LitigationRegion, title: str, basis: str, span: tuple[int, int], group: str = ""
    ) -> None:
        nonlocal counter
        counter += 1
        body = text[span[0] : span[1]]
        first_date = _ANY_DATE.search(_plain(body))
        units.append(
            ProvisionalMatter(
                handle=f"M-{document_id}-O-{counter:03d}",
                title=title[:120],
                basis=basis,
                region_heading=region.heading,
                group=group or region.heading,
                ranges=(span,),
                aliases=() if first_date is None else (first_date.group(0)[:40],),
                source_bytes=len(body.encode("utf-8")),
                family=OPERATIONS_FAMILY,
            )
        )

    for region in regions:
        if region.kind == "ITEM":
            # A dated statement's paragraph is the region and the unit.
            paragraph = Paragraph(
                character_start=region.body_start,
                character_end=region.body_end,
                text=text[region.body_start : region.body_end],
            )
            dated = _dated_sentence(paragraph)
            title = _plain(text[dated[0] : dated[1]]) if dated else _plain(paragraph.text)
            emit(
                region,
                title,
                "dated operations statement",
                (region.body_start, region.body_end),
            )
            continue
        group = ""
        for paragraph, kind in source_blocks(text, region, structure, signpost=_states_operations):
            plain = _plain(paragraph.text)
            span = (paragraph.character_start, paragraph.character_end)
            if kind == "table":
                unassigned.append((*span, "table not carried"))
                continue
            if kind == "rule":
                unassigned.append((*span, "footnote rule"))
                continue
            if kind == "heading":
                group = plain
                unassigned.append((*span, "sub-heading"))
                continue
            if len(plain) <= 200 and _CROSS_REFERENCE_LEAD.match(plain) is not None:
                unassigned.append((*span, "cross-reference"))
                continue
            if _OPERATIONS_CUE.search(plain) is None:
                unassigned.append((*span, "no operations charge or action stated"))
                continue
            dated = _dated_sentence(paragraph)
            emit(
                region,
                _plain(text[dated[0] : dated[1]]) if dated else first_sentence(plain),
                "dated operations statement" if dated else "operations paragraph",
                span,
                group,
            )
    return MatterInventory(
        rules_id=OPERATIONS_INVENTORY_RULES_ID,
        regions=regions,
        matters=tuple(units),
        unassigned=tuple(sorted(unassigned)),
    )


OPERATIONS_CUE = _OPERATIONS_CUE
REFERS_TO_OPERATIONS = _REFERS_TO_OPERATIONS

__all__ = [
    "OPERATIONS_CUE",
    "OPERATIONS_FAMILY",
    "OPERATIONS_INVENTORY_RULES_ID",
    "REFERS_TO_OPERATIONS",
    "dated_statement_regions",
    "operations_inventory",
    "operations_regions",
]
