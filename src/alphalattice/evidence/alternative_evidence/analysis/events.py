"""Inventory dated corporate events in filings.

Corporate-event units: a provisional inventory of the dated events a
filing states in its acquisitions, divestitures and subsequent-events
notes and in the items of a current report (8-K).

The K1 audit (dispatch KEY_ISSUE_AND_MATTER_DISCOVERY_COVERAGE, 2026-09-19)
found that the events of these regions -- a sale, an acquisition, a
definitive agreement, a dividend declared after the period, an 8-K item's
event -- enter the evidence pipeline only as generic passage hits that the
program's allocation does not select: seven of the eight corporate-action
cases of the 45-case set and all eleven development units of the topic
were lost before delivery. This owner gives those units the litigation
inventory's shape -- region discovery from the filing's own headings,
source signposts that open provisional units, region-level statements,
needs, allocation, attribution -- with its own signposts and cues. It
does not borrow the legal vocabulary: an event is a stated action with a
date and a subject, never a proceeding; nothing here decides whether a
transaction closed, what it is worth, or whether it matters.

Consumer: `packet.select_matter_evidence` with `families` naming
`CORPORATE_EVENT`, which merges this inventory's needs with the
litigation inventory's under the one matter allowance; the development
comparison drivers. No model, no query; `retrieval/session.py` issues and
proves the ranges.
"""

from __future__ import annotations

import re
from itertools import pairwise

from ..documents.structure import DocumentStructure
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
)

EVENT_INVENTORY_RULES_ID = "alternative-evidence.corporate-event-inventory.v2"
"""v2 (2026-09-23, Q3): a late-filing notice (NT 10-K, NT 10-Q; Form 12b-25)
is read beside the current reports: its narrative (Part III, why the report
is late) and its other information (Part IV, the change in results it
expects) are each one region and one unit, bounded by the next part or the
signature, like an 8-K item.

v1 (2026-09-19): a note whose heading names acquisitions, divestitures,
dispositions, business combinations, mergers or subsequent events is a
region; so is a note whose lead paragraph dates an event and accounts for
it as an acquisition (goodwill, consideration, purchase price -- the note
named after its counterparty, 'Note 2 - Groq'); so is every item of a
current report other than the exhibits item (9.01). Inside a note, a
sentence that opens with a date and states an event action ('In March
2025, the Company sold', 'On October 11, 2024, we acquired', 'In October
2025, the Company entered into a definitive agreement'), a run-in title
before the date allowed ('Ally Lending On March 1, 2024, we acquired',
the bold sub-heading canonical text keeps inline), opens a unit from that
sentence to the next such sentence or the paragraph's end; the paragraphs
that follow without a dated action continue it -- the purchase-price
allocation, the goodwill, the costs of that event -- until a region-level
paragraph closes it: one that names the events as a body ('the
transactions described above', 'subsequent events through the date'), or
opens by naming the category, the pro forma presentation or the evaluation
of subsequent events. An 8-K item is one unit: the item's whole body, its
date the first date the body states. Identity stays local to a document
and provisional: the same event restated in another filing is another
observation."""

DISCLOSURE_UNIT_INVENTORY_RULES_ID = "alternative-evidence.disclosure-unit-inventory.v1"
"""The combined inventory a record carries when more than one family was
read: the litigation inventory (rules v3) beside the corporate-event
inventory (v2), each unit under its own family."""

CORPORATE_EVENT_FAMILY = "CORPORATE_EVENT"

_EVENT_NOTE = re.compile(
    r"acquisition|divestiture|disposition|business combination|merger|subsequent event",
    re.IGNORECASE,
)
_EXHIBITS_ITEM = re.compile(r"^item\s+9\.01\b", re.IGNORECASE)
LATE_FILING_FORMS = frozenset({"NT 10-K", "NT 10-Q"})
"""The late-filing notices (amendments by their base form) the event family reads."""
_LATE_FILING_PART = re.compile(r"^\W*part\s+(?:iii|iv)\b", re.IGNORECASE)
_MONTH = (
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
)
_DATE = (
    rf"(?:{_MONTH}\s+\d{{1,2}},?\s+\d{{4}}|{_MONTH}\s+\d{{4}}|{_MONTH}\s+and\s+{_MONTH}\s+\d{{4}}|"
    r"the\s+(?:first|second|third|fourth)\s+quarter\s+of\s+(?:fiscal\s+)?\d{4}|"
    r"fiscal\s+\d{4}|\d{4})"
)
_EVENT_VERB = (
    r"(?:acquired|completed(?:\s+the|\s+its|\s+our)?\s+(?:acquisition|sale|divestiture|disposition|purchase)|"
    r"sold|divested|disposed\s+of|refranchised|entered\s+into\s+(?:a|an)\s+[^.]{0,40}?"
    r"(?:agreement|arrangement)|agreed\s+to\s+(?:sell|acquire|purchase|divest)|"
    r"announced|declared|approved|authorized|formed|contributed|received\s+(?:net\s+)?cash\s+proceeds|"
    r"classified\s+as\s+held\s+for\s+sale|closed\s+(?:on\s+)?the\s+(?:sale|acquisition|transaction)|"
    r"issued|redeemed|repaid|paid|recognized\s+a\s+(?:gain|loss)|recorded\s+(?:a|an)\s+[^.]{0,30}?"
    r"(?:charge|gain|loss|impairment)|terminated|amended|launched|spun\s+off|distributed)"
)
_RUN_IN_TITLE = r"(?:(?:[A-Z][A-Za-z0-9&'\u2019.,-]*|of|and|the|for|&)\s+){1,7}"
"""A bold sub-heading the canonical text keeps inline before the sentence
it introduces ("Ally Lending On March 1, 2024, we acquired"): title-case
words, at most seven."""
_DATED_ACTION = re.compile(
    rf"^(?:{_RUN_IN_TITLE})?(?:On|In|During|Effective|As\s+of)\s+(?i:{_DATE}),?\s+"
    rf"[^.]{{0,160}}?\b(?i:{_EVENT_VERB})\b",
    re.DOTALL,
)
"""A sentence that opens with a date and states an event action: the
source's own signpost for one corporate event. Case-sensitive at the
opening so a run-in title is told from prose; the date and the verb are
matched in any case."""
_EVENT_CUE = re.compile(
    r"\b(?:acquisitions?|acquired|divestitures?|divested|dispositions?|sale of|sold|mergers?|"
    r"definitive agreement|joint venture|held for sale|purchase price|consideration|"
    r"dividends?|declared|repurchases?|spin-off|subsequent events?|closing)\b",
    re.IGNORECASE,
)
"""The words a corporate event is described with; a region paragraph that
carries none is not read first."""
_REFERS_TO_EVENTS = re.compile(
    r"\b(?:these|those|such|the)\s+(?:transactions|acquisitions|divestitures|dispositions|"
    r"business\s+combinations|events)\s+(?:described|discussed|referred\s+to)?\s*(?:above|below)?\b"
    r"|\bthe\s+(?:acquisition|divestiture|sale|transaction)s?\s+(?:described|discussed)\s+(?:above|below)\b"
    r"|\bsubsequent\s+events\s+through\b|\bevents\s+subsequent\s+to\b",
    re.IGNORECASE,
)
"""A region-level statement that says it applies to the region's events as
a body ('the transactions described above', 'subsequent events through the
date the financial statements were issued')."""
_EVENT_GENERIC = re.compile(
    r"^(?:(?:acquisitions|divestitures|dispositions|business\s+combinations|subsequent\s+events)\b|"
    r"(?:the\s+following\s+|the\s+)?(?:unaudited\s+)?pro\s+forma|"
    r"we\s+(?:have\s+)?evaluated\s+(?:subsequent\s+)?events|the\s+company\s+(?:has\s+)?evaluated|"
    r"subsequent\s+events\s+(?:have\s+been|were)\s+evaluated|"
    r"the\s+results\s+of\s+operations\s+of\s+(?:the\s+)?acquired)",
    re.IGNORECASE,
)
"""Paragraph openings that speak of the region's events as a body -- the
category, the pro forma presentation, the evaluation of subsequent
events -- rather than continue the one event that is open: region-level,
never a unit's own text. The accounting of the open event (its
purchase-price allocation, its goodwill, its costs) continues it."""
_ACQUISITION_ACCOUNTING = re.compile(
    r"\b(?:goodwill|purchase\s+price|total\s+consideration|business\s+combination|"
    r"acquisition\s+method|net\s+assets\s+acquired)\b",
    re.IGNORECASE,
)
"""What a note's lead says when the dated event it opens with is an
acquisition, whatever the note is called."""
_ANY_DATE = re.compile(_DATE)


def event_regions(text: str, structure: DocumentStructure) -> tuple[LitigationRegion, ...]:
    """The regions the corporate-event inventory reads.

    The regions the corporate-event inventory reads: in a periodic
    report, every note whose heading names the family or whose lead
    paragraph dates an acquisition, bounded by the next note, item or part
    heading; in a current report, every item other than
    the exhibits item, bounded by the next item. `LitigationRegion` is the
    shared region shape; the family is `CORPORATE_EVENT`.
    """
    regions: list[LitigationRegion] = []
    headings = structure.headings
    current_report = structure.family_form == "8-K"
    late_filing = structure.family_form.upper() in LATE_FILING_FORMS
    for index, heading in enumerate(headings):
        if heading.character_start < structure.cover_end:
            continue
        if late_filing:
            # The notice's narrative and other-information parts; the
            # signature line (a sub-heading) closes the last.
            if heading.kind != "PART" or _LATE_FILING_PART.match(heading.text.strip()) is None:
                continue
            closers, kind = {"PART", "SUB"}, "PART"
        elif current_report:
            if heading.kind != "ITEM" or _EXHIBITS_ITEM.match(heading.text.strip()):
                continue
            closers, kind = {"ITEM", "PART"}, "ITEM"
        else:
            if heading.kind != "NOTE":
                continue
            closers, kind = {"PART", "ITEM", "NOTE"}, "NOTE"
        region = bounded_region(
            text,
            headings,
            index,
            closers=closers,
            kind=kind,
            family=CORPORATE_EVENT_FAMILY,
        )
        if region is None:
            continue
        if (
            kind == "NOTE"
            and _EVENT_NOTE.search(heading.text) is None
            and not _lead_dates_an_acquisition(text, region.body_start, region.body_end)
        ):
            continue
        regions.append(region)
    return tuple(regions)


def _lead_dates_an_acquisition(text: str, start: int, end: int) -> bool:
    """Check whether a note opens with a dated acquisition.

    Whether a note's first paragraph opens with a dated event and accounts
    for it as an acquisition: the note named after its counterparty rather
    than the family. A dated debt issue or a policy paragraph is not one.
    """
    paragraphs = _paragraphs(text, start, end, join_cut=True)
    if not paragraphs:
        return False
    lead = _plain(paragraphs[0].text)
    return (
        _DATED_ACTION.match(lead) is not None and _ACQUISITION_ACCOUNTING.search(lead) is not None
    )


def _dated_actions(paragraph: Paragraph) -> tuple[int, ...]:
    """Find dated event actions within one paragraph.

    Offsets inside the paragraph of every sentence that opens with a date
    and states an event action, the first sentence included.
    """
    return tuple(start for start, _end in signposted_sentences(paragraph, _DATED_ACTION))


def _event_title(text: str) -> str:
    """What the source calls the event: its first sentence, plain, bounded."""
    return first_sentence(_plain(text))[:120]


def _region_level(plain: str) -> bool:
    return _EVENT_GENERIC.match(plain) is not None or (
        _DATED_ACTION.match(plain) is None and _REFERS_TO_EVENTS.search(plain) is not None
    )


def event_inventory(
    text: str, structure: DocumentStructure, *, document_id: str = "DOC"
) -> MatterInventory:
    """Inventory provisional corporate-event units and unassigned text.

    The provisional corporate-event units of one canonical text, with the
    unassigned remainder of the event regions. Deterministic; no model.
    Unit handles are `M-<document>-E-<n>`: the same namespace the packet's
    unit records carry, in the event family's own series.
    """
    regions = event_regions(text, structure)
    units: list[ProvisionalMatter] = []
    unassigned: list[tuple[int, int, str]] = []
    counter = 0

    def emit(region: LitigationRegion, lead: tuple[str, str], start: int, end: int) -> None:
        nonlocal counter
        counter += 1
        body = text[start:end]
        first_date = _ANY_DATE.search(_plain(body))
        units.append(
            ProvisionalMatter(
                handle=f"M-{document_id}-E-{counter:03d}",
                title=lead[0],
                basis=lead[1],
                region_heading=region.heading,
                group=region.heading,
                ranges=((start, end),),
                aliases=() if first_date is None else (first_date.group(0)[:40],),
                source_bytes=len(body.encode("utf-8")),
                family=CORPORATE_EVENT_FAMILY,
            )
        )

    for region in regions:
        if region.kind in {"ITEM", "PART"}:
            # A current report's item, or a late-filing notice's part, is one
            # event: the heading is its signpost and its whole body the statement.
            paragraphs = _paragraphs(text, region.body_start, region.body_end, join_cut=True)
            if paragraphs:
                emit(
                    region,
                    (
                        _plain(region.heading)[:120],
                        "8-K item" if region.kind == "ITEM" else "late-filing notice",
                    ),
                    paragraphs[0].character_start,
                    paragraphs[-1].character_end,
                )
            continue
        current: list[tuple[int, int]] = []
        current_lead: tuple[str, str] | None = None

        def close(region: LitigationRegion = region) -> None:
            nonlocal current, current_lead
            if current and current_lead is not None:
                emit(region, current_lead, current[0][0], current[-1][1])
            current, current_lead = [], None

        for paragraph in _paragraphs(text, region.body_start, region.body_end, join_cut=True):
            plain = _plain(paragraph.text)
            cuts = _dated_actions(paragraph)
            if not cuts:
                if current_lead is not None and not _region_level(plain):
                    current.append((paragraph.character_start, paragraph.character_end))
                    continue
                close()
                unassigned.append(
                    (
                        paragraph.character_start,
                        paragraph.character_end,
                        "region-level statement"
                        if _region_level(plain)
                        else "no dated action opened this text",
                    )
                )
                continue
            # Text before the first dated sentence belongs to the unit that
            # is open, or to nothing.
            if cuts[0] > paragraph.character_start:
                head = (paragraph.character_start, cuts[0])
                if current_lead is not None:
                    current.append(head)
                else:
                    unassigned.append((*head, "no dated action opened this text"))
            close()
            bounds = [*cuts, paragraph.character_end]
            for piece_start, piece_end in pairwise(bounds):
                if piece_end <= piece_start:
                    continue
                if piece_end == paragraph.character_end:
                    current, current_lead = (
                        [(piece_start, piece_end)],
                        (_event_title(text[piece_start:piece_end]), "dated event"),
                    )
                else:
                    emit(
                        region,
                        (_event_title(text[piece_start:piece_end]), "dated event"),
                        piece_start,
                        piece_end,
                    )
        close()
    return MatterInventory(
        rules_id=EVENT_INVENTORY_RULES_ID,
        regions=regions,
        matters=tuple(units),
        unassigned=tuple(sorted(unassigned)),
    )


EVENT_CUE = _EVENT_CUE
REFERS_TO_EVENTS = _REFERS_TO_EVENTS

__all__ = [
    "CORPORATE_EVENT_FAMILY",
    "DISCLOSURE_UNIT_INVENTORY_RULES_ID",
    "EVENT_CUE",
    "EVENT_INVENTORY_RULES_ID",
    "REFERS_TO_EVENTS",
    "event_inventory",
    "event_regions",
]
