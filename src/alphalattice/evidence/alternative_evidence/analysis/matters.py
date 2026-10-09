"""Inventory named litigation matters in filings.

Named litigation matters: a provisional reading inventory of a filing's
Legal Proceedings items and contingencies notes, and an accounted reading
plan over it.

This is navigation, not proof of discovery and not extraction. A litigation
region (an Item 3 / Part II Item 1 body, a contingencies or litigation note)
is split into provisional matters by the source's own signposts -- the
sub-headings inside it, a caption lead that opens a paragraph ("Cruz
Litigation.", "Connecticut:", "On May 10, 2021, <party> filed a
complaint"), an italicised caption with " v. ", and the aliases a filing
defines for a matter ("(the "Edmonds")"). Text no signpost claims stays
visible as UNASSIGNED, and a region whose end the structure cannot see is
said to be uncertain. Names, parties and courts alone never prove identity:
a matter handle is local to one document, distinct provisional matters
stay distinct, and nothing here links a matter across filings.

The reading plan hands the matters to the verified reader as exact source
ranges, in windows that fit one read each, visiting every matter once
before any matter gets a second window, so that a verbose matter cannot
spend the budget the others need. What a budget leaves unread is returned
as pending windows the caller can issue next -- never as read.

"""

from __future__ import annotations

import re
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise

from ..documents.structure import DocumentStructure, StructureHeading
from ..retrieval.session import sentence_ends

MATTER_INVENTORY_RULES_ID = "alternative-evidence.litigation-matter-inventory.v2"
MATTER_SEGMENTATION_RULES_ID = "alternative-evidence.litigation-matter-inventory.v3"
"""v3 (2026-09-19; the candidate's inventory, the production default stays v2
pending the comparison): a paragraph the source cut inside a sentence (a page
break) is read whole with the next; a dated opening in the source's own
phrasings opens a matter ("Beginning in December 2017, the ... Panel ...
consolidated numerous cases", "Between September 25 and October 31, 2023,
five class action suits were filed", "In October and November 2025, two class
actions were filed"); a paragraph whose first sentence names a caption, a
docket or a case number and a case-opening filing opens one whatever its
word order ("The plaintiffs in the putative securities class action lawsuit,
captioned 4:18-cv-07669-HSG, initially filed on December 21, 2018 ..."); a
"second"/"another ... was filed" sentence with its own caption or case number
opens one inside a paragraph (Madera, then Howell). Identity stays local to a
document and provisional: a signpost opened it, nothing proved it."""
"""v2 (2026-09-18): a run-in label set off by a dash opens a matter as one
set off by a colon does ("Legal Proceedings -- NEE, FPL ... are the named
defendants in a ... lawsuit filed in ..."); "et al." inside a dated
case-opening filing's caption does not end the lead; "U.S. District Court
for the ... District of ..." is a court, with or without a regional
division ("for the District of Massachusetts"). NEE's securities class action,
derivative actions and antitrust suit were unassigned text under v1; "v."
inside a caption does not end the lead either."""
TOPIC_LANES_ALLOCATION_ID = "alternative-evidence.matter-evidence-allocation.topic-lanes.v1"
"""The integrated pipeline's allocation, v1 (2026-09-22). The needs are dealt
in one sealed service order (`service_order`), a function of the plan's needs
alone, so a later session resumes it where the chain stopped. Lanes are one
per issuer and *topic* (the primary topic the routing gives each need: a
litigation need the legal topic, a financing need the liquidity topic, an
event need the topic its item or note names). The order cycles the issuers in
the request's order; an issuer's turn serves one need from its lanes taken in
turn in the topics' declared order, skipping a lane with nothing to serve, so
an issuer's first windows are one per topic that has a need before any
topic's second, and an exhausted lane relinquishes its turns at once. A
lane's turn serves, alternately from an opening: an opening -- the first part
of an unopened disclosure: a unit's opening, or the whole disclosure of a
filing that names no unit, first; then the cued region statements, the
located reference targets and the uncued region text -- and a completion --
the region-level statements that say they apply to the units of a region in
which a unit has been opened (its qualifications: the accrual position, the
materiality closing), and the next part of a disclosure whose earlier parts
this chain has served, the earliest in source order first. The lane
alternates while both exist and serves whichever remains when one is spent.
Inside a lane, openings of one class are dealt latest filing first (by the
filing's own acceptance clock, the set's order for undated filings), then in
source order. Nothing is weighted or scored; the tie-break is the order stated
here. A need whose range touches a window already chosen in its document joins
it when the two fit the window ceiling, so one excerpt serves both and no byte
is read twice. A need whose unit an exact repeat in a later filing restates
(the same normalized text under the same identifier, as the comparison rules
prove) is passed over as delivered once the later unit's window has been read
in this chain, and is recorded as provided by that excerpt with both sources
named: one read, two pointers, no inferred change. A continuation whose
opening this chain has not served is refused by name. The retired selections'
allocations (`...unmet-needs.v2`, `...plan-prefix.v1`) are named by their
sealed records and read back as strings; nothing is dealt under them."""

_LEGAL_ITEM = re.compile(r"legal\s+proceedings", re.IGNORECASE)
_LEGAL_NOTE = re.compile(
    r"contingenc|litigation|legal\s+proceedings|legal\s+matters", re.IGNORECASE
)
_EMPHASIS = re.compile(r"\*+")
_WHITESPACE = re.compile(r"\s+")
_CAPTION_LEAD = re.compile(r"^(?P<title>[A-Z](?:[^.\n]|\.(?=[A-Za-z])){2,110}?)\.\s+(?=[A-Z(])")
"""'Cruz Litigation. On August 27, 2020, ...': a short title-cased phrase
that opens a paragraph and ends in a period before the statement."""
_SMALL_WORDS = frozenset(
    {"of", "the", "and", "in", "for", "v.", "vs.", "at", "on", "to", "a", "an", "de", "del"}
)
_JURISDICTION_LEAD = re.compile(
    r"^(?P<title>[A-Z][A-Za-z.\- ]{2,40}?)\s*(?::|\u2013|\u2014)\s+(?=[A-Z])"
)
"""'Connecticut: In January 2024, ...'; 'Legal Proceedings -- NEE, FPL, ...'
(an en or em dash in the source):
a run-in label set off by a colon or a dash."""
_DATE_WORDS = (
    r"(?:[A-Z][a-z]+\s+\d{1,2}(?:,?\s+(?:and\s+)?\d{1,2})*,\s+\d{4}|[A-Z][a-z]+\s+\d{4}|\d{4})"
)
_OPENING = (
    r"(?:complaints?|lawsuits?|actions?|petitions?|suits?|proceedings?|class\s+actions?|"
    r"charges?|claims?|writs?)"
)
_PROCEDURAL = re.compile(
    r"\b(?:amended|motion|brief|notice|answer|reply|opposition|supplemental|counterclaims?|"
    r"appeal)\b",
    re.IGNORECASE,
)
_FILED_LEAD = re.compile(
    r"^(?:On|In)\s+"
    + _DATE_WORDS
    + r"(?:,?\s+(?:and\s+)?"
    + _DATE_WORDS
    + r")*,?\s+(?:respectively,\s+)?"
    r"(?P<rest>(?:[^.]|\.(?=[A-Za-z)])|(?:(?<=\bInc)|(?<=Corp)|(?<=\bLtd)|(?<=\bLLC)|(?<=\bal)|(?<=\bv)|(?<=\bvs))\.){0,260}?"
    r"\b(?:filed|commenced|initiated|brought)\b)(?P<after>[^.]{0,80})",
    re.DOTALL,
)
"""'On May 10, 2021, Surgical Instrument Service Company, Inc. ("SIS")
filed a complaint ...': a dated filing of a case-opening pleading opens a
matter; a dated motion, brief, notice or amended complaint continues one
(`_OPENING` must name the pleading, `_PROCEDURAL` must not)."""
_OPENING_WORD = re.compile(r"\b" + _OPENING + r"\b", re.IGNORECASE)
_FILINGS_LEAD = re.compile(
    r"^(?P<rest>(?:[A-Z][a-z]+\s+)?(?:putative\s+|purported\s+|shareholder\s+|securities\s+|"
    r"antitrust\s+|derivative\s+|class\s+action\s+|individual\s+)*"
    + _OPENING
    + r")\s+(?:were|was|have\s+been|has\s+been)\s+filed\b",
    re.IGNORECASE,
)
"""'Three class action complaints were filed against the Company ...'"""
_FILED_VERB = re.compile(
    r"\s+(?:were\s+|was\s+|has\s+been\s+|have\s+been\s+)?"
    r"(?:filed|commenced|initiated|brought)\b.*$",
    re.IGNORECASE | re.DOTALL,
)
_CAPTION_V = re.compile(r"\*([^*\n]{3,160}?\bv\.?\s[^*\n]{2,160}?)\*")
_DEFINED_ALIAS = re.compile(r"\(\s*(?:the\s+)?[\"\u201c]([^\"\u201d\n]{2,80})[\"\u201d]\s*\)")
_CAPTIONED = re.compile(r"captioned\s+[^.]{0,80}?as\s+[\"\u201c]([^\"\u201d\n]{3,160})[\"\u201d]")
_CASE_NUMBER = re.compile(
    r"(?:Case\s+)?No\.\s*[0-9]{1,2}:[0-9]{2}-[a-z]{2}-[0-9]{3,6}(?:-[A-Z\-]{2,12})?"
    r"|Case\s+No\.\s*[0-9A-Z\-]{4,20}",
    re.IGNORECASE,
)
_COURT = re.compile(
    r"(?:(?:United States|U\.S\.) District Court for the (?:[A-Z][A-Za-z ]+?)?District of "
    r"[A-Z][a-z]+(?: [A-Z][a-z]+)?"
    r"|[A-Z][a-z]+ District of [A-Z][a-z]+(?: [A-Z][a-z]+)? Court"
    r"|(?:U\.S\.|United States) Court of Appeals for the [A-Z][A-Za-z ]+?Circuit"
    r"|[A-Z][a-z]+ Circuit Court of Appeals"
    r"|Chancery Court for [A-Z][a-z]+ County, [A-Z][a-z]+"
    r"|Circuit Court of the [A-Z][a-z]+ Judicial Circuit in [A-Z][a-z]+ County, [A-Z][a-z]+"
    r"|[A-Z][a-z]+ state court|(?:U\.S\.|United States) Supreme Court"
    r"|Court of International Trade|Judicial Panel on Multidistrict Litigation)"
)
_REPORT_SECTION = re.compile(
    r"^(?:report\s+of\s+independent|management[\u2019']s\s+discussion|results\s+of\s+operations|"
    r"liquidity\s+and\s+capital\s+resources|executive\s+overview|quantitative\s+and\s+qualitative|"
    r"controls\s+and\s+procedures)",
    re.IGNORECASE,
)
"""Section titles that never sit inside a note: one of them after a note
heading means the note ended before the structure saw its end."""
_NAMED_HEADING = re.compile(r"litigation|lawsuit|settlement|\bv\.\s", re.IGNORECASE)
_LITIGATION_CUE = re.compile(
    r"\b(?:v\.|vs\.|lawsuits?|complaints?|plaintiffs?|defendants?|courts?|settle(?:d|ment)?|"
    r"dismiss(?:ed|al)?|appeals?|class\s+actions?|jury|verdict|arbitration|MDL|multidistrict|"
    r"proceedings?|enforcement|consent\s+decree|penalt(?:y|ies)|sanction(?:s|ing)?|"
    r"notice\s+of\s+(?:potential\s+)?violation|judicial\s+review|petition)\b",
    re.IGNORECASE,
)
"""The words a legal proceeding is described with; "motion" and "trial"
alone are not among them (a store trial, a motion to reconsider a
budget), so an MD&A caption does not become a matter."""
_DATED_OPENING = re.compile(
    r"^(?:Beginning|Between|In|On|During)\s+(?:in\s+)?"
    r"(?:(?:[A-Z][a-z]+|\d{1,2},?|and|through|to)\s+){0,8}?\d{4}\b"
    r"[^.]{0,220}?\b(?:were filed|was filed|filed|commenced|consolidated numerous cases)\b",
    re.DOTALL,
)
"""A dated opening in the source's own phrasings (rules v3): a date or a
span of dates, then a filing or a consolidation, with a caption, a docket or
a case number somewhere in the paragraph."""
_DOCKET = re.compile(r"\d:\d{2}-cv-\d{4,6}")
_CAPTION_FREE = re.compile(
    r"\b[A-Z][A-Za-z.,&'\u2019\- ]{1,80}?\s+v\.\s+[A-Z][A-Za-z.,&'\u2019\- ]{1,80}"
)
_IN_RE = re.compile(r"\bIn re\b|\bMDL No\.\s*\d+", re.IGNORECASE)
_CASE_OPENING_FILED = re.compile(
    r"\b(?:initially\s+)?(?:filed|commenced|initiated|brought)\b(?:\s+(?:on|in)\b)?"
    r"(?P<object>[^.]{0,60})"
)
"""The filing verb of a captioned opening, with what follows it: a case
opening ("filed on December 21, 2018", "filed a lawsuit") and not a motion,
an amended complaint or an appeal ("filed an amended complaint")."""
_FILING_SENTENCE = re.compile(
    r"\b(?:filed|commenced|initiated|brought)\s+(?:a\s+|an\s+|the\s+|its\s+|their\s+)?"
    r"(?:putative\s+|purported\s+|shareholder\s+|securities\s+|derivative\s+|class\s+action\s+|"
    r"consolidated\s+|related\s+|separate\s+|second\s+|third\s+|new\s+)*"
    r"(?:suit|lawsuit|complaint|action|petition|class\s+action|proceeding|claim|writ)s?\b"
    r"|\b(?:suit|lawsuit|complaint|action|petition|class\s+action|proceeding)s?\s+"
    r"(?:was|were)\s+filed\b",
    re.IGNORECASE,
)
"""A sentence that states the filing of a case-opening pleading ('filed a
lawsuit', 'filed suit', 'a second class action was filed'), as opposed to
a motion, a brief, an answer, an appeal or an amended complaint, which
`_PROCEDURAL` names."""
_TERMINAL = re.compile(r"[.:;!?\"\u201d\u2019)\]]\s*$")
_REFERS_TO_MATTERS = re.compile(
    r"\b(?:matters?|proceedings?|legal\s+proceedings|claims?|actions?|lawsuits?|cases?)\s+"
    r"(?:not\s+)?(?:described|discussed|referred\s+to|set\s+forth|listed|disclosed)\s+"
    r"(?:above|below|herein|in\s+this\s+note)\b"
    r"|\b(?:these|those|such|all\s+of\s+the|some\s+or\s+all\s+of\s+the|any\s+of\s+the|"
    r"all\s+such|any\s+such)\s+(?:pending\s+)?"
    r"(?:matters|proceedings|legal\s+proceedings|actions|cases|lawsuits|claims)\b"
    r"|\bany\s+pending\s+(?:claim|proceeding|litigation|matter|legal\s+proceeding)s?\b"
    r"|\bcurrently\s+pending\s+(?:legal\s+)?(?:proceedings|matters|claims|litigation)\b"
    r"|\bpreviously\s+reported\b|\bmatters\s+not\s+described\b",
    re.IGNORECASE,
)
"""A region-level statement that says it applies to the region's matters as
a body: 'the legal proceedings described above', 'some matters described
below', 'these matters', 'some or all of the matters', 'any pending claim,
proceeding or litigation', 'the previously reported ... matter'. Only such
a statement is associated with the matters; the rest stays visible at the
region level with its scope unresolved. 'Described above' alone is not
enough: a matter's own paragraph says 'the methodology described above'
of itself."""
_CROSS_REFERENCE = re.compile(
    r"(?:[Ss]ee|[Rr]efer to|described in|discussed in|set forth in|contained in|incorporated)"
    r"\s+(?:the\s+)?(?:discussion\s+of\s+[^.]{0,60}?\s+in\s+)?"
    r"(?:the\s+[a-z]+\s+paragraph\s+of\s+)?(?:the\s+)?"
    r"(?P<target>Note\s+\d+[A-Z]?|Part\s+[IV]+,?\s*(?:\u201c|\")?Item\s+\d+[A-Z]?|Item\s+\d+[A-Z]?"
    r"|\u201c[^\u201d]{3,80}\u201d|\"[^\"]{3,80}\")",
)
"""An explicit reference a litigation region makes to another location of
the same filing: another note or item, or a quoted section title."""
_OF_ANOTHER_FILING = re.compile(
    r"[^.]{0,80}?\b(?:in|of)\s+(?:our|its|the|the\s+Company[\u2019\']s)\s+"
    r"(?:Annual|Quarterly)\s+Report\s+on\s+Form\s+10-[KQ]\b"
)
_FILING_REFERENCE = re.compile(
    r"(?:(?P<section>(?:Part\s+[IV]+,?\s*)?[\u201c\"]?(?:Item|Note)\s+\d+[A-Z]?[^\u201d\"]{0,60}?)"
    r"[\u201d\"]?\s+(?:in|of)\s+)?"
    r"\b(?:in|to|of|see)?\s*(?:our|its|the|the\s+Company[\u2019\']s)\s+"
    r"(?P<target>(?:Quarterly\s+Report\s+on\s+Form\s+10-Q|Annual\s+Report\s+on\s+Form\s+10-K)"
    r",?\s+for\s+the\s+(?:fiscal\s+)?(?:quarter|period|year)\s+ended\s+"
    r"[A-Z][a-z]+\s+\d{1,2},\s+\d{4})"
)
"""A reference to another filing of the issuer by form and period: what a
10-Q's Legal Proceedings item says it updates ('as updated in Part II,
'Item 1. Legal Proceedings' in our Quarterly Report on Form 10-Q for the
quarter ended April 3, 2026")."""
_GENERIC_LEAD = re.compile(
    r"^(?:from\s+time\s+to\s+time|we\s+are\s+(?:and\s+may\s+be\s+)?(?:involved|subject|a\s+party)|"
    r"the\s+company\s+is\s+(?:currently\s+)?(?:involved|subject|a\s+party)|"
    r"a\s+liability|based\s+on\s+(?:information\s+currently\s+available|currently\s+available)|"
    r"we\s+(?:also\s+)?have\s+certain\s+other|the\s+company\s+(?:accrues|is\s+subject)|"
    r"we\s+have\s+included\s+information|for\s+(?:further\s+)?information)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class LitigationRegion:
    """One Legal Proceedings item or contingencies/litigation note body."""

    kind: str
    """`ITEM` or `NOTE`."""
    heading: str
    heading_start: int
    body_start: int
    body_end: int
    end_basis: str
    """What bounded the body: the next governing heading, or the end of the
    text -- in which case the end is uncertain, not proved."""
    family: str = "LITIGATION"
    """The inventory family the region belongs to: `LITIGATION` here,
    `CORPORATE_EVENT` for `events.py`, which shares this shape."""

    @property
    def end_uncertain(self) -> bool:
        """Report whether this region has an uncertain end."""
        return self.end_basis.startswith("end of text")


@dataclass(frozen=True, slots=True)
class Paragraph:
    """Hold one paragraph's exact source range and text."""

    character_start: int
    character_end: int
    text: str


@dataclass(frozen=True, slots=True)
class ProvisionalMatter:
    """One provisional matter.

    One provisional matter: where its text is, what the source calls it,
    and the references it states. Provisional: a signpost opened it, and a
    different signpost may open text that belongs to the same case.
    """

    handle: str
    title: str
    basis: str
    """The signpost that opened it: `sub-heading`, `caption`, `jurisdiction`,
    `dated filing`, `filings`, `italic caption` or `caption reference`."""
    region_heading: str
    group: str
    """The sub-heading the matter sits under, or the region heading."""
    ranges: tuple[tuple[int, int], ...]
    aliases: tuple[str, ...] = ()
    case_numbers: tuple[str, ...] = ()
    courts: tuple[str, ...] = ()
    source_bytes: int = 0
    """UTF-8 bytes of the matter's source text (character counts are not
    bytes: a curly quote is three)."""
    family: str = "LITIGATION"
    """`LITIGATION` or `CORPORATE_EVENT`: which inventory opened the unit."""

    @property
    def character_count(self) -> int:
        """Count characters assigned to this provisional matter."""
        return sum(end - start for start, end in self.ranges)

    @property
    def named(self) -> bool:
        """Check whether a source names a specific proceeding.

        Whether the source names this as a proceeding -- a caption, a
        jurisdiction lead, a dated case-opening filing, an italic caption, a
        captioned reference, a stated court or case number, or a heading
        that itself names litigation, a lawsuit or a settlement -- as
        opposed to a topical sub-heading group ("Environmental Matters",
        "Tax Claims"). The reading plan visits named matters first; a
        topical group is not a case.
        """
        return (
            self.basis != "sub-heading"
            or bool(self.courts or self.case_numbers)
            or _NAMED_HEADING.search(self.title) is not None
        )


@dataclass(frozen=True, slots=True)
class MatterInventory:
    """Hold named matters and unassigned litigation text."""

    rules_id: str
    regions: tuple[LitigationRegion, ...]
    matters: tuple[ProvisionalMatter, ...]
    unassigned: tuple[tuple[int, int, str], ...]
    """`(start, end, why)` ranges inside the regions that no matter claims:
    boilerplate, accrual policy, a group's own lead-in, or text no signpost
    opened. Visible, and readable on demand."""

    @property
    def region_characters(self) -> int:
        """Count characters across the inventoried litigation regions."""
        return sum(value.body_end - value.body_start for value in self.regions)

    @property
    def unassigned_characters(self) -> int:
        """Count litigation characters left outside named matters."""
        return sum(end - start for start, end, _ in self.unassigned)


def _plain(text: str) -> str:
    return _WHITESPACE.sub(" ", _EMPHASIS.sub("", text.replace("\xa0", " "))).strip()


def _title_like(title: str) -> bool:
    words = title.split()
    if not 1 <= len(words) <= 12:
        return False
    return all(
        word[0].isupper()
        or word[0].isdigit()
        or word.casefold() in _SMALL_WORDS
        or not word[0].isalpha()
        for word in words
    )


def litigation_regions(text: str, structure: DocumentStructure) -> tuple[LitigationRegion, ...]:
    """Find the filing's legal proceedings and litigation notes.

    The Legal Proceedings items and contingencies/litigation notes past
    the cover, each bounded by the next governing heading.
    """
    regions: list[LitigationRegion] = []
    headings = structure.headings
    for index, heading in enumerate(headings):
        if heading.character_start < structure.cover_end:
            continue
        if heading.kind == "ITEM" and _LEGAL_ITEM.search(heading.text):
            closers = {"PART", "ITEM"}
        elif heading.kind == "NOTE" and _LEGAL_NOTE.search(heading.text):
            closers = {"PART", "ITEM", "NOTE"}
        else:
            continue
        body_end = len(text)
        basis = "end of text (no later governing heading: the end is uncertain)"
        for later in headings[index + 1 :]:
            if later.kind in closers:
                body_end = later.character_start
                basis = f"next {later.kind} heading: {later.text[:60]}"
                break
            if later.kind in {"SUB", "SUB_UNCERTAIN"} and _REPORT_SECTION.match(later.text):
                # Another part of the report opens before the note's own end
                # was seen: the note ends here at the latest, uncertainly.
                body_end = later.character_start
                basis = (
                    "end of text (another part of the report opens: "
                    f"{later.text[:50]}; the note's own end was not seen)"
                )
                break
        if not _plain(text[heading.character_end : body_end]):
            continue
        regions.append(
            LitigationRegion(
                kind=heading.kind,
                heading=heading.text,
                heading_start=heading.character_start,
                body_start=heading.character_end,
                body_end=body_end,
                end_basis=basis,
            )
        )
    return tuple(regions)


def _paragraphs(
    text: str, start: int, end: int, *, join_cut: bool = False
) -> tuple[Paragraph, ...]:
    """Split a source range into blank-line paragraphs.

    Blank-line paragraphs of `text[start:end]`. With `join_cut` (rules v3)
    a paragraph that ends inside a sentence -- no terminal punctuation, a
    letter last: a page break in the source -- is one paragraph with the
    next, over the same source offsets, so a lead broken by the page is
    seen and no quotation is invented.
    """
    out: list[Paragraph] = []
    for match in re.finditer(r"[^\n]+(?:\n(?!\s*\n)[^\n]+)*", text[start:end]):
        body = match.group(0)
        lead = len(body) - len(body.lstrip())
        core = body.strip()
        if not core or re.fullmatch(r"\d{1,3}", core):
            continue
        paragraph = Paragraph(
            character_start=start + match.start() + lead,
            character_end=start + match.start() + lead + len(core),
            text=core,
        )
        if (
            join_cut
            and out
            and _TERMINAL.search(out[-1].text) is None
            and out[-1].text[-1:].isalpha()
        ):
            previous = out.pop()
            joined_start, joined_end = previous.character_start, paragraph.character_end
            paragraph = Paragraph(joined_start, joined_end, text[joined_start:joined_end].strip())
        out.append(paragraph)
    return tuple(out)


_TABLE_PLACEHOLDER = re.compile(
    r"^\[Table of \d+ rows? not carried into the canonical text", re.IGNORECASE
)
_FOOTNOTE_RULE = re.compile(r"^_{5,}$")
_SUB_HEADING_LINE = re.compile(r"^(?:[A-Z][\w&'\u2019/,-]*\s*){1,8}$")
_BULLET = re.compile(r"^[\u2022\u00b7\u25e6\u25aa\u2013\u2014\-\*]\s*")


def _is_sub_heading(
    paragraph: Paragraph, structure: DocumentStructure, signpost: Callable[[str], bool]
) -> bool:
    """Identify a subheading that groups litigation units.

    A sub-heading line of the region (the structure's SUB heading at the
    paragraph's start, or a short title-case line the family's signpost
    does not open): the group of the units under it, not text to read.
    """
    plain = _plain(paragraph.text)
    if not plain or len(plain) > 80 or plain.endswith((".", ":", ";")):
        return False
    for heading in structure.headings:
        if (
            heading.kind in {"SUB", "SUB_UNCERTAIN"}
            and heading.character_start == paragraph.character_start
            and _plain(heading.text) == plain
        ):
            return True
    return _SUB_HEADING_LINE.match(plain) is not None and not signpost(plain)


def source_blocks(
    text: str,
    region: LitigationRegion,
    structure: DocumentStructure,
    *,
    signpost: Callable[[str], bool],
) -> list[tuple[Paragraph, str]]:
    """The region's paragraphs as the source groups them.

    The region's paragraphs as the source groups them: a bulleted line
    continues the paragraph that introduced the list, and a paragraph cut
    by a page break (no terminal punctuation, a lowercase letter or a comma
    last, not a sub-heading) continues into the next. Each block carries
    its kind: `heading`, `table`, `rule` or `text`. `signpost` is the
    family's own recognizer of a unit's text (an instrument phrase, an
    operations charge), which a sub-heading line never carries. Shared by
    the financing and operations inventories.
    """
    out: list[tuple[Paragraph, str]] = []
    for paragraph in _paragraphs(text, region.body_start, region.body_end, join_cut=False):
        plain = _plain(paragraph.text)
        if not plain:
            continue
        if _TABLE_PLACEHOLDER.match(plain):
            out.append((paragraph, "table"))
            continue
        if _FOOTNOTE_RULE.match(plain):
            out.append((paragraph, "rule"))
            continue
        if _is_sub_heading(paragraph, structure, signpost):
            out.append((paragraph, "heading"))
            continue
        if out:
            previous, kind = out[-1]
            previous_plain = _plain(previous.text)
            if kind == "text" and (
                _BULLET.match(plain) is not None
                or (
                    (previous_plain[-1].islower() or previous_plain.endswith(","))
                    and not _BULLET.match(previous_plain)
                    and previous_plain[-1] not in ".;:!?)\"'"
                )
            ):
                joined = Paragraph(
                    character_start=previous.character_start,
                    character_end=paragraph.character_end,
                    text=text[previous.character_start : paragraph.character_end],
                )
                out[-1] = (joined, "text")
                continue
        out.append((paragraph, "text"))
    return out


def _first_sentence(plain: str) -> str:
    ends = sentence_ends(plain)
    return plain[: ends[0]] if ends else plain


def _names_a_case(text: str) -> bool:
    return (
        _CAPTION_FREE.search(text) is not None
        or _CASE_NUMBER.search(text) is not None
        or _DOCKET.search(text) is not None
        or _IN_RE.search(text) is not None
    )


def _case_title(text: str, fallback: str) -> str:
    caption = _CAPTION_FREE.search(text)
    if caption is not None:
        return _plain(caption.group(0))[:120]
    in_re = re.search(r"In re [^.,(]{3,120}", text)
    if in_re is not None:
        return _plain(in_re.group(0))[:120]
    number = _CASE_NUMBER.search(text) or _DOCKET.search(text)
    if number is not None:
        return number.group(0)
    return _plain(fallback)[:120]


def _segmentation_lead(paragraph: Paragraph, *, cue_required: bool) -> tuple[str, str] | None:
    """The rules-v3 openings a v2 lead does not see.

    The rules-v3 openings a v2 lead does not see: a dated opening in the
    source's own phrasing, or a first sentence that names a case and a
    case-opening filing, whatever its word order.
    """
    plain = _plain(paragraph.text)
    if _GENERIC_LEAD.match(plain) or (cue_required and _LITIGATION_CUE.search(plain) is None):
        return None
    if _DATED_OPENING.match(plain) is not None and _names_a_case(plain):
        return _case_title(plain, plain[:80]), "dated opening"
    sentence = _first_sentence(plain)
    filed = _CASE_OPENING_FILED.search(sentence)
    if (
        filed is not None
        and _names_a_case(sentence)
        and _PROCEDURAL.search(filed.group("object")) is None
    ):
        return _case_title(sentence, plain[:80]), "captioned opening"
    return None


def _region_level(plain: str) -> bool:
    """Whether a paragraph that opens no matter is a region-level statement (rules v3).

    Whether a paragraph that opens no matter is a region-level statement
    (rules v3): it names no case and either opens generically ('From time
    to time', 'We are subject to', 'The Company accrues') or refers to the
    region's matters as a body ('some or all of the matters', 'the legal
    proceedings described above'). Such a paragraph closes the matter
    before it: the materiality closing after the last matter is the
    region's, not that matter's.
    """
    return not _names_a_case(plain) and (
        _GENERIC_LEAD.match(plain) is not None or _REFERS_TO_MATTERS.search(plain) is not None
    )


def _second_filings(paragraph: Paragraph) -> tuple[int, ...]:
    """Find later case-opening filings within one paragraph.

    Offsets inside the paragraph where a sentence after the first states
    the filing of a case-opening pleading and names the case -- a caption,
    a docket or a case number in that sentence or the next -- and so opens
    a matter of its own (rules v3): 'A second class action was filed in
    March 2026 ... Howell v. Costco ...', 'In the coverage action, five
    plaintiff insurance companies filed suit (Century Indemnity Company,
    et al. v. Aqua-Chem, Inc. ...)'. A motion, an amended complaint or an
    appeal continues the matter it is filed in.
    """
    text = paragraph.text
    ends = [0, *sentence_ends(text)]
    if ends[-1] < len(text):
        ends.append(len(text))
    sentences = [(ends[i], ends[i + 1]) for i in range(len(ends) - 1)]
    cuts: list[int] = []
    for index in range(1, len(sentences)):
        start, end = sentences[index]
        sentence = text[start:end]
        filing = _FILING_SENTENCE.search(sentence)
        if filing is None or _PROCEDURAL.search(sentence) is not None:
            continue
        following = text[end : sentences[index + 1][1]] if index + 1 < len(sentences) else ""
        if _names_a_case(sentence) or _names_a_case(following):
            lead = start + (len(sentence) - len(sentence.lstrip()))
            cuts.append(paragraph.character_start + lead)
    return tuple(cuts)


def _lead(paragraph: Paragraph, *, cue_required: bool) -> tuple[str, str] | None:
    """`(title, basis)` when the paragraph opens a provisional matter.

    `(title, basis)` when the paragraph opens a provisional matter. A
    caption or jurisdiction lead needs a litigation cue where the region
    holds mixed content (a note); inside a Legal Proceedings item every
    caption names a proceeding.
    """
    plain = _plain(paragraph.text)
    if _GENERIC_LEAD.match(plain):
        return None
    cued = _LITIGATION_CUE.search(plain) is not None or not cue_required
    filed = _FILED_LEAD.match(plain)
    if (
        filed is not None
        and _OPENING_WORD.search(filed.group("rest") + filed.group("after")) is not None
        and _PROCEDURAL.search(filed.group("rest") + filed.group("after")) is None
    ):
        captioned = _CAPTIONED.search(plain)
        italic = _CAPTION_V.search(paragraph.text)
        if captioned is not None:
            return captioned.group(1).strip(" ."), "caption reference"
        if italic is not None:
            return _plain(italic.group(1))[:120], "italic caption"
        title = _FILED_VERB.sub("", filed.group("rest")).strip(" ,")
        title = re.sub(r"^(?:the\s+following\s+|the\s+|a\s+|an\s+)", "", title, flags=re.I)
        return title[:120], "dated filing"
    filings = _FILINGS_LEAD.match(plain)
    if filings is not None:
        captioned = _CAPTIONED.search(plain)
        if captioned is not None:
            return captioned.group(1).strip(" ."), "caption reference"
        return filings.group("rest")[:120], "filings"
    caption = _CAPTION_LEAD.match(plain)
    if caption is not None and cued and _title_like(caption.group("title")):
        return caption.group("title"), "caption"
    jurisdiction = _JURISDICTION_LEAD.match(plain)
    if jurisdiction is not None and cued and _title_like(jurisdiction.group("title")):
        return jurisdiction.group("title"), "jurisdiction"
    return None


def matter_inventory(
    text: str,
    structure: DocumentStructure,
    *,
    document_id: str = "DOC",
    rules: str = MATTER_INVENTORY_RULES_ID,
) -> MatterInventory:
    """Inventory provisional litigation matters and unassigned text.

    The provisional matters of every litigation region of one canonical
    text, with the unassigned remainder. Deterministic; no model. `rules`
    names the segmentation: the production rules (v2) by default, or the
    candidate's (`MATTER_SEGMENTATION_RULES_ID`), which the record carries
    as `rules_id` either way.
    """
    if rules not in {MATTER_INVENTORY_RULES_ID, MATTER_SEGMENTATION_RULES_ID}:
        raise ValueError("alternative_evidence.matter_inventory_rules_unknown")
    segmentation = rules == MATTER_SEGMENTATION_RULES_ID
    regions = litigation_regions(text, structure)
    matters: list[ProvisionalMatter] = []
    unassigned: list[tuple[int, int, str]] = []
    counter = 0

    def close(
        region: LitigationRegion,
        group: str,
        grouped: bool,
        lead: tuple[str, str] | None,
        paragraphs: list[Paragraph],
    ) -> None:
        if not paragraphs:
            return
        if lead is None:
            # Under a sub-heading the heading is the signpost; under the
            # region's own lead-in nothing is, and the text stays visible.
            cued = [p for p in paragraphs if _LITIGATION_CUE.search(p.text) is not None]
            if grouped and cued and not all(_GENERIC_LEAD.match(_plain(p.text)) for p in cued):
                lead = (group, "sub-heading")
            else:
                for paragraph in paragraphs:
                    why = (
                        "no signpost opened this text"
                        if _LITIGATION_CUE.search(paragraph.text) is not None
                        else "lead-in or boilerplate (no litigation cue)"
                    )
                    unassigned.append((paragraph.character_start, paragraph.character_end, why))
                return
        emit(region, group, lead, paragraphs[0].character_start, paragraphs[-1].character_end)

    def emit(
        region: LitigationRegion, group: str, lead: tuple[str, str], start: int, end: int
    ) -> None:
        nonlocal counter
        counter += 1
        plain = _plain(text[start:end])
        matters.append(
            ProvisionalMatter(
                handle=f"M-{document_id}-{counter:03d}",
                title=lead[0],
                basis=lead[1],
                region_heading=region.heading,
                group=group,
                # One contiguous range: the matter's paragraphs are consecutive
                # by construction, and the page numbers or blank lines between
                # them are part of the text a reader must be given whole.
                ranges=((start, end),),
                aliases=tuple(
                    dict.fromkeys(_DEFINED_ALIAS.findall(plain) + _CAPTIONED.findall(plain))
                )[:12],
                case_numbers=tuple(dict.fromkeys(m.group(0) for m in _CASE_NUMBER.finditer(plain)))[
                    :12
                ],
                courts=tuple(dict.fromkeys(m.group(0) for m in _COURT.finditer(plain)))[:8],
                source_bytes=len(text[start:end].encode("utf-8")),
            )
        )

    for region in regions:
        subs = [
            heading
            for heading in structure.headings
            if heading.kind in {"SUB", "SUB_UNCERTAIN"}
            and region.body_start <= heading.character_start < region.body_end
        ]
        starts = [region.body_start, *(sub.character_end for sub in subs)]
        ends = [*(sub.character_start for sub in subs), region.body_end]
        titles = [region.heading, *(sub.text for sub in subs)]
        for position, (group_start, group_end, group) in enumerate(
            zip(starts, ends, titles, strict=True)
        ):
            # A sub-heading is a signpost; so is a note heading that names
            # litigation or a settlement, unlike "Contingencies" or "Legal
            # Proceedings", which name the item, not a matter.
            grouped = position > 0 or (
                region.kind == "NOTE"
                and re.search(r"litigation|settlement|lawsuit", region.heading, re.I) is not None
            )
            current: list[Paragraph] = []
            current_lead: tuple[str, str] | None = None
            after_statement = False
            for paragraph in _paragraphs(text, group_start, group_end, join_cut=segmentation):
                cue_required = region.kind == "NOTE"
                lead = _lead(paragraph, cue_required=cue_required)
                if lead is None and segmentation:
                    lead = _segmentation_lead(paragraph, cue_required=cue_required)
                if lead is None:
                    if segmentation and (current_lead is not None or after_statement):
                        # A region-level statement after a matter is the region's
                        # (rules v3): it closes the matter and stays visible as
                        # unassigned; text after it with no signpost of its own
                        # belongs to no matter either.
                        if _region_level(_plain(paragraph.text)):
                            close(region, group, grouped, current_lead, current)
                            current, current_lead = [], None
                            unassigned.append(
                                (
                                    paragraph.character_start,
                                    paragraph.character_end,
                                    "region-level statement",
                                )
                            )
                            after_statement = True
                            continue
                        if after_statement:
                            unassigned.append(
                                (
                                    paragraph.character_start,
                                    paragraph.character_end,
                                    "no signpost opened this text",
                                )
                            )
                            continue
                    current.append(paragraph)
                    continue
                close(region, group, grouped, current_lead, current)
                current, current_lead = [paragraph], lead
                after_statement = False
                if not segmentation:
                    continue
                # A further filing with its own caption inside the paragraph
                # opens a matter of its own from that sentence on.
                cuts = _second_filings(paragraph)
                if cuts:
                    pieces = [paragraph.character_start, *cuts, paragraph.character_end]
                    for piece_start, piece_end in pairwise(pieces[:-1]):
                        piece_lead = (
                            current_lead
                            if piece_start == paragraph.character_start
                            else (
                                _case_title(
                                    text[piece_start:piece_end],
                                    text[piece_start : piece_start + 80],
                                ),
                                "second filing",
                            )
                        )
                        emit(region, group, piece_lead, piece_start, piece_end)
                    last_start = pieces[-2]
                    current = [
                        Paragraph(
                            last_start,
                            paragraph.character_end,
                            text[last_start : paragraph.character_end],
                        )
                    ]
                    current_lead = (
                        _case_title(
                            text[last_start : paragraph.character_end],
                            text[last_start : last_start + 80],
                        ),
                        "second filing",
                    )
            close(region, group, grouped, current_lead, current)
    return MatterInventory(
        rules_id=rules,
        regions=regions,
        matters=tuple(matters),
        unassigned=tuple(sorted(unassigned)),
    )


@dataclass(frozen=True, slots=True)
class RegionStatement:
    """One region-level paragraph no matter claimed.

    One region-level paragraph no matter claimed: the accrual policy, the
    generic ordinary-course statement, a materiality closing, a
    qualification that follows the matters. `position` says where it sits
    relative to the region's named matters; `refers_to_matters` whether it
    says it applies to them ('described above', 'these matters'); the
    association stays unresolved otherwise.
    """

    region_heading: str
    character_start: int
    character_end: int
    position: str
    """`LEAD_IN` (before the first named matter), `TRAILING` (after the last)
    or `BETWEEN`; `WHOLE` when the region has no named matter."""
    cued: bool
    refers_to_matters: bool
    reference_text: str
    group: str = ""
    """The sub-heading the statement sits under when it is a paragraph of a
    topical group no case opened ('Legal Proceedings' as a note's lead-in,
    'Accounting for Loss Contingencies'); empty for unassigned text."""

    @property
    def scope(self) -> str:
        """Distinguish region-wide from unresolved statement scope.

        `REGION` when the statement says it applies to the region's matters,
        `UNRESOLVED` when the source leaves its scope open.
        """
        return "REGION" if self.refers_to_matters else "UNRESOLVED"


def region_statements(
    text: str,
    inventory: MatterInventory,
    *,
    cue: re.Pattern[str] | None = None,
    refers: re.Pattern[str] | None = None,
) -> tuple[RegionStatement, ...]:
    """Collect region-level statements outside named matters.

    Every region-level paragraph of every region, in order -- the
    unassigned text and the paragraphs of the topical groups no unit
    opened -- with its position among the named units and its explicit
    reference to them, if any. `cue` and `refers` are the family's own
    phrasings (the litigation ones by default; `events.py` passes the
    corporate-event ones).
    """
    cue = _LITIGATION_CUE if cue is None else cue
    refers = _REFERS_TO_MATTERS if refers is None else refers
    out: list[RegionStatement] = []
    for region in inventory.regions:
        named = [
            m.ranges[0] for m in inventory.matters if m.named and m.region_heading == region.heading
        ]
        first = min((s for s, _e in named), default=None)
        last = max((e for _s, e in named), default=None)
        pieces: list[tuple[int, int, str]] = [
            (start, end, "")
            for start, end, _why in inventory.unassigned
            if region.body_start <= start < region.body_end
        ]
        for matter in inventory.matters:
            if matter.named or matter.region_heading != region.heading:
                continue
            pieces.extend(
                (paragraph.character_start, paragraph.character_end, matter.group)
                for paragraph in _paragraphs(
                    text, matter.ranges[0][0], matter.ranges[-1][1], join_cut=True
                )
            )
        for start, end, group in sorted(pieces):
            body = text[start:end]
            if first is None:
                position = "WHOLE"
            elif end <= first:
                position = "LEAD_IN"
            elif last is not None and start >= last:
                position = "TRAILING"
            else:
                position = "BETWEEN"
            reference = refers.search(body)
            out.append(
                RegionStatement(
                    region_heading=region.heading,
                    character_start=start,
                    character_end=end,
                    position=position,
                    cued=cue.search(body) is not None,
                    refers_to_matters=reference is not None,
                    reference_text="" if reference is None else _plain(reference.group(0)),
                    group=group,
                )
            )
    return tuple(out)


@dataclass(frozen=True, slots=True)
class CrossReference:
    """An explicit reference a litigation region makes to another location."""

    region_heading: str
    character_start: int
    character_end: int
    target: str
    kind: str
    """`NOTE`, `ITEM`, `FILING` or `TITLE`."""
    section: str = ""
    """For a `FILING` reference, the item or note the citation names inside
    that filing ('Part I, "Item 3. Legal Proceedings"'), or empty."""


def cross_references(text: str, inventory: MatterInventory) -> tuple[CrossReference, ...]:
    """Every explicit reference each litigation region makes, in source order.

    Every explicit reference each litigation region makes, in source
    order: to a note or item of the same filing, to a quoted title, or to
    another filing of the issuer by form and period.
    """
    out: list[CrossReference] = []
    for region in inventory.regions:
        body = text[region.body_start : region.body_end]
        found: list[tuple[int, int, str, str]] = []
        for match in _CROSS_REFERENCE.finditer(body):
            if _OF_ANOTHER_FILING.match(body, match.end()) is not None:
                # 'Part I, "Item 3. Legal Proceedings" in our Annual Report on
                # Form 10-K ...': the item belongs to the filing named after
                # it, which the filing reference below records.
                continue
            target = _plain(match.group("target")).strip('\u201c\u201d"')
            lowered = target.casefold()
            if lowered.startswith("note"):
                kind = "NOTE"
            elif lowered.startswith(("item", "part")):
                kind = "ITEM"
            else:
                kind = "TITLE"
            found.append((match.start(), match.end(), target, kind))
        sections: dict[tuple[int, int], str] = {}
        for match in _FILING_REFERENCE.finditer(body):
            found.append((match.start(), match.end(), _plain(match.group("target")), "FILING"))
            section = match.group("section")
            sections[(match.start(), match.end())] = (
                "" if section is None else _plain(section).strip('\u201c\u201d" ')
            )
        out.extend(
            CrossReference(
                region_heading=region.heading,
                character_start=region.body_start + start,
                character_end=region.body_start + end,
                target=target,
                kind=kind,
                section=sections.get((start, end), ""),
            )
            for start, end, target, kind in sorted(found)
        )
    return tuple(out)


def _window_ranges(
    text: str,
    ranges: tuple[tuple[int, int], ...],
    *,
    window_bytes: int,
    overlap_bytes: int,
) -> tuple[tuple[int, int], ...]:
    """Windows over exact ranges.

    Windows over exact ranges: each at most
    `window_bytes` of UTF-8, ending at a sentence boundary inside its last
    `overlap_bytes` where one falls, the next starting back at that boundary
    minus the overlap.
    """
    windows: list[tuple[int, int]] = []
    for start, end in ranges:
        cursor = start
        while cursor < end:
            stop = _byte_bounded_end(text, cursor, end, window_bytes)
            if stop < end:
                tail = text[cursor:stop]
                floor = _byte_bounded_end(text, cursor, stop, max(window_bytes - overlap_bytes, 1))
                inside = [
                    cursor + value
                    for value in sentence_ends(tail)
                    if value <= len(tail) and cursor + value >= floor
                ]
                if inside and inside[-1] > cursor:
                    stop = inside[-1]
            windows.append((cursor, stop))
            if stop >= end:
                break
            back = _byte_bounded_start(text, stop, cursor, overlap_bytes)
            cursor = back if back > cursor else stop
    return tuple(windows)


def _byte_bounded_end(text: str, start: int, end: int, budget: int) -> int:
    """Find the longest byte-bounded prefix of a range.

    The largest `stop` in `(start, end]` such that `text[start:stop]`
    encodes to at most `budget` UTF-8 bytes (at least one character).
    """
    stop = min(end, start + budget)
    while stop > start + 1 and len(text[start:stop].encode("utf-8")) > budget:
        excess = len(text[start:stop].encode("utf-8")) - budget
        stop = max(start + 1, stop - max(1, excess // 3))
    return stop


def _byte_bounded_start(text: str, end: int, floor: int, budget: int) -> int:
    """Find the shortest byte-bounded suffix of a range.

    The smallest `start` in `[floor, end)` such that `text[start:end]`
    encodes to at most `budget` UTF-8 bytes.
    """
    start = max(floor, end - budget)
    while start < end - 1 and len(text[start:end].encode("utf-8")) > budget:
        excess = len(text[start:end].encode("utf-8")) - budget
        start = min(end - 1, start + max(1, excess // 3))
    return start


@dataclass(frozen=True, slots=True)
class EvidenceNeed:
    """One thing a complete reading of a filing's litigation regions needs.

    One thing a complete reading of a filing's litigation regions needs.
    A need with a range and a priority is read as a window of its own
    (`LEAD`: a named matter's opening; `CONTINUATION`: a later part of a
    matter or of a statement; `REGION_STATEMENT`: a region-level paragraph;
    `REFERENCE` with a range: the located target of an explicit reference
    outside the regions). A need with priority zero is derived: a matter's
    `QUALIFICATION` is the region statement that says it applies to the
    region's matters, provided when that statement is; a `REFERENCE` without
    a range is satisfied by what it points at, or not at all.
    """

    document_key: str
    matter_handle: str | None
    kind: str
    priority: int
    """The allocation class (1: a region's opening and closing statements
    that say they apply to its matters, and the whole disclosure of a filing
    that names no matter; 2: matter openings; 3: further parts and the
    statements between matters; 4: cued region statements whose scope is not
    stated; 5: reference targets outside the regions; 6: uncued region
    text); 0 for a derived need that is never a window of its own."""
    character_start: int | None
    character_end: int | None
    source_bytes: int
    part: int
    part_count: int
    region_heading: str
    detail: str
    refers_to_matters: bool = False
    reference: CrossReference | None = None
    target: ReferenceTarget | None = None
    family: str = "LITIGATION"
    """The family whose inventory stated the need: the lane it is served in."""
    disclosure: str = ""
    """The disclosure this need is a part of -- a unit's handle, or a region
    statement's own start -- so an extension follows its opening."""

    @property
    def window(self) -> tuple[int, int] | None:
        """Return the source window this evidence need requests."""
        if self.character_start is None or self.character_end is None:
            return None
        return self.character_start, self.character_end


@dataclass(frozen=True, slots=True)
class ReferenceTarget:
    """Where an explicit reference resolved to, inside its own document.

    Where an explicit reference resolved to, inside its own document:
    a litigation region (`REGION`), a provisional matter (`MATTER`), a
    heading outside the regions with the range its body governs
    (`HEADING`), another filing the document set may or may not hold
    (`FILING`), or nowhere the structure could see (`UNLOCATED`).
    """

    kind: str
    label: str
    matter_handle: str | None = None
    character_start: int | None = None
    character_end: int | None = None
    section: str = ""
    """For `FILING`: the item or note the citation names inside that filing."""


def required_reference_needs(
    target: ReferenceTarget, needs: tuple[EvidenceNeed, ...], *, section: str = ""
) -> tuple[EvidenceNeed, ...]:
    """Find the needs that deliver a referenced target.

    The needs whose delivery satisfies a reference to `target` inside the
    document `needs` describe: for a region, its opening and closing
    qualifications and every named matter's opening (class 1 and 2; every
    window need of the region when it has none of those); for a matter,
    its opening; for another filing (`needs` being that filing's), the
    same over the region the citation names, or over every litigation
    region when it names none. A precise target is what suffices: a
    reference never demands a whole filing, and a touched region is not a
    delivered one.
    """
    if target.kind == "MATTER":
        return tuple(
            need
            for need in needs
            if need.kind == "LEAD" and need.matter_handle == target.matter_handle
        )
    if target.kind == "REGION":
        headings = {target.label}
    elif target.kind == "FILING":
        wanted = _section_number(section or target.section)
        headings = {
            need.region_heading
            for need in needs
            if need.region_heading
            and (wanted is None or _section_number(need.region_heading) == wanted)
        }
    else:
        return ()
    windows = tuple(
        need
        for need in needs
        if need.priority > 0
        and need.region_heading in headings
        and need.kind in {"LEAD", "REGION_STATEMENT", "CONTINUATION"}
    )
    core = tuple(need for need in windows if need.priority in {1, 2})
    return core or windows


def _section_number(text: str) -> tuple[str, str] | None:
    """Read an item or note number from a heading or citation.

    `('ITEM' | 'NOTE', number)` named at the start of a heading or a citation
    ('Item 3. Legal Proceedings', 'NOTE 12: COMMITMENTS'), or None.
    """
    for kind, pattern in (("NOTE", _NOTE_NUMBER), ("ITEM", _ITEM_NUMBER)):
        match = pattern.search(text)
        if match is not None:
            return kind, match.group(1).casefold()
    return None


def reference_target_state(
    required: tuple[EvidenceNeed, ...], delivered: tuple[tuple[int, int], ...]
) -> tuple[str, int]:
    """`(state, delivered count)` of a located target.

    `(state, delivered count)` of a located target: `DELIVERED` when every
    required range lies whole inside a delivered range, `PARTIALLY_READ`
    when some do, `LOCATED` when none does. An empty requirement is a
    located target whose evidence could not be identified, never delivered.
    """
    if not required:
        return "LOCATED", 0
    count = sum(
        1
        for need in required
        if need.character_start is not None
        and need.character_end is not None
        and any(
            start <= need.character_start and need.character_end <= end for start, end in delivered
        )
    )
    if count == len(required):
        return "DELIVERED", count
    return ("PARTIALLY_READ" if count else "LOCATED"), count


def search_hit_corresponds(target: str, preview: str) -> bool:
    """Whether a search hit is the referenced location and not a passage that merely mentions it.

    Whether a search hit is the referenced location and not a passage that
    merely mentions it: the target (a note or item number, or a quoted
    title) opens a line of the hit's text, as a heading does. A document
    handle alone proves nothing.
    """
    wanted = _plain(target.strip('\u201c\u201d"')).casefold()
    if not wanted:
        return False
    for line in preview.splitlines():
        head = _plain(line).casefold()
        if head.startswith(wanted) and not head[len(wanted) : len(wanted) + 1].isdigit():
            return True
    return False


@dataclass(frozen=True, slots=True)
class DocumentNeeds:
    """A filing's litigation regions as needs.

    A filing's litigation regions as needs: the inventory, the region
    statements and cross-references it was read from, and every need in
    source order. `text` is the verified canonical text the ranges index.
    """

    document_key: str
    issuer: str
    text: str
    inventory: MatterInventory
    statements: tuple[RegionStatement, ...]
    references: tuple[CrossReference, ...]
    needs: tuple[EvidenceNeed, ...]

    @property
    def named_matters(self) -> tuple[ProvisionalMatter, ...]:
        """List the named matters in this document's evidence needs."""
        return tuple(m for m in self.inventory.matters if m.named)


_NOTE_NUMBER = re.compile(r"note\s+(\d+[a-z]?)", re.IGNORECASE)
_ITEM_NUMBER = re.compile(r"item\s+(\d+[a-z]?)", re.IGNORECASE)
_PART_NUMBER = re.compile(r"part\s+([iv]+)", re.IGNORECASE)


def bounded_region(
    text: str,
    headings: Sequence[StructureHeading],
    index: int,
    *,
    closers: Collection[str],
    kind: str,
    family: str,
) -> LitigationRegion | None:
    """The region a family's signposted heading opens.

    The region a family's signposted heading opens: its body runs to
    the next heading of a governing kind (`closers`), or to the end of the
    text with the end recorded as uncertain; None when the body holds no
    text. The construction the corporate-event, financing and operations
    inventories share (record section Y); the litigation inventory keeps
    its own, which also closes a note at another part of the report.
    """
    heading = headings[index]
    body_end = len(text)
    basis = "end of text (no later governing heading: the end is uncertain)"
    for later in headings[index + 1 :]:
        if later.kind in closers:
            body_end = later.character_start
            basis = f"next {later.kind} heading: {later.text[:60]}"
            break
    if not _plain(text[heading.character_end : body_end]):
        return None
    return LitigationRegion(
        kind=kind,
        heading=heading.text,
        heading_start=heading.character_start,
        body_start=heading.character_end,
        body_end=body_end,
        end_basis=basis,
        family=family,
    )


def signposted_sentences(
    paragraph: Paragraph, signpost: re.Pattern[str]
) -> tuple[tuple[int, int], ...]:
    """Locate sentences opened by a matching signpost.

    The sentences of a paragraph whose plain text the signpost matches
    at its opening, as absolute character ranges from the sentence's first
    non-blank character to its end, in order. The event and operations
    inventories read their dated statements with it.
    """
    text = paragraph.text
    ends = [0, *sentence_ends(text)]
    if ends[-1] < len(text):
        ends.append(len(text))
    found: list[tuple[int, int]] = []
    for start, end in pairwise(ends):
        sentence = text[start:end]
        lead = len(sentence) - len(sentence.lstrip())
        if signpost.match(_plain(sentence)) is not None:
            found.append(
                (paragraph.character_start + start + lead, paragraph.character_start + end)
            )
    return tuple(found)


def first_sentence(plain: str) -> str:
    """A plain paragraph's first sentence, or the whole when it has one."""
    ends = sentence_ends(plain)
    return plain[: ends[0]] if ends else plain


def _heading_body_end(structure: DocumentStructure, index: int, closers: set[str]) -> int:
    for later in structure.headings[index + 1 :]:
        if later.kind in closers:
            return int(later.character_start)
    return int(structure.length)


def _region_of(inventory: MatterInventory, heading: StructureHeading) -> LitigationRegion | None:
    for region in inventory.regions:
        if region.heading_start == heading.character_start:
            return region
    return None


def resolve_reference(
    reference: CrossReference, inventory: MatterInventory, structure: DocumentStructure
) -> ReferenceTarget:
    """Where a cross-reference points, by the document's own structure.

    Where a cross-reference points, by the document's own structure: a
    note or item heading (a litigation region when the inventory holds it,
    otherwise the heading and the body it governs), a quoted title that
    names a provisional matter, its group or a sub-heading, or another
    filing. Nothing is searched here.
    """
    target = reference.target
    if reference.kind == "FILING":
        return ReferenceTarget(kind="FILING", label=target, section=reference.section)
    if reference.kind in {"NOTE", "ITEM"}:
        pattern = _NOTE_NUMBER if reference.kind == "NOTE" else _ITEM_NUMBER
        wanted = pattern.search(target)
        if wanted is None:
            return ReferenceTarget(kind="UNLOCATED", label=target)
        number = wanted.group(1).casefold()
        part_wanted = _PART_NUMBER.search(target)
        part: str | None = None
        for index, heading in enumerate(structure.headings):
            if heading.kind == "PART":
                found = _PART_NUMBER.search(heading.text)
                part = None if found is None else found.group(1).upper()
                continue
            if heading.kind != reference.kind or heading.character_start < structure.cover_end:
                continue
            found = pattern.search(heading.text)
            if found is None or found.group(1).casefold() != number:
                continue
            if part_wanted is not None and part != part_wanted.group(1).upper():
                continue
            region = _region_of(inventory, heading)
            if region is not None:
                return ReferenceTarget(kind="REGION", label=region.heading)
            closers = {"PART", "ITEM"} if reference.kind == "ITEM" else {"PART", "ITEM", "NOTE"}
            return ReferenceTarget(
                kind="HEADING",
                label=heading.text[:160],
                character_start=heading.character_end,
                character_end=_heading_body_end(structure, index, closers),
            )
        return ReferenceTarget(kind="UNLOCATED", label=target)
    quoted = _plain(target.strip('"\u201c\u201d')).casefold()
    if not quoted:
        return ReferenceTarget(kind="UNLOCATED", label=target)
    for matter in inventory.matters:
        names = (matter.title, matter.group, *matter.aliases)
        if any(quoted == name.casefold() or quoted in name.casefold() for name in names if name):
            return ReferenceTarget(
                kind="MATTER", label=matter.title[:160], matter_handle=matter.handle
            )
    for index, heading in enumerate(structure.headings):
        if (
            heading.character_start < structure.cover_end
            or quoted not in _plain(heading.text).casefold()
        ):
            continue
        for region in inventory.regions:
            if region.body_start <= heading.character_start < region.body_end:
                return ReferenceTarget(kind="REGION", label=region.heading)
        region = _region_of(inventory, heading)
        if region is not None:
            return ReferenceTarget(kind="REGION", label=region.heading)
        return ReferenceTarget(
            kind="HEADING",
            label=heading.text[:160],
            character_start=heading.character_end,
            character_end=_heading_body_end(structure, index, {"PART", "ITEM", "NOTE"}),
        )
    return ReferenceTarget(kind="UNLOCATED", label=target)


def document_needs(
    text: str,
    structure: DocumentStructure,
    inventory: MatterInventory,
    *,
    document_key: str,
    issuer: str,
    window_bytes: int,
    overlap_bytes: int = 240,
    cue: re.Pattern[str] | None = None,
    refers: re.Pattern[str] | None = None,
    family: str = "LITIGATION",
) -> DocumentNeeds:
    """Every need of one filing's litigation regions, in source order.

    Every need of one filing's litigation regions, in source order: the
    named matters' openings and further parts, the region statements
    (first part by cue, further parts after), each matter's qualifications
    (the region statements that say they apply to the matters), and each
    explicit reference with where it resolved. Deterministic; no model.
    """
    statements = region_statements(text, inventory, cue=cue, refers=refers)
    references = cross_references(text, inventory)
    needs: list[EvidenceNeed] = []
    for matter in inventory.matters:
        if not matter.named:
            continue
        windows = _window_ranges(
            text, matter.ranges, window_bytes=window_bytes, overlap_bytes=overlap_bytes
        )
        for part, (start, end) in enumerate(windows, start=1):
            needs.append(
                EvidenceNeed(
                    document_key=document_key,
                    matter_handle=matter.handle,
                    kind="LEAD" if part == 1 else "CONTINUATION",
                    priority=2 if part == 1 else 3,
                    character_start=start,
                    character_end=end,
                    source_bytes=len(text[start:end].encode("utf-8")),
                    part=part,
                    part_count=len(windows),
                    region_heading=matter.region_heading,
                    detail=f"{matter.basis}: {matter.title[:80]}",
                    family=family,
                    disclosure=matter.handle,
                )
            )
        for statement in statements:
            if statement.region_heading == matter.region_heading and statement.refers_to_matters:
                needs.append(
                    EvidenceNeed(
                        document_key=document_key,
                        matter_handle=matter.handle,
                        kind="QUALIFICATION",
                        priority=0,
                        character_start=statement.character_start,
                        character_end=statement.character_end,
                        source_bytes=0,
                        part=1,
                        part_count=1,
                        region_heading=matter.region_heading,
                        detail=(
                            f"{statement.position.lower()} statement of the region: "
                            f"'{statement.reference_text[:60]}'"
                        ),
                        refers_to_matters=True,
                        family=family,
                        disclosure=matter.handle,
                    )
                )
    # A region's qualifications of its matters as a body are its first
    # lead-in and its last closing that say so; a statement between two
    # matters qualifies its neighbours and is read with them.
    opening: dict[str, RegionStatement] = {}
    closing: dict[str, RegionStatement] = {}
    for statement in statements:
        if not statement.refers_to_matters:
            continue
        if statement.position == "LEAD_IN":
            opening.setdefault(statement.region_heading, statement)
        elif statement.position == "TRAILING":
            closing[statement.region_heading] = statement
    region_level = {id(s) for s in (*opening.values(), *closing.values())}
    for statement in statements:
        windows = _window_ranges(
            text,
            ((statement.character_start, statement.character_end),),
            window_bytes=window_bytes,
            overlap_bytes=overlap_bytes,
        )
        for part, (start, end) in enumerate(windows, start=1):
            if part > 1:
                priority = 3
            elif statement.position == "WHOLE" or id(statement) in region_level:
                priority = 1
            elif statement.refers_to_matters:
                priority = 3
            else:
                priority = 4 if statement.cued else 6
            scope = (
                f"refers to the matters: '{statement.reference_text[:60]}'"
                if statement.refers_to_matters
                else "scope not stated"
            )
            needs.append(
                EvidenceNeed(
                    document_key=document_key,
                    matter_handle=None,
                    kind="REGION_STATEMENT" if part == 1 else "CONTINUATION",
                    priority=priority,
                    character_start=start,
                    character_end=end,
                    source_bytes=len(text[start:end].encode("utf-8")),
                    part=part,
                    part_count=len(windows),
                    region_heading=statement.region_heading,
                    detail=(
                        f"{statement.position.lower()} statement"
                        + (f" under '{statement.group[:40]}'" if statement.group else "")
                        + f"; {scope}"
                    ),
                    refers_to_matters=statement.refers_to_matters,
                    family=family,
                    disclosure=f"S:{statement.character_start}",
                )
            )
    seen: set[tuple[str, str]] = set()
    for reference in references:
        key = (reference.region_heading, reference.target.casefold())
        if key in seen:
            continue
        seen.add(key)
        target = resolve_reference(reference, inventory, structure)
        window: tuple[int, int] | None = None
        if (
            target.kind == "HEADING"
            and target.character_start is not None
            and target.character_end is not None
            and _plain(text[target.character_start : target.character_end])
        ):
            first = _window_ranges(
                text,
                ((target.character_start, target.character_end),),
                window_bytes=window_bytes,
                overlap_bytes=overlap_bytes,
            )
            window = first[0]
        needs.append(
            EvidenceNeed(
                document_key=document_key,
                matter_handle=None,
                kind="REFERENCE",
                priority=5 if window is not None else 0,
                character_start=None if window is None else window[0],
                character_end=None if window is None else window[1],
                source_bytes=0
                if window is None
                else len(text[window[0] : window[1]].encode("utf-8")),
                part=1,
                part_count=1,
                region_heading=reference.region_heading,
                detail=f"{reference.kind.lower()} reference '{reference.target[:40]}' -> "
                f"{target.kind.lower()}: {target.label[:40]}"
                + (f" [{target.section[:40]}]" if target.section else ""),
                reference=reference,
                target=target,
                family=family,
                disclosure=f"R:{reference.character_start}",
            )
        )
    order = {need: index for index, need in enumerate(needs)}
    needs.sort(
        key=lambda need: (
            need.character_start if need.character_start is not None else -1,
            order[need],
        )
    )
    return DocumentNeeds(
        document_key=document_key,
        issuer=issuer,
        text=text,
        inventory=inventory,
        statements=statements,
        references=references,
        needs=tuple(needs),
    )


def combine_needs(first: DocumentNeeds, second: DocumentNeeds, *, rules_id: str) -> DocumentNeeds:
    """One document's needs from two inventories (two families) as one.

    One document's needs from two inventories (two families) as one: the
    regions, units, unassigned text, statements, references and needs of
    both, the needs in source order, under the combined rules id. The
    allocation then deals both families' needs through one queue and one
    allowance, and an excerpt is attributed to whichever units it carries.
    """
    if (first.document_key, first.issuer) != (second.document_key, second.issuer):
        raise ValueError("alternative_evidence.needs_document_mismatch")
    inventory = MatterInventory(
        rules_id=rules_id,
        regions=(*first.inventory.regions, *second.inventory.regions),
        matters=(*first.inventory.matters, *second.inventory.matters),
        unassigned=tuple(sorted((*first.inventory.unassigned, *second.inventory.unassigned))),
    )
    needs = [*first.needs, *second.needs]
    order = {id(need): index for index, need in enumerate(needs)}
    needs.sort(
        key=lambda need: (
            need.character_start if need.character_start is not None else -1,
            order[id(need)],
        )
    )
    return DocumentNeeds(
        document_key=first.document_key,
        issuer=first.issuer,
        text=first.text,
        inventory=inventory,
        statements=tuple(
            sorted((*first.statements, *second.statements), key=lambda s: s.character_start)
        ),
        references=tuple(
            sorted((*first.references, *second.references), key=lambda r: r.character_start)
        ),
        needs=tuple(needs),
    )


@dataclass(frozen=True, slots=True)
class AllocatedWindow:
    """One window the allocation chose.

    One window the allocation chose: an exact range of one document and
    the needs it serves, in the order they joined it. A window serving two
    needs is one excerpt read once.
    """

    document_key: str
    character_start: int
    character_end: int
    source_bytes: int
    needs: tuple[EvidenceNeed, ...]


@dataclass(frozen=True, slots=True)
class Allocation:
    """Record admitted and refused litigation reading windows."""

    rules_id: str
    windows: tuple[AllocatedWindow, ...]
    """In the order the windows were opened."""
    refused: tuple[tuple[EvidenceNeed, str], ...]
    """Every need with a window of its own the allocation could not admit,
    with why (`window allowance` or `byte budget`), in the order refused."""
    admitted: tuple[EvidenceNeed, ...] = ()
    """Every need admitted, in the order the allocation dealt them: the
    declared tie-break made visible."""
    served_by_repeat: tuple[tuple[EvidenceNeed, EvidenceNeed], ...] = ()
    """`(need, representative)`: needs passed over because the later unit
    whose text they restate word for word is admitted in this walk or
    delivered in the chain -- served by that reading, not read again."""

    @property
    def source_bytes(self) -> int:
        """Count source bytes across allocated windows."""
        return sum(window.source_bytes for window in self.windows)


OPENING_CLASSES = (2, 4, 5, 6)
"""The classes whose first part opens a disclosure, in the order a lane
opens them: unit openings (and the whole disclosure of a filing naming no
unit), cued region statements, located reference targets, uncued region
text."""


@dataclass(slots=True)
class _Lane:
    """One issuer's needs of one family.

    One issuer's needs of one family: the openings in the order the lane
    opens them, and the completions -- qualifications and further parts --
    each eligible once what it completes has been opened.
    """

    openings: list[EvidenceNeed]
    completions: list[EvidenceNeed]
    phase: str = "OPEN"

    def _eligible(self, started: set[str]) -> int | None:
        return next(
            (i for i, need in enumerate(self.completions) if _completes(need) in started), None
        )

    def has_work(self, started: set[str]) -> bool:
        return bool(self.openings) or self._eligible(started) is not None

    def take(self, started: set[str]) -> EvidenceNeed:
        completion = self._eligible(started)
        opening = self.openings[0] if self.openings else None
        if self.phase == "OPEN":
            if opening is not None:
                after = started | _opens(opening)
                if any(_completes(need) in after for need in self.completions):
                    self.phase = "EXTEND"
                return self.openings.pop(0)
            assert completion is not None
            return self.completions.pop(completion)
        if completion is not None:
            if opening is not None:
                self.phase = "OPEN"
            return self.completions.pop(completion)
        assert opening is not None
        return self.openings.pop(0)


def _region_key(need: EvidenceNeed) -> str:
    return f"REGION:{need.document_key}:{need.region_heading}"


def _disclosure_key(need: EvidenceNeed) -> str:
    """A disclosure's progress key, qualified by its document.

    A disclosure's progress key, qualified by its document: a region
    statement's key is its own start offset (`S:<start>`), and two filings
    of two issuers can state one at the same offset -- unqualified, one
    filing's opening marked the other's statement started and its
    continuation was dealt before its own opening (the lead's finding of
    2026-09-19, reproduced at this boundary). A unit handle is already
    document-unique; the qualification is uniform so no key is guessed.
    """
    return f"{need.document_key}|{need.disclosure}"


def _opens(need: EvidenceNeed) -> set[str]:
    """What serving this opening starts.

    What serving this opening starts: its disclosure, and its region --
    both qualified by the document they belong to.
    """
    return {_disclosure_key(need), _region_key(need)}


def _completes(need: EvidenceNeed) -> str:
    """What a completion waits for.

    What a completion waits for: a qualification waits for a unit of its
    region; a further part waits for its own disclosure's opening.
    """
    return _region_key(need) if need.part == 1 else _disclosure_key(need)


def service_order(
    documents: tuple[DocumentNeeds, ...],
    *,
    lane_of: Callable[[EvidenceNeed], str],
    lanes_in_order: tuple[str, ...],
    recency: Mapping[str, int] | None = None,
) -> tuple[EvidenceNeed, ...]:
    """`TOPIC_LANES_ALLOCATION_ID`'s sealed service order over every need with a window and a class.

    `TOPIC_LANES_ALLOCATION_ID`'s sealed service order over every need
    with a window and a class: a function of the needs alone (see the
    rule's statement), so a session and every continuation after it walk
    the same order. `lane_of`, `lanes_in_order` and `recency` are the
    routing owner's topic of each need, the topics' order and each
    filing's rank latest first (the set's order when absent): the lanes
    are (issuer, topic) and a lane's openings of one class are dealt
    latest filing first.
    """
    issuers: list[str] = []
    lanes: dict[tuple[str, str], _Lane] = {}
    families = lanes_in_order
    filing_index: dict[str, int] = {}
    unit_regions: set[str] = set()
    for document in documents:
        for need in document.needs:
            if need.kind == "LEAD" and need.window is not None:
                unit_regions.add(_region_key(need))
    for index, document in enumerate(documents):
        filing_index[document.document_key] = index
        if document.issuer not in issuers:
            issuers.append(document.issuer)
        for need in document.needs:
            if need.priority == 0 or need.window is None:
                continue
            lane_name = lane_of(need)
            lane = lanes.setdefault((document.issuer, lane_name), _Lane([], []))
            qualifies = (
                need.priority == 1 and need.refers_to_matters and _region_key(need) in unit_regions
            )
            if need.part > 1 or qualifies:
                lane.completions.append(need)
            else:
                lane.openings.append(need)

    def position(need: EvidenceNeed) -> tuple[int, int, int]:
        return (
            (
                filing_index[need.document_key]
                if recency is None
                else recency.get(need.document_key, filing_index[need.document_key])
            ),
            need.character_start if need.character_start is not None else -1,
            need.part,
        )

    for lane in lanes.values():
        lane.openings.sort(
            key=lambda n: (
                OPENING_CLASSES.index(n.priority) if n.priority in OPENING_CLASSES else 0,
                *position(n),
            )
        )
        lane.completions.sort(key=position)
    by_issuer: dict[str, list[_Lane]] = {
        issuer: [lanes[(issuer, family)] for family in families if (issuer, family) in lanes]
        for issuer in issuers
    }
    turn = dict.fromkeys(issuers, 0)
    started: set[str] = set()
    order: list[EvidenceNeed] = []
    while True:
        served_any = False
        for issuer in issuers:
            members = by_issuer[issuer]
            for _attempt in range(len(members)):
                lane = members[turn[issuer] % len(members)]
                turn[issuer] += 1
                if not lane.has_work(started):
                    continue
                need = lane.take(started)
                order.append(need)
                if need.part == 1:
                    started |= _opens(need)
                served_any = True
                break
        if not served_any:
            break
    return tuple(order)


def allocate_windows(
    documents: tuple[DocumentNeeds, ...],
    *,
    window_allowance: int,
    window_bytes: int,
    byte_budget: int | None = None,
    delivered: Callable[[EvidenceNeed], bool] | None = None,
    lane_of: Callable[[EvidenceNeed], str],
    lanes_in_order: tuple[str, ...],
    recency: Mapping[str, int] | None = None,
    representative_of: Callable[[EvidenceNeed], EvidenceNeed | None] | None = None,
) -> Allocation:
    """`TOPIC_LANES_ALLOCATION_ID`.

    `TOPIC_LANES_ALLOCATION_ID`: walk the sealed service order of every
    need of `documents`; a need `delivered` earlier in the chain is passed
    over (it still counts as this chain's progress: its disclosure is
    started, and it is not dealt again); a need whose range touches a
    window already open in its document joins that window when the union
    fits `window_bytes`, otherwise it opens a window of its own while the
    allowance and the budget admit one; an extension of a disclosure whose
    opening this chain has not served is refused by name. What is refused
    stays listed. With `representative_of`, a need whose unit an exact repeat restates is
    passed over once its representative is admitted in this walk or
    delivered in the chain, and refused as "repeat pending" until then.
    """
    by_key = {document.document_key: document for document in documents}
    opened: list[_OpenWindow] = []
    refused: list[tuple[EvidenceNeed, str]] = []
    admitted: list[EvidenceNeed] = []
    served: list[tuple[EvidenceNeed, EvidenceNeed]] = []
    admitted_keys: set[tuple[str, int | None, int | None, str]] = set()
    started: set[str] = set()
    spent = 0

    def admit(document: DocumentNeeds, need: EvidenceNeed) -> bool:
        nonlocal spent
        window = need.window
        if window is None:
            return False
        start, end = window
        text = document.text
        for entry in opened:
            if entry.document_key != document.document_key:
                continue
            w_start, w_end = entry.character_start, entry.character_end
            touching = (
                (start <= w_end and end >= w_start)
                or (start >= w_end and not text[w_end:start].strip())
                or (end <= w_start and not text[end:w_start].strip())
            )
            if not touching:
                continue
            merged_start, merged_end = min(start, w_start), max(end, w_end)
            merged_bytes = len(text[merged_start:merged_end].encode("utf-8"))
            if merged_bytes > window_bytes:
                continue
            delta = merged_bytes - entry.source_bytes
            if byte_budget is not None and spent + delta > byte_budget:
                refused.append((need, "byte budget"))
                return False
            entry.character_start, entry.character_end = merged_start, merged_end
            entry.source_bytes = merged_bytes
            entry.needs = (*entry.needs, need)
            admitted.append(need)
            spent += delta
            return True
        if len(opened) >= window_allowance:
            refused.append((need, "window allowance"))
            return False
        if byte_budget is not None and spent + need.source_bytes > byte_budget:
            refused.append((need, "byte budget"))
            return False
        opened.append(_OpenWindow(document.document_key, start, end, need.source_bytes, (need,)))
        admitted.append(need)
        spent += need.source_bytes
        return True

    for need in service_order(
        documents, lane_of=lane_of, lanes_in_order=lanes_in_order, recency=recency
    ):
        if delivered is not None and delivered(need):
            if need.part == 1:
                started |= _opens(need)
            continue
        representative = None if representative_of is None else representative_of(need)
        if representative is not None:
            key = (
                representative.document_key,
                representative.character_start,
                representative.character_end,
                representative.kind,
            )
            if key in admitted_keys or (delivered is not None and delivered(representative)):
                served.append((need, representative))
                if need.part == 1:
                    started |= _opens(need)
            else:
                refused.append((need, "repeat pending"))
            continue
        if need.part > 1 and _disclosure_key(need) not in started:
            # An extension of a disclosure this chain has not opened: the
            # allowance is the reason once it is spent, its opening before.
            why = "window allowance" if len(opened) >= window_allowance else "opening pending"
            refused.append((need, why))
            continue
        if admit(by_key[need.document_key], need):
            admitted_keys.add(
                (need.document_key, need.character_start, need.character_end, need.kind)
            )
            if need.part == 1:
                started |= _opens(need)
    return Allocation(
        rules_id=TOPIC_LANES_ALLOCATION_ID,
        windows=tuple(
            AllocatedWindow(
                document_key=entry.document_key,
                character_start=entry.character_start,
                character_end=entry.character_end,
                source_bytes=entry.source_bytes,
                needs=entry.needs,
            )
            for entry in opened
        ),
        refused=tuple(refused),
        admitted=tuple(admitted),
        served_by_repeat=tuple(served),
    )


@dataclass(slots=True)
class _OpenWindow:
    document_key: str
    character_start: int
    character_end: int
    source_bytes: int
    needs: tuple[EvidenceNeed, ...]


@dataclass(frozen=True, slots=True)
class Attribution:
    """What a delivered excerpt carries at one exact range.

    What a delivered excerpt carries at one exact range: a named matter's
    opening or a later part of it, or a region-level statement with its
    scope. The excerpt is transport; this is the claim.
    """

    matter_handle: str | None
    role: str
    scope: str
    basis: str
    region_heading: str
    character_start: int
    character_end: int


def attribute_range(
    document: DocumentNeeds, *, delivered: tuple[int, int], planned: tuple[int, int]
) -> tuple[Attribution, ...]:
    """The attributions of one delivered excerpt.

    The attributions of one delivered excerpt: its exact intersections
    with every named matter's text and every region statement, each saying
    whether the planned window held it or only the reader's growth to
    sentence boundaries did. Two matters in one excerpt are two
    attributions at two ranges; a matter's statement is never carried under
    another matter's handle.
    """
    text = document.text
    d_start, d_end = delivered
    out: list[Attribution] = []

    def basis_of(start: int, end: int) -> str:
        return "planned window" if planned[0] <= start and end <= planned[1] else "reader expansion"

    for matter in document.named_matters:
        m_start, m_end = matter.ranges[0][0], matter.ranges[-1][1]
        start, end = max(m_start, d_start), min(m_end, d_end)
        if end <= start or not text[start:end].strip():
            continue
        out.append(
            Attribution(
                matter_handle=matter.handle,
                role="CORE" if start == m_start else "CONTINUATION",
                scope="MATTER",
                basis=basis_of(start, end),
                region_heading=matter.region_heading,
                character_start=start,
                character_end=end,
            )
        )
    for statement in document.statements:
        start = max(statement.character_start, d_start)
        end = min(statement.character_end, d_end)
        if end <= start or not text[start:end].strip():
            continue
        scope_basis = (
            f"refers to the matters: '{statement.reference_text[:50]}'"
            if statement.refers_to_matters
            else "scope not stated"
        )
        out.append(
            Attribution(
                matter_handle=None,
                role="REGION_STATEMENT",
                scope=statement.scope,
                basis=(
                    f"{basis_of(start, end)}; {statement.position.lower()} statement"
                    + (f" under '{statement.group[:30]}'" if statement.group else "")
                    + f"; {scope_basis}"
                )[:160],
                region_heading=statement.region_heading,
                character_start=start,
                character_end=end,
            )
        )
    out.sort(key=lambda value: (value.character_start, value.character_end))
    return tuple(out)


__all__ = [
    "MATTER_INVENTORY_RULES_ID",
    "MATTER_SEGMENTATION_RULES_ID",
    "TOPIC_LANES_ALLOCATION_ID",
    "AllocatedWindow",
    "Allocation",
    "Attribution",
    "CrossReference",
    "DocumentNeeds",
    "EvidenceNeed",
    "LitigationRegion",
    "MatterInventory",
    "Paragraph",
    "ProvisionalMatter",
    "ReferenceTarget",
    "RegionStatement",
    "allocate_windows",
    "attribute_range",
    "combine_needs",
    "cross_references",
    "document_needs",
    "litigation_regions",
    "matter_inventory",
    "reference_target_state",
    "region_statements",
    "required_reference_needs",
    "resolve_reference",
    "search_hit_corresponds",
    "source_blocks",
]
