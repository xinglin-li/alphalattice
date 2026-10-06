"""Inventory financing disclosures in filings.

Financing-disclosure units: a provisional inventory of what a filing's
debt, borrowings, credit-facility and financing notes state about named
instruments, programs and facilities.

The frozen definition (`fd-financing-family-definition.json`, dispatch
FINANCING_DISCLOSURE_DISCOVERY_AND_PRODUCT_DELIVERY, 2026-09-20) came from
the source-first audit: ten development units in the debt and borrowings
notes -- commercial paper balances and their explicit absence, unused and
undrawn capacities, a program's dated enlargement, notes issued and repaid,
a facility matured -- were held and represented and never inventoried. This
owner gives them the litigation inventory's shape (regions from the
filing's own headings, units the source's own paragraphs open, region-level
statements, needs, allocation, attribution) with the family's own
signposts: an instrument phrase as written. It parses no amount, rate,
maturity, balance or covenant into a field, converts no capacity into a
balance and no authorization into a borrowing, closes no facility on a
repayment, infers no date from a period, and asserts no identity across
filings: the unit is the source's paragraph with the instruments it names,
delivered whole with its qualifications.

Consumer: `packet.select_matter_evidence` with `families` naming
`FINANCING`, which merges this inventory's needs with the other families'
under the one matter allowance; the development comparison drivers. No
model, no query; `retrieval/session.py` issues and proves the ranges.
"""

from __future__ import annotations

import re

from ..documents.structure import DocumentStructure
from .matters import (
    LitigationRegion,
    MatterInventory,
    ProvisionalMatter,
    _plain,
    bounded_region,
    source_blocks,
)

FINANCING_INVENTORY_RULES_ID = "alternative-evidence.financing-disclosure-inventory.v3"
"""v3 (2026-09-22): a clause that names its instrument by a pronoun ('the
Company borrowed $200 million under it'; 'it was used to issue $750 million
of notes') is a mention of that instrument: positive when it states an
amount or a drawing, issuance, repayment or outstanding balance without an
absence, an absence when it states one ('no amounts had been drawn under
it'); a pronominal clause stating neither ('It matures in 2028') says
nothing about polarity. v2 skipped every clause without an instrument
phrase, so a paragraph that stated an absence and then a draw under the
same facility read as an explicit negative -- a unit-wide negation the
source did not make (the lead's finding of 2026-09-19).
v2 (2026-09-21): a stated absence has the scope of the clause that states
it. A unit is an `explicit negative` only when every clause naming an
instrument states an absence and nothing contradicts it: no exception
('other than', 'except'), and no amount beside a second instrument in the
clause; a sentence that states an absence for one instrument beside an
amount outstanding under another ("no commercial paper was outstanding,
but $25 million remained outstanding under the revolving credit facility";
"no borrowings under the facility and $25 million of commercial paper
outstanding"), a qualified absence, or an absence beside a positive
mention elsewhere in the paragraph is a `mixed statement`: read as text,
never a unit-wide negative. One instrument's own size or capacity beside
its absence ("a capacity of $25.0 billion, with no amounts outstanding")
is that instrument's absence. v1 accepted any negative phrase in an
instrument-bearing sentence as negating every instrument the sentence
named.
v1 (2026-09-20): a note whose heading names debt, borrowings, financing
arrangements, loans or notes payable, credit facilities or agreements,
long-term debt, short-term borrowings or indebtedness is a region, bounded
by the next note, item or part heading; MD&A, the litigation notes and
current reports are not. Inside a region, every prose paragraph that names
an instrument, program or facility -- commercial paper and its program,
lines and backup lines of credit, credit facilities and agreements,
revolving and term loans, senior, guaranteed, unsecured, convertible or
subordinated notes and notes due a year, debentures, securitization
financings and programs, discount-window capacity -- is one unit: the
paragraph whole, titled by the first instrument as written, every other
instrument it names listed beside it; a paragraph whose every instrument
mention is negated ('no commercial paper was outstanding') is a unit of
basis `explicit negative` (v2 narrows this to the clause's scope). A
sub-heading line names the group of the units
under it; a table placeholder is unassigned as a table not carried; a
paragraph naming no instrument is a region-level statement. Nothing is
parsed into a field, and a dated action is source text inside its unit."""

FINANCING_FAMILY = "FINANCING"

_FINANCING_NOTE = re.compile(
    r"\bdebt\b|borrowing|financing arrangement|loans?(?: and notes)? payable|notes payable|"
    r"credit (?:agreement|facilit)|indebtedness|long-term debt|short-term borrowing",
    re.IGNORECASE,
)
_NOT_FINANCING_NOTE = re.compile(
    r"debt securities|investments?|marketable|available-for-sale|held-to-maturity", re.IGNORECASE
)
"""A note about securities the filer holds is an asset note, not a
borrowing: `DEBT SECURITIES` names no financing of the filer's own."""
_QUALIFIER = (
    r"(?:senior|guaranteed|unsecured|secured|convertible|subordinated|exchangeable|"
    r"fixed-to-floating rate|floating[- ]rate|fixed[- ]rate|term)"
)
_INSTRUMENT = re.compile(
    r"\bcommercial paper(?: program| borrowings| notes)?\b"
    r"|\b(?:corporate )?backup lines? of credit\b"
    r"|\b(?:unused |undrawn |committed |uncommitted )?lines? of credit\b"
    r"|\b(?:revolving |unsecured |secured |senior )*(?:credit (?:agreement|facility|facilities)|"
    r"loan (?:facility|facilities|agreement))\b"
    r"|\bterm loans?\b"
    rf"|\b{_QUALIFIER}\s+(?:{_QUALIFIER}\s+)*"
    r"(?:notes|debentures)"
    r"(?! to (?:the |our )?(?:condensed |consolidated |unaudited )*financial statements)\b"
    r"(?:\s+due\s+(?:\w+\s+)?\d{4})?"
    r"|\b\d+(?:\.\d+)?\s?%\s+(?:\w+\s+){0,4}?(?:notes|debentures)(?:\s+due\s+(?:\w+\s+)?\d{4})?\b"
    r"|\b(?:notes|debentures) due (?:\w+\s+)?\d{4}\b"
    r"|\b(?:19|20)\d\d\s+(?:Senior\s+)?Notes\b"
    r"|\bsecuritiz(?:ation|ed) (?:financings?|programs?|borrowings?|entities)\b"
    r"|\bdiscount window\b",
    re.IGNORECASE,
)
"""An instrument, program or facility as the source names it: the family's
signpost. A bare reference (`our notes`, `the outstanding notes`, `these
facilities`) names nothing and stays a region-level statement that refers
to the region's instruments. What a paragraph names is listed as written;
nothing is parsed."""
_NEGATED = re.compile(
    r"\bno\s+(?:outstanding\s+)?(?:\w+\s+){0,4}?(?:outstanding|borrowings?|amounts?|balances?)\b"
    r"|\bnone\s+(?:was|were)\s+outstanding\b"
    r"|\bno\s+(?:amounts?|borrowings?)\s+(?:were|was)?\s*outstanding\b"
    r"|\bno\s+borrowings\s+under\b"
    r"|\bhad\s+no\s+(?:outstanding\s+)?(?:\w+\s+){0,3}?(?:borrowings|balances?|amounts?)\b"
    r"|\bwith\s+no\s+amounts?\s+outstanding\b",
    re.IGNORECASE,
)
"""A stated absence: the sentence says nothing is outstanding or borrowed."""
_CLAUSE_BREAK = re.compile(
    r";\s+|,?\s+(?:but|however|while|whereas|although|though)\s+", re.IGNORECASE
)
"""Where one sentence turns to another assertion: the absence stated before
the turn says nothing about what follows it."""
_EXCEPTION = re.compile(
    r"\b(?:other than|except(?:ing)?(?: for)?|excluding|aside from|apart from|net of)\b",
    re.IGNORECASE,
)
"""An absence with an exception is not a whole absence."""
_AMOUNT_BESIDE = re.compile(
    r"\$\s?\d|\b\d[\d,]*(?:\.\d+)?\s*(?:million|billion|thousand)\b|\b\d+(?:\.\d+)?\s?%"
)
"""An amount beside an absence in a clause naming two instruments: the
clause asserts more than the absence, and which instrument the amount
belongs to is the source's to say, not a unit-wide negative's. Beside one
instrument the amount is that instrument's own size or capacity."""
_PRONOMINAL = re.compile(
    r"\b(?:under|of|on|from|through|against)\s+(?:it|them)\b|\bit\s+(?:was|is|has|had|will|may)\b"
    r"|\bthereunder\b|\bits\s+(?:availability|capacity|proceeds|maturity)\b",
    re.IGNORECASE,
)
"""A clause that names the instrument of a neighbouring clause by a pronoun
('under it', 'it was used', 'thereunder'): a mention of that instrument,
read for the polarity it states."""
_POSITIVE_STATE = re.compile(
    r"\bborrow(?:ed|ings?)?\b|\bdr(?:ew|awn|aws?)\b|\bissu(?:ed|es|ance)\b|\brepaid\b"
    r"|\brepay(?:ment|ments)?\b|\boutstanding\b|\butili[sz]ed\b|\bused\s+to\b|\bfunded\b",
    re.IGNORECASE,
)
"""What a pronominal clause must state, with or without an amount, to be a
positive mention: a drawing, an issuance, a repayment or a balance."""
_FINANCING_CUE = re.compile(
    r"\b(?:covenants?|compl(?:y|ied|iance)|fair value|interest paid|maturit(?:y|ies)|"
    r"outstanding|borrowings?|redeem|redemption|indentures?|facility|facilities|notes|"
    r"debentures|commercial paper|lines? of credit|principal|repay|repaid|issued|issuance|"
    r"liquidity|capacity)\b",
    re.IGNORECASE,
)
"""The words a financing disclosure is described with; a region paragraph
that carries none is not read first."""
_REFERS_TO_FINANCING = re.compile(
    r"\b(?:the|these|those|such|our|its)\s+(?:outstanding\s+|above\s+)?"
    r"(?:borrowings|facilities|credit facilities|instruments|debt|indebtedness|"
    r"financial arrangements|financing arrangements|arrangements|notes and debentures)\b",
    re.IGNORECASE,
)
"""A region-level statement that says it applies to the region's instruments
as a body ('these credit facilities', 'our borrowings', 'the above notes and
debentures'). A reference to one kind alone ('our notes', 'the outstanding
notes') names no unit -- the notes it means may live only in a table the
canonical text does not carry -- so it stays a cued statement of the region
with its scope not stated, never a qualification the source did not make."""


def financing_regions(text: str, structure: DocumentStructure) -> tuple[LitigationRegion, ...]:
    """The regions the financing inventory reads.

    The regions the financing inventory reads: in a periodic report, every
    note whose heading names the family, bounded by the next note, item or
    part heading. A current report has none.
    """
    if structure.family_form.upper() == "8-K":
        return ()
    regions: list[LitigationRegion] = []
    headings = structure.headings
    for index, heading in enumerate(headings):
        if heading.character_start < structure.cover_end or heading.kind != "NOTE":
            continue
        if (
            _FINANCING_NOTE.search(heading.text) is None
            or _NOT_FINANCING_NOTE.search(heading.text) is not None
        ):
            continue
        region = bounded_region(
            text,
            headings,
            index,
            closers={"PART", "ITEM", "NOTE"},
            kind="NOTE",
            family=FINANCING_FAMILY,
        )
        if region is not None:
            regions.append(region)
    return tuple(regions)


def _instruments(plain: str) -> tuple[str, ...]:
    """List distinct instrument phrases in source order.

    Every instrument phrase a paragraph names, as written, first
    occurrence first, without repeats (case-insensitively).
    """
    found: list[str] = []
    seen: set[str] = set()
    for match in _INSTRUMENT.finditer(plain):
        phrase = " ".join(match.group(0).split())
        key = phrase.casefold()
        if key in seen:
            continue
        seen.add(key)
        found.append(phrase)
    return tuple(found)


def _sentences(plain: str) -> list[str]:
    return [piece.strip() for piece in re.split(r"(?<=[.;:])\s+", plain) if piece.strip()]


def _clauses(sentence: str) -> list[str]:
    return [piece.strip() for piece in _CLAUSE_BREAK.split(sentence) if piece.strip()]


def _polarity_basis(plain: str) -> str:
    """The unit's basis from the polarity of its instrument mentions, clause by clause.

    The unit's basis from the polarity of its instrument mentions, clause
    by clause: `explicit negative` when every clause naming an instrument
    states an absence with no exception and no amount beside a second
    instrument; `mixed statement` when an absence is stated under an
    exception, beside an amount and another instrument in its clause, or
    beside a clause that names an instrument without one; `instrument
    paragraph` when no absence is stated. The scope of an absence is the
    clause that states it: what follows a turn ('but', ';') is another
    assertion, so one instrument's absence negates no other. A clause that
    names its instrument by a pronoun is read for the polarity it states
    (`_PRONOMINAL`, `_POSITIVE_STATE`); one stating neither is not read.
    """
    absent = positive = unproved = 0
    for sentence in _sentences(plain):
        for clause in _clauses(sentence):
            instruments = len({m.group(0).casefold() for m in _INSTRUMENT.finditer(clause)})
            if not instruments:
                # A pronominal mention of the instrument a neighbouring
                # clause named: its polarity counts as that instrument's.
                if _PRONOMINAL.search(clause) is None:
                    continue
                if _NEGATED.search(clause) is not None:
                    absent += 1
                elif _AMOUNT_BESIDE.search(clause) is not None or _POSITIVE_STATE.search(clause):
                    positive += 1
                continue
            if _NEGATED.search(clause) is None:
                positive += 1
            elif _EXCEPTION.search(clause) is not None or (
                instruments > 1 and _AMOUNT_BESIDE.search(clause) is not None
            ):
                unproved += 1
            else:
                absent += 1
    if absent == 0 and unproved == 0:
        return "instrument paragraph"
    if positive == 0 and unproved == 0:
        return "explicit negative"
    return "mixed statement"


def financing_inventory(
    text: str, structure: DocumentStructure, *, document_id: str = "DOC"
) -> MatterInventory:
    """Inventory provisional financing units and unassigned text.

    The provisional financing units of one canonical text, with the
    unassigned remainder of the financing regions. Deterministic; no model.
    Unit handles are `M-<document>-F-<n>`: the same namespace the packet's
    unit records carry, in the family's own series.
    """
    regions = financing_regions(text, structure)
    units: list[ProvisionalMatter] = []
    unassigned: list[tuple[int, int, str]] = []
    counter = 0
    for region in regions:
        group = ""
        for paragraph, kind in source_blocks(
            text, region, structure, signpost=lambda plain: _INSTRUMENT.search(plain) is not None
        ):
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
            instruments = _instruments(plain)
            if not instruments:
                # A region-level statement or uncued text: `region_statements`
                # reads it from the unassigned ranges with its position and scope.
                unassigned.append((*span, "no instrument named"))
                continue
            counter += 1
            body = text[span[0] : span[1]]
            units.append(
                ProvisionalMatter(
                    handle=f"M-{document_id}-F-{counter:03d}",
                    title=instruments[0][:120],
                    basis=_polarity_basis(plain),
                    region_heading=region.heading,
                    group=group or region.heading,
                    ranges=(span,),
                    aliases=tuple(value[:80] for value in instruments[:12]),
                    source_bytes=len(body.encode("utf-8")),
                    family=FINANCING_FAMILY,
                )
            )
    return MatterInventory(
        rules_id=FINANCING_INVENTORY_RULES_ID,
        regions=regions,
        matters=tuple(units),
        unassigned=tuple(sorted(unassigned)),
    )


FINANCING_CUE = _FINANCING_CUE
REFERS_TO_FINANCING = _REFERS_TO_FINANCING

__all__ = [
    "FINANCING_CUE",
    "FINANCING_FAMILY",
    "FINANCING_INVENTORY_RULES_ID",
    "REFERS_TO_FINANCING",
    "financing_inventory",
    "financing_regions",
]
