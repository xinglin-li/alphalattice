"""Locate parts, items, notes, and headings in canonical filings.

Structure of a canonical filing, read from its own text: parts, items,
notes and sub-headings, the scale statements bound to their note or page,
and the navigational family each source window falls in.

The canonical text of a recorded filing carries no Markdown headings: a
heading is a short line of its own -- ``PART I. FINANCIAL INFORMATION``,
``ITEM 1. LEGAL PROCEEDINGS``, ``5. DEBT``, ``A. ACCOUNTS RECEIVABLE``, a
capitalised title -- and a page repeats its running header (part, registrant,
``NOTES TO ... (CONTINUED)``, ``(In thousands, ...)``) at every page break.
What this owner proves is where those lines sit (exact character ranges in
the text they were read from) and which of them govern a given window. It
does not read the window's meaning.

Families are a navigational taxonomy over filing responsibilities, declared
once below, not a list of answers. A window whose path maps to none of them
is ``OTHER_UNMAPPED``: an explicit state with its own opportunity, never a
silent loss. Front matter, contents pages, signatures and exhibit indexes
are ``FRONT_MATTER``, also explicit, and not evidence.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
from datetime import date

FAMILY_OPERATIONS = "OPERATIONS_RESULTS_OUTLOOK"
FAMILY_FINANCING = "FINANCING_CAPITAL"
FAMILY_TRANSACTIONS = "TRANSACTIONS"
FAMILY_LEGAL = "LEGAL_REGULATORY_TAX"
FAMILY_CONTROLS = "CONTROLS_GOVERNANCE_INSIDER"
FAMILY_COMMITMENTS = "COMMITMENTS_RESTRUCTURING_PENSIONS"
FAMILY_SUBSEQUENT = "SUBSEQUENT_EVENTS_AMENDMENTS"
FAMILY_RISK = "RISK_FACTORS"
FAMILY_STATEMENTS = "FINANCIAL_STATEMENTS"
FAMILY_FRONT_MATTER = "FRONT_MATTER"
FAMILY_OTHER = "OTHER_UNMAPPED"

FAMILIES: tuple[str, ...] = (
    FAMILY_OPERATIONS,
    FAMILY_FINANCING,
    FAMILY_TRANSACTIONS,
    FAMILY_LEGAL,
    FAMILY_CONTROLS,
    FAMILY_COMMITMENTS,
    FAMILY_SUBSEQUENT,
    FAMILY_RISK,
    FAMILY_STATEMENTS,
    FAMILY_OTHER,
    FAMILY_FRONT_MATTER,
)
"""Declared order: the coverage pass visits an issuer's families in this
order, so two runs over the same generation choose the same opportunities."""

EVIDENCE_FAMILIES: frozenset[str] = frozenset(FAMILIES) - {FAMILY_FRONT_MATTER}
"""Every family that may hold evidence; front matter is inventoried and
reported, never selected."""

STRUCTURE_RULES_ID = "alternative-evidence.structure-rules.v4"
"""v4 (2026-09-22): a periodic report's period is its cover's statement --
the front matter before the first Part or Item heading -- never a year its
body names in the same words. Under v3 the whole text was read and the
annual form took the latest date: the first-release acceptance found four
10-Ks for years ended 2025 whose typed statements carried report periods
of 2026-12-31, 2027-12-31 and 2028-12-31 (a later fiscal year named in the
body) and 2024-11-02 (a cover the pattern does not match, the body naming
only the predecessor year), stated as resolved. A cover that states no
period is ABSENT; a filing without a Part or Item heading has no bounded
cover and is read whole, as before.

v3 (2026-09-20): two forms of an item heading the retained book holds and
v2 missed. An item line that closes with a parenthetical statement (COST's
MD&A: "Item 7—Management's Discussion and Analysis of Financial Condition
and Results of Operations (amounts in millions, except per share, share,
percentages and warehouse count data)", 176 characters) is the item at the
length of its title, the parenthetical apart -- and, when it is a scale
statement, that statement is the section's unit declaration; under v2 the
line exceeded the heading bound, the MD&A sat under "Item 6—Reserved" and
a liquidity statement was unroutable. A 10-K or 10-Q whose body labels no
item at all and heads each with its official title alone (SYF's 10-K:
"Cybersecurity", "Controls and Procedures", "Other Information", seven of
them) has those titles as its item headings: the first line that is one of
the form's official titles opens that item, its official part implied,
only when the document carries no labelled item heading and at least three
official titles; a filing with labelled items keeps every bare title a
sub-heading (COST's "Legal Proceedings" inside its contingencies note).
The heading records its item number before the source title, at the
source line's exact range. A filing with no Part or Item heading bounds
its cover by its body: a cover-like line after the first substantive
paragraph no longer extends the cover (the recorded DG 8-K: the cover ran
to 4,271 of 4,546 characters and its vote results routed nowhere).
v2 (2026-09-18): the cover page ends inside the front matter, before the
body's first Part or Item heading; a cover-like line later in the text (a
combined registrant's closing statement that no proxy material was sent to
the co-registrant's holders) no longer makes the whole filing front matter.
A note the word Note names by a letter ("NOTE K - COMMITMENTS AND
CONTINGENCIES") is a note heading, so a Legal Proceedings item that refers
to it reaches a region; a bare letter stays a sub-item."""

_DASHES = "\\-\u2013\u2014"
_PART = re.compile(r"^part\s+([ivx]+)\b\.?\s*(.*)$", re.IGNORECASE)
_ITEM = re.compile(rf"^item\s+(\d{{1,2}}[a-c]?)\s*[.:{_DASHES}]?\s*(.*)$", re.IGNORECASE)
_ITEM_8K = re.compile(rf"^item\s+(\d\.\d{{2}})\s*[.:{_DASHES}]?\s*(.*)$", re.IGNORECASE)
_NOTE = re.compile(
    rf"^(?:note\s+(?P<lettered>[A-Z])|(?:note\s+)?(?P<numbered>\d{{1,2}}))"
    rf"\s*[.:{_DASHES}]\s*(?P<title>[A-Za-z].*)$",
    re.IGNORECASE,
)
"""A note heading: numbered ("15. Commitments and Contingencies", "Note 14 -
Contingencies") or, when the word Note names it, lettered ("NOTE K -
COMMITMENTS AND CONTINGENCIES", "Note N: ..."); a bare letter is a sub-item."""
_SUB_LETTER = re.compile(r"^([a-z])\s*[.)]\s+([A-Z].*)$")
_UNIT = re.compile(
    r"^\((?:in|amounts in|dollars in|\$ in|expressed in)\s+(?:thousands|millions|billions)"
    r"[^()\n]{0,120}\)",
    re.IGNORECASE,
)
_CONTINUED = re.compile(r"\(\s*continued\s*\)\s*$", re.IGNORECASE)
_TABLE_OF_CONTENTS = re.compile(r"^(table of contents|index)$", re.IGNORECASE)
_EMPHASIS = re.compile(r"\*+")
_MARKDOWN_HEADING = re.compile(r"^#{1,6}\s+")
_PAGE_NUMBER = re.compile(r"^\d{1,3}$")
_COVER = re.compile(
    r"check mark|emerging growth company|securities registered pursuant|trading symbol|"
    r"written communications pursuant|soliciting material|pre-commencement|"
    r"commission file number|i\.r\.s\. employer|exact name of registrant|"
    r"registrant.s telephone|address of principal executive offices|"
    r"large accelerated filer|shell company",
    re.IGNORECASE,
)
_MAX_HEADING_LENGTH = 160
_BODY_LINE_LENGTH = 200
_TRAILING_PARENTHETICAL = re.compile(r"^(?P<head>.*?\S)\s*(?P<tail>\([^()]{1,200}\))\s*$")
_ITEM_PREFIX = re.compile(r"^item\s+\d", re.IGNORECASE)
_OFFICIAL_ITEM_TITLES: dict[str, tuple[tuple[str, str, str], ...]] = {
    # (item, official part, title as the form spells it), per form.
    "10-K": (
        ("1", "I", "Business"),
        ("1A", "I", "Risk Factors"),
        ("1B", "I", "Unresolved Staff Comments"),
        ("1C", "I", "Cybersecurity"),
        ("2", "I", "Properties"),
        ("3", "I", "Legal Proceedings"),
        ("4", "I", "Mine Safety Disclosures"),
        (
            "5",
            "II",
            "Market for Registrant's Common Equity, Related Stockholder Matters and Issuer "
            "Purchases of Equity Securities",
        ),
        ("6", "II", "Reserved"),
        (
            "7",
            "II",
            "Management's Discussion and Analysis of Financial Condition and Results of Operations",
        ),
        ("7A", "II", "Quantitative and Qualitative Disclosures About Market Risk"),
        ("8", "II", "Financial Statements and Supplementary Data"),
        (
            "9",
            "II",
            "Changes in and Disagreements with Accountants on Accounting and Financial Disclosure",
        ),
        ("9A", "II", "Controls and Procedures"),
        ("9B", "II", "Other Information"),
        ("9C", "II", "Disclosure Regarding Foreign Jurisdictions that Prevent Inspections"),
        ("10", "III", "Directors, Executive Officers and Corporate Governance"),
        ("11", "III", "Executive Compensation"),
        (
            "12",
            "III",
            "Security Ownership of Certain Beneficial Owners and Management and Related "
            "Stockholder Matters",
        ),
        ("13", "III", "Certain Relationships and Related Transactions, and Director Independence"),
        ("14", "III", "Principal Accountant Fees and Services"),
        ("15", "IV", "Exhibits and Financial Statement Schedules"),
        ("16", "IV", "Form 10-K Summary"),
    ),
    "10-Q": (
        ("1", "I", "Financial Statements"),
        (
            "2",
            "I",
            "Management's Discussion and Analysis of Financial Condition and Results of Operations",
        ),
        ("3", "I", "Quantitative and Qualitative Disclosures About Market Risk"),
        ("4", "I", "Controls and Procedures"),
        ("1", "II", "Legal Proceedings"),
        ("1A", "II", "Risk Factors"),
        ("2", "II", "Unregistered Sales of Equity Securities and Use of Proceeds"),
        ("3", "II", "Defaults Upon Senior Securities"),
        ("4", "II", "Mine Safety Disclosures"),
        ("5", "II", "Other Information"),
        ("6", "II", "Exhibits"),
    ),
}
"""The official item titles of the periodic forms (Regulation S-K and the
forms' instructions), matched with punctuation and case set aside, and
the accountant/accounting spelling of Item 14 both admitted."""
_MINIMUM_BARE_TITLES = 3


def _title_key(text: str) -> str:
    key = re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()
    return key.replace("principal accounting fees", "principal accountant fees")


_OFFICIAL_TITLE_KEYS: dict[str, dict[str, tuple[str, str]]] = {
    form: {_title_key(title): (item, part) for item, part, title in titles}
    for form, titles in _OFFICIAL_ITEM_TITLES.items()
}


def _item_head(stripped: str) -> tuple[str, str | None]:
    """Separate an Item title from its closing parenthetical.

    An item line that closes with a parenthetical statement, split: the
    item title without it, and the parenthetical. Any other line whole.
    """
    if _ITEM_PREFIX.match(stripped) is None:
        return stripped, None
    match = _TRAILING_PARENTHETICAL.match(stripped)
    if match is None:
        return stripped, None
    return match.group("head"), match.group("tail")


_MAX_NORMALIZED_HEADING_LENGTH = 120
_MAX_ITEM_8K_HEADING_LENGTH = 180
"""A current report's item heading is the item's official title, and the
longest of them (5.02, officer and director changes) runs to 156
characters after the item number; measured on the retained book, five of
the sixty-nine item lines of thirty-six 8-Ks (three 5.02, two 2.03) exceeded
the general bound and their items fell into the preceding item's region or
the cover."""
_DATE_TEXT = r"([A-Z][a-z]+\.?\s+\d{1,2},?\s+\d{4})"
_REPORT_PERIOD_QUARTERLY = re.compile(
    rf"for\s+the\s+quarterly\s+period\s+ended[\s*]*{_DATE_TEXT}", re.IGNORECASE
)
_REPORT_PERIOD_ANNUAL = re.compile(
    rf"for\s+the\s+(?:fiscal\s+)?year\s+ended[\s*]*{_DATE_TEXT}", re.IGNORECASE
)
_MONTHS = {
    name: number
    for number, names in enumerate(
        (
            ("january", "jan"),
            ("february", "feb"),
            ("march", "mar"),
            ("april", "apr"),
            ("may",),
            ("june", "jun"),
            ("july", "jul"),
            ("august", "aug"),
            ("september", "sep", "sept"),
            ("october", "oct"),
            ("november", "nov"),
            ("december", "dec"),
        ),
        start=1,
    )
    for name in names
}
_DATE_PARTS = re.compile(r"([A-Za-z]+)\.?\s+(\d{1,2}),?\s+(\d{4})")


def parse_source_date(text: str) -> date | None:
    """Parse a calendar date only when the filing spells it explicitly.

    A calendar date written the way a filing writes one ("June 30, 2026",
    "Dec. 31, 2025"); ``None`` for anything else, never a guess.
    """
    match = _DATE_PARTS.search(text.replace("\xa0", " "))
    if match is None:
        return None
    month = _MONTHS.get(match.group(1).casefold())
    if month is None:
        return None
    try:
        return date(int(match.group(3)), month, int(match.group(2)))
    except ValueError:
        return None


def _normalized_heading(stripped: str) -> tuple[str, str]:
    r"""Remove heading emphasis while retaining its kind.

    The heading text without its emphasis markers and non-breaking spaces,
    and the emphasis it carried (`BOLD`, `ITALIC` or none): some agents render
    every heading as ``**ITEM 1A.\xa0RISK FACTORS**`` or ``*Revolving
    Facility*``, with the bold runs broken mid-word (``**1.****Basis of
    presentation**``); the text between the markers is the heading.
    """
    emphasis = ""
    if stripped.startswith("**") and stripped.endswith("**"):
        emphasis = "BOLD"
    elif stripped.startswith("*") and stripped.endswith("*"):
        emphasis = "ITALIC"
    elif _MARKDOWN_HEADING.match(stripped):
        emphasis = "BOLD"
        stripped = _MARKDOWN_HEADING.sub("", stripped)
    text = _EMPHASIS.sub("", stripped).replace("\xa0", " ")
    return " ".join(text.split()), emphasis


def heading_kind(text: str, *, document_type: str) -> tuple[str, str] | None:
    """Recognize a short Part or Item heading under a filing form.

    `("PART", "II")` or `("ITEM", "2")` when a short line is a Part or an
    Item heading under this form's rules, else None: the one definition of
    those headings, shared with the table-carry policy that runs over the
    source markup before the canonical text exists.
    """
    stripped = text.strip()
    family_form = document_type.upper().removesuffix("/A")
    raw_bound = _MAX_ITEM_8K_HEADING_LENGTH + 20 if family_form == "8-K" else _MAX_HEADING_LENGTH
    if family_form != "8-K" and len(stripped) > raw_bound:
        stripped, _parenthetical = _item_head(stripped)
    if not stripped or len(stripped) > raw_bound:
        return None
    normalized, _emphasis = _normalized_heading(stripped)
    if family_form == "8-K":
        # The item's official title, at its own length.
        if len(normalized) > _MAX_ITEM_8K_HEADING_LENGTH:
            return None
        item_8k = _ITEM_8K.match(normalized)
        if item_8k is not None:
            return ("ITEM", item_8k.group(1).upper())
    if len(normalized) > _MAX_NORMALIZED_HEADING_LENGTH:
        return None
    part = _PART.match(normalized)
    if part is not None:
        return ("PART", part.group(1).upper())
    if family_form != "8-K":
        item = _ITEM.match(normalized)
        if item is not None:
            return ("ITEM", item.group(1).upper())
    return None


@dataclass(frozen=True, slots=True)
class StructureHeading:
    """One heading line at its exact range in the canonical text."""

    kind: str
    """`PART`, `ITEM`, `NOTE`, `SUB`, `SUB_UNCERTAIN`, `UNIT` or `TITLE`."""

    text: str
    line: int
    character_start: int
    character_end: int
    continued: bool = False
    implied_part: str | None = None
    """The official part of an item heading recognized by its bare title in
    a filing that labels no item (`ITEM` only); None for a labelled one."""


@dataclass(frozen=True, slots=True)
class WindowStructure:
    """Describe the filing structure governing one text window.

    What governs a window: its heading path, the family the path maps to,
    the heading that decided the family, and the scale statement bound to its
    note or page, when the source states one within that scope.
    """

    path: tuple[str, ...]
    family: str
    family_basis: str
    unit_declaration: tuple[str, int, int] | None
    """`(text, character_start, character_end)` of the governing statement."""

    @property
    def mapped(self) -> bool:
        """Report whether this window maps to a recognized filing heading."""
        return self.family != FAMILY_OTHER


@dataclass(frozen=True, slots=True)
class ItemRegion:
    """Bound the source range governed by an Item heading.

    One item heading and the source range it governs: from the end of
    its heading line to the next part or item heading (or the end of the
    text). ``part`` is the part state at the heading, ``item`` the item
    number as the heading spells it (``9A``, ``4``), ``title`` the rest of
    the heading line. A heading that is only a contents entry governs no
    text of its own, which the reader of the range decides, not this owner.
    """

    part: str | None
    item: str
    title: str
    heading_start: int
    heading_end: int
    body_start: int
    body_end: int


@dataclass(frozen=True, slots=True)
class ReportPeriod:
    """Record a report's stated period and source range.

    The report's own statement of the period it covers, quoted at its
    range: ``for the quarterly period ended June 30, 2026`` on a 10-Q, ``for
    the fiscal year ended December 31, 2025`` on a 10-K -- read on its cover,
    the front matter before the first Part or Item heading, because the body
    names other years in the same words (the predecessors of a comparison, a
    later fiscal year of a covenant or a plan). ``status`` is ``RESOLVED``
    when the cover states one such date, ``ABSENT`` when it states none the
    reader can find, ``CONFLICTING`` when a quarterly cover names more than
    one; the annual cover's statement is the latest of the dates it names in
    that form, and the basis says so.
    """

    status: str
    period_end: date | None
    text: str
    character_start: int
    character_end: int
    basis: str
    candidates: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _State:
    part: str | None
    item: str | None
    note: str | None
    sub: str | None
    unit: tuple[str, int, int] | None
    front_matter: bool


_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (FAMILY_SUBSEQUENT, ("subsequent event", "explanatory note", "amendment no")),
    (
        FAMILY_LEGAL,
        (
            "legal proceeding",
            "litigation",
            "contingenc",
            "regulatory",
            "environmental",
            "income tax",
            "taxes",
            "antitrust",
            "proceeding",
        ),
    ),
    (
        FAMILY_TRANSACTIONS,
        (
            "acquisition",
            "divestiture",
            "business combination",
            "joint venture",
            "noncontrolling",
            "discontinued",
            "disposition",
            "investments in",
            "equity method",
        ),
    ),
    (
        FAMILY_COMMITMENTS,
        (
            "restructuring",
            "impairment",
            "pension",
            "retirement",
            "benefit plan",
            "postretirement",
            "commitments",
            "guarantee",
            "lease",
            "transformation",
        ),
    ),
    (
        FAMILY_FINANCING,
        (
            "debt",
            "borrowing",
            "credit agreement",
            "credit facilit",
            "notes payable",
            "senior notes",
            "term loan",
            "liquidity",
            "capital resources",
            "financing",
            "equity",
            "stockholders",
            "shareholders",
            "share repurchase",
            "repurchase",
            "dividend",
            "unregistered sales",
            "market for registrant",
            "securities",
            "capital stock",
            "earnings per share",
            "securitization",
            "financial instruments and debt",
        ),
    ),
    (
        FAMILY_CONTROLS,
        (
            "controls and procedures",
            "internal control",
            "disclosure controls",
            "cybersecurity",
            "other information",
            "directors",
            "executive officers",
            "corporate governance",
            "10b5-1",
            "annual meeting",
            "shareholder vote",
            "bylaw",
            "auditor",
            "accountant",
            "audit committee",
        ),
    ),
    (
        FAMILY_OPERATIONS,
        (
            "results of operations",
            "management's discussion",
            "management" + chr(0x2019) + "s discussion",
            "segment",
            "revenue",
            "outlook",
            "guidance",
            "business",
            "operations",
            "overview",
            "sales",
        ),
    ),
    (FAMILY_RISK, ("risk factor", "forward-looking", "cautionary")),
    (
        FAMILY_STATEMENTS,
        (
            "financial statements",
            "balance sheet",
            "statements of operations",
            "statements of income",
            "statements of cash flows",
            "statements of comprehensive",
            "statements of equity",
            "accounting polic",
            "basis of presentation",
            "fair value",
            "goodwill",
            "intangible",
            "inventor",
            "receivable",
            "property, plant",
            "accumulated other comprehensive",
            "general",
            "summary of significant",
            "supplemental",
            "quantitative and qualitative",
        ),
    ),
    (
        FAMILY_FRONT_MATTER,
        (
            "table of contents",
            "exhibit",
            "signature",
            "cover page",
            "form 10-",
            "form 8-k",
            "mine safety",
            "properties",
            "unresolved staff comments",
            "reserved",
        ),
    ),
)

_ITEM_FAMILY_10K: dict[str, str] = {
    "1": FAMILY_OPERATIONS,
    "1A": FAMILY_RISK,
    "1B": FAMILY_FRONT_MATTER,
    "1C": FAMILY_CONTROLS,
    "2": FAMILY_FRONT_MATTER,
    "3": FAMILY_LEGAL,
    "4": FAMILY_FRONT_MATTER,
    "5": FAMILY_FINANCING,
    "6": FAMILY_FRONT_MATTER,
    "7": FAMILY_OPERATIONS,
    "7A": FAMILY_OTHER,
    "8": FAMILY_STATEMENTS,
    "9": FAMILY_CONTROLS,
    "9A": FAMILY_CONTROLS,
    "9B": FAMILY_CONTROLS,
    "9C": FAMILY_CONTROLS,
    "10": FAMILY_CONTROLS,
    "11": FAMILY_CONTROLS,
    "12": FAMILY_FINANCING,
    "13": FAMILY_CONTROLS,
    "14": FAMILY_CONTROLS,
    "15": FAMILY_FRONT_MATTER,
    "16": FAMILY_FRONT_MATTER,
}
_ITEM_FAMILY_10Q_PART_I: dict[str, str] = {
    "1": FAMILY_STATEMENTS,
    "2": FAMILY_OPERATIONS,
    "3": FAMILY_OTHER,
    "4": FAMILY_CONTROLS,
}
_ITEM_FAMILY_10Q_PART_II: dict[str, str] = {
    "1": FAMILY_LEGAL,
    "1A": FAMILY_RISK,
    "2": FAMILY_FINANCING,
    "3": FAMILY_FINANCING,
    "4": FAMILY_FRONT_MATTER,
    "5": FAMILY_CONTROLS,
    "6": FAMILY_FRONT_MATTER,
}
_ITEM_FAMILY_8K: dict[str, str] = {
    "1": FAMILY_TRANSACTIONS,
    "2": FAMILY_OPERATIONS,
    "3": FAMILY_FINANCING,
    "4": FAMILY_CONTROLS,
    "5": FAMILY_CONTROLS,
    "6": FAMILY_FINANCING,
    "7": FAMILY_SUBSEQUENT,
    "8": FAMILY_SUBSEQUENT,
    "9": FAMILY_FRONT_MATTER,
}
_ITEM_8K_SPECIAL: dict[str, str] = {
    "2.02": FAMILY_OPERATIONS,
    "2.03": FAMILY_FINANCING,
    "2.04": FAMILY_FINANCING,
    "2.05": FAMILY_COMMITMENTS,
    "2.06": FAMILY_COMMITMENTS,
    "2.01": FAMILY_TRANSACTIONS,
    "1.01": FAMILY_TRANSACTIONS,
    "1.02": FAMILY_TRANSACTIONS,
    "1.03": FAMILY_FINANCING,
    "3.02": FAMILY_FINANCING,
    "3.03": FAMILY_FINANCING,
    "5.07": FAMILY_CONTROLS,
    "8.01": FAMILY_SUBSEQUENT,
    "7.01": FAMILY_SUBSEQUENT,
}


def _keyword_family(text: str) -> str | None:
    lowered = text.casefold()
    for family, keywords in _KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            return family
    return None


def _is_capitalised_title(text: str) -> bool:
    letters = [character for character in text if character.isalpha()]
    if len(letters) < 4 or "$" in text or len(text) > 90:
        return False
    upper = sum(1 for character in letters if character.isupper())
    return upper / len(letters) >= 0.95


def _is_title_case(text: str) -> bool:
    if len(text) > 60 or len(text) < 4 or "$" in text:
        return False
    stripped = text.rstrip(".:")
    words = stripped.split()
    if not 1 <= len(words) <= 8:
        return False
    capitalised = sum(1 for word in words if word[:1].isupper())
    function_words = {"and", "of", "the", "for", "to", "in", "on", "or", "a", "an", "with"}
    content = [word for word in words if word.casefold() not in function_words]
    return capitalised >= max(1, len(content))


class DocumentStructure:
    """Index the headings and scale statements of one canonical filing.

    Headings, page headers and scale statements of one canonical text, and
    the structure that governs any character position in it.
    """

    def __init__(self, text: str, *, document_type: str) -> None:
        """Index headings and report periods from one canonical filing."""
        self.document_type = document_type.upper()
        self.family_form = self.document_type.removesuffix("/A")
        self.length = len(text)
        self.headings: tuple[StructureHeading, ...] = self._promote_bare_titles(self._scan(text))
        self.cover_end: int = self._cover_end(text, self._body_start())
        self._states: list[tuple[int, _State]] = self._states_at_headings()
        self.report_period: ReportPeriod = self._report_period(text)

    def _report_period(self, text: str) -> ReportPeriod:
        quarterly = self.family_form == "10-Q"
        pattern = _REPORT_PERIOD_QUARTERLY if quarterly else _REPORT_PERIOD_ANNUAL
        found: list[tuple[date, int, int, str]] = []
        # The cover alone states the report's period; the body is past the
        # first Part or Item heading (the whole text when there is none).
        for match in pattern.finditer(text, 0, self._body_start()):
            parsed = parse_source_date(match.group(1))
            if parsed is not None:
                found.append((parsed, match.start(), match.end(), match.group(0)))
        if not found:
            return ReportPeriod(
                status="ABSENT",
                period_end=None,
                text="",
                character_start=0,
                character_end=0,
                basis="the document states no period in its cover form",
            )
        distinct = sorted({value[0] for value in found})
        if quarterly and len(distinct) > 1:
            return ReportPeriod(
                status="CONFLICTING",
                period_end=None,
                text="",
                character_start=0,
                character_end=0,
                basis="the quarterly report names more than one period in its cover form",
                candidates=tuple(value.isoformat() for value in distinct),
            )
        chosen = found[0] if quarterly else max(found, key=lambda value: (value[0], -value[1]))
        return ReportPeriod(
            status="RESOLVED",
            period_end=chosen[0],
            text=" ".join(chosen[3].split()),
            character_start=chosen[1],
            character_end=chosen[2],
            basis=(
                "the report's own statement of the quarterly period"
                if quarterly
                else "the latest fiscal year the document names in its cover form"
            ),
            candidates=tuple(value.isoformat() for value in distinct),
        )

    def _promote_bare_titles(
        self, headings: tuple[StructureHeading, ...]
    ) -> tuple[StructureHeading, ...]:
        """Recognize official Item titles when headings omit Item labels.

        A periodic filing whose body labels no item heads its items with their
        official titles alone: the first heading line that is one of the form's
        titles opens that item, its official part implied. Only when no labelled
        item exists and at least `_MINIMUM_BARE_TITLES` titles are present; a
        later repeat of a title stays what the scan made it.
        """
        titles = _OFFICIAL_TITLE_KEYS.get(self.family_form)
        if titles is None or any(h.kind == "ITEM" for h in headings):
            return headings
        found: dict[str, int] = {}
        for index, heading in enumerate(headings):
            if heading.kind not in {"SUB", "SUB_UNCERTAIN", "TITLE"}:
                continue
            key = _title_key(heading.text)
            if key in titles and key not in found:
                found[key] = index
        if len(found) < _MINIMUM_BARE_TITLES:
            return headings
        promoted = list(headings)
        for key, index in found.items():
            item, part = titles[key]
            heading = headings[index]
            promoted[index] = StructureHeading(
                kind="ITEM",
                text=f"Item {item}. {heading.text}",
                line=heading.line,
                character_start=heading.character_start,
                character_end=heading.character_end,
                continued=heading.continued,
                implied_part=f"PART {part}",
            )
        return tuple(promoted)

    def item_regions(self) -> tuple[ItemRegion, ...]:
        """Every item heading with the range it governs, in source order."""
        regions: list[ItemRegion] = []
        part: str | None = None
        for index, heading in enumerate(self.headings):
            if heading.kind == "PART":
                match = _PART.match(heading.text)
                part = f"PART {match.group(1).upper()}" if match else heading.text
                continue
            if heading.kind != "ITEM":
                continue
            if heading.implied_part is not None:
                part = heading.implied_part
            match = (
                _ITEM_8K.match(heading.text)
                if self.family_form == "8-K"
                else _ITEM.match(heading.text)
            )
            if match is None:
                continue
            body_end = self.length
            for later in self.headings[index + 1 :]:
                if later.kind in {"PART", "ITEM"}:
                    body_end = later.character_start
                    break
            regions.append(
                ItemRegion(
                    part=part,
                    item=match.group(1).upper(),
                    title=" ".join(match.group(2).split()).strip(" ."),
                    heading_start=heading.character_start,
                    heading_end=heading.character_end,
                    body_start=heading.character_end,
                    body_end=body_end,
                )
            )
        return tuple(regions)

    def _body_start(self) -> int:
        """Find the first Part or Item heading after front matter.

        Where the body can first begin: the start of the first Part or Item
        heading, or the end of the text when the filing carries none.
        """
        for heading in self.headings:
            if heading.kind in {"PART", "ITEM"}:
                return heading.character_start
        return self.length

    @staticmethod
    def _cover_end(text: str, limit: int) -> int:
        """Find the end of cover-page boilerplate before filing content.

        Where the cover page ends: the end of the last line before `limit`
        that reads as cover boilerplate. The cover is front matter, so a
        cover-like line past the body's first heading -- NEE's combined 10-K
        closes with "no ... proxy soliciting material has been sent to
        security holders of FPL" -- is body text, not the cover. A filing
        whose body carries no detectable heading is still a body after this
        point -- an explicit unmapped state, not front matter.
        """
        position = 0
        end = 0
        # A filing with no Part or Item heading bounds its cover by its body:
        # once a substantive line (a paragraph past `_BODY_LINE_LENGTH` that is
        # not cover boilerplate) has been read, a later cover-like line -- a
        # current report's "(c) Shell company transactions. N/A" under its
        # Item 9.01, a signature -- no longer extends the cover. Measured on
        # the casebook's recorded DG 8-K (no item line in its canonical text):
        # the cover ran to 4,271 of 4,546 characters and the vote results
        # in its body were routed to no topic.
        headingless = limit >= len(text)
        body_seen = False
        for line in text.split("\n"):
            if position >= limit:
                break
            stripped = line.strip()
            if stripped and _COVER.search(stripped) and len(stripped) <= 400:
                if not (headingless and body_seen):
                    end = position + len(line)
            elif headingless and len(stripped) >= _BODY_LINE_LENGTH:
                body_seen = True
            position += len(line) + 1
        return end

    def _scan(self, text: str) -> tuple[StructureHeading, ...]:
        lines = text.split("\n")
        counts = Counter(line.strip() for line in lines if line.strip())
        headings: list[StructureHeading] = []
        position = 0
        for number, line in enumerate(lines, start=1):
            stripped = line.strip()
            start = position + (len(line) - len(line.lstrip()))
            end = start + len(stripped)
            position += len(line) + 1
            # The same bounds `heading_kind` applies: a current report's item
            # heading at the length of its official title, every other
            # heading at the general bound.
            current_report = self.family_form == "8-K"
            raw_bound = _MAX_ITEM_8K_HEADING_LENGTH + 20 if current_report else _MAX_HEADING_LENGTH
            parenthetical: str | None = None
            if not current_report and len(stripped) > raw_bound:
                stripped, parenthetical = _item_head(stripped)
                if parenthetical is not None:
                    end = start + len(stripped)
            if not stripped or len(stripped) > raw_bound:
                continue
            text, emphasis = _normalized_heading(stripped)
            if not text or _PAGE_NUMBER.match(text):
                continue
            item_title = (
                current_report
                and len(text) <= _MAX_ITEM_8K_HEADING_LENGTH
                and _ITEM_8K.match(text) is not None
            )
            if not item_title and len(text) > _MAX_NORMALIZED_HEADING_LENGTH:
                continue
            kind = self._classify(text, emphasis=emphasis, repeated=counts[stripped] >= 3)
            if kind is None:
                continue
            headings.append(
                StructureHeading(
                    kind=kind,
                    text=text,
                    line=number,
                    character_start=start,
                    character_end=end,
                    continued=bool(_CONTINUED.search(text)),
                )
            )
            if parenthetical is not None and kind == "ITEM" and _UNIT.match(parenthetical):
                # The item line's own scale statement governs the item as a
                # unit declaration, at the parenthetical's exact range.
                unit_start = position - len(line) - 1 + line.find(parenthetical)
                headings.append(
                    StructureHeading(
                        kind="UNIT",
                        text=parenthetical,
                        line=number,
                        character_start=unit_start,
                        character_end=unit_start + len(parenthetical),
                    )
                )
        return tuple(headings)

    def _classify(self, text: str, *, emphasis: str, repeated: bool) -> str | None:
        if _UNIT.match(text):
            return "UNIT"
        if _PART.match(text):
            return "PART"
        if self.family_form == "8-K":
            if _ITEM_8K.match(text):
                return "ITEM"
        elif _ITEM.match(text):
            return "ITEM"
        if _TABLE_OF_CONTENTS.match(text):
            return "TITLE"
        note = _NOTE.match(text)
        if note is not None and (
            _is_capitalised_title(note.group("title"))
            or (_is_title_case(note.group("title")) and not text.endswith("."))
            or (emphasis == "BOLD" and not text.endswith("."))
        ):
            return "NOTE"
        if _SUB_LETTER.match(text):
            return "SUB"
        if _is_capitalised_title(text):
            # A registrant's name and a statement's title repeat as running
            # page headers; they do not open a section.
            return "TITLE" if repeated else "SUB"
        if emphasis == "BOLD" and len(text) <= 80 and not text.endswith(".") and not repeated:
            return "SUB"
        if emphasis == "ITALIC" and len(text) <= 80 and not text.endswith(".") and not repeated:
            return "SUB_UNCERTAIN"
        if _is_title_case(text) and not repeated:
            return "SUB_UNCERTAIN"
        return None

    def _states_at_headings(self) -> list[tuple[int, _State]]:
        states: list[tuple[int, _State]] = []
        part = item = note = sub = None
        unit: tuple[str, int, int] | None = None
        front_matter = True
        for heading in self.headings:
            if heading.kind == "PART":
                match = _PART.match(heading.text)
                label = f"PART {match.group(1).upper()}" if match else heading.text
                if label != part:
                    part = label
                    item = note = sub = None
                    unit = None
                front_matter = False
            elif heading.kind == "ITEM":
                if heading.implied_part is not None and heading.implied_part != part:
                    part = heading.implied_part
                item = heading.text
                note = sub = None
                unit = None
                front_matter = False
            elif heading.kind == "NOTE":
                # A new note keeps the page's scale statement: the notes'
                # "(In thousands ...)" heads the page, and a note that begins
                # mid-page is stated in the same scale until an item changes
                # or a new statement declares its own.
                if not heading.continued:
                    sub = None
                note = _CONTINUED.sub("", heading.text).strip()
                front_matter = False
            elif heading.kind in {"SUB", "SUB_UNCERTAIN"}:
                if not heading.continued:
                    sub = _CONTINUED.sub("", heading.text).strip()
            elif heading.kind == "UNIT":
                unit = (heading.text, heading.character_start, heading.character_end)
            elif heading.kind == "TITLE" and _TABLE_OF_CONTENTS.match(heading.text):
                front_matter = True
            states.append(
                (
                    heading.character_start,
                    _State(
                        part=part,
                        item=item,
                        note=note,
                        sub=sub,
                        unit=unit,
                        front_matter=front_matter,
                    ),
                )
            )
        return states

    def locate(self, character_start: int, character_end: int | None = None) -> WindowStructure:
        """Return the filing structure governing a character window.

        The structure governing a window. A heading inside the window's
        first half opens the section the window belongs to; one in its
        second half belongs to the next window. A window past the cover with
        no governing heading is an explicit unmapped state.
        """
        pivot = (
            character_start
            if character_end is None
            else character_start + (character_end - character_start) // 2
        )
        index = bisect_right([start for start, _ in self._states], pivot) - 1
        state = (
            self._states[index][1]
            if index >= 0
            else _State(None, None, None, None, None, front_matter=True)
        )
        path = tuple(value for value in (state.part, state.item, state.note, state.sub) if value)
        family, basis = self._family(state)
        if (
            family == FAMILY_FRONT_MATTER
            and character_start >= self.cover_end
            and state.item is None
            and state.note is None
        ):
            family, basis = FAMILY_OTHER, "body after the cover with no detected heading"
        return WindowStructure(
            path=path, family=family, family_basis=basis, unit_declaration=state.unit
        )

    def _family(self, state: _State) -> tuple[str, str]:
        if state.front_matter and state.item is None and state.note is None:
            return FAMILY_FRONT_MATTER, "front matter before the first part or item"
        # The most specific heading decides, so a note's family wins over its
        # item's and a sub-heading's over its note's, where each maps.
        for label, text in (("sub", state.sub), ("note", state.note)):
            if text:
                family = _keyword_family(text)
                if family is not None:
                    return family, f"{label}: {text}"
        if state.item:
            family = self._item_family(state.item, state.part)
            if family is not None:
                return family, f"item: {state.item}"
            family = _keyword_family(state.item)
            if family is not None:
                return family, f"item: {state.item}"
        if state.note or state.sub:
            return FAMILY_OTHER, f"unmapped heading: {state.sub or state.note}"
        return FAMILY_OTHER, "no governing item or note"

    def _item_family(self, item_text: str, part: str | None) -> str | None:
        if self.family_form == "8-K":
            match = _ITEM_8K.match(item_text)
            if match is None:
                return None
            number = match.group(1)
            return _ITEM_8K_SPECIAL.get(number) or _ITEM_FAMILY_8K.get(number.split(".")[0])
        match = _ITEM.match(item_text)
        if match is None:
            return None
        number = match.group(1).upper()
        if self.family_form == "10-Q":
            if part == "PART II":
                return _ITEM_FAMILY_10Q_PART_II.get(number)
            return _ITEM_FAMILY_10Q_PART_I.get(number)
        if self.family_form == "10-K":
            return _ITEM_FAMILY_10K.get(number)
        return None


__all__ = [
    "EVIDENCE_FAMILIES",
    "FAMILIES",
    "FAMILY_COMMITMENTS",
    "FAMILY_CONTROLS",
    "FAMILY_FINANCING",
    "FAMILY_FRONT_MATTER",
    "FAMILY_LEGAL",
    "FAMILY_OPERATIONS",
    "FAMILY_OTHER",
    "FAMILY_RISK",
    "FAMILY_STATEMENTS",
    "FAMILY_SUBSEQUENT",
    "FAMILY_TRANSACTIONS",
    "STRUCTURE_RULES_ID",
    "DocumentStructure",
    "ItemRegion",
    "ReportPeriod",
    "StructureHeading",
    "WindowStructure",
    "heading_kind",
    "parse_source_date",
]
