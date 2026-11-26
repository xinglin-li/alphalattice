"""Extract typed disclosures from three closed filing families.

Typed disclosure extraction: three closed families read from the sections
a filing keeps them in, as source assertions with their subject, action,
polarity, period and qualifiers bound to exact text ranges.

The unit is an assertion under a defined subject, property and time scope,
not a sentence that resembles a template. Each family is a small reviewed
rule set derived from its definition and from the varied source forms in the
admitted corpus; a keyword match with an unresolved negation, an unbound
subject, a truncated sentence or a conflicting neighbour is never accepted.
What the rules cannot read is returned as an explicit state -- not found,
reference required, ambiguous, source unavailable, not applicable -- with
the text that caused it kept visible. Absence of text never becomes an
explicit none. A parsed assertion is what the source says; it is not a
certified fact, an actor's check outcome or a CRO judgment.

Rule identities are frozen here (`TYPED_DISCLOSURE_RULES_ID`, one rule id
per family) and the definitions table is hashed into every record, so a
later rule version rotates only the derived observations it produces. The
identity names what the rules read, not only the definitions table: an
edit that changes how a sentence is parsed moves it, so that observations
sealed under one identity never compare as a corporate change against
observations sealed under another (`compare_typed_disclosures`).

v2 (2026-09-18): a negated or conditional conclusion verb ("has not
concluded", "was unable to conclude") yields no conclusion; a none
statement's exception is read with the family's affirmative rule or kept
visible as unresolved, never dropped into a bare none; an action date is
told from a period boundary date.

v3 (2026-09-18): a list glyph that opens a sentence is layout, not its first
word -- "•On December 9, 2025, Terrell Kirk Crews II, Executive Vice
President, Chief Risk Officer, adopted a Rule 10b5-1 trading arrangement ..."
is the dated statement the rule reads; NEE's four adoptions and two
terminations were each an unrecognised statement. The inspected scope a
non-extracted observation delivers ends at a paragraph boundary and runs
through a table carried into the item -- its rows and the footnotes that
qualify them -- within the reader's per-span ceiling, and the observation
says when the item continues beyond the delivered scope.

v4 (2026-09-20, the eight-topic completeness assignment): two families
the loss trace showed returned by the residual search and never
selected, or unrouted -- the customer-concentration statement of the
segment and concentration notes (KO's "No bottlers or customers
represented 10% or more of our net operating revenues", NVDA's direct
customers at 22% and 14%, DRI's "We do not rely on any major customers")
and Item 1C's statement of whether cybersecurity threats or incidents
have materially affected the registrant (KO's, SYF's). A region may now
be a note the spec names by heading words (`RegionSpec.note_keywords`)
rather than an item. The three earlier families' rules are unchanged
and keep their rule identities; the definitions table moves.

v5 (2026-09-20, the first-release safety closeout, finding A): the
cyber-effect rule's negation governs the asserted predicate only. A
correlative ("have not only materially affected our operations but also
increased costs") is not a negation and its "but" is not an exception:
the effect is asserted. An inability or a determination not yet made
("unable to determine whether", "have not yet determined whether") is
no conclusion: no polarity is read, the sentence stays visible and the
observation is AMBIGUOUS. Under v4 the first read as a qualified
negative and the second as a negative or an affirmative.

v6 (2026-09-20, the predicate-scope correction): the cyber-effect rule
binds a negation to the effect predicate before reading it. A negation
counts only inside the predicate's own verb group ("have not materially
affected", "did not have a material"), on a negative subject that
governs the predicate ("no cybersecurity incident has materially
affected"), or on an awareness or experience verb whose object is the
cyber subject the predicate describes ("are not aware of any
cybersecurity incidents that have materially affected", "have not
experienced any cybersecurity incident that has materially affected").
A negation of another predicate ("were not prevented by our controls
and have materially affected our operations") never negates the
effect; because the rule cannot prove what it does govern, the polarity
stays unread and the sentence visible as SCOPE_UNRESOLVED -- the
parser's limit, told apart from the registrant's own NO_CONCLUSION.
Idioms whose "not" negates nothing ("including but not limited to")
are inert. Under v5 the prefix of the clause was searched for any
negative word, so the prevention sentence read as a certain negative.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from enum import StrEnum

from alphalattice.kernel.shared_kernel.identity import canonical_hash

from ..documents.structure import (
    DocumentStructure,
    ItemRegion,
    ReportPeriod,
    parse_source_date,
)
from ..retrieval.session import sentence_ends

TYPED_DISCLOSURE_RULES_ID = "alternative-evidence.typed-disclosures.v6"


class TypedDisclosureFamily(StrEnum):
    """Name a closed family of typed SEC disclosures."""

    DISCLOSURE_CONTROLS_CONCLUSION = "DISCLOSURE_CONTROLS_CONCLUSION"
    UNREGISTERED_EQUITY_SALES = "UNREGISTERED_EQUITY_SALES"
    INSIDER_TRADING_ARRANGEMENTS = "INSIDER_TRADING_ARRANGEMENTS"
    CUSTOMER_CONCENTRATION = "CUSTOMER_CONCENTRATION"
    CYBERSECURITY_THREAT_EFFECT = "CYBERSECURITY_THREAT_EFFECT"


class TypedDisclosureState(StrEnum):
    """Record a typed disclosure's observed state."""

    EXTRACTED = "EXTRACTED"
    EXPLICIT_NONE = "EXPLICIT_NONE"
    NOT_FOUND = "NOT_FOUND"
    REFERENCE_REQUIRED = "REFERENCE_REQUIRED"
    AMBIGUOUS = "AMBIGUOUS"
    SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True, slots=True)
class RegionSpec:
    """Where a family's disclosure lives on one form.

    Where a family's disclosure lives on one form: a navigation prior, not
    a guarantee of presence or an exclusive location.
    """

    form: str
    part: str
    item: str
    title_keywords: tuple[str, ...]
    mandatory: bool
    """Whether the form requires the item to be present, so that its absence
    from an otherwise complete text says the text is incomplete rather than
    that the registrant omitted an inapplicable item."""
    note_keywords: tuple[str, ...] = ()
    """When set, the family's regions are the notes past the cover whose
    heading contains one of these words (a segment or concentration note),
    each bounded by the next note, item or part heading; `item` then names
    the item the notes belong to and `title_keywords` is unused."""


@dataclass(frozen=True, slots=True)
class FamilyDefinition:
    """Bind a disclosure family to its reading rule and source regions."""

    family: TypedDisclosureFamily
    rule_id: str
    subject: str
    cardinality: str
    definition: str
    regions: tuple[RegionSpec, ...]

    @property
    def forms(self) -> tuple[str, ...]:
        """List the SEC forms supported by this family."""
        return tuple(dict.fromkeys(spec.form for spec in self.regions))


DEFINITIONS: tuple[FamilyDefinition, ...] = (
    FamilyDefinition(
        family=TypedDisclosureFamily.DISCLOSURE_CONTROLS_CONCLUSION,
        rule_id="typed-disclosures.disclosure-controls-conclusion.v3",
        subject="management's conclusion on the effectiveness of the registrant's "
        "disclosure controls and procedures (Exchange Act Rules 13a-15(e)/15d-15(e))",
        cardinality="SCALAR",
        definition=(
            "The principal executive and financial officers' stated conclusion that the "
            "registrant's disclosure controls and procedures were or were not effective as "
            "of an evaluation date. Not the same as management's assessment of internal "
            "control over financial reporting, a change in that control, an auditor's "
            "opinion or a remediation status; a qualified or unclear conclusion stays "
            "qualified or unclear."
        ),
        regions=(
            RegionSpec("10-K", "PART II", "9A", ("controls and procedures",), True),
            RegionSpec("10-Q", "PART I", "4", ("controls and procedures",), True),
        ),
    ),
    FamilyDefinition(
        family=TypedDisclosureFamily.UNREGISTERED_EQUITY_SALES,
        rule_id="typed-disclosures.unregistered-equity-sales.v3",
        subject="the registrant's sales of equity securities not registered under the "
        "Securities Act during the quarterly report's period (Regulation S-K Item 701)",
        cardinality="LIST",
        definition=(
            "Each sale of unregistered equity securities the registrant reports for the "
            "period covered by its quarterly report, with its date, securities, purchaser, "
            "consideration and exemption where stated. An explicit none applies only to "
            "the subject, period and scope it states. Issuer repurchases (Item 2(c)) are "
            "not unregistered sales. Information said to be previously reported on Form "
            "8-K is a reference, not the sale itself."
        ),
        regions=(RegionSpec("10-Q", "PART II", "2", ("unregistered",), False),),
    ),
    FamilyDefinition(
        family=TypedDisclosureFamily.INSIDER_TRADING_ARRANGEMENTS,
        rule_id="typed-disclosures.insider-trading-arrangements.v3",
        subject="adoption, modification and termination of Rule 10b5-1 and non-Rule "
        "10b5-1 trading arrangements by directors and officers during the fiscal quarter "
        "(Regulation S-K Item 408(a))",
        cardinality="LIST",
        definition=(
            "Each director or officer the registrant reports as having adopted, modified "
            "or terminated a Rule 10b5-1 or non-Rule 10b5-1 trading arrangement during the "
            "report's last fiscal quarter, with the person, role, action, arrangement kind, "
            "date and stated terms. A statement that none adopted or terminated one in the "
            "quarter says nothing about arrangements already in force. Nothing here infers "
            "a trade or a violation."
        ),
        regions=(
            RegionSpec("10-Q", "PART II", "5", ("other information",), False),
            RegionSpec("10-K", "PART II", "9B", ("other information",), False),
        ),
    ),
    FamilyDefinition(
        family=TypedDisclosureFamily.CUSTOMER_CONCENTRATION,
        rule_id="typed-disclosures.customer-concentration.v1",
        subject="the registrant's statement of customer concentration: a customer's "
        "share of revenue or receivables, or the explicit absence of any customer at "
        "the stated threshold (ASC 280-10-50-42 major customers; Regulation S-K "
        "Item 101(c))",
        cardinality="LIST",
        definition=(
            "Each sentence of the segment, revenue or concentration notes that states a "
            "customer's, bottler's, distributor's or client's percentage of the "
            "registrant's revenue, sales or receivables, or states that no customer "
            "reached a threshold or that the registrant relies on no major customer. "
            "The percentage and the words are the source's; no customer is named by "
            "the rules, no share is summed, and a statement about one period says "
            "nothing about another. An absence applies only to the threshold, subject "
            "and period it states."
        ),
        regions=(
            RegionSpec(
                "10-K",
                "PART II",
                "8",
                ("financial statements",),
                False,
                note_keywords=("segment", "concentration", "customer"),
            ),
            RegionSpec(
                "10-Q",
                "PART I",
                "1",
                ("financial statements",),
                False,
                note_keywords=("segment", "concentration", "customer"),
            ),
        ),
    ),
    FamilyDefinition(
        family=TypedDisclosureFamily.CYBERSECURITY_THREAT_EFFECT,
        rule_id="typed-disclosures.cybersecurity-threat-effect.v3",
        subject="the registrant's statement of whether risks from cybersecurity "
        "threats, including previous incidents, have materially affected or are "
        "reasonably likely to materially affect it (Regulation S-K Item 106(b)(2))",
        cardinality="SCALAR",
        definition=(
            "The Item 1C sentence that states whether cybersecurity threats or "
            "incidents have materially affected, or are reasonably likely to materially "
            "affect, the registrant, its business strategy, results of operations or "
            "financial condition, with the period or scope it states. A statement that "
            "none has is not proof of no incident; a risk-factor sentence about what "
            "could happen is not this statement; a qualified statement stays qualified."
        ),
        regions=(RegionSpec("10-K", "PART I", "1C", ("cybersecurity",), False),),
    ),
)


def definitions_hash() -> str:
    """The identity of the frozen definitions table."""
    return str(
        canonical_hash(
            {
                "rules_id": TYPED_DISCLOSURE_RULES_ID,
                "definitions": [
                    {
                        "family": value.family.value,
                        "rule_id": value.rule_id,
                        "subject": value.subject,
                        "cardinality": value.cardinality,
                        "definition": value.definition,
                        "regions": [
                            {
                                "form": spec.form,
                                "part": spec.part,
                                "item": spec.item,
                                "title_keywords": list(spec.title_keywords),
                                "mandatory": spec.mandatory,
                                **(
                                    {"note_keywords": list(spec.note_keywords)}
                                    if spec.note_keywords
                                    else {}
                                ),
                            }
                            for spec in value.regions
                        ],
                    }
                    for value in DEFINITIONS
                ],
            }
        )
    )


# --------------------------------------------------------------------------
# Drafts: what the pure extraction returns before spans are issued and read.


@dataclass(frozen=True, slots=True)
class DraftField:
    """Hold one provisional field and its source range."""

    name: str
    value: str | None
    text: str
    character_start: int
    character_end: int
    basis: str = ""


@dataclass(frozen=True, slots=True)
class DraftInstance:
    """Hold one provisional disclosure instance."""

    subject: str
    action: str
    polarity: str
    period_end: date | None
    period_text: str
    period_basis: str
    character_start: int
    character_end: int
    statement_text: str
    fields: tuple[DraftField, ...] = ()
    qualifiers: tuple[str, ...] = ()
    unknown_fields: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DraftObservation:
    """Hold a provisional family observation before sealing."""

    family: TypedDisclosureFamily
    rule_id: str
    state: TypedDisclosureState
    reason: str
    part: str | None
    item: str | None
    region_title: str
    report_period: ReportPeriod
    instances: tuple[DraftInstance, ...] = ()
    unrecognized: tuple[str, ...] = ()
    references: tuple[str, ...] = ()
    scope_range: tuple[int, int] | None = None
    inspected_ranges: tuple[tuple[int, int], ...] = ()
    context: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DocumentExtraction:
    """Summarize each family's observation and document completeness.

    Every family's observation on one document plus what the document's
    form and completeness say about all of them.
    """

    observations: tuple[DraftObservation, ...]
    incomplete_reason: str | None
    amendment_scope: tuple[str, ...] | None


# --------------------------------------------------------------------------
# Shared readers.

_WHITESPACE = re.compile(r"\s+")
_DATE = re.compile(r"[A-Z][a-z]+\.?\s+\d{1,2},?\s+\d{4}")
_EMPHASIS_MARKS = re.compile(r"\*+")
_HEADING_LINE = re.compile(r"^[*\s]*(?:item|part)\s+[0-9ivx]+", re.IGNORECASE)
_TERMINATED = re.compile("[.!?;:][\"'\u201d\u2019)\\]]*$")
_EXPLANATORY_NOTE = re.compile(r"EXPLANATORY\s+NOTE", re.IGNORECASE)
_AMENDMENT_SOLELY = re.compile(
    r"(?:amendment|10-K/A|10-Q/A)[^.]{0,200}?\bis being filed solely to\b"
    r"(?P<scope>[^\n]{0,1200}?)(?:\n\n|$)",
    re.IGNORECASE | re.DOTALL,
)
_AMENDMENT_ITEM = re.compile(r"Part\s+([IVX]+),?\s+Item\s+(\d{1,2}[A-C]?)", re.IGNORECASE)
_ORIGINAL_FILING = re.compile(
    r"originally filed its (?P<form>Annual|Quarterly) Report on Form (?P<kind>10-[KQ])[^.]{0,160}?"
    r"on (?P<date>[A-Z][a-z]+\s+\d{1,2},\s+\d{4})",
    re.IGNORECASE,
)


def _collapse(text: str) -> str:
    return _WHITESPACE.sub(" ", text.replace("\xa0", " ")).strip()


_LIST_GLYPH = re.compile(r"^[\u2022\u25e6\u25aa\u25cf\u2013\u2014\-]\s*")


def _plain(text: str) -> str:
    """Normalised for matching only.

    Normalised for matching only: emphasis marks, a list glyph opening the
    text and layout whitespace removed; offsets are never taken from this
    form.
    """
    return _LIST_GLYPH.sub("", _collapse(_EMPHASIS_MARKS.sub("", text)))


@dataclass(frozen=True, slots=True)
class Sentence:
    text: str
    character_start: int
    character_end: int
    terminated: bool
    """Whether the sentence ends in sentence punctuation; a family cue that
    runs into a heading or the end of its region without one is a truncated
    source, not an assertion."""

    @property
    def plain(self) -> str:
        return _plain(self.text)


_STATEMENT_WORDS = re.compile(
    r"\b(?:during|on|none|no|there|we|concluded|adopted|terminated|not|did|were|was|is|are|"
    r"accounted|represented|have|has|rely)\b",
    re.IGNORECASE,
)


def _looks_like_heading(sentence: Sentence) -> bool:
    """Identify unclassified disclosure subheadings.

    A short line with no sentence punctuation and none of the words a
    statement of these families is made of ("Rule 10b5-1 Plans", "Insider
    Trading Arrangements") is a sub-heading the structure scan did not
    classify, not a truncated assertion.
    """
    plain = sentence.plain
    return (
        len(plain) <= 80
        and _STATEMENT_WORDS.search(plain) is None
        and (not sentence.terminated or plain.endswith("."))
        and "," not in plain
    )


def _sentences(text: str, offset: int) -> tuple[Sentence, ...]:
    """The sentences of one region body at their absolute ranges.

    The sentences of one region body at their absolute ranges. Paragraph
    breaks end a sentence too, so a truncated line before a blank line is
    returned as its own, unterminated sentence.
    """
    out: list[Sentence] = []
    for paragraph in re.finditer(r"[^\n]+(?:\n(?!\s*\n)[^\n]+)*", text):
        body = paragraph.group(0)
        start = 0
        ends = [end for end in sentence_ends(body + " ") if end <= len(body) + 1]
        for end in ends:
            piece = body[start:end]
            if piece.strip():
                out.append(_sentence(piece, paragraph.start() + start, offset))
            start = end
        rest = body[start:]
        if rest.strip():
            out.append(_sentence(rest, paragraph.start() + start, offset))
    return tuple(out)


def _sentence(piece: str, start: int, offset: int) -> Sentence:
    lead = len(piece) - len(piece.lstrip())
    trail = len(piece) - len(piece.rstrip())
    core = piece[lead : len(piece) - trail] if trail else piece[lead:]
    plain = _plain(core)
    terminated = _TERMINATED.search(plain) is not None
    return Sentence(
        text=core,
        character_start=offset + start + lead,
        character_end=offset + start + lead + len(core),
        terminated=terminated,
    )


def _field(
    name: str,
    sentence: Sentence,
    match: re.Match[str] | None,
    *,
    value: str | None = None,
    basis: str = "",
) -> DraftField | None:
    if match is None:
        return None
    # Matches are made on the plain form; the field's range is found again in
    # the source text by its words so the range is exact.
    located = _locate(sentence, match.group(0))
    return DraftField(
        name=name,
        value=value if value is not None else _collapse(match.group(0)),
        text=_collapse(match.group(0)),
        character_start=located[0],
        character_end=located[1],
        basis=basis,
    )


def _locate(sentence: Sentence, words: str) -> tuple[int, int]:
    """Locate source words across emphasis and layout whitespace.

    The source range of `words` inside the sentence, tolerant of the
    emphasis marks and layout whitespace the plain form removed.
    """
    pattern = r"[\s*]*".join(re.escape(token) for token in _collapse(words).split(" "))
    match = re.search(pattern, sentence.text.replace("\xa0", " "))
    if match is None:
        return sentence.character_start, sentence.character_end
    return sentence.character_start + match.start(), sentence.character_start + match.end()


_PERIOD_EXPLICIT = re.compile(
    r"(?:ended|ending|as of)\s+(?:the\s+)?(" + _DATE.pattern + r")", re.IGNORECASE
)
_PERIOD_COVERED = re.compile(
    r"(?:the\s+)?(?:end\s+of\s+)?(?:the\s+)?(?:fiscal\s+)?"
    r"(?:period|periods|quarter|quarterly period|year)"
    r"\s+covered\s+by\s+this\s+(?:quarterly\s+|annual\s+)?report(?:\s+on\s+form\s+10-[kq])?",
    re.IGNORECASE,
)
_PERIOD_ANAPHORA = re.compile(
    r"(?:as\s+of\s+)?(?:the\s+end\s+of\s+such\s+period|such\s+period|the\s+evaluation\s+date)"
    r"|based\s+(?:up)?on\s+(?:that|such|this)\s+evaluation|based\s+on\s+the\s+foregoing",
    re.IGNORECASE,
)
_EVALUATION_DATE_DEFINITION = re.compile(
    r"(" + _DATE.pattern + r")\s*\(\s*the\s+[\"\u201c]evaluation\s+date[\"\u201d]\s*\)",
    re.IGNORECASE,
)


def _resolve_period(
    sentence: Sentence, *, region_sentences: tuple[Sentence, ...], report_period: ReportPeriod
) -> tuple[date | None, str, str]:
    """`(period_end, period text, basis)` for one sentence's stated period.

    `(period_end, period text, basis)` for one sentence's stated period:
    a date the sentence names, the report's own period when the sentence
    says "the period covered by this report", an antecedent when it says
    "such period" or "the Evaluation Date", else unknown.
    """
    plain = sentence.plain
    explicit = _PERIOD_EXPLICIT.search(plain)
    if explicit is not None:
        parsed = parse_source_date(explicit.group(1))
        if parsed is not None:
            return parsed, _collapse(explicit.group(0)), "stated in the sentence"
    covered = _PERIOD_COVERED.search(plain)
    if covered is not None:
        return (
            report_period.period_end,
            _collapse(covered.group(0)),
            f"the period covered by this report: {report_period.text or report_period.status}"
            if report_period.period_end is not None
            else f"the period covered by this report; the report period is {report_period.status}",
        )
    anaphora = _PERIOD_ANAPHORA.search(plain)
    if anaphora is not None:
        for earlier in reversed(
            [value for value in region_sentences if value.character_end <= sentence.character_start]
        ):
            defined = _EVALUATION_DATE_DEFINITION.search(earlier.plain)
            if defined is not None:
                parsed = parse_source_date(defined.group(1))
                if parsed is not None:
                    return parsed, _collapse(defined.group(0)), "antecedent in the same section"
            antecedent = _PERIOD_COVERED.search(earlier.plain)
            if antecedent is not None:
                return (
                    report_period.period_end,
                    _collapse(antecedent.group(0)),
                    "antecedent in the same section; the period covered by this report: "
                    + (report_period.text or report_period.status),
                )
            explicit = _PERIOD_EXPLICIT.search(earlier.plain)
            if explicit is not None:
                parsed = parse_source_date(explicit.group(1))
                if parsed is not None:
                    return parsed, _collapse(explicit.group(0)), "antecedent in the same section"
        return None, _collapse(anaphora.group(0)), "antecedent not found"
    return None, "", "no period stated in the sentence"


# --------------------------------------------------------------------------
# Family 1: management's disclosure-controls conclusion.

_CONCLUDED = re.compile(r"\bconclud(?:ed|es|e)\b", re.IGNORECASE)
_NO_CONCLUSION = re.compile(
    r"(?:\bnot|\bnever|\bunable\s+to|\bcannot|\bcan\s+not|\bcould\s+not|\bhas\s+yet\s+to"
    r"|\bhave\s+yet\s+to|\bwould\s+have|\bcould\s+have|\bmight\s+have|\bmay\s+have)"
    r"\s+(?:yet\s+|have\s+|been\s+able\s+to\s+|able\s+to\s+)?$",
    re.IGNORECASE,
)
"""The words that, immediately before "concluded", negate or condition the
conclusion itself: "has not concluded", "was unable to conclude", "would
have concluded". Such a sentence states no conclusion -- neither effective
nor not effective -- whatever predicate follows."""
_EFFECTIVE = re.compile(
    r"\b(?:(?:is|are|was|were|remain(?:s|ed)?|to be)\s+(?:not\s+|no\s+longer\s+)?effective"
    r"|ineffective)\b",
    re.IGNORECASE,
)
_DCP_PHRASE = re.compile(r"disclosure\s+controls?\s+and\s+procedures", re.IGNORECASE)
_ICFR_PHRASE = re.compile(r"internal\s+control\s+over\s+financial\s+reporting", re.IGNORECASE)
_DCP_QUALIFIER = re.compile(
    r"\beffective\b\s*(?P<qualifier>(?:at\s+the\s+reasonable\s+assurance\s+level|to\s+provide\s+reasonable\s+assurance|"
    r"in\s+\(i\)|for\s+the\s+purpose[s]?\s+for\s+which|in\s+ensuring|in\s+providing|except|other\s+than|"
    r"because\s+of|due\s+to|subject\s+to|but\b|although|as\s+a\s+result\s+of)[^.]*)",
    re.IGNORECASE,
)
_MATERIAL_WEAKNESS = re.compile(r"material\s+weakness", re.IGNORECASE)
_REASON_QUALIFIER = re.compile(r"^(?:because\s+of|due\s+to|as\s+a\s+result\s+of)", re.IGNORECASE)
_EXCEPTION_QUALIFIER = re.compile(
    r"^(?:except|other\s+than|because\s+of|due\s+to|subject\s+to|but\b|although|as\s+a\s+result\s+of)",
    re.IGNORECASE,
)


def _concluded_by(plain: str, concluded_at: int) -> str:
    """The officers named as concluding.

    The officers named as concluding: the clause before "concluded", after
    the last comma-bounded evaluation lead-in.
    """
    before = plain[:concluded_at].rstrip()
    lead = re.split(
        r"(?i)based\s+(?:up)?on\s+(?:that|such|this|the\s+foregoing|the)\s*(?:evaluation)?,?\s*",
        before,
    )
    tail = lead[-1].strip(" ,")
    return tail[-160:]


def _extract_dcp(
    sentences: tuple[Sentence, ...], report_period: ReportPeriod
) -> tuple[tuple[DraftInstance, ...], tuple[str, ...], tuple[str, ...]]:
    """The conclusion sentences of one Controls and Procedures section.

    The conclusion sentences of one Controls and Procedures section:
    "concluded ... disclosure controls and procedures ... (not) effective",
    with the control type that the effectiveness predicate applies to read
    from the nearest control phrase before it, so an ICFR conclusion in the
    same section is never taken for the DCP one.
    """
    instances: list[DraftInstance] = []
    unrecognized: list[str] = []
    context: list[str] = []
    for sentence in sentences:
        plain = sentence.plain
        concluded = _CONCLUDED.search(plain)
        if concluded is None:
            continue
        predicate = _EFFECTIVE.search(plain, concluded.end())
        if predicate is None:
            if _DCP_PHRASE.search(plain):
                unrecognized.append(_collapse(plain)[:300])
            continue
        clause = plain[concluded.end() : predicate.start()]
        dcp_positions = [match.start() for match in _DCP_PHRASE.finditer(clause)]
        icfr_positions = [match.start() for match in _ICFR_PHRASE.finditer(clause)]
        if not dcp_positions and not icfr_positions:
            # The controls named before "concluded" ("... of the design and
            # operation of our disclosure controls and procedures ... concluded
            # that ... were effective"): nearest control phrase in the sentence.
            head = plain[: concluded.start()]
            dcp_positions = [match.start() for match in _DCP_PHRASE.finditer(head)]
            icfr_positions = [match.start() for match in _ICFR_PHRASE.finditer(head)]
        if not dcp_positions and not icfr_positions:
            unrecognized.append(_collapse(plain)[:300])
            continue
        if max(icfr_positions or [-1]) > max(dcp_positions or [-1]):
            context.append("ICFR conclusion in the same section (not this family)")
            continue
        if not sentence.terminated:
            unrecognized.append("TRUNCATED: " + _collapse(plain)[:300])
            continue
        if _NO_CONCLUSION.search(plain[max(0, concluded.start() - 48) : concluded.start()]):
            # The conclusion is what is negated: the sentence states that
            # none was reached, so no polarity is read from its predicate.
            unrecognized.append("NO_CONCLUSION: " + _collapse(plain)[:300])
            continue
        negated = bool(
            re.search(
                r"\b(?:not|no\s+longer)\s+effective|ineffective", predicate.group(0), re.IGNORECASE
            )
        )
        polarity = "NOT_EFFECTIVE" if negated else "EFFECTIVE"
        qualifier_match = _DCP_QUALIFIER.search(plain, predicate.start())
        qualifiers: list[str] = []
        if qualifier_match is not None:
            qualifier = _collapse(qualifier_match.group("qualifier"))
            qualifiers.append(qualifier[:400])
            limiting = _EXCEPTION_QUALIFIER.match(qualifier)
            if limiting is not None and not (negated and _REASON_QUALIFIER.match(qualifier)):
                polarity = f"{polarity}_QUALIFIED"
        if _MATERIAL_WEAKNESS.search(plain):
            qualifiers.append("material weakness named in the conclusion sentence")
            if not polarity.endswith("_QUALIFIED") and not negated:
                polarity = f"{polarity}_QUALIFIED"
        period_end, period_text, period_basis = _resolve_period(
            sentence, region_sentences=sentences, report_period=report_period
        )
        fields: list[DraftField] = []
        concluded_by = _concluded_by(plain, concluded.start())
        fields.append(
            DraftField(
                name="concluded_by",
                value=concluded_by,
                text=concluded_by,
                character_start=sentence.character_start,
                character_end=sentence.character_end,
            )
        )
        located = _locate(sentence, predicate.group(0))
        fields.append(
            DraftField(
                name="effectiveness_predicate",
                value=_collapse(predicate.group(0)),
                text=_collapse(predicate.group(0)),
                character_start=located[0],
                character_end=located[1],
            )
        )
        unknown: list[str] = []
        if period_end is None:
            unknown.append("evaluation_date")
        # The antecedent sentence, when the as-of clause points back to it,
        # travels with the conclusion so the reader sees the whole assertion.
        start = sentence.character_start
        if period_basis.startswith("antecedent"):
            earlier = [
                value for value in sentences if value.character_end <= sentence.character_start
            ]
            if earlier:
                start = earlier[-1].character_start
        instances.append(
            DraftInstance(
                subject="the registrant's disclosure controls and procedures",
                action="CONCLUDED",
                polarity=polarity,
                period_end=period_end,
                period_text=period_text,
                period_basis=period_basis,
                character_start=start,
                character_end=sentence.character_end,
                statement_text=_collapse(sentence.text),
                fields=tuple(fields),
                qualifiers=tuple(qualifiers),
                unknown_fields=tuple(unknown),
            )
        )
    return tuple(instances), tuple(unrecognized), tuple(dict.fromkeys(context))


# --------------------------------------------------------------------------
# Negation with an exception, shared by the two list families.

_EXCEPTION_CUE = re.compile(
    r"\b(?:except(?:\s+(?:as|that|for|with\s+respect\s+to|in\s+connection\s+with))?|other\s+than"
    r"|save\s+(?:for|that|as)|with\s+the\s+exception\s+of|excluding|but|however"
    r"|provided,?\s+(?:however,?\s+)?that)\b",
    re.IGNORECASE,
)
_FORWARD_EXCEPTION = re.compile(
    r"^(?:except|other\s+than|save|excluding|but)\s+(?:as\s+|for\s+|that\s+)?"
    r"(?:follows\b|(?:set\s+forth|described|disclosed|discussed|noted|provided|reported|listed|"
    r"shown|indicated|stated)\s+(?:below|herein|above|in\s+the\s+(?:table|following))"
    r"|the\s+following\b|below\b)",
    re.IGNORECASE,
)
"""An exception that points to the statements that follow ("except as
follows:", "other than as set forth below"): the exception is those
statements, read by the family's own rule, not an inline clause."""


@dataclass(frozen=True, slots=True)
class ExceptionClause:
    """What a none statement excepts.

    What a none statement excepts: the exception's words (for the
    qualifier) and, when the exception is stated inline rather than pointing
    to the statements that follow, the clause as its own source-ranged
    sentence, so the family's affirmative rule can read it.
    """

    text: str
    clause: Sentence | None


def _exception_clause(
    sentence: Sentence, plain: str, *, negation_start: int
) -> ExceptionClause | None:
    """The exception a none statement carries, if any.

    The exception a none statement carries, if any. The scope of a "no
    ... except ..." assertion is the negation minus its exception: the
    exception is never dropped, and a none with an exception is never a bare
    none. An inline clause is returned as a sentence over the exact source
    range of its words; "except as follows" and its kin return no clause,
    because the exception is the statements that follow.
    """
    cue = _EXCEPTION_CUE.search(plain)
    if cue is None:
        return None
    text = _collapse(plain[cue.start() : cue.start() + 200])
    if _FORWARD_EXCEPTION.match(plain[cue.start() :]):
        return ExceptionClause(text=text, clause=None)
    if cue.start() < negation_start:
        clause_plain = plain[cue.end() : negation_start]
    else:
        clause_plain = plain[cue.end() :]
    clause_plain = clause_plain.strip(" ,;:").rstrip(".").strip()
    if not clause_plain:
        return ExceptionClause(text=text, clause=None)
    start, end = _locate(sentence, clause_plain)
    if (start, end) == (sentence.character_start, sentence.character_end):
        return ExceptionClause(text=text, clause=None)
    local_start, local_end = start - sentence.character_start, end - sentence.character_start
    return ExceptionClause(
        text=text,
        clause=Sentence(
            text=sentence.text[local_start:local_end],
            character_start=start,
            character_end=end,
            terminated=True,
        ),
    )


# --------------------------------------------------------------------------
# Family 2: unregistered sales of equity securities.

_UNREGISTERED = re.compile(
    r"unregistered|not\s+registered\s+under\s+the\s+securities\s+act", re.IGNORECASE
)
_SALES_NONE = re.compile(
    r"(?:there\s+(?:were|have\s+been|was)\s+no\s+(?:sales\s+of\s+)?unregistered|"
    r"no\s+(?:sales\s+of\s+)?unregistered\s+(?:sales|equity|securities)|"
    r"(?:we|the\s+company|the\s+registrant)\s+(?:did\s+not|has\s+not|have\s+not)\s+(?:sell|sold|issue|issued)\s+any\s+"
    r"(?:unregistered|equity\s+securities\s+(?:that|which)\s+were\s+not\s+registered)|"
    r"(?:no|none\s+of\s+the)\s+(?:equity\s+)?securities\s+(?:were|was)\s+(?:sold|issued)\s+[^.]{0,80}?without\s+registration)",
    re.IGNORECASE,
)
_SALES_NONE_SCOPED = re.compile(
    r"no\s+options\s+to\s+purchase\s+shares[^.]{0,200}?\bwere\s+exercised\s+for\s+which\s+the\s+purchase\s+price\s+was\s+so\s+paid",
    re.IGNORECASE,
)
_BARE_NONE = re.compile(r"^(?:none|not\s+applicable|n/a)\.?$", re.IGNORECASE)
_SALE_VERB = re.compile(r"\b(?:issued|sold|granted)\b", re.IGNORECASE)
_SECURITY = re.compile(
    r"\b(?:shares?|units?|warrants?|notes|options|restricted\s+stock|common\s+stock|preferred\s+stock|securities)\b",
    re.IGNORECASE,
)
_EXEMPTION = re.compile(
    r"(?:exempt(?:ion)?\s+from\s+(?:the\s+)?registration|section\s+4\(a\)\(2\)|section\s+4\(2\)|regulation\s+d\b|rule\s+506|"
    r"section\s+3\(a\)\(9\)|regulation\s+s\b|private\s+placement|not\s+registered\s+under\s+the\s+securities\s+act)",
    re.IGNORECASE,
)
_QUANTITY = re.compile(r"\b(\d{1,3}(?:,\d{3})*|\d+)\s+(?:shares|units|warrants)", re.IGNORECASE)
_CONSIDERATION = re.compile(
    r"(?:for\s+(?:an\s+)?(?:aggregate\s+)?(?:purchase\s+price\s+of\s+|consideration\s+of\s+|cash\s+consideration\s+of\s+)?"
    r"\$\s?[\d,]+(?:\.\d+)?(?:\s+(?:million|billion))?|in\s+exchange\s+for\s+[^,.]{1,120})",
    re.IGNORECASE,
)
_COUNTERPARTY = re.compile(
    r"\bto\s+((?:an?|the)\s+[^,.]{3,120}?|[A-Z][^,.]{2,120}?)(?=,|\s+in\b|\s+for\b|\s+pursuant\b|\.)"
)
_FORM_8K_REFERENCE = re.compile(
    r"(?:previously\s+(?:reported|disclosed|furnished)|as\s+(?:reported|disclosed))[^.]{0,120}?"
    r"(?:current\s+report\s+on\s+)?form\s+8-k(?:[^.]{0,80}?(?:filed|dated)\s+(?:on\s+)?("
    + _DATE.pattern
    + r"))?",
    re.IGNORECASE,
)
_REPURCHASE = re.compile(
    r"repurchase|issuer\s+purchases|purchases\s+of\s+equity\s+securities|purchased\b", re.IGNORECASE
)
_PERIOD_DATE_LEAD = re.compile(
    r"(?:ended|ending|as\s+of|through|until|between)\s*(?:the\s+)?$", re.IGNORECASE
)


def _action_date(plain: str) -> re.Match[str] | None:
    """The first date in the sentence that is not a period boundary ("ended June 30, 2026").

    The first date in the sentence that is not a period boundary ("ended
    June 30, 2026"): the date of the action itself, when one is stated.
    """
    for match in _DATE.finditer(plain):
        if _PERIOD_DATE_LEAD.search(plain[max(0, match.start() - 16) : match.start()]) is None:
            return match
    return None


def _is_sale(plain: str) -> bool:
    return bool(_SALE_VERB.search(plain) and _SECURITY.search(plain) and _EXEMPTION.search(plain))


def _sale_instance(
    sentence: Sentence, sentences: tuple[Sentence, ...], report_period: ReportPeriod
) -> DraftInstance:
    """Extract one reported security sale from a sentence.

    One reported sale from a sentence -- or an exception clause read as
    one -- that names a sale verb, a security and an exemption (`_is_sale`).
    """
    plain = sentence.plain
    fields: list[DraftField] = []
    unknown: list[str] = []
    qualifiers: list[str] = []
    for name, match in (
        ("date", _action_date(plain)),
        ("quantity", _QUANTITY.search(plain)),
        ("consideration", _CONSIDERATION.search(plain)),
        ("purchaser", _COUNTERPARTY.search(plain)),
    ):
        value = _field(name, sentence, match)
        if value is None:
            unknown.append(name)
        else:
            fields.append(value)
    exemptions = [_collapse(match.group(0)) for match in _EXEMPTION.finditer(plain)]
    exemption = _field("exemption", sentence, _EXEMPTION.search(plain), value="; ".join(exemptions))
    if exemption is not None:
        fields.append(exemption)
    period_end, period_text, period_basis = _resolve_period(
        sentence, region_sentences=sentences, report_period=report_period
    )
    if period_end is None and report_period.period_end is not None:
        period_end, period_text, period_basis = (
            report_period.period_end,
            report_period.text,
            "the report period (the sentence states none)",
        )
    stated = next((value.value for value in fields if value.name == "date"), None)
    sale_date = parse_source_date(stated or "")
    if sale_date is not None and period_end is not None and sale_date > period_end:
        qualifiers.append(
            "the stated sale date is after the report period's end: a later event, "
            "not a sale in the period"
        )
    return DraftInstance(
        subject="sale of unregistered equity securities",
        action="SOLD_UNREGISTERED",
        polarity="AFFIRMATIVE",
        period_end=period_end,
        period_text=period_text,
        period_basis=period_basis,
        character_start=sentence.character_start,
        character_end=sentence.character_end,
        statement_text=_collapse(sentence.text),
        fields=tuple(fields),
        qualifiers=tuple(qualifiers),
        unknown_fields=tuple(unknown),
    )


def _extract_sales(
    sentences: tuple[Sentence, ...], region_plain: str, report_period: ReportPeriod
) -> tuple[tuple[DraftInstance, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    instances: list[DraftInstance] = []
    unrecognized: list[str] = []
    references: list[str] = []
    context: list[str] = []
    body_lines = [line for line in region_plain.split("\n") if line.strip()]
    if len(body_lines) == 1 and _BARE_NONE.match(body_lines[0].strip()):
        sentence = sentences[0]
        instances.append(
            DraftInstance(
                subject="unregistered sales of equity securities (the item's whole answer)",
                action="NONE_STATED",
                polarity="NONE",
                period_end=report_period.period_end,
                period_text="the item as a whole",
                period_basis="the report period; the item states only " + _collapse(body_lines[0]),
                character_start=sentence.character_start,
                character_end=sentence.character_end,
                statement_text=_collapse(sentence.text),
                qualifiers=("stated as " + _collapse(body_lines[0]).rstrip("."),),
                unknown_fields=() if report_period.period_end else ("period",),
            )
        )
        return tuple(instances), (), (), ()
    if _REPURCHASE.search(region_plain):
        context.append(
            "issuer repurchase disclosure present in the same item (Item 2(c); not this family)"
        )
    for sentence in sentences:
        plain = sentence.plain
        reference = _FORM_8K_REFERENCE.search(plain)
        if reference is not None:
            references.append(_collapse(reference.group(0))[:200])
            continue
        none = _SALES_NONE.search(plain)
        scoped = _SALES_NONE_SCOPED.search(plain) if none is None else None
        negation = none if none is not None else scoped
        if negation is not None:
            if not sentence.terminated:
                unrecognized.append("TRUNCATED: " + _collapse(plain)[:300])
                continue
            period_end, period_text, period_basis = _resolve_period(
                sentence, region_sentences=sentences, report_period=report_period
            )
            qualifiers: list[str] = []
            subject = "unregistered sales of equity securities"
            if scoped is not None:
                subject = (
                    "option exercises paid by delivery of shares already owned "
                    "(the issuer's stated scope)"
                )
                qualifiers.append(
                    "scope narrower than the family definition: " + _collapse(scoped.group(0))[:200]
                )
            if re.search(r"\bnor\s+did\s+we\s+repurchase|repurchase", plain, re.IGNORECASE):
                qualifiers.append("the sentence also addresses repurchases, a separate subject")
            exception = _exception_clause(sentence, plain, negation_start=negation.start())
            action, polarity = "NONE_STATED", "NONE"
            if exception is not None:
                qualifiers.append("negation with an exception: " + exception.text[:160])
                action, polarity = "NONE_STATED_WITH_EXCEPTION", "NONE_WITH_EXCEPTION"
            instances.append(
                DraftInstance(
                    subject=subject,
                    action=action,
                    polarity=polarity,
                    period_end=period_end,
                    period_text=period_text,
                    period_basis=period_basis,
                    character_start=sentence.character_start,
                    character_end=sentence.character_end,
                    statement_text=_collapse(sentence.text),
                    qualifiers=tuple(qualifiers),
                    unknown_fields=() if period_end else ("period",),
                )
            )
            # The exception is read by the same rule as a stated sale; an
            # exception the rule cannot read stays visible and unresolved.
            if exception is not None and exception.clause is not None:
                if _is_sale(exception.clause.plain):
                    instances.append(_sale_instance(exception.clause, sentences, report_period))
                else:
                    unrecognized.append("EXCEPTION: " + exception.clause.plain[:300])
            continue
        if _is_sale(plain):
            if not sentence.terminated:
                unrecognized.append("TRUNCATED: " + _collapse(plain)[:300])
                continue
            instances.append(_sale_instance(sentence, sentences, report_period))
            continue
        if _UNREGISTERED.search(plain) and not _REPURCHASE.search(plain):
            unrecognized.append(_collapse(plain)[:300])
    return tuple(instances), tuple(unrecognized), tuple(references), tuple(context)


# --------------------------------------------------------------------------
# Family 3: director and officer trading arrangements.

_ARRANGEMENT_CUE = re.compile(r"10b5-1|trading\s+arrangement|trading\s+plan", re.IGNORECASE)
_DIRECTORS_OFFICERS = re.compile(
    r"directors?\s+(?:or|and)\s+officers?|director\s+or\s+officer", re.IGNORECASE
)
_ARRANGEMENT_NONE = re.compile(
    r"(?:there\s+were\s+no|no\s+(?:director|officer)|none\s+of\s+(?:our|the\s+company[\u2019']s)\s+directors?\s+(?:or|and)\s+officers?"
    r"|(?:no|neither\s+any)\s+[^.]{0,40}?directors?\s+(?:or|and)\s+officers?)",
    re.IGNORECASE,
)
_ACTION_WORDS = re.compile(
    r"\b(adopt(?:ed|ion)|terminat(?:ed|ion)|modif(?:ied|ication)|amend(?:ed|ment))\b", re.IGNORECASE
)
_INFORMED = re.compile(
    r"\binformed\s+(?:us|the\s+company)\b|\bnotified\b|\bto\s+(?:our|the\s+company[\u2019']s)\s+knowledge\b",
    re.IGNORECASE,
)
_KIND_RULE = re.compile(r"(?<!non-)\brule\s+10b5-1", re.IGNORECASE)
_KIND_NON_RULE = re.compile(r"non-rule\s+10b5-1", re.IGNORECASE)
_PERSON_LEAD = re.compile(
    r"^On\s+(?P<date>" + _DATE.pattern + r"),\s+(?P<rest>.+)$",
    re.IGNORECASE | re.DOTALL,
)
_PERSON_TAIL = re.compile(
    r"^(?P<rest>.+?)\s+on\s+(?P<date>" + _DATE.pattern + r")(?:[,.]|$)",
    re.IGNORECASE | re.DOTALL,
)
_CREDENTIALS = ("ph.d.", "m.d.", "jr.", "sr.", "ii", "iii", "iv", "esq.", "cpa", "cfa", "j.d.")
_ACTION_VERB = re.compile(
    r"\b(adopted|terminated|modified|amended|entered\s+into)\b", re.IGNORECASE
)
_SHARES_UP_TO = re.compile(r"up\s+to\s+(\d{1,3}(?:,\d{3})*)\s+shares", re.IGNORECASE)
_DURATION = re.compile(
    r"(?:until\s+"
    + _DATE.pattern
    + r"|through\s+"
    + _DATE.pattern
    + r"|between\s+"
    + _DATE.pattern
    + r"\s+and\s+"
    + _DATE.pattern
    + r"|expir(?:es|ing|ation)\s+(?:on\s+)?"
    + _DATE.pattern
    + r")",
    re.IGNORECASE,
)
_ISSUER_SUBJECT = re.compile(
    r"^(?:we|the\s+company|the\s+registrant|our\s+board|the\s+board)\b", re.IGNORECASE
)
_PERSON_NAME = re.compile(
    r"^(?:Mr\.|Ms\.|Mrs\.|Dr\.|Messrs\.)?\s*[A-Z][\w.'\u2019-]*(?:\s+[A-Z][\w.'\u2019-]*){0,4}$"
)


def _split_person(rest: str) -> tuple[str, str, re.Match[str]] | None:
    """Identify a named officer or director and action phrase.

    `(name, role, action match)` from "<Name>, <role>, adopted ..." with
    a credential after the name kept with it; None when the subject is the
    issuer or no action verb follows.
    """
    if _ISSUER_SUBJECT.match(rest.strip()):
        return None
    action = _ACTION_VERB.search(rest)
    if action is None:
        return None
    head = rest[: action.start()].strip()
    parts = [value.strip() for value in head.split(",")]
    if not parts or not parts[0]:
        return None
    name = parts[0]
    if not _PERSON_NAME.match(name):
        return None
    index = 1
    while index < len(parts) and parts[index].casefold().rstrip(",") in _CREDENTIALS:
        name = f"{name}, {parts[index]}"
        index += 1
    role = ", ".join(value for value in parts[index:] if value)
    role = re.sub(
        r"^(?:our|the\s+company[\u2019']s|a|an)\s+", "", role, flags=re.IGNORECASE
    ).strip()
    return name, role, action


def _person_instance(
    sentence: Sentence,
    sentences: tuple[Sentence, ...],
    index: int,
    report_period: ReportPeriod,
    consumed: set[int],
) -> DraftInstance | None:
    """Extract one officer or director action from a sentence.

    One director's or officer's reported action from a sentence -- or an
    exception clause read as one -- of the shape "On <date>, <Name>, <role>,
    adopted/terminated/modified ...". The terms follow in the sentences that
    speak of this person's plan (after `index`); they travel with the
    assertion as its complete evidence and are marked consumed.
    """
    plain = sentence.plain
    person = None
    stated_date: str | None = None
    lead = _PERSON_LEAD.match(plain)
    if lead is not None:
        person = _split_person(lead.group("rest"))
        stated_date = lead.group("date")
    if person is None:
        tail = _PERSON_TAIL.match(plain)
        if tail is not None:
            person = _split_person(tail.group("rest"))
            stated_date = tail.group("date")
    if person is None or not _ARRANGEMENT_CUE.search(plain):
        return None
    name, role, action = person
    verb = action.group(1).casefold()
    action_kind = (
        "ADOPTED"
        if verb in {"adopted", "entered into"}
        else "TERMINATED"
        if verb == "terminated"
        else "MODIFIED"
    )
    kinds = []
    if _KIND_RULE.search(plain):
        kinds.append("RULE_10B5_1")
    if _KIND_NON_RULE.search(plain):
        kinds.append("NON_RULE_10B5_1")
    end_index = index
    surname = name.split(",")[0].split()[-1].rstrip(".")
    for later in range(index + 1, len(sentences)):
        later_plain = sentences[later].plain
        next_lead = _PERSON_LEAD.match(later_plain)
        if next_lead is not None and _split_person(next_lead.group("rest")):
            break
        if re.search(re.escape(surname) + r"[\u2019']s?(?!\w)", later_plain) or re.match(
            r"^(?:This|The)\s+(?:trading\s+)?(?:plan|arrangement)",
            later_plain,
            re.IGNORECASE,
        ):
            end_index = later
            consumed.add(later)
            continue
        break
    terms_text = " ".join(
        [plain, *(sentences[value].plain for value in range(index + 1, end_index + 1))]
    )
    fields: list[DraftField] = []
    unknown: list[str] = []
    fields.append(
        DraftField(
            name="person",
            value=name,
            text=name,
            character_start=_locate(sentence, name)[0],
            character_end=_locate(sentence, name)[1],
        )
    )
    if role:
        fields.append(
            DraftField(
                name="role",
                value=role,
                text=role,
                character_start=_locate(sentence, role)[0],
                character_end=_locate(sentence, role)[1],
            )
        )
    else:
        unknown.append("role")
    parsed_date = parse_source_date(stated_date or "")
    if stated_date and parsed_date is not None:
        located = _locate(sentence, stated_date)
        fields.append(
            DraftField(
                name="action_date",
                value=parsed_date.isoformat(),
                text=stated_date,
                character_start=located[0],
                character_end=located[1],
                basis="stated in the sentence",
            )
        )
    else:
        unknown.append("action_date")
    if kinds:
        fields.append(
            DraftField(
                name="arrangement_kind",
                value="+".join(kinds),
                text=" / ".join(kinds),
                character_start=sentence.character_start,
                character_end=sentence.character_end,
            )
        )
    else:
        unknown.append("arrangement_kind")
    terms_end = max(sentence.character_end, sentences[end_index].character_end)
    shares = [match.group(1) for match in _SHARES_UP_TO.finditer(terms_text)]
    if shares:
        fields.append(
            DraftField(
                name="shares_up_to",
                value="; ".join(shares),
                text="; ".join(shares),
                character_start=sentence.character_start,
                character_end=terms_end,
            )
        )
    duration = _DURATION.search(terms_text)
    if duration is not None:
        fields.append(
            DraftField(
                name="duration",
                value=_collapse(duration.group(0)),
                text=_collapse(duration.group(0)),
                character_start=sentence.character_start,
                character_end=terms_end,
            )
        )
    qualifiers: list[str] = []
    if "former" in role.casefold():
        qualifiers.append("the role is stated as former")
    if (
        re.search(r"will\s+terminate|terminates\s+on", terms_text, re.IGNORECASE)
        and action_kind != "TERMINATED"
    ):
        qualifiers.append("the stated termination is a term of the plan, not a termination action")
    statement = " ".join(
        [sentence.text, *(sentences[value].text for value in range(index + 1, end_index + 1))]
    )
    return DraftInstance(
        subject=f"{name} ({role})" if role else name,
        action=action_kind,
        polarity="AFFIRMATIVE",
        period_end=report_period.period_end,
        period_text=report_period.text,
        period_basis="the report's last fiscal quarter (the action date is stated)",
        character_start=sentence.character_start,
        character_end=terms_end,
        statement_text=_collapse(statement),
        fields=tuple(fields),
        qualifiers=tuple(qualifiers),
        unknown_fields=tuple(unknown),
    )


def _extract_arrangements(
    sentences: tuple[Sentence, ...], report_period: ReportPeriod
) -> tuple[tuple[DraftInstance, ...], tuple[str, ...], tuple[str, ...]]:
    instances: list[DraftInstance] = []
    unrecognized: list[str] = []
    context: list[str] = []
    consumed: set[int] = set()
    for index, sentence in enumerate(sentences):
        if index in consumed:
            continue
        plain = sentence.plain
        cue = bool(_ARRANGEMENT_CUE.search(plain) or _DIRECTORS_OFFICERS.search(plain))
        if not cue:
            continue
        if not sentence.terminated:
            unrecognized.append("TRUNCATED: " + _collapse(plain)[:300])
            consumed.add(index)
            continue
        person = _person_instance(sentence, sentences, index, report_period, consumed)
        if person is not None:
            instances.append(person)
            consumed.add(index)
            continue
        none = _ARRANGEMENT_NONE.search(plain)
        if none is not None and _ACTION_WORDS.search(plain) and (_ARRANGEMENT_CUE.search(plain)):
            period_end, period_text, period_basis = _resolve_period(
                sentence, region_sentences=sentences, report_period=report_period
            )
            qualifiers = []
            exception = _exception_clause(sentence, plain, negation_start=none.start())
            if exception is not None:
                qualifiers.append("negation with an exception: " + exception.text[:160])
            informed = _INFORMED.search(plain)
            if informed is not None:
                qualifiers.append("as informed to the registrant: " + _collapse(informed.group(0)))
            actions = sorted(
                {match.group(1).casefold()[:5] for match in _ACTION_WORDS.finditer(plain)}
            )
            kinds = []
            if _KIND_RULE.search(plain):
                kinds.append("RULE_10B5_1")
            if _KIND_NON_RULE.search(plain):
                kinds.append("NON_RULE_10B5_1")
            fields = [
                DraftField(
                    name="actions_negated",
                    value="+".join(actions),
                    text="+".join(actions),
                    character_start=sentence.character_start,
                    character_end=sentence.character_end,
                ),
                DraftField(
                    name="arrangement_kind",
                    value="+".join(kinds) if kinds else None,
                    text=" / ".join(kinds),
                    character_start=sentence.character_start,
                    character_end=sentence.character_end,
                ),
            ]
            instances.append(
                DraftInstance(
                    subject="directors and officers (as defined in Rule 16a-1(f))",
                    action="NONE_STATED" if exception is None else "NONE_STATED_WITH_EXCEPTION",
                    polarity="NONE" if exception is None else "NONE_WITH_EXCEPTION",
                    period_end=period_end,
                    period_text=period_text,
                    period_basis=period_basis,
                    character_start=sentence.character_start,
                    character_end=sentence.character_end,
                    statement_text=_collapse(sentence.text),
                    fields=tuple(fields),
                    qualifiers=tuple(qualifiers),
                    unknown_fields=(() if period_end else ("period",))
                    + (() if kinds else ("arrangement_kind",)),
                )
            )
            consumed.add(index)
            # An inline exception is read by the same rule as a stated
            # action; one the rule cannot read stays visible and unresolved.
            if exception is not None and exception.clause is not None:
                excepted = _person_instance(
                    exception.clause, sentences, index, report_period, consumed
                )
                if excepted is None:
                    unrecognized.append("EXCEPTION: " + exception.clause.plain[:300])
                else:
                    instances.append(excepted)
            continue
        if _ARRANGEMENT_CUE.search(plain):
            unrecognized.append(_collapse(plain)[:300])
            consumed.add(index)
    return tuple(instances), tuple(unrecognized), tuple(context)


# --------------------------------------------------------------------------
# Document-level extraction.


_SECTION_HEADINGS = frozenset({"PART", "ITEM", "NOTE", "UNIT", "TITLE"})


def _body_has_text(text: str, region: ItemRegion, structure: DocumentStructure) -> bool:
    """Whether the region governs any text that is not a heading line."""
    body = text[region.body_start : region.body_end]
    for heading in structure.headings:
        if (
            heading.kind in _SECTION_HEADINGS
            and region.body_start <= heading.character_start < region.body_end
        ):
            local = heading.character_start - region.body_start
            body = (
                body[:local]
                + " " * (heading.character_end - heading.character_start)
                + body[local + (heading.character_end - heading.character_start) :]
            )
    return bool(_plain(body))


def _regions_for(
    spec: RegionSpec, structure: DocumentStructure, text: str
) -> tuple[tuple[ItemRegion, ...], tuple[ItemRegion, ...]]:
    """`(body regions, contents-only regions)` of the spec's item on this document.

    `(body regions, contents-only regions)` of the spec's item on this
    document: the item number must match; the title decides when it names
    the item, else the part does (a 10-Q has an Item 4 in both parts).
    """
    body: list[ItemRegion] = []
    contents: list[ItemRegion] = []
    if spec.note_keywords:
        for region in _note_regions(spec, structure, text):
            (body if _body_has_text(text, region, structure) else contents).append(region)
        return tuple(body), tuple(contents)
    for region in structure.item_regions():
        if region.item != spec.item:
            continue
        title = region.title.casefold()
        if title:
            if not any(keyword in title for keyword in spec.title_keywords):
                continue
        elif region.part is not None and region.part != spec.part:
            continue
        (body if _body_has_text(text, region, structure) else contents).append(region)
    return tuple(body), tuple(contents)


def _note_regions(
    spec: RegionSpec, structure: DocumentStructure, text: str
) -> tuple[ItemRegion, ...]:
    """Find qualifying notes beyond the document cover.

    The notes past the cover whose heading carries one of the spec's
    note words, each as a region bounded by the next note, item or part
    heading, in source order; the item is the spec's, the title the
    note's heading.
    """
    regions: list[ItemRegion] = []
    headings = structure.headings
    for index, heading in enumerate(headings):
        if heading.kind != "NOTE" or heading.character_start < structure.cover_end:
            continue
        title = _plain(heading.text)
        if not any(keyword in title.casefold() for keyword in spec.note_keywords):
            continue
        body_end = len(text)
        for later in headings[index + 1 :]:
            if later.kind in {"PART", "ITEM", "NOTE"}:
                body_end = later.character_start
                break
        regions.append(
            ItemRegion(
                part=spec.part,
                item=spec.item,
                title=title[:160],
                heading_start=heading.character_start,
                heading_end=heading.character_end,
                body_start=heading.character_end,
                body_end=body_end,
            )
        )
    return tuple(regions)


def _amendment_scope(
    text: str, structure: DocumentStructure
) -> tuple[tuple[str, ...] | None, tuple[int, int] | None, str | None]:
    """An amendment's own statement of what it amends.

    An amendment's own statement of what it amends: the items its
    explanatory note says it is filed solely to provide, the note's range,
    and the original filing it names. None when the amendment states no
    scope, which is uncertainty, never applicability.
    """
    if not structure.document_type.endswith("/A"):
        return None, None, None
    note = _EXPLANATORY_NOTE.search(text)
    window = text[note.start() : note.start() + 3000] if note is not None else text[:6000]
    solely = _AMENDMENT_SOLELY.search(window)
    if solely is None:
        return None, None, None
    scope = tuple(
        dict.fromkeys(
            f"PART {part.upper()} ITEM {item.upper()}"
            for part, item in _AMENDMENT_ITEM.findall(solely.group("scope"))
        )
    )
    base = note.start() if note is not None else 0
    original = _ORIGINAL_FILING.search(window)
    named = (
        None
        if original is None
        else f"the original Form {original.group('kind').upper()} filed {original.group('date')}"
    )
    return scope, (base + solely.start(), base + solely.end()), named


def extract_document(text: str, structure: DocumentStructure) -> DocumentExtraction:
    """Every family's observation on one canonical filing text, from its own sections.

    Every family's observation on one canonical filing text, from its own
    sections. Guard order: the document's form and completeness, the family's
    applicability to the form, the amendment's stated scope, the region, then
    the sentences.
    """
    form = structure.family_form
    report_period = structure.report_period
    observations: list[DraftObservation] = []
    if form not in {"10-K", "10-Q"}:
        return DocumentExtraction((), None, None)
    amendment_scope, scope_range, original = _amendment_scope(text, structure)
    incomplete: str | None = None
    if not structure.document_type.endswith("/A"):
        for definition in DEFINITIONS:
            for spec in definition.regions:
                if spec.form == form and spec.mandatory:
                    body, _contents = _regions_for(spec, structure, text)
                    if not body and not _cue_anywhere(definition.family, text):
                        incomplete = (
                            f"{spec.part} Item {spec.item} is required on Form {form} and "
                            "neither its heading nor its statement is in the canonical text: "
                            "the source as admitted is incomplete for these families"
                        )
    for definition in DEFINITIONS:
        specs = [spec for spec in definition.regions if spec.form == form]
        if not specs:
            observations.append(
                DraftObservation(
                    family=definition.family,
                    rule_id=definition.rule_id,
                    state=TypedDisclosureState.NOT_APPLICABLE,
                    reason=(
                        f"the family is defined for {', '.join(definition.forms)}; "
                        f"this is a {structure.document_type}"
                    ),
                    part=None,
                    item=None,
                    region_title="",
                    report_period=report_period,
                )
            )
            continue
        spec = specs[0]
        body, contents = _regions_for(spec, structure, text)
        if amendment_scope is not None:
            wanted = f"{spec.part} ITEM {spec.item}"
            if wanted not in amendment_scope and not body:
                observations.append(
                    DraftObservation(
                        family=definition.family,
                        rule_id=definition.rule_id,
                        state=TypedDisclosureState.REFERENCE_REQUIRED,
                        reason=(
                            "the amendment states it is filed solely for "
                            f"{', '.join(amendment_scope) or 'other items'}; {spec.part} Item "
                            f"{spec.item} is outside that scope and the information stays in "
                            + (original or "the original filing, which is not admitted")
                        ),
                        part=spec.part,
                        item=spec.item,
                        region_title="",
                        report_period=report_period,
                        references=(original or "the original filing (not admitted)",),
                        scope_range=scope_range,
                    )
                )
                continue
        if not body:
            if structure.document_type.endswith("/A"):
                state, reason = (
                    TypedDisclosureState.AMBIGUOUS,
                    f"an amendment with no stated scope and no {spec.part} Item {spec.item}: "
                    "whether the item was amended or omitted requires reading the original filing",
                )
            elif incomplete is not None:
                state, reason = TypedDisclosureState.SOURCE_UNAVAILABLE, incomplete
            elif spec.mandatory:
                state, reason = (
                    TypedDisclosureState.SOURCE_UNAVAILABLE,
                    f"{spec.part} Item {spec.item} is required on Form {form} and is not in "
                    "the canonical text",
                )
            elif spec.note_keywords:
                state, reason = (
                    TypedDisclosureState.NOT_FOUND,
                    "no note headed by "
                    + ", ".join(repr(word) for word in spec.note_keywords)
                    + f" in the canonical text of {spec.part} Item {spec.item}; the statement "
                    "may sit under another heading, and its absence here is not an explicit none",
                )
            else:
                state, reason = (
                    TypedDisclosureState.NOT_FOUND,
                    f"no {spec.part} Item {spec.item} in the canonical text; Form {form} lets a "
                    "registrant omit an inapplicable or negative Part II item, and an omission is "
                    "not an explicit none" + ("; contents entries only" if contents else ""),
                )
            observations.append(
                DraftObservation(
                    family=definition.family,
                    rule_id=definition.rule_id,
                    state=state,
                    reason=reason,
                    part=spec.part,
                    item=spec.item,
                    region_title="",
                    report_period=report_period,
                    inspected_ranges=tuple(
                        (value.heading_start, value.body_end) for value in contents[:4]
                    ),
                )
            )
            continue
        observations.append(
            _observe_regions(definition, spec, body, text, structure, report_period, incomplete)
        )
    return DocumentExtraction(tuple(observations), incomplete, amendment_scope)


# --------------------------------------------------------------------------
# Family 4: customer concentration (v4).

_CUSTOMER = re.compile(
    r"\b(?:customers?|bottlers?|distributors?|clients?|wholesalers?|retailers?|resellers?|"
    r"purchasers?|payers?|partners?)\b",
    re.IGNORECASE,
)
_SHARE = re.compile(
    r"\b(?:\d{1,2}(?:\.\d+)?\s?(?:%|percent)|ten\s+percent)"
    r"(?:\s+(?:or\s+more|or\s+greater|or\s+less))?",
    re.IGNORECASE,
)
_REVENUE_BASE = re.compile(
    r"\b(?:revenues?|sales|receivables?|billings|net\s+operating\s+revenues?)\b", re.IGNORECASE
)
_CONCENTRATION_ABSENT = re.compile(
    r"\b(?:no|none\s+of\s+(?:our|the|its))\s+(?:single\s+|one\s+|individual\s+|major\s+|"
    r"significant\s+|external\s+|direct\s+)?(?:customers?|bottlers?|distributors?|clients?|"
    r"purchasers?)(?:\s+or\s+\w+)?\s+(?:individually\s+|that\s+)?(?:accounted|accounts|"
    r"represented|represents|comprised|comprises|constituted|exceeded|contributed|generated|"
    r"had|has|have|was|were|is|are)\b"
    r"|\b(?:do|does|did)\s+not\s+(?:rely|depend)\s+(?:on|upon)\s+any\s+"
    r"(?:single\s+|one\s+|major\s+|significant\s+)?(?:customers?|clients?)\b",
    re.IGNORECASE,
)
"""An absence of concentration as the source states it -- the customer
word followed by what it did not do: "No bottlers or customers represented
10% or more", "no single customer accounted for", "We do not rely on any
major customers". "There were no revenues from external customers in the
U.S." names no customer's share and is not read."""
_CONCENTRATION_VERB = re.compile(
    r"\b(?:accounted|represented|comprised|constituted|exceeded|contributed|generated|"
    r"rely|relies|relied|depend|depends|depended|concentrat\w+|significant|major|"
    r"individually|collectively|approximately|aggregate)\b",
    re.IGNORECASE,
)
_COUNTED_CUSTOMER = re.compile(
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+|single|no|any|major|"
    r"significant|largest|principal|individual|each|a|an|another|other|top\s+\w+)\s+"
    r"(?:\w+\s+){0,2}?(?:customers?|bottlers?|distributors?|clients?|wholesalers?|"
    r"retailers?|resellers?|purchasers?|payers?|partners?)\b",
    re.IGNORECASE,
)
"""A share is a customer concentration when the source counts or names
the customers it belongs to ("one bottler", "two direct customers", "no
single customer"); a share of "customers headquartered outside the
United States" is a geographic statement the rule leaves visible."""
_PERIOD_WORDS = re.compile(
    r"\b(?:fiscal|year|years|quarter|months|period|periods|ended|ending|as\s+of)\b", re.IGNORECASE
)


def _extract_concentration(
    sentences: tuple[Sentence, ...], report_period: ReportPeriod
) -> tuple[tuple[DraftInstance, ...], tuple[str, ...], tuple[str, ...]]:
    """The customer-concentration sentences of one segment, revenue or concentration note.

    The customer-concentration sentences of one segment, revenue or
    concentration note: a customer word beside a percentage of revenue,
    sales or receivables (`STATED`), or an explicit absence at a threshold
    or of reliance (`ABSENT`). A share beside a customer word but no
    revenue base, or a customer sentence with a threshold and no absence
    or verb, is kept visible as unrecognised. Nothing names the customer
    or sums the shares.
    """
    instances: list[DraftInstance] = []
    unrecognized: list[str] = []
    context: list[str] = []
    for sentence in sentences:
        plain = sentence.plain
        if _CUSTOMER.search(plain) is None:
            continue
        shares = list(_SHARE.finditer(plain))
        absent = _CONCENTRATION_ABSENT.search(plain)
        # The base is the revenue word the share is of ("22% of total
        # revenue"): the first after the first share, else the first at all.
        base = (_REVENUE_BASE.search(plain, shares[0].end()) if shares else None) or (
            _REVENUE_BASE.search(plain)
        )
        if absent is None and not shares:
            continue
        if not sentence.terminated:
            unrecognized.append("TRUNCATED: " + _collapse(plain)[:300])
            continue
        if absent is not None and (base is not None or shares or _CONCENTRATION_VERB.search(plain)):
            polarity, action = "ABSENT", "CONCENTRATION_ABSENT"
        elif (
            shares
            and base is not None
            and _CONCENTRATION_VERB.search(plain)
            and _COUNTED_CUSTOMER.search(plain) is not None
        ):
            polarity, action = "STATED", "CONCENTRATION_STATED"
        else:
            unrecognized.append(_collapse(plain)[:300])
            continue
        period_end, period_text, period_basis = _resolve_period(
            sentence, region_sentences=sentences, report_period=report_period
        )
        if period_end is None and _PERIOD_WORDS.search(plain) is not None:
            period_text = period_text or "a period the sentence names in its own words"
            period_basis = "stated in the sentence; not resolved to a date"
        fields: list[DraftField] = []
        for position, share in enumerate(shares[:8], start=1):
            field = _field(f"share_{position}", sentence, share)
            if field is not None:
                fields.append(field)
        if absent is not None and shares:
            threshold = _field("threshold", sentence, shares[0])
            fields = [] if threshold is None else [threshold]
        base_field = _field("base", sentence, base)
        if base_field is not None:
            fields.append(base_field)
        qualifiers: list[str] = []
        exception = _EXCEPTION_CUE.search(plain)
        if exception is not None and absent is not None:
            qualifiers.append(_collapse(plain[exception.start() :])[:400])
            polarity = "ABSENT_QUALIFIED"
        instances.append(
            DraftInstance(
                subject="customer concentration of the registrant's "
                + (_collapse(base.group(0)).lower() if base is not None else "revenue"),
                action=action,
                polarity=polarity,
                period_end=period_end,
                period_text=period_text,
                period_basis=period_basis,
                character_start=sentence.character_start,
                character_end=sentence.character_end,
                statement_text=_collapse(sentence.text),
                fields=tuple(fields),
                qualifiers=tuple(qualifiers),
                unknown_fields=("period",) if period_end is None else (),
            )
        )
    return tuple(instances), tuple(unrecognized), tuple(dict.fromkeys(context))


# --------------------------------------------------------------------------
# Family 5: the cybersecurity-effect statement of Item 1C (v4).

_CYBER_SUBJECT = re.compile(
    r"\b(?:cybersecurity|cyber|information\s+security|data\s+security)\s+"
    r"(?:threats?|incidents?|risks?|attacks?|events?)\b"
    r"|\brisks?\s+from\s+(?:known\s+)?cybersecurity\s+threats?\b",
    re.IGNORECASE,
)
_MATERIALLY_AFFECT = re.compile(
    r"\bmaterially\s+affect(?:ed|s|ing)?\b|\bmaterial(?:ly)?\s+(?:adverse\s+)?"
    r"(?:effect|impact)s?\b",
    re.IGNORECASE,
)
_CLAUSE_TURN = re.compile(
    r";\s+|,?\s+(?:but|although|though|while|whereas)\s+|,?\s+however\s+(?!,)",
    re.IGNORECASE,
)
"""Where a sentence turns to another assertion: a negation before the turn
says nothing about the clause after it. A "however" set off by commas
("no such incident, however, has ...") is a parenthetical, not a turn."""
_BELIEF = re.compile(
    r"\b(?:aware|believe|believes|belief|knowledge|identified|determined|concluded)\b",
    re.IGNORECASE,
)
"""An effect stated through the registrant's awareness or belief ("we are
not aware of", "we do not believe") is a qualified statement: what the
registrant knows, not what happened."""
_CORRELATIVE = re.compile(r"\bnot\s+(?:only|just|merely|simply)\b", re.IGNORECASE)
"""'not only ... but also': a correlative that asserts both halves; its
"not" negates nothing and its "but" excepts nothing."""
_CYBER_NO_CONCLUSION = re.compile(
    r"\b(?:unable|not\s+(?:yet\s+)?(?:able|been\s+able|in\s+a\s+position))\s+to\s+"
    r"(?:determine|assess|conclude|estimate|ascertain|quantify)\b"
    r"|\b(?:cannot|can\s+not|could\s+not)\s+(?:yet\s+)?(?:determine|assess|conclude|"
    r"estimate|ascertain|quantify)\b"
    r"|\b(?:have|has|had)\s+not\s+(?:yet\s+)?(?:determined|assessed|concluded|ascertained)\b"
    r"|\bnot\s+yet\s+(?:determined|known|assessed)\b|\bno\s+determination\b"
    r"|\b(?:remains?|remained)\s+(?:under\s+(?:assessment|evaluation|review|investigation)|"
    r"to\s+be\s+(?:determined|assessed))\b|\bundetermined\b",
    re.IGNORECASE,
)
"""The registrant says it has not concluded: an inability to determine,
a determination not yet made, an assessment still open. No polarity is
read from such a sentence; it stays visible as no conclusion."""
_NEGATION_BEFORE = re.compile(
    r"\b(?:not|no|none|nothing|never|neither|nor|without|cannot)\b|\bn\'t\b|\bunaware\b",
    re.IGNORECASE,
)
_INERT_NEGATION = re.compile(
    r"\b(?:including\s+)?(?:but\s+)?not\s+limited\s+to\b|\bwithout\s+limitation\b"
    r"|\bno\s+later\s+than\b|\bnot?\s+(?:less|more|fewer|earlier)\s+than\b",
    re.IGNORECASE,
)
"""Idioms whose negative word negates no predicate."""
_FINITE_AUXILIARY = re.compile(
    r"\b(?:is|are|was|were|be|been|being|has|have|had|does|do|did|will|would|could|can|"
    r"cannot|may|might|shall|should|must)\b",
    re.IGNORECASE,
)
_PREDICATE_CHAIN_WORDS = frozenset(
    {
        "has",
        "have",
        "had",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "does",
        "do",
        "did",
        "will",
        "would",
        "hasn't",
        "haven't",
        "hadn't",
        "isn't",
        "aren't",
        "wasn't",
        "weren't",
        "doesn't",
        "don't",
        "didn't",
        "not",
        "never",
        "nor",
        "neither",
        "none",
        "nothing",
        "yet",
        "ever",
        "also",
        "otherwise",
        "previously",
        "historically",
        "since",
        "already",
        "subsequently",
        "then",
        "that",
        "which",
        "who",
        "and",
        "or",
        "to",
        "date",
        "it",
        "they",
        "we",
        "this",
        "each",
    }
)
"""The function words a verb group is made of: walking back from the
effect words over them finds the predicate's own auxiliaries, negation
and adverbs; the first content word ends the group."""
_CYBER_SUBJECT_AHEAD = (
    r"(?=(?:cybersecurity|cyber|information\s+security|data\s+security)\s+"
    r"(?:threats?|incidents?|risks?|attacks?|events?)\b"
    r"|risks?\s+from\s+(?:known\s+)?cybersecurity\s+threats?\b)"
)
_NEGATIVE_SUBJECT = re.compile(
    r"\b(?:no|none\s+of(?:\s+(?:the|these|those|such|our|its|their))?|neither|"
    r"not\s+(?:one|any)\s+of(?:\s+(?:the|these|those|such|our|its|their))?)\s+"
    r"(?!(?:knowledge|awareness|indication|evidence|assurance|reason)\b)"
    r"(?:[\w-]+\s+){0,3}?" + _CYBER_SUBJECT_AHEAD,
    re.IGNORECASE,
)
"""A negative determiner on the cyber subject itself ("no cybersecurity
incident", "none of the cybersecurity incidents", "neither ... threats");
a negated awareness noun ("no knowledge of any ...") is the verb rule's."""
_NEGATED_GOVERNING_VERB = re.compile(
    r"(?:\b(?:has|have|had|is|are|was|were|do|does|did)\s+)?(?:\bnot\b|\bnever\b|n't\b|\bno\b)"
    r"\s+(?:been\s+|yet\s+|ever\s+|otherwise\s+|previously\s+)?"
    r"\b(?P<verb>experienced|experience|encountered|encounter|suffered|suffer|sustained|"
    r"sustain|undergone|had|have|seen|aware|believe|believes|belief|knowledge|identified|"
    r"identify|determined|determine|concluded|conclude|detected|detect|discovered|"
    r"discover|observed|observe)\b",
    re.IGNORECASE,
)
"""A negated verb whose object may be the cyber subject: an experience
verb ("have not experienced any cybersecurity incident that has
materially affected") negates the incident's existence -- a negative
subject in another shape; an awareness verb ("are not aware of", "have
not identified") states what the registrant knows, so the reading is
qualified. Nothing else is bound: "did not prevent" governs prevention."""
_AWARENESS_VERBS = frozenset(
    {
        "aware",
        "believe",
        "believes",
        "belief",
        "knowledge",
        "identified",
        "identify",
        "determined",
        "determine",
        "concluded",
        "conclude",
        "detected",
        "detect",
        "discovered",
        "discover",
        "observed",
        "observe",
    }
)
_BRIDGE_WORDS = frozenset(
    {
        "of",
        "that",
        "whether",
        "any",
        "a",
        "an",
        "such",
        "the",
        "these",
        "those",
        "our",
        "its",
        "their",
        "currently",
        "presently",
        "to",
        "date",
        "there",
        "is",
        "are",
        "was",
        "were",
        "has",
        "have",
        "had",
        "been",
        "known",
        "previous",
        "prior",
        "other",
        "material",
        "significant",
        "related",
        "actual",
        "recent",
        "specific",
        "reported",
        "new",
        "additional",
        "identified",
        "particular",
        "past",
        "risks",
        "risk",
        "from",
        "or",
        "potential",
        "including",
        "current",
        "existing",
        "ongoing",
        "individual",
        "individually",
        "aggregate",
        "in",
        "and",
    }
)
"""The words that may stand between a governing verb and its cyber object
("aware of any", "believe that there are currently any known")."""
_LIKELY = re.compile(r"\breasonably\s+likely\b|\bcould\b|\bmay\b|\bmight\b", re.IGNORECASE)
_HAVE_AFFECTED = re.compile(
    r"\b(?:have|has|had)\s+(?:not\s+)?(?:\w+\s+){0,2}?materially\s+affected\b"
    r"|\b(?:did|does|do)\s+not\s+(?:\w+\s+){0,3}?material(?:ly)?\b"
    r"|\bnot\s+(?:been\s+)?materially\s+(?:affected|impacted)\b"
    r"|\bwere\s+not\s+material\b|\bhad\s+(?:a|any)\s+material\b"
    r"|(?<!\bbe )materially\s+affected\b",
    re.IGNORECASE,
)
"""The past-tense or present-perfect clause Item 106(b)(2) asks for: what
has happened, as distinct from what could ("could be materially affected"
is the modal passive, a risk statement)."""
_CYBER_PERIOD = re.compile(
    r"\b(?:during|in|over|for)\s+(?:the\s+)?(?:past|last|prior|preceding|previous)\s+"
    r"(?:\w+\s+)?(?:fiscal\s+)?(?:years?|quarters?|months?|periods?)\b"
    r"|\bto\s+date\b|\bas\s+of\s+the\s+date\s+of\s+this\s+(?:report|filing)\b"
    r"|\b(?:during|in)\s+(?:fiscal\s+)?(?:year\s+)?\d{4}\b"
    r"|\b(?:during|in)\s+the\s+(?:first|second|third|fourth)\s+(?:fiscal\s+)?quarter\s+of\s+"
    r"(?:fiscal\s+)?(?:year\s+)?\d{4}\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class _NegationScope:
    """What the rule proved about the negations before the effect predicate.

    What the rule proved about the negations before the effect predicate:
    `binding` names where the polarity was read -- `NONE` (no negation in
    the clause), `PREDICATE` (inside the predicate's own verb group),
    `SUBJECT` (a negative determiner on the cyber subject governing the
    predicate), `VERB` (an awareness or experience verb whose object is the
    cyber subject) -- or None when a negation stands in the clause that
    none of these binds, so the polarity is unproved.
    """

    binding: str | None
    negated: bool
    through_awareness: bool = False
    verb: re.Match[str] | None = None
    negation_start: int = 0
    negation_end: int = 0


def _mask(text: str, *patterns: re.Pattern[str]) -> str:
    """The text with every match of the patterns blanked at its own offsets."""
    out = text
    for pattern in patterns:
        out = pattern.sub(lambda match: " " * (match.end() - match.start()), out)
    return out


def _predicate_chain_start(masked: str, clause_start: int, predicate_start: int) -> int:
    """Where the effect predicate's own verb group begins.

    Where the effect predicate's own verb group begins: the function
    words (auxiliaries, negation, adverbs, a relative pronoun or a
    coordinator) standing immediately before the effect words, up to the
    first content word or clause break.
    """
    start = predicate_start
    for token in reversed(list(re.finditer(r"\S+", masked[clause_start:predicate_start]))):
        if ";" in token.group(0):
            break
        word = token.group(0).strip(",.:()'\"").lower()
        if word not in _PREDICATE_CHAIN_WORDS:
            break
        start = clause_start + token.start()
    return start


def _governs(masked: str, subject_end: int, chain_start: int, *, allow_nor: bool) -> bool:
    """Check whether a cyber subject governs an effect predicate.

    Whether the cyber subject ending at `subject_end` is the subject the
    predicate group starting at `chain_start` describes: nothing between
    them turns the clause, negates, or opens another finite verb group, and
    the gap is short (a parenthetical, a modifier, a relative pronoun).
    """
    if subject_end > chain_start:
        return False
    between = masked[subject_end:chain_start]
    if _CLAUSE_TURN.search(between) is not None or _FINITE_AUXILIARY.search(between):
        return False
    negations = [
        m
        for m in _NEGATION_BEFORE.finditer(between)
        if not (allow_nor and m.group(0).lower() == "nor")
    ]
    return not negations and len(between.split()) <= 16


def _negation_scope(masked: str, *, clause_start: int, predicate: re.Match[str]) -> _NegationScope:
    """Bind clause negations to the predicates they govern.

    Bind the negations of the clause to what they govern, in the order
    the bindings can be proved: the predicate's own verb group; a negative
    determiner on the cyber subject that governs the predicate; a negated
    awareness or experience verb whose object is the cyber subject the
    predicate describes. A clause with no negation reads affirmative; a
    clause with a negation none of these binds is unproved.
    """
    chain_start = _predicate_chain_start(masked, clause_start, predicate.start())
    inside = _NEGATION_BEFORE.search(masked, chain_start, predicate.end())
    if inside is not None:
        return _NegationScope(
            "PREDICATE", True, negation_start=inside.start(), negation_end=inside.end()
        )
    stray = _NEGATION_BEFORE.search(masked, clause_start, chain_start)
    if stray is None:
        return _NegationScope("NONE", False)
    for determiner in _NEGATIVE_SUBJECT.finditer(masked, clause_start, chain_start):
        subject = _CYBER_SUBJECT.match(masked, determiner.end())
        if subject is None:
            continue
        allow_nor = determiner.group(0).lower().startswith("neither")
        if _governs(masked, subject.end(), chain_start, allow_nor=allow_nor):
            return _NegationScope(
                "SUBJECT", True, negation_start=determiner.start(), negation_end=subject.end()
            )
    governing = list(_NEGATED_GOVERNING_VERB.finditer(masked, clause_start, chain_start))
    # Every negation of the clause must sit inside a governing verb's own
    # group ("have not identified ... and have not identified ..."), or the
    # clause holds one the rule has not read.
    unbound = [
        m
        for m in _NEGATION_BEFORE.finditer(masked, clause_start, chain_start)
        if not any(verb.start() <= m.start() < verb.end() for verb in governing)
    ]
    if not unbound:
        for verb in reversed(governing):
            awareness = verb.group("verb").lower() in _AWARENESS_VERBS
            subject = _CYBER_SUBJECT.search(masked, verb.end(), chain_start)
            if subject is not None:
                bridge = masked[verb.end() : subject.start()].replace(",", " ").split()
                if (
                    len(bridge) <= 10
                    and all(word.lower() in _BRIDGE_WORDS for word in bridge)
                    and _governs(masked, subject.end(), chain_start, allow_nor=False)
                ):
                    return _NegationScope(
                        "VERB",
                        True,
                        through_awareness=awareness,
                        verb=verb,
                        negation_start=verb.start(),
                        negation_end=subject.end(),
                    )
            # A belief verb's complement ("do not believe that any of these
            # incidents have materially affected") is what is not believed,
            # whatever names the subject inside it.
            complement = re.compile(r"\s+(?:that|whether)\b", re.IGNORECASE).match(
                masked, verb.end()
            )
            if (
                awareness
                and complement is not None
                and _governs(masked, complement.end(), chain_start, allow_nor=False)
            ):
                return _NegationScope(
                    "VERB",
                    True,
                    through_awareness=True,
                    verb=verb,
                    negation_start=verb.start(),
                    negation_end=complement.end(),
                )
    return _NegationScope(None, False, negation_start=stray.start(), negation_end=stray.end())


def _extract_cyber_effect(
    sentences: tuple[Sentence, ...], report_period: ReportPeriod
) -> tuple[tuple[DraftInstance, ...], tuple[str, ...], tuple[str, ...]]:
    """Classify a stated cybersecurity effect without inferring unresolved polarity.

    The sentence of Item 1C that states whether cybersecurity threats or
    incidents have materially affected the registrant: `NOT_AFFECTED` when
    a negation the rule can bind governs the effect (the predicate's own
    verb group, a negative subject, a negated awareness or experience verb
    over the cyber subject), `AFFECTED` when the clause holds no negation
    (a correlative's "not only" and an idiom's "not limited to" are none),
    either `_QUALIFIED` when an exception follows or the effect is stated
    through awareness or belief; a sentence whose negation the rule cannot
    bind is unproved (visible as SCOPE_UNRESOLVED, no polarity); a sentence
    that says the determination is not made is no conclusion (visible, no
    polarity); a sentence that speaks only of what could or may happen is a
    risk statement the rule leaves visible.
    """
    instances: list[DraftInstance] = []
    unrecognized: list[str] = []
    context: list[str] = []
    for sentence in sentences:
        plain = sentence.plain
        if _CYBER_SUBJECT.search(plain) is None:
            continue
        effect = _MATERIALLY_AFFECT.search(plain)
        if effect is None:
            continue
        if not sentence.terminated:
            unrecognized.append("TRUNCATED: " + _collapse(plain)[:300])
            continue
        realized = _HAVE_AFFECTED.search(plain)
        if realized is None:
            if _LIKELY.search(plain) is not None:
                context.append("a forward-looking cybersecurity sentence (not this family)")
            else:
                unrecognized.append(_collapse(plain)[:300])
            continue
        if _CYBER_NO_CONCLUSION.search(plain) is not None:
            # What the registrant has not determined has no polarity: the
            # sentence is kept whole as no conclusion.
            unrecognized.append("NO_CONCLUSION: " + _collapse(plain)[:300])
            continue
        # The clause the effect words sit in: from the sentence's start or
        # its last turn to the effect words. A correlative's "not" ("not
        # only ... but also") and an idiom's ("but not limited to") govern
        # nothing and are masked at their own offsets.
        masked = _mask(plain, _CORRELATIVE, _INERT_NEGATION)
        turns = list(_CLAUSE_TURN.finditer(masked[: realized.start()]))
        clause_start = turns[-1].end() if turns else 0
        correlative = _CORRELATIVE.search(plain, clause_start, realized.end())
        scope = _negation_scope(masked, clause_start=clause_start, predicate=realized)
        if scope.binding is None:
            # A negation the rule cannot bind to the effect or to another
            # assertion: the sentence asserts something, but not provably
            # which polarity, so it stays visible and unread.
            unrecognized.append("SCOPE_UNRESOLVED: " + _collapse(plain)[:300])
            continue
        polarity = "NOT_AFFECTED" if scope.negated else "AFFECTED"
        qualifiers: list[str] = []
        clause = plain[clause_start : realized.end()]
        belief = _BELIEF.search(clause)
        if belief is not None or scope.through_awareness:
            cue = belief if belief is not None else scope.verb
            assert cue is not None
            qualifiers.append(
                "stated through the registrant's awareness or belief: "
                + _collapse(plain[max(0, cue.start() - 24) : cue.end() + 24])[:120]
            )
            polarity = f"{polarity}_QUALIFIED"
        exception_from = realized.end()
        if correlative is not None:
            # The correlative's second half ("but also ...") completes the
            # assertion; an exception is what follows it, if anything.
            second = re.compile(r"\bbut\b(?:\s+also)?", re.IGNORECASE).search(plain, realized.end())
            if second is not None:
                exception_from = second.end()
        exception = _EXCEPTION_CUE.search(plain, exception_from)
        if exception is not None:
            qualifiers.append(_collapse(plain[exception.start() :])[:400])
            if not polarity.endswith("_QUALIFIED"):
                polarity = f"{polarity}_QUALIFIED"
        if _LIKELY.search(plain) is not None:
            qualifiers.append("the sentence also states a forward-looking likelihood")
        period_end, period_text, period_basis = _resolve_period(
            sentence, region_sentences=sentences, report_period=report_period
        )
        stated = _CYBER_PERIOD.search(plain)
        if stated is not None and period_end is None:
            period_text = _collapse(stated.group(0))
            period_basis = "the period the sentence states, in its own words"
        fields: list[DraftField] = []
        subject_field = _field("subject", sentence, _CYBER_SUBJECT.search(plain))
        if subject_field is not None:
            fields.append(subject_field)
        effect_field = _field("effect_clause", sentence, realized)
        if effect_field is not None:
            fields.append(effect_field)
        if scope.binding not in {"PREDICATE", "NONE"}:
            qualifiers.append(
                "the negation is read on the "
                + ("subject" if scope.binding == "SUBJECT" else "governing verb")
                + ": "
                + _collapse(plain[scope.negation_start : scope.negation_end])[:120]
            )
        instances.append(
            DraftInstance(
                subject="the effect of cybersecurity threats and incidents on the registrant",
                action="EFFECT_STATED",
                polarity=polarity,
                period_end=period_end,
                period_text=period_text,
                period_basis=period_basis,
                character_start=sentence.character_start,
                character_end=sentence.character_end,
                statement_text=_collapse(sentence.text),
                fields=tuple(fields),
                qualifiers=tuple(qualifiers),
                unknown_fields=("period",) if period_end is None and stated is None else (),
            )
        )
    return tuple(instances), tuple(unrecognized), tuple(dict.fromkeys(context))


_SCOPE_CHARACTERS = 2400
_SCOPE_BYTES = 3_900
"""Under the reader's per-span ceiling (16 KiB over four typed spans a read,
4,096 bytes) with room for the sentence context it adds, so the delivered
scope is read whole and never bounded mid-row."""
_TABLE_ROW_LINE = re.compile(r"^[^\n:]{1,160}: [^\n]*?; [^\n:]{1,160}: ")
"""A table row as canonical extraction rules v4 carry it: `heading: value;
heading: value`."""
_TABLE_NOTE_LINE = re.compile(r"^(?:\(\w{1,3}\)|[\u2014\u2013\-_]{3,}$|\(in )")
"""A footnote, a rule line or a units line that belongs to the table above."""
_SCOPE_PARAGRAPH_LIMIT = 64


@dataclass(frozen=True, slots=True)
class _Scope:
    """The delivered inspected scope of a non-extracted observation.

    The delivered inspected scope of a non-extracted observation: where
    it ends (None when not even the item's first unit fits the reader's
    ceiling), the first unit that could not be delivered whole (its
    character position and UTF-8 size), and whether the item goes on past
    the delivered scope.
    """

    end: int | None
    undelivered: tuple[int, int] | None
    continues: bool


def _paragraphs(text: str, start: int, body_end: int) -> list[tuple[int, int]]:
    """Split item paragraphs into complete source units.

    The item's paragraphs as (start, end) ranges over whole units,
    the heading line first; blank runs are no unit.
    """
    units: list[tuple[int, int]] = []
    cursor = start
    while cursor < body_end:
        following = text.find("\n\n", cursor)
        if following == -1 or following > body_end:
            following = body_end
        if text[cursor:following].strip():
            units.append((cursor, following))
        cursor = following + 2
    return units


def _scope_end(text: str, start: int, body_end: int) -> _Scope:
    """Where the delivered inspected scope of a non-extracted observation ends.

    Where the delivered inspected scope of a non-extracted observation
    ends: whole paragraphs from the item's heading, taken in order while
    they fit the reader's per-span ceiling in UTF-8 bytes, through the first
    2,400 characters and then through the table those paragraphs are part
    of -- a repurchase table's "Period: ...; Total Number of Shares
    Purchased: ..." rows and the footnotes that qualify them. Every path
    checks the bytes: a short region with wide characters, a paragraph
    crossing the 2,400th character, a single row larger than the ceiling.
    A unit is delivered whole or not at all; the first unit that does not
    fit is named, and when it is the item's first unit after the heading
    nothing is delivered rather than a fragment.
    """
    units = _paragraphs(text, start, body_end)
    if not units:
        return _Scope(None, None, False)
    end: int | None = None
    undelivered: tuple[int, int] | None = None
    in_table = False
    delivered_units = 0
    for unit_start, unit_end in units[:_SCOPE_PARAGRAPH_LIMIT]:
        paragraph = text[unit_start:unit_end].strip()
        is_row = bool(_TABLE_ROW_LINE.match(paragraph))
        is_note = in_table and bool(_TABLE_NOTE_LINE.match(paragraph))
        reached = end is not None and end - start >= _SCOPE_CHARACTERS
        if reached and not (is_row or is_note):
            break
        if len(text[start:unit_end].encode("utf-8")) > _SCOPE_BYTES:
            undelivered = (unit_start, len(text[unit_start:unit_end].encode("utf-8")))
            break
        end = unit_end
        delivered_units += 1
        in_table = in_table or is_row
    continues = end is None or end < units[-1][1]
    if delivered_units <= 1 and len(units) > 1:
        # The heading alone is no inspected scope.
        return _Scope(None, undelivered, True)
    return _Scope(end, undelivered, continues)


def _cue_anywhere(family: TypedDisclosureFamily, text: str) -> bool:
    plain = _plain(text)
    if family is TypedDisclosureFamily.DISCLOSURE_CONTROLS_CONCLUSION:
        return any(
            _CONCLUDED.search(plain, max(0, match.start() - 400), match.start() + 400)
            for match in _DCP_PHRASE.finditer(plain)
        )
    return False


def _observe_regions(
    definition: FamilyDefinition,
    spec: RegionSpec,
    regions: tuple[ItemRegion, ...],
    text: str,
    structure: DocumentStructure,
    report_period: ReportPeriod,
    incomplete: str | None,
) -> DraftObservation:
    instances: list[DraftInstance] = []
    unrecognized: list[str] = []
    references: list[str] = []
    context: list[str] = []
    inspected: list[tuple[int, int]] = []
    for region in regions:
        inspected.append((region.heading_start, region.body_end))
        sentences = _sentences(text[region.body_start : region.body_end], region.body_start)
        heading_ranges = [
            (heading.character_start, heading.character_end)
            for heading in structure.headings
            if heading.kind in _SECTION_HEADINGS
            and region.body_start <= heading.character_start < region.body_end
        ]
        sentences = tuple(
            value
            for value in sentences
            if not _HEADING_LINE.match(value.text)
            and not _looks_like_heading(value)
            and not any(
                start <= value.character_start < end
                or value.character_start <= start < value.character_end
                for start, end in heading_ranges
            )
        )
        body_lines = _plain_lines(text[region.body_start : region.body_end])
        bare = [line for line in body_lines.split("\n") if line.strip()]
        if definition.family is TypedDisclosureFamily.DISCLOSURE_CONTROLS_CONCLUSION:
            found, odd, notes = _extract_dcp(sentences, report_period)
            refs: tuple[str, ...] = ()
        elif definition.family is TypedDisclosureFamily.UNREGISTERED_EQUITY_SALES:
            found, odd, refs, notes = _extract_sales(sentences, body_lines, report_period)
        elif definition.family is TypedDisclosureFamily.CUSTOMER_CONCENTRATION:
            found, odd, notes = _extract_concentration(sentences, report_period)
            refs = ()
        elif definition.family is TypedDisclosureFamily.CYBERSECURITY_THREAT_EFFECT:
            found, odd, notes = _extract_cyber_effect(sentences, report_period)
            refs = ()
        elif len(bare) == 1 and _BARE_NONE.match(bare[0].strip()):
            # "Item 5. Other Information -- None." answers the whole item, which
            # covers more than this family: not an explicit none for it.
            found, odd, refs, notes = (
                (),
                (),
                (),
                (f"the item's whole answer is {bare[0].strip()!r}",),
            )
        else:
            found, odd, notes = _extract_arrangements(sentences, report_period)
            refs = ()
        instances.extend(found)
        unrecognized.extend(odd)
        references.extend(refs)
        context.extend(notes)
    truncated = [value for value in unrecognized if value.startswith("TRUNCATED: ")]
    no_conclusion = [value for value in unrecognized if value.startswith("NO_CONCLUSION: ")]
    unresolved = [value for value in unrecognized if value.startswith("SCOPE_UNRESOLVED: ")]
    region_title = regions[0].title
    scope = _scope_end(text, regions[0].heading_start, regions[0].body_end)
    scope_range = None if scope.end is None else (regions[0].heading_start, scope.end)
    if truncated and not instances:
        state, reason = (
            TypedDisclosureState.SOURCE_UNAVAILABLE,
            "the family's statement breaks off before its verb or object; the canonical text "
            "is truncated: " + truncated[0][len("TRUNCATED: ") :][:200],
        )
    elif instances:
        polarities = {value.polarity.replace("_QUALIFIED", "") for value in instances}
        asserted = [
            value for value in instances if value.polarity not in {"NONE", "NONE_WITH_EXCEPTION"}
        ]
        excepted = [value for value in instances if value.polarity == "NONE_WITH_EXCEPTION"]
        negative = [value for value in instances if value.polarity == "NONE"]
        # A scalar conclusion conflicts only with another conclusion for the
        # same period; a prior period's conclusion stated beside the current
        # one keeps its own as-of date and is reported as such.
        by_period: dict[date | None, set[str]] = {}
        for value in instances:
            by_period.setdefault(value.period_end, set()).add(
                value.polarity.replace("_QUALIFIED", "")
            )
        conflicting = [period for period, kinds in by_period.items() if len(kinds) > 1]
        if definition.cardinality == "SCALAR" and conflicting:
            state, reason = (
                TypedDisclosureState.AMBIGUOUS,
                "the section states conclusions of different polarity for the same period: "
                + ", ".join(sorted(polarities)),
            )
        elif definition.cardinality == "SCALAR" and no_conclusion:
            state, reason = (
                TypedDisclosureState.AMBIGUOUS,
                "the section both states a conclusion and states that none was reached: "
                + no_conclusion[0][len("NO_CONCLUSION: ") :][:200],
            )
        elif definition.cardinality == "SCALAR" and unresolved:
            state, reason = (
                TypedDisclosureState.AMBIGUOUS,
                "the section states a conclusion beside a sentence whose negation the rule "
                "cannot bind to the effect or to another assertion, so no polarity is read: "
                + unresolved[0][len("SCOPE_UNRESOLVED: ") :][:200],
            )
        elif definition.cardinality == "SCALAR" and len(by_period) > 1:
            state, reason = (
                TypedDisclosureState.EXTRACTED,
                f"{len(instances)} conclusions for different periods, each with its own as-of "
                "date; the current one is the report period's",
            )
        elif definition.cardinality == "LIST" and negative and (asserted or excepted):
            # "no ... adopted or terminated" beside a named adoption without
            # an exception clause is a conflict, not a list.
            state, reason = (
                TypedDisclosureState.AMBIGUOUS,
                "the section both negates the activity and reports an instance of it",
            )
        elif negative and not asserted:
            state, reason = (
                TypedDisclosureState.EXPLICIT_NONE,
                "the source explicitly negates the scoped activity",
            )
        elif excepted and not asserted:
            # A none with an exception the rules did not read is neither a
            # none nor an assertion: the exception is the answer's scope.
            state, reason = (
                TypedDisclosureState.AMBIGUOUS,
                "the source negates the activity except as stated, and no reviewed rule reads "
                "the exception; it requires reading: " + excepted[0].qualifiers[-1][:200]
                if excepted[0].qualifiers
                else "the source negates the activity except as stated; the exception requires "
                "reading",
            )
        else:
            state, reason = (
                TypedDisclosureState.EXTRACTED,
                f"{len(asserted)} source assertion(s) parsed under the definition"
                + ("; stated as the exception to a none" if excepted else ""),
            )
        if unrecognized:
            reason += f"; {len(unrecognized)} relevant sentence(s) not recognised, kept visible"
    elif references:
        state, reason = (
            TypedDisclosureState.REFERENCE_REQUIRED,
            "the item refers to another filing for the activity: " + references[0],
        )
    elif no_conclusion:
        state, reason = (
            TypedDisclosureState.AMBIGUOUS,
            "the source states that no conclusion was reached, so no polarity is read: "
            + no_conclusion[0][len("NO_CONCLUSION: ") :][:200],
        )
    elif unresolved:
        state, reason = (
            TypedDisclosureState.AMBIGUOUS,
            "the sentence asserts the effect beside a negation the rule cannot bind to the "
            "effect or to another assertion, so no polarity is read: "
            + unresolved[0][len("SCOPE_UNRESOLVED: ") :][:200],
        )
    elif unrecognized:
        state, reason = (
            TypedDisclosureState.AMBIGUOUS,
            "a relevant statement was found that no reviewed rule reads; it requires reading",
        )
    else:
        state, reason = (
            TypedDisclosureState.NOT_FOUND,
            f"{spec.part} Item {spec.item} ({region_title}) is present and states nothing the "
            "rules read for this family" + ("; " + "; ".join(context) if context else ""),
        )
    if incomplete is not None and state in {
        TypedDisclosureState.NOT_FOUND,
        TypedDisclosureState.AMBIGUOUS,
    }:
        state, reason = (
            TypedDisclosureState.SOURCE_UNAVAILABLE,
            incomplete + f" (this item: {reason})",
        )
    elif incomplete is not None:
        context.append("the canonical text is incomplete elsewhere: " + incomplete)
    region_text = text[regions[0].heading_start : regions[0].body_end].rstrip()
    if not instances or state is not TypedDisclosureState.EXTRACTED:
        if scope.undelivered is not None:
            position, size = scope.undelivered
            context.append(
                f"a unit of the item at character {position - regions[0].heading_start:,} "
                f"({size:,} bytes) exceeds the reader's per-span ceiling and is not delivered"
            )
        if scope.end is None:
            context.append(
                "no inspected scope is delivered: the item's first unit does not fit the "
                "reader's per-span ceiling; the item stays in the canonical text"
            )
        elif scope.continues:
            context.append(
                f"the delivered scope is the item's first {scope.end - regions[0].heading_start:,} "
                f"of {len(region_text):,} characters; the item continues in the canonical text"
            )
    return DraftObservation(
        family=definition.family,
        rule_id=definition.rule_id,
        state=state,
        reason=reason[:600],
        part=regions[0].part or spec.part,
        item=regions[0].item,
        region_title=region_title,
        report_period=report_period,
        instances=tuple(instances),
        unrecognized=tuple(unrecognized[:8]),
        references=tuple(dict.fromkeys(references))[:4],
        scope_range=scope_range,
        inspected_ranges=tuple(inspected[:4]),
        context=tuple(dict.fromkeys(context))[:6],
    )


def _plain_lines(text: str) -> str:
    return "\n".join(_plain(line) for line in text.split("\n"))


# --------------------------------------------------------------------------
# Explicit comparison with a selected prior observation.

CHANGE_CONTINUING = "CONTINUING_ASSERTION"
CHANGE_CHANGED = "CHANGED_ASSERTION"
CHANGE_NEWLY_REPORTED = "NEWLY_REPORTED_ACTION"
CHANGE_NEW_BASELINE = "NEWLY_OBSERVED_NO_BASELINE"
CHANGE_NOT_OBSERVED = "NO_LONGER_OBSERVED"
CHANGE_RULE_OR_SOURCE = "RULE_OR_SOURCE_CHANGE"
CHANGE_NOT_COMPARABLE = "NOT_COMPARABLE"


@dataclass(frozen=True, slots=True)
class TypedDisclosureChange:
    """Compare one family's current and selected prior observation.

    One family's movement between a current and an explicitly selected
    prior observation of the same issuer, form and item. Never a judgment:
    a disappearance is reported as not observed, not as resolved; a change
    under different rules or from the same source bytes is named as that,
    not as a new event. `changed_fields` names, per assertion, the material
    fields whose values moved ("<assertion>: shares_up_to 12,000 -> 20,000")
    and the assertions newly stated or no longer stated.
    """

    key: str
    kind: str
    detail: str
    current_observation_id: str | None
    prior_observation_id: str | None
    current_state: str | None
    prior_state: str | None
    changed_fields: tuple[str, ...] = ()


def _observation_key(observation: TypedDisclosureObservationLike) -> str:
    form = observation.document_type.upper().removesuffix("/A")
    return f"{observation.entity_id}|{observation.family}|{form}|{observation.item or '-'}"


class TypedDisclosureObservationLike:
    """Describe the sealed observation fields used for comparison.

    The attributes the comparison reads from a sealed observation; kept as
    a structural type so the sealed contract owns its own fields.
    """

    observation_id: str
    entity_id: str
    document_type: str
    revision_label: str
    family: str
    rule_id: str
    item: str | None
    state: str
    report_period_end: str | None
    instances: tuple[object, ...]


_EVIDENCE_ONLY_FIELDS = frozenset({"concluded_by", "effectiveness_predicate"})
"""Fields that locate the assertion's words rather than state its content:
their wording differs between filings that assert the same thing."""


Content = tuple[tuple[str, str], ...]
Assertions = dict[str, tuple[Content, ...]]


def _assertions(observation: TypedDisclosureObservationLike) -> Assertions:
    """What an observation asserts, for comparison.

    What an observation asserts, for comparison: each assertion's identity
    (subject, action, stated date) mapped to the material content of every
    instance that carries it -- the polarity, every parsed field but the
    evidence-only ones (quantity, consideration, purchaser, exemption, person,
    role, arrangement kind, shares, duration, negated actions), the qualifiers
    and the fields the source left unknown. Two sales on one day are two
    assertions of one identity; only a genuinely identical repetition is
    folded into one. The report period is not content: a later filing
    restating the same assertion continues it.
    """
    assertions: dict[str, list[Content]] = {}
    for instance in observation.instances:
        fields = {
            str(getattr(field, "name", "")): getattr(field, "value", None)
            for field in getattr(instance, "fields", ())
        }
        when = fields.get("action_date") or fields.get("date") or ""
        identity = f"{getattr(instance, 'subject', '')}|{getattr(instance, 'action', '')}|{when}"
        content: list[tuple[str, str]] = [("polarity", str(getattr(instance, "polarity", "")))]
        content.extend(
            (name, "" if value is None else str(value))
            for name, value in sorted(fields.items())
            if name not in _EVIDENCE_ONLY_FIELDS
        )
        content.append(
            ("qualifiers", "; ".join(sorted(str(v) for v in getattr(instance, "qualifiers", ()))))
        )
        unknown = sorted(str(v) for v in getattr(instance, "unknown_fields", ()))
        content.append(("unknown_fields", "+".join(unknown)))
        held = assertions.setdefault(identity, [])
        # A genuinely identical repetition is the same assertion; anything
        # that differs in one field is another assertion of this identity.
        if tuple(content) not in held:
            held.append(tuple(content))
    return {identity: tuple(values) for identity, values in assertions.items()}


def _field_moves(
    head: Assertions, base: Assertions
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Compare assertion fields between two observations.

    `(changed, newly stated, no longer stated)` between two assertion
    maps, each entry naming the assertion and, for a change, the fields.
    Under one identity, identical contents pair off first; one remaining
    assertion on each side is that assertion changed; a remaining
    assertion with no counterpart is newly or no longer stated; more than
    one remaining on either side is a correspondence the source does not
    establish, reported as unresolved rather than paired by guess.
    """
    changed: list[str] = []
    new: list[str] = []
    gone: list[str] = []
    for identity in sorted(set(head) | set(base)):
        now = list(head.get(identity, ()))
        then = list(base.get(identity, ()))
        for content in list(now):
            if content in then:
                now.remove(content)
                then.remove(content)
        if not now and not then:
            continue
        if len(now) == 1 and len(then) == 1:
            after, before = dict(now[0]), dict(then[0])
            moved = [
                f"{name} {before.get(name, '') or '-'} -> {after.get(name, '') or '-'}"
                for name in sorted(set(after) | set(before))
                if after.get(name, "") != before.get(name, "")
            ]
            changed.append(f"{identity}: " + "; ".join(moved))
        elif not then:
            new.extend(f"{identity}: newly stated" for _ in now)
        elif not now:
            gone.extend(f"{identity}: no longer stated" for _ in then)
        else:
            changed.append(
                f"{identity}: UNRESOLVED CORRESPONDENCE -- {len(then)} prior and {len(now)} "
                "current assertions of this identity differ and the source does not say "
                "which is which"
            )
    return tuple(changed), tuple(new), tuple(gone)


def compare_typed_disclosures(
    *,
    current: tuple[TypedDisclosureObservationLike, ...],
    prior: tuple[TypedDisclosureObservationLike, ...],
    current_rules: tuple[str, str],
    prior_rules: tuple[str, str],
) -> tuple[TypedDisclosureChange, ...]:
    """Compare current and prior observations by issuer, family, form, and item.

    Compare the latest observation of each (issuer, family, form, item) on
    the current side with the prior side's. `current_rules`/`prior_rules` are
    each `(rules_id, definitions_hash)`; when they differ -- or when the two
    observations were read under different family rule identities -- every
    such pair is a rule change, because nothing else can be told apart.

    Readable pairs are compared assertion by assertion on their material
    content (`_assertions`), not on event labels alone: the same person,
    action and date restated with a different quantity, term or qualifier
    is a changed assertion. Whether a difference is a correction of the same
    period or a newly occurring action is decided by the report period: a
    re-reported period (an amendment, a re-acquired revision) only ever
    changes what that period states; a later period may newly report an
    action, and its restatements of earlier assertions are compared as such.
    """

    def latest(
        values: tuple[TypedDisclosureObservationLike, ...],
    ) -> dict[str, TypedDisclosureObservationLike]:
        chosen: dict[str, TypedDisclosureObservationLike] = {}
        for value in values:
            if value.state == TypedDisclosureState.NOT_APPLICABLE.value:
                continue
            key = _observation_key(value)
            held = chosen.get(key)
            if held is None or (value.report_period_end or "") > (held.report_period_end or ""):
                chosen[key] = value
        return chosen

    now, then = latest(current), latest(prior)
    changes: list[TypedDisclosureChange] = []
    for key in sorted(set(now) | set(then)):
        head, base = now.get(key), then.get(key)
        if head is None:
            assert base is not None
            changes.append(
                TypedDisclosureChange(
                    key=key,
                    kind=CHANGE_NOT_OBSERVED,
                    detail="no current observation for this scope; not thereby resolved or none",
                    current_observation_id=None,
                    prior_observation_id=base.observation_id,
                    current_state=None,
                    prior_state=base.state,
                )
            )
            continue
        if base is None:
            changes.append(
                TypedDisclosureChange(
                    key=key,
                    kind=CHANGE_NEW_BASELINE,
                    detail="no prior observation of this scope was selected; first use",
                    current_observation_id=head.observation_id,
                    prior_observation_id=None,
                    current_state=head.state,
                    prior_state=None,
                )
            )
            continue
        if current_rules != prior_rules or head.rule_id != base.rule_id:
            rule_ids = f"; {base.rule_id} -> {head.rule_id}" if head.rule_id != base.rule_id else ""
            changes.append(
                _change(
                    key,
                    head,
                    base,
                    CHANGE_RULE_OR_SOURCE,
                    f"extraction rules differ ({prior_rules[0]} -> {current_rules[0]}{rule_ids}); "
                    "a difference here is not a new event",
                )
            )
            continue
        changed, new, gone = _field_moves(_assertions(head), _assertions(base))
        moves = (*changed, *new, *gone)
        # The same assertions in any order, under the same state, continue.
        same = head.state == base.state and not moves
        if head.revision_label == base.revision_label:
            changes.append(
                _change(
                    key,
                    head,
                    base,
                    CHANGE_CONTINUING if same else CHANGE_RULE_OR_SOURCE,
                    "the same source revision"
                    + (
                        ""
                        if same
                        else "; the observation differs, so the source bytes or the parse "
                        "changed, not the filing"
                    ),
                    () if same else moves,
                )
            )
            continue
        readable = {TypedDisclosureState.EXTRACTED.value, TypedDisclosureState.EXPLICIT_NONE.value}
        if head.state not in readable or base.state not in readable:
            changes.append(
                _change(
                    key,
                    head,
                    base,
                    CHANGE_NOT_COMPARABLE,
                    f"current {head.state}, prior {base.state}: an unresolved state on "
                    "either side compares nothing; disappearance is not resolution",
                )
            )
            continue
        period = f"period {base.report_period_end} -> {head.report_period_end}"
        if same:
            changes.append(
                _change(
                    key,
                    head,
                    base,
                    CHANGE_CONTINUING,
                    f"the same assertion in a later filing ({period})",
                )
            )
            continue
        same_period = (
            head.report_period_end is not None and head.report_period_end == base.report_period_end
        )
        list_family = head.family in {
            TypedDisclosureFamily.INSIDER_TRADING_ARRANGEMENTS.value,
            TypedDisclosureFamily.UNREGISTERED_EQUITY_SALES.value,
        }
        if same_period:
            kind = CHANGE_CHANGED
            detail = (
                f"the same period ({head.report_period_end}) is re-reported with a different "
                f"statement ({len(changed)} changed, {len(new)} newly stated, {len(gone)} no "
                "longer stated): a changed disclosure or correction of that period, not a new "
                "event"
            )
        elif list_family and new:
            kind = CHANGE_NEWLY_REPORTED
            detail = (
                f"{len(new)} action(s) reported for the current period that the prior did not "
                "report; the prior period's actions are that period's, not withdrawn"
                + (f"; {len(changed)} restated assertion(s) changed" if changed else "")
                + f" ({period})"
            )
        elif changed:
            kind = CHANGE_CHANGED
            detail = (
                f"{len(changed)} assertion(s) restated with changed content, same subject, "
                f"action and date: {'; '.join(changed)[:320]} ({period})"
            )
        else:
            kind = CHANGE_CHANGED
            detail = (
                f"the current period reports {head.state} where the prior reported "
                f"{base.state} ({period})"
                if head.state != base.state
                else f"the current period states {len(gone)} prior assertion(s) no longer "
                f"({period}); that is this period's statement, not a withdrawal"
            )
        changes.append(_change(key, head, base, kind, detail, moves))
    return tuple(changes)


def _change(
    key: str,
    head: TypedDisclosureObservationLike,
    base: TypedDisclosureObservationLike,
    kind: str,
    detail: str,
    changed_fields: tuple[str, ...] = (),
) -> TypedDisclosureChange:
    return TypedDisclosureChange(
        key=key,
        kind=kind,
        detail=detail,
        current_observation_id=head.observation_id,
        prior_observation_id=base.observation_id,
        current_state=head.state,
        prior_state=base.state,
        changed_fields=changed_fields,
    )


__all__ = [
    "CHANGE_CHANGED",
    "CHANGE_CONTINUING",
    "CHANGE_NEWLY_REPORTED",
    "CHANGE_NEW_BASELINE",
    "CHANGE_NOT_COMPARABLE",
    "CHANGE_NOT_OBSERVED",
    "CHANGE_RULE_OR_SOURCE",
    "DEFINITIONS",
    "TYPED_DISCLOSURE_RULES_ID",
    "DocumentExtraction",
    "DraftField",
    "DraftInstance",
    "DraftObservation",
    "ExceptionClause",
    "FamilyDefinition",
    "RegionSpec",
    "TypedDisclosureChange",
    "TypedDisclosureFamily",
    "TypedDisclosureObservationLike",
    "TypedDisclosureState",
    "compare_typed_disclosures",
    "definitions_hash",
    "extract_document",
]
