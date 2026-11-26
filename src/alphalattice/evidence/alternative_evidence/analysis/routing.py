"""One routing owner for the eight evidence topics of a book.

A topic is a business question the obligation asks about an issuer; an
inventory family reads a source shape (the litigation regions, the event
notes and current-report items, the financing notes); a typed rule reads
one fixed disclosure; a table view reads a table the canonical text did
not carry; a residual search reads the routed prose no inventory
addressed. Sources, topics and methods are three axes, and this owner is
where they meet: for every (issuer, topic) cell it names the source
regions the filing's own structure routes to the topic, the methods that
apply, the tables those regions hold, the residual scope a search may
read, and the gaps -- an issuer with no document held, a table without its
retained original, a topic no region routes to. Routes are many-to-many:
a financing unit serves the liquidity topic and, when it names an
issuance, the capital topic; a current report's item serves the topic its
item number names; a navigation family serves several topics. A route
never excludes a passage from another topic, and a familiar heading never
suppresses a region: what no route reaches stays counted.

Nothing here reads the meaning of a passage. A cell's states are machine
facts about sources, representation, discovery and delivery, kept apart
from what an actor later reports having checked.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Protocol

from ..documents.comparison import (
    COMPARISON_RULES_ID,
    ComparableUnit,
    UnitCorrespondence,
    compare_filings,
)
from ..documents.structure import (
    FAMILY_COMMITMENTS,
    FAMILY_CONTROLS,
    FAMILY_FINANCING,
    FAMILY_LEGAL,
    FAMILY_OPERATIONS,
    FAMILY_SUBSEQUENT,
    FAMILY_TRANSACTIONS,
    STRUCTURE_RULES_ID,
    DocumentStructure,
)
from ..documents.tables import TablePlaceholder
from .contracts import ComparedUnitRecord, EvidenceTopic, FilingComparisonRecord
from .events import CORPORATE_EVENT_FAMILY, LATE_FILING_FORMS
from .financing import FINANCING_FAMILY
from .matters import DocumentNeeds, EvidenceNeed
from .operations import OPERATIONS_FAMILY

ROUTING_RULES_ID = "alternative-evidence.topic-routing.v6"
"""v6 (2026-09-24, W1): an issuer without a periodic filing held is no source
gap -- the window is what each issuer filed recently, and nothing is read as
a baseline; an issuer with no document held still is.

v5 (2026-09-23, Q3): a late-filing notice (NT 10-K, NT 10-Q) serves the
governance topic -- the report the issuer could not file on time -- its
narrative and other-information parts each an event unit; and Item 3.01, a
delisting notice or a failure to meet a listing standard, serves the
corporate-action and listing topic (under v4 the capital topic, with the
unregistered sales and the changes to holders' rights of Items 3.02-3.03).

v4 (2026-09-20, the eight-topic completeness assignment): the
operations inventory (`operations.py`) serves the operations topic as a
unit family -- the restructuring, impairment and operating-charge notes
and the dated operations statements of the narrative items -- so a
closure, an impairment or an other-operating-charge statement is
delivered as a unit window, not searched for; the liquidity cue names a
covenant heading and the operations cue the operating-charge, exit and
disposal and held-for-sale headings the trace found unrouted (KO's
`SIGNIFICANT OPERATING AND NONOPERATING ITEMS`); and a sub-heading the
structure is uncertain of (`SUB_UNCERTAIN`) whose text a topic's cue
names opens a region to the next heading of any rank, inside the MD&A
items only (SYF's MD&A `Covenants` and `Dividend and Share
Repurchases`, unrouted under v3 because only item, note and
sub-headings were read; measured on the development filings, admitting
every uncertain sub-heading a cue names grew the residual scope 4.6% in
241 regions and the fixed pair budget then skipped one more strong-topic
question on three of five units, while the MD&A bound keeps the two
statements at +1.4% in 61 regions); the customer-
concentration and cybersecurity-effect typed families serve the
commercial and product topics.
v3 (the shared-gap closeout): an inventoried region withdraws a topic's
residual search only where the family's own units there serve that
topic; the table share is dealt topic-fair.
v2 (2026-09-20): a document whose structure yields no part, item or
note heading past its cover -- a current report whose item lines the
canonical text does not carry, a short unstructured filing; a sub-heading
alone maps to no family -- has no shape to route by,
so its whole body joins every topic's residual scope as an `unstructured
document` region: the questions read it as the unconditional program did,
and its cost is its size. Measured under v1 on the regression copy: the
two former held-out cases the integrated arms lost at the first response
(a tender-offer item and a bylaws item) sat in 8-K texts with no item
heading, outside every routed range and every inventory, unreachable by
any question; nothing else changes.
v1 (2026-09-22): the eight topics routed by source shape as `TOPIC_ROUTES`
states -- the inventories' regions by unit family (a current report's
item number and a periodic note's heading naming the topic), the
structure's navigation families and heading cues for the residual scope,
the typed families, and the tables inside routed regions -- with the
residual scope of a cell being its routed ranges less every inventoried
region, or one bounded broader pass over the issuer's documents when no
region routes to the topic. Frozen before the integrated candidate's first
measurement; a change is a successor, never an edit."""

LITIGATION_FAMILY = "LITIGATION"
TOPICS: tuple[EvidenceTopic, ...] = tuple(EvidenceTopic)
"""The declared order: lanes, cells and reports walk the topics in it."""

TABLE_VIEWS_PER_SESSION = 8
"""Pages of table views one session delivers, inside the matter window
allowance (64) -- two reads of four -- so tables cost the envelope nothing
new and never displace more than eight unit windows."""


@dataclass(frozen=True, slots=True)
class TopicRoute:
    """Where one topic's evidence is looked for, and by which methods."""

    topic: EvidenceTopic
    structure_families: tuple[str, ...]
    """The navigation families whose sections the residual search may read
    for the topic -- an inclusion, never an exclusion of another topic."""
    heading_cue: re.Pattern[str]
    """A note, item or sub-heading that opens a preferred region of the
    topic (an equity note, a liquidity section, a concentration paragraph):
    its body joins the residual scope whatever family the path maps to."""
    unit_families: tuple[str, ...]
    """The inventories whose units serve the topic."""
    typed_families: tuple[str, ...]
    """The typed disclosure families whose observations serve the topic."""
    tables: bool
    """Whether tables inside the routed regions are evidence of the topic."""
    preferred: str
    deficit: str
    """What the routing does not establish, in the plan's own words."""


TOPIC_ROUTES: dict[EvidenceTopic, TopicRoute] = {
    EvidenceTopic.LIQUIDITY_GOING_CONCERN: TopicRoute(
        topic=EvidenceTopic.LIQUIDITY_GOING_CONCERN,
        structure_families=(FAMILY_FINANCING, FAMILY_SUBSEQUENT),
        heading_cue=re.compile(
            r"liquidity|going concern|capital resources|\bdebt\b|borrowing|credit "
            r"(?:agreement|facilit)|financing|notes payable|indebtedness|basis of presentation|"
            r"covenant",
            re.IGNORECASE,
        ),
        unit_families=(FINANCING_FAMILY, CORPORATE_EVENT_FAMILY),
        typed_families=(),
        tables=True,
        preferred="debt and borrowing notes; MD&A liquidity; going-concern statements; "
        "financing events (8-K items 1.03, 2.03, 2.04)",
        deficit="the financing inventory is not cash, maturity, covenant or going-concern "
        "completeness; a liquidity section is read as prose, not as a balance",
    ),
    EvidenceTopic.CAPITAL_DILUTION: TopicRoute(
        topic=EvidenceTopic.CAPITAL_DILUTION,
        structure_families=(FAMILY_FINANCING,),
        heading_cue=re.compile(
            r"earnings per share|(?:stockholders|shareholders)['\u2019]?\s+equity|capital "
            r"stock|repurchase|unregistered sales|market for (?:the )?registrant|dividend|"
            r"convertible|equity compensation|stock-based|share-based",
            re.IGNORECASE,
        ),
        unit_families=(FINANCING_FAMILY, CORPORATE_EVENT_FAMILY),
        typed_families=("UNREGISTERED_EQUITY_SALES",),
        tables=True,
        preferred="equity and EPS notes; issuance and repurchase disclosures; unregistered "
        "sales; financing and capital-return events (8-K items 3.02, 3.03)",
        deficit="a repurchase is not a sale; preferred and convertible features and share "
        "tables are not inferred from a heading",
    ),
    EvidenceTopic.LEGAL_REGULATORY: TopicRoute(
        topic=EvidenceTopic.LEGAL_REGULATORY,
        structure_families=(FAMILY_LEGAL, FAMILY_COMMITMENTS),
        heading_cue=re.compile(
            r"legal proceeding|litigation|contingenc|income tax|regulat|environmental",
            re.IGNORECASE,
        ),
        unit_families=(LITIGATION_FAMILY, CORPORATE_EVENT_FAMILY),
        typed_families=(),
        tables=False,
        preferred="Legal Proceedings items; contingencies and tax notes; legal events",
        deficit="unnamed and umbrella matters stay eligible; tax narrative is prose, not a matter",
    ),
    EvidenceTopic.OPERATIONS_SUPPLY: TopicRoute(
        topic=EvidenceTopic.OPERATIONS_SUPPLY,
        structure_families=(FAMILY_OPERATIONS, FAMILY_COMMITMENTS, FAMILY_SUBSEQUENT),
        heading_cue=re.compile(
            r"restructuring|impairment|segment|supply|manufactur|production|transformation|"
            r"cost (?:reduction|saving)|exit|closure|subsequent event|results of operations|"
            r"other operating charges|significant operating|nonoperating items|"
            r"held for sale|disposal activit|store closing",
            re.IGNORECASE,
        ),
        unit_families=(OPERATIONS_FAMILY, CORPORATE_EVENT_FAMILY),
        typed_families=(),
        tables=True,
        preferred="Business and MD&A; restructuring, impairment and operating-charge notes "
        "and dated operations statements (the operations inventory); subsequent "
        "events; operations events (8-K items 2.02, 2.05, 2.06)",
        deficit="no universal project recognizer: grounded region evidence is delivered, "
        "never fabricated project fields",
    ),
    EvidenceTopic.PRODUCT_SAFETY_CYBER: TopicRoute(
        topic=EvidenceTopic.PRODUCT_SAFETY_CYBER,
        structure_families=(FAMILY_CONTROLS, FAMILY_LEGAL, FAMILY_OPERATIONS),
        heading_cue=re.compile(
            r"cybersecurity|product (?:liability|safety|recall)|recall|data (?:breach|"
            r"security|privacy)|information security|quality",
            re.IGNORECASE,
        ),
        unit_families=(LITIGATION_FAMILY, CORPORATE_EVENT_FAMILY),
        typed_families=("CYBERSECURITY_THREAT_EFFECT",),
        tables=False,
        preferred="cybersecurity (Item 1C) and product disclosures; contingencies; events",
        deficit="risk-factor language is not an incident, and no match is not proof of no incident",
    ),
    EvidenceTopic.GOVERNANCE_CONTROLS: TopicRoute(
        topic=EvidenceTopic.GOVERNANCE_CONTROLS,
        structure_families=(FAMILY_CONTROLS,),
        heading_cue=re.compile(
            r"controls and procedures|internal control|disclosure controls|material "
            r"weakness|auditor|accountant|annual meeting|shareholder vote|bylaw|10b5-1|"
            r"directors|executive officers|corporate governance",
            re.IGNORECASE,
        ),
        unit_families=(CORPORATE_EVENT_FAMILY,),
        typed_families=("DISCLOSURE_CONTROLS_CONCLUSION", "INSIDER_TRADING_ARRANGEMENTS"),
        tables=True,
        preferred="controls items; auditor and officer changes; votes; trading arrangements "
        "(8-K items 4.01, 4.02, 5.01-5.08)",
        deficit="DCP, ICFR, the audit opinion, remediation and changes are different claims; "
        "proxy material not held is a source gap",
    ),
    EvidenceTopic.COMMERCIAL_COUNTERPARTY: TopicRoute(
        topic=EvidenceTopic.COMMERCIAL_COUNTERPARTY,
        structure_families=(FAMILY_OPERATIONS, FAMILY_TRANSACTIONS),
        heading_cue=re.compile(
            r"concentration|customer|supplier|distribut|material (?:definitive )?agreement|"
            r"contract|counterpart|offtake|license agreement|revenue",
            re.IGNORECASE,
        ),
        unit_families=(CORPORATE_EVENT_FAMILY,),
        typed_families=("CUSTOMER_CONCENTRATION",),
        tables=True,
        preferred="customer and supplier concentration; agreements; business and MD&A; "
        "agreement events (8-K items 1.01, 1.02)",
        deficit="counterparty identity is not inferred from a generic phrase; repeated risk "
        "language is not a default",
    ),
    EvidenceTopic.CORPORATE_ACTION_LISTING: TopicRoute(
        topic=EvidenceTopic.CORPORATE_ACTION_LISTING,
        structure_families=(FAMILY_TRANSACTIONS, FAMILY_SUBSEQUENT),
        heading_cue=re.compile(
            r"acquisition|divestiture|business combination|joint venture|disposition|"
            r"discontinued|spin-?off|merger|listing|delist|reverse (?:stock )?split|tender "
            r"offer|exchange offer|subsequent event|redemption",
            re.IGNORECASE,
        ),
        unit_families=(CORPORATE_EVENT_FAMILY, FINANCING_FAMILY),
        typed_families=(),
        tables=True,
        preferred="acquisition, disposal, venture and subsequent-event notes; transaction "
        "and other events (8-K items 2.01, 7.01, 8.01)",
        deficit="an item label is navigation, not a completed transaction; exhibits may be "
        "necessary and are a dependency",
    ),
}
"""All eight rows are executable: every one has a residual route at least,
and a row whose only method is the residual search is a row with a gap
named, not an omission."""


_ITEM_8K_TOPICS: tuple[tuple[re.Pattern[str], tuple[EvidenceTopic, ...]], ...] = (
    (
        re.compile(r"^\W*item\s+1\.0[12]\b", re.IGNORECASE),
        (EvidenceTopic.COMMERCIAL_COUNTERPARTY, EvidenceTopic.CORPORATE_ACTION_LISTING),
    ),
    (re.compile(r"^\W*item\s+1\.03\b", re.IGNORECASE), (EvidenceTopic.LIQUIDITY_GOING_CONCERN,)),
    (re.compile(r"^\W*item\s+1\.0[45]\b", re.IGNORECASE), (EvidenceTopic.PRODUCT_SAFETY_CYBER,)),
    (re.compile(r"^\W*item\s+2\.01\b", re.IGNORECASE), (EvidenceTopic.CORPORATE_ACTION_LISTING,)),
    (re.compile(r"^\W*item\s+2\.02\b", re.IGNORECASE), (EvidenceTopic.OPERATIONS_SUPPLY,)),
    (
        re.compile(r"^\W*item\s+2\.0[34]\b", re.IGNORECASE),
        (EvidenceTopic.LIQUIDITY_GOING_CONCERN, EvidenceTopic.CAPITAL_DILUTION),
    ),
    (re.compile(r"^\W*item\s+2\.0[56]\b", re.IGNORECASE), (EvidenceTopic.OPERATIONS_SUPPLY,)),
    (
        re.compile(r"^\W*item\s+3\.01\b", re.IGNORECASE),
        (EvidenceTopic.CORPORATE_ACTION_LISTING,),
    ),
    (re.compile(r"^\W*item\s+3\.0[23]\b", re.IGNORECASE), (EvidenceTopic.CAPITAL_DILUTION,)),
    (re.compile(r"^\W*item\s+4\.0[12]\b", re.IGNORECASE), (EvidenceTopic.GOVERNANCE_CONTROLS,)),
    (re.compile(r"^\W*item\s+5\.0[1-8]\b", re.IGNORECASE), (EvidenceTopic.GOVERNANCE_CONTROLS,)),
    (
        re.compile(r"^\W*item\s+[78]\.01\b", re.IGNORECASE),
        (EvidenceTopic.CORPORATE_ACTION_LISTING,),
    ),
)
"""A current report's item number names the topic its event serves."""

_ISSUANCE = re.compile(
    r"\bissu(?:ed|ance|es|ing)\b|\boffering\b|\bconvertible\b|\bshares?\b|\brepurchase|"
    r"\bdividend",
    re.IGNORECASE,
)
_PRODUCT = re.compile(r"product|recall|cyber|data breach|privacy|safety", re.IGNORECASE)
_LEGAL_EVENT = re.compile(
    r"settle|lawsuit|litigation|complaint|court|regulator|consent (?:decree|order)|"
    r"investigation|subpoena",
    re.IGNORECASE,
)


def topics_of_need(
    need: EvidenceNeed, *, document_type: str, text: str
) -> tuple[EvidenceTopic, ...]:
    """The topics a need serves, the primary first.

    A litigation need serves the legal topic (and the product topic when
    its title names a product, recall, privacy or cyber matter). A
    financing need serves the liquidity topic and, when its own text names
    an issuance, offering, shares, a repurchase or a dividend, the capital
    topic too. An event need of a current report serves the topic its item
    number names, the corporate-action topic when the item is not mapped;
    an event note of a periodic filing serves the corporate-action topic,
    with the operations topic for a restructuring, impairment or
    subsequent-event note, the legal topic when its title names a
    settlement or a proceeding, and the liquidity and capital topics when
    its title names a financing.
    """
    window = need.window
    return topics_of_unit(
        family=need.family,
        region_heading=need.region_heading,
        title=need.detail,
        document_type=document_type,
        excerpt="" if window is None else text[window[0] : window[1]],
    )


def topics_of_unit(
    *,
    family: str,
    region_heading: str,
    title: str,
    document_type: str,
    excerpt: str,
) -> tuple[EvidenceTopic, ...]:
    """Determine the topics served by one inventoried unit.

    The topics a unit of an inventory family serves, from the same facts
    a delivered window carries -- its family, its region's heading, the
    unit's title and the excerpt's words -- so a sealed window is routed
    exactly as its need was.
    """
    if family == LITIGATION_FAMILY:
        topics = [EvidenceTopic.LEGAL_REGULATORY]
        if _PRODUCT.search(title) is not None:
            topics.append(EvidenceTopic.PRODUCT_SAFETY_CYBER)
        return tuple(topics)
    if family == FINANCING_FAMILY:
        topics = [EvidenceTopic.LIQUIDITY_GOING_CONCERN]
        if _ISSUANCE.search(excerpt) is not None:
            topics.append(EvidenceTopic.CAPITAL_DILUTION)
        return tuple(topics)
    if family == OPERATIONS_FAMILY:
        return (EvidenceTopic.OPERATIONS_SUPPLY,)
    return _event_topics(region_heading, title, document_type)


def _event_topics(heading: str, detail: str, document_type: str) -> tuple[EvidenceTopic, ...]:
    family_form = document_type.upper().removesuffix("/A")
    if family_form in LATE_FILING_FORMS:
        # The report the issuer could not file on time: a controls matter first.
        return (EvidenceTopic.GOVERNANCE_CONTROLS,)
    if family_form == "8-K":
        for pattern, named in _ITEM_8K_TOPICS:
            if pattern.match(heading.strip()) is not None:
                if named == (EvidenceTopic.CORPORATE_ACTION_LISTING,) and (
                    _LEGAL_EVENT.search(detail) is not None
                ):
                    return (EvidenceTopic.CORPORATE_ACTION_LISTING, EvidenceTopic.LEGAL_REGULATORY)
                return named
        return (EvidenceTopic.CORPORATE_ACTION_LISTING,)
    lowered = f"{heading} {detail}".casefold()
    topics = [EvidenceTopic.CORPORATE_ACTION_LISTING]
    if any(word in lowered for word in ("restructuring", "impairment", "subsequent")):
        topics.append(EvidenceTopic.OPERATIONS_SUPPLY)
    if _LEGAL_EVENT.search(lowered) is not None:
        topics.append(EvidenceTopic.LEGAL_REGULATORY)
    if _ISSUANCE.search(lowered) is not None or "debt" in lowered or "credit" in lowered:
        topics.append(EvidenceTopic.LIQUIDITY_GOING_CONCERN)
        topics.append(EvidenceTopic.CAPITAL_DILUTION)
    return tuple(topics)


@dataclass(frozen=True, slots=True)
class RoutedDocument:
    """One admitted filing as routing sees it.

    One admitted filing as routing sees it: its verified text and
    structure, the issuer, and the times its sealed reference states --
    acceptance (the official clock), publication with its precision, the
    report period -- so a cell can order its filings latest first and a
    correspondence can carry both sources' times.
    """

    document_key: str
    document_handle: str
    entity_id: str
    document_type: str
    text: str
    structure: DocumentStructure
    accepted_at: datetime | None
    published_at: datetime | None
    report_period_end: date | None

    @property
    def clock(self) -> datetime | None:
        """The filing's time for ordering.

        The filing's time for ordering: acceptance (the official clock),
        else publication; None when the reference states neither.
        """
        return self.accepted_at or self.published_at


@dataclass(frozen=True, slots=True)
class RoutedRegion:
    """One source range routed to a topic, with the basis it was read by.

    One source range routed to a topic, with the basis it was read by:
    an inventory's region (`inventory:<family>`), or a heading cue of the
    structure (`heading cue:<kind>`).
    """

    document_key: str
    entity_id: str
    topic: EvidenceTopic
    heading: str
    character_start: int
    character_end: int
    basis: str


@dataclass(frozen=True, slots=True)
class TableNeed:
    """A table inside a routed region whose original is retained.

    A table inside a routed region whose original is retained: a view
    the session can deliver. `rank` orders a topic's own queue for the
    session's table share: the latest filing before an earlier one, source
    order inside a filing. v3: the basis a region was routed by (an
    inventory family's recognition, a heading cue) grants no priority --
    the share is dealt across the topics that have tables, one table a
    turn, a table serving several topics taken once (`topic_fair_views`).
    Measured on the retained book under the basis rank, one topic held 2,467
    routed tables and was dealt none.
    """

    document_key: str
    entity_id: str
    topics: tuple[EvidenceTopic, ...]
    placeholder: TablePlaceholder
    region_heading: str
    rank: tuple[int, int] = (0, 0)


@dataclass(frozen=True, slots=True)
class IssuerTopicCell:
    """The machine states of one (issuer, topic) cell before delivery."""

    entity_id: str
    topic: EvidenceTopic
    forms_held: tuple[str, ...]
    regions: int
    regions_by_basis: tuple[tuple[str, int], ...]
    unit_needs: int
    """Window needs of the inventories whose unit serves the topic."""
    typed_observations: int
    tables: int
    tables_without_original: int
    residual: str
    """`QUEUED`: the routed ranges no inventory addressed are the residual
    scope; `COVERED`: every routed range is an inventoried region, so no
    residual search is needed; `BROADER`: no region routes to the topic in
    this issuer's documents, and one bounded broader pass over them is the
    scope; `NO_SOURCE`: the issuer holds no admitted document, so nothing
    can be searched and the gap is the source."""
    residual_ranges: tuple[tuple[str, int, int], ...]
    """`(document key, start, end)` character ranges of the residual scope."""
    gaps: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TopicRouting:
    """The routing of one document set.

    The routing of one document set: every cell, every routed region,
    the table needs, the unit correspondences across filings, the exact
    repeats the allocation collapses, and the topics each need serves.
    """

    rules_id: str
    comparison_rules_id: str
    cells: tuple[IssuerTopicCell, ...]
    regions: tuple[RoutedRegion, ...]
    table_needs: tuple[TableNeed, ...]
    correspondences: tuple[UnitCorrespondence, ...]
    repeats: Mapping[tuple[str, str], tuple[str, str]]
    """`(document key, unit handle)` of an earlier filing's unit whose text
    an exact repeat in a later filing restates -> `(document key, unit
    handle)` of that later unit: the later is read, the earlier is a
    pointer served by the same excerpt, both sources and times kept."""
    topics_by_need: Mapping[int, tuple[EvidenceTopic, ...]] = field(default_factory=dict)
    """By `id(need)`: the topics each inventoried need serves."""
    recency: Mapping[str, int] = field(default_factory=dict)
    comparison_bindings: tuple[tuple[str, str], ...] = ()
    """(closure hash, record hash) of every sealed comparison this routing
    read back or sealed: what the receipt binds for the chain."""
    """Document key -> rank among the issuer's filings, 0 the latest."""

    def lane_of(self, need: EvidenceNeed) -> str:
        """The allocation lane a need is served in: its primary topic."""
        topics = self.topics_by_need.get(id(need))
        return str(topics[0]) if topics else str(EvidenceTopic.CORPORATE_ACTION_LISTING)

    def cell(self, entity_id: str, topic: EvidenceTopic) -> IssuerTopicCell | None:
        """Find one issuer-topic cell in the routing."""
        return next((c for c in self.cells if c.entity_id == entity_id and c.topic == topic), None)


_GOVERNING = {"PART", "ITEM", "NOTE", "SUB"}
_CLOSERS = {
    "ITEM": {"PART", "ITEM"},
    "NOTE": {"PART", "ITEM", "NOTE"},
    "SUB": {"PART", "ITEM", "NOTE", "SUB"},
    "SUB_UNCERTAIN": {"PART", "ITEM", "NOTE", "SUB", "SUB_UNCERTAIN"},
}
"""What closes a cue region opened by a heading of each kind. v4: a
sub-heading the structure is uncertain of opens a region too, closed by
the next heading of any rank, so a false heading only splits a region
and never extends one; it opens one inside the MD&A items only."""
_MDA_TITLE = re.compile(r"management.{0,3}s\s+discussion", re.IGNORECASE)
"""The MD&A item, by its title, on either periodic form."""


def _segments(structure: DocumentStructure) -> tuple[tuple[int, int, str], ...]:
    """Mark body stretches by governing heading family.

    `(start, end, family)` of every stretch of the body governed by one
    heading state, past the cover: the family a chunk inside it would carry.
    """
    headings = [h for h in structure.headings if h.kind in _GOVERNING]
    segments: list[tuple[int, int, str]] = []
    for index, heading in enumerate(headings):
        if heading.character_start < structure.cover_end:
            continue
        end = headings[index + 1].character_start if index + 1 < len(headings) else structure.length
        if end <= heading.character_end:
            continue
        family = structure.locate(heading.character_end, end).family
        segments.append((heading.character_end, end, family))
    return tuple(segments)


def _cue_regions(
    structure: DocumentStructure, cue: re.Pattern[str]
) -> tuple[tuple[int, int, str, str], ...]:
    """Find bounded heading regions matching a topic cue.

    `(start, end, heading text, kind)` of every item, note or sub-heading
    past the cover whose text the cue matches, bounded by the next heading
    of its own rank or above (an uncertain sub-heading: by the next heading
    of any rank, and only inside an MD&A item).
    """
    regions: list[tuple[int, int, str, str]] = []
    headings = structure.headings
    mda = [
        (region.body_start, region.body_end)
        for region in structure.item_regions()
        if _MDA_TITLE.search(region.title) is not None
    ]
    for index, heading in enumerate(headings):
        if heading.character_start < structure.cover_end or heading.kind not in _CLOSERS:
            continue
        if cue.search(heading.text) is None:
            continue
        if heading.kind == "SUB_UNCERTAIN" and not any(
            start <= heading.character_start < end for start, end in mda
        ):
            continue
        end = structure.length
        for later in headings[index + 1 :]:
            if later.kind in _CLOSERS[heading.kind]:
                end = later.character_start
                break
        if end > heading.character_end:
            regions.append((heading.character_end, end, heading.text, heading.kind))
    return tuple(regions)


def _subtract(
    ranges: Sequence[tuple[int, int]], holes: Sequence[tuple[int, int]]
) -> list[tuple[int, int]]:
    """The parts of `ranges` outside every hole, merged and in order."""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    result: list[tuple[int, int]] = []
    for start, end in merged:
        cursor = start
        for h_start, h_end in sorted(holes):
            if h_end <= cursor or h_start >= end:
                continue
            if h_start > cursor:
                result.append((cursor, h_start))
            cursor = max(cursor, h_end)
        if cursor < end:
            result.append((cursor, end))
    return result


class ComparisonMemo(Protocol):
    """Read or seal a bound filing comparison.

    Where a bound filing comparison is read back and a computed one
    sealed (`runtime.reuse.SealedComparisons`).
    """

    def find(self, closure_hash: str, record_hash: str) -> tuple[ComparedUnitRecord, ...] | None:
        """Read a comparison record bound to this closure, when present."""
        ...

    def seal(self, record: FilingComparisonRecord) -> str | None:
        """Stage a computed comparison and return its record hash."""
        ...

    def commit(self) -> frozenset[str]:
        """Commit staged comparisons under one storage admission.

        Place what `seal` staged since the last commit, under one storage
        admission; the record hashes refused, which stay unbound.
        """
        ...


ComparisonBindings = Mapping[str, str]
"""Closure hash -> record hash: the comparisons a sealed receipt of the
chain resolved, the only way a routing reuses one."""


def route_documents(
    *,
    documents: Sequence[RoutedDocument],
    needs_by_document: Mapping[str, DocumentNeeds],
    placeholders: Mapping[str, Sequence[TablePlaceholder]],
    originals: Mapping[str, bool],
    typed_by_issuer: Mapping[str, Mapping[str, int]],
    entity_ids: Sequence[str] = (),
    comparisons: ComparisonMemo | None = None,
    unit_rules: tuple[str, ...] = (),
    bindings: ComparisonBindings | None = None,
) -> TopicRouting:
    """Route every (issuer, topic) cell of a document set from one shared discovery.

    Route every (issuer, topic) cell of a document set from one shared
    discovery: `needs_by_document` are the inventories' needs where a
    family read the filing (the litigation, event and financing regions,
    already combined per document); `placeholders` each document's
    uncarried tables; `originals` whether a markup original stands behind
    the document; `typed_by_issuer` the typed observations by family. No
    session, no model work: the structure is read, the texts are compared,
    nothing is issued. `entity_ids` are the unit's issuers: one that holds
    no admitted document still gets its eight cells, each a source gap.
    """
    by_issuer: dict[str, list[RoutedDocument]] = {}
    for entity_id in entity_ids:
        by_issuer.setdefault(entity_id, [])
    for document in documents:
        by_issuer.setdefault(document.entity_id, []).append(document)
    # Latest first by the filing's own clock; a filing without one sorts
    # after every dated filing, in the set's order, so a known time always
    # precedes an unknown one and no instant is invented.
    recency: dict[str, int] = {}
    for entries in by_issuer.values():
        dated = sorted(
            (d for d in entries if d.clock is not None),
            key=lambda d: (d.clock.isoformat() if d.clock is not None else "", d.document_key),
            reverse=True,
        )
        undated = [d for d in entries if d.clock is None]
        for position, document in enumerate([*dated, *undated]):
            recency[document.document_key] = position
    topics_by_need: dict[int, tuple[EvidenceTopic, ...]] = {}
    for document in documents:
        needs = needs_by_document.get(document.document_key)
        if needs is None:
            continue
        for need in needs.needs:
            topics_by_need[id(need)] = topics_of_need(
                need, document_type=document.document_type, text=document.text
            )
    segments = {d.document_key: _segments(d.structure) for d in documents}
    structured = {
        d.document_key: any(
            h.kind in {"PART", "ITEM", "NOTE"} and h.character_start >= d.structure.cover_end
            for h in d.structure.headings
        )
        for d in documents
    }
    # v3: an inventoried region withdraws a topic's residual search only
    # where the family's own units there serve that topic. A structural
    # inventory replaces search for the requirement it supplies; it does
    # not suppress another topic's statements because they share the
    # region (the litigation note is prose to the product topic, the debt
    # note to the capital topic until a unit there names an issuance).
    inventoried: dict[str, list[tuple[int, int, frozenset[EvidenceTopic]]]] = {}
    for key, needs in needs_by_document.items():
        served: dict[tuple[int, int], set[EvidenceTopic]] = {
            (r.body_start, r.body_end): set() for r in needs.inventory.regions
        }
        for need in needs.needs:
            window = need.window
            if window is None:
                continue
            for (start, end), topics in served.items():
                if start <= window[0] and window[1] <= end:
                    topics.update(topics_by_need.get(id(need), ()))
        inventoried[key] = [
            (start, end, frozenset(topics)) for (start, end), topics in served.items()
        ]
    cells: list[IssuerTopicCell] = []
    regions: list[RoutedRegion] = []
    table_needs: dict[tuple[str, int], TableNeed] = {}
    for entity_id in sorted(by_issuer):
        entries = sorted(by_issuer[entity_id], key=lambda d: recency[d.document_key])
        forms = tuple(sorted({d.document_type.upper() for d in entries}))
        for topic in TOPICS:
            route = TOPIC_ROUTES[topic]
            issuer_regions: list[RoutedRegion] = []
            residual: list[tuple[str, int, int]] = []
            unit_needs = 0
            tables = 0
            tables_without_original = 0
            any_inventoried = False
            for document in entries:
                key = document.document_key
                needs = needs_by_document.get(key)
                routed_ranges: list[tuple[int, int]] = []
                if needs is not None:
                    for region in needs.inventory.regions:
                        if region.family not in route.unit_families:
                            continue
                        if region.family == CORPORATE_EVENT_FAMILY and topic not in _event_topics(
                            region.heading, "", document.document_type
                        ):
                            continue
                        issuer_regions.append(
                            RoutedRegion(
                                document_key=key,
                                entity_id=entity_id,
                                topic=topic,
                                heading=region.heading[:160],
                                character_start=region.body_start,
                                character_end=region.body_end,
                                basis=f"inventory:{region.family}",
                            )
                        )
                        any_inventoried = True
                    for need in needs.needs:
                        if need.priority > 0 and topic in topics_by_need.get(id(need), ()):
                            unit_needs += 1
                for start, end, family in segments[key]:
                    if family in route.structure_families:
                        routed_ranges.append((start, end))
                if not structured[key] and document.structure.length > document.structure.cover_end:
                    # No part, item or note heading past the cover (a sub-heading
                    # maps to no family): no shape to route by, so the body is
                    # every topic's residual scope.
                    routed_ranges.append((document.structure.cover_end, document.structure.length))
                    issuer_regions.append(
                        RoutedRegion(
                            document_key=key,
                            entity_id=entity_id,
                            topic=topic,
                            heading="(no governing heading past the cover)",
                            character_start=document.structure.cover_end,
                            character_end=document.structure.length,
                            basis="unstructured document",
                        )
                    )
                for start, end, heading_text, kind in _cue_regions(
                    document.structure, route.heading_cue
                ):
                    routed_ranges.append((start, end))
                    issuer_regions.append(
                        RoutedRegion(
                            document_key=key,
                            entity_id=entity_id,
                            topic=topic,
                            heading=heading_text[:160],
                            character_start=start,
                            character_end=end,
                            basis=f"heading cue:{kind}",
                        )
                    )
                holes = [
                    (start, end)
                    for start, end, served_topics in inventoried.get(key, ())
                    if topic in served_topics
                ]
                for start, end in _subtract(routed_ranges, holes):
                    residual.append((key, start, end))
                if route.tables:
                    covering = [
                        (r.character_start, r.character_end, r.heading, r.basis)
                        for r in issuer_regions
                        if r.document_key == key
                    ]
                    for placeholder in placeholders.get(key, ()):
                        holding = [
                            (h, b)
                            for s, e, h, b in covering
                            if s <= placeholder.character_start and placeholder.character_end <= e
                        ]
                        if not holding:
                            continue
                        if not originals.get(key, False):
                            tables_without_original += 1
                            continue
                        tables += 1
                        heading = holding[0][0]
                        table_rank = (recency[key], placeholder.ordinal)
                        held = table_needs.get((key, placeholder.ordinal))
                        table_needs[(key, placeholder.ordinal)] = TableNeed(
                            document_key=key,
                            entity_id=entity_id,
                            topics=(*(held.topics if held else ()), topic),
                            placeholder=placeholder,
                            region_heading=held.region_heading if held else heading,
                            rank=min(table_rank, held.rank) if held else table_rank,
                        )
            if not entries:
                residual_state = "NO_SOURCE"
            elif residual:
                residual_state = "QUEUED"
            elif any_inventoried or issuer_regions:
                residual_state = "COVERED"
            else:
                residual_state = "BROADER"
                residual = [
                    (d.document_key, d.structure.cover_end, d.structure.length)
                    for d in entries
                    if d.structure.length > d.structure.cover_end
                ]
            by_basis: dict[str, int] = {}
            for routed in issuer_regions:
                by_basis[routed.basis] = by_basis.get(routed.basis, 0) + 1
            typed = sum(
                typed_by_issuer.get(entity_id, {}).get(family, 0) for family in route.typed_families
            )
            gaps: list[str] = []
            # A periodic report outside the window is no gap: the window is
            # what each issuer filed recently, and nothing is read as a baseline.
            if not entries:
                gaps.append("SOURCE_GAP: no admitted document held for the issuer")
            if tables_without_original:
                gaps.append(
                    f"REPRESENTATION_GAP: {tables_without_original} table(s) in routed regions "
                    "without a retained original"
                )
            if residual_state == "BROADER":
                gaps.append(
                    "ROUTE_GAP: no region routes to the topic; one bounded broader pass over "
                    "the issuer's documents"
                )
            cells.append(
                IssuerTopicCell(
                    entity_id=entity_id,
                    topic=topic,
                    forms_held=forms,
                    regions=len(issuer_regions),
                    regions_by_basis=tuple(sorted(by_basis.items())),
                    unit_needs=unit_needs,
                    typed_observations=typed,
                    tables=tables,
                    tables_without_original=tables_without_original,
                    residual=residual_state,
                    residual_ranges=tuple(residual),
                    gaps=tuple(gaps),
                )
            )
            regions.extend(issuer_regions)
    correspondences, repeats, comparison_bindings = _correspondences(
        by_issuer,
        needs_by_document,
        recency,
        comparisons=comparisons,
        unit_rules=unit_rules,
        bindings=bindings or {},
    )
    if comparisons is not None:
        # The computed records of this routing placed together, under one
        # admission; a refused batch is used for this routing and bound to
        # nothing, so the chain computes those pairs again.
        refused = comparisons.commit()
        if refused:
            comparison_bindings = tuple(
                binding for binding in comparison_bindings if binding[1] not in refused
            )
    return TopicRouting(
        rules_id=ROUTING_RULES_ID,
        comparison_rules_id=COMPARISON_RULES_ID,
        cells=tuple(cells),
        regions=tuple(regions),
        table_needs=tuple(
            sorted(table_needs.values(), key=lambda need: (need.rank, need.document_key))
        ),
        correspondences=correspondences,
        repeats=repeats,
        topics_by_need=topics_by_need,
        recency=recency,
        comparison_bindings=comparison_bindings,
    )


def topic_fair_views(
    table_needs: Sequence[TableNeed],
    *,
    share: int,
    renderable: Callable[[TableNeed], bool],
) -> list[TableNeed]:
    """Deal the session's table share across the topics that have tables.

    Deal the session's table share across the topics that have tables:
    the topics in their declared order, each taking the best-ranked table
    of its own queue it has not been dealt, a table serving several topics
    taken once and counted for all of them, a topic with no table left
    passed over, until the share is spent or no table remains. A table the
    boundary cannot render is passed over by name (`renderable`), never
    counted. Deterministic: the same needs deal the same views.
    """
    queues: dict[EvidenceTopic, list[TableNeed]] = {topic: [] for topic in TOPICS}
    for need in sorted(table_needs, key=lambda n: (n.rank, n.document_key)):
        for topic in need.topics:
            queues[topic].append(need)
    positions: dict[EvidenceTopic, int] = dict.fromkeys(TOPICS, 0)
    dealt: set[tuple[str, int]] = set()
    refused: set[tuple[str, int]] = set()
    views: list[TableNeed] = []
    while len(views) < share:
        progressed = False
        for topic in TOPICS:
            queue = queues[topic]
            while positions[topic] < len(queue):
                need = queue[positions[topic]]
                positions[topic] += 1
                key = (need.document_key, need.placeholder.ordinal)
                if key in dealt or key in refused:
                    continue
                if not renderable(need):
                    refused.add(key)
                    continue
                dealt.add(key)
                views.append(need)
                progressed = True
                break
            if len(views) >= share:
                break
        if not progressed:
            break
    return views


def _units_of(document: RoutedDocument, needs: DocumentNeeds) -> tuple[ComparableUnit, ...]:
    units: list[ComparableUnit] = []
    for matter in needs.inventory.matters:
        text = "\n".join(document.text[start:end] for start, end in matter.ranges)
        units.append(
            ComparableUnit(
                document_key=document.document_key,
                handle=matter.handle,
                family=matter.family,
                title=matter.title,
                text=text,
                case_numbers=matter.case_numbers,
                aliases=matter.aliases,
            )
        )
    return tuple(units)


def _compare_pair(
    *,
    entity_id: str,
    family: str,
    later: RoutedDocument,
    later_units: Sequence[ComparableUnit],
    earlier: RoutedDocument,
    earlier_units: Sequence[ComparableUnit],
    digests: dict[str, str],
    comparisons: ComparisonMemo | None,
    unit_rules: tuple[str, ...],
    bindings: ComparisonBindings,
    bound: list[tuple[str, str]],
) -> tuple[UnitCorrespondence, ...]:
    """The pair's correspondences.

    The pair's correspondences: read back from the sealed record the
    chain's receipt binds to the pair's closure when it names one and the
    record is there and verifies, else computed and sealed (the binding
    appended to `bound` for the receipt either way). A closure the chain
    binds nothing to is computed, never looked up: a record on disk is a
    claim, the receipt's binding is the authority. Document keys are this
    routing's; the record keeps only what is the pair's own (the unit
    handles, the states, the measures).
    """
    if comparisons is None:
        return compare_filings(entity_id=entity_id, later=later_units, earlier=earlier_units)
    for document in (later, earlier):
        if document.document_key not in digests:
            digests[document.document_key] = hashlib.sha256(
                document.text.encode("utf-8")
            ).hexdigest()

    def units_hash(units: Sequence[ComparableUnit]) -> str:
        return FilingComparisonRecord.units_hash_of(
            tuple(
                (u.handle, u.family, hashlib.sha256(u.text.encode("utf-8")).hexdigest())
                for u in units
            )
        )

    closure = {
        "entity_id": entity_id,
        "family": family,
        "later_content_sha256": digests[later.document_key],
        "earlier_content_sha256": digests[earlier.document_key],
        "later_units_hash": units_hash(later_units),
        "earlier_units_hash": units_hash(earlier_units),
        "comparison_rules_id": COMPARISON_RULES_ID,
        "structure_rules_id": STRUCTURE_RULES_ID,
    }
    key = FilingComparisonRecord.closure_hash(**closure, unit_rules=unit_rules)
    record_hash = bindings.get(key)
    held = None if record_hash is None else comparisons.find(key, record_hash)
    if held is not None and record_hash is not None:
        bound.append((key, record_hash))
        return tuple(
            UnitCorrespondence(
                entity_id=entity_id,
                family=family,
                later_document_key=later.document_key,
                later_handle=unit.later_handle,
                earlier_document_key=(
                    earlier.document_key if unit.state != "FIRST_OBSERVED" else None
                ),
                earlier_handle=unit.earlier_handle,
                state=unit.state,
                ratio=unit.ratio,
                changed_characters=unit.changed_characters,
                basis=unit.basis,
            )
            for unit in held
        )
    computed = compare_filings(entity_id=entity_id, later=later_units, earlier=earlier_units)
    units = tuple(
        ComparedUnitRecord(
            later_handle=value.later_handle,
            earlier_handle=value.earlier_handle,
            state=value.state,
            ratio=value.ratio,
            changed_characters=value.changed_characters,
            basis=value.basis[:200],
        )
        for value in computed
    )
    content = {
        **closure,
        "unit_rules": list(unit_rules),
        "units": [unit.model_dump(mode="json") for unit in units[:4096]],
        "results_hash": FilingComparisonRecord.results_hash_of(units[:4096]),
        "comparison_hash": key,
    }
    record = FilingComparisonRecord(
        **closure,
        unit_rules=unit_rules,
        units=units[:4096],
        results_hash=FilingComparisonRecord.results_hash_of(units[:4096]),
        comparison_hash=key,
        record_hash=FilingComparisonRecord.record_hash_of(content),
    )
    if comparisons.seal(record) is not None:
        bound.append((key, record.record_hash))
    return computed


def _correspondences(
    by_issuer: Mapping[str, Sequence[RoutedDocument]],
    needs_by_document: Mapping[str, DocumentNeeds],
    recency: Mapping[str, int],
    *,
    comparisons: ComparisonMemo | None = None,
    unit_rules: tuple[str, ...] = (),
    bindings: ComparisonBindings | None = None,
) -> tuple[
    tuple[UnitCorrespondence, ...],
    dict[tuple[str, str], tuple[str, str]],
    tuple[tuple[str, str], ...],
]:
    """Compare each filing's units with its prior issuer filing.

    Each filing's units against the previous filing of the same issuer
    (by acceptance) that inventoried the same family; the exact repeats as
    the allocation's collapse map. With `comparisons`, a pair's comparison
    is read back from the sealed record the chain's `bindings` name for
    its closure and sealed when computed (`FilingComparisonRecord`); the
    third result is every (closure, record) the routing stands on, for
    the receipt to bind.
    """
    results: list[UnitCorrespondence] = []
    repeats: dict[tuple[str, str], tuple[str, str]] = {}
    digests: dict[str, str] = {}
    bound: list[tuple[str, str]] = []
    for entity_id in sorted(by_issuer):
        ordered = sorted(by_issuer[entity_id], key=lambda d: recency[d.document_key])
        with_units = [
            (document, _units_of(document, needs_by_document[document.document_key]))
            for document in ordered
            if document.document_key in needs_by_document
        ]
        for index, (document, units) in enumerate(with_units):
            families = {unit.family for unit in units}
            for family in sorted(families):
                later = [u for u in units if u.family == family]
                earlier_units: tuple[ComparableUnit, ...] = ()
                earlier_document: RoutedDocument | None = None
                for previous, previous_units in with_units[index + 1 :]:
                    candidates = [u for u in previous_units if u.family == family]
                    if candidates:
                        earlier_units = tuple(candidates)
                        earlier_document = previous
                        break
                if not earlier_units or earlier_document is None:
                    continue
                for value in _compare_pair(
                    entity_id=entity_id,
                    family=family,
                    later=document,
                    later_units=later,
                    earlier=earlier_document,
                    earlier_units=earlier_units,
                    digests=digests,
                    comparisons=comparisons,
                    unit_rules=unit_rules,
                    bindings=bindings or {},
                    bound=bound,
                ):
                    results.append(value)
                    if (
                        value.state == "EXACT_REPEAT"
                        and value.earlier_document_key is not None
                        and value.earlier_handle is not None
                        and value.later_handle is not None
                    ):
                        repeats[(value.earlier_document_key, value.earlier_handle)] = (
                            document.document_key,
                            value.later_handle,
                        )
    # A repeat of a repeat points at the latest reading.
    for key in list(repeats):
        target = repeats[key]
        seen = {key}
        while target in repeats and target not in seen:
            seen.add(target)
            target = repeats[target]
        repeats[key] = target
    return tuple(results), repeats, tuple(bound)


__all__ = [
    "LITIGATION_FAMILY",
    "ROUTING_RULES_ID",
    "TABLE_VIEWS_PER_SESSION",
    "TOPICS",
    "TOPIC_ROUTES",
    "ComparisonBindings",
    "ComparisonMemo",
    "IssuerTopicCell",
    "RoutedDocument",
    "RoutedRegion",
    "TableNeed",
    "TopicRoute",
    "TopicRouting",
    "route_documents",
    "topics_of_need",
    "topics_of_unit",
]
