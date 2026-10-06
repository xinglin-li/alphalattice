"""The deterministic evidence packet: an installed query program, run by the Host.

The analyst no longer searches. The Host runs one fixed program of topic
queries over the frozen generation, reads the top spans, and hands the analyst
the result as a bounded packet. Two consequences are the point:

- which spans reach the analyst is a function of the generation and the program,
  so it is reproducible, auditable and testable offline (recall of a planted
  claim is a measurable property of the program, not of a model's search); and
- one semantic execution is enough, because there is nothing left to iterate
  over.

The program is analysis policy: its identity enters the Host analysis policy
binding through this module's source closure and the receipt's
`query_program_hash`.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from typing import NamedTuple

from alphalattice.kernel.shared_kernel.identity import canonical_hash

from ..contracts import (
    AlternativeEvidenceClass,
    AlternativeEvidenceRequest,
    AlternativeEvidenceSnapshot,
)
from ..documents.contracts import AlternativeEvidenceDocumentSet
from ..retrieval.contracts import (
    AlternativeEvidenceResolvedSpan,
    AlternativeEvidenceRetrievalGeneration,
    RetrievalGenerationRecord,
    TableViewBinding,
)
from ..retrieval.session import (
    MAXIMUM_ISSUED_CANDIDATE_WINDOWS,
    MAXIMUM_ISSUED_MATTER_WINDOWS,
    MAXIMUM_MATTER_READS,
    AlternativeEvidenceRetrievalSession,
    InspectedDocument,
    passage_hash,
)
from .contracts import (
    CANDIDATE_FRONTIER_LIMIT,
    AlternativeEvidenceResearchObligation,
    AlternativeEvidenceRetrievalAccessReceipt,
    CandidateFrontier,
    ComparisonBindingRecord,
    EvidenceNeedRecord,
    EvidenceQueryRecord,
    EvidenceSpanGroup,
    EvidenceTopic,
    IssuerTopicCellRecord,
    LitigationMatterDocument,
    LitigationMatterRecord,
    LitigationRegionRecord,
    MatterAttributionRecord,
    MatterWindowRecord,
    PendingCandidateRecord,
    ProvisionalMatterRecord,
    SelectedWindowFacet,
    TableViewRecord,
    TableViewRefusalRecord,
    TopicRoutingRecord,
    TypedDisclosureField,
    TypedDisclosureInstance,
    TypedDisclosureObservation,
    TypedDisclosureRecord,
    UnassignedRangeRecord,
    UnitCorrespondenceRecord,
)
from .disclosures import (
    TYPED_DISCLOSURE_RULES_ID,
    DraftInstance,
    DraftObservation,
    TypedDisclosureState,
    definitions_hash,
    extract_document,
)
from .events import (
    CORPORATE_EVENT_FAMILY,
    DISCLOSURE_UNIT_INVENTORY_RULES_ID,
    EVENT_CUE,
    EVENT_INVENTORY_RULES_ID,
    LATE_FILING_FORMS,
    REFERS_TO_EVENTS,
    event_inventory,
)
from .financing import (
    FINANCING_CUE,
    FINANCING_FAMILY,
    FINANCING_INVENTORY_RULES_ID,
    REFERS_TO_FINANCING,
    financing_inventory,
)
from .matters import (
    MATTER_INVENTORY_RULES_ID,
    MATTER_SEGMENTATION_RULES_ID,
    TOPIC_LANES_ALLOCATION_ID,
    AllocatedWindow,
    Allocation,
    DocumentNeeds,
    EvidenceNeed,
    MatterInventory,
    allocate_windows,
    attribute_range,
    combine_needs,
    document_needs,
    matter_inventory,
    reference_target_state,
    required_reference_needs,
    search_hit_corresponds,
)
from .operations import (
    OPERATIONS_CUE,
    OPERATIONS_FAMILY,
    OPERATIONS_INVENTORY_RULES_ID,
    REFERS_TO_OPERATIONS,
    operations_inventory,
)
from .routing import (
    TABLE_VIEWS_PER_SESSION,
    TOPICS,
    ComparisonMemo,
    RoutedDocument,
    TableNeed,
    TopicRouting,
    route_documents,
    topic_fair_views,
)


@dataclass(frozen=True, slots=True)
class EvidenceQuery:
    """Declare one bounded question for evidence search."""

    query_id: str
    topic: EvidenceTopic
    text: str
    kind: str = "ADVERSE"
    """`ADVERSE` (the topic's downside events) or `STATE` (its compliance,
    resolution and routine-event families)."""


EVIDENCE_QUERY_PROGRAM_ID = "alternative-evidence.query-program.v3"
"""v3: the obligation's two checks are asked, not one. Each topic keeps its
focused adverse-event question and gains one or two *state* questions for the
topic's compliance, resolution and routine-event families; each query's best
twenty per issuer is read twelve deep per filing; same-filing hits whose whole
text is the same are grouped; each issuer's batch takes every question's best
passage before any question's second. (v2 asked the adverse half only,
read six deep, grouped on previews and filled the batch by score across
questions.)"""

# One question per event family, phrased as a question.
#
# Focused, because a question that lists three unrelated families -- a lawsuit,
# an application to a regulator, a settlement with a government -- scored every
# one of them badly: measured on the admitted corpus the merged legal question
# lost the two litigation facts that its focused form kept. The cross-encoder
# that orders the final cut was trained on question-form queries against
# passages; measured on the admitted corpus, rewriting the program as
# questions moved the annotated facts up within their own issuer for fifteen
# of nineteen comparable occurrences and down for four.
#
# Two kinds of question per topic, because the obligation requires two checks:
# supporting evidence and contradicting evidence. The eight adverse questions
# ask for the topic's downside events. Measured on the seven source-confirmed
# facts the eight alone never delivered (a covenant compliance statement, new
# secured financing, a joint venture, a tender offer, amended bylaws, a
# shareholder vote, an outstanding balance), none was asked about: every one is
# a compliance, financing, venture, liability-management or governance *state*
# fact. The twelve state questions ask for those families -- one family per
# question, or two that are read as one (settled/dismissed/resolved with
# not-material; buybacks with dividends) -- and a de-risk statement is what a
# state question returns, so contradicting evidence is found on purpose.
# Nothing here names a company, an amount, a document, an offset or an
# expected answer. Twenty queries: the session's search budget is twenty-four.
EVIDENCE_QUERY_PROGRAM: tuple[EvidenceQuery, ...] = (
    EvidenceQuery(
        "Q-LIQUIDITY",
        EvidenceTopic.LIQUIDITY_GOING_CONCERN,
        "Is there substantial doubt about the company's ability to continue as a going "
        "concern, a covenant breach or waiver, or a draw on its credit facility?",
    ),
    EvidenceQuery(
        "Q-DILUTION",
        EvidenceTopic.CAPITAL_DILUTION,
        "Did the company issue shares, sell convertible notes, or take an action that "
        "dilutes existing holders?",
    ),
    EvidenceQuery(
        "Q-LEGAL",
        EvidenceTopic.LEGAL_REGULATORY,
        "Was a lawsuit or class action filed against the company, and in which court?",
    ),
    EvidenceQuery(
        "Q-OPERATIONS",
        EvidenceTopic.OPERATIONS_SUPPLY,
        "Did the company record an impairment or write-down of a facility, asset group "
        "or disposal group, or was production interrupted, curtailed or idled by a "
        "supplier interruption or supply constraint?",
    ),
    EvidenceQuery(
        "Q-PRODUCT",
        EvidenceTopic.PRODUCT_SAFETY_CYBER,
        "Did the company recall a product, report injuries, or suffer a cybersecurity "
        "incident or data breach?",
    ),
    EvidenceQuery(
        "Q-GOVERNANCE",
        EvidenceTopic.GOVERNANCE_CONTROLS,
        "Did the company identify a material weakness, restate results, or have an "
        "executive or auditor resign?",
    ),
    EvidenceQuery(
        "Q-COMMERCIAL",
        EvidenceTopic.COMMERCIAL_COUNTERPARTY,
        "Did a significant customer terminate its contract, or did a counterparty "
        "default or file for bankruptcy?",
    ),
    EvidenceQuery(
        "Q-CORPORATE",
        EvidenceTopic.CORPORATE_ACTION_LISTING,
        "Did the company complete or announce an acquisition, divestiture, spin-off, "
        "delisting notice or reverse split?",
    ),
    EvidenceQuery(
        "Q-LIQUIDITY-COVENANTS",
        EvidenceTopic.LIQUIDITY_GOING_CONCERN,
        "Does the company state that it was in compliance with the financial covenants "
        "under its credit agreement and indentures as of the end of the period?",
        kind="STATE",
    ),
    EvidenceQuery(
        "Q-LIQUIDITY-CAPACITY",
        EvidenceTopic.LIQUIDITY_GOING_CONCERN,
        "What borrowing capacity, undrawn commitments, cash or liquidity does the company "
        "report as available, and did it amend or extend a credit facility or "
        "securitization program?",
        kind="STATE",
    ),
    EvidenceQuery(
        "Q-CAPITAL-FINANCING",
        EvidenceTopic.CAPITAL_DILUTION,
        "Did the company enter into a new credit agreement, term loan, mortgage loan or "
        "notes offering, and at what principal amount, interest rate and maturity?",
        kind="STATE",
    ),
    EvidenceQuery(
        "Q-CAPITAL-RETURN",
        EvidenceTopic.CAPITAL_DILUTION,
        "Did the company repurchase shares, authorize or expand a share repurchase "
        "program, or declare, increase or suspend a dividend?",
        kind="STATE",
    ),
    EvidenceQuery(
        "Q-LEGAL-DERISK",
        EvidenceTopic.LEGAL_REGULATORY,
        "Was a lawsuit, claim or regulatory proceeding settled, dismissed or resolved, or "
        "does the company state that it does not expect a pending matter to have a "
        "material adverse effect?",
        kind="STATE",
    ),
    EvidenceQuery(
        "Q-OPERATIONS-PROGRAM",
        EvidenceTopic.OPERATIONS_SUPPLY,
        "Did the company complete or wind down a restructuring, transformation or cost "
        "program, and what charges were recorded for it or are no longer expected?",
        kind="STATE",
    ),
    EvidenceQuery(
        "Q-PRODUCT-RESOLVED",
        EvidenceTopic.PRODUCT_SAFETY_CYBER,
        "Does the company state that a product recall, safety matter or cybersecurity "
        "incident was remediated, closed or had no material effect on its operations?",
        kind="STATE",
    ),
    EvidenceQuery(
        "Q-GOVERNANCE-CONTROLS",
        EvidenceTopic.GOVERNANCE_CONTROLS,
        "Did management conclude that the company's disclosure controls and procedures "
        "and internal control over financial reporting were effective, with no material "
        "changes?",
        kind="STATE",
    ),
    EvidenceQuery(
        "Q-GOVERNANCE-MEETING",
        EvidenceTopic.GOVERNANCE_CONTROLS,
        "Did the company amend its bylaws or charter, or report the results of "
        "shareholder votes on directors, the auditor or shareholder proposals at a "
        "meeting?",
        kind="STATE",
    ),
    EvidenceQuery(
        "Q-CORPORATE-LIABILITY-MANAGEMENT",
        EvidenceTopic.CORPORATE_ACTION_LISTING,
        "Did the company launch a tender offer or exchange offer for its debt securities, "
        "or redeem, repurchase or repay its notes before maturity?",
        kind="STATE",
    ),
    EvidenceQuery(
        "Q-CORPORATE-VENTURE",
        EvidenceTopic.CORPORATE_ACTION_LISTING,
        "Did the company form a joint venture or partnership with another party, or "
        "record noncontrolling interests from a new venture?",
        kind="STATE",
    ),
    EvidenceQuery(
        "Q-COMMERCIAL-AGREEMENTS",
        EvidenceTopic.COMMERCIAL_COUNTERPARTY,
        "Did the company sign, renew, extend or amend a significant customer, supplier, "
        "distribution or offtake agreement, or report a concentration of revenue?",
        kind="STATE",
    ),
)

# Each query returns its own best twenty -- per issuer. This is the cut that
# decides what the packet can contain, because it happens before allocation:
# the reranker scores every candidate the retriever admits, and whatever a
# query does not return is unreachable no matter how the packet is shared out.
# Measured on the admitted corpus, holding the per-issuer batch at ten: twelve
# recalled three of six casebook facts, twenty recalled four, and forty, eighty
# and the whole scored set recalled the same four. Twenty is where the recall
# saturates and is the retrieval contract's ceiling per group. The session
# hands the kernel the filings grouped by issuer and each group takes its own
# twenty of the reranked order. It costs no further inference -- the passages
# it admits were already scored.
QUERY_TOP_K = 20
# How deep into each filing the cross-encoder reads for a question: the
# filing's best twelve fused candidates, not the index spec's six. Measured on
# the admitted corpus, a compliance statement a direct question ranked sixth
# by the dense channel and twelfth by bm25 fused ninth in its 300-chunk
# filing and was never scored; at twelve every one of the seven
# source-confirmed facts reached a query's return, at six five did. Twelve is
# a query-time choice of this program, carried on the request, so the index
# and its generations keep their identity. Twice the pairs of six.
RERANKER_DEPTH_PER_DOCUMENT = 12
# Sixteen per issuer for twenty questions: each question's best passage for
# the issuer, with the four weakest questions' passages dropped, before any
# question's second. 128 for eight issuers; thirty-two reads of four.
SPANS_PER_ISSUER = 16
MAXIMUM_PACKET_SPANS = 128
SPANS_PER_READ = 4
# The residual search's cold rerank-pair budget for one session: half of
# what the unconditional program can spend on a full unit (twenty questions,
# twelve pairs per filing, twenty-four filings). A question that would start
# after the budget is spent is recorded as skipped for budget, never run
# over a narrower depth; the routing record reports what was spent.
RESIDUAL_RERANK_PAIR_BUDGET = 20 * RERANKER_DEPTH_PER_DOCUMENT * 24 // 2
# The integrated selection deals each issuer's sixteen residual spans by
# issuer-topic cell -- every cell's best distinct candidate before any
# cell's second -- and seals what the batch had no room for as a pending
# plan a later session of the chain reads without a search. Measured on
# the retained book under the question-round rule (section Q, W0): ten of
# the 63 development units and three of the 45 cases were returned whole
# or in part and cut by one issuer's sixteen question-best passages; none
# by the return limit, none by the packet cap. v2: inside a cell the
# candidates are dealt by their questions' own ranks in rounds -- every
# question's first, then every question's second -- never by a score
# compared across questions (v1 ordered the cell by score, and a
# question whose scores run lower ranked its first hit 21st in a cell).
# v3: a candidate whose range a span of another channel already delivered
# is COVERED, not read: its questions travel to the covering span and its
# turn goes to the cell's next candidate (measured on the release copy,
# 18% of the residual reads repeated a range another channel held). Every
# returned candidate the batch did not read is sealed in the frontier, not
# six a cell (75% of the paid, reranked candidates were dropped uncounted
# by name); a continuation reads the frontier round by round and skips
# what the chain has covered since. v4: COVERED means delivered whole --
# the exact union of the same filing's unit windows, typed statements and
# read candidates holds every character of the candidate's range (v3 took
# a span holding 80% of it, which discharged a candidate whose undelivered
# tail could hold the qualification or the counter-statement); a table
# page covers nothing (its range is a placeholder line, its content rows
# rendered from the original); a partial overlap leaves the candidate
# pending and is reported apart.
RESIDUAL_SELECTION_RULES_ID = "alternative-evidence.residual-selection.issuer-topic-cells.v4"
# The context-complete allocation (`allocation=CONTEXT_COMPLETE`, section
# V's Gate A: a cell's later turns completed the source paragraph of the
# focal it read before broadening; measured at +1 cumulative unit and -1
# displaced first-response unit) is retired without a supported consumer
# (record section X); the receipts dealt under it name
# `contracts.CONTEXT_COMPLETE_SELECTION_RULES_ID` and read back unchanged.
MAXIMUM_SEARCHES = 24
"""The receipt's bound on searches a session runs (`search_call_count`).
The receipt's `queries` hold up to 26 records: the program's twenty, run
or skipped by name, and -- on receipts sealed under the retired
gap-directed residual search (`residual_search=GAP_DIRECTED`, section V,
Gate C: up to six region-scoped searches after the bank, recorded as
`Q-<TOPIC>-GAP<n>`; falsified as tested and retired, record section X)
-- up to six more, which still read back."""


CANDIDATE_READS_PER_ISSUER = SPANS_PER_ISSUER
# A continuation session reads sealed candidates as a first session reads
# its residual spans: at most sixteen an issuer, under the program's
# thirty-two reads of four, no search, no pair.


def topic_fair_order(program: tuple[EvidenceQuery, ...]) -> tuple[EvidenceQuery, ...]:
    """The order the residual bank runs in under a pair budget.

    The order the residual bank runs in under a pair budget: every topic's
    adverse question in program order, then every topic's first state
    question before any topic's second, topics in program order within a
    round -- so the budget's stop falls on the topics' last questions, not
    on the last topics. Measured on the book under the program's own order,
    the stop fell on the same three questions of two topics on every unit
    (`Q-CORPORATE-LIABILITY-MANAGEMENT`, `Q-CORPORATE-VENTURE`,
    `Q-COMMERCIAL-AGREEMENTS`) while liquidity ran three. The program's own
    identity and the unconditional path are untouched: this is the routed
    selection's execution order alone.
    """
    adverse = tuple(query for query in program if query.kind != "STATE")
    rounds: dict[int, list[EvidenceQuery]] = {}
    seen: dict[EvidenceTopic, int] = {}
    for query in program:
        if query.kind != "STATE":
            continue
        position = seen.get(query.topic, 0)
        seen[query.topic] = position + 1
        rounds.setdefault(position, []).append(query)
    return (*adverse, *(query for position in sorted(rounds) for query in rounds[position]))


def query_program_hash(program: tuple[EvidenceQuery, ...] = EVIDENCE_QUERY_PROGRAM) -> str:
    """Hash the declared evidence query program."""
    return str(
        canonical_hash(
            {
                "program_id": EVIDENCE_QUERY_PROGRAM_ID,
                "queries": tuple(
                    {
                        "query_id": value.query_id,
                        "topic": value.topic,
                        "text": value.text,
                        "kind": value.kind,
                    }
                    for value in program
                ),
                "top_k": QUERY_TOP_K,
                "top_k_scope": "per-issuer document group",
                "reranker_depth_per_document": RERANKER_DEPTH_PER_DOCUMENT,
                "maximum_spans": MAXIMUM_PACKET_SPANS,
                "spans_per_issuer": SPANS_PER_ISSUER,
                "spans_per_read": SPANS_PER_READ,
                "allocation": "each question's best per issuer first, by score within a round",
                "duplicates": "same-filing identical whole passages grouped, recorded, not deleted",
            }
        )
    )


@dataclass(frozen=True, slots=True)
class AlternativeEvidencePacket:
    """Everything an analyst may reason over. Exact spans, no full documents."""

    request: AlternativeEvidenceRequest
    obligation: AlternativeEvidenceResearchObligation
    snapshot: AlternativeEvidenceSnapshot
    document_set: AlternativeEvidenceDocumentSet
    receipt: AlternativeEvidenceRetrievalAccessReceipt
    spans: tuple[AlternativeEvidenceResolvedSpan, ...]


def span_reading_order(
    spans: Sequence[AlternativeEvidenceResolvedSpan], entity_ids: Sequence[str]
) -> tuple[AlternativeEvidenceResolvedSpan, ...]:
    """The order an agent reads a packet's excerpts in.

    The order an agent reads a packet's excerpts in: by issuer, in the
    request's order (an excerpt of no requested issuer last); within an
    issuer the latest filing first; within a filing by position.
    """
    rank = {entity_id: index for index, entity_id in enumerate(entity_ids)}
    return tuple(
        sorted(
            spans,
            key=lambda value: (
                rank.get(value.entity_id, len(rank)),
                value.entity_id,
                -value.available_at.timestamp(),
                value.document_handle,
                value.character_start,
                value.span_handle,
            ),
        )
    )


def span_aliases(
    spans: Sequence[AlternativeEvidenceResolvedSpan], entity_ids: Sequence[str]
) -> dict[str, AlternativeEvidenceResolvedSpan]:
    """Assign stable aliases to delivered span handles.

    The name every view gives each delivered excerpt -- `S1`, `S2`.. in
    reading order -- and the excerpt the Host maps it back to. A function of
    the packet alone, so every view and every sealer agree.
    """
    return {
        f"S{index}": value
        for index, value in enumerate(span_reading_order(spans, entity_ids), start=1)
    }


CellCandidate = tuple[int, int, int, str, str, str, str, str]
"""One candidate of the integrated selection: the issuer-local rank the
question gave it, the question's program order, the global rank, span
handle, issuer, document handle, whole-passage content hash, query id."""


DeliveredRanges = Mapping[str, Sequence[tuple[int, int, str]]]
"""Per document handle, the character ranges delivered spans of other
channels hold, each with its span handle."""


def delivered_ranges(
    spans: Sequence[AlternativeEvidenceResolvedSpan],
) -> dict[str, list[tuple[int, int, str]]]:
    """The ranges the delivered spans hold, by document handle.

    The ranges the delivered spans hold, by document handle: unit
    windows, typed statements and candidates read -- each span's range is
    the excerpt actually delivered, bounded reads included. A span of the
    residual batch itself is not a cover for another candidate, and a
    table page is not: its range is the placeholder line of the canonical
    text, while what it delivers is rows rendered from the original, so it
    proves nothing about a passage's characters.
    """
    out: dict[str, list[tuple[int, int, str]]] = {}
    for span in spans:
        if span.span_handle.startswith(("SPAN-S", "SPAN-X")):
            continue
        out.setdefault(span.document_handle, []).append(
            (span.character_start, span.character_end, span.span_handle)
        )
    return out


def covering_span_handles(
    delivered: DeliveredRanges, document_handle: str, start: int, end: int
) -> tuple[str, ...]:
    """Find spans that jointly cover a candidate source range.

    The delivered spans of the same filing whose ranges, joined, hold
    the candidate's whole range `[start, end)` -- in range order, the one
    holding the first character first; empty when any character of the
    candidate is undelivered. Ranges are character offsets of the
    canonical text on both sides, so a multibyte character is one unit
    here as it is in the excerpt. A span holding most of the candidate
    is not a cover: the undelivered tail may hold the qualification or
    the counter-statement that decides the passage, so an overlap that
    leaves any of the range unread leaves the candidate pending and is
    reported apart (`partially overlapping`).
    """
    if end <= start:
        return ()
    held = sorted(
        (h_start, h_end, handle)
        for h_start, h_end, handle in delivered.get(document_handle, ())
        if h_start < end and start < h_end
    )
    reached = start
    for h_start, h_end, _handle in held:
        if h_start > reached:
            return ()
        reached = max(reached, h_end)
        if reached >= end:
            break
    if reached < end:
        return ()
    return tuple(handle for _s, _e, handle in held)


def overlapping_span_handles(
    delivered: DeliveredRanges, document_handle: str, start: int, end: int
) -> tuple[str, ...]:
    """Find spans intersecting a candidate range without covering it.

    The delivered spans of the same filing that intersect the range
    without jointly holding it whole: the partial overlaps a reader is
    told about beside a pending candidate.
    """
    if covering_span_handles(delivered, document_handle, start, end):
        return ()
    return tuple(
        handle
        for h_start, h_end, handle in sorted(delivered.get(document_handle, ()))
        if h_start < end and start < h_end
    )


def _allocate_by_cell(
    ranked: list[CellCandidate],
    cell_of: Mapping[str, EvidenceTopic],
    covered_by: Callable[[CellCandidate], tuple[str, ...]] | None = None,
) -> tuple[
    tuple[str, ...],
    tuple[EvidenceSpanGroup, ...],
    dict[tuple[str, EvidenceTopic], list[tuple[CellCandidate, tuple[str, ...]]]],
]:
    """Deal residual candidates by issuer-topic cell.

    The integrated selection's batch per issuer, dealt by issuer-topic cell
    (`RESIDUAL_SELECTION_RULES_ID`): the issuer's candidates -- one per span
    identity, holding the best key any question gave it: the lowest
    issuer-local rank, then the earliest question -- grouped by the cell of
    the question that holds that key and ordered inside the cell by that
    key, so the cell is dealt by its questions' own ranks in rounds (every
    question's first, then every question's second); the batch is walked
    in rounds over the cells in the topics' declared order, each round
    taking every cell's next candidate, so no cell's second passage is
    read before every cell's first and no score is compared across cells
    or questions. The same
    sixteen per issuer and 128 per packet hold; a same-content repeat from
    the same filing is grouped under the passage already in the batch and
    costs the cell no turn; a candidate `covered_by` a span another
    channel delivered costs the cell no turn either and is returned with
    the covering handle. What the batch had no room for is returned per
    cell, in the cell's order, for the receipt to seal as the frontier.
    """
    by_issuer: dict[str, list[CellCandidate]] = {}
    for item in ranked:
        by_issuer.setdefault(item[4], []).append(item)
    selected: list[CellCandidate] = []
    grouped: dict[str, list[str]] = {}
    pending: dict[tuple[str, EvidenceTopic], list[tuple[CellCandidate, tuple[str, ...]]]] = {}
    covered: dict[str, tuple[str, ...]] = {}
    for issuer in sorted(by_issuer):
        cells: dict[EvidenceTopic, list[CellCandidate]] = {}
        for item in by_issuer[issuer]:
            cells.setdefault(cell_of[item[7]], []).append(item)
        positions: dict[EvidenceTopic, int] = dict.fromkeys(cells, 0)
        batch: list[CellCandidate] = []
        dealt: set[str] = set()
        full = False

        def deal(
            item: CellCandidate,
            *,
            batch: list[CellCandidate] = batch,
            dealt: set[str] = dealt,
        ) -> bool:
            """One candidate into the batch.

            One candidate into the batch: grouped under a same-content
            passage already there, skipped with its cover named, or
            read; True when it took a read.
            """
            representative = next(
                (kept for kept in batch if kept[5] == item[5] and kept[6] == item[6]),
                None,
            )
            if representative is not None:
                grouped.setdefault(representative[3], []).append(item[3])
                dealt.add(item[3])
                return False
            cover = () if covered_by is None else covered_by(item)
            if cover:
                covered[item[3]] = cover
                dealt.add(item[3])
                return False
            batch.append(item)
            dealt.add(item[3])
            return True

        while not full:
            progressed = False
            for topic in TOPICS:
                items = cells.get(topic)
                if items is None:
                    continue
                if (
                    len(batch) >= SPANS_PER_ISSUER
                    or len(selected) + len(batch) >= MAXIMUM_PACKET_SPANS
                ):
                    full = True
                    break
                # The cell's next candidate; one grouped or covered costs
                # the cell no turn.
                while positions[topic] < len(items):
                    item = items[positions[topic]]
                    positions[topic] += 1
                    if not deal(item):
                        continue
                    progressed = True
                    break
            if not progressed:
                break
        selected.extend(batch)
        for topic in TOPICS:
            items = cells.get(topic)
            if items is None:
                continue
            # The cell's order past the batch: the candidates the turns
            # found covered, then the rest, each with its cover if any.
            rest = [
                (item, covered.get(item[3], ()))
                for item in items
                if item not in batch
                and not any(kept[5] == item[5] and kept[6] == item[6] for kept in batch)
            ]
            if rest:
                pending[(issuer, topic)] = rest
    groups = tuple(
        EvidenceSpanGroup(representative_span_handle=handle, member_span_handles=tuple(members))
        for handle, members in grouped.items()
    )
    return tuple(item[3] for item in selected), groups, pending


TYPED_SPANS_PER_READ = 4
_FAMILY_CODES = {
    "DISCLOSURE_CONTROLS_CONCLUSION": "DCP",
    "UNREGISTERED_EQUITY_SALES": "UES",
    "INSIDER_TRADING_ARRANGEMENTS": "ITA",
    "CUSTOMER_CONCENTRATION": "CCN",
    "CYBERSECURITY_THREAT_EFFECT": "CYB",
}


def select_typed_disclosures(
    *,
    session: AlternativeEvidenceRetrievalSession,
) -> tuple[TypedDisclosureRecord, tuple[AlternativeEvidenceResolvedSpan, ...]]:
    """Read typed disclosures within their own verified budget.

    Read the typed disclosure families from every admitted 10-K and
    10-Q of the session's document set, issue an exact-range span for each
    assertion and for each observation's inspected scope, and read them
    through the verified reader under the typed families' own budget.

    Deterministic: the same generation and rules give the same observations
    and spans. No query, no embedding, no reranker pair. Instances beyond
    the typed span budget keep their text and range in the record with no
    handle, counted per observation, never dropped. v4 (the completeness
    assignment): the budget serves every observation's assertions first,
    document by document, then the inspected scopes of the observations
    that name an unread statement (ambiguous, source unavailable, a
    reference required: NVDA's Item 9B trading arrangements in a table
    layout the rules cannot read), then the not-found and none scopes --
    measured under v3 the scopes of the earlier documents' none and
    not-found observations displaced the later documents' assertions --
    and an observation of a family the form does
    not define (NOT_APPLICABLE) is not sealed: the definitions table the
    record hashes says which forms each family reads.
    """
    documents = session.inspect_documents()
    observations: list[TypedDisclosureObservation] = []
    pending: list[tuple[int, int | None, tuple[int, int]]] = []
    ranges: list[tuple[int | None, tuple[int, int]]]
    """(observation index, instance index or None for the scope, range)."""
    drafts: list[tuple[str, DraftObservation, list[DraftInstance]]] = []
    skipped = 0
    issuers_with_periodic: set[str] = set()
    for document in documents:
        extraction = extract_document(document.text, document.structure)
        if not extraction.observations:
            skipped += 1
            continue
        issuers_with_periodic.add(document.entity_id)
        for draft in extraction.observations:
            if draft.state is TypedDisclosureState.NOT_APPLICABLE:
                continue
            drafts.append((document.document_id, draft, list(draft.instances)))
    # Spans are issued in two passes under the session's typed budget --
    # every observation's assertions document by document, then the
    # inspected scopes of the non-extracted observations; what the budget
    # refuses stays recorded without a handle.
    handles_by_key: dict[tuple[int, int | None], str] = {}
    exhausted = False
    passes: list[tuple[int, str, list[tuple[int | None, tuple[int, int]]]]] = []
    for index, (document_id, _draft, instances) in enumerate(drafts):
        passes.append(
            (
                index,
                document_id,
                [
                    (position, (instance.character_start, instance.character_end))
                    for position, instance in enumerate(instances)
                ],
            )
        )
    # The scopes: the observations that name an unread statement first (a
    # relevant sentence no rule reads, a truncated or incomplete item, a
    # reference to another filing), then the not-found and none scopes.
    unread_first = (
        TypedDisclosureState.AMBIGUOUS,
        TypedDisclosureState.SOURCE_UNAVAILABLE,
        TypedDisclosureState.REFERENCE_REQUIRED,
    )
    for wanted in (True, False):
        for index, (document_id, draft, instances) in enumerate(drafts):
            if (draft.state in unread_first) != wanted:
                continue
            if draft.scope_range is not None and (
                not instances or draft.state is not TypedDisclosureState.EXTRACTED
            ):
                passes.append((index, document_id, [(None, draft.scope_range)]))
    for index, document_id, ranges in passes:
        if exhausted or not ranges:
            continue
        try:
            issued = session.issue_source_spans(document_id, tuple(value for _, value in ranges))
        except ValueError as error:
            if "source_span_budget_exhausted" not in str(error):
                raise
            # The batch did not fit: name what still fits, one range at a time.
            issued = ()
            for _position, span_range in ranges:
                try:
                    issued = (*issued, *session.issue_source_spans(document_id, (span_range,)))
                except ValueError as inner:
                    if "source_span_budget_exhausted" not in str(inner):
                        raise
                    exhausted = True
                    break
        for (slot, span_range), issued_handle in zip(ranges, issued, strict=False):
            handles_by_key[(index, slot)] = issued_handle
            pending.append((index, slot, span_range))
    ordered_handles = [handles_by_key[(index, position)] for index, position, _ in pending]
    spans: list[AlternativeEvidenceResolvedSpan] = []
    for start in range(0, len(ordered_handles), TYPED_SPANS_PER_READ):
        batch = tuple(ordered_handles[start : start + TYPED_SPANS_PER_READ])
        spans.extend(session.read_typed_spans(span_handles=batch))
    read_handles = {value.span_handle for value in spans}
    documents_by_id = {value.document_id: value for value in documents}
    for index, (document_id, draft, instances) in enumerate(drafts):
        document = documents_by_id[document_id]
        sealed_instances: list[TypedDisclosureInstance] = []
        undelivered = 0
        for position, instance in enumerate(instances):
            handle = handles_by_key.get((index, position))
            if handle is not None and handle not in read_handles:
                raise ValueError("alternative_evidence.typed_disclosure_span_unread")
            if handle is None:
                undelivered += 1
            sealed_instances.append(
                TypedDisclosureInstance(
                    instance_id=(
                        f"T-{document.entity_id}-{_FAMILY_CODES[draft.family.value]}"
                        f"-{index:03d}-{position + 1:02d}"
                    ).upper(),
                    span_handle=handle,
                    character_start=instance.character_start,
                    character_end=instance.character_end,
                    subject=instance.subject[:200],
                    action=instance.action,
                    polarity=instance.polarity,
                    period_end=None
                    if instance.period_end is None
                    else instance.period_end.isoformat(),
                    period_text=instance.period_text[:200],
                    period_basis=instance.period_basis[:240],
                    statement_text=instance.statement_text[:2400],
                    fields=tuple(
                        TypedDisclosureField(
                            name=value.name,
                            value=None if value.value is None else value.value[:240],
                            text=value.text[:400] or value.name,
                            character_start=value.character_start,
                            character_end=max(value.character_end, value.character_start + 1),
                            basis=value.basis[:240],
                        )
                        for value in instance.fields[:16]
                    ),
                    qualifiers=tuple(value[:400] for value in instance.qualifiers[:8]),
                    unknown_fields=instance.unknown_fields[:8],
                )
            )
        scope_handle = handles_by_key.get((index, None))
        period = draft.report_period
        observations.append(
            TypedDisclosureObservation(
                observation_id=(
                    f"TO-{document.entity_id}-{document.revision_label}"
                    f"-{_FAMILY_CODES[draft.family.value]}"
                )
                .upper()
                .replace("/", "-")
                .replace("_", "-"),
                entity_id=document.entity_id,
                document_handle=document.document_handle,
                document_type=document.document_type,
                revision_label=document.revision_label,
                family=draft.family.value,
                rule_id=draft.rule_id,
                state=draft.state.value,
                reason=draft.reason[:600],
                part=draft.part,
                item=draft.item,
                region_title=draft.region_title[:160],
                report_period_end=None
                if period.period_end is None
                else period.period_end.isoformat(),
                report_period_status=period.status,
                report_period_basis=period.basis[:200],
                instances=tuple(sealed_instances[:32]),
                unrecognized=tuple(value[:300] for value in draft.unrecognized[:8]),
                references=tuple(value[:200] for value in draft.references[:4]),
                context=tuple(value[:200] for value in draft.context[:6]),
                scope_span_handle=scope_handle,
                inspected_ranges=draft.inspected_ranges[:4],
                undelivered_instance_count=undelivered + max(0, len(sealed_instances) - 32),
            )
        )
    issuers = tuple(
        dict.fromkeys(
            reference.entity_id
            for reference in session.document_set.documents
            if reference.entity_id not in issuers_with_periodic
        )
    )
    record = TypedDisclosureRecord(
        rules_id=TYPED_DISCLOSURE_RULES_ID,
        definitions_hash=definitions_hash(),
        documents_inspected=len(documents),
        documents_skipped_by_form=skipped,
        observations=tuple(observations[:96]),
        span_handles=tuple(ordered_handles),
        read_call_count=session.typed_span_read_count,
        issuers_without_periodic_filing=issuers[:8],
    )
    return record, tuple(spans)


MATTER_WINDOWS_PER_READ = 4
MATTER_WINDOW_BYTES = 3900
"""UTF-8 bytes a matter window may hold: under the reader's 4 KiB per span
at four spans a read, so every window read is delivered whole."""
_MATTER_NEXT_READS = 8


def _plan_corpus(generation: RetrievalGenerationRecord) -> str:
    """What a reading plan's identity binds of the index it reads.

    What a reading plan's identity binds of the index it reads: the
    corpus commitment -- the same bytes under another request are the
    same plan, which is what lets a reused selection be continued -- and,
    for a legacy generation that commits to no corpus, the generation
    itself.
    """
    if isinstance(generation, AlternativeEvidenceRetrievalGeneration):
        return generation.corpus_hash
    return generation.generation_hash


def _litigation_document(
    document: InspectedDocument,
    inventory: MatterInventory,
    *,
    records: Sequence[MatterWindowRecord],
    matter_planned: Mapping[str, int],
    matter_read: Mapping[str, int],
    planned_windows: int,
) -> LitigationMatterDocument:
    """Read one filing's inventory and matter-selection state.

    One filing's inventory and reading state as the record carries it,
    whichever order chose the windows: `inspection` is COMPLETE only when
    every window planned over it is a proved whole read.
    """
    read = [r for r in records if r.status == "READ"]
    pending = [r for r in records if r.status == "PENDING"]
    covered = [r for r in records if r.status in {"READ", "PRIOR"}]
    bounded = [r for r in covered if not r.delivered_whole]
    qualifications: list[str] = []
    for region in inventory.regions:
        if region.end_uncertain:
            qualifications.append(f"region end uncertain: {region.heading[:60]}")
    if inventory.unassigned:
        qualifications.append(
            f"{len(inventory.unassigned)} unassigned range(s) no signpost claimed"
        )
    if bounded:
        qualifications.append(f"{len(bounded)} window(s) delivered bounded, not whole")
    region_bytes = sum(
        len(document.text[r.body_start : r.body_end].strip().encode("utf-8"))
        for r in inventory.regions
    )
    if not planned_windows and region_bytes:
        qualifications.append("region text opened no paragraph to plan")
    if not inventory.regions:
        inspection = "NO_LITIGATION_REGION"
    elif not pending and not bounded and (covered or not region_bytes):
        # Every planned window is a proved whole read: coverage of the
        # plan, and only of the plan -- the qualifications say the rest.
        inspection = "COMPLETE"
    elif covered:
        inspection = "PARTIAL"
    else:
        inspection = "NONE"
    period = document.structure.report_period.period_end
    return LitigationMatterDocument(
        document_handle=document.document_handle,
        entity_id=document.entity_id,
        document_type=document.document_type,
        revision_label=document.revision_label,
        report_period_end=None if period is None else period.isoformat(),
        regions=tuple(
            LitigationRegionRecord(
                kind=region.kind,
                heading=region.heading[:160],
                character_start=region.body_start,
                character_end=region.body_end,
                end_basis=region.end_basis[:240],
                end_uncertain=region.end_uncertain,
                family=region.family,
            )
            for region in inventory.regions[:8]
        ),
        matters=tuple(
            ProvisionalMatterRecord(
                handle=matter.handle,
                title=matter.title[:160] or matter.handle,
                basis=matter.basis[:32],
                named=matter.named,
                region_heading=matter.region_heading[:160],
                group=matter.group[:160],
                character_start=matter.ranges[0][0],
                character_end=matter.ranges[-1][1],
                source_bytes=max(1, matter.source_bytes),
                aliases=tuple(value[:80] for value in matter.aliases[:12]),
                case_numbers=tuple(value[:40] for value in matter.case_numbers[:12]),
                courts=tuple(value[:120] for value in matter.courts[:8]),
                planned_windows=max(1, matter_planned.get(matter.handle, 0)),
                read_windows=matter_read.get(matter.handle, 0),
                family=matter.family,
            )
            for matter in inventory.matters[:160]
        ),
        unassigned=tuple(
            UnassignedRangeRecord(character_start=start, character_end=end, reason=reason[:80])
            for start, end, reason in inventory.unassigned[:64]
        ),
        inspection=inspection,
        qualifications=tuple(value[:120] for value in qualifications[:8]),
        planned_windows=planned_windows,
        read_windows=len(read),
        pending_windows=len(pending),
        region_source_bytes=region_bytes,
        read_source_bytes=sum(r.source_bytes for r in read),
        pending_source_bytes=sum(r.source_bytes for r in pending),
    )


MAXIMUM_REFERENCE_SEARCHES = 2
"""Searches a matter selection may spend on explicit references its
structure could not locate: a fallback for a named gap, never a discovery
step; the integrated selection spends none."""
_FILING_REFERENCE = re.compile(
    r"Form\s+(?P<form>10-[KQ])\s+for\s+the\s+(?:fiscal\s+)?(?:quarter|period|year)\s+ended\s+"
    r"(?P<month>[A-Z][a-z]+)\s+(?P<day>\d{1,2}),\s+(?P<year>\d{4})"
)
_MONTHS = {
    name: number
    for number, name in enumerate(
        (
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ),
        start=1,
    )
}


@dataclass(frozen=True, slots=True)
class MatterEvidenceTrace:
    """What one matter selection did, for the development comparison and the boundary tests.

    What one matter selection did, for the development comparison and
    the boundary tests: every filing's needs, the allocation, and the
    reference searches spent. Memory only; the record carries the compact
    accounting.
    """

    documents: tuple[DocumentNeeds, ...]
    allocation: Allocation
    reference_spans: tuple[AlternativeEvidenceResolvedSpan, ...]
    """Spans a reference search read under the program's budget: they are
    the caller's to account for in the receipt's `read_span_handles`."""
    searches: int


def _referenced_filing(target: str) -> tuple[str, date] | None:
    match = _FILING_REFERENCE.search(target)
    if match is None:
        return None
    try:
        when = date(
            int(match.group("year")), _MONTHS[match.group("month")], int(match.group("day"))
        )
    except (KeyError, ValueError):
        return None
    return match.group("form"), when


LITIGATION_FAMILY = "LITIGATION"
MATTER_FAMILIES = (
    LITIGATION_FAMILY,
    CORPORATE_EVENT_FAMILY,
    FINANCING_FAMILY,
    OPERATIONS_FAMILY,
)
"""The disclosure-unit families the integrated selection inventories, in
the order a record names them: the litigation regions of the periodic
filings (`matters.py`), the corporate-event regions of the periodic
filings and the items of the current reports (`events.py`), the
financing regions of the periodic filings (`financing.py`), and the
operations regions of the periodic filings -- the restructuring,
impairment and operating-charge notes and the dated operations
statements of the narrative items (`operations.py`). The integrated
method inventories them all; a request's `MatterSelectionPolicy` keeps
its three-family spelling as that method's identity (the workspace
manifests bind it), so no sealed request or manifest moves."""


def _inventory_documents(
    session: AlternativeEvidenceRetrievalSession,
    *,
    rules: str,
    families: tuple[str, ...],
) -> tuple[dict[str, InspectedDocument], dict[str, DocumentNeeds], int]:
    """Every admitted filing's needs under the requested families.

    Every admitted filing's needs under the requested families: the
    periodic filings' litigation regions and -- when the corporate-event
    family is asked for -- their event regions and the items of the current
    reports, the two families' needs of one filing combined under one queue
    (`combine_needs`). A filing no requested family reads is skipped by
    form.
    """
    litigation = LITIGATION_FAMILY in families
    events = CORPORATE_EVENT_FAMILY in families
    financing = FINANCING_FAMILY in families
    operations = OPERATIONS_FAMILY in families
    inspected: dict[str, InspectedDocument] = {}
    needs_by_document: dict[str, DocumentNeeds] = {}
    skipped = 0
    for document in session.inspect_documents():
        periodic = document.document_type.upper().startswith("10-")
        current = document.structure.family_form.upper() == "8-K"
        late = document.structure.family_form.upper() in LATE_FILING_FORMS
        if not (
            (periodic and (litigation or financing or operations))
            or (events and (periodic or current or late))
        ):
            skipped += 1
            continue
        document_id = document.document_handle[4:]
        parts: list[DocumentNeeds] = []
        if periodic and litigation:
            parts.append(
                document_needs(
                    document.text,
                    document.structure,
                    matter_inventory(
                        document.text, document.structure, document_id=document_id, rules=rules
                    ),
                    document_key=document.document_id,
                    issuer=document.entity_id,
                    window_bytes=MATTER_WINDOW_BYTES,
                )
            )
        if events:
            parts.append(
                document_needs(
                    document.text,
                    document.structure,
                    event_inventory(document.text, document.structure, document_id=document_id),
                    document_key=document.document_id,
                    issuer=document.entity_id,
                    window_bytes=MATTER_WINDOW_BYTES,
                    cue=EVENT_CUE,
                    refers=REFERS_TO_EVENTS,
                    family=CORPORATE_EVENT_FAMILY,
                )
            )
        if periodic and financing:
            parts.append(
                document_needs(
                    document.text,
                    document.structure,
                    financing_inventory(document.text, document.structure, document_id=document_id),
                    document_key=document.document_id,
                    issuer=document.entity_id,
                    window_bytes=MATTER_WINDOW_BYTES,
                    cue=FINANCING_CUE,
                    refers=REFERS_TO_FINANCING,
                    family=FINANCING_FAMILY,
                )
            )
        if periodic and operations:
            parts.append(
                document_needs(
                    document.text,
                    document.structure,
                    operations_inventory(
                        document.text, document.structure, document_id=document_id
                    ),
                    document_key=document.document_id,
                    issuer=document.entity_id,
                    window_bytes=MATTER_WINDOW_BYTES,
                    cue=OPERATIONS_CUE,
                    refers=REFERS_TO_OPERATIONS,
                    family=OPERATIONS_FAMILY,
                )
            )
        needs = parts[0]
        for part in parts[1:]:
            needs = combine_needs(needs, part, rules_id=DISCLOSURE_UNIT_INVENTORY_RULES_ID)
        inspected[document.document_id] = document
        needs_by_document[document.document_id] = needs
    return inspected, needs_by_document, skipped


def select_matter_evidence(
    *,
    session: AlternativeEvidenceRetrievalSession,
    inventory: SharedInventory,
    routing: TopicRouting,
    byte_budget: int | None = None,
    reference_searches: int = 0,
    prior: LitigationMatterRecord | None = None,
    continued_from: str | None = None,
    session_limit: int | None = None,
    window_limit: int | None = None,
    allowance_reserve: int = 0,
    pending_elsewhere: bool = False,
) -> tuple[
    LitigationMatterRecord, tuple[AlternativeEvidenceResolvedSpan, ...], MatterEvidenceTrace
]:
    """The integrated selection's delivery of the inventoried disclosure units.

    The integrated selection's delivery of the inventoried disclosure
    units: over the shared `inventory` of every admitted filing (the unit
    families of `MATTER_FAMILIES` under `MATTER_SEGMENTATION_RULES_ID`) and
    each filing's needs (every unit's opening and further parts, every
    region-level statement, each unit's qualifications and every explicit
    reference), choose windows in `TOPIC_LANES_ALLOCATION_ID`'s sealed
    service order -- the `routing`'s issuer-topic lanes, the filings'
    recency and the exact repeats -- under the session's one matter
    allowance less the `allowance_reserve` kept for table views, and an
    optional byte budget; read them through the verified reader; and record
    what each delivered excerpt carries at its exact ranges
    (`attributions`) and what each need got (`needs`).

    A reference the structure could not locate may spend at most
    `reference_searches` (<= `MAXIMUM_REFERENCE_SEARCHES`) bounded searches,
    accepting only a hit in the same filing that opens with the cited
    heading; the spans such a read returns are in the trace, for the
    caller's receipt.

    A continuation takes the `prior` sealed record of this very plan (with
    `continued_from`, the hash of the receipt that sealed it) and reads the
    needs the chain has not delivered: the plan's identity (`plan_hash`) is
    the needs -- rules, families, allocation, generation and every need's
    document, kind, unit and range -- not the windows one session chose, so
    a later session continues the same plan under its own allowance. The
    prior's proved windows are carried as PRIOR with the handles their own
    sessions issued and count as delivered; a prior of another plan (a
    retired method's record among them), a prior whose proved windows serve
    no need of this plan or claim more reads than they carry, a prior with
    nothing pending (unless `pending_elsewhere` says the receipt's other
    channels still owe reads), and a chain that would exceed
    `session_limit` or `window_limit` refuse by name. A continuation spends
    no search. Deterministic: the same generation and routing give the same
    needs, windows and reads. No model work otherwise.

    The production plan's prefix and the candidate needs allocation this
    function also dealt were retired by the first-release integration
    (2026-09-22): their sealed records read back, and nothing new is dealt
    under them.
    """
    allocation = TOPIC_LANES_ALLOCATION_ID
    families = MATTER_FAMILIES
    record_rules = MATTER_SEGMENTATION_RULES_ID
    if not 0 <= reference_searches <= MAXIMUM_REFERENCE_SEARCHES:
        raise ValueError("alternative_evidence.matter_reference_searches_invalid")
    allowance = min(MAXIMUM_ISSUED_MATTER_WINDOWS, MAXIMUM_MATTER_READS * MATTER_WINDOWS_PER_READ)
    if not 0 <= allowance_reserve < allowance:
        raise ValueError("alternative_evidence.matter_allowance_reserve_invalid")
    allowance -= allowance_reserve
    inspected, needs_by_document, skipped = inventory
    documents = tuple(needs_by_document.values())
    # An exact repeat's earlier copy is served by the later unit's reading:
    # the need it is passed over as, by the later need at the same part.
    repeat_of: dict[tuple[str, int | None, int | None, str], EvidenceNeed] = {}
    if routing.repeats:
        by_unit: dict[tuple[str, str, int], EvidenceNeed] = {}
        for filing in documents:
            for need in filing.needs:
                if need.matter_handle is not None and need.window is not None:
                    by_unit[(need.document_key, need.matter_handle, need.part)] = need
        for filing in documents:
            for need in filing.needs:
                if need.matter_handle is None or need.window is None:
                    continue
                later_unit = routing.repeats.get((need.document_key, need.matter_handle))
                if later_unit is None:
                    continue
                representative = by_unit.get((later_unit[0], later_unit[1], need.part))
                if representative is not None and representative.window is not None:
                    repeat_of[_need_key(need)] = representative
    plan_hash = str(
        canonical_hash(
            {
                "rules_id": record_rules,
                "allocation_rules_id": allocation,
                "families": list(families),
                "window_bytes": MATTER_WINDOW_BYTES,
                "corpus": _plan_corpus(session.generation),
                "needs": [
                    (
                        inspected[need.document_key].document_handle,
                        need.kind,
                        need.matter_handle,
                        need.character_start,
                        need.character_end,
                        need.part,
                        need.part_count,
                        need.priority,
                    )
                    for filing in documents
                    for need in filing.needs
                ],
            }
        )
    )
    # What the chain has delivered: the prior's proved windows first, this
    # session's reads after them. Each entry is the span, its delivered
    # range and the window it served.
    delivered: dict[str, list[tuple[str, int, int, AllocatedWindow]]] = {}

    def covering(document_key: str, start: int, end: int) -> tuple[str, bool] | None:
        """Find a delivered span covering an exact matter range.

        The first delivered span of the document whose delivered range
        holds `[start, end)`, and whether that span serves other needs.
        """
        for handle, d_start, d_end, window in delivered.get(document_key, []):
            if d_start <= start and end <= d_end:
                own = any(
                    n.character_start == start and n.character_end == end for n in window.needs
                )
                shared = len(window.needs) > 1 or not own
                return handle, shared
        return None

    def repeat_covering(need: EvidenceNeed) -> tuple[str, EvidenceNeed] | None:
        """Find the delivered span for an exact repeated unit.

        The span that read the later unit an exact repeat of this need's
        unit restates, when the chain has read it: the need is then served
        by that excerpt, shared, with its own source named.
        """
        representative = repeat_of.get(_need_key(need))
        if representative is None or representative.window is None:
            return None
        found = covering(representative.document_key, *representative.window)
        return None if found is None else (found[0], representative)

    prior_records: list[MatterWindowRecord] = []
    carried: list[EvidenceNeedRecord] = []
    plan_offset, session_index = 0, 1
    if prior is None:
        if continued_from is not None or session_limit is not None or window_limit is not None:
            raise ValueError("alternative_evidence.litigation_continuation_without_prior")
    else:
        if continued_from is None:
            raise ValueError("alternative_evidence.litigation_continuation_without_prior")
        if prior.allocation_rules_id is None:
            # A record of the retired production plan: read as sealed,
            # never continued.
            raise ValueError("alternative_evidence.litigation_continuation_allocation_unsupported")
        if prior.historical_reference_needs:
            # A candidate record sealed before references carried a target
            # state: its provided references meant any overlap, which no
            # current plan accepts as delivered. Read under its own contract,
            # never continued, never backfilled.
            raise ValueError("alternative_evidence.litigation_continuation_prior_incompatible")
        if (
            prior.rules_id != record_rules
            or prior.allocation_rules_id != allocation
            or tuple(prior.families) != families
            or prior.plan_hash != plan_hash
        ):
            raise ValueError("alternative_evidence.litigation_plan_changed")
        plan_offset = prior.plan_offset + prior.read_windows
        session_index = prior.session_index + 1
        proved = [w for w in prior.windows if w.status in {"READ", "PRIOR"}]
        if len(proved) != plan_offset:
            raise ValueError("alternative_evidence.litigation_prior_window_unproved")
        keys_by_handle = {value.document_handle: key for key, value in inspected.items()}
        for proved_window in proved:
            key = keys_by_handle.get(proved_window.document_handle)
            if proved_window.span_handle is None or key is None:
                raise ValueError("alternative_evidence.litigation_prior_window_unproved")
            served = tuple(
                need
                for need in needs_by_document[key].needs
                if need.window is not None
                and proved_window.character_start <= need.window[0]
                and need.window[1] <= proved_window.character_end
            )
            if not served:
                # A window that serves no need of this plan is not this
                # plan's progress, whatever record carries it.
                raise ValueError("alternative_evidence.litigation_prior_window_unproved")
            delivered.setdefault(key, []).append(
                (
                    proved_window.span_handle,
                    proved_window.character_start,
                    proved_window.character_end,
                    AllocatedWindow(
                        document_key=key,
                        character_start=proved_window.character_start,
                        character_end=proved_window.character_end,
                        source_bytes=proved_window.source_bytes,
                        needs=served,
                    ),
                )
            )
            prior_records.append(proved_window.model_copy(update={"status": "PRIOR"}))
        if not pending_elsewhere and not any(
            need.priority > 0
            and need.window is not None
            and covering(need.document_key, *need.window) is None
            for filing in documents
            for need in filing.needs
        ):
            # A plan every window of which the chain has read continues only
            # when the receipt's other channels -- sealed residual candidates,
            # table pages -- still owe reads; then this record carries the
            # plan forward with nothing dealt.
            raise ValueError("alternative_evidence.litigation_continuation_nothing_pending")
        if prior.session_index > 1 and (prior.session_limit, prior.window_limit) != (
            session_limit,
            window_limit,
        ):
            raise ValueError("alternative_evidence.litigation_continuation_limits_changed")
        if session_limit is None or window_limit is None:
            raise ValueError("alternative_evidence.litigation_continuation_limits_required")
        if session_index > session_limit or plan_offset >= window_limit:
            raise ValueError("alternative_evidence.litigation_continuation_budget_exhausted")
        allowance = min(allowance, window_limit - plan_offset)
        # A reference the prior resolved by a search keeps that resolution:
        # the span is in the chain's set, and a continuation spends none.
        carried = [
            need
            for need in prior.needs
            if need.kind == "REFERENCE"
            and need.status == "PROVIDED"
            and need.span_handle is not None
            and not need.span_handle.startswith("SPAN-M")
        ]
    # The whole plan walks its sealed service order; what the chain has
    # delivered is passed over as progress, never dealt again, and an exact
    # repeat's earlier copy is delivered by the later unit's reading, once
    # that reading is in the chain.
    chosen = allocate_windows(
        documents,
        window_allowance=allowance,
        window_bytes=MATTER_WINDOW_BYTES,
        byte_budget=byte_budget,
        delivered=lambda need: (
            need.window is not None
            and (
                covering(need.document_key, *need.window) is not None
                or repeat_covering(need) is not None
            )
        ),
        lane_of=routing.lane_of,
        lanes_in_order=tuple(str(topic) for topic in TOPICS),
        recency=routing.recency,
        representative_of=(
            None if not repeat_of else (lambda need: repeat_of.get(_need_key(need)))
        ),
    )
    # Issue per document in allocation order, read four at a time.
    series = f"M{session_index:02d}"
    handles: dict[int, str] = {}
    by_document: dict[str, list[int]] = {}
    for position, window in enumerate(chosen.windows):
        by_document.setdefault(window.document_key, []).append(position)
    for document_id, positions in by_document.items():
        issued = session.issue_matter_windows(
            document_id,
            tuple(
                (chosen.windows[p].character_start, chosen.windows[p].character_end)
                for p in positions
            ),
            series=series,
        )
        for position, issued_handle in zip(positions, issued, strict=True):
            handles[position] = issued_handle
    ordered = [handles[position] for position in range(len(chosen.windows))]
    spans: list[AlternativeEvidenceResolvedSpan] = []
    for start in range(0, len(ordered), MATTER_WINDOWS_PER_READ):
        batch = tuple(ordered[start : start + MATTER_WINDOWS_PER_READ])
        spans.extend(session.read_matter_windows(span_handles=batch))
    by_handle = {span.span_handle: span for span in spans}
    if set(by_handle) != set(ordered):
        raise ValueError("alternative_evidence.litigation_window_unread")
    # What each excerpt carries, and what each need got.
    window_records: list[MatterWindowRecord] = [*prior_records]
    for position, window in enumerate(chosen.windows):
        span = by_handle[handles[position]]
        document = inspected[window.document_key]
        attributions = attribute_range(
            needs_by_document[window.document_key],
            delivered=(span.character_start, span.character_end),
            planned=(window.character_start, window.character_end),
        )
        first = window.needs[0]
        window_records.append(
            MatterWindowRecord(
                document_handle=document.document_handle,
                matter_handle=first.matter_handle or "UNASSIGNED",
                part=first.part,
                part_count=first.part_count,
                character_start=window.character_start,
                character_end=window.character_end,
                source_bytes=window.source_bytes,
                status="READ",
                span_handle=span.span_handle,
                delivered_whole=(
                    span.character_start <= window.character_start
                    and span.character_end >= window.character_end
                    and not any("bounded" in value for value in span.limitations)
                ),
                attributions=tuple(
                    MatterAttributionRecord(
                        matter_handle=value.matter_handle,
                        role=value.role,
                        scope=value.scope,
                        basis=value.basis[:160],
                        region_heading=value.region_heading[:160],
                        character_start=value.character_start,
                        character_end=value.character_end,
                    )
                    for value in attributions[:64]
                ),
            )
        )
        delivered.setdefault(window.document_key, []).append(
            (span.span_handle, span.character_start, span.character_end, window)
        )
    refused_why = {_need_key(need): why for need, why in chosen.refused}
    for need, _why in chosen.refused:
        document = inspected[need.document_key]
        need_window = need.window
        if need_window is None:
            continue
        window_records.append(
            MatterWindowRecord(
                document_handle=document.document_handle,
                matter_handle=need.matter_handle or "UNASSIGNED",
                part=need.part,
                part_count=need.part_count,
                character_start=need_window[0],
                character_end=need_window[1],
                source_bytes=max(1, need.source_bytes),
                status="PENDING",
            )
        )
    reference_spans: list[AlternativeEvidenceResolvedSpan] = []
    searches = 0
    need_records: list[EvidenceNeedRecord] = []
    statuses: dict[tuple[str, int | None, int | None, str], tuple[str, str | None]] = {}
    for filing in documents:
        for need in filing.needs:
            need_window = need.window
            if need.priority == 0 or need_window is None:
                continue
            found = covering(need.document_key, need_window[0], need_window[1])
            repeated = repeat_covering(need) if found is None else None
            if found is None and repeated is not None:
                handle, representative = repeated
                status = "PROVIDED_SHARED"
                detail = (
                    f"{need.detail}; exact repeat of {representative.matter_handle} in "
                    f"{inspected[representative.document_key].document_handle}: served by "
                    "that reading, this filing's own text and time kept"
                )
            elif found is None:
                why = refused_why.get(_need_key(need), "not reached")
                status, handle = "PENDING_ALLOWANCE", None
                detail = f"{need.detail}; {why}"
            else:
                handle, shared = found
                status = "PROVIDED_SHARED" if shared else "PROVIDED"
                detail = need.detail
            statuses[(need.document_key, need.character_start, need.character_end, need.kind)] = (
                status,
                handle,
            )
            need_records.append(
                _need_record(
                    inspected[need.document_key],
                    need,
                    status,
                    handle,
                    detail,
                    # A located heading outside the regions is its own window:
                    # delivered when read, located otherwise.
                    target_state=(
                        None
                        if need.kind != "REFERENCE"
                        else ("DELIVERED" if handle is not None else "LOCATED")
                    ),
                )
            )
    for filing in documents:
        for need in filing.needs:
            if need.priority != 0:
                continue
            document_record = inspected[need.document_key]
            if need.kind == "QUALIFICATION":
                statement = statuses.get(
                    (
                        need.document_key,
                        need.character_start,
                        need.character_end,
                        "REGION_STATEMENT",
                    )
                )
                if statement is not None and statement[1] is not None:
                    need_records.append(
                        _need_record(
                            document_record, need, "PROVIDED_SHARED", statement[1], need.detail
                        )
                    )
                else:
                    need_records.append(
                        _need_record(
                            document_record,
                            need,
                            "PENDING_ALLOWANCE",
                            None,
                            f"{need.detail}; the statement was not delivered",
                        )
                    )
                continue
            target = need.target
            if target is None:
                continue
            # A reference is satisfied only by its target's required evidence
            # as actually delivered: a touched region, a filing with any window
            # read, or a search hit in the same filing proves location, not
            # delivery. The state names which it is.
            status, handle, detail, state = "UNRESOLVED", None, need.detail, "UNRESOLVED"
            kept = next(
                (
                    value
                    for value in carried
                    if value.document_handle == document_record.document_handle
                    and value.detail.startswith(need.detail[:120])
                ),
                None,
            )
            if target.kind in {"REGION", "MATTER"}:
                required = required_reference_needs(target, filing.needs)
                state, count = reference_target_state(
                    required, _delivered_ranges(delivered, need.document_key)
                )
                status, handle, detail = _reference_outcome(
                    need.detail, state, count, len(required), delivered, need.document_key, required
                )
            elif target.kind == "FILING":
                wanted = _referenced_filing(target.label)
                held = None
                if wanted is not None:
                    for other in inspected.values():
                        period = other.structure.report_period.period_end
                        if (
                            other.entity_id == filing.issuer
                            and other.document_type.upper().startswith(wanted[0])
                            and period == wanted[1]
                        ):
                            held = other
                            break
                if held is None:
                    detail = f"{need.detail}; referenced filing not held in the document set"
                else:
                    required = required_reference_needs(
                        target, needs_by_document[held.document_id].needs
                    )
                    if not required:
                        detail = (
                            f"{need.detail}; held as {held.document_handle}, whose cited "
                            "section is not a region inventoried here"
                        )
                    else:
                        state, count = reference_target_state(
                            required, _delivered_ranges(delivered, held.document_id)
                        )
                        status, handle, detail = _reference_outcome(
                            f"{need.detail}; held as {held.document_handle}",
                            state,
                            count,
                            len(required),
                            delivered,
                            held.document_id,
                            required,
                        )
            elif target.kind == "UNLOCATED" and kept is not None:
                status, handle, state = "PROVIDED", kept.span_handle, "DELIVERED"
                detail = kept.detail
            elif (
                target.kind == "UNLOCATED"
                and searches < reference_searches
                and need.reference is not None
            ):
                searches += 1
                query = need.reference.target.strip('"“”')[:200]
                result = session.search(query=query, top_k=3)
                same = [
                    hit
                    for hit in result.hits
                    if hit.document_handle == document_record.document_handle
                    and search_hit_corresponds(need.reference.target, hit.preview)
                ]
                if same:
                    read = session.read_spans(span_handles=(same[0].span_handle,))
                    reference_spans.extend(read)
                    status, handle, state = "PROVIDED", read[0].span_handle, "DELIVERED"
                    detail = (
                        f"{need.detail}; located by one bounded search of the same filing, "
                        "the hit opening with the cited heading"
                    )
                elif result.hits:
                    detail = (
                        f"{need.detail}; one bounded search found no hit in the filing that "
                        "opens with the cited heading"
                    )
                else:
                    detail = f"{need.detail}; one bounded search found nothing in the filing"
            elif target.kind == "UNLOCATED":
                detail = f"{need.detail}; not located by structure, no search spent"
            need_records.append(
                _need_record(document_record, need, status, handle, detail, target_state=state)
            )
    read_by_document: dict[str, list[MatterWindowRecord]] = {}
    for record in window_records:
        read_by_document.setdefault(record.document_handle, []).append(record)
    document_records = []
    for document_id, document in inspected.items():
        needs = needs_by_document[document_id]
        matter_planned: dict[str, int] = {}
        matter_read: dict[str, int] = {}
        for need in needs.needs:
            if need.kind in {"LEAD", "CONTINUATION"} and need.matter_handle is not None:
                matter_planned[need.matter_handle] = matter_planned.get(need.matter_handle, 0) + 1
                if statuses.get(_need_key(need), ("PENDING_ALLOWANCE", None))[1] is not None:
                    matter_read[need.matter_handle] = matter_read.get(need.matter_handle, 0) + 1
        document_records.append(
            _litigation_document(
                document,
                needs.inventory,
                records=read_by_document.get(document.document_handle, []),
                matter_planned=matter_planned,
                matter_read=matter_read,
                planned_windows=sum(1 for n in needs.needs if n.priority > 0),
            )
        )
    read_records = [r for r in window_records if r.status == "READ"]
    pending_records = [r for r in window_records if r.status == "PENDING"]
    record = LitigationMatterRecord(
        rules_id=record_rules,
        allocation_rules_id=allocation,
        families=families,
        plan_hash=plan_hash,
        plan_offset=plan_offset,
        continued_from=continued_from,
        session_index=session_index,
        cumulative_read_windows=plan_offset + len(read_records),
        session_limit=session_limit,
        window_limit=window_limit,
        window_allowance=allowance,
        read_allowance=MAXIMUM_MATTER_READS,
        window_byte_ceiling=MATTER_WINDOW_BYTES,
        documents_inspected=len(inspected),
        documents_skipped_by_form=skipped,
        documents=tuple(document_records[:64]),
        windows=tuple(window_records[:1024]),
        span_handles=tuple(ordered),
        read_call_count=session.matter_read_count,
        planned_windows=len(window_records),
        read_windows=len(read_records),
        pending_windows=len(pending_records),
        prior_windows=len(prior_records),
        planned_source_bytes=sum(r.source_bytes for r in window_records),
        read_source_bytes=sum(r.source_bytes for r in read_records),
        pending_source_bytes=sum(r.source_bytes for r in pending_records),
        needs=tuple(need_records[:1024]),
    )
    trace = MatterEvidenceTrace(
        documents=documents,
        allocation=chosen,
        reference_spans=tuple(reference_spans),
        searches=searches,
    )
    return record, tuple(spans), trace


def _delivered_ranges(
    delivered: Mapping[str, Sequence[tuple[str, int, int, AllocatedWindow]]], document_key: str
) -> tuple[tuple[int, int], ...]:
    return tuple((start, end) for _h, start, end, _w in delivered.get(document_key, ()))


def _reference_outcome(
    detail: str,
    state: str,
    count: int,
    required: int,
    delivered: Mapping[str, Sequence[tuple[str, int, int, AllocatedWindow]]],
    document_key: str,
    needs: tuple[EvidenceNeed, ...],
) -> tuple[str, str | None, str]:
    """Report delivery status and target span for a referenced need.

    The need status, the span that delivers the target (when it is
    delivered) and the detail for a located target: DELIVERED is
    PROVIDED_SHARED on the first span holding one of its ranges;
    PARTIALLY_READ and LOCATED are unread evidence, named with its count.
    """
    if state == "DELIVERED":
        first = needs[0]
        handle = next(
            (
                h
                for h, start, end, _w in delivered.get(document_key, ())
                if first.character_start is not None
                and first.character_end is not None
                and start <= first.character_start
                and first.character_end <= end
            ),
            None,
        )
        return (
            "PROVIDED_SHARED",
            handle,
            f"{detail}; the target's {required} required range(s) delivered",
        )
    if state == "PARTIALLY_READ":
        return (
            "PENDING_ALLOWANCE",
            None,
            f"{detail}; target partially read: {count} of {required} required range(s) "
            "delivered, the rest unread",
        )
    if required:
        return (
            "PENDING_ALLOWANCE",
            None,
            f"{detail}; target located, none of its {required} required range(s) read",
        )
    return "PENDING_ALLOWANCE", None, f"{detail}; target located, its evidence not identified"


def _need_key(need: EvidenceNeed) -> tuple[str, int | None, int | None, str]:
    return need.document_key, need.character_start, need.character_end, need.kind


def _need_record(
    document: InspectedDocument,
    need: EvidenceNeed,
    status: str,
    span_handle: str | None,
    detail: str,
    *,
    target_state: str | None = None,
) -> EvidenceNeedRecord:
    # The current write boundary: every reference states what its target
    # got. A record without one is historical (`486cea54`), never new.
    if need.kind == "REFERENCE" and target_state is None:
        raise ValueError("alternative_evidence.evidence_need_target_state_required")
    return EvidenceNeedRecord(
        document_handle=document.document_handle,
        matter_handle=need.matter_handle,
        kind=need.kind,
        status=status,
        character_start=need.character_start,
        character_end=need.character_end,
        span_handle=span_handle,
        detail=detail[:240],
        target_state=target_state,
    )


class SharedInventory(NamedTuple):
    """Share one filing inventory across routing and delivery.

    One discovery of a session's admitted filings, shared by the routing
    and the delivery of the integrated selection: the inspected documents,
    every filing's needs and the documents no requested family reads.
    """

    inspected: dict[str, InspectedDocument]
    needs_by_document: dict[str, DocumentNeeds]
    skipped: int


def shared_inventory(session: AlternativeEvidenceRetrievalSession) -> SharedInventory:
    """The integrated selection's one discovery.

    The integrated selection's one discovery: every unit family of
    `MATTER_FAMILIES` under the unit segmentation rules
    (`MATTER_SEGMENTATION_RULES_ID`).
    """
    return SharedInventory(
        *_inventory_documents(session, rules=MATTER_SEGMENTATION_RULES_ID, families=MATTER_FAMILIES)
    )


def routed_documents(
    session: AlternativeEvidenceRetrievalSession, inventory: SharedInventory
) -> tuple[RoutedDocument, ...]:
    """The inspected filings with the times their sealed references state."""
    references = {
        str(reference.workspace_document_id): reference
        for reference in session.document_set.documents
    }
    routed: list[RoutedDocument] = []
    for key, document in inventory.inspected.items():
        reference = references.get(key)
        routed.append(
            RoutedDocument(
                document_key=key,
                document_handle=document.document_handle,
                entity_id=document.entity_id,
                document_type=document.document_type,
                text=document.text,
                structure=document.structure,
                accepted_at=None if reference is None else reference.accepted_at,
                published_at=None if reference is None else reference.published_at,
                report_period_end=None if reference is None else reference.report_period_end,
            )
        )
    return tuple(routed)


UNIT_INVENTORY_RULES: tuple[str, ...] = (
    MATTER_INVENTORY_RULES_ID,
    MATTER_SEGMENTATION_RULES_ID,
    EVENT_INVENTORY_RULES_ID,
    DISCLOSURE_UNIT_INVENTORY_RULES_ID,
    FINANCING_INVENTORY_RULES_ID,
    OPERATIONS_INVENTORY_RULES_ID,
)
"""The rules that read the units a filing comparison aligns: part of the
closure a sealed comparison is keyed on."""


def route_session(
    session: AlternativeEvidenceRetrievalSession,
    inventory: SharedInventory,
    typed: TypedDisclosureRecord | None,
    *,
    entity_ids: Sequence[str] = (),
    comparisons: ComparisonMemo | None = None,
    bindings: Mapping[str, str] | None = None,
) -> TopicRouting:
    """The routing of the session's document set from the shared inventory.

    The routing of the session's document set from the shared inventory:
    the tables each filing's canonical text left as placeholders, whether
    an original stands behind each, and the typed observations by issuer
    and family. No search, no read, nothing issued. With `comparisons`,
    each filing pair's temporal comparison is read back from the sealed
    record the chain's `bindings` (closure -> record hash, from the prior
    receipt) name for it, and sealed when computed; a first response
    binds nothing and computes every pair.
    """
    typed_by_issuer: dict[str, dict[str, int]] = {}
    for observation in () if typed is None else typed.observations:
        if observation.state == "EXTRACTED":
            counts = typed_by_issuer.setdefault(observation.entity_id, {})
            counts[observation.family] = counts.get(observation.family, 0) + 1
    return route_documents(
        documents=routed_documents(session, inventory),
        needs_by_document=inventory.needs_by_document,
        placeholders={key: session.table_placeholders(key) for key in inventory.inspected},
        originals={key: session.original_available(key) for key in inventory.inspected},
        typed_by_issuer=typed_by_issuer,
        entity_ids=entity_ids,
        comparisons=comparisons,
        unit_rules=UNIT_INVENTORY_RULES,
        bindings=bindings,
    )


@dataclass(frozen=True, slots=True)
class RoutedSelection:
    """What the integrated selection delivered, for the runtime to seal.

    What the integrated selection delivered, for the runtime to seal: the
    residual questions' records, the residual reads and their groups, the
    typed record, the matter record under the topic lanes, the routing
    record, and every span in packet order (residual, typed, matter, table
    views).
    """

    queries: tuple[EvidenceQueryRecord, ...]
    read_span_handles: tuple[str, ...]
    span_groups: tuple[EvidenceSpanGroup, ...]
    typed: TypedDisclosureRecord
    matters: LitigationMatterRecord
    routing: TopicRoutingRecord
    spans: tuple[AlternativeEvidenceResolvedSpan, ...]
    trace: MatterEvidenceTrace


def select_routed_evidence(
    *,
    session: AlternativeEvidenceRetrievalSession,
    entity_ids: Sequence[str] = (),
    program: tuple[EvidenceQuery, ...] = EVIDENCE_QUERY_PROGRAM,
    pair_budget: int = RESIDUAL_RERANK_PAIR_BUDGET,
    collapse_repeats: bool = True,
    table_share: int = TABLE_VIEWS_PER_SESSION,
    comparisons: ComparisonMemo | None = None,
) -> RoutedSelection:
    """The integrated selection of one unit.

    The integrated selection of one unit: shared discovery once (the three
    unit families' inventories), the typed families read under their own
    budget, the eight topics routed by source shape, the units dealt in
    issuer-topic lanes latest filing first with exact repeats collapsed
    (`TOPIC_LANES_ALLOCATION_ID`) under the matter allowance less a table
    share, up to `TABLE_VIEWS_PER_SESSION` table views rendered from the
    retained originals inside that share, and the question bank run only
    over each topic's residual scope -- the routed ranges no inventory
    addressed -- under `pair_budget` cross-encoder pairs, each question's
    hits read as the program's are. A question with no residual scope is
    skipped for scope; one the budget cannot admit is skipped for budget;
    both are recorded, and a selection whose questions were all skipped is
    a legitimate zero-query selection with its routing saying why.

    `collapse_repeats` and `table_share` are the measurement's knobs, never
    the wire's: the product runs with both (the exact repeats collapsed,
    up to `TABLE_VIEWS_PER_SESSION` views); an arm that isolates the
    routing and the scoped search from the temporal collapse and the
    table delivery runs without them, and the receipt records what ran.
    """
    inventory = shared_inventory(session)
    typed, typed_spans = select_typed_disclosures(session=session)
    routing = route_session(
        session, inventory, typed, entity_ids=entity_ids, comparisons=comparisons
    )
    if not collapse_repeats:
        routing = replace(routing, repeats={})
    # Table views: the first page of each routed table with an original, in
    # routing order, inside the reserved share of the matter allowance --
    # chosen before the units are dealt, so the share is exactly what will
    # be delivered and the units keep the rest. A table the boundary
    # refuses by name (no heading row; a correspondence the original does
    # not prove) is a representation gap, recorded, and the share moves to
    # the next table.
    table_spans: list[AlternativeEvidenceResolvedSpan] = []
    delivered_tables: list[tuple[str, int]] = []
    views: list[TableNeed] = []
    refusals: list[TableViewRefusalRecord] = []
    refused_by_document: dict[str, dict[int, str]] = {}

    renderable = table_view_renderable(session, routing, inventory, refusals, refused_by_document)
    if table_share:
        views = topic_fair_views(routing.table_needs, share=table_share, renderable=renderable)
    unrenderable = {
        (key, ordinal) for key, codes in refused_by_document.items() for ordinal in codes
    }
    matters, matter_spans, trace = select_matter_evidence(
        session=session,
        inventory=inventory,
        routing=routing,
        allowance_reserve=len(views),
    )
    by_document: dict[str, list[int]] = {}
    for position, need in enumerate(views):
        by_document.setdefault(need.document_key, []).append(position)
    handles: dict[int, str] = {}
    series = f"X{matters.session_index:02d}"
    for document_key, positions in by_document.items():
        issued = session.issue_table_views(
            document_key,
            tuple((views[p].placeholder, 1) for p in positions),
            series=series,
        )
        for position, handle in zip(positions, issued, strict=True):
            handles[position] = handle
    ordered = [handles[position] for position in range(len(views))]
    for start in range(0, len(ordered), MATTER_WINDOWS_PER_READ):
        table_spans.extend(
            session.read_table_views(
                span_handles=tuple(ordered[start : start + MATTER_WINDOWS_PER_READ])
            )
        )
    delivered_tables = [(need.document_key, need.placeholder.ordinal) for need in views]
    pages_by_handle = {span.span_handle: span.table_view for span in table_spans}
    # The residual search: each question over its topic's residual scope.
    records: list[EvidenceQueryRecord] = []
    best: dict[tuple[str, int, str, int, int], CellCandidate] = {}
    scopes: dict[EvidenceTopic, tuple[tuple[str, int, int], ...]] = {}
    scope_windows: dict[tuple[str, EvidenceTopic], int] = {}
    routed_chunks: set[tuple[str, int]] = set()
    for cell in routing.cells:
        ranges: dict[str, list[tuple[int, int]]] = {}
        for document_key, start, end in cell.residual_ranges:
            ranges.setdefault(document_key, []).append((start, end))
        entries = session.scope_for(ranges) if ranges else ()
        scope_windows[(cell.entity_id, cell.topic)] = sum(
            last - first + 1 for _key, first, last in entries
        )
        for key, first, last in entries:
            routed_chunks.update((key, ordinal) for ordinal in range(first, last + 1))
        scopes[cell.topic] = tuple(sorted({*scopes.get(cell.topic, ()), *entries}))
    run = skipped_scope = skipped_budget = 0
    query_topics: dict[str, EvidenceTopic] = {}
    found_by: dict[tuple[str, int, str, int, int], list[tuple[int, int, str]]] = {}
    """Every question that returned a passage, as (issuer-local rank, global
    rank, query id): the provenance the sealed pending plan keeps."""
    for order, query in enumerate(topic_fair_order(program)):
        query_topics[query.query_id] = query.topic
        scope = scopes.get(query.topic, ())
        if not scope:
            skipped_scope += 1
            records.append(
                EvidenceQueryRecord(
                    query_id=query.query_id,
                    topic=query.topic,
                    text=query.text,
                    hit_span_handles=(),
                    kind=query.kind,
                    disposition="SKIPPED_NO_SCOPE",
                    scope_windows=0,
                )
            )
            continue
        if session.reranked_pairs >= pair_budget:
            skipped_budget += 1
            records.append(
                EvidenceQueryRecord(
                    query_id=query.query_id,
                    topic=query.topic,
                    text=query.text,
                    hit_span_handles=(),
                    kind=query.kind,
                    disposition="SKIPPED_BUDGET",
                    scope_windows=sum(last - first + 1 for _k, first, last in scope),
                )
            )
            continue
        result = session.search(
            query=query.text,
            top_k=QUERY_TOP_K,
            reranker_depth=RERANKER_DEPTH_PER_DOCUMENT,
            scope=scope,
        )
        run += 1
        records.append(
            EvidenceQueryRecord(
                query_id=query.query_id,
                topic=query.topic,
                text=query.text,
                hit_span_handles=tuple(hit.span_handle for hit in result.hits),
                kind=query.kind,
                scope_windows=sum(last - first + 1 for _k, first, last in scope),
                reranked_pairs=result.reranked_pairs,
            )
        )
        local_ranks: dict[str, int] = {}
        for hit in result.hits:
            identity = session.span_identity(hit.span_handle)
            local_ranks[hit.entity_id] = local_ranks.get(hit.entity_id, 0) + 1
            found_by.setdefault(identity, []).append(
                (local_ranks[hit.entity_id], hit.rank, query.query_id)
            )
            candidate: CellCandidate = (
                local_ranks[hit.entity_id],
                order,
                hit.rank,
                hit.span_handle,
                hit.entity_id,
                hit.document_handle,
                session.span_content_hash(hit.span_handle),
                query.query_id,
            )
            held = best.get(identity)
            if held is None or candidate < held:
                best[identity] = candidate
    ranked = sorted(best.values())
    # A candidate whose range a unit window, a typed statement or a table
    # page already delivered holds is covered: not read again, its
    # questions carried to the covering span.
    delivered = delivered_ranges((*typed_spans, *matter_spans, *table_spans))

    def cover_of(item: CellCandidate) -> tuple[str, ...]:
        document_key, start, end = session.span_range(item[3])
        return covering_span_handles(
            delivered, inventory.inspected[document_key].document_handle, start, end
        )

    selected, groups, pending_by_cell = _allocate_by_cell(ranked, query_topics, cover_of)
    # The frontier: every cell's candidates the batch did not read, in the
    # cell's order, with the source range and the passage hash a later
    # session must reproduce, every question that returned them, and the
    # covering span of those another channel delivered.
    pending_records: list[PendingCandidateRecord] = []
    candidates_beyond = 0
    for (issuer, topic), items in sorted(
        pending_by_cell.items(), key=lambda pair: (pair[0][0], TOPICS.index(pair[0][1]))
    ):
        for position, (item, cover) in enumerate(items, start=1):
            if len(pending_records) >= CANDIDATE_FRONTIER_LIMIT:
                candidates_beyond += 1
                continue
            document_key, start, end = session.span_range(item[3])
            provenance = sorted(found_by[session.span_identity(item[3])])
            pending_records.append(
                PendingCandidateRecord(
                    entity_id=issuer,
                    topic=topic,
                    document_handle=inventory.inspected[document_key].document_handle,
                    character_start=start,
                    character_end=end,
                    passage_hash=passage_hash(inventory.inspected[document_key].text, start, end),
                    found_by=tuple(dict.fromkeys(q for _l, _g, q in provenance))[:8],
                    best_global_rank=provenance[0][1],
                    best_local_rank=provenance[0][0],
                    order=position,
                    state="COVERED" if cover else "PENDING",
                    session_index=1 if cover else None,
                    covered_by=cover[0] if cover else None,
                    covered_with=cover[1:],
                )
            )
    residual_spans: list[AlternativeEvidenceResolvedSpan] = []
    for start in range(0, len(selected), SPANS_PER_READ):
        residual_spans.extend(
            session.read_spans(span_handles=selected[start : start + SPANS_PER_READ])
        )
    residual_by_cell: dict[tuple[str, EvidenceTopic], int] = {}
    query_of = {item[3]: item[7] for item in ranked}
    for span in residual_spans:
        topic = query_topics[query_of[span.span_handle]]
        residual_by_cell[(span.entity_id, topic)] = (
            residual_by_cell.get((span.entity_id, topic), 0) + 1
        )
    # What no topic's scope reached, among the proved windows past the cover.
    everything = session.scope_for(
        {
            key: [(document.structure.cover_end, document.structure.length)]
            for key, document in inventory.inspected.items()
            if document.structure.length > document.structure.cover_end
        }
    )
    all_chunks = {
        (key, ordinal) for key, first, last in everything for ordinal in range(first, last + 1)
    }
    inventoried_ranges = {
        key: [(r.body_start, r.body_end) for r in needs.inventory.regions]
        for key, needs in inventory.needs_by_document.items()
    }
    inventoried_chunks = {
        (key, ordinal)
        for key, first, last in session.scope_for(inventoried_ranges)
        for ordinal in range(first, last + 1)
    }
    unrouted = len(all_chunks - routed_chunks - inventoried_chunks)
    record = routing_record(
        routing,
        matters=matters,
        inspected=inventory.inspected,
        needs_by_document=inventory.needs_by_document,
        table_view_span_handles=tuple(ordered),
        table_views=tuple(
            _table_view_record(
                handles[position],
                need,
                inventory.inspected[need.document_key].document_handle,
                pages_by_handle.get(handles[position]),
                session_index=1,
            )
            for position, need in enumerate(views)
        ),
        delivered_tables=delivered_tables,
        candidates=tuple(pending_records),
        candidates_beyond=candidates_beyond,
        unrenderable=unrenderable,
        refusals=tuple(refusals),
        scope_windows=scope_windows,
        residual_by_cell=residual_by_cell,
        pair_budget=pair_budget,
        reranked_pairs=session.reranked_pairs,
        questions=(run, skipped_scope, skipped_budget),
        unrouted_windows=unrouted,
        residual_selection_rules_id=RESIDUAL_SELECTION_RULES_ID,
    )
    return RoutedSelection(
        queries=tuple(records),
        read_span_handles=tuple(span.span_handle for span in residual_spans),
        span_groups=groups,
        typed=typed,
        matters=matters,
        routing=record,
        spans=(*residual_spans, *typed_spans, *matter_spans, *table_spans),
        trace=trace,
    )


def routing_record(
    routing: TopicRouting,
    *,
    matters: LitigationMatterRecord,
    inspected: Mapping[str, InspectedDocument],
    needs_by_document: Mapping[str, DocumentNeeds],
    table_view_span_handles: tuple[str, ...],
    table_views: tuple[TableViewRecord, ...],
    delivered_tables: Sequence[tuple[str, int]],
    scope_windows: Mapping[tuple[str, EvidenceTopic], int],
    unrenderable: Collection[tuple[str, int]] = (),
    refusals: tuple[TableViewRefusalRecord, ...] = (),
    residual_by_cell: Mapping[tuple[str, EvidenceTopic], int],
    pair_budget: int,
    reranked_pairs: int,
    questions: tuple[int, int, int],
    unrouted_windows: int,
    candidates: tuple[PendingCandidateRecord, ...] = (),
    candidates_beyond: int = 0,
    candidate_span_handles: tuple[str, ...] = (),
    residual_selection_rules_id: str = RESIDUAL_SELECTION_RULES_ID,
) -> TopicRoutingRecord:
    """The routing as a sealed record.

    The routing as a sealed record: every cell with what the delivery
    provided of its unit needs (from the matter record's need statuses),
    its tables and its residual hits; the correspondences; the budget.
    """
    provided: set[tuple[str, int | None, int | None, str]] = set()
    for record in matters.needs:
        if record.status in {"PROVIDED", "PROVIDED_SHARED"}:
            provided.add(
                (record.document_handle, record.character_start, record.character_end, record.kind)
            )
    delivered_by_cell: dict[tuple[str, EvidenceTopic], int] = {}
    for key, needs in needs_by_document.items():
        handle = inspected[key].document_handle
        for need in needs.needs:
            if need.priority == 0 or need.window is None:
                continue
            if (handle, need.character_start, need.character_end, need.kind) not in provided:
                continue
            for topic in routing.topics_by_need.get(id(need), ()):
                cell_key = (needs.issuer, topic)
                delivered_by_cell[cell_key] = delivered_by_cell.get(cell_key, 0) + 1
    tables_by_cell: dict[tuple[str, EvidenceTopic], int] = {}
    unrenderable_by_cell: dict[tuple[str, EvidenceTopic], int] = {}
    delivered_set = set(delivered_tables)
    unrenderable_set = set(unrenderable)
    for table in routing.table_needs:
        table_key = (table.document_key, table.placeholder.ordinal)
        for topic in table.topics:
            if table_key in delivered_set:
                tables_by_cell[(table.entity_id, topic)] = (
                    tables_by_cell.get((table.entity_id, topic), 0) + 1
                )
            elif table_key in unrenderable_set:
                unrenderable_by_cell[(table.entity_id, topic)] = (
                    unrenderable_by_cell.get((table.entity_id, topic), 0) + 1
                )
    cells = tuple(
        IssuerTopicCellRecord(
            entity_id=cell.entity_id,
            topic=cell.topic,
            forms_held=cell.forms_held[:8],
            regions=cell.regions,
            regions_by_basis=cell.regions_by_basis[:8],
            unit_needs=cell.unit_needs,
            unit_needs_delivered=delivered_by_cell.get((cell.entity_id, cell.topic), 0),
            typed_observations=cell.typed_observations,
            tables=cell.tables,
            tables_delivered=tables_by_cell.get((cell.entity_id, cell.topic), 0),
            tables_without_original=cell.tables_without_original,
            tables_unrenderable=unrenderable_by_cell.get((cell.entity_id, cell.topic), 0),
            residual=cell.residual,
            residual_windows=scope_windows.get((cell.entity_id, cell.topic), 0),
            residual_hits=residual_by_cell.get((cell.entity_id, cell.topic), 0),
            candidates_pending=sum(
                1
                for c in candidates
                if c.entity_id == cell.entity_id and c.topic == cell.topic and c.state == "PENDING"
            ),
            candidates_read=sum(
                1
                for c in candidates
                if c.entity_id == cell.entity_id and c.topic == cell.topic and c.state == "READ"
            ),
            candidates_covered=sum(
                1
                for c in candidates
                if c.entity_id == cell.entity_id and c.topic == cell.topic and c.state == "COVERED"
            ),
            gaps=cell.gaps[:8],
        )
        for cell in routing.cells
    )
    correspondences = tuple(
        UnitCorrespondenceRecord(
            entity_id=value.entity_id,
            family=value.family,
            later_document_handle=inspected[value.later_document_key].document_handle,
            later_unit_handle=value.later_handle,
            earlier_document_handle=(
                None
                if value.earlier_document_key is None
                else inspected[value.earlier_document_key].document_handle
            ),
            earlier_unit_handle=value.earlier_handle,
            state=value.state,
            ratio=None if value.ratio is None else round(value.ratio, 4),
            changed_characters=value.changed_characters,
            basis=value.basis[:200],
        )
        for value in routing.correspondences[:512]
    )
    return TopicRoutingRecord(
        rules_id=routing.rules_id,
        comparison_rules_id=routing.comparison_rules_id,
        allocation_rules_id=TOPIC_LANES_ALLOCATION_ID,
        cells=cells[:64],
        correspondences=correspondences,
        comparisons=tuple(
            ComparisonBindingRecord(closure_hash=closure, record_hash=record)
            for closure, record in routing.comparison_bindings[:256]
        ),
        exact_repeats_collapsed=len(routing.repeats),
        table_view_span_handles=table_view_span_handles,
        table_views=table_views,
        table_view_refusals=refusals,
        table_views_pending=max(
            0, len(routing.table_needs) - len(delivered_tables) - len(unrenderable_set)
        ),
        residual_rerank_pair_budget=pair_budget,
        residual_reranked_pairs=reranked_pairs,
        questions_run=questions[0],
        questions_skipped_no_scope=questions[1],
        questions_skipped_budget=questions[2],
        unrouted_windows=unrouted_windows,
        residual_selection_rules_id=residual_selection_rules_id,
        candidate_frontier=CandidateFrontier.from_records(candidates) if candidates else None,
        candidates_beyond_plan=candidates_beyond,
        candidate_span_handles=candidate_span_handles,
    )


def _table_view_record(
    handle: str,
    need: TableNeed,
    document_handle: str,
    page: TableViewBinding | None,
    *,
    session_index: int,
) -> TableViewRecord:
    """One delivered page as the routing record seals it.

    One delivered page as the routing record seals it: the table, its
    filing, topics and region, and the rows the page holds -- so a later
    session resumes the table at the next row, never at a count.
    """
    return TableViewRecord(
        span_handle=handle,
        document_handle=document_handle,
        entity_id=need.entity_id,
        topics=need.topics,
        region_heading=need.region_heading[:160],
        table_ordinal=need.placeholder.ordinal,
        rows_total=need.placeholder.rows_total,
        rows_from=0 if page is None else page.rows_from,
        rows_to=0 if page is None else page.rows_to,
        remaining_rows=0 if page is None else page.remaining_rows,
        rows_clipped=0 if page is None else page.rows_clipped,
        session_index=session_index,
    )


@dataclass(frozen=True, slots=True)
class ContinuedReads:
    """What a continuation session read of the sealed pending plan without a search.

    What a continuation session read of the sealed pending plan without
    a search: the candidates read (records marked READ with this session's
    handles), the candidate spans, the table pages read (resumed pages and
    newly dealt tables) with their records and spans, the refusals the
    share met, and the table identities delivered.
    """

    pending_candidates: tuple[PendingCandidateRecord, ...]
    candidate_span_handles: tuple[str, ...]
    candidate_spans: tuple[AlternativeEvidenceResolvedSpan, ...]
    table_view_span_handles: tuple[str, ...]
    table_views: tuple[TableViewRecord, ...]
    table_spans: tuple[AlternativeEvidenceResolvedSpan, ...]
    refusals: tuple[TableViewRefusalRecord, ...]
    delivered_tables: tuple[tuple[str, int], ...]
    unrenderable: tuple[tuple[str, int], ...]


def delivered_table_identities(
    views: Sequence[TableViewRecord], inspected: Mapping[str, InspectedDocument]
) -> tuple[tuple[str, int], ...]:
    """List sealed table identities delivered by a chain.

    The tables a chain delivered, by the sealed identity of each page
    (its filing and table ordinal) -- never a prefix of the routing's table
    needs, which after the topic-fair share are not the tables dealt.
    """
    keys = {document.document_handle: key for key, document in inspected.items()}
    return tuple(
        dict.fromkeys(
            (keys[view.document_handle], view.table_ordinal)
            for view in views
            if view.document_handle in keys
        )
    )


def plan_table_pages(
    *,
    prior: TopicRoutingRecord,
    routing: TopicRouting,
    inspected: Mapping[str, InspectedDocument],
    renderable: Callable[[TableNeed], bool],
    share: int = TABLE_VIEWS_PER_SESSION,
) -> tuple[tuple[TableNeed, int], ...]:
    """The table pages a continuation session reads inside its share.

    The table pages a continuation session reads inside its share: first
    the next page of every table the chain delivered in part and can
    resume (the sealed pages' progress, `TableProgress.next_row`: the row
    after the rows proved delivered contiguously from row 1), in the order
    the tables first appear in the sealed pages; then, in the share left,
    the tables the chain has not dealt, topic-fair as the first session
    dealt them, the delivered and the refused excluded by identity. A table
    whose progress is unknown (a page sealed before pages carried rows) or
    whose remainder is a clipped row no page can hold is not resumed; the
    continuation scope names it as a remainder.
    """
    keys = {document.document_handle: key for key, document in inspected.items()}
    needs_by_identity = {
        (need.document_key, need.placeholder.ordinal): need for need in routing.table_needs
    }
    pages: list[tuple[TableNeed, int]] = []
    progress = prior.table_progress()
    seen: set[tuple[str, int]] = set()
    for view in prior.table_views:
        key = keys.get(view.document_handle)
        identity = (view.document_handle, view.table_ordinal)
        if key is None or identity in seen:
            continue
        seen.add(identity)
        need = needs_by_identity.get((key, view.table_ordinal))
        next_row = progress[identity].next_row
        if need is None or next_row is None or progress[identity].state != "PARTIAL":
            continue
        if len(pages) >= share:
            break
        pages.append((need, next_row))
    delivered = set(delivered_table_identities(prior.table_views, inspected))
    refused = {
        (keys[refusal.document_handle], refusal.table_ordinal)
        for refusal in prior.table_view_refusals
        if refusal.document_handle in keys
    }
    remaining = [
        need
        for need in routing.table_needs
        if (need.document_key, need.placeholder.ordinal) not in delivered
        and (need.document_key, need.placeholder.ordinal) not in refused
    ]
    for need in topic_fair_views(
        remaining, share=max(0, share - len(pages)), renderable=renderable
    ):
        pages.append((need, 1))
    return tuple(pages)


def continue_routed_reads(
    *,
    session: AlternativeEvidenceRetrievalSession,
    prior: TopicRoutingRecord,
    inventory: SharedInventory,
    routing: TopicRouting,
    session_index: int,
    pages: tuple[tuple[TableNeed, int], ...],
    refusals: tuple[TableViewRefusalRecord, ...],
    unrenderable: Collection[tuple[str, int]],
    candidate_reads_per_issuer: int = CANDIDATE_READS_PER_ISSUER,
    candidate_read_limit: int = MAXIMUM_ISSUED_CANDIDATE_WINDOWS,
    delivered: DeliveredRanges | None = None,
) -> ContinuedReads:
    """Read the sealed pending plan in a continuation session.

    Read the sealed pending plan in a continuation session: the pages
    planned (`plan_table_pages`) under the matter allowance's table share,
    and the pending residual candidates -- per issuer, the cells in the
    topics' declared order, round by round as the first session dealt its
    batch, up to `candidate_reads_per_issuer` an issuer and
    `candidate_read_limit` a session -- each issued at its sealed range
    (series `C<session>`), proved by the sealed passage hash before it is
    read, and read through the program's own reader and budget. No query
    is embedded, no pair scored, no passage embedded: a continuation is a
    read. A candidate whose range no longer holds the sealed passage is a
    refusal by name, never a read. A pending candidate whose range the
    chain has delivered since through another channel (`delivered`: the
    prior spans and this session's windows and pages) is covered by that
    span and not read.
    """
    keys = {document.document_handle: key for key, document in inspected_keys(inventory)}
    records = list(prior.candidates())
    if delivered is not None:
        for index, record in enumerate(records):
            if record.state != "PENDING":
                continue
            cover = covering_span_handles(
                delivered, record.document_handle, record.character_start, record.character_end
            )
            if cover:
                records[index] = record.model_copy(
                    update={
                        "state": "COVERED",
                        "covered_by": cover[0],
                        "covered_with": cover[1:],
                        "session_index": session_index,
                    }
                )
    # ---- the candidates.
    by_issuer: dict[str, dict[EvidenceTopic, list[int]]] = {}
    for index, record in enumerate(records):
        if record.state != "PENDING":
            continue
        by_issuer.setdefault(record.entity_id, {}).setdefault(record.topic, []).append(index)
    chosen: list[int] = []
    for issuer in sorted(by_issuer):
        cells = by_issuer[issuer]
        positions: dict[EvidenceTopic, int] = dict.fromkeys(cells, 0)
        taken = 0
        while taken < candidate_reads_per_issuer and len(chosen) < candidate_read_limit:
            progressed = False
            for topic in TOPICS:
                indexes = cells.get(topic)
                if indexes is None:
                    continue
                if taken >= candidate_reads_per_issuer or len(chosen) >= candidate_read_limit:
                    break
                if positions[topic] >= len(indexes):
                    continue
                index = indexes[positions[topic]]
                chosen.append(index)
                positions[topic] += 1
                taken += 1
                progressed = True
            if not progressed:
                break
    series = f"C{session_index:02d}"
    handle_of: dict[int, str] = {}
    by_document: dict[str, list[int]] = {}
    for index in chosen:
        by_document.setdefault(records[index].document_handle, []).append(index)
    for document_handle, indexes in by_document.items():
        key = keys.get(document_handle)
        if key is None:
            raise ValueError("alternative_evidence.pending_candidate_source_missing")
        issued = session.issue_candidate_windows(
            key,
            tuple((records[i].character_start, records[i].character_end) for i in indexes),
            series=series,
        )
        for index, handle in zip(indexes, issued, strict=True):
            if session.issued_passage_hash(handle) != records[index].passage_hash:
                raise ValueError("alternative_evidence.pending_candidate_source_moved")
            handle_of[index] = handle
    ordered = [handle_of[index] for index in chosen]
    candidate_spans: list[AlternativeEvidenceResolvedSpan] = []
    for start in range(0, len(ordered), SPANS_PER_READ):
        candidate_spans.extend(
            session.read_spans(span_handles=tuple(ordered[start : start + SPANS_PER_READ]))
        )
    read = {span.span_handle for span in candidate_spans}
    for index in chosen:
        if handle_of[index] not in read:
            raise ValueError("alternative_evidence.pending_candidate_unread")
        records[index] = records[index].model_copy(
            update={
                "state": "READ",
                "span_handle": handle_of[index],
                "session_index": session_index,
            }
        )
    # ---- the table pages.
    page_handles: dict[int, str] = {}
    by_key: dict[str, list[int]] = {}
    for position, (need, _row) in enumerate(pages):
        by_key.setdefault(need.document_key, []).append(position)
    view_series = f"X{session_index:02d}"
    for document_key, positions_ in by_key.items():
        issued = session.issue_table_views(
            document_key,
            tuple((pages[p][0].placeholder, pages[p][1]) for p in positions_),
            series=view_series,
        )
        for position, handle in zip(positions_, issued, strict=True):
            page_handles[position] = handle
    page_order = [page_handles[position] for position in range(len(pages))]
    table_spans: list[AlternativeEvidenceResolvedSpan] = []
    for start in range(0, len(page_order), MATTER_WINDOWS_PER_READ):
        table_spans.extend(
            session.read_table_views(
                span_handles=tuple(page_order[start : start + MATTER_WINDOWS_PER_READ])
            )
        )
    pages_by_handle = {span.span_handle: span.table_view for span in table_spans}
    view_records = tuple(
        _table_view_record(
            page_handles[position],
            need,
            inventory.inspected[need.document_key].document_handle,
            pages_by_handle.get(page_handles[position]),
            session_index=session_index,
        )
        for position, (need, _row) in enumerate(pages)
    )
    return ContinuedReads(
        pending_candidates=tuple(records),
        candidate_span_handles=(*prior.candidate_span_handles, *ordered),
        candidate_spans=tuple(candidate_spans),
        table_view_span_handles=(*prior.table_view_span_handles, *page_order),
        table_views=(*prior.table_views, *view_records),
        table_spans=tuple(table_spans),
        refusals=refusals,
        delivered_tables=delivered_table_identities(
            (*prior.table_views, *view_records), inventory.inspected
        ),
        unrenderable=tuple(unrenderable),
    )


def inspected_keys(inventory: SharedInventory) -> tuple[tuple[str, InspectedDocument], ...]:
    return tuple(inventory.inspected.items())


def table_view_renderable(
    session: AlternativeEvidenceRetrievalSession,
    routing: TopicRouting,
    inventory: SharedInventory,
    refusals: list[TableViewRefusalRecord],
    refused_by_document: dict[str, dict[int, str]],
) -> Callable[[TableNeed], bool]:
    """Check once whether a routed table can render.

    The boundary's answer whether a routed table renders, asked once per
    document (the original verified and parsed once, no handle, no read);
    a refusal is recorded by name as the representation gap it is. One
    closure for the first session's share and a continuation's pages.
    """

    def renderable(need: TableNeed) -> bool:
        known = refused_by_document.get(need.document_key)
        if known is None:
            candidates = tuple(
                n.placeholder for n in routing.table_needs if n.document_key == need.document_key
            )
            codes = session.table_view_refusals(need.document_key, candidates)
            known = {
                placeholder.ordinal: code
                for placeholder, code in zip(candidates, codes, strict=True)
                if code is not None
            }
            refused_by_document[need.document_key] = known
        code = known.get(need.placeholder.ordinal)
        if code is None:
            return True
        if len(refusals) < 64 and not any(
            r.table_ordinal == need.placeholder.ordinal
            and r.document_handle == inventory.inspected[need.document_key].document_handle
            for r in refusals
        ):
            refusals.append(
                TableViewRefusalRecord(
                    document_handle=inventory.inspected[need.document_key].document_handle,
                    entity_id=need.entity_id,
                    table_ordinal=need.placeholder.ordinal,
                    code=code.split(":")[0][:80],
                )
            )
        return False

    return renderable


def matter_views(
    receipt: AlternativeEvidenceRetrievalAccessReceipt,
) -> dict[str, tuple[LitigationMatterDocument, ProvisionalMatterRecord | None, MatterWindowRecord]]:
    """Resolve the document, matter, and window behind each span.

    The document, provisional matter (None for unassigned text) and window
    behind each matter span handle.
    """
    matters = receipt.litigation_matters
    if matters is None:
        return {}
    documents = {value.document_handle: value for value in matters.documents}
    views: dict[
        str, tuple[LitigationMatterDocument, ProvisionalMatterRecord | None, MatterWindowRecord]
    ] = {}
    for window in matters.windows:
        if window.span_handle is None:
            continue
        document = documents[window.document_handle]
        matter = next((m for m in document.matters if m.handle == window.matter_handle), None)
        views[window.span_handle] = (document, matter, window)
    return views


def matter_view(
    document: LitigationMatterDocument,
    matter: ProvisionalMatterRecord | None,
    window: MatterWindowRecord,
) -> dict[str, object]:
    """A matter window as a reader sees it beside its excerpt.

    A matter window as a reader sees it beside its excerpt: which
    provisional matter it belongs to, which part of that matter's text it
    is, and how much of the matter and the filing remains unread. Navigation
    and accounting, never a finding.
    """
    return {
        "matter_handle": window.matter_handle,
        "family": None if matter is None else matter.family,
        "title": None if matter is None else matter.title,
        "basis": "unassigned text" if matter is None else matter.basis,
        "named_proceeding": None if matter is None else matter.named,
        "region_heading": None if matter is None else matter.region_heading,
        "aliases": () if matter is None else matter.aliases,
        "case_numbers": () if matter is None else matter.case_numbers,
        "part": window.part,
        "part_count": window.part_count,
        "character_range": [window.character_start, window.character_end],
        "delivered_whole": window.delivered_whole,
        **(
            {}
            if not window.attributions
            else {
                "carries": [
                    {
                        "matter_handle": value.matter_handle,
                        "role": value.role,
                        "scope": value.scope,
                        "basis": value.basis,
                        "region_heading": value.region_heading,
                        "character_range": [value.character_start, value.character_end],
                    }
                    for value in window.attributions
                ],
                "carries_rule": MATTER_CARRIES_RULE,
            }
        ),
        "matter_windows_read": None if matter is None else matter.read_windows,
        "matter_windows_planned": None if matter is None else matter.planned_windows,
        "document_inspection": document.inspection,
        "document_pending_windows": document.pending_windows,
        "rule": MATTER_WINDOW_RULE,
    }


MATTER_WINDOW_RULE = (
    "A window is the source's own words for one part of a provisionally inventoried "
    "matter; read every part before treating a matter's status as stated, and treat "
    "pending windows as unread."
)
MATTER_CARRIES_RULE = (
    "The excerpt is transport; each entry is what it carries at an exact "
    "range: a matter's opening or a later part of it under that matter's "
    "handle, or a region-level statement whose scope is REGION (the source "
    "says it applies to the region's matters) or UNRESOLVED (the source "
    "does not say). Two matters in one excerpt are two entries; read each "
    "at its own range and attribute nothing across them."
)


def matter_window_view(
    matter: ProvisionalMatterRecord | None, window: MatterWindowRecord
) -> dict[str, object]:
    """A matter window beside its excerpt in a packet part.

    A matter window beside its excerpt in a packet part: which matter it
    belongs to (its family and title inline; the matter's descriptors once
    in the part's `litigation_matters.matters_in_part` under the handle),
    which part of that matter's text it is, and what the excerpt carries at
    exact ranges (the rules once in the part's matter block). Measured on a
    real unit, the whole view on every window was 17% of the bytes a
    consumer received, most of it one matter's descriptors, two rule
    sentences and one filing's counts restated on every window.
    """
    return {
        "matter_handle": window.matter_handle,
        "family": None if matter is None else matter.family,
        "title": None if matter is None else matter.title,
        "part": window.part,
        "part_count": window.part_count,
        "character_range": [window.character_start, window.character_end],
        "delivered_whole": window.delivered_whole,
        **(
            {}
            if not window.attributions
            else {
                "carries": [
                    {
                        "matter_handle": value.matter_handle,
                        "role": value.role,
                        "scope": value.scope,
                        "basis": value.basis,
                        "region_heading": value.region_heading,
                        "character_range": [value.character_start, value.character_end],
                    }
                    for value in window.attributions
                ]
            }
        ),
    }


def matters_in_part(
    views: Mapping[
        str, tuple[LitigationMatterDocument, ProvisionalMatterRecord | None, MatterWindowRecord]
    ],
    span_handles: Sequence[str],
) -> dict[str, object]:
    """Each matter one of the part's window spans belongs to, stated once under its handle.

    Each matter one of the part's window spans belongs to, stated once
    under its handle: the descriptors every window of it used to restate,
    and the filing's inspection state. Unassigned text is its own entry
    under the window's handle.
    """
    entries: dict[str, object] = {}
    for span_handle in span_handles:
        held = views.get(span_handle)
        if held is None:
            continue
        document, matter, window = held
        if window.matter_handle in entries:
            continue
        entries[window.matter_handle] = {
            "document_handle": document.document_handle,
            "family": None if matter is None else matter.family,
            "title": None if matter is None else matter.title,
            "basis": "unassigned text" if matter is None else matter.basis,
            "named_proceeding": None if matter is None else matter.named,
            "region_heading": None if matter is None else matter.region_heading,
            "aliases": () if matter is None else matter.aliases,
            "case_numbers": () if matter is None else matter.case_numbers,
            "matter_windows_read": None if matter is None else matter.read_windows,
            "matter_windows_planned": None if matter is None else matter.planned_windows,
            "document_inspection": document.inspection,
            "document_pending_windows": document.pending_windows,
        }
    return entries


def litigation_matter_summary(
    receipt: AlternativeEvidenceRetrievalAccessReceipt,
) -> dict[str, object] | None:
    """Summarize matter inventory across delivery parts.

    The packet-level accounting of the matter inventory, compact enough
    to travel with every delivery part: the allowance and the chain of
    reading sessions, what was planned, read, left pending or read earlier,
    every filing's inspection state with its qualifications and counts, the
    next reads and the cost of the next session. Each matter window carries
    its own matter context beside its excerpt; the per-matter inventory stays
    in the sealed receipt. None when the inventory was not run.
    """
    matters = receipt.litigation_matters
    if matters is None:
        return None
    pending = [w for w in matters.windows if w.status == "PENDING"]
    return {
        "rules_id": matters.rules_id,
        "allowance": {
            "windows_per_session": matters.window_allowance,
            "reads_per_session": matters.read_allowance,
            "window_byte_ceiling": matters.window_byte_ceiling,
            "scope": "one session, which serves one unit of the book; never per issuer",
        },
        "reading_chain": {
            "session_index": matters.session_index,
            "continues_earlier_session": matters.continued_from is not None,
            "cumulative_read_windows": matters.cumulative_read_windows,
            "session_limit": matters.session_limit,
            "window_limit": matters.window_limit,
            # What the declared allowance still admits: sessions after this
            # one and matter windows under the cumulative limit; null on a
            # first reading, whose continuation declares the limits.
            "remaining": (
                None
                if matters.session_limit is None or matters.window_limit is None
                else {
                    "sessions": max(0, matters.session_limit - matters.session_index),
                    "windows": max(
                        0,
                        matters.window_limit
                        - (
                            matters.plan_offset + matters.read_windows
                            if matters.cumulative_read_windows is None
                            else matters.cumulative_read_windows
                        ),
                    ),
                }
            ),
            "rule": "Limits are the cumulative allowance a continuation was requested "
            "under; null for a first reading, which has no continuation allowance of its "
            "own. An exhausted allowance leaves the pending scope pending: it says nothing "
            "about how much of the sources was read.",
        },
        "windows": {
            "planned": matters.planned_windows,
            "read": matters.read_windows,
            "pending": matters.pending_windows,
            "read_earlier": matters.prior_windows,
        },
        "source_bytes": {
            "planned": matters.planned_source_bytes,
            "read": matters.read_source_bytes,
            "pending": matters.pending_source_bytes,
            "measure": "UTF-8 bytes of the source windows; not characters, not the serialized "
            "response",
        },
        "documents_inspected": matters.documents_inspected,
        "documents_skipped_by_form": matters.documents_skipped_by_form,
        "documents": [
            {
                "document_handle": document.document_handle,
                "entity_id": document.entity_id,
                "document_type": document.document_type,
                "report_period_end": document.report_period_end,
                "inspection": document.inspection,
                "qualifications": list(document.qualifications),
                "regions": len(document.regions),
                "matters_inventoried": len(document.matters),
                "named_matters": sum(1 for matter in document.matters if matter.named),
                "unassigned_ranges": len(document.unassigned),
                "windows": {
                    "planned": document.planned_windows,
                    "read": document.read_windows,
                    "pending": document.pending_windows,
                    "read_earlier": document.planned_windows
                    - document.read_windows
                    - document.pending_windows,
                },
                "source_bytes": {
                    "regions": document.region_source_bytes,
                    "read": document.read_source_bytes,
                    "pending": document.pending_source_bytes,
                },
            }
            for document in matters.documents
        ],
        "next_reads": [
            {
                "document_handle": window.document_handle,
                "matter_handle": window.matter_handle,
                "part": [window.part, window.part_count],
                "character_range": [window.character_start, window.character_end],
                "source_bytes": window.source_bytes,
            }
            for window in pending[:_MATTER_NEXT_READS]
        ],
        **(
            {}
            if not matters.needs
            else {
                "allocation_rules_id": matters.allocation_rules_id,
                "families": list(matters.families),
                "needs": {
                    "target_completeness_unspecified": matters.historical_reference_needs,
                    "by_status": {
                        status: sum(1 for need in matters.needs if need.status == status)
                        for status in (
                            "PROVIDED",
                            "PROVIDED_SHARED",
                            "PENDING_ALLOWANCE",
                            "UNRESOLVED",
                        )
                    },
                    "unprovided": [
                        {
                            "document_handle": need.document_handle,
                            "matter_handle": need.matter_handle,
                            "kind": need.kind,
                            "status": need.status,
                            **(
                                {}
                                if need.target_state is None
                                else {"target_state": need.target_state}
                            ),
                            "character_range": (
                                None
                                if need.character_start is None
                                else [need.character_start, need.character_end]
                            ),
                            "detail": need.detail,
                        }
                        for need in matters.needs
                        if need.status in {"PENDING_ALLOWANCE", "UNRESOLVED"}
                    ][: _MATTER_NEXT_READS * 4],
                    "rule": (
                        "A need is one thing a complete reading of a filing's litigation "
                        "regions requires: a matter's opening and its further parts, a "
                        "region-level statement, the qualification a matter depends on, an "
                        "explicit reference. What is PENDING_ALLOWANCE is unread source; what "
                        "is UNRESOLVED points outside what was held or located. Neither is "
                        "evidence of anything, and an absent statement is not a resolution. A "
                        "reference counted under target_completeness_unspecified was sealed "
                        "before target states were recorded: whether its target's evidence "
                        "was delivered is unknown, and this record cannot be continued."
                    ),
                },
            }
        ),
        "continuation": litigation_continuation_scope(receipt),
        "rule": (
            "The inventory is navigation, not proof of discovery: a matter absent from it is "
            "not absent from the filing, unassigned text is unread text, a PENDING window is "
            "unread source and not evidence, and PARTIAL or NONE inspection is not a review of "
            "the filing's proceedings. Nothing here states a status, an outcome or a liability."
        ),
    }


def litigation_continuation_scope(
    receipt: AlternativeEvidenceRetrievalAccessReceipt,
) -> dict[str, object] | None:
    """Describe one continuation's pending litigation reads and cost.

    What one more reading session over this receipt's pending scope would
    read and cost, and what a continuation must declare: the explanation
    that travels beside a delivery's `continuation_request`, never inside
    it -- the request holds only the operation's own parameters. None when
    the inventory was not run.
    """
    matters = receipt.litigation_matters
    if matters is None:
        return None
    pending = [w for w in matters.windows if w.status == "PENDING"]
    channels = routed_pending_work(receipt)
    anything = bool(pending) or any(channels.values())
    remainders = reading_remainders(receipt)
    # The pages this session's share would read: the partial tables
    # first, then the tables not yet dealt, inside the share.
    pages = min(
        TABLE_VIEWS_PER_SESSION,
        channels["partial_table_pages"] + channels["tables_not_dealt"],
    )
    next_session = pending[: max(0, matters.window_allowance - pages)]
    candidates = 0
    if receipt.routing is not None:
        by_issuer: dict[str, int] = {}
        for c in receipt.routing.candidates():
            if c.state == "PENDING":
                by_issuer[c.entity_id] = by_issuer.get(c.entity_id, 0) + 1
        candidates = min(
            MAXIMUM_ISSUED_CANDIDATE_WINDOWS,
            sum(min(n, CANDIDATE_READS_PER_ISSUER) for n in by_issuer.values()),
        )
    return {
        # PENDING: a continuation reads more of the sealed plan.
        # NOTHING_RESUMABLE: no continuation can read more, and the
        # remainders name what the plan leaves unread. COMPLETE: the
        # sealed plan is exhausted and nothing it named remains -- of
        # the plan, never of the sources (see `scope_rule`).
        "state": ("PENDING" if anything else "NOTHING_RESUMABLE" if remainders else "COMPLETE"),
        "pending_windows": len(pending),
        "pending_source_bytes": sum(w.source_bytes for w in pending),
        "pending_candidates": channels["candidates"],
        "partial_table_pages": channels["partial_table_pages"],
        "tables_not_dealt": channels["tables_not_dealt"],
        "remainders": remainders,
        "next_session": {
            "windows": len(next_session),
            "reads": -(-len(next_session) // MATTER_WINDOWS_PER_READ),
            "source_bytes": sum(w.source_bytes for w in next_session),
            "table_pages": pages,
            "candidates": candidates,
            "candidate_reads": -(-candidates // SPANS_PER_READ),
            "model_work": "NONE",
        },
        # A first reading has no continuation allowance of its own: the
        # request's limit slots are null and the consumer declares them.
        "declare": (
            ["session_limit", "window_limit"] if anything and matters.session_limit is None else []
        ),
        "rule": (
            "More source is read only through an explicit continuation request bound "
            "to this packet's sealed receipt and span set (the delivery's "
            "continuation_request, which holds only the operation's parameters; fill "
            "the limits it names under declare); reading another part of this delivery "
            "is not that request and reads no source."
        ),
        "scope_rule": (
            "The states name the sealed reading plan -- the inventoried units, the "
            "routed tables and the returned candidates the plan holds -- never the "
            "sources: passages no question returned, text outside the routed regions "
            "and tables without a retained original were never in the plan, and a "
            "COMPLETE plan or an exhausted budget does not say every source was read."
        ),
    }


def routed_pending_work(receipt: AlternativeEvidenceRetrievalAccessReceipt) -> dict[str, int]:
    """Count pending work available to a continuation.

    The sealed pending plan's work beside the matter windows, as counts
    of what a continuation can read: candidates the chain has not read,
    tables delivered in part that a next page resumes (the sealed pages'
    progress), tables routed and renderable that no session dealt. Zero on
    a receipt that did not route. A table whose progress is unknown or
    whose remainder is a clipped row is not counted here: nothing resumes
    it, and `reading_remainders` names it.
    """
    routing = receipt.routing
    if routing is None:
        return {"candidates": 0, "partial_table_pages": 0, "tables_not_dealt": 0}
    return {
        "candidates": sum(1 for c in routing.candidates() if c.state == "PENDING"),
        "partial_table_pages": sum(
            1 for value in routing.table_progress().values() if value.resumable
        ),
        "tables_not_dealt": routing.table_views_pending,
    }


def reading_remainders(receipt: AlternativeEvidenceRetrievalAccessReceipt) -> dict[str, int]:
    """What the sealed plan leaves unread that no continuation reads, as non-zero counts by name.

    What the sealed plan leaves unread that no continuation reads, as
    non-zero counts by name: tables delivered in part whose remainder is a
    clipped row (`tables_partial_unresumable`), tables whose page progress
    is unknown (`tables_progress_unknown`), tables the boundary refused
    (`tables_unrenderable`), tables in routed regions without a retained
    original (`tables_without_original`), returned candidates the plan had
    no room for (`candidates_beyond_plan`), residual questions the pair
    budget skipped (`questions_skipped_budget`) and references no filing
    resolved (`references_unresolved`). Empty when nothing remains; a
    continuation scope carries it beside its state.
    """
    routing = receipt.routing
    counts: dict[str, int] = {}
    if routing is not None:
        progress = routing.table_progress().values()
        counts["tables_partial_unresumable"] = sum(
            1 for v in progress if v.state == "PARTIAL" and not v.resumable
        )
        counts["tables_progress_unknown"] = sum(
            1 for v in progress if v.state == "UNKNOWN_PROGRESS"
        )
        counts["tables_unrenderable"] = len(routing.table_view_refusals)
        counts["tables_without_original"] = sum(
            cell.tables_without_original for cell in routing.cells
        )
        counts["candidates_beyond_plan"] = routing.candidates_beyond_plan
        counts["questions_skipped_budget"] = routing.questions_skipped_budget
    matters = receipt.litigation_matters
    if matters is not None:
        counts["references_unresolved"] = sum(
            1 for need in matters.needs if need.kind == "REFERENCE" and need.status == "UNRESOLVED"
        )
    return {name: value for name, value in counts.items() if value}


DELIVERY_MEASUREMENT = (
    "utf-8 bytes of the whole JSON response body as the local Web route serializes it "
    "(json.dumps with sort_keys, ASCII-escaped), plus a fixed transport allowance; "
    "no token count is claimed"
)
"""How a delivery is measured. Tokens depend on the consumer's tokenizer, which
the Host does not know for an external consumer; bytes of the actual response
are exact, and a consumer that knows its tokenizer declares its own budget in
bytes. The measurement covers the product's schema, metadata and continuation
overhead, not only the packet text."""

DEFAULT_DELIVERY_BUDGET_BYTES = 256 * 1024
"""The conservative budget a response is admitted against when the consumer
declares none. A budget bounds this response only; it claims nothing about
the consumer's whole conversation."""

MINIMUM_DELIVERY_BUDGET_BYTES = 32 * 1024
MAXIMUM_DELIVERY_BUDGET_BYTES = 16 * 1024 * 1024
"""A declared budget outside these bounds is refused by name."""

DELIVERY_ENVELOPE_ALLOWANCE_BYTES = 4 * 1024
"""Headroom for the transport's own envelope (the client's `data`, URL and
navigation keys), so a body admitted here fits the message that carries it."""


class DeliveryBudgetBelowMinimumUnit(ValueError):
    """Not even one complete item fits the budget.

    Not even one complete item fits the budget: a typed, actionable outcome
    (declare a larger budget), never a cut item under its old identity.
    """

    def __init__(self, item: str, byte_length: int, budget_bytes: int) -> None:
        """Name the complete item that exceeds the delivery budget."""
        super().__init__(
            f"alternative_evidence.delivery_budget_below_minimum_unit:{item}:"
            f"{byte_length}>{budget_bytes}"
        )
        self.item = item
        self.byte_length = byte_length
        self.budget_bytes = budget_bytes


def admit_delivery_budget(budget_bytes: int | None) -> tuple[int, str]:
    """The budget a response is measured against and where it came from."""
    if budget_bytes is None:
        return DEFAULT_DELIVERY_BUDGET_BYTES, "HOST_CONSERVATIVE_DEFAULT"
    if not MINIMUM_DELIVERY_BUDGET_BYTES <= budget_bytes <= MAXIMUM_DELIVERY_BUDGET_BYTES:
        bounds = f"{MINIMUM_DELIVERY_BUDGET_BYTES}..{MAXIMUM_DELIVERY_BUDGET_BYTES}"
        raise ValueError(
            f"alternative_evidence.delivery_budget_invalid:{budget_bytes} outside {bounds}"
        )
    return budget_bytes, "CONSUMER_DECLARED"


def serialized_response_bytes(body: Mapping[str, object]) -> int:
    """The bytes the local Web route puts on the wire for this body."""
    return len(json.dumps(body, default=str, sort_keys=True).encode("utf-8"))


def delivery_part_plan(
    items: tuple[str, ...], part_items: tuple[str, ...], part: int, part_count: int
) -> tuple[bool, tuple[str, ...]]:
    """One part of a delivery over complete items.

    One part of a delivery over complete items: whether it is the whole
    delivery, and the items that remain after it. Parts are contiguous in
    plan order, so what remains is what follows the part's last item.
    """
    whole = part_count == 1 and len(part_items) == len(items)
    end = items.index(part_items[-1]) + 1 if part_items else 0
    return whole, items[end:]


def next_part_request(
    part: int,
    part_count: int,
    *,
    delivery_budget_bytes: int | None,
    **fields: object,
) -> dict[str, object] | None:
    """The request that asks for the next part, or None after the last.

    The request that asks for the next part, or None after the last: the
    caller's fields (the operation, the selector, the identities the part
    answers to), the next part number, and the budget the consumer asked
    with, when it asked with one.
    """
    if part >= part_count:
        return None
    return {
        **fields,
        "delivery_part": part + 1,
        **(
            {}
            if delivery_budget_bytes is None
            else {"delivery_budget_bytes": delivery_budget_bytes}
        ),
    }


def delivery_facts(
    *,
    budget_bytes: int,
    budget_source: str,
    mode: str,
    part: int,
    part_count: int,
    next_request: dict[str, object] | None,
    claim: str,
    **facts: object,
) -> dict[str, object]:
    """Report the bounded delivery facts for a public envelope.

    The `delivery` block every public envelope carries, measured by the
    one delivery mechanism after it is built (`response_bytes` is filled
    there): the budget and its source, the envelope allowance, the mode and
    the part, the builder's own delivered and remaining facts, the
    continuation and the claim that bounds this response alone.
    """
    return {
        "measurement": DELIVERY_MEASUREMENT,
        "budget_bytes": budget_bytes,
        "budget_source": budget_source,
        "envelope_allowance_bytes": DELIVERY_ENVELOPE_ALLOWANCE_BYTES,
        "response_bytes": None,
        "delivery_mode": mode,
        "part": part,
        "part_count": part_count,
        **facts,
        "next_request": next_request,
        "claim": claim,
    }


def plan_delivery(
    items: Sequence[str],
    *,
    fits: Callable[[tuple[str, ...]], bool],
    byte_length: Callable[[tuple[str, ...]], int],
    budget_bytes: int,
) -> tuple[tuple[str, ...], ...]:
    """Cut an ordered sequence of complete items into parts that each fit.

    `fits(items)` says whether the actual response holding exactly those
    items, with everything the response carries beside them, is within the
    budget; the planner never sees bytes it did not measure through it.
    Items keep their order and are never split: a part is the longest prefix
    of what remains that fits (a bounded search over prefixes, since adding
    an item can only grow the response). An item that does not fit on its
    own is a typed refusal, never a truncation. An empty sequence is one part
    when the empty response fits.
    """
    if not items:
        if fits(()):
            return ((),)
        raise DeliveryBudgetBelowMinimumUnit("<empty>", byte_length(()), budget_bytes)
    parts: list[tuple[str, ...]] = []
    start = 0
    while start < len(items):
        first = (items[start],)
        if not fits(first):
            raise DeliveryBudgetBelowMinimumUnit(items[start], byte_length(first), budget_bytes)
        low, high = 1, len(items) - start
        while low < high:
            middle = (low + high + 1) // 2
            if fits(tuple(items[start : start + middle])):
                low = middle
            else:
                high = middle - 1
        parts.append(tuple(items[start : start + low]))
        start += low
    return tuple(parts)


def span_provenance(
    receipt: AlternativeEvidenceRetrievalAccessReceipt,
) -> tuple[dict[str, tuple[str, ...]], dict[str, SelectedWindowFacet]]:
    """What found each read span, as the packet and the finding package both say it.

    What found each read span, as the packet and the finding package both
    say it: `<query_id>:<kind>` for every question that returned it,
    `W01:STRUCTURE:<family>` for a structural scan's admission and
    `T01:TYPED:<family>:<state>` for a typed disclosure rule's binding,
    `M01:MATTER:<matter handle>` for a litigation matter window and
    `M01:EVENT:<unit handle>` for a corporate-event unit's; and the
    facet of each structurally read span.
    """
    found_by: dict[str, tuple[str, ...]] = {}
    for record in receipt.queries:
        for handle in record.hit_span_handles:
            found_by[handle] = (*found_by.get(handle, ()), f"{record.query_id}:{record.kind}")
    scan = receipt.structural_scan
    facets = {} if scan is None else {facet.span_handle: facet for facet in scan.facets}
    for handle, facet in facets.items():
        found_by[handle] = (*found_by.get(handle, ()), f"W01:STRUCTURE:{facet.family}")
    for handle, (observation, _instance) in typed_views(receipt).items():
        found_by[handle] = (
            *found_by.get(handle, ()),
            f"T01:TYPED:{observation.family}:{observation.state}",
        )
    for handle, (document, _matter, window) in matter_views(receipt).items():
        families = {value.handle: value.family for value in document.matters}
        found_by[handle] = (
            *found_by.get(handle, ()),
            f"M01:{_unit_label(families.get(window.matter_handle))}:{window.matter_handle}",
        )
        for value in window.attributions:
            entry = (
                f"M01:{_unit_label(families.get(value.matter_handle))}:{value.matter_handle}"
                if value.matter_handle is not None
                else f"M01:REGION:{value.scope}"
            )
            if entry not in found_by[handle]:
                found_by[handle] = (*found_by[handle], entry)
    # A sealed candidate read in a later session keeps every question that
    # returned it in the first, and names the session that read it; a
    # candidate another channel's span covers lends that span every
    # question that returned it, and names the cover.
    kinds = {query.query_id: query.kind for query in EVIDENCE_QUERY_PROGRAM}
    kinds.update({record.query_id: record.kind for record in receipt.queries})
    if receipt.routing is not None:
        for candidate in receipt.routing.candidates():
            if candidate.span_handle is not None:
                found_by[candidate.span_handle] = (
                    *(f"{q}:{kinds.get(q, 'STATE')}" for q in candidate.found_by),
                    f"C{candidate.session_index or 0:02d}:PENDING_PLAN:{candidate.order}",
                )
            elif candidate.covered_by is not None:
                # Every span of the covering union carries the questions and
                # the marker: the evidence is read across them.
                for cover in (candidate.covered_by, *candidate.covered_with):
                    entries = list(found_by.get(cover, ()))
                    for q in candidate.found_by:
                        entry = f"{q}:{kinds.get(q, 'STATE')}"
                        if entry not in entries:
                            entries.append(entry)
                    marker = (
                        f"C{candidate.session_index or 0:02d}:COVERED_CANDIDATE:{candidate.order}"
                    )
                    if marker not in entries:
                        entries.append(marker)
                    found_by[cover] = tuple(entries)
    return found_by, facets


def _unit_label(family: str | None) -> str:
    """Name the source unit family of a delivered window.

    `MATTER` for a litigation matter (and for unassigned text, which
    the litigation series has always carried), `EVENT` for a corporate-
    event unit, `FINANCING` for a financing unit, `OPERATIONS` for an
    operations unit.
    """
    if family == CORPORATE_EVENT_FAMILY:
        return "EVENT"
    if family == FINANCING_FAMILY:
        return "FINANCING"
    if family == OPERATIONS_FAMILY:
        return "OPERATIONS"
    return "MATTER"


def typed_views(
    receipt: AlternativeEvidenceRetrievalAccessReceipt,
) -> dict[str, tuple[TypedDisclosureObservation, TypedDisclosureInstance | None]]:
    """The typed observation (and instance, for an assertion's span) behind each typed span handle.

    The typed observation (and instance, for an assertion's span) behind
    each typed span handle.
    """
    typed = receipt.typed_disclosures
    if typed is None:
        return {}
    views: dict[str, tuple[TypedDisclosureObservation, TypedDisclosureInstance | None]] = {}
    for observation in typed.observations:
        if observation.scope_span_handle is not None:
            views[observation.scope_span_handle] = (observation, None)
        for instance in observation.instances:
            if instance.span_handle is not None:
                views[instance.span_handle] = (observation, instance)
    return views


def typed_view(
    observation: TypedDisclosureObservation, instance: TypedDisclosureInstance | None
) -> dict[str, object]:
    """The typed assertion or scope as a reader sees it beside the excerpt.

    The typed assertion or scope as a reader sees it beside the excerpt:
    the source's own words bound to fields, never a judgment.
    """
    view: dict[str, object] = {
        "family": observation.family,
        "state": observation.state,
        "rule_id": observation.rule_id,
        "section": {
            "part": observation.part,
            "item": observation.item,
            "title": observation.region_title,
        },
        "report_period_end": observation.report_period_end,
    }
    if instance is None:
        view["scope"] = observation.reason
        view["references"] = observation.references
        view["unrecognized"] = observation.unrecognized
        return view
    view["assertion"] = {
        "instance_id": instance.instance_id,
        "subject": instance.subject,
        "action": instance.action,
        "polarity": instance.polarity,
        "period_end": instance.period_end,
        "period_text": instance.period_text,
        "period_basis": instance.period_basis,
        "character_range": [instance.character_start, instance.character_end],
        "fields": {value.name: value.value for value in instance.fields},
        "qualifiers": instance.qualifiers,
        "unknown_fields": instance.unknown_fields,
    }
    return view


def typed_disclosure_summary(
    receipt: AlternativeEvidenceRetrievalAccessReceipt,
) -> dict[str, object] | None:
    """The packet-level accounting of the typed families.

    The packet-level accounting of the typed families: every observation's
    state, section and handles, the issuers with no periodic filing, and the
    rule the reader must apply. None when the families were not run.
    """
    typed = receipt.typed_disclosures
    if typed is None:
        return None
    # Semantics only, as the whole packet: the rules' name, never the
    # definitions' hash, which the sealed receipt carries.
    return {
        "rules_id": typed.rules_id,
        "documents_inspected": typed.documents_inspected,
        "documents_skipped_by_form": typed.documents_skipped_by_form,
        "issuers_without_periodic_filing": typed.issuers_without_periodic_filing,
        "span_handles": typed.span_handles,
        "read_call_count": typed.read_call_count,
        "observations": tuple(
            {
                "observation_id": value.observation_id,
                "entity_id": value.entity_id,
                "document_handle": value.document_handle,
                "document_type": value.document_type,
                "family": value.family,
                "state": value.state,
                "reason": value.reason,
                "section": {"part": value.part, "item": value.item, "title": value.region_title},
                "report_period_end": value.report_period_end,
                "report_period_status": value.report_period_status,
                "instance_count": len(value.instances),
                "instance_span_handles": [i.span_handle for i in value.instances],
                "undelivered_instance_count": value.undelivered_instance_count,
                "scope_span_handle": value.scope_span_handle,
                "references": value.references,
                "unrecognized": value.unrecognized,
                "context": value.context,
            }
            for value in typed.observations
        ),
        "rule": (
            "A typed observation is what the filing's own section states under a frozen "
            "definition, bound to verified spans; it is not a check outcome, a disposition "
            "or a judgment. EXPLICIT_NONE is scoped to its stated subject and period; "
            "NOT_FOUND, REFERENCE_REQUIRED, AMBIGUOUS and SOURCE_UNAVAILABLE are unresolved "
            "scope to read or report, never evidence of absence."
        ),
    }


def structure_view(facet: SelectedWindowFacet) -> dict[str, object]:
    """The facet as a reader sees it beside the excerpt."""
    return {
        "path": facet.path,
        "family": facet.family,
        "role": facet.role,
        "gaps": facet.gaps,
        "related_span_handles": facet.related_span_handles,
    }


def _carried(section: object, delivery_part: tuple[int, int] | None) -> object:
    """A section every part would repeat unchanged.

    A section every part would repeat unchanged: itself in the whole
    packet and in part 1, a compact reference in a later part -- the part
    that carries it and the content hash the consumer can check it by.
    Nothing is cut: part 1 holds the section whole.
    """
    if delivery_part is None or delivery_part[0] <= 1:
        return section
    return {
        "carried_in_part": 1,
        "content_hash": str(canonical_hash(json.loads(json.dumps(section, default=str)))),
        "entries": len(section) if isinstance(section, (tuple, list)) else None,
        "rule": (
            "This section is identical in every part of the delivery; part 1 carries "
            "it whole under this content hash."
        ),
    }


def render_evidence_packet(
    packet: AlternativeEvidencePacket,
    *,
    span_handles: tuple[str, ...] | None = None,
    delivery_part: tuple[int, int] | None = None,
) -> str:
    """Render the packet for the page and the audit read.

    Render the packet for the page and the audit read. Semantics only; no
    physical identity. An agent reads the bundle (`views.render_analyst_bundle`),
    not this: nothing here instructs a reader.

    `span_handles` renders one delivery part: the named spans with the whole
    packet's obligation, coverage and index, so a part reads as the packet
    does and cites the same handles; `delivery_part` is `(part, part_count)`,
    and a later part names the typed families' accounting by content hash
    (`carried_in_part`), since a consumer of part n holds part 1. Shared
    context is stated once a part, never once a span: a matter's descriptors
    and the window rules in the matter block (`matters_in_part`), the
    limitation sentences in `limitation_texts` (each span lists its codes), a
    document's name, title and time in the index (each span names its
    document handle, issuer, form and availability).
    """
    selected = (
        packet.spans
        if span_handles is None
        else tuple(value for value in packet.spans if value.span_handle in span_handles)
    )
    typed_facts = tuple(
        value.model_view()
        for value in packet.snapshot.citations
        if value.evidence_class is AlternativeEvidenceClass.SEC_COMPANYFACTS
    )
    # Which questions returned each read span (its kind tells the reader which
    # check it is a candidate for), and which same-filing passages it stands for.
    found_by, facets = span_provenance(packet.receipt)
    typed = typed_views(packet.receipt)
    typed_summary = typed_disclosure_summary(packet.receipt)
    matters = matter_views(packet.receipt)
    matter_summary = litigation_matter_summary(packet.receipt)
    grouped = {
        group.representative_span_handle: group.member_span_handles
        for group in packet.receipt.span_groups
    }
    limitation_texts: dict[str, str] = {}
    alias_of = {
        value.span_handle: alias
        for alias, value in span_aliases(packet.spans, packet.request.ordered_entity_ids).items()
    }

    def limitation_codes(limitations: Sequence[str]) -> list[str]:
        codes = []
        for sentence in limitations:
            code = f"L-{hashlib.sha256(sentence.encode('utf-8')).hexdigest()[:8]}"
            limitation_texts[code] = sentence
            codes.append(code)
        return codes

    span_rows = tuple(
        {
            "alias": alias_of[value.span_handle],
            "span_handle": value.span_handle,
            "document_handle": value.document_handle,
            "entity_id": value.entity_id,
            "document_type": value.document_type,
            "available_at": value.available_at.isoformat(),
            "excerpt": value.excerpt,
            "found_by": found_by.get(value.span_handle, ()),
            "stands_for": grouped.get(value.span_handle, ()),
            **(
                {"structure": structure_view(facets[value.span_handle])}
                if value.span_handle in facets
                else {}
            ),
            **(
                {"typed_disclosure": typed_view(*typed[value.span_handle])}
                if value.span_handle in typed
                else {}
            ),
            **(
                {"litigation_matter": matter_window_view(*matters[value.span_handle][1:])}
                if value.span_handle in matters
                else {}
            ),
            **(
                {"unit_declaration": value.unit_declaration.model_dump(mode="json")}
                if value.unit_declaration is not None
                else {}
            ),
            "limitations": limitation_codes(value.limitations),
        }
        for value in selected
    )
    if matter_summary is not None:
        matter_summary = {
            **matter_summary,
            "matters_in_part": matters_in_part(
                matters, tuple(value.span_handle for value in selected)
            ),
            "window_rule": MATTER_WINDOW_RULE,
            "carries_rule": MATTER_CARRIES_RULE,
        }
    payload = {
        "research_obligation": {
            "question": packet.obligation.question,
            "issuer_axis": packet.obligation.ordered_entity_ids,
            "evidence_as_of": packet.obligation.evidence_as_of.isoformat(),
            "approved_source_families": packet.obligation.approved_source_families,
            "required_checks": packet.obligation.required_checks,
        },
        "source_coverage": {
            "status": packet.snapshot.status,
            "expected": packet.snapshot.expected_source_count,
            "available": packet.snapshot.available_source_count,
            "missing": packet.snapshot.missing_source_count,
            "failed": packet.snapshot.failed_source_count,
            "limitations": packet.snapshot.limitations,
            "admitted_documents": len(packet.document_set.documents),
            "rejected_documents": len(packet.document_set.rejections),
        },
        "document_index": tuple(value.model_view() for value in packet.document_set.documents),
        "document_rejections": tuple(
            {
                "document_handle": value.semantic_handle,
                "entity_id": value.entity_id,
                "reason": value.code,
                "summary": value.summary,
            }
            for value in packet.document_set.rejections
        ),
        "typed_facts": typed_facts,
        **(
            {}
            if typed_summary is None
            else {"typed_disclosures": _carried(typed_summary, delivery_part)}
        ),
        **({} if matter_summary is None else {"litigation_matters": matter_summary}),
        "spans": span_rows,
        "limitation_texts": limitation_texts,
    }
    return (
        "# Alternative Evidence Extraction Task\n\n"
        + json.dumps(payload, sort_keys=True, default=str)
        + "\n\nReturn one answer of atomic, source-grounded findings."
    )


__all__ = [
    "DEFAULT_DELIVERY_BUDGET_BYTES",
    "DELIVERY_ENVELOPE_ALLOWANCE_BYTES",
    "DELIVERY_MEASUREMENT",
    "EVIDENCE_QUERY_PROGRAM",
    "EVIDENCE_QUERY_PROGRAM_ID",
    "LITIGATION_FAMILY",
    "MATTER_FAMILIES",
    "MATTER_WINDOWS_PER_READ",
    "MATTER_WINDOW_BYTES",
    "MAXIMUM_DELIVERY_BUDGET_BYTES",
    "MAXIMUM_PACKET_SPANS",
    "MAXIMUM_REFERENCE_SEARCHES",
    "MINIMUM_DELIVERY_BUDGET_BYTES",
    "QUERY_TOP_K",
    "RERANKER_DEPTH_PER_DOCUMENT",
    "RESIDUAL_RERANK_PAIR_BUDGET",
    "RESIDUAL_SELECTION_RULES_ID",
    "SPANS_PER_ISSUER",
    "SPANS_PER_READ",
    "AlternativeEvidencePacket",
    "ContinuedReads",
    "DeliveryBudgetBelowMinimumUnit",
    "EvidenceQuery",
    "MatterEvidenceTrace",
    "RoutedSelection",
    "SharedInventory",
    "admit_delivery_budget",
    "continue_routed_reads",
    "covering_span_handles",
    "delivered_ranges",
    "delivered_table_identities",
    "delivery_facts",
    "delivery_part_plan",
    "litigation_continuation_scope",
    "litigation_matter_summary",
    "matter_view",
    "matter_views",
    "next_part_request",
    "overlapping_span_handles",
    "plan_delivery",
    "plan_table_pages",
    "query_program_hash",
    "reading_remainders",
    "render_evidence_packet",
    "route_session",
    "routed_pending_work",
    "routing_record",
    "select_matter_evidence",
    "select_routed_evidence",
    "select_typed_disclosures",
    "serialized_response_bytes",
    "shared_inventory",
    "span_aliases",
    "span_provenance",
    "span_reading_order",
    "structure_view",
    "table_view_renderable",
    "typed_disclosure_summary",
    "typed_view",
    "typed_views",
]
