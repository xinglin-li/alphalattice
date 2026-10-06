"""The one evidence read model the external Analyst and the UI consume.

A packet's sealed receipt and span set already state everything the
delivery proved; the projections here restate it as bundles a reader can
filter without acquisition, embedding, reranking or interpretation. Each
bundle is one delivered span with its issuer, the topics it serves and the
method that found it, the unit or typed statement it belongs to, its
document's times at their stated precision, the comparison basis its unit
has across filings, the references it leaves unresolved, and how far its
delivery got. The issuer-topic ledger restates the routing's cells and
gaps; the time view resolves an interval on the acceptance-time basis
(publication at its own precision beside it, date-only kept date-only,
unknown time reported as unknown) and keeps the unfiltered coverage in
sight, so a small recent view cannot pass for a complete review.

Source observation, machine-derived metadata and actor interpretation stay
distinguishable: nothing here reports a check as done, a matter as
resolved or a risk as changed. The finding package, the dossier and the
report carry the actor's recorded support and contradiction; this model
carries what was delivered and what was not.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from typing import Any

from alphalattice.evidence.alternative_evidence.publication.analysis import (
    AlternativeEvidenceAnalysisLineage,
    AlternativeEvidenceAnalysisPublicationView,
)

from ..contracts import AlternativeEvidenceRequest
from ..documents.contracts import AlternativeEvidenceDocumentReference
from ..retrieval.contracts import AlternativeEvidenceResolvedSpan
from .contracts import (
    AlternativeEvidenceRetrievalAccessReceipt,
    EvidenceTopic,
    IssuerTopicCellRecord,
    TableProgress,
    TopicRoutingRecord,
    cell_table_states,
)
from .packet import (
    EVIDENCE_QUERY_PROGRAM,
    AlternativeEvidencePacket,
    delivered_ranges,
    matter_view,
    matter_views,
    overlapping_span_handles,
    span_provenance,
    structure_view,
    typed_view,
    typed_views,
)
from .routing import TOPIC_ROUTES, TOPICS, topics_of_unit

EVIDENCE_BUNDLE_VIEW_ID = "alternative-evidence.evidence-bundle-view.v2"
"""v2 (2026-09-20): a table bundle carries the table's progress from the
sealed pages (`table.progress`, `rows_clipped`) and `delivery.whole` is
true only when the table's declared rows are proved delivered, not when
this page happens to be the last."""
DEFAULT_VIEW_DAYS = 30
MAXIMUM_VIEW_DAYS = 3660

_QUERY_TOPICS = {query.query_id: query.topic for query in EVIDENCE_QUERY_PROGRAM}
_TYPED_TOPICS: dict[str, tuple[EvidenceTopic, ...]] = {}
for _route in TOPIC_ROUTES.values():
    for _family in _route.typed_families:
        _TYPED_TOPICS[_family] = (*_TYPED_TOPICS.get(_family, ()), _route.topic)

METHODS = {
    "S": "RESIDUAL_SEARCH",
    "C": "RESIDUAL_SEARCH",
    "W": "STRUCTURAL_SCAN",
    "T": "TYPED_RULE",
    "M": "UNIT_WINDOW",
    "X": "TABLE_VIEW",
}
"""What found a span, by its handle's series: the question bank (or the
unconditional program) -- a search of the first session (`S`) or a sealed
candidate of that search read in a later session of the chain (`C`) --,
the structural scan, a typed disclosure rule, a unit window of an inventory
family, a table view of the original."""


def span_method(span_handle: str) -> str:
    """Name the method that delivered a span handle."""
    return METHODS.get(span_handle.split("-")[1][0], "UNKNOWN")


def _references(
    packet: AlternativeEvidencePacket,
) -> dict[str, AlternativeEvidenceDocumentReference]:
    return {value.semantic_handle: value for value in packet.document_set.documents}


def span_topics(
    receipt: AlternativeEvidenceRetrievalAccessReceipt, span: AlternativeEvidenceResolvedSpan
) -> tuple[str, ...]:
    """The topics a delivered span serves, primary first.

    The topics a delivered span serves, primary first: the questions'
    topics for a searched span, the typed family's for a typed span, the
    unit's for a window, the record's for a table view. Empty for a
    structural scan's span (its family names its section, not a topic).
    """
    handle = span.span_handle
    method = span_method(handle)
    topics: list[str] = []
    if method == "RESIDUAL_SEARCH":
        for record in receipt.queries:
            if handle in record.hit_span_handles:
                topic = str(_QUERY_TOPICS.get(record.query_id, record.topic))
                if topic not in topics:
                    topics.append(topic)
        if receipt.routing is not None:
            for candidate in receipt.routing.candidates():
                if candidate.span_handle == handle:
                    for query_id in (candidate.topic, *candidate.found_by):
                        topic = str(_QUERY_TOPICS.get(str(query_id), query_id))
                        if topic not in topics:
                            topics.append(topic)
    elif method == "TYPED_RULE":
        typed = typed_views(receipt).get(handle)
        if typed is not None:
            topics.extend(str(t) for t in _TYPED_TOPICS.get(typed[0].family, ()))
    elif method == "UNIT_WINDOW":
        found = matter_views(receipt).get(handle)
        if found is not None:
            document, matter, window = found
            family = "LITIGATION" if matter is None else matter.family
            topics.extend(
                str(t)
                for t in topics_of_unit(
                    family=family,
                    region_heading=(
                        matter.region_heading
                        if matter is not None
                        else next(
                            (a.region_heading for a in window.attributions if a.region_heading),
                            "",
                        )
                    ),
                    title="" if matter is None else matter.title,
                    document_type=document.document_type,
                    excerpt=span.excerpt,
                )
            )
    elif method == "TABLE_VIEW" and receipt.routing is not None:
        for view in receipt.routing.table_views:
            if view.span_handle == handle:
                topics.extend(str(t) for t in view.topics)
    return tuple(topics)


def comparison_basis(
    routing: TopicRoutingRecord | None, document_handle: str, unit_handle: str | None
) -> dict[str, object] | None:
    """What the comparison rules recorded for a unit against the issuer's other filings.

    What the comparison rules recorded for a unit against the issuer's
    other filings: the later side's state against the earlier unit, or --
    for an earlier unit -- that a later filing restated or changed it.
    """
    if routing is None or unit_handle is None:
        return None
    for value in routing.correspondences:
        if (
            value.later_document_handle == document_handle
            and value.later_unit_handle == unit_handle
        ):
            return {
                "state": value.state,
                "against": value.earlier_document_handle,
                "against_unit": value.earlier_unit_handle,
                "ratio": value.ratio,
                "changed_characters": value.changed_characters,
                "basis": value.basis,
                "rules_id": routing.comparison_rules_id,
            }
        if (
            value.earlier_document_handle == document_handle
            and value.earlier_unit_handle == unit_handle
        ):
            return {
                "state": "RESTATED_LATER" if value.state == "EXACT_REPEAT" else "COMPARED_LATER",
                "by": value.later_document_handle,
                "by_unit": value.later_unit_handle,
                "later_state": value.state,
                "ratio": value.ratio,
                "changed_characters": value.changed_characters,
                "basis": value.basis,
                "rules_id": routing.comparison_rules_id,
            }
    return {
        "state": "NOT_COMPARED",
        "basis": "no other filing of the issuer inventoried the unit's family",
    }


def _time_of(reference: AlternativeEvidenceDocumentReference | None) -> dict[str, object]:
    if reference is None:
        return {
            "accepted_at": None,
            "published_at": None,
            "published_precision": "UNKNOWN",
            "available_at": None,
            "availability_basis": "REFERENCE_ABSENT",
            "report_period_end": None,
        }
    return reference.temporal_view()


def evidence_bundles(packet: AlternativeEvidencePacket) -> tuple[dict[str, Any], ...]:
    """Every delivered span of the packet as one bundle, in packet order."""
    return bundles_of(packet.receipt, _references(packet), packet.spans)


def bundles_of(
    receipt: AlternativeEvidenceRetrievalAccessReceipt,
    references: Mapping[str, AlternativeEvidenceDocumentReference],
    spans: Sequence[AlternativeEvidenceResolvedSpan],
) -> tuple[dict[str, Any], ...]:
    """Collect delivered span bundles and their document times.

    The bundles of delivered spans under a sealed receipt, with the
    document references their times come from -- the packet's, or a
    replayed publication's lineage.
    """
    found_by, facets = span_provenance(receipt)
    typed = typed_views(receipt)
    matters = matter_views(receipt)
    routing = receipt.routing
    progress = {} if routing is None else routing.table_progress()
    unresolved: dict[str, list[str]] = {}
    pending_by_unit: dict[tuple[str, str], int] = {}
    if receipt.litigation_matters is not None:
        for need in receipt.litigation_matters.needs:
            if need.kind == "REFERENCE" and need.status == "UNRESOLVED":
                unresolved.setdefault(need.document_handle, []).append(need.detail)
        for window in receipt.litigation_matters.windows:
            if window.status == "PENDING":
                key = (window.document_handle, window.matter_handle)
                pending_by_unit[key] = pending_by_unit.get(key, 0) + 1
    bundles: list[dict[str, Any]] = []
    for span in spans:
        handle = span.span_handle
        method = span_method(handle)
        unit: dict[str, object] | None = None
        statement: dict[str, object] | None = None
        delivery: dict[str, object] = {"state": "DELIVERED", "whole": True}
        unit_handle: str | None = None
        if handle in matters:
            document, matter, window = matters[handle]
            unit_handle = window.matter_handle
            unit = {
                "unit_handle": window.matter_handle,
                "family": None if matter is None else matter.family,
                "title": None if matter is None else matter.title,
                "region_heading": None if matter is None else matter.region_heading,
                "named_proceeding": None if matter is None else matter.named,
                "case_numbers": () if matter is None else matter.case_numbers,
            }
            delivery = {
                "state": "DELIVERED",
                "whole": window.delivered_whole,
                "part": window.part,
                "part_count": window.part_count,
                "unit_windows_pending": pending_by_unit.get(
                    (window.document_handle, window.matter_handle), 0
                ),
                "document_inspection": document.inspection,
                "document_windows_pending": document.pending_windows,
            }
        if handle in typed:
            observation, instance = typed[handle]
            statement = {
                "family": observation.family,
                "state": observation.state,
                "rule_id": observation.rule_id,
                "section": {"part": observation.part, "item": observation.item},
                "report_period_end": observation.report_period_end,
                **(
                    {}
                    if instance is None
                    else {
                        "subject": instance.subject,
                        "action": instance.action,
                        "polarity": instance.polarity,
                        "period_end": instance.period_end,
                    }
                ),
            }
        table = None
        if span.table_view is not None:
            table_progress = progress.get((span.document_handle, span.table_view.table_ordinal))
            table = {
                "table_ordinal": span.table_view.table_ordinal,
                "rows_total": span.table_view.rows_total,
                "rows_from": span.table_view.rows_from,
                "rows_to": span.table_view.rows_to,
                "remaining_rows": span.table_view.remaining_rows,
                "rows_clipped": span.table_view.rows_clipped,
                "headings": span.table_view.headings,
                "scale": span.table_view.scale,
                "footnote_count": span.table_view.footnote_count,
                "original_content_hash": span.table_view.parent_source_content_hash,
                # The whole table's state from every sealed page of the chain
                # (this page is one of them); a span read outside a routed
                # selection has no sealed pages and its progress is unknown.
                "progress": _table_progress_view(table_progress),
            }
            delivery = {
                "state": "DELIVERED",
                "whole": table_progress is not None and table_progress.state == "COMPLETE",
                "remaining_rows": span.table_view.remaining_rows,
            }
        reference = references.get(span.document_handle)
        bundles.append(
            {
                "span_handle": handle,
                "entity_id": span.entity_id,
                "document_handle": span.document_handle,
                "document_type": span.document_type,
                "method": method,
                "found_by": found_by.get(handle, ()),
                "topics": span_topics(receipt, span),
                "unit": unit,
                "statement": statement,
                "table": table,
                "structure_family": (facets[handle].family if handle in facets else None),
                "time": _time_of(reference),
                "unit_declaration": (
                    None
                    if span.unit_declaration is None
                    else span.unit_declaration.model_dump(mode="json")
                ),
                "qualifications": list(span.limitations),
                "comparison": comparison_basis(routing, span.document_handle, unit_handle),
                "unresolved_references": tuple(unresolved.get(span.document_handle, ())),
                "delivery": delivery,
                "character_range": [span.character_start, span.character_end],
                "excerpt_bytes": len(span.excerpt.encode("utf-8")),
            }
        )
    return tuple(bundles)


def _table_progress_view(value: TableProgress | None) -> dict[str, object]:
    if value is None:
        return {"state": "UNKNOWN_PROGRESS", "rows_declared": 0, "rows_delivered": 0}
    return {
        "state": value.state,
        "rows_declared": value.rows_declared,
        "rows_delivered": value.rows_delivered,
        "next_row": value.next_row,
        "pages": value.pages,
        "gap": value.gap,
        "rows_clipped": value.rows_clipped,
    }


def _cell_view(
    cell: IssuerTopicCellRecord,
    progress: Mapping[tuple[str, int], TableProgress],
    overlapping: int | None = None,
) -> dict[str, object]:
    tables = cell_table_states(progress, cell.entity_id, cell.topic)
    tables_pending = max(0, cell.tables - cell.tables_delivered - cell.tables_unrenderable)
    # Why the cell is not COMPLETE, by name; COMPLETE is the empty list.
    incomplete: list[str] = []
    if cell.unit_needs_delivered < cell.unit_needs:
        incomplete.append("UNIT_NEEDS_UNREAD")
    if tables_pending:
        incomplete.append("TABLES_NOT_DEALT")
    if tables["PARTIAL"]:
        incomplete.append("TABLES_DELIVERED_IN_PART")
    if tables["UNKNOWN_PROGRESS"]:
        incomplete.append("TABLES_PROGRESS_UNKNOWN")
    if cell.tables_unrenderable:
        incomplete.append("TABLES_UNRENDERABLE")
    if cell.tables_without_original:
        incomplete.append("TABLES_WITHOUT_ORIGINAL")
    if cell.candidates_pending:
        incomplete.append("CANDIDATES_UNREAD")
    if cell.residual == "QUEUED":
        incomplete.append("RESIDUAL_SCOPE_QUEUED")
    if cell.residual == "BROADER":
        # No region routes to the topic: the broader pass is a bounded
        # search over the issuer's documents, never a delivered scope.
        incomplete.append("ROUTE_GAP")
    if any(gap.startswith("SOURCE_GAP") for gap in cell.gaps):
        incomplete.append("SOURCE_GAP")
    delivered_anything = bool(
        cell.unit_needs_delivered
        or cell.tables_delivered
        or cell.residual_hits
        or cell.candidates_read
        or cell.typed_observations
    )
    return {
        "entity_id": cell.entity_id,
        "topic": str(cell.topic),
        "forms_held": list(cell.forms_held),
        "regions": cell.regions,
        "regions_by_basis": [list(pair) for pair in cell.regions_by_basis],
        "unit_needs": cell.unit_needs,
        "unit_needs_delivered": cell.unit_needs_delivered,
        "unit_needs_pending": max(0, cell.unit_needs - cell.unit_needs_delivered),
        "typed_observations": cell.typed_observations,
        "tables": cell.tables,
        "tables_delivered": cell.tables_delivered,
        "tables_complete": tables["COMPLETE"],
        "tables_partial": tables["PARTIAL"],
        "tables_progress_unknown": tables["UNKNOWN_PROGRESS"],
        "tables_without_original": cell.tables_without_original,
        "tables_unrenderable": cell.tables_unrenderable,
        "tables_pending": tables_pending,
        "residual": cell.residual,
        "residual_windows": cell.residual_windows,
        "residual_hits": cell.residual_hits,
        "candidates_pending": cell.candidates_pending,
        "candidates_read": cell.candidates_read,
        "candidates_covered": cell.candidates_covered,
        # Pending candidates some delivered span of the same filing
        # intersects without holding whole: still unread, reported apart
        # from the covered (delivered whole) so a partial overlap is never
        # read as delivery. Absent when the ledger was built without spans.
        **({} if overlapping is None else {"candidates_overlapping": overlapping}),
        "gaps": list(cell.gaps),
        "incomplete": incomplete,
        "delivery_state": (
            "NO_SOURCE"
            if cell.residual == "NO_SOURCE"
            else "NO_ROUTE"
            if cell.residual == "BROADER" and cell.unit_needs == 0 and cell.typed_observations == 0
            else "COMPLETE"
            if not incomplete
            else "PARTIAL"
            if delivered_anything
            else "PENDING"
        ),
    }


def topic_coverage_ledger(
    receipt: AlternativeEvidenceRetrievalAccessReceipt,
    spans: Sequence[AlternativeEvidenceResolvedSpan] | None = None,
) -> dict[str, Any] | None:
    """The issuer-topic ledger of an integrated selection.

    The issuer-topic ledger of an integrated selection: every cell with
    its source, representation, discovery and delivery states and gaps;
    the per-topic aggregate; the residual search's budget; and what no
    route reached. None for a selection that did not route. With the
    delivered `spans`, each cell and the candidates block also count the
    pending candidates a delivered span partially overlaps (never
    delivered whole; still unread).
    """
    routing = receipt.routing
    if routing is None:
        return None
    progress = routing.table_progress()
    sealed_candidates = routing.candidates()
    overlapping_by_cell: dict[tuple[str, str], int] | None = None
    if spans is not None:
        held = delivered_ranges(spans)
        overlapping_by_cell = {}
        for candidate in sealed_candidates:
            if candidate.state != "PENDING":
                continue
            if overlapping_span_handles(
                held,
                candidate.document_handle,
                candidate.character_start,
                candidate.character_end,
            ):
                cell_key = (candidate.entity_id, str(candidate.topic))
                overlapping_by_cell[cell_key] = overlapping_by_cell.get(cell_key, 0) + 1
    cells = [
        _cell_view(
            cell,
            progress,
            None
            if overlapping_by_cell is None
            else overlapping_by_cell.get((cell.entity_id, str(cell.topic)), 0),
        )
        for cell in routing.cells
    ]
    by_topic: dict[str, dict[str, int]] = {}
    for cell in cells:
        topic = str(cell["topic"])
        aggregate = by_topic.setdefault(
            topic,
            {
                "issuers": 0,
                "unit_needs": 0,
                "unit_needs_delivered": 0,
                "tables": 0,
                "tables_delivered": 0,
                "tables_complete": 0,
                "tables_partial": 0,
                "tables_progress_unknown": 0,
                "tables_without_original": 0,
                "residual_hits": 0,
                "candidates_pending": 0,
                "candidates_read": 0,
                "cells_with_gaps": 0,
                "cells_no_route": 0,
                "cells_no_source": 0,
            },
        )
        aggregate["issuers"] += 1
        for key in (
            "unit_needs",
            "unit_needs_delivered",
            "tables",
            "tables_delivered",
            "tables_complete",
            "tables_partial",
            "tables_progress_unknown",
            "tables_without_original",
            "residual_hits",
            "candidates_pending",
            "candidates_read",
        ):
            aggregate[key] += int(cell[key])  # type: ignore[call-overload]
        aggregate["cells_with_gaps"] += 1 if cell["gaps"] else 0
        aggregate["cells_no_route"] += 1 if cell["delivery_state"] == "NO_ROUTE" else 0
        aggregate["cells_no_source"] += 1 if cell["delivery_state"] == "NO_SOURCE" else 0
    return {
        "rules_id": routing.rules_id,
        "comparison_rules_id": routing.comparison_rules_id,
        "allocation_rules_id": routing.allocation_rules_id,
        "cells": cells,
        "topics": [{"topic": str(topic), **by_topic.get(str(topic), {})} for topic in TOPICS],
        "residual_search": {
            "questions_run": routing.questions_run,
            "questions_skipped_no_scope": routing.questions_skipped_no_scope,
            "questions_skipped_budget": routing.questions_skipped_budget,
            "rerank_pair_budget": routing.residual_rerank_pair_budget,
            "reranked_pairs": routing.residual_reranked_pairs,
        },
        "tables": {
            # Pages delivered, and the tables they prove: complete, delivered
            # in part (resumable or not), of unknown progress; the tables
            # not dealt a view; the tables refused or without an original.
            "pages_delivered": len(routing.table_view_span_handles),
            "complete": sum(1 for v in progress.values() if v.state == "COMPLETE"),
            "partial": sum(1 for v in progress.values() if v.state == "PARTIAL"),
            "partial_resumable": sum(1 for v in progress.values() if v.resumable),
            "progress_unknown": sum(1 for v in progress.values() if v.state == "UNKNOWN_PROGRESS"),
            "not_dealt": routing.table_views_pending,
            "unrenderable": len(routing.table_view_refusals),
            "without_original": sum(cell.tables_without_original for cell in routing.cells),
        },
        "candidates": {
            # The residual channel's sealed plan: every returned candidate
            # the batch did not read (six a cell on a record sealed before the
            # frontier), what the chain read of it, what another channel's
            # span covers, and what no plan could seal.
            "rules_id": routing.residual_selection_rules_id,
            "sealed": len(sealed_candidates),
            "pending": sum(1 for c in sealed_candidates if c.state == "PENDING"),
            "read": len(routing.candidate_span_handles),
            "covered": sum(1 for c in sealed_candidates if c.state == "COVERED"),
            **(
                {}
                if overlapping_by_cell is None
                else {"overlapping": sum(overlapping_by_cell.values())}
            ),
            "beyond_plan": routing.candidates_beyond_plan,
        },
        "exact_repeats_collapsed": routing.exact_repeats_collapsed,
        "unrouted_windows": routing.unrouted_windows,
        "states_rule": (
            "Machine states of sources, representation, discovery and delivery: COMPLETE "
            "means every routed unit of the cell was read, every routed table's declared "
            "rows are proved delivered by the sealed pages (a page is not a table), no "
            "table lacks an original or a rendering, no returned candidate is unread and "
            "no residual search is queued and the cell carries no source or route gap -- "
            "not that the topic has been checked, and not that every passage of the "
            "sources was read; each cell's `incomplete` names what keeps it from "
            "COMPLETE. PARTIAL names a cell with something delivered and something "
            "unread, undeliverable or missing; PENDING a routed cell with nothing "
            "delivered yet; NO_ROUTE a topic no region of the issuer's documents routes "
            "to and no typed observation serves, searched by one bounded broader pass; "
            "NO_SOURCE an issuer with no admitted document. An actor's check is recorded "
            "elsewhere and never erases a gap named here."
        ),
    }


def _interval(
    *,
    cutoff: datetime,
    last_days: int | None,
    view_from: datetime | None,
    view_to: datetime | None,
) -> tuple[datetime, datetime, str]:
    """The resolved view interval.

    The resolved view interval: explicit bounds when given (the end never
    later than the cutoff), else the last `last_days` ending at the cutoff.
    """
    end = cutoff if view_to is None or view_to > cutoff else view_to
    if view_from is not None:
        start = view_from
        basis = "EXPLICIT_INTERVAL"
    else:
        days = DEFAULT_VIEW_DAYS if last_days is None else last_days
        if not 1 <= days <= MAXIMUM_VIEW_DAYS:
            raise ValueError(f"alternative_evidence.view_interval_invalid:{days}")
        start = end - timedelta(days=days)
        basis = f"LAST_{days}_DAYS_ENDING_AT_CUTOFF" if view_to is None else f"LAST_{days}_DAYS"
    if start >= end:
        raise ValueError("alternative_evidence.view_interval_invalid:empty")
    return start.astimezone(UTC), end.astimezone(UTC), basis


def _placement(time: dict[str, object], start: datetime, end: datetime) -> tuple[str, str]:
    """`(placement, basis)`.

    `(placement, basis)`: IN, OUT, BOUNDARY (a date-only publication whose
    day touches the interval's bounds) or UNKNOWN (no acceptance and no
    publication time).
    """
    accepted = time.get("accepted_at")
    if isinstance(accepted, str):
        instant = datetime.fromisoformat(accepted)
        return ("IN" if start <= instant <= end else "OUT"), "ACCEPTANCE"
    published = time.get("published_at")
    if isinstance(published, str):
        instant = datetime.fromisoformat(published)
        if time.get("published_precision") == "DATE":
            day: date = instant.date()
            if start.date() < day < end.date():
                return "IN", "PUBLICATION_DATE"
            if day in {start.date(), end.date()}:
                return "BOUNDARY", "PUBLICATION_DATE"
            return "OUT", "PUBLICATION_DATE"
        return ("IN" if start <= instant <= end else "OUT"), "PUBLICATION_INSTANT"
    return "UNKNOWN", "NO_STATED_TIME"


def time_view(
    packet: AlternativeEvidencePacket,
    *,
    entity_id: str | None = None,
    topic: str | None = None,
    last_days: int | None = None,
    view_from: datetime | None = None,
    view_to: datetime | None = None,
) -> dict[str, Any]:
    """The read-only time view over a prepared packet's delivered evidence.

    The read-only time view over a prepared packet's delivered evidence:
    the resolved interval (ending at the run's evidence cutoff unless an
    earlier end is asked for), the bundles whose document was accepted in
    it -- publication at its own precision when acceptance is unknown, a
    date-only day on a bound reported as BOUNDARY, no stated time reported
    as UNKNOWN -- the older bundles that in-interval units restate or
    change as labelled historical context, the bundles outside the
    interval counted, and the unfiltered ledger. Filtering reads sealed
    artifacts only: no acquisition, no embedding, no reranking.
    """
    request: AlternativeEvidenceRequest = packet.request
    start, end, basis = _interval(
        cutoff=request.evidence_as_of, last_days=last_days, view_from=view_from, view_to=view_to
    )
    if topic is not None and topic not in {str(t) for t in TOPICS}:
        raise ValueError(f"alternative_evidence.view_topic_unknown:{topic}")
    bundles = evidence_bundles(packet)
    selected = [
        b
        for b in bundles
        if (entity_id is None or b["entity_id"] == entity_id)
        and (topic is None or topic in b["topics"])
    ]
    inside: list[dict[str, Any]] = []
    boundary: list[dict[str, Any]] = []
    unknown: list[dict[str, Any]] = []
    outside = 0
    for bundle in selected:
        placement, time_basis = _placement(bundle["time"], start, end)
        placed = {**bundle, "placement": placement, "time_basis": time_basis}
        if placement == "IN":
            inside.append(placed)
        elif placement == "BOUNDARY":
            boundary.append(placed)
        elif placement == "UNKNOWN":
            unknown.append(placed)
        else:
            outside += 1
    # Historical context: the earlier units an in-interval unit restates or
    # changes -- as delivered bundles when this packet read them, else as
    # pointers to the earlier source with its own time (an exact repeat is
    # served by the later reading and read once) -- labelled, never recent.
    context: list[dict[str, Any]] = []
    references = _references(packet)
    by_unit = {
        (b["document_handle"], b["unit"]["unit_handle"]): b
        for b in bundles
        if b["unit"] is not None
    }
    named: set[tuple[str, str]] = set()
    for bundle in inside:
        comparison = bundle.get("comparison") or {}
        against = comparison.get("against")
        against_unit = comparison.get("against_unit")
        if not (isinstance(against, str) and isinstance(against_unit, str)):
            continue
        if (against, against_unit) in named:
            continue
        named.add((against, against_unit))
        earlier = by_unit.get((against, against_unit))
        if earlier is not None:
            context.append(
                {
                    **earlier,
                    "placement": "HISTORICAL_CONTEXT",
                    "context_for": bundle["span_handle"],
                    "relation": comparison.get("state"),
                }
            )
        else:
            context.append(
                {
                    "span_handle": bundle["span_handle"],
                    "entity_id": bundle["entity_id"],
                    "document_handle": against,
                    "unit": {"unit_handle": against_unit},
                    "time": _time_of(references.get(against)),
                    "placement": "HISTORICAL_CONTEXT",
                    "context_for": bundle["span_handle"],
                    "relation": comparison.get("state"),
                    "delivery": {
                        "state": "SERVED_BY_LATER_READING"
                        if comparison.get("state") == "EXACT_REPEAT"
                        else "NOT_DELIVERED_IN_PACKET",
                        "whole": False,
                    },
                }
            )
    ledger = topic_coverage_ledger(packet.receipt, packet.spans)
    return {
        "view_id": EVIDENCE_BUNDLE_VIEW_ID,
        "interval": {
            "from": start.isoformat(),
            "to": end.isoformat(),
            "timezone": "UTC",
            "basis": basis,
            "evidence_cutoff": request.evidence_as_of.astimezone(UTC).isoformat(),
            "time_basis": "ACCEPTANCE",
            "rule": (
                "A document is in the interval by its official acceptance time; when the "
                "source states no acceptance, by its publication at the stated precision "
                "-- a date-only publication on a bound is BOUNDARY, never an invented "
                "instant; a document with no stated time is UNKNOWN. Retrieval time and "
                "the wall clock place nothing. Nothing after the evidence cutoff exists in "
                "this packet."
            ),
        },
        "filters": {"entity_id": entity_id, "topic": topic},
        "bundles": inside,
        "boundary": boundary,
        "unknown_time": unknown,
        "historical_context": context,
        "outside_interval": outside,
        "selected": len(selected),
        "delivered_total": len(bundles),
        "coverage": ledger,
        "claim": (
            "A view of one sealed evidence set: filtering completes no check, changes no "
            "scope and conceals no gap; the unfiltered coverage travels with it."
        ),
    }


def finding_filings(
    lineage: AlternativeEvidenceAnalysisLineage,
) -> dict[str, frozenset[tuple[str, str]]]:
    """Map each finding of an analysis to the filings it cites.

    Args:
        lineage: The analysis's verified lineage.

    Returns:
        Each finding's handle with the filings its supporting and contradicting spans cite,
        as (issuer, accession).
    """
    documents = {
        value.semantic_handle: (value.entity_id, value.revision_label)
        for value in lineage.document_set.documents
    }
    spans = {
        value.span_handle: documents.get(value.document_handle)
        for value in lineage.cro_package.verified_spans
    }
    return {
        finding.finding_handle: frozenset(
            filing
            for handle in (*finding.supporting_span_handles, *finding.contradicting_span_handles)
            if (filing := spans.get(handle)) is not None
        )
        for finding in lineage.brief.findings
    }


def earlier_findings(
    readings: tuple[
        tuple[AlternativeEvidenceAnalysisPublicationView, frozenset[tuple[str, str]]], ...
    ],
) -> tuple[tuple[str, str], ...]:
    """Write one line of context per earlier finding on a filing still in the window.

    By issuer: what was found and when, never the filing again (W3).

    Args:
        readings: Each earlier analysis with the filings it read that are still in the window.

    Returns:
        (issuer, line) pairs, each once, in the readings' order.
    """
    lines: list[tuple[str, str]] = []
    for view, filings in readings:
        cited = finding_filings(view.lineage)
        found = view.lineage.request.evidence_as_of.date().isoformat()
        for finding in view.lineage.brief.findings:
            for entity in finding.affected_entities:
                if any(filing[0] == entity for filing in cited[finding.finding_handle] & filings):
                    summary = " ".join(finding.summary.split())
                    lines.append(
                        (
                            entity,
                            f"{entity} (found {found}; {finding.topic.value}, "
                            f"{finding.direction.value}): "
                            + (summary if len(summary) <= 300 else summary[:299] + "\u2026"),
                        )
                    )
    return tuple(dict.fromkeys(lines))


def finding_excerpts(
    view: AlternativeEvidenceAnalysisPublicationView,
    *handle_groups: tuple[str, ...],
    qualified: bool,
) -> tuple[list[dict[str, object]], ...]:
    """Deliver the verified excerpts a finding cites, each with its document and provenance.

    For each span: its document's type, revision and times, the excerpt and its range, the
    question that found it (and its kind), the same-filing passages it stood for, its structure,
    typed disclosure and litigation matter where the receipt holds them, and the shared read
    model's memberships (topics, method, comparison basis), beside the actor's claim, never
    adjudicating it. A handle the publication did not deliver is named as such.

    Args:
        view: The analysis publication the finding came from.
        *handle_groups: Each group of span handles to deliver (a finding's support, its
            contrary spans).
        qualified: Whether the handles carry a child publication's qualifier (an aggregate
            dossier's `P<hash>:` prefix).

    Returns:
        One list of excerpts per group, in the groups' order.
    """
    receipt = view.lineage.access_receipt
    found_by, facets = span_provenance(receipt)
    typed = typed_views(receipt)
    matters = matter_views(receipt)
    stands_for = {
        group.representative_span_handle: group.member_span_handles for group in receipt.span_groups
    }
    documents = {value.semantic_handle: value for value in view.lineage.document_set.documents}
    spans = {value.span_handle: value for value in view.lineage.cro_package.verified_spans}

    def local(handle: str) -> str:
        return handle.split(":", 1)[1] if qualified and ":" in handle else handle

    def excerpts(handles: tuple[str, ...]) -> list[dict[str, object]]:
        items: list[dict[str, object]] = []
        for qualified in handles:
            span = spans.get(local(qualified))
            if span is None:
                items.append({"span_handle": qualified, "status": "NOT_DELIVERED_IN_PACKET"})
                continue
            document = documents.get(span.document_handle)
            unit_handle = (
                matters[local(qualified)][2].matter_handle if local(qualified) in matters else None
            )
            items.append(
                {
                    "span_handle": qualified,
                    "entity_id": span.entity_id,
                    "document_handle": span.document_handle,
                    "document_type": span.document_type,
                    "revision_label": span.revision_label,
                    "title": span.title,
                    "available_at": span.available_at.isoformat(),
                    "published_at": None
                    if span.published_at is None
                    else span.published_at.isoformat(),
                    **({} if document is None else {"time": document.temporal_view()}),
                    "amendment": span.document_type.upper().endswith("/A"),
                    "character_range": [span.character_start, span.character_end],
                    "excerpt": span.excerpt,
                    "found_by": found_by.get(local(qualified), ()),
                    "stands_for": stands_for.get(local(qualified), ()),
                    **(
                        {"structure": structure_view(facets[local(qualified)])}
                        if local(qualified) in facets
                        else {}
                    ),
                    **(
                        {"typed_disclosure": typed_view(*typed[local(qualified)])}
                        if local(qualified) in typed
                        else {}
                    ),
                    **(
                        {"litigation_matter": matter_view(*matters[local(qualified)])}
                        if local(qualified) in matters
                        else {}
                    ),
                    **(
                        {"unit_declaration": span.unit_declaration.model_dump(mode="json")}
                        if span.unit_declaration is not None
                        else {}
                    ),
                    "limitations": span.limitations,
                    "source_verified": document is not None,
                    # The shared read model's memberships: machine-derived,
                    # beside the actor's claim, never adjudicating it.
                    "topics": list(span_topics(receipt, span)),
                    "method": span_method(local(qualified)),
                    **(
                        {}
                        if receipt.routing is None
                        else {
                            "comparison": comparison_basis(
                                receipt.routing, span.document_handle, unit_handle
                            )
                        }
                    ),
                }
            )
        return items

    return tuple(excerpts(group) for group in handle_groups)


__all__ = [
    "DEFAULT_VIEW_DAYS",
    "EVIDENCE_BUNDLE_VIEW_ID",
    "MAXIMUM_VIEW_DAYS",
    "METHODS",
    "bundles_of",
    "comparison_basis",
    "earlier_findings",
    "evidence_bundles",
    "finding_excerpts",
    "finding_filings",
    "span_method",
    "span_topics",
    "time_view",
    "topic_coverage_ledger",
]
