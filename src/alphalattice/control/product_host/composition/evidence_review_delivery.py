"""Deliver bounded Evidence and CRO read views through one deterministic owner.

The Evidence & CRO section's bounded deliveries: the analyst packet and its
evidence views, the book ledger, the CRO dossier, a finding's evidence package and the
published review's export. One owner, composed by the review application and read by the
operation owner; each delivery builds its surface's body and shares one part loop.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast
from uuid import UUID

from pydantic import ValidationError

from alphalattice.control.task_control.registry import TaskNotFoundError
from alphalattice.evidence.alternative_evidence.analysis.contracts import (
    AlternativeEvidenceAnalystBrief,
    AlternativeEvidenceRetrievalAccessReceipt,
)
from alphalattice.evidence.alternative_evidence.analysis.packet import (
    DELIVERY_ENVELOPE_ALLOWANCE_BYTES,
    AlternativeEvidencePacket,
    DeliveryBudgetBelowMinimumUnit,
    admit_delivery_budget,
    delivery_facts,
    delivery_part_plan,
    litigation_continuation_scope,
    next_part_request,
    plan_delivery,
    render_evidence_packet,
    routed_pending_work,
    serialized_response_bytes,
)
from alphalattice.evidence.alternative_evidence.analysis.read_model import (
    bundles_of,
    evidence_bundles,
    finding_excerpts,
    time_view,
    topic_coverage_ledger,
)
from alphalattice.evidence.alternative_evidence.publication.analysis import (
    AlternativeEvidenceAnalysisLineage,
    AlternativeEvidenceAnalysisPublicationView,
)
from alphalattice.evidence.alternative_evidence.publication.artifacts import (
    AlternativeEvidencePublicationError,
)
from alphalattice.evidence.alternative_evidence.publication.contracts import (
    AlternativeEvidenceAnalysisPublication,
)
from alphalattice.evidence.alternative_evidence.runtime.coverage import (
    coverage_unit_ids,
)
from alphalattice.interface.local_application.cli_contract import refusal_words
from alphalattice.interface.local_application.failure_codes import public_failure
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.oversight.chief_risk_officer.decision.book_evidence import (
    BookSelector,
    PortfolioEvidenceReviewError,
    assessment_schema,
    assessment_schema_hash,
    qualify_span_handle,
)
from alphalattice.oversight.chief_risk_officer.decision.portfolio_review import (
    PortfolioReviewDossier,
    finding_aliases,
    review_dispositions,
)
from alphalattice.oversight.chief_risk_officer.publication.portfolio_review import (
    PortfolioReviewView,
)

from .evidence_review_application import _CHECKS_ITEM, ReviewOutcome

SEALED_EXPORT_REUSE_SECONDS = 600
"""How long the export the Host last sealed answers its next pages before one is sealed again: a
page slices what this Host built and verified in full, and the bound keeps it from outliving new
evidence for long (V94)."""

if TYPE_CHECKING:
    from alphalattice.evidence.alternative_evidence.contracts import AlternativeEvidenceRequest
    from alphalattice.evidence.alternative_evidence.documents.contracts import (
        AlternativeEvidenceDocumentSet,
    )
    from alphalattice.evidence.alternative_evidence.retrieval.contracts import (
        AlternativeEvidenceResolvedSpan,
    )

    from .evidence_review_application import EvidenceReviewApplication


@dataclass(frozen=True, slots=True)
class _PublishedReadingPacket:
    """The sealed facts the packet read model consumes, from a publication's exact lineage.

    A historical read invents no research obligation or preparation. The existing read model
    reads only these four packet properties, each supplied by the verified publication.
    """

    lineage: AlternativeEvidenceAnalysisLineage

    @property
    def request(self) -> AlternativeEvidenceRequest:
        """The exact analysis request and its information cutoff."""
        return self.lineage.request

    @property
    def document_set(self) -> AlternativeEvidenceDocumentSet:
        """The source documents with their stated time precision."""
        return self.lineage.document_set

    @property
    def receipt(self) -> AlternativeEvidenceRetrievalAccessReceipt:
        """The access receipt that supplied the published analysis."""
        return self.lineage.access_receipt

    @property
    def spans(self) -> tuple[AlternativeEvidenceResolvedSpan, ...]:
        """Every exact span the published analysis delivered."""
        return self.lineage.cro_package.verified_spans


class EvidenceReviewDelivery:
    """Every bounded delivery the section hands out, over the application it serves."""

    def __init__(self, app: EvidenceReviewApplication) -> None:
        """Bind bounded evidence deliveries to the retained review application.

        Args:
            app: Deterministic evidence review application owner.
        """
        self.app = app

    def export_analysis_packet(
        self,
        *,
        selector: BookSelector | None,
        task_id: UUID,
        unit_id: str | None = None,
        delivery_part: int | None = None,
        delivery_budget_bytes: int | None = None,
        context_hash: str | None = None,
        evidence_detail: str | None = None,
        view_entity_id: str | None = None,
        view_topic: str | None = None,
        view_last_days: int | None = None,
        view_from: str | None = None,
        view_to: str | None = None,
    ) -> dict[str, object] | ReviewOutcome:
        """Deliver an exact prepared packet within its measured response byte budget.

        One prepared packet: a single preparation's, or one unit's of a run,
        delivered within a byte budget measured on the actual response.

        Two different continuations travel with a delivery and are never
        confused: `next_request` reads the next part of this same prepared
        evidence and reads no source; `continuation_request` asks the
        application for one more bounded source-reading session over the
        packet's pending scope, bound to the receipt and span set the packet
        sealed. `evidence_detail="session_windows"` delivers only the matter windows
        the packet's own reading session read -- what a continuation's
        consumer, holding every earlier part, has not yet received;
        `evidence_detail="topic_coverage"` delivers the issuer-topic ledger
        with every delivered bundle, and `evidence_detail="time_view"` the
        bundles accepted in a view interval ending at the evidence cutoff
        (the `view_*` filters), both read from the sealed artifacts alone.

        The packet must be one of this book's units at its own cutoff -- the
        same issuers, the same question -- or it is not this book's packet.
        The whole response is delivered when it fits the declared or default
        budget; otherwise complete spans are delivered in parts, in packet
        order, each part carrying the whole packet's schema, index and
        submission template and an executable request for the next part.
        Nothing is cut under its handle; a span that does not fit on its own
        is a typed refusal. `context_hash` binds a continuation to the packet
        it started from: a changed packet refuses the continuation.
        """
        chosen = self.app._review_selector(selector)
        if isinstance(chosen, ReviewOutcome):
            return chosen
        adapter = self.app.evidence_task_adapter
        if adapter is None:
            raise PortfolioEvidenceReviewError("product_host.evidence_task_adapter_absent")
        packet, context, resolved, units, expected = self.app._book_packet(
            chosen, task_id=task_id, unit_id=unit_id
        )
        if context_hash is not None and context_hash != context["analysis_context_hash"]:
            raise PortfolioEvidenceReviewError("alternative_evidence.delivery_continuation_stale")
        budget_bytes, budget_source = admit_delivery_budget(delivery_budget_bytes)
        selector_fields = chosen.request_fields()
        matters = packet.receipt.litigation_matters
        # The continuation request is executable as returned: only the
        # operation's own parameters, with the limit slots null on a first
        # reading (the consumer declares them; the operation refuses by name
        # until it does). What it would read and cost sits beside it. It is
        # offered exactly when the operation would read something: a pending
        # matter window, a sealed candidate or a table page the plan resumes.
        continuation_request: dict[str, object] | None = None
        if matters is not None and (
            matters.pending_windows or any(routed_pending_work(packet.receipt).values())
        ):
            receipt_hash, span_set_hash = adapter.prepared_receipt_identity(
                task_id, unit_id=unit_id
            )
            continuation_request = {
                "operation": "EVIDENCE_CONTINUE",
                **selector_fields,
                "task_id": str(task_id),
                **({} if unit_id is None else {"evidence_unit_id": unit_id}),
                "continuation_of": receipt_hash,
                "continuation_spans": span_set_hash,
                "session_limit": matters.session_limit,
                "window_limit": matters.window_limit,
            }
        constant: dict[str, object] = {
            **{key: value for key, value in context.items() if key != "packet"},
            "book": asdict(self.app._book_projection(resolved)),
            "coverage_unit": {
                "unit_id": expected[0],
                "unit_count": len(units),
                "ordered_entity_ids": list(packet.request.ordered_entity_ids),
            },
            "continuation_request": continuation_request,
            "continuation_scope": litigation_continuation_scope(packet.receipt),
            "submission_template": self.app._analysis_submission(
                chosen,
                task_id=task_id,
                unit_id=unit_id,
                context_hash=str(context["analysis_context_hash"]),
            ),
            "external_host_usage": "UNAVAILABLE",
            "claim": "EXTERNAL_FINDINGS_ARE_NOT_A_CRO_REVIEW_OR_TRADE_AUTHORITY",
        }
        if evidence_detail not in {
            None,
            "session_windows",
            "topic_coverage",
            "time_view",
        }:
            raise PortfolioEvidenceReviewError(
                f"alternative_evidence.evidence_detail_unknown:{evidence_detail}"
            )
        if evidence_detail in {"topic_coverage", "time_view"}:
            return self._deliver_evidence_view(
                packet,
                detail=evidence_detail,
                constant=constant,
                selector_fields=selector_fields,
                task_id=task_id,
                unit_id=unit_id,
                context_hash=str(context["analysis_context_hash"]),
                budget_bytes=budget_bytes,
                budget_source=budget_source,
                delivery_budget_bytes=delivery_budget_bytes,
                delivery_part=delivery_part,
                view_entity_id=view_entity_id,
                view_topic=view_topic,
                view_last_days=view_last_days,
                view_from=view_from,
                view_to=view_to,
            )
        handles = tuple(value.span_handle for value in packet.spans)
        session_windows = evidence_detail == "session_windows"
        if session_windows:
            # Only the matter windows the packet's own reading session read:
            # a continuation's consumer already holds every earlier part.
            if matters is None or not matters.span_handles:
                raise PortfolioEvidenceReviewError("alternative_evidence.session_windows_absent")
            own = set(matters.span_handles)
            handles = tuple(handle for handle in handles if handle in own)
        by_entity: dict[str, list[str]] = {}
        for value in packet.spans:
            by_entity.setdefault(value.entity_id, []).append(value.span_handle)

        def body_for(
            part_handles: tuple[str, ...], part: int, part_count: int
        ) -> dict[str, object]:
            whole, remaining = delivery_part_plan(handles, part_handles, part, part_count)
            whole = whole and not session_windows
            delivered = set(part_handles)
            return {
                "status": "EVIDENCE_ANALYST_PACKET_READY"
                if whole
                else "EVIDENCE_ANALYST_PACKET_PART",
                **constant,
                "packet": render_evidence_packet(
                    packet,
                    span_handles=None if whole else part_handles,
                    delivery_part=None if whole else (part, part_count),
                ),
                "delivery": delivery_facts(
                    budget_bytes=budget_bytes,
                    budget_source=budget_source,
                    mode="WHOLE_PACKET" if whole else "PARTS",
                    part=part,
                    part_count=part_count,
                    delivery_scope="SESSION_WINDOWS" if session_windows else "PACKET",
                    scope_span_count=len(handles),
                    delivered_span_handles=list(part_handles),
                    delivered_entity_ids=[
                        entity
                        for entity in packet.request.ordered_entity_ids
                        if by_entity.get(entity) and all(h in delivered for h in by_entity[entity])
                    ],
                    remaining_span_count=len(remaining),
                    remaining_span_handles=list(remaining),
                    next_request=next_part_request(
                        part,
                        part_count,
                        delivery_budget_bytes=delivery_budget_bytes,
                        operation="EVIDENCE_PACKET",
                        **selector_fields,
                        task_id=str(task_id),
                        **({} if unit_id is None else {"evidence_unit_id": unit_id}),
                        **({"evidence_detail": "session_windows"} if session_windows else {}),
                        analysis_context_hash=context["analysis_context_hash"],
                    ),
                    claim=(
                        "Bounds this response only; every part cites the same handles and "
                        "checks, and nothing is cut under its handle."
                    ),
                ),
            }

        return self._deliver(
            handles,
            body_for=body_for,
            budget_bytes=budget_bytes,
            delivery_part=delivery_part,
            owner="alternative_evidence",
        )

    def _deliver_evidence_view(
        self,
        packet: AlternativeEvidencePacket,
        *,
        detail: str,
        constant: dict[str, object],
        selector_fields: dict[str, str],
        task_id: UUID,
        unit_id: str | None,
        context_hash: str,
        budget_bytes: int,
        budget_source: str,
        delivery_budget_bytes: int | None,
        delivery_part: int | None,
        view_entity_id: str | None,
        view_topic: str | None,
        view_last_days: int | None,
        view_from: str | None,
        view_to: str | None,
    ) -> dict[str, object]:
        """The shared Analyst/UI read model of one prepared packet, delivered
        through the one delivery mechanism: the issuer-topic ledger with every
        bundle (`topic_coverage`), or the bundles of a resolved time interval
        with their boundary, unknown-time and historical-context lists and
        the unfiltered ledger (`time_view`). Filtering reads the sealed
        receipt, span set and document set: no acquisition, no embedding, no
        reranking, no interpretation; the bundles are the items paged."""

        try:
            if detail == "topic_coverage":
                if any(
                    value is not None
                    for value in (view_entity_id, view_topic, view_last_days, view_from, view_to)
                ):
                    raise PortfolioEvidenceReviewError(
                        "alternative_evidence.view_filters_not_applicable:topic_coverage"
                    )
                view: dict[str, object] = {
                    "coverage": topic_coverage_ledger(packet.receipt, packet.spans),
                    "bundles": list(evidence_bundles(packet)),
                }
                paged_keys: tuple[str, ...] = ("bundles",)
            else:
                view = time_view(
                    packet,
                    entity_id=view_entity_id,
                    topic=view_topic,
                    last_days=view_last_days,
                    view_from=None if view_from is None else datetime.fromisoformat(view_from),
                    view_to=None if view_to is None else datetime.fromisoformat(view_to),
                )
                paged_keys = ("bundles", "boundary", "unknown_time", "historical_context")
        except ValueError as error:
            raise PortfolioEvidenceReviewError(str(error)) from error
        if view.get("coverage") is None:
            raise PortfolioEvidenceReviewError(
                "alternative_evidence.topic_coverage_absent:the packet's selection did not route"
            )
        # The paged items: every bundle of every paged list, in view order,
        # keyed by the handle that names it (a span appears in one list).
        items: list[tuple[str, str, dict[str, object]]] = []
        for key in paged_keys:
            for bundle in cast(list[dict[str, object]], view[key]):
                items.append((f"{key}:{bundle['span_handle']}", key, bundle))
        by_item = {name: (key, bundle) for name, key, bundle in items}
        handles = tuple(name for name, _key, _bundle in items)
        status = "EVIDENCE_TOPIC_COVERAGE" if detail == "topic_coverage" else "EVIDENCE_TIME_VIEW"
        filters = {
            key: value
            for key, value in (
                ("view_entity_id", view_entity_id),
                ("view_topic", view_topic),
                ("view_last_days", view_last_days),
                ("view_from", view_from),
                ("view_to", view_to),
            )
            if value is not None
        }

        def body_for(
            part_handles: tuple[str, ...], part: int, part_count: int
        ) -> dict[str, object]:
            whole, remaining = delivery_part_plan(handles, part_handles, part, part_count)
            lists: dict[str, list[dict[str, object]]] = {key: [] for key in paged_keys}
            for name in part_handles:
                key, bundle = by_item[name]
                lists[key].append(bundle)
            return {
                "status": status,
                **constant,
                "evidence_view": {
                    **{key: value for key, value in view.items() if key not in paged_keys},
                    **lists,
                    "view_rule": (
                        "One authoritative read model of one sealed evidence set: source "
                        "observation (the excerpt and its document's stated times), "
                        "machine-derived metadata (topics, method, comparison, delivery) "
                        "and actor interpretation (recorded elsewhere) stay distinct; a "
                        "renderer infers no completion, clearance or severity from a "
                        "state, a missing row, a hit or a date."
                    ),
                },
                "delivery": delivery_facts(
                    budget_bytes=budget_bytes,
                    budget_source=budget_source,
                    mode="WHOLE_VIEW" if whole else "PARTS",
                    part=part,
                    part_count=part_count,
                    delivered_items=list(part_handles),
                    remaining_items=list(remaining),
                    model_work="NONE",
                    next_request=next_part_request(
                        part,
                        part_count,
                        delivery_budget_bytes=delivery_budget_bytes,
                        operation="EVIDENCE_PACKET",
                        **selector_fields,
                        task_id=str(task_id),
                        **({} if unit_id is None else {"evidence_unit_id": unit_id}),
                        evidence_detail=detail,
                        **filters,
                        analysis_context_hash=context_hash,
                    ),
                    claim="Bounds this response only; nothing is cut under its handle.",
                ),
            }

        return self._deliver(
            handles,
            body_for=body_for,
            budget_bytes=budget_bytes,
            delivery_part=delivery_part,
            owner="alternative_evidence",
        )

    def book_ledger(
        self, selector: BookSelector | None = None, *, page: int | None = None
    ) -> dict[str, object] | ReviewOutcome:
        """Read bounded issuer-topic groups from the book's exact sealed coverage packets.

        The book's issuer-topic ledger in one read, group by group, as the
        Reading map needs it: each prepared group's cells (issuer, topic,
        delivery state, incomplete reasons, gaps), its counts and its
        continuation state; a group without a readable packet named by its
        state; a group whose selection did not route refused by name. A page
        holds twenty groups -- sized by the groups, never by the packets.

        The groups and their packets are the section's: the book's newest
        coverage Task (as `coverage_progress` names them for a wider book).
        Each group is read from its sealed receipt and the span set that receipt
        delivered, proved as a reuse proves it (`SealedSelections.spans_of`);
        no excerpt is shown, and a group's packet read still verifies its
        source bytes. No acquisition, embedding, reranking or
        interpretation.
        """
        chosen = self.app._review_selector(selector)
        if isinstance(chosen, ReviewOutcome):
            return chosen
        adapter = self.app.evidence_task_adapter
        if adapter is None:
            raise PortfolioEvidenceReviewError("product_host.evidence_task_adapter_absent")
        resolved = self.app.resolve_book(chosen)
        scope = resolved.scope
        units = self.app._units(scope)
        ids = coverage_unit_ids(units)
        # (group id, issuers, packet Task, packet unit, state, failure code)
        groups: list[tuple[str, tuple[str, ...], str | None, str | None, str, str | None]] = []
        latest = next(self.app._run_tasks(scope), None)
        states: dict[tuple[str, ...], tuple[str, dict[str, object]]] = {}
        run_task: str | None = None
        if latest is not None:
            run_task = str(latest.task_id)
            states = {
                tuple(cast(tuple[str, ...], value["ordered_entity_ids"])): (unit_id, value)
                for unit_id, value in adapter.unit_states(latest).items()
            }
        for entities in units:
            found = states.get(entities)
            state = "NOT_STARTED" if found is None else str(found[1]["state"])
            groups.append(
                (
                    ids[entities],
                    entities,
                    run_task if state == "PREPARED" else None,
                    None if found is None else found[0],
                    state,
                    None if found is None else cast("str | None", found[1].get("failure_code")),
                )
            )
        per_page = 20
        page_count = max(1, math.ceil(len(groups) / per_page))
        number = 1 if page is None else page
        if not 1 <= number <= page_count:
            raise PortfolioEvidenceReviewError(
                f"alternative_evidence.ledger_page_out_of_range:{number} of {page_count}"
            )
        body: list[dict[str, object]] = []
        selector_fields = chosen.request_fields()

        def refuse_group(
            group: dict[str, object],
            *,
            code: str,
            detail: str,
            task_id: str | None,
        ) -> None:
            words = refusal_words(f"{code}:{group['group_id']}")
            group.update(
                status="REFUSED",
                failure_code=code,
                refusal=code,
                detail=detail or str(words.get("detail") or "This group could not be read."),
                next_action=words.get("next_action"),
                next_requests={
                    "ledger": {
                        "operation": "EVIDENCE_LEDGER",
                        **selector_fields,
                        "ledger_page": number,
                    },
                    **(
                        {}
                        if task_id is None
                        else {"task": {"operation": "TASK_RECOVERY", "task_id": task_id}}
                    ),
                    "storage": {"operation": "STORAGE_READBACK"},
                    "workspace": {"operation": "WORKSPACE_SHOW"},
                    "backups": {"operation": "WORKSPACE_BACKUPS"},
                },
            )

        def refuse_read_error(group: dict[str, object], error: BaseException, task_id: str) -> None:
            if isinstance(error, TaskNotFoundError):
                code = "task_control.task_not_found"
                detail = (
                    f"Group {group['group_id']} names Task {task_id}, but its Task record is "
                    "not present in this workspace. Recover the Task record before using the "
                    "group."
                )
            elif isinstance(error, FileNotFoundError):
                code = "alternative_evidence.artifact_missing"
                detail = (
                    f"Group {group['group_id']} on Task {task_id} names a sealed record that is "
                    "not held. Inspect stored files and workspace backups before relying on "
                    "this group."
                )
            elif isinstance(error, OSError):
                code = "alternative_evidence.artifact_unavailable"
                detail = (
                    f"Group {group['group_id']} on Task {task_id} could not read one of its "
                    "sealed records. Inspect stored files and workspace backups before relying "
                    "on this group."
                )
            else:
                code = public_failure(error, "alternative_evidence.artifact_tampered")
                detail = (
                    f"Group {group['group_id']} on Task {task_id} could not verify one of its "
                    "sealed records. Inspect stored files and workspace backups before relying "
                    "on this group."
                )
            refuse_group(group, code=code, detail=detail, task_id=task_id)

        for group_id, entities, task_id, unit_id, state, failure in groups[
            (number - 1) * per_page : number * per_page
        ]:
            group: dict[str, object] = {
                "group_id": group_id,
                "ordered_entity_ids": list(entities),
                "state": state,
                "failure_code": failure,
                "packet_task_id": task_id,
                "packet_unit_id": unit_id,
            }
            if task_id is not None:
                try:
                    receipt_hash, _span_set_hash = adapter.prepared_receipt_identity(
                        UUID(task_id), unit_id=unit_id
                    )
                except (TaskNotFoundError, OSError, ValueError) as error:
                    refuse_read_error(group, error, task_id)
                    body.append(group)
                    continue
                try:
                    receipt = adapter.runtime.artifacts.load(
                        "retrieval-access-receipts",
                        receipt_hash,
                        AlternativeEvidenceRetrievalAccessReceipt,
                    )
                except (
                    OSError,
                    AlternativeEvidencePublicationError,
                    ValidationError,
                ) as error:
                    refuse_read_error(group, error, task_id)
                    body.append(group)
                    continue
                try:
                    # The spans the receipt delivered, proved the way a reuse proves them:
                    # the set it names, of its request and generation.
                    spans = adapter.runtime.sealed_selections.spans_of(receipt)
                except (OSError, ValueError) as error:
                    # spans_of converts each present-but-invalid span set to a typed ValueError.
                    refuse_read_error(group, error, task_id)
                    body.append(group)
                    continue
                if spans is None:
                    refuse_group(
                        group,
                        code="alternative_evidence.artifact_missing",
                        detail=(
                            f"Group {group_id} names Task {task_id}, but its sealed span set is "
                            "not held. The group remains refused until the named records are "
                            "restored and read again."
                        ),
                        task_id=task_id,
                    )
                    body.append(group)
                    continue
                ledger = topic_coverage_ledger(receipt, spans)
                if ledger is None:
                    refuse_group(
                        group,
                        code="alternative_evidence.topic_coverage_absent",
                        detail=(
                            f"Group {group_id} on Task {task_id} has no topic coverage because "
                            "the packet's selection did not route."
                        ),
                        task_id=task_id,
                    )
                    body.append(group)
                    continue
                try:
                    read = self.export_analysis_packet(
                        selector=chosen,
                        task_id=UUID(task_id),
                        unit_id=unit_id,
                        evidence_detail="topic_coverage",
                        delivery_part=1,
                    )
                except (
                    TaskNotFoundError,
                    OSError,
                    AlternativeEvidencePublicationError,
                    ValidationError,
                ) as error:
                    refuse_read_error(group, error, task_id)
                    body.append(group)
                    continue
                if isinstance(read, ReviewOutcome):
                    group["packet_read"] = {
                        "disposition": read.disposition,
                        "failure_code": read.failure_code,
                    }
                    if read.disposition.startswith("REFUSED") or read.failure_code:
                        refuse_group(
                            group,
                            code=read.failure_code or "alternative_evidence.packet_read_refused",
                            detail=read.detail,
                            task_id=task_id,
                        )
                        body.append(group)
                        continue
                else:
                    for key in ("coverage_unit", "delivery", "continuation_request"):
                        group[key] = read.get(key)

                # Only attach derived cells after the selected receipt, span set and packet read
                # all succeeded. Calculation errors propagate as calculation errors.
                cells = cast(list[dict[str, object]], ledger["cells"])
                group["rules_id"] = ledger["rules_id"]
                group["cells"] = cells
                group["topics"] = ledger["topics"]
                counts: dict[str, int] = {}
                for cell in cells:
                    key = str(cell["delivery_state"])
                    counts[key] = counts.get(key, 0) + 1
                group["cells_by_state"] = counts
                group["continuation"] = litigation_continuation_scope(receipt)
            body.append(group)
        return {
            "status": "EVIDENCE_BOOK_LEDGER",
            "book": asdict(self.app._book_projection(resolved)),
            "groups": body,
            "page": number,
            "page_count": page_count,
            "groups_per_page": per_page,
            "total_groups": len(groups),
            "next_request": None
            if number >= page_count
            else {"operation": "EVIDENCE_LEDGER", **selector_fields, "ledger_page": number + 1},
            "claim": (
                "Derived annotations of the sealed receipts and span sets; no excerpt, "
                "no model work. A group's packet read verifies its source bytes."
            ),
        }

    def export_dossier(
        self,
        selector: BookSelector | None = None,
        *,
        delivery_part: int | None = None,
        delivery_budget_bytes: int | None = None,
        review_dossier_hash: str | None = None,
        review_read_at: datetime | None = None,
    ) -> dict[str, object] | ReviewOutcome:
        """Deliver exact CRO cognition inputs and answer contracts within a measured byte budget.

        Read-only, credential-independent exact cognition input and response
        contract, delivered within a byte budget measured on the actual response.

        A whole-book dossier that does not fit is delivered as parts over its
        findings, in dossier order, each with the citations those findings
        cite (uncited citations ride with part 1), and every part carrying the
        issuers, the children, the coverage, the schema and the submission
        template; `review_dossier_hash` binds a continuation to the dossier
        it started from, and `review_read_at` resolves it as read then (V255). Beside the
        sealed dossier: `evidence_sources` on
        every part, and each issuer's reported checks (`issuer_checks`) as
        items after the findings, each in exactly one part, so no part grows
        with the book.
        """
        chosen = self.app._review_selector(selector)
        if isinstance(chosen, ReviewOutcome):
            return chosen
        dossier, read_at = self.app._review_dossier_read(chosen, read_at=review_read_at)
        if isinstance(dossier, ReviewOutcome):
            return dossier
        # Bind the continuation to the cutoff the owner actually resolved from
        # its records, not to the instant this delivery happens to be read.
        assert read_at is not None
        if review_dossier_hash is not None and review_dossier_hash != dossier.dossier_hash:
            raise PortfolioEvidenceReviewError("chief_risk_officer.delivery_continuation_stale")
        budget_bytes, budget_source = admit_delivery_budget(delivery_budget_bytes)
        sources, issuer_checks = self._dossier_evidence_sources(dossier)
        schema = assessment_schema(dossier)
        policy_hash = self.app._decision_policy().binding_hash
        schema_hash = assessment_schema_hash(dossier)
        selector_fields = self.app._selector_for_dossier(dossier).request_fields()
        constant: dict[str, object] = {
            "decision_policy_hash": policy_hash,
            "assessment_schema": schema,
            "assessment_schema_hash": schema_hash,
            # The name an answer cites each finding by; the Host maps it back.
            "finding_aliases": finding_aliases(dossier),
            "submission_template": self.app._review_submission(dossier, read_at=read_at),
            "claim": "EXTERNAL_ASSESSMENT_IS_NOT_A_ROUTE_OR_PROTECTED_VALIDATION",
            "external_host_usage": "UNAVAILABLE",
            # Beside the sealed dossier, never inside it: its hash and every
            # review bound to it are unchanged by what is read here.
            "evidence_sources": sources,
        }
        full = dossier.model_dump(mode="json")
        handles = tuple(value.finding_handle for value in dossier.findings)
        finding_handles = frozenset(handles)
        # An issuer's checks are an item of the delivery after the findings
        # (a finding handle never starts with the prefix).
        items = (*handles, *(_CHECKS_ITEM + entity for entity in issuer_checks))
        cited_by: dict[str, str] = {}
        for finding in dossier.findings:
            for span in (*finding.supporting_span_handles, *finding.contradicting_span_handles):
                cited_by.setdefault(span, finding.finding_handle)

        def body_for(part_items: tuple[str, ...], part: int, part_count: int) -> dict[str, object]:
            whole, remaining = delivery_part_plan(items, part_items, part, part_count)
            part_handles = tuple(item for item in part_items if item in finding_handles)
            part_checks = {
                entity: issuer_checks[entity]
                for entity in (
                    item.removeprefix(_CHECKS_ITEM)
                    for item in part_items
                    if item not in finding_handles
                )
            }
            remaining_findings = [item for item in remaining if item in finding_handles]
            delivered = set(part_handles)
            if whole:
                dossier_body: dict[str, object] = full
            else:
                dossier_body = {
                    **full,
                    "findings": [
                        value for value in full["findings"] if value["finding_handle"] in delivered
                    ],
                    "citations": [
                        value
                        for value in full["citations"]
                        if cited_by.get(value["span_handle"]) in delivered
                        or (part == 1 and value["span_handle"] not in cited_by)
                    ],
                    "unresolved_questions": full.get("unresolved_questions", [])
                    if part == 1
                    else [],
                    "dossier_partial": {
                        "part": part,
                        "part_count": part_count,
                        "finding_handles": list(part_handles),
                        "issuer_check_entities": list(part_checks),
                        "rule": (
                            "The sealed dossier's hash covers every part; answer every "
                            "qualified finding after reading every part. Each issuer's "
                            "reported checks are in exactly one part."
                        ),
                    },
                }
            return {
                "status": "CRO_DOSSIER_READY" if whole else "CRO_DOSSIER_PART",
                "dossier": dossier_body,
                **constant,
                "issuer_checks": part_checks,
                "delivery": delivery_facts(
                    budget_bytes=budget_bytes,
                    budget_source=budget_source,
                    mode="WHOLE_DOSSIER" if whole else "PARTS",
                    part=part,
                    part_count=part_count,
                    delivered_finding_handles=list(part_handles),
                    remaining_finding_count=len(remaining_findings),
                    remaining_finding_handles=remaining_findings,
                    remaining_issuer_check_count=len(remaining) - len(remaining_findings),
                    next_request=next_part_request(
                        part,
                        part_count,
                        delivery_budget_bytes=delivery_budget_bytes,
                        operation="CRO_REVIEW_DOSSIER",
                        **selector_fields,
                        review_dossier_hash=dossier.dossier_hash,
                        review_read_at=read_at.isoformat(),
                    ),
                    claim=(
                        "Bounds this response only; every part answers to the same dossier "
                        "hash and schema, and no finding or citation is cut."
                    ),
                ),
            }

        return self._deliver(
            items,
            body_for=body_for,
            budget_bytes=budget_bytes,
            delivery_part=delivery_part,
            owner="chief_risk_officer",
        )

    def review_standing(self, selector: BookSelector | None) -> dict[str, object]:
        """Read the exact book's latest published review for an activation decision (V614).

        Read sealed review and analysis facts only: no current eligibility, dossier compiler
        or review actor. Preparation covers only analysis-linked packets with an exact
        preparation receipt; the publication carries no whole-book prepared-only snapshot.
        Missing review is explicit; a broken published lineage still refuses its read.
        """
        empty = {
            "published_at": None,
            "evidence_as_of": None,
            "coverage": None,
            "cro": None,
        }
        if selector is None:
            return {
                **empty,
                "status": "UNAVAILABLE",
                "detail": "No sealed book is named for this review standing.",
            }
        # Scope compilation needs a current issuer authority; historical standing does not.
        book = self.app._open_book(self.app.inputs(), selector)
        view = self.app.published_review_for_book(
            book, result_hash=book.result_hash, use_selected_analysis=False
        )
        if view is None:
            return {
                **empty,
                "status": "NOT_REVIEWED",
                "detail": "This book has not been reviewed.",
                "book_selector": selector.request_fields(),
            }
        if view.publication.result_hash != book.result_hash:
            raise PortfolioEvidenceReviewError(
                "product_host.evidence_review_export_subject_mismatch"
            )
        dossier, recommendation = view.dossier, view.recommendation
        for publication_hash in dossier.evidence_publication_hashes:
            self.app._require_evidence_publications().verify(publication_hash)
        sources, checks = self._dossier_evidence_sources(dossier)
        states: dict[str, int] = {}
        for check in checks.values():
            state = str(check["state"])
            states[state] = states.get(state, 0) + 1
        coverage = dossier.coverage.model_dump(mode="json")
        return {
            "status": "REVIEWED",
            "detail": "This book has a published Evidence and CRO review.",
            "book_selector": selector.request_fields(),
            "review_publication_hash": view.publication.publication_hash,
            "published_at": view.publication.published_at.isoformat(),
            "evidence_as_of": dossier.evidence_as_of.isoformat(),
            "evidence_expires_at": dossier.evidence_expires_at.isoformat(),
            "under_installed_policy": view.under_installed_policy,
            "coverage": {
                "prepared": {
                    "scope": "ANALYSIS_LINKED_PACKETS",
                    "packets": len(sources),
                    "with_preparation_receipt": sum(
                        source["prepared_task_id"] is not None for source in sources
                    ),
                    "preparation_unknown": sum(
                        source["prepared_task_id"] is None for source in sources
                    ),
                    "detail": (
                        "Preparation coverage counts only this review's analysis-linked "
                        "packets with a recorded preparation receipt; whole-book "
                        "prepared-only coverage was not recorded."
                    ),
                },
                "analyzed": {
                    "publications": len(sources),
                    "scope_issuers": len(dossier.issuers),
                    "issuer_states": {
                        state: sum(issuer.review_state == state for issuer in dossier.issuers)
                        for state in sorted({issuer.review_state for issuer in dossier.issuers})
                    },
                    "check_states": states,
                    "completion_schema": dossier.completion_schema,
                },
                "reviewed": {
                    **{
                        key: value
                        for key, value in coverage.items()
                        if key not in {"missing_evidence", "unavailable_reasons"}
                    },
                    "missing_evidence_count": len(dossier.coverage.missing_evidence),
                    "unavailable_reason_count": len(dossier.coverage.unavailable_reasons),
                },
            },
            "cro": {
                "route": recommendation.route.value,
                "review_state": recommendation.review_state.value,
                "reasons": list(recommendation.reasons),
            },
            "next_requests": {
                "review": {
                    "operation": "EVIDENCE_CRO_EXPORT",
                    **selector.request_fields(),
                    "review_publication_hash": view.publication.publication_hash,
                }
            },
        }

    def _dossier_evidence_sources(
        self, dossier: PortfolioReviewDossier
    ) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
        """For each evidence publication the dossier names: the preparation
        Task (and unit) that sealed the receipt its analysis answered, and
        the brief's completion schema and required checks; and for each
        issuer, the checks as that analysis reported them -- executed,
        deferred with the actor's note, unreported -- under that schema
        (under v1, `executed` meant every required check where spans were
        delivered). Read from the sealed publication, the sealed brief and
        the Task records."""

        store = self.app._require_evidence_publications().store
        units = {
            child.analysis_publication_hash: child.unit_id for child in dossier.evidence_children
        }
        preparing = (
            {}
            if self.app.evidence_task_adapter is None
            else self.app.evidence_task_adapter.prepared_receipts()
        )
        sources: list[dict[str, object]] = []
        checks: dict[str, dict[str, object]] = {}
        for publication_hash in dossier.evidence_publication_hashes:
            publication = store.load(
                "analysis-publications", publication_hash, AlternativeEvidenceAnalysisPublication
            )
            brief = store.load(
                "analyst-briefs", publication.analyst_brief_hash, AlternativeEvidenceAnalystBrief
            )
            task = preparing.get(publication.access_receipt_hash)
            completion = brief.completion
            sources.append(
                {
                    "analysis_publication_hash": publication_hash,
                    "unit_id": units.get(publication_hash),
                    "access_receipt_hash": publication.access_receipt_hash,
                    "prepared_task_id": None if task is None else task[0],
                    "prepared_unit_id": None if task is None else task[1],
                    "completion_schema": None
                    if completion is None
                    else completion.completion_schema,
                    "required_checks": []
                    if completion is None
                    else list(completion.required_checks),
                }
            )
            if completion is None:
                continue
            for outcome in completion.issuers:
                checks[outcome.entity_id] = {
                    "state": outcome.state.value,
                    "executed": list(outcome.checks_executed),
                    "deferred": [
                        {"check": value.check, "note": value.note}
                        for value in outcome.checks_deferred
                    ],
                    "unreported": list(outcome.checks_unreported),
                    "analysis_publication_hash": publication_hash,
                }
        return sources, checks

    def export_finding_evidence(
        self,
        selector: BookSelector | None = None,
        *,
        finding_handle: str,
        review_dossier_hash: str | None = None,
        review_publication_hash: str | None = None,
        review_read_at: datetime | None = None,
    ) -> dict[str, object] | ReviewOutcome:
        """Deliver one exact finding and its verified cited evidence as a bounded package.

        One finding as a bounded evidence package: its atomic claim as the
        analyst stated it, the exact verified excerpts it cites for and
        against, each with its document's type, revision and availability, the
        question that found each passage (and its kind), the same-filing
        passages a read span stood for, the unit's open questions, the
        reviewer's disposition when a published review is named, and the
        qualified references to the producing publication.

        Read-only. `review_dossier_hash` binds the read to the dossier the
        consumer is reviewing (a changed dossier refuses); with
        `review_publication_hash` the finding is read from that sealed review
        instead of the current dossier, so a historical disposition is read
        back exactly. Nothing here is a judgment: the package is what the
        product can prove was delivered and what was submitted about it.
        """
        chosen = self.app._review_selector(selector)
        if isinstance(chosen, ReviewOutcome):
            return chosen
        disposition: dict[str, object] | None = None
        if review_publication_hash is not None:
            view = self.app.review_publications.read(review_publication_hash)
            dossier = view.dossier
            for value in review_dispositions(
                view.receipt.submission,
                answered=view.receipt.answer is not None,
                finding_handles=tuple(f.finding_handle for f in dossier.findings),
            ):
                if value.finding_handle == finding_handle:
                    disposition = value.model_dump(mode="json")
        else:
            resolved_dossier = self.app._resolve_review_dossier(chosen, read_at=review_read_at)
            if isinstance(resolved_dossier, ReviewOutcome):
                return resolved_dossier
            dossier = resolved_dossier
        if review_dossier_hash is not None and review_dossier_hash != dossier.dossier_hash:
            raise PortfolioEvidenceReviewError("chief_risk_officer.delivery_continuation_stale")
        finding = next(
            (value for value in dossier.findings if value.finding_handle == finding_handle), None
        )
        if finding is None:
            raise PortfolioEvidenceReviewError("chief_risk_officer.finding_handle_unknown")
        service = self.app._require_evidence_publications()
        now = self.app.clock()
        children = dossier.evidence_children
        # The child publication the finding came from: the qualifier in an
        # aggregate dossier, the one publication otherwise.
        publications = dossier.evidence_publication_hashes
        if children:
            qualifier = finding.finding_handle.split("-")[1]
            publication_hash = next(
                (value for value in publications if f"P{value[:8].upper()}" == qualifier), None
            )
            if publication_hash is None:
                raise PortfolioEvidenceReviewError("chief_risk_officer.finding_handle_unknown")
            unit_id = next(
                (c.unit_id for c in children if c.analysis_publication_hash == publication_hash),
                None,
            )
        else:
            publication_hash = publications[0]
            unit_id = None
        replayed = service.replay(publication_hash, now=now)
        support, contrary = finding_excerpts(
            replayed,
            finding.supporting_span_handles,
            finding.contradicting_span_handles,
            qualified=bool(children),
        )
        questions = [
            q.text for q in dossier.unresolved_questions if unit_id is None or q.unit_id == unit_id
        ]
        return {
            "status": "FINDING_EVIDENCE_PACKAGE",
            "review_dossier_hash": dossier.dossier_hash,
            "finding": finding.model_dump(mode="json"),
            "atomic_claim": finding.summary,
            "affected_entities": list(finding.affected_entities),
            "event_state": {
                "topic": str(finding.topic),
                "lifecycle": str(finding.lifecycle),
                "direction": str(finding.direction),
                "structure": str(finding.structure),
            },
            "support": support,
            "contrary": contrary,
            "unresolved_questions": questions,
            "disposition": disposition,
            "producing_publication": {
                "unit_id": unit_id,
                "analysis_publication_hash": publication_hash,
                "evidence_as_of": replayed.lineage.request.evidence_as_of.isoformat(),
                "published_at": replayed.publication.published_at.isoformat(),
                "expires_at": replayed.publication.expires_at.isoformat(),
                "document_count": len(replayed.lineage.document_set.documents),
                "delivered_span_count": len(replayed.lineage.cro_package.verified_spans),
            },
            "claim": (
                "Excerpts are verified source text delivered to the analyst; the claim is the "
                "analyst's; the disposition is the reviewer's; nothing here is adjudicated "
                "by the product."
            ),
        }

    def export_review(
        self,
        selector: BookSelector | None,
        publication_hash: str | None,
        *,
        prior_publication_hash: str | None = None,
    ) -> dict[str, object]:
        """Export explicitly selected immutable facts, never a newly chosen review."""
        return self.verified_export(
            selector, publication_hash, prior_publication_hash=prior_publication_hash
        )[0]

    def export_page_source(
        self,
        selector: BookSelector | None,
        publication_hash: str | None,
        *,
        prior_publication_hash: str | None = None,
    ) -> tuple[dict[str, object], str]:
        """The export a citation page slices, and how it was verified (V94).

        A page names the book and the review; the export last sealed for them (its
        `export_hash`) answers each page, built and verified in full by the first request that
        needs it -- a whole export or a first page -- and kept `SEALED_EXPORT_REUSE_SECONDS`, the
        Host's own export in its own memory, not a file it trusts. The Host keeps the last
        export it sealed and no other, so no cache size is set here. The basis says which:
        `FULL`, or `SEALED_EXPORT <time>`.
        """
        key = self._export_key(selector, publication_hash, prior_publication_hash)
        with self.app._sealed_export_lock:
            kept = self.app._sealed_export
        if (
            kept is not None
            and kept[0] == key
            and (self.app.clock() - kept[1]).total_seconds() <= SEALED_EXPORT_REUSE_SECONDS
        ):
            return kept[2], f"SEALED_EXPORT {kept[1].isoformat()}"
        exported = self.export_review(
            selector, publication_hash, prior_publication_hash=prior_publication_hash
        )
        return exported, "FULL"

    def _export_key(
        self,
        selector: BookSelector | None,
        publication_hash: str | None,
        prior_publication_hash: str | None,
    ) -> str:
        chosen = self.app.default_selector(selector)
        return str(
            canonical_hash(
                {
                    "book": None if chosen is None else chosen.request_fields(),
                    "review": publication_hash,
                    "prior": prior_publication_hash,
                }
            )
        )

    def _seal(self, key: str, exported: dict[str, object]) -> None:
        # What a page reads: never the portfolio or the rendered HTML.
        kept = {k: v for k, v in exported.items() if k not in {"portfolio", "html"}}
        with self.app._sealed_export_lock:
            self.app._sealed_export = (key, self.app.clock(), kept)

    def verified_export(
        self,
        selector: BookSelector | None,
        publication_hash: str | None,
        *,
        prior_publication_hash: str | None = None,
    ) -> tuple[dict[str, object], PortfolioReviewView | None]:
        """The export and the review view it verified.

        A caller reads on the view rather than opening the review again (V151). Each export
        verified in full is sealed for its citation pages (V94).
        """
        exported, view, _analyses = self.verified_read_view(
            selector, publication_hash, prior_publication_hash=prior_publication_hash
        )
        return exported, view

    def published_reading(
        self,
        view: PortfolioReviewView,
        *,
        verified: tuple[AlternativeEvidenceAnalysisPublicationView, ...] = (),
        entity_id: str | None = None,
        topic: str | None = None,
        last_days: int | None = None,
    ) -> dict[str, object]:
        """An unsealed reading of exactly the analyses a review read, under its view filters.

        Reuse an analysis already verified for this request only by its publication hash;
        carried analyses absent from that read are replayed by their sealed references.
        No current preparation receipt or actor finding supplies a passage's membership.
        """
        known = {value.publication.publication_hash: value for value in verified}
        service = self.app._require_evidence_publications()
        now = self.app.clock()
        children = {
            child.analysis_publication_hash: child.unit_id
            for child in view.dossier.evidence_children
        }
        lists: dict[str, list[dict[str, object]]] = {
            key: [] for key in ("bundles", "boundary", "unknown_time", "historical_context")
        }
        analyses: list[dict[str, object]] = []
        delivered_total = selected = outside = 0
        for publication in view.dossier.evidence_publication_hashes:
            value = (
                known[publication] if publication in known else service.replay(publication, now=now)
            )
            lineage = value.lineage
            if last_days is None:
                # No interval was requested: every delivered passage remains readable, including
                # older, date-only and unknown-time sources, with no implicit thirty-day cut.
                bundles = bundles_of(
                    lineage.access_receipt,
                    {d.semantic_handle: d for d in lineage.document_set.documents},
                    lineage.cro_package.verified_spans,
                )
                chosen = [
                    bundle
                    for bundle in bundles
                    if (entity_id is None or bundle["entity_id"] == entity_id)
                    and (topic is None or topic in bundle["topics"])
                ]
                reading: dict[str, Any] = {
                    "bundles": [{**bundle, "placement": "DELIVERED"} for bundle in chosen],
                    "boundary": [],
                    "unknown_time": [],
                    "historical_context": [],
                    "delivered_total": len(bundles),
                    "selected": len(chosen),
                    "outside_interval": 0,
                    "interval": None,
                }
            else:
                # time_view consumes only request, document_set, receipt and spans. This adapter
                # supplies those exact facts without manufacturing a preparation or obligation.
                reading = time_view(
                    cast(AlternativeEvidencePacket, _PublishedReadingPacket(lineage)),
                    entity_id=entity_id,
                    topic=topic,
                    last_days=last_days,
                )
            unit_id = children.get(publication)
            analyses.append(
                {
                    "analysis_publication_hash": publication,
                    "unit_id": unit_id,
                    "interval": reading["interval"],
                    "evidence_cutoff": lineage.request.evidence_as_of.isoformat(),
                }
            )
            for key in lists:
                for bundle in reading[key]:
                    projected = {**bundle, "analysis_publication_hash": publication}
                    if children:
                        projected["span_handle"] = qualify_span_handle(
                            bundle["span_handle"], publication_hash=publication
                        )
                        projected["unit_id"] = unit_id
                        if bundle.get("context_for"):
                            projected["context_for"] = qualify_span_handle(
                                bundle["context_for"], publication_hash=publication
                            )
                    lists[key].append(projected)
            delivered_total += reading["delivered_total"]
            selected += reading["selected"]
            outside += reading["outside_interval"]
        return {
            "review_publication_hash": view.publication.publication_hash,
            "filters": {"entity_id": entity_id, "topic": topic, "last_days": last_days},
            "analyses": analyses,
            **lists,
            "delivered_total": delivered_total,
            "selected": selected,
            "outside_interval": outside,
            "claim": (
                "A view of the published review's exact delivered evidence; filtering changes "
                "no scope, completes no check and leaves the immutable export whole."
            ),
        }

    def verified_read_view(
        self,
        selector: BookSelector | None,
        publication_hash: str | None,
        *,
        prior_publication_hash: str | None = None,
    ) -> tuple[
        dict[str, object],
        PortfolioReviewView | None,
        tuple[AlternativeEvidenceAnalysisPublicationView, ...],
    ]:
        """The immutable export, its review and the exact analyses its read view consumes.

        The analyses supply a separate requested read view only. Its annotations never enter
        the export's bytes, hash or sealed citation-page source, and share its verification.
        """
        from alphalattice.interface.local_application.evidence_cro import render_review_export
        from alphalattice.kernel.shared_kernel.identity import canonical_hash

        chosen = self.app.default_selector(selector)
        if chosen is None:
            raise PortfolioEvidenceReviewError("product_host.evidence_review_book_unselected")
        book = self.app._open_book(self.app.inputs(), chosen)
        view = (
            None
            if publication_hash is None
            else self.app.review_publications.read(publication_hash)
        )
        # Freeze changes the authority for new reviews, not the identity of the
        # already sealed result/report an explicitly selected old review named.
        historical_result = (
            view is not None
            and book.result_hash is not None
            and view.publication.result_hash == book.result_hash
            and book.authority in {"DEVELOPMENT_RESULT", "FROZEN_CANDIDATE"}
            and view.publication.book_authority in {"DEVELOPMENT_RESULT", "FROZEN_CANDIDATE"}
        )
        if view is not None and (
            view.publication.report_hash != book.report_hash
            or (view.publication.book_authority is not book.authority and not historical_result)
            or (
                view.publication.result_hash is not None
                and view.publication.result_hash != book.result_hash
            )
            or view.publication.update_subject != book.update_subject
            or view.publication.experiment_subject != book.experiment_subject
        ):
            raise PortfolioEvidenceReviewError(
                "product_host.evidence_review_export_subject_mismatch"
            )
        if book.experiment_subject is not None:
            from alphalattice.interface.local_application.experiment_report import (
                render_experiment_report,
            )

            assert self.app.read_experiment is not None
            experiment_subject = book.experiment_subject
            portfolio = self.app.read_experiment(
                experiment_subject.experiment_task_id,
                experiment_subject.portfolio_session.isoformat(),
            )
            base_html: str | None = render_experiment_report(portfolio)
        elif book.update_subject is not None:
            assert self.app.read_update is not None
            subject = book.update_subject
            portfolio = self.app.read_update(
                subject.update_task_id, subject.update_publication_hash
            )
            base_html = cast(str | None, portfolio.get("html"))
        else:
            assert book.report is not None
            portfolio = {"report": book.report.model_dump(mode="json")}
            base_html = None
        evidence = None
        review = None
        analyses: tuple[AlternativeEvidenceAnalysisPublicationView, ...] = ()
        if view is not None:
            # One publication for the ordinary dossier; one per unit for a book
            # analysed as several, each replayed exactly and listed under its
            # unit so a qualified span handle in the review resolves to its
            # own publication.
            service = self.app._require_evidence_publications()
            now = self.app.clock()
            children = view.dossier.evidence_children
            replayed = [
                (
                    None if not children else children[index].unit_id,
                    service.replay(publication_hash, now=now),
                )
                for index, publication_hash in enumerate(view.dossier.evidence_publication_hashes)
            ]
            analyses = tuple(value for _unit, value in replayed)

            # Each verified span carries its document's temporal view: the
            # export is read away from the packet, so the document index it
            # would otherwise resolve through travels compactly with the span.
            def timed(value: Any) -> dict[str, dict[str, object]]:
                documents = value.lineage.document_set.documents
                return {d.semantic_handle: d.temporal_view() for d in documents}

            if not children:
                verified = replayed[0][1]
                times = timed(verified)
                evidence = {
                    "publication": verified.publication.model_dump(mode="json"),
                    "verified_spans": [
                        {**v.model_dump(mode="json"), "time": times.get(v.document_handle)}
                        for v in verified.lineage.cro_package.verified_spans
                    ],
                }
            else:
                evidence = {
                    "aggregate_publication_hash": view.dossier.analysis_publication_hash,
                    "children": [
                        {
                            "unit_id": unit_id,
                            "publication": value.publication.model_dump(mode="json"),
                        }
                        for unit_id, value in replayed
                    ],
                    # Every child's spans under the handle the dossier cites
                    # them by, so a citation resolves to one excerpt.
                    "verified_spans": [
                        {
                            **span.model_dump(mode="json"),
                            "span_handle": qualify_span_handle(
                                span.span_handle,
                                publication_hash=value.publication.publication_hash,
                            ),
                            "unit_id": unit_id,
                            "time": timed(value).get(span.document_handle),
                        }
                        for unit_id, value in replayed
                        for span in value.lineage.cro_package.verified_spans
                    ],
                }
            review = {
                "publication": view.publication.model_dump(mode="json"),
                "dossier": view.dossier.model_dump(mode="json"),
                "recommendation": view.recommendation.model_dump(mode="json"),
                "receipt": view.receipt.model_dump(mode="json"),
            }
        changes = (
            None
            if publication_hash is None or prior_publication_hash is None
            else self.app.review_changes(publication_hash, prior_publication_hash)
        )
        snapshot: dict[str, object] = {
            "kind": "PortfolioEvidenceReviewExport",
            "book_authority": str(
                book.authority if view is None else view.publication.book_authority
            ),
            "update_subject": None
            if book.update_subject is None
            else book.update_subject.model_dump(mode="json"),
            "report_hash": book.report_hash,
            **(
                {}
                if book.experiment_subject is None
                else {"experiment_subject": book.experiment_subject.model_dump(mode="json")}
            ),
            "portfolio": portfolio,
            "review": review,
            "evidence": evidence,
            "changes_since_prior_review": changes,
            "review_status": "NOT_REVIEWED" if view is None else "EXACT_HISTORICAL_READBACK",
            # Only when false, so every export under the installed policy keeps its hash.
            **(
                {}
                if view is None or view.under_installed_policy
                else {"review_under_installed_policy": False}
            ),
            "claim": "Readback grants no current evidence eligibility, forward advice, "
            "protected validation or trade authority.",
        }
        # Dynamic task status is not part of the immutable exported portfolio.
        if book.update_subject is not None:
            snapshot["portfolio"] = {
                k: portfolio[k]
                for k in ("publication", "history", "listing_labels", "strategy_package_id")
            }
        rendered = render_review_export(snapshot, base_html)
        exported = {
            **snapshot,
            "html": rendered,
            "export_hash": str(canonical_hash({**snapshot, "html": rendered})),
        }
        self._seal(self._export_key(selector, publication_hash, prior_publication_hash), exported)
        return exported, view, analyses

    def _deliver(
        self,
        items: tuple[str, ...],
        *,
        body_for: Callable[[tuple[str, ...], int, int], dict[str, object]],
        budget_bytes: int,
        delivery_part: int | None,
        owner: str,
    ) -> dict[str, object]:
        """One delivery mechanism for both consumer-facing inputs: the whole
        response when it fits, else the requested part of a plan over complete
        items, each part measured as the response it is."""

        allowance = DELIVERY_ENVELOPE_ALLOWANCE_BYTES

        def measured(body: dict[str, object]) -> dict[str, object]:
            # The recorded count is part of the body it counts: iterate until
            # the number's own digits are included (two passes settle it).
            delivery = dict(cast(dict[str, object], body["delivery"]))
            count = 0
            for _ in range(4):
                candidate = {**body, "delivery": {**delivery, "response_bytes": count}}
                measured_bytes = serialized_response_bytes(candidate)
                if measured_bytes == count:
                    return candidate
                count = measured_bytes
            raise PortfolioEvidenceReviewError("alternative_evidence.delivery_measurement_unstable")

        def byte_length(part_items: tuple[str, ...], part: int, part_count: int) -> int:
            return serialized_response_bytes(body_for(part_items, part, part_count)) + allowance

        whole = body_for(items, 1, 1)
        if serialized_response_bytes(whole) + allowance <= budget_bytes:
            if delivery_part not in (None, 1):
                raise PortfolioEvidenceReviewError(f"{owner}.delivery_part_unknown:{delivery_part}")
            return measured(whole)
        # A part is measured with a continuation in place (the last part
        # measures slightly large, never small); part numbers are unknown
        # until the plan exists, so a plan-time part is measured as part 1 of 2.
        try:
            parts = plan_delivery(
                items,
                fits=lambda part_items: byte_length(part_items, 1, 2) <= budget_bytes,
                byte_length=lambda part_items: byte_length(part_items, 1, 2),
                budget_bytes=budget_bytes,
            )
        except DeliveryBudgetBelowMinimumUnit as error:
            raise PortfolioEvidenceReviewError(str(error)) from error
        index = 1 if delivery_part is None else delivery_part
        if not 1 <= index <= len(parts):
            raise PortfolioEvidenceReviewError(
                f"{owner}.delivery_part_unknown:{index} of {len(parts)}"
            )
        return measured(body_for(parts[index - 1], index, len(parts)))
