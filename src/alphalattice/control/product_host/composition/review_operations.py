"""Actor-neutral review operations, loaded on request."""

from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime
from functools import partial
from typing import TYPE_CHECKING, Any, cast

from alphalattice.capabilities.data.errors import DataError
from alphalattice.control.product_host.composition import committee
from alphalattice.control.product_host.composition.evidence_review_application import ReviewOutcome
from alphalattice.control.product_host.composition.evidence_review_bundles import (
    PACKET_SELECTOR_CODES,
    EvidenceReviewBundles,
    agent_bundle_refusal,
)
from alphalattice.control.product_host.composition.evidence_review_delivery import (
    EvidenceReviewDelivery,
)
from alphalattice.control.product_host.composition.host_answer_submission import (
    submit_agent_answer,
    submit_evidence_analysis,
)
from alphalattice.control.product_host.composition.plain_refusals import refused, unit_refusal
from alphalattice.control.product_host.composition.portfolio_research_operations import (
    _citation_page,
    _pages_citations,
    _read_at,
    _selector,
)
from alphalattice.control.product_host.composition.research_review_outcome import (
    _review_outcome_body,
)
from alphalattice.control.product_host.composition.review_updates import bundle_update
from alphalattice.interface.local_application.failure_codes import public_failure
from alphalattice.oversight.chief_risk_officer.decision.book_evidence import (
    BookSelector,
    PortfolioEvidenceReviewError,
)
from alphalattice.protocols.actor_execution.bundles import AgentRole

if TYPE_CHECKING:
    from alphalattice.interface.local_application import portfolio_research as requests
    from alphalattice.protocols.actor_execution.answers import AgentRun
    from alphalattice.protocols.actor_execution.bundles import AgentBundleRecord

    from . import portfolio_research_operations as host
    from .operation_route_families import WorkspaceOperationCall
    from .portfolio_research_operations import PortfolioResearchOperations


def execute(
    operations: host.PortfolioResearchOperations,
    request: requests.PortfolioResearchOperationRequest,
    *,
    caller: requests.OperationCaller,
    read_files: tuple[str, ...] | None,
    agent_run: AgentRun | None,
    accepted_bundle: AgentBundleRecord | None,
) -> dict[str, object]:
    """Dispatch an Evidence or committee review operation."""
    if operations.review is None:
        raise ValueError("product_host.evidence_review_not_configured")
    selector = _selector(request)
    match request.operation:
        case "AGENT_BUNDLE_PREPARE":
            assert request.agent_role is not None and request.bundle_directory is not None
            # A role's view of the date's positions, its floor or review route.
            dated = None
            if request.task_id is not None:
                with bundle_update(request.task_id):
                    dated = operations._dated(str(request.task_id))
            now, store, file = (
                operations.dispatcher.clock(),
                operations.goals.store,
                operations._file_floor,
            )
            parts = committee.bundle_parts(
                request.agent_role, dated, store, now, request.bundle_directory
            )
            try:
                prepared = EvidenceReviewBundles(operations.review).prepare_agent_bundle(
                    role=request.agent_role,
                    selector=selector,
                    directory=request.bundle_directory,
                    task_id=request.task_id,
                    unit_id=request.evidence_unit_id,
                    dispatcher=operations.dispatcher,
                    view=parts[0],
                    route=parts[1],
                    issue_view=lambda: operations.data_issues.readback(
                        limit=50, delegated_by=operations._data_decisions(caller)
                    ),
                )
            except (PortfolioEvidenceReviewError, ValueError) as error:
                refusal = _review_bundle_refusal(error, request.agent_role, selector)
                if refusal is None:
                    raise
                return refusal
            return committee.issued(store, _review_outcome_body(prepared), now, file)
        case "AGENT_ANSWER_SUBMIT":
            return submit_agent_answer(operations, request, caller)
        case "EVIDENCE_CRO":
            section = operations.evidence_cro(
                selector,
                request.review_publication_hash,
                evidence_detail=request.evidence_detail,
                view_entity_id=request.view_entity_id,
                view_topic=request.view_topic,
                view_last_days=request.view_last_days,
            )
            if not _pages_citations(request):  # an agent's dated review offers its committee
                return committee.offer(operations, section, selector, caller)
            progress = cast("dict[str, object] | None", section.get("coverage_progress"))
            return {
                "status": "EVIDENCE_CRO_CITATIONS_PAGE",
                "book": section.get("book"),
                "review_publication_hash": section.get("review_publication_hash"),
                **_citation_page(
                    cast(list[dict[str, object]], section.get("citations") or []),
                    request,
                    groups={
                        str(unit["unit_id"]): tuple(cast(list[str], unit["entity_ids"]))
                        for unit in cast(
                            list[dict[str, object]], (progress or {}).get("units") or []
                        )
                    },
                    key="citations",
                ),
            }
        case "EVIDENCE_CRO_EXPORT":
            delivery = EvidenceReviewDelivery(operations.review)
            if not _pages_citations(request):
                return delivery.export_review(
                    selector,
                    request.review_publication_hash,
                    prior_publication_hash=request.prior_review_publication_hash,
                )
            # A page slices the export sealed for this book and review.
            exported, basis = delivery.export_page_source(
                selector,
                request.review_publication_hash,
                prior_publication_hash=request.prior_review_publication_hash,
            )
            exported_dossier = cast(
                dict[str, object], cast(dict[str, object], exported["review"])["dossier"]
            )
            return {
                "status": "EVIDENCE_CRO_EXPORT_SPANS_PAGE",
                "export_hash": exported.get("export_hash"),
                "verification_basis": basis,
                "review_publication_hash": cast(
                    dict[str, object],
                    cast(dict[str, object], exported["review"])["publication"],
                ).get("publication_hash"),
                **_citation_page(
                    cast(
                        list[dict[str, object]],
                        cast(dict[str, object], exported["evidence"]).get("verified_spans") or [],
                    ),
                    request,
                    groups={
                        str(child["unit_id"]): tuple(cast(list[str], child["ordered_entity_ids"]))
                        for child in cast(
                            list[dict[str, object]],
                            exported_dossier.get("evidence_children") or [],
                        )
                    },
                    key="verified_spans",
                ),
            }
        case "EVIDENCE_REFRESH":
            return operations.evidence_refresh(selector, caller=caller)
        case "EVIDENCE_PREPARE":
            outcome = operations.review.refresh_evidence(
                dispatcher=operations.dispatcher,
                selector=selector,
                evidence_as_of=(
                    None
                    if request.evidence_as_of is None
                    else datetime.fromisoformat(request.evidence_as_of)
                ),
                prepare_only=True,
                preparation_binding_hash=request.preparation_binding_hash,
                caller=caller,
            )
            body = _review_outcome_body(outcome)
            if outcome.task_id is not None:
                packets = operations.review.packet_requests(selector, task_id=outcome.task_id)
                chosen = operations.review.default_selector(selector)
                # Nothing new: the CRO's review carries forward with its Task and book.
                carried: dict[str, dict[str, Any]] = {
                    "task": {"operation": "STATUS", "task_id": str(outcome.task_id)},
                    **(
                        {"book": {"operation": "EVIDENCE_CRO", **chosen.request_fields()}}
                        if chosen is not None
                        else {}
                    ),
                }
                body["next_requests"] = {
                    **(outcome.next_requests or {}),
                    **(packets or carried),
                }
            return body
        case "EVIDENCE_PREVIEW":
            try:
                return operations.review.preview_evidence(selector, caller=caller)
            except DataError as error:
                if error.failure.code != "data_layer.knowledge_snapshot_missing":
                    raise
                chosen = operations.review.default_selector(selector)
                return {
                    **refused(error.failure.code),
                    "detail": "An earlier Evidence preparation names a snapshot missing from "
                    "this data root. Prepare the book's Evidence again.",
                    "next_requests": {
                        "prepare": {
                            "operation": "EVIDENCE_PREPARE",
                            **(chosen.request_fields() if chosen else {}),
                        }
                    },
                }
        case "EVIDENCE_LEDGER":
            return _review_outcome_body(
                EvidenceReviewDelivery(operations.review).book_ledger(
                    selector, page=request.ledger_page
                )
            )
        case "EVIDENCE_DOCUMENTS":
            return operations.review.documents(page=request.documents_page)
        case "EVIDENCE_PACKET":
            assert request.task_id is not None
            try:
                delivered = EvidenceReviewDelivery(operations.review).export_analysis_packet(
                    selector=selector,
                    task_id=request.task_id,
                    unit_id=request.evidence_unit_id,
                    delivery_part=request.delivery_part,
                    delivery_budget_bytes=request.delivery_budget_bytes,
                    context_hash=request.analysis_context_hash,
                    evidence_detail=request.evidence_detail,
                    view_entity_id=request.view_entity_id,
                    view_topic=request.view_topic,
                    view_last_days=request.view_last_days,
                    view_from=request.view_from,
                    view_to=request.view_to,
                )
            except (PortfolioEvidenceReviewError, ValueError) as error:
                # An unprepared unit says why and where prepared ones are.
                unit = unit_refusal(
                    public_failure(error, "alternative_evidence.refused"),
                    None if selector is None else selector.request_fields(),
                )
                if unit is None:
                    raise
                return unit
            return _review_outcome_body(delivered)
        case "EVIDENCE_CONTINUE":
            assert request.task_id is not None
            assert request.continuation_of is not None
            assert request.continuation_spans is not None
            assert request.session_limit is not None
            assert request.window_limit is not None
            outcome = operations.review.continue_evidence(
                dispatcher=operations.dispatcher,
                selector=selector,
                task_id=request.task_id,
                unit_id=request.evidence_unit_id,
                continuation_of=request.continuation_of,
                continuation_spans=request.continuation_spans,
                session_limit=request.session_limit,
                window_limit=request.window_limit,
            )
            body = _review_outcome_body(outcome)
            if outcome.task_id is not None:
                # The continuation's packet is its own Task's; its unit rides in its input.
                # Its consumer holds every earlier part already, so the
                # session's own windows are offered beside the whole.
                packets = operations.review.packet_requests(selector, task_id=outcome.task_id)
                body["next_requests"] = {
                    **(outcome.next_requests or {}),
                    **packets,
                    "packet_session_windows": {
                        **packets["packet"],
                        "evidence_detail": "session_windows",
                    },
                }
            return body
        case "EVIDENCE_ANALYSIS_SUBMIT":
            return submit_evidence_analysis(
                operations,
                request,
                selector,
                caller,
                read_files,
                agent_run,
                accepted_bundle,
                _review_outcome_body,
            )
        case "EVIDENCE_SELECT":
            assert request.analysis_publication_hash is not None
            return operations.evidence_select(
                selector,
                analysis_publication_hash=request.analysis_publication_hash,
                chosen_by=caller,
            )
        case "CRO_REVIEW":
            return operations.cro_review(selector)
        case "CRO_REVIEW_DOSSIER":
            dossier = EvidenceReviewDelivery(operations.review).export_dossier(
                selector,
                delivery_part=request.delivery_part,
                delivery_budget_bytes=request.delivery_budget_bytes,
                review_dossier_hash=request.review_dossier_hash,
                review_read_at=_read_at(request),
            )
            return _review_outcome_body(dossier) if isinstance(dossier, ReviewOutcome) else dossier
        case "CRO_REVIEW_FINDING":
            assert request.finding_handle is not None
            package = EvidenceReviewDelivery(operations.review).export_finding_evidence(
                selector,
                finding_handle=request.finding_handle,
                review_dossier_hash=request.review_dossier_hash,
                review_publication_hash=request.review_publication_hash,
                review_read_at=_read_at(request),
            )
            return _review_outcome_body(package) if isinstance(package, ReviewOutcome) else package
        case "CRO_REVIEW_SUBMIT":
            assert request.review_dossier_hash is not None
            assert request.review_policy_hash is not None
            assert request.review_schema_hash is not None
            assert request.review_answer is not None
            with (
                operations.workspace_session.mutation_gate.hold()
                if accepted_bundle is not None
                else nullcontext()
            ):
                body = _review_outcome_body(
                    operations.review.submit_assessment(
                        dispatcher=operations.dispatcher,
                        selector=selector,
                        dossier_hash=request.review_dossier_hash,
                        policy_hash=request.review_policy_hash,
                        schema_hash=request.review_schema_hash,
                        answer=request.review_answer,
                        caller=caller,
                        read_files=read_files,
                        agent_run=agent_run,
                        read_at=_read_at(request),
                    )
                )
                accepted_receipt = operations._observe(
                    lambda: operations.capture_accepted_answer_context(accepted_bundle, body)
                )
            return operations.observe_accepted_answer(
                body, accepted_bundle, accepted_receipt=accepted_receipt
            )
    raise ValueError("portfolio_research.operation_unknown")


def _review_bundle_refusal(
    error: Exception, role: AgentRole | None, selector: BookSelector | None
) -> dict[str, object] | None:
    """A bundle selector names its missing fields; an unavailable unit names its read."""
    code = public_failure(error, "agent_bundle.refused")
    # A packet named without its Task or unit is told how to name it.
    if str(error).partition(":")[0] in PACKET_SELECTOR_CODES | {
        "agent_bundle.specialist_task_required",
        "agent_bundle.reference_bound_exceeded",
    }:
        return agent_bundle_refusal(code, role=role)
    # A unit that was not prepared says why and where the prepared ones are.
    unit = unit_refusal(code, None if selector is None else selector.request_fields())
    if unit is None:
        return None
    return unit


def route(call: WorkspaceOperationCall[PortfolioResearchOperations]) -> dict[str, object]:
    """Serve a Review operation, keeping direct submissions' original error handling."""
    run = partial(
        execute,
        call.operations,
        call.request,
        caller=call.caller,
        read_files=call.read_files,
        agent_run=call.agent_run,
        accepted_bundle=call.accepted_bundle,
    )
    if call.raw_review:
        return run()
    return call.operations._execute_review(call.request, run)
