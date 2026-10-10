"""Reference-selected research delivery; existing owners verify every constituent."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import Any, cast
from uuid import UUID

from pydantic_core import to_jsonable_python

from alphalattice.control.product_host.research_authoring.comparison import (
    compare_experiment_reports,
)
from alphalattice.control.product_host.research_authoring.risk_reports import RiskReportLinks
from alphalattice.interface.local_application.failure_codes import located_failure
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchOperationRequest,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.oversight.chief_risk_officer.decision.book_evidence import BookSelector
from alphalattice.protocols.research_authoring.contracts import AuthoringError

from .evidence_review_application import EvidenceReviewApplication
from .evidence_review_delivery import EvidenceReviewDelivery
from .research_experiments import ResearchExperimentApplication


def export_research_delivery(
    *,
    request: PortfolioResearchOperationRequest,
    experiments: ResearchExperimentApplication,
    review: EvidenceReviewApplication | None,
    caller: str,
) -> dict[str, object]:
    """Read-only composition: no implicit latest choice, publication, fit or Task."""
    from alphalattice.interface.local_application.experiment_report import render_research_delivery

    if (
        request.task_id is None
        or request.portfolio_session is None
        or request.experiment_receipt_hash is None
    ):
        raise AuthoringError("research_delivery.subject_required")
    selector = BookSelector(
        experiment_task_id=request.task_id,
        experiment_receipt_hash=request.experiment_receipt_hash,
        portfolio_session=request.portfolio_session,
    )
    # Share only already verified values during this one export. The existing
    # review owner still asks for its reads in its original refusal-first order.
    opened: dict[tuple[UUID, str | None], dict[str, Any]] = {}

    def read(task_id: UUID, session: str | None = None) -> dict[str, Any]:
        if task_id == request.task_id and session is None:
            session = request.portfolio_session
        key = task_id, session
        if key not in opened:
            opened[key] = experiments.readback(task_id, session)
        return opened[key]

    review_export: dict[str, Any] | None = None
    if request.review_publication_hash is not None:
        if review is None:
            raise AuthoringError("research_delivery.review_runtime_unavailable")
        review_export = EvidenceReviewDelivery(replace(review, read_experiment=read)).export_review(
            selector, request.review_publication_hash
        )
        portfolio = cast(dict[str, Any], review_export["portfolio"])
    else:
        portfolio = read(request.task_id, request.portfolio_session)
    if portfolio.get("status") != "EXPERIMENT_PUBLISHED" or not portfolio.get("portfolio_source"):
        raise AuthoringError("research_delivery.completed_portfolio_required")
    if (
        portfolio["task_id"] != str(request.task_id)
        or portfolio["receipt"]["receipt_hash"] != request.experiment_receipt_hash
        or portfolio["position"]["session"] != request.portfolio_session
    ):
        raise AuthoringError("research_delivery.portfolio_subject_mismatch")

    source = portfolio["portfolio_source"]
    alpha = read(UUID(source["alpha_task_id"]))
    if (
        alpha.get("status") != "EXPERIMENT_PUBLISHED"
        or alpha["receipt"]["receipt_hash"] != source["alpha_receipt_hash"]
        or alpha["program"]["program_hash"] != source["alpha_program_hash"]
        or alpha["input_binding_hash"] != source["input_binding_hash"]
        or not any(
            v["candidate_id"] == source["candidate_id"] for v in alpha["result"]["candidates"]
        )
    ):
        raise AuthoringError("research_delivery.alpha_source_mismatch")
    alpha_source = alpha["alpha_source"]
    factor = read(UUID(alpha_source["factor_task_id"]))
    if (
        factor.get("status") != "EXPERIMENT_PUBLISHED"
        or factor["receipt"]["receipt_hash"] != alpha_source["factor_receipt_hash"]
        or factor["input_binding_hash"] != alpha_source["factor_binding"]["binding_hash"]
    ):
        raise AuthoringError("research_delivery.factor_source_mismatch")
    sections: dict[str, dict[str, Any]] = {
        "portfolio": {"status": "PRESENT", "value": portfolio},
        "alpha": {"status": "PRESENT", "value": alpha},
        "factor": {"status": "PRESENT", "value": factor},
        "foundation": {
            "status": "NOT_PUBLISHED",
            "detail": "No Foundation admission named by this Alpha source.",
        },
        "comparison": {"status": "NOT_SELECTED"},
        "risk": {
            "status": "NOT_SELECTED",
            "detail": "No linked Risk report selected; this is not zero risk.",
        },
        "evidence_cro": {
            "status": "NOT_SELECTED",
            "detail": "No review selected; this is not clearance.",
        },
    }
    foundation_hash = alpha_source.get("foundation_admission_hash")
    if foundation_hash is not None:
        foundation = experiments.operate(
            PortfolioResearchOperationRequest(
                operation="EXPERIMENT_FOUNDATION_READBACK",
                foundation_admission_hash=foundation_hash,
            ),
            caller=caller,
        )
        admission = cast(dict[str, Any], foundation["admission"])
        if (
            admission["factor_task_id"] != str(alpha_source["factor_task_id"])
            or admission["curation_receipt_hash"] != alpha_source["curation_receipt_hash"]
        ):
            raise AuthoringError("research_delivery.foundation_source_mismatch")
        sections["foundation"] = {"status": "PRESENT", "value": foundation}
    if request.left_task_id is not None and request.right_task_id is not None:
        if request.task_id not in {request.left_task_id, request.right_task_id}:
            raise AuthoringError("research_delivery.comparison_subject_mismatch")
        try:
            if request.left_task_id == request.right_task_id:
                raise AuthoringError("portfolio_research.comparison_requires_two_results")
            left = (
                portfolio
                if request.left_task_id == request.task_id
                else read(request.left_task_id, request.portfolio_session)
            )
            right = (
                portfolio
                if request.right_task_id == request.task_id
                else read(request.right_task_id, request.portfolio_session)
            )
            compared = compare_experiment_reports(left, right)
            sections["comparison"] = {"status": "PRESENT", "value": compared}
        except ValueError as error:
            if str(error) not in {
                "portfolio_research.comparison_requires_two_results",
                "portfolio_research.comparison_completed_books_required",
                "portfolio_research.comparison_input_or_support_mismatch",
                "portfolio_research.session_outside_report",
            }:
                raise
            sections["comparison"] = {
                "status": "INCOMPATIBLE",
                **located_failure(error, "research_delivery.comparison_incompatible"),
                "left_task_id": str(request.left_task_id),
                "right_task_id": str(request.right_task_id),
            }
    if request.risk_report_hash is not None:
        linked = RiskReportLinks(experiments.session.workspace, read).read_link(
            request.task_id, request.risk_report_hash
        )
        sections["risk"] = {
            "status": "PRESENT",
            "value": {key: value for key, value in linked.items() if key != "portfolio"},
            "selected_session_covered": request.portfolio_session
            in cast(dict[str, Any], linked["risk"])["risk_surface"]["formation_sessions"],
        }
    if review_export is not None:
        # The review's stated expiry: a carried reading's is its window's, and a
        # book read as several units has no one publication to read it from (Z1).
        expiry = review_export["review"]["recommendation"]["evidence_expires_at"]
        assert review is not None
        sections["evidence_cro"] = {
            "status": "EXPIRED"
            if datetime.fromisoformat(expiry) < review.clock()
            else "HISTORICAL_REVIEW",
            "value": {
                k: v
                for k, v in review_export.items()
                if k not in {"kind", "portfolio", "html", "export_hash"}
            },
            "source_kind": review_export["kind"],
            "source_export_hash": review_export["export_hash"],
        }
    _, allowed = request.field_contract(request.operation)
    selection = to_jsonable_python(
        {
            "operation": request.operation,
            **{
                name: getattr(request, name)
                for name in sorted(allowed)
                if getattr(request, name) is not None
            },
        }
    )
    snapshot: dict[str, object] = {
        "kind": "ResearchDeliveryExport",
        "status": "RESEARCH_DELIVERY_EXPORTED",
        "selection": selection,
        "input": {
            "research_input_id": portfolio["research_input_id"],
            "input_binding_hash": portfolio["input_binding_hash"],
            "panel_snapshot_hash": source["panel_snapshot_hash"],
            "outcome_snapshot_hash": source["outcome_snapshot_hash"],
            "universe_revision": source["universe_revision"],
        },
        "question": request.delivery_question,
        "question_status": "PROVIDED_FOR_THIS_DELIVERY_NOT_ORIGINAL_EXPERIMENT_INTENT"
        if request.delivery_question
        else "NOT_RECORDED",
        "sections": sections,
        "summary": {
            "portfolio_task_id": str(request.task_id),
            "portfolio_receipt_hash": request.experiment_receipt_hash,
            "portfolio_session": request.portfolio_session,
            "section_status": {name: value["status"] for name, value in sections.items()},
            "portfolio_result": portfolio["result"],
            "comparison_tasks": None
            if request.left_task_id is None
            else {
                "left": str(request.left_task_id),
                "right": str(request.right_task_id),
            },
            "risk_reference": None
            if sections["risk"]["status"] != "PRESENT"
            else {
                "selected_session_covered": sections["risk"]["selected_session_covered"],
                **(
                    {"report_window": sections["risk"]["value"]["link"]["report_window"]}
                    if sections["risk"]["value"]["link"].get("report_window")
                    else {}
                ),
                **{
                    key: sections["risk"]["value"]["link"][key]
                    for key in (
                        "link_hash",
                        "risk_formation_count",
                        "portfolio_formation_count",
                        "claim",
                    )
                },
            },
            "review": None
            if review_export is None
            else {
                "publication_hash": review_export["review"]["publication"]["publication_hash"],
                "route": review_export["review"]["recommendation"]["route"],
                "completeness": review_export["review"]["recommendation"]["review_state"],
                "coverage": review_export["review"]["dossier"]["coverage"],
            },
        },
        "commentary": [v.model_dump(mode="json") for v in request.delivery_commentary or ()],
        "commentary_provenance": {
            "submitted_by": caller,
            "attribution": "CALLER_SUPPLIED_NOT_VERIFIED_HOST_IDENTITIES",
        },
        "next_requests": {"reopen": selection},
        "claim": (
            "Verified historical research with separately attributed commentary. "
            "No winner, current advice, scientific approval or trade authority. "
            "Optional missing sections are not zero risk."
        ),
    }
    rendered = render_research_delivery(snapshot)
    body = {**snapshot, "html": rendered}
    return {**body, "export_hash": str(canonical_hash(body))}


def export_update_delivery(
    *,
    request: PortfolioResearchOperationRequest,
    readback: dict[str, Any],
    review: EvidenceReviewApplication | None,
    committee: list[dict[str, str]] | None,
) -> dict[str, object]:
    """A date's published positions as the delivery.

    Its publication, the exact review it names and, once the committee closed, its floor as the
    commentary, which the Host accepted.
    """
    from alphalattice.interface.local_application.experiment_report import render_research_delivery

    publication = readback.get("publication")
    subject = {
        "update_task_id": str(request.update_task_id),
        "update_publication_hash": request.update_publication_hash,
        "position_basis": request.position_basis,
    }
    context = readback.get("committee_context")
    if (
        not isinstance(publication, dict)
        or publication["content_hash"] != request.update_publication_hash
        or request.position_basis is None
        or readback.get("review_selector") != subject
        or (context and context.get("review_selector") != subject)
    ):
        raise AuthoringError("research_delivery.update_subject_mismatch")
    selection: dict[str, object] = {
        "operation": "EXPERIMENT_DELIVERY_EXPORT",
        **subject,
    }
    review_export: dict[str, Any] | None = None
    if request.review_publication_hash is not None:
        if review is None:
            raise AuthoringError("research_delivery.review_runtime_unavailable")
        review_export = EvidenceReviewDelivery(review).export_review(
            BookSelector(
                update_task_id=request.update_task_id,
                update_publication_hash=request.update_publication_hash,
                position_basis=request.position_basis,
            ),
            request.review_publication_hash,
        )
        selection["review_publication_hash"] = request.review_publication_hash
    evidence: dict[str, Any] = {"status": "PRESENT" if review_export else "NOT_SELECTED"}
    if review_export:
        evidence["value"] = review_export
    snapshot: dict[str, object] = {
        "kind": "ResearchDeliveryExport",
        "status": "RESEARCH_DELIVERY_EXPORTED",
        "selection": selection,
        "input": {"strategy_package_id": readback.get("strategy_package_id"), **selection},
        "question": request.delivery_question,
        "question_status": "PROVIDED_FOR_THIS_DELIVERY_NOT_ORIGINAL_EXPERIMENT_INTENT"
        if request.delivery_question
        else "NOT_RECORDED",
        "sections": {
            "positions": {
                "status": "PRESENT",
                "value": {
                    "html": readback.get("html"),
                    "basis": request.position_basis,
                    **{
                        k: readback[k]
                        for k in ("position_rows", "schedule", "date_risk")
                        if k in readback
                    },
                },
            },
            "evidence_cro": evidence,
        },
        "summary": {"section_status": {"positions": "PRESENT", "evidence_cro": evidence["status"]}},
        "commentary": committee or [],
        **({"committee_context": context} if context else {}),
        "commentary_provenance": {"submitted_by": "HOST", "attribution": "COMMITTEE_FLOOR"},
        "next_requests": {"reopen": selection},
        "claim": (
            "Research positions for a date, with the committee's attributed commentary. No "
            "orders, current advice, scientific approval or trade authority."
        ),
    }
    rendered = render_research_delivery(snapshot)
    body = {**snapshot, "html": rendered}
    return {**body, "export_hash": str(canonical_hash(body))}
