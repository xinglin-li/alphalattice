"""The research case page's compatibility routes over goals.

The Workbench's case page still saves a research case document and reads a case record. This
module translates both ways, so the page's `/api/research/case*` calls keep working while goals
replace research cases.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final
from uuid import UUID

from alphalattice.control.product_host.publication.goals import GoalStore
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchOperationRequest,
)

CASE_ROUTES: Final = (
    ("GET", "/api/research/cases", "CASE_LIST"),
    ("GET", "/api/research/case", "CASE_READBACK"),
    ("GET", "/api/research/case/narrative", "CASE_NARRATIVE"),
    ("GET", "/api/research/case/export", "CASE_EXPORT"),
    ("POST", "/api/research/case/save", "CASE_SAVE"),
    ("POST", "/api/research/case/attach", "CASE_ATTACH"),
    ("POST", "/api/research/case/note", "CASE_NOTE"),
)
"""The case page's own calls (`live-cases.js`, `live-team.js`, the UI walk's census)."""

_STATUS: Final = {
    "GOAL_SHOW": "CASE_READBACK",
    "GOAL_NARRATIVE": "CASE_NARRATIVE",
    "GOAL_EXPORTED": "CASE_EXPORTED",
    "GOAL_SAVED": "CASE_SAVED",
}


def declaration_of(document: Mapping[str, Any]) -> dict[str, Any]:
    """A research case document as a research goal's declaration."""
    return {
        "title": document.get("title"),
        "objective": document.get("question"),
        "kind": "RESEARCH",
        "scope": document.get("scope") or "",
        "constraints": list(document.get("constraints") or ()),
        "criteria": [
            {"criterion_id": f"c{index}", "text": text}
            for index, text in enumerate(document.get("criteria") or (), start=1)
        ],
        "budget": document.get("budget"),
        "research": {
            "purpose": document.get("purpose"),
            "comparison_design": document.get("comparison_design"),
            "required_stages": list(document.get("required_stages") or ()),
        },
    }


def case_request(operation: str, fields: Mapping[str, Any]) -> PortfolioResearchOperationRequest:
    """The goal operation a case route's request names, its fields translated."""
    try:
        goal_id = UUID(str(fields["case_id"])) if fields.get("case_id") else None
    except ValueError as error:
        raise ValueError("research_case.case_id_invalid") from error
    goal_hash = fields.get("case_hash")
    reason = fields.get("change_reason")
    if operation == "CASE_LIST":
        limit = fields.get("history_limit")
        return PortfolioResearchOperationRequest(
            operation="GOAL_LIST",
            history_limit=int(limit) if limit is not None else None,
            history_cursor=fields.get("history_cursor"),
        )
    if operation in {"CASE_READBACK", "CASE_NARRATIVE", "CASE_EXPORT"}:
        name = {"CASE_READBACK": "GOAL_SHOW", "CASE_NARRATIVE": "GOAL_NARRATIVE"}
        return PortfolioResearchOperationRequest(
            operation=name.get(operation, "GOAL_EXPORT"),  # type: ignore[arg-type]
            goal_hash=goal_hash,
        )
    if operation == "CASE_SAVE":
        declaration = declaration_of(fields.get("case_document") or {})
        if goal_hash:
            return PortfolioResearchOperationRequest(
                operation="GOAL_REVISE",
                goal_id=goal_id,
                goal_hash=goal_hash,
                goal_declaration=declaration,
                change_reason=reason,
            )
        if goal_id is None:
            raise ValueError("research_case.case_id_invalid")
        return PortfolioResearchOperationRequest(
            operation="GOAL_OPEN",
            goal_id=goal_id,
            goal_declaration=declaration,
            change_reason=reason,
        )
    if operation == "CASE_ATTACH":
        return PortfolioResearchOperationRequest(
            operation="GOAL_ATTACH",
            goal_hash=goal_hash,
            goal_reference=fields.get("case_reference"),
            change_reason=reason,
        )
    if operation == "CASE_NOTE":
        return PortfolioResearchOperationRequest(
            operation="GOAL_NOTE",
            goal_hash=goal_hash,
            goal_statement=fields.get("case_statement"),
            change_reason=reason,
        )
    raise ValueError("research_case.operation_not_supported")


def case_document(declaration: Mapping[str, Any]) -> dict[str, Any]:
    """A goal's declaration in the case page's words."""
    research = declaration.get("research") or {}
    return {
        "title": declaration["title"],
        "question": declaration["objective"],
        "purpose": research.get("purpose", "NEW_RESEARCH"),
        "scope": declaration.get("scope", ""),
        "comparison_design": research.get("comparison_design", ""),
        "criteria": [c["text"] for c in declaration.get("criteria", ())],
        "constraints": list(declaration.get("constraints", ())),
        "budget": declaration.get("budget")
        or {"maximum_tasks": 0, "maximum_numerical_calls": 0, "maximum_model_calls": 0},
        "required_stages": list(research.get("required_stages", ())),
    }


def case_record(goal: Mapping[str, Any]) -> dict[str, Any]:
    """A goal revision in the shape of a research case revision."""
    return {
        "kind": "ResearchCase",
        "case_id": goal["goal_id"],
        "workspace_id": goal["workspace_id"],
        "parent_hash": goal["parent_hash"],
        "revision": goal["revision"],
        "recorded_at": goal["recorded_at"],
        "intent_registered_at": goal["intent_registered_at"],
        "submitted_by": goal["submitted_by"],
        "change_reason": goal["change_reason"],
        "document": case_document(goal["declaration"]),
        "references": goal["references"],
        "statements": goal["statements"],
        # Assignments are Team's now, in the goal's conversation; the UI pass shows them.
        "assignments": [],
        "case_hash": goal["goal_hash"],
    }


def _case_next(requests: Mapping[str, Any]) -> dict[str, Any]:
    names = {"GOAL_SHOW": "CASE_READBACK", "GOAL_NARRATIVE": "CASE_NARRATIVE"}
    out: dict[str, Any] = {}
    for key, value in requests.items():
        if not isinstance(value, Mapping) or not str(value.get("operation", "")).startswith(
            "GOAL_"
        ):
            out[key] = value
        elif value["operation"] in names and value.get("goal_hash"):
            out[key] = {"operation": names[value["operation"]], "case_hash": value["goal_hash"]}
    return out


def case_answer(operation: str, body: Mapping[str, Any], store: GoalStore) -> dict[str, Any]:
    """A goal operation's answer in the case page's shape; a refusal passes through as it is."""
    if body.get("status") == "REFUSED":
        return dict(body)
    if operation == "CASE_LIST":
        cases = []
        for row in body["goals"]:
            head = store.head(UUID(row["goal_id"]))
            cases.append(
                {
                    "case_id": row["goal_id"],
                    "case_hash": row["goal_hash"],
                    "title": row["title"],
                    "question": row["objective"],
                    "purpose": row["research_purpose"] or "NEW_RESEARCH",
                    "revision": row["revision"],
                    "recorded_at": row["recorded_at"],
                    "reference_count": row["reference_count"],
                    "statement_count": row["statement_count"],
                    "outcome": head.outcome() if head else "QUESTION_OPEN",
                }
            )
        return {
            "status": body["status"],
            "cases": cases,
            "next_cursor": body["next_cursor"],
            "claim": body["claim"],
        }
    answer = {
        key: value
        for key, value in body.items()
        if key not in {"goal_id", "goal_hash", "goal", "state", "record", "goal_prompt"}
        and key not in {"bound_session", "next_requests", "statement_context"}
    }
    answer["status"] = _STATUS.get(str(body.get("status")), body.get("status"))
    answer["case_id"] = body["goal_id"]
    answer["case_hash"] = body["goal_hash"]
    if "goal" in body:
        answer["case"] = case_record(body["goal"])
    if "statement_context" in body:
        answer["statement_context"] = {
            key: {"case_hash": v["goal_hash"], "question": v["objective"], "design": v["design"]}
            for key, v in body["statement_context"].items()
        }
    if "next_requests" in body:
        answer["next_requests"] = _case_next(body["next_requests"])
    return answer
