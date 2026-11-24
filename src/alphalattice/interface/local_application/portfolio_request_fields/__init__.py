"""Request field order and the models that supply fields by route family."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal, cast

from pydantic.fields import FieldInfo

type RequestFamily = Literal[
    "workspace", "goals", "research", "data", "feature", "review", "portfolio", "task"
]

# This order is part of the public schema and validation error output.
FIELD_OWNER: dict[str, RequestFamily] = {
    "schema_operation": "workspace",
    "goal_id": "goals",
    "goal_hash": "goals",
    "goal_reference_id": "goals",
    "goal_declaration": "goals",
    "goal_reference": "goals",
    "goal_statement": "goals",
    "goal_submission": "goals",
    "change_reason": "goals",
    "history_limit": "research",
    "history_cursor": "data",
    "history_entry_id": "research",
    "history_kind": "research",
    "experience_hypothesis": "research",
    "experience_method": "research",
    "experience_workspace": "research",
    "experience_trial": "research",
    "experience_width": "research",
    "experience_known_at": "research",
    "experience_budget": "research",
    "experience_query": "research",
    "experience_limit": "research",
    "review_subject": "research",
    "review_verdict": "research",
    "review_note": "research",
    "left_task_id": "research",
    "right_task_id": "research",
    "left_candidate_id": "research",
    "right_candidate_id": "research",
    "delivery_question": "research",
    "delivery_commentary": "research",
    "research_input_plan_hash": "research",
    "origin_task_id": "research",
    "factor_task_id": "research",
    "preparation_plan_hash": "data",
    "data_issue_case_token": "data",
    "data_issue_evidence_hash": "data",
    "data_issue_option_id": "data",
    "data_issue_option_hash": "data",
    "data_issue_grant_hash": "data",
    "storage_plan_hash": "data",
    "evidence_index_id": "data",
    "input_binding_hash": "research",
    "input_pinned": "data",
    "research_input_id": "research",
    "experiment_document": "research",
    "feature_document": "feature",
    "feature_plan_hash": "feature",
    "feature_preparation_hash": "research",
    "feature_output": "feature",
    "experiment_kind": "research",
    "risk_task_id": "research",
    "risk_report_hash": "research",
    "experiment_yaml": "research",
    "experiment_plan_hash": "research",
    "risk_report_scope": "research",
    "experiment_curation": "research",
    "curation_receipt_hash": "research",
    "foundation_admission_hash": "research",
    "candidate_id": "research",
    "portfolio_session": "review",
    "performance": "portfolio",
    "portfolio_scope": "portfolio",
    "spec": "portfolio",
    "strategy_package_id": "portfolio",
    "component_id": "research",
    "model_lifecycle": "research",
    "model_id": "research",
    "task_id": "research",
    "expected_task_hash": "task",
    "recovery_task_id": "portfolio",
    "recovery_task_hash": "portfolio",
    "incident_key": "task",
    "remedy": "task",
    "result_hash": "review",
    "left_result_hash": "portfolio",
    "right_result_hash": "portfolio",
    "candidate_hash": "portfolio",
    "handoff_hash": "review",
    "update_task_id": "review",
    "experiment_task_id": "review",
    "experiment_receipt_hash": "review",
    "update_publication_hash": "review",
    "position_basis": "review",
    "review_publication_hash": "review",
    "analysis_publication_hash": "review",
    "review_dossier_hash": "review",
    "review_policy_hash": "review",
    "review_schema_hash": "review",
    "review_read_at": "review",
    "review_answer": "review",
    "analysis_context_hash": "review",
    "analysis_answer": "review",
    "agent_role": "workspace",
    "bundle_directory": "workspace",
    "agent_answer": "workspace",
    "evidence_as_of": "review",
    "preparation_binding_hash": "review",
    "evidence_unit_id": "review",
    "delivery_part": "review",
    "delivery_budget_bytes": "review",
    "evidence_detail": "review",
    "view_entity_id": "review",
    "view_topic": "review",
    "view_last_days": "review",
    "view_from": "review",
    "view_to": "review",
    "ledger_page": "review",
    "documents_page": "review",
    "citation_entity_id": "review",
    "citation_unit_id": "review",
    "citation_page": "review",
    "continuation_of": "review",
    "continuation_spans": "review",
    "session_limit": "review",
    "window_limit": "review",
    "finding_handle": "review",
    "prior_review_publication_hash": "review",
    "update_plan_hash": "data",
    "score_plan_hash": "portfolio",
    "score_snapshot_hash": "portfolio",
    "calibration_plan_hash": "portfolio",
    "prepared_input_hash": "portfolio",
    "observed_through": "portfolio",
    "positions_date": "portfolio",
    "formation_session": "portfolio",
    "automation_enabled": "portfolio",
    "network_enabled": "workspace",
    "evidence_documents_per_issuer": "review",
    "evidence_total_documents": "review",
    "evidence_total_bytes": "review",
    "evidence_setup": "review",
    "usage_reading_enabled": "workspace",
    "committee_daily_enabled": "goals",
    "analyst_reads_mode": "workspace",
    "analyst_reads_bundles": "workspace",
    "automation_package_ids": "portfolio",
    "upgrade_set_hash": "workspace",
    "feature_trial_id": "feature",
    "feature_factor_id": "feature",
    "extensions_page": "feature",
    "agent_session": "goals",
    "wait_seconds": "task",
    "status_section": "task",
    "after": "workspace",
    "limit": "workspace",
    "watch": "workspace",
    "event": "workspace",
    "wake_thread": "workspace",
    "wake_read": "workspace",
    "wake_codex_path": "workspace",
    "storage_cap_bytes": "data",
    "cpu_budget": "workspace",
    "tasks_waiting": "workspace",
    "tasks_running": "workspace",
    "backup_generations_kept": "data",
    "committee_role": "goals",
    "committee_key": "goals",
    "committee_message": "goals",
    "committee_seen": "goals",
    "person_confirmation": "data",
}


_FAMILIES_HELD = 8
"""One entry per request family."""


@lru_cache(maxsize=_FAMILIES_HELD)
def request_fields(family: RequestFamily) -> dict[str, FieldInfo]:
    """Load the fields of a family only when a request needs one of them."""
    model: type[Any]
    match family:
        case "workspace":
            from . import workspace  # lazy: workspace request

            model = workspace.RequestFields
        case "goals":
            from . import goals  # lazy: goals request

            model = goals.RequestFields
        case "research":
            from . import research  # lazy: research request

            model = research.RequestFields
        case "data":
            from . import data  # lazy: data request

            model = data.RequestFields
        case "feature":
            from . import feature  # lazy: feature request

            model = feature.RequestFields
        case "review":
            from . import review  # lazy: review request

            model = review.RequestFields
        case "portfolio":
            from . import portfolio  # lazy: portfolio request

            model = portfolio.RequestFields
        case "task":
            from . import task  # lazy: task request

            model = task.RequestFields
    return cast(dict[str, FieldInfo], model.model_fields)
