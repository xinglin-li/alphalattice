"""Required and permitted fields for each local research operation."""

data_issue_fields = frozenset(
    {
        "data_issue_case_token",
        "data_issue_evidence_hash",
        "data_issue_option_id",
        "data_issue_option_hash",
    }
)
requirements = {
    "GOAL_LIST": (
        frozenset(),
        frozenset({"agent_session", "history_limit", "history_cursor", "task_id"}),
    ),
    "GOAL_OPEN": (
        frozenset({"goal_declaration"}),
        frozenset({"goal_id", "goal_declaration", "change_reason"}),
    ),
    # Bound sessions default to their Goal.
    "GOAL_REVISE": (
        frozenset({"goal_declaration", "change_reason"}),
        frozenset({"goal_id", "goal_hash", "goal_declaration", "change_reason"}),
    ),
    **{
        # A saved revision or the current head.
        op: (frozenset(), frozenset({"goal_id", "goal_hash"}))
        for op in ("GOAL_SHOW", "GOAL_NARRATIVE", "GOAL_PROFILE", "GOAL_EXPORT")
    },
    "GOAL_REFERENCE": (
        frozenset({"goal_hash", "goal_reference_id"}),
        frozenset({"goal_hash", "goal_reference_id"}),
    ),
    **{
        op: (
            frozenset({field, "change_reason"}),
            frozenset({"goal_id", "goal_hash", field, "change_reason"}),
        )
        for op, field in (
            ("GOAL_ATTACH", "goal_reference"),
            ("GOAL_NOTE", "goal_statement"),
        )
    },
    "GOAL_CONTINUE": (
        frozenset({"goal_hash", "task_id"}),
        frozenset({"goal_hash", "task_id", "research_input_id", "input_binding_hash"}),
    ),
    "GOAL_TAKE": (frozenset({"goal_id"}), frozenset({"goal_id"})),
    "GOAL_SUBMIT": (
        frozenset({"goal_submission"}),
        frozenset({"goal_id", "goal_hash", "goal_submission"}),
    ),
    "GOAL_ABANDON": (
        frozenset({"change_reason"}),
        frozenset({"goal_id", "change_reason"}),
    ),
    "DATA_ISSUES": (frozenset(), frozenset({"history_limit", "history_cursor"})),
    "DATA_ISSUE_PREVIEW": (data_issue_fields, data_issue_fields),
    "DATA_ISSUE_CONFIRM": (
        data_issue_fields,
        data_issue_fields | {"data_issue_grant_hash"},
    ),
    "DATA_ISSUE_DELEGATE": (
        data_issue_fields | {"task_id"},
        data_issue_fields | {"task_id"},
    ),
    "DATA_ISSUE_REVOKE": (
        frozenset({"data_issue_grant_hash"}),
        frozenset({"data_issue_grant_hash"}),
    ),
    "RESEARCH_HISTORY": (
        frozenset(),
        frozenset(
            {
                "history_limit",
                "history_cursor",
                "history_entry_id",
                "history_kind",
                "strategy_package_id",
                "research_input_id",
                "input_binding_hash",
            }
        ),
    ),
    "EXPERIENCE_READ": (
        frozenset(),
        frozenset(
            {
                "goal_id",
                "experience_hypothesis",
                "experience_method",
                "experience_workspace",
                "history_limit",
            }
        ),
    ),
    "EXPERIENCE_BRIEF": (
        frozenset({"goal_id"}),
        frozenset({"goal_id", "experience_budget", "experience_known_at"}),
    ),
    "EXPERIENCE_CLASS": (
        frozenset({"experience_trial"}),
        frozenset({"experience_trial", "experience_width", "experience_known_at"}),
    ),
    "EXPERIENCE_RECALL": (
        frozenset({"experience_query"}),
        frozenset(
            {"experience_query", "experience_workspace", "experience_known_at", "experience_limit"}
        ),
    ),
    "EXPERIENCE_RECORD": (
        frozenset({"review_subject", "review_verdict", "review_note"}),
        frozenset({"review_subject", "review_verdict", "review_note"}),
    ),
    "EXPERIMENT_COMPARE": (
        frozenset({"left_task_id", "right_task_id"}),
        frozenset({"left_task_id", "right_task_id", "portfolio_session"}),
    ),
    "EXPERIMENT_ALPHA_COMPARE": (
        frozenset(
            {
                "left_task_id",
                "right_task_id",
                "left_candidate_id",
                "right_candidate_id",
            }
        ),
        frozenset(
            {
                "left_task_id",
                "right_task_id",
                "left_candidate_id",
                "right_candidate_id",
            }
        ),
    ),
    "EXPERIMENT_DELIVERY_EXPORT": (
        # A study's book or a date's published positions (`ALTERNATIVES`).
        frozenset(),
        frozenset(
            {
                "task_id",
                "experiment_receipt_hash",
                "portfolio_session",
                "update_task_id",
                "update_publication_hash",
                "position_basis",
                "left_task_id",
                "right_task_id",
                "risk_report_hash",
                "review_publication_hash",
                "delivery_question",
                "delivery_commentary",
            }
        ),
    ),
    "EXPERIMENT_PORTFOLIO_DRAFT": (
        frozenset({"task_id", "candidate_id"}),
        frozenset({"task_id", "candidate_id"}),
    ),
    "EXPERIMENT_FOUNDATION_PREVIEW": (
        frozenset({"task_id", "curation_receipt_hash", "research_input_id", "input_binding_hash"}),
        frozenset({"task_id", "curation_receipt_hash", "research_input_id", "input_binding_hash"}),
    ),
    **{
        op: (
            frozenset({"foundation_admission_hash"}),
            frozenset({"foundation_admission_hash"}),
        )
        for op in (
            "EXPERIMENT_FOUNDATION_SEAL",
            "EXPERIMENT_FOUNDATION_READBACK",
            "EXPERIMENT_FOUNDATION_SUMMARY",
            "EXPERIMENT_FOUNDATION_EXPORT",
            "EXPERIMENT_FOUNDATION_DRAFT",
        )
    },
    "RESEARCH_INPUT_PLAN": (
        frozenset({"research_input_id"}),
        frozenset({"research_input_id"}),
    ),
    "RESEARCH_INPUT_CONFIRM": (
        frozenset({"research_input_plan_hash"}),
        frozenset({"research_input_plan_hash"}),
    ),
    "RESEARCH_INPUT_READBACK": (frozenset({"task_id"}), frozenset({"task_id"})),
    "EXPERIMENT_DRAFT": (
        frozenset({"task_id"}),
        frozenset({"task_id", "input_binding_hash", "research_input_id"}),
    ),
    "EXPERIMENT_PROMOTE": (frozenset({"task_id"}), frozenset({"task_id"})),
    "EXPERIMENT_CONTINUE": (frozenset({"task_id"}), frozenset({"task_id"})),
    "ACTIVITY_REFUSALS": (frozenset(), frozenset({"view_last_days"})),
    "NETWORK_ACCESS_SET": (frozenset({"network_enabled"}), frozenset({"network_enabled"})),
    "EVIDENCE_INSTALL": (frozenset({"evidence_setup"}), frozenset({"evidence_setup"})),
    "EVIDENCE_CONSENT_SET": (
        frozenset({"evidence_documents_per_issuer"}),
        frozenset(
            {
                "evidence_documents_per_issuer",
                "evidence_total_documents",
                "evidence_total_bytes",
            }
        ),
    ),
    "WORKSPACE_PREPARE_READBACK": (frozenset(), frozenset({"task_id"})),
    "WORKSPACE_PREPARE_CONFIRM": (
        frozenset({"preparation_plan_hash"}),
        frozenset({"preparation_plan_hash", "data_issue_grant_hash"}),
    ),
    "STORAGE_CAP_SET": (frozenset({"storage_cap_bytes"}), frozenset({"storage_cap_bytes"})),
    "STORAGE_CONFIRM": (frozenset({"storage_plan_hash"}), frozenset({"storage_plan_hash"})),
    "STORAGE_PIN": (
        frozenset({"input_binding_hash", "input_pinned"}),
        frozenset({"input_binding_hash", "input_pinned"}),
    ),
    "STORAGE_EVIDENCE_REBUILD": (
        frozenset({"evidence_index_id"}),
        frozenset({"evidence_index_id"}),
    ),
    "EXPERIMENT_CONTROLS": (
        frozenset(),
        frozenset(
            {
                "research_input_id",
                "input_binding_hash",
                "experiment_kind",
                "component_id",
                "feature_preparation_hash",
                "model_lifecycle",
            }
        ),
    ),
    "FEATURE_CATALOG_CONTROLS": (
        frozenset({"input_binding_hash"}),
        frozenset({"input_binding_hash", "feature_plan_hash"}),
    ),
    "FEATURE_CATALOG_BUILD": (
        frozenset({"feature_plan_hash"}),
        frozenset({"feature_plan_hash", "feature_output"}),
    ),
    "FEATURE_CATALOG_BUILD_READBACK": (frozenset({"task_id"}), frozenset({"task_id"})),
    "FEATURE_TRIAL": (
        frozenset({"feature_plan_hash", "task_id"}),
        frozenset({"feature_plan_hash", "task_id"}),
    ),
    "FEATURE_TRIAL_READBACK": (
        frozenset({"feature_trial_id"}),
        frozenset({"feature_trial_id"}),
    ),
    "FEATURE_REVIEW": (
        frozenset({"feature_plan_hash", "feature_factor_id"}),
        frozenset({"feature_plan_hash", "feature_factor_id"}),
    ),
    "FEATURE_ACTIVATE": (
        frozenset({"feature_plan_hash", "feature_factor_id"}),
        frozenset({"feature_plan_hash", "feature_factor_id"}),
    ),
    "FEATURE_DEACTIVATE": (
        frozenset({"feature_factor_id"}),
        frozenset({"feature_factor_id"}),
    ),
    # Every declared formula factor, a page at a time.
    "FEATURE_EXTENSIONS": (frozenset(), frozenset({"extensions_page"})),
    "FEATURE_CATALOG_PLAN": (
        frozenset({"feature_document"}),
        frozenset({"feature_document"}),
    ),
    "FEATURE_CATALOG_READBACK": (
        frozenset({"feature_plan_hash"}),
        frozenset({"feature_plan_hash"}),
    ),
    "MODEL_TRAINING_INPUT_PLAN": (
        frozenset({"research_input_id", "component_id"}),
        frozenset({"research_input_id", "input_binding_hash", "component_id", "model_lifecycle"}),
    ),
    "RESEARCH_STRATEGY_PLAN": (
        frozenset({"experiment_document"}),
        frozenset({"experiment_document"}),
    ),
    "RESEARCH_STRATEGY_PREPARE": (
        frozenset({"experiment_plan_hash"}),
        frozenset({"experiment_plan_hash"}),
    ),
    "RESEARCH_STRATEGY_READBACK": (frozenset({"task_id"}), frozenset({"task_id"})),
    "RESEARCH_STRATEGY_INSTALL": (frozenset({"task_id"}), frozenset({"task_id"})),
    "MODEL_TRAINING_INPUT_PREPARE": (
        frozenset({"experiment_plan_hash"}),
        frozenset({"experiment_plan_hash"}),
    ),
    "MODEL_TRAINING_INPUT_READBACK": (
        frozenset({"task_id"}),
        frozenset({"task_id"}),
    ),
    "EXPERIMENT_LINK_RISK": (
        frozenset({"task_id", "risk_task_id"}),
        frozenset({"task_id", "risk_task_id", "risk_report_scope"}),
    ),
    "EXPERIMENT_RISK_LINKS": (
        frozenset({"task_id"}),
        frozenset({"task_id"}),
    ),
    "EXPERIMENT_RISK_EXPORT": (
        frozenset({"task_id", "risk_report_hash"}),
        frozenset({"task_id", "risk_report_hash"}),
    ),
    "EXPERIMENT_PLAN": (
        frozenset(),
        frozenset(
            {
                "research_input_id",
                "input_binding_hash",
                "experiment_document",
                "experiment_yaml",
                "origin_task_id",
                "factor_task_id",
                "curation_receipt_hash",
            }
        ),
    ),
    "EXPERIMENT_RUN": (
        frozenset({"experiment_plan_hash"}),
        frozenset({"experiment_plan_hash", "experience_hypothesis"}),
    ),
    "EXPERIMENT_PREVIEW_READBACK": (
        frozenset({"experiment_plan_hash"}),
        frozenset({"experiment_plan_hash"}),
    ),
    "EXPERIMENT_SUMMARY": (frozenset({"task_id"}), frozenset({"task_id"})),
    "EXPERIMENT_READBACK": (
        frozenset({"task_id"}),
        frozenset({"task_id", "portfolio_session"}),
    ),
    "EXPERIMENT_REPLAY": (frozenset({"task_id"}), frozenset({"task_id"})),
    "EXPERIMENT_EXPORT": (
        frozenset({"task_id"}),
        frozenset({"task_id", "portfolio_session"}),
    ),
    "EXPERIMENT_CURATION": (frozenset({"task_id"}), frozenset({"task_id"})),
    "EXPERIMENT_CURATE": (
        frozenset({"task_id", "experiment_curation"}),
        frozenset({"task_id", "experiment_curation"}),
    ),
    "EXPERIMENT_HANDOFF_PREVIEW": (
        frozenset({"task_id", "curation_receipt_hash"}),
        frozenset(
            {
                "task_id",
                "curation_receipt_hash",
                "input_binding_hash",
                "research_input_id",
                "experiment_document",
                "experiment_yaml",
            }
        ),
    ),
    "DATA_CHANGE_CONFIRM": (
        frozenset({"update_plan_hash"}),
        frozenset({"update_plan_hash"}),
    ),
    "RESEARCH_UPDATE_AUTOMATION_CONFIGURE": (
        frozenset({"automation_enabled", "automation_package_ids"}),
        frozenset({"automation_enabled", "automation_package_ids"}),
    ),
    "RESEARCH_UPDATE_PLAN": (
        frozenset({"strategy_package_id"}),
        frozenset({"strategy_package_id", "observed_through", "positions_date"}),
    ),
    "RESEARCH_UPDATE_RUN": (
        frozenset({"update_plan_hash"}),
        frozenset({"update_plan_hash"}),
    ),
    # A Task, or a strategy's latest update, never another strategy's.
    "RESEARCH_UPDATE_READBACK": (
        frozenset(),
        frozenset({"task_id", "strategy_package_id"}),
    ),
    # The client's own commands, operations like any other (OP1).
    "OPERATION_LIST": (frozenset(), frozenset({"strategy_package_id", "schema_operation"})),
    "ACTIVITY_LIST": (frozenset(), frozenset({"after", "limit", "watch"})),
    "ACTIVITY_RECENT": (frozenset(), frozenset({"limit"})),
    "EVENT_DECLARE": (frozenset({"event"}), frozenset({"event"})),
    "COMMITTEE_OPEN": (frozenset({"update_task_id"}), frozenset({"update_task_id"})),
    "COMMITTEE_SUBMIT": (
        frozenset({"update_task_id", "committee_role", "committee_key", "committee_message"}),
        frozenset({"update_task_id", "committee_role", "committee_key", "committee_message"}),
    ),
    "COMMITTEE_READ": (
        frozenset({"update_task_id"}),
        frozenset({"update_task_id", "committee_role", "committee_key", "committee_seen"}),
    ),
    "WAKE_REGISTER": (
        frozenset({"task_id", "wake_thread", "wake_read"}),
        frozenset({"task_id", "wake_thread", "wake_read", "wake_codex_path"}),
    ),
    "USAGE_READING_SET": (frozenset({"usage_reading_enabled"}),) * 2,
    "COMMITTEE_DAILY_SET": (frozenset({"committee_daily_enabled"}),) * 2,
    "ANALYST_READS": (frozenset(), frozenset({"analyst_reads_bundles"})),
    "ANALYST_READS_SET": (frozenset({"analyst_reads_mode", "analyst_reads_bundles"}),) * 2,
    # One of the three, which the owner holds to: a setting a request.
    "CPU_BUDGET_SET": (
        frozenset(),
        frozenset({"cpu_budget", "tasks_waiting", "tasks_running"}),
    ),
    # The held state's backup outside the workspace; its restore is the
    # client's own, with no Host.
    "WORKSPACE_BACKUP": (frozenset(), frozenset({"backup_generations_kept"})),
    "MODEL_ACTIVATE": (frozenset({"model_id"}), frozenset({"model_id"})),
    "MODEL_DEACTIVATE": (frozenset({"model_id"}), frozenset({"model_id"})),
    # A person runs a reviewed research book's strategy forward, or stops it (LS1).
    "STRATEGY_ACTIVATE": (frozenset({"task_id"}), frozenset({"task_id"})),
    "STRATEGY_DEACTIVATE": (
        frozenset({"strategy_package_id"}),
        frozenset({"strategy_package_id"}),
    ),
    "PORTFOLIO_UPDATE_PLAN": (
        frozenset({"strategy_package_id"}),
        frozenset({"strategy_package_id", "prepared_input_hash", "observed_through"}),
    ),
    "PORTFOLIO_UPDATE_RUN": (
        frozenset({"update_plan_hash"}),
        frozenset({"update_plan_hash"}),
    ),
    # A Task, or a strategy's latest update, never another strategy's.
    "PORTFOLIO_UPDATE_READBACK": (
        frozenset(),
        frozenset({"task_id", "strategy_package_id"}),
    ),
    "STRATEGY_CALIBRATION_PLAN": (
        frozenset({"strategy_package_id", "score_snapshot_hash"}),
        frozenset({"strategy_package_id", "score_snapshot_hash"}),
    ),
    "STRATEGY_CALIBRATION_RUN": (
        frozenset({"calibration_plan_hash"}),
        frozenset({"calibration_plan_hash"}),
    ),
    # A Task, or a strategy's latest, never another strategy's.
    "STRATEGY_CALIBRATION_READBACK": (
        frozenset(),
        frozenset({"task_id", "strategy_package_id"}),
    ),
    "STRATEGY_SCORE_PLAN": (
        frozenset({"strategy_package_id"}),
        frozenset({"strategy_package_id", "formation_session", "component_id"}),
    ),
    "STRATEGY_SCORE_RUN": (frozenset({"score_plan_hash"}), frozenset({"score_plan_hash"})),
    # A Task, or a strategy's latest, never another strategy's.
    "STRATEGY_SCORE_READBACK": (frozenset(), frozenset({"task_id", "strategy_package_id"})),
    "DATA_UPDATE_RUN": (frozenset({"update_plan_hash"}), frozenset({"update_plan_hash"})),
    "DATA_UPDATE_READBACK": (frozenset(), frozenset({"task_id"})),
    "CONTROLS": (frozenset(), frozenset({"strategy_package_id"})),
    "PLAN": (frozenset({"spec"}), frozenset({"spec"})),
    "RUN": (frozenset({"spec"}), frozenset({"spec"})),
    "STATUS": (frozenset({"task_id"}), frozenset({"task_id", "wait_seconds", "status_section"})),
    "TASK_RECOVERY": (frozenset({"task_id"}), frozenset({"task_id"})),
    "TASK_REMEDIATE": (
        frozenset({"task_id", "incident_key", "remedy"}),
        frozenset({"task_id", "incident_key", "remedy", "expected_task_hash"}),
    ),
    # A confirmation may carry the Task version it was made against; the owner
    # refuses to act on any other version instead of acting on whatever the Task became.
    "RECOVER": (frozenset({"task_id"}), frozenset({"task_id", "expected_task_hash"})),
    "TASKS": (
        frozenset(),
        frozenset({"agent_session", "history_limit", "history_cursor"}),
    ),
    "UPGRADE_ACKNOWLEDGE": (
        frozenset({"upgrade_set_hash"}),
        frozenset({"upgrade_set_hash"}),
    ),
    "CANCEL": (frozenset({"task_id"}), frozenset({"task_id", "expected_task_hash"})),
    "RESULTS": (frozenset(), frozenset({"task_id"})),
    "REPORT": (frozenset({"result_hash"}), frozenset({"result_hash", "portfolio_session"})),
    "PORTFOLIO_READBACK": (
        frozenset({"task_id"}),
        frozenset({"task_id", "portfolio_session", "performance", "portfolio_scope"}),
    ),
    "EXPERIMENT_DECLARATION_CONVERT": (
        frozenset(),
        frozenset({"experiment_document", "experiment_yaml"}),
    ),
    "COMPARE": (
        frozenset({"left_result_hash", "right_result_hash"}),
        frozenset({"left_result_hash", "right_result_hash"}),
    ),
    "FREEZE": (frozenset({"result_hash"}), frozenset({"result_hash"})),
    "FINALIZATION": (
        frozenset({"candidate_hash"}),
        frozenset({"candidate_hash"}),
    ),
    "EXPORT": (frozenset({"result_hash"}), frozenset({"result_hash"})),
    # The Evidence & CRO operations name at most one sealed book; with no
    # book named the owner picks the workspace default.
    "EVIDENCE_CRO": (
        frozenset(),
        frozenset(
            {
                "result_hash",
                "handoff_hash",
                "review_publication_hash",
                "citation_entity_id",
                "citation_unit_id",
                "citation_page",
                "evidence_detail",
                "view_entity_id",
                "view_topic",
                "view_last_days",
            }
        ),
    ),
    "EVIDENCE_CRO_EXPORT": (
        frozenset(),
        frozenset(
            {
                "result_hash",
                "handoff_hash",
                "review_publication_hash",
                "prior_review_publication_hash",
                "citation_entity_id",
                "citation_unit_id",
                "citation_page",
            }
        ),
    ),
    "EVIDENCE_REFRESH": (frozenset(), frozenset({"result_hash", "handoff_hash"})),
    # A captured preparation intent carries its cutoff and binding; a
    # request without them prepares as of now.
    "EVIDENCE_PREPARE": (
        frozenset(),
        frozenset({"result_hash", "handoff_hash", "evidence_as_of", "preparation_binding_hash"}),
    ),
    "EVIDENCE_PREVIEW": (frozenset(), frozenset({"result_hash", "handoff_hash"})),
    # The book's ledger is read a page of groups at a time.
    "EVIDENCE_LEDGER": (
        frozenset(),
        frozenset({"result_hash", "handoff_hash", "ledger_page"}),
    ),
    # The workspace's retained documents, every issuer's, a page at a time (A6).
    "EVIDENCE_DOCUMENTS": (frozenset(), frozenset({"documents_page"})),
    # A packet is one preparation's, or one unit's of a coverage run;
    # the unit rides with the Task id, and the answer names it back.
    # A part of a packet's delivery names its part, its budget and the
    # context it continues; a continuation for another packet is stale.
    "EVIDENCE_PACKET": (
        frozenset({"task_id"}),
        frozenset(
            {
                "task_id",
                "evidence_unit_id",
                "result_hash",
                "handoff_hash",
                "delivery_part",
                "delivery_budget_bytes",
                "analysis_context_hash",
                "evidence_detail",
                "view_entity_id",
                "view_topic",
                "view_last_days",
                "view_from",
                "view_to",
            }
        ),
    ),
    # One more bounded source-reading session over a prepared packet:
    # names the receipt and span set the packet sealed and the
    # cumulative allowance; never a page, never a position.
    "EVIDENCE_CONTINUE": (
        frozenset(
            {
                "task_id",
                "continuation_of",
                "continuation_spans",
                "session_limit",
                "window_limit",
            }
        ),
        frozenset(
            {
                "task_id",
                "evidence_unit_id",
                "continuation_of",
                "continuation_spans",
                "session_limit",
                "window_limit",
                "result_hash",
                "handoff_hash",
            }
        ),
    ),
    "EVIDENCE_ANALYSIS_SUBMIT": (
        frozenset({"task_id", "analysis_context_hash", "analysis_answer"}),
        frozenset(
            {
                "task_id",
                "evidence_unit_id",
                "analysis_context_hash",
                "analysis_answer",
                "result_hash",
                "handoff_hash",
            }
        ),
    ),
    "CRO_REVIEW": (frozenset(), frozenset({"result_hash", "handoff_hash"})),
    "CRO_REVIEW_DOSSIER": (
        frozenset(),
        frozenset(
            {
                "result_hash",
                "handoff_hash",
                "delivery_part",
                "delivery_budget_bytes",
                "review_dossier_hash",
                "review_read_at",
            }
        ),
    ),
    "CRO_REVIEW_FINDING": (
        frozenset({"finding_handle"}),
        frozenset(
            {
                "finding_handle",
                "review_dossier_hash",
                "review_read_at",
                "review_publication_hash",
                "result_hash",
                "handoff_hash",
            }
        ),
    ),
    "CRO_REVIEW_SUBMIT": (
        frozenset(
            {
                "review_dossier_hash",
                "review_policy_hash",
                "review_schema_hash",
                "review_answer",
            }
        ),
        frozenset(
            {
                "review_dossier_hash",
                "review_policy_hash",
                "review_schema_hash",
                "review_read_at",
                "review_answer",
                "result_hash",
                "handoff_hash",
            }
        ),
    ),
    # A bundle for one agent role: the Analyst's names its prepared
    # packet (Task and unit), the CRO's the book; the other specialists
    # name one retained Task. An answer names only the bundle's directory;
    # the Host holds its assignment and exact permitted references.
    "AGENT_BUNDLE_PREPARE": (
        frozenset({"agent_role", "bundle_directory"}),
        frozenset(
            {
                "agent_role",
                "bundle_directory",
                "task_id",
                "evidence_unit_id",
                "result_hash",
                "handoff_hash",
            }
        ),
    ),
    "AGENT_ANSWER_SUBMIT": (
        frozenset({"bundle_directory", "agent_answer"}),
        frozenset({"bundle_directory", "agent_answer", "committee_key"}),
    ),
    # Choosing a review's analysis alone must name a publication; the book is optional.
    "EVIDENCE_SELECT": (
        frozenset({"analysis_publication_hash"}),
        frozenset({"analysis_publication_hash", "result_hash", "handoff_hash"}),
    ),
}


def field_contract(
    operation: str, recovery: frozenset[str], person_decisions: frozenset[str]
) -> tuple[frozenset[str], frozenset[str]]:
    """Return an operation's required and allowed request fields."""
    # An operation the table does not name takes no field.
    required, allowed = requirements.get(operation, (frozenset(), frozenset()))
    if operation in {
        "EVIDENCE_CRO",
        "EVIDENCE_PREPARE",
        "EVIDENCE_PREVIEW",
        "EVIDENCE_PACKET",
        "EVIDENCE_LEDGER",
        "EVIDENCE_CONTINUE",
        "EVIDENCE_ANALYSIS_SUBMIT",
        "EVIDENCE_REFRESH",
        "EVIDENCE_SELECT",
        "CRO_REVIEW",
        "CRO_REVIEW_DOSSIER",
        "CRO_REVIEW_FINDING",
        "CRO_REVIEW_SUBMIT",
        "EVIDENCE_CRO_EXPORT",
        "AGENT_BUNDLE_PREPARE",
    }:
        allowed |= {
            "update_task_id",
            "update_publication_hash",
            "position_basis",
            "experiment_task_id",
            "experiment_receipt_hash",
            "portfolio_session",
        }
    if operation in recovery:
        allowed |= {"recovery_task_id", "recovery_task_hash"}
    if operation in person_decisions:
        allowed |= {"person_confirmation"}
    return required, frozenset(allowed)
