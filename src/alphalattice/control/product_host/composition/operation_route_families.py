"""Compose the one operation table from the Host's existing family owners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from alphalattice.interface.local_application.operations import OPERATIONS
from alphalattice.kernel.door import CLI_OPERATION, Door, DoorRefusal, FamilyRoute

if TYPE_CHECKING:
    from alphalattice.interface.local_application.cli_invocation import ForwardedCli
    from alphalattice.interface.local_application.portfolio_research import (
        OperationCaller,
        PortfolioResearchOperationRequest,
    )
    from alphalattice.protocols.actor_execution.answers import AgentRun
    from alphalattice.protocols.actor_execution.bundles import AgentBundleRecord
    from alphalattice.protocols.actor_execution.contracts import AgentExecutionBinding


@dataclass(frozen=True, slots=True)
class WorkspaceOperationCall[Owner]:
    """The Host, typed request and caller delivered to a workspace route family."""

    operations: Owner
    request: PortfolioResearchOperationRequest
    caller: OperationCaller
    agent_execution: AgentExecutionBinding | None = None
    raw_review: bool = False
    read_files: tuple[str, ...] | None = None
    agent_run: AgentRun | None = None
    accepted_bundle: AgentBundleRecord | None = None


_ROUTES = (
    FamilyRoute(
        (
            "CANCEL",
            "RECOVER",
            "STATUS",
            "TASKS",
            "TASK_GUARDIAN",
            "TASK_INCIDENTS",
            "TASK_RECOVERY",
            "TASK_REMEDIATE",
        ),
        "task_operations",
        "route",
    ),
    FamilyRoute(
        (
            "DATA_CHANGE_CONFIRM",
            "DATA_ISSUES",
            "DATA_ISSUE_CONFIRM",
            "DATA_ISSUE_DELEGATE",
            "DATA_ISSUE_PREVIEW",
            "DATA_ISSUE_REVOKE",
            "DATA_UPDATE_PLAN",
            "DATA_UPDATE_READBACK",
            "DATA_UPDATE_RUN",
            "STORAGE_CAP_SET",
            "STORAGE_CAP_SHOW",
            "STORAGE_CONFIRM",
            "STORAGE_EVIDENCE_REBUILD",
            "STORAGE_PIN",
            "STORAGE_PLAN",
            "STORAGE_READBACK",
        ),
        "data_operations",
        "route",
    ),
    FamilyRoute(
        (
            "FEATURE_CATALOG_BUILD",
            "FEATURE_CATALOG_BUILD_READBACK",
            "FEATURE_CATALOG_CONTROLS",
            "FEATURE_CATALOG_PLAN",
            "FEATURE_CATALOG_READBACK",
            "FEATURE_TRIAL",
            "FEATURE_TRIALS",
            "FEATURE_TRIAL_READBACK",
        ),
        "feature_operations",
        "route",
    ),
    FamilyRoute(
        (
            "MODEL_TRAINING_INPUT_PLAN",
            "MODEL_TRAINING_INPUT_PREPARE",
            "MODEL_TRAINING_INPUT_READBACK",
            "RESEARCH_INPUTS",
            "RESEARCH_INPUT_CONFIRM",
            "RESEARCH_INPUT_PLAN",
            "RESEARCH_INPUT_READBACK",
            "RESEARCH_STRATEGY_CONTROLS",
            "RESEARCH_STRATEGY_INSTALL",
            "RESEARCH_STRATEGY_PLAN",
            "RESEARCH_STRATEGY_PREPARE",
            "RESEARCH_STRATEGY_READBACK",
            "WORKSPACE_PREPARE_CONFIRM",
            "WORKSPACE_PREPARE_PLAN",
            "WORKSPACE_PREPARE_READBACK",
        ),
        "research_operations",
        "route",
    ),
    FamilyRoute(
        (
            "COMPARE",
            "CONTROLS",
            "EXPORT",
            "FINALIZATION",
            "FREEZE",
            "PLAN",
            "PORTFOLIO_UPDATE_PLAN",
            "PORTFOLIO_UPDATE_READBACK",
            "PORTFOLIO_UPDATE_RUN",
            "REPORT",
            "RESEARCH_UPDATE_AUTOMATION_CONFIGURE",
            "RESEARCH_UPDATE_AUTOMATION_READBACK",
            "RESEARCH_UPDATE_PLAN",
            "RESEARCH_UPDATE_READBACK",
            "RESEARCH_UPDATE_RUN",
            "RESULTS",
            "RUN",
            "STRATEGY_CALIBRATION_PLAN",
            "STRATEGY_CALIBRATION_READBACK",
            "STRATEGY_CALIBRATION_RUN",
            "STRATEGY_SCORE_PLAN",
            "STRATEGY_SCORE_READBACK",
            "STRATEGY_SCORE_RUN",
        ),
        "portfolio_operations",
        "route",
    ),
    FamilyRoute(
        (
            "AGENT_ANSWER_SUBMIT",
            "AGENT_BUNDLE_PREPARE",
            "CRO_REVIEW",
            "CRO_REVIEW_DOSSIER",
            "CRO_REVIEW_FINDING",
            "CRO_REVIEW_SUBMIT",
            "EVIDENCE_ANALYSIS_SUBMIT",
            "EVIDENCE_CONTINUE",
            "EVIDENCE_CRO",
            "EVIDENCE_CRO_EXPORT",
            "EVIDENCE_DOCUMENTS",
            "EVIDENCE_LEDGER",
            "EVIDENCE_PACKET",
            "EVIDENCE_PREPARE",
            "EVIDENCE_PREVIEW",
            "EVIDENCE_REFRESH",
            "EVIDENCE_SELECT",
        ),
        "review_operations",
        "route",
    ),
    FamilyRoute(
        (
            "ACTIVITY_LIST",
            "ACTIVITY_RECENT",
            "ANALYST_READS",
            "ANALYST_READS_SET",
            "CPU_BUDGET_SET",
            "CPU_BUDGET_SHOW",
            "EVENT_DECLARE",
            "HOST_STOP",
            "OPERATION_LIST",
            "SESSION_USAGE_READ",
            "USAGE_READING",
            "USAGE_READING_SET",
            "WAKE_REGISTER",
            "WORKSPACE_BACKUP",
            "WORKSPACE_BACKUPS",
            "WORKSPACE_SHOW",
        ),
        "portfolio_research_operations",
        "route_client",
    ),
    FamilyRoute(
        (
            "GOAL_ABANDON",
            "GOAL_ATTACH",
            "GOAL_CONTINUE",
            "GOAL_EXPORT",
            "GOAL_LIST",
            "GOAL_NARRATIVE",
            "GOAL_NOTE",
            "GOAL_OPEN",
            "GOAL_PROFILE",
            "GOAL_REFERENCE",
            "GOAL_REVISE",
            "GOAL_SCHEMA",
            "GOAL_SHOW",
            "GOAL_SUBMIT",
            "GOAL_TAKE",
        ),
        "portfolio_research_operations",
        "route_goal",
    ),
    FamilyRoute(
        (
            "EXPERIMENTS",
            "EXPERIMENT_ALPHA_COMPARE",
            "EXPERIMENT_COMPARE",
            "EXPERIMENT_CONTINUE",
            "EXPERIMENT_CONTROLS",
            "EXPERIMENT_CURATE",
            "EXPERIMENT_CURATION",
            "EXPERIMENT_DECLARATION_CONVERT",
            "EXPERIMENT_DELIVERY_EXPORT",
            "EXPERIMENT_DRAFT",
            "EXPERIMENT_EXPORT",
            "EXPERIMENT_FOUNDATIONS",
            "EXPERIMENT_FOUNDATION_DRAFT",
            "EXPERIMENT_FOUNDATION_EXPORT",
            "EXPERIMENT_FOUNDATION_PREVIEW",
            "EXPERIMENT_FOUNDATION_READBACK",
            "EXPERIMENT_FOUNDATION_SEAL",
            "EXPERIMENT_FOUNDATION_SUMMARY",
            "EXPERIMENT_HANDOFF_PREVIEW",
            "EXPERIMENT_LINK_RISK",
            "EXPERIMENT_PLAN",
            "EXPERIMENT_PORTFOLIO_DRAFT",
            "EXPERIMENT_PREVIEW_READBACK",
            "EXPERIMENT_PROMOTE",
            "EXPERIMENT_READBACK",
            "EXPERIMENT_REPLAY",
            "EXPERIMENT_RISK_EXPORT",
            "EXPERIMENT_RISK_LINKS",
            "EXPERIMENT_RUN",
            "EXPERIMENT_SUMMARY",
            "EXPERIMENT_VERIFY_ALL",
        ),
        "research_experiments",
        "route",
    ),
    FamilyRoute(
        (
            "EXPERIENCE_BRIEF",
            "EXPERIENCE_CLASS",
            "EXPERIENCE_READ",
            "EXPERIENCE_RECALL",
            "EXPERIENCE_RECORD",
            "EXPERIENCE_SQL",
        ),
        "experience_read",
        "route",
    ),
    FamilyRoute(
        (
            "COMMITTEE_DAILY_SET",
            "COMMITTEE_OPEN",
            "COMMITTEE_READ",
            "COMMITTEE_SUBMIT",
        ),
        "portfolio_research_operations",
        "route_committee",
    ),
    FamilyRoute(
        (
            "ACTIVITY_REFUSALS",
            "EVIDENCE_CONSENT",
            "EVIDENCE_CONSENT_SET",
            "EVIDENCE_INSTALL",
            "FEATURE_ACTIVATE",
            "FEATURE_DEACTIVATE",
            "FEATURE_EXTENSIONS",
            "FEATURE_REVIEW",
            "MODEL_ACTIVATE",
            "MODEL_DEACTIVATE",
            "MODEL_EXTENSIONS",
            "NETWORK_ACCESS",
            "NETWORK_ACCESS_SET",
            "PENDING_DECISIONS",
            "PORTFOLIO_READBACK",
            "RESEARCH_HISTORY",
            "STRATEGY_ACTIVATE",
            "STRATEGY_DEACTIVATE",
            "UPGRADE_ACKNOWLEDGE",
            "UPGRADE_OVERVIEW",
        ),
        "portfolio_research_operations",
        "route_host",
    ),
)


def host_door() -> Door[WorkspaceOperationCall[object] | ForwardedCli, dict[str, object]]:
    """Bind the existing operation vocabulary to composition's lazy family registrations."""
    families = (
        FamilyRoute(route.operations, f"{__package__}.{route.module}", route.handler)
        for route in _ROUTES
    )
    invocation = FamilyRoute(
        (CLI_OPERATION,), "alphalattice.interface.local_application.cli", "execute_forwarded"
    )
    door: Door[WorkspaceOperationCall[object] | ForwardedCli, dict[str, object]] = Door(
        (*families, invocation)
    )
    public = {*OPERATIONS, CLI_OPERATION}
    if missing := public - door.operations:
        raise DoorRefusal("door.operation_unregistered", min(missing))
    if added := door.operations - public:
        raise DoorRefusal("door.operation_unrecognized", min(added))
    return door


def workspace_operation(operation: str) -> bool:
    """Whether this is one of the workspace families used by the preparation readers."""
    return operation in _WORKSPACE_OPERATIONS


def client_operation(operation: str) -> bool:
    """Whether the operation reads or controls the Host without reopening research inputs."""
    return operation in _CLIENT_OPERATIONS


_WORKSPACE_OPERATIONS = frozenset(
    operation
    for family in _ROUTES
    if family.module
    in {"task_operations", "data_operations", "feature_operations", "research_operations"}
    for operation in family.operations
)
_CLIENT_OPERATIONS = frozenset(
    operation
    for family in _ROUTES
    if family.handler == "route_client"
    for operation in family.operations
)
