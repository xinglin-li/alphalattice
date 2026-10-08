"""The Local Application Service: one product boundary over the Host application.

Desktop, the thin CLI and an installed Agent tool all reach the Portfolio path
through this service. It adds no validation, no default and no formula of its
own -- it composes typed facts and hands them on -- so the three entry points
cannot drift into three slightly different products.

The Host application arrives as a **port**, not an import. Product Host is a
composition root and nothing in `src/` may depend on it; the structural guard
enforces that, and it is the right rule -- a composition root that other packages
import is no longer a root. The concrete application is wired in at the entry
point, which is also what makes this service testable without a workspace.

What the service adds is the researcher's vocabulary: `plan`, `run`, `compare`,
`with_study_window`, `full_support_reference`, `readouts`, `open_html`, `export`.
Each is one sentence over the port, and each returns a contract rather than a
rendered string, because the renderer is a consumer of facts and not a source of
them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from datetime import date, datetime
from typing import Any, Literal, Protocol, Self, get_args
from uuid import UUID

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.capabilities.portfolio_backtesting.metrics import (
    evaluate_net_simple_return_path,
)
from alphalattice.evidence.alternative_evidence.analysis.contracts import (
    AlternativeEvidenceAnalystAnswer,
    EvidenceTopic,
)
from alphalattice.investment.portfolio_strategy_lab.application.advancement import (
    WatermarkAdvancementProgram,
    WatermarkAdvancementReceipt,
)
from alphalattice.investment.portfolio_strategy_lab.application.contracts import (
    WORKSPACE_DEFAULT_STRATEGY_PACKAGE_ID,
    PortfolioDeclaredPathReport,
    PortfolioEconomicLedger,
    PortfolioExecutionLedger,
    PortfolioExecutionProgram,
    PortfolioPlanPreview,
    PortfolioReadouts,
    PortfolioResearchResult,
    PortfolioResearchSpec,
    PortfolioScheduleGuard,
    PortfolioStudyWindowGuard,
    PortfolioSupportCoverage,
    ScoreSourceMode,
)
from alphalattice.investment.portfolio_strategy_lab.application.controls import (
    FULL_SUPPORT_SELECTION,
    INSTALLED_PUBLIC_CONTROL_CATALOG,
    PublicControlCatalog,
    PublicControlDescriptor,
)
from alphalattice.investment.portfolio_strategy_lab.application.report_projection import (
    project_dated_position,
)
from alphalattice.oversight.chief_risk_officer.decision.portfolio_review import (
    PortfolioReviewAnswer,
)
from alphalattice.protocols.actor_execution.bundles import AgentRole


class LocalApplicationError(ValueError):
    """Stable refusal for a Local Application Service boundary failure."""


class FactorCurationChoice(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)
    factor_id: str = Field(min_length=1)
    role: Literal["CORE", "CONDITIONAL"]
    rationale: str = Field(min_length=1, max_length=500)


class FactorCurationRequest(BaseModel):  # type: ignore[misc]
    """Actor intent and the report it answers, never caller-supplied authority."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    expected_receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    choices: tuple[FactorCurationChoice, ...]
    limitations_acknowledged: tuple[str, ...]


class ResearchDeliveryCommentary(BaseModel):  # type: ignore[misc]
    """Attributed caller text, never a product-verified actor identity or decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    attribution: str = Field(min_length=1, max_length=160)
    text: str = Field(min_length=1, max_length=20_000)


type PortfolioResearchOperation = Literal[
    "GOAL_SCHEMA",
    "GOAL_LIST",
    "GOAL_OPEN",
    "GOAL_REVISE",
    "GOAL_SHOW",
    "GOAL_NARRATIVE",
    "GOAL_REFERENCE",
    "GOAL_ATTACH",
    "GOAL_NOTE",
    "GOAL_EXPORT",
    "GOAL_CONTINUE",
    "GOAL_TAKE",
    "GOAL_SUBMIT",
    "GOAL_ABANDON",
    "RESEARCH_HISTORY",
    "EXPERIMENT_COMPARE",
    "EXPERIMENT_ALPHA_COMPARE",
    "EXPERIMENT_DELIVERY_EXPORT",
    "RESEARCH_INPUTS",
    "RESEARCH_INPUT_PLAN",
    "RESEARCH_INPUT_CONFIRM",
    "RESEARCH_INPUT_READBACK",
    "MODEL_TRAINING_INPUT_PLAN",
    "MODEL_TRAINING_INPUT_PREPARE",
    "MODEL_TRAINING_INPUT_READBACK",
    "FEATURE_CATALOG_CONTROLS",
    "FEATURE_CATALOG_PLAN",
    "FEATURE_CATALOG_READBACK",
    "FEATURE_CATALOG_BUILD",
    "FEATURE_CATALOG_BUILD_READBACK",
    "FEATURE_TRIAL",
    "FEATURE_REVIEW",
    "FEATURE_ACTIVATE",
    "FEATURE_DEACTIVATE",
    "FEATURE_EXTENSIONS",
    "FEATURE_TRIAL_READBACK",
    "FEATURE_TRIALS",
    "RESEARCH_STRATEGY_PLAN",
    "RESEARCH_STRATEGY_CONTROLS",
    "RESEARCH_STRATEGY_PREPARE",
    "RESEARCH_STRATEGY_READBACK",
    "RESEARCH_STRATEGY_INSTALL",
    "EXPERIMENT_DRAFT",
    "EXPERIMENT_PROMOTE",
    "EXPERIMENT_CONTINUE",
    "WORKSPACE_PREPARE_READBACK",
    "WORKSPACE_PREPARE_PLAN",
    "WORKSPACE_PREPARE_CONFIRM",
    "STORAGE_CAP_SHOW",
    "STORAGE_CAP_SET",
    "STORAGE_READBACK",
    "STORAGE_PLAN",
    "STORAGE_CONFIRM",
    "STORAGE_PIN",
    "STORAGE_EVIDENCE_REBUILD",
    "EXPERIMENT_CONTROLS",
    "EXPERIMENT_LINK_RISK",
    "EXPERIMENT_RISK_LINKS",
    "EXPERIMENT_RISK_EXPORT",
    "EXPERIMENT_PLAN",
    "EXPERIMENT_PREVIEW_READBACK",
    "EXPERIMENT_RUN",
    "EXPERIMENTS",
    "EXPERIMENT_READBACK",
    "EXPERIMENT_SUMMARY",
    "EXPERIMENT_REPLAY",
    "EXPERIMENT_EXPORT",
    "EXPERIMENT_CURATION",
    "EXPERIMENT_CURATE",
    "EXPERIMENT_HANDOFF_PREVIEW",
    "EXPERIMENT_FOUNDATIONS",
    "EXPERIMENT_FOUNDATION_PREVIEW",
    "EXPERIMENT_FOUNDATION_SEAL",
    "EXPERIMENT_FOUNDATION_READBACK",
    "EXPERIMENT_FOUNDATION_SUMMARY",
    "EXPERIMENT_FOUNDATION_EXPORT",
    "EXPERIMENT_FOUNDATION_DRAFT",
    "EXPERIMENT_PORTFOLIO_DRAFT",
    "EXPERIMENT_VERIFY_ALL",
    "CONTROLS",
    "PLAN",
    "RUN",
    "STATUS",
    "TASK_RECOVERY",
    "TASK_GUARDIAN",
    "TASK_INCIDENTS",
    "TASK_REMEDIATE",
    "RECOVER",
    "TASKS",
    "UPGRADE_OVERVIEW",
    "UPGRADE_ACKNOWLEDGE",
    "NETWORK_ACCESS",
    "NETWORK_ACCESS_SET",
    "PENDING_DECISIONS",
    "ACTIVITY_REFUSALS",
    "CANCEL",
    "RESULTS",
    "REPORT",
    "COMPARE",
    "FREEZE",
    "FINALIZATION",
    "EXPORT",
    "EVIDENCE_CRO",
    "EVIDENCE_CRO_EXPORT",
    "EVIDENCE_REFRESH",
    "EVIDENCE_PREPARE",
    "EVIDENCE_PREVIEW",
    "EVIDENCE_PACKET",
    "EVIDENCE_LEDGER",
    "EVIDENCE_DOCUMENTS",
    "EVIDENCE_CONTINUE",
    "EVIDENCE_ANALYSIS_SUBMIT",
    "EVIDENCE_SELECT",
    "CRO_REVIEW",
    "CRO_REVIEW_DOSSIER",
    "CRO_REVIEW_FINDING",
    "CRO_REVIEW_SUBMIT",
    "AGENT_BUNDLE_PREPARE",
    "AGENT_ANSWER_SUBMIT",
    "DATA_UPDATE_PLAN",
    "DATA_UPDATE_RUN",
    "DATA_UPDATE_READBACK",
    "DATA_CHANGE_CONFIRM",
    "DATA_ISSUES",
    "DATA_ISSUE_PREVIEW",
    "DATA_ISSUE_CONFIRM",
    "DATA_ISSUE_DELEGATE",
    "DATA_ISSUE_REVOKE",
    "STRATEGY_SCORE_PLAN",
    "STRATEGY_SCORE_RUN",
    "STRATEGY_SCORE_READBACK",
    "STRATEGY_CALIBRATION_PLAN",
    "STRATEGY_CALIBRATION_RUN",
    "STRATEGY_CALIBRATION_READBACK",
    "PORTFOLIO_UPDATE_PLAN",
    "PORTFOLIO_UPDATE_RUN",
    "PORTFOLIO_UPDATE_READBACK",
    "RESEARCH_UPDATE_PLAN",
    "RESEARCH_UPDATE_RUN",
    "RESEARCH_UPDATE_READBACK",
    "RESEARCH_UPDATE_AUTOMATION_READBACK",
    "RESEARCH_UPDATE_AUTOMATION_CONFIGURE",
    "WORKSPACE_SHOW",
    "OPERATION_LIST",
    "ACTIVITY_LIST",
    "ACTIVITY_RECENT",
    "EVENT_DECLARE",
    "SESSION_USAGE_READ",
    "USAGE_READING",
    "USAGE_READING_SET",
    "CPU_BUDGET_SHOW",
    "CPU_BUDGET_SET",
    "WORKSPACE_BACKUP",
    "WORKSPACE_BACKUPS",
    "MODEL_EXTENSIONS",
    "MODEL_ACTIVATE",
    "MODEL_DEACTIVATE",
    "STRATEGY_ACTIVATE",
    "STRATEGY_DEACTIVATE",
]

_RECOVERY_CONTEXT_OPERATIONS = frozenset(
    {
        "PLAN",
        "RUN",
        "EXPERIMENT_PLAN",
        "EXPERIMENT_RUN",
        "EXPERIMENT_VERIFY_ALL",
        "PORTFOLIO_UPDATE_PLAN",
        "PORTFOLIO_UPDATE_RUN",
        "RESEARCH_UPDATE_PLAN",
        "RESEARCH_UPDATE_RUN",
        "STRATEGY_CALIBRATION_PLAN",
        "STRATEGY_CALIBRATION_RUN",
        "STRATEGY_SCORE_PLAN",
        "STRATEGY_SCORE_RUN",
        "RESEARCH_INPUT_PLAN",
        "RESEARCH_INPUT_CONFIRM",
        "MODEL_TRAINING_INPUT_PLAN",
        "MODEL_TRAINING_INPUT_PREPARE",
        "WORKSPACE_PREPARE_PLAN",
        "WORKSPACE_PREPARE_CONFIRM",
        "DATA_UPDATE_PLAN",
        "DATA_UPDATE_RUN",
        "RESEARCH_STRATEGY_PLAN",
        "RESEARCH_STRATEGY_PREPARE",
        "FEATURE_CATALOG_PLAN",
        "FEATURE_CATALOG_BUILD",
        "EVIDENCE_PREVIEW",
        "EVIDENCE_PREPARE",
    }
)
"""Existing Task planner/admission doors that can carry common recovery context."""


@dataclass(frozen=True, slots=True)
class PortfolioResearchOperationRequest:
    """Actor-neutral request shared by browser and optional Agent adapter."""

    operation: PortfolioResearchOperation
    goal_id: UUID | None = None
    goal_hash: str | None = None
    goal_reference_id: str | None = None
    goal_declaration: dict[str, Any] | None = None
    goal_reference: dict[str, Any] | None = None
    goal_statement: dict[str, Any] | None = None
    goal_submission: dict[str, Any] | None = None
    change_reason: str | None = None
    history_limit: int | None = None
    history_cursor: str | None = None
    history_entry_id: str | None = None
    history_kind: str | None = None
    left_task_id: UUID | None = None
    right_task_id: UUID | None = None
    left_candidate_id: str | None = None
    right_candidate_id: str | None = None
    delivery_question: str | None = None
    delivery_commentary: tuple[ResearchDeliveryCommentary, ...] | None = None
    research_input_plan_hash: str | None = None
    origin_task_id: UUID | None = None
    factor_task_id: UUID | None = None
    preparation_plan_hash: str | None = None
    data_issue_case_token: str | None = None
    data_issue_evidence_hash: str | None = None
    data_issue_option_id: str | None = None
    data_issue_option_hash: str | None = None
    data_issue_grant_hash: str | None = None
    storage_plan_hash: str | None = None
    input_binding_hash: str | None = None
    input_pinned: bool | None = None
    evidence_index_id: str | None = None
    research_input_id: str | None = None
    experiment_document: dict[str, Any] | None = None
    feature_document: dict[str, Any] | None = None
    feature_plan_hash: str | None = None
    feature_output: Literal["RAW_VALUES", "PREPROCESSED_VALUES"] | None = None
    feature_preparation_hash: str | None = None
    experiment_kind: str | None = None
    risk_task_id: UUID | None = None
    risk_report_hash: str | None = None
    risk_report_scope: Literal["POST_OBSERVED_PORTFOLIO_WINDOW"] | None = None
    experiment_yaml: str | None = None
    experiment_plan_hash: str | None = None
    experiment_curation: FactorCurationRequest | None = None
    curation_receipt_hash: str | None = None
    foundation_admission_hash: str | None = None
    candidate_id: str | None = None
    portfolio_session: str | None = None
    spec: dict[str, object] | None = None
    strategy_package_id: str | None = None
    component_id: str | None = None
    model_id: str | None = None
    task_id: UUID | None = None
    expected_task_hash: str | None = None
    recovery_task_id: UUID | None = None
    recovery_task_hash: str | None = None
    incident_key: str | None = None
    """An incident the Supervisor keeps (GY2), by its record's key."""
    remedy: Literal["CANCEL", "RECOVER", "REPLAN"] | None = None
    """The remedy the Host offered for that incident, chosen."""
    result_hash: str | None = None
    left_result_hash: str | None = None
    right_result_hash: str | None = None
    candidate_hash: str | None = None
    handoff_hash: str | None = None
    update_task_id: UUID | None = None
    experiment_task_id: UUID | None = None
    experiment_receipt_hash: str | None = None
    update_publication_hash: str | None = None
    position_basis: str | None = None
    review_publication_hash: str | None = None
    analysis_publication_hash: str | None = None
    review_dossier_hash: str | None = None
    review_policy_hash: str | None = None
    review_schema_hash: str | None = None
    review_read_at: str | None = None
    """When the review's dossier was read (an aware ISO time), sealed with its bundle or its
    first part: a later read of the same review resolves the dossier at that time, never the
    clock."""
    review_answer: Any = None
    """The CRO's answer as written (`PortfolioReviewAnswer`); the Host screens
    it item by item, so it is carried raw."""
    analysis_context_hash: str | None = None
    analysis_answer: Any = None
    """The Analyst's answer as written (`AlternativeEvidenceAnalystAnswer`);
    the Host screens it item by item, so it is carried raw."""
    agent_role: AgentRole | None = None
    bundle_directory: str | None = None
    """The absolute directory a bundle was written to: named when it is
    prepared, and named again -- never a hash -- when an answer to it is sent."""
    agent_answer: Any = None
    """An answer to the named bundle, as written; the bundle's role decides
    which answer it is."""
    evidence_as_of: str | None = None
    preparation_binding_hash: str | None = None
    evidence_unit_id: str | None = None
    """One unit of a coverage run (`u01`..): the packet asked for, the packet answered."""
    delivery_part: int | None = None
    """Which part of a bounded delivery to read; absent means the whole, or part 1."""
    delivery_budget_bytes: int | None = None
    """The consumer's own byte budget for one response; absent means the Host default."""
    evidence_detail: str | None = None
    """A bounded detail of a prepared packet delivered instead of the whole
    packet: `session_windows` (only the matter windows the packet's own reading
    session read), `topic_coverage` (the issuer-topic ledger with every
    delivered bundle) or `time_view` (the bundles whose document was
    accepted in a view interval ending at the evidence cutoff, with the
    unfiltered ledger beside them); absent means the packet. On `EVIDENCE_CRO`,
    `time_view` requests a separate reading of the published review's delivered evidence."""
    view_entity_id: str | None = None
    view_topic: str | None = None
    view_last_days: int | None = None
    view_from: str | None = None
    view_to: str | None = None
    """The time view's filters: one issuer, one topic, the last N days ending
    at the run's evidence cutoff (30 when none is given for a packet), or an explicit
    ISO-8601 interval whose end never passes the cutoff. A published Reading with no
    day interval reads all delivered passages; each analysis keeps its own cutoff. Read-only."""
    ledger_page: int | None = None
    """The page of a book ledger's groups (`EVIDENCE_LEDGER`), from 1."""
    documents_page: int | None = None
    """The page of the workspace's retained documents (`EVIDENCE_DOCUMENTS`), from 1."""
    citation_entity_id: str | None = None
    citation_unit_id: str | None = None
    citation_page: int | None = None
    """A page of the section's citations (`EVIDENCE_CRO`) or of a review
    export's verified spans (`EVIDENCE_CRO_EXPORT`), for one issuer or one
    group, from 1; the answer is the page alone."""
    continuation_of: str | None = None
    continuation_spans: str | None = None
    """The access receipt and span set a source-reading continuation continues:
    the ones the packet's own delivery named, never a position."""
    session_limit: int | None = None
    window_limit: int | None = None
    """The cumulative allowance a continuation is requested under: reading
    sessions including the first, and matter windows over all of them."""
    finding_handle: str | None = None
    """One finding of the current dossier (or of a named review), for its evidence package."""
    prior_review_publication_hash: str | None = None
    """An explicitly selected earlier review an export reports its changes against."""
    update_plan_hash: str | None = None
    score_plan_hash: str | None = None
    score_snapshot_hash: str | None = None
    calibration_plan_hash: str | None = None
    prepared_input_hash: str | None = None
    observed_through: str | None = None
    formation_session: str | None = None
    automation_enabled: bool | None = None
    network_enabled: bool | None = None
    usage_reading_enabled: bool | None = None
    automation_package_ids: tuple[str, ...] | None = None
    upgrade_set_hash: str | None = None
    """The installed identity set an upgrade overview showed, when acknowledging it."""
    feature_trial_id: str | None = None
    """A feature trial, by the ID its request answered with."""
    feature_factor_id: str | None = None
    """A formula factor a research plan declares, which its review and activation name."""
    extensions_page: int | None = None
    """The page of the declared formula factors (`FEATURE_EXTENSIONS`), from 1."""
    agent_session: str | None = None
    """An agent session, whose submitted Tasks a Task listing shows, and whose goals a goal
    listing shows."""
    wait_seconds: float | None = None
    """How long a Task's status may wait for the Task to move on (a stage, a lifecycle)
    before it answers; at most 20 seconds. A follow asks with it instead of polling."""
    after: str | None = None
    limit: int | None = None
    watch: tuple[UUID, ...] | None = None
    event: dict[str, Any] | None = None
    storage_cap_bytes: str | None = None
    """Automatic workspace capacity or a positive whole byte count, never a sealed input."""
    cpu_budget: str | None = None
    tasks_waiting: str | None = None
    backup_generations_kept: int | None = None

    def __post_init__(self) -> None:
        """Validate operation fields and normalize delivery commentary."""
        if (self.recovery_task_id is None) != (self.recovery_task_hash is None):
            raise LocalApplicationError("portfolio_research.recovery_context_pair_required")
        if self.recovery_task_id is not None and not isinstance(self.recovery_task_id, UUID):
            raise LocalApplicationError("portfolio_research.recovery_task_id_invalid")
        if self.operation == "EXPERIMENT_DELIVERY_EXPORT" and (
            (self.left_task_id is None) != (self.right_task_id is None)
        ):
            raise LocalApplicationError("research_delivery.comparison_pair_required")
        if self.delivery_question is not None and (
            not isinstance(self.delivery_question, str)
            or not 1 <= len(self.delivery_question) <= 2400
        ):
            raise LocalApplicationError("research_delivery.question_invalid")
        if self.delivery_commentary is not None:
            if len(self.delivery_commentary) > 16:
                raise LocalApplicationError("research_delivery.commentary_limit_exceeded")
            object.__setattr__(
                self,
                "delivery_commentary",
                tuple(
                    value
                    if isinstance(value, ResearchDeliveryCommentary)
                    else ResearchDeliveryCommentary.model_validate(value)
                    for value in self.delivery_commentary
                ),
            )
        if self.history_limit is not None and (
            type(self.history_limit) is not int or not 1 <= self.history_limit <= 50
        ):
            raise LocalApplicationError("research_history.limit_outside_1_50")
        for value in (
            self.goal_hash,
            self.preparation_plan_hash,
            self.data_issue_case_token,
            self.data_issue_evidence_hash,
            self.data_issue_option_hash,
            self.data_issue_grant_hash,
            self.storage_plan_hash,
            self.input_binding_hash,
            self.research_input_plan_hash,
            self.foundation_admission_hash,
            self.risk_report_hash,
            self.review_dossier_hash,
            self.review_policy_hash,
            self.review_schema_hash,
            self.analysis_context_hash,
            self.preparation_binding_hash,
            self.evidence_index_id,
            self.continuation_of,
            self.continuation_spans,
            self.upgrade_set_hash,
            self.feature_trial_id,
            self.recovery_task_hash,
        ):
            if value is not None and (
                not isinstance(value, str)
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise LocalApplicationError("workspace.operation_hash_invalid")
        if self.input_pinned is not None and type(self.input_pinned) is not bool:
            raise LocalApplicationError("workspace.input_pin_flag_invalid")
        if self.agent_role is not None and self.agent_role not in get_args(AgentRole.__value__):
            raise LocalApplicationError("agent_bundle.role_unknown")
        if self.bundle_directory is not None and (
            not isinstance(self.bundle_directory, str)
            or not 1 <= len(self.bundle_directory) <= 1024
        ):
            raise LocalApplicationError("agent_bundle.directory_invalid")
        if self.evidence_unit_id is not None and (
            not isinstance(self.evidence_unit_id, str)
            or not 3 <= len(self.evidence_unit_id) <= 4
            or self.evidence_unit_id[0] != "u"
            or not self.evidence_unit_id[1:].isdigit()
        ):
            raise LocalApplicationError("alternative_evidence.coverage_unit_id_invalid")
        for name in ("delivery_part", "delivery_budget_bytes"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise LocalApplicationError("alternative_evidence.delivery_selector_invalid")
        if self.ledger_page is not None and (
            type(self.ledger_page) is not int or self.ledger_page < 1
        ):
            raise LocalApplicationError("alternative_evidence.ledger_page_invalid")
        if self.documents_page is not None and (
            type(self.documents_page) is not int or self.documents_page < 1
        ):
            raise LocalApplicationError("alternative_evidence.documents_page_invalid")
        if self.extensions_page is not None and (
            type(self.extensions_page) is not int or self.extensions_page < 1
        ):
            raise LocalApplicationError("feature_extension.extensions_page_invalid")
        if self.citation_page is not None and (
            type(self.citation_page) is not int or self.citation_page < 1
        ):
            raise LocalApplicationError("portfolio_research.citation_page_invalid")
        if self.citation_unit_id is not None and (
            not isinstance(self.citation_unit_id, str)
            or not 3 <= len(self.citation_unit_id) <= 4
            or self.citation_unit_id[0] != "u"
            or not self.citation_unit_id[1:].isdigit()
        ):
            raise LocalApplicationError("alternative_evidence.coverage_unit_id_invalid")
        if self.evidence_detail is not None and self.evidence_detail not in {
            "session_windows",
            "topic_coverage",
            "time_view",
        }:
            raise LocalApplicationError("alternative_evidence.evidence_detail_unknown")
        if self.view_last_days is not None and (
            type(self.view_last_days) is not int or not 1 <= self.view_last_days <= 3660
        ):
            raise LocalApplicationError("alternative_evidence.view_interval_invalid")
        for name in ("view_from", "view_to"):
            value = getattr(self, name)
            if value is not None:
                try:
                    parsed = datetime.fromisoformat(str(value))
                except ValueError as error:
                    raise LocalApplicationError(
                        "alternative_evidence.view_interval_invalid"
                    ) from error
                if parsed.tzinfo is None:
                    raise LocalApplicationError("alternative_evidence.view_interval_invalid")
        if self.view_topic is not None and self.view_topic not in {
            value.value for value in EvidenceTopic
        }:
            raise LocalApplicationError("alternative_evidence.view_topic_unknown")
        if self.wait_seconds is not None and (
            isinstance(self.wait_seconds, bool)
            or not isinstance(self.wait_seconds, int | float)
            or not 0 < self.wait_seconds <= 20
        ):
            raise LocalApplicationError("local_application.wait_seconds_invalid")
        for name, ceiling in (("session_limit", 99), ("window_limit", 4096)):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or not 1 <= value <= ceiling):
                raise LocalApplicationError("alternative_evidence.continuation_limit_invalid")
        if self.evidence_as_of is not None:
            try:
                parsed = datetime.fromisoformat(str(self.evidence_as_of))
            except ValueError as error:
                raise LocalApplicationError("alternative_evidence.cutoff_invalid") from error
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                raise LocalApplicationError("alternative_evidence.cutoff_invalid")
        if self.review_read_at is not None:
            try:
                parsed = datetime.fromisoformat(str(self.review_read_at))
            except ValueError as error:
                raise LocalApplicationError("chief_risk_officer.review_read_at_invalid") from error
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                raise LocalApplicationError("chief_risk_officer.review_read_at_invalid")
        if self.experiment_curation is not None and not isinstance(
            self.experiment_curation, FactorCurationRequest
        ):
            object.__setattr__(
                self,
                "experiment_curation",
                FactorCurationRequest.model_validate(self.experiment_curation),
            )
        if self.experiment_document is not None and not isinstance(self.experiment_document, dict):
            raise LocalApplicationError("research_experiment.document_not_a_mapping")
        if self.experiment_yaml is not None and not isinstance(self.experiment_yaml, str):
            raise LocalApplicationError("research_experiment.yaml_not_text")
        required, allowed = self.field_contract(self.operation)
        supplied = {
            field.name
            for field in fields(self)
            if field.name != "operation" and getattr(self, field.name) is not None
        }
        missing = sorted(required - supplied)
        if missing:
            raise LocalApplicationError(
                "portfolio_research.operation_field_required:" + ",".join(missing)
            )
        unexpected = sorted(supplied - allowed)
        if unexpected:
            raise LocalApplicationError(
                "portfolio_research.operation_field_not_allowed:" + ",".join(unexpected)
            )

    @staticmethod
    def field_contract(
        operation: PortfolioResearchOperation,
    ) -> tuple[frozenset[str], frozenset[str]]:
        """The same required/allowed fields drive validation and discovery."""
        data_issue_fields = frozenset(
            {
                "data_issue_case_token",
                "data_issue_evidence_hash",
                "data_issue_option_id",
                "data_issue_option_hash",
            }
        )
        requirements = {
            "GOAL_SCHEMA": (frozenset(), frozenset()),
            "GOAL_LIST": (
                frozenset(),
                frozenset({"agent_session", "history_limit", "history_cursor"}),
            ),
            "GOAL_OPEN": (
                frozenset({"goal_declaration"}),
                frozenset({"goal_id", "goal_declaration", "change_reason"}),
            ),
            # A bound session's goal and head are the defaults, so neither is required (V391).
            "GOAL_REVISE": (
                frozenset({"goal_declaration", "change_reason"}),
                frozenset({"goal_id", "goal_hash", "goal_declaration", "change_reason"}),
            ),
            **{
                # An exact revision by its hash, or the goal's current head by its id.
                op: (frozenset(), frozenset({"goal_id", "goal_hash"}))
                for op in ("GOAL_SHOW", "GOAL_NARRATIVE", "GOAL_EXPORT")
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
                frozenset({"task_id", "experiment_receipt_hash", "portfolio_session"}),
                frozenset(
                    {
                        "task_id",
                        "experiment_receipt_hash",
                        "portfolio_session",
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
            "EXPERIMENT_FOUNDATIONS": (frozenset(), frozenset()),
            "EXPERIMENT_VERIFY_ALL": (frozenset(), frozenset()),
            "EXPERIMENT_FOUNDATION_PREVIEW": (
                frozenset(
                    {"task_id", "curation_receipt_hash", "research_input_id", "input_binding_hash"}
                ),
                frozenset(
                    {"task_id", "curation_receipt_hash", "research_input_id", "input_binding_hash"}
                ),
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
            "RESEARCH_INPUTS": (frozenset(), frozenset()),
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
            "NETWORK_ACCESS": (frozenset(), frozenset()),
            "PENDING_DECISIONS": (frozenset(), frozenset()),
            "ACTIVITY_REFUSALS": (frozenset(), frozenset({"view_last_days"})),
            "NETWORK_ACCESS_SET": (frozenset({"network_enabled"}), frozenset({"network_enabled"})),
            "WORKSPACE_PREPARE_READBACK": (frozenset(), frozenset({"task_id"})),
            "WORKSPACE_PREPARE_PLAN": (frozenset(), frozenset()),
            "WORKSPACE_PREPARE_CONFIRM": (
                frozenset({"preparation_plan_hash"}),
                frozenset({"preparation_plan_hash", "data_issue_grant_hash"}),
            ),
            "STORAGE_CAP_SHOW": (frozenset(), frozenset()),
            "STORAGE_CAP_SET": (frozenset({"storage_cap_bytes"}), frozenset({"storage_cap_bytes"})),
            "STORAGE_READBACK": (frozenset(), frozenset()),
            "STORAGE_PLAN": (frozenset(), frozenset()),
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
            "FEATURE_TRIALS": (frozenset(), frozenset()),
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
            # Every declared formula factor, a page at a time (U56).
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
                frozenset({"research_input_id", "input_binding_hash", "component_id"}),
            ),
            "RESEARCH_STRATEGY_PLAN": (
                frozenset({"experiment_document"}),
                frozenset({"experiment_document"}),
            ),
            "RESEARCH_STRATEGY_CONTROLS": (frozenset(), frozenset()),
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
                frozenset({"experiment_plan_hash"}),
            ),
            "EXPERIMENT_PREVIEW_READBACK": (
                frozenset({"experiment_plan_hash"}),
                frozenset({"experiment_plan_hash"}),
            ),
            "EXPERIMENTS": (frozenset(), frozenset()),
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
            "RESEARCH_UPDATE_AUTOMATION_READBACK": (frozenset(), frozenset()),
            "RESEARCH_UPDATE_AUTOMATION_CONFIGURE": (
                frozenset({"automation_enabled", "automation_package_ids"}),
                frozenset({"automation_enabled", "automation_package_ids"}),
            ),
            "RESEARCH_UPDATE_PLAN": (
                frozenset({"strategy_package_id"}),
                frozenset({"strategy_package_id", "observed_through"}),
            ),
            "RESEARCH_UPDATE_RUN": (
                frozenset({"update_plan_hash"}),
                frozenset({"update_plan_hash"}),
            ),
            # A Task, or a strategy's latest update, never another strategy's (V595).
            "RESEARCH_UPDATE_READBACK": (
                frozenset(),
                frozenset({"task_id", "strategy_package_id"}),
            ),
            # The client's own commands, operations like any other (V266, OP1).
            "WORKSPACE_SHOW": (frozenset(), frozenset()),
            "OPERATION_LIST": (frozenset(), frozenset({"strategy_package_id"})),
            "ACTIVITY_LIST": (frozenset(), frozenset({"after", "limit", "watch"})),
            "ACTIVITY_RECENT": (frozenset(), frozenset({"limit"})),
            "EVENT_DECLARE": (frozenset({"event"}), frozenset({"event"})),
            "SESSION_USAGE_READ": (frozenset(), frozenset()),
            "USAGE_READING": (frozenset(), frozenset()),
            "USAGE_READING_SET": (
                frozenset({"usage_reading_enabled"}),
                frozenset({"usage_reading_enabled"}),
            ),
            "CPU_BUDGET_SHOW": (frozenset(), frozenset()),
            # One of the two, which the owner holds to: a setting a request (V100).
            "CPU_BUDGET_SET": (frozenset(), frozenset({"cpu_budget", "tasks_waiting"})),
            # The held state's backup outside the workspace (V209); its restore is the
            # client's own, with no Host (V328).
            "WORKSPACE_BACKUP": (frozenset(), frozenset({"backup_generations_kept"})),
            "WORKSPACE_BACKUPS": (frozenset(), frozenset()),
            "MODEL_EXTENSIONS": (frozenset(), frozenset()),
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
            # A Task, or a strategy's latest update, never another strategy's (V595).
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
            # A Task, or a strategy's latest, never another strategy's (V595).
            "STRATEGY_CALIBRATION_READBACK": (
                frozenset(),
                frozenset({"task_id", "strategy_package_id"}),
            ),
            "STRATEGY_SCORE_PLAN": (
                frozenset({"strategy_package_id"}),
                frozenset({"strategy_package_id", "formation_session", "component_id"}),
            ),
            "STRATEGY_SCORE_RUN": (frozenset({"score_plan_hash"}), frozenset({"score_plan_hash"})),
            # A Task, or a strategy's latest, never another strategy's (V595).
            "STRATEGY_SCORE_READBACK": (frozenset(), frozenset({"task_id", "strategy_package_id"})),
            "DATA_UPDATE_PLAN": (frozenset(), frozenset()),
            "DATA_UPDATE_RUN": (frozenset({"update_plan_hash"}), frozenset({"update_plan_hash"})),
            "DATA_UPDATE_READBACK": (frozenset(), frozenset({"task_id"})),
            "CONTROLS": (frozenset(), frozenset({"strategy_package_id"})),
            "PLAN": (frozenset({"spec"}), frozenset({"spec"})),
            "RUN": (frozenset({"spec"}), frozenset({"spec"})),
            "STATUS": (frozenset({"task_id"}), frozenset({"task_id", "wait_seconds"})),
            "TASK_RECOVERY": (frozenset({"task_id"}), frozenset({"task_id"})),
            "TASK_GUARDIAN": (frozenset(), frozenset()),
            "TASK_INCIDENTS": (frozenset(), frozenset()),
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
            "UPGRADE_OVERVIEW": (frozenset(), frozenset()),
            "UPGRADE_ACKNOWLEDGE": (
                frozenset({"upgrade_set_hash"}),
                frozenset({"upgrade_set_hash"}),
            ),
            "CANCEL": (frozenset({"task_id"}), frozenset({"task_id", "expected_task_hash"})),
            "RESULTS": (frozenset(), frozenset({"task_id"})),
            "REPORT": (frozenset({"result_hash"}), frozenset({"result_hash", "portfolio_session"})),
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
                frozenset(
                    {"result_hash", "handoff_hash", "evidence_as_of", "preparation_binding_hash"}
                ),
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
                frozenset({"bundle_directory", "agent_answer"}),
            ),
            # Choosing which analysis a review reads is the one operation that
            # must name a publication; the book stays optional as above.
            "EVIDENCE_SELECT": (
                frozenset({"analysis_publication_hash"}),
                frozenset({"analysis_publication_hash", "result_hash", "handoff_hash"}),
            ),
        }
        required, allowed = requirements[operation]
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
        if operation in _RECOVERY_CONTEXT_OPERATIONS:
            allowed |= {"recovery_task_id", "recovery_task_hash"}
        return required, frozenset(allowed)


class PortfolioResearchRequestDocument(BaseModel):  # type: ignore[misc]
    """One typed external document for the existing actor-neutral operation request.

    Each field's meaning is its docstring, which `alphalattice schema show` prints (SC3).
    """

    model_config = ConfigDict(extra="forbid", frozen=True, use_attribute_docstrings=True)

    operation: PortfolioResearchOperation
    """The operation asked for; `alphalattice schema show <name>` lists its fields."""
    goal_id: UUID | None = None
    """The goal, by the id its GOAL_OPEN answered with."""
    goal_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The goal's version as last read; a goal that moved since refuses the change."""
    goal_reference_id: str | None = Field(default=None, min_length=1, max_length=64)
    """The reference ID saved in the exact goal revision; only that reference is verified."""
    goal_declaration: dict[str, Any] | None = None
    """The goal as written: its statement, scope and completion rule (`goal schema`)."""
    goal_reference: dict[str, Any] | None = None
    """What to attach to the goal: a Task, a study, a result or a publication, by reference."""
    goal_statement: dict[str, Any] | None = None
    """A note for the goal's record, written by the agent working for it."""
    goal_submission: dict[str, Any] | None = None
    """The completion document claiming the goal is met, naming its evidence."""
    change_reason: str | None = None
    """Why the goal changes, kept in its record beside the change."""
    history_limit: int | None = Field(default=None, ge=1, le=50, strict=True)
    """How many entries one page reads, 1 to 50."""
    history_cursor: str | None = None
    """Where a list reads on: the `next_cursor` its previous page answered."""
    history_entry_id: str | None = Field(default=None, json_schema_extra={"reference": "issued"})
    """One research history entry, by the id the history listed."""
    history_kind: str | None = None
    """Only the history entries of this kind."""
    left_task_id: UUID | None = None
    """The first of two studies to compare, by its Task."""
    right_task_id: UUID | None = None
    """The second of two studies to compare, by its Task."""
    left_candidate_id: str | None = Field(default=None, min_length=1)
    """The first of two Alpha candidates to compare."""
    right_candidate_id: str | None = Field(default=None, min_length=1)
    """The second of two Alpha candidates to compare."""
    delivery_question: str | None = Field(default=None, min_length=1, max_length=2400)
    """The research question a delivery answers, in the researcher's words."""
    delivery_commentary: tuple[ResearchDeliveryCommentary, ...] | None = Field(
        default=None, max_length=16
    )
    """The researcher's comments a delivery carries, each on the part it names."""
    research_input_plan_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The research input plan to confirm, by the hash its PLAN answered."""
    origin_task_id: UUID | None = None
    """The study a new plan continues from, by its Task."""
    factor_task_id: UUID | None = None
    """The Factor study an Alpha or Foundation plan builds on, by its Task."""
    preparation_plan_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The workspace preparation plan to confirm, by the hash its PLAN answered."""
    data_issue_case_token: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The data issue, by the case token the issue list gave."""
    data_issue_evidence_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The evidence the issue was shown with; an issue that moved is refused."""
    data_issue_option_id: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_-]{2,119}$")
    """The option chosen for the data issue, one of those it offered."""
    data_issue_option_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The chosen option's hash as offered; a changed option is refused."""
    data_issue_grant_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """A person's recorded grant for a data decision, by its hash."""
    storage_plan_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The storage clean-up plan to confirm, by the hash its PLAN answered."""
    evidence_index_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The Evidence retrieval index to rebuild, by the id the storage view listed."""
    input_binding_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The research input's version, by its binding hash; with `research_input_id` it names the
    data a study reads."""
    input_pinned: bool | None = Field(default=None, strict=True)
    """true keeps the research input's version from clean-up; false lets clean-up release it."""
    research_input_id: str | None = None
    """The research input, by its id (RESEARCH_INPUTS lists them)."""
    experiment_document: dict[str, Any] | None = None
    """The study's declaration as a document, as `study controls` or `study draft`
    writes it."""
    feature_document: dict[str, Any] | None = None
    """A Feature definition as a document, for the Feature catalog to plan."""
    feature_plan_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """A Feature catalog plan, by the hash its PLAN answered."""
    feature_preparation_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The prepared Feature input whose controls a study reads, by its hash."""
    feature_output: Literal["RAW_VALUES", "PREPROCESSED_VALUES"] | None = None
    """What a Feature build writes: `RAW_VALUES`, or `PREPROCESSED_VALUES` after the catalog's
    preprocessing."""
    experiment_kind: str | None = None
    """The study kind whose controls to show, as `study controls` lists them; Factor when
    omitted."""
    risk_task_id: UUID | None = None
    """The Risk study to link, by its Task."""
    risk_report_hash: str | None = None
    """A Risk report, by its hash, to export or deliver."""
    experiment_yaml: str | None = None
    """The study's declaration as YAML text; send it or `experiment_document`, not both."""
    experiment_plan_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """A saved plan, by the hash its PLAN answered."""
    risk_report_scope: Literal["POST_OBSERVED_PORTFOLIO_WINDOW"] | None = None
    """`POST_OBSERVED_PORTFOLIO_WINDOW` links the Risk report over the Portfolio's observed
    window."""
    experiment_curation: FactorCurationRequest | None = None
    """The Factor curation decision: which factors go on, and why."""
    curation_receipt_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The Factor curation decision, by its receipt hash."""
    foundation_admission_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The Foundation, by its admission hash."""
    candidate_id: str | None = Field(default=None, min_length=1)
    """The Alpha candidate a Portfolio draft is built from."""
    portfolio_session: str | None = None
    """The session a dated view reads, as YYYY-MM-DD; for a book, names the book with the other
    book fields, as a book's offered requests fill them."""
    spec: dict[str, object] | None = None
    """The Portfolio book's controls (`strategy-book controls` lists them); those left out keep
    their defaults."""
    strategy_package_id: str | None = Field(default=None, min_length=1)
    """The installed strategy package, by its id."""
    component_id: str | None = Field(default=None, min_length=1)
    """The strategy component, by its id."""
    model_id: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_]*$")
    """An Alpha model, by the id its declaration names (`model list` lists them)."""
    task_id: UUID | None = None
    """The Task, by its id."""
    expected_task_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The Task's version as last read; a Task that moved since refuses the request."""
    recovery_task_id: UUID | None = None
    """The stopped Task whose owner replan this existing planner/admission continues."""
    recovery_task_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The exact stopped Task version named by `recovery_task_id`."""
    incident_key: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The Guanyin incident, by the key TASK_INCIDENTS listed."""
    remedy: Literal["CANCEL", "RECOVER", "REPLAN"] | None = None
    """The remedy to perform, one the incident offers: `CANCEL`, `RECOVER` or `REPLAN`."""
    result_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """A Portfolio result, by its hash; for a book, names the book with the other book fields, as
    a book's offered requests fill them."""
    left_result_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The first of two Portfolio results to compare."""
    right_result_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The second of two Portfolio results to compare."""
    candidate_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The frozen candidate to finalize, by its hash."""
    handoff_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The research handoff a book comes from. Names the book with the other book fields, as a
    book's offered requests fill them."""
    update_task_id: UUID | None = None
    """The Portfolio update Task a book comes from. Names the book with the other book fields, as
    a book's offered requests fill them."""
    experiment_task_id: UUID | None = None
    """The Portfolio study Task a book comes from. Names the book with the other book fields, as
    a book's offered requests fill them."""
    experiment_receipt_hash: str | None = None
    """The Portfolio study's result, by its receipt hash. Names the book with the other book
    fields, as a book's offered requests fill them."""
    update_publication_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The Portfolio update's publication a book comes from. Names the book with the other book
    fields, as a book's offered requests fill them."""
    position_basis: str | None = None
    """The positions a book's update published them on. Names the book with the other book
    fields, as a book's offered requests fill them."""
    review_publication_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """A published CRO review, by its hash."""
    analysis_publication_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """A published Evidence analysis, by its hash."""
    review_dossier_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The CRO dossier a review answers, by its hash."""
    review_policy_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The review policy the CRO answers under, as the dossier named it."""
    review_schema_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The review answer's schema, as the dossier named it."""
    review_read_at: str | None = Field(default=None, min_length=1, max_length=64)
    """When the review's dossier was read, as its bundle or first part sealed it."""
    review_answer: PortfolioReviewAnswer | dict[str, Any] | None = None
    """The schema is the answer's, so capabilities describe it; an answer the
    schema would refuse is still carried, as written, for the Host to name
    each problem by item."""
    analysis_context_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The Evidence packet's context an analysis answers, by its hash."""
    analysis_answer: AlternativeEvidenceAnalystAnswer | dict[str, Any] | None = None
    """The Analyst's answer to an Evidence packet."""
    agent_role: AgentRole | None = None
    """Whose bundle to prepare: `ANALYST`, `CRO`, `ALPHA`, `DATA`, `FACTOR`,
    `PORTFOLIO` or `RISK`. The five research specialists name a retained Task."""
    bundle_directory: str | None = Field(default=None, min_length=1, max_length=1024)
    """Where the agent's bundle is written, and its answer read back."""
    agent_answer: dict[str, Any] | None = None
    """An answer to the named bundle as written. Analyst and CRO retain their
    domain schemas; the other specialists supply bounded `text` and exact
    `references` listed in the bundle. The Host checks shape and bindings,
    never scientific content, and names each problem."""
    evidence_as_of: str | None = Field(default=None, min_length=1, max_length=64)
    """The time Evidence is prepared as of, in ISO 8601."""
    preparation_binding_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The Evidence preparation's binding, by its hash."""
    evidence_unit_id: str | None = Field(default=None, pattern=r"^u[0-9]{2,3}$")
    """One Evidence unit, by its id (u01, u02 ...)."""
    delivery_part: int | None = Field(default=None, ge=1, le=100_000, strict=True)
    """Which part of an answer delivered in parts to read, from 1."""
    delivery_budget_bytes: int | None = Field(default=None, ge=1, strict=True)
    """The most bytes one delivered part may hold."""
    evidence_detail: str | None = Field(
        default=None,
        pattern=r"^(litigation_inventory|session_windows|topic_coverage|time_view)$",
    )
    """Which detail of the packet to read: `litigation_inventory`, `session_windows`,
    `topic_coverage` or `time_view`; `EVIDENCE_CRO` also takes `time_view` for the
    published reading."""
    view_entity_id: str | None = Field(default=None, min_length=1, max_length=32)
    """Only this entity's Evidence."""
    view_topic: str | None = Field(default=None, pattern=r"^[A-Z_]{1,40}$")
    """Only this topic's Evidence."""
    view_last_days: int | None = Field(default=None, ge=1, le=3660, strict=True)
    """Only the last this many days."""
    view_from: str | None = Field(default=None, min_length=1, max_length=64)
    """The first day read, as YYYY-MM-DD."""
    view_to: str | None = Field(default=None, min_length=1, max_length=64)
    """The last day read, as YYYY-MM-DD."""
    ledger_page: int | None = Field(default=None, ge=1, le=100_000, strict=True)
    """Which page of the Evidence ledger to read, from 1."""
    documents_page: int | None = Field(default=None, ge=1, le=100_000, strict=True)
    """Which page of the workspace's retained documents to read, from 1."""
    citation_entity_id: str | None = Field(default=None, min_length=1, max_length=32)
    """The entity whose citation to open."""
    citation_unit_id: str | None = Field(default=None, pattern=r"^u[0-9]{2,3}$")
    """The Evidence unit the citation belongs to."""
    citation_page: int | None = Field(default=None, ge=1, le=100_000, strict=True)
    """Which page of the cited source to read, from 1."""
    continuation_of: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The prepared packet a continuation reads on from, by the receipt it sealed."""
    continuation_spans: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The packet's sealed span set, by its hash."""
    session_limit: int | None = Field(default=None, ge=1, le=99, strict=True)
    """The most reading sessions the continued chain may hold."""
    window_limit: int | None = Field(default=None, ge=1, le=4096, strict=True)
    """The most windows the continued chain may read."""
    finding_handle: str | None = Field(default=None, pattern=r"^FIND-[A-Z0-9-]{1,80}$")
    """A CRO finding, by its handle (FIND-...)."""
    prior_review_publication_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """An earlier CRO review the export compares with."""
    update_plan_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """An update plan, by the hash its PLAN answered."""
    score_plan_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """A score plan, by the hash its PLAN answered."""
    score_snapshot_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The scores a calibration reads, by their snapshot hash."""
    calibration_plan_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """A calibration plan, by the hash its PLAN answered."""
    prepared_input_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The prepared input a Portfolio update reads, by its hash."""
    observed_through: str | None = None
    """The last session the update observes, as YYYY-MM-DD."""
    formation_session: str | None = None
    """The session the scores are formed on, as YYYY-MM-DD."""
    automation_enabled: bool | None = Field(default=None, strict=True)
    """true lets the daily research update run by itself; false stops it."""
    network_enabled: bool | None = Field(default=None, strict=True)
    """true lets this workspace reach the network; false keeps it offline."""
    usage_reading_enabled: bool | None = Field(default=None, strict=True)
    """true lets the Host read the bound agent Sessions' own files for usage; false reads
    nothing."""
    automation_package_ids: tuple[str, ...] | None = None
    """The strategy packages the automatic research update covers."""
    upgrade_set_hash: str | None = None
    """The installed identity set an upgrade overview showed, when acknowledging it."""
    feature_trial_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """A feature trial, by the ID its request answered with."""
    feature_factor_id: str | None = Field(default=None, min_length=1, max_length=160)
    """A formula factor a research plan declares, which its review and activation name."""
    extensions_page: int | None = Field(default=None, ge=1, le=100_000, strict=True)
    """Which page of the declared formula factors to read, from 1."""
    agent_session: str | None = Field(default=None, min_length=1, max_length=200)
    """An agent session: a Task listing shows the Tasks it submitted, 50 to a page, and a goal
    listing the goals whose record names it."""
    wait_seconds: float | None = Field(default=None, gt=0, le=20)
    """How long a Task's status may wait for the Task to move on; at most 20 seconds."""
    after: str | None = Field(default=None, min_length=1, max_length=256)
    """Where the activity feed reads on: the cursor its previous page answered."""
    limit: int | None = Field(default=None, ge=1, le=200, strict=True)
    """The most a page reads: 1 to 200 rows of the activity feed, 1 to 50 recent groups."""
    watch: tuple[UUID, ...] | None = Field(default=None, max_length=16)
    """Tasks, by id, whose current projection the activity page joins; a short id the compact
    display gave reads as its whole id."""
    event: dict[str, Any] | None = None
    """The event this client declares about its own work (`ExternalActivityEventDocument`)."""
    storage_cap_bytes: str | None = Field(default=None, min_length=1, max_length=32)
    """`auto`, or a positive whole byte cap for managed writes; changes no result identity."""
    cpu_budget: str | None = Field(default=None, min_length=1, max_length=16)
    """`auto`, or a whole number of cores a book's preparation may use."""
    tasks_waiting: str | None = Field(default=None, min_length=1, max_length=16)
    """`auto`, or how many Tasks may wait behind the running one."""
    backup_generations_kept: int | None = Field(default=None, ge=1, le=100, strict=True)
    """How many backup generations to keep: seven unless a request keeps another count."""
    # Deliberately no actor field. Who is choosing is decided by the bridge
    # this envelope arrives at, never by the envelope.

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_operation_fields(self) -> Self:
        """Validate the operation envelope against the typed request contract."""
        self.to_operation_request()
        return self

    def to_operation_request(self) -> PortfolioResearchOperationRequest:
        """Retain typed values while delegating operation-specific validation."""
        return PortfolioResearchOperationRequest(
            **{
                field.name: getattr(self, field.name)
                for field in fields(PortfolioResearchOperationRequest)
            }
        )


OperationCaller = Literal["HUMAN", "INSTALLED_AGENT", "EXTERNAL_AUTOMATION", "SERVICE_AUTOMATION"]
"""Who is running this operation, stated by the entry that runs it.

Never a request field. A caller that could name itself in its own payload could
name the other one, and an Agent's choice would be recorded as a person's.
"""


def public_control_document(
    control: PublicControlDescriptor,
    *,
    scope: str,
    minimum: str | None = None,
    maximum: str | None = None,
) -> dict[str, object]:
    """One browser-ready control contract; clients parse no admission prose."""
    return {
        "control_id": control.control_id,
        "label": control.label,
        "unit": control.unit,
        "value_kind": control.value_kind,
        "options": list(control.options),
        "min": str(control.min) if minimum is None and control.min is not None else minimum,
        "max": str(control.max) if maximum is None and control.max is not None else maximum,
        "step": None if control.step is None else str(control.step),
        "default_display": control.default_display,
        "default_value": control.default_value,
        "help": control.help,
        "guard": control.guard,
        "refusal": control.refusal,
        "disposition": control.disposition,
        "invalidation_class": control.invalidation_class,
        "scope": scope,
    }


class PortfolioPlanPort(Protocol):
    """What a plan hands back. Structural, so the Host satisfies it without knowing."""

    @property
    def preview(self) -> PortfolioPlanPreview:
        """Return the plan's preview without running it."""
        ...


class PortfolioRunPort(Protocol):
    """Expose the result of a completed portfolio run."""

    @property
    def result(self) -> PortfolioResearchResult:
        """Return the sealed run result."""
        ...


class PortfolioAdvancementPort(Protocol):
    """The advancement half of the Host surface, kept structurally separate.

    Advancing inputs and studying an admitted ledger are different operations,
    so they arrive as different ports. A service that took one object with both
    would let a caller reach a fit through a reporting request.
    """

    def plan_advancement(self) -> PortfolioAdvancementPlanPort:
        """Preview advancement across admitted input owners."""
        ...

    def advance(self) -> WatermarkAdvancementReceipt:
        """Advance or reuse each input and return its receipt."""
        ...


class PortfolioAdvancementPlanPort(Protocol):
    """Expose the program and sessions proposed for advancement."""

    @property
    def program(self) -> WatermarkAdvancementProgram:
        """Return the proposed advancement program."""
        ...

    @property
    def target_sessions(self) -> tuple[date, ...]:
        """Return the sessions the plan would advance."""
        ...

    @property
    def blocking_owner_id(self) -> str | None:
        """Identify an input owner that blocks advancement, if any."""
        ...


class PortfolioTaskLineagePort(Protocol):
    """Resolve a result to its originating task."""

    def originating_task(self, result_hash: str) -> UUID | None:
        """Return the task that produced the given result, if recorded."""
        ...


class PortfolioLedgerPort(Protocol):
    """Read sealed portfolio artifacts by their durable identities."""

    def load_result(self, result_hash: str) -> PortfolioResearchResult:
        """Open a sealed result by hash."""
        ...

    def load_report(self, report_hash: str) -> PortfolioDeclaredPathReport:
        """Open the declared-path report by hash."""
        ...

    def load_economics(self, economic_hash: str) -> PortfolioEconomicLedger:
        """Open the economic ledger by hash."""
        ...

    def load_execution(self, ledger_hash: str) -> PortfolioExecutionLedger:
        """Open the execution ledger by hash."""
        ...

    def load_lane(self, *, category: str, content_hash: str) -> bytes:
        """Read one categorized content-addressed lane."""
        ...

    def load_opening_reference(self, ledger: PortfolioExecutionLedger) -> npt.NDArray[np.float64]:
        """Read the opening reference for an execution ledger."""
        ...

    def load_html_by_uri(self, uri: str) -> str:
        """Read a saved self-contained HTML report by URI."""
        ...


class PortfolioApplicationPort(Protocol):
    """The exact Host surface this service consumes, and nothing wider."""

    @property
    def workspace_id(self) -> str:
        """Identify the workspace bound to this application."""
        ...

    @property
    def ledger(self) -> PortfolioLedgerPort:
        """Return the artifact reader for this workspace."""
        ...

    def plan(self, spec: PortfolioResearchSpec) -> PortfolioPlanPort:
        """Plan a study from the supplied specification."""
        ...

    def run(self, *, spec: PortfolioResearchSpec) -> PortfolioRunPort:
        """Run the admitted portfolio study."""
        ...

    def report(self, result_hash: str) -> PortfolioDeclaredPathReport:
        """Read the report associated with a result."""
        ...

    def export(self, result_hash: str) -> str:
        """Export the saved result as a self-contained page."""
        ...


@dataclass(frozen=True, slots=True)
class PortfolioComparisonMetric:
    """One typed fact shown on both sides of a comparison."""

    label: str
    unit: str | None
    left: str | int | float | bool | tuple[str, ...] | None
    right: str | int | float | bool | tuple[str, ...] | None


@dataclass(frozen=True, slots=True)
class PortfolioComparisonDimension:
    """A researcher-facing group of facts; it carries no preference or score."""

    dimension: Literal[
        "HOLDINGS",
        "CONCENTRATION",
        "COST",
        "TURNOVER",
        "PERFORMANCE",
        "RISK",
        "LIMITATIONS",
    ]
    metrics: tuple[PortfolioComparisonMetric, ...]


@dataclass(frozen=True, slots=True)
class PortfolioComparison:
    """Two declared paths, side by side, with no ranking.

    The service compares and never selects. A comparison that named a winner
    would be a promotion, and promotion is exactly what a bounded exploration is
    not allowed to do.
    """

    left: PortfolioDeclaredPathReport
    right: PortfolioDeclaredPathReport
    shares_execution_ledger: bool
    differing_controls: tuple[str, ...]
    dimensions: tuple[PortfolioComparisonDimension, ...]
    disposition: Literal["DECLARED_PATH_COMPARISON_NO_SELECTION"] = (
        "DECLARED_PATH_COMPARISON_NO_SELECTION"
    )


@dataclass(frozen=True, slots=True)
class FrozenCandidateProjection:
    """What `Freeze` shows: the exact candidate and where it came from.

    Identities only, and every one of them read off the Gate 9B owner rather
    than recomputed here. A projection that derived any of these would be a
    second freeze.
    """

    candidate_hash: str
    workspace_id: str
    development_result_hash: str
    development_task_id: str
    development_run_hash: str
    program_hash: str
    spec_hash: str
    control_receipt_hash: str
    pre_protected_state_hash: str
    last_formation_session: date
    formation_count: int
    frozen_at: datetime


FinalizationDisposition = Literal[
    "NOT_FROZEN",
    "AWAITING_PROTECTED_AUTHORITY",
    "RELEASED",
]
"""Where a candidate stands, in the only three states a local build can be in.

`AWAITING_PROTECTED_AUTHORITY` is the ordinary answer and is a *state*, not an
error: a development workspace holds no protected fixture and no Stage 11
permit, so a frozen candidate waits. The UI renders the wait; it never renders a
terminal recommendation, and there is no route here that could manufacture one.
"""


@dataclass(frozen=True, slots=True)
class FinalizationStatusProjection:
    """What `Finalize` shows. Validation status lives here, not in a second Desk."""

    candidate_hash: str
    disposition: FinalizationDisposition
    detail: str
    package_hash: str | None = None
    validation_receipt_hash: str | None = None
    handoff_hash: str | None = None
    released_result_hash: str | None = None
    released_report_hash: str | None = None
    closure: str | None = None


@dataclass(frozen=True, slots=True)
class PortfolioExportManifest:
    """One compact record of what was asked, what answered, and how to ask again.

    The command needs the workspace's folder and the spec, saved from ``spec_document``, and
    nothing else. The predecessor's root/hash argument list is deliberately not restored: a
    hash on a command line is an identity being replayed as an authority, and the compiler
    never saw it.
    """

    workspace_id: str
    spec_hash: str
    holdings_spec_hash: str
    program_hash: str
    numerical_input_assembly_hash: str | None
    result_hash: str
    execution_ledger_hash: str
    economic_ledger_hash: str
    report_hash: str
    authorities_hash: str | None
    originating_task_id: str | None
    execution_mode: str
    controls: tuple[tuple[str, str], ...]
    command: str
    spec_document: dict[str, object]

    def as_json(self) -> str:
        """Serialize the manifest with stable keys and readable indentation."""
        return json.dumps(
            {
                "kind": "PortfolioExportManifest",
                "workspace_id": self.workspace_id,
                "spec_hash": self.spec_hash,
                "holdings_spec_hash": self.holdings_spec_hash,
                "program_hash": self.program_hash,
                "numerical_input_assembly_hash": self.numerical_input_assembly_hash,
                "result_hash": self.result_hash,
                "execution_ledger_hash": self.execution_ledger_hash,
                "economic_ledger_hash": self.economic_ledger_hash,
                "report_hash": self.report_hash,
                "authorities_hash": self.authorities_hash,
                "originating_task_id": self.originating_task_id,
                "execution_mode": self.execution_mode,
                "controls": [list(pair) for pair in self.controls],
                "command": self.command,
                "spec": self.spec_document,
            },
            indent=2,
            sort_keys=True,
        )


class PortfolioFreezePort(Protocol):
    """The Gate 9B freeze owner, as much of it as a local UI may reach."""

    def freeze(self, *, result_hash: str) -> FrozenCandidateProjection:
        """Freeze the identified development result through its owner."""
        ...

    def open_frozen(self, candidate_hash: str) -> FrozenCandidateProjection | None:
        """Open a frozen candidate when that identity exists."""
        ...

    def finalization_status(self, candidate_hash: str) -> FinalizationStatusProjection:
        """Read finalization status for one frozen candidate."""
        ...


class PortfolioProgramPort(Protocol):
    """Open the execution program bound to a saved result."""

    def program(self, result_hash: str) -> PortfolioExecutionProgram:
        """Return the saved execution program for a result."""
        ...


class LocalPortfolioResearchService:
    """One service per workspace session; owns no state beyond the injected port."""

    def __init__(
        self,
        *,
        application: PortfolioApplicationPort,
        advancement: PortfolioAdvancementPort | None = None,
        lineage: PortfolioTaskLineagePort | None = None,
        freeze: PortfolioFreezePort | None = None,
        programs: PortfolioProgramPort | None = None,
    ) -> None:
        """Bind application and optional owners for this workspace session."""
        self.application = application
        self._advancement = advancement
        self._lineage = lineage
        self._freeze = freeze
        self._programs = programs

    @property
    def workspace_id(self) -> str:
        """The workspace this service writes into, as its export manifest names it."""
        return self.application.workspace_id

    def plan_advancement(self) -> PortfolioAdvancementPlanPort:
        """What advancing would cost, without asking any owner to advance."""
        return self._require_advancement().plan_advancement()

    def advance(self) -> WatermarkAdvancementReceipt:
        """Ask every input owner to advance or exactly reuse, once."""
        return self._require_advancement().advance()

    def originating_task(self, result_hash: str) -> UUID | None:
        """The task that produced a result, so a report can cite its own run."""
        if self._lineage is None:
            raise LocalApplicationError("local_application.task_lineage_not_wired")
        return self._lineage.originating_task(result_hash)

    def _require_advancement(self) -> PortfolioAdvancementPort:
        if self._advancement is None:
            raise LocalApplicationError("local_application.advancement_not_wired")
        return self._advancement

    @property
    def controls(self) -> PublicControlCatalog:
        """The surface a UI renders. It is not built here, only handed on."""
        return INSTALLED_PUBLIC_CONTROL_CATALOG

    def plan(self, spec: PortfolioResearchSpec) -> PortfolioPlanPreview:
        """Return a portfolio study preview without running it."""
        return self.application.plan(spec).preview

    def run(self, spec: PortfolioResearchSpec) -> PortfolioResearchResult:
        """Run the admitted study and return its sealed result."""
        return self.application.run(spec=spec).result

    def open_result(self, result_hash: str) -> PortfolioResearchResult:
        """The result itself, so an operation can pass it on instead of the hash.

        Deliberately not the report as well. Comparing a result with itself, or
        exporting it against a spec it did not come from, is decidable from the
        result alone; opening its report first would turn those refusals into
        reads and would fail differently when the report is unreadable.
        """
        return self.application.ledger.load_result(result_hash)

    def report_of(self, result: PortfolioResearchResult) -> PortfolioDeclaredPathReport:
        """The report an opened result names -- one read, taken when it is needed.

        Every value passed here is opened by the operation that calls it and
        discarded with it. Nothing is keyed, stored or shared between
        operations, so a later request opens its own artifacts and can never
        answer from bytes an earlier one happened to hold.
        """
        return self.application.ledger.load_report(result.report_hash)

    def report(self, result_hash: str) -> PortfolioDeclaredPathReport:
        """Read the saved report for an identified result."""
        return self.application.report(result_hash)

    def export(self, result_hash: str) -> str:
        """Export an identified result through its application owner."""
        return self.application.export(result_hash)

    def open_html(self, result_hash: str) -> str:
        """The self-contained page, read back from the ledger rather than re-rendered."""
        result = self.application.ledger.load_result(result_hash)
        return self.application.ledger.load_html_by_uri(result.html_uri)

    def schedule_guard(self, result_hash: str) -> PortfolioScheduleGuard:
        """Return the schedule guard recorded with a result's report."""
        return self.report(result_hash).schedule_guard

    def window_guard(self, result_hash: str) -> PortfolioStudyWindowGuard:
        """Return the study window guard recorded with a result's report."""
        return self.report(result_hash).window_guard

    def full_support_reference(self, spec: PortfolioResearchSpec) -> PortfolioSupportCoverage:
        """The reference a descriptive window must always be shown against."""
        return self.plan(spec).coverage

    def with_study_window(
        self,
        spec: PortfolioResearchSpec,
        *,
        study_start: date | None,
        study_end: date | None,
    ) -> PortfolioResearchSpec:
        """Re-slice an existing request. Nothing upstream is invalidated by this."""
        return PortfolioResearchSpec.create(
            strategy_package_id=spec.strategy_package_id,
            score_source_mode=spec.score_source_mode,
            top_k=spec.top_k,
            tranches=spec.tranches,
            exit_rank=spec.exit_rank,
            weight_rule=spec.weight_rule,
            cost_bps_per_side=str(spec.cost.cost_bps_per_side),
            secondary_benchmark_view=spec.secondary_benchmark_view,
            report_unit=spec.report_unit,
            study_start=study_start,
            study_end=study_end,
        )

    def readouts(self, result_hash: str) -> PortfolioReadouts:
        """Build readouts from the saved report for one result."""
        return self.readouts_of(self.report_of(self.open_result(result_hash)))

    def path_readback_of(
        self,
        report: PortfolioDeclaredPathReport,
        portfolio_session: date | None = None,
        *,
        execution: PortfolioExecutionLedger | None = None,
        economics: PortfolioEconomicLedger | None = None,
    ) -> dict[str, object]:
        """A dated view of this saved report, never a new window or numerical RUN."""
        store = self.application.ledger
        ledger = execution or store.load_execution(report.execution_ledger_hash)
        day = portfolio_session or report.window_guard.selected_end
        if (
            day not in ledger.formation_sessions
            or not report.window_guard.selected_start <= day <= report.window_guard.selected_end
        ):
            raise LocalApplicationError("portfolio_application.session_outside_report")
        if (
            ledger.ledger_hash != report.execution_ledger_hash
            or ledger.program_hash != report.program_hash
        ):
            raise LocalApplicationError("portfolio_application.report_execution_binding_mismatch")
        economics = economics or store.load_economics(report.economic_ledger_hash)
        if economics.economic_ledger_hash != report.economic_ledger_hash:
            raise LocalApplicationError("portfolio_application.report_economic_binding_mismatch")
        row = ledger.formation_sessions.index(day)
        position = project_dated_position(
            ledger=ledger,
            economics=economics,
            formation_row=row,
            executed_weights=store.load_lane(
                category="executed-weights", content_hash=ledger.executed_weights_hash
            ),
            target_weights=None
            if ledger.target_weights_hash is None
            else store.load_lane(
                category="target-weights", content_hash=ledger.target_weights_hash
            ),
            # Noninitial dates use the preceding weights row, not the opening
            # boundary. Do not demand an irrelevant legacy boundary artifact.
            opening_reference=store.load_opening_reference(ledger) if row == 0 else np.empty(0),
        )
        return {
            "position": position.body(),
            "series": [
                {
                    "session": session.isoformat(),
                    "gross_simple_return": ledger.gross_simple_returns[i],
                    "net_simple_return": economics.net_simple_returns[i],
                    "benchmark_simple_return": ledger.anchor_simple_returns[i],
                    "one_way_turnover": ledger.one_way_turnovers[i],
                }
                for i, session in enumerate(ledger.formation_sessions)
                if report.window_guard.selected_start <= session <= report.window_guard.selected_end
            ],
            "execution_ledger_hash": ledger.ledger_hash,
            "economic_ledger_hash": economics.economic_ledger_hash,
            "weights_hash": ledger.executed_weights_hash,
            "targets_hash": ledger.target_weights_hash,
        }

    def selected_window_performance_of(
        self,
        report: PortfolioDeclaredPathReport,
        *,
        execution: PortfolioExecutionLedger,
        economics: PortfolioEconomicLedger,
    ) -> dict[str, object]:
        """Derive optional return metrics from this verified report's sealed net path.

        The selected window is descriptive. This read changes no report, model,
        cost or benchmark and delegates all return arithmetic to Backtesting.
        """
        if (
            execution.ledger_hash != report.execution_ledger_hash
            or execution.program_hash != report.program_hash
            or economics.economic_ledger_hash != report.economic_ledger_hash
            or economics.execution_ledger_hash != execution.ledger_hash
            or len(economics.net_simple_returns) != len(execution.formation_sessions)
        ):
            raise LocalApplicationError("portfolio_application.report_economic_binding_mismatch")
        window = report.window_guard
        net: npt.NDArray[np.float64] = np.asarray(
            [
                economics.net_simple_returns[i]
                for i, session in enumerate(execution.formation_sessions)
                if window.selected_start <= session <= window.selected_end
            ],
            dtype=np.float64,
        )
        if net.size != window.selected_session_count:
            raise LocalApplicationError("portfolio_application.report_window_axis_mismatch")
        metrics: dict[str, float] = {}
        absences: dict[str, object] = {}
        names = (
            "annualized_return",
            "annualized_volatility",
            "maximum_drawdown",
            "sharpe",
            "sortino",
        )
        if net.size < 2:
            absences = {
                name: {
                    "status": "UNAVAILABLE",
                    "reason": "INSUFFICIENT_RETURN_OBSERVATIONS",
                    "detail": "The shared daily-return metric owner requires at least two "
                    "observations in the selected sealed window.",
                }
                for name in names
            }
        else:
            values = evaluate_net_simple_return_path(net_simple_returns=net).model_dump()
            for name in names:
                value = float(values[name])
                if np.isfinite(value):
                    metrics[name] = value
                else:
                    zero_downside = name == "sortino" and not bool(np.any(net < 0.0))
                    absences[name] = {
                        "status": "UNAVAILABLE",
                        "reason": "ZERO_DOWNSIDE_DEVIATION"
                        if zero_downside
                        else "NONFINITE_NET_RETURN_METRIC",
                        "detail": "The selected sealed net-return path has no downside "
                        "deviation against the owner's zero threshold."
                        if zero_downside
                        else "The shared owner returned a nonfinite value for this window.",
                    }
        return {
            "selected_window_metrics": metrics,
            "selected_window_metric_absences": absences,
            "selected_window_metric_provenance": {
                "status": "DERIVED_FROM_SEALED_NET_RETURN_PATH",
                "report_hash": report.report_hash,
                "execution_ledger_hash": execution.ledger_hash,
                "economic_ledger_hash": economics.economic_ledger_hash,
                "window_hash": window.guard_hash,
                "selected_start": window.selected_start.isoformat(),
                "selected_end": window.selected_end.isoformat(),
                "observation_count": int(net.size),
                "return_unit": "FRACTION",
                "annualization_sessions_per_year": 252,
                "volatility_degrees_of_freedom": 1,
                "sharpe_cash_return_per_session": 0.0,
                "sortino_downside_threshold": 0.0,
            },
        }

    def readouts_of(
        self,
        report: PortfolioDeclaredPathReport,
        *,
        economics: PortfolioEconomicLedger | None = None,
    ) -> PortfolioReadouts:
        """The readouts for an already-open report and optional opened economics.

        Split from `readouts` so an operation that has already opened the report
        does not open it again to be told the same numbers. An operation already
        reading economics can pass that verified value; otherwise this opens it
        once, keeping it out of operations that do not need it.
        """
        economics = economics or self.application.ledger.load_economics(report.economic_ledger_hash)
        if economics.economic_ledger_hash != report.economic_ledger_hash:
            raise LocalApplicationError("portfolio_application.report_economic_binding_mismatch")
        return PortfolioReadouts.create(
            distinct_names_held=report.window_end_distinct_names,
            effective_n=report.window_end_effective_n,
            one_way_turnover_per_trading_session=report.mean_one_way_turnover,
            aggregate_cap_binding_sessions=report.aggregate_cap_binding_sessions,
            aggregate_cap_binding_names_total=report.aggregate_cap_binding_names_total,
            # The window's wealth, not the path's: every other readout beside it
            # describes the window end, and mixing the two would be the quiet
            # kind of wrong.
            cumulative_net_wealth=report.window_cumulative_net_wealth,
            cost_bps_per_side=economics.cost_bps_per_side,
            # Both conventions, always labelled, because a report that shows one
            # of them unlabelled has published the wrong number. A broker quotes
            # per side; a round trip is both sides; the engine charges a rate
            # against one-way turnover. Round trip and the platform rate are the
            # same figure here by construction -- `2 * per_side` -- and they are
            # carried separately so a reader is never asked to infer that.
            cost_bps_round_trip=economics.platform_one_way_cost_bps,
            platform_one_way_cost_bps=economics.platform_one_way_cost_bps,
            median_holding_adv20_dollar_volume=(
                report.window_end_median_holding_adv20_dollar_volume
            ),
            industry_attribution_available=bool(report.risk_facts),
        )

    # ------------------------------------------------------------- freeze

    def freeze(self, result_hash: str) -> FrozenCandidateProjection:
        """Freeze one exact development result through the Gate 9B owner."""
        return self._require_freeze().freeze(result_hash=result_hash)

    def frozen_candidate(self, candidate_hash: str) -> FrozenCandidateProjection | None:
        """Open a frozen candidate if its identity is present."""
        return self._require_freeze().open_frozen(candidate_hash)

    def finalization_status(self, candidate_hash: str) -> FinalizationStatusProjection:
        """Validation status, inside `Finalize`, for exactly one candidate."""
        return self._require_freeze().finalization_status(candidate_hash)

    def _require_freeze(self) -> PortfolioFreezePort:
        if self._freeze is None:
            raise LocalApplicationError("local_application.freeze_not_wired")
        return self._freeze

    # ------------------------------------------------------------- export

    def export_manifest(
        self, result_hash: str, spec: PortfolioResearchSpec
    ) -> PortfolioExportManifest:
        """Everything needed to re-ask this question, and nothing to replay as authority.

        The spec document is the *values*, so a reader can inspect and re-enter
        them. The identities beside it are what the run produced, so a reader can
        check that a re-ask landed on the same path -- which is a comparison, not
        an instruction.
        """
        result = self.open_result(result_hash)
        self._refuse_unless_this_results_spec(result, spec)
        # Only now the children, and in the order they were read before this
        # chain was consolidated: the report, the Program port, the lineage.
        return self._manifest(
            result,
            spec,
            report=self.report_of(result),
            program=self._program_of(result_hash),
            originating_task=self._optional_task(result_hash),
        )

    def _refuse_unless_this_results_spec(
        self, result: PortfolioResearchResult, spec: PortfolioResearchSpec
    ) -> None:
        """The one export refusal the result alone decides.

        A named check rather than an inline comparison, because both entry
        points have to run it *before* they touch anything else, and a check
        that lived only inside `export_manifest_from` could not: an argument is
        evaluated before the call it is passed to, so resolving the lineage for
        that argument beat the refusal to it.
        """
        if result.spec_hash != spec.spec_hash:
            raise LocalApplicationError("local_application.export_spec_is_not_this_result")

    def _program_of(self, result_hash: str) -> PortfolioExecutionProgram | None:
        """The Program record when that port is wired, and `None` when it is not."""
        return None if self._programs is None else self._programs.program(result_hash)

    def _optional_task(self, result_hash: str) -> UUID | None:
        """The lineage when one is wired, and `None` when one is not.

        `originating_task` refuses an unwired lineage, which is right for a
        caller that asked for the task. Export never did: it publishes `None`
        for that field, and a service built with nothing but an application must
        keep exporting.
        """
        return None if self._lineage is None else self._lineage.originating_task(result_hash)

    def export_manifest_from(
        self,
        result: PortfolioResearchResult,
        spec: PortfolioResearchSpec,
        *,
        originating_task: UUID | None,
    ) -> PortfolioExportManifest:
        """The same manifest, for a result an operation has already opened.

        The spec check runs first here too, before the report or the Program
        port: this method is public, and the Host reaches it directly. The
        originating task is a parameter rather than another lookup, since the
        caller that resolved this result's spec already read the lineage.
        """
        self._refuse_unless_this_results_spec(result, spec)
        return self._manifest(
            result,
            spec,
            report=self.report_of(result),
            program=self._program_of(result.result_hash),
            originating_task=originating_task,
        )

    def _manifest(
        self,
        result: PortfolioResearchResult,
        spec: PortfolioResearchSpec,
        *,
        report: PortfolioDeclaredPathReport,
        program: PortfolioExecutionProgram | None,
        originating_task: UUID | None,
    ) -> PortfolioExportManifest:
        """The manifest itself, from values its callers have already read.

        It reads nothing, which is the point: each entry point states its own
        order of reads, and neither can smuggle one in here where the refusals
        cannot see it.
        """
        return PortfolioExportManifest(
            workspace_id=self.workspace_id,
            spec_hash=spec.spec_hash,
            holdings_spec_hash=spec.holdings_spec_hash,
            program_hash=result.program_hash,
            numerical_input_assembly_hash=(
                None if program is None else program.numerical_input_assembly_hash
            ),
            result_hash=result.result_hash,
            execution_ledger_hash=result.execution_ledger_hash,
            economic_ledger_hash=result.economic_ledger_hash,
            report_hash=result.report_hash,
            authorities_hash=None if program is None else program.authorities_hash,
            originating_task_id=(None if originating_task is None else str(originating_task)),
            execution_mode=spec.execution_mode,
            controls=tuple(report.control_receipt.selected),
            # The product's own command, the spec saved from `spec_document` (OP2, V401).
            command="alphalattice --workspace <dir> strategy-book run --file <spec.json>",
            spec_document=_spec_document(spec),
        )

    def compare(self, left_result_hash: str, right_result_hash: str) -> PortfolioComparison:
        """Compare two saved results with different configurations."""
        left_result = self.open_result(left_result_hash)
        right_result = self.open_result(right_result_hash)
        if left_result.spec_hash == right_result.spec_hash:
            raise LocalApplicationError("local_application.comparison_requires_two_configurations")
        # Only now: two configurations is decidable from the results alone, and
        # refusing after opening their reports would read for nothing.
        left = self.report_of(left_result)
        right = self.report_of(right_result)
        # Named from the durable control receipts, not by diffing report fields.
        # Most controls never reach a report -- `top_k`, `exit_rank`, the cost --
        # so a field diff would silently under-report what actually differs.
        differing = left.control_receipt.changed_against(right.control_receipt)
        left_readouts = self.readouts_of(left)
        right_readouts = self.readouts_of(right)
        left_risk = left.risk_facts[-1] if left.risk_facts else None
        right_risk = right.risk_facts[-1] if right.risk_facts else None

        def metric(
            label: str,
            unit: str | None,
            left_value: str | int | float | bool | tuple[str, ...] | None,
            right_value: str | int | float | bool | tuple[str, ...] | None,
        ) -> PortfolioComparisonMetric:
            return PortfolioComparisonMetric(
                label=label,
                unit=unit,
                left=left_value,
                right=right_value,
            )

        dimensions = (
            PortfolioComparisonDimension(
                dimension="HOLDINGS",
                metrics=(
                    metric(
                        "Names held",
                        "names",
                        left.window_end_book.held_count,
                        right.window_end_book.held_count,
                    ),
                    metric(
                        "Names opened at the window end",
                        "names",
                        left.window_end_book.opened_count,
                        right.window_end_book.opened_count,
                    ),
                    metric(
                        "Names exited at the window end",
                        "names",
                        left.window_end_book.exited_count,
                        right.window_end_book.exited_count,
                    ),
                    metric(
                        "Absolute weight change",
                        "portfolio weight",
                        left.window_end_book.absolute_weight_change_total,
                        right.window_end_book.absolute_weight_change_total,
                    ),
                ),
            ),
            PortfolioComparisonDimension(
                dimension="CONCENTRATION",
                metrics=(
                    metric(
                        "Effective N",
                        None,
                        left.window_end_effective_n,
                        right.window_end_effective_n,
                    ),
                    metric(
                        "Cap-binding sessions",
                        "sessions",
                        left.aggregate_cap_binding_sessions,
                        right.aggregate_cap_binding_sessions,
                    ),
                    metric(
                        "Median holding ADV20 dollar volume",
                        "dollars",
                        left.window_end_median_holding_adv20_dollar_volume,
                        right.window_end_median_holding_adv20_dollar_volume,
                    ),
                ),
            ),
            PortfolioComparisonDimension(
                dimension="COST",
                metrics=(
                    metric(
                        "Cost per side",
                        "bps per side",
                        left_readouts.cost_bps_per_side,
                        right_readouts.cost_bps_per_side,
                    ),
                    metric(
                        "Round-trip cost",
                        "bps",
                        left_readouts.cost_bps_round_trip,
                        right_readouts.cost_bps_round_trip,
                    ),
                ),
            ),
            PortfolioComparisonDimension(
                dimension="TURNOVER",
                metrics=(
                    metric(
                        "Mean one-way turnover",
                        "per trading session",
                        left.mean_one_way_turnover,
                        right.mean_one_way_turnover,
                    ),
                ),
            ),
            PortfolioComparisonDimension(
                dimension="PERFORMANCE",
                metrics=(
                    metric(
                        "Cumulative net wealth",
                        "wealth multiple",
                        left.window_cumulative_net_wealth,
                        right.window_cumulative_net_wealth,
                    ),
                    metric("Report unit", None, left.report_unit, right.report_unit),
                ),
            ),
            PortfolioComparisonDimension(
                dimension="RISK",
                metrics=(
                    metric(
                        "Risk attribution available",
                        None,
                        left_risk is not None,
                        right_risk is not None,
                    ),
                    metric(
                        "Predicted volatility",
                        "declared-path volatility",
                        None if left_risk is None else left_risk.total_predicted_volatility,
                        None if right_risk is None else right_risk.total_predicted_volatility,
                    ),
                    metric(
                        "Systematic variance",
                        "variance",
                        None if left_risk is None else left_risk.systematic_variance,
                        None if right_risk is None else right_risk.systematic_variance,
                    ),
                    metric(
                        "Idiosyncratic variance",
                        "variance",
                        None if left_risk is None else left_risk.idiosyncratic_variance,
                        None if right_risk is None else right_risk.idiosyncratic_variance,
                    ),
                ),
            ),
            PortfolioComparisonDimension(
                dimension="LIMITATIONS",
                metrics=(
                    metric("Declared limitations", None, left.limitations, right.limitations),
                ),
            ),
        )
        return PortfolioComparison(
            left=left,
            right=right,
            shares_execution_ledger=left.execution_ledger_hash == right.execution_ledger_hash,
            differing_controls=differing,
            dimensions=dimensions,
        )


def _spec_document(spec: PortfolioResearchSpec) -> dict[str, object]:
    """The spec as values a caller can read, edit and hand back as a file."""
    return {
        "strategy_package_id": spec.strategy_package_id,
        "score_source_mode": spec.score_source_mode,
        "top_k": spec.top_k,
        "tranches": spec.tranches,
        "exit_rank": spec.exit_rank,
        "weight_rule": spec.weight_rule,
        "cost_bps_per_side": str(spec.cost.cost_bps_per_side),
        "secondary_benchmark_view": spec.secondary_benchmark_view,
        "report_unit": spec.report_unit,
        "study_start": None if spec.study_start is None else spec.study_start.isoformat(),
        "study_end": None if spec.study_end is None else spec.study_end.isoformat(),
    }


def spec_from_document(
    document: dict[str, object],
    *,
    default_strategy_package_id: str = WORKSPACE_DEFAULT_STRATEGY_PACKAGE_ID,
    default_score_source_mode: ScoreSourceMode = "HISTORICAL_ARRAY_REPLAY",
) -> PortfolioResearchSpec:
    """Rebuild a request from exported or entered values. The round trip's half.

    A control the caller left at the catalog's own default *display* is treated
    as absent, because some of those displays are descriptions rather than
    literals -- `exit_rank` reads `2 * top_k`, and a study bound reads
    `full common support`. Resolving them here rather than in a browser keeps one
    owner for what a default means.

    A study bound has a second spelling of absent: `FULL_SUPPORT`, which is what
    a stored control receipt writes, since a receipt must record a value for
    every control. Reading it here is what lets a document rebuilt from a
    published result land on the same spec as the one that produced it.
    """
    catalog_defaults = {
        control.control_id: control.default_display
        for control in INSTALLED_PUBLIC_CONTROL_CATALOG.controls
    }

    def _given(key: str) -> str | None:
        value = document.get(key)
        if value is None:
            return None
        text = str(value).strip()
        if not text or text == catalog_defaults.get(key):
            return None
        return text

    def _date(key: str) -> date | None:
        value = _given(key)
        if value is None or value == FULL_SUPPORT_SELECTION:
            return None
        return date.fromisoformat(value)

    # A control the caller did not supply is *omitted*, never passed as a
    # placeholder, so the spec owner keeps sole authority over what its own
    # default is. That is what makes the values mixed in kind here: the document
    # arrives untyped from JSON, and `PortfolioResearchSpec.create` is the
    # validator by design -- it turns an unadmitted string into a stable product
    # refusal rather than a schema traceback.
    supplied: dict[str, Any] = {}
    strategy_package_id = _given("strategy_package_id") or default_strategy_package_id
    score_source_mode = _given("score_source_mode") or default_score_source_mode
    for key, read in (
        ("top_k", int),
        ("tranches", int),
        ("exit_rank", int),
        ("weight_rule", str),
        ("cost_bps_per_side", str),
        ("secondary_benchmark_view", str),
        ("report_unit", str),
    ):
        value = _given(key)
        if value is not None:
            supplied[key] = read(value)
    return PortfolioResearchSpec.create(
        **supplied,
        strategy_package_id=strategy_package_id,
        score_source_mode=score_source_mode,  # type: ignore[arg-type]
        study_start=_date("study_start"),
        study_end=_date("study_end"),
    )


__all__ = [
    "FinalizationDisposition",
    "FinalizationStatusProjection",
    "FrozenCandidateProjection",
    "LocalApplicationError",
    "LocalPortfolioResearchService",
    "PortfolioAdvancementPlanPort",
    "PortfolioAdvancementPort",
    "PortfolioApplicationPort",
    "PortfolioComparison",
    "PortfolioComparisonDimension",
    "PortfolioComparisonMetric",
    "PortfolioExportManifest",
    "PortfolioFreezePort",
    "PortfolioLedgerPort",
    "PortfolioPlanPort",
    "PortfolioProgramPort",
    "PortfolioResearchOperation",
    "PortfolioResearchOperationRequest",
    "PortfolioResearchRequestDocument",
    "PortfolioRunPort",
    "PortfolioTaskLineagePort",
    "public_control_document",
    "spec_from_document",
]
