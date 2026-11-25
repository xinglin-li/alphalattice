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

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, fields
from datetime import date, datetime
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol, Self, cast, get_args
from uuid import UUID

from pydantic import ConfigDict, create_model, model_validator

from alphalattice.capabilities.domain_contracts.identity import canonical_hash
from alphalattice.interface.local_application.portfolio_export import PortfolioExportManifest
from alphalattice.interface.local_application.portfolio_research_fields import field_contract
from alphalattice.interface.local_application.questions import (
    PersonConfirmation as PersonConfirmation,
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
from alphalattice.protocols.actor_execution.bundles import AgentRole

from .portfolio_request_fields import FIELD_OWNER, request_fields
from .portfolio_request_fields.contracts import PortfolioResearchOperation, RequestFieldModel

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from alphalattice.investment.portfolio_strategy_lab.application.advancement import (
        WatermarkAdvancementProgram,
        WatermarkAdvancementReceipt,
    )
    from alphalattice.investment.portfolio_strategy_lab.application.controls import (
        PublicControlCatalog,
        PublicControlDescriptor,
    )

    from .portfolio_request_fields.goals import CommitteeMessage as CommitteeMessage
    from .portfolio_request_fields.research import (
        FactorCurationChoice as FactorCurationChoice,
    )
    from .portfolio_request_fields.research import (
        FactorCurationRequest as FactorCurationRequest,
    )
    from .portfolio_request_fields.research import (
        ResearchDeliveryCommentary as ResearchDeliveryCommentary,
    )


class LocalApplicationError(ValueError):
    """Stable refusal for a Local Application Service boundary failure."""

    allowed: list[str] | None = None  # the fields its operation takes


_REQUEST_OPERATIONS = frozenset(get_args(PortfolioResearchOperation.__value__))

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

PERSON_ONLY: Final[frozenset[str]] = frozenset(
    {
        "NETWORK_ACCESS_SET",
        "EVIDENCE_CONSENT_SET",
        "EVIDENCE_INSTALL",
        "USAGE_READING_SET",
        "ANALYST_READS_SET",
        "COMMITTEE_DAILY_SET",
        "STORAGE_CONFIRM",
        "STORAGE_PIN",
        "RESEARCH_INPUT_CONFIRM",
        "DATA_ISSUE_DELEGATE",
        "DATA_ISSUE_REVOKE",
        "MODEL_ACTIVATE",
        "MODEL_DEACTIVATE",
        "FEATURE_ACTIVATE",
        "FEATURE_DEACTIVATE",
        "RESEARCH_UPDATE_AUTOMATION_CONFIGURE",
        "STRATEGY_ACTIVATE",
        "STRATEGY_DEACTIVATE",
    }
)
"""The operations only a person decides.

Each owner refuses every other caller, an Agent's included, unless the request relays the person's
yes (`person_confirmation`, `PERSON_DECISIONS`) (LS1, OW12). The CLI marks each, so an agent asks
the person in one line first; a test holds this set to the owners' refusals.

Data change confirmation reads its plan kind: membership requires the person's approval,
while an exactly scoped full-history audit also admits the installed agent. It is not
unconditionally person-only (person-stops row 49).
"""

FIRST_USE_STEPS: Final[frozenset[str]] = frozenset(
    {
        "NETWORK_ACCESS_SET",
        "EVIDENCE_CONSENT_SET",
        "EVIDENCE_INSTALL",
        "WORKSPACE_PREPARE_CONFIRM",
        "DATA_ISSUE_CONFIRM",
        "DATA_CHANGE_CONFIRM",
        "STRATEGY_ACTIVATE",
    }
)
"""The person's steps a first-use goal delegates to the agent that runs it (OP19):
opening the network for the first preparation, SEC source consent within the default budget,
confirming that preparation and its resumes,
deciding its data issues, confirming its membership changes, and activating its book, which the
person deactivates in one click. A deactivation, a model's
or Feature's activation, a storage decision, an automation, a revocation and anything paid stay a
person's."""

PERSON_DECISIONS: Final[frozenset[str]] = PERSON_ONLY | FIRST_USE_STEPS
"""The person's decisions an agent sends with the person's yes, asked in one line
(`person_confirmation`): each person-only operation, and each first-use step outside its goal's
delegation. A paid action or a real order never joins it."""


@dataclass(frozen=True, slots=True)
class PortfolioResearchOperationRequest:
    """Shared request."""

    operation: PortfolioResearchOperation
    schema_operation: PortfolioResearchOperation | None = None
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
    experience_hypothesis: str | None = None
    experience_method: str | None = None
    experience_workspace: str | None = None
    experience_trial: str | None = None
    experience_width: str | None = None
    experience_known_at: str | None = None
    experience_budget: int | None = None
    experience_query: str | None = None
    experience_limit: int | None = None
    review_subject: str | None = None
    review_verdict: str | None = None
    review_note: dict[str, Any] | None = None
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
    performance: Literal["latest"] | None = None
    portfolio_scope: Literal["holdings"] | None = None
    spec: dict[str, object] | None = None
    strategy_package_id: str | None = None
    component_id: str | None = None
    model_lifecycle: Literal["LIGHT", "FULL"] | None = None
    """A component's lifecycle: the light default, or the component's FULL lifecycle by name."""
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
    positions_date: str | None = None
    """The trade date whose positions an update is planned for; its as-of is the close before."""
    formation_session: str | None = None
    automation_enabled: bool | None = None
    network_enabled: bool | None = None
    evidence_documents_per_issuer: int | None = None
    evidence_total_documents: int | None = None
    evidence_total_bytes: int | None = None
    evidence_setup: str | None = None
    usage_reading_enabled: bool | None = None
    committee_daily_enabled: bool | None = None
    analyst_reads_mode: Literal["SUBAGENTS", "IN_SESSION"] | None = None
    analyst_reads_bundles: int | None = None
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
    status_section: Literal["storage"] | None = None
    """The optional Task status section to compute on an explicit read."""
    after: str | None = None
    limit: int | None = None
    watch: tuple[UUID, ...] | None = None
    event: dict[str, Any] | None = None
    wake_thread: str | None = None
    wake_read: str | None = None
    wake_codex_path: str | None = None
    storage_cap_bytes: str | None = None
    """Automatic workspace capacity or a positive whole byte count, never a sealed input."""
    cpu_budget: str | None = None
    tasks_waiting: str | None = None
    tasks_running: str | None = None
    backup_generations_kept: int | None = None
    person_confirmation: PersonConfirmation | None = None
    """The person's yes this request relays, on one of their decisions (`PERSON_DECISIONS`)."""
    committee_role: Literal["PM", "ALPHA", "RISK", "CRO"] | None = None
    committee_key: str | None = None
    committee_message: CommitteeMessage | None = None
    committee_seen: int | None = None
    goal_seen: str | None = None

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
            from .portfolio_request_fields import research  # lazy: delivery commentary

            if len(self.delivery_commentary) > 16:
                raise LocalApplicationError("research_delivery.commentary_limit_exceeded")
            object.__setattr__(
                self,
                "delivery_commentary",
                tuple(
                    value
                    if isinstance(value, research.ResearchDeliveryCommentary)
                    else research.ResearchDeliveryCommentary.model_validate(value)
                    for value in self.delivery_commentary
                ),
            )
        if self.committee_message is not None:
            from .portfolio_request_fields import goals  # lazy: committee message

            if not isinstance(self.committee_message, goals.CommitteeMessage):
                object.__setattr__(
                    self,
                    "committee_message",
                    goals.CommitteeMessage.model_validate(self.committee_message),
                )
        if self.person_confirmation is not None and not isinstance(
            self.person_confirmation, PersonConfirmation
        ):
            object.__setattr__(
                self,
                "person_confirmation",
                PersonConfirmation.model_validate(self.person_confirmation),
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
        if self.view_topic is not None:
            from ...evidence.alternative_evidence.analysis import contracts  # lazy: evidence topic

            if self.view_topic not in {value.value for value in contracts.EvidenceTopic}:
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
        if self.experience_known_at is not None:
            try:
                parsed = datetime.fromisoformat(str(self.experience_known_at))
            except ValueError as error:
                raise LocalApplicationError("experience.known_at_invalid") from error
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                raise LocalApplicationError("experience.known_at_invalid")
        if self.experience_query is not None and (
            type(self.experience_query) is not str
            or not 1 <= len(self.experience_query) <= 512
            or (
                self.experience_limit is not None
                and (type(self.experience_limit) is not int or not 1 <= self.experience_limit <= 20)
            )
        ):
            raise LocalApplicationError("experience.recall_query_invalid")
        if self.review_read_at is not None:
            try:
                parsed = datetime.fromisoformat(str(self.review_read_at))
            except ValueError as error:
                raise LocalApplicationError("chief_risk_officer.review_read_at_invalid") from error
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                raise LocalApplicationError("chief_risk_officer.review_read_at_invalid")
        if self.experiment_curation is not None:
            from .portfolio_request_fields import research  # lazy: Factor curation

            if not isinstance(self.experiment_curation, research.FactorCurationRequest):
                object.__setattr__(
                    self,
                    "experiment_curation",
                    research.FactorCurationRequest.model_validate(self.experiment_curation),
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
        for code, names in (("required", required - supplied), ("not_allowed", supplied - allowed)):
            if names:
                named = ",".join(sorted(names))
                bad = LocalApplicationError(f"portfolio_research.operation_field_{code}:{named}")
                bad.allowed = sorted(allowed)
                raise bad

    @staticmethod
    def field_contract(
        operation: PortfolioResearchOperation,
    ) -> tuple[frozenset[str], frozenset[str]]:
        """The same required/allowed fields drive validation and discovery."""
        return field_contract(operation, _RECOVERY_CONTEXT_OPERATIONS, PERSON_DECISIONS)


class _RequestDocument(RequestFieldModel):
    """Actor-neutral document; field docstrings feed `alphalattice schema show` (SC3)."""

    model_config = ConfigDict(extra="forbid", frozen=True, use_attribute_docstrings=True)

    operation: PortfolioResearchOperation
    """The operation asked for; `alphalattice schema show <name>` lists its fields."""

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_operation_fields(self) -> Self:
        """Validate the operation envelope against the typed request contract."""
        self.to_operation_request()
        return self

    def to_operation_request(self) -> PortfolioResearchOperationRequest:
        """Validate typed fields."""
        values: dict[str, Any] = {
            field.name: getattr(self, field.name, None)
            for field in fields(PortfolioResearchOperationRequest)
            if field.name != "operation"
        }
        return PortfolioResearchOperationRequest(operation=self.operation, **values)


_DOCUMENTS_HELD = 256
"""Request document models kept: above one per operation's admitted fields plus the full set."""


@lru_cache(maxsize=_DOCUMENTS_HELD)
def _request_document(fields_to_read: frozenset[str]) -> type[_RequestDocument]:
    definitions = {}
    for name, family in FIELD_OWNER.items():
        if name in fields_to_read:
            field = request_fields(family)[name]
            definitions[name] = (field.annotation, deepcopy(field))
    return cast(
        type[_RequestDocument],
        create_model(
            "PortfolioResearchRequestDocument",
            __base__=_RequestDocument,
            __doc__=_RequestDocument.__doc__,
            __module__=__name__,
            **definitions,
        ),
    )


def full_request_document() -> type[_RequestDocument]:
    """The complete, ordered request contract used for discovery."""
    return _request_document(frozenset(FIELD_OWNER))


def request_document_schema(fields_to_read: frozenset[str]) -> dict[str, Any]:
    """Describe a selected operation using just its declared request fields."""
    return _request_document(fields_to_read).model_json_schema()


def validate_request_document(value: Mapping[str, Any]) -> _RequestDocument:
    """Use a single operation's admitted fields for regular requests."""
    operation = value.get("operation")
    if isinstance(operation, str) and operation in _REQUEST_OPERATIONS:
        _, allowed = PortfolioResearchOperationRequest.field_contract(
            cast(PortfolioResearchOperation, operation)
        )
        if allowed.union({"operation"}).issuperset(value):
            model = _request_document(allowed)
            return model.model_validate(value)
    # Complete validation keeps the order and kinds of refusals across operations.
    return full_request_document().model_validate(value)


def __getattr__(name: str) -> object:
    """Keep typed request and nested document imports available on demand."""
    if name == "PortfolioResearchRequestDocument":
        return full_request_document()
    if name == "CommitteeMessage":
        from .portfolio_request_fields import goals  # lazy: committee document

        return goals.CommitteeMessage
    if name in {"FactorCurationChoice", "FactorCurationRequest", "ResearchDeliveryCommentary"}:
        from .portfolio_request_fields import research  # lazy: research document

        return getattr(research, name)
    raise AttributeError(name)


if TYPE_CHECKING:
    PortfolioResearchRequestDocument = _RequestDocument


def decision_hash(request: Mapping[str, object]) -> str:
    """Hash the request a relayed yes binds."""
    document = validate_request_document(
        {
            name: value
            for name, value in request.items()
            if name != "person_confirmation" and value is not None
        }
    )
    return str(canonical_hash(document.model_dump(mode="json", exclude_none=True)))


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
        from alphalattice.investment.portfolio_strategy_lab.application import (  # lazy: catalog
            controls,
        )

        return controls.INSTALLED_PUBLIC_CONTROL_CATALOG

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
        import numpy as np  # lazy: only position readback constructs this array

        from alphalattice.investment.portfolio_strategy_lab.application import (  # lazy: readback
            report_projection,
        )

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
        position = report_projection.project_dated_position(
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
        import numpy as np  # lazy: only this performance read evaluates the selected return window

        from alphalattice.capabilities.portfolio_backtesting import (  # lazy: performance metrics
            metrics as backtest_metrics,
        )

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
            values = backtest_metrics.evaluate_net_simple_return_path(
                net_simple_returns=net
            ).model_dump()
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
            # An unwired lineage remains optional for export.
            originating_task=(
                None if self._lineage is None else self._lineage.originating_task(result_hash)
            ),
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
            # The product's own command, the spec saved from `spec_document` (OP2).
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
    from alphalattice.investment.portfolio_strategy_lab.application import (  # lazy: spec controls
        controls,
    )

    catalog_defaults = {
        control.control_id: control.default_display
        for control in controls.INSTALLED_PUBLIC_CONTROL_CATALOG.controls
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
        if value is None or value == controls.FULL_SUPPORT_SELECTION:
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
