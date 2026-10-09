"""Actor-neutral Portfolio Research operations shared by Local Web and Agent.

HTTP owns authentication and encoding; the Agent bridge owns model-facing tool
shape.  This module owns neither.  It composes the same application, dispatcher,
Task registry and readback owners into one finite set of product operations.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable, Mapping
from contextlib import nullcontext, suppress
from dataclasses import dataclass, field, fields, replace
from datetime import date, datetime
from functools import partial
from pathlib import Path
from time import monotonic, sleep
from typing import Any, Final, Literal, Protocol, TypeVar, cast, runtime_checkable
from uuid import UUID

from pydantic import ValidationError

from alphalattice.control.data_platform.maintenance.contracts import WorkspaceDataUpdatePlan
from alphalattice.control.observation_runtime.telemetry.progress import WorkspaceProgressPublisher
from alphalattice.control.product_host.composition.application_session import (
    WorkspaceApplicationSession,
)
from alphalattice.control.product_host.composition.decision_advancement import (
    SCHEMA as DECISION_ADVANCEMENT_SCHEMA,
)
from alphalattice.control.product_host.composition.decision_advancement import (
    DecisionAdvancementApplication,
    DecisionAdvancementCommand,
)
from alphalattice.control.product_host.composition.evidence_review_application import (
    EvidenceReviewApplication,
    ReviewOutcome,
    agent_answer_result,
)
from alphalattice.control.product_host.composition.evidence_review_bundles import (
    ANSWER_FIELDS,
    PACKET_SELECTOR_CODES,
    STALE_BUNDLE_CODES,
    EvidenceReviewBundles,
    agent_bundle_refusal,
    sealed_bounds_exceeded,
)
from alphalattice.control.product_host.composition.evidence_review_delivery import (
    EvidenceReviewDelivery,
)
from alphalattice.control.product_host.composition.evidence_review_projection import (
    EvidenceCroProjector,
)
from alphalattice.control.product_host.composition.feature_trials import (
    FeatureTrials,
    trial_reads_once,
)
from alphalattice.control.product_host.composition.goals import GoalApplication
from alphalattice.control.product_host.composition.pending_decisions import (
    all_data_issues,
    pending_decisions,
)
from alphalattice.control.product_host.composition.plain_refusals import (
    SELECTOR_CODES,
    STALE_PACKET,
    deferral,
    explain,
    refused,
    stale_packet_refusal,
    task_record_refusal,
    unit_refusal,
)
from alphalattice.control.product_host.composition.portfolio_application import (
    AdmittedPortfolioResearch,
    PlannedPortfolioResearch,
    PortfolioResearchApplication,
)
from alphalattice.control.product_host.composition.portfolio_result_context import (
    installed_temporal_statements,
    saved_portfolio_context,
)
from alphalattice.control.product_host.composition.portfolio_updates import (
    TASK_KIND as PORTFOLIO_UPDATE_TASK_KIND,
)
from alphalattice.control.product_host.composition.portfolio_updates import (
    PortfolioUpdateApplication,
    PortfolioUpdateCommand,
)
from alphalattice.control.product_host.composition.research_experiments import (
    LEDGER_READ_OPERATIONS,
    ResearchExperimentApplication,
    review_requests,
    verified_study_evidence,
)
from alphalattice.control.product_host.composition.research_experiments import (
    TASK_KIND as RESEARCH_EXPERIMENT_TASK_KIND,
)
from alphalattice.control.product_host.composition.research_history import ResearchHistory
from alphalattice.control.product_host.composition.research_update_automation import (
    ResearchUpdateAutomation,
)
from alphalattice.control.product_host.composition.research_workspace import (
    RESEARCH_WORKSPACE_MANIFEST_NAME,
    ResearchWorkspaceError,
    ResearchWorkspaceManifest,
    ResearchWorkspaceManifestHolder,
    admit_research_workspace,
    read_research_workspace_manifest,
    update_research_workspace_manifest,
)
from alphalattice.control.product_host.composition.resource_estimates import (
    MEMORY_INSUFFICIENT,
    gate_for,
)
from alphalattice.control.product_host.composition.strategy_activation import (
    StrategyActivation,
)
from alphalattice.control.product_host.composition.strategy_calibration import (
    TASK_KIND as STRATEGY_CALIBRATION_TASK_KIND,
)
from alphalattice.control.product_host.composition.strategy_calibration import (
    StrategyCalibrationApplication,
    StrategyCalibrationCommand,
)
from alphalattice.control.product_host.composition.strategy_scoring import (
    TASK_KIND as STRATEGY_SCORE_TASK_KIND,
)
from alphalattice.control.product_host.composition.strategy_scoring import (
    StrategyScoreCommand,
    StrategyScoringApplication,
)
from alphalattice.control.product_host.composition.strategy_tasks import (
    StrategyRead,
    latest_by_strategy,
    planned,
    strategy_task,
)
from alphalattice.control.product_host.composition.task_recovery import (
    TaskAttentionFact,
    TaskRecoveryView,
    build_task_recovery_view,
    stop_detail,
)
from alphalattice.control.product_host.composition.task_supervision import TaskSupervisor
from alphalattice.control.product_host.composition.upgrade_overview import (
    acknowledge_upgrade,
    installed_study_identities,
    upgrade_overview,
)
from alphalattice.control.product_host.composition.verification_sweep import (
    StudyVerificationSweep,
    StudyVerificationSweepCommand,
)
from alphalattice.control.product_host.data_preparation.application import (
    TASK_KIND as PREPARATION_TASK_KIND,
)
from alphalattice.control.product_host.data_preparation.application import (
    WorkspacePreparationApplication,
    WorkspacePreparationCommand,
)
from alphalattice.control.product_host.data_preparation.feature_research import (
    ResearchFeatureBuildApplication,
)
from alphalattice.control.product_host.data_preparation.input_capture import (
    ResearchInputCaptureApplication,
)
from alphalattice.control.product_host.data_preparation.model_training import (
    ModelTrainingInputApplication,
)
from alphalattice.control.product_host.data_preparation.remediation import (
    WorkspaceDataIssueApplication,
)
from alphalattice.control.product_host.data_preparation.research_strategy import (
    ResearchStrategyPreparation,
)
from alphalattice.control.product_host.maintenance.data_update import (
    DATA_UPDATE_TASK_KIND,
    WorkspaceDataUpdateApplication,
    WorkspaceDataUpdateCommand,
    installed_data_update_binding,
    read_workspace_inputs,
)
from alphalattice.control.product_host.publication.goals import GoalStore
from alphalattice.control.product_host.research_authoring.feature_research import (
    ResearchFeatureDefinitions,
)
from alphalattice.control.product_host.research_authoring.risk_reports import RiskReportLinks
from alphalattice.control.product_host.research_authoring.timing import binding_temporal_scope
from alphalattice.control.product_host.storage.backup import (
    GENERATIONS_KEPT,
    WorkspaceBackupError,
    WorkspaceBackups,
    backup_answer,
    take_automatic_backup,
)
from alphalattice.control.product_host.storage.input_references import ResearchInputStorage
from alphalattice.control.product_host.storage.inventory import StorageInventoryError
from alphalattice.control.product_host.storage.retention import (
    StorageRetentionError,
    require_no_pending_cleanup,
)
from alphalattice.control.research_program.authoring.document import (
    load_authoring_document,
)
from alphalattice.control.task_control.contracts import (
    TaskLifecycle,
    TaskRecord,
    TaskReplan,
    TaskSafeProjection,
)
from alphalattice.control.task_control.queue import (
    read_queue_setting,
    waiting_places,
    write_queue_setting,
)
from alphalattice.control.task_control.registry import (
    LEDGER_REBUILT_NEXT,
    DuckDbTaskControlRegistry,
    TaskNotFoundError,
    TaskQueueFull,
    TaskQueueHeadAuthorityError,
    TaskRecordAuthorityError,
    TaskTransitionRejected,
    TaskVersionStale,
)
from alphalattice.control.task_control.runner import TaskHeartbeatReader, TaskHeartbeatReadout
from alphalattice.control.task_control.timing import read_stage_spans, task_timing
from alphalattice.control.workspace_runtime.content_store import (
    CommittedIndex,
    ContentAddressedStoreError,
    verified_model_read_scope,
)
from alphalattice.control.workspace_runtime.network_access import (
    network_access,
    set_network_access,
)
from alphalattice.evidence.alternative_evidence.publication.artifacts import (
    verified_evidence_records,
)
from alphalattice.evidence.alternative_evidence.runtime.execution import (
    MAXIMUM_CPU_BUDGET,
    CpuBudgetStore,
    budget_cores,
    machine_load,
    plan_execution,
)
from alphalattice.evidence.alternative_evidence.runtime.progress import publish_evidence_work
from alphalattice.evidence.alternative_evidence.storage.inventory import EvidenceStorageBinding
from alphalattice.foundation.feature_engine.catalog.service import FeatureCatalogCrudError
from alphalattice.interface.local_application.activity import (
    ACTIVITY_REFUSAL_DAYS,
    ActivityReadQuery,
    ExternalActivityEventDocument,
    OperationObserver,
    observed_operation,
    refusal_code,
)
from alphalattice.interface.local_application.cli_contract import (
    REQUEST_PROVENANCE,
    RequestProvenance,
    refusal_words,
    worded_refusal,
)
from alphalattice.interface.local_application.cli_contract import (
    refused as is_refusal,
)
from alphalattice.interface.local_application.dispatcher import (
    CommandAdmission,
    CommandSubmission,
    LocalBackgroundDispatcher,
    TaskVersionMovedError,
)
from alphalattice.interface.local_application.evidence_cro import evidence_cro_body
from alphalattice.interface.local_application.failure_codes import (
    located_failure,
    owner_failure_code,
    public_failure,
    safe_failure_code,
    typed_failures,
)
from alphalattice.interface.local_application.goals import (
    Goal,
    GoalAcceptedAnswerReceipt,
    GoalSession,
)
from alphalattice.interface.local_application.native_bridge import (
    HOSTS,
    NativeBridgeError,
    NativeResearchBinding,
    deliver_accepted_answer,
    file_bound_fact,
    session_project,
    set_usage_reading,
    usage_reading_answer,
)
from alphalattice.interface.local_application.native_setup import (
    admitted_session_project,
    autobind_root,
    autobind_session,
)
from alphalattice.interface.local_application.operations import GRAMMAR, OPERATIONS, PERSON_ONLY
from alphalattice.interface.local_application.portfolio_research import (
    FrozenCandidateProjection,
    LocalApplicationError,
    LocalPortfolioResearchService,
    OperationCaller,
    PortfolioResearchOperationRequest,
    PortfolioResearchRequestDocument,
    _spec_document,
    public_control_document,
    spec_from_document,
)
from alphalattice.investment.alpha_research.scores.model_renewal import (
    verified_lifecycle_admissions,
)
from alphalattice.investment.portfolio_strategy_lab.application.contracts import (
    PortfolioResearchResult,
    PortfolioResearchSpec,
)
from alphalattice.investment.portfolio_strategy_lab.application.decision_updates import (
    PortfolioUpdatePublication,
    portfolio_update_positions,
)
from alphalattice.investment.portfolio_strategy_lab.application.resolution import (
    StrategyPortfolioResolver,
)
from alphalattice.investment.portfolio_strategy_lab.application.strategy_package import (
    FrozenStrategyPackage,
)
from alphalattice.investment.portfolio_strategy_lab.application.task import (
    portfolio_research_task_input,
)
from alphalattice.investment.portfolio_strategy_lab.reporting.static import (
    format_book_change,
    format_book_weight,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.oversight.chief_risk_officer.decision.book_evidence import (
    BookSelector,
    PortfolioEvidenceReviewError,
)
from alphalattice.oversight.chief_risk_officer.runtime.portfolio_review_task import (
    TASK_KIND as CRO_REVIEW_TASK_KIND,
)
from alphalattice.protocols.actor_execution.answers import AgentAnswerRecord, AgentRun
from alphalattice.protocols.actor_execution.bundles import AgentBundleRecord, answer_read
from alphalattice.protocols.actor_execution.contracts import AgentExecutionBinding
from alphalattice.protocols.research_authoring.contracts import AuthoringError

PORTFOLIO_RUN_COMMAND = "portfolio_public_development_replay"


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    """One result an owner read back after a command returned."""

    artifact_kind: str
    artifact_hash: str


_PUBLISHED_ARTIFACTS: dict[str, str] = {
    PORTFOLIO_RUN_COMMAND: "PortfolioResearchResult",
    RESEARCH_EXPERIMENT_TASK_KIND: "ResearchExecutionEvidence",
}
"""The command kinds whose result `artifact_reference` opens, and the artifact each publishes."""
_Observed = TypeVar("_Observed")


@runtime_checkable
class _AdmitsTasks(Protocol):
    """An owner that admits Tasks declares how each kind it admits is planned again."""

    @property
    def replans(self) -> tuple[TaskReplan, ...]: ...


@dataclass
class PortfolioRunCommand:
    """One admitted Portfolio run, as the shared dispatcher sees it."""

    application: PortfolioResearchApplication
    spec: PortfolioResearchSpec | None = None
    planned: PlannedPortfolioResearch | None = None
    recovering: bool = False
    _admitted: AdmittedPortfolioResearch | None = field(default=None, init=False, repr=False)

    @property
    def command_kind(self) -> str:
        """Identify the portfolio run command consumed by the background dispatcher.

        Returns:
            Installed portfolio run command kind.
        """
        return PORTFOLIO_RUN_COMMAND

    def admit(self) -> CommandAdmission:
        """Admit the explicit portfolio spec once and retain its exact admission.

        Returns:
            Task identity and lifecycle supplied by the deterministic application.

        Raises:
            ValueError: This is a recovery command or its declared spec is absent.
        """
        if self.recovering:
            raise ValueError("portfolio_application.recovery_command_cannot_admit")
        if self.spec is None:
            raise ValueError("portfolio_application.command_spec_absent")
        self._admitted = self.application.admit(spec=self.spec, planned=self.planned)
        return CommandAdmission(task_id=self._admitted.task_id, lifecycle=self._admitted.lifecycle)

    def execute(self, task_id: UUID, *, expected_task_hash: str | None = None) -> None:
        """Execute or recover the exact admitted portfolio task.

        Task Control owns cancellation races. Only the named dispatch/result cancellation refusals
        are suppressed when the task is already cancelling or cancelled; unrelated failures
        propagate.

        Args:
            task_id: Exact task to execute or recover.
            expected_task_hash: Optional optimistic task identity.
        """
        if self.recovering:
            self.application.recover(task_id=task_id, expected_task_hash=expected_task_hash)
            return
        if self.spec is None:
            raise ValueError("portfolio_application.command_spec_absent")
        admitted = self._admitted
        if admitted is None or admitted.task_id != task_id:
            admitted = self.application.admit(spec=self.spec)
        try:
            self.application.execute(admitted, expected_task_hash=expected_task_hash)
        except ValueError as error:
            # Cancellation may win between the application's QUEUED read and
            # the runner's claim. Task Control owns that outcome; it is not a
            # failed worker. Keep the synchronous application's strict result
            # contract and never hide an unrelated error beside a cancellation.
            lifecycle = self.application.session.task_control_registry.task(task_id).lifecycle
            if lifecycle in {TaskLifecycle.CANCEL_REQUESTED, TaskLifecycle.CANCELLED} and str(
                error
            ) in {
                "portfolio_application.task_not_dispatched",
                "portfolio_application.task_not_succeeded:CANCEL_REQUESTED",
                "portfolio_application.task_not_succeeded:CANCELLED",
            }:
                return
            raise

    @classmethod
    def recover(cls, *, application: PortfolioResearchApplication) -> PortfolioRunCommand:
        """Drive owed work from the recovery-required Task's durable input."""
        return cls(application=application, recovering=True)


def _read_at(request: PortfolioResearchOperationRequest) -> datetime | None:
    """When a review's dossier was read, as its bundle or first part sealed it (V255)."""
    return (
        None if request.review_read_at is None else datetime.fromisoformat(request.review_read_at)
    )


RUN_FORWARD_WORDS: Final = (
    "A book that runs forward to next positions comes from the research strategy: its "
    "controls name the required Alpha and Risk studies over their whole support, which need no "
    "Factor study; then prepare and install it, run its whole-support book and review that "
    "book. A Lab book is research only and is never activated."
)
"""The first use's shortest way, ahead of the inputs' Lab flows (FLOW-3)."""
INSTALLED_BOOK_WORDS: Final = (
    "The installed strategy runs its whole-support historical book from its controls; review "
    "that book, then read its exact activation offer."
)


@dataclass(slots=True)
class PortfolioResearchOperations:
    """One product operation owner for every human and Agent caller."""

    service: LocalPortfolioResearchService | None
    dispatcher: LocalBackgroundDispatcher
    application: PortfolioResearchApplication | None
    workspace_manifest: ResearchWorkspaceManifest
    workspace_session: WorkspaceApplicationSession
    review: EvidenceReviewApplication | None
    heartbeats: TaskHeartbeatReader
    """Read-only access to the runners' operational heartbeat sidecars, by execution id;
    what it reads never raises past its boundary and never changes what Task Control says."""
    data_update: WorkspaceDataUpdateApplication | None = None
    manifests: ResearchWorkspaceManifestHolder | None = None
    """The one holder of the workspace manifest the applications read (V182); made from
    `workspace_manifest` when the composition passes none."""
    recover_task: Callable[..., tuple[UUID, ...]] | None = None
    """`(task_id, *, expected_task_hash=None)`: the service's own resume. A confirmed
    version rides the dispatcher's queue item to Task Control's start boundary."""
    recoverable_task_kinds: Callable[[], frozenset[str]] | None = None
    """The Task kinds this service can resume, read at the moment a recovery view is built."""
    resume_refusal: Callable[[TaskRecord], str | None] | None = None
    """Why a Task cannot resume under what is installed now, asked before any resume; the
    service's own check (`LocalPortfolioWebSession.resume_refusal`). Nothing is written."""
    observer: OperationObserver | None = None
    """Records entry, return and refusal of the operations it classifies as
    activity. Optional: every path below runs identically without one, and an
    observer that raises is counted here and otherwise ignored."""

    read_native_usage: Callable[[], dict[str, object]] | None = None
    """Reads the bound Sessions' usage now (`NativeUsageReader.read`); None without a Host."""
    native_projects: set[Path] = field(default_factory=set)
    """Agent projects this Host admitted a Session binding from, for a workspace kept outside
    them; found again after a restart by the next bind or milestone (FLOW-1)."""
    autobound: set[tuple[str, str]] = field(default_factory=set)
    """The Sessions this Host checked for a binding, once each per run (AUTOBIND)."""
    goals_named: set[tuple[str, str, str]] = field(default_factory=set)
    """Each Session and goal whose holding this Host filed, once per run (STOPS-1)."""

    observer_failures: int = field(default=0, init=False)
    last_observer_failure: str | None = field(default=None, init=False)
    _manifest_stamp: tuple[int, int] | None = field(default=None, init=False)
    """The manifest file's stamp when it last read as the held manifest (W10)."""
    """Exception class of the last observer failure; never its message."""

    scoring: StrategyScoringApplication | None = field(init=False, default=None)
    calibration: StrategyCalibrationApplication | None = field(init=False, default=None)
    updates: PortfolioUpdateApplication | None = field(init=False, default=None)
    research_updates: DecisionAdvancementApplication | None = field(init=False, default=None)
    automation: ResearchUpdateAutomation | None = field(init=False, default=None)
    activations: StrategyActivation | None = field(init=False, default=None)
    experiments: ResearchExperimentApplication = field(init=False)
    model_contracts: dict[str, dict[str, Any]] = field(init=False, default_factory=dict)
    """Each Alpha model's contract answer by its identity, for the Host's life (V353)."""
    goals: GoalApplication = field(init=False)
    preparation: WorkspacePreparationApplication = field(init=False)
    data_issues: WorkspaceDataIssueApplication = field(init=False)
    input_capture: ResearchInputCaptureApplication = field(init=False)
    model_training_inputs: ModelTrainingInputApplication = field(init=False)
    feature_builds: ResearchFeatureBuildApplication = field(init=False)
    trials: FeatureTrials = field(init=False)
    research_strategies: ResearchStrategyPreparation = field(init=False)
    storage: ResearchInputStorage = field(init=False)
    _packages: dict[str, FrozenStrategyPackage] = field(init=False, repr=False)
    _study_identities: dict[str, str] | None = field(default=None, init=False, repr=False)
    sweep: StudyVerificationSweep = field(init=False)
    supervisor: TaskSupervisor = field(init=False)

    def __post_init__(self) -> None:
        """Compose workspace owners and resume retained feature trials.

        Preparation, storage, data, experiments and goals share the retained writer session. When
        portfolio research is installed, wire its scoring, calibration, update and decision owners
        and the bounded dispatcher idle callback.
        """
        if self.manifests is None:
            self.manifests = ResearchWorkspaceManifestHolder(self.workspace_manifest)
        self.experiments = ResearchExperimentApplication(
            session=self.workspace_session,
            manifest=self.manifests,
            dispatcher=self.dispatcher,
            clock=self.dispatcher.clock,
        )
        self.supervisor = TaskSupervisor(self)
        self.sweep = StudyVerificationSweep(
            session=self.workspace_session,
            experiments=self.experiments,
            installed=lambda: str(canonical_hash(self._installed_study_identities())),
            clock=self.dispatcher.clock,
        )
        self.goals = GoalApplication(
            GoalStore(
                self.workspace_session.workspace / "artifacts", self.workspace_manifest.workspace_id
            ),
            self.dispatcher.clock,
            lambda request, actor: self.execute(request, caller=cast(OperationCaller, actor)),
            lambda task_id: self.workspace_session.task_control_registry.task(task_id).admitted_at,
            self._task_facts,
            workspace=self.workspace_session.workspace,
        )
        self.preparation = WorkspacePreparationApplication(
            self.workspace_session,
            clock=self.dispatcher.clock,
            provider=self.data_update.provider if self.data_update else None,
            source_loader=self.data_update.source_loader if self.data_update else None,
        )
        self.storage = ResearchInputStorage(
            self.workspace_session,
            evidence=self._evidence_storage_binding(),
            clock=self.dispatcher.clock,
        )
        self._bind_evidence_storage_admission()
        self.data_issues = WorkspaceDataIssueApplication(
            self.workspace_session, self.dispatcher.clock
        )
        self.input_capture = ResearchInputCaptureApplication(
            self.workspace_session, clock=self.dispatcher.clock
        )
        self.model_training_inputs = ModelTrainingInputApplication(
            self.workspace_session, clock=self.dispatcher.clock
        )
        self.feature_builds = ResearchFeatureBuildApplication(
            self.workspace_session, clock=self.dispatcher.clock
        )
        self.research_strategies = ResearchStrategyPreparation(
            self.workspace_session,
            clock=self.dispatcher.clock,
            read_experiment=lambda task_id: self.experiments.readback(task_id),
            list_experiments=self.experiments.listing,
        )
        self.trials = FeatureTrials(
            workspace=self.workspace_session.workspace,
            experiments=self.experiments,
            feature_builds=self.feature_builds,
            dispatcher=self.dispatcher,
            clock=self.dispatcher.clock,
        )
        self.dispatcher.on_idle = self._worker_idle
        # A trial the Host was running when it stopped moves on when it starts again, and a
        # data update's backup it never took is taken.
        self.trials.advance_all()
        self._take_backup()
        self._packages = {}
        if self.review is not None:
            self.review.read_experiment = self.experiments.readback
        if self.application is None:
            return
        assert self.review is not None
        self._packages = {
            package.strategy_id: package
            for package in self.application.resolver.installed_packages().values()
        }
        self.scoring = StrategyScoringApplication(
            session=self.workspace_session,
            manifest=self.manifests,
            packages=lambda: self._packages,
            clock=self.dispatcher.clock,
        )
        self.calibration = StrategyCalibrationApplication(
            scoring=self.scoring, clock=self.dispatcher.clock
        )
        self.updates = PortfolioUpdateApplication(
            application=self.application,
            calibration=self.calibration,
            manifest=self.manifests,
            clock=self.dispatcher.clock,
        )
        if self.data_update is not None:
            self.data_update.portfolio_obligations = self.updates.valuation_obligations
        if self.data_update is not None:
            self.research_updates = DecisionAdvancementApplication(self.updates, self.data_update)
        manifests = self.manifests
        assert manifests is not None
        self.activations = StrategyActivation(
            session=self.workspace_session,
            manifest=lambda: manifests.current,
            packages=lambda: self._packages,
            ledger=self.application.ledger,
            read_experiment=self.experiments.readback,
            read_information=self.experiments.summary,
            read_review=self._book_review_standing,
            clock=self.dispatcher.clock,
            hold=self._hold_activation,
        )
        self.review.read_update = self._read_review_update
        self.review.installed_temporal_statements = partial(
            installed_temporal_statements, self.workspace_session.workspace, self.workspace_manifest
        )
        self.automation = ResearchUpdateAutomation(
            index=CommittedIndex(self.application.ledger.root, self.application.ledger.content),
            workspace_manifest_hash=self.workspace_manifest.manifest_hash,
            installed_package_ids=tuple(p for p in self._packages if self._update_configured(p)),
            clock=self.dispatcher.clock,
            execute=lambda request: self.execute(request, caller="SERVICE_AUTOMATION"),
        )
        self.dispatcher.on_idle = self._worker_idle

    def _worker_idle(self) -> None:
        """The Task worker went idle: wake the update automation, move feature trials on, admit
        the sweep if it is due, then take a data update's waiting backup while no Task waits."""

        if self.automation is not None:
            self.automation.command_completed()
        self.trials.advance_all()
        self.sweep_if_due()
        self._take_backup()

    def _take_backup(self) -> None:
        take_automatic_backup(
            self.workspace_session.workspace,
            clock=self.dispatcher.clock,
            gate=self.workspace_session.mutation_gate,
            stop=lambda: self.dispatcher.closing or self._tasks_wait(),
        )

    def _tasks_wait(self) -> bool:
        return any(
            task.lifecycle in {TaskLifecycle.QUEUED, TaskLifecycle.RUNNING, TaskLifecycle.DEFERRED}
            for task in self.workspace_session.task_control_registry.tasks()
        )

    def sweep_if_due(self) -> None:
        """Admit the verification sweep when it is due and nothing else waits (V89).

        The workspace's lowest priority, it runs only in time the Tasks leave idle: never while
        a deferral holds the running place, which it would only queue behind (V604).
        """
        with self.sweep.admission:
            if self.sweep.due() and not self._tasks_wait():
                self.dispatcher.submit(StudyVerificationSweepCommand(self.sweep, "DUE"))

    def verify_all(self) -> dict[str, object]:
        """Admit a whole sweep at once: every saved study verified in full (V89)."""
        studies = len(self.sweep.saved_studies())
        sent = self.dispatcher.submit(StudyVerificationSweepCommand(self.sweep, "ALL"))
        if sent.task_id is None:
            return {
                "status": "REFUSED",
                "failure_code": "task_control.queue_full"
                if sent.disposition == "REFUSED_QUEUE_FULL"
                else "study_verification_sweep.not_admitted",
                "detail": sent.refusal_detail,
                "saved_studies": studies,
            }
        return {
            "status": "ADMITTED",
            "task_id": str(sent.task_id),
            "lifecycle": sent.lifecycle,
            "saved_studies": studies,
            "next_requests": {"status": {"operation": "STATUS", "task_id": str(sent.task_id)}},
        }

    def _data_decisions(self, caller: str) -> str | None:
        """The first-use delegation the reader's goal gives it over its preparation's data issues.

        A read is attributed to no goal, so the goal is the one the request names or its
        session holds, as the confirm it leads to will be attributed.
        """
        try:
            goal = self.goals.attributed_goal(REQUEST_PROVENANCE.get())
        except ValueError:
            goal = None
        return self.goals.data_decisions(goal, caller)

    def _task_facts(self, task_id: UUID) -> tuple[str, str, datetime | None] | None:
        """A Task's kind, lifecycle and canonical update clock; nothing when absent."""
        try:
            task = self.workspace_session.task_control_registry.task(task_id)
        except TaskNotFoundError:
            return None
        return task.task_kind, task.lifecycle.value, task.updated_at

    def execute(
        self,
        request: PortfolioResearchOperationRequest,
        *,
        caller: OperationCaller = "HUMAN",
        agent_execution: AgentExecutionBinding | None = None,
    ) -> dict[str, object]:
        """Run one operation, recorded under the open goal it works for, if any (OP13).

        A request works for the goal its client names, else for the goal its agent session
        took; a recorded operation is then kept in the goal's record, and its activity rows
        name the goal. Reads and the goal's own operations are never attributed.
        """
        provenance = REQUEST_PROVENANCE.get()
        try:
            goal = (
                None
                if request.operation.startswith("GOAL_")
                or not observed_operation(request.operation)
                else self.goals.attributed_goal(provenance)
            )
            if goal is None and _opens_goal(request.operation, provenance):
                goal = self._open_session_goal(request.operation, provenance)
            if not request.operation.startswith("GOAL_"):
                # Before the work, which may deliver to the Session's binding (AUTOBIND).
                self._autobind(provenance, goal)
                self._name_goal(provenance, goal)
        except ValueError as error:
            # A request naming a goal it cannot count toward is refused before it runs.
            code = public_failure(error, "goal.refused")
            return worded_refusal(
                {
                    "status": "REFUSED",
                    "failure_code": code,
                    "refused": code,
                    "next_action": "NAME_AN_OPEN_GOAL_OR_NONE",
                }
            )
        if goal is None:
            # Every refusal leaves with its words and a way on (OP4, V449).
            answered = worded_refusal(
                _carried(
                    self._execute_observed(request, caller=caller, agent_execution=agent_execution),
                    request,
                ),
                workspace=self.workspace_session.workspace,
            )
            if request.operation.startswith("GOAL_"):
                # After a goal's own operation, so the bound fact is filed under it.
                held = None
                if request.operation in {"GOAL_OPEN", "GOAL_TAKE"}:
                    with suppress(Exception):
                        held = self.goals.attributed_goal(provenance)
                self._autobind(provenance, held)
                self._name_goal(provenance, held)
            return answered
        delegation = self.goals.delegation(goal, request.operation, caller)
        scope = REQUEST_PROVENANCE.set(
            replace(
                provenance or RequestProvenance(),
                goal_id=str(goal.goal_id),
                delegation=delegation,
            )
        )
        try:
            # The observer records the actual executor; delegation supplies authority only
            # when the existing dispatcher checks the person's step.
            body = self._execute_observed(
                request,
                caller=caller,
                agent_execution=agent_execution,
            )
        finally:
            REQUEST_PROVENANCE.reset(scope)
        self._observe(
            lambda: self.goals.attribute(goal, request, body, provenance, delegation=delegation)
        )
        self._settle_first_use(goal)
        # The goal the request counted toward, which the answer's context names (V401).
        return worded_refusal(
            _carried({**body, "attributed_goal_id": str(goal.goal_id)}, request),
            workspace=self.workspace_session.workspace,
        )

    def _open_session_goal(
        self, operation: str, provenance: RequestProvenance | None
    ) -> Goal | None:
        """Open a goal for an identified Session's first research request when it holds none.

        Under the person's hands-off rule the work is recorded from the first call: the goal
        names the request that began it, and the Session holds it as `goal open` would. The
        Session may open its own goal, the person's sentence for a first use, to replace it.
        Nothing here refuses the request (AUTOBIND).
        """
        assert provenance is not None and provenance.vendor is not None
        command = GRAMMAR[operation]
        name = "Claude Code" if provenance.vendor == "claude-code" else "Codex"
        declaration = {
            "title": f"Session work: {command.noun} {command.verb}",
            "objective": (
                f"The work this {name} session began with `{command.noun} {command.verb}`: "
                f"{command.purpose}"
            )[:2400],
            "kind": "RESEARCH",
            "scope": (
                "Opened by the Host at the session's first research request, which held no "
                "goal; the session may open its own goal to replace it."
            ),
            "criteria": [
                {
                    "criterion_id": "recorded",
                    "text": "The session's requests, Tasks and results are recorded here.",
                }
            ],
        }
        try:
            opened = self.goals.operate(
                PortfolioResearchOperationRequest(
                    operation="GOAL_OPEN", goal_declaration=declaration
                ),
                "EXTERNAL_AUTOMATION",
                provenance,
            )
            if opened.get("status") != "GOAL_SAVED":
                return None
            return self.goals.store.head(UUID(str(opened["goal_id"])))
        except Exception:
            return None

    def _autobind(self, provenance: RequestProvenance | None, goal: Goal | None = None) -> None:
        """Bind the Session a request names, once, when nothing binds it to this workspace.

        A Session is bound by working: no `session bind`, no configure. The binding lives in
        the workspace, the bound fact names the Session in Team and its Goal from the first
        call, and its usage is read in the background. A Codex child binds its lead. A goal
        the Session already holds is named by the same fact (STOPS-1). Any failure leaves the
        request as it is (AUTOBIND).
        """
        if provenance is None or provenance.vendor not in HOSTS or not provenance.session:
            return
        key = (provenance.vendor, provenance.session)
        if key in self.autobound:
            return
        self.autobound.add(key)
        workspace = Path(self.workspace_session.workspace)
        try:
            if self._served(key, workspace):
                return
            above = None
            with suppress(NativeBridgeError):
                above = session_project(workspace.resolve(), key[0])
            root, lead = autobind_session(workspace, *key, project=above)
            if above is not None:
                self.native_projects.add(above)
            self.autobound.add((key[0], lead))
            binding = NativeResearchBinding.read(root, session=(key[0], lead))
            observer = self.observer
            if binding is not None and observer is not None:
                file_bound_fact(
                    root,
                    binding,
                    at=self.dispatcher.clock(),
                    publish=lambda document: self.declare_event(observer, document),
                    goal_hash=None if goal is None else goal.goal_hash,
                )
                if goal is not None:
                    self.goals_named.add((*key, str(goal.goal_id)))
        except Exception:
            return
        if self.read_native_usage is not None:
            threading.Thread(target=self.read_native_usage, daemon=True).start()

    def _name_goal(self, provenance: RequestProvenance | None, goal: Goal | None) -> None:
        """File, once, the goal a bound Session holds, so its Sessions row reads it (STOPS-1).

        A goal the Host opened for the Session, or one it opened or took itself, is named by
        the same product fact as its binding; the Session never has to declare it. Any
        failure leaves the request as it is.
        """
        observer = self.observer
        if (
            observer is None
            or provenance is None
            or provenance.vendor not in HOSTS
            or not provenance.session
            or goal is None
        ):
            return
        with suppress(Exception):
            key = (provenance.vendor, provenance.session, str(goal.goal_id))
            if key in self.goals_named:
                return
            self.goals_named.add(key)
            workspace = Path(self.workspace_session.workspace)
            found = self._session_binding((provenance.vendor, provenance.session), workspace)
            if found is not None:
                file_bound_fact(
                    *found,
                    at=self.dispatcher.clock(),
                    publish=lambda document: self.declare_event(observer, document),
                    goal_hash=goal.goal_hash,
                )

    def _session_binding(
        self, session: tuple[str, str], workspace: Path
    ) -> tuple[Path, NativeResearchBinding] | None:
        """This exact Session's binding of this workspace, wherever it is kept.

        The configured project above the workspace, a project this Host admitted, or the
        workspace's own folder where the Host bound the Session itself (AUTOBIND).
        """
        projects: list[Path] = []
        with suppress(NativeBridgeError):
            above = session_project(workspace.resolve(), session[0])
            admitted_session_project(workspace, above, session[0])
            projects.append(above)
        projects.extend(sorted(self.native_projects))
        projects.append(autobind_root(workspace))
        for project in projects:
            with suppress(NativeBridgeError, OSError):
                binding = NativeResearchBinding.read(project, session=session)
                if binding is not None and binding.workspace.resolve() == workspace.resolve():
                    return project, binding
        return None

    def _served(self, session: tuple[str, str], workspace: Path) -> bool:
        """Whether a binding of this Session, or of its lead, already serves this workspace."""
        projects = {autobind_root(workspace), *self.native_projects}
        with suppress(NativeBridgeError):
            projects.add(session_project(workspace.resolve(), session[0]))
        return any(
            binding.workspace.resolve() == workspace.resolve() and binding.serves(session)
            for project in projects
            for binding in NativeResearchBinding.bindings(project)
        )

    def _delegating_goal(self) -> Goal | None:
        """The first-use goal whose delegation this request runs under, if any (V452)."""
        provenance = REQUEST_PROVENANCE.get()
        if provenance is None or provenance.delegation is None or provenance.goal_id is None:
            return None
        return self.goals.store.head(UUID(provenance.goal_id))

    def _settle_first_use(self, goal: Goal | None) -> None:
        """Close the network a first-use goal's delegation opened, once the goal is over.

        The goal grants nothing after it (V452): its end, by submission, abandonment or its
        hours passing, closes what its delegation opened, recorded in its ledger.
        """
        if goal is None or not self.goals.network_left_open(goal):
            return
        # Only a setting the delegation still holds is undone: a person who set the network
        # since keeps theirs (V460).
        held = network_access(self.workspace_session.workspace).set_by
        if held is None or held.get("delegation") != f"first-use-goal:{goal.goal_id}":
            return
        set_network_access(
            self.workspace_session.workspace,
            enabled=False,
            delegation=f"first-use-goal:{goal.goal_id}",
            until=self.goals.delegation_ends(goal),
        )
        self.goals.store.attribute(
            goal.goal_id,
            {
                "recorded_at": self.goals.clock().isoformat(),
                "operation": "NETWORK_ACCESS_SET",
                "status": "NETWORK_ACCESS",
                "task_id": None,
                "agent_vendor": None,
                "agent_session": None,
                "delegation": f"first-use-goal:{goal.goal_id}",
                "network_enabled": False,
            },
        )

    def _execute_observed(
        self,
        request: PortfolioResearchOperationRequest,
        *,
        caller: OperationCaller = "HUMAN",
        agent_execution: AgentExecutionBinding | None = None,
    ) -> dict[str, object]:
        """Run one finite operation; all domain authority remains below here.

        `caller` is supplied by the entry boundary that ran this call and is not
        readable from the request, so a page or a model cannot claim to be the
        other actor. Only operations that record who acted consult it.

        The observer, when present, sees the same request and caller before the
        owner runs and the owner's unaltered answer or exception afterwards. It
        cannot change either: a refusal is re-raised exactly as it was raised,
        and an observer that itself fails is counted and stepped over.
        """

        observer = self.observer
        provenance = REQUEST_PROVENANCE.get()
        # Delegated authority affects admission, never the observed caller.
        authority = "HUMAN" if provenance is not None and provenance.delegation else caller
        span = (
            None
            if observer is None
            else self._observe(lambda: observer.entered(request, caller=caller))
        )
        try:
            recovery = self._recovery_context(request)
            # One operation proves each lifecycle admission it reads, and verifies
            # each Evidence record it loads, once; the proof is released with the
            # operation, never kept across requests.
            if isinstance(recovery, dict):
                body = recovery
            else:
                execution_request = (
                    request
                    if recovery is None or request.operation == "DATA_UPDATE_PLAN"
                    else replace(request, recovery_task_id=None, recovery_task_hash=None)
                )
                with (
                    verified_model_read_scope(
                        reuse_verified=request.operation
                        in (
                            LEDGER_READ_OPERATIONS
                            | {
                                "REPORT",
                                "RESEARCH_UPDATE_READBACK",
                                "PORTFOLIO_UPDATE_READBACK",
                                "GOAL_SHOW",
                                "GOAL_EXPORT",
                                "GOAL_REFERENCE",
                            }
                        )
                    ),
                    verified_lifecycle_admissions(),
                    verified_study_evidence(
                        reuse_verified=request.operation in LEDGER_READ_OPERATIONS
                    ),
                    verified_evidence_records(),
                    trial_reads_once(),
                ):
                    body = typed_failures(
                        self._execute_within_memory(
                            execution_request, caller=authority, agent_execution=agent_execution
                        )
                    )
                if recovery is not None:
                    source, declaration, is_preview, normalized = recovery
                    body = (
                        self._recovery_preview_answer(source, declaration, body)
                        if is_preview
                        else self._recovery_admission_answer(source, normalized, body)
                    )
        except BaseException as error:
            if isinstance(error, TaskRecordAuthorityError):
                body = {
                    **refused("task_control.database_authority_unreadable"),
                    "refusals": [
                        task_record_refusal(task_id) for task_id in error.refused_task_ids
                    ],
                    "next_requests": {
                        "workspace": {"operation": "WORKSPACE_SHOW"},
                        "backups": {"operation": "WORKSPACE_BACKUPS"},
                    },
                }
            elif isinstance(error, TaskQueueHeadAuthorityError):
                body = self.task_queue_authority_refusal(error)
            elif isinstance(error, TaskNotFoundError) and (
                (
                    error.task_id == request.task_id
                    and request.operation in {"STATUS", "TASK_RECOVERY", "CANCEL", "RECOVER"}
                )
                or (
                    error.task_id == request.recovery_task_id
                    and request.recovery_task_id is not None
                )
            ):
                body = {
                    "status": "REFUSED",
                    "failure_code": "task_control.task_not_found",
                    "refused": "task_control.task_not_found",
                    "task_id": str(error.task_id),
                    "next_action": "DISCOVER_TASK_IN_CURRENT_WORKSPACE",
                    **explain("task_control.task_not_found"),
                    **(
                        {
                            "next_requests": {
                                "recovery": {
                                    "operation": "TASK_RECOVERY",
                                    "task_id": str(error.task_id),
                                }
                            }
                        }
                        if request.recovery_task_id is not None
                        else {}
                    ),
                }
            else:
                code = owner_failure_code(error)
                if observer is not None and span is not None:
                    failure = error
                    self._observe(lambda: observer.failed(span, failure))
                if observer is not None and code is not None:
                    raised = code
                    self._observe(
                        lambda: observer.refused(request.operation, raised, caller=caller)
                    )
                # Observing a refusal cannot turn its raised transport response into
                # an operation return. The entry boundary maps the original exception.
                raise
        if observer is not None and span is not None:
            self._observe(lambda: observer.returned(span, body))
        # Every refusal is counted, reads included: the one durable trace of where
        # requests fail (AC, OP14).
        refusal = refusal_code(body)
        if observer is not None and refusal is not None:
            self._observe(lambda: observer.refused(request.operation, refusal, caller=caller))
        return body

    def _observe(self, step: Callable[[], _Observed]) -> _Observed | None:
        """One observer step. Its failure is a fact about the observer, kept here."""

        try:
            return step()
        except Exception as error:
            self.observer_failures += 1
            self.last_observer_failure = type(error).__name__
            return None

    def _execute_within_memory(
        self,
        request: PortfolioResearchOperationRequest,
        *,
        caller: OperationCaller,
        agent_execution: AgentExecutionBinding | None,
    ) -> dict[str, object]:
        """A heavy plan's answer with its resource estimate; a heavy run refused before
        admission when its estimated peak exceeds the machine's available memory."""

        gate = gate_for(self.workspace_session.workspace)
        plan_hash = (
            request.experiment_plan_hash
            or request.update_plan_hash
            or request.preparation_plan_hash
        )
        refusal = gate.run_refusal(
            request.operation, plan_hash, self.workspace_session.task_control_registry
        )
        if refusal is not None:
            return refusal
        body = self._execute(request, caller=caller, agent_execution=agent_execution)
        gate.plan_answered(request.operation, body)
        task_id = body.get("task_id")
        if isinstance(task_id, str):
            gate.task_admitted(request.operation, plan_hash, task_id)
        return body

    def _execute(
        self,
        request: PortfolioResearchOperationRequest,
        *,
        caller: OperationCaller,
        agent_execution: AgentExecutionBinding | None,
    ) -> dict[str, object]:
        if caller == "SERVICE_AUTOMATION" and request.operation not in {
            "RESEARCH_UPDATE_PLAN",
            "RESEARCH_UPDATE_RUN",
            "RESEARCH_UPDATE_READBACK",
            # The Host's Forward prewarm reads only retained results and updates.
            "REPORT",
            "EXPERIMENT_READBACK",
            "PORTFOLIO_UPDATE_READBACK",
        }:
            raise ValueError("research_update.automation_operation_not_admitted")
        client = self._client_operation(request, caller=caller)
        if client is not None:
            return client
        self._refresh_prepared_inputs()
        try:
            if request.operation in {
                "GOAL_OPEN",
                "GOAL_REVISE",
                "GOAL_ATTACH",
                "GOAL_NOTE",
                "GOAL_TAKE",
                "GOAL_SUBMIT",
                "GOAL_ABANDON",
                "RUN",
                "RECOVER",
                "EXPERIMENT_RUN",
                "EXPERIMENT_PROMOTE",
                "EXPERIMENT_CONTINUE",
                "EXPERIMENT_CURATE",
                "EXPERIMENT_LINK_RISK",
                "EXPERIMENT_FOUNDATION_SEAL",
                "FREEZE",
                "DATA_CHANGE_CONFIRM",
                "DATA_ISSUE_CONFIRM",
                "DATA_ISSUE_DELEGATE",
                "DATA_ISSUE_REVOKE",
                "DATA_UPDATE_RUN",
                "WORKSPACE_PREPARE_CONFIRM",
                "RESEARCH_INPUT_CONFIRM",
                "MODEL_TRAINING_INPUT_PREPARE",
                "FEATURE_CATALOG_BUILD",
                "FEATURE_TRIAL",
                "RESEARCH_STRATEGY_PREPARE",
                "RESEARCH_STRATEGY_INSTALL",
                "RESEARCH_UPDATE_RUN",
                "RESEARCH_UPDATE_AUTOMATION_CONFIGURE",
                "STRATEGY_SCORE_RUN",
                "STRATEGY_CALIBRATION_RUN",
                "PORTFOLIO_UPDATE_RUN",
            }:
                require_no_pending_cleanup(self.workspace_session.workspace)
            common = self._workspace_operation(request, caller=caller)
        except (StorageRetentionError, StorageInventoryError) as error:
            return {"status": "REFUSED", "failure_code": error.failure_code}
        if common is not None:
            return common
        if request.operation.startswith("GOAL_"):
            try:
                if (
                    request.operation == "GOAL_OPEN"
                    and isinstance(request.goal_declaration, dict)
                    and request.goal_declaration.get("kind") == "FIRST_USE"
                    and any(t.lifecycle.value == "SUCCEEDED" for t in self.preparation.tasks())
                ):
                    # A first use is the one before the workspace's first preparation (V452).
                    raise ValueError("goal.first_use_after_preparation")
                answer = self.goals.operate(request, caller, REQUEST_PROVENANCE.get())
                if request.operation in {"GOAL_SUBMIT", "GOAL_ABANDON"}:
                    self._settle_first_use(self.goals.first_use())
                return answer
            except ValidationError as error:
                return {
                    "status": "REFUSED",
                    **located_failure(error, "goal.document_invalid"),
                    "next_action": "READ_GOAL_SCHEMA_AND_CORRECT_DOCUMENT",
                    "next_requests": {"schema": {"operation": "GOAL_SCHEMA"}},
                }
            except (ValueError, KeyError, OSError) as error:
                return goal_refusal(error, request)
        if request.operation == "PENDING_DECISIONS":
            return self.pending_decisions(caller=caller)
        if request.operation == "ACTIVITY_REFUSALS":
            if self.observer is None:
                return refused("activity.storage_unavailable")
            return self.observer.refusals(days=request.view_last_days or ACTIVITY_REFUSAL_DAYS)
        if request.operation == "NETWORK_ACCESS":
            return network_access(self.workspace_session.workspace).body()
        if request.operation == "NETWORK_ACCESS_SET":
            # Network access is an authority, like the background update settings: a
            # person sets it; a client or an Agent reads it.
            if caller != "HUMAN":
                return refused("local_application.network_access_human_only")
            assert request.network_enabled is not None
            # A first-use goal's step holds only until its delegation ends (OP19, V452).
            delegated = self._delegating_goal()
            return set_network_access(
                self.workspace_session.workspace,
                enabled=request.network_enabled,
                delegation=None if delegated is None else f"first-use-goal:{delegated.goal_id}",
                until=None if delegated is None else self.goals.delegation_ends(delegated),
            ).body()
        if request.operation in {"UPGRADE_OVERVIEW", "UPGRADE_ACKNOWLEDGE"}:
            return self.upgrade(request)
        if request.operation == "RESEARCH_HISTORY":
            try:
                answer = ResearchHistory(
                    self.workspace_session,
                    self.experiments,
                    None if self.application is None else self.application.pipeline,
                    self.updates,
                    self.research_updates,
                    self.review,
                ).listing(
                    strategy=request.strategy_package_id,
                    input_id=request.research_input_id,
                    input_hash=request.input_binding_hash,
                    kind=request.history_kind,
                    limit=request.history_limit or 25,
                    cursor=request.history_cursor,
                    entry_id=request.history_entry_id,
                    installed_packages=tuple(self._packages),
                )
                if "entries" not in answer:
                    return answer
                records = self.workspace_session.task_control_registry.record_collection().records
                attention = self.supervisor.attention(records)
                by_id = {record.task_id: record for record in records}
                for entry in cast("list[dict[str, Any]]", answer["entries"]):
                    if entry["task_id"] is not None:
                        task_id = UUID(entry["task_id"])
                        fact = attention.get(task_id)
                        if fact is not None:
                            entry["attention"] = fact.model_dump(mode="json")
                        record = by_id.get(task_id)
                        if record is not None:
                            entry["task_kind"] = record.task_kind
                            entry["task_record_hash"] = record.record_hash
                        if (
                            record is not None
                            and record.failure_code == "task_control.ledger_rebuilt"
                        ):
                            entry["detail"] = stop_detail(
                                record.task_kind, record.failure_code, "TASK_CONTROL"
                            )
                            if self._task_replan(record) is not None:
                                entry["stop_next"] = LEDGER_REBUILT_NEXT
                return answer
            except (ValueError, KeyError, OSError) as error:
                return {
                    "status": "REFUSED",
                    **located_failure(error, "research_history.refused"),
                }
        if request.operation.startswith("MODEL_"):
            return self._model_operation(request, caller)
        if request.operation in {"STRATEGY_ACTIVATE", "STRATEGY_DEACTIVATE"}:
            return self._strategy_activation(request, caller)
        if request.operation in {
            "FEATURE_EXTENSIONS",
            "FEATURE_REVIEW",
            "FEATURE_ACTIVATE",
            "FEATURE_DEACTIVATE",
        }:
            return self._feature_extension_operation(request, caller)
        if request.operation.startswith("EXPERIMENT"):
            try:
                if request.operation == "EXPERIMENT_FOUNDATION_SUMMARY":
                    from .foundation_summary import read_foundation_summary

                    assert request.foundation_admission_hash is not None
                    return read_foundation_summary(
                        self.workspace_session.workspace, request.foundation_admission_hash
                    )
                if request.operation == "EXPERIMENT_VERIFY_ALL":
                    return self.verify_all()
                if request.operation == "EXPERIMENT_DELIVERY_EXPORT":
                    from .research_delivery import export_research_delivery

                    return export_research_delivery(
                        request=request,
                        experiments=self.experiments,
                        review=self.review,
                        caller=caller,
                    )
                return self.experiments.operate(
                    request, caller=caller, agent_execution=agent_execution
                )
            except ValidationError as error:
                return {
                    "status": "REFUSED",
                    **located_failure(error, "research_experiment.document_invalid"),
                    "next_action": "CORRECT_DECLARATION_AND_REPLAN",
                    "next_requests": {"inputs": {"operation": "RESEARCH_INPUTS"}},
                }
            except (ValueError, KeyError, OSError, StorageRetentionError) as error:
                code = (
                    "task_control.task_not_found"
                    if isinstance(error, TaskNotFoundError)
                    else str(getattr(error, "failure_code", None))
                    if getattr(error, "failure_code", None)
                    else public_failure(error, "research_experiment.refused")
                )
                if code in {
                    "research_experiment.input_not_admitted",
                    "research_experiment.input_selection_required",
                }:
                    return {
                        "status": "REFUSED",
                        "failure_code": code,
                        "fields": [["research_input_id"]],
                        "message": (
                            "Use a research input ID from RESEARCH_INPUTS; "
                            "the Universe handle is a different declaration field."
                            if code == "research_experiment.input_not_admitted"
                            and request.research_input_id is not None
                            else "Choose one admitted research input ID from RESEARCH_INPUTS; "
                            "prepare the workspace first if none exists."
                        ),
                        "next_action": "RESEARCH_INPUTS",
                        "next_requests": {"inputs": {"operation": "RESEARCH_INPUTS"}},
                    }
                located = getattr(error, "document_location", None)
                return {
                    **refused(
                        code,
                        origin=str(request.origin_task_id) if request.origin_task_id else None,
                        portfolio=_portfolio_source(request),
                        **self.experiments.study_facts(
                            request.task_id,
                            moved=code == "research_experiment.execution_binding_changed",
                        ),
                    ),
                    # Where a declaration sent as text stopped parsing (V145).
                    **({"document_location": located} if located else {}),
                    # What the refusing owner knows each field may hold (V292).
                    **(
                        {"expected": error.expected}
                        if isinstance(error, AuthoringError) and error.expected
                        else {}
                    ),
                    # A missing prerequisite result names the flow's and the way on (V367).
                    **self.experiments.refusal_prerequisites(code, request),
                }
        try:
            review_result = self._review_operation(request, caller=caller)
        except PortfolioEvidenceReviewError as error:
            stray = "product_host.evidence_review_update_selector_invalid:position_basis"
            if str(error) == stray:
                return refused(stray, book=_request_fields(request, "position_basis"))
            worded = self._selector_refusal(str(error), request)
            if worded is not None:
                return worded
            code = "product_host.evidence_review_evidence_not_current"
            if str(error) != code:
                raise
            selector = _selector(request)
            return refused(code, book=None if selector is None else selector.request_fields())
        if review_result is not None:
            return review_result
        if not self.installed():
            return refused("research_workspace.strategy_not_installed")
        assert self.service is not None and self.review is not None
        assert self.scoring is not None and self.calibration is not None
        assert self.updates is not None and self.automation is not None
        match request.operation:
            case "RESEARCH_UPDATE_AUTOMATION_READBACK":
                return self._automation_answer(self.automation.readback())
            case "RESEARCH_UPDATE_AUTOMATION_CONFIGURE":
                assert caller != "SERVICE_AUTOMATION"
                if caller == "EXTERNAL_AUTOMATION":
                    # A person turns it on or off, as the registry marks it (PERSON_ONLY, V407).
                    raise ValueError("research_update.human_confirmation_required")
                if (
                    type(request.automation_enabled) is not bool
                    or request.automation_package_ids is None
                ):
                    raise ValueError("research_update.automation_fields_invalid")
                return self._automation_answer(
                    self.automation.configure(
                        enabled=request.automation_enabled,
                        package_ids=request.automation_package_ids,
                        chosen_by=caller,
                    )
                )
            case "RESEARCH_UPDATE_PLAN" | "RESEARCH_UPDATE_RUN" | "RESEARCH_UPDATE_READBACK":
                try:
                    if self.research_updates is None:
                        raise ValueError("research_update.not_installed")
                    if request.operation == "RESEARCH_UPDATE_PLAN":
                        assert request.strategy_package_id is not None
                        return self.research_updates.plan(
                            request.strategy_package_id,
                            None
                            if request.observed_through is None
                            else date.fromisoformat(request.observed_through),
                        )
                    if request.operation == "RESEARCH_UPDATE_READBACK":
                        chosen = self._strategy_task(request)
                        if isinstance(chosen, dict):
                            return chosen
                        return _position_rows(
                            self._research_update_way(
                                chosen, self.research_updates.readback(chosen)
                            )
                        )
                    assert request.update_plan_hash is not None
                    plan = self.research_updates.prepare(request.update_plan_hash)
                    reused_task = self.research_updates.reusable(plan)
                    if reused_task is not None:
                        return {
                            "status": "REUSED_EXACT",
                            "task_id": None,
                            "publication_task_id": str(reused_task),
                        }
                    pending = self.research_updates.in_flight(plan)
                    if pending is not None:
                        return {
                            "status": "REUSED_IN_FLIGHT",
                            "task_id": str(pending.task_id),
                            "lifecycle": pending.lifecycle,
                        }
                    # A stopped update's Task, reopened where its stop allows, answers as it is now
                    # (V600).
                    return self._run_answer(
                        self.dispatcher.submit(
                            DecisionAdvancementCommand(self.research_updates, plan)
                        ),
                        plan.data_plan,
                    )
                except (ValueError, KeyError, FileNotFoundError) as error:
                    failure = located_failure(error, "research_update.refused")
                    body = {
                        "status": "REFUSED",
                        **failure,
                        **explain(str(failure["failure_code"])),
                    }
                    if (
                        failure["failure_code"] == "portfolio_update.not_installed"
                        and request.strategy_package_id is not None
                        and request.strategy_package_id in self._packages
                        and self.activations is not None
                    ):
                        body["activation"] = self._activation_offer(request.strategy_package_id)
                    return self._inputs_way(body)
            case "PORTFOLIO_UPDATE_PLAN" | "PORTFOLIO_UPDATE_RUN" | "PORTFOLIO_UPDATE_READBACK":
                try:
                    if request.operation == "PORTFOLIO_UPDATE_PLAN":
                        assert request.strategy_package_id is not None
                        return self.updates.plan(
                            request.strategy_package_id,
                            request.prepared_input_hash,
                            None
                            if request.observed_through is None
                            else date.fromisoformat(request.observed_through),
                        )
                    if request.operation == "PORTFOLIO_UPDATE_READBACK":
                        chosen = self._strategy_task(request)
                        if isinstance(chosen, dict):
                            return chosen
                        return _position_rows(self.updates.readback(chosen))
                    assert request.update_plan_hash is not None
                    plan = self.updates.prepare(request.update_plan_hash)
                    reused = self.updates.reusable(plan)
                    if reused is not None:
                        return {
                            "status": "REUSED_EXACT",
                            "task_id": None,
                            "publication_hash": reused.content_hash,
                            **reused_read(
                                self.workspace_session.task_control_registry,
                                plan.plan_hash,
                                "PORTFOLIO_UPDATE_READBACK",
                            ),
                        }
                    admitted = self.dispatcher.submit(PortfolioUpdateCommand(self.updates, plan))
                    return {
                        "status": admitted.disposition,
                        "task_id": str(admitted.task_id) if admitted.task_id else None,
                        "lifecycle": admitted.lifecycle,
                        "failure_code": admitted.refusal_detail,
                    }
                except FileNotFoundError:
                    return {
                        "status": "REFUSED",
                        "failure_code": "portfolio_update.artifact_missing",
                    }
                except (ValueError, KeyError) as error:
                    return {
                        "status": "REFUSED",
                        **located_failure(error, "portfolio_update.refused"),
                    }
            case (
                "STRATEGY_CALIBRATION_PLAN"
                | "STRATEGY_CALIBRATION_RUN"
                | "STRATEGY_CALIBRATION_READBACK"
            ):
                try:
                    if request.operation == "STRATEGY_CALIBRATION_PLAN":
                        assert (
                            request.strategy_package_id is not None
                            and request.score_snapshot_hash is not None
                        )
                        return self.calibration.plan(
                            request.strategy_package_id, request.score_snapshot_hash
                        )
                    if request.operation == "STRATEGY_CALIBRATION_READBACK":
                        chosen = self._strategy_task(request)
                        if isinstance(chosen, dict):
                            return chosen
                        return self.calibration.readback(chosen)
                    assert request.calibration_plan_hash is not None
                    plan = self.calibration.prepare(request.calibration_plan_hash)
                    reused = self.calibration.reusable(plan)
                    if reused is not None:
                        return {
                            "status": "REUSED_EXACT",
                            "task_id": None,
                            # The record its reuse is, by its identity (OP18).
                            "calibration_input_hash": reused.content_hash,
                            "input": reused.model_dump(mode="json"),
                            "readout": reused.readout(),
                            **reused_read(
                                self.workspace_session.task_control_registry,
                                plan.plan_hash,
                                "STRATEGY_CALIBRATION_READBACK",
                            ),
                        }
                    result = self.dispatcher.submit(
                        StrategyCalibrationCommand(self.calibration, plan)
                    )
                    return {
                        "status": result.disposition,
                        "task_id": str(result.task_id) if result.task_id else None,
                        "lifecycle": result.lifecycle,
                        "failure_code": result.refusal_detail,
                    }
                except FileNotFoundError:
                    return {
                        "status": "REFUSED",
                        "failure_code": "portfolio_calibration.artifact_missing",
                    }
                except (ValueError, KeyError) as error:
                    if (
                        str(error) == "portfolio_calibration.plan_required"
                        and request.calibration_plan_hash is not None
                    ):
                        return {
                            **refused("portfolio_calibration.plan_required"),
                            "next_requests": self.calibration.replan_requests(
                                request.calibration_plan_hash
                            ),
                        }
                    failure = located_failure(error, "portfolio_calibration.refused")
                    return {
                        "status": "REFUSED",
                        **failure,
                        **explain(str(failure["failure_code"])),
                    }
            case "STRATEGY_SCORE_PLAN" | "STRATEGY_SCORE_RUN" | "STRATEGY_SCORE_READBACK":
                try:
                    if request.operation == "STRATEGY_SCORE_PLAN":
                        assert request.strategy_package_id is not None
                        return self.scoring.plan(
                            request.strategy_package_id,
                            date.fromisoformat(request.formation_session)
                            if request.formation_session
                            else None,
                            component_id=request.component_id,
                        )
                    if request.operation == "STRATEGY_SCORE_READBACK":
                        chosen = self._strategy_task(request)
                        if isinstance(chosen, dict):
                            return chosen
                        return self.scoring.readback(chosen)
                    assert request.score_plan_hash is not None
                    plan = self.scoring.prepare(request.score_plan_hash)
                    reused = self.scoring.reusable(plan)
                    if reused is not None:
                        return {
                            "status": "REUSED_EXACT",
                            "task_id": None,
                            # The score it reuses, which a calibration plan takes (OP18).
                            "score_snapshot_hash": reused.snapshot_hash,
                            "score": reused.model_dump(mode="json"),
                            **reused_read(
                                self.workspace_session.task_control_registry,
                                plan.plan_hash,
                                "STRATEGY_SCORE_READBACK",
                            ),
                        }
                    result = self.dispatcher.submit(StrategyScoreCommand(self.scoring, plan))
                    return {
                        "status": result.disposition,
                        "task_id": str(result.task_id) if result.task_id else None,
                        "lifecycle": result.lifecycle,
                        "failure_code": result.refusal_detail,
                    }
                except FileNotFoundError:
                    return {"status": "REFUSED", "failure_code": "strategy_score.artifact_missing"}
                except (ValueError, KeyError) as error:
                    if (
                        str(error) == "strategy_score.plan_required"
                        and request.score_plan_hash is not None
                    ):
                        return {
                            **refused("strategy_score.plan_required"),
                            "next_requests": self.scoring.replan_requests(request.score_plan_hash),
                        }
                    failure = (
                        # The Alpha store says it in a sentence, inside the study's closure: a
                        # stored artifact missing, stale or tampered (V449).
                        {"failure_code": "strategy_score.alpha_artifact_unreadable"}
                        if type(error).__name__ == "AlphaDevelopmentArtifactReadbackError"
                        and safe_failure_code(str(error)) is None
                        else located_failure(error, "strategy_score.refused")
                    )
                    return {
                        "status": "REFUSED",
                        **failure,
                        **explain(str(failure["failure_code"])),
                    }
            case "CONTROLS" | "PLAN" | "RUN":
                try:
                    if request.operation == "CONTROLS":
                        return self.controls(request.strategy_package_id)
                    assert request.spec is not None
                    return (
                        self.plan(request.spec)
                        if request.operation == "PLAN"
                        else self.run(request.spec)
                    )
                except LocalApplicationError as error:
                    code = str(error)
                    if code not in {
                        "strategy_book.strategy_package_required",
                        "research_workspace.strategy_not_installed",
                    } and not code.startswith("local_application.strategy_package_not_installed:"):
                        raise
                    return refused(code, installed_packages=tuple(sorted(self._packages)))
            case "RESULTS":
                return self.results(request.task_id)
            case "REPORT":
                assert request.result_hash is not None
                try:
                    return self.report(request.result_hash, request.portfolio_session)
                except ContentAddressedStoreError as error:
                    rows = cast("list[dict[str, object]]", self.results()["results"])
                    listed = {str(v["result_hash"]) for v in rows}
                    return refused(
                        "portfolio_research.result_not_found"
                        if (
                            str(error).partition(":")[0] == "content_store.artifact_missing"
                            and request.result_hash not in listed
                        )
                        else public_failure(error, "portfolio_research.report_unavailable")
                    )
                except LocalApplicationError as error:
                    if str(error) != "portfolio_application.session_outside_report":
                        raise
                    return refused(
                        public_failure(error, "portfolio_research.report_unavailable"),
                        result_hash=request.result_hash,
                    )
            case "COMPARE":
                assert request.left_result_hash is not None
                assert request.right_result_hash is not None
                return self.compare(request.left_result_hash, request.right_result_hash)
            case "FREEZE":
                assert request.result_hash is not None
                return self.freeze(request.result_hash)
            case "FINALIZATION":
                assert request.candidate_hash is not None
                return self.finalization(request.candidate_hash)
            case "EXPORT":
                assert request.result_hash is not None
                return self.export(request.result_hash)
        raise ValueError("portfolio_research.operation_unknown")

    def _agent_run(self) -> AgentRun | None:
        """The Session that submitted an answer, recorded beside it (V300, AU3, LAWS.md ID7).

        The lead submits every answer; which of its children wrote it is not observed, so the
        record names the host and Session alone, never a guess. None for a request that names
        no agent Session.
        """
        provenance = REQUEST_PROVENANCE.get()
        if provenance is None or provenance.vendor is None or provenance.session is None:
            return None
        run: AgentRun = AgentRun.model_validate(
            {"host": provenance.vendor, "session_id": provenance.session, "basis": "NOT_OBSERVED"}
        )
        return run

    def record_agent_answer(
        self,
        bundle: AgentBundleRecord | None,
        body: Mapping[str, object],
        *,
        accepted_receipt: GoalAcceptedAnswerReceipt | None = None,
    ) -> dict[str, object] | None:
        """Wire one sealed answer to the shared native observation owner (V691)."""
        view = body.get("answer")
        delivery = view.get("accepted_delivery") if isinstance(view, dict) else None
        if body.get("disposition") not in {"ADMITTED", "REUSED_EXACT"}:
            return None
        provenance = REQUEST_PROVENANCE.get()
        references: dict[str, object] = {
            **({"host": provenance.vendor} if provenance is not None and provenance.vendor else {}),
            **(
                {"session_id": provenance.session}
                if provenance is not None and provenance.session
                else {}
            ),
            **({"bundle_reference": bundle.record_hash} if bundle is not None else {}),
            **({"task_id": body["task_id"]} if body.get("task_id") else {}),
        }

        def unavailable(reason: str, *missing: str) -> dict[str, object]:
            code = f"native_bridge.{reason}"
            return {
                "status": "UNAVAILABLE",
                "reason": code,
                **(refusal_words(code) or refusal_words("native_bridge.accepted_delivery_failed")),
                "missing": list(missing),
                **references,
            }

        if not isinstance(delivery, dict):
            if isinstance(view, dict) and view.get("verdict") in {"ACCEPTED", "DONE"}:
                return unavailable("accepted_delivery_unavailable", "accepted_delivery")
            return None
        if delivery.get("status") == "UNAVAILABLE":
            raise ValueError("native_bridge.accepted_delivery_unavailable")
        record = AgentAnswerRecord.model_validate(delivery["answer_record"])
        if record.verdict.value not in {"ACCEPTED", "DONE"}:
            return None
        references.update(answer_reference=record.record_hash, task_id=delivery["task_id"])
        run = record.agent_run
        if run is not None:
            references.update(host=run.host, session_id=run.session_id)
        if bundle is None:
            return unavailable("accepted_bundle_unavailable", "accepted_bundle")
        if run is None:
            return unavailable("accepted_author_not_observed", "accepted_author")
        if provenance is None or (provenance.vendor, provenance.session) != (
            run.host,
            run.session_id,
        ):
            return unavailable("parent_session_mismatch", "parent_session")
        task = self.workspace_session.task_control_registry.task(UUID(str(delivery["task_id"])))
        original_session = GoalSession(vendor=run.host, session_id=run.session_id)
        if accepted_receipt is None:
            return unavailable("accepted_context_unavailable", "first_accepted_context")
        receipt = accepted_receipt
        if (
            receipt.session != original_session
            or receipt.task_id != task.task_id
            or receipt.bundle_reference != bundle.record_hash
            or receipt.answer_reference != record.record_hash
            or receipt.bundle_role != bundle.role
            or receipt.verdict != record.verdict.value
        ):
            return unavailable("accepted_context_unavailable", "first_accepted_context")
        original_goal = receipt.goal_id
        if self.observer is None:
            return unavailable("history_unavailable", "native_history")
        observer = self.observer
        workspace = self.workspace_session.workspace
        found = self._session_binding(
            (original_session.vendor, original_session.session_id), Path(workspace)
        )
        if found is None:
            return unavailable("binding_mismatch", "native_binding")
        project, binding = found
        result = deliver_accepted_answer(
            project,
            binding,
            bundle=bundle,
            answer=record,
            contribution=delivery["contribution"],
            task_id=str(task.task_id),
            admitted_at=task.admitted_at,
            publish=lambda document: self.declare_event(
                observer, document, accepted_receipt=receipt
            ),
        )
        if result is not None and result.get("status") == "DELIVERED":
            result.update(
                original_session_host=original_session.vendor,
                original_session_id=original_session.session_id,
                original_goal_id=None if original_goal is None else str(original_goal),
            )
        return result

    def capture_accepted_answer_context(
        self, bundle: AgentBundleRecord | None, body: Mapping[str, object]
    ) -> GoalAcceptedAnswerReceipt | None:
        """Freeze the first genuine acceptance's prior packet before optional delivery.

        Called while the scientific submission holds the workspace mutation gate. This
        reads only product owner records; it never discovers native files or contacts a Host.
        The caller isolates failures so the accepted scientific answer remains usable.
        """
        view = body.get("answer")
        delivery = view.get("accepted_delivery") if isinstance(view, dict) else None
        if bundle is None or not isinstance(delivery, dict) or "answer_record" not in delivery:
            return None
        record = AgentAnswerRecord.model_validate(delivery["answer_record"])
        run, provenance = record.agent_run, REQUEST_PROVENANCE.get()
        if (
            record.verdict.value not in {"ACCEPTED", "DONE"}
            or run is None
            or provenance is None
            or (provenance.vendor, provenance.session) != (run.host, run.session_id)
        ):
            return None
        task = self.workspace_session.task_control_registry.task(UUID(str(delivery["task_id"])))
        session = GoalSession(vendor=run.host, session_id=run.session_id)
        context = self.goals.accepted_answer_context(
            bundle_reference=bundle.record_hash,
            answer_reference=record.record_hash,
            session=session,
            first_submission=delivery.get("first_submission") is True,
            provenance=provenance,
        )
        return GoalAcceptedAnswerReceipt(
            **context.model_dump(),
            session=session,
            task_id=task.task_id,
            bundle_reference=bundle.record_hash,
            answer_reference=record.record_hash,
            bundle_role=bundle.role,
            verdict="ACCEPTED" if record.verdict.value == "ACCEPTED" else "DONE",
        )

    def observe_accepted_answer(
        self,
        body: dict[str, object],
        bundle: AgentBundleRecord | None,
        *,
        accepted_receipt: GoalAcceptedAnswerReceipt | None = None,
    ) -> dict[str, object]:
        """Keep observation optional at every registered accepted-answer return door."""
        failures = self.observer_failures
        conversation = self._observe(
            lambda: self.record_agent_answer(bundle, body, accepted_receipt=accepted_receipt)
        )
        if conversation is not None:
            body["conversation"] = conversation
        elif self.observer_failures > failures:
            view = body.get("answer")
            delivery = view.get("accepted_delivery") if isinstance(view, dict) else None
            unavailable_delivery = (
                isinstance(delivery, dict) and delivery.get("status") == "UNAVAILABLE"
            )
            reason = (
                "native_bridge.accepted_delivery_unavailable"
                if unavailable_delivery
                else "native_bridge.accepted_delivery_failed"
            )
            provenance = REQUEST_PROVENANCE.get()
            body["conversation"] = {
                "status": "UNAVAILABLE",
                "reason": reason,
                **refusal_words(reason),
                "missing": [
                    "accepted_delivery" if unavailable_delivery else "accepted_answer_delivery"
                ],
                **({"bundle_reference": bundle.record_hash} if bundle is not None else {}),
                **({"task_id": body["task_id"]} if body.get("task_id") else {}),
                **(
                    {"host": provenance.vendor}
                    if provenance is not None and provenance.vendor
                    else {}
                ),
                **(
                    {"session_id": provenance.session}
                    if provenance is not None and provenance.session
                    else {}
                ),
            }
        return body

    def _review_operation(
        self,
        request: PortfolioResearchOperationRequest,
        *,
        caller: OperationCaller,
        read_files: tuple[str, ...] | None = None,
        agent_run: AgentRun | None = None,
        accepted_bundle: AgentBundleRecord | None = None,
    ) -> dict[str, object] | None:
        """Research review does not require installing a frozen strategy.

        ``read_files`` is what an agent's answer named as read and ``agent_run`` who made it,
        both kept with the answer (V260, V300).
        """
        if request.operation not in {
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
        }:
            return None
        if self.review is None:
            raise ValueError("product_host.evidence_review_not_configured")
        selector = _selector(request)
        match request.operation:
            case "AGENT_BUNDLE_PREPARE":
                assert request.agent_role is not None and request.bundle_directory is not None
                try:
                    prepared = EvidenceReviewBundles(self.review).prepare_agent_bundle(
                        role=request.agent_role,
                        selector=selector,
                        directory=request.bundle_directory,
                        task_id=request.task_id,
                        unit_id=request.evidence_unit_id,
                        dispatcher=self.dispatcher,
                    )
                except (PortfolioEvidenceReviewError, ValueError) as error:
                    code = public_failure(error, "agent_bundle.refused")
                    # A packet named without its Task or unit is told how to name it (V242).
                    if str(error).partition(":")[0] in PACKET_SELECTOR_CODES | {
                        "agent_bundle.specialist_task_required",
                        "agent_bundle.reference_bound_exceeded",
                    }:
                        return agent_bundle_refusal(code, role=request.agent_role)
                    # A unit that was not prepared says why and where the prepared ones are (V388).
                    unit = unit_refusal(
                        code, None if selector is None else selector.request_fields()
                    )
                    if unit is None:
                        raise
                    return unit
                return _review_outcome_body(prepared)
            case "AGENT_ANSWER_SUBMIT":
                # The bundle's own submission, completed with the answer as
                # written: the agent names its bundle, never a hash.
                assert request.bundle_directory is not None
                bundle = EvidenceReviewBundles(self.review).agent_bundle(request.bundle_directory)
                if bundle is None:
                    return agent_bundle_refusal("agent_bundle.not_prepared")
                # The files the agent names as read whole are its own word, kept with
                # its answer as provenance; the answer is read whatever they are (OP11).
                answer, read = answer_read(request.agent_answer or {})
                agent_run = self._agent_run()
                if bundle.role not in ANSWER_FIELDS:
                    try:
                        with self.workspace_session.mutation_gate.hold():
                            generic_body = self.review.submit_specialist_answer(
                                bundle=bundle, answer=answer, read_files=read, agent_run=agent_run
                            )
                            accepted_receipt = self._observe(
                                lambda: self.capture_accepted_answer_context(bundle, generic_body)
                            )
                    except PortfolioEvidenceReviewError as error:
                        if str(error) not in {
                            "agent_bundle.specialist_task_changed",
                            "agent_bundle.answer_settled",
                        }:
                            raise
                        return agent_bundle_refusal(str(error), role=bundle.role)
                    generic_body = self.observe_accepted_answer(
                        generic_body, bundle, accepted_receipt=accepted_receipt
                    )
                    answered = agent_answer_result(bundle.role, generic_body)
                    if answered["status"] in {"ACCEPTED", "DONE"}:
                        answered["bundle_reference"] = bundle.record_hash
                    if agent_run is not None and answered["status"] not in {"ACCEPTED", "DONE"}:
                        answered["recorded_agent"] = agent_run.model_dump(mode="json")
                    if "conversation" in generic_body:
                        answered["conversation"] = generic_body["conversation"]
                    return answered
                bound = PortfolioResearchRequestDocument.model_validate(
                    {**bundle.submission, ANSWER_FIELDS[bundle.role]: answer}
                ).to_operation_request()
                try:
                    body = self._review_operation(
                        bound,
                        caller=caller,
                        read_files=read,
                        agent_run=agent_run,
                        accepted_bundle=bundle,
                    )
                except ValidationError as error:
                    # The answer passed its format and the record it would seal did
                    # not: named by place, count and bound, never a bare field code.
                    over = sealed_bounds_exceeded(error)
                    if not over:
                        raise
                    return agent_bundle_refusal(
                        "agent_bundle.seal_bound_exceeded", role=bundle.role, bounds=over
                    )
                except PortfolioEvidenceReviewError as error:
                    if str(error) == "agent_bundle.answer_settled":  # V417
                        return agent_bundle_refusal(str(error), role=bundle.role)
                    if str(error) not in STALE_BUNDLE_CODES:
                        raise
                    return agent_bundle_refusal("agent_bundle.preparation_stale")
                except ValueError as error:
                    # A packet prepared under authority the Host no longer holds -- an install
                    # replaced its package -- is refused by what moved, with the book's preview
                    # that prepares the unit again (V547).
                    if not str(error).startswith(STALE_PACKET):
                        raise
                    return stale_packet_refusal(
                        str(error),
                        role=bundle.role,
                        book=_bundle_book(bundle.submission),
                        unit_id=bundle.submission.get("evidence_unit_id"),
                    )
                assert body is not None
                book = _bundle_book(bundle.submission)
                answered = agent_answer_result(bundle.role, body, book)
                if agent_run is not None and answered["status"] not in {"ACCEPTED", "DONE"}:
                    answered["recorded_agent"] = agent_run.model_dump(mode="json")
                if "conversation" in body:
                    answered["conversation"] = body["conversation"]
                return answered
            case "EVIDENCE_CRO":
                section = self.evidence_cro(
                    selector,
                    request.review_publication_hash,
                    evidence_detail=request.evidence_detail,
                    view_entity_id=request.view_entity_id,
                    view_topic=request.view_topic,
                    view_last_days=request.view_last_days,
                )
                if not _pages_citations(request):
                    return section
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
                delivery = EvidenceReviewDelivery(self.review)
                if not _pages_citations(request):
                    return delivery.export_review(
                        selector,
                        request.review_publication_hash,
                        prior_publication_hash=request.prior_review_publication_hash,
                    )
                # A page slices the export sealed for this book and review (V94).
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
                            cast(dict[str, object], exported["evidence"]).get("verified_spans")
                            or [],
                        ),
                        request,
                        groups={
                            str(child["unit_id"]): tuple(
                                cast(list[str], child["ordered_entity_ids"])
                            )
                            for child in cast(
                                list[dict[str, object]],
                                exported_dossier.get("evidence_children") or [],
                            )
                        },
                        key="verified_spans",
                    ),
                }
            case "EVIDENCE_REFRESH":
                return self.evidence_refresh(selector)
            case "EVIDENCE_PREPARE":
                outcome = self.review.refresh_evidence(
                    dispatcher=self.dispatcher,
                    selector=selector,
                    evidence_as_of=(
                        None
                        if request.evidence_as_of is None
                        else datetime.fromisoformat(request.evidence_as_of)
                    ),
                    prepare_only=True,
                    preparation_binding_hash=request.preparation_binding_hash,
                )
                body = _review_outcome_body(outcome)
                if outcome.task_id is not None:
                    packets = self.review.packet_requests(selector, task_id=outcome.task_id)
                    chosen = self.review.default_selector(selector)
                    # Nothing new, the CRO's review carried forward: its Task and its book go
                    # on, never a packet of the review's Task (V434).
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
                return self.review.preview_evidence(selector)
            case "EVIDENCE_LEDGER":
                return _review_outcome_body(
                    EvidenceReviewDelivery(self.review).book_ledger(
                        selector, page=request.ledger_page
                    )
                )
            case "EVIDENCE_DOCUMENTS":
                return self.review.documents(page=request.documents_page)
            case "EVIDENCE_PACKET":
                assert request.task_id is not None
                try:
                    delivered = EvidenceReviewDelivery(self.review).export_analysis_packet(
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
                    # A unit that was not prepared says why and where the prepared ones are
                    # (V388); any other refusal is answered as before.
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
                outcome = self.review.continue_evidence(
                    dispatcher=self.dispatcher,
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
                    # The continuation's packet is its own Task's, single: the
                    # unit it continued rides in its input, not in the request.
                    # Its consumer holds every earlier part already, so the
                    # session's own windows are offered beside the whole.
                    packets = self.review.packet_requests(selector, task_id=outcome.task_id)
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
                assert request.task_id is not None
                assert request.analysis_context_hash is not None
                assert request.analysis_answer is not None
                with (
                    self.workspace_session.mutation_gate.hold()
                    if accepted_bundle is not None
                    else nullcontext()
                ):
                    analysis_body = _review_outcome_body(
                        self.review.submit_analysis(
                            dispatcher=self.dispatcher,
                            selector=selector,
                            task_id=request.task_id,
                            context_hash=request.analysis_context_hash,
                            answer=request.analysis_answer,
                            caller=caller,
                            unit_id=request.evidence_unit_id,
                            read_files=read_files,
                            agent_run=agent_run,
                        )
                    )
                    accepted_receipt = self._observe(
                        lambda: self.capture_accepted_answer_context(accepted_bundle, analysis_body)
                    )
                return self.observe_accepted_answer(
                    analysis_body, accepted_bundle, accepted_receipt=accepted_receipt
                )
            case "EVIDENCE_SELECT":
                assert request.analysis_publication_hash is not None
                return self.evidence_select(
                    selector,
                    analysis_publication_hash=request.analysis_publication_hash,
                    chosen_by=caller,
                )
            case "CRO_REVIEW":
                return self.cro_review(selector)
            case "CRO_REVIEW_DOSSIER":
                dossier = EvidenceReviewDelivery(self.review).export_dossier(
                    selector,
                    delivery_part=request.delivery_part,
                    delivery_budget_bytes=request.delivery_budget_bytes,
                    review_dossier_hash=request.review_dossier_hash,
                    review_read_at=_read_at(request),
                )
                return (
                    _review_outcome_body(dossier) if isinstance(dossier, ReviewOutcome) else dossier
                )
            case "CRO_REVIEW_FINDING":
                assert request.finding_handle is not None
                package = EvidenceReviewDelivery(self.review).export_finding_evidence(
                    selector,
                    finding_handle=request.finding_handle,
                    review_dossier_hash=request.review_dossier_hash,
                    review_publication_hash=request.review_publication_hash,
                    review_read_at=_read_at(request),
                )
                return (
                    _review_outcome_body(package) if isinstance(package, ReviewOutcome) else package
                )
            case "CRO_REVIEW_SUBMIT":
                assert request.review_dossier_hash is not None
                assert request.review_policy_hash is not None
                assert request.review_schema_hash is not None
                assert request.review_answer is not None
                with (
                    self.workspace_session.mutation_gate.hold()
                    if accepted_bundle is not None
                    else nullcontext()
                ):
                    body = _review_outcome_body(
                        self.review.submit_assessment(
                            dispatcher=self.dispatcher,
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
                    accepted_receipt = self._observe(
                        lambda: self.capture_accepted_answer_context(accepted_bundle, body)
                    )
                return self.observe_accepted_answer(
                    body, accepted_bundle, accepted_receipt=accepted_receipt
                )
        raise ValueError("portfolio_research.operation_unknown")

    def _refresh_prepared_inputs(self) -> None:
        # Asked on every operation, so the manifest file's own stat answers first whether
        # anything could have changed: two Task counts per request under Task Control's one
        # connection lock queued every light read at twenty requests a second (W10, measured
        # with py-spy), and the counts themselves are answered again while nothing is written.
        # They stay the first check, not a listing that validates every Task record (~0.2 s
        # per poll on thirty-five Tasks); a workspace with no preparation reads no manifest.
        stamp = _file_stamp(self.workspace_session.workspace / RESEARCH_WORKSPACE_MANIFEST_NAME)
        if stamp is not None and stamp == self._manifest_stamp:
            return
        if (
            not self.preparation.has_tasks()
            and not self.workspace_session.task_control_registry.has_tasks(
                self.model_training_inputs.task_kind
            )
        ):
            return
        current = read_research_workspace_manifest(self.workspace_session.workspace)
        if current.manifest_hash == self.workspace_manifest.manifest_hash:
            self._manifest_stamp = stamp
            return
        training_owned = self.model_training_inputs.owns_published_manifest(current.manifest_hash)
        if not training_owned and not self.preparation.owns_published_manifest(
            current.manifest_hash
        ):
            raise ValueError("research_workspace.configuration_changed_restart_required")
        # Reload only the fields the verified preparation publication owns.
        # Neither path may install or replace strategy/default authority.
        if (
            current.with_bindings(
                **(
                    {"model_training_inputs": self.workspace_manifest.model_training_inputs}
                    if training_owned
                    else {
                        "experiment_inputs": self.workspace_manifest.experiment_inputs,
                        "data_update": self.workspace_manifest.data_update,
                    }
                ),
            ).manifest_hash
            != self.workspace_manifest.manifest_hash
        ):
            raise ValueError("research_workspace.configuration_changed_restart_required")
        self._hold(current)
        self._manifest_stamp = stamp

    def _hold(self, current: ResearchWorkspaceManifest) -> None:
        """Refresh the one workspace manifest holder every application reads.

        The applications read `manifests.current`, so none keeps a copy of its own to fall
        behind and refuse its next plan as `workspace_manifest_changed` though its inputs
        had not moved (V182). The update automation keeps only the hash its settings are
        approved under, so a new manifest asks for their approval again, as a restart would.

        Args:
            current: The manifest just published and verified.
        """
        self.workspace_manifest = current
        assert self.manifests is not None
        self.manifests.current = current  # every application reads this one holder
        if self.automation is not None:
            self.automation.manifest_hash = current.manifest_hash

    def installed(self) -> bool:
        """Read whether a strategy is installed, from the packages this Host holds.

        Returns:
            Whether any package is installed; every "nothing installed" answer reads this.
        """
        return bool(self._packages)

    def _install(self, task_id: UUID) -> dict[str, object]:
        """Install a prepared research strategy and load it into this running Host.

        The resolver takes the installed catalog and the package mapping is swapped once, so the
        package's PLAN and RUN are served at once: no restart, no person. A book, score,
        calibration or update Task in progress reads the installed packages, so installing
        waits for it, refused by name.
        """
        running = next(
            (
                task
                for task in self.workspace_session.task_control_registry.tasks()
                if task.task_kind in _PORTFOLIO_TASK_KINDS and task.lifecycle in _LIVE_LIFECYCLES
            ),
            None,
        )
        if running is not None:
            return refused(
                f"research_strategy.installation_waits_for_portfolio_tasks:{running.task_id}",
                task_id=str(running.task_id),
            )
        answer = self.research_strategies.install(task_id)
        resolver = None if self.application is None else self.application.resolver
        if isinstance(resolver, StrategyPortfolioResolver):
            admitted = admit_research_workspace(self.workspace_session.workspace)
            resolver.catalog = admitted.catalog
            self._packages = {
                package.strategy_id: package for package in resolver.installed_packages().values()
            }
            self._hold_activation(admitted.manifest)
        return answer

    def _hold_activation(self, current: ResearchWorkspaceManifest) -> None:
        """Hold the manifest a person's activation or deactivation published (LS1).

        The Host wrote it itself, under its gate, so it is held as published; the daily update
        offers the packages it now binds, and asks the person again (V182).
        """
        self._hold(current)
        self._manifest_stamp = _file_stamp(
            self.workspace_session.workspace / RESEARCH_WORKSPACE_MANIFEST_NAME
        )
        if self.automation is not None:
            self.automation.installed = tuple(
                p for p in self._packages if self._update_configured(p)
            )

    def _book_review_standing(self, task_id: UUID) -> dict[str, object]:
        """Compose the Task publication and the review owner's exact-result standing (V614)."""
        assert self.activations is not None and self.review is not None
        result_hash = self.activations.book_result_hash(task_id)
        return EvidenceReviewDelivery(self.review).review_standing(
            None if result_hash is None else BookSelector(result_hash=result_hash)
        )

    def _strategy_activation(
        self, request: PortfolioResearchOperationRequest, caller: OperationCaller
    ) -> dict[str, object]:
        """A person runs a reviewed research book's strategy forward, or stops it (LS1, OW12).

        As a person activates a model (OW11), every other caller is refused by name, before
        anything else is asked. A first-use goal's delegation carries the activation of a book
        with a published review, and only of one (STOPS-1); the person deactivates it.
        """
        if caller != "HUMAN":
            return cast(
                dict[str, object], refused("strategy_activation.human_confirmation_required")
            )
        if self.activations is None:
            return cast(
                dict[str, object], refused("strategy_activation.research_strategy_required")
            )
        provenance = REQUEST_PROVENANCE.get()
        try:
            if request.operation == "STRATEGY_ACTIVATE":
                assert request.task_id is not None
                if (
                    provenance is not None
                    and provenance.delegation is not None
                    and (
                        self.review is None
                        or self._book_review_standing(request.task_id)["status"] != "REVIEWED"
                    )
                ):
                    raise ValueError("strategy_activation.review_required")
                return self.activations.activate(request.task_id)
            assert request.strategy_package_id is not None
            return self.activations.deactivate(request.strategy_package_id)
        except Exception as error:
            if owner_failure_code(error) is None and not isinstance(
                error, (ValueError, OSError, KeyError)
            ):
                raise
            failure = located_failure(error, "strategy_activation.refused")
            code = str(failure["failure_code"])
            return {
                "status": "REFUSED",
                **failure,
                **explain(code),
                **(
                    {
                        "next_requests": {
                            "cap": {"operation": "STORAGE_CAP_SHOW"},
                            "storage": {"operation": "STORAGE_READBACK"},
                            "cleanup": {"operation": "STORAGE_PLAN"},
                        }
                    }
                    if code == "storage.managed_capacity_exceeded"
                    else {}
                ),
            }

    def _workspace_operation(
        self,
        request: PortfolioResearchOperationRequest,
        *,
        caller: OperationCaller,
    ) -> dict[str, object] | None:
        match request.operation:
            case "FEATURE_CATALOG_BUILD":
                assert request.feature_plan_hash is not None
                return self.feature_builds.build(
                    request.feature_plan_hash,
                    caller=caller,
                    dispatcher=self.dispatcher,
                    preprocess=request.feature_output == "PREPROCESSED_VALUES",
                )
            case "FEATURE_TRIAL" | "FEATURE_TRIAL_READBACK" | "FEATURE_TRIALS":
                try:
                    if request.operation == "FEATURE_TRIALS":
                        return self.trials.listing()
                    if request.operation == "FEATURE_TRIAL_READBACK":
                        assert request.feature_trial_id is not None
                        return self.trials.readback(request.feature_trial_id)
                    assert request.feature_plan_hash is not None and request.task_id is not None
                    return self.trials.start(
                        request.feature_plan_hash, request.task_id, caller=caller
                    )
                except ValueError as error:
                    code = public_failure(error, "feature_trial.refused")
                    return refused(
                        code,
                        task_id=str(request.task_id or "") or None,
                        # A study the trial cannot run against names the way on (V354).
                        trial=self.trials.way_on(request.feature_plan_hash)
                        if code == "feature_trial.study_not_from_factor_evidence"
                        and request.feature_plan_hash is not None
                        else None,
                    )
            case "FEATURE_CATALOG_BUILD_READBACK":
                assert request.task_id is not None
                build = self.feature_builds.readback(request.task_id)
                # The readback checked the Task's contract; its input is the build's source.
                binding = str(
                    self.workspace_session.task_control_registry.task(
                        request.task_id
                    ).input.payload["input_binding_hash"]
                )
                # What the build's source can claim about time, from its input's Panel (V347).
                return {
                    **build,
                    "temporal_scope": binding_temporal_scope(
                        self.workspace_session.workspace,
                        binding,
                        window_start=None,
                        window_end=None,
                    ),
                }
            case "FEATURE_CATALOG_CONTROLS" | "FEATURE_CATALOG_PLAN" | "FEATURE_CATALOG_READBACK":
                definitions = ResearchFeatureDefinitions(
                    self.workspace_session.workspace,
                )
                try:
                    if request.operation == "FEATURE_CATALOG_CONTROLS":
                        assert request.input_binding_hash is not None
                        return {
                            **definitions.controls(
                                request.input_binding_hash, request.feature_plan_hash
                            ),
                            # What a trial of a factor on this input runs against, before
                            # anything is built (V367, V354).
                            **self.experiments.prerequisites_of(
                                "FEATURE_TRIAL", binding_hash=request.input_binding_hash
                            ),
                        }
                    if request.operation == "FEATURE_CATALOG_PLAN":
                        assert request.feature_document is not None
                        with self.workspace_session.mutation_gate.hold():
                            plan = definitions.plan(request.feature_document)
                        return _with_trial(definitions.readback(plan.plan_hash))
                    assert request.feature_plan_hash is not None
                    return _with_trial(definitions.readback(request.feature_plan_hash))
                except ValidationError as error:
                    return {
                        "status": "REFUSED",
                        **located_failure(error, "feature_research.document_invalid"),
                        "next_action": "READ_FEATURE_CATALOG_CONTROLS_AND_CORRECT_DOCUMENT",
                    }
                except FeatureCatalogCrudError as error:
                    return {
                        "status": "REFUSED",
                        "failure_code": error.failure_code,
                        "message": str(error),
                    }
                except (ValueError, OSError, KeyError) as error:
                    named = request.input_binding_hash or str(
                        (request.feature_document or {}).get("input_binding_hash") or ""
                    )
                    held = [
                        v.binding_hash for v in self.experiments.manifest.experiment_inputs or ()
                    ]
                    if isinstance(error, FileNotFoundError) and named and named not in held:
                        # A binding the workspace does not hold names the field, the bindings
                        # it holds and each one's controls; nothing is repaired (V380).
                        return {
                            "status": "REFUSED",
                            "failure_code": "feature_research.input_binding_unresolved",
                            "fields": [["input_binding_hash"]],
                            "expected": {"input_binding_hash": held},
                            **explain("feature_research.input_binding_unresolved"),
                            "next_requests": {
                                **{
                                    f"controls:{binding[:12]}": {
                                        "operation": "FEATURE_CATALOG_CONTROLS",
                                        "input_binding_hash": binding,
                                    }
                                    for binding in held
                                },
                                "inputs": {"operation": "WORKSPACE_SHOW"},
                            },
                        }
                    return {
                        "status": "REFUSED",
                        "failure_code": public_failure(
                            error, "feature_research.definition_or_source_unavailable"
                        ),
                    }
            case "RESEARCH_STRATEGY_CONTROLS":
                return self.research_strategies.controls()
            case (
                "RESEARCH_STRATEGY_PLAN"
                | "RESEARCH_STRATEGY_PREPARE"
                | "RESEARCH_STRATEGY_READBACK"
                | "RESEARCH_STRATEGY_INSTALL"
            ):
                try:
                    if request.operation == "RESEARCH_STRATEGY_PLAN":
                        assert request.experiment_document is not None
                        return self.research_strategies.plan(request.experiment_document)
                    if request.operation == "RESEARCH_STRATEGY_PREPARE":
                        assert request.experiment_plan_hash is not None
                        return self.research_strategies.prepare(
                            request.experiment_plan_hash, caller=caller, dispatcher=self.dispatcher
                        )
                    assert request.task_id is not None
                    if request.operation == "RESEARCH_STRATEGY_INSTALL":
                        return self._install(request.task_id)
                    return self.research_strategies.readback(request.task_id)
                except ValidationError as error:
                    return {
                        "status": "REFUSED",
                        **located_failure(error, "research_strategy.declaration_invalid"),
                        "next_action": "CORRECT_DECLARATION_AND_REPLAN",
                    }
                except (ValueError, OSError, KeyError) as error:
                    if (
                        str(error) == "research_strategy.preview_required"
                        and request.experiment_plan_hash is not None
                    ):
                        return {
                            **refused("research_strategy.preview_required"),
                            "next_requests": self.research_strategies.replan_requests(
                                request.experiment_plan_hash
                            ),
                        }
                    refusal: dict[str, object] = {
                        "status": "REFUSED",
                        **located_failure(error, "research_strategy.refused"),
                    }
                    if str(refusal.get("failure_code", "")).startswith(
                        (
                            "research_strategy.risk_parent_support_incomplete",
                            "research_strategy.risk_history_insufficient",
                        )
                    ):
                        recovery = self.research_strategies.controls(
                            (request.experiment_document or {}).get("input_binding_hash")
                        )
                        refusal["risk_windows"] = recovery["risk_windows"]
                        refusal["next_requests"] = recovery["next_requests"]
                    return refusal
            case "MODEL_TRAINING_INPUT_PLAN":
                assert request.research_input_id is not None and request.component_id is not None
                return self.model_training_inputs.plan(
                    request.research_input_id,
                    request.input_binding_hash,
                    request.component_id,
                    **(
                        {"model_lifecycle": request.model_lifecycle}
                        if request.model_lifecycle
                        else {}
                    ),
                )
            case "MODEL_TRAINING_INPUT_PREPARE":
                assert request.experiment_plan_hash is not None
                try:
                    return self.model_training_inputs.prepare(
                        request.experiment_plan_hash, caller=caller, dispatcher=self.dispatcher
                    )
                except ValueError as error:
                    if str(error) != "model_training.preview_required":
                        raise
                    return {
                        **refused("model_training.preview_required"),
                        "next_requests": self.model_training_inputs.replan_requests(
                            request.experiment_plan_hash
                        ),
                    }
            case "MODEL_TRAINING_INPUT_READBACK":
                assert request.task_id is not None
                return self.model_training_inputs.readback(request.task_id)
            case "DATA_ISSUES":
                return cast(
                    dict[str, object],
                    self.data_issues.readback(
                        limit=request.history_limit or 25,
                        cursor=request.history_cursor,
                        delegated_by=self._data_decisions(caller),
                    ),
                )
            case "DATA_ISSUE_REVOKE":
                assert request.data_issue_grant_hash is not None
                return cast(
                    dict[str, object],
                    self.data_issues.revoke(
                        grant_hash=request.data_issue_grant_hash, caller=caller
                    ),
                )
            case "DATA_ISSUE_CONFIRM" | "DATA_ISSUE_PREVIEW" | "DATA_ISSUE_DELEGATE":
                assert request.data_issue_case_token is not None
                assert request.data_issue_evidence_hash is not None
                assert request.data_issue_option_id is not None
                assert request.data_issue_option_hash is not None
                if request.operation == "DATA_ISSUE_DELEGATE":
                    assert request.task_id is not None
                choice = dict(
                    case_token=request.data_issue_case_token,
                    evidence_hash=request.data_issue_evidence_hash,
                    option_id=request.data_issue_option_id,
                    option_hash=request.data_issue_option_hash,
                )
                return cast(
                    dict[str, object],
                    self.data_issues.preview(**choice, delegated_by=self._data_decisions(caller))
                    if request.operation == "DATA_ISSUE_PREVIEW"
                    else self.data_issues.delegate(**choice, task_id=request.task_id, caller=caller)
                    if request.operation == "DATA_ISSUE_DELEGATE"
                    else self.data_issues.confirm(
                        **choice, caller=caller, grant_hash=request.data_issue_grant_hash
                    ),
                )
            case "RESEARCH_INPUTS":
                # Discovery isolates each unreadable publication; exact input selection remains
                # strict in ResearchInputRevisions.select/lineage.
                return self.input_capture.revisions.versions()
            case "RESEARCH_INPUT_PLAN":
                assert request.research_input_id is not None
                return self.input_capture.plan(request.research_input_id)
            case "RESEARCH_INPUT_CONFIRM":
                assert request.research_input_plan_hash is not None
                try:
                    return self.input_capture.confirm(
                        request.research_input_plan_hash, caller=caller, dispatcher=self.dispatcher
                    )
                except ValueError as error:
                    if str(error) != "research_input.preview_required":
                        raise
                    return {
                        **refused("research_input.preview_required"),
                        "next_requests": self.input_capture.replan_requests(
                            request.research_input_plan_hash
                        ),
                    }
            case "RESEARCH_INPUT_READBACK":
                assert request.task_id is not None
                return self.input_capture.readback(request.task_id)
            case "WORKSPACE_PREPARE_READBACK":
                return self._preparation_readback(request.task_id)
            case "WORKSPACE_PREPARE_PLAN":
                return self.preparation.plan()
            case "WORKSPACE_PREPARE_CONFIRM":
                assert request.preparation_plan_hash is not None
                self.preparation.require_confirmation_caller(
                    request.preparation_plan_hash,
                    caller=caller,
                    grant_hash=request.data_issue_grant_hash,
                )
                previous = next(
                    (
                        t
                        for t in self.preparation.tasks()
                        if t.input.payload["plan"]["plan_hash"] == request.preparation_plan_hash
                        and t.lifecycle.value != "CANCELLED"
                    ),
                    None,
                )
                if previous is not None:
                    if previous.lifecycle.value == "SUCCEEDED":
                        return {
                            "status": "REUSED_EXACT",
                            "task_id": None,
                            "publication_task_id": str(previous.task_id),
                        }
                    if previous.lifecycle in _PREPARATION_IN_FLIGHT:
                        return {"status": "REUSED_IN_FLIGHT", "task_id": str(previous.task_id)}
                sent = self.dispatcher.submit(
                    WorkspacePreparationCommand(
                        self.preparation,
                        request.preparation_plan_hash,
                        caller=caller,
                        grant_hash=request.data_issue_grant_hash,
                    )
                )
                return {
                    # The preparation's Task as it is now, a stopped one confirmed again reopened
                    # (V600).
                    **self._run_answer(sent),
                    # A refused admission's words, a provider's wait among them (V375).
                    **(
                        explain(sent.refusal_detail, workspace=self.workspace_session.workspace)
                        if sent.refusal_detail
                        else {}
                    ),
                    # A preview past its hour or superseded: preview the preparation again (V543).
                    **(
                        {"next_requests": {"replan": {"operation": "WORKSPACE_PREPARE_PLAN"}}}
                        if sent.refusal_detail == "workspace_preparation.preview_required"
                        else {}
                    ),
                }
            case "STORAGE_CAP_SHOW":
                return self.storage.cap_readback()
            case "STORAGE_CAP_SET":
                assert request.storage_cap_bytes is not None
                return self.storage.set_cap(request.storage_cap_bytes, caller=caller)
            case "STORAGE_READBACK":
                return self.storage.readback()
            case "STORAGE_PLAN":
                return self.storage.plan()
            case "STORAGE_CONFIRM":
                assert request.storage_plan_hash is not None
                return self.storage.confirm(request.storage_plan_hash, caller=caller)
            case "STORAGE_PIN":
                assert request.input_binding_hash is not None and type(request.input_pinned) is bool
                return self.storage.pin(
                    request.input_binding_hash, pinned=request.input_pinned, caller=caller
                )
            case "STORAGE_EVIDENCE_REBUILD":
                assert request.evidence_index_id is not None
                return self.storage.rebuild_evidence_index(request.evidence_index_id, caller=caller)
            case "DATA_CHANGE_CONFIRM":
                if self.data_update is None or request.update_plan_hash is None:
                    raise ValueError("workspace_data_update.not_configured")
                admitted = self.data_update.confirm(request.update_plan_hash, caller=caller)
                return {
                    "status": "APPROVED",
                    "task_id": str(admitted.task_id),
                    "lifecycle": admitted.lifecycle,
                    "next_action": "DATA_UPDATE_RUN",
                    "next_requests": {
                        "run": {
                            "operation": "DATA_UPDATE_RUN",
                            "update_plan_hash": request.update_plan_hash,
                        }
                    },
                }
            case "DATA_UPDATE_PLAN" | "DATA_UPDATE_RUN" | "DATA_UPDATE_READBACK":
                try:
                    if self.data_update is None:
                        raise ValueError("workspace_data_update.not_configured")
                    if request.operation == "DATA_UPDATE_PLAN":
                        return self.data_update.plan(recovery_task_id=request.recovery_task_id)
                    if request.operation == "DATA_UPDATE_READBACK":
                        return self._data_update_readback(request.task_id)
                    assert request.update_plan_hash is not None
                    plan = self.data_update.prepare(request.update_plan_hash)
                    reused = self.data_update.reusable(plan)
                    if reused is not None:
                        # The update it reused, by the Task that sealed it, which a read follows
                        # (V449: every state has its exit).
                        sealed = self.data_update.receipt_task(reused)
                        return {
                            "status": "REUSED_EXACT",
                            "task_id": None,
                            **(
                                {}
                                if sealed is None
                                else {
                                    "publication_task_id": str(sealed),
                                    "next_requests": {
                                        "read": {
                                            "operation": "DATA_UPDATE_READBACK",
                                            "task_id": str(sealed),
                                        }
                                    },
                                }
                            ),
                            "receipt": reused.model_dump(mode="json"),
                        }
                    submitted = self.dispatcher.submit(
                        WorkspaceDataUpdateCommand(self.data_update, plan)
                    )
                    # The update's Task as it is now, a stopped one reopened where its stop allows
                    # (V600).
                    answer = self._run_answer(submitted, plan)
                    return {
                        **answer,
                        # A refused admission's words, a provider's wait among them (V375).
                        **(
                            explain(
                                submitted.refusal_detail, workspace=self.workspace_session.workspace
                            )
                            if submitted.refusal_detail
                            else {}
                        ),
                        # The update it started, read by its Task (V418).
                        **(
                            {
                                "next_requests": {
                                    **cast(dict[str, Any], answer.get("next_requests") or {}),
                                    "show": {
                                        "operation": "DATA_UPDATE_READBACK",
                                        "task_id": str(submitted.task_id),
                                    },
                                }
                            }
                            if submitted.task_id
                            else {}
                        ),
                    }
                except FileNotFoundError:
                    return {
                        "status": "REFUSED",
                        "failure_code": "workspace_data_update.artifact_missing",
                    }
                except (ValueError, KeyError) as error:
                    update_refusal: dict[str, object] = {
                        "status": "REFUSED",
                        **located_failure(error, "workspace_data_update.refused"),
                    }
                    if update_refusal["failure_code"] == "workspace_data_update.plan_required":
                        # A plan past its hour or never kept here: plan the update again (V543).
                        update_refusal["next_requests"] = {
                            "replan": {"operation": "DATA_UPDATE_PLAN"}
                        }
                    return update_refusal
            case "STATUS":
                assert request.task_id is not None
                return self.status(request.task_id, wait_seconds=request.wait_seconds)
            case "TASK_RECOVERY":
                assert request.task_id is not None
                return self.recovery_view(request.task_id)
            case "TASK_GUARDIAN":
                return self.guardian()
            case "TASK_INCIDENTS":
                return self.supervisor.incidents()
            case "TASK_REMEDIATE":
                return self.supervisor.remediate(
                    request,
                    selected_by="USER_COMMAND"
                    if REQUEST_PROVENANCE.get() is None
                    else "AGENT_PROPOSAL",
                )
            case "RECOVER":
                assert request.task_id is not None
                before = self.status(request.task_id)
                stale = self._stale_confirmation(request, before)
                if stale is not None:
                    return stale
                confirmed = request.expected_task_hash
                retry: dict[str, object] | None = None
                if before["lifecycle"] == "BLOCKED":
                    # A BLOCKED Task is reopened only by its owner's judgement,
                    # against the version the person confirmed, and only when
                    # the refusal was an artifact that can have been repaired.
                    record = self.workspace_session.task_control_registry.task(request.task_id)
                    raised = _runner_retry_reason(record) is not None
                    if record.task_kind != RESEARCH_EXPERIMENT_TASK_KIND and not raised:
                        return {**before, "disposition": "NOT_RECOVERY_REQUIRED"}
                    if confirmed is None:
                        return {
                            "status": "REFUSED",
                            "failure_code": "local_application.expected_task_hash_required",
                            "task_id": before["task_id"],
                            "task_record_hash": before["task_record_hash"],
                            # The view that offers it bound to this version (V443).
                            "next_requests": {
                                "recovery": {
                                    "operation": "TASK_RECOVERY",
                                    "task_id": str(request.task_id),
                                }
                            },
                        }
                    # The confirmed version travels into Task Control's write;
                    # a Task that moved while its plan was re-checked is refused
                    # there, as stale, with nothing applied.
                    retry = (
                        self._reopen_runner_block(record, confirmed)
                        if raised
                        else self.experiments.retry_blocked(
                            request.task_id,
                            now=self.dispatcher.clock(),
                            expected_task_hash=confirmed,
                        )
                    )
                    if retry.get("failure_code") == "task_control.recovery_version_stale":
                        return {
                            "status": "REFUSED",
                            "failure_code": "local_application.confirmation_stale",
                            "task_id": str(retry["task_id"]),
                            "lifecycle": retry["lifecycle"],
                            "task_record_hash": retry["task_record_hash"],
                            "refused_at": "TASK_CONTROL",
                        }
                    if retry.get("disposition") != "RETRY_ADMITTED":
                        return {**before, **retry}
                    # The reopened version, the direct successor of the confirmed
                    # one, is what the resume below is confirmed against.
                    confirmed = str(retry["task_record_hash"])
                    before = self.status(request.task_id)
                if before["lifecycle"] not in {"RECOVERY_REQUIRED", "QUEUED"}:
                    return {**before, "disposition": "NOT_RECOVERY_REQUIRED"}
                if before["resume_refusal"] is not None:
                    return self._resume_refused(before)
                if self.recover_task is None:
                    return {
                        **before,
                        **refused(
                            "local_application.task_recovery_not_configured",
                            task_id=str(request.task_id),
                        ),
                    }
                resumed = self.recover_task(request.task_id, expected_task_hash=confirmed)
                if request.task_id not in resumed:
                    failure = self.dispatcher.failure(request.task_id)
                    if failure:
                        return {
                            **self.status(request.task_id),
                            "status": "REFUSED",
                            "failure_code": failure,
                            "detail": (
                                "The resume was not handed to the worker; the Task's recovery "
                                "view says what stopped it and what may resume it."
                            ),
                            "next_requests": {
                                "recovery": {
                                    "operation": "TASK_RECOVERY",
                                    "task_id": str(request.task_id),
                                }
                            },
                        }
                return {
                    **self.status(request.task_id),
                    **({"retry": retry} if retry is not None else {}),
                    "resumed_task_ids": [str(v) for v in resumed],
                }
            case "TASKS":
                return self.tasks(
                    agent_session=request.agent_session,
                    limit=request.history_limit or _TASK_PAGE_ROWS,
                    cursor=request.history_cursor,
                )
            case "CANCEL":
                assert request.task_id is not None
                stale = self._stale_confirmation(request, self.status(request.task_id))
                if stale is not None:
                    return stale
                return self.cancel(request.task_id, expected_task_hash=request.expected_task_hash)
        return None

    @staticmethod
    def _stale_confirmation(
        request: PortfolioResearchOperationRequest, current: dict[str, object]
    ) -> dict[str, object] | None:
        """A confirmation made against one Task version acts on that version or not at all.

        The registry guards its own write with the version it reads at the moment
        of the write; this guard is the caller's: what the person or Agent
        confirmed is what the owner is asked to change. A refusal changes nothing
        and hands back the current version so the choice can be renewed.
        """

        expected = request.expected_task_hash
        if expected is None:
            return None
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            return {
                "status": "REFUSED",
                "failure_code": "local_application.expected_task_hash_invalid",
                "task_id": current["task_id"],
            }
        if expected == current["task_record_hash"]:
            return None
        return {
            "status": "REFUSED",
            "failure_code": "local_application.confirmation_stale",
            "task_id": current["task_id"],
            "lifecycle": current["lifecycle"],
            "task_record_hash": current["task_record_hash"],
            "refused_at": "OPERATION_ENTRY",
        }

    def workspace_context(self, *, caller: OperationCaller) -> dict[str, object]:
        """Join existing read operations on demand; no cached readiness or new authority."""
        sections: dict[str, object] = {}
        requests = (
            (
                "goals",
                PortfolioResearchOperationRequest(operation="GOAL_LIST", history_limit=5),
            ),
            ("inputs", PortfolioResearchOperationRequest(operation="RESEARCH_INPUTS")),
            (
                "recent_research",
                PortfolioResearchOperationRequest(operation="RESEARCH_HISTORY", history_limit=5),
            ),
            ("tasks", PortfolioResearchOperationRequest(operation="TASKS")),
            ("data_update", PortfolioResearchOperationRequest(operation="DATA_UPDATE_READBACK")),
            # Whether the workspace is prepared, preparing or unprepared, read
            # once with the session so an entry page can say so and name the
            # next step without a second request.
            (
                "preparation",
                PortfolioResearchOperationRequest(operation="WORKSPACE_PREPARE_READBACK"),
            ),
        )
        # Six readers over two stores: one read boundary holds each store's
        # instance across them instead of an engine open per reader. If a Data
        # writer already holds the gate, do not block the entry page behind it:
        # saved-input/Task metadata uses its ordinary owners, while mutable Data
        # discovery stays explicitly unread, not empty or cached as ready.
        with self.workspace_session.reads(timeout_seconds=0.05) as batched:
            for name, request in requests:
                if not batched and name == "data_update":
                    sections[name] = {
                        "status": "BUSY",
                        "failure_code": "workspace_context.writer_busy",
                        "message": "Current data state is being changed; Task progress and saved "
                        "inputs remain readable.",
                        "next_action": "DATA_UPDATE_READBACK",
                    }
                    continue
                try:
                    sections[name] = self.execute(request, caller=caller)
                except (ValueError, KeyError, OSError) as error:
                    sections[name] = {
                        "status": "REFUSED",
                        **located_failure(error, "portfolio_research.section_refused"),
                    }
        return {
            **sections,
            "verification": "DISCOVERY_ONLY_SELECTED_OPERATIONS_REVALIDATE",
            "actions": [
                {
                    "action": "update_data",
                    "label": "Update source data",
                    "entry_operation": "DATA_UPDATE_PLAN",
                    "configured": self.data_update is not None
                    and self.workspace_manifest.data_update is not None,
                    "selection": [],
                    "effect": "May fetch data and rebuild affected Features after confirmation; "
                    "does not change sealed research inputs.",
                    "confirmation": "HUMAN",
                    "panel": "workspace",
                },
                {
                    "action": "capture_input",
                    "label": "Seal a research input version",
                    "entry_operation": "RESEARCH_INPUT_PLAN",
                    "selection": ["research_input_id"],
                    "effect": "Capture prepared local data as an immutable input, or reuse an "
                    "unchanged version; does not run an experiment.",
                    "confirmation": "HUMAN",
                    "panel": "workspace",
                },
                {
                    "action": "research",
                    "label": "Research with an existing input",
                    "entry_operation": "EXPERIMENT_CONTROLS",
                    "selection": ["research_input_id", "input_binding_hash", "experiment_kind"],
                    "effect": "Obtain a declaration, then PLAN before RUN. Selected sealed "
                    "inputs remain fixed; browsing does not download data or fit models.",
                    "confirmation": "INSPECT_PLAN_BEFORE_RUN",
                    "panel": "research",
                },
            ],
        }

    def session_projection(
        self,
        *,
        session_token: str,
        include_context: bool = False,
        caller: OperationCaller = "HUMAN",
    ) -> dict[str, object]:
        # Context executes its readers inside one read scope; their first
        # operation refreshes the manifest there. A separate refresh here
        # reopened Task Control twice before that same scope.
        """Read session mode, capacity and installed strategy context.

        Args:
            session_token: Explicit connection session token to project.
            include_context: Whether to include the workspace context projection.
            caller: Declared operation caller.

        Returns:
            Preparation/research mode, bounded execution capacity and exact available strategy
            views.
        """
        if include_context:
            context = {"research_context": self.workspace_context(caller=caller)}
        else:
            self._refresh_prepared_inputs()
            context = {}
        if not self.installed():
            return {
                **context,
                "workspace_id": self.workspace_manifest.workspace_id,
                "workspace_manifest_hash": self.workspace_manifest.manifest_hash,
                "execution_mode": "RESEARCH_PREPARATION",
                "session_token": session_token,
                "capacity": {"active": 1, "queued": 1},
                "strategy": None,
                "installed_strategies": [],
            }
        assert self.service is not None
        package = (
            self._package(self.workspace_manifest.default_strategy_package_id)
            if self.workspace_manifest.default_strategy_package_id is not None
            else None
        )
        return {
            **context,
            "workspace_id": self.service.workspace_id,
            "workspace_manifest_hash": self.workspace_manifest.manifest_hash,
            "execution_mode": "DEVELOPMENT_REPLAY",
            "session_token": session_token,
            "capacity": {"active": 1, "queued": 1},
            "strategy": None
            if package is None
            else {
                "strategy_id": package.strategy_id,
                "package_hash": package.package_hash,
                "score_source_mode": self.workspace_manifest.default_score_source_mode,
                "score_source": package.score_source(
                    self.workspace_manifest.default_score_source_mode
                    or package.default_score_source_mode
                ).description,
                "merge_semantics": package.merge_semantics,
                "component_count": len(package.component_plan),
            },
            "installed_strategies": [
                {
                    "strategy_id": value.strategy_id,
                    "package_hash": value.package_hash,
                    "score_source_modes": list(value.installed_modes),
                    "research_update_configured": self._update_configured(value.strategy_id),
                    "strategy_dates": None
                    if self.activations is None
                    else self.activations.dates(value.strategy_id),
                }
                for value in sorted(self._packages.values(), key=lambda item: item.strategy_id)
            ],
        }

    def _update_configured(self, package_id: str) -> bool:
        assert self.updates is not None
        if self.workspace_manifest.data_update is None or not all(
            any(v.strategy_package_id == package_id for v in bindings or ())
            for bindings in (
                self.workspace_manifest.score_inputs,
                self.workspace_manifest.decision_updates,
            )
        ):
            return False
        checkpoint = self.updates._checkpoint(package_id, current=False)
        if not self.updates.model_bindings_match(checkpoint):
            return False
        return not any(c.weight_rule == "mu.iv0" for c in checkpoint.recipe.components) or any(
            v.strategy_package_id == package_id
            for v in self.workspace_manifest.calibration_inputs or ()
        )

    def controls(self, strategy_package_id: str | None) -> dict[str, object]:
        """Project the installed portfolio controls and explicit frozen/refused choices.

        Args:
            strategy_package_id: Optional explicit package selection; otherwise use the declared
                default.

        Returns:
            Service catalog, selected package presentation, admitted controls and frozen
            controls/refusals.
        """
        assert self.service is not None
        catalog = self.service.controls
        package = self._package(
            strategy_package_id or self.workspace_manifest.default_strategy_package_id
        )
        rows, frozen = self._package_controls(package)
        return {
            "catalog_hash": catalog.catalog_hash,
            "presentation_hash": catalog.presentation_hash,
            "package_hash": package.package_hash,
            "strategy_package_id": package.strategy_id,
            "report_reference_selection": RiskReportLinks.selection(
                task_kind="INSTALLED_STRATEGY_BOOK", completed_portfolio=False
            ),
            "strategy_dates": None
            if self.activations is None
            else self.activations.dates(package.strategy_id),
            "default_score_source_mode": package.default_score_source_mode,
            "score_source_modes": list(package.installed_modes),
            "controls": rows,
            "frozen": frozen,
            "template": {
                "strategy_package_id": package.strategy_id,
                "score_source_mode": package.default_score_source_mode,
                **{str(row["control_id"]): row["default_value"] for row in rows},
            },
            "next_requests": {
                "preview": {
                    "operation": "PLAN",
                    "spec": {"strategy_package_id": package.strategy_id},
                }
            },
            "refused": [
                {
                    "control_id": refusal.control_id,
                    "refusal_code": refusal.refusal_code,
                    "reason": refusal.reason,
                    "evidence_reference": refusal.evidence_reference,
                }
                for refusal in catalog.refusals
            ],
            # Whether the package runs forward, from which book and through when, and what a
            # person can do with it (LS1, U73).
            "activation": {"status": "INACTIVE"}
            if self.activations is None
            else self._activation_offer(package.strategy_id),
        }

    def _strategy_task(
        self, request: PortfolioResearchOperationRequest
    ) -> UUID | dict[str, object]:
        """The Task a readback of work planned per strategy reads, or its answer without one.

        Its Task, or its strategy's latest, never another strategy's; unselected, refused where two
        strategies hold Tasks (V595). The owner then reads the Task it is handed.
        """
        registry = self.workspace_session.task_control_registry
        return strategy_task(
            _STRATEGY_READS[request.operation],
            request.operation,
            registry.tasks(),
            registry.task,
            request.task_id,
            request.strategy_package_id,
        )

    def _automation_answer(self, body: dict[str, object]) -> dict[str, object]:
        """The daily update with the strategies it runs forward, and a person's two requests
        on it: on for exactly those strategies, or off (U73). It offers no other package, and
        refuses one by name (`research_update.automation_package_not_installed`)."""
        assert self.automation is not None
        ids = sorted(self.automation.installed)
        # Each strategy's own latest update, found as its readback finds it and its read bound to
        # it: a reader of one running forward never takes another's (V595).
        read = _STRATEGY_READS["RESEARCH_UPDATE_READBACK"]
        latest = latest_by_strategy(
            (
                task
                for task in self.workspace_session.task_control_registry.tasks()
                if read.kind(task)
            ),
            read.path,
        )
        rows = [
            {
                "strategy_package_id": package_id,
                **(
                    {"status": "ACTIVE"}
                    if self.activations is None
                    else self.activations.summary(package_id)
                ),
                "latest_update": (
                    None
                    if (task := latest.get(package_id)) is None
                    else {
                        **(_latest_update(task) or {}),
                        **(
                            ResearchHistory.latest_update(self.research_updates, task)
                            if task.lifecycle is TaskLifecycle.SUCCEEDED
                            and self.research_updates is not None
                            else {}
                        ),
                    }
                ),
            }
            for package_id in ids
        ]
        settings = body.get("settings")
        on = (
            body.get("status") == "ENABLED_SERVICE_LIFETIME"
            and isinstance(settings, dict)
            and list(settings.get("package_ids") or ()) == ids
        )
        requests: dict[str, object] = {}
        if ids and not on:
            requests["enable"] = {
                "operation": "RESEARCH_UPDATE_AUTOMATION_CONFIGURE",
                "automation_enabled": True,
                "automation_package_ids": ids,
            }
        if body.get("status") != "DISABLED":
            requests["disable"] = {
                "operation": "RESEARCH_UPDATE_AUTOMATION_CONFIGURE",
                "automation_enabled": False,
                "automation_package_ids": [],
            }
        return {**body, "runs_forward": rows, "next_requests": requests}

    def _forward_intents(self) -> list[dict[str, object]]:
        """Each installed research strategy's way forward, beside the inputs' flows (V471).

        A strategy that runs forward offers its daily update's plan; one that does not names its
        activation, a person's, with the book it would activate or the book it holds. A default
        installation's packages have no activation and are not listed.
        """
        if not self.installed():
            # No strategy is installed: the way forward begins at the research strategy's
            # controls, which name each missing component's first step (V505, RR5). It is the
            # first intent, the shortest way to positions, ahead of the inputs' Lab flows.
            return [
                {
                    "flow": "RUN_FORWARD",
                    "status": "NO_RESEARCH_STRATEGY_INSTALLED",
                    "detail": RUN_FORWARD_WORDS,
                    "next_requests": {
                        "strategy_controls": {"operation": "RESEARCH_STRATEGY_CONTROLS"}
                    },
                }
            ]
        if self.activations is None:
            return []
        intents: list[dict[str, object]] = []
        for package_id in sorted(self._packages):
            offer = self._activation_offer(package_id)
            if (
                offer["status"] == "INACTIVE"
                and "next_requests" not in offer
                and "held" not in offer
            ):
                # Installed with no book yet: its whole-support book comes next (FLOW-3).
                intents.append(
                    {
                        "flow": "RUN_FORWARD",
                        "strategy_package_id": package_id,
                        "status": "NO_BOOK_YET",
                        "detail": INSTALLED_BOOK_WORDS,
                        "next_requests": {
                            "books": {"operation": "CONTROLS", "strategy_package_id": package_id}
                        },
                    }
                )
                continue
            intents.append(
                {
                    "flow": "RUN_FORWARD",
                    "strategy_package_id": package_id,
                    "activation": offer,
                    "next_requests": {
                        "update": {
                            "operation": "RESEARCH_UPDATE_PLAN",
                            "strategy_package_id": package_id,
                        }
                    }
                    if offer["status"] == "ACTIVE"
                    else {"books": {"operation": "CONTROLS", "strategy_package_id": package_id}},
                }
            )
        return intents

    def _reopen_runner_block(self, record: TaskRecord, confirmed: str) -> dict[str, object]:
        """Reopen a Task the runner itself blocked, as its RECOVER confirms (V475, PERF-1).

        The block is Task Control's own (`TASK_STAGE_RAISED`, or the memory check before a
        stage), so any Task kind reopens: the cause is fixed or memory is free, the confirmed
        version reopens it as interrupted, and the resume runs the stage again (with its
        budget of raises whole, and the memory check first).
        """
        registry = self.workspace_session.task_control_registry
        try:
            reopened = registry.mark_recovery_required(
                task_id=record.task_id,
                failure_code=str(record.failure_code),
                observed_at=self.dispatcher.clock(),
                allow_blocked=True,
                expected_task_hash=confirmed,
            )
        except TaskVersionStale:
            current = registry.task(record.task_id)
            return {
                "failure_code": "task_control.recovery_version_stale",
                "task_id": str(record.task_id),
                "lifecycle": current.lifecycle.value,
                "task_record_hash": current.record_hash,
            }
        return {
            "disposition": "RETRY_ADMITTED",
            "task_id": str(record.task_id),
            "task_record_hash": reopened.record_hash,
        }

    def _activation_offer(self, package_id: str) -> dict[str, object]:
        """The activation's offer, a held book with its refusal's words (U73)."""
        assert self.activations is not None
        offer: dict[str, object] = self.activations.offer(package_id)
        held = offer.get("held")
        if isinstance(held, dict):
            detail = explain(str(held["failure_code"])).get("detail")
            if detail is not None:
                held["detail"] = detail
        return offer

    def plan(self, document: dict[str, object]) -> dict[str, object]:
        """Resolve an explicit portfolio declaration into its exact execution preview.

        Args:
            document: Authored portfolio spec fields.

        Returns:
            Resolved identities, cache disposition, intervals, admitted controls, estimates and
            comparison limits; no task is executed.
        """
        assert self.application is not None
        spec = self._spec(document)
        planned = self.application.plan(spec)
        preview = planned.preview
        package = self._package(preview.strategy_package_id)
        controls, frozen = self._package_controls(
            package,
            spec=spec,
            eligible_count=preview.eligible_count,
            support=(preview.full_support_start, preview.full_support_end),
        )
        next_actions = {
            "RESULT_HIT": ("OPEN_CACHED_RESULT", "ADJUST_DECLARED_CONTROLS"),
            "EXECUTION_LEDGER_HIT": (
                "RUN_REPORT_DESCENDANTS",
                "ADJUST_DECLARED_CONTROLS",
            ),
            "FULL_NUMERICAL_MISS": (
                "ADMIT_DECLARED_PATH_RUN",
                "ADJUST_DECLARED_CONTROLS",
            ),
        }[preview.cache_state]
        return {
            "spec_hash": preview.spec_hash,
            "admission_hash": preview.admission_hash,
            "authorities_hash": preview.authorities_hash,
            "strategy_package_id": preview.strategy_package_id,
            "strategy_package_hash": preview.strategy_package_hash,
            "score_source_mode": preview.score_source_mode,
            "strategy": {
                "strategy_package_id": package.strategy_id,
                "package_hash": package.package_hash,
                "score_source_mode": preview.score_source_mode,
                "score_source": package.score_source(preview.score_source_mode).description,
                "risk_disposition": package.risk_disposition,
                "component_count": len(package.component_plan),
            },
            "available_controls": controls,
            "frozen_controls": frozen,
            "available_interval": {
                "start": preview.full_support_start,
                "end": preview.full_support_end,
                "sessions": preview.coverage.common_session_count,
                "selected_start": preview.selected_study_start,
                "selected_end": preview.selected_study_end,
            },
            "cache": {
                "state": preview.cache_state,
                "exact_hit": preview.exact_cache_hit,
                "result_hash": preview.cached_result_hash,
                "execution_ledger_coverage": preview.execution_ledger_coverage,
            },
            "estimated_work": {
                "basis": preview.work_estimate_basis,
                "formations": preview.prefix_work_formation_count,
                "score_replays": preview.estimated_score_replays,
                "risk_surface_builds": preview.estimated_risk_surface_builds,
                "optimizer_calls": preview.optimizer_call_count,
            },
            "limitations": {
                "strategy_claim_limits": list(package.claim_limits),
                "plan_refusals": list(preview.refusals),
                "refused_capability_count": len(preview.refusals),
                "comparison": preview.comparison_plan,
                "capacity": "NOT_MODELED",
            },
            "next_lawful_actions": list(next_actions),
            "next_requests": {"run": {"operation": "RUN", "spec": _spec_document(spec)}},
            "exact_cache_hit": preview.exact_cache_hit,
            "cached_result_hash": preview.cached_result_hash,
            "optimizer_call_count": preview.optimizer_call_count,
            "prefix_work_formation_count": preview.prefix_work_formation_count,
            "estimated_score_replays": preview.estimated_score_replays,
            "estimated_risk_surface_builds": preview.estimated_risk_surface_builds,
            "legal_recovery": list(preview.legal_recovery),
            "refusals": list(preview.refusals),
            "coverage": {
                "common_watermark_start": preview.coverage.common_watermark_start,
                "common_watermark_end": preview.coverage.common_watermark_end,
                "common_session_count": preview.coverage.common_session_count,
            },
        }

    def run(self, document: dict[str, object]) -> dict[str, object]:
        """Check bounded capacity, plan once and reuse or dispatch exact portfolio research.

        Args:
            document: Authored portfolio spec fields.

        Returns:
            Queue refusal, exact reused result or newly submitted task metadata.
        """
        assert self.application is not None
        spec = self._spec(document)
        # A full queue answers before the plan resolves sources, scope and caches (V100); the
        # preview answers whether a run would be reused, which takes no place.
        try:
            self.workspace_session.task_control_registry.check_capacity()
        except TaskQueueFull as error:
            return {
                "disposition": "REFUSED_QUEUE_FULL",
                "command_kind": PORTFOLIO_RUN_COMMAND,
                "task_id": None,
                "refusal_detail": str(error)[:200],
                "lifecycle": None,
            }
        # A preview's cache state can change after publication (or tamper).
        # Revalidate through the existing read-only owner once per RUN; pass the
        # resulting authorities to admission rather than resolving them again.
        planned = self.application.plan(spec)
        # The one reuse rule, which the standalone script's run keeps too (V195).
        reused = self.application.reused(spec, planned)
        if reused is not None:
            return {
                "disposition": "REUSED_EXACT",
                "command_kind": PORTFOLIO_RUN_COMMAND,
                "task_id": None,
                "refusal_detail": None,
                "lifecycle": None,
                "result_hash": reused.result.result_hash,
            }
        submission = self.dispatcher.submit(
            PortfolioRunCommand(application=self.application, spec=spec, planned=planned)
        )
        return {
            "disposition": submission.disposition,
            "command_kind": submission.command_kind,
            "task_id": None if submission.task_id is None else str(submission.task_id),
            "refusal_detail": submission.refusal_detail,
            "lifecycle": submission.lifecycle,
        }

    def status(self, task_id: UUID, *, wait_seconds: float | None = None) -> dict[str, object]:
        """Read one task and optionally wait for a bounded state transition.

        Args:
            task_id: Exact retained task identity.
            wait_seconds: Optional bounded long-poll duration.

        Returns:
            Task lifecycle, failure/recovery facts, work-item timing and open supervisory remedies.
        """
        projection = self.dispatcher.status(task_id)
        if wait_seconds is not None:
            projection = self._moved_on(task_id, projection, wait_seconds)
        body = task_status_body(projection)
        body.update(self._evidence_completion(projection))
        body["worker_failure"] = self.dispatcher.failure(task_id)
        body["task_record_hash"] = projection.task_record_hash
        body["resume_refusal"] = self._resume_refusal(projection)
        stopped = self._stopped_answer(projection)
        if stopped:
            body.update(stopped)
        elif projection.lifecycle.value == "DEFERRED":
            # A deferred Task waits on its retry time and then on its plan sent again, which no
            # waiter outlasts: its read says when and by which request, as its owner's readback
            # words it, and a waiter ends on it with that (V507; V449: every state has its exit).
            body.update(self._deferred_way(projection.task_kind, task_id))
        registry = self.workspace_session.task_control_registry
        try:
            # One version of both, read once while nothing is written (W10).
            record, items = registry.task_with_work_items(task_id)
        except (KeyError, ValueError, TaskNotFoundError):
            return body
        body["timing"] = task_timing(
            record,
            items,
            now=self.dispatcher.clock(),
            spans=read_stage_spans(self.workspace_session.runtime_path, task_id),
        )
        body["attention"] = self.supervisor.attention((record,))[task_id].model_dump(mode="json")
        # What the stopped stage's owner saw beside its code, as its work item keeps it (V444).
        cause = next((item.failure_cause for item in items if item.failure_cause), None)
        if cause is not None:
            body["failure_cause"] = cause.model_dump(mode="json")
        # The agent session that submitted it, where its request named one (U33).
        agent = registry.submitted_by(task_id)
        body["submitted_by"] = None if agent is None else agent.model_dump(mode="json")
        # Its open incident, as the Supervisor last found it: a follower wakes on it (GY2, WK).
        incident = self.supervisor.open_incident(task_id)
        body["incident"] = (
            None
            if incident is None
            else {
                "key": incident.key,
                "code": incident.code,
                "detected_at": incident.incident.detected_at.isoformat(),
                "detail": incident.incident.user_safe_detail,
                "remedies": [r.action for r in incident.remedies if r.available],
                "read": {"operation": "TASK_INCIDENTS"},
            }
        )
        return body

    def _preparation_readback(self, task_id: UUID | None) -> dict[str, Any]:
        """A preparation's readback; one the provider deferred says why, when, and how it
        resumes (V375): its owner takes the same plan again once the time has passed."""
        prepared = self.preparation.readback(task_id)
        if prepared.get("status") != "DEFERRED":
            return prepared
        return deferral(
            prepared,
            failure_code=prepared.get("failure_code"),
            retry_after_at=(prepared.get("progress") or {}).get("retry_after_at"),
            resume={
                "operation": "WORKSPACE_PREPARE_CONFIRM",
                "preparation_plan_hash": str(prepared["plan_hash"]),
            }
            if prepared.get("plan_hash")
            else None,
            held=bool(prepared.get("inputs")),
        )

    def _data_update_readback(self, task_id: UUID | None) -> dict[str, Any]:
        """A data update's readback; one the provider deferred says why, when, and how it
        resumes, and that the published inputs stand meanwhile (V375)."""
        if self.data_update is None:
            raise ValueError("workspace_data_update.not_configured")
        updated = self.data_update.readback(task_id)
        if updated.get("task_lifecycle") == "BLOCKED" and updated.get("task_id"):
            # A stop its plan's rerun resumes offers that rerun, as a deferral offers its resume
            # (V600).
            offer = self._resume_offer(UUID(str(updated["task_id"])))
            offered = cast(dict[str, Any], updated.get("next_requests") or {})
            task = self.workspace_session.task_control_registry.task(UUID(str(updated["task_id"])))
            way = (
                self._stopped_way(task.task_id, task.failure_code, "BLOCKED")
                if task.failure_code
                else {}
            )
            return {
                **updated,
                **way,
                "next_requests": {**offered, **(way.get("next_requests") or {}), **offer},
            }
        if updated.get("task_lifecycle") != "DEFERRED":
            return updated
        deferred = self.workspace_session.task_control_registry.task(UUID(str(updated["task_id"])))
        return deferral(
            updated,
            failure_code=deferred.failure_code,
            retry_after_at=None
            if updated.get("retry_after_at") is None
            else str(updated["retry_after_at"]),
            resume={
                "operation": "DATA_UPDATE_RUN",
                "update_plan_hash": str(updated["plan_hash"]),
            }
            if updated.get("plan_hash")
            else None,
            held=updated.get("inputs") is not None,
        )

    def _research_update_readback(self, task_id: UUID | None) -> dict[str, Any]:
        """A research update's readback by its Task, with its way on (STATUS's deferral reader)."""
        assert self.research_updates is not None and task_id is not None
        return self._research_update_way(task_id, self.research_updates.readback(task_id))

    def _research_update_way(self, task_id: UUID, read: dict[str, Any]) -> dict[str, Any]:
        """A research update's read, a stopped or deferred one with its way on.

        A stop its plan's rerun resumes offers that rerun (V600). A deferral -- the provider's
        wait in its data stage -- says why, when and how it resumes, as a data update's does: the
        same plan sent again once `retry_after_at` has passed (V601). The read is the owner's,
        taken where the door reads it (V595). A cancelled one keeps its stop and offers only
        the recorded Task, without a run or replan.
        """
        assert self.research_updates is not None
        if read.get("status") == "CANCELLED":
            task = self.workspace_session.task_control_registry.task(task_id)
            return {
                **read,
                "detail": read.get("detail")
                or stop_detail(task.task_kind, task.failure_code or "", "TASK_CONTROL"),
                "next_requests": {
                    **cast(dict[str, Any], read.get("next_requests") or {}),
                    "task": {"operation": "STATUS", "task_id": str(task_id)},
                },
            }
        if read.get("status") == "DEFERRED":
            task = self.workspace_session.task_control_registry.task(task_id)
            plan = self.research_updates.prepare(str(task.input.payload["plan"]["content_hash"]))
            retry = (
                None if self.data_update is None else self.data_update.retry_after(plan.data_plan)
            )
            return deferral(
                read,
                failure_code=task.failure_code,
                retry_after_at=None if retry is None else retry.isoformat(),
                resume={"operation": "RESEARCH_UPDATE_RUN", "update_plan_hash": plan.content_hash},
                held=True,
            )
        offer = self._resume_offer(task_id) if read.get("status") == "BLOCKED" else {}
        code = read.get("failure_code")
        way = (
            self._stopped_way(task_id, code, "BLOCKED")
            if read.get("status") == "BLOCKED" and isinstance(code, str)
            else {}
        )
        if not (offer or way):
            return read
        offered = cast(dict[str, Any], read.get("next_requests") or {})
        return {
            **read,
            **way,
            "next_requests": {
                **offered,
                **cast(dict[str, Any], way.get("next_requests") or {}),
                **offer,
            },
        }

    def _inputs_way(self, answer: dict[str, Any]) -> dict[str, Any]:
        """A refusal while the workspace's inputs are not ready, named by their state, with the
        request that settles them (V604): the update that holds the running place with its data
        stage, whose status names its way on (a deferral resumes once due); with none, a data
        update, which settles a provider's deferral a cancelled update left and names any data
        case. Another kind holding the place, the verification sweep among them, holds no
        inputs."""
        code = str(answer.get("failure_code") or "")
        if code != "strategy_score.workspace_inputs_not_ready":
            return answer
        workspace = self.workspace_session.workspace
        try:
            state = read_workspace_inputs(
                workspace, installed_data_update_binding(workspace)
            ).readiness_status
        except (ValueError, OSError):
            state = None
        holder = self.workspace_session.task_control_registry.running_place_holder()
        updates = {DATA_UPDATE_TASK_KIND, DecisionAdvancementApplication.task_kind}
        return {
            **answer,
            "failure_code": code if state is None else f"{code}:{state}",
            "next_requests": {"status": {"operation": "STATUS", "task_id": str(holder.task_id)}}
            if holder is not None and holder.task_kind in updates
            else {"data_update": {"operation": "DATA_UPDATE_PLAN"}},
        }

    def _deferred_way(self, task_kind: str, task_id: UUID) -> dict[str, object]:
        """A deferred Task's way on: its owner's words, its retry time, the request that resumes
        it and the readback that says so (V507). A kind without a deferral readback goes on by
        its recovery view; a readback that refuses leaves its read, which says why."""
        if task_kind == PREPARATION_TASK_KIND:
            operation, reader = "WORKSPACE_PREPARE_READBACK", self._preparation_readback
        elif task_kind == DATA_UPDATE_TASK_KIND:
            operation, reader = "DATA_UPDATE_READBACK", self._data_update_readback
        elif task_kind == DecisionAdvancementApplication.task_kind:
            # A research update the provider deferred in its data stage (V601).
            operation, reader = "RESEARCH_UPDATE_READBACK", self._research_update_readback
        else:
            recovery = {"operation": "TASK_RECOVERY", "task_id": str(task_id)}
            return {"next_requests": {"recovery": recovery}}
        read = {"operation": operation, "task_id": str(task_id)}
        try:
            owned = reader(task_id)
        except (FileNotFoundError, KeyError, ValueError):
            return {"next_requests": {"read": read}}
        resume = (owned.get("next_requests") or {}).get("resume")
        return {
            "detail": owned.get("detail"),
            "retry_after_at": owned.get("retry_after_at"),
            "next_requests": {**({"resume": resume} if resume else {}), "read": read},
        }

    def _moved_on(
        self, task_id: UUID, seen: TaskSafeProjection, wait_seconds: float
    ) -> TaskSafeProjection:
        """The Task's status once it moves on from ``seen`` (its lifecycle or a verified
        stage), or after ``wait_seconds``: a follow learns of a change when it happens
        instead of at its next poll (binding plan N8). Read every tenth of a second, and
        nothing is held between reads, so the Task's own writes never wait on a follower."""

        deadline = monotonic() + wait_seconds
        mark = (seen.lifecycle, seen.verified_stage_count)
        current = seen
        while current.lifecycle in _LIVE_LIFECYCLES:
            remaining = deadline - monotonic()
            if remaining <= 0:
                break
            sleep(min(_FOLLOW_READ_SECONDS, remaining))
            current = self.dispatcher.status(task_id)
            if (current.lifecycle, current.verified_stage_count) != mark:
                break
        return current

    def pending_decisions(self, *, caller: OperationCaller = "HUMAN") -> dict[str, object]:
        """What waits on a person, each with the request that takes it (N5)."""
        # One reading of the Tasks serves the whole page: the studies waiting, the
        # stopped Tasks and the upgrade overview answer at one moment (V119).
        batch = self.workspace_session.task_control_registry.record_collection()
        tasks = batch.records
        task_refusals = [task_record_refusal(task_id) for task_id in batch.refused_task_ids]
        goal_attribution = self.goals.decision_attribution()
        for task in tasks:
            if task.task_kind == CRO_REVIEW_TASK_KIND and (
                goal_ids := goal_attribution.get(("task_id", str(task.task_id)))
            ):
                goal_attribution.setdefault(("review_key", task.input.input_hash), set()).update(
                    goal_ids
                )
        # The workspace's own state is its owners' answers, read in one boundary; a store a
        # writer holds leaves its part unread rather than read as empty (V45).
        owners: dict[str, dict[str, object]] = {}
        with self.workspace_session.reads(timeout_seconds=0.05) as batched:
            for name, request in (
                (
                    "preparation",
                    PortfolioResearchOperationRequest(operation="WORKSPACE_PREPARE_READBACK"),
                ),
                ("inputs", PortfolioResearchOperationRequest(operation="RESEARCH_INPUTS")),
                (
                    "data_update",
                    PortfolioResearchOperationRequest(operation="DATA_UPDATE_READBACK"),
                ),
            ):
                if not batched and name == "data_update":
                    continue
                try:
                    owners[name] = self.execute(request, caller=caller)
                except (ValueError, KeyError, OSError):
                    continue
        awaiting = self.experiments.awaiting(tasks)
        if task_refusals:
            # An unreadable Task may already promote a study or admit a preview.
            # Keep readable curation, but do not infer either kind of absence.
            awaiting["promotion"] = []
        first = self.goals.first_use()
        answer = pending_decisions(
            first_use=None
            if first is None or not self.goals.first_use_delegation(first)["active"]
            else {
                "goal_id": str(first.goal_id),
                "goal_hash": first.goal_hash,
                "objective": first.declaration.objective,
                **self.goals.first_use_delegation(first),
                "delegated_steps": self.goals.record(first).get("delegated_steps", []),
            },
            tasks=tasks,
            goal_attribution=goal_attribution,
            attention=self.supervisor.attention(tasks),
            awaiting=awaiting,
            data_issues=all_data_issues(
                lambda cursor: self.data_issues.readback(limit=50, cursor=cursor)
            ),
            overview=self._upgrade_overview(tasks, task_refusals),
            previews=[]
            if task_refusals
            else self.experiments.waiting_previews(awaiting["admitted"]),
            preparation=owners.get("preparation"),
            inputs=owners.get("inputs"),
            data_update=owners.get("data_update"),
        )
        if task_refusals:
            cast(list[dict[str, Any]], answer["decisions"]).extend(
                # An unreadable record is a defect the agent reads and repairs (STOPS-1).
                {
                    "kind": "TASK_RECORD_UNREADABLE",
                    **item,
                    "waits_on": "AGENT",
                    "goal_ids": sorted(goal_attribution.get(("task_id", item["task_id"]), ())),
                }
                for item in task_refusals
            )
            cast(dict[str, int], answer["counts"])["TASK_RECORD_UNREADABLE"] = len(task_refusals)
            answer["detail"] = task_refusals[0]["detail"]
            answer["refusals"] = task_refusals
        return answer

    def upgrade(self, request: PortfolioResearchOperationRequest) -> dict[str, object]:
        """What an upgrade touched (`upgrade_overview`), or its acknowledgement."""
        registry = self.workspace_session.task_control_registry
        if request.operation == "UPGRADE_OVERVIEW":
            batch = registry.record_collection()
            return self._upgrade_overview(
                batch.records, [task_record_refusal(task_id) for task_id in batch.refused_task_ids]
            )
        overview = self._upgrade_overview(registry.tasks())
        assert request.upgrade_set_hash is not None
        return acknowledge_upgrade(
            self.workspace_session.workspace,
            overview,
            confirmed=request.upgrade_set_hash,
            now=self.dispatcher.clock(),
        )

    def _installed_study_identities(self) -> dict[str, str]:
        if self._study_identities is None:
            # The installed study identities are the code this Host serves: asked once.
            self._study_identities = installed_study_identities()
        return self._study_identities

    def _upgrade_overview(
        self, tasks: tuple[TaskRecord, ...], task_refusals: list[dict[str, Any]] | None = None
    ) -> dict[str, object]:
        self._installed_study_identities()
        return upgrade_overview(
            workspace=self.workspace_session.workspace,
            registry=self.workspace_session.task_control_registry,
            review=self.review,
            resume_refusal=self.resume_refusal or (lambda _task: None),
            command_running=self.dispatcher.command_running,
            tasks=tasks,
            task_refusals=task_refusals or (),
            study_identities=self._study_identities,
            failed_studies=self.sweep.failed_studies(),
        )

    def _resume_refusal(self, projection: TaskSafeProjection) -> str | None:
        """Why a Task no command is driving cannot resume under what is installed now.

        Asked only of an interrupted or queued Task this Host is not already driving:
        a queued Task whose command waits its turn is not stalled."""

        if (
            self.resume_refusal is None
            or projection.lifecycle not in {TaskLifecycle.RECOVERY_REQUIRED, TaskLifecycle.QUEUED}
            or self.dispatcher.command_running(projection.task_id)
        ):
            return None
        return self.resume_refusal(
            self.workspace_session.task_control_registry.task(projection.task_id)
        )

    @staticmethod
    def _resume_refused(before: dict[str, object]) -> dict[str, object]:
        return {
            **before,
            "status": "REFUSED",
            "failure_code": before["resume_refusal"],
            "disposition": "REFUSED_CHANGED_SINCE_ADMISSION",
            "detail": (
                "This Task cannot resume: what it was admitted under has changed since (the "
                "installed code, or the binding it was planned against), and its owner refuses "
                "it as recorded. Its verified stages and artifacts stay as recorded. Cancel it "
                "and plan the same work again; the new plan is made under what is installed now."
            ),
            "next_requests": {
                "cancel": {
                    "operation": "CANCEL",
                    "task_id": before["task_id"],
                    "expected_task_hash": before["task_record_hash"],
                },
                "recovery": {"operation": "TASK_RECOVERY", "task_id": before["task_id"]},
            },
        }

    def replans(self) -> dict[str, TaskReplan]:
        """Each Task kind's re-PLAN, as the owner that admits it declares it (V188).

        Returns:
            The declarations of every owner this service composes, by Task kind.
        """
        owners = (
            owner
            for item in fields(self)
            if isinstance(owner := getattr(self, item.name), _AdmitsTasks)
        )
        return {replan.task_kind: replan for owner in owners for replan in owner.replans}

    def recovery_view(self, task_id: UUID) -> dict[str, object]:
        """What stopped, what stays verified and what the owners permit, at one moment.

        Task Control is read once under its own lock (`board`); the dispatcher's
        two facts are read beside it. The `status` block is that same projection as
        the operation reader sees it (the dispatcher's operation-return mask), not a
        second read, so the document never mixes Task versions.
        """
        snapshot = self.workspace_session.task_control_registry.board(task_id)
        replan = self._task_replan(snapshot.task)
        resume_refusal = self._resume_refusal(snapshot.projection)
        attention = self.supervisor.attention((snapshot.task,)).get(task_id)
        heartbeats = (
            TaskHeartbeatReadout(signals=(), unreadable=())
            if snapshot.execution is None
            else self.heartbeats.read(snapshot.execution.execution_id)
        )
        view = build_task_recovery_view(
            replans=self.replans(),
            snapshot=snapshot,
            running=self.dispatcher.command_running(task_id),
            worker_failure=self.dispatcher.failure(task_id),
            heartbeats=heartbeats,
            recoverable_kinds=(
                self.recoverable_task_kinds() if self.recoverable_task_kinds else frozenset()
            ),
            observed_at=self.dispatcher.clock(),
            blocked_retry_reason=self.experiments.blocked_retry_reason(snapshot.task)
            or _runner_retry_reason(snapshot.task),
            resume_refusal=resume_refusal,
            replan_refusal=str(replan["detail"]) if replan and "failure_code" in replan else None,
            attention=attention,
        )
        status = task_status_body(self.dispatcher.masked(snapshot.projection))
        status["worker_failure"] = self.dispatcher.failure(task_id)
        status["task_record_hash"] = snapshot.projection.task_record_hash
        status["resume_refusal"] = resume_refusal
        body: dict[str, object] = {**view.model_dump(mode="json"), "status": status}
        # Each permitted action that takes a version, bound to the Task and the version this
        # view read, so `recovery run --from` sends what was seen; a Task that moved since is
        # refused as stale, nothing applied (V443).
        versioned = {"RECOVER", "CANCEL"}  # the actions whose request takes the version
        offered = {
            str(action["action"]).lower(): {
                "operation": action["operation"],
                "task_id": str(task_id),
                "expected_task_hash": snapshot.projection.task_record_hash,
            }
            for action in cast(list[dict[str, object]], body.get("actions") or [])
            if action.get("available") and action.get("operation") in versioned
        }
        # The owner's verified durable declaration supplies every field. A stopped source
        # version is carried only when its declaration has a distinct preview and admission.
        for action in cast(list[dict[str, object]], body.get("actions") or []):
            if action.get("action") != "REPLAN" or not action.get("available"):
                continue
            if replan and "operation" in replan:
                offered["replan"] = replan
        if replan and "failure_code" in replan:
            offered.update(cast(dict[str, Any], replan.get("next_requests") or {}))
        # A blocked Task's owner's own way on, as its status offers it (V511: RR5's recovery
        # view offered only a generic re-plan).
        code = snapshot.projection.latest_failure_code
        owned = (
            cast(dict[str, Any], self._stopped_way(task_id, code, "BLOCKED").get("next_requests"))
            if code and snapshot.projection.lifecycle is TaskLifecycle.BLOCKED
            else None
        )
        if offered or owned:
            body["next_requests"] = {**(owned or {}), **offered}
        return body

    def _task_replan(self, task: TaskRecord) -> dict[str, object] | None:
        """Ask the same composed owner that declares this Task's preview to bind it."""
        owned = self._task_replan_owner(task)
        if owned is None:
            return None
        declared, owner = owned
        if declared.preview is None:
            return None
        _required, allowed = PortfolioResearchOperationRequest.field_contract(declared.preview)
        choices = allowed - {"recovery_task_id", "recovery_task_hash"}
        request = (
            {"operation": declared.preview}
            if not choices
            else cast(dict[str, object], cast(Any, owner).replan_request(task))
        )
        if task.lifecycle in {
            TaskLifecycle.BLOCKED,
            TaskLifecycle.CANCELLED,
            TaskLifecycle.RECOVERY_REQUIRED,
        }:
            return {
                **request,
                "recovery_task_id": str(task.task_id),
                "recovery_task_hash": task.record_hash,
            }
        return request

    def _task_replan_owner(self, task: TaskRecord) -> tuple[TaskReplan, Any] | None:
        """Find the owner behind the current composed replan declaration."""
        declaration = self.replans().get(task.task_kind)
        if declaration is None:
            return None
        for item in fields(self):
            owner = getattr(self, item.name)
            if isinstance(owner, _AdmitsTasks) and any(
                replan == declaration for replan in owner.replans
            ):
                return declaration, owner
        return None

    @staticmethod
    def _normalized_operation_request(
        request: PortfolioResearchOperationRequest,
    ) -> dict[str, object]:
        """Return the JSON form of an operation, without recovery provenance fields."""
        values = {
            item.name: getattr(request, item.name)
            for item in fields(request)
            if item.name not in {"recovery_task_id", "recovery_task_hash"}
            and getattr(request, item.name) is not None
        }
        return cast(
            dict[str, object],
            PortfolioResearchRequestDocument.model_validate(values).model_dump(
                mode="json", exclude_none=True
            ),
        )

    def _normalized_replan_request(self, request: Mapping[str, object]) -> dict[str, object]:
        """Validate and normalize one owner-supplied operation request."""
        typed = PortfolioResearchRequestDocument.model_validate(
            dict(request)
        ).to_operation_request()
        return self._normalized_operation_request(typed)

    def _recovery_context(
        self, request: PortfolioResearchOperationRequest
    ) -> tuple[TaskRecord, TaskReplan, bool, dict[str, object]] | dict[str, object] | None:
        """Validate a replan preview or its exact previously offered admission."""
        source_id = request.recovery_task_id
        source_hash = request.recovery_task_hash
        if source_id is None and source_hash is None:
            return None
        if source_id is None or source_hash is None:
            return self._stale_recovery_context(
                source_id, code="portfolio_research.recovery_context_pair_required"
            )
        registry = self.workspace_session.task_control_registry
        try:
            source = registry.task(source_id)
        except TaskNotFoundError:
            raise
        if source.record_hash != source_hash:
            return self._stale_recovery_context(source_id)
        if source.lifecycle not in {
            TaskLifecycle.BLOCKED,
            TaskLifecycle.CANCELLED,
            TaskLifecycle.RECOVERY_REQUIRED,
        }:
            return self._stale_recovery_context(
                source_id, code="portfolio_research.recovery_request_not_offered"
            )
        owned = self._task_replan_owner(source)
        if owned is None:
            return self._stale_recovery_context(
                source_id, code="portfolio_research.recovery_request_not_offered"
            )
        declaration, _owner = owned
        if declaration.preview is None or request.operation not in {
            declaration.preview,
            declaration.admitting,
        }:
            return self._stale_recovery_context(
                source_id, code="portfolio_research.recovery_request_not_offered"
            )
        try:
            normalized = self._normalized_operation_request(request)
            if request.operation == declaration.preview:
                expected_offer = self._task_replan(source)
                if expected_offer is None:
                    return self._stale_recovery_context(
                        source_id, code="portfolio_research.recovery_request_not_offered"
                    )
                expected = self._normalized_replan_request(expected_offer)
                if normalized != expected:
                    return self._stale_recovery_context(
                        source_id, code="portfolio_research.recovery_request_not_offered"
                    )
                if declaration.preview == declaration.admitting:
                    try:
                        registry.record_recovery_link(
                            source_task_id=source.task_id,
                            source_record_hash=source.record_hash,
                            admission_request=cast(dict[str, Any], normalized),
                            observed_at=self.dispatcher.clock(),
                        )
                    except (TaskVersionStale, TaskTransitionRejected):
                        return self._stale_recovery_context(source_id)
                    return source, declaration, False, normalized
                return source, declaration, True, normalized
        except (KeyError, TypeError, ValueError, ValidationError):
            return self._stale_recovery_context(
                source_id, code="portfolio_research.recovery_request_not_offered"
            )
        # The registry read is outside the malformed-request handler: database authority
        # failures must retain the Task Control refusal instead of being called a bad request.
        previews = registry.recovery_links(source_id)
        if not any(
            link.source_record_hash == source_hash
            and link.successor_task_id is None
            and link.admission_request == normalized
            for link in previews
        ):
            return self._stale_recovery_context(
                source_id, code="portfolio_research.recovery_request_not_offered"
            )
        return source, declaration, False, normalized

    def _stale_recovery_context(
        self,
        source_id: UUID | None,
        *,
        code: str = "local_application.confirmation_stale",
    ) -> dict[str, object]:
        """Return the located refusal with a request to refresh this Task's recovery view."""
        answer: dict[str, object] = {
            **refused(code),
            "refused_at": "OPERATION_ENTRY",
            "next_requests": {
                "recovery": {
                    "operation": "TASK_RECOVERY",
                    **({"task_id": str(source_id)} if source_id is not None else {}),
                }
            },
        }
        if source_id is None:
            return answer
        answer["task_id"] = str(source_id)
        try:
            source = self.workspace_session.task_control_registry.task(source_id)
        except TaskNotFoundError:
            return answer
        answer["lifecycle"] = source.lifecycle.value
        answer["task_record_hash"] = source.record_hash
        return answer

    @staticmethod
    def _rewrite_admission_offer(
        value: object, operation: str, source_id: UUID, source_hash: str
    ) -> tuple[object, list[dict[str, object]]]:
        """Bind the matching owner request anywhere in its mapping or list of offers."""
        if isinstance(value, Mapping):
            if value.get("operation") == operation:
                offer = dict(value)
                original = dict(offer)
                offer["recovery_task_id"] = str(source_id)
                offer["recovery_task_hash"] = source_hash
                return offer, [original]
            found: list[dict[str, object]] = []
            rewritten: dict[object, object] = {}
            for key, item in value.items():
                updated, matches = PortfolioResearchOperations._rewrite_admission_offer(
                    item, operation, source_id, source_hash
                )
                rewritten[key] = updated
                found.extend(matches)
            return (rewritten if found else value), found
        if isinstance(value, list):
            found = []
            rewritten_items: list[object] = []
            for item in value:
                updated, matches = PortfolioResearchOperations._rewrite_admission_offer(
                    item, operation, source_id, source_hash
                )
                rewritten_items.append(updated)
                found.extend(matches)
            return (rewritten_items if found else value), found
        return value, []

    def _recovery_preview_answer(
        self, source: TaskRecord, declaration: TaskReplan, body: dict[str, object]
    ) -> dict[str, object]:
        """Persist each actual filled admission request and carry its exact source pair."""
        if declaration.preview == "WORKSPACE_PREPARE_PLAN" and body.get(
            "predecessor_task_id"
        ) != str(source.task_id):
            return self._stale_recovery_context(
                source.task_id, code="portfolio_research.recovery_request_not_offered"
            )
        next_requests = body.get("next_requests")
        if next_requests is None:
            return body
        rewritten, matches = self._rewrite_admission_offer(
            next_requests,
            declaration.admitting,
            source.task_id,
            source.record_hash,
        )
        if not matches:
            return body
        registry = self.workspace_session.task_control_registry
        try:
            for match in matches:
                admission_request = self._normalized_replan_request(match)
                registry.record_recovery_link(
                    source_task_id=source.task_id,
                    source_record_hash=source.record_hash,
                    admission_request=cast(dict[str, Any], admission_request),
                    observed_at=self.dispatcher.clock(),
                )
        except (TaskVersionStale, TaskTransitionRejected):
            return self._stale_recovery_context(source.task_id)
        return {**body, "next_requests": rewritten}

    def _recovery_admission_answer(
        self,
        source: TaskRecord,
        admission_request: dict[str, object],
        body: dict[str, object],
    ) -> dict[str, object]:
        """Link only the distinct canonical same-kind Task the admitted operation returned."""
        value = body.get("task_id")
        if not isinstance(value, str):
            return body
        try:
            successor_id = UUID(value)
        except ValueError:
            return body
        if successor_id == source.task_id:
            return body
        try:
            successor = self.workspace_session.task_control_registry.task(successor_id)
        except TaskNotFoundError:
            return body
        if successor.task_kind != source.task_kind:
            return body
        self.workspace_session.task_control_registry.record_recovery_link(
            source_task_id=source.task_id,
            source_record_hash=source.record_hash,
            admission_request=cast(dict[str, Any], admission_request),
            successor_task_id=successor_id,
            observed_at=self.dispatcher.clock(),
        )
        return body

    def _run_answer(
        self, sent: CommandSubmission, data_plan: WorkspaceDataUpdatePlan | None = None
    ) -> dict[str, Any]:
        """A run's answer by its Task as every read of it sees it now (V600).

        The admission carries the state the Task was admitted or found in; one found stopped and
        reopened, or already taken up by the worker, has moved since. So the answer reads the
        Task as its status does, and one that stands stopped names its stop's code, words and way
        on, as that status names them. A refused admission names its own code, and a deferral
        refused before its retry time names that time, read for the plan's data stage,
        `data_plan` (V601).
        """
        if sent.task_id is None:
            refused: dict[str, Any] = {
                "status": sent.disposition,
                "task_id": None,
                "lifecycle": None,
                "failure_code": sent.refusal_detail,
            }
            if (
                sent.refusal_detail == "workspace_data_update.retry_not_due"
                and data_plan is not None
                and self.data_update is not None
            ):
                retry = self.data_update.retry_after(data_plan)
                refused["retry_after_at"] = None if retry is None else retry.isoformat()
            return refused
        projection = self.dispatcher.status(sent.task_id)
        return {
            "status": sent.disposition,
            "task_id": str(sent.task_id),
            "lifecycle": projection.lifecycle.value,
            "failure_code": None,
            **self._stopped_answer(projection),
        }

    def _stopped_answer(self, projection: TaskSafeProjection) -> dict[str, Any]:
        """A stopped Task's code, words and way on, or one owing a recovery's way on, as every
        read of it names them; nothing for a Task in any other state.

        A blocked one says why, as a refusal does, since a read of it answers REFUSED (V424); a
        stop its plan's rerun resumes offers that rerun beside its recovery (V600).
        """
        state = projection.lifecycle.value
        if state not in {"BLOCKED", "RECOVERY_REQUIRED"}:
            return {}
        task_id, code = projection.task_id, projection.latest_failure_code
        way = self._stopped_way(task_id, code, state) if code else {}
        answer: dict[str, Any] = {
            key: way[key] for key in ("next_action", "network_access") if key in way
        }
        answer["detail"] = way.get("detail") or stop_detail(
            projection.task_kind, code or "", "TASK_CONTROL"
        )
        if state == "BLOCKED":
            answer["failure_code"] = code
        elif code == "task_control.child_start_failed":
            answer["failure_code"] = code
            answer.update({key: way[key] for key in ("detail", "next_action") if key in way})
        answer["next_requests"] = {
            **cast(dict[str, Any], way.get("next_requests") or {}),
            **(self._resume_offer(task_id) if state == "BLOCKED" else {}),
            "recovery": {"operation": "TASK_RECOVERY", "task_id": str(task_id)},
        }
        return answer

    def _resume_offer(self, task_id: UUID) -> dict[str, object]:
        """A stopped update's resume: its own plan run again, where its stop is one a rerun
        resumes (V600).

        A data update and a research update stop in the data update's stages, whose owner says
        which stops a rerun resumes; each is offered through the door that reruns its kind.
        """
        if self.data_update is None:
            return {}
        task = self.workspace_session.task_control_registry.task(task_id)
        door = (
            "DATA_UPDATE_RUN"
            if task.task_kind == DATA_UPDATE_TASK_KIND
            else "RESEARCH_UPDATE_RUN"
            if task.input.input_schema_id == DECISION_ADVANCEMENT_SCHEMA
            else None
        )
        plan = task.input.payload.get("plan")
        if (
            door is None
            or task.lifecycle is not TaskLifecycle.BLOCKED
            or not self.data_update.resumes(task.failure_code)
            or not isinstance(plan, dict)
        ):
            return {}
        return {"resume": {"operation": door, "update_plan_hash": str(plan["content_hash"])}}

    def _stopped_way(self, task_id: UUID, code: str, lifecycle: str) -> dict[str, Any]:
        """A stopped Task's owner's words and way on, a Portfolio study's bound to the Alpha
        study and candidate its plan names (V511)."""
        book = (
            self.experiments.book_source(task_id)
            if code.startswith("portfolio_research.")
            else None
        )
        way = explain(
            code,
            task_id=str(task_id),
            lifecycle=lifecycle,
            portfolio=book,
            workspace=self.workspace_session.workspace,
        )
        # A code the door's table words reads in those words when its owner adds none, as a
        # refusal does at the door (OP4): a walk refused before it opens says why (V519).
        way = way or dict(refusal_words(code) or {})
        request = self._task_replan(self.workspace_session.task_control_registry.task(task_id))
        if request:
            offered = cast(dict[str, Any], way.get("next_requests") or {})
            continuation = (
                {"replan": request}
                if "operation" in request
                else cast(dict[str, Any], request.get("next_requests") or {})
            )
            way = {**way, "next_requests": {**offered, **continuation}}
        return way

    def guardian(self) -> dict[str, object]:
        """Read every Task not finished at once, as Guanyin sees it, for the PM.

        Each is its own recovery view, projected: its liveness, its real progress in stages,
        the work it keeps, its incidents and the recovery its owners permit (GY, V78).
        Nothing here repairs or retries; a finished Task is its readback's, not this read's.
        """
        entries = []
        counts: dict[str, int] = {}
        batch = self.workspace_session.task_control_registry.record_collection()
        refusals = [task_record_refusal(task_id) for task_id in batch.refused_task_ids]
        for task in batch.records:
            if task.lifecycle in _FINISHED_LIFECYCLES:
                continue
            try:
                view = self.recovery_view(task.task_id)
            except TaskQueueHeadAuthorityError:
                raise  # the unreadable dependency belongs to the queue head, not this peer
            except (ValueError, KeyError, OSError):
                refusals.append(task_record_refusal(str(task.task_id)))
                continue
            if (
                task.lifecycle is TaskLifecycle.RECOVERY_REQUIRED
                and not cast(dict[str, object], view["attention"])["unresolved"]
            ):
                continue
            stages = cast(list[dict[str, object]], view["stages"])
            health = cast(dict[str, object], view["health"])
            counts[str(health["status"])] = counts.get(str(health["status"]), 0) + 1
            entries.append(
                {
                    "task_id": view["task_id"],
                    "task_kind": view["task_kind"],
                    "lifecycle": view["lifecycle"],
                    "health": health,
                    "liveness": view["liveness"],
                    "progress": {
                        "verified_stage_count": view["verified_stage_count"],
                        "total_stage_count": view["total_stage_count"],
                        "current_stage": cast(dict[str, object], view["status"])["current_stage"],
                    },
                    "preserved_stages": [
                        stage["stage_id"] for stage in stages if stage["lifecycle"] == "VERIFIED"
                    ],
                    "incidents": view["incidents"],
                    "actions": [
                        action
                        for action in cast(list[dict[str, object]], view["actions"])
                        if action["available"]
                    ],
                    "next_requests": {
                        "recovery": {"operation": "TASK_RECOVERY", "task_id": view["task_id"]}
                    },
                }
            )
        return {
            "status": "TASK_GUARDIAN",
            **({"refusals": refusals} if refusals else {}),
            # Guanyin's own mode, read from its view rather than restated here.
            "guardian_mode": TaskRecoveryView.model_fields["guardian_mode"].default,
            "tasks": entries,
            "counts": counts,
            "detail": (
                f"{len(entries)} Task{'' if len(entries) == 1 else 's'} not finished."
                if entries
                else "Some Task records could not be read."
                if refusals
                else "No Task is running, waiting or stopped."
            ),
        }

    def tasks(
        self,
        *,
        agent_session: str | None = None,
        limit: int = 50,
        cursor: str | None = None,
    ) -> dict[str, object]:
        """The workspace's Tasks; or one agent session's, a page at a time (U54).

        A session's page names each Task's submitter, its final lifecycle and the artifact its
        owner reads back when this answers -- the activity feed's own readback, never the feed.

        Args:
            agent_session: The agent session whose Tasks to list; none lists every Task.
            limit: How many of the session's Tasks one page holds, at most 50.
            cursor: The `next_cursor` the session's previous page answered.

        Returns:
            The rows; a session's listing adds `next_cursor`.
        """
        registry = self.workspace_session.task_control_registry
        batch = registry.record_collection()
        attention = self.supervisor.attention(batch.records)
        if agent_session is None:
            all_tasks: dict[str, object] = {
                "tasks": [
                    self._task_row(value, attention.get(value.task_id))
                    for value in self.dispatcher.active()
                ]
            }
            if batch.refused_task_ids:
                all_tasks["refusals"] = [
                    task_record_refusal(task_id) for task_id in batch.refused_task_ids
                ]
            return all_tasks
        submitted = registry.submitted_by_session(agent_session)
        if cursor is not None:
            position = next(
                (i for i, (task_id, _) in enumerate(submitted) if str(task_id) == cursor), None
            )
            if position is None:
                return {"status": "REFUSED", "failure_code": "tasks.cursor_moved_reload"}
            submitted = submitted[position + 1 :]
        page = submitted[: min(limit, _TASK_PAGE_ROWS)]
        projection_batch = registry.projection_collection(task_id for task_id, _ in page)
        projected = {value.task_id: value for value in projection_batch.projections}
        refused_ids = frozenset(projection_batch.refused_task_ids)
        canonical_refused_ids = frozenset(batch.refused_task_ids)
        rows: list[dict[str, object]] = []
        for task_id, agent in page:
            if task_id in refused_ids or str(task_id) in canonical_refused_ids:
                continue
            projection = projected.get(task_id) or self.dispatcher.status(task_id)
            final = self.dispatcher.final_lifecycle(projection)
            rows.append(
                {
                    **self._task_row(projection, attention.get(task_id)),
                    "submitted_by": agent.model_dump(mode="json"),
                    "final_state": None if final is None else final.value,
                    "artifact": (
                        self._read_back(projection) if final is TaskLifecycle.SUCCEEDED else None
                    ),
                }
            )
        answer: dict[str, object] = {
            "tasks": rows,
            "next_cursor": str(page[-1][0]) if len(submitted) > len(page) else None,
        }
        refusals = [
            task_record_refusal(str(task_id))
            for task_id, _ in page
            if str(task_id) in canonical_refused_ids and task_id not in refused_ids
        ]
        refusals.extend(
            self.task_projection_refusal(task_id) for task_id in projection_batch.refused_task_ids
        )
        if refusals:
            answer["refusals"] = refusals
        return answer

    @staticmethod
    def task_queue_authority_refusal(error: TaskQueueHeadAuthorityError) -> dict[str, object]:
        """Name the unreadable queue dependency without assigning a peer a false failure."""
        return {
            **refused("task_control.database_authority_unreadable"),
            "queue_head_task_id": error.task_id,
            "next_requests": {
                "workspace": {"operation": "WORKSPACE_SHOW"},
                "backups": {"operation": "WORKSPACE_BACKUPS"},
            },
        }

    @staticmethod
    def task_projection_refusal(task_id: UUID) -> dict[str, object]:
        """Name an unreadable Task projection without inventing its state."""
        code = "task_control.projection_unavailable"
        return {
            "task_id": str(task_id),
            "status": "REFUSED",
            "failure_code": code,
            **refusal_words(code),
            "next_requests": {
                "workspace": {"operation": "WORKSPACE_SHOW"},
                "backups": {"operation": "WORKSPACE_BACKUPS"},
            },
        }

    def _task_row(
        self, value: TaskSafeProjection, attention: TaskAttentionFact | None = None
    ) -> dict[str, object]:
        return {
            **task_status_body(value),
            **self._evidence_completion(value),
            "task_record_hash": value.task_record_hash,
            **({"attention": attention.model_dump(mode="json")} if attention is not None else {}),
            **(
                {"detail": stop_detail(value.task_kind, value.latest_failure_code, "TASK_CONTROL")}
                if value.latest_failure_code == "task_control.ledger_rebuilt"
                else {}
            ),
            **(
                {"stop_next": LEDGER_REBUILT_NEXT}
                if value.latest_failure_code == "task_control.ledger_rebuilt"
                and "operation"
                in (
                    self._task_replan(
                        self.workspace_session.task_control_registry.task(value.task_id)
                    )
                    or {}
                )
                else {}
            ),
            "resume_refusal": self._resume_refusal(value),
            # A finished Task's span is fixed, so a listing reads the same twice; a
            # running one is its running_since (CLI-4).
            "running_seconds": None
            if value.running_since is None or value.lifecycle in _LIVE_LIFECYCLES
            else round(
                max(0.0, (value.last_activity_at - value.running_since).total_seconds()),
                3,
            ),
        }

    def _evidence_completion(self, projection: TaskSafeProjection) -> dict[str, object]:
        """Evidence's completion words, shared by STATUS/--wait and Task listings."""
        adapter = None if self.review is None else self.review.evidence_task_adapter
        if (
            adapter is None
            or projection.task_kind != adapter.task_kind
            or projection.lifecycle is not TaskLifecycle.SUCCEEDED
        ):
            return {}
        task = self.workspace_session.task_control_registry.task(projection.task_id)
        assert self.review is not None
        return EvidenceCroProjector(self.review).completed_preparation(task)

    def _read_back(self, projection: TaskSafeProjection) -> dict[str, object] | None:
        """The artifact a succeeded Task's owner opens now; unavailable when it cannot (U54)."""
        published = _PUBLISHED_ARTIFACTS.get(projection.task_kind)
        if published is None:
            return None
        try:
            reference = self.artifact_reference(projection.task_kind, projection.task_id)
        except (ValueError, KeyError, OSError) as error:
            code = str(located_failure(error, "tasks.artifact_readback_failed")["failure_code"])
        else:
            if reference is not None:
                return {
                    "artifact_kind": reference.artifact_kind,
                    "artifact_hash": reference.artifact_hash,
                    "availability": "AVAILABLE",
                    "failure_code": None,
                }
            code = "tasks.artifact_readback_absent"
        return {
            "artifact_kind": published,
            "artifact_hash": None,
            "availability": "UNAVAILABLE",
            "failure_code": code,
        }

    def artifact_reference(self, command_kind: str, task_id: UUID) -> ArtifactReference | None:
        """Open the owner's published result for a returned command, if it has one.

        Only the two kinds whose result the workbench opens directly are named;
        every other kind is read through its own readback route. A kind not
        listed answers `None`, which the observer records as lifecycle only.
        The reference is the *opened* artifact's own identity: the Portfolio
        result is read through the same owner readback REPORT uses, so an index
        entry alone never counts; an experiment's evidence is the one its Task's
        own last stage read back through its verifier and sealed in a VERIFIED
        receipt, so no second readback runs in the Host after the Task (W10).
        The activity feed and a session's Task listing read through this one (U54).

        Args:
            command_kind: The command's kind, which is its Task's kind.
            task_id: The Task.

        Returns:
            The opened artifact's reference, or None when it has none to open.

        Raises:
            ValueError: The opened result is not the one the index named for this Task.
        """
        published = _PUBLISHED_ARTIFACTS.get(command_kind)
        if published is None:
            return None
        if command_kind == PORTFOLIO_RUN_COMMAND:
            if self.service is None or self.application is None:
                return None
            for manifest in self.application.pipeline.manifests():
                if manifest.task_id != task_id:
                    continue
                # The index names the result; opening it is the verification.
                result = self.service.open_result(manifest.result_hash)
                if (
                    result.result_hash != manifest.result_hash
                    or self.service.originating_task(result.result_hash) != task_id
                ):
                    raise ValueError("portfolio_application.result_readback_mismatch")
                return ArtifactReference(published, result.result_hash)
            return None
        evidence_hash = self.experiments.verified_evidence_hash(task_id)
        return None if evidence_hash is None else ArtifactReference(published, evidence_hash)

    def cancel(self, task_id: UUID, *, expected_task_hash: str | None = None) -> dict[str, object]:
        """Request task cancellation with exact optimistic version enforcement.

        Request cancellation; a confirmed version is enforced by Task Control's own
        transaction, and a Task that moved meanwhile is refused with nothing applied.
        """
        try:
            accepted = self.dispatcher.request_cancel(
                task_id, expected_task_hash=expected_task_hash
            )
        except TaskVersionMovedError:
            current = self.status(task_id)
            return {
                "status": "REFUSED",
                "failure_code": "local_application.confirmation_stale",
                "task_id": current["task_id"],
                "lifecycle": current["lifecycle"],
                "task_record_hash": current["task_record_hash"],
                "refused_at": "TASK_CONTROL",
            }
        body = task_status_body(self.dispatcher.status(task_id))
        body["cancel_requested"] = accepted
        return body

    def results(self, task_id: UUID | None = None) -> dict[str, object]:
        """List readable retained results and name unreadable discovery records.

        Args:
            task_id: Optional exact producer/reuser; reads its metadata association only.

        Returns:
            Published result/report/program/task identities and completion metadata, plus typed
            item refusals when a by-result index or its sealed manifest could not be verified.
            Such a refusal does not establish whether a result or its Task is absent.
        """
        assert self.application is not None
        if task_id is None:
            manifests, unreadable = self.application.pipeline.manifest_collection()
        else:
            try:
                manifest = self.application.pipeline.find_for_task(task_id)
                manifests = () if manifest is None else (manifest,)
                unreadable = ()
            except (ValueError, OSError, KeyError, TypeError, ContentAddressedStoreError) as error:
                return {
                    "results": [],
                    "refusals": [
                        {
                            "status": "REFUSED",
                            "record_id": f"task:{task_id}",
                            "task_id": str(task_id),
                            "failure_code": public_failure(
                                error, "research_history.entry_unreadable"
                            ),
                            "detail": (
                                "Research History could not verify the recorded metadata "
                                f"for Task {task_id}. "
                                "Read that Task's status before treating "
                                "its result or publication as absent."
                            ),
                            "next_requests": {
                                "task": {"operation": "STATUS", "task_id": str(task_id)}
                            },
                        }
                    ],
                }
        refusals: list[dict[str, object]] = []
        for issue in unreadable:
            identifier = issue.get("result_hash") or issue["index_file"]
            requests: dict[str, dict[str, object]] = {
                "workspace": {"operation": "WORKSPACE_SHOW"},
                "backups": {"operation": "WORKSPACE_BACKUPS"},
            }
            if result_hash := issue.get("result_hash"):
                requests["report"] = {"operation": "REPORT", "result_hash": result_hash}
            row: dict[str, object] = {
                **issue,
                "record_id": identifier,
                "detail": (
                    f"The saved result record {identifier} could not be verified. Its originating "
                    "Task and whether the result can be read remain unknown. Read `workspace "
                    "show` and `backup list`; if a listed verified generation holds it, restore "
                    "that generation into a new directory with `alphalattice backup restore "
                    "--dir <new directory> --generation <verified generation hash> "
                    "--workspace-id <workspace id> --root <backup root>`, then read it there."
                ),
                "next_requests": requests,
            }
            refusals.append(row)
        body: dict[str, object] = {
            "results": [
                {
                    "result_hash": manifest.result_hash,
                    "report_hash": manifest.report_hash,
                    "program_hash": manifest.program_hash,
                    "task_id": str(manifest.task_id),
                    "completed_at": manifest.completed_at.isoformat(),
                }
                for manifest in manifests
            ],
        }
        if refusals:
            body["refusals"] = refusals
        return body

    def report(self, result_hash: str, portfolio_session: str | None = None) -> dict[str, object]:
        # One opened result for the whole answer: the readouts are derived from
        # the report already in hand rather than re-derived from the hash, which
        # is what made a single REPORT open the result and the report twice each.
        """Open one exact research result and project its report and declared positions.

        Args:
            result_hash: Exact retained result identity.
            portfolio_session: Optional session for declared-path position readback.

        Returns:
            Report, originating tasks, window, controls, unit-bearing readouts and optional
            positions; no research is rerun.
        """
        assert self.service is not None
        result = self.service.open_result(result_hash)
        report = self.service.report_of(result)
        selector = {
            "result_hash": result_hash,
            **({"portfolio_session": portfolio_session} if portfolio_session else {}),
        }
        assert self.application is not None
        program = self.application.ledger.load_program(result.program_hash)
        execution = self.application.ledger.load_execution(result.execution_ledger_hash)
        packages = self.application.resolver.installed_packages()
        package = packages.get(program.strategy_package_hash or "")
        economics = self.application.ledger.load_economics(report.economic_ledger_hash)
        readouts = self.service.readouts_of(report, economics=economics)
        performance = self.service.selected_window_performance_of(
            report, execution=execution, economics=economics
        )
        task = self.service.originating_task(result_hash)
        assert self.activations is not None and self.review is not None
        standing = self.activations.standing(task, result_hash=result_hash).model_dump(mode="json")
        review = EvidenceReviewDelivery(self.review).review_standing(
            BookSelector(result_hash=result_hash)
        )
        position = (
            self.service.path_readback_of(
                report,
                date.fromisoformat(portfolio_session),
                execution=execution,
                economics=economics,
            )
            if portfolio_session is not None
            else {}
        )
        if position and self.application is not None:
            assert portfolio_session is not None
            position["reading_kind"] = "INSTALLED_RESULT"
            position["reading_context"] = saved_portfolio_context(
                workspace=self.application.workspace,
                packages=packages,
                manifest=self.workspace_manifest,
                program=program,
                session=date.fromisoformat(portfolio_session),
            )
            position["spec"] = _spec_document(self._result_spec(result, task))
        return {
            **position,
            "standing": standing,
            "review_standing": review,
            "selected_window_metrics": {
                "cumulative_return": report.window_cumulative_net_wealth - 1.0,
                "cost_bps": float(readouts.platform_one_way_cost_bps),
                **cast(dict[str, float], performance["selected_window_metrics"]),
            },
            "selected_window_metric_absences": {
                **cast(dict[str, object], performance["selected_window_metric_absences"]),
                **{
                    metric: {
                        "status": "UNAVAILABLE",
                        "reason": "NOT_RECORDED_IN_DECLARED_PATH_REPORT",
                        "detail": "The saved installed-strategy report does not record this metric "
                        "for its selected window; this read does not estimate it.",
                    }
                    for metric in (
                        "information_ratio",
                        "benchmark_relative_return",
                        "beta",
                        "tracking_error",
                        "zero_cash_jensen_alpha",
                    )
                },
            },
            "selected_window_metric_provenance": performance["selected_window_metric_provenance"],
            "strategy_dates": None
            if self.activations is None or package is None
            else self.activations.dates(
                package.strategy_id, book_sessions=execution.formation_sessions
            ),
            "result_hash": result_hash,
            "report_reference_selection": RiskReportLinks.selection(
                task_kind="INSTALLED_STRATEGY_BOOK", completed_portfolio=False
            ),
            "report_hash": report.report_hash,
            "originating_task_id": None if task is None else str(task),
            # Every Task that published this result, the first included (V189).
            "used_by_task_ids": (
                []
                if self.application is None
                else [
                    str(value) for value in self.application.pipeline.tasks_for_result(result_hash)
                ]
            ),
            "readouts": {
                "distinct_names_held": readouts.distinct_names_held,
                "effective_n": readouts.effective_n,
                "one_way_turnover_per_trading_session": (
                    readouts.one_way_turnover_per_trading_session
                ),
                "cumulative_net_wealth": readouts.cumulative_net_wealth,
                "cost_bps_per_side": readouts.cost_bps_per_side,
                "cost_bps_round_trip": readouts.cost_bps_round_trip,
                "platform_one_way_cost_bps": readouts.platform_one_way_cost_bps,
                "aggregate_cap_binding_sessions": readouts.aggregate_cap_binding_sessions,
                "aggregate_cap_binding_names_total": readouts.aggregate_cap_binding_names_total,
                "median_holding_adv20_dollar_volume": (readouts.median_holding_adv20_dollar_volume),
                "industry_attribution_available": readouts.industry_attribution_available,
            },
            "window": {
                "selected_start": report.window_guard.selected_start.isoformat(),
                "selected_end": report.window_guard.selected_end.isoformat(),
                "selected_session_count": report.window_guard.selected_session_count,
                "clamped_to_materialized_path": report.window_guard.clamped_to_materialized_path,
            },
            "schedule": {
                "guard_family": report.schedule_guard.guard_family,
                "tranches": report.schedule_guard.tranches,
            },
            "book": {
                "formation_session": report.window_end_book.formation_session.isoformat(),
                "change_boundary": report.window_end_book.change_boundary,
                "preceding_formation_session": (
                    None
                    if report.window_end_book.preceding_formation_session is None
                    else report.window_end_book.preceding_formation_session.isoformat()
                ),
                "held_count": report.window_end_book.held_count,
                "opened_count": report.window_end_book.opened_count,
                "exited_count": report.window_end_book.exited_count,
                "absolute_weight_change_total": format_book_weight(
                    report.window_end_book.absolute_weight_change_total
                ),
                "positions": [
                    {
                        "listing_id": position.listing_id,
                        "weight": format_book_weight(position.weight),
                        "preceding_weight": format_book_weight(position.preceding_weight),
                        "weight_change_bp": format_book_change(position.weight_change),
                        "disposition": position.disposition,
                    }
                    for position in report.window_end_book.positions
                ],
            },
            "controls": [list(pair) for pair in report.control_receipt.selected],
            "limitations": list(report.limitations),
            "unit_rows": [list(row) for row in report.window_unit_rows],
            "report_unit": report.report_unit,
            # The book this report read goes on to its own review, never the default's (V598).
            "review_selector": selector,
            "next_requests": review_requests(selector),
        }

    def compare(self, left_result_hash: str, right_result_hash: str) -> dict[str, object]:
        """Compare two retained results through the deterministic comparison owner.

        Args:
            left_result_hash: Exact left result.
            right_result_hash: Exact right result.

        Returns:
            Comparison disposition, common ledger, differing controls, task lineage and unit-bearing
            metrics.
        """
        assert self.service is not None
        comparison = self.service.compare(left_result_hash, right_result_hash)
        left_task = self.service.originating_task(left_result_hash)
        right_task = self.service.originating_task(right_result_hash)
        return {
            "kind": "INSTALLED_RESULT_COMPARISON",
            "disposition": comparison.disposition,
            "shares_execution_ledger": comparison.shares_execution_ledger,
            "differing_controls": list(comparison.differing_controls),
            "left": {
                "result_hash": left_result_hash,
                "task_id": None if left_task is None else str(left_task),
                "window": comparison.left.window_guard.model_dump(mode="json"),
                "report_hash": comparison.left.report_hash,
                "controls": [list(value) for value in comparison.left.control_receipt.selected],
            },
            "right": {
                "result_hash": right_result_hash,
                "task_id": None if right_task is None else str(right_task),
                "window": comparison.right.window_guard.model_dump(mode="json"),
                "report_hash": comparison.right.report_hash,
                "controls": [list(value) for value in comparison.right.control_receipt.selected],
            },
            "dimensions": [
                {
                    "dimension": dimension.dimension,
                    "metrics": [
                        {
                            "label": metric.label,
                            "unit": metric.unit,
                            "left": metric.left,
                            "right": metric.right,
                        }
                        for metric in dimension.metrics
                    ],
                }
                for dimension in comparison.dimensions
            ],
        }

    def freeze(self, result_hash: str) -> dict[str, object]:
        """Freeze one exact visible result through the finalization service.

        Args:
            result_hash: Exact result selected for freezing.

        Returns:
            Frozen candidate read view.
        """
        assert self.service is not None
        return _frozen_body(self.service.freeze(result_hash))

    def finalization(self, candidate_hash: str) -> dict[str, object]:
        """Read frozen candidate release state and its next permitted action.

        Args:
            candidate_hash: Exact candidate identity.

        Returns:
            Package, receipt, handoff and release lineage; this projection grants no final-stage
            action.
        """
        assert self.service is not None
        status = self.service.finalization_status(candidate_hash)
        frozen = self.service.frozen_candidate(candidate_hash)
        next_action = {
            "NOT_FROZEN": "FREEZE_A_VISIBLE_DEVELOPMENT_RESULT",
            "AWAITING_PROTECTED_AUTHORITY": "WAIT_FOR_PROTECTED_AUTHORITY",
            "RELEASED": "READ_RELEASED_RESULT",
        }[status.disposition]
        return {
            "candidate": None if frozen is None else _frozen_body(frozen),
            "disposition": status.disposition,
            "detail": status.detail,
            "package_hash": status.package_hash,
            "validation_receipt_hash": status.validation_receipt_hash,
            "handoff_hash": status.handoff_hash,
            "released_result_hash": status.released_result_hash,
            "released_report_hash": status.released_report_hash,
            "closure": status.closure,
            "next_lawful_action": next_action,
            "stage_11_action_available": False,
        }

    def export(self, result_hash: str) -> dict[str, object]:
        # The spec check and the manifest both need this result; opening it once
        # is the difference between four reads and eight.
        """Read the exact result export manifest without recomputing research.

        Args:
            result_hash: Exact retained result identity.

        Returns:
            Decoded export manifest object.

        Raises:
            ValueError: Retained export JSON is not an object.
        """
        assert self.service is not None
        result = self.service.open_result(result_hash)
        task = self.service.originating_task(result_hash)
        manifest = self.service.export_manifest_from(
            result, self._result_spec(result, task), originating_task=task
        )
        document = json.loads(manifest.as_json())
        if not isinstance(document, dict):  # pragma: no cover - owner always emits an object
            raise ValueError("portfolio_application.export_document_invalid")
        return cast(dict[str, object], document)

    # ------------------------------------------------------- evidence & CRO

    def evidence_cro(
        self,
        selector: BookSelector | None,
        publication_hash: str | None = None,
        *,
        evidence_detail: str | None = None,
        view_entity_id: str | None = None,
        view_topic: str | None = None,
        view_last_days: int | None = None,
    ) -> dict[str, object]:
        """The Evidence & CRO section for one sealed book, or the default book."""
        assert self.review is not None
        if evidence_detail is not None and evidence_detail != "time_view":
            raise PortfolioEvidenceReviewError(
                "alternative_evidence.view_filters_not_applicable:" + evidence_detail
            )
        body = evidence_cro_body(
            EvidenceCroProjector(self.review).projection(
                selector,
                publication_hash=publication_hash,
                reading=evidence_detail == "time_view"
                or any(v is not None for v in (view_entity_id, view_topic, view_last_days)),
                entity_id=view_entity_id,
                topic=view_topic,
                last_days=view_last_days,
            )
        )
        if publication_hash is not None:
            body["review_publication_hash"] = publication_hash
        return body

    def _read_review_update(self, task_id: UUID, publication_hash: str) -> dict[str, object]:
        assert self.updates is not None
        task = self.workspace_session.task_control_registry.task(task_id)
        if task.input.input_schema_id == DECISION_ADVANCEMENT_SCHEMA:
            if self.research_updates is None:
                raise ValueError("product_host.evidence_review_update_not_configured")
            body = self.research_updates.readback(task_id, publication_hash=publication_hash)
        elif task.task_kind == PORTFOLIO_UPDATE_TASK_KIND:
            body = self.updates.readback(task_id, publication_hash=publication_hash)
        else:
            raise ValueError("product_host.evidence_review_update_task_mismatch")
        plan = cast(dict[str, object], task.input.payload["plan"])
        if publication_hash == plan.get("parent_hash"):
            value = PortfolioUpdatePublication.model_validate(body["publication"])
            producers = [self.updates.publication_task(value)]
            if self.research_updates is not None:
                producers.append(self.research_updates.publication_task(publication_hash))
            found = [v for v in producers if v is not None]
            if len(found) != 1:
                raise ValueError("product_host.evidence_review_publication_origin_unavailable")
            # A later Task can display this parent; it must not give the exact
            # same publication a new review identity or buy duplicate cognition.
            return self._read_review_update(found[0], publication_hash)
        return body

    def evidence_refresh(self, selector: BookSelector | None) -> dict[str, object]:
        """`Refresh evidence`: one real Alternative Evidence Task, or a typed refusal."""
        assert self.review is not None
        return _review_outcome_body(
            self.review.refresh_evidence(dispatcher=self.dispatcher, selector=selector)
        )

    def evidence_select(
        self,
        selector: BookSelector | None,
        *,
        analysis_publication_hash: str,
        chosen_by: OperationCaller,
    ) -> dict[str, object]:
        """Record which admitted analysis this book's review is read against.

        `chosen_by` has no default: the one caller that could forget to state it
        is the one that would silently record the wrong actor.
        """
        assert self.review is not None
        if chosen_by == "SERVICE_AUTOMATION":
            raise ValueError("research_update.automation_operation_not_admitted")
        book = self.review.default_selector(selector) or BookSelector()
        selection = self.review.select_evidence(
            selector=book,
            analysis_publication_hash=analysis_publication_hash,
            chosen_by=chosen_by,
        )
        return {
            "selection_hash": selection.selection_hash,
            "analysis_publication_hash": selection.analysis_publication_hash,
            "issuer_scope_hash": selection.issuer_scope_hash,
            "evidence_as_of": selection.evidence_as_of.isoformat(),
            "chosen_at": selection.chosen_at.isoformat(),
            "chosen_by": selection.chosen_by,
            # The book the choice was recorded for, and its review read against it (V487).
            "review_selector": book.request_fields(),
            "next_requests": {
                **review_requests(book.request_fields()),
                # This choice already names an admitted analysis. Its exact
                # dossier stays readable independently of further source work.
                "dossier": {"operation": "CRO_REVIEW_DOSSIER", **book.request_fields()},
            },
        }

    def cro_review(self, selector: BookSelector | None) -> dict[str, object]:
        """`Review with CRO`: exact reuse first, then one real Task."""
        assert self.review is not None
        return _review_outcome_body(
            self.review.review(dispatcher=self.dispatcher, selector=selector)
        )

    def _bind_evidence_storage_admission(self) -> None:
        """Every evidence write is admitted by the workspace's own storage budget,
        and a preparation's counts reach the workspace's progress projection.

        The evidence runtime is composed before the storage owner exists; the
        admission is bound here, once both do, so a source set, a canonical
        blob, a vector payload or an index is refused by the same rule that
        refuses a Data or Factor write when the workspace is at its cap.
        """

        adapter = None if self.review is None else self.review.evidence_task_adapter
        if adapter is not None:
            adapter.runtime.storage_admission = self.storage.admit_evidence_bytes
            # A preparation's counts reach the workspace's progress projection, as a
            # data preparation's do (the stage, its count and unit, the unit it is).
            publisher = WorkspaceProgressPublisher(self.workspace_session.workspace / "artifacts")
            adapter.progress.sink = partial(publish_evidence_work, publisher)

    def cpu_budget(self) -> dict[str, object]:
        """Read the declared CPU budget, observed machine and current preparation capacity.

        The CPU budget as set, the machine as found (processors, their load, memory,
        the book preparations running) and what the budget gives a book now; the last
        preparation's receipt. What an operator or agent reads to choose a budget.

        An execution setting: `cpu-budget show` and `set` read and set it (V266); how fast,
        never what (the final close-out, F1).
        """
        store = CpuBudgetStore(self.workspace_session.workspace / "runtime")
        budget = store.read()
        adapter = None if self.review is None else self.review.evidence_task_adapter
        machine = machine_load(
            preparations_running=0 if adapter is None else adapter.preparations_running()
        )
        now = plan_execution(
            budget,
            machine,
            units=MAXIMUM_CPU_BUDGET,
            task_id=None,
            planned_at=self.dispatcher.clock(),
        )
        last = store.last()
        cores, reason = budget_cores(budget, machine)
        last_task, last_fits = store.last_task(), store.last_model_fits()
        queue = read_queue_setting(self.workspace_session.workspace / "runtime")
        places, places_reason = waiting_places(queue)
        tasks = self.workspace_session.task_control_registry.tasks()
        return {
            "status": "CPU_BUDGET",
            **budget.model_dump(mode="json", exclude={"schema_version"}),
            "task_queue": {
                **queue.model_dump(mode="json", exclude={"schema_version"}),
                "places": places,
                "reason": places_reason,
                "waiting_now": sum(task.lifecycle is TaskLifecycle.QUEUED for task in tasks),
            },
            "machine": machine.model_dump(mode="json"),
            "a_book_now": {
                "cores": now.cores,
                "units_at_once_at_most": now.units_at_once,
                "threads_first_unit": now.threads_first_unit,
                "threads_per_session": now.threads_per_session,
                "reason": now.reason,
            },
            "a_task_now": {"cores": cores, "reason": reason},
            "last_preparation": None if last is None else last.model_dump(mode="json"),
            "last_task": None if last_task is None else last_task.model_dump(mode="json"),
            "last_model_fits": None if last_fits is None else last_fits.model_dump(mode="json"),
            "guidance": (
                "auto takes the processors not in use when a preparation or a Task starts. "
                "Before a heavy Task the agent keeps auto unless the machine is busy with other "
                "work or small; then it sets a number of cores itself, without asking, and says "
                "in one line what it set and how to change it. A budget changes how "
                "fast work runs, never what it computes: a book's "
                "model sessions prove the retrieval canary when they load, a Task's DuckDB and "
                "Arrow reads give the same values at any count, and an Alpha study's LightGBM "
                "fits prove their canary at the count they run on. The agent never changes a "
                "study's model, window or bounds to save time. tasks_waiting sets how many "
                "Tasks may wait behind the running one (auto: one place per four processors); a "
                "request past the last place is refused before its planning work, and a Task "
                "waiting for its recovery holds no running place."
            ),
        }

    def set_cpu_budget(
        self, value: object, *, chosen_by: Literal["HUMAN", "EXTERNAL_AUTOMATION"]
    ) -> dict[str, object]:
        """Set the CPU budget used when the next preparation is planned.

        Set the workspace's CPU budget: `auto` or a whole number of cores. The next
        preparation plans with it; one running keeps its plan.
        """
        try:
            CpuBudgetStore(self.workspace_session.workspace / "runtime").write(
                value, chosen_by=chosen_by, chosen_at=self.dispatcher.clock()
            )
        except ValueError as error:
            return {
                "status": "REFUSED",
                **located_failure(error, "execution.cpu_budget_invalid"),
                "next_action": "SET_AUTO_OR_A_WHOLE_NUMBER_OF_CORES",
            }
        return self.cpu_budget()

    def set_tasks_waiting(
        self, value: object, *, chosen_by: Literal["HUMAN", "EXTERNAL_AUTOMATION"]
    ) -> dict[str, object]:
        """Set how many Tasks may wait behind the running one: `auto` or a whole number.

        The next admission counts with it; the Tasks already waiting keep their places (V100).
        """
        try:
            write_queue_setting(
                self.workspace_session.workspace / "runtime",
                value,
                chosen_by=chosen_by,
                chosen_at=self.dispatcher.clock(),
            )
        except ValueError as error:
            return {
                "status": "REFUSED",
                **located_failure(error, "task_control.tasks_waiting_invalid"),
                "next_action": "SET_AUTO_OR_A_WHOLE_NUMBER_OF_TASKS",
            }
        return self.cpu_budget()

    def _model_operation(
        self, request: PortfolioResearchOperationRequest, caller: OperationCaller
    ) -> dict[str, object]:
        """The Alpha models this workspace admits beyond the installed ones (EX).

        Anyone reads the review packets; a person activates or deactivates a model, as a
        person sets network access (V143).
        """
        from alphalattice.control.product_host.research_authoring.model_extensions import (
            ModelExtensions,
        )

        models = ModelExtensions(
            self.workspace_session.workspace,
            self.dispatcher.clock,
            contracts=self.model_contracts,
        )
        try:
            if request.operation == "MODEL_EXTENSIONS":
                return models.review()
            if caller != "HUMAN":
                return refused("model_extension.human_confirmation_required")
            assert request.model_id is not None
            if request.operation == "MODEL_ACTIVATE":
                return models.activate(request.model_id)
            return models.deactivate(request.model_id)
        except (ValueError, OSError) as error:
            failure = located_failure(error, "model_extension.refused")
            return {
                "status": "REFUSED",
                **failure,
                **explain(str(failure["failure_code"])),
            }

    def _feature_extension_operation(
        self, request: PortfolioResearchOperationRequest, caller: OperationCaller
    ) -> dict[str, object]:
        """The formula factors this workspace's daily catalog admits (EX).

        Anyone reads a review packet; a person activates or deactivates a factor, as a person
        activates a model.
        """
        from alphalattice.control.product_host.research_authoring.feature_extensions import (
            FeatureExtensions,
        )

        features = FeatureExtensions(
            self.workspace_session.workspace,
            self.dispatcher.clock,
            trials=self.trials,
            goals=self.goals.store,
            rebind=self._rebind_feature_catalog,
        )
        try:
            if request.operation == "FEATURE_EXTENSIONS":
                return features.listing(page=request.extensions_page)
            assert request.feature_factor_id is not None
            if request.operation == "FEATURE_REVIEW":
                assert request.feature_plan_hash is not None
                return features.review(request.feature_plan_hash, request.feature_factor_id)
            if caller != "HUMAN":
                return refused("feature_extension.human_confirmation_required")
            if request.operation == "FEATURE_ACTIVATE":
                assert request.feature_plan_hash is not None
                return features.activate(request.feature_plan_hash, request.feature_factor_id)
            return features.deactivate(request.feature_factor_id)
        except (ValueError, OSError) as error:
            failure = located_failure(error, "feature_extension.refused")
            return {
                "status": "REFUSED",
                **failure,
                **explain(str(failure["failure_code"])),
            }

    def _rebind_feature_catalog(self) -> None:
        """The workspace's data update binds the catalog its activations make (EX).

        The next update finds its active Panel under another catalog the workspace made and
        rebuilds it under this one; a workspace with no data update has nothing to rebind.
        """
        from alphalattice.control.product_host.maintenance.data_update import (
            installed_data_update_binding,
        )

        workspace = self.workspace_session.workspace
        binding = installed_data_update_binding(workspace)
        _, updated = update_research_workspace_manifest(
            workspace,
            lambda current: (
                current
                if current.data_update is None or current.data_update == binding
                else current.with_bindings(data_update=binding)
            ),
            gate=self.workspace_session.mutation_gate,
        )
        self._hold(updated)

    def _selector_refusal(
        self, code: str, request: PortfolioResearchOperationRequest
    ) -> dict[str, object] | None:
        """A review request's book selector refused, worded with the request that works (V546).

        Parts of a study's selector beside another book do not apply, and the same request
        without them is offered (V290's way, for the study's parts); a selector missing a part,
        or naming two books, says which parts name a book. A study's selector that names a
        Task of another kind -- a public development replay, whose book is its published
        result -- offers the request by the receipt it named, where that receipt opens a book.

        Args:
            code: The refusal's code.
            request: The refused request.

        Returns:
            The worded refusal; None for a refusal that is not the selector's.
        """
        base, _sep, subject = code.partition(":")
        if base == "product_host.evidence_review_experiment_selector_invalid" and subject:
            return refused(code, book=_request_fields(request, *subject.split(",")))
        if base == "product_host.evidence_review_study_selector_not_a_study":
            receipt = request.experiment_receipt_hash
            book: dict[str, str] | None = None
            if receipt is not None and self.review is not None:
                try:
                    self.review.resolve_book(BookSelector(result_hash=receipt))
                except (PortfolioEvidenceReviewError, ValueError, KeyError):
                    book = None
                else:
                    study = ("experiment_task_id", "experiment_receipt_hash", "portfolio_session")
                    book = {**_request_fields(request, *study), "result_hash": receipt}
            return refused(code, book=book, task_id=str(request.experiment_task_id), kind=subject)
        if base in SELECTOR_CODES:
            return refused(code)
        return None

    def _backup_operation(
        self, request: PortfolioResearchOperationRequest, *, make: bool = True
    ) -> dict[str, object]:
        """Back up the held state outside the workspace (V209), or read what is kept.

        `WORKSPACE_BACKUPS` reads the root and the kept generations and makes none, so the
        Storage page can show them before a person chooses "back up now" (V337, U48).
        """
        workspace = self.workspace_session.workspace
        try:
            backups = WorkspaceBackups(
                workspace,
                workspace_id=read_research_workspace_manifest(workspace).workspace_id,
                clock=self.dispatcher.clock,
            )
            generation = (
                backups.create(
                    reason="REQUEST",
                    generations_kept=request.backup_generations_kept or GENERATIONS_KEPT,
                )
                if make
                else None
            )
        except (WorkspaceBackupError, ResearchWorkspaceError, OSError) as error:
            return {"status": "REFUSED", **located_failure(error, "workspace_backup.refused")}
        return backup_answer(backups, generation)

    def _client_operation(
        self, request: PortfolioResearchOperationRequest, *, caller: OperationCaller
    ) -> dict[str, object] | None:
        """The client's own commands, answered as operations like any other (V266, OP1).

        The workspace, the operations, the activity feed and its event ingress, the CPU budget;
        None for any other operation.
        """
        operation = request.operation
        if operation in {"WORKSPACE_SHOW", "OPERATION_LIST"}:
            return self.client_session(request, caller=caller)
        if operation == "CPU_BUDGET_SHOW":
            return self.cpu_budget()
        if operation == "USAGE_READING":
            return usage_reading_answer(self.workspace_session.workspace)
        if operation == "USAGE_READING_SET":
            # The person's privacy choice: an Agent reads it and never turns reading back on.
            if caller != "HUMAN":
                return refused("native_bridge.usage_reading_human_only")
            assert request.usage_reading_enabled is not None
            return set_usage_reading(
                self.workspace_session.workspace, enabled=request.usage_reading_enabled
            )
        if operation == "SESSION_USAGE_READ":
            # Optional reading: a missing reader or a failed read never refuses research.
            if self.read_native_usage is None:
                return {"status": "UNAVAILABLE", "reason": "native_bridge.lead_usage_read_failed"}
            return self.read_native_usage()
        if operation in {"WORKSPACE_BACKUP", "WORKSPACE_BACKUPS"}:
            return self._backup_operation(request, make=operation == "WORKSPACE_BACKUP")
        if operation == "CPU_BUDGET_SET":
            chosen_by: Literal["HUMAN", "EXTERNAL_AUTOMATION"] = (
                "HUMAN" if caller == "HUMAN" else "EXTERNAL_AUTOMATION"
            )
            # One setting a request (V100): the budget or the Tasks that may wait.
            if (request.cpu_budget is None) == (request.tasks_waiting is None):
                return refused("execution.cpu_budget_invalid")
            if request.tasks_waiting is not None:
                return self.set_tasks_waiting(request.tasks_waiting, chosen_by=chosen_by)
            return self.set_cpu_budget(request.cpu_budget, chosen_by=chosen_by)
        if operation == "WAKE_REGISTER":
            # A Codex turn ends its shell's children, so the Host holds the lead's wake in the
            # Task's journal and the activity sends it (R1, WAKE).
            assert request.task_id is not None
            assert request.wake_thread is not None and request.wake_read is not None
            try:
                wake = self.workspace_session.task_control_registry.register_wake(
                    request.task_id,
                    request.wake_thread,
                    request.wake_read,
                    observed_at=self.dispatcher.clock(),
                )
            except TaskNotFoundError:
                return refused("task_control.task_not_found")
            return {"status": "WAKE_REGISTERED", "task_id": str(request.task_id), "wake": wake}
        if operation not in {"ACTIVITY_LIST", "ACTIVITY_RECENT", "EVENT_DECLARE"}:
            return None
        observer = self.observer
        if observer is None:
            return refused("activity.storage_unavailable")
        if operation == "EVENT_DECLARE":
            return self.declare_event(observer, request.event)
        try:
            if operation == "ACTIVITY_RECENT":
                return observer.recent(limit=20 if request.limit is None else request.limit)
            # The feed's own parser reads the page as its query names it.
            query: dict[str, list[str]] = {}
            if request.after is not None:
                query["after"] = [request.after]
            if request.limit is not None:
                query["limit"] = [str(request.limit)]
            if request.watch:
                query["watch"] = [",".join(str(task) for task in request.watch)]
            return observer.read(ActivityReadQuery.from_query(query))
        except TaskQueueHeadAuthorityError as error:
            return self.task_queue_authority_refusal(error)
        except ValueError as error:
            return refused(public_failure(error, "activity.read_refused"))

    def client_session(
        self, request: PortfolioResearchOperationRequest, *, caller: OperationCaller
    ) -> dict[str, object]:
        """The workspace as a client reads it (`workspace show`), or with its operations.

        `operation list` adds every operation's fields and notes, and a named package's controls.
        """
        body = self.session_projection(
            session_token="",
            include_context=request.operation == "WORKSPACE_SHOW",
            caller=caller,
        )
        body.pop("session_token", None)
        body["caller"] = caller
        body["permission_notes"] = [
            "Contracts describe requests, not grants; inspect owner readiness before work.",
            (
                "Data decisions accept an exact local Human grant for EXTERNAL_AUTOMATION; "
                "other Human confirmations remain Human-only."
            ),
            "External automation cannot change background update settings.",
            (
                "Each operation marked person_only is completed by a person, in the "
                "Workbench; a client's or an Agent's request for it is refused."
            ),
            (
                "Each operation marked first_use is the person's too, and the agent running the "
                "workspace's first-use goal, opened from the person's sentence, completes it for "
                "the person while that goal is open and within its hours."
            ),
        ]
        if request.operation == "WORKSPACE_SHOW":
            # The first read is the path: every standard flow on each input, what it needs,
            # holds and asks next, each request ready to send (V376).
            body["intents"] = [*self._forward_intents(), *self.experiments.intents()]
            return body
        contracts: dict[str, dict[str, object]] = {}
        for operation in OPERATIONS:
            required, allowed = PortfolioResearchOperationRequest.field_contract(operation)  # type: ignore[arg-type]
            contracts[operation] = {"required": sorted(required), "allowed": sorted(allowed)}
            if operation in PERSON_ONLY:  # a person completes it, in the Workbench (V143)
                contracts[operation]["person_only"] = True
        body["operation_schema"] = PortfolioResearchRequestDocument.model_json_schema()
        body["operation_fields"] = contracts
        if request.strategy_package_id is not None:
            body["package_controls"] = self._execute(
                PortfolioResearchOperationRequest(
                    operation="CONTROLS", strategy_package_id=request.strategy_package_id
                ),
                caller=caller,
                agent_execution=None,
            )
        return body

    def declare_event(
        self,
        observer: OperationObserver,
        event: dict[str, Any] | None,
        *,
        accepted_receipt: GoalAcceptedAnswerReceipt | None = None,
    ) -> dict[str, object]:
        """Record one event a client declares about its own work (`event declare`).

        Team's record and the goal's timeline are one record (GR2, OP13): the event keeps
        the goal its session holds, and an admitted one joins that goal's record.
        """
        try:
            document = ExternalActivityEventDocument.model_validate(event)
        except ValidationError as error:
            return {
                "status": "REFUSED",
                **located_failure(error, "activity.event_invalid"),
                "next_action": "READ_ACTIVITY_EVENT_CONTRACT",
            }
        with self.workspace_session.mutation_gate.hold(), self.goals.store.lock:
            return self._file_declared_event(observer, document, accepted_receipt=accepted_receipt)

    def _file_declared_event(
        self,
        observer: OperationObserver,
        document: ExternalActivityEventDocument,
        *,
        accepted_receipt: GoalAcceptedAnswerReceipt | None,
    ) -> dict[str, object]:
        """File and attribute one admitted event while the existing owner locks are held."""
        try:
            filed, goal, session = self.goals.file_event(
                document, REQUEST_PROVENANCE.get(), accepted_receipt=accepted_receipt
            )
        except ValueError as error:
            return {
                "status": "REFUSED",
                "failure_code": safe_failure_code(str(error)) or "goal.event_not_filed",
                "next_action": "NAME_AN_OPEN_GOAL_OR_NONE",
            }
        answer = observer.admit_external_event(filed)
        if goal is None or answer.get("status") not in {"APPENDED", "REUSED_EXACT"}:
            return answer
        if answer["status"] == "REUSED_EXACT":
            original = next(
                (
                    entry
                    for entry in self.goals.store.attributed(goal.goal_id)
                    if entry.get("observation_id") == answer["observation_id"]
                ),
                None,
            )
            if original is not None:
                return {**answer, "goal_id": str(goal.goal_id)}
        # The ledger can have appended before a failed Goal attribution. A verified
        # exact retry repairs that missing projection once, without another event.
        return {
            **answer,
            **self.goals.record_event(goal, filed, str(answer["observation_id"]), session),
        }

    def _evidence_storage_binding(self) -> EvidenceStorageBinding | None:
        """The evidence store's roots and its owner's own evict/rebuild, for storage.

        The roots follow the Host's composition of the evidence runtime; the
        index owner's actions are bound only when a runtime is admitted, so
        without one the bytes are still accounted and nothing is released.
        """

        adapter = None if self.review is None else self.review.evidence_task_adapter
        if adapter is None:
            workspace = self.workspace_session.workspace
            knowledge_root = workspace / "runtime" / "evidence-knowledge"
            artifact_root = workspace / "runtime" / "artifacts" / "alternative-evidence"
            if not knowledge_root.is_dir() and not artifact_root.is_dir():
                return None
            return EvidenceStorageBinding(
                knowledge_root=knowledge_root, artifact_root=artifact_root
            )
        runtime = adapter.runtime
        return EvidenceStorageBinding(
            knowledge_root=runtime.retrieval.workspace.root,
            artifact_root=runtime.artifacts.root,
            active_index_ids=runtime.retrieval.active_index_ids,
            evict=lambda relative, plan_hash, evicted_at: runtime.retrieval.evict_generation(
                relative, plan_hash=plan_hash, evicted_at=evicted_at
            ),
            rebuild=runtime.rebuild_retrieval,
        )

    def _spec(self, document: dict[str, object]) -> PortfolioResearchSpec:
        request = dict(document)
        explicit_strategy = request.get("strategy_package_id")
        strategy_package_id = (
            self.workspace_manifest.default_strategy_package_id
            if explicit_strategy is None
            else str(explicit_strategy)
        )
        package = self._package(strategy_package_id)
        for fixed in package.controls.frozen:
            request.setdefault(fixed.control_id, fixed.frozen_display)
        default_mode = (
            self.workspace_manifest.default_score_source_mode
            if explicit_strategy is None
            else package.default_score_source_mode
        )
        if default_mode is None:
            default_mode = package.default_score_source_mode
        return spec_from_document(
            request,
            default_strategy_package_id=package.strategy_id,
            default_score_source_mode=default_mode,
        )

    def _package(self, strategy_package_id: str | None) -> FrozenStrategyPackage:
        if strategy_package_id is None:
            raise LocalApplicationError(
                "strategy_book.strategy_package_required"
                if self.installed()
                else "research_workspace.strategy_not_installed"
            )
        package = self._packages.get(strategy_package_id)
        if package is None:
            raise LocalApplicationError(
                f"local_application.strategy_package_not_installed:{strategy_package_id}"
            )
        return package

    def _package_controls(
        self,
        package: FrozenStrategyPackage,
        *,
        spec: PortfolioResearchSpec | None = None,
        eligible_count: int | None = None,
        support: tuple[str, str] | None = None,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        assert self.service is not None
        rows: list[dict[str, object]] = []
        frozen: list[dict[str, object]] = []
        surface = package.controls
        for control in self.service.controls.controls:
            fixed = surface.frozen_control(control.control_id)
            if fixed is not None:
                frozen.append(
                    {
                        "control_id": fixed.control_id,
                        "label": control.label,
                        "unit": control.unit,
                        "frozen_display": fixed.frozen_display,
                        "help": control.help,
                        "guard": control.guard,
                        "refusal": fixed.refusal_code,
                        "reason": fixed.reason,
                    }
                )
                continue
            minimum: str | None = None
            maximum: str | None = None
            if control.control_id == "exit_rank" and spec is not None:
                floor, ceiling = spec.exit_rank_band()
                minimum = str(floor)
                maximum = str(min(ceiling, eligible_count or ceiling))
            elif control.control_id in {"study_start", "study_end"} and support is not None:
                minimum, maximum = support
            rows.append(
                public_control_document(
                    control,
                    scope=(
                        "SHARED" if control.control_id in surface.shared else "PACKAGE_ADMITTED"
                    ),
                    minimum=minimum,
                    maximum=maximum,
                )
            )
        return rows, frozen

    def _result_spec(
        self, result: PortfolioResearchResult, task_id: UUID | None
    ) -> PortfolioResearchSpec:
        """The spec this result was admitted under, from its own durable task.

        Takes the opened result rather than a hash: the caller has it, and
        re-opening it here to compare one field was a read for nothing. The
        comparison itself is unchanged and still refuses a result whose task
        names a different configuration.
        """

        if task_id is None:
            raise LocalApplicationError("local_application.result_has_no_originating_task")
        task = self.workspace_session.task_control_registry.task(task_id)
        durable = portfolio_research_task_input(task)
        if result.spec_hash != durable.spec.spec_hash:
            raise LocalApplicationError("local_application.result_task_spec_mismatch")
        return durable.spec


def goal_refusal(error: Exception, request: PortfolioResearchOperationRequest) -> dict[str, object]:
    """A goal operation's refusal, in words where the Host has them.

    A closed goal names its standing, read by the goal the request named or the session's own
    (V391), beside the follow-up's start: the refusal stays, and the record stays as sealed
    (V437).

    Args:
        error: What the goal owner raised.
        request: The request it refused.

    Returns:
        The refusal, its words and the requests that go on from it.
    """
    failure = located_failure(error, "goal.refused")
    code = str(failure["failure_code"])
    answer: dict[str, object] = {"status": "REFUSED", **failure, **explain(code)}
    if code.partition(":")[0] == "goal.closed_open_a_follow_up":
        named = {
            name: str(value)
            for name, value in (("goal_id", request.goal_id), ("goal_hash", request.goal_hash))
            if value is not None
        }
        offered = cast(dict[str, object], answer.get("next_requests") or {})
        answer["next_requests"] = {"show": {"operation": "GOAL_SHOW", **named}, **offered}
    elif code == "goal.reference_not_found" and request.goal_hash is not None:
        answer["next_requests"] = {
            "narrative": {"operation": "GOAL_NARRATIVE", "goal_hash": request.goal_hash}
        }
    elif code == "goal.revision_conflict_read_latest" and request.goal_id is not None:
        # The goal's latest revision, by its id alone: the request's hash names the revision
        # another session replaced, and the session's own goal may be another (V527).
        offered = cast(dict[str, object], answer.get("next_requests") or {})
        answer["next_requests"] = {
            "show": {"operation": "GOAL_SHOW", "goal_id": str(request.goal_id)},
            **offered,
        }
    return answer


def reused_read(
    registry: DuckDbTaskControlRegistry, plan_hash: str, read: str
) -> dict[str, object]:
    """An exact reuse's Task and its read: the succeeded Task that ran the plan reused.

    So a read continued from the reuse reads what it reused, never another latest Task (V491,
    as the data update's reuse does since V449); nothing when no retained Task ran the plan, as
    for a score an admitted publication supplied.
    """
    for task in reversed(registry.tasks()):
        stored = task.input.payload.get("plan")
        if (
            task.lifecycle is TaskLifecycle.SUCCEEDED
            and isinstance(stored, dict)
            and stored.get("plan_hash") == plan_hash
        ):
            return {
                "publication_task_id": str(task.task_id),
                "next_requests": {"read": {"operation": read, "task_id": str(task.task_id)}},
            }
    return {}


_STRATEGY_READS: Final[dict[str, StrategyRead]] = {
    "RESEARCH_UPDATE_READBACK": StrategyRead(
        "research_update",
        "package_id",
        "NO_RESEARCH_UPDATE",
        lambda task: task.input.input_schema_id == DECISION_ADVANCEMENT_SCHEMA,
    ),
    "PORTFOLIO_UPDATE_READBACK": StrategyRead(
        "portfolio_update",
        "strategy_package_id",
        "NO_PORTFOLIO_UPDATE",
        lambda task: task.task_kind == PORTFOLIO_UPDATE_TASK_KIND,
    ),
    "STRATEGY_SCORE_READBACK": StrategyRead(
        "strategy_score",
        "binding.strategy_package_id",
        "NO_SCORE_PUBLICATION",
        lambda task: task.task_kind == STRATEGY_SCORE_TASK_KIND,
    ),
    "STRATEGY_CALIBRATION_READBACK": StrategyRead(
        "strategy_calibration",
        "binding.strategy_package_id",
        "NO_CALIBRATION_PUBLICATION",
        lambda task: task.task_kind == STRATEGY_CALIBRATION_TASK_KIND,
    ),
}
"""Each readback of work planned per strategy, read by its Task or by a strategy's latest, never
another strategy's (V595): the one place a reader of "the latest" of such work picks it."""


def _latest_update(task: TaskRecord | None) -> dict[str, object] | None:
    """A strategy's latest research update Task, its read bound to it; none before its first."""
    if task is None:
        return None
    return {
        "task_id": str(task.task_id),
        "lifecycle": task.lifecycle.value,
        "target_session": planned(task, "target"),
        "next_requests": {
            "readback": {"operation": "RESEARCH_UPDATE_READBACK", "task_id": str(task.task_id)}
        },
    }


def _position_rows(body: dict[str, object]) -> dict[str, object]:
    publication = body.get("publication")
    if isinstance(publication, dict):
        value = PortfolioUpdatePublication.model_validate(publication)
        history = tuple(
            PortfolioUpdatePublication.model_validate(v)
            for v in cast(list[object], body.get("history", []))
        )
        positions = portfolio_update_positions(value, history)
        weights, changes = positions.weights, positions.changes
        previous = positions.preceding or weights
        labels = cast(dict[str, str], body["listing_labels"])
        body["position_rows"] = [
            {
                "listing_id": listing,
                "name": label,
                "weight": format_book_weight(weights[i]),
                "change": "Not an estimate" if changes is None else format_book_change(changes[i]),
                "basis": "Close estimate; conditional execution"
                if positions.basis == "CONDITIONAL_ESTIMATE"
                else "Observed research entry",
            }
            for i, (listing, label) in enumerate(labels.items())
            if weights[i] > 0 or previous[i] > 0
        ]
        body["review_selector"] = {
            "update_task_id": body["task_id"],
            "update_publication_hash": value.content_hash,
            "position_basis": positions.basis,
        }
        # The day's positions go on to their own review, never the default book's (V483).
        body["next_requests"] = {
            **cast(dict[str, object], body.get("next_requests") or {}),
            **review_requests(cast(dict[str, str], body["review_selector"])),
        }
    return body


def _file_stamp(path: Path) -> tuple[int, int] | None:
    """A file's modification time and size, as its own record of a rewrite; none when unread."""
    try:
        status = path.stat()
    except OSError:
        return None
    return status.st_mtime_ns, status.st_size


def _request_fields(request: PortfolioResearchOperationRequest, *left: str) -> dict[str, str]:
    """A request's plain fields as its document writes them, `left` left out (V290)."""
    return {
        item.name: str(value)
        for item in fields(request)
        if item.name not in left
        and isinstance(value := getattr(request, item.name), str | int | float | UUID)
        and not isinstance(value, bool)
    }


def _opens_goal(operation: str, provenance: RequestProvenance | None) -> bool:
    """Whether this request is an identified Session's research request that opens a goal."""
    return (
        provenance is not None
        and provenance.vendor in HOSTS
        and bool(provenance.session)
        and provenance.goal_id is None
        and not operation.startswith("GOAL_")
        and observed_operation(operation)
        and operation not in {"EVENT_DECLARE", "SESSION_USAGE_READ", "USAGE_READING_SET"}
    )


def _portfolio_source(request: PortfolioResearchOperationRequest) -> dict[str, str | None]:
    """The Alpha study and candidate a refused Portfolio declaration named (V322)."""
    try:
        document = (
            load_authoring_document(request.experiment_yaml)
            if request.experiment_yaml is not None
            else request.experiment_document
        )
    except ValueError:
        document = None
    section = document.get("portfolio") if isinstance(document, Mapping) else None
    if not isinstance(section, Mapping):
        return {}
    return {
        "task_id": _text(section.get("alpha_task_id")),
        "candidate_id": _text(section.get("candidate_id")),
    }


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _runner_retry_reason(task: TaskRecord) -> str | None:
    """Why a Task the runner itself blocked may be reopened, or nothing (V475, PERF-1)."""
    if task.lifecycle is not TaskLifecycle.BLOCKED:
        return None
    if task.failure_code == MEMORY_INSUFFICIENT:
        return (
            "The stage did not start: its estimated peak memory exceeded what the machine "
            "had available. Once running Tasks have ended or memory is free, RECOVER with "
            "this version reopens the same Task: its verified stages are kept and the memory "
            "check runs again before the stage."
        )
    if not str(task.failure_code or "").startswith("TASK_STAGE_RAISED:"):
        return None
    return (
        "The stage raised the same unnamed error on three attempts in a row; its type and "
        "message are beside the code. Once its cause is fixed, RECOVER with this version "
        "reopens the same Task: its verified stages are kept and the stage runs again."
    )


_CARRIED_TARGETS: Final = ("strategy_package_id",)
_CARRIED_BOOK: Final = (
    "result_hash",
    "handoff_hash",
    "update_task_id",
    "update_publication_hash",
    "experiment_task_id",
    "experiment_receipt_hash",
    "position_basis",
    "portfolio_session",
)


def _carried(
    body: dict[str, object], request: PortfolioResearchOperationRequest
) -> dict[str, object]:
    """A refusal's way on carries the target its request named (V474).

    The refusal knows what was asked for: a package, a book. A next request it offers that
    takes the same target and names none is bound to it, so a client following it reads that
    package or book, never the workspace's default. A next request naming its own target keeps
    it, and a book's fields travel together or not at all.
    """
    offered = body.get("next_requests")
    if not is_refusal(body) or not isinstance(offered, dict) or not offered:
        return body
    asked = {
        name: value if isinstance(value, str) else str(value)
        for name in (*_CARRIED_TARGETS, *_CARRIED_BOOK)
        if (value := getattr(request, name, None)) is not None
    }
    target = {k: asked[k] for k in _CARRIED_TARGETS if k in asked}
    book = {k: asked[k] for k in _CARRIED_BOOK if k in asked}
    changed: dict[str, object] = {}
    for name, offer in offered.items():
        if not isinstance(offer, dict) or not isinstance(offer.get("operation"), str):
            continue
        try:
            _required, allowed = PortfolioResearchOperationRequest.field_contract(
                offer["operation"]
            )
        except (KeyError, ValueError):
            continue
        add = {k: v for k, v in target.items() if k in allowed and k not in offer}
        if book and not set(_CARRIED_BOOK) & set(offer) and set(book) <= allowed:
            add.update(book)
        if add:
            changed[name] = {**offer, **add}
    return {**body, "next_requests": {**offered, **changed}} if changed else body


def _bundle_book(submission: Mapping[str, Any]) -> dict[str, Any]:
    """The book a bundle's submission names, by its selector's fields."""
    return {
        name: value
        for name, value in submission.items()
        if name in {item.name for item in fields(BookSelector)} and value is not None
    }


def _selector(request: PortfolioResearchOperationRequest) -> BookSelector | None:
    if all(
        v is None
        for v in (
            request.result_hash,
            request.handoff_hash,
            request.update_task_id,
            request.update_publication_hash,
            request.position_basis,
            request.experiment_task_id,
            request.experiment_receipt_hash,
            request.portfolio_session,
        )
    ):
        return None
    return BookSelector(
        result_hash=request.result_hash,
        handoff_hash=request.handoff_hash,
        update_task_id=request.update_task_id,
        update_publication_hash=request.update_publication_hash,
        position_basis=request.position_basis,
        experiment_task_id=request.experiment_task_id,
        experiment_receipt_hash=request.experiment_receipt_hash,
        portfolio_session=request.portfolio_session,
    )


_CITATIONS_PER_PAGE = 20


def _pages_citations(request: PortfolioResearchOperationRequest) -> bool:
    return any(
        value is not None
        for value in (
            request.citation_entity_id,
            request.citation_unit_id,
            request.citation_page,
        )
    )


def _citation_page(
    items: list[dict[str, object]],
    request: PortfolioResearchOperationRequest,
    *,
    groups: dict[str, tuple[str, ...]],
    key: str,
) -> dict[str, object]:
    """One page of cited items for one issuer, one group or all of them,
    twenty a page, with the totals; a group of a book of one unit is the
    whole book. Refused by name: both filters at once, an unknown group, a
    page past the last."""

    if request.citation_entity_id is not None and request.citation_unit_id is not None:
        raise ValueError("portfolio_research.citation_filter_ambiguous")
    if request.citation_unit_id is not None:
        if groups and request.citation_unit_id not in groups:
            raise ValueError(
                f"portfolio_research.citation_group_unknown:{request.citation_unit_id}"
            )
        members = set(groups.get(request.citation_unit_id, ()))
        chosen = [item for item in items if not groups or item.get("entity_id") in members]
    elif request.citation_entity_id is not None:
        chosen = [item for item in items if item.get("entity_id") == request.citation_entity_id]
    else:
        chosen = list(items)
    page_count = max(1, -(-len(chosen) // _CITATIONS_PER_PAGE))
    page = 1 if request.citation_page is None else request.citation_page
    if not 1 <= page <= page_count:
        raise ValueError(f"portfolio_research.citation_page_out_of_range:{page} of {page_count}")
    return {
        key: chosen[(page - 1) * _CITATIONS_PER_PAGE : page * _CITATIONS_PER_PAGE],
        "page": page,
        "page_count": page_count,
        "per_page": _CITATIONS_PER_PAGE,
        "total": len(chosen),
        "total_unfiltered": len(items),
        "filter": {
            "entity_id": request.citation_entity_id,
            "unit_id": request.citation_unit_id,
        },
    }


def _with_trial(readback: dict[str, Any]) -> dict[str, Any]:
    """A planned feature's readback with its trial offered, the study left to choose (V354).

    What a trial runs against is stated before any build, so an agent holding only Factor
    studies learns it before it builds rather than at the trial's refusal.
    """
    return {
        **readback,
        "trial_baseline": (
            "A completed Alpha study handed off from a Factor study on this input, or the "
            "Portfolio study built on one."
        ),
        "next_requests": {
            **readback["next_requests"],
            "trial": {
                "operation": "FEATURE_TRIAL",
                "feature_plan_hash": readback["plan_hash"],
                "task_id": None,
            },
        },
    }


def _review_outcome_body(outcome: ReviewOutcome | dict[str, object]) -> dict[str, object]:
    """One JSON shape for both evidence actions, including their refusals.

    `task_id` is present exactly when a Task was admitted, so a reader can tell
    an admitted refresh from a reuse or a refusal without parsing the detail.
    """

    if isinstance(outcome, dict):
        return outcome
    body: dict[str, object] = {"disposition": outcome.disposition, "detail": outcome.detail}
    if outcome.task_id is not None:
        body["task_id"] = str(outcome.task_id)
    if outcome.lifecycle is not None:
        body["lifecycle"] = outcome.lifecycle
    if outcome.review is not None:
        body["review_key"] = outcome.review.publication.review_key
        body["review_publication_hash"] = outcome.review.publication.publication_hash
    if outcome.evidence_as_of is not None:
        body["evidence_as_of"] = outcome.evidence_as_of.isoformat()
    if outcome.failure_code is not None:
        body["failure_code"] = outcome.failure_code
    if outcome.next_requests:
        body["next_requests"] = dict(outcome.next_requests)
    if outcome.evidence_unit_id is not None:
        body["evidence_unit_id"] = outcome.evidence_unit_id
    if outcome.answer is not None:
        body["answer"] = dict(outcome.answer)
    if outcome.network_access is not None:
        body["network_access"] = dict(outcome.network_access)
    if outcome.source_network_access is not None:
        body["source_network_access"] = dict(outcome.source_network_access)
    if outcome.source_ways is not None:
        body["source_ways"] = dict(outcome.source_ways)
    return body


_FOLLOW_READ_SECONDS = 0.1
"""How often a waiting status reads the Task again."""

_TASK_PAGE_ROWS = 50
"""How many Tasks one page of an agent session's listing holds at most, and by default (U54)."""

_LIVE_LIFECYCLES = frozenset(
    {
        TaskLifecycle.QUEUED,
        TaskLifecycle.RUNNING,
        TaskLifecycle.DEFERRED,
        TaskLifecycle.CANCEL_REQUESTED,
    }
)
_PREPARATION_IN_FLIGHT = frozenset(
    {
        TaskLifecycle.QUEUED,
        TaskLifecycle.RUNNING,
        TaskLifecycle.CANCEL_REQUESTED,
    }
)
"""A preparation Task still on its way: its plan confirmed again answers it, never another
(V430: the check named `IN_PROGRESS`, which no Task holds, and missed every running one). A
deferred one is not on its way: its plan confirmed again goes to its owner, which refuses it
before the retry time and resumes the same Task after (V375; answered here as in flight,
the offered `resume` never reached the owner)."""
_FINISHED_LIFECYCLES = frozenset({TaskLifecycle.SUCCEEDED, TaskLifecycle.CANCELLED})
_PORTFOLIO_TASK_KINDS = frozenset(
    {
        PORTFOLIO_RUN_COMMAND,
        STRATEGY_SCORE_TASK_KIND,
        STRATEGY_CALIBRATION_TASK_KIND,
        PORTFOLIO_UPDATE_TASK_KIND,
    }
)
"""The Task kinds that read the installed packages while they run."""
"""A Task done with: its readback answers for it, not the Guardian read (GY)."""


def task_status_body(projection: TaskSafeProjection) -> dict[str, object]:
    """The one JSON shape of a Task Control projection every reader receives."""
    return {
        "task_id": str(projection.task_id),
        "task_kind": projection.task_kind,
        "lifecycle": projection.lifecycle.value,
        "goal_summary": projection.goal_summary,
        "current_stage": projection.current_stage,
        "verified_stage_count": projection.verified_stage_count,
        "total_stage_count": projection.total_stage_count,
        "running_since": (
            projection.running_since.isoformat() if projection.running_since is not None else None
        ),
        "last_activity_at": projection.last_activity_at.isoformat(),
        "cancel_available": projection.cancel_available,
        "cancel_pending": projection.cancel_pending,
        "queued_next_task_id": (
            None if projection.queued_next_task_id is None else str(projection.queued_next_task_id)
        ),
        "latest_failure_code": projection.latest_failure_code,
    }


def _frozen_body(frozen: FrozenCandidateProjection) -> dict[str, object]:
    return {
        "candidate_hash": frozen.candidate_hash,
        "workspace_id": frozen.workspace_id,
        "development_result_hash": frozen.development_result_hash,
        "development_task_id": frozen.development_task_id,
        "development_run_hash": frozen.development_run_hash,
        "program_hash": frozen.program_hash,
        "spec_hash": frozen.spec_hash,
        "control_receipt_hash": frozen.control_receipt_hash,
        "pre_protected_state_hash": frozen.pre_protected_state_hash,
        "last_formation_session": frozen.last_formation_session.isoformat(),
        "formation_count": frozen.formation_count,
        "frozen_at": frozen.frozen_at.isoformat(),
    }


__all__ = [
    "PORTFOLIO_RUN_COMMAND",
    "PortfolioResearchOperations",
    "PortfolioRunCommand",
    "task_status_body",
]
