"""Compose the local Web session, applications, dispatcher and routes.

Task Control owns lifecycle; Portfolio owns numbers. This composition retains
one workspace session and Task registry. Reports serve the executor's sealed
pages; Evidence and CRO share the browser and generic Agent operation owner.
"""

from __future__ import annotations

import json
import secrets
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from functools import partial
from http import HTTPStatus
from pathlib import Path
from threading import Event, Lock, Thread
from typing import TYPE_CHECKING, Any, Final, cast, get_args
from urllib.parse import urlencode
from uuid import UUID

from pydantic import TypeAdapter, ValidationError

from alphalattice.control.product_host.composition.application_session import (
    WorkspaceApplicationSession,
)
from alphalattice.control.product_host.composition.first_start import (
    FirstStart,
    refusal_reason,
)
from alphalattice.control.product_host.composition.goal_case_routes import (
    CASE_ROUTES,
    case_answer,
    case_request,
)
from alphalattice.control.product_host.composition.local_web_connection import (
    publish_client_connection,
    remove_client_connection,
    workspace_web,
)
from alphalattice.control.product_host.composition.plain_refusals import refused
from alphalattice.control.product_host.composition.reference_prefixes import ReferenceLedger
from alphalattice.control.product_host.composition.research_workspace import (
    ResearchWorkspaceEvidenceReview,
    ResearchWorkspaceManifest,
    ResearchWorkspaceManifestHolder,
    admit_research_workspace,
    initialize_research_workspace,
    manifest_fields_hash,
    named_workspace_id,
    read_research_workspace_manifest,
)
from alphalattice.control.product_host.composition.task_kinds import (
    EVIDENCE_INSTALL_TASK_KIND,
    EVIDENCE_REVIEW_RUNTIME_FILE,
    PORTFOLIO_UPDATE_TASK_KIND,
    STRATEGY_CALIBRATION_TASK_KIND,
    STRATEGY_SCORE_TASK_KIND,
    STUDY_VERIFICATION_TASK_KIND,
)
from alphalattice.control.product_host.composition.task_kinds import (
    RESEARCH_EXPERIMENT_TASK_KIND as STUDY_TASK_KIND,
)
from alphalattice.control.product_host.maintenance.signals import (
    MaintenanceBackgroundHost,
    MaintenanceWakeController,
)
from alphalattice.control.task_control.contracts import TaskLifecycle, TaskRecord
from alphalattice.control.task_control.queue import read_queue_setting, running_places
from alphalattice.control.task_control.registry import (
    TaskQueueFull,
    TaskTransitionRejected,
    TaskVersionStale,
)
from alphalattice.control.task_control.runner import TaskHeartbeatReader
from alphalattice.control.workspace_runtime.content_store import verified_model_read_scope
from alphalattice.control.workspace_runtime.database import (
    WORKSPACE_MARKET_DATA_DATABASE_FILENAME,
    keep_workspace_databases,
)
from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.evidence.alternative_evidence.sources.admission import (
    OfficialSourceAdmission,
    admit_workspace_source,
)
from alphalattice.interface.local_application import native_setup
from alphalattice.interface.local_application.activity import (
    READ_OPERATIONS,
    ActivityReadQuery,
    ExternalActivityReadQuery,
    refusal_code,
    returned_status,
)
from alphalattice.interface.local_application.call_span import CALL_PATH
from alphalattice.interface.local_application.cli_contract import (
    REQUEST_PROVENANCE,
    SECTOR_DEVELOPMENT_PUBLISH_PATH,
    RequestProvenance,
    agent_session,
    refusal_words,
)
from alphalattice.interface.local_application.cli_invocation import invoke as invoke_cli
from alphalattice.interface.local_application.client import (
    LocalResearchConnection,
)
from alphalattice.interface.local_application.dispatcher import (
    LocalApplicationCommand,
    LocalBackgroundDispatcher,
)
from alphalattice.interface.local_application.failure_codes import (
    FAILURE_DETAIL_WITHHELD,
    located_failure,
    owner_failure_code,
    safe_failure_code,
)
from alphalattice.interface.local_application.native_bridge import (
    HOSTS,
    NativeBridgeError,
    NativeResearchBinding,
    autobind_root,
    read_session_usage,
    session_binding,
)
from alphalattice.interface.local_application.portfolio_research import (
    FinalizationStatusProjection,
    FrozenCandidateProjection,
    LocalPortfolioResearchService,
    OperationCaller,
    PortfolioResearchOperation,
    PortfolioResearchOperationRequest,
    validate_request_document,
)
from alphalattice.interface.local_application.thin_cli import CLI_OPERATION
from alphalattice.interface.local_application.web import (
    DEFAULT_IDLE_TIMEOUT_SECONDS,
    LocalWebApplication,
    LocalWebError,
    LocalWebResponse,
    LocalWebService,
    report_policy,
)
from alphalattice.interface.local_application.workbench_assets import serve_assets
from alphalattice.investment.alpha_research.experiments.authoring import ALPHA_EXPERIMENT_KIND
from alphalattice.investment.alpha_research.scores.model_renewal import (
    verified_lifecycle_admissions,
)
from alphalattice.investment.portfolio_strategy_lab.application.advancement_task import (
    ADVANCEMENT_TASK_KIND,
)
from alphalattice.investment.portfolio_strategy_lab.application.finalization import (
    ValidatedPortfolioHandoff,
)
from alphalattice.investment.portfolio_strategy_lab.application.resolution import (
    PortfolioExecutionResolver,
    SharedPortfolioInputResolver,
    StrategyPortfolioResolver,
)
from alphalattice.investment.portfolio_strategy_lab.application.task import (
    PORTFOLIO_PUBLIC_TASK_KIND,
)
from alphalattice.investment.portfolio_strategy_lab.publication.finalization_ledger import (
    PortfolioFinalizationStore,
)
from alphalattice.kernel.clock import utc_now
from alphalattice.kernel.config import current as process_config
from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer.history import (
    data_root,
    held,
    move_carried,
    release_history,
    render_in_background,
)
from alphalattice.kernel.data_layer.store import preload_grouping
from alphalattice.kernel.data_layer.upgrade import UpgradeRunner
from alphalattice.kernel.shared_kernel.project_layout import resolve_playpen_root

if TYPE_CHECKING:
    from alphalattice.control.data_platform.readiness import SourceLoader
    from alphalattice.control.product_host.composition.evidence_review_application import (
        EvidenceReviewApplication,
    )
    from alphalattice.control.product_host.composition.native_usage_reader import NativeUsageReader
    from alphalattice.control.product_host.composition.portfolio_application import (
        PortfolioResearchApplication,
    )
    from alphalattice.control.product_host.composition.portfolio_finalization import (
        PortfolioFinalizationApplication,
    )
    from alphalattice.control.product_host.composition.portfolio_research_operations import (
        PortfolioResearchOperations,
    )
    from alphalattice.control.product_host.composition.workspace_activity import WorkspaceActivity
    from alphalattice.control.product_host.maintenance.data_update import (
        WorkspaceDataUpdateApplication,
    )
    from alphalattice.evidence.alternative_evidence.contracts import SecIssuerRegistrySnapshot
    from alphalattice.evidence.alternative_evidence.publication.analysis import (
        AlternativeEvidenceAnalysisPublicationService,
    )
    from alphalattice.evidence.alternative_evidence.publication.artifacts import (
        AlternativeEvidenceArtifactStore,
    )
    from alphalattice.evidence.alternative_evidence.runtime.policy import (
        AdmittedEvidencePolicy,
    )
    from alphalattice.evidence.alternative_evidence.runtime.service import (
        AlternativeEvidenceDocumentIntelligenceRuntime,
    )
    from alphalattice.evidence.alternative_evidence.runtime.task_adapter import (
        AlternativeEvidenceDocumentTaskResources,
    )
    from alphalattice.foundation.market_data_ops.sources.providers import MarketDataProvider
    from alphalattice.investment.portfolio_strategy_lab.application.contracts import (
        PortfolioExecutionProgram,
    )
    from alphalattice.investment.portfolio_strategy_lab.reporting.static import (
        PortfolioReportContext,
    )
    from alphalattice.oversight.chief_risk_officer.portfolio_evidence.contracts import (
        AdmittedListingTickerAuthority,
    )
    from alphalattice.oversight.chief_risk_officer.runtime.portfolio_review_task import (
        PortfolioReviewActor,
    )

ASSET_ROOT = Path(__file__).resolve().parents[3] / "interface" / "local_application" / "assets"


def _utc_now() -> datetime:
    """The product clock: one function, aware, used by every owner on the path."""

    return utc_now()


_utc_now.clock_domain = "UTC"  # type: ignore[attr-defined]  # the default Host uses the system wall clock


"""Local files, served from the installed package. No CDN, no font host."""


@dataclass(slots=True)
class HostPortfolioFreeze:
    """The Gate 9B freeze and finalization owners, projected for a local UI.

    Every field below is read off an owner. Nothing is derived here, and there is
    no route through this class that could evaluate protected evidence: freezing
    is a read of a published development result plus one registry write, and the
    finalization status is whatever the finalization store already holds.
    """

    session: WorkspaceApplicationSession
    application: PortfolioResearchApplication
    store: PortfolioFinalizationStore
    _finalization: PortfolioFinalizationApplication | None = field(
        init=False, default=None, repr=False
    )
    _finalization_lock: Lock = field(init=False, default_factory=Lock, repr=False)

    def _protected_finalization(self) -> PortfolioFinalizationApplication:

        # Only the status of an existing candidate needs the protected path.
        from .portfolio_finalization import (  # lazy: on use
            PortfolioFinalizationApplication,
        )

        with self._finalization_lock:
            if self._finalization is None:
                self._finalization = PortfolioFinalizationApplication(
                    workspace_id=self.application.workspace_id,
                    session=self.session,
                    continuation=None,  # type: ignore[arg-type]
                    finalization_store=self.store,
                )
            return self._finalization

    def freeze(self, *, result_hash: str) -> FrozenCandidateProjection:
        """Freeze a published development result through its originating Task owner.

        Args:
            result_hash: Exact published development result to freeze.

        Returns:
            Projection of the candidate recorded by the finalization store.

        Raises:
            LocalWebError: The result has no originating development Task.
        """
        manifest = self.application.pipeline.find_for_result(result_hash)
        if manifest is None:
            raise LocalWebError("local_web.result_has_no_originating_task")

        # Only a freeze request needs the candidate publication and Task adapters.
        from ....investment.portfolio_strategy_lab.application import (  # lazy: freeze
            candidate_freeze,
        )
        from .portfolio_finalization import (  # lazy: on use
            HostCompletedDevelopmentTasks,
        )

        candidate = candidate_freeze.freeze_candidate(
            self.application.ledger,
            self.store,
            workspace_id=self.application.workspace_id,
            result_hash=result_hash,
            development_task_id=str(manifest.task_id),
            tasks=HostCompletedDevelopmentTasks(self.session.task_control_registry),
        )
        return _projected(candidate)

    def open_frozen(self, candidate_hash: str) -> FrozenCandidateProjection | None:
        """Read a frozen candidate by its exact identity.

        Args:
            candidate_hash: Candidate identity recorded by the finalization owner.

        Returns:
            The candidate projection, or None when no candidate is recorded.
        """
        candidate = self.store.open_candidate(candidate_hash)
        return None if candidate is None else _projected(candidate)

    def finalization_status(self, candidate_hash: str) -> FinalizationStatusProjection:
        """Project the recorded permit, closure and release status of a candidate.

        Args:
            candidate_hash: Exact frozen candidate identity to inspect.

        Returns:
            Status distinguishing an absent candidate, awaiting protected authority
            or publication, and the released package with its recorded handoff.
        """
        candidate = self.store.open_candidate(candidate_hash)
        if candidate is None:
            return FinalizationStatusProjection(
                candidate_hash=candidate_hash,
                disposition="NOT_FROZEN",
                detail="no candidate was frozen under this identity",
            )
        gate = self._protected_finalization().gate
        permit = gate.reopen_permit(candidate=candidate)
        if permit is None:
            return FinalizationStatusProjection(
                candidate_hash=candidate_hash,
                disposition="AWAITING_PROTECTED_AUTHORITY",
                detail=(
                    "this candidate is frozen and waiting. A protected evaluation needs a "
                    "one-time permit over admitted protected evidence, and this workspace "
                    "holds neither."
                ),
            )
        package = self.store.find_package_for_permit(permit.permit_hash)
        receipt = gate.reopen_receipt(permit=permit)
        if package is None or receipt is None:
            return FinalizationStatusProjection(
                candidate_hash=candidate_hash,
                disposition="AWAITING_PROTECTED_AUTHORITY",
                detail="a permit exists; the protected path has not been sealed and closed",
            )
        release = self.store.find_release_for_package(package.package_hash)
        handoff = self.store.find_handoff_for_package(package.package_hash)
        if release is None or handoff is None:
            return FinalizationStatusProjection(
                candidate_hash=candidate_hash,
                disposition="AWAITING_PROTECTED_AUTHORITY",
                detail="closed, and not yet released",
                package_hash=package.package_hash,
                validation_receipt_hash=receipt.receipt_hash,
                closure=receipt.closure,
            )
        return FinalizationStatusProjection(
            candidate_hash=candidate_hash,
            disposition="RELEASED",
            detail="released; the exact handoff is readable",
            package_hash=package.package_hash,
            validation_receipt_hash=receipt.receipt_hash,
            handoff_hash=handoff.handoff_hash,
            released_result_hash=release.released_result_hash,
            released_report_hash=release.released_report_hash,
            closure=receipt.closure,
        )


def _projected(candidate: object) -> FrozenCandidateProjection:
    state = candidate.pre_protected_state  # type: ignore[attr-defined]
    return FrozenCandidateProjection(
        candidate_hash=candidate.candidate_hash,  # type: ignore[attr-defined]
        workspace_id=candidate.workspace_id,  # type: ignore[attr-defined]
        development_result_hash=candidate.development_result_hash,  # type: ignore[attr-defined]
        development_task_id=candidate.development_task_id,  # type: ignore[attr-defined]
        development_run_hash=candidate.development_run_hash,  # type: ignore[attr-defined]
        program_hash=candidate.program_hash,  # type: ignore[attr-defined]
        spec_hash=candidate.spec_hash,  # type: ignore[attr-defined]
        control_receipt_hash=candidate.control_receipt_hash,  # type: ignore[attr-defined]
        pre_protected_state_hash=state.state_hash,
        last_formation_session=state.last_formation_session,
        formation_count=state.formation_count,
        frozen_at=candidate.frozen_at,  # type: ignore[attr-defined]
    )


class LocalWebSessionError(RuntimeError):
    """A local session could not be started or stopped without leaving state."""


@dataclass(frozen=True, slots=True)
class EvidenceReviewAuthority:
    """What a host must be *given* before the Evidence & CRO section can work.

    An issuer registry and a listing/ticker authority are admitted evidence
    inputs; an evidence runtime and a review actor are admitted capabilities; a
    validated handoff is released by Portfolio. None of them can be inferred
    from a workspace, so a service that was not handed them serves typed
    zero-work refusals -- which is exactly what an ordinary development
    workspace does until its owner admits them.
    """

    registry: SecIssuerRegistrySnapshot
    listing_authority: AdmittedListingTickerAuthority
    artifacts: AlternativeEvidenceArtifactStore
    handoff: ValidatedPortfolioHandoff | None = None
    evidence_publications: AlternativeEvidenceAnalysisPublicationService | None = None
    evidence_runtime: AlternativeEvidenceDocumentIntelligenceRuntime | None = None
    """The admitted Alternative Evidence runtime, or nothing.

    A *runtime*, not an adapter. An adapter is bound to a Task Control registry,
    and that registry does not exist until a workspace session has been acquired,
    so the two session-independent halves are held here and the composition
    owner binds them to the session's registry once it has one.
    """

    evidence_resources: AlternativeEvidenceDocumentTaskResources | None = None
    evidence_policy: AdmittedEvidencePolicy | None = None
    review_actor: PortfolioReviewActor | None = None
    model_authority_admitted: bool = True
    """False when the workspace verified but no Provider credential is admitted."""

    selected_analysis_publication_hash: str | None = None


class PortfolioForwardPrewarm:
    """One coalescing background reader for the latest active book in this Host."""

    def __init__(self, operations: PortfolioResearchOperations) -> None:
        """Retain existing read owners without creating a Task or publication.

        Args:
            operations: This Host's admitted operation owners.
        """
        self.operations = operations
        self._wake = MaintenanceWakeController()
        self._stopping = Event()
        self._started = False
        self._host = MaintenanceBackgroundHost(wake=self._wake, run_once=self._read_once)

    def attach(self, dispatcher: LocalBackgroundDispatcher) -> None:
        """Preserve command-return observations and add a non-blocking update wake.

        Args:
            dispatcher: The Host dispatcher whose activity observer is already attached.
        """
        observed = dispatcher.on_command_returned

        def command_returned(kind: str, task_id: UUID, failure: str | None) -> None:
            try:
                if observed is not None:
                    observed(kind, task_id, failure)
            finally:
                if failure is None and not self._stopping.is_set():
                    self._wake_after_update(kind, task_id)

        dispatcher.on_command_returned = command_returned

    def _wake_after_update(self, kind: str, task_id: UUID) -> None:
        from ..maintenance.data_update import (  # lazy: successful data or Portfolio update
            DATA_UPDATE_TASK_KIND,
        )

        if kind not in {DATA_UPDATE_TASK_KIND, ADVANCEMENT_TASK_KIND, PORTFOLIO_UPDATE_TASK_KIND}:
            return
        # Only terminal success warms new data, since a returned worker may have stopped.
        try:
            task = self.operations.workspace_session.task_control_registry.task(task_id)
        except Exception:
            return
        if task is not None and task.lifecycle is TaskLifecycle.SUCCEEDED:
            self.wake()

    def start(self) -> None:
        """Start the serial reader and queue the initial startup read."""
        self._host.start()
        self._started = True
        self.wake()

    def wake(self) -> None:
        """Coalesce another read request without doing any read in the caller."""
        if not self._stopping.is_set():
            self._wake.event.set()

    def close(self, *, timeout: float | None = None) -> bool:
        """Stop accepting work and join before this Host releases its read resources.

        Args:
            timeout: Bounded join allowance, or None to wait for the current read.

        Returns:
            Whether no background read remains live.
        """
        self._stopping.set()
        if not self._started:
            return True
        quiet = self._host.close(timeout=0.0 if timeout is None else timeout)
        if timeout is None and not quiet:
            self._host.wait_stopped()
            return True
        return quiet

    @property
    def last_error_code(self) -> str | None:
        """Return the existing maintenance owner's bounded read failure code."""
        return self._host.last_error_code

    @verified_model_read_scope()
    @verified_lifecycle_admissions()
    def _read_once(self) -> None:
        if self._stopping.is_set() or self.operations.activations is None:
            return
        manifests = self.operations.manifests
        assert manifests is not None
        candidates: list[tuple[datetime, str, UUID]] = []
        for binding in manifests.current.decision_updates or ():
            state = self.operations.activations.summary(binding.strategy_package_id)
            if state.get("status") != "ACTIVE":
                continue
            book = state.get("book_task_id")
            activated = state.get("activated_at")
            if not isinstance(book, str) or not isinstance(activated, str):
                continue
            task_id = UUID(book)
            when = datetime.fromisoformat(activated)
            if when.tzinfo is None:
                continue
            candidates.append((when, binding.strategy_package_id, task_id))
        if candidates and not self._stopping.is_set():
            # One exact book per cycle bounds retained work. A wake during this
            # read requests at most one following cycle on the same serial worker.
            task_id = max(candidates, key=lambda row: (row[0], row[1], str(row[2])))[2]
            read_workbench_portfolio(
                self.operations,
                {"task_id": [str(task_id)], "performance": ["latest"]},
                caller="SERVICE_AUTOMATION",
            )


class _RecoveryCommands(Mapping[str, LocalApplicationCommand]):
    """Restore ownership by kind without constructing a command for settled Tasks."""

    def __init__(
        self,
        factories: dict[str, Callable[[], LocalApplicationCommand]],
        verified: dict[str, LocalApplicationCommand],
    ) -> None:
        self._factories = factories
        self._issued = verified
        self._kinds = dict.fromkeys((*factories, *verified))

    def __contains__(self, kind: object) -> bool:
        return kind in self._kinds

    def __iter__(self) -> Iterator[str]:
        return iter(self._kinds)

    def __len__(self) -> int:
        return len(self._kinds)

    def __getitem__(self, kind: str) -> LocalApplicationCommand:
        try:
            return self._issued[kind]
        except KeyError:
            command = self._factories[kind]()
            self._issued[kind] = command
            return command


def _recovery_command[Owner](
    constructor: Callable[[Owner], LocalApplicationCommand], owner: Owner | None
) -> LocalApplicationCommand:
    assert owner is not None
    return constructor(owner)


def _portfolio_recovery(
    operations: PortfolioResearchOperations, kind: str
) -> LocalApplicationCommand:

    from . import (  # lazy: only owed Portfolio work opens the family's recovery commands.
        portfolio_operations,
    )

    return portfolio_operations.recovery_command(operations, kind)


def _experiment_recovery(operations: PortfolioResearchOperations) -> LocalApplicationCommand:

    # Only this recovery command needs its owner's adapter.
    from .research_experiments import (  # lazy: on use
        ResearchExperimentCommand,
    )

    return ResearchExperimentCommand(operations.experiments)


def _sweep_recovery(operations: PortfolioResearchOperations) -> LocalApplicationCommand:

    from .verification_sweep import (  # lazy: only this recovery command needs its owner's adapter.
        StudyVerificationSweepCommand,
    )

    return StudyVerificationSweepCommand(operations.sweep)


def _evidence_install_recovery(operations: PortfolioResearchOperations) -> LocalApplicationCommand:

    # Only this recovery command needs its owner's adapter.
    from .evidence_authority_setup import (  # lazy: on use
        EvidenceInstallCommand,
    )

    return EvidenceInstallCommand(operations.evidence_install)


@dataclass(slots=True)
class LocalPortfolioWebSession:
    """Everything the local product needs, started and stopped as one unit."""

    workspace: Path
    workspace_manifest: ResearchWorkspaceManifest
    resolver: PortfolioExecutionResolver | None
    port: int = 0
    idle_timeout_seconds: int = field(default=DEFAULT_IDLE_TIMEOUT_SECONDS, kw_only=True)
    upgrade: UpgradeRunner | None = field(default=None, kw_only=True)
    """A runner given in place of the first start's own (`FirstStart`); it finishes before the
    Host binds."""
    first_start: FirstStart | None = field(default=None, init=False, repr=False)
    """The upgrade this start owes: while it is not done, the Host serves only its routes."""
    stop_host: Callable[[], dict[str, object]] | None = field(default=None, kw_only=True)
    """How `host stop` stops this Host, given to its operations when they compose."""
    _upgrading: Thread | None = field(default=None, init=False, repr=False)
    _admitting: Lock = field(default_factory=Lock, init=False, repr=False)

    review_authority: EvidenceReviewAuthority | None = None
    data_provider: MarketDataProvider | None = None
    data_source_loader: SourceLoader | None = None
    clock: Callable[[], datetime] = _utc_now
    """The service's own timezone-aware clock, shared with everything it builds."""
    session: WorkspaceApplicationSession | None = field(default=None, init=False)
    dispatcher: LocalBackgroundDispatcher | None = field(default=None, init=False)
    operations: PortfolioResearchOperations | None = field(default=None, init=False)
    activity: WorkspaceActivity | None = field(default=None, init=False)
    forward_prewarm: PortfolioForwardPrewarm | None = field(default=None, init=False)
    native_usage_reader: NativeUsageReader | None = field(default=None, init=False)
    web: LocalWebService | None = field(default=None, init=False)
    client_connection: LocalResearchConnection | None = field(default=None, init=False, repr=False)
    _history: bool = field(default=False, init=False, repr=False)
    _kept: Callable[[], None] | None = field(default=None, init=False, repr=False)
    """The end of this Host's storage session: each workspace file opened once while it serves."""
    resumed_task_ids: tuple[UUID, ...] = field(default=(), init=False)
    stop_requested: Callable[[], bool] = field(default=bool, init=False)
    _workspace_review_authority: bool = field(default=False, init=False)
    _official_source: OfficialSourceAdmission | None = field(default=None, init=False)

    @property
    def application(self) -> PortfolioResearchApplication | None:
        """Open the workspace Portfolio application on first use."""
        return None if self.operations is None else self.operations.application

    @property
    def service(self) -> LocalPortfolioResearchService | None:
        """Open the workspace Portfolio service and its application on first use."""
        return None if self.operations is None else self.operations.service

    @property
    def review(self) -> EvidenceReviewApplication | None:
        """Open the workspace Evidence review on first use."""
        return None if self.operations is None else self.operations.review

    @property
    def workspace_id(self) -> str:
        """Return the admitted workspace manifest's identity."""
        return self.workspace_manifest.workspace_id

    @classmethod
    def from_workspace(
        cls,
        workspace: Path,
        *,
        port: int = 0,
        review_authority: EvidenceReviewAuthority | None = None,
        clock: Callable[[], datetime] = _utc_now,
        official_source: OfficialSourceAdmission | None = None,
        upgrade: UpgradeRunner | None = None,
    ) -> LocalPortfolioWebSession:
        """Ordinary composition: one path in, verified manifest and catalog out.

        The official SEC source is admitted at runtime from the workspace's consent, its network
        control and the product's contact (`admit_workspace_source`), with no restart.
        `official_source` replaces that admission with a fixed one, a rehearsal's own.
        """
        initialize_research_workspace(workspace)
        if (named := named_workspace_id(workspace)) is not None:
            move_carried(workspace.resolve(), named)  # a copy's lanes, before admission reads them
        admitted = admit_research_workspace(workspace)
        value = cls(
            workspace=admitted.root,
            workspace_manifest=admitted.manifest,
            # Composed with or without an installed strategy: an installation sets its
            # catalog on the running Host, so no restart is needed.
            resolver=StrategyPortfolioResolver(
                shared=SharedPortfolioInputResolver(), catalog=admitted.catalog
            ),
            port=port,
            review_authority=review_authority,
            clock=clock,
            upgrade=upgrade,
        )
        value._workspace_review_authority = (
            review_authority is None and admitted.manifest.evidence_review is not None
        )
        value._official_source = official_source
        return value

    def start(self) -> str:
        """Acquire, compose, bind -- or leave nothing behind: a failure part-way tears down what
        already started (the history, a lease, a non-daemon worker) in a healthy stop's order,
        and is re-raised, so no held workspace is left with no URL to stop it through."""
        try:
            # The session writes the market store (its data updates): it opens writable once, at
            # its first use, the history's own reads included.
            self._kept = keep_workspace_databases(
                writes=(self.workspace / WORKSPACE_MARKET_DATA_DATABASE_FILENAME,)
            )
            WorkspaceApplicationSession.open_history(
                self.workspace_manifest.workspace_id, self.workspace, self.clock
            )
            self._history = True
            return self._start()
        except BaseException:
            self._teardown(timeout=None)
            raise

    def _start(self) -> str:
        if self._workspace_review_authority and self.review_authority is None:
            binding = self.workspace_manifest.evidence_review
            assert binding is not None
            self.review_authority = self._workspace_authority(binding)
        if self.upgrade is None:
            first = FirstStart(data_root(), held(), self.workspace)
            if first.plan is not None:
                self.first_start = first
                restricted = self._upgrade_web(first)
                self.web = LocalWebService(application=restricted, port=self.port)
                url = self.web.start()
                self._publish_connection(restricted, url)
                if first.state == "RUNNING":  # acknowledged when it started: it resumes
                    self._upgrade_in_background()
                return url
        return self._compose()

    def _upgrade_web(self, first: FirstStart) -> LocalWebApplication:
        """The Host while its upgrade is owed: the Workbench's assets, /api/session, the plan and
        its progress on /api/upgrade, and its acknowledgement, which the CLI's `upgrade show` and
        `upgrade acknowledge` reach through /api/client/operations. Every other path and
        operation refuses `upgrade.in_progress` with the progress."""
        web = workspace_web(self.workspace, None)
        serve_assets(web, ASSET_ROOT)
        # Nothing it answers writes the log: a span or a CLI call's end would land between the
        # data root's backup and the run's check that the backup holds the log's head. A CLI's
        # call end is refused with the progress, which the CLI does not read.
        del web.routes[("POST", CALL_PATH)]
        routes = {"UPGRADE_OVERVIEW": ("GET", "/api/upgrade")}
        routes["UPGRADE_ACKNOWLEDGE"] = ("POST", "/api/upgrade/acknowledge")
        web.withheld = lambda: {
            "failure_code": "upgrade.in_progress",
            "upgrade": first.view()["first_start"],
        }

        @web.route("GET", "/api/session", session=True)
        def _session(_query: Mapping[str, list[str]], _payload: dict[str, Any]) -> object:
            return {
                "workspace_id": self.workspace_id,
                "workspace_manifest_hash": self.workspace_manifest.manifest_hash,
                "execution_mode": "UPGRADE",
                "session_token": web.session_token,
                "upgrade": first.view()["first_start"],
                "routes": {op: {"method": m, "path": path} for op, (m, path) in routes.items()},
            }

        @web.route(*routes["UPGRADE_OVERVIEW"])
        def _overview(_query: Mapping[str, list[str]], _payload: dict[str, Any]) -> object:
            return first.view()

        @web.route(*routes["UPGRADE_ACKNOWLEDGE"], mutates=True, traced=False)
        def _acknowledge(_query: Mapping[str, list[str]], payload: dict[str, Any]) -> object:
            with self._admitting:
                first.admit(payload.get("upgrade_set_hash"))
                self._upgrade_in_background()
            return first.view()

        @web.route(
            "POST", "/api/client/operations", mutates=True, external_client=True, traced=False
        )
        def _client(query: Mapping[str, list[str]], payload: dict[str, Any]) -> object:
            try:
                request = validate_request_document(payload)
            except ValidationError as error:
                failure = located_failure(error, "local_client.request_invalid", request=payload)
                return {"status": "REFUSED", **failure}
            operation = request.to_operation_request()
            try:
                if operation.operation == "UPGRADE_OVERVIEW":
                    return first.view()
                if operation.operation == "UPGRADE_ACKNOWLEDGE":
                    return _acknowledge(query, {"upgrade_set_hash": operation.upgrade_set_hash})
            except DataError as error:
                return refused(error.failure.code)
            return refused("upgrade.in_progress") | {"upgrade": first.view()["first_start"]}

        return web

    def _publish_connection(
        self, web: LocalWebApplication, url: str, serving_host: dict[str, object] | None = None
    ) -> None:
        """Publish the connection the CLI reads to reach this Host's `web` at `url`."""
        connection = LocalResearchConnection(
            workspace=str(self.workspace.resolve()),
            workspace_id=self.workspace_id,
            url=url,
            instance=web.external_instance,
            token=web.external_token,
        )
        publish_client_connection(connection, serving_host=serving_host)
        self.client_connection = connection
        preload_grouping()  # the admission's grouping engine loads once the Host serves

    def _upgrade_in_background(self) -> None:
        """Run the acknowledged upgrade off the request thread, then compose and serve in full."""

        def finish() -> None:
            first = self.first_start
            assert first is not None
            try:
                first.run()
                self._compose()
            except Exception as error:  # kept for /api/upgrade; the Host stays restricted
                first.state, first.failure = "FAILED", first.failure or refusal_reason(error)

        self._upgrading = Thread(target=finish, name="first-start-upgrade", daemon=False)
        self._upgrading.start()

    def upgraded(self, timeout: float | None = None) -> bool:
        """Wait up to `timeout` for the acknowledged first-start upgrade to end; True once the
        Host serves its full composition."""
        if self._upgrading is not None:
            self._upgrading.join(timeout)
        return self.operations is not None

    def _compose(self) -> str:
        from .native_usage_reader import attach_native_usage  # lazy: full Host only
        from .operation_route_families import HostRouteFamilies  # lazy: full Host only
        from .portfolio_research_operations import (  # lazy: full Host only
            PortfolioResearchOperations,
        )
        from .workspace_activity import WorkspaceActivity  # lazy: full Host only

        session = WorkspaceApplicationSession.acquire(self.workspace, self.upgrade).__enter__()
        self.session = session
        # The one holder of the workspace manifest every application reads; the operations
        # refresh it when a publication moves the manifest.
        manifests = ResearchWorkspaceManifestHolder(self.workspace_manifest)
        resolver = self.resolver
        open_portfolio = None
        finalization_store = None
        pipeline = None
        if resolver is not None:
            # The light history reader opens the publication store without compiling a book.
            from ..publication.portfolio_research import (  # lazy: bound workspace
                PortfolioResearchPipelineStore,
            )

            root = session.workspace / "runtime" / "artifacts"
            history = PortfolioResearchPipelineStore(root)
            candidate_store = PortfolioFinalizationStore(root)
            pipeline, finalization_store = history, candidate_store

            def open_portfolio() -> tuple[
                PortfolioResearchApplication, LocalPortfolioResearchService
            ]:

                # The Portfolio application compiles only when its routes or recovery use it.
                from . import (  # lazy: on use
                    portfolio_application,
                )

                def report_context(
                    program: PortfolioExecutionProgram, day: date
                ) -> PortfolioReportContext:

                    # The numerical report reader belongs to a published Portfolio result.
                    from . import (  # lazy: on use
                        portfolio_result_context,
                    )

                    return portfolio_result_context.portfolio_report_context(
                        workspace=self.workspace,
                        packages=resolver.installed_packages(),
                        manifest=manifests.current,
                        program=program,
                        session=day,
                    )

                application = portfolio_application.PortfolioResearchApplication(
                    workspace_id=self.workspace_id,
                    workspace=self.workspace,
                    manifest_binding=lambda: manifest_fields_hash(
                        manifests.current, portfolio_application.PLAN_FIELDS
                    ),
                    session=session,
                    resolver=resolver,
                    report_context=report_context,
                    clock=self.clock,
                    pipeline=history,
                )
                freeze = HostPortfolioFreeze(
                    session=session, application=application, store=candidate_store
                )
                return application, LocalPortfolioResearchService(
                    application=application,
                    lineage=application,
                    freeze=freeze,
                    programs=application,
                )

        self.dispatcher = LocalBackgroundDispatcher(
            status_port=session.task_control_registry,
            queue_full_error=TaskQueueFull,
            version_moved_error=TaskTransitionRejected,
            version_stale_error=TaskVersionStale,
            clock=self.clock,
            stop_requested=lambda: self.stop_requested(),
            workers=lambda: running_places(read_queue_setting(self.workspace / "runtime"))[0],
        )
        session.task_control_registry.overlapping = runs_beside_others
        session.task_control_registry.started = self.dispatcher.drive_waiting
        self.dispatcher.start()
        # The light history reader and the review owner share their publication store.
        # Bound workspace.
        from ....evidence.alternative_evidence.publication.artifacts import (  # lazy: on use
            AlternativeEvidenceArtifactStore,
        )

        authority = self.review_authority
        review_artifacts = (
            authority.artifacts
            if authority is not None
            else AlternativeEvidenceArtifactStore(self.workspace / "runtime" / "artifacts")
        )

        def open_review() -> EvidenceReviewApplication:
            review = self._review_application(
                session, finalization_store, artifacts=review_artifacts
            )
            if self._workspace_review_authority and self._official_source is not None:
                review.use_source(self._official_source)
            review.refresh_source()
            return review

        update_clock = self.clock
        data_provider = self.data_provider
        data_source_loader = self.data_source_loader

        def open_data_update() -> WorkspaceDataUpdateApplication:
            from ..maintenance.data_update import (  # lazy: first workspace data update use
                WorkspaceDataUpdateApplication,
            )

            return WorkspaceDataUpdateApplication(
                session=session,
                manifest=manifests,
                clock=update_clock,
                provider=data_provider,
                source_loader=data_source_loader,
            )

        self.operations = PortfolioResearchOperations(
            dispatcher=self.dispatcher,
            workspace_manifest=self.workspace_manifest,
            route_families=HostRouteFamilies(),
            portfolio_factory=open_portfolio,
            portfolio_pipeline=pipeline,
            installed_packages=None if resolver is None else resolver.installed_packages,
            manifests=manifests,
            workspace_session=session,
            review_factory=open_review,
            review_artifacts=review_artifacts,
            recover_task=self.resume,
            recoverable_task_kinds=self.recoverable_task_kinds,
            resume_refusal=self.resume_refusal,
            install_review=self.serve_installed_review,
            official_source=self._official_source,
            heartbeats=TaskHeartbeatReader(
                session.runtime_path,
                self.workspace / "runtime" / EVIDENCE_REVIEW_RUNTIME_FILE,
            ),
            data_update_factory=open_data_update,
        )
        # The activity observer is composed after every owner it reads and before the socket opens,
        # so the first request is already observed. Activity is optional infrastructure: a store
        # that cannot open leaves the observer UNAVAILABLE on every read and the product usable.
        activity = WorkspaceActivity(
            workspace=self.workspace,
            workspace_id=self.workspace_id,
            gate=session.mutation_gate,
            instance=secrets.token_hex(16),
            clock=self.clock,
        )
        self.activity = activity
        self.operations.observer = activity
        activity.attach(
            dispatcher=self.dispatcher,
            registry=session.task_control_registry,
            artifacts=self.operations.artifact_reference,
            operations=self.operations,
        )
        self.forward_prewarm = PortfolioForwardPrewarm(self.operations)
        self.forward_prewarm.attach(self.dispatcher)
        attach_native_usage(self, self.operations, activity)
        web_application = build_local_web_application(operations=self.operations, activity=activity)
        web_application.admission = self.dispatcher._admit
        _agent_telemetry_routes(web_application, self.native_usage_reader, self.workspace_id)
        self.operations.stop_host = self.stop_host
        with self.operations.recover_after_connection():
            if self.web is None:
                self.web = LocalWebService(application=web_application, port=self.port)
                url = self.web.start()
            else:  # the upgrade's socket, its session kept: the browser that opened it stays in
                self.web.serve(web_application)
                url = self.web.url
            self._publish_connection(web_application, url, activity.serving_host)
        self.resumed_task_ids = self.resume()
        # A Codex lead's wake whose Task ended, stopped or deferred while no Host ran (WAKE).
        activity.replay_wakes()
        if (automation := self.operations.automation) is not None:
            automation.start()
        if not self.resumed_task_ids:
            # A Host that starts idle sweeps as one that went idle would.
            self.operations.sweep_if_due()
        self.operations.start_file_scrub()
        self.operations.supervisor.start()
        self.forward_prewarm.start()
        # A note a stopped Host left unrendered, or one a changed renderer rewrites, is rendered
        # once the Host serves in full: a start that fails, or one restricted to its upgrade,
        # starts no render and writes nothing to the log through one.
        render_in_background()
        return url

    def resume(
        self, only_task_id: UUID | None = None, *, expected_task_hash: str | None = None
    ) -> tuple[UUID, ...]:
        """Pick up whatever the previous service left `RECOVERY_REQUIRED`.

        Every command is rebuilt from the Task that is actually recovery-required.
        `expected_task_hash` (with `only_task_id`) is a confirmed version; it rides the
        queue item to Task Control's start boundary and is enforced there.
        """
        if self.dispatcher is None:
            return ()
        return self.dispatcher.resume(
            self._recovery_commands(),
            only_task_id=only_task_id,
            expected_task_hash=expected_task_hash,
            refusal=self.resume_refusal,
            stopping=self.stop_requested,
        )

    def resume_refusal(self, task: TaskRecord) -> str | None:
        """Why this Task cannot resume under the code installed now, or None.

        A Task keeps the compatibility its owner computed when it was admitted or
        last started. Resuming recomputes it, and after an upgrade the owner refuses
        (its typed code) or the value moved against the latest execution's (Task
        Control's refusal). Both used to surface in the worker, after the resume was
        accepted, where nobody read them: the Task stayed RECOVERY_REQUIRED, was
        retried at every start and held the workspace's active place. Asked here,
        first, the answer is the Task's visible state. Nothing is written, and a Task
        whose recovery is not driven through an owner's adapter (the Portfolio run,
        the evidence and review Tasks) answers None.
        """
        if self.session is None or task.lifecycle not in {
            TaskLifecycle.RECOVERY_REQUIRED,
            TaskLifecycle.QUEUED,
        }:
            return None
        command = self._recovery_commands(owed_only=False).get(task.task_kind)
        # The Host's own commands hold the owner's application, which is its Task
        # adapter; the Portfolio run and the review applications are not adapters.
        compatibility_of = getattr(getattr(command, "application", None), "compatibility", None)
        if not callable(compatibility_of):
            return None
        try:
            compatibility = compatibility_of(task)
        except Exception as error:
            code = owner_failure_code(error)
            if code is None and not isinstance(error, ValueError):
                raise
            return code or "task_control.resume_compatibility_unreadable"
        if task.plan.workflow_definition_hash != compatibility.workflow_definition_hash:
            return "task_control.workflow_changed"
        if (
            task.lifecycle is TaskLifecycle.RECOVERY_REQUIRED
            and task.latest_execution_id is not None
            and self.session.task_control_registry.execution(task.latest_execution_id).compatibility
            != compatibility
        ):
            return "task_control.execution_compatibility_changed"
        return None

    def recoverable_task_kinds(self) -> frozenset[str]:
        """The Task kinds this service can resume, as a fact for a recovery view.

        The same commands `resume` would build, plus the installed Portfolio run
        whenever its application is composed (resume builds that one only for a
        Task actually owed). No command is enqueued by asking.
        """
        return frozenset(self._recovery_commands(owed_only=False))

    def _workspace_authority(
        self, binding: ResearchWorkspaceEvidenceReview
    ) -> EvidenceReviewAuthority:
        """The workspace's recorded package; the review admits the official source at runtime."""

        # Only a workspace with recorded review authority needs its admission.
        from .evidence_review_workspace import (  # lazy: on use
            admit_evidence_review_workspace,
        )

        admitted = admit_evidence_review_workspace(workspace=self.workspace, binding=binding)
        return EvidenceReviewAuthority(
            registry=admitted.registry,
            listing_authority=admitted.listing_authority,
            artifacts=admitted.runtime.artifacts,
            evidence_publications=admitted.runtime.publications,
            evidence_runtime=admitted.runtime,
            evidence_resources=admitted.resources,
            evidence_policy=admitted.evidence_policy,
            review_actor=admitted.review_actor,
            model_authority_admitted=admitted.model_authority_admitted,
        )

    def _review_application(
        self,
        session: WorkspaceApplicationSession,
        finalization_store: PortfolioFinalizationStore | None,
        *,
        artifacts: AlternativeEvidenceArtifactStore | None = None,
    ) -> EvidenceReviewApplication:
        return build_evidence_review_application(
            workspace=self.workspace,
            workspace_id=self.workspace_id,
            session=session,
            application=self.application,
            finalization_store=finalization_store,
            artifacts=artifacts,
            authority=self.review_authority,
            official_source=(
                partial(admit_workspace_source, self.workspace, clock=self.clock)
                if self._workspace_review_authority and self._official_source is None
                else None
            ),
            clock=self.clock,
        )

    def serve_installed_review(self) -> AlternativeEvidenceDocumentIntelligenceRuntime | None:
        """Serve the Evidence package an install Task just bound, in place, with no restart.

        The review keeps its object and swaps the package in at once (`adopt`), so every holder
        reads it; the runtime it replaces is answered, for the caller to close once the storage
        reads the new one. A swap that fails leaves the review as it was.
        """
        binding = read_research_workspace_manifest(self.workspace).evidence_review
        if binding is None or self.session is None or self.review is None:
            raise ValueError("evidence_review.install_failed")
        held = self.review_authority, self._workspace_review_authority
        authority = self._workspace_authority(binding)
        self.review_authority, self._workspace_review_authority = authority, True
        try:
            self.review.adopt(self._review_application(self.session, self.review.finalization))
        except BaseException:
            self.review_authority, self._workspace_review_authority = held
            if authority.evidence_runtime is not None:
                authority.evidence_runtime.close()
            raise
        if self._official_source is not None:
            self.review.use_source(self._official_source)
        replaced = held[0] if held[1] else None
        return None if replaced is None else replaced.evidence_runtime

    def _recovery_commands(
        self, *, owed_only: bool = True
    ) -> Mapping[str, LocalApplicationCommand]:

        # Recovery builds its commands once a queued or recovery-required Task needs them.
        from ..data_preparation.application import (  # lazy: on use
            WorkspacePreparationCommand,
        )
        from ..data_preparation.feature_research import (  # lazy: Task recovery catalog
            ResearchFeatureBuildCommand,
        )
        from ..data_preparation.input_capture import (  # lazy: Task recovery catalog
            ResearchInputCaptureCommand,
        )
        from ..data_preparation.model_training import (  # lazy: Task recovery catalog
            ModelTrainingInputCommand,
        )
        from ..data_preparation.research_strategy import (  # lazy: Task recovery catalog
            ResearchStrategyCommand,
        )
        from ..maintenance.data_update import (  # lazy: Task recovery catalog
            WorkspaceDataUpdateCommand,
        )
        from .portfolio_research_operations import (  # lazy: Task recovery catalog
            PORTFOLIO_RUN_COMMAND,
            PortfolioRunCommand,
        )

        factories: dict[str, Callable[[], LocalApplicationCommand]] = {}
        operations = self.operations
        if operations is not None:
            factories.update(
                {
                    STUDY_TASK_KIND: partial(_experiment_recovery, operations),
                    WorkspacePreparationCommand.command_kind: lambda: WorkspacePreparationCommand(
                        operations.preparation
                    ),
                    ResearchInputCaptureCommand.command_kind: lambda: ResearchInputCaptureCommand(
                        operations.input_capture
                    ),
                    ModelTrainingInputCommand.command_kind: lambda: ModelTrainingInputCommand(
                        operations.model_training_inputs
                    ),
                    ResearchStrategyCommand.command_kind: lambda: ResearchStrategyCommand(
                        operations.research_strategies
                    ),
                    ResearchFeatureBuildCommand.command_kind: lambda: ResearchFeatureBuildCommand(
                        operations.feature_builds
                    ),
                    STUDY_VERIFICATION_TASK_KIND: partial(_sweep_recovery, operations),
                    EVIDENCE_INSTALL_TASK_KIND: partial(_evidence_install_recovery, operations),
                }
            )
            # Read what the workspace binds from the same holder as its owners.
            manifest = (
                self.workspace_manifest
                if operations.manifests is None
                else operations.manifests.current
            )
            if operations.has_portfolio:
                if operations.has_data_update:
                    factories[ADVANCEMENT_TASK_KIND] = partial(
                        _portfolio_recovery, operations, ADVANCEMENT_TASK_KIND
                    )
                if manifest.decision_updates:
                    factories[PORTFOLIO_UPDATE_TASK_KIND] = partial(
                        _portfolio_recovery, operations, PORTFOLIO_UPDATE_TASK_KIND
                    )
                if manifest.calibration_inputs:
                    factories[STRATEGY_CALIBRATION_TASK_KIND] = partial(
                        _portfolio_recovery, operations, STRATEGY_CALIBRATION_TASK_KIND
                    )
                if manifest.score_inputs:
                    factories[STRATEGY_SCORE_TASK_KIND] = partial(
                        _portfolio_recovery, operations, STRATEGY_SCORE_TASK_KIND
                    )
            if operations.has_data_update and manifest.data_update is not None:
                factories[WorkspaceDataUpdateCommand.command_kind] = lambda: _recovery_command(
                    WorkspaceDataUpdateCommand, operations.data_update
                )
        assert self.session is not None
        registry = self.session.task_control_registry
        records = registry.tasks() if owed_only else registry.record_collection().records
        owed_kinds = {
            task.task_kind
            for task in records
            if task.lifecycle in {TaskLifecycle.RECOVERY_REQUIRED, TaskLifecycle.QUEUED}
        }
        if (
            operations is not None
            and operations.has_portfolio
            and (not owed_only or PORTFOLIO_PUBLIC_TASK_KIND in owed_kinds)
        ):

            def open_run() -> LocalApplicationCommand:
                application = operations.application
                assert application is not None
                return PortfolioRunCommand.recover(application=application)

            factories[PORTFOLIO_RUN_COMMAND] = open_run
        verified: dict[str, LocalApplicationCommand] = {}
        if owed_kinds:
            # Only owed Evidence or CRO work needs the review runtime.
            from ....evidence.alternative_evidence.runtime.task_adapter import (  # lazy: on use
                TASK_KIND as evidence_kind,
            )
            from ....oversight.chief_risk_officer.runtime import (  # lazy: CRO recovery
                portfolio_review_task,
            )

            cro_kind = portfolio_review_task.TASK_KIND

            if owed_kinds & {evidence_kind, cro_kind} and (review := self.review) is not None:
                # Keep the review's durable Task input and complete-or-refused authority check.
                for kind, command in review.recovery_commands(owed_only=owed_only).items():
                    verified[kind] = cast(LocalApplicationCommand, command)
        return _RecoveryCommands(factories, verified)

    def stop(self, *, timeout: float | None = None) -> None:
        """Join writers before releasing the workspace lease.

        A bounded stop that cannot finish raises and retains the lease for a retry.
        """
        if self.dispatcher is not None:
            self.dispatcher._begin_drain()
        if not self._teardown(timeout=timeout):
            raise LocalWebSessionError("local_web_session.writer_still_live")

    def _teardown(self, *, timeout: float | None) -> bool:
        """Join work and requests before releasing the lease. True when complete."""

        quiet = True
        if self._upgrading is not None:
            self._upgrading.join(timeout)
            if self._upgrading.is_alive():
                return False
        if self.forward_prewarm is not None:
            quiet = self.forward_prewarm.close(timeout=timeout) and quiet
        if self.operations is not None:
            self.operations.supervisor.stop()
            quiet = self.operations.stop_file_scrub(timeout) and quiet
        if self.operations is not None and (automation := self.operations._automation) is not None:
            quiet = automation.close(timeout=timeout) and quiet
        if self.dispatcher is not None:
            quiet = self.dispatcher.close(timeout=timeout) and quiet
        if self.web is not None and quiet:
            # Read-only status stays available while the owned work reaches a safe point.
            # Request threads are joined before the workspace lease is released.
            quiet = self.web.stop(timeout=timeout) and quiet
        if not quiet:
            return False
        if self.activity is not None:
            # After the worker: its command-return observation is the last
            # append this service makes. Closed before the lease is released.
            self.activity.close()
            self.activity = None
        if (
            self._workspace_review_authority
            and self.review_authority is not None
            and self.review_authority.evidence_runtime is not None
        ):
            self.review_authority.evidence_runtime.close()
            self.review_authority = None
        if self.operations is not None and self.operations._review is not None:
            self.operations._review.close_source()
        if self._kept is not None:  # before the lease: every kept file closed and sealed
            self._kept()
            self._kept = None
        if self._history:
            self._history = False
            release_history()
        if self.client_connection is not None:  # a Host serving only its upgrade published one
            remove_client_connection(self.client_connection)
            self.client_connection = None
        if self.session is not None:
            self.session.__exit__(None, None, None)
        self.web = None
        self.dispatcher = None
        self.session = None
        self.operations = None
        self.forward_prewarm = None
        self.native_usage_reader = None
        return True

    def __enter__(self) -> LocalPortfolioWebSession:
        """Start the local HTTP service and enter this session.

        Returns:
            This started session.
        """
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        """Stop the session while allowing a context exception to propagate."""
        self.stop()

    @property
    def url(self) -> str:
        """Return the running local HTTP service's URL.

        Raises:
            LocalWebError: The service has not been started.
        """
        if self.web is None:
            raise LocalWebError("local_web.service_not_started")
        return self.web.url

    @property
    def launch_url(self) -> str:
        """The URL that gives the person's browser its session.

        The launcher prints and opens it, and a script or the QA kit takes it from
        that output (HB). The Host never writes it to disk.
        """
        if self.web is None:
            raise LocalWebError("local_web.service_not_started")
        named = agent_session(process_config().public)
        goal = (
            self.operations.goals.attributed_goal(
                RequestProvenance(vendor=named[0], session=named[1])
            )
            if self.operations is not None and named is not None
            else None
        )
        follow = "latest" if goal is None else f"goal:{goal.goal_id}"
        return self.web.launch_url + "#" + urlencode({"follow": follow})


def runs_beside_others(task: TaskRecord) -> bool:
    """Whether a Task may run beside others.

    A Risk or Factor study writes only its own folder, and an Evidence install only its
    package. An Alpha study runs alone: its fold metrics are unscoped dot products over a
    fold's rows, long enough that the BLAS splits their sums by the process's thread count,
    which another Task's one-thread scope would narrow mid-run. A qualification reads the
    studies of its family, so it runs alone too.
    """
    if task.task_kind == EVIDENCE_INSTALL_TASK_KIND:
        return True
    plan = task.input.payload.get("plan") if task.task_kind == STUDY_TASK_KIND else None
    return (
        isinstance(plan, dict)
        and plan.get("qualification_family") is None
        and plan.get("program", {}).get("kind") != ALPHA_EXPERIMENT_KIND
    )


def build_evidence_review_application(
    *,
    workspace: Path,
    workspace_id: str,
    session: WorkspaceApplicationSession,
    application: PortfolioResearchApplication | None,
    finalization_store: PortfolioFinalizationStore | None,
    authority: EvidenceReviewAuthority | None = None,
    artifacts: AlternativeEvidenceArtifactStore | None = None,
    official_source: Callable[[], OfficialSourceAdmission] | None = None,
    clock: Callable[[], datetime] = _utc_now,
) -> EvidenceReviewApplication:
    """The one review owner this service holds.

    Built for every workspace, with or without admitted evidence authority: the
    section has to be able to answer "what is the state" in both, and a route
    that constructed the owner only when it already had an answer could never
    report the wait.
    """

    # Evidence and CRO compose their readers together at first use.
    from alphalattice.evidence.alternative_evidence.runtime.policy import (  # lazy: admitted review
        AdmittedEvidencePolicy,
    )

    from ....evidence.alternative_evidence.publication.artifacts import (  # lazy: admitted review
        AlternativeEvidenceArtifactStore,
    )
    from ....evidence.alternative_evidence.runtime.task_adapter import (  # lazy: admitted review
        AlternativeEvidenceDocumentTaskAdapter,
    )

    # Admitted review.
    from ....oversight.chief_risk_officer.publication.portfolio_review import (  # lazy: on use
        PortfolioReviewPublicationService,
    )
    from .evidence_review_application import (  # lazy: on use
        EvidenceReviewApplication,
    )

    if authority is not None:
        artifacts = authority.artifacts
    elif artifacts is None:
        artifacts = AlternativeEvidenceArtifactStore(workspace / "runtime" / "artifacts")
    adapter: AlternativeEvidenceDocumentTaskAdapter | None = None
    if (
        authority is not None
        and authority.evidence_runtime is not None
        and authority.evidence_resources is not None
    ):
        adapter = AlternativeEvidenceDocumentTaskAdapter(
            runtime=authority.evidence_runtime,
            registry=session.task_control_registry,
            resources=authority.evidence_resources,
            clock=clock,
        )

    def _latest() -> str | None:
        manifests = () if application is None else application.pipeline.manifests()
        return manifests[0].result_hash if manifests else None

    return EvidenceReviewApplication(
        workspace_id=workspace_id,
        workspace=workspace,
        session=session,
        ledger=None if application is None else application.ledger,
        artifacts=artifacts,
        evidence_publications=(authority.evidence_publications if authority is not None else None),
        review_publications=PortfolioReviewPublicationService(artifacts),
        # The repository root, not `src`: the review's decision policy hashes
        # source files by repo-relative path.
        playpen_root=resolve_playpen_root(Path(__file__)),
        latest_result_hash=_latest,
        finalization=finalization_store,
        registry=authority.registry if authority is not None else None,
        listing_authority=authority.listing_authority if authority is not None else None,
        handoff=authority.handoff if authority is not None else None,
        evidence_task_adapter=adapter,
        evidence_policy=(
            authority.evidence_policy or AdmittedEvidencePolicy()
            if authority is not None
            else AdmittedEvidencePolicy()
        ),
        review_actor=authority.review_actor if authority is not None else None,
        model_authority_admitted=(
            authority.model_authority_admitted if authority is not None else True
        ),
        selected_analysis_publication_hash=(
            authority.selected_analysis_publication_hash if authority is not None else None
        ),
        official_source=official_source,
        recorded_policy=authority.evidence_policy if authority is not None else None,
        clock=clock,
    )


OPERATION_ROUTES: Final[tuple[tuple[str, str, str], ...]] = (
    ("POST", "/api/experiments/declaration", "EXPERIMENT_DECLARATION_CONVERT"),
    ("GET", "/api/evidence/preview", "EVIDENCE_PREVIEW"),
    ("POST", "/api/evidence/prepare", "EVIDENCE_PREPARE"),
    ("GET", "/api/evidence/packet", "EVIDENCE_PACKET"),
    ("GET", "/api/evidence-cro/ledger", "EVIDENCE_LEDGER"),
    ("GET", "/api/evidence/documents", "EVIDENCE_DOCUMENTS"),
    ("POST", "/api/evidence/continue", "EVIDENCE_CONTINUE"),
    ("POST", "/api/evidence/analysis", "EVIDENCE_ANALYSIS_SUBMIT"),
    ("GET", "/api/cro/dossier", "CRO_REVIEW_DOSSIER"),
    ("GET", "/api/cro/finding", "CRO_REVIEW_FINDING"),
    ("POST", "/api/cro/assessment", "CRO_REVIEW_SUBMIT"),
    ("POST", "/api/research/delivery", "EXPERIMENT_DELIVERY_EXPORT"),
    ("POST", "/api/recover", "RECOVER"),
    ("GET", "/api/tasks", "TASKS"),
    ("GET", "/api/tasks/recovery", "TASK_RECOVERY"),
    ("GET", "/api/tasks/guardian", "TASK_GUARDIAN"),
    ("GET", "/api/research-history", "RESEARCH_HISTORY"),
    ("GET", "/api/experience", "EXPERIENCE_READ"),
    ("GET", "/api/experience-brief", "EXPERIENCE_BRIEF"),
    ("GET", "/api/experience-class", "EXPERIENCE_CLASS"),
    ("GET", "/api/experience-recall", "EXPERIENCE_RECALL"),
    ("POST", "/api/experience", "EXPERIENCE_RECORD"),
    ("GET", "/api/upgrade", "UPGRADE_OVERVIEW"),
    ("POST", "/api/upgrade/acknowledge", "UPGRADE_ACKNOWLEDGE"),
    ("GET", "/api/workspace/network", "NETWORK_ACCESS"),
    ("GET", "/api/decisions", "PENDING_DECISIONS"),
    ("GET", "/api/activity/refusals", "ACTIVITY_REFUSALS"),
    ("POST", "/api/workspace/network", "NETWORK_ACCESS_SET"),
    ("GET", "/api/evidence/consent", "EVIDENCE_CONSENT"),
    ("POST", "/api/evidence/consent", "EVIDENCE_CONSENT_SET"),
    ("POST", "/api/evidence/install", "EVIDENCE_INSTALL"),
    ("GET", "/api/experiments/compare", "EXPERIMENT_COMPARE"),
    ("GET", "/api/experiments/alpha-compare", "EXPERIMENT_ALPHA_COMPARE"),
    ("POST", "/api/experiments/risk-link", "EXPERIMENT_LINK_RISK"),
    ("GET", "/api/experiments/risk-links", "EXPERIMENT_RISK_LINKS"),
    ("GET", "/api/experiments/risk-export", "EXPERIMENT_RISK_EXPORT"),
    ("GET", "/api/research-inputs", "RESEARCH_INPUTS"),
    ("POST", "/api/research-inputs/plan", "RESEARCH_INPUT_PLAN"),
    ("POST", "/api/experiments/training-inputs/plan", "MODEL_TRAINING_INPUT_PLAN"),
    ("POST", "/api/research-strategies/plan", "RESEARCH_STRATEGY_PLAN"),
    ("GET", "/api/research-strategies/controls", "RESEARCH_STRATEGY_CONTROLS"),
    ("GET", "/api/features/controls", "FEATURE_CATALOG_CONTROLS"),
    ("POST", "/api/features/plan", "FEATURE_CATALOG_PLAN"),
    ("GET", "/api/features/readback", "FEATURE_CATALOG_READBACK"),
    ("POST", "/api/features/build", "FEATURE_CATALOG_BUILD"),
    ("GET", "/api/features/build", "FEATURE_CATALOG_BUILD_READBACK"),
    ("POST", "/api/feature-trials", "FEATURE_TRIAL"),
    ("GET", "/api/feature-trials/readback", "FEATURE_TRIAL_READBACK"),
    ("GET", "/api/feature-trials", "FEATURE_TRIALS"),
    ("POST", "/api/research-strategies/prepare", "RESEARCH_STRATEGY_PREPARE"),
    ("GET", "/api/research-strategies/readback", "RESEARCH_STRATEGY_READBACK"),
    ("POST", "/api/research-strategies/install", "RESEARCH_STRATEGY_INSTALL"),
    ("POST", "/api/experiments/training-inputs/prepare", "MODEL_TRAINING_INPUT_PREPARE"),
    (
        "GET",
        "/api/experiments/training-inputs/readback",
        "MODEL_TRAINING_INPUT_READBACK",
    ),
    ("POST", "/api/research-inputs/confirm", "RESEARCH_INPUT_CONFIRM"),
    ("GET", "/api/research-inputs/readback", "RESEARCH_INPUT_READBACK"),
    ("POST", "/api/experiments/draft", "EXPERIMENT_DRAFT"),
    ("GET", "/api/workspace/preparation", "WORKSPACE_PREPARE_READBACK"),
    ("GET", "/api/workspace/data-issues", "DATA_ISSUES"),
    ("POST", "/api/workspace/data-issues/preview", "DATA_ISSUE_PREVIEW"),
    ("POST", "/api/workspace/data-issues/confirm", "DATA_ISSUE_CONFIRM"),
    ("POST", "/api/workspace/data-issues/delegate", "DATA_ISSUE_DELEGATE"),
    ("POST", "/api/workspace/data-issues/revoke", "DATA_ISSUE_REVOKE"),
    ("POST", "/api/workspace/preparation/plan", "WORKSPACE_PREPARE_PLAN"),
    ("POST", "/api/workspace/preparation/confirm", "WORKSPACE_PREPARE_CONFIRM"),
    ("GET", "/api/workspace/storage/cap", "STORAGE_CAP_SHOW"),
    ("POST", "/api/workspace/storage/cap", "STORAGE_CAP_SET"),
    ("GET", "/api/workspace/storage", "STORAGE_READBACK"),
    ("POST", "/api/workspace/storage/plan", "STORAGE_PLAN"),
    ("POST", "/api/workspace/storage/confirm", "STORAGE_CONFIRM"),
    ("POST", "/api/workspace/storage/pin", "STORAGE_PIN"),
    ("POST", "/api/workspace/storage/evidence-rebuild", "STORAGE_EVIDENCE_REBUILD"),
    ("GET", "/api/experiments/controls", "EXPERIMENT_CONTROLS"),
    ("GET", "/api/experiments", "EXPERIMENTS"),
    ("POST", "/api/experiments/plan", "EXPERIMENT_PLAN"),
    ("GET", "/api/experiments/preview", "EXPERIMENT_PREVIEW_READBACK"),
    ("POST", "/api/experiments/run", "EXPERIMENT_RUN"),
    ("POST", "/api/experiments/promote", "EXPERIMENT_PROMOTE"),
    ("POST", "/api/experiments/continue", "EXPERIMENT_CONTINUE"),
    ("GET", "/api/experiments/readback", "EXPERIMENT_READBACK"),
    ("GET", "/api/experiments/summary", "EXPERIMENT_SUMMARY"),
    ("POST", "/api/experiments/replay", "EXPERIMENT_REPLAY"),
    ("GET", "/api/experiments/export", "EXPERIMENT_EXPORT"),
    ("GET", "/api/experiments/curation", "EXPERIMENT_CURATION"),
    ("POST", "/api/experiments/curation", "EXPERIMENT_CURATE"),
    ("POST", "/api/experiments/handoff", "EXPERIMENT_HANDOFF_PREVIEW"),
    ("GET", "/api/experiments/foundations", "EXPERIMENT_FOUNDATIONS"),
    ("POST", "/api/experiments/foundations/preview", "EXPERIMENT_FOUNDATION_PREVIEW"),
    ("POST", "/api/experiments/foundations/seal", "EXPERIMENT_FOUNDATION_SEAL"),
    ("GET", "/api/experiments/foundations/readback", "EXPERIMENT_FOUNDATION_READBACK"),
    ("GET", "/api/experiments/foundations/summary", "EXPERIMENT_FOUNDATION_SUMMARY"),
    ("GET", "/api/experiments/foundations/export", "EXPERIMENT_FOUNDATION_EXPORT"),
    ("POST", "/api/experiments/foundations/draft", "EXPERIMENT_FOUNDATION_DRAFT"),
    ("POST", "/api/experiments/portfolio-draft", "EXPERIMENT_PORTFOLIO_DRAFT"),
    # The CPU budget: how much of the machine a book's preparation uses, the operator's;
    # the Task queue's waiting places are set on the same route.
    ("GET", "/api/workspace/cpu-budget", "CPU_BUDGET_SHOW"),
    ("POST", "/api/workspace/cpu-budget", "CPU_BUDGET_SET"),
    ("GET", "/api/workspace/backup", "WORKSPACE_BACKUPS"),
    # A Goal or Team page reads the bound Sessions' usage when it opens.
    ("POST", "/api/workspace/session-usage", "SESSION_USAGE_READ"),
    ("GET", "/api/workspace/usage-reading", "USAGE_READING"),
    ("POST", "/api/workspace/usage-reading", "USAGE_READING_SET"),
    ("POST", "/api/workspace/backup", "WORKSPACE_BACKUP"),
    ("GET", "/api/models", "MODEL_EXTENSIONS"),
    ("POST", "/api/models/activate", "MODEL_ACTIVATE"),
    ("POST", "/api/models/deactivate", "MODEL_DEACTIVATE"),
    # A person runs a reviewed research book's strategy forward, or stops it (LS1, OW12).
    ("POST", "/api/strategy/activate", "STRATEGY_ACTIVATE"),
    ("POST", "/api/strategy/deactivate", "STRATEGY_DEACTIVATE"),
    ("GET", "/api/feature-research", "FEATURE_EXTENSIONS"),
    ("GET", "/api/feature-research/review", "FEATURE_REVIEW"),
    ("POST", "/api/feature-research/activate", "FEATURE_ACTIVATE"),
    ("POST", "/api/feature-research/deactivate", "FEATURE_DEACTIVATE"),
    # What the interface reads: the goals a person keeps (U23, U32; taking, submitting
    # and continuing stay an agent's, through the client's route), the saved studies'
    # sweep and Guanyin's incidents with their remedies.
    ("GET", "/api/goals", "GOAL_LIST"),
    ("GET", "/api/committee", "COMMITTEE_READ"),
    ("GET", "/api/goals/schema", "GOAL_SCHEMA"),
    ("GET", "/api/goals/show", "GOAL_SHOW"),
    ("GET", "/api/goals/narrative", "GOAL_NARRATIVE"),
    ("GET", "/api/goals/profile", "GOAL_PROFILE"),
    ("GET", "/api/goals/reference", "GOAL_REFERENCE"),
    ("GET", "/api/goals/export", "GOAL_EXPORT"),
    ("POST", "/api/goals/open", "GOAL_OPEN"),
    ("POST", "/api/goals/revise", "GOAL_REVISE"),
    ("POST", "/api/goals/attach", "GOAL_ATTACH"),
    ("POST", "/api/goals/note", "GOAL_NOTE"),
    ("POST", "/api/goals/abandon", "GOAL_ABANDON"),
    ("POST", "/api/experiments/verify-all", "EXPERIMENT_VERIFY_ALL"),
    ("GET", "/api/tasks/incidents", "TASK_INCIDENTS"),
    ("POST", "/api/tasks/remediate", "TASK_REMEDIATE"),
)
"""Every operation the Local Web serves by name, with its method and path: the one table the
Host routes and the session answer names (U13, OW10), so a page turns a next request into its
route without a map of its own."""

HANDLED_OPERATION_ROUTES: Final[tuple[tuple[str, str, str], ...]] = (
    ("GET", "/api/controls", "CONTROLS"),
    ("GET", "/api/workbench/portfolio", "PORTFOLIO_READBACK"),
    ("POST", "/api/data-update/plan", "DATA_UPDATE_PLAN"),
    ("POST", "/api/data-update/confirm", "DATA_CHANGE_CONFIRM"),
    ("POST", "/api/data-update/run", "DATA_UPDATE_RUN"),
    ("POST", "/api/strategy-score/plan", "STRATEGY_SCORE_PLAN"),
    ("POST", "/api/strategy-score/run", "STRATEGY_SCORE_RUN"),
    ("POST", "/api/research-update/plan", "RESEARCH_UPDATE_PLAN"),
    ("POST", "/api/research-update/run", "RESEARCH_UPDATE_RUN"),
    ("POST", "/api/portfolio-update/plan", "PORTFOLIO_UPDATE_PLAN"),
    ("POST", "/api/portfolio-update/run", "PORTFOLIO_UPDATE_RUN"),
    ("POST", "/api/strategy-calibration/plan", "STRATEGY_CALIBRATION_PLAN"),
    ("POST", "/api/strategy-calibration/run", "STRATEGY_CALIBRATION_RUN"),
    ("POST", "/api/plan", "PLAN"),
    ("POST", "/api/run", "RUN"),
    ("POST", "/api/evidence-refresh", "EVIDENCE_REFRESH"),
    ("POST", "/api/evidence-select", "EVIDENCE_SELECT"),
    ("POST", "/api/cro-review", "CRO_REVIEW"),
    ("GET", "/api/evidence-cro/export", "EVIDENCE_CRO_EXPORT"),
    ("GET", "/api/research-update/automation", "RESEARCH_UPDATE_AUTOMATION_READBACK"),
    ("POST", "/api/research-update/automation", "RESEARCH_UPDATE_AUTOMATION_CONFIGURE"),
)
"""Dedicated handlers' routes, also published in the session's operation table."""

_HANDLED_ROUTE: Final[dict[str, tuple[str, str]]] = {
    operation: (method, path) for method, path, operation in HANDLED_OPERATION_ROUTES
}


def read_workbench_portfolio(
    operations: PortfolioResearchOperations,
    query: Mapping[str, list[str]],
    *,
    caller: OperationCaller = "HUMAN",
) -> dict[str, Any]:
    """Translate the browser's exact selectors into the shared Portfolio read."""
    if set(query) - {"task_id", "portfolio_session", "performance", "scope"}:
        raise LocalWebError("workbench.query_field_unknown")
    task = _optional(query, "task_id")
    if task is None:
        raise LocalWebError("workbench.task_id_required")
    performance = _optional(query, "performance")
    if performance not in {None, "latest"}:
        raise LocalWebError("workbench.performance_selection_invalid")
    scope = _optional(query, "scope")
    if scope not in {None, "holdings"}:
        raise LocalWebError("local_web.query_parameter_invalid:scope")
    request = PortfolioResearchOperationRequest(
        operation="PORTFOLIO_READBACK",
        task_id=UUID(task),
        portfolio_session=_optional(query, "portfolio_session"),
        performance="latest" if performance == "latest" else None,
        portfolio_scope="holdings" if scope == "holdings" else None,
    )
    body = (
        operations.execute(request)
        if caller == "HUMAN"
        else operations.execute(request, caller=caller)
    )
    if caller == "SERVICE_AUTOMATION" and (returned_status(body) or "").startswith("REFUSED"):
        raise LocalWebError(safe_failure_code(refusal_code(body)) or FAILURE_DETAIL_WITHHELD)
    return body


def mount_sector_development_publish(
    web: LocalWebApplication, workspace: Path, mutation_gate: WorkspaceMutationGate
) -> None:
    """Publish and verify through the served workspace's existing Host writer."""

    @web.route("POST", SECTOR_DEVELOPMENT_PUBLISH_PATH, mutates=True, external_client=True)
    def publish(_query: Mapping[str, list[str]], payload: dict[str, Any]) -> object:
        from alphalattice.control.workspace_runtime.artifacts import (  # lazy: publish
            ArtifactResolver,
        )
        from alphalattice.foundation.causal_outcomes.execution.readers import (  # lazy: publish
            CausalExecutionOutcomeDevelopmentReader,
        )
        from alphalattice.investment.sector_research.experiments.service import (  # lazy: publish
            SectorExperimentRequest,
            SectorResearchDevelopmentService,
        )

        try:
            request = TypeAdapter(SectorExperimentRequest).validate_python(payload, extra="forbid")
        except ValidationError as error:
            raise LocalWebError("local_client.request_invalid") from error

        artifacts = workspace / "artifacts"
        service = SectorResearchDevelopmentService(
            artifact_root=artifacts,
            outcome_reader=CausalExecutionOutcomeDevelopmentReader(artifacts),
            resolver=ArtifactResolver(artifacts),
        )
        try:
            with mutation_gate.hold():
                published = service.publish(request)
                lineage = service.verify(experiment_hash=published.experiment.experiment_hash)
        except FileNotFoundError as error:
            if owner_failure_code(error) is not None:
                raise
            return {
                "action": "UNRESOLVED",
                "failure_code": "sector_research.evidence_not_published",
                "current_pointer_writes": 0,
            }

        return {
            "action": "PUBLISHED",
            "experiment_hash": published.experiment.experiment_hash,
            "sector_target_evidence_hash": published.target_evidence.evidence_hash,
            "forecast_surface_hash": published.surface.surface_hash,
            "evaluation_hash": published.evaluation.evaluation_hash,
            "causal_outcome_snapshot_hash": lineage.outcome_snapshot_hash,
            "outcome_method_binding_hash": lineage.outcome_method_binding_hash,
            "maturity_lag_sessions": lineage.maturity_lag_sessions,
            "method_id": published.surface.recipe.method_id,
            "formation_count": len(published.formation_sessions),
            "forecast_formation_count": len(published.forecast_formation_sessions),
            "sector_count": len(published.ordered_sectors),
            "current_pointer_writes": 0,
            "alpha_model_fits": 0,
        }


def build_local_web_application(
    *,
    operations: PortfolioResearchOperations,
    activity: WorkspaceActivity | None = None,
) -> LocalWebApplication:
    """Register the routes. Every one of them is a projection or a command."""
    web = workspace_web(
        operations.workspace_session.workspace, activity.instance if activity else None
    )
    if activity is not None:
        _activity_routes(web, activity)
    _native_session_routes(web, operations)
    session = operations.workspace_session
    mount_sector_development_publish(web, session.workspace, session.mutation_gate)

    # Every hash and id answered to a client, which reads them back from a beginning.
    references = ReferenceLedger(operations.workspace_session.workspace)

    def refuse_forwarded_serve(_arguments: list[str]) -> int:
        raise ValueError("the Host does not serve its own launcher through CLI_INVOKE")

    @web.route("POST", "/api/client/operations", mutates=True, external_client=True)
    def external_operation(_query: Mapping[str, list[str]], payload: dict[str, Any]) -> object:
        if payload.get("operation") == CLI_OPERATION:
            from alphalattice.interface.local_application.cli import main  # lazy: CLI dispatch

            return invoke_cli(
                payload, session.workspace, lambda argv: main(argv, serve=refuse_forwarded_serve)
            )
        try:
            document = validate_request_document(payload)
        except ValidationError as error:
            if activity is not None:
                # A request that does not meet its schema is counted like any refusal, under
                # the operation it named (AC, OP14).
                named = payload.get("operation")
                activity.refused(
                    named
                    if isinstance(named, str)
                    and named in get_args(cast(Any, PortfolioResearchOperation).__value__)
                    else "UNKNOWN",
                    "local_client.request_invalid",
                    caller="EXTERNAL_AUTOMATION",
                )
            return {
                "status": "REFUSED",
                **located_failure(error, "local_client.request_invalid", request=payload),
            }
        answer = operations.execute(document.to_operation_request(), caller="EXTERNAL_AUTOMATION")
        references.record(answer)
        return answer

    serve_assets(web, ASSET_ROOT)

    # The token goes only to the browser holding the cookie the launch URL set (HB).
    @web.route("GET", "/api/session", session=True)
    def _session(_query: Mapping[str, list[str]], _payload: dict[str, Any]) -> object:
        return {
            **operations.session_projection(
                session_token=web.session_token, include_context=_query.get("context") == ["1"]
            ),
            # Each operation's route, from the table the Host serves.
            "routes": {
                operation: {"method": method, "path": path}
                for method, path, operation in (*OPERATION_ROUTES, *HANDLED_OPERATION_ROUTES)
            },
        }

    @web.route("GET", "/api/workbench/portfolio")
    def _workbench_portfolio(query: Mapping[str, list[str]], _payload: dict[str, Any]) -> object:
        return read_workbench_portfolio(operations, query)

    @web.route(*_HANDLED_ROUTE["DATA_UPDATE_PLAN"], mutates=True)
    def _data_plan(_query: Mapping[str, list[str]], payload: dict[str, Any]) -> object:
        if payload:
            raise LocalWebError("workspace_data_update.no_plan_controls")
        return operations.execute(PortfolioResearchOperationRequest(operation="DATA_UPDATE_PLAN"))

    @web.route(*_HANDLED_ROUTE["DATA_UPDATE_RUN"], mutates=True)
    def _data_run(_query: Mapping[str, list[str]], payload: dict[str, Any]) -> object:
        if set(payload) != {"update_plan_hash"} or not isinstance(payload["update_plan_hash"], str):
            raise LocalWebError("workspace_data_update.plan_hash_required")
        return operations.execute(
            PortfolioResearchOperationRequest(
                operation="DATA_UPDATE_RUN", update_plan_hash=payload["update_plan_hash"]
            )
        )

    _maintenance_routes(web, operations)
    _planning_routes(web, operations)

    @web.route("GET", "/api/status")
    def _status(query: Mapping[str, list[str]], _payload: dict[str, Any]) -> object:
        return _task_read_response(
            operations.execute(
                PortfolioResearchOperationRequest(
                    operation="STATUS",
                    task_id=_task_uuid(_one(query, "task_id")),
                    wait_seconds=_wait_seconds(query.get("wait_seconds")),
                )
            )
        )

    _result_routes(web, operations)
    _dispatch_routes(web, operations)

    def case_route(operation: str):  # type: ignore[no-untyped-def]
        """The case page's call, answered by the goal it names until a later interface update."""

        def handle(query: Mapping[str, list[str]], payload: dict[str, Any]) -> object:
            fields = {**payload, **{key: _one(query, key) for key in query}}
            fields.pop("operation", None)
            try:
                request = case_request(operation, fields)
            except (TypeError, ValueError) as error:
                raise LocalWebError(owner_failure_code(error) or str(error)) from error
            body = operations.execute(request)
            return case_answer(operation, body, operations.goals.store)

        return handle

    for method, path, operation in CASE_ROUTES:
        web.route(method, path, mutates=method == "POST")(case_route(operation))

    return web


def _agent_telemetry_routes(
    web: LocalWebApplication, reader: NativeUsageReader | None, workspace_id: str
) -> None:
    """Receive native exports on the Host's existing authenticated loopback socket."""
    for signal in ("logs", "traces", "metrics"):

        @web.route(
            "POST", f"/v1/{signal}", mutates=True, external_client=True, otlp=(signal, workspace_id)
        )
        def receive(_query: Mapping[str, list[str]], payload: dict[str, Any]) -> object:
            if reader is None:
                raise LocalWebError("native_bridge.lead_usage_read_failed")
            reader.receive(payload["rows"])
            content_type = payload["content_type"]
            return LocalWebResponse(
                HTTPStatus.OK, b"{}" if content_type == "application/json" else b"", content_type
            )


def _maintenance_routes(web: LocalWebApplication, operations: PortfolioResearchOperations) -> None:
    @web.route(
        "GET",
        "/api/data-update",
    )
    def _data_readback(query: Mapping[str, list[str]], _payload: dict[str, Any]) -> object:
        task = query.get("task_id")
        return operations.execute(
            PortfolioResearchOperationRequest(
                operation="DATA_UPDATE_READBACK",
                task_id=UUID(_one(query, "task_id")) if task else None,
            )
        )

    @web.route(*_HANDLED_ROUTE["STRATEGY_SCORE_PLAN"], mutates=True)
    def _score_plan(_query: Mapping[str, list[str]], payload: dict[str, Any]) -> object:
        if not {"strategy_package_id"}.issubset(payload) or set(payload) - {
            "strategy_package_id",
            "formation_session",
            "component_id",
        }:
            raise LocalWebError("strategy_score.plan_fields_invalid")
        if any(not isinstance(value, str) for value in payload.values()):
            raise LocalWebError("strategy_score.plan_fields_invalid")
        return operations.execute(
            PortfolioResearchOperationRequest(operation="STRATEGY_SCORE_PLAN", **payload)
        )

    @web.route(*_HANDLED_ROUTE["STRATEGY_SCORE_RUN"], mutates=True)
    def _score_run(_query: Mapping[str, list[str]], payload: dict[str, Any]) -> object:
        if set(payload) != {"score_plan_hash"} or not isinstance(payload["score_plan_hash"], str):
            raise LocalWebError("strategy_score.plan_hash_required")
        return operations.execute(
            PortfolioResearchOperationRequest(operation="STRATEGY_SCORE_RUN", **payload)
        )

    @web.route("GET", "/api/strategy-score")
    def _score_readback(query: Mapping[str, list[str]], _payload: dict[str, Any]) -> object:
        return operations.execute(
            PortfolioResearchOperationRequest(
                operation="STRATEGY_SCORE_READBACK",
                task_id=UUID(_one(query, "task_id")) if query.get("task_id") else None,
                strategy_package_id=_one(query, "strategy_package_id")
                if query.get("strategy_package_id")
                else None,
            )
        )

    @web.route(*_HANDLED_ROUTE["RESEARCH_UPDATE_AUTOMATION_READBACK"])
    def _automation_readback(_query: Mapping[str, list[str]], _payload: dict[str, Any]) -> object:
        return operations.execute(
            PortfolioResearchOperationRequest(operation="RESEARCH_UPDATE_AUTOMATION_READBACK")
        )

    @web.route(*_HANDLED_ROUTE["RESEARCH_UPDATE_AUTOMATION_CONFIGURE"], mutates=True)
    def _automation_configure(_query: Mapping[str, list[str]], payload: dict[str, Any]) -> object:
        if (
            set(payload) != {"automation_enabled", "automation_package_ids"}
            or type(payload["automation_enabled"]) is not bool
            or not isinstance(payload["automation_package_ids"], list)
            or any(not isinstance(v, str) for v in payload["automation_package_ids"])
        ):
            raise LocalWebError("research_update.automation_fields_invalid")
        return operations.execute(
            PortfolioResearchOperationRequest(
                operation="RESEARCH_UPDATE_AUTOMATION_CONFIGURE",
                automation_enabled=payload["automation_enabled"],
                automation_package_ids=tuple(payload["automation_package_ids"]),
            )
        )

    @web.route(*_HANDLED_ROUTE["DATA_CHANGE_CONFIRM"], mutates=True)
    def _data_change_confirm(_query: Mapping[str, list[str]], payload: dict[str, Any]) -> object:
        if set(payload) != {"update_plan_hash"} or not isinstance(payload["update_plan_hash"], str):
            raise LocalWebError("workspace_data_update.proposal_required")
        return operations.execute(
            PortfolioResearchOperationRequest(operation="DATA_CHANGE_CONFIRM", **payload)
        )

    @web.route(*_HANDLED_ROUTE["RESEARCH_UPDATE_PLAN"], mutates=True)
    def _research_update_plan(_query: Mapping[str, list[str]], payload: dict[str, Any]) -> object:
        if (
            "strategy_package_id" not in payload
            or set(payload) - {"strategy_package_id", "observed_through"}
            or any(not isinstance(v, str) for v in payload.values())
        ):
            raise LocalWebError("research_update.plan_fields_invalid")
        return operations.execute(
            PortfolioResearchOperationRequest(operation="RESEARCH_UPDATE_PLAN", **payload)
        )

    @web.route(*_HANDLED_ROUTE["RESEARCH_UPDATE_RUN"], mutates=True)
    def _research_update_run(_query: Mapping[str, list[str]], payload: dict[str, Any]) -> object:
        if set(payload) != {"update_plan_hash"} or not isinstance(payload["update_plan_hash"], str):
            raise LocalWebError("research_update.plan_hash_required")
        return operations.execute(
            PortfolioResearchOperationRequest(operation="RESEARCH_UPDATE_RUN", **payload)
        )

    @web.route("GET", "/api/research-update")
    def _research_update_readback(
        query: Mapping[str, list[str]], _payload: dict[str, Any]
    ) -> object:
        return operations.execute(
            PortfolioResearchOperationRequest(
                operation="RESEARCH_UPDATE_READBACK",
                task_id=UUID(_one(query, "task_id")) if query.get("task_id") else None,
                strategy_package_id=_one(query, "strategy_package_id")
                if query.get("strategy_package_id")
                else None,
            )
        )


def _planning_routes(web: LocalWebApplication, operations: PortfolioResearchOperations) -> None:
    def _spec_document(payload: Mapping[str, Any]) -> dict[str, object]:
        document = payload.get("spec")
        if document is None:
            document = {}
        if not isinstance(document, dict):
            raise LocalWebError("local_web.spec_not_an_object")
        return cast(dict[str, object], document)

    @web.route(*_HANDLED_ROUTE["PORTFOLIO_UPDATE_PLAN"], mutates=True)
    def _portfolio_update_plan(_query: Mapping[str, list[str]], payload: dict[str, Any]) -> object:
        if (
            "strategy_package_id" not in payload
            or set(payload) - {"strategy_package_id", "prepared_input_hash", "observed_through"}
            or any(
                not isinstance(v, str) and not (key == "prepared_input_hash" and v is None)
                for key, v in payload.items()
            )
        ):
            raise LocalWebError("portfolio_update.plan_fields_invalid")
        return operations.execute(
            PortfolioResearchOperationRequest(operation="PORTFOLIO_UPDATE_PLAN", **payload)
        )

    @web.route(*_HANDLED_ROUTE["PORTFOLIO_UPDATE_RUN"], mutates=True)
    def _portfolio_update_run(_query: Mapping[str, list[str]], payload: dict[str, Any]) -> object:
        if set(payload) != {"update_plan_hash"} or not isinstance(payload["update_plan_hash"], str):
            raise LocalWebError("portfolio_update.plan_hash_required")
        return operations.execute(
            PortfolioResearchOperationRequest(operation="PORTFOLIO_UPDATE_RUN", **payload)
        )

    @web.route("GET", "/api/portfolio-update")
    def _portfolio_update_readback(
        query: Mapping[str, list[str]], _payload: dict[str, Any]
    ) -> object:
        return operations.execute(
            PortfolioResearchOperationRequest(
                operation="PORTFOLIO_UPDATE_READBACK",
                task_id=UUID(_one(query, "task_id")) if query.get("task_id") else None,
                strategy_package_id=_one(query, "strategy_package_id")
                if query.get("strategy_package_id")
                else None,
            )
        )

    @web.route(*_HANDLED_ROUTE["STRATEGY_CALIBRATION_PLAN"], mutates=True)
    def _calibration_plan(_query: Mapping[str, list[str]], payload: dict[str, Any]) -> object:
        if set(payload) != {"strategy_package_id", "score_snapshot_hash"} or any(
            not isinstance(v, str) for v in payload.values()
        ):
            raise LocalWebError("portfolio_calibration.plan_fields_invalid")
        return operations.execute(
            PortfolioResearchOperationRequest(operation="STRATEGY_CALIBRATION_PLAN", **payload)
        )

    @web.route(*_HANDLED_ROUTE["STRATEGY_CALIBRATION_RUN"], mutates=True)
    def _calibration_run(_query: Mapping[str, list[str]], payload: dict[str, Any]) -> object:
        if set(payload) != {"calibration_plan_hash"} or not isinstance(
            payload["calibration_plan_hash"], str
        ):
            raise LocalWebError("portfolio_calibration.plan_hash_required")
        return operations.execute(
            PortfolioResearchOperationRequest(operation="STRATEGY_CALIBRATION_RUN", **payload)
        )

    @web.route("GET", "/api/strategy-calibration")
    def _calibration_readback(query: Mapping[str, list[str]], _payload: dict[str, Any]) -> object:
        return operations.execute(
            PortfolioResearchOperationRequest(
                operation="STRATEGY_CALIBRATION_READBACK",
                task_id=UUID(_one(query, "task_id")) if query.get("task_id") else None,
                strategy_package_id=_one(query, "strategy_package_id")
                if query.get("strategy_package_id")
                else None,
            )
        )

    @web.route(*_HANDLED_ROUTE["CONTROLS"])
    def _controls(query: Mapping[str, list[str]], _payload: dict[str, Any]) -> object:
        """The shared catalog, projected through the selected package's surface.

        A control the package froze is returned as a *fact* with the value it is
        fixed at, not as a widget the page will offer and the compiler will then
        refuse. That mismatch is the defect this route exists to close: the
        successor's page advertised the predecessor's `mu.iv1` default and every
        holdings control beside it.
        """

        selected = query.get("strategy_package_id")
        return operations.execute(
            PortfolioResearchOperationRequest(
                operation="CONTROLS",
                strategy_package_id=(
                    None if selected is None else _one(query, "strategy_package_id")
                ),
            )
        )

    @web.route(*_HANDLED_ROUTE["PLAN"], mutates=True)
    def _plan(_query: Mapping[str, list[str]], payload: dict[str, Any]) -> object:
        return operations.execute(
            PortfolioResearchOperationRequest(operation="PLAN", spec=_spec_document(payload))
        )

    @web.route(*_HANDLED_ROUTE["RUN"], mutates=True)
    def _run(_query: Mapping[str, list[str]], payload: dict[str, Any]) -> object:
        return operations.execute(
            PortfolioResearchOperationRequest(operation="RUN", spec=_spec_document(payload))
        )


def _result_routes(web: LocalWebApplication, operations: PortfolioResearchOperations) -> None:
    @web.route("POST", "/api/cancel", mutates=True)
    def _cancel(_query: Mapping[str, list[str]], payload: dict[str, Any]) -> object:
        task_id = payload.get("task_id")
        if not isinstance(task_id, str):
            raise LocalWebError("local_web.task_id_required")
        if set(payload) - {"task_id", "expected_task_hash"}:
            raise LocalWebError("local_web.operation_fields_invalid")
        return operations.execute(
            PortfolioResearchOperationRequest(
                operation="CANCEL",
                task_id=UUID(task_id),
                expected_task_hash=payload.get("expected_task_hash"),
            )
        )

    @web.route("GET", "/api/results")
    def _results(query: Mapping[str, list[str]], _payload: dict[str, Any]) -> object:
        task = _optional(query, "task_id")
        return operations.execute(
            PortfolioResearchOperationRequest(
                operation="RESULTS", task_id=_task_uuid(task) if task else None
            )
        )

    @web.route("GET", "/api/report")
    def _report(query: Mapping[str, list[str]], _payload: dict[str, Any]) -> object:
        return operations.execute(
            PortfolioResearchOperationRequest(
                operation="REPORT",
                result_hash=_one(query, "result_hash"),
                portfolio_session=_optional(query, "portfolio_session"),
            )
        )

    @web.route("GET", "/api/compare")
    def _compare(query: Mapping[str, list[str]], _payload: dict[str, Any]) -> object:
        return operations.execute(
            PortfolioResearchOperationRequest(
                operation="COMPARE",
                left_result_hash=_one(query, "left"),
                right_result_hash=_one(query, "right"),
            )
        )

    @web.route("POST", "/api/freeze", mutates=True)
    def _freeze(_query: Mapping[str, list[str]], payload: dict[str, Any]) -> object:
        result_hash = payload.get("result_hash")
        if not isinstance(result_hash, str):
            raise LocalWebError("local_web.result_hash_required")
        return operations.execute(
            PortfolioResearchOperationRequest(operation="FREEZE", result_hash=result_hash)
        )

    @web.route("GET", "/api/finalization")
    def _finalization(query: Mapping[str, list[str]], _payload: dict[str, Any]) -> object:
        return operations.execute(
            PortfolioResearchOperationRequest(
                operation="FINALIZATION",
                candidate_hash=_one(query, "candidate_hash"),
            )
        )

    @web.route("GET", "/api/evidence-cro")
    def _evidence_cro(query: Mapping[str, list[str]], _payload: dict[str, Any]) -> object:
        """The Evidence & CRO section beside the selected book.

        Answered by the same operation owner the Agent tool uses. With no book
        named the owner picks the workspace default: a given handoff, else the
        most recent completed result.
        """

        return operations.execute(
            PortfolioResearchOperationRequest(
                operation="EVIDENCE_CRO",
                review_publication_hash=_optional(query, "review_publication_hash"),
                **_citation_fields(query),
                **_reading_fields(query),
                **_review_fields(
                    {name: _optional(query, name) for name in _REVIEW_SELECTOR_FIELDS}
                ),
            )
        )

    @web.route(*_HANDLED_ROUTE["EVIDENCE_REFRESH"], mutates=True)
    def _evidence_refresh(_query: Mapping[str, list[str]], payload: dict[str, Any]) -> object:
        return operations.execute(
            PortfolioResearchOperationRequest(
                operation="EVIDENCE_REFRESH",
                **_review_fields(payload),
            )
        )

    @web.route(*_HANDLED_ROUTE["EVIDENCE_SELECT"], mutates=True)
    def _evidence_select(_query: Mapping[str, list[str]], payload: dict[str, Any]) -> object:
        chosen = payload.get("analysis_publication_hash")
        if not isinstance(chosen, str):
            raise LocalWebError("local_web.analysis_publication_hash_required")
        return operations.execute(
            PortfolioResearchOperationRequest(
                operation="EVIDENCE_SELECT",
                analysis_publication_hash=chosen,
                **_review_fields(payload),
            )
        )

    @web.route(*_HANDLED_ROUTE["CRO_REVIEW"], mutates=True)
    def _cro_review(_query: Mapping[str, list[str]], payload: dict[str, Any]) -> object:
        return operations.execute(
            PortfolioResearchOperationRequest(
                operation="CRO_REVIEW",
                **_review_fields(payload),
            )
        )


def _dispatch_routes(
    web: LocalWebApplication,
    operations: PortfolioResearchOperations,
) -> None:
    @web.route(*_HANDLED_ROUTE["EVIDENCE_CRO_EXPORT"])
    def _review_export(query: Mapping[str, list[str]], _payload: dict[str, Any]) -> object:
        return operations.execute(
            PortfolioResearchOperationRequest(
                operation="EVIDENCE_CRO_EXPORT",
                review_publication_hash=_optional(query, "review_publication_hash"),
                prior_review_publication_hash=_optional(query, "prior_review_publication_hash"),
                **_citation_fields(query),
                **_review_fields(
                    {name: _optional(query, name) for name in _REVIEW_SELECTOR_FIELDS}
                ),
            )
        )

    @web.route("GET", "/api/export")
    def _export(query: Mapping[str, list[str]], _payload: dict[str, Any]) -> object:
        return operations.execute(
            PortfolioResearchOperationRequest(
                operation="EXPORT", result_hash=_one(query, "result_hash")
            )
        )

    @web.route("GET", "/report")
    @web.route("GET", "/api/client/report", external_client=True)
    def _html(query: Mapping[str, list[str]], _payload: dict[str, Any]) -> object:
        """The page the executor already sealed, read back and served as-is.

        Not re-rendered. There is one renderer and it ran at publication time;
        rendering again here would be a second report owner with a different
        input set.
        """

        service = operations.service
        if service is None:
            raise LocalWebError("research_workspace.strategy_not_installed")
        page = service.open_html(_one(query, "result_hash"))
        return LocalWebResponse(
            HTTPStatus.OK,
            page.encode("utf-8"),
            "text/html; charset=utf-8",
            # The sealed page styles itself inline, and the default policy forbids inline style.
            # Naming its exact style by hash keeps the page byte-identical and the policy narrow.
            content_security_policy=report_policy(page),
        )

    def operation_route(operation: Any):  # type: ignore[no-untyped-def]
        def handle(query: Mapping[str, list[str]], payload: dict[str, Any]) -> object:
            fields = dict(payload)
            fields.update({key: _one(query, key) for key in query})
            # A request document the product composed names its operation;
            # this route is that operation, and never another one.
            named = fields.pop("operation", None)
            if named is not None and named != operation:
                raise LocalWebError(f"local_web.operation_mismatch:{named}")
            for name in (
                "task_id",
                "origin_task_id",
                "recovery_task_id",
                "factor_task_id",
                "risk_task_id",
                "left_task_id",
                "right_task_id",
            ):
                if name in fields:
                    fields[name] = _task_uuid(str(fields[name]))
            if "goal_id" in fields:
                # The request is a dataclass: a route gives it the goal's UUID, as its Tasks'.
                try:
                    fields["goal_id"] = UUID(str(fields["goal_id"]))
                except ValueError as error:
                    raise LocalWebError(
                        "local_web.goal_id_invalid",
                        next_action="COPY_THE_GOAL_ID_FROM_THE_GOAL_LIST",
                    ) from error
            for name in (
                "history_limit",
                "experience_limit",
                "delivery_part",
                "delivery_budget_bytes",
                "session_limit",
                "window_limit",
                "view_last_days",
                "ledger_page",
                "citation_page",
                "documents_page",
                "extensions_page",
            ):
                # A null slot is the operation's to refuse by name.
                if fields.get(name) is not None:
                    try:
                        fields[name] = int(fields[name])
                    except (TypeError, ValueError) as error:
                        raise LocalWebError("local_web.operation_fields_invalid") from error
            try:
                request = PortfolioResearchOperationRequest(operation=operation, **fields)
            except TypeError as error:
                raise LocalWebError("local_web.operation_fields_invalid") from error
            body = operations.execute(request)
            if operation == "DATA_ISSUE_PREVIEW" and body.get("status") == "REFUSED":
                # Keep the route's 400 refusal while retaining the Data owner's
                # typed next request for both CLI and browser readers.
                return LocalWebResponse(
                    HTTPStatus.BAD_REQUEST,
                    json.dumps(body, sort_keys=True).encode("utf-8"),
                    "application/json; charset=utf-8",
                )
            return _task_read_response(body) if operation == "TASK_RECOVERY" else body

        return handle

    # A route's write check follows its operation in the registry, never a flag written
    # beside it (HB): a POST whose operation is not one of the registry's reads
    # takes it. A GET never writes.
    for method, path, operation in OPERATION_ROUTES:
        web.route(method, path, mutates=method == "POST" and operation not in READ_OPERATIONS)(
            operation_route(operation)
        )


def _native_session_refusal(code: str) -> dict[str, Any]:
    return {"status": "REFUSED", "failure_code": code, **refusal_words(code)}


def _native_session_routes(
    web: LocalWebApplication, operations: PortfolioResearchOperations
) -> None:
    """Register the served workspace's native binding and usage routes."""
    workspace = operations.workspace_session.workspace.resolve()

    @web.route("POST", "/api/client/session/bind", mutates=True, external_client=True)
    def bind(_query: Mapping[str, list[str]], payload: dict[str, Any]) -> object:
        try:
            request = native_setup.NativeSessionBindRequest.model_validate(payload)
        except ValidationError:
            return _native_session_refusal("native_bridge.request_invalid")
        provenance = REQUEST_PROVENANCE.get()
        if provenance is None or provenance.vendor is None or provenance.session is None:
            return _native_session_refusal("local_client.session_unnamed")
        project: Path | None = None
        try:
            project = native_setup.admitted_session_project(
                workspace, request.project, provenance.vendor
            )
            operations.native_projects.add(project)
            return native_setup.bind_session(
                project,
                host=provenance.vendor,
                session_id=provenance.session,
                workspace=workspace,
                usage=request.usage,
            )
        except NativeBridgeError as error:
            return _native_session_refusal(str(error))
        except OSError as error:
            if owner_failure_code(error) is not None:
                raise
            return native_setup.bound_in_workspace(error, project, workspace, request)

    web.route("POST", "/api/client/session/event", mutates=True, external_client=True)(
        partial(_native_session_event, operations, workspace)
    )


def _native_session_event(
    operations: PortfolioResearchOperations,
    workspace: Path,
    _query: Mapping[str, list[str]],
    payload: dict[str, Any],
) -> object:
    """Deliver the caller's own usage through its exact binding and Goal authority."""
    try:
        request = native_setup.NativeEventDeliveryRequest.model_validate(payload)
    except ValidationError:
        return _native_session_refusal("native_bridge.request_invalid")
    provenance = REQUEST_PROVENANCE.get()
    if (
        provenance is None
        or provenance.vendor is None
        or provenance.vendor not in HOSTS
        or provenance.session is None
    ):
        return _native_session_refusal("native_bridge.event_scope_invalid")
    project: Path | None = None
    try:
        # The caller's verified binding chooses the Session and workspace, never the body.
        selected_project = request.project
        # The Host's canonical workspace folder is already its binding owner;
        # it is not a configured project, nor a caller-selected alternative.
        if selected_project != autobind_root(workspace):
            selected_project = native_setup.admitted_session_project(
                workspace, selected_project, provenance.vendor
            )
            operations.native_projects.add(selected_project)
        project = selected_project
        found = session_binding(
            selected_project, workspace, (provenance.vendor, provenance.session)
        )
        if found is None:
            raise NativeBridgeError("native_bridge.not_bound")
        bound_project, binding = found
        if bound_project != selected_project:
            raise NativeBridgeError("native_bridge.project_mismatch")
        if binding.workspace.resolve() != workspace:
            raise NativeBridgeError("native_bridge.workspace_mismatch")
        parent_provenance = RequestProvenance(
            vendor=binding.host,
            session=binding.session_id,
            goal_id=provenance.goal_id,
        )
        goal = operations.goals.attributed_goal(parent_provenance)
        goal_id = None if goal is None else str(goal.goal_id)

        def publish(document: dict[str, Any]) -> dict[str, Any]:
            with (
                operations.workspace_session.mutation_gate.hold(),
                operations.goals.store.lock,
            ):
                current = operations.goals.attributed_goal(parent_provenance)
                current_id = None if current is None else str(current.goal_id)
                caller_goal = operations.goals.attributed_goal(provenance)
                if (
                    current_id != goal_id
                    or (goal_id is None and caller_goal is not None)
                    or NativeResearchBinding.read(
                        selected_project, session=(binding.host, binding.session_id)
                    )
                    != binding
                ):
                    return _native_session_refusal("native_bridge.event_scope_invalid")
                token = REQUEST_PROVENANCE.set(
                    RequestProvenance(
                        vendor=provenance.vendor,
                        session=provenance.session,
                        goal_id=goal_id,
                        delegation=provenance.delegation,
                    )
                )
                try:
                    return operations.execute(
                        validate_request_document(
                            {"operation": "EVENT_DECLARE", "event": document}
                        ).to_operation_request(),
                        caller="EXTERNAL_AUTOMATION",
                    )
                finally:
                    REQUEST_PROVENANCE.reset(token)

        # Only a milestone's read of the caller's own Session enters here.
        if request.event != {"source": "native_usage_read"} or (
            provenance.vendor,
            provenance.session,
        ) != (binding.host, binding.session_id):
            return _native_session_refusal("native_bridge.event_scope_invalid")
        return read_session_usage(
            project,
            binding,
            workspace_id=operations.workspace_manifest.workspace_id,
            publish=publish,
            goal_id=goal_id,
        )
    except NativeBridgeError as error:
        return _native_session_refusal(str(error))
    except (KeyError, TypeError, ValueError) as error:
        if owner_failure_code(error) is not None:
            raise
        return _native_session_refusal("native_bridge.event_invalid")
    except OSError as error:
        if owner_failure_code(error) is not None:
            raise
        return native_setup.files_unavailable(
            error, project=project or workspace, workspace=workspace
        )


def _activity_routes(web: LocalWebApplication, activity: WorkspaceActivity) -> None:
    """The browser's bounded activity feed and the declared sessions' readback. Reads never
    record; a client reads the feed and declares its events as operations."""

    def _read(query: Mapping[str, list[str]], _payload: dict[str, Any]) -> object:
        return activity.read(ActivityReadQuery.from_query(query))

    web.route("GET", "/api/activity")(_read)

    # The declared sessions' own readback (the Team page's owner read): one kind, walked back.
    def _read_external(query: Mapping[str, list[str]], _payload: dict[str, Any]) -> object:
        return activity.read_external(ExternalActivityReadQuery.from_query(query))

    web.route("GET", "/api/activity/external")(_read_external)


_REVIEW_SELECTOR_FIELDS = (
    "result_hash",
    "handoff_hash",
    "update_task_id",
    "update_publication_hash",
    "position_basis",
    "experiment_task_id",
    "experiment_receipt_hash",
    "portfolio_session",
)


def _review_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {key: _optional_field(payload, key) for key in _REVIEW_SELECTOR_FIELDS}
    if values["update_task_id"] is not None:
        values["update_task_id"] = UUID(values["update_task_id"])
    if values["experiment_task_id"] is not None:
        values["experiment_task_id"] = UUID(values["experiment_task_id"])
    return values


def _reading_fields(query: Mapping[str, list[str]]) -> dict[str, Any]:
    """Forward a published Reading view's typed filters to its operation owner."""
    days = _optional(query, "view_last_days")
    try:
        last_days = None if days is None else int(days)
    except ValueError as error:
        raise LocalWebError("alternative_evidence.view_interval_invalid") from error
    return {
        "evidence_detail": _optional(query, "evidence_detail"),
        "view_entity_id": _optional(query, "view_entity_id"),
        "view_topic": _optional(query, "view_topic"),
        "view_last_days": last_days,
    }


def _citation_fields(query: Mapping[str, list[str]]) -> dict[str, Any]:
    """One page of the section's citations or an export's verified spans,
    when one is asked for (an issuer, a group, a page from 1); none of
    them asks for the whole answer as before."""

    page = _optional(query, "citation_page")
    try:
        number = None if page is None else int(page)
    except ValueError as error:
        raise LocalWebError("local_web.query_parameter_invalid:citation_page") from error
    return {
        "citation_entity_id": _optional(query, "citation_entity_id"),
        "citation_unit_id": _optional(query, "citation_unit_id"),
        "citation_page": number,
    }


def _optional(query: Mapping[str, list[str]], key: str) -> str | None:
    values = query.get(key) or []
    if not values:
        return None
    if len(values) != 1 or not values[0]:
        raise LocalWebError(f"local_web.query_parameter_invalid:{key}")
    return values[0]


def _optional_field(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise LocalWebError(f"local_web.payload_field_invalid:{key}")
    return value


def _task_uuid(value: str) -> UUID:
    try:
        return UUID(value)
    except ValueError as error:
        raise LocalWebError(
            "local_web.task_id_invalid", next_action="COPY_FULL_TASK_ID_FROM_HISTORY"
        ) from error


def _task_read_response(body: dict[str, object]) -> object:
    """Legacy GET readers retain HTTP failure status; shared operations keep typed JSON."""
    if body.get("failure_code") == "task_control.task_not_found":
        return LocalWebResponse(
            HTTPStatus.NOT_FOUND,
            json.dumps(body).encode("utf-8"),
            "application/json; charset=utf-8",
        )
    return body


def _wait_seconds(values: list[str] | None) -> float | None:
    """A status read's wait, from its query; the request refuses a value out of range."""

    if not values:
        return None
    try:
        return float(values[-1])
    except ValueError as error:
        raise LocalWebError("local_application.wait_seconds_invalid") from error


def _one(query: Mapping[str, list[str]], key: str) -> str:
    values = query.get(key) or []
    if len(values) != 1 or not values[0]:
        raise LocalWebError(f"local_web.query_parameter_required:{key}")
    return values[0]


__all__ = [
    "ASSET_ROOT",
    "EvidenceReviewAuthority",
    "HostPortfolioFreeze",
    "LocalPortfolioWebSession",
    "LocalWebSessionError",
    "PortfolioForwardPrewarm",
    "build_evidence_review_application",
    "build_local_web_application",
    "read_workbench_portfolio",
]
