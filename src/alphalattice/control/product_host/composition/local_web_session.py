"""Compose the local Web product: session, application, dispatcher, routes.

This is the only place that knows all of it at once, which is the point. The
Local Application boundary declares ports and owns no wiring; Task Control owns
lifecycle; Portfolio owns numbers; the browser owns nothing. Composition holds
the workspace session open, builds one application over it, hands the service its
ports, and registers the routes a projection can reach.

Three things are deliberately *not* here. There is no second report owner -- the
`/report/*.html` route serves the page the executor already sealed. There is no
second Task registry -- the dispatcher admits through the session's. And the
Evidence & CRO routes are not special: they are three more operations of the one
operation owner, so the browser and the generic Agent tool share them.
"""

from __future__ import annotations

import json
import os
import secrets
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from functools import partial
from http import HTTPStatus
from pathlib import Path
from threading import Event, Lock
from typing import Any, Final, cast, get_args
from urllib.parse import urlencode
from uuid import UUID

from pydantic import ValidationError

from alphalattice.control.data_platform.readiness import SourceLoader
from alphalattice.control.product_host.composition.application_session import (
    WorkspaceApplicationSession,
)
from alphalattice.control.product_host.composition.decision_advancement import (
    DecisionAdvancementCommand,
)
from alphalattice.control.product_host.composition.evidence_authority_setup import (
    EVIDENCE_INSTALL_TASK_KIND,
    EvidenceInstallCommand,
)
from alphalattice.control.product_host.composition.evidence_review_application import (
    EvidenceReviewApplication,
)
from alphalattice.control.product_host.composition.evidence_review_workspace import (
    admit_evidence_review_workspace,
)
from alphalattice.control.product_host.composition.goal_case_routes import (
    CASE_ROUTES,
    case_answer,
    case_request,
)
from alphalattice.control.product_host.composition.local_web_connection import (
    publish_client_connection,
    remove_client_connection,
)
from alphalattice.control.product_host.composition.portfolio_application import (
    PLAN_FIELDS as PORTFOLIO_PLAN_FIELDS,
)
from alphalattice.control.product_host.composition.portfolio_application import (
    PortfolioResearchApplication,
)
from alphalattice.control.product_host.composition.portfolio_finalization import (
    HostCompletedDevelopmentTasks,
    PortfolioFinalizationApplication,
)
from alphalattice.control.product_host.composition.portfolio_research_operations import (
    PORTFOLIO_RUN_COMMAND,
    PortfolioResearchOperations,
    PortfolioRunCommand,
)
from alphalattice.control.product_host.composition.portfolio_result_context import (
    portfolio_report_context,
)
from alphalattice.control.product_host.composition.portfolio_updates import (
    TASK_KIND as PORTFOLIO_UPDATE_TASK_KIND,
)
from alphalattice.control.product_host.composition.portfolio_updates import PortfolioUpdateCommand
from alphalattice.control.product_host.composition.reference_prefixes import ReferenceLedger
from alphalattice.control.product_host.composition.research_experiments import (
    TASK_KIND as STUDY_TASK_KIND,
)
from alphalattice.control.product_host.composition.research_experiments import (
    ResearchExperimentCommand,
)
from alphalattice.control.product_host.composition.research_workspace import (
    ResearchWorkspaceEvidenceReview,
    ResearchWorkspaceManifest,
    ResearchWorkspaceManifestHolder,
    admit_research_workspace,
    initialize_research_workspace,
    manifest_fields_hash,
    read_research_workspace_manifest,
)
from alphalattice.control.product_host.composition.strategy_calibration import (
    TASK_KIND as CALIBRATION_TASK_KIND,
)
from alphalattice.control.product_host.composition.strategy_calibration import (
    StrategyCalibrationCommand,
)
from alphalattice.control.product_host.composition.strategy_scoring import (
    TASK_KIND as STRATEGY_SCORE_TASK_KIND,
)
from alphalattice.control.product_host.composition.strategy_scoring import (
    StrategyScoreCommand,
)
from alphalattice.control.product_host.composition.verification_sweep import (
    StudyVerificationSweepCommand,
)
from alphalattice.control.product_host.composition.workspace_activity import WorkspaceActivity
from alphalattice.control.product_host.data_preparation.application import (
    WorkspacePreparationCommand,
)
from alphalattice.control.product_host.data_preparation.feature_research import (
    ResearchFeatureBuildCommand,
)
from alphalattice.control.product_host.data_preparation.input_capture import (
    ResearchInputCaptureCommand,
)
from alphalattice.control.product_host.data_preparation.model_training import (
    ModelTrainingInputCommand,
)
from alphalattice.control.product_host.data_preparation.research_strategy import (
    ResearchStrategyCommand,
)
from alphalattice.control.product_host.maintenance.data_update import (
    DATA_UPDATE_TASK_KIND,
    WorkspaceDataUpdateApplication,
    WorkspaceDataUpdateCommand,
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
from alphalattice.evidence.alternative_evidence.contracts import SecIssuerRegistrySnapshot
from alphalattice.evidence.alternative_evidence.publication.analysis import (
    AlternativeEvidenceAnalysisPublicationService,
)
from alphalattice.evidence.alternative_evidence.publication.artifacts import (
    AlternativeEvidenceArtifactStore,
)
from alphalattice.evidence.alternative_evidence.runtime.policy import AdmittedEvidencePolicy
from alphalattice.evidence.alternative_evidence.runtime.service import (
    AlternativeEvidenceDocumentIntelligenceRuntime,
)
from alphalattice.evidence.alternative_evidence.runtime.task_adapter import (
    AlternativeEvidenceDocumentTaskAdapter,
    AlternativeEvidenceDocumentTaskResources,
)
from alphalattice.evidence.alternative_evidence.sources.admission import (
    OfficialSourceAdmission,
    admit_workspace_source,
)
from alphalattice.foundation.market_data_ops.sources.providers import MarketDataProvider
from alphalattice.interface.local_application.activity import (
    READ_OPERATIONS,
    ActivityReadQuery,
    ExternalActivityReadQuery,
    refusal_code,
    returned_status,
)
from alphalattice.interface.local_application.cli_contract import (
    REQUEST_PROVENANCE,
    RequestProvenance,
    agent_session,
)
from alphalattice.interface.local_application.client import (
    LocalResearchConnection,
    workspace_connection_key,
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
    read_session_usage,
    session_project,
    usage_reading,
)
from alphalattice.interface.local_application.native_setup import (
    admitted_session_project,
    autobind_root,
)
from alphalattice.interface.local_application.portfolio_research import (
    FinalizationStatusProjection,
    FrozenCandidateProjection,
    LocalPortfolioResearchService,
    OperationCaller,
    PortfolioResearchOperation,
    PortfolioResearchOperationRequest,
    PortfolioResearchRequestDocument,
)
from alphalattice.interface.local_application.web import (
    LocalWebApplication,
    LocalWebError,
    LocalWebResponse,
    LocalWebService,
    report_policy,
)
from alphalattice.investment.alpha_research.experiments.authoring import ALPHA_EXPERIMENT_KIND
from alphalattice.investment.alpha_research.scores.model_renewal import (
    verified_lifecycle_admissions,
)
from alphalattice.investment.portfolio_strategy_lab.application.advancement_task import (
    ADVANCEMENT_TASK_KIND,
)
from alphalattice.investment.portfolio_strategy_lab.application.candidate_freeze import (
    freeze_candidate,
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
from alphalattice.kernel.shared_kernel.project_layout import resolve_playpen_root
from alphalattice.oversight.chief_risk_officer.portfolio_evidence.contracts import (
    AdmittedListingTickerAuthority,
)
from alphalattice.oversight.chief_risk_officer.publication.portfolio_review import (
    PortfolioReviewPublicationService,
)
from alphalattice.oversight.chief_risk_officer.runtime.portfolio_review_task import (
    PortfolioReviewActor,
)

ASSET_ROOT = Path(__file__).resolve().parents[3] / "interface" / "local_application" / "assets"


def _utc_now() -> datetime:
    """The product clock: one function, aware, used by every owner on the path."""

    return datetime.now(tz=UTC)


"""Local files, served from the installed package. No CDN, no font host."""


@dataclass(frozen=True, slots=True)
class HostPortfolioFreeze:
    """The Gate 9B freeze and finalization owners, projected for a local UI.

    Every field below is read off an owner. Nothing is derived here, and there is
    no route through this class that could evaluate protected evidence: freezing
    is a read of a published development result plus one registry write, and the
    finalization status is whatever the finalization store already holds.
    """

    session: WorkspaceApplicationSession
    application: PortfolioResearchApplication
    finalization: PortfolioFinalizationApplication

    @property
    def _store(self) -> PortfolioFinalizationStore:
        return self.finalization.store

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
        candidate = freeze_candidate(
            self.application.ledger,
            self._store,
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
        candidate = self._store.open_candidate(candidate_hash)
        return None if candidate is None else _projected(candidate)

    def finalization_status(self, candidate_hash: str) -> FinalizationStatusProjection:
        """Project the recorded permit, closure and release status of a candidate.

        Args:
            candidate_hash: Exact frozen candidate identity to inspect.

        Returns:
            Status distinguishing an absent candidate, awaiting protected authority
            or publication, and the released package with its recorded handoff.
        """
        candidate = self._store.open_candidate(candidate_hash)
        if candidate is None:
            return FinalizationStatusProjection(
                candidate_hash=candidate_hash,
                disposition="NOT_FROZEN",
                detail="no candidate was frozen under this identity",
            )
        permit = self.finalization.gate.reopen_permit(candidate=candidate)
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
        package = self._store.find_package_for_permit(permit.permit_hash)
        receipt = self.finalization.gate.reopen_receipt(permit=permit)
        if package is None or receipt is None:
            return FinalizationStatusProjection(
                candidate_hash=candidate_hash,
                disposition="AWAITING_PROTECTED_AUTHORITY",
                detail="a permit exists; the protected path has not been sealed and closed",
            )
        release = self._store.find_release_for_package(package.package_hash)
        handoff = self._store.find_handoff_for_package(package.package_hash)
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


class NativeUsageReader:
    """Read the bound Sessions' usage when research asks, never on a timer (FLOW-1).

    A milestone -- a goal's take or submission, an answer's submission -- or a Goal or Team page
    opening asks. Each Session bound to this workspace whose reading is on is read from its own
    file and the children that file records (`read_session_usage`); no other Session's file is
    opened. One read runs at a time, a Session read within `RECENT` is not read again, and a
    failure is named in `state`, never a refusal of research.
    """

    RECENT = timedelta(seconds=10)

    def __init__(self, operations: PortfolioResearchOperations) -> None:
        """Retain the served workspace's binding, Goal and observation owners.

        Args:
            operations: This Host's admitted operation and Goal owners.
        """
        self.operations = operations
        self.workspace = operations.workspace_session.workspace.resolve()
        self.clock = operations.dispatcher.clock
        self._reading = Lock()
        self._read_at: dict[tuple[str, str], datetime] = {}
        self._status = "NOT_BOUND"
        self._reason: str | None = None

    def state(self) -> dict[str, object]:
        """The last reading's standing, never native lifecycle or completeness proof.

        Returns:
            Bounded status and failure code without participant ids or file paths.
        """
        return {
            "status": "UNAVAILABLE" if self._reason is not None else self._status,
            "reason": self._reason,
        }

    def _bindings(self) -> tuple[tuple[Path, NativeResearchBinding], ...]:
        found: list[tuple[Path, NativeResearchBinding]] = []
        projects: set[tuple[Path, str]] = set()
        for host in HOSTS:
            try:
                projects.add((session_project(self.workspace, host), host))
            except NativeBridgeError as error:
                if str(error) != "native_bridge.project_declaration_missing":
                    self._reason = self._reason or safe_failure_code(str(error))
            # A workspace kept outside its project: the projects this Host admitted it from.
            projects.update((admitted, host) for admitted in self.operations.native_projects)
        for candidate, host in sorted(projects, key=lambda item: (str(item[0]), item[1])):
            try:
                project = admitted_session_project(self.workspace, candidate, host)
                bindings, diagnostics = NativeResearchBinding.binding_entries(project)
                for diagnostic in diagnostics:
                    self._reason = self._reason or safe_failure_code(diagnostic["failure_code"])
                found.extend(
                    (project, binding)
                    for binding in bindings
                    if binding.host == host and binding.workspace.resolve() == self.workspace
                )
            except NativeBridgeError as error:
                if str(error) != "native_bridge.project_declaration_missing":
                    self._reason = self._reason or safe_failure_code(str(error))
            except Exception as error:
                self._reason = self._reason or (
                    owner_failure_code(error) or "native_bridge.lead_usage_read_failed"
                )
        # The bindings the Host made itself, kept in the workspace (AUTOBIND).
        root = autobind_root(self.workspace)
        try:
            held, _diagnostics = NativeResearchBinding.binding_entries(root)
            found.extend(
                (root, binding) for binding in held if binding.workspace.resolve() == self.workspace
            )
        except Exception as error:
            self._reason = self._reason or (
                owner_failure_code(error) or "native_bridge.lead_usage_read_failed"
            )
        return tuple(found)

    def read_binding(self, project: Path, binding: NativeResearchBinding) -> dict[str, object]:
        """Read one bound Session now, filed under the Goal its Session holds.

        Args:
            project: The admitted project holding the binding.
            binding: The exact bound Session of this workspace.

        Returns:
            The reading's receipt, one entry per participant.
        """
        provenance = RequestProvenance(vendor=binding.host, session=binding.session_id)
        goal = self.operations.goals.attributed_goal(provenance)
        goal_id = None if goal is None else str(goal.goal_id)

        def publish(document: dict[str, Any]) -> dict[str, Any]:
            # Pin even an explicitly absent Goal through filing. The same-store RLock
            # prevents a concurrent Goal take from changing file_event's fallback binding.
            with (
                self.operations.workspace_session.mutation_gate.hold(),
                self.operations.goals.store.lock,
            ):
                current = self.operations.goals.attributed_goal(provenance)
                if (None if current is None else str(current.goal_id)) != goal_id or (
                    NativeResearchBinding.read(project, session=(binding.host, binding.session_id))
                    != binding
                ):
                    return {
                        "status": "REFUSED",
                        "failure_code": "native_bridge.event_scope_invalid",
                    }
                token = REQUEST_PROVENANCE.set(
                    None if goal_id is None else RequestProvenance(goal_id=goal_id)
                )
                try:
                    return self.operations.execute(
                        PortfolioResearchRequestDocument(
                            operation="EVENT_DECLARE", event=document
                        ).to_operation_request(),
                        caller="EXTERNAL_AUTOMATION",
                    )
                finally:
                    REQUEST_PROVENANCE.reset(token)

        self._read_at[(binding.host, binding.session_id)] = self.clock()
        return read_session_usage(project, binding, publish=publish, goal_id=goal_id)

    def read(self) -> dict[str, object]:
        """Read every Session bound to this workspace whose reading is on, once.

        Returns:
            ``READ`` with one receipt per Session, ``RECENT`` for one read moments ago,
            ``OFF`` or ``NOT_BOUND``; ``BUSY`` while another read runs.
        """
        if not self._reading.acquire(blocking=False):
            return {"status": "BUSY", "sessions": []}
        try:
            self._reason = None
            if not usage_reading(self.workspace):
                # The person's switch: no binding is even looked for.
                self._status = "OFF"
                return {"status": "OFF", "sessions": []}
            found = self._bindings()
            active = tuple(item for item in found if item[1].usage != "OFF")
            if not active:
                self._status = "NOT_BOUND" if not found else "OFF"
                return {"status": self._status, "sessions": []}
            self._status = "OBSERVING"
            now = self.clock()
            sessions: list[dict[str, object]] = []
            for project, binding in active:
                key = (binding.host, binding.session_id)
                last = self._read_at.get(key)
                if last is not None and now - last < self.RECENT:
                    sessions.append({"host": binding.host, "status": "RECENT"})
                    continue
                try:
                    receipt = self.read_binding(project, binding)
                except Exception as error:
                    receipt = {
                        "status": "UNAVAILABLE",
                        "reason": owner_failure_code(error)
                        or "native_bridge.lead_usage_read_failed",
                    }
                if receipt.get("status") not in {"DELIVERED", "SKIPPED"}:
                    self._reason = self._reason or safe_failure_code(receipt.get("reason"))
                sessions.append({"host": binding.host, **receipt})
            return {"status": "READ", "sessions": sessions}
        finally:
            self._reading.release()


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
                if (
                    not self._stopping.is_set()
                    and failure is None
                    and kind
                    in {
                        DATA_UPDATE_TASK_KIND,
                        ADVANCEMENT_TASK_KIND,
                        PORTFOLIO_UPDATE_TASK_KIND,
                    }
                ):
                    # A returned worker may have deferred, blocked or cancelled its
                    # Task. Only the registry's terminal success warms new data.
                    try:
                        task = self.operations.workspace_session.task_control_registry.task(task_id)
                    except Exception:
                        task = None
                    if task is not None and task.lifecycle is TaskLifecycle.SUCCEEDED:
                        self.wake()

        dispatcher.on_command_returned = command_returned

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

    @verified_model_read_scope(reuse_verified=True)
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


@dataclass(slots=True)
class LocalPortfolioWebSession:
    """Everything the local product needs, started and stopped as one unit."""

    workspace: Path
    workspace_manifest: ResearchWorkspaceManifest
    resolver: PortfolioExecutionResolver | None
    port: int = 0

    review_authority: EvidenceReviewAuthority | None = None
    data_provider: MarketDataProvider | None = None
    data_source_loader: SourceLoader | None = None
    clock: Callable[[], datetime] = _utc_now
    """The service's own timezone-aware clock, shared with everything it builds."""
    session: WorkspaceApplicationSession | None = field(default=None, init=False)
    application: PortfolioResearchApplication | None = field(default=None, init=False)
    service: LocalPortfolioResearchService | None = field(default=None, init=False)
    dispatcher: LocalBackgroundDispatcher | None = field(default=None, init=False)
    operations: PortfolioResearchOperations | None = field(default=None, init=False)
    activity: WorkspaceActivity | None = field(default=None, init=False)
    forward_prewarm: PortfolioForwardPrewarm | None = field(default=None, init=False)
    native_usage_reader: NativeUsageReader | None = field(default=None, init=False)
    web: LocalWebService | None = field(default=None, init=False)
    client_connection: LocalResearchConnection | None = field(default=None, init=False, repr=False)
    review: EvidenceReviewApplication | None = field(default=None, init=False)
    resumed_task_ids: tuple[UUID, ...] = field(default=(), init=False)
    _workspace_review_authority: bool = field(default=False, init=False)
    _official_source: OfficialSourceAdmission | None = field(default=None, init=False)

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
    ) -> LocalPortfolioWebSession:
        """Ordinary composition: one path in, verified manifest and catalog out.

        The official SEC source is admitted at runtime from the workspace's consent, its network
        control and the product's contact (`admit_workspace_source`), with no restart.
        `official_source` replaces that admission with a fixed one, a rehearsal's own.
        """
        initialize_research_workspace(workspace)
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
        )
        value._workspace_review_authority = (
            review_authority is None and admitted.manifest.evidence_review is not None
        )
        value._official_source = official_source
        return value

    def start(self) -> str:
        """Acquire, compose, bind -- or leave nothing behind.

        Half of this is a lease and a worker thread, and the other half is a
        socket. A failure part-way through used to leave the first half live
        with no URL to stop it through, which is a held workspace and a
        non-daemon thread the process cannot exit past. So the whole thing is
        one transaction: anything already started is torn down, in the order a
        healthy stop uses, before the failure is re-raised.
        """
        try:
            return self._start()
        except BaseException:
            self._teardown(timeout=None)
            raise

    def _start(self) -> str:
        if self._workspace_review_authority and self.review_authority is None:
            binding = self.workspace_manifest.evidence_review
            assert binding is not None
            self.review_authority = self._workspace_authority(binding)
        session = WorkspaceApplicationSession.acquire(self.workspace).__enter__()
        self.session = session
        # The one holder of the workspace manifest every application reads; the operations
        # refresh it when a publication moves the manifest (V182).
        manifests = ResearchWorkspaceManifestHolder(self.workspace_manifest)
        if self.resolver is not None:
            resolver = self.resolver
            application = PortfolioResearchApplication(
                workspace_id=self.workspace_id,
                workspace=self.workspace,
                manifest_binding=lambda: manifest_fields_hash(
                    manifests.current, PORTFOLIO_PLAN_FIELDS
                ),
                session=session,
                resolver=self.resolver,
                report_context=lambda program, day: portfolio_report_context(
                    workspace=self.workspace,
                    packages=resolver.installed_packages(),
                    manifest=manifests.current,
                    program=program,
                    session=day,
                ),
                clock=self.clock,  # the one service clock, so its runner's heartbeats and
                # the readers that judge them share it
            )
            self.application = application
            finalization = PortfolioFinalizationApplication(
                workspace_id=self.workspace_id,
                session=session,
                continuation=None,  # type: ignore[arg-type]
            )
            freeze = HostPortfolioFreeze(
                session=session, application=application, finalization=finalization
            )
            self.service = LocalPortfolioResearchService(
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
            workers=lambda: running_places(read_queue_setting(self.workspace / "runtime"))[0],
        )
        session.task_control_registry.overlapping = runs_beside_others
        session.task_control_registry.started = self.dispatcher.drive_waiting
        self.dispatcher.start()
        self.review = self._review_application(
            session, None if self.resolver is None else finalization.store
        )
        if self._workspace_review_authority and self._official_source is not None:
            self.review.use_source(self._official_source)
        self.review.refresh_source()
        self.operations = PortfolioResearchOperations(
            service=self.service,
            dispatcher=self.dispatcher,
            application=self.application,
            workspace_manifest=self.workspace_manifest,
            manifests=manifests,
            workspace_session=session,
            review=self.review,
            recover_task=self.resume,
            recoverable_task_kinds=self.recoverable_task_kinds,
            resume_refusal=self.resume_refusal,
            install_review=self.serve_installed_review,
            official_source=self._official_source,
            heartbeats=TaskHeartbeatReader(
                session.runtime_path,
                *([self.review.runner_runtime_path()] if self.review is not None else []),
            ),
            data_update=WorkspaceDataUpdateApplication(
                session=session,
                manifest=manifests,
                clock=self.clock,
                provider=self.data_provider,
                source_loader=self.data_source_loader,
            ),
        )
        # The activity observer is composed after every owner it reads and
        # before the socket opens, so the first request is already observed.
        # Activity is optional infrastructure: a store that cannot open leaves
        # the observer UNAVAILABLE on every read and the product usable.
        activity = WorkspaceActivity(
            workspace=self.workspace,
            workspace_id=self.workspace_id,
            gate=session.mutation_gate,
            instance=secrets.token_hex(16),
            clock=self.clock,
            storage_cap_reader=self.operations.storage.capacity_cap_bytes,
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
        web_application = build_local_web_application(
            service=self.service,
            operations=self.operations,
            activity=activity,
        )
        self.web = LocalWebService(application=web_application, port=self.port)
        url = self.web.start()
        connection = LocalResearchConnection(
            workspace=str(self.workspace.resolve()),
            workspace_id=self.workspace_id,
            url=url,
            instance=web_application.external_instance,
            token=web_application.external_token,
        )
        publish_client_connection(connection)
        self.client_connection = connection
        self.resumed_task_ids = self.resume()
        # A Codex lead's wake whose Task ended, stopped or deferred while no Host ran (WAKE).
        activity.replay_wakes()
        if self.operations.automation is not None:
            self.operations.automation.start()
        if not self.resumed_task_ids:
            # A Host that starts idle sweeps as one that went idle would (V89).
            self.operations.sweep_if_due()
        # Guanyin's Supervisor reads the unfinished Tasks from here on (GY2).
        self.operations.supervisor.start()
        self.forward_prewarm.start()
        # Optional reading owns no research authority and must not prevent serving it.
        constructor_failure: str | None = None
        try:
            self.native_usage_reader = NativeUsageReader(self.operations)
            self.operations.read_native_usage = self.native_usage_reader.read
        except Exception as error:
            self.native_usage_reader = None
            constructor_failure = (
                owner_failure_code(error) or "native_bridge.lead_usage_read_failed"
            )

        def native_usage_state() -> dict[str, object]:
            """Keep the last reading's standing visible through the activity owner's readback."""
            if self.native_usage_reader is not None:
                return self.native_usage_reader.state()
            return {
                "status": "UNAVAILABLE",
                "reason": constructor_failure or "native_bridge.lead_usage_read_failed",
            }

        activity.native_usage_state = native_usage_state
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
    ) -> EvidenceReviewApplication:
        return build_evidence_review_application(
            workspace=self.workspace,
            workspace_id=self.workspace_id,
            session=session,
            application=self.application,
            finalization_store=finalization_store,
            authority=self.review_authority,
            official_source=(
                partial(admit_workspace_source, self.workspace)
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

    def _recovery_commands(self, *, owed_only: bool = True) -> dict[str, LocalApplicationCommand]:
        commands: dict[str, LocalApplicationCommand] = {}
        if self.operations is not None:
            experiment_command = ResearchExperimentCommand(self.operations.experiments)
            commands[experiment_command.command_kind] = experiment_command
            preparation_command = WorkspacePreparationCommand(self.operations.preparation)
            commands[preparation_command.command_kind] = preparation_command
            capture_command = ResearchInputCaptureCommand(self.operations.input_capture)
            commands[capture_command.command_kind] = capture_command
            training_command = ModelTrainingInputCommand(self.operations.model_training_inputs)
            commands[training_command.command_kind] = training_command
            research_strategy_command = ResearchStrategyCommand(self.operations.research_strategies)
            commands[research_strategy_command.command_kind] = research_strategy_command
            feature_command = ResearchFeatureBuildCommand(self.operations.feature_builds)
            commands[feature_command.command_kind] = feature_command
            sweep_command = StudyVerificationSweepCommand(self.operations.sweep)
            commands[sweep_command.command_kind] = sweep_command
            install_command = EvidenceInstallCommand(self.operations.evidence_install)
            commands[install_command.command_kind] = install_command
        if self.operations is not None and self.operations.research_updates is not None:
            advancement_command = DecisionAdvancementCommand(self.operations.research_updates)
            commands[advancement_command.command_kind] = advancement_command
        # What the workspace holds now, read from the one holder the applications read (V182).
        manifest = (
            self.workspace_manifest
            if self.operations is None or self.operations.manifests is None
            else self.operations.manifests.current
        )
        if (
            self.operations is not None
            and self.operations.updates is not None
            and manifest.decision_updates
        ):
            commands[PORTFOLIO_UPDATE_TASK_KIND] = PortfolioUpdateCommand(self.operations.updates)
        if (
            self.operations is not None
            and self.operations.calibration is not None
            and manifest.calibration_inputs
        ):
            commands[CALIBRATION_TASK_KIND] = StrategyCalibrationCommand(
                self.operations.calibration
            )
        if (
            self.operations is not None
            and self.operations.scoring is not None
            and manifest.score_inputs
        ):
            commands[STRATEGY_SCORE_TASK_KIND] = StrategyScoreCommand(self.operations.scoring)
        if (
            self.operations is not None
            and self.operations.data_update is not None
            and manifest.data_update is not None
        ):
            commands[DATA_UPDATE_TASK_KIND] = WorkspaceDataUpdateCommand(
                self.operations.data_update
            )
        assert self.session is not None
        if self.application is not None and (
            not owed_only
            or any(
                task.lifecycle in {TaskLifecycle.RECOVERY_REQUIRED, TaskLifecycle.QUEUED}
                and task.task_kind == PORTFOLIO_PUBLIC_TASK_KIND
                for task in self.session.task_control_registry.tasks()
            )
        ):
            commands[PORTFOLIO_RUN_COMMAND] = PortfolioRunCommand.recover(
                application=self.application
            )
        if self.review is not None:
            # Resume the admitted Task's durable identity, verified stages and inputs.
            # Rebuild from the recovery-required Task so it cannot answer another question.
            for kind, command in self.review.recovery_commands(owed_only=owed_only).items():
                commands[kind] = cast(LocalApplicationCommand, command)
        return commands

    def stop(self, *, timeout: float | None = None) -> None:
        """Fail closed. Unbounded by default; a bounded stop that cannot finish raises.

        The workspace lease is released last and only once every thread that
        could write through it has ended. A bounded caller that cannot get there
        is told so, and the session stays exactly as it is -- lease held, worker
        joinable -- so calling `stop()` again finishes the job rather than
        starting from a half-released state.
        """
        if not self._teardown(timeout=timeout):
            raise LocalWebSessionError("local_web_session.writer_still_live")

    def _teardown(self, *, timeout: float | None) -> bool:
        """Stop the socket, then the worker, then the lease. True when complete."""

        quiet = True
        if self.forward_prewarm is not None:
            quiet = self.forward_prewarm.close(timeout=timeout) and quiet
        if self.operations is not None:
            self.operations.supervisor.stop()
        if self.operations is not None and self.operations.automation is not None:
            quiet = self.operations.automation.close(timeout=timeout) and quiet
        if self.web is not None:
            # Request threads can admit tasks, so they are writers too and are
            # joined before anything else is released.
            quiet = self.web.stop(timeout=timeout) and quiet
        if self.dispatcher is not None:
            quiet = self.dispatcher.close(timeout=timeout) and quiet
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
        if self.review is not None:
            self.review.close_source()
        if self.session is not None:
            if self.client_connection is not None:
                remove_client_connection(self.client_connection)
                self.client_connection = None
            self.session.__exit__(None, None, None)
        self.web = None
        self.dispatcher = None
        self.session = None
        self.application = None
        self.service = None
        self.operations = None
        self.forward_prewarm = None
        self.native_usage_reader = None
        # The review application holds this session's registry and its adapter,
        # both of which are now closed. Keeping it would leave a stopped service
        # able to answer questions about a workspace it no longer holds.
        self.review = None
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
        named = agent_session(os.environ)
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
    official_source: Callable[[], OfficialSourceAdmission] | None = None,
    clock: Callable[[], datetime] = _utc_now,
) -> EvidenceReviewApplication:
    """The one review owner this service holds.

    Built for every workspace, with or without admitted evidence authority: the
    section has to be able to answer "what is the state" in both, and a route
    that constructed the owner only when it already had an answer could never
    report the wait.
    """
    artifacts = (
        authority.artifacts
        if authority is not None
        else AlternativeEvidenceArtifactStore(workspace / "runtime" / "artifacts")
    )
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
    # the Task queue's waiting places are set on the same route (V100, V266).
    ("GET", "/api/workspace/cpu-budget", "CPU_BUDGET_SHOW"),
    ("POST", "/api/workspace/cpu-budget", "CPU_BUDGET_SET"),
    ("GET", "/api/workspace/backup", "WORKSPACE_BACKUPS"),
    # A Goal or Team page reads the bound Sessions' usage when it opens (FLOW-1).
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
    # What the UI pass reads (V321): the goals a person keeps (U23, U32; taking, submitting
    # and continuing stay an agent's, through the client's route), the saved studies'
    # sweep (U37) and Guanyin's incidents with their remedies (U38).
    ("GET", "/api/goals", "GOAL_LIST"),
    ("GET", "/api/committee", "COMMITTEE_READ"),
    ("GET", "/api/goals/schema", "GOAL_SCHEMA"),
    ("GET", "/api/goals/show", "GOAL_SHOW"),
    ("GET", "/api/goals/narrative", "GOAL_NARRATIVE"),
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
"""The operations served by handlers of their own rather than the table's route: their
handlers read their paths here, and the session answer names them beside the table's (U13)."""

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


def build_local_web_application(
    *,
    service: LocalPortfolioResearchService | None,
    operations: PortfolioResearchOperations,
    activity: WorkspaceActivity | None = None,
) -> LocalWebApplication:
    """Register the routes. Every one of them is a projection or a command."""
    workspace_key = workspace_connection_key(operations.workspace_session.workspace)
    if activity is None:
        web = LocalWebApplication(external_workspace_id=workspace_key)
    else:
        # One launch identity: the client connection's instance and the
        # observations' source id name the same service start.
        web = LocalWebApplication(
            external_workspace_id=workspace_key, external_instance=activity.instance
        )
        _activity_routes(web, activity)
    _native_session_routes(web, operations)

    # Every hash and id answered to a client, which reads them back from a beginning (V393).
    references = ReferenceLedger(operations.workspace_session.workspace)

    @web.route("POST", "/api/client/operations", mutates=True, external_client=True)
    def external_operation(_query: Mapping[str, list[str]], payload: dict[str, Any]) -> object:
        try:
            document = PortfolioResearchRequestDocument.model_validate(payload)
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
                **located_failure(error, "local_client.request_invalid"),
                "next_action": "READ_CAPABILITIES_SCHEMA",
            }
        answer = operations.execute(document.to_operation_request(), caller="EXTERNAL_AUTOMATION")
        references.record(answer)
        return answer

    def _asset_manifest() -> dict[str, str]:
        """The build's map of logical asset names to their content-hashed paths; empty when
        the assets were built without one (an older build), in which case only the plain
        names are served."""
        path = ASSET_ROOT / "workbench-manifest.json"
        if not path.is_file():
            return {}
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return {str(k): str(v) for k, v in loaded.items()} if isinstance(loaded, dict) else {}

    # The workbench is the product's one web entry, served at `/` and at its own name (older
    # links). The original UI was retired on 2026-09-19; the operations only it drove (freeze,
    # finalization, the update families) remain the Agent tool's and the automation API's.
    web.add_asset("/", (ASSET_ROOT / "workbench.html").read_bytes(), "text/html; charset=utf-8")
    # Each built asset answers under its plain name (revalidated every time, so an older link
    # reads the current build) and, when the build's manifest names one, under its
    # content-hashed path -- immutable, so a warm load fetches nothing (round 96).
    manifest = _asset_manifest()
    for name, content_type in (
        ("workbench.html", "text/html; charset=utf-8"),
        ("workbench-prelude.js", "text/javascript; charset=utf-8"),
        ("workbench.css", "text/css; charset=utf-8"),
        ("workbench.js", "text/javascript; charset=utf-8"),
        ("workbench.zh.js", "text/javascript; charset=utf-8"),
        # The workbench's one designed face (Geist, SIL OFL): served here, never fetched.
        ("fonts/geist-latin.woff2", "font/woff2"),
        ("fonts/geist-mono-latin.woff2", "font/woff2"),
    ):
        asset = ASSET_ROOT / name
        if asset.is_file():
            body = asset.read_bytes()
            web.add_asset(f"/{name}", body, content_type)
            hashed = manifest.get(name)
            if hashed and hashed != f"/{name}" and name != "workbench.html":
                web.add_asset(hashed, body, content_type, immutable=True)
    # Browsers request this implicitly. Answer locally so a clean page does not
    # produce a misleading network error for an asset the product never needs.
    web.add_asset("/favicon.ico", b"", "image/x-icon")

    def _spec_document(payload: Mapping[str, Any]) -> dict[str, object]:
        document = payload.get("spec")
        if document is None:
            document = {}
        if not isinstance(document, dict):
            raise LocalWebError("local_web.spec_not_an_object")
        return cast(dict[str, object], document)

    # The token goes only to the browser holding the cookie the launch URL set (HB).
    @web.route("GET", "/api/session", session=True)
    def _session(_query: Mapping[str, list[str]], _payload: dict[str, Any]) -> object:
        return {
            **operations.session_projection(
                session_token=web.session_token, include_context=_query.get("context") == ["1"]
            ),
            # Each operation's route, from the table the Host serves (U13).
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

    @web.route("GET", "/api/controls")
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

        if service is None:
            raise LocalWebError("research_workspace.strategy_not_installed")
        page = service.open_html(_one(query, "result_hash"))
        return LocalWebResponse(
            HTTPStatus.OK,
            page.encode("utf-8"),
            "text/html; charset=utf-8",
            # The sealed page styles itself inline, and the default policy
            # forbids inline style. Naming its exact style by hash keeps the page
            # byte-identical and the policy narrow.
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
    # beside it (HB, V184): a POST whose operation is not one of the registry's reads
    # takes it. A GET never writes.
    for method, path, operation in OPERATION_ROUTES:
        web.route(method, path, mutates=method == "POST" and operation not in READ_OPERATIONS)(
            operation_route(operation)
        )

    def case_route(operation: str):  # type: ignore[no-untyped-def]
        """The case page's call, answered by the goal it names until the UI pass (U23)."""

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


def _native_session_routes(
    web: LocalWebApplication, operations: PortfolioResearchOperations
) -> None:
    """Write native attachment metadata only in this served workspace's exact project."""
    from alphalattice.interface.local_application import native_setup
    from alphalattice.interface.local_application.cli_contract import (
        REQUEST_PROVENANCE,
        refusal_words,
    )
    from alphalattice.interface.local_application.failure_codes import owner_failure_code
    from alphalattice.interface.local_application.native_bridge import (
        NativeBridgeError,
        NativeResearchBinding,
    )
    from alphalattice.interface.local_application.native_setup import (
        NativeEventDeliveryRequest,
        NativeSessionBindRequest,
        bind_session,
        files_unavailable,
    )

    workspace = operations.workspace_session.workspace.resolve()

    def refusal(code: str) -> dict[str, Any]:
        return {"status": "REFUSED", "failure_code": code, **refusal_words(code)}

    @web.route("POST", "/api/client/session/bind", mutates=True, external_client=True)
    def bind(_query: Mapping[str, list[str]], payload: dict[str, Any]) -> object:
        try:
            request = NativeSessionBindRequest.model_validate(payload)
        except ValidationError:
            return refusal("native_bridge.request_invalid")
        provenance = REQUEST_PROVENANCE.get()
        if provenance is None or provenance.vendor is None or provenance.session is None:
            return refusal("local_client.session_unnamed")
        project: Path | None = None
        try:
            project = native_setup.admitted_session_project(
                workspace, request.project, provenance.vendor
            )
            operations.native_projects.add(project)
            return bind_session(
                project,
                host=provenance.vendor,
                session_id=provenance.session,
                workspace=workspace,
                usage=request.usage,
            )
        except NativeBridgeError as error:
            return refusal(str(error))
        except OSError as error:
            if owner_failure_code(error) is not None:
                raise
            return files_unavailable(error, project=project or workspace, workspace=workspace)

    @web.route("POST", "/api/client/session/event", mutates=True, external_client=True)
    def event(_query: Mapping[str, list[str]], payload: dict[str, Any]) -> object:
        try:
            request = NativeEventDeliveryRequest.model_validate(payload)
        except ValidationError:
            return refusal("native_bridge.request_invalid")
        provenance = REQUEST_PROVENANCE.get()
        if (
            provenance is None
            or provenance.vendor is None
            or provenance.vendor not in HOSTS
            or provenance.session is None
        ):
            return refusal("native_bridge.event_scope_invalid")
        project: Path | None = None
        try:
            # The real caller selects an exact binding or its native-verified parent within
            # this admitted project. The event body cannot select a Session or workspace.
            selected_project = native_setup.admitted_session_project(
                workspace, request.project, provenance.vendor
            )
            operations.native_projects.add(selected_project)
            project = selected_project
            found = NativeResearchBinding.find(
                selected_project, session=(provenance.vendor, provenance.session)
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
                        return refusal("native_bridge.event_scope_invalid")
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
                            PortfolioResearchRequestDocument(
                                operation="EVENT_DECLARE", event=document
                            ).to_operation_request(),
                            caller="EXTERNAL_AUTOMATION",
                        )
                    finally:
                        REQUEST_PROVENANCE.reset(token)

            # A milestone's reading of the caller's own Session; no other event enters here.
            if request.event != {"source": "native_usage_read"} or (
                provenance.vendor,
                provenance.session,
            ) != (binding.host, binding.session_id):
                return refusal("native_bridge.event_scope_invalid")
            return read_session_usage(project, binding, publish=publish, goal_id=goal_id)
        except NativeBridgeError as error:
            return refusal(str(error))
        except (KeyError, TypeError, ValueError) as error:
            if owner_failure_code(error) is not None:
                raise
            return refusal("native_bridge.event_invalid")
        except OSError as error:
            if owner_failure_code(error) is not None:
                raise
            return files_unavailable(error, project=project or workspace, workspace=workspace)


def _activity_routes(web: LocalWebApplication, activity: WorkspaceActivity) -> None:
    """The browser's bounded activity feed and the declared sessions' readback. Reads never
    record; a client reads the feed and declares its events as operations (V266)."""

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
        import json

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
