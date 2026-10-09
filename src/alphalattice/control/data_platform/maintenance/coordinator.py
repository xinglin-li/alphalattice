"""Concrete onboarding and daily-maintenance coordinator for one workspace."""

from __future__ import annotations

import gc
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from itertools import pairwise
from typing import Any, Protocol

import pyarrow as pa
from pydantic import TypeAdapter

from alphalattice.control.data_platform.contracts import (
    DataRemediationActorError,
    DataRemediationActorSubmission,
    DataRemediationExecutionReceipt,
)
from alphalattice.control.data_platform.delegation import DataIssueDelegation
from alphalattice.control.data_platform.maintenance.contracts import (
    ActionAuditChainReceipt,
    ActionAuditScope,
    AgentExecutionBudget,
    DataRemediationFailureReceipt,
    ListingMarketDataChange,
    MaintenanceFreshnessProjection,
    MaintenancePhase,
    MaintenanceStatus,
    MarketDataChangeSet,
    WorkspaceMaintenanceCycle,
    WorkspaceMaintenanceOutcome,
    WorkspaceMaintenanceRequest,
    remediation_failure_reason,
)
from alphalattice.control.data_platform.maintenance.invalidation import FeatureInvalidationTopology
from alphalattice.control.data_platform.preflight import (
    resolve_trading_session_authority,
)
from alphalattice.control.data_platform.readiness import (
    WorkspaceReadinessGate,
    WorkspaceReadinessStatus,
)
from alphalattice.control.data_platform.remediation_case import (
    raw_retention_decision_proofs,
    seal_validated_data_remediation_execution,
)
from alphalattice.control.observation_runtime.telemetry.progress import WorkProgressUpdate
from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.foundation.feature_engine.catalog.contracts import desktop_core_feature_bundle
from alphalattice.foundation.feature_engine.contracts import (
    FeatureBuildRequest,
    FeatureBuildStatus,
    FeatureInvalidation,
    canonical_hash,
)
from alphalattice.foundation.feature_engine.inputs.contracts import ListingQuarantine
from alphalattice.foundation.feature_engine.inputs.gateway import (
    FeatureInputAdmission,
    FeatureInputAgentCase,
    FeatureInputAssessment,
    FeatureInputExecutionStatus,
    FeatureInputGatewayResult,
    FeatureInputGovernanceService,
    FeatureInputPolicyExecution,
    FeatureInputRemediationExecutor,
    ListingEligibilityDecision,
    ListingQualityEvidence,
    PanelImpactProjection,
    ProviderCohortDeferred,
)
from alphalattice.foundation.feature_engine.inputs.quality import (
    FeatureInputQualityEvaluator,
    RawCloseSeries,
)
from alphalattice.foundation.feature_engine.producers.preprocessing.robust_cross_section import (
    allows_missing_source_rows,
)
from alphalattice.foundation.feature_engine.publication.snapshots import (
    FeaturePanelSnapshotPublisher,
    PublishedFeaturePanelSnapshot,
)
from alphalattice.foundation.feature_engine.runtime.service import FeatureFoundationService
from alphalattice.foundation.feature_engine.storage.repositories import (
    FeatureStateRepository,
    PanelStateRepository,
)
from alphalattice.foundation.market_data_ops.runtime.remediation import (
    PolicyDecision,
    RemediationAction,
)
from alphalattice.foundation.market_data_ops.runtime.remediation import (
    canonical_hash as remediation_hash,
)
from alphalattice.foundation.market_data_ops.runtime.universe_maintenance import (
    CurrentUniverseMaintenance,
    CurrentUniverseMaintenanceStatus,
    current_universe_maintenance_id,
)
from alphalattice.foundation.market_data_ops.runtime.universe_onboarding import (
    CurrentUniverseOnboardingStatus,
)
from alphalattice.foundation.market_data_ops.sources.manifest import (
    UniverseManifest,
    build_quality_filtered_research_manifest,
    qualification_obligation,
)
from alphalattice.foundation.market_data_ops.sources.membership import (
    UniverseBootstrapRecord,
    journal_members,
    membership_effective_session,
    membership_events_for_transition,
)
from alphalattice.foundation.market_data_ops.sources.price_integrity import (
    TradingSessionAuthority,
)
from alphalattice.foundation.market_data_ops.sources.providers import MarketDataProvider
from alphalattice.foundation.market_data_ops.storage.duckdb import (
    ActionAuditReceipt,
    MarketDataRepository,
)
from alphalattice.protocols.actor_execution.contracts import ActorKind

DataEngineerDiagnoser = Callable[
    [
        object,
        tuple[ListingQualityEvidence, ...],
        PanelImpactProjection,
        AgentExecutionBudget,
    ],
    DataRemediationActorSubmission,
]


@dataclass
class _CycleHold:
    """What one cycle holds between its network edges: the single-writer gate
    and one retained workspace database instance.

    Everything a cycle does around the Universe source, the Provider fetch,
    the Feature phase and a diagnoser call is a few dozen small reads and
    short transactions that used to open, read the metadata of and checkpoint
    the workspace database each. Taken once, the instance serves them all;
    each network edge releases it first and takes it again afterwards, so the
    gate is never held while the process waits on the outside world (a stage
    that retains the instance itself, as the data update does, keeps it open
    across the edge; writable, that is no lock). The gate is
    taken first so no other thread can wait on this thread's gate while this
    thread waits on their connection mode.
    """

    gate: WorkspaceMutationGate
    market_data: MarketDataRepository
    _stack: ExitStack | None = field(default=None, init=False, repr=False)

    def take(self) -> None:
        if self._stack is not None:
            return
        stack = ExitStack()
        try:
            stack.enter_context(self.gate.hold())
            stack.enter_context(self.market_data.database.retain(read_only=False))
        except BaseException:
            stack.close()
            raise
        self._stack = stack

    def release(self) -> None:
        stack, self._stack = self._stack, None
        if stack is not None:
            stack.close()


def _raw_close_points(series: pa.Table) -> dict[str, RawCloseSeries]:
    """Split one listing-ordered close series into each listing's chronological closes.

    The series is ordered by listing then session, so a listing's rows are
    one contiguous run; the runs are found on the dictionary-encoded listing
    column without materializing a Python string per row, and each listing's
    closes stay columns (``RawCloseSeries``), a slice of the series' own.
    """

    if series.num_rows == 0:
        return {}
    encoded = series.column("listing_id").combine_chunks().dictionary_encode()
    indices = encoded.indices.to_numpy()
    listings = encoded.dictionary.to_pylist()
    changes = (indices[1:] != indices[:-1]).nonzero()[0]
    boundaries = [0, *(int(value) + 1 for value in changes), len(indices)]
    sessions = series.column("session_date").to_numpy().astype("datetime64[D]")
    closes = series.column("close").to_numpy().astype("float64")
    return {
        str(listings[int(indices[start])]): RawCloseSeries(sessions[start:stop], closes[start:stop])
        for start, stop in pairwise(boundaries)
    }


def _feature_failure_cause(coverage_summary: Mapping[str, object]) -> dict[str, object] | None:
    """What a Feature build that failed saw, in the stage's words, or None (V444).

    The build keeps its cause in its own summary (`materialization_failure`); the cycle's
    outcome carries it beside the code so the Task that ran the cycle keeps it too.
    """
    failure = coverage_summary.get("materialization_failure")
    if not isinstance(failure, Mapping):
        return None
    return {
        "exception_type": failure.get("exception_type"),
        "detail": failure.get("detail"),
        "step": failure.get("stage"),
        "unit": failure.get("listing_id"),
        "first_session": failure.get("first_target_session"),
        "last_session": failure.get("last_target_session"),
    }


@dataclass(frozen=True)
class _GovernanceEvaluation:
    manifest: UniverseManifest
    result: FeatureInputGatewayResult
    evidence: tuple[ListingQualityEvidence, ...]
    panel_impact: PanelImpactProjection | None
    # The manifest whose Sector evidence served the judgement when the
    # governed manifest has none of its own (a parent bound for a candidate
    # recheck): partial evidence, failures only, no admission.
    sector_source: UniverseManifest | None = None


class WorkspaceMaintenanceRecords(Protocol):
    """The records a maintenance cycle keeps: the cycle, its action audits and its scope.

    The Host passes its DuckDB registry, the records' authority; the coordinator reads and
    writes the cycle's records through these methods only.
    """

    def admit(
        self, request: WorkspaceMaintenanceRequest, *, observed_at: datetime
    ) -> WorkspaceMaintenanceCycle:
        """Admits the request's cycle, or returns the cycle already admitted for it."""
        ...

    def cycle(self, cycle_id: str) -> WorkspaceMaintenanceCycle:
        """Reads one cycle."""
        ...

    def update(
        self,
        cycle_id: str,
        *,
        phase: MaintenancePhase,
        status: MaintenanceStatus,
        observed_at: datetime,
        change_set: MarketDataChangeSet | None = None,
        child_task_refs: tuple[str, ...] | None = None,
        effect_receipts: tuple[str, ...] | None = None,
        retry_after_at: datetime | None = None,
        failure_code: str | None = None,
        transport_workers: int | None = None,
    ) -> WorkspaceMaintenanceCycle:
        """Moves a cycle to its next phase and status."""
        ...

    def record_market_data_scope(
        self,
        cycle_id: str,
        *,
        maintenance_id: str,
        manifest_revision: str,
        as_of_session: date,
        observed_at: datetime,
    ) -> None:
        """Records the maintenance run a cycle admitted, in the cycle's own event log."""
        ...

    def record_action_audit(self, receipt: ActionAuditChainReceipt) -> bool:
        """Appends an action audit to its listing's chain."""
        ...

    def record_data_remediation_failure(self, receipt: DataRemediationFailureReceipt) -> bool:
        """Persists a terminal Agent failure without recording an effect."""
        ...

    def latest_action_audit(
        self, listing_id: str, provider: str, *, full_only: bool = False
    ) -> ActionAuditChainReceipt | None:
        """Reads a listing's newest action audit."""
        ...

    def latest_action_audits(
        self,
        listing_ids: tuple[str, ...],
        provider: str,
        *,
        full_only: bool = False,
    ) -> dict[str, ActionAuditChainReceipt]:
        """Reads the newest action audit per listing in one scan."""
        ...

    def action_audits(
        self,
        listing_ids: tuple[str, ...],
        provider: str,
        *,
        full_only: bool = False,
    ) -> tuple[ActionAuditChainReceipt, ...]:
        """Reads the listings' audit chains, newest first."""
        ...


@dataclass(frozen=True)
class _QualityEvidence:
    """The quality step's judged evidence, before the gateway records its assessment (V109)."""

    manifest: UniverseManifest
    evidence: tuple[ListingQualityEvidence, ...]
    sector_by_listing_id: dict[str, str]
    boundary: Any
    observed_at: datetime
    panel_impact: PanelImpactProjection | None
    sector_unknown: frozenset[str]
    sector_source: UniverseManifest | None


@dataclass
class WorkspaceMaintenanceCoordinator:
    """Drive existing deterministic owners without creating another graph."""

    market_data: MarketDataRepository
    feature_state: FeatureStateRepository
    panel_state: PanelStateRepository
    manifest: UniverseManifest
    provider: MarketDataProvider
    mutation_gate: WorkspaceMutationGate
    readiness_gate: WorkspaceReadinessGate
    feature_foundation: FeatureFoundationService
    registry: WorkspaceMaintenanceRecords
    snapshot_publisher: FeaturePanelSnapshotPublisher | None = None
    feature_input: FeatureInputGovernanceService | None = None
    quality_evaluator: FeatureInputQualityEvaluator | None = None
    diagnose: DataEngineerDiagnoser | None = None
    publish_adjusted_return_revision: Callable[[], object] | None = None
    agent_budget: AgentExecutionBudget = field(default_factory=AgentExecutionBudget)
    historical_revision_detection: str = "EVIDENCE_TRIGGERED_NO_PROVIDER_CHANGE_FEED"
    _session_authority_cache: dict[tuple[date, date], tuple[datetime, TradingSessionAuthority]] = (
        field(default_factory=dict, init=False, repr=False)
    )
    _maintenance_runner: tuple[tuple[object, ...], CurrentUniverseMaintenance] | None = field(
        default=None, init=False, repr=False
    )
    clock: Callable[[], datetime] | None = None

    def __post_init__(self) -> None:
        """Require the installed revision-detection policy and resolve quality/invalidation owners.

        Raises:
            ValueError: The declared historical-revision detection policy is unsupported.
        """
        if self.historical_revision_detection != "EVIDENCE_TRIGGERED_NO_PROVIDER_CHANGE_FEED":
            raise ValueError("unsupported historical-revision detection policy")
        self.quality_evaluator = self.quality_evaluator or FeatureInputQualityEvaluator()
        self.topology = FeatureInvalidationTopology(self.feature_foundation.catalog)

    def run(
        self,
        request: WorkspaceMaintenanceRequest,
        *,
        observed_at: datetime,
        onboarding_work_budget: int | None = None,
        maintenance_work_budget: int | None = None,
    ) -> WorkspaceMaintenanceOutcome:
        """Run bounded maintenance while retaining the workspace cycle hold.

        Args:
            request: Sealed request whose scope this operation executes or projects.
            observed_at: Operational observation clock; callers supply an aware instant.
            onboarding_work_budget: Optional bound on candidate onboarding work in this call.
            maintenance_work_budget: Optional bound on incremental maintenance work in this call.

        Returns:
            The maintenance outcome after the cycle hold is released.

        Raises:
            ValueError: The operational observation clock is invalid.
        """
        now = self._utc(observed_at)
        hold = _CycleHold(self.mutation_gate, self.market_data)
        hold.take()
        try:
            return self._run(
                request,
                now,
                hold,
                onboarding_work_budget=onboarding_work_budget,
                maintenance_work_budget=maintenance_work_budget,
            )
        finally:
            hold.release()

    def _run(
        self,
        request: WorkspaceMaintenanceRequest,
        now: datetime,
        hold: _CycleHold,
        *,
        onboarding_work_budget: int | None,
        maintenance_work_budget: int | None,
    ) -> WorkspaceMaintenanceOutcome:
        cycle = self.registry.admit(request, observed_at=now)
        # A cycle that stopped inside quality governance -- waiting for a
        # decision, pending review, refused while binding the evidence of the
        # child the decision derived, or cancelled there -- resumes governance
        # when it is run again. Market data is already current then, and
        # without this the cycle would read an older admission for the same
        # revision as clearance and build on the parent, silently dropping
        # the decision.
        resuming_quality_deferred = (
            cycle.status
            in {
                MaintenanceStatus.DEFERRED,
                MaintenanceStatus.REVIEW_PENDING,
                MaintenanceStatus.BLOCKED,
                MaintenanceStatus.CANCELLED,
            }
            and cycle.phase is MaintenancePhase.QUALITY
        )
        if cycle.status is MaintenanceStatus.COMPLETED:
            return WorkspaceMaintenanceOutcome(
                cycle.cycle_id,
                MaintenanceStatus.NOOP,
                MaintenancePhase.COMPLETED,
                change_set_hash=(
                    cycle.market_data_change_set.change_set_hash
                    if cycle.market_data_change_set
                    else None
                ),
                child_task_refs=cycle.child_task_refs,
            )
        if (
            cycle.status is MaintenanceStatus.DEFERRED
            and cycle.retry_after_at is not None
            and now < cycle.retry_after_at
        ):
            return WorkspaceMaintenanceOutcome(
                cycle.cycle_id,
                cycle.status,
                cycle.phase,
                change_set_hash=(
                    cycle.market_data_change_set.change_set_hash
                    if cycle.market_data_change_set
                    else None
                ),
                child_task_refs=cycle.child_task_refs,
                retry_after_at=cycle.retry_after_at,
                failure_code=cycle.failure_code,
            )
        if cycle.status is MaintenanceStatus.DEFERRED:
            cycle = self.registry.update(
                cycle.cycle_id,
                phase=cycle.phase,
                status=MaintenanceStatus.RUNNING,
                observed_at=now,
                failure_code=None,
                transport_workers=cycle.transport_workers,
            )
        readiness = self.readiness_gate.refresh_sources_if_due(
            observed_at=now,
            before_fetch=hold.release,
            target_session=request.target_market_session,
            operation_started_at=request.request_clock,
        )
        hold.take()
        if readiness.status is WorkspaceReadinessStatus.INITIALIZATION_REQUIRED:
            return self._finish(
                cycle.cycle_id,
                phase=MaintenancePhase.PREFLIGHT,
                status=MaintenanceStatus.BLOCKED,
                observed_at=now,
                failure_code="workspace_maintenance.initialization_consent_required",
            )
        if readiness.status is WorkspaceReadinessStatus.MANIFEST_UPDATE_PENDING:
            return self._finish(
                cycle.cycle_id,
                phase=MaintenancePhase.PREFLIGHT,
                status=MaintenanceStatus.BLOCKED,
                observed_at=now,
                failure_code="workspace_maintenance.manifest_update_consent_required",
            )
        if readiness.status is WorkspaceReadinessStatus.SOURCE_VERIFICATION_BLOCKED:
            return self._finish(
                cycle.cycle_id,
                phase=MaintenancePhase.PREFLIGHT,
                status=MaintenanceStatus.BLOCKED,
                observed_at=now,
                failure_code=readiness.failure_code,
            )
        if readiness.status is WorkspaceReadinessStatus.ONBOARDING_IN_PROGRESS:
            # First-use onboarding fetches on its own schedule; it runs outside
            # the hold like every other Provider edge.
            hold.release()
            onboarding = self.readiness_gate.resume_onboarding()
            outcome = onboarding.runner.run(
                observed_at=now,
                work_budget=onboarding_work_budget,
            )
            if outcome.status is CurrentUniverseOnboardingStatus.DEFERRED:
                return self._finish(
                    cycle.cycle_id,
                    phase=MaintenancePhase.MARKET_DATA,
                    status=MaintenanceStatus.DEFERRED,
                    observed_at=now,
                    retry_after_at=outcome.deferred.retry_after_at if outcome.deferred else None,
                    failure_code=outcome.failure_code,
                )
            if outcome.status is CurrentUniverseOnboardingStatus.BLOCKED:
                return self._finish(
                    cycle.cycle_id,
                    phase=MaintenancePhase.MARKET_DATA,
                    status=MaintenanceStatus.BLOCKED,
                    observed_at=now,
                    failure_code=outcome.failure_code,
                )
            if outcome.status is not CurrentUniverseOnboardingStatus.COMPLETED:
                return self._finish(
                    cycle.cycle_id,
                    phase=MaintenancePhase.MARKET_DATA,
                    status=MaintenanceStatus.RUNNING,
                    observed_at=now,
                )
            readiness = self.readiness_gate.complete_onboarding(
                onboarding, outcome, observed_at=now
            )
            hold.take()
        if readiness.manifest is not None:
            self._bind_manifest(
                readiness.manifest,
                observed_at=now,
                effective_session=request.target_market_session,
                authority="workspace_readiness",
            )
        working = None
        try:
            if request.candidate_data_recheck is not None:
                # The request carries the recheck for every cycle of the Task.
                # A recheck that already completed answers from the owner's
                # memory under this cycle's retained instance; only an
                # unfinished one reaches the Provider, and nothing is held
                # across that.
                remembered = self.readiness_gate.candidate_data.completed_if_unchanged(
                    request.candidate_data_recheck, target_session=request.target_market_session
                )
                if remembered is not None:
                    outcome, working = remembered
                else:
                    hold.release()
                    outcome, working = self.readiness_gate.candidate_data.advance(
                        request.candidate_data_recheck,
                        target_session=request.target_market_session,
                        observed_at=now,
                        work_budget=onboarding_work_budget,
                    )
                    hold.take()
                if outcome.status is not CurrentUniverseOnboardingStatus.COMPLETED:
                    return self._finish(
                        cycle.cycle_id,
                        phase=MaintenancePhase.MARKET_DATA,
                        status=MaintenanceStatus.BLOCKED
                        if outcome.status is CurrentUniverseOnboardingStatus.BLOCKED
                        else MaintenanceStatus.DEFERRED
                        if outcome.deferred is not None
                        else MaintenanceStatus.RUNNING,
                        observed_at=now,
                        retry_after_at=outcome.deferred.retry_after_at
                        if outcome.deferred
                        else None,
                        failure_code=outcome.failure_code,
                    )
            if request.candidate_recheck is not None:
                # Due Feature/Sector work is executed whatever the raw retry
                # admitted: a retry that admitted nothing leaves the parent as
                # planned, one that admitted candidates activated a merged
                # parent the recheck rides on. Resolved here, every cycle,
                # from the stored request, so a cycle that stopped, was
                # cancelled or was interrupted between the two finds the
                # recheck still owed when it runs again.
                working = self._feature_recheck_working_manifest(request, prepared=working)
        except ValueError as error:
            if not str(error).startswith("workspace_maintenance.candidate_"):
                raise
            # A recheck whose scope no longer describes the source (the
            # candidate document, parent or admission moved since the plan)
            # stops the cycle by that name for a new plan; it is not an
            # interruption of the Task.
            return self._finish(
                cycle.cycle_id,
                phase=MaintenancePhase.PREFLIGHT,
                status=MaintenanceStatus.BLOCKED,
                observed_at=now,
                # The owner's own code; the interface withholds other text when it serves it.
                failure_code=str(error),
            )
        if (
            working is not None
            and self.manifest.revision_sha256 != working.revision_sha256
            and not self.panel_state.admits_feature_candidate_subset(
                candidate_revision=working.revision_sha256,
                result_revision=self.manifest.revision_sha256,
                as_of_session=request.target_market_session,
                allow_sector_preparation=True,
            )
        ):
            if self.manifest.revision_sha256 != request.membership_revision:
                return self._finish(
                    cycle.cycle_id,
                    phase=MaintenancePhase.PREFLIGHT,
                    status=MaintenanceStatus.BLOCKED,
                    observed_at=now,
                    failure_code="workspace_maintenance.candidate_recheck_source_changed",
                )
            self._bind_manifest(
                working,
                observed_at=now,
                effective_session=request.target_market_session,
                authority="candidate_preparation",
            )
        return self._run_cycle(
            cycle,
            request,
            now,
            resuming_quality_deferred=resuming_quality_deferred,
            maintenance_work_budget=maintenance_work_budget,
            hold=hold,
        )

    def _run_cycle(
        self,
        cycle: WorkspaceMaintenanceCycle,
        request: WorkspaceMaintenanceRequest,
        now: datetime,
        *,
        resuming_quality_deferred: bool,
        maintenance_work_budget: int | None,
        hold: _CycleHold,
    ) -> WorkspaceMaintenanceOutcome:
        # After U0 the request must bind exact active membership. Initialization
        # may finish qualifying its own candidate subset; that bounded exception
        # cannot authorize a broader pool or an unrelated stale request.
        if (
            request.membership_revision != self.manifest.revision_sha256
            and not self._request_admits_working_manifest(request)
        ):
            return self._finish(
                cycle.cycle_id,
                phase=MaintenancePhase.PREFLIGHT,
                status=MaintenanceStatus.BLOCKED,
                observed_at=now,
                failure_code="workspace_maintenance.membership_revision_mismatch",
            )

        # A quality-governance child changes membership identity, not provider
        # observations. Rebind only evidence that still matches the child's
        # current mapping and source hashes so a verified child does not repeat
        # network work merely because its manifest identity changed.
        self.mutation_gate.run(
            self.market_data.bind_available_action_audit_receipts_to_manifest,
            self.manifest,
            requested_as_of=request.target_market_session,
            now=now,
        )
        self.feature_foundation.sector_activation.bind_to_manifest(self.manifest)

        active_panel = self.panel_state.active_feature_panel(request.market_profile_id)
        needs_feature_initialization = active_panel is None
        # A predecessor Panel does not supply Sector evidence for newly admitted
        # names. The existing Feature build establishes that evidence; quality
        # governance must then run before this maintenance cycle can complete.
        sector_available = self.feature_state.current_sector_state(self.manifest) is not None
        awaiting_transition_quality = needs_feature_initialization or not sector_available
        maintenance_operation_id = current_universe_maintenance_id(
            self.manifest,
            as_of_session=request.target_market_session,
            authorized_full_history_listing_ids=tuple(sorted(request.full_history_listing_ids)),
        )
        requested_full_listing_ids = self._authorized_full_audit_listing_ids(request)
        authorized_full_listing_ids = self._pending_authorized_full_audit_listing_ids(
            request,
            requested_full_listing_ids,
        )
        required_full_listing_ids = tuple(
            sorted(
                set(
                    self._required_full_audit_listing_ids(
                        now,
                        as_of_session=request.target_market_session,
                        progress_operation_id=maintenance_operation_id,
                    )
                )
                - set(authorized_full_listing_ids)
            )
        )
        maintenance_change_set = cycle.market_data_change_set or MarketDataChangeSet()
        retry_grants = self._elapsed_wait_retry_grants(now=now)
        # Daily maintenance owns the currently admitted research membership.
        # Acquisition candidates that previously failed quality cannot make an
        # otherwise fresh workspace re-download the whole active universe on
        # every launch; they return only through their governed requalification
        # path or a newly approved candidate transition.
        candidate_raw_through = self.market_data.manifest_raw_through(self.manifest)
        candidate_adjusted_through = self.market_data.manifest_provider_adjusted_through(
            self.manifest
        )
        market_data_needs_refresh = (
            candidate_raw_through is None
            or candidate_raw_through < request.target_market_session
            or candidate_adjusted_through is None
            or candidate_adjusted_through < request.target_market_session
            or bool(authorized_full_listing_ids)
            or bool(required_full_listing_ids)
        )
        # A Panel is unreadable by Factor and Risk without a Feature Input
        # Gateway admission, so the absence of one is itself a reason to govern.
        # Refresh alone is not enough: a freshly initialized workspace skips
        # governance on its first cycle for want of an active Panel, and every
        # later cycle finds market data already current, which would leave the
        # admission permanently unreachable.
        needs_quality_admission = self.feature_input is not None and (
            self.panel_state.feature_input_quality_disclosure(
                result_manifest_revision=self.manifest.revision_sha256
            ).get("gateway_qualified")
            is not True
            or resuming_quality_deferred
            or self.panel_state.has_unresolved_feature_input(self.manifest.revision_sha256)
        )
        governed = False
        if not needs_feature_initialization and market_data_needs_refresh:
            prior_listing_ids = {item.listing_id for item in self.manifest.listings}
            raw_range = self.market_data.manifest_raw_range(self.manifest)
            maintenance = self._maintenance_for(
                request,
                now,
                history_start=raw_range[0] if raw_range else request.target_market_session,
                authorized_full_listing_ids=authorized_full_listing_ids,
                required_full_listing_ids=required_full_listing_ids,
                requested_full_listing_ids=requested_full_listing_ids,
                transport_workers=cycle.transport_workers,
                retry_grants=retry_grants,
            )
            # The run this cycle keys its listing units under is the cycle's own
            # record: the manifest bound here, which a transition's request does
            # not name, is what a later reader of this cycle's units resolves.
            self.registry.record_market_data_scope(
                cycle.cycle_id,
                maintenance_id=maintenance.maintenance_id,
                manifest_revision=self.manifest.revision_sha256,
                as_of_session=request.target_market_session,
                observed_at=now,
            )
            # The runner releases the hold itself right before it reaches
            # the Provider and takes it again once its last fetch is back, so
            # this preflight, the runner's own bookkeeping and the cycle's
            # epilogue share one retained instance around the fetch.
            maintenance_outcome = maintenance.run(
                observed_at=now,
                work_budget=maintenance_work_budget,
                before_fetch=hold.release,
                after_fetch=hold.take,
            )
            if maintenance_outcome.status is CurrentUniverseMaintenanceStatus.DEFERRED:
                return self._finish(
                    cycle.cycle_id,
                    phase=MaintenancePhase.MARKET_DATA,
                    status=MaintenanceStatus.DEFERRED,
                    observed_at=now,
                    retry_after_at=now + timedelta(minutes=5),
                    failure_code=maintenance_outcome.failure_code,
                    transport_workers=max(1, cycle.transport_workers // 2),
                )
            if maintenance_outcome.status is CurrentUniverseMaintenanceStatus.RUNNING:
                return self._finish(
                    cycle.cycle_id,
                    phase=MaintenancePhase.MARKET_DATA,
                    status=MaintenanceStatus.RUNNING,
                    observed_at=now,
                )
            maintenance_change_set = self._merge_change_sets(
                maintenance_change_set,
                self._record_change_set(
                    maintenance_outcome.listing_changes,
                    request=request,
                    observed_at=now,
                ),
            )
            approval_required = tuple(
                sorted(
                    item.listing_id
                    for item in self.market_data.current_universe_maintenance_listings(
                        maintenance.maintenance_id
                    )
                    if item.failure_code == "data.full_history_audit_approval_required"
                )
            )
            if approval_required:
                return self._finish(
                    cycle.cycle_id,
                    phase=MaintenancePhase.MARKET_DATA,
                    status=MaintenanceStatus.BLOCKED,
                    observed_at=now,
                    failure_code="data.full_history_audit_approval_required",
                    change_set=maintenance_change_set,
                    effect_receipts=(
                        canonical_hash(
                            {
                                "kind": "FullHistoryAuditRequirement",
                                "market_profile_id": request.market_profile_id,
                                "target_market_session": request.target_market_session,
                                "listing_ids": approval_required,
                            }
                        ),
                    ),
                )
            governance = (
                self._govern_quality(
                    maintenance.maintenance_id,
                    request=request,
                    observed_at=now,
                    macro_retry_exhausted=resuming_quality_deferred,
                    observed_workers=cycle.transport_workers,
                )
                if sector_available
                else self._govern_failed_members_before_feature_build(
                    maintenance.maintenance_id,
                    request=request,
                    observed_at=now,
                    macro_retry_exhausted=resuming_quality_deferred,
                    observed_workers=cycle.transport_workers,
                )
            )
            governed = sector_available and governance is not None
            # Handling a governance result may hand a case to a diagnoser (a
            # model call); nothing is held across it.
            hold.release()
            if (
                governance is not None
                and not sector_available
                and not governance.result.agent_cases
                and governance.result.deferred is None
            ):
                # Failures-only governance found nothing to decide. Members
                # still failed (every failure held by a standing quarantine):
                # the day cannot complete, and the cause is the one the
                # derived binding would have stopped by after the build.
                # No member failed (the earlier cases were cleared by this
                # evidence): the build proceeds.
                guarded: tuple[MaintenanceStatus, datetime | None, str | None] | None = (
                    (MaintenanceStatus.BLOCKED, None, "data.listing_updates_incomplete")
                    if self._failed_members(maintenance.maintenance_id)
                    else None
                )
            else:
                guarded = (
                    self._handle_governance_result(
                        governance,
                        maintenance_id=maintenance.maintenance_id,
                        request=request,
                        observed_at=now,
                        observed_workers=cycle.transport_workers,
                    )
                    if governance is not None
                    else None
                )
            if guarded is not None:
                return self._finish(
                    cycle.cycle_id,
                    phase=MaintenancePhase.QUALITY,
                    status=guarded[0],
                    observed_at=now,
                    retry_after_at=guarded[1],
                    failure_code=guarded[2],
                    change_set=maintenance_change_set,
                )
            current_listing_ids = {item.listing_id for item in self.manifest.listings}
            if current_listing_ids != prior_listing_ids:
                maintenance_change_set = MarketDataChangeSet(
                    listing_changes=maintenance_change_set.listing_changes,
                    membership_additions=tuple(sorted(current_listing_ids - prior_listing_ids)),
                    membership_removals=tuple(sorted(prior_listing_ids - current_listing_ids)),
                    receipt_hashes=maintenance_change_set.receipt_hashes,
                )
                active_panel = self.panel_state.active_feature_panel(request.market_profile_id)

        # Governance retains on its own and may hand a case to a diagnoser (a
        # model call); the hold ends here in every branch.
        hold.release()
        if not governed and needs_quality_admission and sector_available:
            governance = self._govern_quality(
                None,
                request=request,
                observed_at=now,
                macro_retry_exhausted=resuming_quality_deferred,
                observed_workers=cycle.transport_workers,
            )
            guarded = self._handle_governance_result(
                governance,
                maintenance_id=None,
                request=request,
                observed_at=now,
                observed_workers=cycle.transport_workers,
            )
            governed = True
            if guarded is not None:
                return self._finish(
                    cycle.cycle_id,
                    phase=MaintenancePhase.QUALITY,
                    status=guarded[0],
                    observed_at=now,
                    retry_after_at=guarded[1],
                    failure_code=guarded[2],
                    change_set=maintenance_change_set,
                )
            active_panel = self.panel_state.active_feature_panel(request.market_profile_id)
            # The disclosure is sealed into the Panel manifest at publication, so
            # a Panel published before the admission still reads as unqualified.
            # Republish from the existing Feature rows; no rebuild is required.
            # Only while the Panel still answers for the current manifest: an
            # admission that derives a new child revision leaves the Panel stale,
            # and rebuilding it against that revision belongs to the feature
            # phase below, which already plans a panel_binding_change.
            if (
                active_panel is not None
                and self.snapshot_publisher is not None
                and str(active_panel["manifest_revision"]) == self.manifest.revision_sha256
            ):
                self._publish_snapshot_with_bounded_retry(
                    history_start=self._history_start(),
                    as_of_session=request.target_market_session,
                    observed_at=now,
                )

        if active_panel is not None:
            try:
                maintenance_change_set = self._merge_change_sets(
                    maintenance_change_set,
                    self._unconsumed_feature_source_change_set(
                        active_panel,
                        through=request.target_market_session,
                    ),
                )
            except ValueError:
                return self._finish(
                    cycle.cycle_id,
                    phase=MaintenancePhase.MARKET_DATA,
                    status=MaintenanceStatus.BLOCKED,
                    observed_at=now,
                    failure_code=("workspace_maintenance.feature_source_reconciliation_failed"),
                    change_set=maintenance_change_set,
                )

        if not sector_available:
            # A candidate recheck whose candidates still have no Sector keeps them quarantined
            # without refreshing every member's Sector or building the Features twice.
            applied, refused = self._candidate_only_sector_recheck(request, observed_at=now)
            if applied:
                return self._finish(
                    cycle.cycle_id,
                    phase=MaintenancePhase.FEATURE,
                    status=MaintenanceStatus.BLOCKED if refused else MaintenanceStatus.RUNNING,
                    observed_at=now,
                    failure_code=refused or "sector.recoverable_exclusion_applied",
                    change_set=maintenance_change_set,
                )

        # The Feature phase reaches the Provider (SPY) and its own worker
        # threads; the hold must not span it.
        hold.release()
        old_spy = self.feature_state.market_reference("SPY")
        spy_outcome = self.feature_foundation.refresh_spy(
            as_of_session=request.target_market_session,
            observed_at=now,
        )
        if spy_outcome.status is FeatureBuildStatus.DEFERRED:
            return self._finish(
                cycle.cycle_id,
                phase=MaintenancePhase.FEATURE,
                status=MaintenanceStatus.DEFERRED,
                observed_at=now,
                retry_after_at=spy_outcome.retry_after_at,
                failure_code=spy_outcome.failure_code,
                change_set=maintenance_change_set,
            )
        if spy_outcome.status is not FeatureBuildStatus.COMPLETED:
            return self._finish(
                cycle.cycle_id,
                phase=MaintenancePhase.FEATURE,
                status=MaintenanceStatus.BLOCKED,
                observed_at=now,
                failure_code=spy_outcome.failure_code,
                change_set=maintenance_change_set,
            )
        current_spy = self.feature_state.market_reference("SPY")
        if current_spy is None:
            raise AssertionError("completed SPY refresh did not publish a reference")
        spy_changed = (
            old_spy is not None and old_spy["revision_hash"] != current_spy["revision_hash"]
        ) or (
            active_panel is not None
            and str(active_panel["spy_revision"]) != str(current_spy["revision_hash"])
        )
        spy_correction_start = None
        spy_return_change_sessions: tuple[date, ...] = ()
        spy_action_receipt = current_spy.get("action_audit_receipt_hash")
        if spy_changed and spy_action_receipt:
            adjusted_revision = self.market_data.provider_adjusted_revision(str(spy_action_receipt))
            if adjusted_revision is not None:
                spy_return_change_sessions = adjusted_revision.changed_return_sessions
                if spy_return_change_sessions:
                    spy_correction_start = min(spy_return_change_sessions)
        change_set = MarketDataChangeSet(
            listing_changes=maintenance_change_set.listing_changes,
            spy_correction_start=spy_correction_start,
            spy_return_change_sessions=spy_return_change_sessions,
            sector_revision_changed=maintenance_change_set.sector_revision_changed,
            membership_additions=maintenance_change_set.membership_additions,
            membership_removals=maintenance_change_set.membership_removals,
            receipt_hashes=tuple(
                sorted(
                    set(
                        maintenance_change_set.receipt_hashes
                        + ((str(spy_action_receipt),) if spy_action_receipt else ())
                    )
                )
            ),
        )
        if needs_feature_initialization:
            from alphalattice.foundation.feature_engine.contracts import FeatureInvalidation

            invalidations: tuple[FeatureInvalidation, ...] = (
                FeatureInvalidation("initial_backfill"),
            )
        else:
            invalidations = self.topology.plan(
                change_set,
                history_start=self._history_start(),
                as_of_session=request.target_market_session,
            ).invalidations
            if active_panel is not None and (
                str(active_panel["manifest_revision"]) != self.manifest.revision_sha256
                or str(active_panel["catalog_hash"])
                != self.feature_foundation.catalog.binding.catalog_hash
            ):
                from alphalattice.foundation.feature_engine.contracts import FeatureInvalidation

                # A catalog change recomputes everything. A manifest change is
                # a new governance relationship over the same rows: the Panel
                # is rebuilt under its binding and reuses every partition whose
                # cross-sections it still computes.
                binding_kind = (
                    "catalog_binding_change"
                    if str(active_panel["catalog_hash"])
                    != self.feature_foundation.catalog.binding.catalog_hash
                    else "panel_binding_change"
                )
                invalidations = tuple(
                    sorted(
                        {
                            *invalidations,
                            FeatureInvalidation(
                                binding_kind,
                                earliest_session=self._history_start(),
                            ),
                        },
                        key=lambda item: (
                            item.kind,
                            item.listing_id or "",
                            item.earliest_session or request.target_market_session,
                        ),
                    )
                )
        try:
            quality_change = self._source_eligibility_invalidation(request)
        except (ValueError, OSError):
            return self._finish(
                cycle.cycle_id,
                phase=MaintenancePhase.QUALITY,
                status=MaintenanceStatus.BLOCKED,
                observed_at=now,
                failure_code="feature.source_eligibility_unavailable",
                change_set=change_set,
            )
        if quality_change is not None:
            invalidations = (*invalidations, quality_change)
        current_members = {item.listing_id for item in self.manifest.listings}
        if any(
            item.listing_id is not None and item.listing_id not in current_members
            for item in invalidations
        ):
            axis, _scopes = self.feature_foundation._axis_listings()
            requested = self.market_data.load_universe_manifest_revision(
                request.membership_revision
            )
            # A saved candidate can fail qualification without ever joining the
            # Panel's axis. Historical members keep their owed source work;
            # unknown names remain for the Feature owner's refusal.
            never_admitted = {item.listing_id for item in requested.listings} - {
                item.listing_id for item in axis
            }
            invalidations = tuple(
                item
                for item in invalidations
                if item.kind == "panel_binding_change" or item.listing_id not in never_admitted
            )
        if (
            not invalidations
            and active_panel is not None
            and active_panel["as_of_session"] >= request.target_market_session
        ):
            qualification = self._qualify_features_for_membership(request=request, observed_at=now)
            if qualification is not None:
                return self._finish(
                    cycle.cycle_id,
                    phase=MaintenancePhase.QUALITY,
                    status=qualification[0],
                    failure_code=qualification[1],
                    failure_cause=qualification[2],
                    observed_at=now,
                    change_set=change_set,
                )
            snapshot = self.panel_state.feature_panel_snapshot_for_active(request.market_profile_id)
            if snapshot is None or governed:
                if self.snapshot_publisher is None:
                    return self._finish(
                        cycle.cycle_id,
                        phase=MaintenancePhase.FEATURE,
                        status=MaintenanceStatus.BLOCKED,
                        observed_at=now,
                        failure_code="workspace_maintenance.snapshot_publisher_missing",
                        change_set=change_set,
                    )
                published = self._publish_snapshot_with_bounded_retry(
                    history_start=self._history_start(),
                    as_of_session=request.target_market_session,
                    observed_at=now,
                )
                if published is None:
                    return self._finish(
                        cycle.cycle_id,
                        phase=MaintenancePhase.FEATURE,
                        status=MaintenanceStatus.BLOCKED,
                        observed_at=now,
                        failure_code="workspace_maintenance.snapshot_publication_failed",
                        change_set=change_set,
                    )
                return self._finish(
                    cycle.cycle_id,
                    phase=MaintenancePhase.COMPLETED,
                    status=MaintenanceStatus.COMPLETED,
                    observed_at=now,
                    change_set=change_set,
                    effect_receipts=(published.manifest.snapshot_hash,),
                )
            # The Panel is published and there is no Feature work to do, which is
            # also the state a Panel published before Gateway admission stays in
            # forever: its semantic index was withheld on purpose, the admission
            # has since arrived, and no other path ever revisits it. Completing
            # the handoff here reads the immutable manifest and chunks that
            # already exist -- no Feature value is recomputed, and nothing is
            # republished -- and it is a no-op once the index resolves.
            handed_off: str | None = None
            if self.snapshot_publisher is not None:
                try:
                    handed_off = self.snapshot_publisher.complete_semantic_index_handoff(
                        manifest=self.manifest
                    )
                except (OSError, ValueError):
                    return self._finish(
                        cycle.cycle_id,
                        phase=MaintenancePhase.FEATURE,
                        status=MaintenanceStatus.BLOCKED,
                        observed_at=now,
                        failure_code="workspace_maintenance.semantic_index_handoff_failed",
                        change_set=change_set,
                    )
            return self._finish(
                cycle.cycle_id,
                phase=MaintenancePhase.COMPLETED,
                status=MaintenanceStatus.COMPLETED,
                observed_at=now,
                change_set=change_set,
                effect_receipts=(handed_off,) if handed_off else None,
            )
        if needs_feature_initialization and sector_available:
            # U0's baseline qualification answers from computed base values.
            # When an earlier build of this first use (one refused at its
            # Panel, or one over a membership governance then replaced) has
            # already materialized every candidate's as-of row, the judgement
            # is made here, before a Panel is composed for a membership it
            # would derive away from -- but an exclusion is taken from a row
            # only when that row's materialization still applies to the
            # current inputs (the qualification owner's proof); a row the
            # source has moved under, or one that is not yet computed, leaves
            # the judgement to the build, exactly as before. Nothing is
            # published before the judgement on either path.
            qualification = self._qualify_features_for_membership(
                request=request, observed_at=now, only_when_materialized=True
            )
            if qualification is not None:
                return self._finish(
                    cycle.cycle_id,
                    phase=MaintenancePhase.QUALITY,
                    status=qualification[0],
                    failure_code=qualification[1],
                    failure_cause=qualification[2],
                    observed_at=now,
                    change_set=change_set,
                )
        feature_request = FeatureBuildRequest.create(
            manifest_revision=self.manifest.revision_sha256,
            catalog=self.feature_foundation.catalog.binding,
            spy_revision=str(current_spy["revision_hash"]),
            history_start=self._history_start(),
            as_of_session=request.target_market_session,
            invalidations=invalidations,
        )
        # The build is this cycle's step, under the Task that runs the cycle (W10, V102): its
        # per-listing receipts make a build that stopped cheap to run again, and a deferred one is
        # built again by the next cycle once its retry time has passed. A first use whose
        # membership waits for its baseline qualification composes no Panel for it (V311).
        compose_panel = not (
            needs_feature_initialization and self._baseline_awaits_qualification(request)
        )
        outcome = self.feature_foundation.build(
            feature_request,
            observed_at=self._utc(self.clock() if self.clock is not None else datetime.now(UTC)),
            refresh_sector=True,
            compose_panel=compose_panel,
        )
        if outcome.status is FeatureBuildStatus.COMPLETED:
            qualification = self._qualify_features_for_membership(request=request, observed_at=now)
            if qualification is not None:
                return self._finish(
                    cycle.cycle_id,
                    phase=MaintenancePhase.QUALITY,
                    status=qualification[0],
                    failure_code=qualification[1],
                    failure_cause=qualification[2],
                    observed_at=now,
                    change_set=change_set,
                )
            if not compose_panel:
                return self._finish(
                    cycle.cycle_id,
                    phase=MaintenancePhase.FEATURE,
                    status=MaintenanceStatus.RUNNING,
                    observed_at=now,
                    change_set=change_set,
                )
            if self.snapshot_publisher is None:
                return self._finish(
                    cycle.cycle_id,
                    phase=MaintenancePhase.FEATURE,
                    status=MaintenanceStatus.BLOCKED,
                    observed_at=now,
                    failure_code="workspace_maintenance.snapshot_publisher_missing",
                    change_set=change_set,
                )
            published = self._publish_snapshot_with_bounded_retry(
                history_start=self._history_start(),
                as_of_session=request.target_market_session,
                observed_at=now,
            )
            if published is None:
                return self._finish(
                    cycle.cycle_id,
                    phase=MaintenancePhase.FEATURE,
                    status=MaintenanceStatus.BLOCKED,
                    observed_at=now,
                    failure_code="workspace_maintenance.snapshot_publication_failed",
                    change_set=change_set,
                )
            return self._finish(
                cycle.cycle_id,
                phase=MaintenancePhase.QUALITY
                if awaiting_transition_quality and needs_quality_admission and not governed
                else MaintenancePhase.COMPLETED,
                status=MaintenanceStatus.RUNNING
                if awaiting_transition_quality and needs_quality_admission and not governed
                else MaintenanceStatus.COMPLETED,
                observed_at=now,
                change_set=change_set,
                effect_receipts=tuple(
                    value
                    for value in (outcome.receipt_hash, published.manifest.snapshot_hash)
                    if value
                ),
            )
        status = (
            MaintenanceStatus.DEFERRED
            if outcome.status is FeatureBuildStatus.DEFERRED
            else MaintenanceStatus.BLOCKED
        )
        if outcome.failure_code in {
            "feature.panel_coverage_not_ready",
            "feature.membership_admission_required",
        }:
            qualification = self._qualify_features_for_membership(request=request, observed_at=now)
            if qualification is not None:
                return self._finish(
                    cycle.cycle_id,
                    phase=MaintenancePhase.QUALITY,
                    status=qualification[0],
                    failure_code=qualification[1],
                    failure_cause=qualification[2],
                    observed_at=now,
                    change_set=change_set,
                )
        failure_code = outcome.failure_code
        if (
            status is MaintenanceStatus.BLOCKED
            and outcome.failure_code == "sector.partial_current_sector"
        ):
            refused = self._apply_sector_partial_exclusion(request=request, observed_at=now)
            if refused is None:
                return self._finish(
                    cycle.cycle_id,
                    phase=MaintenancePhase.FEATURE,
                    status=MaintenanceStatus.RUNNING,
                    observed_at=now,
                    failure_code="sector.recoverable_exclusion_applied",
                    change_set=change_set,
                )
            failure_code = refused
        return self._finish(
            cycle.cycle_id,
            phase=MaintenancePhase.FEATURE,
            status=status,
            observed_at=now,
            failure_code=failure_code,
            failure_cause=(
                _feature_failure_cause(outcome.coverage_summary)
                if failure_code == outcome.failure_code
                else None
            ),
            change_set=change_set,
            retry_after_at=outcome.retry_after_at,
        )

    def _source_eligibility_invalidation(
        self, request: WorkspaceMaintenanceRequest
    ) -> FeatureInvalidation | None:
        """A changed dated mask needs Panel work even when no price changed."""
        if self.snapshot_publisher is None:
            return None
        snapshot = self.panel_state.feature_panel_snapshot_for_active(request.market_profile_id)
        if snapshot is None:
            return None
        payload = self.snapshot_publisher.resolver.load_feature_panel_manifest(
            str(snapshot["manifest_uri"])
        )
        old = payload.get("safe_summary", {}).get("membership", {}).get("source_exclusions", ())
        axis, _scopes = self.feature_foundation._axis_listings()
        current = self.panel_state.panel_source_exclusions(
            market_profile_id=request.market_profile_id,
            sessions=(
                date.fromisoformat(str(payload["history_start"])),
                request.target_market_session,
            ),
            listing_ids=tuple(item.listing_id for item in axis),
        )
        previous = tuple(
            sorted(
                (str(item["listing_id"]), str(item["first_session"]), str(item["last_session"]))
                for item in old
            )
        )
        following = tuple(
            sorted(
                (item.listing_id, item.first_session.isoformat(), item.last_session.isoformat())
                for item in current
            )
        )
        if previous == following:
            return None
        return FeatureInvalidation(
            "panel_binding_change",
            earliest_session=min(date.fromisoformat(item[1]) for item in (*previous, *following)),
            source_receipt_hash=canonical_hash({"source_eligibility": following}),
        )

    def _refused_binding_cause(
        self, request: WorkspaceMaintenanceRequest, listing_ids: tuple[str, ...]
    ) -> str:
        """The stable code for a derived membership whose evidence would not bind."""

        maintenance_id = current_universe_maintenance_id(
            self.manifest,
            as_of_session=request.target_market_session,
            authorized_full_history_listing_ids=tuple(sorted(request.full_history_listing_ids)),
        )
        failed = {
            item.listing_id
            for item in self.market_data.current_universe_maintenance_listings(maintenance_id)
            if item.state == "FAILED"
        }
        if failed & set(listing_ids):
            return "data.listing_updates_incomplete"
        return "workspace_maintenance.derived_manifest_evidence_incomplete"

    def _feature_recheck_working_manifest(
        self, request: WorkspaceMaintenanceRequest, *, prepared: UniverseManifest | None
    ) -> UniverseManifest:
        """The manifest the request's Feature/Sector recheck prepares.

        The readiness owner resolves the scope: against the parent as planned
        when nothing moved it, across the request's own raw-retry transition
        when that retry admitted candidates, and by refusal otherwise. The
        manifest the raw retry says it prepared must be the one the gate
        resolves; a disagreement between the two owners is a changed source,
        not a manifest to work on.
        """

        scope = request.candidate_recheck
        assert scope is not None
        working = self.readiness_gate.resolve_feature_candidate_recheck(
            scope,
            prior_revision=request.membership_revision,
            data_recheck=request.candidate_data_recheck,
            target_session=request.target_market_session,
        )
        if prepared is not None and prepared.revision_sha256 != working.revision_sha256:
            raise ValueError("workspace_maintenance.candidate_recheck_source_changed")
        return working

    def _request_admits_working_manifest(self, request: WorkspaceMaintenanceRequest) -> bool:
        """Admit declared preparation work separately from qualified research membership.

        After U0 preparation requires an exact dated prerequisite admission and
        retains nominal members. ENTRY still needs baseline proof; mere subset
        shape cannot authorize a stale or different population.
        """
        candidate_revision = request.membership_revision
        working = None
        if request.candidate_data_recheck is not None:
            working = self.readiness_gate.candidate_data.completed(
                request.candidate_data_recheck, target_session=request.target_market_session
            )
        if request.candidate_recheck is not None:
            working = self._feature_recheck_working_manifest(request, prepared=working)
        if working is not None:
            candidate_revision = working.revision_sha256
            if self.manifest.revision_sha256 == candidate_revision:
                return True  # Base preparation only; the Panel still requires journal admission.
        bootstrap = self.market_data.universe_bootstrap(request.market_profile_id)
        if bootstrap is not None:
            nominal = journal_members(
                bootstrap, self.market_data.membership_events(request.market_profile_id)
            )
            return set(nominal) <= {
                item.listing_id for item in self.manifest.listings
            } and self.panel_state.admits_feature_candidate_subset(
                candidate_revision=candidate_revision,
                result_revision=self.manifest.revision_sha256,
                as_of_session=request.target_market_session,
                allow_sector_preparation=True,
            )
        obligation = qualification_obligation(
            "baseline_features", desktop_core_feature_bundle().bundle_hash
        )
        if obligation not in self.manifest.qualification_obligations:
            return False
        try:
            parent = self.market_data.load_universe_manifest_revision(request.membership_revision)
            derived = build_quality_filtered_research_manifest(
                parent,
                eligible_listing_ids=tuple(item.listing_id for item in self.manifest.listings),
                qualification_obligations=self.manifest.qualification_obligations,
            )
        except ValueError:
            return False
        return bool(
            parent.profile == self.manifest.profile
            and derived.revision_sha256 == self.manifest.revision_sha256
        )

    def _baseline_awaits_qualification(self, request: WorkspaceMaintenanceRequest) -> bool:
        """Whether a first use's membership still waits for its baseline qualification (V311).

        With no universe bootstrap and no baseline obligation recorded, the judgement after the
        build binds a qualified manifest (`_qualify_features_for_membership`), so a Panel composed
        for this membership would never be published: the build materializes the base values the
        judgement reads, activates the Sector revision and composes nothing, and the next cycle
        composes the Panel once, for the qualified membership.
        """
        if not self.manifest.listings:
            return False
        if self.market_data.universe_bootstrap(request.market_profile_id) is not None:
            return False
        obligation = qualification_obligation(
            "baseline_features", desktop_core_feature_bundle().bundle_hash
        )
        return obligation not in self.manifest.qualification_obligations

    def _qualify_features_for_membership(
        self,
        *,
        request: WorkspaceMaintenanceRequest,
        observed_at: datetime,
        only_when_materialized: bool = False,
    ) -> tuple[MaintenanceStatus, str, Mapping[str, object] | None] | None:
        """Qualify U0 or new candidates from already-computed base values before entry.

        Raw onboarding is only candidate preparation. The shipped base bundle
        is the baseline contract; arbitrary installed research extensions do not
        silently become extra Universe requirements. This admission does not
        replace the Gateway's separate full-quality audit.

        ``only_when_materialized`` answers ``None`` while any candidate's
        as-of row is still pending -- not yet computed, or computed from
        inputs that are no longer current (the qualification owner reports
        such an exclusion as pending) -- leaving the judgement to the call
        after the build that computes it.

        The assessment is made at the coordinator's clock now, the same
        operational instant `_quality_evidence` and the Sector
        exclusion take: the cycle's entry time predates a Feature build run
        inside the cycle, whose Sector observation is stamped at the runner's
        later clock, and a knowledge boundary sealed at the entry time refused
        that observation as if it came from the future (2026-09-15, run 1). An
        explicit `request.knowledge_cutoff_at` still binds and still refuses a
        later observation; a source observed after this instant is still refused.
        """
        observed_at = self._utc(self.clock() if self.clock is not None else observed_at)
        bootstrap = self.market_data.universe_bootstrap(request.market_profile_id)
        prior = (
            set(
                journal_members(
                    bootstrap, self.market_data.membership_events(request.market_profile_id)
                )
            )
            if bootstrap is not None
            else set()
        )
        candidates = tuple(
            item.listing_id for item in self.manifest.listings if item.listing_id not in prior
        )
        if not candidates:
            return None
        if self.feature_input is None:
            return MaintenanceStatus.BLOCKED, "feature.baseline_gateway_required", None
        parent = self.manifest
        bundle = desktop_core_feature_bundle()
        if not set(bundle.factor_ids).issubset(self.feature_foundation.catalog.factor_ids):
            return MaintenanceStatus.BLOCKED, "feature.baseline_bundle_unavailable", None
        obligation = qualification_obligation("baseline_features", bundle.bundle_hash)
        qualification = self.feature_foundation.qualify_features(
            listing_ids=candidates,
            factor_ids=bundle.factor_ids,
            as_of_session=request.target_market_session,
            require_current_source=only_when_materialized,
            observed_at=observed_at,
        )
        if only_when_materialized and qualification.pending_listing_ids:
            return None
        if (
            bootstrap is None
            and obligation in parent.qualification_obligations
            and not (qualification.exclusions or qualification.pending_listing_ids)
        ):
            return None
        sector = self.feature_state.current_sector_state(parent)
        if sector is None:
            return MaintenanceStatus.BLOCKED, "feature.baseline_sector_required", None
        from alphalattice.foundation.feature_engine.contracts import TemporalKnowledgeBoundary

        boundary = TemporalKnowledgeBoundary(
            market_as_of_session=request.target_market_session,
            knowledge_cutoff_at=request.knowledge_cutoff_at or observed_at,
            materialized_at=observed_at,
            universe_source_observed_at=datetime.combine(
                parent.profile.manifest_as_of, datetime.min.time(), tzinfo=UTC
            ),
            sector_source_observed_at=sector.sector_observed_at,
            universe_point_in_time_qualified=False,
            sector_point_in_time_qualified=False,
        )
        try:
            result = self.feature_input.admit_feature_baseline(
                candidate_manifest=parent,
                qualification=qualification,
                sector=sector,
                temporal_boundary=boundary,
                observed_at=observed_at,
            )
        except ValueError as exc:
            if not str(exc).startswith("feature."):
                raise
            # A short population says how short and why, beside its code.
            return MaintenanceStatus.BLOCKED, str(exc), getattr(exc, "cause", None)
        reduced = result.research_manifest
        assert reduced is not None
        if not self._bind_derived_manifest_evidence(
            parent,
            reduced,
            requested_as_of=request.target_market_session,
            observed_at=observed_at,
        ):
            return MaintenanceStatus.BLOCKED, "feature.baseline_evidence_binding_failed", None
        self._bind_manifest(
            reduced,
            observed_at=observed_at,
            effective_session=request.target_market_session,
            authority="baseline_feature_qualification"
            if bootstrap is None
            else "qualified_source_membership",
            reference_hash=qualification.evidence_hash,
        )
        return MaintenanceStatus.RUNNING, "feature.baseline_qualification_applied", None

    def _candidate_only_sector_recheck(
        self, request: WorkspaceMaintenanceRequest, *, observed_at: datetime
    ) -> tuple[bool, str | None]:
        """Recheck only the candidates' Sector while the members' Sector is current.

        A candidate recheck works on the active membership plus its candidates, a revision no
        Sector reference covers, so the Feature build refreshed every member's Sector (one
        provider call a listing) to learn the candidates', and built again over the reduction
        when a candidate still had none. While the active membership holds a complete Sector
        revision inside its refresh interval, only the candidates are fetched. When none of
        them has a Sector, they stay quarantined by the recoverable exclusion over the members'
        carried evidence, which keeps its observation time; the answer is (True, its refusal or
        None). Every other case answers (False, None) and the full refresh runs as before: no
        recheck, a stale or incomplete reference, a candidate outside the recheck, a provider
        failure other than a missing Sector, a candidate that now has a Sector, or a reduction
        that would not be the active membership.
        """
        scope = request.candidate_recheck
        if scope is None or self.feature_input is None:
            return False, None
        working = self.manifest
        active = self.market_data.load_universe_manifest_revision(request.membership_revision)
        member_ids = {item.listing_id for item in active.listings}
        candidates = tuple(item for item in working.listings if item.listing_id not in member_ids)
        if (
            not candidates
            or not member_ids < {item.listing_id for item in working.listings}
            or not {item.listing_id for item in candidates} <= set(scope.listing_ids)
            or self.feature_state.current_sector_state(active) is None
            or self.feature_state.sector_revision_refresh_due(active, observed_at=observed_at)
            or self._sector_reduced_manifest(working, tuple(sorted(member_ids))).revision_sha256
            != active.revision_sha256
        ):
            return False, None
        evidence = self.feature_state.bindable_sector_evidence(active)
        observed = self.feature_foundation.observe_candidate_sectors(candidates)
        if evidence is None or observed is None:
            return False, None
        observations, failures, provider_wide = observed
        if (
            provider_wide
            or observations
            or set(failures) != {item.listing_id for item in candidates}
            or any(
                item["failure_code"] != "sector.missing_current_sector"
                for item in failures.values()
            )
        ):
            return False, None
        return True, self._apply_sector_partial_exclusion(
            request=request, observed_at=observed_at, carried=evidence.observations
        )

    def _sector_reduced_manifest(
        self, parent: UniverseManifest, admitted_ids: tuple[str, ...]
    ) -> UniverseManifest:
        """The membership a Sector exclusion leaves: the parent's admitted listings."""
        assert self.feature_input is not None
        # The parent's obligations are kept and the Sector requirement is
        # added by name; the identity is the set, so re-satisfying it over the
        # same listings is the same manifest.
        return build_quality_filtered_research_manifest(
            parent,
            eligible_listing_ids=admitted_ids,
            qualification_obligations=(
                qualification_obligation(
                    "feature_input_policy", self.feature_input.gateway.policy.policy_hash
                ),
                qualification_obligation("sector", "YAHOO_CURRENT_SECTOR:require_nonempty"),
            ),
        )

    def _apply_sector_partial_exclusion(
        self,
        *,
        request: WorkspaceMaintenanceRequest,
        observed_at: datetime,
        carried: tuple[Mapping[str, object], ...] | None = None,
    ) -> str | None:
        """Quarantine unadmitted Sector candidates and activate a safe subset.

        The provider has already exhausted its bounded per-listing retry.  The
        only automatic reduction is of candidates, never established members.
        No Sector value is invented and no existing manifest is mutated.
        Answers None once the reduced membership is active, otherwise the
        stable code the cycle stops by: the Sector shortfall itself when no
        safe reduction exists, the data cause when the reduction could not
        bind its evidence because members failed the day's refresh, and the
        evidence shortfall when it could not bind for another reason.

        ``carried`` is the active members' Sector evidence when only the candidates were
        rechecked: the reduction is then the active membership, which keeps its revision.
        """

        if self.feature_input is None:
            return "sector.partial_current_sector"
        observed_at = self._utc(self.clock() if self.clock is not None else observed_at)
        parent = self.manifest
        observations = (
            carried
            if carried is not None
            else self.feature_state.staged_sector_observations(parent.revision_sha256)
        )
        observation_by_id = {str(item["listing_id"]): item for item in observations}
        expected_ids = {item.listing_id for item in parent.listings}
        admitted_ids = tuple(sorted(expected_ids & set(observation_by_id)))
        excluded_ids = tuple(sorted(expected_ids - set(observation_by_id)))
        if not admitted_ids or not excluded_ids:
            return "sector.partial_current_sector"
        bootstrap = self.market_data.universe_bootstrap(request.market_profile_id)
        if bootstrap is not None:
            nominal = set(
                journal_members(
                    bootstrap, self.market_data.membership_events(request.market_profile_id)
                )
            )
            if nominal.intersection(excluded_ids):
                # Missing Sector evidence cannot become a nominal EXIT.
                return "sector.partial_current_sector"
        sector_distribution: dict[str, int] = {}
        for listing_id in admitted_ids:
            sector = str(observation_by_id[listing_id]["sector_name"])
            sector_distribution[sector] = sector_distribution.get(sector, 0) + 1
        policy = self.feature_input.gateway.policy
        # Before admission, candidate pass rate is not the Panel denominator.
        # Existing members cannot be removed above; the numerical Panel retains
        # its own coverage and Sector floors over the actual admitted Universe.
        if any(count < policy.minimum_sector_size for count in sector_distribution.values()):
            return "sector.partial_current_sector"

        from alphalattice.foundation.feature_engine.contracts import TemporalKnowledgeBoundary

        boundary = TemporalKnowledgeBoundary(
            market_as_of_session=request.target_market_session,
            knowledge_cutoff_at=request.knowledge_cutoff_at or observed_at,
            materialized_at=observed_at,
            universe_source_observed_at=datetime.combine(
                parent.profile.manifest_as_of,
                datetime.min.time(),
                tzinfo=UTC,
            ),
            sector_source_observed_at=observed_at,
            universe_point_in_time_qualified=False,
            sector_point_in_time_qualified=False,
        )
        quarantines = tuple(
            ListingQuarantine.create(
                listing_id=listing_id,
                reason_codes=("SECTOR_CLASSIFICATION_UNAVAILABLE",),
                evidence_hash=canonical_hash(
                    [parent.revision_sha256, listing_id, "sector.missing_current_sector"]
                ),
                execution_receipt_hash=canonical_hash(
                    [
                        "host-policy",
                        "recoverable-sector-exclusion",
                        parent.revision_sha256,
                        listing_id,
                    ]
                ),
                recheck_after_at=observed_at + timedelta(days=1),
            )
            for listing_id in excluded_ids
        )
        reduced = self._sector_reduced_manifest(parent, admitted_ids)
        admission = FeatureInputAdmission.create(
            candidate_manifest_revision=parent.revision_sha256,
            admitted_listing_ids=admitted_ids,
            quarantines=quarantines,
            quality_policy_hash=policy.policy_hash,
            temporal_boundary=boundary,
            evidence_scope="SECTOR_ONLY",
        )
        decisions = tuple(
            ListingEligibilityDecision.create(
                listing_id=listing.listing_id,
                assessment=(
                    FeatureInputAssessment.ADMITTED
                    if listing.listing_id in observation_by_id
                    else FeatureInputAssessment.QUARANTINED
                ),
                reason_codes=(
                    ()
                    if listing.listing_id in observation_by_id
                    else ("SECTOR_CLASSIFICATION_UNAVAILABLE",)
                ),
                evidence_hash=(
                    str(observation_by_id[listing.listing_id]["evidence_hash"])
                    if listing.listing_id in observation_by_id
                    else next(
                        item.evidence_hash
                        for item in quarantines
                        if item.listing_id == listing.listing_id
                    )
                ),
                valid_until=(
                    None
                    if listing.listing_id in observation_by_id
                    else observed_at + timedelta(days=1)
                ),
                requalification_conditions=(
                    ()
                    if listing.listing_id in observation_by_id
                    else ("provider_returns_verified_nonempty_sector",)
                ),
            )
            for listing in parent.listings
        )
        result = FeatureInputGatewayResult(
            FeatureInputAssessment.QUARANTINED,
            decisions,
            admission=admission,
            research_manifest=reduced,
            quarantines=quarantines,
        )
        self.mutation_gate.run(
            self.panel_state.record_feature_input_gateway_result,
            candidate_manifest=parent,
            result=result,
            temporal_boundary=boundary,
            observed_at=observed_at,
        )
        try:
            self.mutation_gate.run(
                self.market_data.bind_action_audit_receipts_to_manifest,
                parent,
                reduced,
                requested_as_of=request.target_market_session,
                now=observed_at,
            )
        except ValueError:
            # A member of the reduced child holds no reusable audit receipt
            # for this session: the exclusion cannot bind its evidence and
            # is not applied; a later run re-attempts it, as the governed
            # child binding does (an escaped refusal used to interrupt the
            # Task). The cycle stops by the cause the user can act on: the
            # day's refresh failed for members of the reduction (a stale
            # payload seals no audit), or the evidence is otherwise missing,
            # expired or changed since the parent's audit.
            return self._refused_binding_cause(request, admitted_ids)
        if carried is None:
            # Through the activation coordinator, as every activation: its map and receipt,
            # the reclassifications' effective session among them, reach the ledger (V346).
            # Carried evidence is the active membership's own revision, already activated.
            self.feature_foundation.sector_activation.activate(
                manifest=reduced,
                observations=tuple(observation_by_id[item] for item in admitted_ids),
                observed_at=observed_at,
            )
        self.mutation_gate.run(
            self.feature_state.clear_sector_staging,
            parent.revision_sha256,
        )
        self._bind_manifest(
            reduced,
            observed_at=observed_at,
            effective_session=request.target_market_session,
            authority="sector_partial_exclusion",
            reference_hash=admission.admission_hash,
        )
        return None

    def freshness_projection(
        self, request: WorkspaceMaintenanceRequest
    ) -> MaintenanceFreshnessProjection:
        """Project cycle state beside current panel freshness and workspace readiness.

        Args:
            request: Sealed request whose scope this operation executes or projects.

        Returns:
            The target/current session and recorded maintenance/readiness state.
        """
        cycle_id = canonical_hash(["workspace-maintenance", request.request_hash])
        cycle = self.registry.cycle(cycle_id)
        panel = self.panel_state.active_feature_panel(request.market_profile_id)
        return MaintenanceFreshnessProjection(
            target_market_session=request.target_market_session,
            current_market_session=(panel["as_of_session"] if panel is not None else None),
            phase=cycle.phase,
            status=cycle.status,
            readiness=(self.market_data.readiness.load(request.market_profile_id).status),
            retry_after_at=cycle.retry_after_at,
            failure_code=cycle.failure_code,
        )

    def _authorized_full_audit_listing_ids(
        self, request: WorkspaceMaintenanceRequest
    ) -> tuple[str, ...]:
        explicitly_requested = set(request.full_history_listing_ids)
        known_listing_ids = {item.listing_id for item in self.manifest.listings}
        if explicitly_requested - known_listing_ids:
            raise ValueError("full-history audit contains a listing outside the candidate manifest")
        return tuple(sorted(explicitly_requested))

    def _pending_authorized_full_audit_listing_ids(
        self,
        request: WorkspaceMaintenanceRequest,
        requested_listing_ids: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Resolve an explicit grant to only audits not already proven complete."""

        if not requested_listing_ids:
            return ()
        receipts = self.registry.latest_action_audits(
            requested_listing_ids,
            self.provider.name,
            full_only=True,
        )
        valid_provider_receipts = self.market_data.full_action_audit_receipt_hashes(
            tuple(receipt.provider_receipt_hash for receipt in receipts.values())
        )
        pending = []
        for listing_id in requested_listing_ids:
            receipt = receipts.get(listing_id)
            if (
                receipt is None
                or receipt.provider_receipt_hash not in valid_provider_receipts
                or receipt.covered_through_session < request.target_market_session
                or receipt.data_policy_hash != request.data_policy_hash
            ):
                pending.append(listing_id)
        return tuple(pending)

    def _required_full_audit_listing_ids(
        self,
        now: datetime,
        *,
        as_of_session: date,
        progress_operation_id: str,
    ) -> tuple[str, ...]:
        """Return evidence-triggered requirements without granting authority."""

        listing_ids = tuple(item.listing_id for item in self.manifest.listings)
        total_units = 4

        def publish_progress(
            completed_units: int,
            *,
            current_item: str,
            due_count: int = 0,
            seeded_count: int = 0,
        ) -> None:
            sink = getattr(self.feature_foundation, "progress_sink", None)
            if sink is None:
                return
            sink(
                WorkProgressUpdate(
                    operation_id=progress_operation_id,
                    stage_id="historical_audit_preflight",
                    status=("SUCCEEDED" if completed_units == total_units else "RUNNING"),
                    completed_units=completed_units,
                    total_units=total_units,
                    unit_name="evidence batches",
                    current_item=current_item,
                    counters={
                        "listings": len(listing_ids),
                        "requirements": due_count,
                        "seeded_legacy_receipts": seeded_count,
                    },
                )
            )

        due: list[str] = []
        publish_progress(0, current_item="chain receipts")
        chain_candidates = self.registry.action_audits(
            listing_ids,
            self.provider.name,
            full_only=True,
        )
        valid_provider_receipt_hashes = self.market_data.full_action_audit_receipt_hashes(
            tuple(item.provider_receipt_hash for item in chain_candidates)
        )
        receipts: dict[str, ActionAuditChainReceipt] = {}
        for receipt in chain_candidates:
            if receipt.provider_receipt_hash in valid_provider_receipt_hashes:
                receipts.setdefault(receipt.listing_id, receipt)
        publish_progress(1, current_item="legacy provider receipts")
        missing_ids = tuple(listing_id for listing_id in listing_ids if listing_id not in receipts)
        provider_receipts = self.market_data.latest_action_audit_receipts(
            missing_ids,
            provider=self.provider.name,
        )
        seeded_count = 0
        for listing_id in missing_ids:
            provider_receipt = provider_receipts.get(listing_id)
            if provider_receipt is None:
                continue
            receipt = ActionAuditChainReceipt.create(
                listing_id=listing_id,
                provider=self.provider.name,
                audit_scope=ActionAuditScope.FULL,
                previous_receipt_hash=None,
                full_anchor_receipt_hash=None,
                history_start=provider_receipt.history_start,
                history_end=provider_receipt.history_end,
                covered_through_session=provider_receipt.requested_as_of,
                window_action_hash=provider_receipt.action_set_hash,
                action_set_hash=provider_receipt.action_set_hash,
                raw_evidence_hash=provider_receipt.raw_evidence_hash,
                mapping_revision=provider_receipt.mapping_revision,
                data_policy_hash="onboarding-full-audit-anchor",
                provider_receipt_hash=provider_receipt.receipt_hash,
                observed_at=(
                    provider_receipt.observed_at.replace(tzinfo=UTC)
                    if provider_receipt.observed_at.tzinfo is None
                    else provider_receipt.observed_at
                ),
            )
            if self.registry.record_action_audit(receipt):
                seeded_count += 1
            receipts[listing_id] = receipt
        publish_progress(
            2,
            current_item="mapping revisions",
            seeded_count=seeded_count,
        )
        receipt_ids = tuple(sorted(receipts))
        mapping_revisions = self.market_data.provider_mapping_revisions(
            receipt_ids,
            provider=self.provider.name,
            as_of_session=as_of_session,
        )
        publish_progress(
            3,
            current_item="historical revisions",
            seeded_count=seeded_count,
        )
        historical_revisions = self.market_data.historical_revisions_since(
            {listing_id: receipt.observed_at for listing_id, receipt in receipts.items()},
            provider=self.provider.name,
            before_session=as_of_session - timedelta(days=45),
        )
        for listing in self.manifest.listings:
            held = receipts.get(listing.listing_id)
            if held is None:
                due.append(listing.listing_id)
                continue
            if mapping_revisions[listing.listing_id] != held.mapping_revision:
                due.append(listing.listing_id)
                continue
            if listing.listing_id in historical_revisions:
                due.append(listing.listing_id)
        publish_progress(
            4,
            current_item="complete",
            due_count=len(due),
            seeded_count=seeded_count,
        )
        return tuple(sorted(set(due)))

    def _session_authority(
        self, *, start: date, end: date, as_of_timestamp: datetime
    ) -> TradingSessionAuthority:
        """The qualified session axis for one bounded range, resolved once per range.

        The axis over ten years costs about 0.4 s of calendar materialization,
        which every one-listing run of a cycle used to repeat. The schedule is
        a pure function of the range and of whether ``end`` has closed by
        ``as_of_timestamp``; a cached axis is reused only when it already
        reaches ``end`` and the new clock is not earlier than the one that
        resolved it, so a not-yet-closed target session is never served stale.
        """

        cached = self._session_authority_cache.get((start, end))
        if (
            cached is not None
            and as_of_timestamp >= cached[0]
            and cached[1].sessions
            and cached[1].sessions[-1] == end
        ):
            return cached[1]
        authority = resolve_trading_session_authority(
            start=start, end=end, as_of_timestamp=as_of_timestamp
        )
        self._session_authority_cache[(start, end)] = (as_of_timestamp, authority)
        return authority

    def _maintenance_for(
        self,
        request: WorkspaceMaintenanceRequest,
        now: datetime,
        *,
        history_start: date,
        authorized_full_listing_ids: tuple[str, ...],
        required_full_listing_ids: tuple[str, ...],
        requested_full_listing_ids: tuple[str, ...],
        transport_workers: int,
        retry_grants: Mapping[str, tuple[str, datetime]] | None = None,
    ) -> CurrentUniverseMaintenance:
        """The maintenance runner for this cycle, kept across its bounded runs.

        A runner holds no per-run state; everything it advances is durable
        listing state it re-reads on every run. Keeping the instance lets its
        idempotent admission run once per process instead of once per listing.
        Any change in what identifies the run -- membership, target session,
        authorized or required listings, the transport budget -- builds a new
        runner, exactly as before.
        """

        key: tuple[object, ...] = (
            self.manifest.revision_sha256,
            request.target_market_session,
            tuple(authorized_full_listing_ids),
            tuple(required_full_listing_ids),
            tuple(requested_full_listing_ids),
            transport_workers,
            history_start,
            tuple(sorted((retry_grants or {}).items())),
        )
        if self._maintenance_runner is not None and self._maintenance_runner[0] == key:
            return self._maintenance_runner[1]
        runner = CurrentUniverseMaintenance(
            store=self.market_data,
            manifest=self.manifest,
            provider=self.provider,
            as_of_session=request.target_market_session,
            # Qualification never invents its own calendar: the Host
            # resolves the axis over the range this workspace actually
            # holds and injects it.
            trading_session_authority=self._session_authority(
                start=history_start,
                end=request.target_market_session,
                as_of_timestamp=now,
            ),
            full_audit_listing_ids=frozenset(authorized_full_listing_ids),
            full_history_escalation_listing_ids=frozenset(authorized_full_listing_ids),
            full_history_required_listing_ids=frozenset(required_full_listing_ids),
            authorization_identity_listing_ids=frozenset(requested_full_listing_ids),
            retry_grants=dict(retry_grants or {}),
            max_workers=transport_workers,
            mutation_gate=self.mutation_gate,
            progress_sink=self.feature_foundation.progress_sink,
        )
        self._maintenance_runner = (key, runner)
        return runner

    def _record_change_set(
        self,
        documents: tuple[dict[str, Any], ...],
        *,
        request: WorkspaceMaintenanceRequest,
        observed_at: datetime,
    ) -> MarketDataChangeSet:
        changes: list[ListingMarketDataChange] = []
        receipts: list[str] = []
        for item in documents:
            listing_id = str(item["listing_id"])
            previous = self.registry.latest_action_audit(listing_id, self.provider.name)
            scope = ActionAuditScope(str(item["audit_scope"]).casefold())
            history_start = date.fromisoformat(str(item["history_start"]))
            history_end = date.fromisoformat(str(item["history_end"]))
            if previous is not None and all(
                (
                    previous.audit_scope is scope,
                    previous.history_start == history_start,
                    previous.history_end == history_end,
                    previous.covered_through_session == request.target_market_session,
                    previous.window_action_hash == str(item["window_action_hash"]),
                    previous.action_set_hash == str(item["action_set_hash"]),
                    previous.raw_evidence_hash == str(item["raw_evidence_hash"]),
                    previous.mapping_revision == str(item["mapping_revision"]),
                    previous.data_policy_hash == request.data_policy_hash,
                    previous.provider_receipt_hash == str(item["provider_receipt_hash"]),
                )
            ):
                chain = previous
            else:
                chain = ActionAuditChainReceipt.create(
                    listing_id=listing_id,
                    provider=self.provider.name,
                    audit_scope=scope,
                    previous_receipt_hash=previous.receipt_hash if previous else None,
                    full_anchor_receipt_hash=(
                        previous.full_anchor_receipt_hash if previous is not None else None
                    ),
                    history_start=history_start,
                    history_end=history_end,
                    covered_through_session=request.target_market_session,
                    window_action_hash=str(item["window_action_hash"]),
                    action_set_hash=str(item["action_set_hash"]),
                    raw_evidence_hash=str(item["raw_evidence_hash"]),
                    mapping_revision=str(item["mapping_revision"]),
                    data_policy_hash=request.data_policy_hash,
                    provider_receipt_hash=str(item["provider_receipt_hash"]),
                    observed_at=observed_at,
                )
                self.registry.record_action_audit(chain)
            receipts.append(chain.receipt_hash)
            changes.append(
                ListingMarketDataChange(
                    listing_id=listing_id,
                    new_session_start=self._optional_date(item.get("new_session_start")),
                    new_sessions=tuple(
                        date.fromisoformat(str(value)) for value in item.get("new_sessions", ())
                    ),
                    raw_correction_start=self._optional_date(item.get("raw_correction_start")),
                    action_correction_start=self._optional_date(
                        item.get("action_correction_start")
                    ),
                    raw_correction_sessions=tuple(
                        date.fromisoformat(str(value))
                        for value in item.get("raw_correction_sessions", ())
                    ),
                    adjusted_return_change_sessions=tuple(
                        date.fromisoformat(str(value))
                        for value in item.get("adjusted_return_change_sessions", ())
                    ),
                    source_receipt_hashes=(chain.receipt_hash,),
                )
            )
        return MarketDataChangeSet(
            listing_changes=tuple(
                item
                for item in changes
                if item.new_session_start
                or item.raw_correction_start
                or item.action_correction_start
                or item.adjusted_return_change_sessions
            ),
            receipt_hashes=tuple(receipts),
        )

    @staticmethod
    def _merge_change_sets(
        prior: MarketDataChangeSet,
        current: MarketDataChangeSet,
    ) -> MarketDataChangeSet:
        """Carry a verified market-data prefix across deferred stage resumes."""

        if prior.empty:
            return current
        if current.empty or prior.change_set_hash == current.change_set_hash:
            return prior

        def earliest(left: date | None, right: date | None) -> date | None:
            values = tuple(value for value in (left, right) if value is not None)
            return min(values) if values else None

        by_listing: dict[str, ListingMarketDataChange] = {
            item.listing_id: item for item in prior.listing_changes
        }
        for item in current.listing_changes:
            previous = by_listing.get(item.listing_id)
            if previous is None:
                by_listing[item.listing_id] = item
                continue
            by_listing[item.listing_id] = ListingMarketDataChange(
                listing_id=item.listing_id,
                new_session_start=earliest(previous.new_session_start, item.new_session_start),
                new_sessions=tuple(sorted(set(previous.new_sessions) | set(item.new_sessions))),
                raw_correction_start=earliest(
                    previous.raw_correction_start,
                    item.raw_correction_start,
                ),
                action_correction_start=earliest(
                    previous.action_correction_start,
                    item.action_correction_start,
                ),
                raw_correction_sessions=tuple(
                    sorted(
                        set(previous.raw_correction_sessions) | set(item.raw_correction_sessions)
                    )
                ),
                adjusted_return_change_sessions=tuple(
                    sorted(
                        set(previous.adjusted_return_change_sessions)
                        | set(item.adjusted_return_change_sessions)
                    )
                ),
                source_receipt_hashes=tuple(
                    sorted(set(previous.source_receipt_hashes) | set(item.source_receipt_hashes))
                ),
            )

        additions = set(prior.membership_additions)
        removals = set(prior.membership_removals)
        for listing_id in current.membership_additions:
            removals.discard(listing_id)
            additions.add(listing_id)
        for listing_id in current.membership_removals:
            additions.discard(listing_id)
            removals.add(listing_id)
        return MarketDataChangeSet(
            listing_changes=tuple(by_listing.values()),
            spy_correction_start=earliest(
                prior.spy_correction_start,
                current.spy_correction_start,
            ),
            spy_return_change_sessions=tuple(
                sorted(
                    set(prior.spy_return_change_sessions) | set(current.spy_return_change_sessions)
                )
            ),
            sector_revision_changed=(
                prior.sector_revision_changed or current.sector_revision_changed
            ),
            membership_additions=tuple(sorted(additions)),
            membership_removals=tuple(sorted(removals)),
            receipt_hashes=tuple(sorted(set(prior.receipt_hashes) | set(current.receipt_hashes))),
        )

    def _unconsumed_feature_source_change_set(
        self,
        active_panel: dict[str, object],
        *,
        through: date,
    ) -> MarketDataChangeSet:
        """Reconstruct every source delta after the last verified panel prefix.

        Market hydration and quality governance may complete under one durable
        maintenance cycle while Feature Foundation resumes under another
        request or a derived manifest.  A cycle-local change set is therefore
        evidence, not the consumption cursor.  The active panel activation is
        the verified prefix; immutable raw/adjusted ledgers plus concrete
        feature gaps determine the outstanding work.
        """

        activated_at = active_panel.get("activated_at")
        if not isinstance(activated_at, datetime):
            raise ValueError("active panel has no feature-delta consumption prefix")
        prior_manifest = self.market_data.load_universe_manifest_revision(
            str(active_panel["manifest_revision"])
        )
        prior_listing_ids = {item.listing_id for item in prior_manifest.listings}
        current_listing_ids = {item.listing_id for item in self.manifest.listings}
        # Read the source union, not just the newly activated request roster.
        # Most transitions need one existing scope; mixed ENTRY/EXIT needs two.
        # A removed name can still owe the pre-exit tail or an approved correction.
        scopes = (
            (self.manifest,)
            if prior_listing_ids <= current_listing_ids
            else (prior_manifest,)
            if current_listing_ids <= prior_listing_ids
            else (prior_manifest, self.manifest)
        )
        appends = tuple(
            delta
            for scope in scopes
            for delta in self.market_data.feature_source_appends(
                scope,
                catalog_hash=self.feature_foundation.catalog.binding.catalog_hash,
                through=through,
                include_incomplete=allows_missing_source_rows(
                    self.feature_foundation.panel.policy_hash
                ),
            )
        )
        raw_deltas = tuple(
            delta
            for scope in scopes
            for delta in self.market_data.raw_bar_semantic_deltas_since(
                scope, observed_after=activated_at, through=through
            )
        )
        adjusted_deltas = tuple(
            delta
            for scope in scopes
            for delta in self.market_data.provider_adjusted_semantic_deltas_since(
                scope, observed_after=activated_at, through=through
            )
        )
        verified_session_sets = frozenset(
            listing_id
            for scope in scopes
            for listing_id in self.market_data.verified_feature_source_session_sets(
                scope,
                catalog_hash=self.feature_foundation.catalog.binding.catalog_hash,
                through=through,
            )
        )

        new_by_listing: dict[str, set[date]] = {}
        raw_by_listing: dict[str, set[date]] = {}
        adjusted_by_listing: dict[str, set[date]] = {}
        receipts_by_listing: dict[str, set[str]] = {}

        def ensure(listing_id: str) -> None:
            new_by_listing.setdefault(listing_id, set())
            raw_by_listing.setdefault(listing_id, set())
            adjusted_by_listing.setdefault(listing_id, set())
            receipts_by_listing.setdefault(listing_id, set())

        for delta in appends:
            ensure(delta.listing_id)
            new_by_listing[delta.listing_id].update(delta.sessions)
            receipts_by_listing[delta.listing_id].add(delta.evidence_hash)
        append_sessions = {
            listing_id: frozenset(values) for listing_id, values in new_by_listing.items()
        }
        for delta in raw_deltas:
            ensure(delta.listing_id)
            raw_by_listing[delta.listing_id].update(delta.sessions)
            receipts_by_listing[delta.listing_id].add(delta.evidence_hash)
        for revision in adjusted_deltas:
            changed_sessions = frozenset(revision.changed_return_sessions)
            if revision.uniform_rescale:
                if changed_sessions:
                    raise ValueError("uniform provider rescale contains changed-return sessions")
                continue
            if revision.session_set_changed:
                # Base preparation may already have consumed a tail while the
                # Panel waits for membership admission. No remaining base append
                # is then expected; the immutable Panel prefix is still owed it.
                published_through = date.fromisoformat(str(active_panel["as_of_session"]))
                prepared_unconsumed = (
                    bool(changed_sessions) and min(changed_sessions) > published_through
                )
                if (
                    revision.listing_id in verified_session_sets
                    and changed_sessions
                    and not append_sessions.get(revision.listing_id)
                    and not prepared_unconsumed
                ):
                    # Re-entry can prepare its absent interval before the global
                    # Panel cutoff. Those source rows were never Panel members;
                    # this is lookback coverage, not a rewrite of consumed history.
                    past = tuple(
                        sorted(day for day in changed_sessions if day <= published_through)
                    )
                    membership = self.market_data.membership_schedule(
                        self.manifest.profile.market_profile_id,
                        sessions=past,
                        fallback_listing_ids=tuple(sorted(prior_listing_ids)),
                    )
                    prepared_unconsumed = all(
                        revision.listing_id not in membership.members(day) for day in past
                    )
                if revision.listing_id not in verified_session_sets or (
                    not append_sessions.get(revision.listing_id) and not prepared_unconsumed
                ):
                    raise ValueError("provider session-set change is not a verified append")
            if not changed_sessions:
                continue
            ensure(revision.listing_id)
            adjusted_by_listing[revision.listing_id].update(changed_sessions)
            receipts_by_listing[revision.listing_id].add(revision.receipt_hash)

        listing_changes = []
        receipt_hashes: set[str] = set()
        for listing_id in sorted(new_by_listing):
            new_sessions = tuple(sorted(new_by_listing[listing_id]))
            raw_sessions = tuple(sorted(raw_by_listing[listing_id]))
            adjusted_sessions = tuple(sorted(adjusted_by_listing[listing_id]))
            receipts = tuple(sorted(receipts_by_listing[listing_id]))
            receipt_hashes.update(receipts)
            if not (new_sessions or raw_sessions or adjusted_sessions):
                continue
            listing_changes.append(
                ListingMarketDataChange(
                    listing_id=listing_id,
                    new_session_start=min(new_sessions) if new_sessions else None,
                    new_sessions=new_sessions,
                    raw_correction_start=min(raw_sessions) if raw_sessions else None,
                    raw_correction_sessions=raw_sessions,
                    adjusted_return_change_sessions=adjusted_sessions,
                    source_receipt_hashes=receipts,
                )
            )

        sector = self.feature_state.current_sector_state(self.manifest)
        return MarketDataChangeSet(
            listing_changes=tuple(listing_changes),
            sector_revision_changed=(
                sector is not None
                and str(active_panel["sector_revision"]) != sector.sector_revision
            ),
            membership_additions=tuple(sorted(current_listing_ids - prior_listing_ids)),
            membership_removals=tuple(sorted(prior_listing_ids - current_listing_ids)),
            receipt_hashes=tuple(sorted(receipt_hashes)),
        )

    def _govern_failed_members_before_feature_build(
        self,
        maintenance_id: str,
        *,
        request: WorkspaceMaintenanceRequest,
        observed_at: datetime,
        macro_retry_exhausted: bool,
        observed_workers: int,
    ) -> _GovernanceEvaluation | None:
        """Govern the day's failed members of a manifest that has no Sector evidence yet.

        A parent bound for a candidate recheck gets its entrants' Sector from
        the Feature build, so ordinary governance (which needs complete
        Sector evidence and a Panel-impact projection) cannot run before it,
        and a member whose refresh failed used to stop the day only after
        that build, by the derived binding. Here the existing governance runs
        first over the same evidence -- the prior manifest's Sector evidence
        covers the members it knows, the entrants are declared unknown, and
        the Gateway invents nothing for them: failures only, no exclusion
        option, no admission. Nothing to govern (no member failed, or no
        prior Sector evidence to judge with) answers None and the cycle
        proceeds as before.
        """
        if self.feature_input is None:
            return None
        if not self._failed_members(maintenance_id) and not (
            self.panel_state.has_unresolved_feature_input(self.manifest.revision_sha256)
        ):
            return None
        if request.membership_revision == self.manifest.revision_sha256:
            return None
        try:
            prior = self.market_data.load_universe_manifest_revision(request.membership_revision)
        except ValueError:
            return None
        known = self.feature_state.current_sector_state(prior)
        if (
            prior.profile != self.manifest.profile
            or known is None
            or {item.listing_id for item in self.manifest.listings}
            <= set(known.sector_by_listing_id)
        ):
            # No prior Sector evidence to judge with, or evidence that covers
            # every member (then the manifest's own state would have served).
            return None
        return self._govern_quality(
            maintenance_id,
            request=request,
            observed_at=observed_at,
            macro_retry_exhausted=macro_retry_exhausted,
            observed_workers=observed_workers,
            sector_source=prior,
        )

    def _failed_members(self, maintenance_id: str) -> frozenset[str]:
        """The members whose refresh the maintenance operation records as failed."""
        members = {item.listing_id for item in self.manifest.listings}
        return frozenset(
            item.listing_id
            for item in self.market_data.current_universe_maintenance_listings(maintenance_id)
            if item.failure_code is not None and item.listing_id in members
        )

    def _elapsed_wait_retry_grants(self, *, now: datetime) -> dict[str, tuple[str, datetime]]:
        """The elapsed waits of the manifest's open cases, by listing.

        ``wait_for_provider_recovery`` is one bounded retry after its time:
        the Human's decision grants the listing one attempt past the
        operation's attempt budget. The grant is named here by the
        decision's execution receipt and the wait instant; whether it is
        new, held or already spent is the store's durable answer on the
        listing's row (``MaintenanceRetryGrant``), never this cycle's view.
        """
        grants: dict[str, tuple[str, datetime]] = {}
        for document in self.panel_state.feature_input_case_documents(
            self.manifest.revision_sha256
        ):
            case = FeatureInputAgentCase.read_document(document)
            resolution = self.panel_state.feature_input_resolution(case.case_token)
            effect = (resolution or {}).get("effect")
            if not isinstance(effect, dict) or effect.get("retry_after_at") is None:
                continue
            execution = TypeAdapter(FeatureInputPolicyExecution).validate_python(effect)
            if execution.retry_after_at is None or execution.retry_after_at > now:
                continue
            for listing_id in case.listing_ids:
                held = grants.get(listing_id)
                if held is None or held[1] < execution.retry_after_at:
                    grants[listing_id] = (
                        execution.execution_receipt_hash,
                        execution.retry_after_at,
                    )
        return grants

    def _govern_quality(
        self,
        maintenance_id: str | None,
        *,
        request: WorkspaceMaintenanceRequest,
        observed_at: datetime,
        macro_retry_exhausted: bool,
        observed_workers: int,
        sector_source: UniverseManifest | None = None,
    ) -> _GovernanceEvaluation:
        # The evidence is read and judged under one retained instance and without the
        # workspace's write gate: only the recording writes (RH, V109). The step held the gate
        # over the whole list, about 38 s for 473 names on the first use and still 1.5 s for 60
        # fixture names, while every other write (a Task's progress among them) waited. The
        # maintenance cycle is the only writer of what it reads, and runs this step itself.
        with self.market_data.database.retain(read_only=False):
            prepared = self._quality_evidence(
                maintenance_id,
                request=request,
                observed_at=observed_at,
                sector_source=sector_source,
            )
        with self.mutation_gate.hold(), self.market_data.database.retain(read_only=False):
            return self._record_quality(
                prepared,
                macro_retry_exhausted=macro_retry_exhausted,
                observed_workers=observed_workers,
            )

    def _quality_evidence(
        self,
        maintenance_id: str | None,
        *,
        request: WorkspaceMaintenanceRequest,
        observed_at: datetime,
        sector_source: UniverseManifest | None = None,
    ) -> _QualityEvidence:
        if self.feature_input is None:
            raise ValueError("workspace maintenance requires Feature Input governance")
        observed_at = self._utc(self.clock() if self.clock is not None else observed_at)
        quality_manifest = self.manifest
        sector = self.feature_state.current_sector_state(sector_source or quality_manifest)
        if sector is None:
            raise ValueError("workspace maintenance quality governance requires sector evidence")
        # Partial Sector evidence (another manifest's, over the members it
        # knows) governs failures only: the listings it does not cover are
        # declared unknown to the Gateway, and no Panel impact is projected.
        sector_unknown = frozenset(
            listing.listing_id
            for listing in quality_manifest.listings
            if listing.listing_id not in sector.sector_by_listing_id
        )
        if bool(sector_unknown) != (sector_source is not None):
            raise ValueError("workspace maintenance quality governance requires sector evidence")
        evidence: list[ListingQualityEvidence] = []
        # A maintenance operation contributes per-listing incident observations.
        # Admission-only governance runs without one, so a listing simply has no
        # incident to report and its evidence comes from the stored admission and
        # bars instead.
        by_listing = (
            {}
            if maintenance_id is None
            else {
                item.listing_id: item
                for item in self.market_data.current_universe_maintenance_listings(maintenance_id)
            }
        )
        # A listing's evidence declares the whole range it holds
        # (``range_start`` is its first raw session), and two audits qualify
        # that range: the full-history anchor of its audit chain (the newest
        # full-history provider receipt, ``latest_action_audit_receipts``) and
        # the link sealed for this session in the scope that maintenance ran
        # (a rolling window on an ordinary day; the change document reports
        # it, and without a change document the store holds it). The
        # adjusted-close diagnostic is judged over both -- the anchor keeps a
        # historical mismatch visible on a day whose window is clean, the
        # link keeps a fresh one visible -- and both passes of governance
        # (the maintenance cycle that raises a case, the continuation that
        # applies the decision) read the same two receipts, so a listing's
        # evidence identity does not move between them.
        listing_ids = tuple(listing.listing_id for listing in quality_manifest.listings)
        # One set-scoped read for the whole manifest, then one evaluation per
        # listing over its slice: the same rows, order and evidence identity
        # as one admission, one bar history and one action set read per
        # listing, without a connection request per read or a full-width bar
        # object per session.
        inputs = self.market_data.quality_governance_inputs(
            listing_ids, through=request.target_market_session
        )
        raw_closes = _raw_close_points(inputs.raw_close_series)
        anchors = self.market_data.latest_action_audit_receipts(
            listing_ids, provider=self.provider.name
        )
        session_links: dict[str, ActionAuditReceipt] = (
            {}
            if maintenance_id is not None
            else self.market_data.latest_action_audit_receipts(
                listing_ids,
                provider=self.provider.name,
                requested_as_of=request.target_market_session,
            )
        )
        for listing in quality_manifest.listings:
            item = by_listing.get(listing.listing_id)
            observed_failure_code = item.failure_code if item is not None else None
            admission = inputs.admissions.get(listing.listing_id)
            closes = raw_closes.get(listing.listing_id)
            if admission is None or closes is None or not len(closes):
                evidence.append(
                    ListingQualityEvidence(
                        listing_id=listing.listing_id,
                        provider=self.provider.name,
                        range_start=request.target_market_session,
                        range_end=request.target_market_session,
                        evidence_hash=canonical_hash(
                            [listing.listing_id, observed_failure_code, request.request_hash]
                        ),
                        failure_code=observed_failure_code or "data.quality_evidence_missing",
                        reason_codes=("QUALITY_EVIDENCE_MISSING",),
                        retry_exhausted=True,
                    )
                )
                continue
            change = (item.change_document if item is not None else None) or {}
            failure_code = observed_failure_code
            if failure_code is not None:
                evidence.append(
                    ListingQualityEvidence(
                        listing_id=listing.listing_id,
                        provider=self.provider.name,
                        range_start=closes.session(0),
                        range_end=request.target_market_session,
                        evidence_hash=canonical_hash(
                            [listing.listing_id, failure_code, request.request_hash]
                        ),
                        failure_code=failure_code,
                        reason_codes=(failure_code.upper().replace(".", "_"),),
                        retry_exhausted=True,
                    )
                )
                continue
            assert self.quality_evaluator is not None
            anchor = anchors.get(listing.listing_id)
            link = session_links.get(listing.listing_id)
            diagnostic_bps = self._admission_diagnostic_bps(
                change.get("adjusted_diagnostic_max_bps")
                if change.get("provider_receipt_hash")
                else (link.max_adjusted_close_difference_bps if link is not None else None),
                anchor.max_adjusted_close_difference_bps if anchor is not None else None,
            )
            listing_actions = inputs.actions.get(listing.listing_id)
            if listing_actions is None:
                raise ValueError("listing has no provider mapping")
            actions = tuple(
                action
                for action in listing_actions
                if action.effective_date <= request.target_market_session
            )
            evidence.append(
                self.quality_evaluator.evaluate(
                    admission=admission,
                    provider=self.provider.name,
                    range_start=closes.session(0),
                    range_end=request.target_market_session,
                    raw_closes=closes,
                    corporate_action_sessions=frozenset(
                        action.effective_date for action in actions
                    ),
                    action_evidence_hash=canonical_hash([asdict(action) for action in actions]),
                    action_audit_completed=(
                        bool(change.get("provider_receipt_hash"))
                        or link is not None
                        or anchor is not None
                    ),
                    adjusted_diagnostic_max_bps=diagnostic_bps,
                    identity_verified=True,
                    retry_exhausted=True,
                )
            )
        from alphalattice.foundation.feature_engine.contracts import TemporalKnowledgeBoundary

        boundary = TemporalKnowledgeBoundary(
            market_as_of_session=request.target_market_session,
            knowledge_cutoff_at=request.knowledge_cutoff_at or observed_at,
            materialized_at=observed_at,
            universe_source_observed_at=datetime.combine(
                quality_manifest.profile.manifest_as_of,
                datetime.min.time(),
                tzinfo=UTC,
            ),
            sector_source_observed_at=sector.sector_observed_at,
            universe_point_in_time_qualified=False,
            sector_point_in_time_qualified=False,
        )
        panel_impact = (
            PanelImpactProjection(
                sector_distribution=sector.sector_distribution,
                panel_coverage=1.0,
                available_factor_count=len(self.feature_foundation.catalog.factor_ids),
                lineage_compatible=True,
            )
            if not sector_unknown
            else None
        )
        return _QualityEvidence(
            manifest=quality_manifest,
            evidence=tuple(evidence),
            sector_by_listing_id={
                listing.listing_id: sector.sector_by_listing_id[listing.listing_id]
                for listing in quality_manifest.listings
                if listing.listing_id not in sector_unknown
            },
            boundary=boundary,
            observed_at=observed_at,
            panel_impact=panel_impact,
            sector_unknown=sector_unknown,
            sector_source=sector_source,
        )

    def _record_quality(
        self, prepared: _QualityEvidence, *, macro_retry_exhausted: bool, observed_workers: int
    ) -> _GovernanceEvaluation:
        """The gateway's assessment of the judged evidence, recorded under the write gate."""
        assert self.feature_input is not None
        result = self.feature_input.assess_and_record(
            candidate_manifest=prepared.manifest,
            evidence=prepared.evidence,
            sector_by_listing_id=prepared.sector_by_listing_id,
            temporal_boundary=prepared.boundary,
            observed_at=prepared.observed_at,
            panel_impact=prepared.panel_impact,
            macro_retry_exhausted=macro_retry_exhausted,
            observed_workers=observed_workers,
            sector_unknown_listing_ids=prepared.sector_unknown,
            raw_retention_proofs=raw_retention_decision_proofs(self.panel_state, self.market_data),
        )
        return _GovernanceEvaluation(
            prepared.manifest,
            result,
            prepared.evidence,
            prepared.panel_impact,
            prepared.sector_source,
        )

    @staticmethod
    def _admission_diagnostic_bps(
        session_link_bps: object, full_anchor_bps: object
    ) -> float | None:
        """The adjusted-close diagnostic over the range the evidence declares.

        The session's link audits its own window; the chain's full-history
        anchor audits everything before it. The evidence claims the whole
        range, so the diagnostic is the larger of the two audits that cover
        it; one absent audit leaves the other's, both absent leaves nothing,
        which the evaluator reports as ``ADJUSTED_DIAGNOSTIC_UNAVAILABLE``.
        """

        values = [
            float(value)
            for value in (session_link_bps, full_anchor_bps)
            if isinstance(value, int | float)
        ]
        return max(values) if values else None

    def _handle_governance_result(
        self,
        governance: _GovernanceEvaluation,
        *,
        maintenance_id: str | None,
        request: WorkspaceMaintenanceRequest,
        observed_at: datetime,
        observed_workers: int,
    ) -> tuple[MaintenanceStatus, datetime | None, str | None] | None:
        result = governance.result
        if result.assessment in {
            FeatureInputAssessment.ADMITTED,
            FeatureInputAssessment.QUARANTINED,
        }:
            if result.research_manifest is not None:
                if not self._bind_derived_manifest_evidence(
                    governance.manifest,
                    result.research_manifest,
                    requested_as_of=request.target_market_session,
                    observed_at=observed_at,
                ):
                    return (
                        MaintenanceStatus.BLOCKED,
                        None,
                        "workspace_maintenance.derived_manifest_evidence_incomplete",
                    )
                self._bind_manifest(
                    result.research_manifest,
                    observed_at=observed_at,
                    effective_session=request.target_market_session,
                    authority="feature_input_gateway",
                    reference_hash=(
                        result.admission.admission_hash
                        if result.admission is not None
                        else result.research_manifest.revision_sha256
                    ),
                )
            return None
        if result.deferred is not None:
            return (
                MaintenanceStatus.DEFERRED,
                result.deferred.retry_after_at,
                result.deferred.failure_code,
            )
        if not result.agent_cases or self.feature_input is None:
            if result.failure_reasons:
                return (
                    MaintenanceStatus.BLOCKED,
                    None,
                    "feature_input." + result.failure_reasons[0].lower(),
                )
            return (
                MaintenanceStatus.DEFERRED,
                observed_at + timedelta(minutes=5),
                "workspace_maintenance.data_engineer_diagnosis_deferred",
            )
        executions = []
        awaiting_choice = False
        active_case = None
        try:
            executor = FeatureInputRemediationExecutor(self.feature_input.gateway)
            for original_case in result.agent_cases:
                case = original_case
                active_case = case
                case_evidence = governance.evidence
                case_panel_impact = governance.panel_impact
                actor_submission = None
                saved = self.panel_state.feature_input_resolution(case.case_token)
                if saved is not None and saved.get("effect", {}).get("failure_reasons"):
                    awaiting_choice = True
                    continue
                if saved is not None and saved.get("effect", {}).get("retry_after_at"):
                    previous = TypeAdapter(FeatureInputPolicyExecution).validate_python(
                        saved["effect"]
                    )
                    if (
                        previous.retry_after_at is not None
                        and previous.retry_after_at <= observed_at
                    ):
                        # The wait elapsed, its one retry was spent, and the
                        # listing still fails: a wait is a bounded retry, not
                        # a standing decision, so this is a choice again.
                        awaiting_choice = True
                        continue
                    executions.append(previous)
                    continue
                chosen = (
                    DataRemediationExecutionReceipt.model_validate(saved["receipt"])
                    if saved is not None
                    else None
                )
                if chosen is None and (self.diagnose is None or case_panel_impact is None):
                    # No diagnoser, or no Panel impact to hand one (partial
                    # Sector evidence): the choice is the Human's.
                    awaiting_choice = True
                    continue
                for _diagnosis_attempt in range(2):
                    if chosen is not None:
                        if (
                            chosen.execution_policy_hash
                            != self.feature_input.gateway.policy.policy_hash
                        ):
                            raise ValueError("feature_input.resolution_policy_changed")
                        actor_submission = DataRemediationActorSubmission(
                            submission=chosen.submission,
                            actor_kind=chosen.actor_submission.actor_kind,
                            actor_id=chosen.actor_submission.actor_id,
                            agent_execution=chosen.actor_submission.agent_execution,
                        )
                    else:
                        assert self.diagnose is not None and case_panel_impact is not None
                        actor_submission = self.diagnose(
                            case,
                            case_evidence,
                            case_panel_impact,
                            self.agent_budget,
                        )
                    fresh = self._govern_quality(
                        maintenance_id,
                        request=request,
                        observed_at=observed_at,
                        macro_retry_exhausted=True,
                        observed_workers=observed_workers,
                        sector_source=governance.sector_source,
                    )
                    if fresh.result.assessment in {
                        FeatureInputAssessment.ADMITTED,
                        FeatureInputAssessment.QUARANTINED,
                    }:
                        return self._handle_governance_result(
                            fresh,
                            maintenance_id=maintenance_id,
                            request=request,
                            observed_at=observed_at,
                            observed_workers=observed_workers,
                        )
                    current_scope = tuple(
                        item for item in fresh.evidence if item.listing_id in case.listing_ids
                    )
                    if len(current_scope) != len(case.listing_ids) or any(
                        item.admitted or item.failure_code != case.failure_code
                        for item in current_scope
                    ):
                        return (
                            MaintenanceStatus.DEFERRED,
                            observed_at + timedelta(minutes=5),
                            "data.evidence_changed_during_diagnosis",
                        )
                    reconciled = self.feature_input.gateway.reconcile_stale_case(
                        candidate_manifest=governance.manifest,
                        prior_case=case,
                        current_evidence=current_scope,
                        observed_at=observed_at,
                        observed_workers=observed_workers,
                    )
                    if isinstance(reconciled, ProviderCohortDeferred):
                        return (
                            MaintenanceStatus.DEFERRED,
                            reconciled.retry_after_at,
                            reconciled.failure_code,
                        )
                    if reconciled.case_token == case.case_token:
                        case_evidence = fresh.evidence
                        case_panel_impact = fresh.panel_impact
                        break
                    if chosen is not None:
                        raise ValueError("feature_input.resolution_evidence_changed")
                    case = reconciled
                    active_case = case
                    case_evidence = fresh.evidence
                    case_panel_impact = fresh.panel_impact
                assert actor_submission is not None
                proposal = actor_submission.submission.proposal
                option = self.feature_input.gateway.validate_agent_selection(
                    case,
                    run_id=proposal.run_id,
                    case_token=proposal.case_token,
                    evidence_hash=proposal.evidence_hash,
                    option_id=proposal.option_id,
                    current_evidence_hash=case.evidence_hash,
                    rediagnosis_count=case.rediagnosis_count,
                )
                policy_decision = PolicyDecision(
                    case_token=case.case_token,
                    evidence_hash=case.evidence_hash,
                    option_id=option.option_id,
                    option_hash=option.option_hash,
                    policy_hash=remediation_hash(option.policy_args),
                    disposition=option.disposition,
                )
                neutral_receipt = seal_validated_data_remediation_execution(
                    submission=actor_submission.submission,
                    policy_decision=policy_decision,
                    actor_kind=actor_submission.actor_kind,
                    actor_id=actor_submission.actor_id,
                    agent_execution=actor_submission.agent_execution,
                    execution_policy_hash=self.feature_input.gateway.policy.policy_hash,
                )
                if chosen is not None and neutral_receipt != chosen:
                    raise ValueError("feature_input.resolution_catalog_changed")
                if option.policy_args.action in {
                    RemediationAction.QUARANTINE_LISTING,
                    RemediationAction.EXCLUDE_FROM_NEXT_MANIFEST,
                }:
                    sector = self.feature_state.current_sector_state(governance.manifest)
                    if sector is None or case_panel_impact is None:
                        raise ValueError("feature_input.sector_evidence_missing")
                    excluded = set(option.target_listing_ids) | {
                        item.listing_id
                        for item in self.panel_state.active_listing_quarantines(
                            governance.manifest.revision_sha256
                        )
                    }
                    distribution: dict[str, int] = dict.fromkeys(sector.sector_distribution, 0)
                    for listing_id, name in sector.sector_by_listing_id.items():
                        if listing_id not in excluded:
                            distribution[name] += 1
                    case_panel_impact = PanelImpactProjection(
                        sector_distribution=distribution,
                        panel_coverage=min(
                            case_panel_impact.panel_coverage,
                            sum(distribution.values()) / len(governance.manifest.listings),
                        ),
                        available_factor_count=case_panel_impact.available_factor_count,
                        lineage_compatible=case_panel_impact.lineage_compatible,
                    )
                execution = executor.execute(
                    case=case,
                    option=option,
                    evidence_by_listing_id={item.listing_id: item for item in case_evidence},
                    observed_at=observed_at,
                    panel_impact_after_exclusion=case_panel_impact,
                    proposal_hash=neutral_receipt.receipt_hash,
                    human_confirmed=(
                        actor_submission.actor_kind is ActorKind.HUMAN
                        or (
                            chosen is not None
                            and actor_submission.actor_kind is ActorKind.EXTERNAL_AUTOMATION
                            and DataIssueDelegation(
                                self.market_data.path.parent
                            ).recorded_human_authorization(actor_submission.actor_id, case, option)
                        )
                    ),
                )
                if execution.status is not FeatureInputExecutionStatus.DATA_TRUTH_REVIEW:
                    self.mutation_gate.run(
                        self.panel_state.record_feature_input_resolution,
                        case.case_token,
                        neutral_receipt.model_dump(mode="json"),
                    )
                self.feature_input.record_policy_execution(
                    candidate_manifest_revision=governance.manifest.revision_sha256,
                    execution=execution,
                    observed_at=observed_at,
                    effective_session=request.target_market_session,
                )
                if execution.status is not FeatureInputExecutionStatus.DATA_TRUTH_REVIEW:
                    self.mutation_gate.run(
                        self.panel_state.record_feature_input_effect,
                        case.case_token,
                        TypeAdapter(type(execution)).dump_python(execution, mode="json"),
                    )
                executions.append(execution)
        except Exception as exc:
            failure_code, explanation = remediation_failure_reason(exc)
            execution_attempt_count = (
                exc.execution_attempt_count if isinstance(exc, DataRemediationActorError) else 0
            )
            prior_failure_codes = (
                exc.prior_failure_codes
                if isinstance(exc, DataRemediationActorError)
                else (type(exc).__name__.upper(),)
            )
            agent_execution_hashes = (
                tuple(
                    canonical_hash(value.model_dump(mode="json")) for value in exc.agent_executions
                )
                if isinstance(exc, DataRemediationActorError)
                else ()
            )
            context_audit_hashes = (
                tuple(
                    canonical_hash(value.model_dump(mode="json"))
                    for value in exc.model_context_audits
                )
                if isinstance(exc, DataRemediationActorError)
                else ()
            )
            receipt_values = {
                # Admission-only rechecks have a workspace cycle, not a new
                # provider maintenance run. Preserve that distinction in the reference.
                "maintenance_id": maintenance_id
                or "workspace-cycle:"
                + canonical_hash(["workspace-maintenance", request.request_hash]),
                "case_token": active_case.case_token if active_case is not None else None,
                "evidence_hash": (active_case.evidence_hash if active_case is not None else None),
                "failure_code": failure_code,
                "retryable": True,
                "execution_attempt_count": execution_attempt_count,
                "prior_failure_codes": prior_failure_codes,
                "agent_execution_hashes": agent_execution_hashes,
                "model_context_audit_hashes": context_audit_hashes,
                "observed_at": observed_at,
                "explanation": explanation,
            }
            self.registry.record_data_remediation_failure(
                DataRemediationFailureReceipt.seal(**receipt_values)
            )
            return (
                MaintenanceStatus.DEFERRED,
                observed_at + timedelta(minutes=5),
                failure_code,
            )
        if (
            awaiting_choice
            or any(item.failure_reasons for item in executions)
            or any(
                item.status is FeatureInputExecutionStatus.DATA_TRUTH_REVIEW for item in executions
            )
        ):
            return MaintenanceStatus.REVIEW_PENDING, None, "data.truth_review_required"
        deferred_times = [
            item.retry_after_at
            for item in executions
            if item.status is FeatureInputExecutionStatus.DEFERRED
            and item.retry_after_at is not None
        ]
        if deferred_times:
            return (
                MaintenanceStatus.DEFERRED,
                max(deferred_times),
                "data.remediation_wait",
            )
        if any(
            item.status
            in {
                FeatureInputExecutionStatus.QUARANTINE_READY,
                FeatureInputExecutionStatus.RAW_VALUE_RETAINED,
            }
            for item in executions
        ):
            # These effects require no provider work. Reassess and continue
            # the same command rather than making the user resume it twice.
            fresh = self._govern_quality(
                maintenance_id,
                request=request,
                observed_at=observed_at,
                macro_retry_exhausted=True,
                observed_workers=observed_workers,
                sector_source=governance.sector_source,
            )
            return self._handle_governance_result(
                fresh,
                maintenance_id=maintenance_id,
                request=request,
                observed_at=observed_at,
                observed_workers=observed_workers,
            )
        return (
            MaintenanceStatus.DEFERRED,
            observed_at + timedelta(minutes=5),
            "data.remediation_retry_scheduled",
        )

    def _publish_snapshot_with_bounded_retry(
        self,
        *,
        history_start: date,
        as_of_session: date,
        observed_at: datetime,
    ) -> PublishedFeaturePanelSnapshot | None:
        """Retry one transient publication failure after releasing large frames.

        Publication is content-addressed and idempotent.  Live full-universe
        evidence showed that an immediate publication can fail once after the
        memory-heavy panel build while the same immutable source succeeds in a
        fresh process.  One GC-assisted retry closes that desktop boundary
        without sleeping, changing content identity, or hiding persistent
        source/hash failures.
        """

        if self.snapshot_publisher is None:
            return None
        publication_observed_at = observed_at.astimezone(UTC)
        active_panel = self.panel_state.active_feature_panel(
            self.manifest.profile.market_profile_id
        )
        if active_panel is not None and isinstance(
            active_panel.get("knowledge_cutoff_at"), datetime
        ):
            cutoff = active_panel["knowledge_cutoff_at"]
            assert isinstance(cutoff, datetime)
            cutoff = cutoff.replace(tzinfo=UTC) if cutoff.tzinfo is None else cutoff.astimezone(UTC)
            publication_observed_at = max(publication_observed_at, cutoff)
        for attempt in range(2):
            try:
                published = self.snapshot_publisher.publish(
                    manifest=self.manifest,
                    history_start=history_start,
                    as_of_session=as_of_session,
                    observed_at=publication_observed_at,
                )
            except (OSError, ValueError):
                if attempt == 0:
                    gc.collect()
                continue
            self._record_bootstrap_if_first(published)
            return published
        return None

    def _bind_manifest(
        self,
        manifest: UniverseManifest,
        *,
        observed_at: datetime,
        effective_session: date | None = None,
        authority: str = "workspace_readiness",
        reference_hash: str | None = None,
    ) -> None:
        self.mutation_gate.run(self.market_data.bootstrap, manifest)
        self._journal_membership(
            manifest,
            effective_session=effective_session,
            observed_at=observed_at,
            authority=authority,
            reference_hash=reference_hash or manifest.revision_sha256,
        )
        if manifest.revision_sha256 != self.manifest.revision_sha256:
            record = self.market_data.readiness.load(manifest.profile.market_profile_id)
            if record is not None:
                self.mutation_gate.run(
                    self.market_data.readiness.save,
                    market_profile_id=manifest.profile.market_profile_id,
                    status=WorkspaceReadinessStatus.FEATURE_BUILDING.value,
                    active_manifest_id=manifest.manifest_id,
                    active_manifest_revision=manifest.revision_sha256,
                    active_membership_fingerprint=(
                        manifest.membership_fingerprint or record.active_membership_fingerprint
                    ),
                    active_candidate_manifest_document=record.active_candidate_manifest_document,
                    pending_membership_fingerprint=None,
                    pending_candidate_manifest_document=None,
                    last_checked_at=record.last_checked_at,
                    last_changed_at=record.last_changed_at,
                    failure_code=None,
                    observed_at=observed_at,
                )
        self.manifest = manifest
        self.feature_foundation.manifest = manifest

    def _journal_membership(
        self,
        manifest: UniverseManifest,
        *,
        effective_session: date | None,
        observed_at: datetime,
        authority: str,
        reference_hash: str,
    ) -> None:
        """Append the membership change a bound manifest makes, once the Universe is bootstrapped.

        Before the bootstrap record exists the workspace is still building its
        initial cohort and the manifest under construction is the membership
        of every session. Source membership transitions remain prospective.
        Feature quality, baseline qualification and missing Sector evidence are
        not EXIT/ENTRY authorities: their dated restrictions are separate.
        A bound manifest whose members the journal already holds writes nothing.
        """

        profile_id = manifest.profile.market_profile_id
        bootstrap = self.market_data.universe_bootstrap(profile_id)
        if bootstrap is None:
            return
        if authority in {
            "feature_input_gateway",
            "sector_partial_exclusion",
            "baseline_feature_qualification",
        }:
            return  # Dated source usability is not nominal ENTRY/EXIT authority.
        events = self.market_data.membership_events(profile_id)
        current = journal_members(bootstrap, events)
        following = tuple(sorted(item.listing_id for item in manifest.listings))
        if set(current) == set(following):
            return
        if effective_session is None:
            raise ValueError("workspace_maintenance.membership_change_without_effective_session")
        if set(following) - set(current) and authority != "qualified_source_membership":
            return  # Raw-qualified candidates first owe materialized Feature qualification.
        if set(following) - set(current) and not self.panel_state.admits_feature_candidate_subset(
            candidate_revision=self.manifest.revision_sha256,
            result_revision=manifest.revision_sha256,
            as_of_session=effective_session,
            qualification_hash=reference_hash,
        ):
            raise ValueError("workspace_maintenance.feature_candidate_admission_missing")
        effective_session = membership_effective_session(
            observed_at=observed_at, decided_at=observed_at, not_before=effective_session
        )
        self.mutation_gate.run(
            self.market_data.append_membership_events,
            membership_events_for_transition(
                market_profile_id=profile_id,
                current_listing_ids=current,
                next_listing_ids=following,
                effective_session=effective_session,
                observed_at=observed_at,
                decided_at=observed_at,
                authority=authority,
                reference_hash=reference_hash,
                manifest_revision=manifest.revision_sha256,
                first_sequence=(events[-1].sequence + 1) if events else 1,
            ),
        )

    def _record_bootstrap_if_first(self, published: PublishedFeaturePanelSnapshot) -> None:
        """Freeze U0 and T0 at the first Gateway-qualified publication, once.

        The record is the boundary between the disclosed initial-cohort
        backfill and the forward, as-observed membership; it is written once
        and never re-derived, and the Panel that was just published is the
        one it names.
        """

        profile_id = self.manifest.profile.market_profile_id
        snapshot_manifest = getattr(published, "manifest", None)
        if snapshot_manifest is None or self.market_data.universe_bootstrap(profile_id) is not None:
            return
        summary = snapshot_manifest.safe_summary
        quality = summary.get("quality_governance")
        if not isinstance(quality, dict) or not bool(quality.get("gateway_qualified")):
            return
        policy_hash = (
            self.feature_input.gateway.policy.policy_hash
            if self.feature_input is not None
            else str(quality.get("quality_admission_hash") or "")
        )
        if not policy_hash or self.manifest.qualification_policy_hash is None:
            return
        record = UniverseBootstrapRecord(
            market_profile_id=profile_id,
            t0_session=snapshot_manifest.as_of_session,
            history_start=snapshot_manifest.history_start,
            cohort_listing_ids=tuple(item.listing_id for item in self.manifest.listings),
            cohort_hash="",
            manifest_revision=self.manifest.revision_sha256,
            candidate_manifest_hash=(
                self.manifest.membership_fingerprint or self.manifest.revision_sha256
            ),
            qualification_policy_hash=self.manifest.qualification_policy_hash,
            feature_input_policy_hash=policy_hash,
            source_observed_at=datetime.combine(
                self.manifest.profile.manifest_as_of, datetime.min.time(), tzinfo=UTC
            ),
            admitted_at=published.materialized_at,
            panel_snapshot_hash=snapshot_manifest.snapshot_hash,
            derivation="FIRST_QUALIFIED_PUBLICATION",
        )
        self.mutation_gate.run(self.market_data.record_universe_bootstrap, record)

    def _bind_derived_manifest_evidence(
        self,
        source_manifest: UniverseManifest,
        target_manifest: UniverseManifest,
        *,
        requested_as_of: date,
        observed_at: datetime,
    ) -> bool:
        """Publish child-bound action and sector lineage before activation."""

        if source_manifest.revision_sha256 == target_manifest.revision_sha256:
            return True
        try:
            # Evidence readers join through the candidate membership rows. This
            # registers the governed child without activating it; activation
            # still follows successful action and Sector readback below.
            self.mutation_gate.run(self.market_data.bootstrap, target_manifest)
            self.mutation_gate.run(
                self.market_data.bind_action_audit_receipts_to_manifest,
                source_manifest,
                target_manifest,
                requested_as_of=requested_as_of,
                now=observed_at,
            )
            # Through the activation coordinator, so the ledger holds a sector
            # map for the revision this rebinding activates; the coordinator
            # runs the store mutation through the gate itself.
            sector_binding = self.feature_foundation.sector_activation.bind_to_manifest(
                target_manifest
            )
        except ValueError:
            return False
        return sector_binding is not None

    def _history_start(self) -> date:
        active_panel = self.panel_state.active_feature_panel(
            self.manifest.profile.market_profile_id
        )
        if active_panel is not None:
            inherited = active_panel.get("history_start")
            if isinstance(inherited, date):
                return inherited
            if inherited is not None:
                return date.fromisoformat(str(inherited))
        # Before the first Panel: every member's earliest held session, read
        # in one scan rather than one full history per listing (a first-use
        # cycle asks several times before its Panel exists).
        ranges = self.market_data.listing_raw_ranges(
            tuple(listing.listing_id for listing in self.manifest.listings)
        )
        if any(listing.listing_id not in ranges for listing in self.manifest.listings):
            raise ValueError("feature foundation requires raw history for every listing")
        # Initial onboarding admits only listings qualified for one shared
        # Research History Window.  A later listing must never move that
        # global boundary forward; active revisions inherit it from the panel.
        earliest: date = min(first for first, _last in ranges.values())
        return earliest

    def _finish(
        self,
        cycle_id: str,
        *,
        phase: MaintenancePhase,
        status: MaintenanceStatus,
        observed_at: datetime,
        change_set: MarketDataChangeSet | None = None,
        child_task_refs: tuple[str, ...] | None = None,
        effect_receipts: tuple[str, ...] | None = None,
        retry_after_at: datetime | None = None,
        failure_code: str | None = None,
        failure_cause: Mapping[str, object] | None = None,
        transport_workers: int | None = None,
    ) -> WorkspaceMaintenanceOutcome:
        if (
            change_set is not None
            and not change_set.empty
            and status in {MaintenanceStatus.COMPLETED, MaintenanceStatus.NOOP}
            and self.publish_adjusted_return_revision is not None
        ):
            try:
                self.publish_adjusted_return_revision()
            except Exception:
                phase = MaintenancePhase.MARKET_DATA
                status = MaintenanceStatus.BLOCKED
                retry_after_at = None
                failure_code = "workspace_maintenance.adjusted_return_revision_projection_failed"
                failure_cause = None
        self.registry.update(
            cycle_id,
            phase=phase,
            status=status,
            observed_at=observed_at,
            change_set=change_set,
            child_task_refs=child_task_refs,
            effect_receipts=effect_receipts,
            retry_after_at=retry_after_at,
            failure_code=failure_code,
            transport_workers=transport_workers,
        )
        return WorkspaceMaintenanceOutcome(
            cycle_id=cycle_id,
            status=status,
            phase=phase,
            change_set_hash=change_set.change_set_hash if change_set else None,
            child_task_refs=(
                child_task_refs
                if child_task_refs is not None
                else self.registry.cycle(cycle_id).child_task_refs
            ),
            retry_after_at=retry_after_at,
            failure_code=failure_code,
            failure_cause=failure_cause,
        )

    @staticmethod
    def _optional_date(value: object) -> date | None:
        return date.fromisoformat(str(value)) if value else None

    @staticmethod
    def _utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("workspace maintenance time must be timezone-aware")
        return value.astimezone(UTC)
