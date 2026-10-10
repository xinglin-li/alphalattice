"""Workspace input maintenance over existing owners; never Factor research."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass
from datetime import date, datetime
from hashlib import sha256
from pathlib import Path
from typing import cast
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

import duckdb

from alphalattice.control.data_platform.contracts import DataRemediationExecutionReceipt
from alphalattice.control.data_platform.maintenance.contracts import (
    ListingMarketDataChange,
    MaintenanceStatus,
    MaintenanceTrigger,
    WorkspaceDataUpdateBinding,
    WorkspaceDataUpdatePlan,
    WorkspaceDataUpdateReceipt,
    WorkspaceInputStatus,
    WorkspaceMaintenanceRequest,
    full_history_audit_requirement,
    workspace_maintenance_data_policy_hash,
)
from alphalattice.control.data_platform.maintenance.data_changes import (
    BookObligations,
    WorkspaceDataChanges,
)
from alphalattice.control.data_platform.maintenance.reconciliation import (
    capture_verified_maintenance_completion,
)
from alphalattice.control.data_platform.maintenance.registry import (
    DuckDbWorkspaceMaintenanceRegistry,
)
from alphalattice.control.data_platform.readiness import (
    SourceLoader,
    WorkspaceConsentAction,
    WorkspaceReadinessConsent,
    WorkspaceReadinessGate,
    _latest_common_us_session,
    build_workspace_readiness,
    source_check_due,
    source_verification_failed,
)
from alphalattice.control.data_platform.task_telemetry import TaskTelemetry
from alphalattice.control.product_host.composition.application_session import (
    WorkspaceApplicationSession,
)
from alphalattice.control.product_host.composition.research_workspace import (
    RESEARCH_WORKSPACE_MANIFEST_NAME,
    ResearchWorkspaceManifest,
    ResearchWorkspaceManifestHolder,
    create_research_workspace_manifest,
    held,
    manifest_fields_hash,
    read_research_workspace_manifest,
    update_research_workspace_manifest,
)
from alphalattice.control.product_host.composition.workspace import WorkspaceRuntime
from alphalattice.control.product_host.research_authoring.feature_activations import (
    feature_catalog_for,
    workspace_feature_catalog,
)
from alphalattice.control.product_host.storage.backup import (
    defer_automatic_backup,
    settle_automatic_backup,
)
from alphalattice.control.product_host.storage.inventory import require_storage_capacity
from alphalattice.control.product_host.storage.plan_previews import (
    PLAN_PREVIEWS_DIRECTORY,
    PreviewRegistry,
)
from alphalattice.control.product_host.storage.retention import require_no_pending_cleanup
from alphalattice.control.task_control.contracts import (
    ResearchGoal,
    ResearchPlan,
    StageFailureCause,
    TaskEvidence,
    TaskExecution,
    TaskExecutionCompatibility,
    TaskInputEnvelope,
    TaskLifecycle,
    TaskRecord,
    TaskReplan,
    WorkItemDefinition,
)
from alphalattice.control.task_control.registry import (
    DuckDbTaskControlRegistry,
    resolve_task_control_database,
)
from alphalattice.control.task_control.runner import (
    StageDisposition,
    StageExecutionResult,
)
from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.control.workspace_runtime.content_store import ContentAddressedStoreError
from alphalattice.control.workspace_runtime.database import WorkspaceDatabase
from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.control.workspace_runtime.network_access import network_access
from alphalattice.control.workspace_runtime.writer_lease import WorkspaceWriterLease
from alphalattice.foundation.factor_research.publication.artifacts import (
    FactorResearchArtifactStore,
)
from alphalattice.foundation.feature_engine.contracts import panel_as_of_listing_identity
from alphalattice.foundation.feature_engine.producers.reference_data import MarketReference
from alphalattice.foundation.feature_engine.storage.repositories import (
    FeatureStateRepository,
    PanelStateRepository,
)
from alphalattice.foundation.market_data_ops.runtime.universe_maintenance import (
    CurrentUniverseMaintenance,
    current_universe_maintenance_id,
)
from alphalattice.foundation.market_data_ops.runtime.universe_onboarding import (
    CurrentUniverseOnboardingStatus,
)
from alphalattice.foundation.market_data_ops.sources.providers import (
    MarketDataProvider,
    YFinanceMarketDataProvider,
)
from alphalattice.foundation.market_data_ops.storage.duckdb import (
    CurrentUniverseMaintenanceListing,
    ManifestTransitionRecord,
    MarketDataRepository,
)
from alphalattice.foundation.research_foundation.contracts import (
    PreResearchDeskSafeProjection,
    ResearchFoundationBinding,
)
from alphalattice.interface.local_application.dispatcher import CommandAdmission
from alphalattice.interface.local_application.failure_codes import public_failure
from alphalattice.kernel.shared_kernel.identity import canonical_hash, schema_structure
from alphalattice.kernel.shared_kernel.project_layout import resolve_playpen_root

PROFILE = (
    resolve_playpen_root(Path(__file__)) / "config/market-profiles/us-current-index-research.yaml"
)
PLAN_FIELDS = ("data_update",)
"""The manifest fields a plan reads: the data update binding it runs under and writes. A plan binds
them, never the whole manifest, so a publication of fields it does not read leaves it applicable
(V223, OW10)."""

DATA_UPDATE_TASK_KIND = "workspace_data_update"
_STAGES = ("validate_update_request", "maintain_data_feature", "publish_update_receipt")
# One maintenance cycle advances one network chunk of listings: the runner
# fetches the chunk with its transport workers and applies every completed
# fetch on one retained instance, so the coordinator's preflight, the
# capacity walk, the cancel check and the engine's checkpoint are paid once
# per chunk instead of once per listing. Cancellation is still honoured at
# the cycle boundary and every listing still commits on its own.
_MAINTENANCE_CYCLE_LISTINGS = CurrentUniverseMaintenance.chunk_size
_MEMBERSHIP_REVIEW = "workspace_data_update.membership_changed_review_required"
"""A plan without a declared change met a changed membership; a new plan, not a retry."""
_RECOVERABLE_BLOCKS = {
    "data.truth_review_required",
    "workspace_maintenance.derived_manifest_evidence_incomplete",
    "workspace_maintenance.snapshot_publication_failed",
    "workspace_maintenance.feature_source_reconciliation_failed",
    "data.full_history_audit_approval_required",
    # Its way on is the plan run again once a person allows the network: the rerun resumes the
    # stopped Task, and is refused while the network stays closed (V600).
    "workspace_data_update.source_access_not_admitted",
    # Its way on is the backup's cause resolved and the plan run again: the rerun resumes the
    # stopped Task, whose first stage takes the backup again.
    "workspace_data_update.archive_failed",
}
"""The stops a rerun of the same plan resumes, once a person has cleared their cause."""


def installed_data_update_binding(workspace: Path | None = None) -> WorkspaceDataUpdateBinding:
    """The binding a workspace's data update runs under.

    The profile, the data policy and the daily catalog the workspace's activations make (EX).

    Args:
        workspace: The workspace; none binds the shipped catalog.

    Returns:
        The binding.
    """
    return WorkspaceDataUpdateBinding.seal(
        profile_file_hash=sha256(PROFILE.read_bytes()).hexdigest(),
        data_policy_hash=workspace_maintenance_data_policy_hash(),
        feature_catalog_hash=workspace_feature_catalog(workspace).binding.catalog_hash,
    )


def _confined_workspace(root: Path) -> Path:
    root = root.resolve()
    for name in (
        "market-data.duckdb",
        "artifacts",
        "artifacts/feature-panel/manifests",
        "artifacts/feature-panel/chunks",
        "artifacts/factor-research",
        "yfinance-cache",
        "runtime",
    ):
        try:
            (root / name).resolve().relative_to(root)
        except ValueError as error:
            raise ValueError("workspace_data_update.path_escapes_workspace") from error
    if not (root / "market-data.duckdb").is_file():
        raise ValueError("workspace_data_update.existing_data_required")
    return root


def read_workspace_inputs(
    root: Path, binding: WorkspaceDataUpdateBinding, *, allow_transition: bool = False
) -> WorkspaceInputStatus:
    """Local metadata/immutable-manifest read; never instantiate maintenance."""
    root = _confined_workspace(root)
    if binding != installed_data_update_binding(root):
        raise ValueError("workspace_data_update.installed_binding_mismatch")
    market = MarketDataRepository(root)
    # Dozens of small reads over one database: one retained read-only
    # instance serves them all. A write attempted inside is refused, not
    # upgraded; this is a read-only consumer and stays one.
    with market.database.retain(read_only=True):
        return _read_workspace_inputs(root, binding, market, allow_transition=allow_transition)


def _read_workspace_inputs(
    root: Path,
    binding: WorkspaceDataUpdateBinding,
    market: MarketDataRepository,
    *,
    allow_transition: bool,
) -> WorkspaceInputStatus:
    readiness = market.readiness.load(binding.market_profile_id)
    if readiness is None or readiness.active_manifest_id is None:
        raise ValueError("workspace_data_update.initialization_required")
    manifest = market.load_universe_manifest(readiness.active_manifest_id)
    if (
        manifest.revision_sha256 != readiness.active_manifest_revision
        or manifest.profile.market_profile_id != binding.market_profile_id
    ):
        raise ValueError("workspace_data_update.manifest_mismatch")
    profile = manifest.profile
    if (
        profile.market,
        profile.currency,
        profile.provider,
        profile.calendar_id,
        profile.daily_price_basis,
    ) != ("US", "USD", "yfinance", "XNYS_XNAS", "split_adjusted"):
        raise ValueError("workspace_data_update.profile_not_admitted")
    snapshot = PanelStateRepository(
        market.database, market_data=market
    ).feature_panel_snapshot_for_active(binding.market_profile_id)
    if snapshot is None:
        raise ValueError("workspace_data_update.active_panel_required")
    resolver = ArtifactResolver(root / "artifacts")
    panel = resolver.load_feature_panel_manifest(str(snapshot["manifest_uri"]))
    for chunk in panel.get("chunks", []):
        path = resolver._feature_panel_chunk_path(str(chunk["chunk_hash"]))
        try:
            path.resolve().relative_to(root)
        except ValueError as error:
            raise ValueError("workspace_data_update.path_escapes_workspace") from error
        resolver.resolve_feature_panel_chunk_ref(
            uri=str(chunk["uri"]),
            content_hash=str(chunk["chunk_hash"]),
            metadata_hash=str(chunk["metadata_hash"]),
        )
    if (
        panel["snapshot_hash"] != snapshot["snapshot_hash"]
        # A Panel under another catalog this workspace made (an activation since) is rebuilt
        # by the update under the binding's; a catalog it cannot resolve is refused.
        or (
            snapshot["catalog_hash"] != binding.feature_catalog_hash
            and feature_catalog_for(root, str(snapshot["catalog_hash"])) is None
        )
        or (
            snapshot["manifest_revision"] != manifest.revision_sha256
            and not (allow_transition and readiness.status == "FEATURE_BUILDING")
        )
    ):
        raise ValueError("workspace_data_update.panel_binding_mismatch")
    gate = build_workspace_readiness(market, profile_path=PROFILE)
    local = gate._ready_or_blocked(readiness) if readiness.status == "RESEARCH_READY" else None
    input_status = local.status.value if local else readiness.status
    if (
        input_status == "RESEARCH_READY"
        and PanelStateRepository(market.database, market_data=market)
        .feature_input_quality_disclosure(result_manifest_revision=manifest.revision_sha256)
        .get("gateway_qualified")
        is not True
    ):
        input_status = "FEATURE_BUILDING"
    if PanelStateRepository(market.database, market_data=market).has_unresolved_feature_input(
        manifest.revision_sha256
    ):
        input_status = "DATA_REVIEW_PENDING"
    foundation_hash = foundation_panel = None
    disposition = "MISSING"
    store = FactorResearchArtifactStore(root / "artifacts")
    try:
        projection = PreResearchDeskSafeProjection.model_validate(
            store.pre_research_desk_projection()
        )
        foundation = ResearchFoundationBinding.model_validate(
            store.load_research_foundation(
                store.uri("research-desk/foundations", projection.foundation_hash)
            )
        )
        if foundation.foundation_hash != projection.foundation_hash:
            raise ValueError("foundation identity differs")
        foundation_hash, foundation_panel = (
            foundation.foundation_hash,
            foundation.feature_panel_snapshot_hash,
        )
        disposition = (
            "MATCHING_PANEL" if foundation_panel == snapshot["snapshot_hash"] else "PRIOR_INPUTS"
        )
    except FileNotFoundError:
        pass
    except ValueError:
        disposition = "UNREADABLE"
    return WorkspaceInputStatus.seal(
        manifest_revision=manifest.revision_sha256,
        data_revision_hash=market.maintenance_revision_fingerprint(
            manifest, listing_ids=tuple(market.research_listing_sources(manifest))
        ),
        data_through=market.manifest_raw_through(manifest),
        adjusted_through=market.manifest_provider_adjusted_through(manifest),
        panel_hash=snapshot["snapshot_hash"],
        panel_through=snapshot["as_of_session"],
        readiness_status=input_status,
        sources_checked_at=readiness.last_checked_at,
        source_check_failed_at=(
            readiness.source_check_failed_at
            if source_verification_failed(
                readiness.last_checked_at, readiness.source_check_failed_at
            )
            else None
        ),
        source_disclosure=(
            "LAST_KNOWN_MEMBERSHIP_AFTER_SOURCE_FAILURE"
            if source_verification_failed(
                readiness.last_checked_at, readiness.source_check_failed_at
            )
            else None
        ),
        foundation_hash=foundation_hash,
        foundation_panel_hash=foundation_panel,
        foundation_disposition=disposition,
        panel_manifest_revision=str(snapshot["manifest_revision"])
        if snapshot["manifest_revision"] != manifest.revision_sha256
        else None,
        listing_set_hash=panel_as_of_listing_identity(panel)[0],
    )


def inspect_existing_data_workspace(root: Path) -> dict[str, object]:
    """Read qualification without taking a writer lease or changing the source.

    This is input readback, not binding admission: the mutating command must still
    check outstanding Tasks and revalidate the source under its own lease.
    """
    try:
        state = read_workspace_inputs(root, installed_data_update_binding(root))
    except (ValueError, OSError, RuntimeError, duckdb.Error) as error:
        code = str(error)
        if not code.startswith("workspace_data_update."):
            code = "workspace_data_update.source_unreadable"
        action, detail = {
            "workspace_data_update.existing_data_required": (
                "SELECT_EXISTING_DATA_WORKSPACE",
                "No local market database exists here. "
                "Select a data workspace, not a virtual workspace.",
            ),
            "workspace_data_update.initialization_required": (
                "SELECT_QUALIFIED_SOURCE_OR_PREPARE_SEPARATE_WORKSPACE",
                "No qualified Universe/readiness is recorded. Use a qualified source "
                "or explicitly prepare a separate research workspace.",
            ),
            "workspace_data_update.active_panel_required": (
                "COMPLETE_SOURCE_FEATURE_PREPARATION",
                "The source has no admitted active Feature/Panel snapshot. "
                "Complete its governed preparation before binding.",
            ),
        }.get(
            code,
            (
                "INSPECT_SOURCE_BINDINGS",
                "Source qualification could not be verified; do not bind or overwrite it.",
            ),
        )
        return {
            "status": "NOT_QUALIFIED",
            "failure_code": code[:240],
            "message": detail,
            "next_action": action,
        }
    qualified = state.readiness_status == "RESEARCH_READY"
    return {
        "status": "QUALIFIED_LOCAL_INPUTS" if qualified else "NOT_QUALIFIED",
        "inputs": state.model_dump(mode="json"),
        "next_action": "BIND_EXISTING_DATA_WORKSPACE" if qualified else "OPEN_SOURCE_READINESS",
        "claim": "INPUT_QUALIFICATION_ONLY_NOT_BINDING_OR_FORWARD_ADMISSION",
        "numerical_call_count": 0,
    }


def bind_existing_data_workspace(root: Path) -> ResearchWorkspaceManifest:
    """Local setup only. No market/Task database creation or source acquisition."""
    root = _confined_workspace(root)
    lease = WorkspaceWriterLease.acquire(root)
    try:
        manifest = (
            read_research_workspace_manifest(root)
            if (root / RESEARCH_WORKSPACE_MANIFEST_NAME).exists()
            else None
        )
        # The resolver refuses by name a store at the root that still holds Tasks.
        tasks = DuckDbTaskControlRegistry.read_existing_tasks(resolve_task_control_database(root))
        if any(
            task.lifecycle
            not in {TaskLifecycle.SUCCEEDED, TaskLifecycle.BLOCKED, TaskLifecycle.CANCELLED}
            for task in tasks
        ):
            raise ValueError("workspace_data_update.unsettled_tasks")
        binding = installed_data_update_binding(root)
        state = read_workspace_inputs(root, binding)
        if state.readiness_status != "RESEARCH_READY":
            raise ValueError("workspace_data_update.qualified_workspace_required")
        if manifest is None:
            # Explicit one-time binding, after qualified Data evidence was read.
            # Ordinary launcher startup still refuses unknown nonempty roots.
            return create_research_workspace_manifest(
                root,
                ResearchWorkspaceManifest.research_only(f"research-{uuid4()}").with_bindings(
                    data_update=binding
                ),
            )
        if manifest.data_update == binding:
            return manifest
        # The lease is held, so this process is the only writer; the one write all the same.
        _, updated = update_research_workspace_manifest(
            root,
            lambda current: current.with_bindings(data_update=binding),
            gate=WorkspaceMutationGate(),
        )
        return updated
    finally:
        lease.close()


def _partition_reuse(value: object) -> dict[str, int] | None:
    """The Panel's `partition_reuse` summary as two non-negative counts, or None when the
    Panel records none; any other shape is the Panel's own disclosure error."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("workspace_data_update.partition_reuse_disclosure_invalid")
    counts = {}
    for key in ("reused_partition_count", "composed_partition_count"):
        count = value.get(key)
        if type(count) is not int or count < 0:
            raise ValueError("workspace_data_update.partition_reuse_disclosure_invalid")
        counts[key] = count
    return counts


def _restated_dates(item: ListingMarketDataChange) -> set[date]:
    """The distinct restated sessions of one listing's change: a raw bar corrected in place
    or an adjusted return that moved, on sessions the update did not add. One date named by
    both corrections is one session; a new session's own return is new work."""
    return (set(item.raw_correction_sessions) | set(item.adjusted_return_change_sessions)) - set(
        item.new_sessions
    )


def _task_contract(plan: WorkspaceDataUpdatePlan):  # type: ignore[no-untyped-def]
    envelope = TaskInputEnvelope.create(
        task_kind=DATA_UPDATE_TASK_KIND,
        input_schema_id="workspace-data-update"
        if plan.change is None
        else "workspace-data-update-human-confirmed",
        payload={"plan": plan.model_dump(mode="json")},
    )
    goal = ResearchGoal.create(
        goal_kind="UPDATE_WORKSPACE_INPUTS",
        input_hash=envelope.input_hash,
        deliverable_kind="WorkspaceDataUpdateReceipt",
        summary="Update admitted inputs; retain Foundation research.",
    )
    items = tuple(
        WorkItemDefinition.create(
            stage_id=stage,
            dependency_ids=_STAGES[:index],
            verifier_id=f"workspace_data_update.{stage}",
        )
        for index, stage in enumerate(_STAGES)
    )
    workflow = ResearchPlan.create(
        goal_hash=goal.goal_hash,
        workflow_definition_hash=canonical_hash(_STAGES),
        verifier_catalog_hash=canonical_hash(tuple(item.verifier_id for item in items)),
        work_items=items,
    )
    return envelope, goal, workflow


NETWORK_WORK_WORDS = {
    "DATA": "missing market data through session {session}",
    "MEMBERSHIP": "the membership source check for session {session}",
    "CANDIDATES": "the failed-candidate data retry through session {session}",
    "HISTORY": "the admitted full-history acquisition",
    "REFERENCE": "the SPY market reference through session {session}",
    "SECTOR": "the candidate Sector source check for session {session}",
}
"""Named provider work: the same needs in the plan and a compact Task refusal."""


class WorkspaceDataUpdateApplication:
    """Own admitted Data/Feature maintenance, exact change consent and task-bound receipts."""

    task_kind = DATA_UPDATE_TASK_KIND
    replans = (
        TaskReplan(
            task_kind=DATA_UPDATE_TASK_KIND, preview="DATA_UPDATE_PLAN", admitting="DATA_UPDATE_RUN"
        ),
    )
    """The re-plan of the Task kind this owner admits, which the recovery view offers
    (V188)."""
    portfolio_obligations: Callable[[], BookObligations] | None = None
    recorded_data_confirmation: Callable[[DataRemediationExecutionReceipt], bool] | None = None

    @property
    def manifest(self) -> ResearchWorkspaceManifest:
        """The workspace manifest, read from the one holder the Host refreshes (V182)."""
        return self._manifests.current

    @staticmethod
    def _bound_covers(day: date | None, target: date) -> bool:
        """A published through bound includes its stated session."""
        return day is not None and day >= target

    @classmethod
    def _held_lanes(cls, plan: WorkspaceDataUpdatePlan) -> tuple[bool, ...]:
        """Each through bound includes the stated session's data."""
        bounds = (plan.before.data_through, plan.before.adjusted_through, plan.before.panel_through)
        return tuple(cls._bound_covers(day, plan.request.target_market_session) for day in bounds)

    def _candidate_source_covers(self, plan: WorkspaceDataUpdatePlan) -> bool:
        """A Feature retry's captured parent can need data beyond the held child."""
        scope = plan.request.candidate_recheck
        if scope is None:
            return True
        market = MarketDataRepository(self.session.workspace)
        parent = market.load_universe_manifest_revision(scope.parent_manifest_revision)
        return all(
            self._bound_covers(bound, plan.request.target_market_session)
            for bound in (
                market.manifest_raw_through(parent),
                market.manifest_provider_adjusted_through(parent),
            )
        )

    def _reference_covers(self, plan: WorkspaceDataUpdatePlan) -> bool:
        """Read the Feature owner's verified SPY coverage before local Panel work."""
        market = MarketDataRepository(self.session.workspace)
        manifest = market.current_quality_filtered_research_manifest(
            market_profile_id=plan.binding.market_profile_id
        )
        if manifest is None:
            return False
        return (
            FeatureStateRepository(market.database, market_data=market).verified_market_reference(
                MarketReference.spy(manifest).manifest,
                reference_id="SPY",
                requested_as_of=plan.request.target_market_session,
            )
            is not None
        )

    def _candidate_sector_check_due(self, plan: WorkspaceDataUpdatePlan) -> bool:
        """A captured Feature retry reads its parent's complete Sector reference."""
        scope = plan.request.candidate_recheck
        if scope is None:
            return False
        market = MarketDataRepository(self.session.workspace)
        parent = market.load_universe_manifest_revision(scope.parent_manifest_revision)
        return bool(
            FeatureStateRepository(market.database, market_data=market).sector_revision_refresh_due(
                parent, observed_at=plan.request.request_clock
            )
        )

    def network_work(self, plan: WorkspaceDataUpdatePlan) -> dict[str, str]:
        """Name the planned provider work, apart from local Feature/Panel completion."""
        target = plan.request.target_market_session
        data, adjusted, panel = self._held_lanes(plan)
        work = {}
        if not (data and adjusted):
            work["DATA"] = NETWORK_WORK_WORDS["DATA"].format(session=target.isoformat())
        elif not panel and not self._reference_covers(plan):
            work["REFERENCE"] = NETWORK_WORK_WORDS["REFERENCE"].format(session=target.isoformat())
        if source_check_due(
            plan.before.sources_checked_at,
            plan.request.request_clock,
            reference_session=target,
            failed_at=plan.before.source_check_failed_at,
            operation_started_at=plan.request.request_clock,
        ):
            work["MEMBERSHIP"] = NETWORK_WORK_WORDS["MEMBERSHIP"].format(session=target.isoformat())
        if plan.request.candidate_data_recheck is not None or not self._candidate_source_covers(
            plan
        ):
            work["CANDIDATES"] = NETWORK_WORK_WORDS["CANDIDATES"].format(session=target.isoformat())
        if self._candidate_sector_check_due(plan):
            work["SECTOR"] = NETWORK_WORK_WORDS["SECTOR"].format(session=target.isoformat())
        if plan.request.full_history_listing_ids or (
            plan.change is not None and plan.change.additions
        ):
            work["HISTORY"] = NETWORK_WORK_WORDS["HISTORY"]
        return work

    def historical_inputs_cover(self, plan: WorkspaceDataUpdatePlan) -> bool:
        """Use an inclusive ready snapshot only when no planned upkeep remains."""
        return (
            plan.before.readiness_status == "RESEARCH_READY"
            and all(self._held_lanes(plan))
            and not self.network_work(plan)
            and plan.request.candidate_recheck is None
            and plan.change is None
        )

    @property
    def _provider_requires_network(self) -> bool:
        """A supplied network adapter needs the same permission as the default one."""
        return self.provider is None or isinstance(self.provider, YFinanceMarketDataProvider)

    def source_access(self, plan: WorkspaceDataUpdatePlan) -> str:
        """The plan label and admission share the provider's access boundary."""
        if not self.network_work(plan):
            return "LOCAL_ADMITTED_INPUTS"
        return (
            "EXPLICIT_NETWORK_REQUIRED"
            if self._provider_requires_network
            else "HOST_SUPPLIED_PROVIDER"
        )

    def __init__(
        self,
        *,
        session: WorkspaceApplicationSession,
        manifest: ResearchWorkspaceManifest | ResearchWorkspaceManifestHolder,
        clock: Callable[[], datetime],
        provider: MarketDataProvider | None = None,
        source_loader: SourceLoader | None = None,
    ) -> None:
        """Wire held workspace declarations, explicit source owners and maintenance telemetry.

        Args:
            session: Retained workspace writer/task session.
            manifest: Held workspace declaration.
            clock: Explicit observed-time source.
            provider: Optional admitted market data provider.
            source_loader: Optional admitted source capture loader.
        """
        self.session, self.clock = session, clock
        self._manifests = held(manifest)
        self.provider, self.source_loader = provider, source_loader
        self.last_plan: WorkspaceDataUpdatePlan | None = None
        self.changes = WorkspaceDataChanges(session.workspace, session.mutation_gate)
        self._previews: PreviewRegistry[WorkspaceDataUpdatePlan] = PreviewRegistry(
            model=WorkspaceDataUpdatePlan,
            clock=clock,
            root=session.workspace / "runtime" / PLAN_PREVIEWS_DIRECTORY / "data-update",
            hash_field="content_hash",
        )
        # The Task-bound work progress of the maintenance stage, kept beside the Task
        # (shared with first-use preparation): telemetry, never authority.
        self.telemetry = TaskTelemetry(
            session.workspace, root="runtime/data-update", stages=_STAGES, clock=clock
        )

    def _obligations(self) -> BookObligations:
        return {} if self.portfolio_obligations is None else self.portfolio_obligations()

    def _readiness(self, market: MarketDataRepository) -> WorkspaceReadinessGate:
        return build_workspace_readiness(
            market, profile_path=PROFILE, provider=self.provider, source_loader=self.source_loader
        )

    def _approved_plans(self) -> tuple[WorkspaceDataUpdatePlan, ...]:
        return tuple(
            self._plan_of(t, require_current=False)
            for t in self.session.task_control_registry.tasks()
            if t.task_kind == self.task_kind
            and t.input.input_schema_id == "workspace-data-update-human-confirmed"
            and t.lifecycle is not TaskLifecycle.CANCELLED
        )

    def _binding(self) -> WorkspaceDataUpdateBinding:
        binding = self.manifest.data_update
        if binding is None:
            raise ValueError("workspace_data_update.not_configured")
        if manifest_fields_hash(
            read_research_workspace_manifest(self.session.workspace), PLAN_FIELDS
        ) != manifest_fields_hash(self.manifest, PLAN_FIELDS):
            raise ValueError("workspace_data_update.workspace_manifest_changed")
        if binding != installed_data_update_binding(self.session.workspace):
            raise ValueError("workspace_data_update.installed_binding_mismatch")
        return binding

    def plan(
        self, *, fresh: bool = False, recovery_task_id: UUID | None = None
    ) -> dict[str, object]:
        """Preview exact maintenance, due candidate checks and any explicit membership change.

        Raw candidate retries and Feature/Sector rechecks are planned independently. Planning
        neither runs Factor research nor grants source access. A data update that has not
        ended answers its own plan: its run follows it, or resumes a deferred one once due
        (V601); planned anew, the later update queued behind the deferral (V604).

        Args:
            fresh: Plan anew even while a data update has not ended: a research update's own
                data stage, sealed into its own Task.
            recovery_task_id: The exact stopped Task whose offered continuation is being read.

        Returns:
            Transition refusal, ordinary maintenance plan, human confirmation proposal with source
            access and valuation obligations, or the plan of the update that has not ended.
        """
        waiting = None if fresh or recovery_task_id is not None else self._waiting()
        if waiting is not None:
            planned = self._plan_of(waiting)
            self.last_plan = planned
            return self._plan_answer(planned, self.clock(), admitted=True)
        binding = self._binding()
        state = read_workspace_inputs(self.session.workspace, binding, allow_transition=True)
        now = self.clock()
        stopped = (
            (self.session.task_control_registry.task(recovery_task_id),)
            if recovery_task_id is not None
            else ()
            if fresh
            else reversed(self.session.task_control_registry.tasks())
        )
        audit_parent = None
        for task, planned in self._stopped_changes(state, stopped):
            if task.failure_code == "data.full_history_audit_approval_required":
                # The audit it owes is planned and admitted onto its own cycle.
                audit_parent = planned
                break
            self._require_plan(planned)
            self.last_plan = planned
            return self._plan_answer(planned, now, admitted=True)
        if state.panel_manifest_revision is not None and audit_parent is None:
            raise ValueError("workspace_data_update.transition_not_verified")
        target = (
            _latest_common_us_session(on_or_before=now.date(), observed_at=now)
            if audit_parent is None
            else audit_parent.request.target_market_session
        )
        if state.readiness_status == "ONBOARDING_IN_PROGRESS":
            self.last_plan = None
            return {
                "status": "BLOCKED",
                "inputs": state.model_dump(mode="json"),
                "failure_code": "workspace_data_update.existing_owner_action_required",
                "next_action": "RESOLVE_EXISTING_WORKSPACE_TRANSITION",
            }
        obligations = self._obligations()
        grants = self.changes.active_grants(self._approved_plans(), obligations)
        change = self.changes.proposal(
            state=state,
            readiness=self._readiness(MarketDataRepository(self.session.workspace)),
            observed_at=now,
            obligations=obligations,
            valuation_grants=grants,
            target_session=target,
        )
        if audit_parent is not None and (change is None or change.action != "FULL_HISTORY_AUDIT"):
            raise ValueError("workspace_data_update.audit_requirement_unverified")
        if state.readiness_status == "MANIFEST_UPDATE_PENDING" and change is None:
            self.last_plan = None
            return {
                "status": "BLOCKED",
                "inputs": state.model_dump(mode="json"),
                "failure_code": "workspace_data_update.existing_owner_action_required",
                "next_action": "RESOLVE_EXISTING_WORKSPACE_TRANSITION",
            }
        readiness = self._readiness(MarketDataRepository(self.session.workspace))
        # Two kinds of due work, planned independently: the raw retry of failed
        # candidates and the Feature/Sector recheck of quarantined parent
        # members. A day on which both fall due carries both; the raw retry
        # admitting nothing does not settle the other, and the coordinator
        # resolves the Feature scope against the parent as it stands after
        # the raw retry.
        raw_recheck = (
            readiness.candidate_data.plan(observed_at=now, target_session=target)
            if change is None
            else None
        )
        request = WorkspaceMaintenanceRequest.create(
            market_profile_id=binding.market_profile_id,
            target_market_session=target,
            knowledge_cutoff_at=None,
            requested_at=now,
            trigger=MaintenanceTrigger.USER_REQUEST,
            membership_revision=state.manifest_revision,
            data_policy_hash=binding.data_policy_hash,
            feature_policy_hash=canonical_hash(
                {"catalog": binding.feature_catalog_hash, "invalidation": "domain-topology"}
            ),
            full_history_listing_ids=() if change is None else change.full_history_listing_ids,
            candidate_recheck=(
                readiness.plan_feature_candidate_recheck(observed_at=now)
                if change is None
                else None
            ),
            candidate_data_recheck=raw_recheck,
        )
        # The answer names this plan, never the owner's last one, which a concurrent plan
        # may have replaced in between (V534).
        planned = WorkspaceDataUpdatePlan.seal(
            workspace_id=self.manifest.workspace_id,
            workspace_manifest_hash=manifest_fields_hash(self.manifest, PLAN_FIELDS),
            binding=binding,
            before=state,
            request=request,
            change=change,
            valuation_grants=grants,
        )
        self.last_plan = planned
        # A change plan is kept in the change store, which its person's confirm reads; a
        # maintenance plan is a preview in the plan store, so its run finds it after a Host
        # restart within its hour (V537).
        if change is not None:
            self.changes.save_plan(planned)
        else:
            self._previews.remember(planned)
        return self._plan_answer(planned, now, admitted=False)

    def _partial_transition(
        self, task: TaskRecord, plan: WorkspaceDataUpdatePlan, state: WorkspaceInputStatus
    ) -> bool:
        """Read the exact approved journal that still owes its prior Panel an update (DUPD)."""
        change = plan.change
        if (
            task.input.input_schema_id != "workspace-data-update-human-confirmed"
            or change is None
            or change.action != "UNIVERSE"
            or state.panel_manifest_revision is None
        ):
            return False
        market = MarketDataRepository(self.session.workspace)
        transition = market.manifest_transition(str(change.transition_id))
        readiness = market.readiness.load(plan.binding.market_profile_id)
        if (
            state.panel_hash != plan.before.panel_hash
            or not self._accepts(plan, transition, state.manifest_revision)[0]
        ):
            return False
        if (
            transition.lifecycle != "ACTIVATED"
            or transition.approved_at is None
            or transition.activated_at is None
            or transition.market_profile_id != plan.binding.market_profile_id
            or transition.prior_manifest_revision != plan.before.manifest_revision
            or state.panel_manifest_revision != transition.prior_manifest_revision
            or transition.additions != tuple(sorted(change.additions))
            or transition.removals != tuple(sorted(change.removals))
            or readiness is None
            or canonical_hash(readiness.active_candidate_manifest_document)
            != change.candidate_document_hash
        ):
            raise ValueError("workspace_data_update.transition_not_verified")
        execution = self.changes.execution(plan)
        if execution is not None and (
            execution.request.membership_revision != transition.next_manifest_revision
        ):
            raise ValueError("workspace_data_update.execution_binding_invalid")
        return True

    def _waiting(self) -> TaskRecord | None:
        """The data update that has not ended, if one has not (V604)."""
        return next(
            (
                task
                for task in reversed(self.session.task_control_registry.tasks())
                if task.task_kind == self.task_kind
                and task.lifecycle
                in (
                    TaskLifecycle.QUEUED,
                    TaskLifecycle.RUNNING,
                    TaskLifecycle.RECOVERY_REQUIRED,
                    TaskLifecycle.DEFERRED,
                )
            ),
            None,
        )

    def _plan_answer(
        self, planned: WorkspaceDataUpdatePlan, now: datetime, *, admitted: bool
    ) -> dict[str, object]:
        """A plan's answer: a change waits for its person's approval unless its Task was
        admitted, which its run follows or resumes."""
        state, request, change = planned.before, planned.request, planned.change
        target = request.target_market_session
        confirm = change is not None and not admitted
        grants = planned.valuation_grants
        formation = self.changes.formation_scope(state) if change is not None else None
        network_work = self.network_work(planned)
        return {
            "status": "CONFIRMATION_REQUIRED" if confirm else "PLANNED",
            "plan_hash": planned.content_hash,
            "target_session": target.isoformat(),
            "inputs": state.model_dump(mode="json"),
            "work": (
                "Provider work: " + "; ".join(network_work.values()) + ". "
                if network_work
                else "No provider work: consume held market data. "
            )
            + "Maintain Feature, verify Panel; no Factor research.",
            "source_access": self.source_access(planned),
            "source_check_due": source_check_due(
                state.sources_checked_at,
                now,
                reference_session=target,
                failed_at=state.source_check_failed_at,
            ),
            "next_action": "DATA_CHANGE_CONFIRM" if confirm else "DATA_UPDATE_RUN",
            "next_requests": {
                "confirm" if confirm else "run": {
                    "operation": "DATA_CHANGE_CONFIRM" if confirm else "DATA_UPDATE_RUN",
                    "update_plan_hash": planned.content_hash,
                }
            },
            "candidate_recheck": asdict(request.candidate_recheck)
            if request.candidate_recheck is not None
            else None,
            "candidate_data_recheck": asdict(request.candidate_data_recheck)
            if request.candidate_data_recheck is not None
            else None,
            "change": None if change is None else change.model_dump(mode="json"),
            "audit_labels": []
            if change is None
            else [
                v.symbol
                for v in {
                    v.listing_id: v
                    for m in (
                        MarketDataRepository(
                            self.session.workspace
                        ).load_universe_manifest_revision(state.manifest_revision),
                        *(g.manifest for g in grants),
                        *((formation,) if formation is not None else ()),
                    )
                    for v in m.listings
                }.values()
                if v.listing_id in change.full_history_listing_ids
            ],
        }

    def confirm(self, plan_hash: str, *, caller: str) -> CommandAdmission:
        """Require human consent and revalidate an exact current change under the mutation gate.

        A stock-list change is the person's to approve. The full-history audit its listings
        owe is a default the agent takes and discloses (person-stops row 49), so an agent may
        confirm a `FULL_HISTORY_AUDIT` plan, which a stopped change then resumes on.

        Args:
            plan_hash: Exact retained membership/change proposal.
            caller: HUMAN, or any caller for a full-history audit.

        Returns:
            Exact reused or newly admitted change task.

        Raises:
            ValueError: Caller/change scope is invalid, proposal is stale or another task must
                finish or recover.
        """
        plan = self.changes.load_plan(plan_hash)
        change = plan.change
        if caller != "HUMAN" and (change is None or change.action != "FULL_HISTORY_AUDIT"):
            raise ValueError("workspace_data_update.human_confirmation_required")
        if change is None:
            raise ValueError("workspace_data_update.change_proposal_required")
        self._require_plan(plan)
        registry = self.session.task_control_registry
        envelope, goal, workflow = _task_contract(plan)

        def approve() -> CommandAdmission:
            for task in registry.tasks():
                if task.input == envelope and task.lifecycle is not TaskLifecycle.CANCELLED:
                    return CommandAdmission(task_id=task.task_id, lifecycle=task.lifecycle.value)
                if task.lifecycle not in {
                    TaskLifecycle.SUCCEEDED,
                    TaskLifecycle.BLOCKED,
                    TaskLifecycle.CANCELLED,
                }:
                    raise ValueError(
                        f"workspace_data_update.finish_or_recover_existing_task:{task.task_id}"
                    )
            now = self.clock()
            state = read_workspace_inputs(
                self.session.workspace, plan.binding, allow_transition=True
            )
            current = self.changes.proposal(
                state=state,
                readiness=self._readiness(MarketDataRepository(self.session.workspace)),
                observed_at=now,
                obligations=self._obligations(),
                valuation_grants=self.changes.active_grants(
                    self._approved_plans(), self._obligations()
                ),
                target_session=plan.request.target_market_session,
            )
            parent = self._audit_parent(state) if change.action == "FULL_HISTORY_AUDIT" else None
            if (
                state != plan.before
                or current != plan.change
                or (
                    parent is None
                    and plan.request.target_market_session
                    != _latest_common_us_session(on_or_before=now.date(), observed_at=now)
                )
            ):
                raise ValueError("workspace_data_update.proposal_stale")
            if parent is not None:
                # The stopped change keeps its approval, Task and sealed request; its cycle
                # reads this supplement beside the request's audits when it runs again.
                parent_task, parent_plan = parent
                cycle_id = self._cycle_id(parent_plan)
                ids = change.full_history_listing_ids
                self._registry().record_audit_supplement(
                    cycle_id,
                    listing_ids=ids,
                    requirement_receipt=full_history_audit_requirement(
                        self._registry().cycle(cycle_id).request, ids
                    ),
                    audit_plan_hash=plan.content_hash,
                    observed_at=now,
                )
                return CommandAdmission(
                    task_id=parent_task.task_id, lifecycle=parent_task.lifecycle.value
                )
            task = registry.admit(
                input_envelope=envelope, goal=goal, plan=workflow, observed_at=now
            ).record
            return CommandAdmission(task_id=task.task_id, lifecycle=task.lifecycle.value)

        return cast(CommandAdmission, self.session.mutation_gate.run(approve))

    def _receipt(self, plan_hash: str | None = None) -> WorkspaceDataUpdateReceipt | None:
        return DuckDbWorkspaceMaintenanceRegistry.read_data_update_receipt(
            self.session.workspace / "market-data.duckdb", plan_hash
        )

    def receipt_task(self, receipt: WorkspaceDataUpdateReceipt) -> UUID | None:
        """The update Task that sealed a receipt: the one whose plan the receipt names.

        A reuse answers it, so a read follows the update it reused (V449).

        Args:
            receipt: A sealed update receipt.

        Returns:
            The succeeded Task, or None for a receipt no retained Task sealed.
        """
        for task in reversed(self.session.task_control_registry.tasks()):
            if task.task_kind != self.task_kind or task.lifecycle is not TaskLifecycle.SUCCEEDED:
                continue
            plan = task.input.payload.get("plan")
            if isinstance(plan, dict) and plan.get("content_hash") == receipt.plan_hash:
                sealed: UUID = task.task_id
                return sealed
        return None

    def readback(self, task_id: UUID | None = None) -> dict[str, object]:
        """Read current Data state and an exact selected or latest update task record.

        The workspace's data state and, for one update Task, its record: the latest
        update Task by default, or the Task named by `task_id` -- an id that names no Task,
        or a Task of another kind, is a typed refusal that names the latest, never a
        fall-back to it. `selected` says whether a Task was asked for by id.
        """
        binding = self._binding()
        registry = self.session.task_control_registry
        latest = next(
            (value for value in reversed(registry.tasks()) if value.task_kind == self.task_kind),
            None,
        )
        task = latest
        if task_id is not None:
            try:
                task = registry.task(task_id)
            except KeyError:
                task = None
            if task is None or task.task_kind != self.task_kind:
                return {
                    "status": "REFUSED",
                    "failure_code": "workspace_data_update.task_kind_mismatch"
                    if task is not None
                    else "workspace_data_update.task_not_found",
                    "requested_task_id": str(task_id),
                    "task_kind": task.task_kind if task is not None else None,
                    "latest_task_id": str(latest.task_id) if latest else None,
                }
        plan = self._plan_of(task, require_current=False) if task is not None else None
        receipt = self._receipt(plan.content_hash if plan else None)
        retry_after = (
            self.retry_after(plan)
            if task is not None and plan is not None and task.lifecycle is TaskLifecycle.DEFERRED
            else None
        )
        sector_reference = None
        partition_reuse: dict[str, object] | None = None
        try:
            state = read_workspace_inputs(
                self.session.workspace, binding, allow_transition=True
            ).model_dump(mode="json")
            resolver = ArtifactResolver(self.session.workspace / "artifacts")
            summary = self._panel_summary(resolver, state["panel_hash"])
            sector_reference = summary.get("sector_reference")
            if sector_reference is not None and not isinstance(sector_reference, dict):
                raise ValueError("workspace_data_update.sector_reference_disclosure_invalid")
            # Partition reuse is the account of one Panel's composition. With a receipt it
            # is the receipt's resulting Panel -- the selected update's own work, whatever
            # Panel is active today; without one it is the current Panel's, said so.
            if receipt is None:
                counts = _partition_reuse(summary.get("partition_reuse"))
                if counts is not None:
                    partition_reuse = {**counts, "source": "CURRENT_PANEL"}
            else:
                partition_reuse = self._receipt_partition_reuse(resolver, receipt)
            failure = None
        except FileNotFoundError:
            state, failure = None, "workspace_data_update.artifact_missing"
        except ValueError as error:
            state, failure = None, str(error)
        result = {
            "status": "PUBLISHED" if receipt else "NO_UPDATE_PUBLICATION",
            "inputs": state,
            "current_input_failure": failure,
            "receipt": receipt.model_dump(mode="json") if receipt else None,
            "membership": self._membership_readback(binding.market_profile_id),
            "foundation_action": "UNCHANGED_NO_RESEARCH_OR_REVIEW",
            "task_id": str(task.task_id) if task is not None else None,
            "task_lifecycle": task.lifecycle.value if task is not None else None,
            "selected": task_id is not None,
            "latest_task_id": str(latest.task_id) if latest else None,
            "plan_hash": plan.content_hash if plan else None,
            "retry_after_at": retry_after.isoformat() if retry_after is not None else None,
            "next_action": "DATA_UPDATE_PLAN"
            if state is not None
            and state.get("readiness_status")
            in {"MANIFEST_UPDATE_PENDING", "ONBOARDING_IN_PROGRESS"}
            else None,
        }
        if sector_reference is not None and failure is None:
            result["sector_reference"] = sector_reference
        if failure is None and partition_reuse is not None:
            # One Panel's own account of its composition: how many closed-year partitions
            # it kept and how many it recomposed, with which Panel says so. Reuse, not new
            # work; an unavailable historical account is said so, never a later Panel's.
            result["partition_reuse"] = partition_reuse
        if task is not None and plan is not None:
            # The update's own work, for the selected Task: the maintenance cycle this
            # Task's request admitted (phase, status, what changed), the maintenance
            # runner's durable listing units for its target session, and the Task-bound
            # work progress of its Feature and market-data steps. Read-only facts of the
            # existing owners; none of them is a lifecycle or completion verdict.
            result["cycle"] = self._cycle_readback(plan)
            result["maintenance"] = self._maintenance_readback(plan)
            # This selected update's saved formation disposition, not a later
            # Panel or today's membership. Keep it visible if Feature's coverage
            # floors stop the update before a publication exists.
            result["removed_member_tails"] = [
                value.model_dump(mode="json")
                for value in (
                    receipt.removed_member_tails
                    if receipt is not None
                    else self.changes.removed_member_tails(plan)
                )
            ]
            result["work_progress"] = self.telemetry.work_progress(task)
            # What its stopped step saw beside the code, as Task Control keeps it (V444).
            cause = next(
                (
                    item.failure_cause
                    for item in registry.work_items(task.task_id)
                    if item.failure_cause is not None
                ),
                None,
            )
            if cause is not None:
                result["failure_cause"] = cause.model_dump(mode="json")
        return result

    @staticmethod
    def _panel_summary(resolver: ArtifactResolver, panel_hash: str) -> dict[str, object]:
        panel = resolver.load_feature_panel_manifest(
            resolver.feature_panel_manifest_uri(panel_hash)
        )
        summary = panel["safe_summary"]
        if not isinstance(summary, dict):
            raise ValueError("workspace_data_update.panel_summary_invalid")
        return summary

    def _receipt_partition_reuse(
        self, resolver: ArtifactResolver, receipt: WorkspaceDataUpdateReceipt
    ) -> dict[str, object]:
        """The composition account of the Panel the receipt resulted in. A Panel manifest
        no longer present, or one that records no account, is unavailable by name: the
        selected update's own metadata is missing, and no later Panel stands in for it."""
        try:
            counts = _partition_reuse(
                self._panel_summary(resolver, receipt.after.panel_hash).get("partition_reuse")
            )
        except FileNotFoundError:
            return {
                "source": "RECEIPT_PANEL",
                "availability": "UNAVAILABLE",
                "failure_code": "workspace_data_update.receipt_panel_missing",
                "panel_hash": receipt.after.panel_hash,
            }
        if counts is None:
            return {
                "source": "RECEIPT_PANEL",
                "availability": "UNAVAILABLE",
                "failure_code": "workspace_data_update.receipt_panel_reuse_not_recorded",
                "panel_hash": receipt.after.panel_hash,
            }
        return {**counts, "source": "RECEIPT_PANEL", "panel_hash": receipt.after.panel_hash}

    @staticmethod
    def _restated_sessions(change: dict[str, object] | None) -> int:
        """Distinct restated dates of one listing: a raw bar corrected in place or an
        adjusted return that moved, on sessions the update did not add. One date named by
        both corrections is one restated session; a new session's own return is new work."""
        change = change or {}
        new_sessions = change.get("new_sessions")
        corrections = change.get("raw_correction_sessions")
        adjusted = change.get("adjusted_return_change_sessions")
        added = set(new_sessions) if isinstance(new_sessions, list) else set()
        restated = set(corrections) if isinstance(corrections, list) else set()
        if isinstance(adjusted, list):
            restated |= set(adjusted)
        return len(restated - added)

    def _cycle_readback(self, plan: WorkspaceDataUpdatePlan) -> dict[str, object] | None:
        """The maintenance cycle admitted for this plan's request, or None before it exists."""
        try:
            cycle = self._cycle(plan)
        except ValueError:
            return None
        changes = cycle.market_data_change_set
        return {
            "cycle_id": cycle.cycle_id,
            "phase": cycle.phase.value,
            "status": cycle.status.value,
            "updated_at": self.telemetry.instant(cycle.updated_at).isoformat(),
            # The record's age at the product clock, for a page whose own clock is not it.
            "age_seconds": self.telemetry.age(cycle.updated_at),
            "retry_after_at": (
                cycle.retry_after_at.isoformat() if cycle.retry_after_at is not None else None
            ),
            "failure_code": cycle.failure_code,
            "transport_workers": cycle.transport_workers,
            "change_set": None
            if changes is None
            else {
                "change_set_hash": changes.change_set_hash,
                "listings_with_new_sessions": sum(
                    1 for item in changes.listing_changes if item.new_sessions
                ),
                # Listing-sessions: one per listing per new session, never trading days.
                "new_sessions": sum(len(item.new_sessions) for item in changes.listing_changes),
                # A restatement is a raw bar corrected in place, or an adjusted return that
                # moved on a session the update did not add: the new session's own return
                # is new work, not a correction; one date named by both is one session.
                "listings_with_corrections": sum(
                    1 for item in changes.listing_changes if _restated_dates(item)
                ),
                "restated_sessions": sum(
                    len(_restated_dates(item)) for item in changes.listing_changes
                ),
                "membership_additions": len(changes.membership_additions),
                "membership_removals": len(changes.membership_removals),
                "sector_revision_changed": changes.sector_revision_changed,
                "receipts": len(changes.receipt_hashes),
            },
        }

    _MAINTENANCE_ROWS = 96

    def _maintenance_readback(self, plan: WorkspaceDataUpdatePlan) -> dict[str, object] | None:
        """The maintenance runner's durable listing units of the selected Task's own cycle,
        keyed by the run the cycle recorded when it admitted them (the manifest bound for
        that cycle: a transition or a candidate recheck runs under one the request does not
        name, and today's active manifest is another update's scope). A cycle recorded
        before that record existed is resolved from its own request, and only when the run
        found says the same manifest revision. Counts over every unit, and the newest moved
        units (UPDATED or FAILED, by their own update instant, at most `_MAINTENANCE_ROWS`)
        with what each one changed. None before a cycle is admitted or before the runner
        admitted units for it; a scope the store no longer resolves, or units it cannot
        read, are unavailable by name -- never replaced by current data."""
        try:
            cycle = self._cycle(plan)
        except ValueError:
            return None
        request = cycle.request
        market = MarketDataRepository(self.session.workspace)
        scope = self._registry().market_data_scope(cycle.cycle_id)
        if scope is not None:
            maintenance_id = scope["maintenance_id"]
        else:
            try:
                manifest = market.load_universe_manifest_revision(request.membership_revision)
            except ValueError:
                return {
                    "availability": "UNAVAILABLE",
                    "failure_code": "workspace_data_update.maintenance_manifest_unavailable",
                    "membership_revision": request.membership_revision,
                    "as_of_session": request.target_market_session.isoformat(),
                }
            maintenance_id = current_universe_maintenance_id(
                manifest,
                as_of_session=request.target_market_session,
                authorized_full_history_listing_ids=tuple(sorted(request.full_history_listing_ids)),
            )
        try:
            run = market.current_universe_maintenance_run(maintenance_id)
        except ValueError:
            if scope is None:
                return None
            return {
                "availability": "UNAVAILABLE",
                "failure_code": "workspace_data_update.maintenance_run_missing",
                "maintenance_id": maintenance_id,
                "membership_revision": scope["manifest_revision"],
                "as_of_session": scope["as_of_session"],
            }
        if scope is None and run.research_manifest_revision != request.membership_revision:
            return {
                "availability": "UNAVAILABLE",
                "failure_code": "workspace_data_update.maintenance_scope_not_recorded",
                "membership_revision": request.membership_revision,
                "as_of_session": request.target_market_session.isoformat(),
            }
        try:
            listings = market.current_universe_maintenance_listings(maintenance_id)
        except ValueError as error:
            return {
                "availability": "UNREADABLE",
                "failure_code": "workspace_data_update.maintenance_units_unreadable",
                "maintenance_id": maintenance_id,
                "detail": str(error)[:200],
            }
        moved = sorted(
            (item for item in listings if item.state != "PENDING"),
            key=lambda item: (item.updated_at, item.symbol),
        )
        rows = []
        for item in moved[-self._MAINTENANCE_ROWS :]:
            change = item.change_document or {}
            new_sessions = change.get("new_sessions")
            sentinel = change.get("price_action_sentinel")
            added = set(new_sessions) if isinstance(new_sessions, list) else set()
            restated = self._restated_sessions(change)
            rows.append(
                {
                    "listing_id": item.listing_id,
                    "symbol": item.symbol,
                    "state": item.state,
                    "failure_code": item.failure_code,
                    "raw_through": item.raw_through.isoformat() if item.raw_through else None,
                    "new_sessions": sorted(added),
                    "restated_sessions": restated,
                    "audit_scope": change.get("audit_scope"),
                    "sentinel_disposition": (
                        sentinel.get("disposition") if isinstance(sentinel, dict) else None
                    ),
                    "attempt_count": item.attempt_count,
                    "updated_at": self.telemetry.instant(item.updated_at).isoformat(),
                }
            )

        def has_new_sessions(item: CurrentUniverseMaintenanceListing) -> bool:
            new_sessions = (item.change_document or {}).get("new_sessions")
            return isinstance(new_sessions, list) and bool(new_sessions)

        updated = [item for item in listings if item.state == "UPDATED"]
        return {
            "availability": "AVAILABLE",
            "maintenance_id": maintenance_id,
            "scope_source": "CYCLE_RECORD" if scope is not None else "CYCLE_REQUEST",
            "cycle_id": cycle.cycle_id,
            "lifecycle": run.lifecycle,
            "as_of_session": run.as_of_session.isoformat(),
            "manifest_revision": run.research_manifest_revision,
            "age_seconds": self.telemetry.age(
                max((item.updated_at for item in moved), default=run.updated_at)
            ),
            # Every count is over listings (units), by what the runner recorded for each:
            # processed = updated + failed; updated splits into new sessions fetched,
            # history corrected without a new session, and re-verified with nothing new.
            "counts": {
                "listings": len(listings),
                "processed": len(moved),
                "updated": len(updated),
                "failed": sum(item.state == "FAILED" for item in listings),
                "pending": sum(item.state == "PENDING" for item in listings),
                "with_new_sessions": sum(1 for item in updated if has_new_sessions(item)),
                "corrected": sum(
                    1
                    for item in updated
                    if not has_new_sessions(item) and self._restated_sessions(item.change_document)
                ),
                "reverified": sum(
                    1
                    for item in updated
                    if not has_new_sessions(item)
                    and not self._restated_sessions(item.change_document)
                ),
            },
            "rows": rows,
            "retained": len(rows),
            "moved": len(moved),
        }

    def _membership_readback(self, market_profile_id: str) -> dict[str, object] | None:
        """The Universe journal as the page reads it: the boundary, then the changes.

        References the journal rather than copying the membership: the
        bootstrap cohort by its identity and size, and the most recent
        events with their effective session, authority and reference. The
        full lists live once, in the workspace's Universe tables.
        """

        try:
            market = MarketDataRepository(self.session.workspace)
            bootstrap = market.universe_bootstrap(market_profile_id)
            if bootstrap is None:
                return None
            events = market.membership_events(market_profile_id)
            observations = market.universe_source_observations(market_profile_id)
        except (ValueError, OSError, duckdb.Error):
            return None
        return {
            "bootstrap": {
                "t0_session": bootstrap.t0_session.isoformat(),
                "history_start": bootstrap.history_start.isoformat(),
                "cohort_size": len(bootstrap.cohort_listing_ids),
                "cohort_hash": bootstrap.cohort_hash,
                "derivation": bootstrap.derivation,
                "initialization_assumption": bootstrap.initialization_assumption,
                "admitted_at": bootstrap.admitted_at.isoformat(),
                "record_hash": bootstrap.record_hash,
            },
            "journal_sequence": events[-1].sequence if events else 0,
            "latest_effective_session": (
                events[-1].effective_session.isoformat() if events else None
            ),
            "recent_events": [
                {
                    "sequence": item.sequence,
                    "listing_id": item.listing_id,
                    "kind": item.kind,
                    "effective_session": item.effective_session.isoformat(),
                    "observed_at": item.observed_at.isoformat(),
                    "decided_at": item.decided_at.isoformat(),
                    "authority": item.authority,
                    "reference_hash": item.reference_hash,
                    "manifest_revision": item.manifest_revision,
                }
                for item in events[-20:]
            ],
            "recent_source_observations": [
                {
                    "observation_hash": item.observation_hash,
                    "observed_at": item.observed_at.isoformat(),
                    "previous_observed_at": (
                        item.previous_observed_at.isoformat()
                        if item.previous_observed_at is not None
                        else None
                    ),
                    "candidate_membership_hash": item.candidate_membership_hash,
                    "source_identity_hash": item.source_identity_hash,
                    "first_eligible_session": item.first_eligible_session.isoformat(),
                }
                for item in observations
            ],
        }

    def reusable(self, plan: WorkspaceDataUpdatePlan) -> WorkspaceDataUpdateReceipt | None:
        """Require succeeded exact receipt, current inputs and all due/valuation obligations.

        Args:
            plan: Explicit sealed maintenance plan.

        Returns:
            Verified reusable receipt or None when work, valuation coverage or freshness remains
            owed.
        """
        self._require_plan(plan)
        try:
            current = read_workspace_inputs(self.session.workspace, plan.binding)
        except ValueError:
            return None  # An owed partial update is resumed from its Task, not a new base.
        receipt = self._receipt(plan.content_hash) or self._receipt()
        if (
            receipt is None
            or receipt.after != current
            or receipt.target_session != plan.request.target_market_session
            or source_check_due(
                current.sources_checked_at,
                self.clock(),
                reference_session=plan.request.target_market_session,
                failed_at=current.source_check_failed_at,
            )
        ):
            return None
        for task in self.session.task_control_registry.tasks():
            if task.task_kind == self.task_kind and task.lifecycle is TaskLifecycle.SUCCEEDED:
                original = self._plan_of(task, require_current=False)
                if original.content_hash == receipt.plan_hash and original.binding == plan.binding:
                    self._require_plan(original)
                    if (
                        plan.request.candidate_recheck is not None
                        or plan.request.candidate_data_recheck is not None
                    ) and original.request != plan.request:
                        # A due candidate check is work even when current members are fresh.
                        continue
                    if plan.change is not None and original != plan:
                        continue
                    needed = {
                        v.listing_id
                        for m in self.changes.valuation_scopes(plan)
                        for v in m.listings
                    }
                    covered = {
                        v.listing_id
                        for m in self.changes.valuation_scopes(original)
                        for v in m.listings
                    }
                    if not needed <= covered:
                        continue
                    try:
                        self.changes.verified_valuation(
                            original,
                            self.provider
                            or YFinanceMarketDataProvider(
                                self.session.workspace / "yfinance-cache"
                            ),
                        )
                    except ValueError:
                        continue
                    return receipt
        return None

    def prepare(self, plan_hash: str) -> WorkspaceDataUpdatePlan:
        """Reopen an exact maintenance preview, durable task plan or saved change plan.

        Args:
            plan_hash: Exact retained plan identity.

        Returns:
            Exact retained maintenance plan; task recovery remains possible after process loss.

        Raises:
            ValueError: The plan is past its hour or was never planned in this workspace.
        """
        # A maintenance preview runs within its hour whether or not the Host restarted since it
        # was planned, and never past it from this Host's memory alone (V537, as V536).
        if (kept := self._previews.runnable(plan_hash)) is not None:
            return kept
        # An already admitted request remains recoverable after process loss.
        for task in self.session.task_control_registry.tasks():
            if task.task_kind == self.task_kind:
                plan = self._plan_of(task, require_current=False)
                if plan.content_hash == plan_hash:
                    self._require_plan(plan)
                    return plan
        # A change plan is kept until its person confirms it. A plan kept nowhere is one to plan
        # again, never a tampered artifact (V537).
        try:
            return self.changes.load_plan(plan_hash)
        except ContentAddressedStoreError as error:
            if str(error).partition(":")[0] == "content_store.artifact_missing":
                raise ValueError("workspace_data_update.plan_required") from error
            raise

    def admit(self, plan: WorkspaceDataUpdatePlan) -> CommandAdmission:
        """Require admitted source access and exact current maintenance or confirmed change scope.

        Args:
            plan: Explicit sealed maintenance plan.

        Returns:
            Reused/recoverable or newly admitted task identity/lifecycle.

        Raises:
            ValueError: Cleanup, source access, consent, retry time or exact before-state is
                inadmissible.
        """
        require_no_pending_cleanup(self.session.workspace)
        self._require_plan(plan)
        if self.network_work(plan) and not self._source_access_admitted():
            raise ValueError("workspace_data_update.source_access_not_admitted")
        registry = self.session.task_control_registry
        envelope, goal, workflow = _task_contract(plan)
        existing = next(
            (
                task
                for task in reversed(registry.tasks())
                if task.input.input_hash == envelope.input_hash
                and task.lifecycle is not TaskLifecycle.CANCELLED
            ),
            None,
        )
        if (
            existing is None
            and plan.change is not None
            and plan.change.action == "FULL_HISTORY_AUDIT"
        ):
            supplemented = self._supplemented(plan)
            if supplemented is not None:
                task = self.resume_stopped(*supplemented)
                return CommandAdmission(task_id=task.task_id, lifecycle=task.lifecycle.value)
        if plan.change is not None and existing is None:
            raise ValueError("workspace_data_update.human_confirmation_required")
        if existing is not None:
            existing = self.resume_stopped(existing, plan)
            return CommandAdmission(task_id=existing.task_id, lifecycle=existing.lifecycle.value)
        if (
            read_workspace_inputs(
                self.session.workspace, plan.binding, allow_transition=plan.change is not None
            )
            != plan.before
        ):
            raise ValueError("workspace_data_update.stale_plan")
        admitted = registry.admit(
            input_envelope=envelope, goal=goal, plan=workflow, observed_at=self.clock()
        ).record
        return CommandAdmission(task_id=admitted.task_id, lifecycle=admitted.lifecycle.value)

    def _source_access_admitted(self) -> bool:
        """Whether the update may read its provider: one the Host supplies, or the network."""
        return not self._provider_requires_network or network_access(self.session.workspace).allowed

    @staticmethod
    def resumes(code: str | None) -> bool:
        """Whether a rerun of its plan resumes a Task the update's stages stopped with `code`."""
        return code in _RECOVERABLE_BLOCKS or (
            code is not None
            and code.split(":", 1)[0] == "research_update.input_source_access_not_admitted"
        )

    def resume_stopped(self, task: TaskRecord, plan: WorkspaceDataUpdatePlan) -> TaskRecord:
        """The Task a rerun of its plan finds, resumed where its deferral or stop allows.

        The provider's deferral is reopened once its retry time has passed, and refused before
        it (`workspace_data_update.retry_not_due`) (V601). The update's stop a person clears -- a
        data decision taken, the network allowed -- is reopened as owed a recovery, and refused by
        its code while the network it names stays closed (V600). Either runs on from the stage it
        stopped in, the stop kept in the Task's own record; any other Task is answered as it
        stands.

        Args:
            task: The Task the plan's rerun found: this update's, or one whose stage ran it.
            plan: The data update plan that Task ran.

        Returns:
            The Task, reopened or as it stands.

        Raises:
            ValueError: `workspace_data_update.retry_not_due` before a deferral's retry time, or
                `workspace_data_update.source_access_not_admitted` while the network a stop
                names stays closed.
        """
        if task.lifecycle is TaskLifecycle.DEFERRED:
            retry_after = self.retry_after(plan)
            if retry_after is not None and self.clock() < retry_after:
                raise ValueError("workspace_data_update.retry_not_due")
            return self.session.task_control_registry.mark_recovery_required(
                task_id=task.task_id,
                failure_code=task.failure_code or "DATA_UPDATE_RETRY_DUE",
                observed_at=self.clock(),
            )
        if task.lifecycle is not TaskLifecycle.BLOCKED:
            return task
        if not self.resumes(task.failure_code):
            if plan.change is None:
                return task
            state = read_workspace_inputs(
                self.session.workspace, plan.binding, allow_transition=True
            )
            partial = self._partial_transition(task, plan, state)
            if state.panel_manifest_revision is not None and not partial:
                raise ValueError("workspace_data_update.transition_not_verified")
            if not partial:
                return task
        if (
            task.failure_code is not None
            and task.failure_code.split(":", 1)[0]
            in {
                "workspace_data_update.source_access_not_admitted",
                "research_update.input_source_access_not_admitted",
            }
            and not self._source_access_admitted()
        ):
            raise ValueError(task.failure_code)
        return self.session.task_control_registry.mark_recovery_required(
            task_id=task.task_id,
            failure_code=task.failure_code,
            observed_at=self.clock(),
            allow_blocked=True,
            expected_task_hash=task.record_hash,
        )

    def execute(self, task_id: UUID, *, expected_task_hash: str | None = None) -> None:
        """Execute one exact admitted workspace data update task.

        Args:
            task_id: Exact retained task.
            expected_task_hash: Optional optimistic task identity.
        """
        record = self.session.task_control_registry.task(task_id)
        self._plan_of(record)
        self.session.execute_admitted(record, self, self.clock, expected_task_hash)

    def _require_plan(self, plan: WorkspaceDataUpdatePlan) -> None:
        if (
            plan.workspace_id != self.manifest.workspace_id
            or plan.workspace_manifest_hash != manifest_fields_hash(self.manifest, PLAN_FIELDS)
            or plan.binding != self._binding()
        ):
            raise ValueError("workspace_data_update.task_binding_mismatch")
        approved = (
            {p.content_hash: p for p in self._approved_plans()} if plan.valuation_grants else {}
        )
        for grant in plan.valuation_grants:
            parent = approved.get(grant.approved_plan_hash)
            if (
                parent is None
                or parent.change is None
                or parent.workspace_id != plan.workspace_id
                or parent.workspace_manifest_hash != plan.workspace_manifest_hash
                or grant.manifest != parent.change.valuation_manifest
                or not set(grant.book_roots) <= {r for r, _ in parent.change.book_heads}
                or not set(grant.listing_ids) <= {v.listing_id for v in grant.manifest.listings}
            ):
                raise ValueError("workspace_data_update.valuation_grant_invalid")
        if plan.request.full_history_listing_ids:
            manifest = MarketDataRepository(self.session.workspace).load_universe_manifest_revision(
                plan.before.manifest_revision
            )
            formation = self.changes.formation_scope(plan.before, plan.change)
            allowed = {
                v.listing_id
                for m in (
                    manifest,
                    *self.changes.valuation_scopes(plan),
                    *((formation,) if formation is not None else ()),
                )
                for v in m.listings
            }
            if not set(plan.request.full_history_listing_ids) <= allowed:
                raise ValueError("workspace_data_update.audit_scope_invalid")

    def task_subject(self, task: TaskRecord) -> dict[str, object]:
        """Read this owner's admitted date, independent of the current workspace head."""
        try:
            plan = self._plan_of(task, require_current=False)
        except (KeyError, ValueError) as error:
            return {
                "subject_refusal": {
                    "status": "REFUSED",
                    "failure_code": public_failure(
                        error, "workspace_data_update.task_binding_mismatch"
                    ),
                }
            }
        return {"subject_context": {"date": plan.request.target_market_session.isoformat()}}

    def _plan_of(
        self, task: TaskRecord, *, require_current: bool = True
    ) -> WorkspaceDataUpdatePlan:
        if task.task_kind != self.task_kind:
            raise ValueError("workspace_data_update.task_kind_mismatch")
        plan = WorkspaceDataUpdatePlan.model_validate(task.input.payload["plan"])
        expected, goal, workflow = _task_contract(plan)
        if task.input != expected or task.goal != goal or task.plan != workflow:
            raise ValueError("workspace_data_update.task_binding_mismatch")
        if require_current:
            self._require_plan(plan)
        return cast(WorkspaceDataUpdatePlan, plan)

    def _registry(self) -> DuckDbWorkspaceMaintenanceRegistry:
        return DuckDbWorkspaceMaintenanceRegistry(
            self.session.workspace / "market-data.duckdb", gate=self.session.mutation_gate
        )

    def _accepts(
        self,
        plan: WorkspaceDataUpdatePlan,
        transition: ManifestTransitionRecord,
        active: str | None,
    ) -> tuple[bool, str | None]:
        """Whether recovery accepts the active membership, and an admission to adopt; reads only.

        It accepts the journal's next membership or the one this plan's own cycle recorded
        making active: a stopped change owns the child it derived (a Sector exclusion, a gateway
        quarantine) until its Panel publishes.

        A change stopped part-way by 0.1.3 or earlier kept no record. Its active membership is
        accepted only where the gateway's own admission chain derives it from the journal's next
        membership at the change's session; `_step` records that admission as the cycle's
        working membership, `adopted:<admission_hash>`, once it has verified the transition.
        Nothing else is inferred. Retirement: remove the adoption in 0.2, once no supported
        workspace predates the record.
        """
        following = transition.next_manifest_revision
        if active == following:
            return True, None
        recorded = self._registry().working_manifest(self._cycle_id(plan))
        if recorded is not None or active is None or following is None:
            return active == recorded, None
        market = MarketDataRepository(self.session.workspace)
        admission = PanelStateRepository(market.database, market_data=market).deriving_admission(
            candidate_revision=following,
            result_revision=active,
            as_of_session=plan.request.target_market_session,
        )
        return admission is not None, admission

    def _cancelled(self, plan: WorkspaceDataUpdatePlan) -> StageExecutionResult:
        """The safe checkpoint honoured the Task's cancel; say so on the cycle too.

        The Task lifecycle stays the authority for the cancellation (the runner
        applies it from this result); the maintenance cycle the Task was running
        records the same fact, so no readback presents it as active work while
        the listing progress it made stays with its maintenance operation for
        the next plan. A cycle that never started, or already completed, is
        left as it is.
        """

        try:
            cycle = self._cycle(plan)
        except ValueError:
            return StageExecutionResult(StageDisposition.CANCELLED)
        if cycle.status not in {
            MaintenanceStatus.COMPLETED,
            MaintenanceStatus.NOOP,
            MaintenanceStatus.CANCELLED,
        }:
            self._registry().cancel(cycle.cycle_id, observed_at=self.clock())
        return StageExecutionResult(StageDisposition.CANCELLED)

    def _audit_parent(
        self, state: WorkspaceInputStatus
    ) -> tuple[TaskRecord, WorkspaceDataUpdatePlan] | None:
        """The stopped membership change whose cycle stopped for an audit it may not run."""
        tasks = reversed(self.session.task_control_registry.tasks())
        return next(
            (
                (task, plan)
                for task, plan in self._stopped_changes(state, tasks)
                if task.failure_code == "data.full_history_audit_approval_required"
            ),
            None,
        )

    def _stopped_changes(
        self, state: WorkspaceInputStatus, tasks: Iterable[TaskRecord]
    ) -> Iterator[tuple[TaskRecord, WorkspaceDataUpdatePlan]]:
        """Each stopped membership change among these Tasks that still owes its Panel."""
        for task in tasks:
            if task.task_kind == self.task_kind and task.lifecycle is TaskLifecycle.BLOCKED:
                plan = self._plan_of(task, require_current=False)
                if self._partial_transition(task, plan, state):
                    yield task, plan

    def _supplemented(
        self, audit: WorkspaceDataUpdatePlan
    ) -> tuple[TaskRecord, WorkspaceDataUpdatePlan] | None:
        """The membership change an audit plan's confirmation was admitted onto, if any."""
        registry = self._registry()
        for task in reversed(self.session.task_control_registry.tasks()):
            if task.task_kind != self.task_kind or task.lifecycle in {
                TaskLifecycle.SUCCEEDED,
                TaskLifecycle.CANCELLED,
            }:
                continue
            plan = self._plan_of(task, require_current=False)
            if (
                plan.change is not None
                and plan.change.action == "UNIVERSE"
                and audit.content_hash in registry.admitted_audit_plans(self._cycle_id(plan))
            ):
                return task, plan
        return None

    def _cycle_id(self, plan: WorkspaceDataUpdatePlan) -> str:
        execution = self.changes.execution(plan) if plan.change is not None else None
        request = plan.request if execution is None else execution.request
        return str(canonical_hash(["workspace-maintenance", request.request_hash]))

    def _cycle(self, plan: WorkspaceDataUpdatePlan):  # type: ignore[no-untyped-def]
        return self._registry().cycle(self._cycle_id(plan))

    def retry_after(self, plan: WorkspaceDataUpdatePlan) -> datetime | None:
        """When a deferred update of this plan may run again, as its cycle records it.

        Args:
            plan: The data update plan a deferred Task ran, its own or a research update's.

        Returns:
            The provider's retry time, or None where nothing names one.
        """
        market = MarketDataRepository(self.session.workspace)
        state = market.readiness.load(plan.binding.market_profile_id)
        if (
            plan.change is not None
            and plan.change.action == "UNIVERSE"
            and state is not None
            and state.status == "ONBOARDING_IN_PROGRESS"
        ):
            onboarding = self._readiness(market).resume_onboarding()
            deferred = market.current_hydration_deferred(onboarding.runner.onboarding_id)
            return None if deferred is None else deferred.retry_after_at
        execution = self.changes.execution(plan) if plan.change is not None else None
        return DuckDbWorkspaceMaintenanceRegistry.read_update_retry_after(
            market.database.path,
            plan.request.request_hash if execution is None else execution.request.request_hash,
        )

    def compatibility(self, task: TaskRecord) -> TaskExecutionCompatibility:
        """Bind exact maintenance schema, workflow, Data policy and execution identity.

        Args:
            task: Exact retained maintenance task.

        Returns:
            Deterministic execution compatibility declaration.
        """
        plan = self._plan_of(task)
        return TaskExecutionCompatibility.create(
            task_contract_hash=canonical_hash(schema_structure(WorkspaceDataUpdatePlan)),
            workflow_definition_hash=task.plan.workflow_definition_hash,
            input_schema_id=task.input.input_schema_id,
            domain_policy_hash=plan.binding.content_hash,
            framework_identity_hash=self.session.execution_identity(
                canonical_hash({"workflow": _STAGES})
            ),
        )

    def _completion(self, plan: WorkspaceDataUpdatePlan):  # type: ignore[no-untyped-def]
        market = MarketDataRepository(self.session.workspace)
        snapshot = PanelStateRepository(
            market.database, market_data=market
        ).feature_panel_snapshot_for_active(plan.binding.market_profile_id)
        if snapshot is None:
            raise ValueError(_MEMBERSHIP_REVIEW)
        if plan.change is None and snapshot["manifest_revision"] != plan.before.manifest_revision:
            # A revision that admits the same listings is governance, not a
            # membership change; the plan stops for review only when the
            # members the Panel admits at its as-of session differ from the
            # ones the plan was made against.
            resolver = ArtifactResolver(self.session.workspace / "artifacts")
            panel = resolver.load_feature_panel_manifest(str(snapshot["manifest_uri"]))
            admitted = panel_as_of_listing_identity(panel)[0]
            if plan.before.listing_set_hash is None or admitted != plan.before.listing_set_hash:
                raise ValueError(_MEMBERSHIP_REVIEW)
        return capture_verified_maintenance_completion(
            cycle=self._cycle(plan), panel_snapshot=snapshot
        )

    def execute_stage(
        self, *, task: TaskRecord, execution: TaskExecution, work_item: WorkItemDefinition
    ) -> StageExecutionResult:
        """Execute exact maintenance work with task-bound cancellation and telemetry.

        Args:
            task: Exact admitted maintenance task.
            execution: Current execution declaration.
            work_item: Exact installed workflow stage.

        Returns:
            Deterministic step result for the task's sealed maintenance plan.
        """
        return self.execute_step(
            self._plan_of(task),
            work_item.stage_id,
            cancelled=lambda: (
                self.session.task_control_registry.task(task.task_id).lifecycle
                is TaskLifecycle.CANCEL_REQUESTED
            ),
            bound_to=(task, execution),
        )

    def execute_step(
        self,
        plan: WorkspaceDataUpdatePlan,
        stage: str,
        *,
        cancelled: Callable[[], bool],
        bound_to: tuple[TaskRecord, TaskExecution] | None = None,
    ) -> StageExecutionResult:
        """Domain maintenance shared by standalone and captured advancement Tasks.

        `bound_to` names the Task and execution running this step: their work progress
        is then kept beside the Task as well as in the workspace projection. A captured
        advancement Task passes nothing and keeps the projection alone, as before.
        """
        self._require_plan(plan)
        if stage == _STAGES[0]:
            return self._step(plan, stage, cancelled=cancelled, bound_to=bound_to)
        # The data stage and the receipt's publication each keep one writable instance of the
        # market store from their first read to their last write, so the cycles' holds, the
        # runner's units and every other thread's reads attach to it instead of reopening the
        # file with a cold cache and checkpointing it at each close. Writable, it is no lock:
        # every thread attaches without waiting, and each cycle still releases the write gate
        # at its network edges.
        with WorkspaceDatabase(self.session.workspace).retain(read_only=False):
            return self._step(plan, stage, cancelled=cancelled, bound_to=bound_to)

    def _step(
        self,
        plan: WorkspaceDataUpdatePlan,
        stage: str,
        *,
        cancelled: Callable[[], bool],
        bound_to: tuple[TaskRecord, TaskExecution] | None,
    ) -> StageExecutionResult:
        if stage == _STAGES[0]:
            # The last update's backup, if the Host never took it or it failed, is taken first.
            if settle_automatic_backup(
                self.session.workspace, clock=self.clock, gate=self.session.mutation_gate
            ):
                return StageExecutionResult(
                    StageDisposition.BLOCKED, failure_code="workspace_data_update.archive_failed"
                )
            if (
                read_workspace_inputs(
                    self.session.workspace, plan.binding, allow_transition=plan.change is not None
                )
                != plan.before
            ):
                raise ValueError("workspace_data_update.stale_plan")
            content = plan.content_hash
        elif stage == _STAGES[1]:
            if self.network_work(plan) and not self._source_access_admitted():
                return StageExecutionResult(
                    StageDisposition.BLOCKED,
                    failure_code="workspace_data_update.source_access_not_admitted",
                )
            market = MarketDataRepository(self.session.workspace)
            readiness = market.readiness.load(plan.binding.market_profile_id)
            assert readiness is not None and readiness.active_manifest_id is not None
            manifest = market.load_universe_manifest(readiness.active_manifest_id)
            provider = self.provider or YFinanceMarketDataProvider(
                self.session.workspace / "yfinance-cache"
            )
            if plan.change is not None and plan.change.action == "UNIVERSE":
                gate = self._readiness(market)
                state = gate.assess(observed_at=self.clock())
                if state.status.value == "MANIFEST_UPDATE_PENDING":
                    if (
                        state.proposal is None
                        or state.proposal.transition_id != plan.change.transition_id
                    ):
                        raise ValueError("workspace_data_update.transition_changed")
                    row = market.readiness.load(plan.binding.market_profile_id)
                    if (
                        row is None
                        or canonical_hash(row.pending_candidate_manifest_document)
                        != plan.change.candidate_document_hash
                    ):
                        raise ValueError("workspace_data_update.transition_changed")
                    task = next(
                        t
                        for t in self.session.task_control_registry.tasks()
                        if t.input == _task_contract(plan)[0]
                    )
                    onboarding = gate.start_approved_onboarding(
                        WorkspaceReadinessConsent(
                            consent_id=uuid5(NAMESPACE_URL, plan.content_hash),
                            market_profile_id=plan.binding.market_profile_id,
                            action=WorkspaceConsentAction.REFRESH_MANIFEST,
                            approved_at=task.admitted_at,
                        )
                    )
                elif state.status.value == "ONBOARDING_IN_PROGRESS":
                    onboarding = gate.resume_onboarding()
                else:
                    onboarding = None
                if onboarding is not None:
                    # The new members' hydration advances one chunk per run
                    # like first use does; the cancel is honoured at the
                    # chunk boundary and every listing commits on its own.
                    while True:
                        if cancelled():
                            return self._cancelled(plan)
                        outcome = onboarding.runner.run(
                            observed_at=self.clock(),
                            work_budget=onboarding.runner.hydration_chunk_size,
                        )
                        if outcome.status is not CurrentUniverseOnboardingStatus.RUNNING:
                            break
                    if outcome.status is not CurrentUniverseOnboardingStatus.COMPLETED:
                        return StageExecutionResult(
                            StageDisposition.DEFERRED
                            if outcome.status is CurrentUniverseOnboardingStatus.DEFERRED
                            else StageDisposition.BLOCKED,
                            failure_code=outcome.failure_code
                            or "workspace_data_update.onboarding_incomplete",
                        )
                    gate.complete_onboarding(onboarding, outcome, observed_at=self.clock())
                transition = market.manifest_transition(str(plan.change.transition_id))
                readiness = market.readiness.load(plan.binding.market_profile_id)
                assert readiness is not None and readiness.active_manifest_id is not None
                accepted, adoption = self._accepts(
                    plan, transition, readiness.active_manifest_revision
                )
                if (
                    transition.activated_at is None
                    or transition.prior_manifest_revision != plan.before.manifest_revision
                    or not accepted
                ):
                    raise ValueError("workspace_data_update.transition_not_verified")
                if adoption is not None:
                    self._registry().record_working_manifest(
                        self._cycle_id(plan),
                        manifest_revision=readiness.active_manifest_revision,
                        authority=f"adopted:{adoption}",
                        observed_at=self.clock(),
                    )
                manifest = market.load_universe_manifest(readiness.active_manifest_id)
            execution = (
                self.changes.capture_execution(plan, manifest.revision_sha256)
                if plan.change is not None
                else None
            )
            request = plan.request if execution is None else execution.request
            resolver = ArtifactResolver(self.session.workspace / "artifacts")
            index = resolver.find_feature_panel_semantic_index(
                panel_snapshot_hash=plan.before.panel_hash
            )
            if index is None:
                raise ValueError("workspace_data_update.panel_semantic_index_missing")
            require_storage_capacity(self.session.workspace, additional_bytes=0)
            formation_failure = self.changes.maintain_formation(
                plan=plan, provider=provider, observed_at=self.clock(), cancelled=cancelled
            )
            if (
                formation_failure is not None
                and formation_failure[0] == "workspace_data_update.formation_cancelled"
            ):
                return self._cancelled(plan)
            if formation_failure is not None:
                return StageExecutionResult(
                    StageDisposition.BLOCKED,
                    failure_code=formation_failure[0],
                    failure_cause=StageFailureCause.from_facts(formation_failure[1]),
                )
            runtime = WorkspaceRuntime.create(
                workspace=self.session.workspace,
                manifest=manifest,
                provider=provider,
                artifact_root=self.session.workspace / "artifacts",
                max_live_symbols=1000,
                feature_catalog=workspace_feature_catalog(self.session.workspace),
                application_controls=self.session,
            )
            try:
                if bound_to is not None:
                    # The Feature owner and the maintenance runner report through this one
                    # sink (the coordinator hands it to the runner it builds).
                    runtime.feature_foundation.progress_sink = self.telemetry.bound_progress_sink(
                        *bound_to, stage, runtime.progress_publisher
                    )
                gate = self._readiness(runtime.market_data)
                # A catalog an activation made opens its own closure before its first build.
                runtime.prepare_feature_closure()
                coordinator = runtime.maintenance_coordinator(
                    readiness_gate=gate,
                    clock=self.clock,
                    recorded_data_confirmation=self.recorded_data_confirmation,
                )
                while True:
                    if cancelled():
                        return self._cancelled(plan)
                    require_storage_capacity(
                        self.session.workspace,
                        additional_bytes=0,
                    )
                    outcome = coordinator.run(
                        request,
                        observed_at=self.clock(),
                        maintenance_work_budget=_MAINTENANCE_CYCLE_LISTINGS,
                    )
                    if outcome.status is not MaintenanceStatus.RUNNING:
                        break
                if outcome.status not in {MaintenanceStatus.COMPLETED, MaintenanceStatus.NOOP}:
                    return StageExecutionResult(
                        StageDisposition.DEFERRED
                        if outcome.status is MaintenanceStatus.DEFERRED
                        else StageDisposition.BLOCKED,
                        failure_code=outcome.failure_code
                        or "workspace_data_update.review_required",
                        failure_cause=StageFailureCause.from_facts(outcome.failure_cause),
                    )
                try:
                    content = self._completion(plan).completion_hash
                except ValueError as error:
                    if str(error) != _MEMBERSHIP_REVIEW:
                        raise
                    # The cycle activated a quality-derived child membership
                    # this plan did not declare; the update is applied and the
                    # Task stops here by name for the user to plan against the
                    # new membership, instead of surfacing as an interruption.
                    return StageExecutionResult(
                        StageDisposition.BLOCKED,
                        failure_code=public_failure(error, "workspace_data_update.blocked"),
                    )
                try:
                    self.changes.maintain_valuation(
                        plan=plan, provider=provider, observed_at=self.clock(), cancelled=cancelled
                    )
                except ValueError as error:
                    if str(error) == "workspace_data_update.valuation_cancelled":
                        return self._cancelled(plan)
                    if str(error) not in {
                        "data.full_history_audit_approval_required",
                        "workspace_data_update.held_valuation_unavailable",
                    }:
                        raise
                    return StageExecutionResult(
                        StageDisposition.BLOCKED,
                        failure_code=public_failure(error, "workspace_data_update.blocked"),
                    )
            finally:
                runtime.close()
        elif stage == _STAGES[2]:
            receipt = self._receipt(plan.content_hash)
            if receipt is None:
                try:
                    self._completion(plan)
                except ValueError as error:
                    if str(error) != _MEMBERSHIP_REVIEW:
                        raise
                    return StageExecutionResult(
                        StageDisposition.BLOCKED,
                        failure_code=public_failure(error, "workspace_data_update.blocked"),
                    )
                after = read_workspace_inputs(self.session.workspace, plan.binding)
                if after.foundation_hash != plan.before.foundation_hash:
                    raise ValueError("workspace_data_update.foundation_changed")
                cycle = self._cycle(plan)
                receipt = WorkspaceDataUpdateReceipt.seal(
                    plan_hash=plan.content_hash,
                    maintenance_request_hash=cycle.request.request_hash,
                    cycle_id=cycle.cycle_id,
                    target_session=plan.request.target_market_session,
                    completed_at=self.clock(),
                    before=plan.before,
                    after=after,
                    child_task_refs=cycle.child_task_refs,
                    effect_receipts=cycle.effect_receipts,
                    transition_receipt_hash=None
                    if plan.change is None
                    else self.changes.capture_execution(plan, after.manifest_revision).content_hash,
                    valuation_receipt_hashes=self.changes.verified_valuation(
                        plan,
                        self.provider
                        or YFinanceMarketDataProvider(self.session.workspace / "yfinance-cache"),
                    ),
                    removed_member_tails=self.changes.removed_member_tails(plan),
                )
                self._registry().publish_data_update_receipt(receipt)
                # The backup is archive work: the Host takes it once idle.
                defer_automatic_backup(self.session.workspace, at=self.clock())
            content = receipt.content_hash
        else:
            raise ValueError("workspace_data_update.stage_unknown")
        return StageExecutionResult(
            StageDisposition.READY,
            evidence=(
                TaskEvidence(
                    evidence_kind="workspace_data_update.stage",
                    reference=f"playpen://workspace-data-update/{stage}/{content}",
                    content_hash=content,
                ),
            ),
        )

    def verify_stage(
        self,
        *,
        task: TaskRecord,
        execution: TaskExecution,
        work_item: WorkItemDefinition,
        evidence: tuple[TaskEvidence, ...],
    ) -> tuple[TaskEvidence, ...]:
        """Verify one exact task-bound maintenance stage through its owning verifier.

        Args:
            task: Exact retained task.
            execution: Current execution declaration.
            work_item: Exact installed stage.
            evidence: Declared stage evidence.

        Returns:
            Unchanged verified evidence.
        """
        return self.verify_step(self._plan_of(task), work_item.stage_id, evidence)

    def verify_step(
        self,
        plan: WorkspaceDataUpdatePlan,
        stage: str,
        evidence: tuple[TaskEvidence, ...],
        *,
        require_current: bool = True,
    ) -> tuple[TaskEvidence, ...]:
        """Require exact plan, completed valuation or maintenance receipt evidence.

        Args:
            plan: Exact retained maintenance plan.
            stage: Installed stage identifier.
            evidence: Exact declared stage evidence tuple.
            require_current: Whether to require installed plan bindings during verification.

        Returns:
            Unchanged evidence after stage-specific content and valuation checks.

        Raises:
            ValueError: Plan/stage, completion, receipt or exact evidence is invalid.
        """
        if require_current:
            self._require_plan(plan)
        if stage == _STAGES[0]:
            content = plan.content_hash
        elif stage == _STAGES[1]:
            content = self._completion(plan).completion_hash
            self.changes.verified_valuation(
                plan,
                self.provider
                or YFinanceMarketDataProvider(self.session.workspace / "yfinance-cache"),
            )
        elif stage == _STAGES[2]:
            receipt = self._receipt(plan.content_hash)
            if receipt is None:
                raise ValueError("workspace_data_update.receipt_absent")
            content = receipt.content_hash
        else:
            raise ValueError("workspace_data_update.stage_unknown")
        expected = (
            TaskEvidence(
                evidence_kind="workspace_data_update.stage",
                reference=f"playpen://workspace-data-update/{stage}/{content}",
                content_hash=content,
            ),
        )
        if evidence != expected:
            raise ValueError("workspace_data_update.evidence_invalid")
        return evidence


@dataclass
class WorkspaceDataUpdateCommand:
    """Dispatch one explicit admitted workspace maintenance plan."""

    application: WorkspaceDataUpdateApplication
    plan: WorkspaceDataUpdatePlan | None = None
    command_kind: str = DATA_UPDATE_TASK_KIND

    def admit(self) -> CommandAdmission:
        """Require an exact maintenance plan before deterministic task admission.

        Returns:
            Task admission from the maintenance owner.

        Raises:
            ValueError: Exact plan is absent or its declared admission checks refuse.
        """
        if self.plan is None:
            raise ValueError("workspace_data_update.plan_required")
        return self.application.admit(self.plan)

    def execute(self, task_id: UUID, *, expected_task_hash: str | None = None) -> None:
        """Execute one exact task admitted for workspace maintenance.

        Args:
            task_id: Exact retained task.
            expected_task_hash: Optional optimistic task identity.
        """
        self.application.execute(task_id, expected_task_hash=expected_task_hash)
