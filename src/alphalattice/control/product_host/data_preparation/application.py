"""First-use preparation through existing Data/Feature owners and Task Control."""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Annotated, Any, Literal, Self
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, ValidationError, model_validator

from alphalattice.control.data_platform.contracts import DataRemediationExecutionReceipt
from alphalattice.control.data_platform.delegation import DataIssueDelegation
from alphalattice.control.data_platform.maintenance.contracts import (
    MaintenanceStatus,
    MaintenanceTrigger,
    WorkspaceDataUpdateBinding,
    WorkspaceInputStatus,
    WorkspaceMaintenanceRequest,
    workspace_maintenance_data_policy_hash,
)
from alphalattice.control.data_platform.readiness import (
    SourceLoader,
    WorkspaceConsentAction,
    WorkspaceReadinessConsent,
    WorkspaceReadinessGate,
    _latest_common_us_session,
    build_workspace_readiness,
)
from alphalattice.control.data_platform.task_telemetry import (
    TELEMETRY_REPLACE_DELAYS,
    TaskTelemetry,
)
from alphalattice.control.observation_runtime.telemetry.progress import (
    WorkProgressUpdate,
    WorkspaceProgressPublisher,
)
from alphalattice.control.product_host.composition.application_session import (
    WorkspaceApplicationSession,
)
from alphalattice.control.product_host.composition.research_workspace import (
    ResearchWorkspaceManifest,
    read_research_workspace_manifest,
    update_research_workspace_manifest,
)
from alphalattice.control.product_host.composition.workspace import WorkspaceRuntime
from alphalattice.control.product_host.data_preparation.host import (
    PreFactorWorkspaceHost,
    ProductWorkspacePaths,
)
from alphalattice.control.product_host.data_preparation.remediation import (
    WorkspaceDataIssueApplication,
    superseded_preparations,
)
from alphalattice.control.product_host.maintenance.data_update import (
    PROFILE,
    installed_data_update_binding,
    read_workspace_inputs,
)
from alphalattice.control.product_host.research_authoring.factor_inputs import (
    factor_template,
    factor_workflow,
    normalize_factor_document,
    publish_prepared_factor_inputs,
    read_factor_bundle,
)
from alphalattice.control.product_host.research_authoring.feature_activations import (
    workspace_feature_catalog,
)
from alphalattice.control.product_host.storage.contracts import (
    StorageBudgetResolution,
    storage_input_estimate,
)
from alphalattice.control.product_host.storage.inventory import require_storage_capacity
from alphalattice.control.product_host.storage.plan_previews import (
    PLAN_PREVIEWS_DIRECTORY,
    PreviewRegistry,
)
from alphalattice.control.task_control.child import ChildStartFailed
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
from alphalattice.control.task_control.runner import (
    StageDisposition,
    StageExecutionResult,
)
from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.control.workspace_runtime.network_access import network_access
from alphalattice.foundation.feature_engine.inputs.gateway import FeatureInputPolicy
from alphalattice.foundation.feature_engine.storage.repositories import PanelStateRepository
from alphalattice.foundation.market_data_ops.runtime.universe_onboarding import (
    CurrentUniverseOnboardingOutcome,
    CurrentUniverseOnboardingStatus,
    DataTargetSessionLag,
    ListingUnitObservation,
    listing_outcome_category,
)
from alphalattice.foundation.market_data_ops.sources.providers import (
    MarketDataProvider,
    YFinanceMarketDataProvider,
)
from alphalattice.foundation.market_data_ops.sources.universe import (
    bootstrap_from_candidate_manifest_document,
    candidate_manifest_document,
)
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from alphalattice.interface.local_application.dispatcher import CommandAdmission
from alphalattice.kernel.data.universe import FROZEN_UNIVERSE_SOURCES
from alphalattice.kernel.shared_kernel.domain.errors import WorkspaceConflictError
from alphalattice.kernel.shared_kernel.identity import canonical_hash, schema_structure
from alphalattice.kernel.shared_kernel.identity_successors import is_current
from alphalattice.kernel.shared_kernel.project_layout import resolve_playpen_root
from alphalattice.kernel.shared_kernel.sealing import seal_model
from alphalattice.kernel.shared_kernel.source_identity import source_rule_closure_hash

TASK_KIND = "workspace_preparation"
STAGES = ("freeze_sources", "prepare_data", "prepare_features", "publish_inputs", "verify_inputs")

ListingUnitState = Literal[
    "PENDING",
    "RAW_READY",
    "QUALITY_ELIGIBLE",
    "QUALITY_INELIGIBLE",
    "FEATURE_READY",
    "RAW_FAILED",
    "AUDIT_FAILED",
]


def research_next(manifest: ResearchWorkspaceManifest) -> tuple[str, dict[str, dict[str, str]]]:
    """A prepared workspace's next research step, as its first use takes it (FLOW-3).

    With no research strategy installed, `strategy controls` names the strategy's required
    Alpha and Risk studies, the shortest way to a book that runs forward; once one is
    installed, its book's controls. A study on the prepared input stays offered beside it
    for exploration.
    """
    if manifest.strategy_installation == "NON_DEFAULT_RESEARCH":
        return "CONTROLS", {"strategy_book": {"operation": "CONTROLS"}}
    return "RESEARCH_STRATEGY_CONTROLS", {
        "strategy_controls": {"operation": "RESEARCH_STRATEGY_CONTROLS"}
    }


class ListingActivityRow(BaseModel):  # type: ignore[misc]
    """Retain one announced listing-unit transition and its distinct observation clocks.

    One listing-unit transition as the onboarding runner announced it: the symbol,
    the exact listing, the durable state it took, where its bars came from when that is
    known, and two instants that are not the same thing -- `observed_at` is the runner's
    run instant (the `observed_at` its whole run, and every unit of it, is recorded
    under; not a per-listing completion clock) and `noted_at` is the preparation owner's
    clock when the announcement arrived, right after the durable write. Never a payload.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    listing_id: str = Field(min_length=1, max_length=160)
    symbol: str = Field(min_length=1, max_length=40)
    state: ListingUnitState
    observed_at: AwareDatetime
    noted_at: AwareDatetime
    origin: Literal["ACQUIRED", "RETAINED", "LOCAL"] | None = None
    raw_through: date | None = None
    failure_code: str | None = Field(default=None, max_length=120)
    reasons: tuple[str, ...] = ()
    tail_acquired: bool = False


class ListingActivityCounts(BaseModel):  # type: ignore[misc]
    """Retain runner-derived listing-unit counts at the latest snapshot boundary.

    The runner's own unit counts at the last chunk boundary, advanced by every transition
    announced since: the same quantities as the chunk record (raw availability, quality,
    Feature admission and failures kept apart), at the snapshot's boundary rather than the
    chunk's. Derived from the runner's retained progress and every announcement, never from
    the bounded rows below; reconciled to the runner's count at each chunk boundary.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidates: int = Field(ge=0)
    raw_ready: int = Field(ge=0)
    quality_eligible: int = Field(ge=0)
    feature_ready: int = Field(ge=0)
    failed: int = Field(ge=0)
    exclusion_counts: dict[str, Annotated[int, Field(ge=0)]] | None = None


class ListingActivitySnapshot(BaseModel):  # type: ignore[misc]
    """Retain a bounded atomic activity sidecar for one exact task execution.

    The Task-bound sidecar of recent listing activity: a bounded snapshot of one
    execution's latest unit transitions and its counts at that boundary, replaced
    atomically, never an event history. `observed` counts every transition of the
    execution, `retained` the rows kept, `dropped` the difference (disclosed, not
    reconstructed).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(min_length=1)
    input_hash: str = Field(min_length=1)
    execution_id: str = Field(min_length=1)
    stage: str
    sequence: int = Field(ge=1)
    observed: int = Field(ge=0)
    retained: int = Field(ge=0)
    dropped: int = Field(ge=0)
    written_at: AwareDatetime
    counts: ListingActivityCounts
    rows: tuple[ListingActivityRow, ...]
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_bounds(self) -> ListingActivitySnapshot:
        """Require an installed activity stage and exact observed/retained/dropped counts.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Stage is unknown or counts do not describe retained rows.
        """
        if self.stage not in STAGES:
            raise ValueError("listing activity names an unknown stage")
        if self.retained != len(self.rows) or self.observed != self.retained + self.dropped:
            raise ValueError("listing activity counts do not describe its rows")
        return self


_LISTING_BINDING_KEYS = frozenset({"content_hash", "task_id", "input_hash"})


class _ListingActivityDelivery:
    """Optional, bounded and coalesced: the recent listing-unit transitions of one stage
    execution and their counts, written beside the Task's stage records as one atomically
    replaced snapshot.

    Bounds: at most `RETAIN` rows are kept; one delivery attempt per `COALESCE_UNITS`
    observations or `COALESCE_SECONDS` of the product clock -- whether the last attempt
    succeeded or not -- plus one at each chunk boundary (`flush`); each attempt waits at
    most the telemetry replace budget (`TELEMETRY_REPLACE_DELAYS`, about 30 ms). The
    observer is synchronous in the runner's thread, so those bounds are the cost the
    runner pays: a few milliseconds per unit on average, never a wait per unit and never
    a stop, a retry or a change of the work. A failed attempt is counted with its typed
    cause; the retained rows and counts wait for the next attempt, which carries them all.

    Counts start from the runner's retained progress (the store's units when the stage
    entered) and advance with every announcement by the affected listing's own transition
    -- from the state the run loaded it in, or the state this delivery last heard for it,
    to the announced state -- under the runner's state semantics; the chunk boundary
    reconciles them to the runner's outcome, which the durable chunk record also carries."""

    RETAIN = 96
    COALESCE_UNITS = 5
    COALESCE_SECONDS = 1.0

    def __init__(
        self,
        owner: WorkspacePreparationApplication,
        task: TaskRecord,
        binding: dict[str, str],
        path: Path,
        seed: CurrentUniverseOnboardingOutcome,
    ) -> None:
        self.owner, self.task_key, self.binding, self.path = owner, str(task.task_id), binding, path
        self.delivery = owner.telemetry.delivery(task.task_id, "listing-activity")
        self.rows: deque[dict[str, Any]] = deque(maxlen=self.RETAIN)
        self.counts = self._counts_of(seed)
        self.states: dict[str, str] = {}  # the last state announced per listing, this execution
        self.categories: dict[str, str | None] = {}
        self.observed = 0
        self.sequence = 0
        self.pending = 0
        self.dirty = False
        now = owner.clock()
        self.attempted_at = now
        self.written_at = now

    @staticmethod
    def _counts_of(outcome: CurrentUniverseOnboardingOutcome) -> dict[str, Any]:
        return {
            "candidates": outcome.candidates,
            "raw_ready": outcome.raw_ready,
            "quality_eligible": outcome.quality_eligible,
            "feature_ready": outcome.feature_ready,
            "failed": outcome.failed,
            "exclusion_counts": dict(outcome.exclusion_counts) or None,
        }

    _QUALITY = frozenset({"QUALITY_ELIGIBLE", "FEATURE_READY"})
    _FAILED = frozenset({"RAW_FAILED", "QUALITY_INELIGIBLE", "AUDIT_FAILED"})

    def _count(self, observation: ListingUnitObservation) -> None:
        # The runner counts a unit as raw-ready once it holds bars (a later failure keeps
        # them), as quality-eligible while it is QUALITY_ELIGIBLE or FEATURE_READY, and as
        # failed in RAW_FAILED, QUALITY_INELIGIBLE or AUDIT_FAILED. Each announcement moves
        # the counts by what this listing left and what it entered: a retained listing
        # whose audit fails while it is still PENDING adds a failure and revokes nothing;
        # one that fails after its quality admission gives that admission back.
        prior = self.states.get(observation.listing_id, observation.run_start_state)
        state = observation.state
        self.states[observation.listing_id] = state
        counts = self.counts
        if state == "RAW_READY" and prior == "PENDING":
            counts["raw_ready"] += 1
        counts["quality_eligible"] += int(state in self._QUALITY) - int(prior in self._QUALITY)
        counts["feature_ready"] += int(state == "FEATURE_READY") - int(prior == "FEATURE_READY")
        counts["failed"] += int(state in self._FAILED) - int(prior in self._FAILED)
        category = listing_outcome_category(state, observation.reasons, observation.failure_code)
        previous = self.categories.get(observation.listing_id)
        if (exclusions := counts["exclusion_counts"]) is not None:
            for kind, change in ((previous, -1), (category, 1)):
                if kind is not None and kind.lower() in exclusions:
                    exclusions[kind.lower()] += change
        self.categories[observation.listing_id] = category

    def observe(self, observation: ListingUnitObservation) -> None:
        now = self.owner.clock()
        self.rows.append(
            ListingActivityRow(
                listing_id=observation.listing_id,
                symbol=observation.symbol,
                state=observation.state,
                observed_at=observation.observed_at,
                noted_at=now,
                origin=observation.origin,
                raw_through=observation.raw_through,
                failure_code=observation.failure_code,
                reasons=observation.reasons,
                tail_acquired=observation.tail_acquired,
            ).model_dump(mode="json")
        )
        self._count(observation)
        self.observed += 1
        self.pending += 1
        self.dirty = True
        if (
            self.pending >= self.COALESCE_UNITS
            or (now - self.attempted_at).total_seconds() >= self.COALESCE_SECONDS
        ):
            self._attempt(now)

    def flush(self, outcome: CurrentUniverseOnboardingOutcome | None = None) -> None:
        """The chunk boundary: the runner's own counts replace the replayed ones, and one
        attempt carries everything retained since the last successful delivery."""

        if outcome is not None:
            counts = self._counts_of(outcome)
            if counts != self.counts:
                self.counts = counts
                self.dirty = True
        if self.dirty:
            self._attempt(self.owner.clock())

    def _attempt(self, now: datetime) -> None:
        # The coalescing budget restarts at every attempt, refused or not: a refusal costs
        # one bounded wait per budget, never one per unit.
        self.pending = 0
        self.attempted_at = now
        data: dict[str, Any] = {
            **self.binding,
            "sequence": self.sequence + 1,
            "observed": self.observed,
            "retained": len(self.rows),
            "dropped": self.observed - len(self.rows),
            "written_at": now.isoformat(),
            "counts": dict(self.counts),
            "rows": list(self.rows),
        }
        data["content_hash"] = canonical_hash(data)
        self.delivery["attempts"] += 1
        try:
            self.owner.telemetry.replace_json(self.path, data, delays=TELEMETRY_REPLACE_DELAYS)
        except (OSError, ValueError, RuntimeError) as error:
            self.owner.telemetry.delivery_failed(self.delivery, "LISTING_ACTIVITY", error)
            return
        self.sequence += 1
        self.dirty = False
        self.written_at = now
        self.delivery["delivered"] += 1
        self.delivery["last_delivered_at"] = now.isoformat()
        self.owner.telemetry.last_valid[(self.task_key, "listing-activity")] = data


IMPLEMENTATION_ROLE = "product_host.workspace_preparation"
"""The role a move of this implementation is recorded under in `config/identity-successors.json`,
so a plan sealed under its predecessor stays current (binding plan R1, LAWS.md ID1)."""


def _implementation() -> str:
    root = resolve_playpen_root(Path(__file__))
    return source_rule_closure_hash(
        root=root,
        semantic_owner="product_host",
        numerical_role="WORKSPACE_PREPARATION",
        tracked_paths=(
            "src/alphalattice/control/product_host/data_preparation/application.py",
            "src/alphalattice/control/data_platform/readiness.py",
            "src/alphalattice/control/data_platform/candidate_requalification.py",
            "src/alphalattice/control/product_host/composition/workspace.py",
            "src/alphalattice/control/product_host/research_authoring/factor_inputs.py",
            "src/alphalattice/foundation/feature_engine/runtime/closure_genesis.py",
            "src/alphalattice/kernel/shared_kernel/persistence.py",
        ),
    )


class WorkspacePreparationPlan(BaseModel):  # type: ignore[misc]
    """Seal workspace preparation scope, target session and optional exact predecessor inputs."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    workspace_manifest: ResearchWorkspaceManifest
    binding: WorkspaceDataUpdateBinding
    target_session: date
    planned_at: datetime
    implementation_hash: str
    plan_hash: str
    existing_inputs: WorkspaceInputStatus | None = Field(
        default=None, exclude_if=lambda v: v is None
    )
    predecessor_task_id: UUID | None = Field(default=None, exclude_if=lambda v: v is None)

    @classmethod
    def create(cls, **values: Any) -> Self:
        """Seal an explicit workspace preparation plan.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical plan_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        return seal_model(cls, values, field="plan_hash")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require exact plan identity, aware planning time and qualified reused inputs.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Plan identity/time or reused-input readiness/target differs.
        """
        if (
            self.plan_hash != canonical_hash(self.model_dump(mode="json", exclude={"plan_hash"}))
            or self.planned_at.tzinfo is None
            or (
                self.existing_inputs is not None
                and (
                    self.existing_inputs.readiness_status != "RESEARCH_READY"
                    or self.target_session != self.existing_inputs.panel_through
                )
            )
        ):
            raise ValueError("workspace_preparation.plan_invalid")
        return self


class WorkspacePreparationApplication:
    """Own explicitly confirmed workspace preparation and its task-bound retained checkpoints."""

    task_kind = TASK_KIND
    replans = (
        TaskReplan(
            task_kind=TASK_KIND,
            preview="WORKSPACE_PREPARE_PLAN",
            admitting="WORKSPACE_PREPARE_CONFIRM",
        ),
    )
    """The re-plan of the Task kind this owner admits, which the recovery view offers
    (V188)."""

    def __init__(
        self,
        session: WorkspaceApplicationSession,
        *,
        clock: Callable[[], datetime],
        provider: MarketDataProvider | None = None,
        source_loader: SourceLoader | None = None,
        recorded_data_confirmation: Callable[[DataRemediationExecutionReceipt], bool] | None = None,
    ):
        """Wire retained preparation authority, optional sources and task telemetry.

        Args:
            session: Retained workspace writer/task session.
            clock: Explicit observed-time source.
            provider: Optional admitted market data provider.
            source_loader: Optional admitted source capture loader.
            recorded_data_confirmation: Owner proof of a retained first-use data decision.
        """
        self.session, self.clock = session, clock
        self.provider, self.source_loader = provider, source_loader
        self.recorded_data_confirmation = recorded_data_confirmation
        self.last_plan: WorkspacePreparationPlan | None = None
        # The newest plan, sealed on disk: a confirm after a restart finds it, and a newer
        # plan removes it, one preparation running at a time (V525).
        self._previews: PreviewRegistry[WorkspacePreparationPlan] = PreviewRegistry(
            model=WorkspacePreparationPlan,
            clock=clock,
            capacity=1,
            root=session.workspace / "runtime" / PLAN_PREVIEWS_DIRECTORY / "workspace-preparation",
            single=True,
        )
        self.data_issue_delegation = DataIssueDelegation(session.workspace)
        # The Task-bound telemetry sidecars (work progress, listing activity) beside this
        # owner's stage records; shared with the data update owner.
        self.telemetry = TaskTelemetry(
            session.workspace, root="runtime/preparation", stages=STAGES, clock=clock
        )

    def _gate(self) -> WorkspaceReadinessGate:
        return build_workspace_readiness(
            MarketDataRepository(self.session.workspace),
            profile_path=PROFILE,
            provider=self.provider,
            source_loader=self.source_loader,
        )

    def _sources_allowed(self) -> None:
        if not network_access(self.session.workspace).allowed and (
            self.provider is None
            or isinstance(self.provider, YFinanceMarketDataProvider)
            or self.source_loader is None
        ):
            raise ValueError("workspace_preparation.source_access_not_admitted")

    def tasks(self) -> tuple[TaskRecord, ...]:
        """Read only retained workspace preparation task declarations.

        Returns:
            Ordered preparation tasks from Task Control.
        """
        return tuple(
            t for t in self.session.task_control_registry.tasks() if t.task_kind == TASK_KIND
        )

    def has_tasks(self) -> bool:
        """Whether a preparation Task was ever admitted: one count, no record read."""
        return bool(self.session.task_control_registry.has_tasks(TASK_KIND))

    def task_subject(self, task: TaskRecord) -> dict[str, object]:
        """Read the admitted date from this owner's canonical Task, without delegation IO."""
        from alphalattice.control.product_host.composition.plain_refusals import refused

        try:
            if task.task_kind != TASK_KIND:
                raise ValueError("workspace_preparation.task_not_admitted")
            plan = WorkspacePreparationPlan.model_validate(task.input.payload["plan"])
        except (KeyError, ValidationError):
            return {"subject_refusal": refused("workspace_preparation.task_not_admitted")}
        return {"subject_context": {"date": plan.target_session.isoformat()}}

    def readback(self, task_id: UUID | None = None) -> dict[str, Any]:
        """Read the selected preparation task and its exact retained verified input state.

        The preparation as one Task tells it: the selected Task when `task_id` names
        one, otherwise the latest preparation Task (the discovery a bare entry makes). A
        selected Task that does not exist, or is not a preparation Task, is a typed
        refusal, never a fall-back to the latest. The `inputs` are the selected Task's own
        published and verified input as the workspace still lists it; without a Task they
        are the workspace's inputs.
        """
        manifest = read_research_workspace_manifest(self.session.workspace)
        tasks = self.tasks()
        latest = tasks[-1] if tasks else None
        if task_id is not None:
            selected = next((v for v in tasks if v.task_id == task_id), None)
            if selected is None:
                try:
                    other = self.session.task_control_registry.task(task_id)
                except KeyError:
                    other = None
                return {
                    "status": "REFUSED",
                    "failure_code": "workspace_preparation.task_kind_mismatch"
                    if other is not None
                    else "workspace_preparation.task_not_found",
                    "requested_task_id": str(task_id),
                    "task_kind": other.task_kind if other is not None else None,
                    "latest_task_id": str(latest.task_id) if latest else None,
                }
            latest = selected
        workspace_inputs = [v.model_dump(mode="json") for v in manifest.experiment_inputs or ()]
        published = self._load(latest.task_id, STAGES[4], optional=True) if latest else None
        inputs = (
            [v for v in workspace_inputs if v["binding_hash"] == published["binding_hash"]]
            if published is not None
            else []
            if latest
            else workspace_inputs
        )
        local_data = (self.session.workspace / "market-data.duckdb").is_file()
        successor = next(
            (
                task
                for task in tasks
                if latest is not None
                and task.input.payload.get("source_task_id") == str(latest.task_id)
            ),
            None,
        )
        replan = bool(
            latest
            and successor is None
            and latest.lifecycle is TaskLifecycle.BLOCKED
            and not is_current(
                IMPLEMENTATION_ROLE,
                self._stored_plan(latest).implementation_hash,
                _implementation(),
            )
        )
        truth_review_pending = bool(
            latest
            and successor is None
            and latest.lifecycle is TaskLifecycle.BLOCKED
            and latest.failure_code == "data.truth_review_required"
            and not WorkspaceDataIssueApplication(
                self.session, self.clock
            ).current_decisions_ready()
        )
        return {
            "status": latest.lifecycle.value
            if latest
            else "RESEARCH_INPUTS_READY"
            if inputs
            else "LOCAL_DATA_PRESENT"
            if local_data
            else "INITIALIZATION_REQUIRED",
            "source_mode": "PREVIEW_EXISTING_DATA_REUSE"
            if local_data
            else "ACQUIRE_DECLARED_SOURCES_AFTER_CONFIRMATION",
            "next_action": "WORKSPACE_PREPARE_READBACK"
            if successor is not None
            else "DATA_ISSUES"
            if truth_review_pending
            else "WORKSPACE_PREPARE_PLAN"
            if replan or (not latest and not inputs)
            else research_next(manifest)[0]
            if inputs
            else None,
            "next_requests": {
                "successor": {
                    "operation": "WORKSPACE_PREPARE_READBACK",
                    "task_id": str(successor.task_id),
                }
            }
            if successor is not None
            else {
                **({"issues": {"operation": "DATA_ISSUES"}} if truth_review_pending else {}),
                **(
                    {
                        "confirm": {
                            "operation": "WORKSPACE_PREPARE_CONFIRM",
                            "preparation_plan_hash": latest.input.payload["plan"]["plan_hash"],
                        }
                    }
                    if latest is not None
                    and latest.lifecycle is TaskLifecycle.BLOCKED
                    and not replan
                    and not truth_review_pending
                    else {}
                ),
                **(research_next(manifest)[1] if inputs else {}),
                **{
                    "research_controls:" + item["binding_hash"]: {
                        "operation": "EXPERIMENT_CONTROLS",
                        "research_input_id": item["input_id"],
                        "input_binding_hash": item["binding_hash"],
                    }
                    for item in inputs
                },
            },
            "superseded_by_task_id": str(successor.task_id) if successor else None,
            "superseded_task_ids": sorted(
                {
                    str(task.input.payload["source_task_id"])
                    for task in tasks
                    if task.lifecycle is TaskLifecycle.SUCCEEDED
                    and task.input.payload.get("source_task_id") is not None
                    and str(self._stored_plan(task).predecessor_task_id)
                    == task.input.payload["source_task_id"]
                }
            ),
            "confirmation_available": False
            if replan or successor is not None or truth_review_pending
            else None,
            "execution_binding_changed": replan,
            "task_id": str(latest.task_id) if latest else None,
            "selected": task_id is not None,
            "latest_task_id": str(tasks[-1].task_id) if tasks else None,
            "plan_hash": latest.input.payload["plan"]["plan_hash"] if latest else None,
            "failure_code": latest.failure_code if latest else None,
            "inputs": inputs,
            "published_binding_hash": published["binding_hash"] if published else None,
            "profile": "us-current-index-research",
            "sources": [
                "yfinance",
                *(
                    f"{index.value}: {source.uri}"
                    for index, source in FROZEN_UNIVERSE_SOURCES.items()
                ),
            ],
            "limits": [
                "CURRENT_UNIVERSE_RESEARCH_ONLY",
                "NO_DATA_API_KEY",
                "NO_FOUNDATION_OR_STRATEGY_ACTIVATION",
            ],
            **self.activity_readback(latest),
        }

    def activity_readback(self, task: TaskRecord | None) -> dict[str, Any]:
        """Read task-bound work once, keeping its sample and actual work clocks separate."""
        work = self.telemetry.work_progress(task) if task else None
        listings = self._listing_activity(task) if task else None
        instants = []
        if work and work["availability"] == "BOUND":
            instants.append(self.telemetry.instant(work["updated_at"]))
        if listings and listings["availability"] == "BOUND":
            instants.extend(self.telemetry.instant(row["noted_at"]) for row in listings["rows"])
        return {
            "progress": self._load(task.task_id, "progress", optional=True) if task else None,
            "work_progress": work,
            "listing_activity": listings,
            "activity_timing": {
                "stage": task.active_work_item_id,
                "sampled_at": self.clock().isoformat(),
                "last_work_at": max(instants).isoformat() if instants else None,
            }
            if task
            else None,
        }

    def plan(
        self,
        *,
        network_delegation: str | None = None,
        recovery_task_id: UUID | None = None,
    ) -> dict[str, Any]:
        """Preview preparation or exact checkpoint recovery without acquiring new sources.

        Qualified local inputs are reused without download. Unfinished tasks retain their authority;
        a blocked or cancelled predecessor is resumed only within its exact scope.

        Args:
            network_delegation: Active first-use authority validated by the Host.
            recovery_task_id: The exact stopped Task whose offered continuation is being read.

        Returns:
            Existing preparation state or explicit confirmation preview with source-access refusal,
            exact predecessor and legal next requests.

        Raises:
            ValueError: Existing local data lacks an explicit qualified binding.
        """
        recovery = (
            self.session.task_control_registry.task(recovery_task_id)
            if recovery_task_id is not None
            else None
        )
        if recovery is not None:
            self._stored_plan(recovery)
            if recovery.lifecycle not in {TaskLifecycle.BLOCKED, TaskLifecycle.CANCELLED}:
                raise ValueError("workspace_preparation.resume_scope_changed")
        manifest = read_research_workspace_manifest(self.session.workspace)
        if manifest.experiment_inputs:
            return {
                **self.readback(),
                "status": "ALREADY_PREPARED",
                "next_action": research_next(manifest)[0],
            }
        tasks = self.tasks()
        superseded = superseded_preparations(self.session.task_control_registry, tasks)
        if recovery_task_id is not None and str(recovery_task_id) in superseded:
            raise ValueError("workspace_preparation.resume_scope_changed")
        if recovery is not None:
            tasks = (recovery,)
        unfinished = [
            t
            for t in tasks
            if str(t.task_id) not in superseded
            and t.lifecycle not in {TaskLifecycle.SUCCEEDED, TaskLifecycle.CANCELLED}
        ]
        resume = next(
            (
                t.task_id
                for t in reversed(tasks)
                if t.lifecycle is TaskLifecycle.CANCELLED
                and self._load(t.task_id, STAGES[0], optional=True) is not None
            ),
            None,
        )
        if unfinished:
            previous = unfinished[-1]
            same_binding = self._stored_plan(previous).implementation_hash == _implementation()
            if (
                previous.lifecycle is not TaskLifecycle.BLOCKED
                or self._load(previous.task_id, STAGES[0], optional=True) is None
                or (
                    same_binding
                    and (
                        recovery_task_id is None
                        or previous.failure_code == "data.truth_review_required"
                    )
                    and not (
                        previous.failure_code == "data.truth_review_required"
                        and WorkspaceDataIssueApplication(
                            self.session, self.clock
                        ).current_decisions_ready()
                    )
                )
            ):
                return (
                    {
                        **self.readback(previous.task_id),
                        "predecessor_task_id": str(previous.task_id),
                    }
                    if recovery is not None
                    else self.readback()
                )
            resume = previous.task_id
        existing_inputs = None
        predecessor_plan = None
        if resume is not None:
            predecessor_plan = self._stored_plan(self.session.task_control_registry.task(resume))
            existing_inputs = predecessor_plan.existing_inputs
        elif (self.session.workspace / "market-data.duckdb").exists():
            if manifest.data_update is None:
                raise ValueError("workspace_preparation.existing_data_requires_explicit_binding")
            existing_inputs = read_workspace_inputs(self.session.workspace, manifest.data_update)
            if existing_inputs.readiness_status != "RESEARCH_READY":
                raise ValueError("workspace_preparation.qualified_local_inputs_required")
        now = self.clock()
        candidate_plan = WorkspacePreparationPlan.create(
            workspace_manifest=manifest,
            binding=installed_data_update_binding(self.session.workspace),
            target_session=predecessor_plan.target_session
            if predecessor_plan is not None
            else existing_inputs.panel_through
            if existing_inputs is not None
            else _latest_common_us_session(on_or_before=now.date(), observed_at=now),
            planned_at=now,
            implementation_hash=_implementation(),
            existing_inputs=existing_inputs,
            predecessor_task_id=resume,
        )
        prior = self.last_plan
        plan = (
            prior
            if prior is not None
            and all(
                getattr(prior, field) == getattr(candidate_plan, field)
                for field in (
                    "workspace_manifest",
                    "binding",
                    "target_session",
                    "implementation_hash",
                    "existing_inputs",
                    "predecessor_task_id",
                )
            )
            else candidate_plan
        )
        # One assignment publishes the newest plan for the next plan's comparison; the
        # answer is built from this plan and resume alone (V534).
        self.last_plan = plan
        captured_source = None
        recovery_work = None
        candidate_count = None
        if resume is not None:
            previous, captured_source = self._predecessor(plan)
            candidate_count = len(
                bootstrap_from_candidate_manifest_document(
                    captured_source["candidate"]
                ).candidate_symbols
            )
            retained = [
                item.stage_id
                for item in self.session.task_control_registry.work_items(previous.task_id)
                if item.lifecycle.value == "VERIFIED"
                and self._load(previous.task_id, item.stage_id, optional=True) is not None
            ]
            reused = [stage for stage in retained if stage == STAGES[0]]
            recovery_work = {
                "retained_verified_stages": retained,
                "reused_stages": reused,
                "remaining_stages": [stage for stage in STAGES if stage not in reused],
                "local_units": "REVALIDATE_AND_REUSE_COMPLETED_UNITS",
                "sources_may_be_accessed": ["yfinance"],
                "network_requests": "UNKNOWN_UNTIL_EXECUTION",
                "detail": (
                    "Captured sources are reused; completed local units are revalidated and "
                    "reused; missing units may access yfinance again. Remaining time is unknown."
                ),
            }
        try:
            if plan.existing_inputs is None:
                self._sources_allowed()
            access_failure = None
        except ValueError as error:
            access_failure = str(error)
        self._previews.remember(plan)
        readback = self.readback()
        # One answer, one instruction: admitted sources (a Host-supplied provider included) say
        # nothing about the network, whose own way on (a restart without the offline switch)
        # would contradict the offered confirmation; only a refusal carries it.
        refusal = (
            {}
            if access_failure is None
            else {
                "network_access": network_access(self.session.workspace).body(
                    for_refusal=True, delegation=network_delegation
                ),
                "workspace_path": str(self.session.workspace),
            }
        )
        return {
            "status": "CONFIRMATION_REQUIRED",
            "confirmation_available": access_failure is None,
            "source_access_failure": access_failure,
            "source_access": {
                "status": "ADMITTED" if access_failure is None else "NETWORK_CONTROL_REQUIRED",
                **refusal,
                "authorization_required": access_failure is not None,
                "verified_source_checkpoint_retained": captured_source is not None,
                "network_requests": "UNKNOWN_UNTIL_EXECUTION",
            },
            "plan_hash": plan.plan_hash,
            "resume_from_cancelled_task": str(resume)
            if resume
            and self.session.task_control_registry.task(resume).lifecycle is TaskLifecycle.CANCELLED
            else None,
            "predecessor_task_id": str(resume) if resume else None,
            "recovery_work": recovery_work,
            "target_session": str(plan.target_session),
            "initial_history_years": 10 if plan.existing_inputs is None else None,
            "source_mode": "REUSE_CAPTURED_SOURCES_REVALIDATE_LOCAL_STATE"
            if resume
            else "ACQUIRE_APPROVED_SOURCES"
            if plan.existing_inputs is None
            else "REUSE_QUALIFIED_LOCAL_DATA_NO_DOWNLOAD",
            "candidate_count": candidate_count,
            "candidate_count_basis": "VERIFIED_SOURCE_CHECKPOINT"
            if captured_source is not None
            else "KNOWN_AFTER_SOURCE_CAPTURE",
            "universe": "current S&P 500 union NASDAQ-100 union DJIA",
            "quality": {"maximum_missing_ratio": 0.02, "maximum_consecutive_missing_sessions": 20},
            "sources": [
                "yfinance",
                *(source["source_uri"] for source in captured_source["candidate"]["sources"]),
            ]
            if captured_source is not None
            else readback["sources"],
            "next_action": "WORKSPACE_PREPARE_CONFIRM"
            if access_failure is None
            else "ADMIT_SOURCE_ACCESS_THEN_PREVIEW_AGAIN",
            "next_requests": (
                {
                    "confirm": {
                        "operation": "WORKSPACE_PREPARE_CONFIRM",
                        "preparation_plan_hash": plan.plan_hash,
                    },
                    **{
                        "confirm_with_grant:" + grant["grant_hash"]: {
                            "operation": "WORKSPACE_PREPARE_CONFIRM",
                            "preparation_plan_hash": plan.plan_hash,
                            "data_issue_grant_hash": grant["grant_hash"],
                        }
                        for grant in self.data_issue_delegation.active_grants(now=now)
                        if str(plan.predecessor_task_id) == grant["preparation_task_id"]
                    },
                }
                if access_failure is None
                else {}
            ),
            "limits": readback["limits"],
        }

    def delegated_resume_allowed(self, plan_hash: str, grant_hash: str | None) -> bool:
        """Verify an exact unexpired delegation, recorded decision and preparation predecessor.

        Args:
            plan_hash: Exact requested predecessor or successor plan.
            grant_hash: Optional exact data issue delegation grant.

        Returns:
            Whether the granted external-automation decision authorizes this exact retained resume
            scope.
        """
        if grant_hash is None:
            return False
        grant = self.data_issue_delegation.read(grant_hash, now=self.clock())
        tasks = self.tasks()
        predecessor = next(
            (task for task in tasks if str(task.task_id) == grant["preparation_task_id"]),
            None,
        )
        if predecessor is None or not self.data_issue_delegation.matches_resume(
            grant,
            task_id=str(predecessor.task_id),
            plan_hash=predecessor.input.payload["plan"]["plan_hash"],
            lifecycle=predecessor.lifecycle.value,
            failure_code=predecessor.failure_code,
        ):
            return False
        market = MarketDataRepository(self.session.workspace)
        panel = PanelStateRepository(market.database, market_data=market)
        resolution = panel.feature_input_resolution(grant["case_token"])
        receipt = resolution.get("receipt") if isinstance(resolution, dict) else None
        submission = receipt.get("actor_submission") if isinstance(receipt, dict) else None
        decision = receipt.get("policy_decision") if isinstance(receipt, dict) else None
        if (
            not isinstance(submission, dict)
            or submission.get("actor_kind") != "EXTERNAL_AUTOMATION"
            or submission.get("actor_id") != self.data_issue_delegation.actor_id(grant_hash)
            or not isinstance(decision, dict)
            or any(
                decision.get(field) != grant[source]
                for field, source in (
                    ("case_token", "case_token"),
                    ("evidence_hash", "evidence_hash"),
                    ("option_id", "option_id"),
                    ("option_hash", "option_hash"),
                )
            )
        ):
            return False
        if plan_hash == predecessor.input.payload["plan"]["plan_hash"]:
            return True
        # A grant-bound successor can be repeated or recovered after a Host
        # restart. Its admitted envelope, rather than this process's last
        # preview, proves the exact predecessor and grant scope.
        admitted = next(
            (
                task
                for task in tasks
                if task.input.payload["plan"]["plan_hash"] == plan_hash
                and task.input.payload.get("source_task_id") == str(predecessor.task_id)
                and task.input.payload.get("data_issue_grant_hash") == grant_hash
                and task.input.payload.get("caller") == "EXTERNAL_AUTOMATION"
            ),
            None,
        )
        if admitted is not None:
            self._of(admitted)
            return True
        plan = self._planned(plan_hash)
        if plan is None or plan.predecessor_task_id != predecessor.task_id:
            return False
        self._predecessor(plan)
        self._require(plan)
        return True

    def require_confirmation_caller(
        self, plan_hash: str, *, caller: str, grant_hash: str | None
    ) -> None:
        """Require human confirmation or an exact granted external-automation resume.

        Args:
            plan_hash: Exact preparation plan.
            caller: Declared operation caller.
            grant_hash: Optional exact delegation grant. An external resume of a Task admitted
                under one, naming none, runs under that Task's own grant, checked again (V484).

        Raises:
            ValueError: A human supplies delegation or a non-human lacks the exact admitted resume
                grant.
        """
        if caller == "HUMAN" and grant_hash is not None:
            raise ValueError("workspace_preparation.human_confirmation_does_not_use_delegation")
        if caller == "EXTERNAL_AUTOMATION" and grant_hash is None:
            # The plan's own Task keeps the grant a person admitted it under: its resume needs no
            # second copy, and the validator still checks that grant's term and scope. A new
            # admission has no such Task and names its grant (V484).
            grant_hash = next(
                (
                    str(task.input.payload["data_issue_grant_hash"])
                    for task in self.tasks()
                    if task.input.payload["plan"]["plan_hash"] == plan_hash
                    and task.lifecycle is not TaskLifecycle.CANCELLED
                    and task.input.payload.get("caller") == "EXTERNAL_AUTOMATION"
                    and task.input.payload.get("data_issue_grant_hash")
                ),
                None,
            )
        if caller != "HUMAN" and not (
            caller == "EXTERNAL_AUTOMATION" and self.delegated_resume_allowed(plan_hash, grant_hash)
        ):
            raise ValueError("workspace_preparation.human_confirmation_required")

    def confirm(
        self, plan_hash: str, *, caller: str, grant_hash: str | None = None
    ) -> CommandAdmission:
        """Confirm exact preparation under the shared revocation/admission mutation gate.

        Args:
            plan_hash: Exact current preview or retained task plan.
            caller: Declared human or granted external-automation caller.
            grant_hash: Optional exact resume delegation.

        Returns:
            Exact reused/recoverable or newly admitted task identity/lifecycle.

        Raises:
            ValueError: Caller, retry time, preview/source/predecessor scope is invalid or another
                task must finish or recover.
        """
        with self.session.mutation_gate.hold():
            # Revocation and admission share the mutation gate: a check made before
            # acquiring it cannot authorize work after a concurrent revocation.
            self.require_confirmation_caller(plan_hash, caller=caller, grant_hash=grant_hash)
            for task in self.tasks():
                if (
                    task.input.payload["plan"]["plan_hash"] == plan_hash
                    and task.lifecycle is not TaskLifecycle.CANCELLED
                ):
                    if task.lifecycle in {TaskLifecycle.DEFERRED, TaskLifecycle.BLOCKED}:
                        progress = self._load(task.task_id, "progress", optional=True) or {}
                        retry = progress.get("retry_after_at")
                        if retry and self.clock() < datetime.fromisoformat(retry):
                            raise ValueError("workspace_preparation.retry_not_due")
                        retry_plan = self._of(task)
                        if retry_plan.existing_inputs is None:
                            self._sources_allowed()
                        task = self.session.task_control_registry.mark_recovery_required(
                            task_id=task.task_id,
                            failure_code=task.failure_code or "workspace_preparation.retry_due",
                            observed_at=self.clock(),
                            allow_blocked=task.lifecycle is TaskLifecycle.BLOCKED,
                        )
                    return CommandAdmission(task_id=task.task_id, lifecycle=task.lifecycle.value)
            plan = self._planned(plan_hash)
            if plan is None:
                raise ValueError("workspace_preparation.preview_required")
            resume = plan.predecessor_task_id
            if plan.existing_inputs is None:
                self._sources_allowed()
            self._require(plan)
            for t in self.session.task_control_registry.tasks():
                if (
                    t.lifecycle not in {TaskLifecycle.SUCCEEDED, TaskLifecycle.CANCELLED}
                    and t.task_id != plan.predecessor_task_id
                    and str(t.task_id)
                    not in {v.input.payload.get("source_task_id") for v in self.tasks()}
                ):
                    raise ValueError(
                        f"workspace_preparation.finish_or_recover_existing_task:{t.task_id}"
                    )
            approved_at = self.clock()
            if (
                resume is None
                and plan.existing_inputs is None
                and plan.target_session
                != _latest_common_us_session(
                    on_or_before=approved_at.date(), observed_at=approved_at
                )
            ):
                raise ValueError("workspace_preparation.preview_stale")
            if resume is not None:
                self._predecessor(plan)
            envelope = TaskInputEnvelope.create(
                task_kind=TASK_KIND,
                input_schema_id="workspace-preparation-input",
                payload={
                    "plan": plan.model_dump(mode="json"),
                    "approved_at": approved_at.isoformat(),
                    "caller": caller,
                    **(
                        {"data_issue_grant_hash": grant_hash}
                        if caller == "EXTERNAL_AUTOMATION"
                        else {}
                    ),
                    **({"source_task_id": str(resume)} if resume else {}),
                },
            )
            goal = ResearchGoal.create(
                goal_kind="PREPARE_RESEARCH_WORKSPACE",
                input_hash=envelope.input_hash,
                deliverable_kind="FactorInputBundle",
                summary="Prepare current-universe research inputs, not a strategy or Foundation.",
            )
            workflow = ResearchPlan.create(
                goal_hash=goal.goal_hash,
                workflow_definition_hash=canonical_hash(STAGES),
                verifier_catalog_hash=canonical_hash(STAGES),
                work_items=tuple(
                    WorkItemDefinition.create(
                        stage_id=s,
                        dependency_ids=STAGES[:i],
                        verifier_id=f"workspace_preparation.{s}",
                    )
                    for i, s in enumerate(STAGES)
                ),
            )
            task = self.session.task_control_registry.admit(
                input_envelope=envelope, goal=goal, plan=workflow, observed_at=approved_at
            ).record
            return CommandAdmission(task_id=task.task_id, lifecycle=task.lifecycle.value)

    def _require(self, plan: WorkspacePreparationPlan) -> None:
        current = read_research_workspace_manifest(self.session.workspace)
        restored = current.with_bindings(
            data_update=plan.workspace_manifest.data_update,
            experiment_inputs=plan.workspace_manifest.experiment_inputs,
        )
        if (
            restored != plan.workspace_manifest
            or plan.binding != installed_data_update_binding(self.session.workspace)
            or not is_current(IMPLEMENTATION_ROLE, plan.implementation_hash, _implementation())
        ):
            raise ValueError("workspace_preparation.binding_changed")
        if (
            plan.existing_inputs is not None
            and read_workspace_inputs(self.session.workspace, plan.binding) != plan.existing_inputs
        ):
            raise ValueError("workspace_preparation.local_inputs_changed_new_scope_required")

    def _stored_plan(self, task: TaskRecord) -> WorkspacePreparationPlan:
        if task.task_kind != TASK_KIND:
            raise ValueError("workspace_preparation.task_not_admitted")
        plan: WorkspacePreparationPlan = WorkspacePreparationPlan.model_validate_json(
            json.dumps(task.input.payload["plan"])
        )
        if task.input.payload.get("caller") == "HUMAN":
            return plan
        if DataIssueDelegation.known_successor_receipt(
            task.input.payload
        ) and task.input.payload.get("source_task_id") == str(plan.predecessor_task_id):
            return plan
        grant_hash = task.input.payload.get("data_issue_grant_hash")
        if (
            task.input.payload.get("caller") == "EXTERNAL_AUTOMATION"
            and isinstance(grant_hash, str)
            and plan.predecessor_task_id is not None
            and task.input.payload.get("source_task_id") == str(plan.predecessor_task_id)
            and self.data_issue_delegation.read(grant_hash)["preparation_task_id"]
            == str(plan.predecessor_task_id)
        ):
            return plan
        raise ValueError("workspace_preparation.task_not_admitted")

    def _of(self, task: TaskRecord) -> WorkspacePreparationPlan:
        plan = self._stored_plan(task)
        self._require(plan)
        return plan

    def _predecessor(self, plan: WorkspacePreparationPlan) -> tuple[TaskRecord, dict[str, Any]]:
        if plan.predecessor_task_id is None:
            raise ValueError("workspace_preparation.predecessor_absent")
        previous = self.session.task_control_registry.task(plan.predecessor_task_id)
        original = self._stored_plan(previous)
        if previous.lifecycle not in {TaskLifecycle.CANCELLED, TaskLifecycle.BLOCKED} or any(
            getattr(original, key) != getattr(plan, key)
            for key in ("workspace_manifest", "binding", "target_session", "existing_inputs")
        ):
            raise ValueError("workspace_preparation.resume_scope_changed")
        frozen = self._load(previous.task_id, STAGES[0])
        if frozen is None or not any(
            item.stage_id == STAGES[0] and item.lifecycle.value == "VERIFIED"
            for item in self.session.task_control_registry.work_items(previous.task_id)
        ):
            raise ValueError("workspace_preparation.predecessor_source_unverified")
        return previous, frozen

    def _planned(self, plan_hash: str) -> WorkspacePreparationPlan | None:
        """The plan an answer named while it is runnable, kept by the store whether or not the
        Host restarted since (V525); a newer plan supersedes it. Never this Host's memory alone,
        which ran a plan past its hour that a restarted Host refused (V536)."""
        return self._previews.runnable(plan_hash)

    def _path(self, task_id: UUID, name: str) -> Path:
        return self.telemetry.path(task_id, name)

    def _load(self, task_id: UUID, name: str, *, optional: bool = False) -> dict[str, Any] | None:
        path = self._path(task_id, name)
        if not path.exists() and optional:
            return None
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("workspace_preparation.stage_tampered")
        if data["content_hash"] != canonical_hash(
            {k: v for k, v in data.items() if k != "content_hash"}
        ):
            raise ValueError("workspace_preparation.stage_tampered")
        if data["task_id"] != str(task_id) or data["stage"] != name:
            raise ValueError("workspace_preparation.stage_misfiled")
        task = self.session.task_control_registry.task(task_id)
        if data["input_hash"] != task.input.input_hash:
            raise ValueError("workspace_preparation.stage_input_mismatch")
        item = next(
            (
                v
                for v in self.session.task_control_registry.work_items(task_id)
                if v.stage_id == name
            ),
            None,
        )
        if (
            item is not None
            and item.evidence
            and item.evidence[0].content_hash != data["content_hash"]
        ):
            raise ValueError("workspace_preparation.stage_evidence_changed")
        if "budget_upper_bound" in data:
            try:
                StorageBudgetResolution.model_validate(data["budget_upper_bound"])
            except ValueError as error:
                raise ValueError("workspace_preparation.stage_tampered") from error
        return data

    def _save(
        self, task: TaskRecord, name: str, values: dict[str, Any], *, progress: bool = False
    ) -> dict[str, Any]:
        data = {
            "task_id": str(task.task_id),
            "input_hash": task.input.input_hash,
            "stage": name,
            **values,
        }
        data["content_hash"] = canonical_hash(data)
        existing = self._load(task.task_id, name, optional=True)
        if existing is not None and not progress:
            if existing != data:
                raise ValueError("workspace_preparation.stage_identity_conflict")
            return existing
        self.telemetry.replace_json(self._path(task.task_id, name), data)
        return data

    def _bound_progress_sink(
        self,
        task: TaskRecord,
        execution: TaskExecution,
        stage: str,
        publisher: WorkspaceProgressPublisher,
    ) -> Callable[[WorkProgressUpdate], object]:
        """The Feature owners' work progress kept beside this Task's stage records, bound to
        the Task, its execution and this stage (the shared Task telemetry)."""

        return self.telemetry.bound_progress_sink(task, execution, stage, publisher)

    def _bound_listing_observer(
        self,
        task: TaskRecord,
        execution: TaskExecution,
        stage: str,
        seed: CurrentUniverseOnboardingOutcome,
    ) -> _ListingActivityDelivery:
        """The onboarding runner's listing-unit transitions and running counts, kept beside
        this Task's stage records bound to the Task, its execution and this stage
        (`listing-activity.json`), seeded from the runner's retained progress. Telemetry,
        never authority: a lost write is a staler snapshot; the durable chunk record stays
        the record of the counts at each chunk boundary."""

        return _ListingActivityDelivery(
            self,
            task,
            {
                "task_id": str(task.task_id),
                "input_hash": task.input.input_hash,
                "execution_id": str(execution.execution_id),
                "stage": stage,
            },
            self._path(task.task_id, "listing-activity"),
            seed,
        )

    def _listing_activity(self, task: TaskRecord) -> dict[str, Any] | None:
        """The kept listing activity of one Task: BOUND while the snapshot names the Task's
        current execution and stage and the Task is executing, otherwise NOT_CURRENT (an
        earlier execution's or stage's rows, retained as history). Read, parsed and
        validated as the typed snapshot or said to be UNREADABLE with its cause beside
        the last valid snapshot this process wrote or read; another Task's is UNBOUND;
        nothing is promoted. `age_seconds` is the snapshot's age at the product clock."""

        task_key = str(task.task_id)
        key = (task_key, "listing-activity")
        delivery = self.telemetry.deliveries.get(key)
        report: dict[str, Any] = {} if delivery is None else {"delivery": dict(delivery)}

        def unreadable(cause: str) -> dict[str, Any]:
            retained = self.telemetry.last_valid.get(key)
            values = {"availability": "UNREADABLE", "cause": cause, **report}
            if retained is not None:
                values["last_valid"] = {
                    **{k: v for k, v in retained.items() if k != "content_hash"},
                    "age_seconds": self.telemetry.age(retained["written_at"]),
                }
            return values

        path = self._path(task.task_id, "listing-activity")
        try:
            present = path.exists()
        except OSError as error:
            return unreadable(type(error).__name__)
        if not present:
            return {"availability": "NONE", **report} if report else None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            return unreadable(type(error).__name__)
        if not isinstance(data, dict) or data.get("content_hash") != canonical_hash(
            {k: v for k, v in data.items() if k != "content_hash"}
        ):
            return unreadable("content_hash")
        try:
            snapshot = ListingActivitySnapshot.model_validate(data)
        except ValueError:
            return unreadable("snapshot_invalid")
        if snapshot.task_id != task_key or snapshot.input_hash != task.input.input_hash:
            return {"availability": "UNBOUND", **report}
        age = self.telemetry.age(snapshot.written_at)
        valid = snapshot.model_dump(mode="json")
        self.telemetry.last_valid[key] = valid
        current = (
            snapshot.execution_id == str(task.latest_execution_id)
            and snapshot.stage == task.active_work_item_id
            and task.lifecycle in {TaskLifecycle.RUNNING, TaskLifecycle.CANCEL_REQUESTED}
        )
        return {
            "availability": "BOUND" if current else "NOT_CURRENT",
            **{k: v for k, v in valid.items() if k not in _LISTING_BINDING_KEYS},
            "age_seconds": age,
            **report,
        }

    @staticmethod
    def _progress_values(stage: str, outcome: CurrentUniverseOnboardingOutcome) -> dict[str, Any]:
        return {
            "phase": stage,
            "candidates": outcome.candidates,
            "raw_ready": outcome.raw_ready,
            "quality_eligible": outcome.quality_eligible,
            "failed": outcome.failed,
            "exclusion_counts": dict(outcome.exclusion_counts) or None,
            "retry_after_at": (
                outcome.retry_after_at.isoformat() if outcome.retry_after_at else None
            ),
        }

    def compatibility(self, task: TaskRecord) -> TaskExecutionCompatibility:
        """Bind preparation schema, workflow, data policy and exact installed execution identity.

        Args:
            task: Exact admitted preparation task.

        Returns:
            Deterministic execution compatibility declaration.
        """
        plan = self._of(task)
        return TaskExecutionCompatibility.create(
            task_contract_hash=canonical_hash(schema_structure(WorkspacePreparationPlan)),
            workflow_definition_hash=task.plan.workflow_definition_hash,
            input_schema_id=task.input.input_schema_id,
            domain_policy_hash=plan.binding.content_hash,
            framework_identity_hash=self.session.execution_identity(plan.implementation_hash),
        )

    def execute(self, task_id: UUID, *, expected_task_hash: str | None = None) -> None:
        """Execute the exact retained preparation task through its writer session.

        Args:
            task_id: Exact admitted task.
            expected_task_hash: Optional optimistic task identity.
        """
        task = self.session.task_control_registry.task(task_id)
        self.session.execute_admitted(task, self, self.clock, expected_task_hash)

    def owns_published_manifest(self, manifest_hash: str) -> bool:
        """Check retained preparation publication records for one exact manifest identity.

        Args:
            manifest_hash: Exact manifest identity.

        Returns:
            Whether a retained preparation stage published this manifest.
        """
        return any(
            record is not None and record["manifest_hash"] == manifest_hash
            for task in self.tasks()
            for record in (self._load(task.task_id, STAGES[3], optional=True),)
        )

    def execute_stage(
        self, *, task: TaskRecord, execution: TaskExecution, work_item: WorkItemDefinition
    ) -> StageExecutionResult:
        """Reuse or perform and retain one exact preparation checkpoint.

        Args:
            task: Exact admitted preparation task.
            execution: Current execution declaration.
            work_item: Exact workflow stage.

        Returns:
            Ready checkpoint evidence or a bounded named storage/runtime/validation refusal.
        """
        try:
            plan = self._of(task)
            stage = work_item.stage_id
            existing = self._load(task.task_id, stage, optional=True)
            if existing is None:
                result = self._perform(task, plan, stage, execution)
                if isinstance(result, StageExecutionResult):
                    return result
                existing = self._save(task, stage, result)
            return StageExecutionResult(
                StageDisposition.READY,
                evidence=(
                    TaskEvidence(
                        evidence_kind="workspace_preparation.stage",
                        reference=f"playpen://workspace-preparation/{task.task_id}/{stage}",
                        content_hash=existing["content_hash"],
                    ),
                ),
            )
        except ChildStartFailed:
            raise
        except (ValueError, OSError, RuntimeError) as error:
            return StageExecutionResult(
                StageDisposition.BLOCKED,
                failure_code=(
                    error.failure.code
                    if isinstance(error, WorkspaceConflictError)
                    else str(getattr(error, "failure_code", str(error)))[:120]
                ),
            )

    def _perform(
        self,
        task: TaskRecord,
        plan: WorkspacePreparationPlan,
        stage: str,
        execution: TaskExecution,
    ) -> dict[str, Any] | StageExecutionResult:
        root = self.session.workspace

        def cancelled() -> bool:
            return (
                self.session.task_control_registry.task(task.task_id).lifecycle
                is TaskLifecycle.CANCEL_REQUESTED
            )

        if plan.existing_inputs is not None and stage in STAGES[:3]:
            resolver = ArtifactResolver(root / "artifacts")
            panel_hash = plan.existing_inputs.panel_hash
            ref = resolver.feature_panel_manifest_uri(panel_hash)
            panel = resolver.load_feature_panel_manifest(ref)
            found = resolver.find_feature_panel_semantic_index(panel_snapshot_hash=panel_hash)
            if found is None:
                raise ValueError("workspace_preparation.local_panel_index_absent")
            estimate = storage_input_estimate(
                active_listing_count=int(panel["active_listing_count"]),
                research_session_count=len(found[0]["sessions"]),
            )
            require_storage_capacity(
                root,
                additional_bytes=estimate["estimated_core_bytes"],
            )
            return {
                "source_mode": "REUSED_QUALIFIED_LOCAL_DATA",
                "network_calls": 0,
                "snapshot_hash": panel_hash,
                "manifest_ref": ref,
                "input_estimate": estimate,
            }
        gate = self._gate()
        if stage == STAGES[0]:
            source_task_id = task.input.payload.get("source_task_id")
            if source_task_id is not None:
                previous_id = UUID(source_task_id)
                previous, frozen = self._predecessor(plan)
                if previous.task_id != previous_id:
                    raise ValueError("workspace_preparation.resume_scope_changed")
                return {
                    "captured_at": frozen["captured_at"],
                    "candidate": frozen["candidate"],
                    "input_estimate": {
                        key: value
                        for key, value in frozen.get(
                            "input_estimate", frozen.get("budget_upper_bound", {})
                        ).items()
                        if key
                        in {
                            "active_listing_count",
                            "research_session_count",
                            "listing_session_pairs",
                            "estimated_core_bytes",
                        }
                    },
                }
            self._sources_allowed()
            captured = self.clock()
            bootstrap = gate.source_loader(observed_at=captured)
            estimate = storage_input_estimate(
                active_listing_count=len(bootstrap.candidate_symbols),
                research_session_count=10 * 366,
            )
            require_storage_capacity(
                root,
                additional_bytes=estimate["estimated_core_bytes"],
            )
            return {
                "captured_at": captured.isoformat(),
                "candidate": candidate_manifest_document(bootstrap.source_manifest),
                "input_estimate": estimate,
            }
        if stage == STAGES[1]:
            self._sources_allowed()
            sources = self._load(task.task_id, STAGES[0])
            assert sources is not None
            state = gate.market_data.readiness.load(plan.binding.market_profile_id)
            if state is None:
                approved = datetime.fromisoformat(task.input.payload["approved_at"])
                onboarding = gate.start_approved_onboarding(
                    WorkspaceReadinessConsent(
                        consent_id=uuid5(NAMESPACE_URL, plan.plan_hash),
                        market_profile_id=plan.binding.market_profile_id,
                        action=WorkspaceConsentAction.INITIALIZE,
                        approved_at=approved,
                    ),
                    source_loader=lambda **_: bootstrap_from_candidate_manifest_document(
                        sources["candidate"]
                    ),
                    target_session=plan.target_session,
                )
            elif state.status == "ONBOARDING_IN_PROGRESS":
                onboarding = gate.resume_onboarding()
            else:
                manifest = gate.market_data.current_quality_filtered_research_manifest(
                    market_profile_id=plan.binding.market_profile_id
                )
                if manifest is None:
                    raise ValueError("workspace_preparation.data_not_ready")
                return {"manifest_revision": manifest.revision_sha256}
            # The runner announces each listing unit it records; kept beside this
            # Task bound to this execution and stage, bounded and coalesced, with the
            # counts seeded from the units the store already holds.
            retained = onboarding.runner.retained_progress()
            activity = self._bound_listing_observer(task, execution, stage, retained)
            onboarding.runner.listing_observer = activity.observe
            if self._load(task.task_id, "progress", optional=True) is None:
                # The denominator is known once the sources are captured and the
                # units admitted: say "0 of N" (or a continuation's retained counts)
                # before the first chunk returns, not after it.
                self._save(task, "progress", self._progress_values(stage, retained), progress=True)
            minimum_target_listings = FeatureInputPolicy().minimum_sector_size
            deferred: StageExecutionResult | None = None
            while True:
                if cancelled():
                    return StageExecutionResult(StageDisposition.CANCELLED)
                self._capacity(task)
                # One run advances one hydration chunk: its listings are
                # fetched by the runner's workers and applied on one retained
                # instance, so the capacity walk, the cancel check and the
                # progress file are paid per chunk, not per listing. A cancel
                # is honoured at the next chunk boundary; every listing still
                # commits on its own and a restart resumes from those units.
                try:
                    outcome = onboarding.runner.run(
                        observed_at=self.clock(),
                        work_budget=onboarding.runner.hydration_chunk_size,
                        minimum_target_listings=minimum_target_listings,
                    )
                except DataTargetSessionLag as error:
                    outcome = onboarding.runner.retained_progress()
                    deferred = StageExecutionResult(
                        StageDisposition.DEFERRED,
                        failure_code=error.failure_code,
                        failure_cause=StageFailureCause.from_facts(error.cause),
                    )
                activity.flush(outcome)
                self._save(task, "progress", self._progress_values(stage, outcome), progress=True)
                if deferred is not None:
                    return deferred
                if outcome.status is not CurrentUniverseOnboardingStatus.RUNNING:
                    break
            if outcome.status is not CurrentUniverseOnboardingStatus.COMPLETED:
                return StageExecutionResult(
                    StageDisposition.DEFERRED
                    if outcome.status is CurrentUniverseOnboardingStatus.DEFERRED
                    else StageDisposition.BLOCKED,
                    failure_code=outcome.failure_code or "workspace_preparation.data_incomplete",
                )
            ready = gate.complete_onboarding(onboarding, outcome, observed_at=self.clock())
            assert ready.manifest is not None
            return {"manifest_revision": ready.manifest.revision_sha256}
        if stage == STAGES[2]:
            self._sources_allowed()
            data = self._load(task.task_id, STAGES[1])
            assert data is not None
            # A retry after the recoverable sector exclusion below must compose
            # the runtime over the membership the readiness owner has already
            # activated (the derived child), not the admission ancestor the
            # data stage recorded; on a first pass both name the same revision.
            readiness_record = gate.market_data.readiness.load(plan.binding.market_profile_id)
            active_revision = (
                readiness_record.active_manifest_revision
                if readiness_record is not None
                and readiness_record.active_manifest_revision is not None
                and readiness_record.status == "FEATURE_BUILDING"
                else data["manifest_revision"]
            )
            manifest = gate.market_data.load_universe_manifest_revision(active_revision)
            runtime = WorkspaceRuntime.create(
                workspace=root,
                manifest=manifest,
                provider=gate.provider,
                artifact_root=root / "artifacts",
                max_live_symbols=1000,
                feature_catalog=workspace_feature_catalog(root),
                application_controls=self.session,
            )
            # The Feature owners report their finer work (listings materialized,
            # panel years computed) through the runtime's progress sink; bound
            # here to this Task, execution and stage so the page can show it for
            # the selected Task.
            runtime.feature_foundation.progress_sink = self._bound_progress_sink(
                task, execution, stage, runtime.progress_publisher
            )
            try:
                self._capacity(task)
                self._save(
                    task,
                    "progress",
                    {"phase": stage, "status": "PREPARING_FEATURE_CLOSURE"},
                    progress=True,
                )
                runtime.prepare_feature_closure()
                coordinator = runtime.maintenance_coordinator(
                    readiness_gate=gate,
                    clock=self.clock,
                    recorded_data_confirmation=self.recorded_data_confirmation,
                )
                request = WorkspaceMaintenanceRequest.create(
                    market_profile_id=plan.binding.market_profile_id,
                    target_market_session=plan.target_session,
                    knowledge_cutoff_at=None,
                    requested_at=plan.planned_at,
                    trigger=MaintenanceTrigger.ONBOARDING,
                    membership_revision=manifest.revision_sha256,
                    data_policy_hash=workspace_maintenance_data_policy_hash(),
                    feature_policy_hash=canonical_hash(
                        {
                            "catalog": runtime.feature_foundation.catalog.binding.catalog_hash,
                            "invalidation": "domain-topology",
                            "snapshot": "annual-content-addressed-zstd",
                        }
                    ),
                )
                while True:
                    if cancelled():
                        return StageExecutionResult(StageDisposition.CANCELLED)
                    self._capacity(task)
                    # The coordinator may derive and activate a reduced child
                    # membership during first-use governance (a qualified listing
                    # whose provider reports no current sector is quarantined,
                    # not invented). It answers RUNNING and continues only for a
                    # request bound to that exact active membership, so rebind
                    # before every call instead of re-presenting the admission
                    # ancestor and blocking on membership_revision_mismatch.
                    if coordinator.manifest.revision_sha256 != request.membership_revision:
                        request = WorkspaceMaintenanceRequest.create(
                            market_profile_id=request.market_profile_id,
                            target_market_session=request.target_market_session,
                            knowledge_cutoff_at=request.knowledge_cutoff_at,
                            requested_at=request.requested_at,
                            trigger=request.trigger,
                            membership_revision=coordinator.manifest.revision_sha256,
                            data_policy_hash=request.data_policy_hash,
                            feature_policy_hash=request.feature_policy_hash,
                        )
                    outcome = coordinator.run(
                        request, observed_at=self.clock(), maintenance_work_budget=1
                    )
                    self._save(
                        task,
                        "progress",
                        {
                            "phase": stage,
                            "status": outcome.status.value,
                            # This call may have activated a child, even when it
                            # completes; report the result scope, not its request.
                            "membership_revision": coordinator.manifest.revision_sha256,
                            "retry_after_at": outcome.retry_after_at.isoformat()
                            if outcome.retry_after_at
                            else None,
                        },
                        progress=True,
                    )
                    if outcome.status is not MaintenanceStatus.RUNNING:
                        break
                if outcome.status not in {MaintenanceStatus.COMPLETED, MaintenanceStatus.NOOP}:
                    return StageExecutionResult(
                        StageDisposition.DEFERRED
                        if outcome.status is MaintenanceStatus.DEFERRED
                        else StageDisposition.BLOCKED,
                        failure_code=outcome.failure_code
                        or "workspace_preparation.feature_incomplete",
                        failure_cause=StageFailureCause.from_facts(outcome.failure_cause),
                    )
                snapshot = runtime.panel_state.feature_panel_snapshot_for_active(
                    plan.binding.market_profile_id
                )
                if snapshot is None:
                    raise ValueError("workspace_preparation.panel_absent")
            finally:
                runtime.close()
            check = PreFactorWorkspaceHost(
                ProductWorkspacePaths.at(root), PROFILE
            )._run_data_truth_preflight(
                market_profile_id=plan.binding.market_profile_id,
                as_of_session=plan.target_session,
                now=self.clock(),
            )
            if check is not None and check.blocks_research_ready:
                raise ValueError(check.blocking_failure_code)
            return {
                "snapshot_hash": str(snapshot["snapshot_hash"]),
                "manifest_ref": str(snapshot["manifest_uri"]),
            }
        if stage == STAGES[3]:
            self._capacity(task)
            feature = self._load(task.task_id, STAGES[2])
            assert feature is not None
            binding = publish_prepared_factor_inputs(
                self.session,
                task.task_id,
                input_id="factor-development",
                panel_snapshot_hash=feature["snapshot_hash"],
                completed_at=self.clock(),
                bind_configuration=True,
            )
            # Onto the manifest as it stands: an installation since does not go (V194).
            _, updated = update_research_workspace_manifest(
                root,
                lambda current: current.with_bindings(data_update=plan.binding),
                gate=self.session.mutation_gate,
            )
            return {"binding_hash": binding.binding_hash, "manifest_hash": updated.manifest_hash}
        if stage == STAGES[4]:
            inputs = self._load(task.task_id, STAGES[3])
            assert inputs is not None
            bundle = read_factor_bundle(root, inputs["binding_hash"])
            binding = next(
                v
                for v in read_research_workspace_manifest(root).experiment_inputs or ()
                if v.binding_hash == bundle.binding_hash
            )
            factor_workflow(
                workspace=root,
                binding_hash=bundle.binding_hash,
                document=normalize_factor_document(factor_template(bundle), binding, bundle),
            )
            return {"binding_hash": inputs["binding_hash"], "status": "RESEARCH_INPUTS_READY"}
        raise ValueError("workspace_preparation.stage_unknown")

    def _capacity(self, task: TaskRecord) -> None:
        from alphalattice.control.product_host.storage.retention import require_no_pending_cleanup

        require_no_pending_cleanup(self.session.workspace)
        frozen = self._load(task.task_id, STAGES[0])
        assert frozen is not None
        require_storage_capacity(
            self.session.workspace,
            additional_bytes=0,
        )

    def verify_stage(
        self,
        *,
        task: TaskRecord,
        execution: TaskExecution,
        work_item: WorkItemDefinition,
        evidence: tuple[TaskEvidence, ...],
    ) -> tuple[TaskEvidence, ...]:
        """Require task-bound checkpoint evidence and verify the final factor bundle.

        Args:
            task: Exact retained preparation task.
            execution: Current execution declaration.
            work_item: Exact workflow stage.
            evidence: Declared single-content stage evidence.

        Returns:
            Unchanged verified evidence.

        Raises:
            ValueError: Retained checkpoint input/content differs from the task or evidence.
        """
        self._of(task)
        record = self._load(task.task_id, work_item.stage_id)
        if (
            record is None
            or record["input_hash"] != task.input.input_hash
            or len(evidence) != 1
            or evidence[0].content_hash != record["content_hash"]
        ):
            raise ValueError("workspace_preparation.evidence_mismatch")
        if work_item.stage_id == STAGES[-1]:
            read_factor_bundle(self.session.workspace, record["binding_hash"])
        return evidence


@dataclass
class WorkspacePreparationCommand:
    """Dispatch one explicit preparation confirmation through its deterministic owner."""

    application: WorkspacePreparationApplication
    plan_hash: str | None = None
    caller: str = "HUMAN"
    grant_hash: str | None = None
    command_kind: str = TASK_KIND

    def admit(self) -> CommandAdmission:
        """Require the exact preparation confirmation before task admission.

        Returns:
            Deterministic task admission.

        Raises:
            ValueError: Required plan_hash is absent or its declared admission checks refuse.
        """
        if self.plan_hash is None:
            raise ValueError("workspace_preparation.plan_required")
        return self.application.confirm(
            self.plan_hash, caller=self.caller, grant_hash=self.grant_hash
        )

    def execute(self, task_id: UUID, *, expected_task_hash: str | None = None) -> None:
        """Execute one exact task admitted for this preparation confirmation.

        Args:
            task_id: Exact retained task.
            expected_task_hash: Optional optimistic task identity.
        """
        self.application.execute(task_id, expected_task_hash=expected_task_hash)
