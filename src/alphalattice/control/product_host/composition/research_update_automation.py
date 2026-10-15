"""Service-lifetime consent and wake over the existing Portfolio operations.

It owns the Host's own update admissions: the daily automation the person enables, and the first
use's dated update, admitted in the activation's act or, when its date's information is not yet
available, held until then and run once.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from dataclasses import replace
from datetime import date, datetime, timedelta
from threading import RLock
from typing import Any, Literal, Protocol, Self, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, model_validator

from alphalattice.capabilities.causal_inputs.availability import information_available_at
from alphalattice.control.product_host.composition.application_session import (
    WorkspaceApplicationSession,
)
from alphalattice.control.product_host.composition.goals import GoalApplication
from alphalattice.control.product_host.composition.plain_refusals import explain
from alphalattice.control.product_host.maintenance.signals import (
    MaintenanceBackgroundHost,
    MaintenanceWakeController,
)
from alphalattice.control.product_host.publication.goals import sessions_of
from alphalattice.control.workspace_runtime.content_store import CommittedIndex, CommittedKind
from alphalattice.foundation.causal_outcomes.execution.readers import planned_local_qa_schedule
from alphalattice.interface.local_application.cli_contract import (
    REQUEST_PROVENANCE,
    RequestProvenance,
    command,
    entry,
)
from alphalattice.interface.local_application.dispatcher import LocalBackgroundDispatcher
from alphalattice.interface.local_application.failure_codes import located_failure
from alphalattice.interface.local_application.goals import target_sessions, update_offers
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchOperationRequest,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash


class ResearchUpdateAutomationSettings(BaseModel):  # type: ignore[misc]
    """Seal caller-selected installed update packages with exact manifest and settings lineage."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    workspace_manifest_hash: str
    previous_hash: str | None
    enabled: bool
    package_ids: tuple[str, ...]
    chosen_by: Literal["HUMAN", "INSTALLED_AGENT"]
    content_hash: str

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal explicit research update automation settings.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical content_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        return cls(**values, content_hash=canonical_hash(values))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def identity(self) -> Self:
        """Require ordered unique selected packages and exact enabled settings identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Packages are unordered/duplicated, enabled selection is empty or
                content_hash differs.
        """
        if (
            self.package_ids != tuple(sorted(set(self.package_ids)))
            or (self.enabled and not self.package_ids)
            or self.content_hash
            != canonical_hash(self.model_dump(mode="json", exclude={"content_hash"}))
        ):
            raise ValueError("research_update.automation_settings_invalid")
        return self


_AUTOMATION_SETTINGS = CommittedKind(
    "research-update-automation",
    "research-update-automation",
    ResearchUpdateAutomationSettings,
    "content_hash",
)

_MOVING = frozenset({"QUEUED", "RUNNING", "RECOVERY_REQUIRED", "CANCEL_REQUESTED"})
"""An attended update still on its way, as its readback's status names it."""
_STOPPED = frozenset({"BLOCKED", "CANCELLED"})
"""An attended update that stopped: its words name the way on, never a new plan at once."""
_INPUTS_NOT_READY = "strategy_score.workspace_inputs_not_ready"
"""A plan refused while the workspace's inputs were not ready, as while another update's
provider deferral holds them; the package is tried again once the worker is next idle."""


class ResearchUpdateAutomation:
    """One wake callback, not a Task runtime or an external/background service.

    Settings use the existing write-once index. They grant scheduling only;
    operations retain source access, fit budgets, Task identity and publication.
    """

    def __init__(
        self,
        *,
        index: CommittedIndex,
        workspace_manifest_hash: str,
        installed_package_ids: tuple[str, ...],
        clock: Callable[[], datetime],
        execute: Callable[[PortfolioResearchOperationRequest], dict[str, object]],
        resume: Callable[[], None] | None = None,
    ) -> None:
        """Wire committed settings and a bounded service-lifetime maintenance host.

        Args:
            index: Deterministic committed settings index.
            workspace_manifest_hash: Exact admitted workspace declaration.
            installed_package_ids: Explicit installed strategy packages.
            clock: Explicit observed-time source.
            execute: Deterministic operation dispatch callback.
            resume: At start, what holds the first use's dated update again when still owed.
        """
        self.index, self.manifest_hash = index, workspace_manifest_hash
        self.installed = installed_package_ids
        self.clock, self.execute = clock, execute
        self.wake = MaintenanceWakeController()
        self.host = MaintenanceBackgroundHost(wake=self.wake, run_once=self._cycle, clock=clock)
        # The first use's dated update: the same seam, one shot, never the daily schedule.
        self.held_wake = MaintenanceWakeController()
        self.held_host = MaintenanceBackgroundHost(
            wake=self.held_wake, run_once=self._run_held, clock=clock
        )
        self._held: tuple[datetime, Callable[[], object]] | None = None
        self._resume = resume
        self._lock = RLock()
        self._started = False
        self._closing = False
        self._next_due: datetime | None = None
        self._last_attempt: dict[str, object] | None = None
        self._pending: list[str] = []  # Scheduling order only; Tasks remain durable elsewhere.
        self._round: date | None = None
        """The session whose data readiness last owed every package its cycle."""
        # Attendance: the update a package's cycle admitted or resumed, read once the Task
        # worker is idle; when a deferred one may run again; and the packages refused while the
        # workspace's inputs were not ready, cycled again once the worker is next idle.
        self._watched: dict[str, str] = {}
        self._retry: dict[str, datetime] = {}
        self._after: list[str] = []

    def settings(self) -> ResearchUpdateAutomationSettings | None:
        """Walk committed settings lineage to its exact latest admitted record.

        Returns:
            Latest settings or None without settings.

        Raises:
            ValueError: Parent lineage is broken or cyclic.
        """
        current = None
        seen: set[str] = set()
        while True:
            parent = None if current is None else current.content_hash
            value = self.index.open(_AUTOMATION_SETTINGS, canonical_hash(["SETTINGS", parent]))
            if value is None:
                return current
            if value.previous_hash != parent or value.content_hash in seen:
                raise ValueError("research_update.automation_lineage_invalid")
            seen.add(value.content_hash)
            current = value

    def readback(self) -> dict[str, object]:
        """Read scheduling status, admission, last attempt and maintenance failure.

        Returns:
            Disabled/enabled/reapproval/failure state, settings and next due time; source and fit
            authority remain ungranted.
        """
        with self._lock:
            value = self.settings()
            admitted = value is not None and value.workspace_manifest_hash == self.manifest_hash
            return {
                "status": "DISABLED"
                if value is None or not value.enabled
                else (
                    "AUTOMATION_CHECK_FAILED"
                    if self.host.last_error_code is not None
                    else "ENABLED_SERVICE_LIFETIME"
                    if admitted
                    else "SETTINGS_REAPPROVAL_REQUIRED"
                ),
                "settings": None if value is None else value.model_dump(mode="json"),
                "next_due_at": None
                if self._next_due is None or self.host.last_error_code is not None
                else self._next_due.isoformat(),
                "failure_code": self.host.last_error_code,
                "last_attempt_in_service": self._last_attempt,
                "source_or_fit_authority_granted": False,
                "detail": (
                    "Runs only while this Local Web service is running. Closing the browser "
                    "is safe; exiting the service stops scheduling. "
                    "Existing admitted Tasks may finish."
                ),
            }

    def configure(
        self,
        *,
        enabled: bool,
        package_ids: tuple[str, ...],
        chosen_by: Literal["HUMAN", "INSTALLED_AGENT"],
    ) -> dict[str, object]:
        """Commit an explicit installed-package selection and arm service-lifetime scheduling.

        Args:
            enabled: Whether scheduling is enabled.
            package_ids: Explicit selected installed packages.
            chosen_by: Declared human or installed-agent chooser.

        Returns:
            Current scheduling readback after exact settings reuse or commit.

        Raises:
            ValueError: Service is closing or a selected package is not installed.
        """
        with self._lock:
            if self._closing:
                raise ValueError("research_update.service_closing")
            if not set(package_ids) <= set(self.installed):
                raise ValueError("research_update.automation_package_not_installed")
            old = self.settings()
            if old is None or (old.enabled, old.package_ids, old.workspace_manifest_hash) != (
                enabled,
                tuple(sorted(set(package_ids))),
                self.manifest_hash,
            ):
                value = ResearchUpdateAutomationSettings.create(
                    workspace_manifest_hash=self.manifest_hash,
                    previous_hash=None if old is None else old.content_hash,
                    enabled=enabled,
                    package_ids=tuple(sorted(set(package_ids))),
                    chosen_by=chosen_by,
                )
                self.index.commit(
                    _AUTOMATION_SETTINGS, canonical_hash(["SETTINGS", value.previous_hash]), value
                )
            elif self.host.last_error_code is None:
                return self.readback()
            self._arm(self.clock() if enabled else None)
            self._pending = list(sorted(set(package_ids))) if enabled else []
            self._attend_afresh()
            return self.readback()

    def _attend_afresh(self) -> None:
        """Forget what earlier cycles attended.

        The next cycle reads each package's update from its durable Tasks again, its plan
        answering one that waits.
        """
        self._round = None
        self._watched.clear()
        self._retry.clear()
        self._after.clear()

    def _arm(self, value: datetime | None) -> None:
        self._next_due = value
        self.wake.set_next_due(value)
        self.wake.event.set()

    def start(self) -> None:
        """Start the maintenance host and catch up from admitted settings and durable tasks."""
        with self._lock:
            value = self.settings()
            if (
                value is not None
                and value.enabled
                and value.workspace_manifest_hash == self.manifest_hash
            ):
                self._pending = list(value.package_ids)
                self._attend_afresh()
                self._arm(self.clock())  # Catch up from durable Tasks, not a remembered date.
            self.host.start()
            self.held_host.start()
            self._started = True
        if self._resume is not None:
            self._resume()

    def close(self, *, timeout: float | None) -> bool:
        """Disarm scheduling and wait for the maintenance host within the declared bound.

        Args:
            timeout: Optional stop timeout; None waits for complete stop after the initial bounded
                wait.

        Returns:
            Whether the maintenance host stopped.
        """
        self._closing = True
        self._arm(None)
        self.held_wake.set_next_due(None)
        if not self._started:
            return True
        quiet = self.held_host.close(timeout=5.0 if timeout is None else timeout)
        quiet = self.host.close(timeout=5.0 if timeout is None else timeout) and quiet
        if not quiet and timeout is None:
            self.host.wait_stopped()
            quiet = True
        return quiet

    def hold(self, at: datetime, run: Callable[[], object]) -> None:
        """Hold the first use's dated update until its information is available; run it once.

        It carries out the person's own named request, so the daily automation's setting does not
        gate it, and it never arms the daily schedule.
        """
        with self._lock:
            self._held = (at, run)
        self.held_wake.set_next_due(at)
        self.held_wake.event.set()

    def _run_held(self) -> None:
        with self._lock:
            held, self._held = self._held, None
            if held is not None and (self._closing or held[0] > self.clock()):
                self._held, held = held, None
        if held is not None:
            held[1]()

    def command_completed(self) -> None:
        """Wake once the dispatcher's queue is idle.

        The remaining packages are cycled then, and the updates the automation attends: one it
        admitted or resumed is read then.
        """
        with self._lock:
            if (self._pending or self._watched or self._after) and not self._closing:
                self._arm(self.clock())

    def _cycle(self) -> None:
        with self._lock:
            settings = self.settings()
            if (
                self._closing
                or settings is None
                or not settings.enabled
                or settings.workspace_manifest_hash != self.manifest_hash
            ):
                self._arm(None)
                return
            watched = dict(self._watched)
        # The updates earlier cycles admitted or resumed, read now the worker is idle.
        reads = {
            package: self.execute(
                PortfolioResearchOperationRequest(
                    operation="RESEARCH_UPDATE_READBACK", task_id=UUID(task_id)
                )
            )
            for package, task_id in watched.items()
        }
        with self._lock:
            if self._closing or self.settings() != settings:
                return
            now = self.clock()
            # Data readiness already applies this operational two-hour Yahoo
            # finality delay; this schedules a check, not a data-availability claim.
            ready = [
                (p.formation_session, information_available_at(p.formation_close_at))
                for p in planned_local_qa_schedule(
                    now.date() - timedelta(days=14), now.date() + timedelta(days=14)
                )
            ]
            latest = max((session for session, at in ready if at <= now), default=None)
            for package, read in reads.items():
                self._settle(package, watched[package], read, latest)
            if latest is not None and latest != self._round:
                # A session's data is ready: every package is owed its cycle.
                self._round = latest
                self._pending += [p for p in settings.package_ids if p not in self._pending]
            # A deferred update is resumed once its retry time has passed, and a package refused
            # while the workspace's inputs were not ready is tried again.
            owed = [
                p for p, at in sorted(self._retry.items(), key=lambda item: item[1]) if at <= now
            ]
            for package in owed:
                del self._retry[package]
            self._pending += [
                p for p in dict.fromkeys((*owed, *self._after)) if p not in self._pending
            ]
            self._after.clear()
            self._arm(min([min(at for _, at in ready if at > now), *self._retry.values()]))
            if not self._pending:
                return
            package = self._pending.pop(0)
        plan = self.execute(
            PortfolioResearchOperationRequest(
                operation="RESEARCH_UPDATE_PLAN",
                strategy_package_id=package,
            )
        )
        with self._lock:
            # Disabling while PLAN is opening artifacts prevents admission.
            if self._closing or self.settings() != settings:
                return
            result = plan
            if plan.get("status") == "PLANNED":
                result = self.execute(
                    PortfolioResearchOperationRequest(
                        operation="RESEARCH_UPDATE_RUN",
                        update_plan_hash=str(plan["update_plan_hash"]),
                    )
                )
            self._last_attempt = {
                "strategy_package_id": package,
                "settings_hash": settings.content_hash,
                "observed_at": now.isoformat(),
                "outcome": result,
            }
            if result.get("task_id"):
                self._watched[package] = str(result["task_id"])
            elif result.get("retry_after_at"):
                # Its deferred update, refused before its retry time, is resumed then.
                at = datetime.fromisoformat(str(result["retry_after_at"]))
                self._retry[package] = at
                if self._next_due is None or at < self._next_due:
                    self._arm(at)
            elif str(result.get("failure_code") or "").partition(":")[0] == _INPUTS_NOT_READY:
                self._after.append(package)
            if self._pending and not result.get("task_id"):
                self._arm(self.clock())

    def _settle(
        self, package: str, task_id: str, read: dict[str, object], latest: date | None
    ) -> None:
        """Settle an update this automation admitted or resumed, read once the worker is idle.

        Still on its way, it is read at the next idle; deferred, it is resumed at its retry
        time; published short of the latest ready session, that session is planned at once;
        stopped, its words name the way on and the next ready session plans again.
        """
        status = read.get("status")
        if status in _MOVING or self._watched.get(package) != task_id:
            return
        del self._watched[package]
        if status == "DEFERRED":
            retry = read.get("retry_after_at")
            if retry is not None:
                self._retry[package] = datetime.fromisoformat(str(retry))
            return
        update = read.get("update")
        target = update.get("target_session") if isinstance(update, dict) else None
        if (
            status not in _STOPPED
            and target is not None
            and latest is not None
            and date.fromisoformat(str(target)) < latest
            and package not in self._pending
        ):
            self._pending.append(package)


class _Host(Protocol):
    """What the first use's update reads and acts through: the Host's operations."""

    @property
    def goals(self) -> GoalApplication: ...

    @property
    def dispatcher(self) -> LocalBackgroundDispatcher: ...

    @property
    def workspace_session(self) -> WorkspaceApplicationSession: ...

    @property
    def automation(self) -> ResearchUpdateAutomation | None: ...

    def execute(self, request: PortfolioResearchOperationRequest) -> dict[str, object]: ...


def first_update(
    host: _Host,
    activated: dict[str, object],
    provenance: RequestProvenance | None,
    *,
    held: bool = False,
) -> dict[str, object]:
    """The activated strategy's first update, admitted in the activation's own act.

    The person's activation and the first use's agent's alike plan the update for the first use's
    named date while its goal delegates, else for the latest completed session, and admit its run,
    so neither waits on the other. It is recorded under the first use, whose lead's goal wait sees
    it end, and the Host wakes a Codex lead's session when it does. A named date whose information
    is not yet available is held until it is (`ResearchUpdateAutomation.hold`) and then run once,
    recorded under the goal while it stays open (`held`); the delegation still ends on time.
    """
    first_use = host.goals.first_use()
    live = (
        first_use
        if first_use is not None
        and (
            host.goals.first_use_delegation(first_use)["active"]
            or (held and first_use.state == "OPEN")
        )
        else None
    )
    named = None if live is None else live.declaration.target_date
    scope = (
        REQUEST_PROVENANCE.set(
            replace(provenance or RequestProvenance(), goal_id=str(live.goal_id))
        )
        if live is not None
        else None
    )
    run, dated = None, None
    try:
        dated = None if named is None else target_sessions(named, host.dispatcher.clock())
        plan = host.execute(
            PortfolioResearchOperationRequest(
                operation="RESEARCH_UPDATE_PLAN",
                strategy_package_id=str(activated["strategy_package_id"]),
                observed_through=None if dated is None else str(dated["formation_session"]),
            )
        )
        run = (
            host.execute(
                PortfolioResearchOperationRequest(
                    operation="RESEARCH_UPDATE_RUN",
                    update_plan_hash=str(plan["update_plan_hash"]),
                )
            )
            if plan.get("status") == "PLANNED"
            else None
        )
    except (ValueError, OSError, KeyError) as error:
        # A date outside the calendar or an owner's refusal leaves the activation standing.
        failure = located_failure(error, "research_update.refused")
        plan = {"status": "REFUSED", **failure, **explain(str(failure["failure_code"]))}
    finally:
        if scope is not None:
            REQUEST_PROVENANCE.reset(scope)
    plan = _held_until(host, activated, provenance, plan, dated, live)
    task = None if run is None else run.get("task_id")
    if task and live is not None:
        _wake_codex(host, live.goal_id, str(task))
    offered = dict(cast(dict[str, object], activated.get("next_requests") or {}))
    offered |= update_offers(offered.pop("update", None), named, task)
    kept: tuple[str, ...] = ("status", "target_session", "update_plan_hash", "failure_code")
    kept += ("detail", "next_action", "next_requests", "held_until", "reading")
    return {
        "update": {
            "plan": {key: plan[key] for key in kept if key in plan},
            **({"run": run} if run is not None else {}),
        },
        "next_requests": offered,
    }


def _held_until(
    host: _Host,
    activated: dict[str, object],
    provenance: RequestProvenance | None,
    plan: dict[str, Any],
    dated: dict[str, object] | None,
    live: Any,
) -> dict[str, Any]:
    """The plan, held when its named date's information is not yet available: the update runs
    once at that time, and the goal's ledger keeps the hold for a Host that starts again."""
    code = str(plan.get("failure_code") or "").partition(":")[0]
    if code != "research_update.target_not_completed" or dated is None or live is None:
        return plan
    at = datetime.fromisoformat(str(dated["information_available_at"]))
    if host.automation is None or at <= host.dispatcher.clock():
        return plan
    host.automation.hold(at, lambda: first_update(host, activated, provenance, held=True))
    package = str(activated["strategy_package_id"])
    host.goals.store.attribute(live.goal_id, {"held_update": {"package": package, "at": str(at)}})
    return {**plan, "held_until": at.isoformat(), "reading": dated.get("reading")}


def _wake_codex(host: _Host, goal_id: UUID, task: str) -> None:
    """Register the Codex lead's wake on the admitted update, as a goal's session holds it."""
    read = command(
        {"operation": "STATUS", "task_id": task}, prefix=entry(host.workspace_session.workspace, ())
    )
    for session in sessions_of(host.goals.store.attributed(goal_id)):
        # A wake not registered leaves the admitted update to its offered status read.
        with suppress(ValueError, OSError):
            if session["vendor"] == "codex":
                host.workspace_session.task_control_registry.register_wake(
                    UUID(task),
                    session["session_id"],
                    str(read),
                    observed_at=host.dispatcher.clock(),
                )


def resume_held(host: _Host) -> None:
    """At Host start, hold the first use's dated update again while its goal is open.

    The goal's ledger names the held package; the update's own plan reuses what already ran, so
    holding it again never admits it twice.
    """
    first = host.goals.first_use()
    if first is None or first.state != "OPEN" or host.automation is None:
        return
    entries = host.goals.store.attributed(first.goal_id)
    held = [cast(dict[str, str], e["held_update"]) for e in entries if "held_update" in e]
    if held:
        activated: dict[str, object] = {"strategy_package_id": held[-1]["package"]}
        at = max(datetime.fromisoformat(str(held[-1]["at"])), host.dispatcher.clock())
        host.automation.hold(at, lambda: first_update(host, activated, None, held=True))
