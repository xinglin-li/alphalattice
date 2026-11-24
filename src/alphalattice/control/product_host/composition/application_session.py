"""One process-scoped writer session for a local AlphaLattice workspace."""

from __future__ import annotations

import platform
import sys
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from alphalattice.capabilities.data.errors import DataError
from alphalattice.capabilities.domain_contracts.environment import recorded_environment
from alphalattice.capabilities.domain_contracts.identity import canonical_hash
from alphalattice.control.observation_runtime.telemetry.process_metrics import (
    current_allowed_logical_processors,
    worker_processes,
)
from alphalattice.control.product_host.composition.research_workspace import named_workspace_id
from alphalattice.control.product_host.composition.resource_estimates import gate_for
from alphalattice.control.product_host.storage.backup import WorkspaceBackups
from alphalattice.control.product_host.storage.inventory import storage_capacity_scope
from alphalattice.control.task_control.child import warm_child_workers
from alphalattice.control.task_control.contracts import TaskLifecycle, TaskRecord
from alphalattice.control.task_control.ledger import SubmittingAgent
from alphalattice.control.task_control.queue import read_queue_setting, running_places
from alphalattice.control.task_control.registry import (
    DuckDbTaskControlRegistry,
    resolve_task_control_database,
)
from alphalattice.control.task_control.runner import TaskControlRunner, TaskDomainAdapter
from alphalattice.control.task_control.scratch import own_task_scratch, reclaim_task_scratch
from alphalattice.control.workspace_runtime.database import (
    WORKSPACE_MARKET_DATA_DATABASE_FILENAME,
    retain_workspace_database,
)
from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.control.workspace_runtime.network_access import network_access
from alphalattice.control.workspace_runtime.reader_threads import apply_reader_threads
from alphalattice.control.workspace_runtime.writer_lease import WorkspaceWriterLease
from alphalattice.evidence.alternative_evidence.runtime.execution import (
    TASK_LEASES,
    CpuBudgetStore,
    TaskReaderExecution,
    below_normal_thread,
    budget_cores,
    machine_load,
    platform_identity,
    process_priority,
)
from alphalattice.evidence.alternative_evidence.storage import world as evidence_world
from alphalattice.foundation.data_layer.backup import verify_data_root
from alphalattice.foundation.data_layer.history import open_history
from alphalattice.foundation.data_layer.history_carried_results import CarriedCheck
from alphalattice.foundation.data_layer.history_runtime import start_log_reader
from alphalattice.foundation.data_layer.upgrade import UpgradeRunner
from alphalattice.foundation.market_data_ops.storage import world
from alphalattice.interface.local_application.cli_contract import REQUEST_PROVENANCE
from alphalattice.kernel.clock import utc_now

PORTFOLIO_PUBLIC_RUNTIME_FILENAME = "portfolio-public-task-checkpoints.sqlite"

_TASK_READER_POLICY = Lock()


def _apply_task_reader_policy() -> int:
    """Apply the widest remaining grant to Arrow and workspace database opens."""
    with _TASK_READER_POLICY:
        widest = TASK_LEASES.widest()
        if widest is None:
            return 1  # No Task remains to use the pool; the next admission sets its width.
        threads: int = apply_reader_threads(widest)
        return threads


def verify_upgrade_backups(
    root: Path,
    workspaces: tuple[Path, ...],
    recorded: dict[str, Any] | None = None,
    *,
    machine_generation: Path,
    workspace_backups: tuple[WorkspaceBackups, ...],
) -> dict[str, Any]:
    """The production seam verifies the exact pre-upgrade roots without publishing backups.

    The Host binds generation/repositories with functools.partial. Missing, mismatched or
    damaged backups refuse through their existing owners before any family step begins.
    """
    machine = recorded["machine"] if recorded is not None else None
    if machine is not None and machine["generation"] != str(machine_generation.resolve()):
        raise DataError(
            "The machine generation differs from the recorded pre-upgrade backup; "
            "supply that generation before resuming.",
            code="upgrade.refused",
        )
    manifest = verify_data_root(
        machine_generation,
        source=root,
        pre_upgrade_head=machine["head"] if machine is not None else None,
    )
    backups = {value.workspace: value for value in workspace_backups}
    if set(backups) != {path.resolve() for path in workspaces}:
        raise DataError(
            "A verified backup for every upgrade workspace is required; include each workspace.",
            code="upgrade.refused",
        )
    saved = (
        {row["workspace"]: row for row in recorded["workspaces"]} if recorded is not None else {}
    )
    generations = []
    for path, value in sorted(backups.items()):
        prior = saved.get(str(path)) if recorded is not None else None
        if recorded is not None and (
            prior is None
            or prior["root"] != str(value.root)
            or prior["workspace_id"] != value.workspace_id
        ):
            raise DataError(
                "The workspace backup authority differs from its recorded pre-upgrade "
                "generation; use that backup repository before resuming.",
                code="upgrade.refused",
            )
        chosen = (
            value.verify(generation_hash=prior["generation"])
            if prior is not None
            else value.verify(current=True)
        )
        generations.append(
            dict(
                workspace=str(path),
                workspace_id=value.workspace_id,
                root=str(value.root),
                generation=chosen.generation_hash,
            )
        )
    return dict(
        machine=dict(generation=str(machine_generation.resolve()), head=manifest["head"]),
        workspaces=generations,
    )


def carried_result_checks() -> dict[str, CarriedCheck]:
    """Each result type a copy carries, with its owner's proof that a carried body is the one its
    digest names; a Host's history runs it before admitting the body (`open_history`)."""

    def execution_outcome(workspace: Path, digest: str, profile: Any, table: Any) -> None:

        # Only a carried outcome needs its numerical artifact reader.
        from ....foundation.causal_outcomes.execution.artifacts import (  # lazy: on use
            carried_chunk_check,
        )

        carried_chunk_check(workspace, digest, profile, table)

    def portfolio_benchmark(workspace: Path, digest: str, profile: Any, table: Any) -> None:

        # Only a copied benchmark needs the portfolio-input reader.
        from ....capabilities.portfolio_inputs.benchmark import (  # lazy: on use
            carried_benchmark_check,
        )

        carried_benchmark_check(workspace, digest, profile, table)

    def portfolio_execution_series(workspace: Path, digest: str, profile: Any, table: Any) -> None:

        from ....investment.portfolio_strategy_lab.publication import (  # lazy: copied row
            portfolio_ledger,
        )

        portfolio_ledger.carried_series_check(workspace, digest, profile, table)

    def portfolio_report_holding(workspace: Path, digest: str, profile: Any, table: Any) -> None:

        from ....investment.portfolio_strategy_lab.publication import (  # lazy: copied row
            portfolio_ledger,
        )

        portfolio_ledger.carried_holding_check(workspace, digest, profile, table)

    return {
        "execution_outcome": execution_outcome,
        "portfolio_benchmark": portfolio_benchmark,
        "portfolio_execution_series": portfolio_execution_series,
        "portfolio_report_holding": portfolio_report_holding,
    }


@dataclass
class WorkspaceApplicationSession:
    """The one lease, gate, Task registry, and the runners' runtime file."""

    workspace: Path
    writer_lease: WorkspaceWriterLease
    mutation_gate: WorkspaceMutationGate
    task_control_registry: DuckDbTaskControlRegistry
    runtime_path: Path
    processor: str
    scratch_epoch: str
    scratch_cleanup: dict[str, object]

    @staticmethod
    def open_history(workspace_id: str, workspace: Path, clock: Callable[[], datetime]) -> None:
        """Open Host history with the workspace's network control at the export boundary."""
        budget = CpuBudgetStore(workspace / "runtime").read().cpu_budget
        processors = len(current_allowed_logical_processors()) or 1
        open_history(
            workspace_id,
            workspace.name,
            workspace,
            can_export=lambda: network_access(workspace, now=clock()).allowed,
            carried_checks=carried_result_checks(),
            duckdb_threads=processors if budget == "auto" else min(processors, budget),
        )

    def execute_admitted(
        self,
        task: TaskRecord,
        adapter: TaskDomainAdapter,
        clock: Callable[[], datetime],
        expected_task_hash: str | None = None,
    ) -> TaskRecord | None:
        """Run or recover one admitted Host task with this session's runner."""
        if task.lifecycle not in {TaskLifecycle.QUEUED, TaskLifecycle.RECOVERY_REQUIRED}:
            return None
        gate = gate_for(self.workspace)
        start: Callable[[], None] | None = None

        @contextmanager
        def resources(record: TaskRecord) -> Iterator[None]:
            nonlocal start
            with self.task_resources(record, clock) as start:
                yield

        def stage_gate(record: TaskRecord) -> str | None:
            nonlocal start
            refused = gate.stage_refusal(record)
            if refused is None and start is not None:
                # The runner has claimed the Task and owns child-start recovery here.
                launch, start = start, None
                launch()
            return refused

        runner = TaskControlRunner(
            registry=self.task_control_registry,
            adapters={adapter.task_kind: adapter},
            runtime_path=str(self.runtime_path),
            clock=clock,
            task_scope=resources,
            stage_gate=stage_gate,
            stage_scope=storage_capacity_scope,
        )
        try:
            if task.lifecycle is TaskLifecycle.RECOVERY_REQUIRED:
                return runner.recover(task.task_id, expected_task_hash=expected_task_hash)
            return runner.run_next(
                expected_task_id=task.task_id, expected_task_hash=expected_task_hash
            )
        finally:
            runner.close()

    @contextmanager
    def task_resources(
        self, task: TaskRecord, clock: Callable[[], datetime]
    ) -> Iterator[Callable[[], None] | None]:
        """The CPU budget's cores this Task holds while it runs: its reads' threads and the work
        items it runs at once.

        A Task that may run beside others takes its share of the budget over the running
        places; the leases of the Tasks running at once stay within the budget (`TASK_LEASES`).
        What it ran with is recorded beside it in the workspace's execution log; the reads give
        the same bytes at any count, and a Task's work items answer in their
        order whatever the width (`_WorkPool`).
        """

        store = CpuBudgetStore(self.workspace / "runtime")
        budget = store.read()
        # A fixed budget needs affinity, while automatic sizing also samples load.
        machine = machine_load(sample_busy=budget.cpu_budget == "auto")
        cores, reason = budget_cores(budget, machine)
        ceiling = (
            machine.processors
            if budget.cpu_budget == "auto"
            else min(budget.cpu_budget, machine.processors)
        )
        places, _why = running_places(read_queue_setting(self.workspace / "runtime"))
        if places > 1 and self.task_control_registry.overlapping(task):
            cores = max(1, min(cores, ceiling // places))
            reason += f"; its share of {places} running places"
        with (
            ExitStack() as after_release,
            own_task_scratch(self.workspace, task.task_id, self.scratch_epoch),
            TASK_LEASES.lease(task.task_id, cores, ceiling) as beside,
            below_normal_thread() if beside else nullcontext(False) as lowered,
        ):
            after_release.callback(_apply_task_reader_policy)
            reader_threads = _apply_task_reader_policy()
            store.record(
                TaskReaderExecution(
                    task_id=task.task_id,
                    task_kind=task.task_kind,
                    cpu_budget=budget.cpu_budget,
                    cores=cores,
                    reader_threads=reader_threads,
                    reason=reason,
                    machine=machine,
                    started_at=clock(),
                    platform=platform_identity(processor=self.processor),
                    running_beside=beside,
                    cpu_ceiling=ceiling,
                    # A Task beside another runs below normal, whatever the process runs at.
                    priority=f"{process_priority()}{', thread below_normal' if lowered else ''}",
                )
            )
            start_log_reader()
            prepare = task.task_kind == "workspace_preparation"
            with warm_child_workers(
                worker_processes(cores) if prepare else 0,
                modules=("alphalattice.foundation.feature_engine.runtime.service", "pandas"),
            ) as start_workers:
                yield start_workers if prepare else None

    def execution_identity(self, implementation_hash: str) -> str:
        """Bind a Task's recovery to its owner's numerical implementation, and to nothing else.

        The dispatch that drives an admitted Task (this file, Task Control's runner and
        registry) decides no number: a recovery re-verifies its stage prefix from the
        evidence, and the owner's implementation hash moves when the numbers could. The
        dispatch's own bytes were in this identity until dispatch left the identity closure,
        so every edit here refused the recovery of every waiting Task.
        """
        return str(canonical_hash({"implementation": implementation_hash}))

    @classmethod
    def acquire(
        cls, workspace: Path, upgrade: UpgradeRunner | None = None
    ) -> WorkspaceApplicationSession:
        """Acquire one writer lease and reconcile/rebuild durable task state under that authority.

        Args:
            workspace: Caller-owned workspace root.
            upgrade: The first 0.2.8 start's machine-log runner, before any task or serving
                application is constructed. Its caller holds the machine writer for the Host's
                lifetime. No real family is registered by this mechanism card.

        Returns:
            Session with retained writer lease, mutation gate, task registry and public runtime
            path.

        Raises:
            Exception: Registry/session construction fails; the acquired lease is closed before
                refusal propagates.
        """
        # Resolve the process's platform facts before requests reach the Task runner.
        # CPython caches these values; keep the original processor string in each receipt.
        if sys.platform == "win32":
            recorded_environment()
        processor = platform.processor() or platform.machine()
        root = workspace.resolve()
        runtime = root / "runtime"
        lease = WorkspaceWriterLease.acquire(root)
        try:
            if upgrade is not None:
                if root not in {path.resolve() for path in upgrade.workspaces}:
                    raise DataError(
                        "This workspace is outside the upgrade's backed-up workspace set; "
                        "include its verified backup before starting the Host.",
                        code="upgrade.refused",
                    )
                upgrade.run()
            workspace_id = named_workspace_id(root)
            if workspace_id is not None:  # a Host's open publishes its market data and documents
                # before it answers: what a stage wrote and committed before a crash, or what was
                # written outside a Host since its last publish (world.publish_pending)
                world.publish_pending(root / "market-data.duckdb", workspace_id)
                world.admit_sectors(root / "market-data.duckdb", workspace_id)
                evidence_world.admit(
                    runtime / "artifacts", runtime / "evidence-knowledge", workspace_id
                )
            gate = WorkspaceMutationGate()
            registry = DuckDbTaskControlRegistry(
                resolve_task_control_database(root),
                gate=gate,
                submitted_by=_submitting_agent,
                workspace_id=workspace_id,
            )
            # This process just acquired the exclusive workspace lease: an old
            # RUNNING record cannot still belong to a live workspace writer.
            # Registry construction outside this lease boundary stays passive.
            registry.reconcile_after_writer_acquisition(observed_at=utc_now())
            scratch_epoch = str(uuid4())
            return cls(
                workspace=root,
                writer_lease=lease,
                mutation_gate=gate,
                task_control_registry=registry,
                runtime_path=runtime / PORTFOLIO_PUBLIC_RUNTIME_FILENAME,
                processor=processor,
                scratch_epoch=scratch_epoch,
                scratch_cleanup=reclaim_task_scratch(root, scratch_epoch),
            )
        except Exception:
            lease.close()
            raise

    @contextmanager
    def reads(self, *, timeout_seconds: float | None = None) -> Iterator[bool]:
        """One read boundary for a projection that joins several read operations.

        Every operation inside opens the workspace's stores exactly as it
        does on its own and receives a cursor of an instance this boundary
        holds, instead of a fresh engine instance per open (the session
        context is twenty-four opens, half of its time). Three holds, taken
        in the order the writers take them, so that no writer can wait for
        this boundary's instance while holding a lock the boundary needs
        next: the mutation gate first (a gated writer of the market store
        queues here, before it asks the engine for the instance), then the
        Task registry's read scope (its connection lock, then one read-only
        connection; a control-only writer queues at that lock), then the
        market store retained read-only when it exists (an unprepared
        workspace has none yet). A write inside is refused by the owners'
        own rules, never upgraded; a writer that arrives afterward waits for
        the boundary. A writer already holding the gate is different: with
        a timeout, yield False without retaining either store. The caller can
        read independent metadata and explicitly defer mutable Data discovery.
        True means the batched read boundary was acquired.
        """
        with ExitStack() as holds:
            if timeout_seconds is None:
                holds.enter_context(self.mutation_gate.hold())
            else:
                try:
                    holds.enter_context(
                        self.mutation_gate.try_hold(timeout_seconds=timeout_seconds)
                    )
                except TimeoutError:
                    yield False
                    return
            holds.enter_context(self.task_control_registry.read_scope())
            market = self.workspace / WORKSPACE_MARKET_DATA_DATABASE_FILENAME
            if market.is_file():
                holds.enter_context(retain_workspace_database(market, read_only=True))
            yield True

    def close(self) -> None:
        """Release the retained workspace writer lease."""
        self.writer_lease.close()

    def __enter__(self) -> WorkspaceApplicationSession:
        """Enter the already acquired session context.

        Returns:
            This session without acquiring another lease.
        """
        return self

    def __exit__(self, *_: object) -> None:
        """Release the retained writer lease when the session context ends.

        Args:
            _: Context exception metadata; unused by cleanup.
        """
        self.close()


__all__ = ["PORTFOLIO_PUBLIC_RUNTIME_FILENAME", "WorkspaceApplicationSession"]


def _submitting_agent() -> SubmittingAgent | None:
    """The agent session of the request being admitted, as the Host's boundary read it."""
    provenance = REQUEST_PROVENANCE.get()
    if provenance is None or (provenance.vendor is None and provenance.session is None):
        return None
    return SubmittingAgent(
        vendor=provenance.vendor, session=provenance.session, goal_id=provenance.goal_id
    )
