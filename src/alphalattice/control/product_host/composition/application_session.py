"""One process-scoped writer session for a local AlphaLattice workspace."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from alphalattice.control.task_control.contracts import TaskLifecycle, TaskRecord
from alphalattice.control.task_control.ledger import SubmittingAgent
from alphalattice.control.task_control.registry import (
    DuckDbTaskControlRegistry,
    resolve_task_control_database,
)
from alphalattice.control.task_control.runner import TaskControlRunner, TaskDomainAdapter
from alphalattice.control.workspace_runtime.database import (
    WORKSPACE_MARKET_DATA_DATABASE_FILENAME,
    retain_workspace_database,
)
from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.control.workspace_runtime.reader_threads import apply_reader_threads
from alphalattice.control.workspace_runtime.writer_lease import WorkspaceWriterLease
from alphalattice.evidence.alternative_evidence.runtime.execution import (
    CpuBudgetStore,
    TaskReaderExecution,
    budget_cores,
    machine_load,
    platform_identity,
)
from alphalattice.interface.local_application.cli_contract import REQUEST_PROVENANCE
from alphalattice.kernel.shared_kernel.identity import canonical_hash

PORTFOLIO_PUBLIC_RUNTIME_FILENAME = "portfolio-public-task-checkpoints.sqlite"


@dataclass
class WorkspaceApplicationSession:
    """The one lease, gate, Task registry, and the runners' runtime file."""

    workspace: Path
    writer_lease: WorkspaceWriterLease
    mutation_gate: WorkspaceMutationGate
    task_control_registry: DuckDbTaskControlRegistry
    runtime_path: Path

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
        cores = self._apply_cpu_budget(task, clock)
        runner = TaskControlRunner(
            registry=self.task_control_registry,
            adapters={adapter.task_kind: adapter},
            runtime_path=str(self.runtime_path),
            clock=clock,
            width=cores,
        )
        try:
            if task.lifecycle is TaskLifecycle.RECOVERY_REQUIRED:
                return runner.recover(task.task_id, expected_task_hash=expected_task_hash)
            return runner.run_next(
                expected_task_id=task.task_id, expected_task_hash=expected_task_hash
            )
        finally:
            runner.close()

    def _apply_cpu_budget(self, task: TaskRecord, clock: Callable[[], datetime]) -> int:
        """The CPU budget, as the threads this Task's DuckDB and Arrow reads use and the work
        items it runs at once (V436).

        Applied when the Task starts, and recorded beside it in the workspace's execution
        log; the reads give the same bytes at any count (binding plan, B7), and a Task's work
        items answer in their order whatever the width (`_WorkPool`).

        Returns:
            The cores the budget gives the Task.
        """

        store = CpuBudgetStore(self.workspace / "runtime")
        budget = store.read()
        machine = machine_load()
        cores, reason = budget_cores(budget, machine)
        store.record(
            TaskReaderExecution(
                task_id=task.task_id,
                task_kind=task.task_kind,
                cpu_budget=budget.cpu_budget,
                cores=cores,
                reader_threads=apply_reader_threads(cores),
                reason=reason,
                machine=machine,
                started_at=clock(),
                platform=platform_identity(),
            )
        )
        return cores

    def execution_identity(self, implementation_hash: str) -> str:
        """Bind a Task's recovery to its owner's numerical implementation, and to nothing else.

        The dispatch that drives an admitted Task (this file, Task Control's runner and
        registry) decides no number: a recovery re-verifies its stage prefix from the
        evidence, and the owner's implementation hash moves when the numbers could. The
        dispatch's own bytes were in this identity until the binding plan's B5, so every
        edit here refused the recovery of every waiting Task.
        """
        return str(canonical_hash({"implementation": implementation_hash}))

    @classmethod
    def acquire(cls, workspace: Path) -> WorkspaceApplicationSession:
        """Acquire one writer lease and reconcile/rebuild durable task state under that authority.

        Args:
            workspace: Caller-owned workspace root.

        Returns:
            Session with retained writer lease, mutation gate, task registry and public runtime
            path.

        Raises:
            Exception: Registry/session construction fails; the acquired lease is closed before
                refusal propagates.
        """
        root = workspace.resolve()
        runtime = root / "runtime"
        lease = WorkspaceWriterLease.acquire(root)
        try:
            gate = WorkspaceMutationGate()
            registry = DuckDbTaskControlRegistry(
                resolve_task_control_database(root),
                gate=gate,
                submitted_by=_submitting_agent,
            )
            # This process just acquired the exclusive workspace lease: an old
            # RUNNING record cannot still belong to a live workspace writer.
            # Registry construction outside this lease boundary stays passive.
            registry.reconcile_after_writer_acquisition(observed_at=datetime.now(UTC))
            # A Task whose row the store lost reads again from its frozen request (V181).
            registry.rebuild_from_requests(observed_at=datetime.now(UTC))
            return cls(
                workspace=root,
                writer_lease=lease,
                mutation_gate=gate,
                task_control_registry=registry,
                runtime_path=runtime / PORTFOLIO_PUBLIC_RUNTIME_FILENAME,
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
    """The agent session of the request being admitted, as the Host's boundary read it (U33)."""
    provenance = REQUEST_PROVENANCE.get()
    if provenance is None or (provenance.vendor is None and provenance.session is None):
        return None
    return SubmittingAgent(
        vendor=provenance.vendor, session=provenance.session, goal_id=provenance.goal_id
    )
