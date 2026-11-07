"""One synchronous mutation boundary for one local workspace process."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from threading import RLock, Thread, current_thread
from time import monotonic
from typing import ParamSpec, TypeVar

from alphalattice.kernel.shared_kernel.host_stop import host_stop_scope, stop_predicate

_Parameters = ParamSpec("_Parameters")
_Result = TypeVar("_Result")
_SUBJECT: ContextVar[tuple[str, str | None] | None] = ContextVar(
    "workspace_gate_subject", default=None
)
_DEADLINE: ContextVar[float | None] = ContextVar("workspace_gate_deadline", default=None)
_BACKGROUND: ContextVar[bool] = ContextVar("workspace_gate_background", default=False)
DEFAULT_WAIT_SECONDS = 30.0


class GateBusy(TimeoutError):
    """The gate remained occupied through the caller's bounded wait."""

    code = "workspace_context.writer_busy"

    def __init__(self, holder: dict[str, object] | None = None) -> None:
        """Name the holder the request waited behind.

        Args:
            holder: The gate's holder when the wait ended, or None if it had just freed.
        """
        super().__init__(self.code)
        self.holder = holder


@contextmanager
def request_deadline(deadline: float) -> Iterator[None]:
    """Bound every gate wait of this request, on this thread, by its monotonic `deadline`.

    Background Tasks wait for the current writer without a request deadline.

    Args:
        deadline: The request's own deadline, on `time.monotonic`'s clock.

    Yields:
        Nothing; the bound holds until the block ends.
    """
    scope = _DEADLINE.set(deadline)
    try:
        yield
    finally:
        _DEADLINE.reset(scope)


def current_deadline() -> float | None:
    """The deadline that bounds this thread's gate waits, if a request set one.

    Returns:
        The monotonic deadline, or None outside a bounded request.
    """
    return _DEADLINE.get()


@contextmanager
def gate_subject(operation: str, task_id: object | None = None) -> Iterator[None]:
    """Name the operation, and its Task, this thread runs, so a gate it holds says who holds it.

    Args:
        operation: The operation in flight on this thread.
        task_id: Its Task, where it has one.

    Yields:
        Nothing; the name holds until the block ends.
    """
    scope = _SUBJECT.set((operation, None if task_id is None else str(task_id)))
    try:
        yield
    finally:
        _SUBJECT.reset(scope)


@contextmanager
def background_task(operation: str, task_id: object | None = None) -> Iterator[None]:
    """Mark a background Task and its identity for its own gate waits."""
    scope = _BACKGROUND.set(True)
    try:
        with gate_subject(operation, task_id):
            yield
    finally:
        _BACKGROUND.reset(scope)


def run_background[T](call: Callable[[], T], operation: str, task_id: object | None = None) -> T:
    """Run a background worker with its operation and Task named."""
    with background_task(operation, task_id):
        return call()


def carry_background[T](call: Callable[[], T]) -> Callable[[], T]:
    """Carry only background Task identity to a concurrent work-item thread."""
    if not _BACKGROUND.get():
        return call
    operation, task_id = _SUBJECT.get() or ("BACKGROUND_TASK", None)
    stopping = stop_predicate()

    def carried() -> T:
        with host_stop_scope(stopping):
            return run_background(call, operation, task_id)

    return carried


class WorkspaceMutationGate:
    """Serialize local database mutations made by a single desktop host process.

    The selected database engine remains the owner of transaction isolation.
    This gate supplies the application-level single-writer ordering required by
    task events and data writes without serializing read-only connections.
    """

    def __init__(self) -> None:
        """Create one reentrant synchronous mutation boundary."""
        self._lock = RLock()
        self._depth = 0
        self._holder: tuple[str | None, str | None, str, float] | None = None
        self._observation = RLock()
        self._waiters: dict[Thread, tuple[str | None, str | None, str, float]] = {}

    def holder(self) -> dict[str, object] | None:
        """Who holds the gate now, read without taking it.

        A wait behind the gate can say whom it waited for instead of guessing.

        Returns:
            The holder's operation and Task as its thread named them, the thread, and the
            seconds held; None when the gate is free.
        """
        with self._observation:
            held = self._holder
        if held is None:
            return None
        operation, task_id, thread, since = held
        return {
            "operation": operation,
            "task_id": task_id,
            "thread": thread,
            "held_seconds": round(monotonic() - since, 3),
        }

    def waiting(self) -> dict[str, object] | None:
        """Return live waiters and the writer without taking the gate.

        A waiter's held_seconds is its elapsed wait. The gate and its records are process-local.
        """
        with self._observation:
            self._waiters = {
                thread: row for thread, row in self._waiters.items() if thread.is_alive()
            }
            if not self._waiters:
                return None
            now = monotonic()
            return {
                "holder": self.holder(),
                "waiters": [
                    {
                        "operation": operation,
                        "task_id": task_id,
                        "thread": name,
                        "held_seconds": round(now - since, 3),
                    }
                    for operation, task_id, name, since in self._waiters.values()
                ],
            }

    @contextmanager
    def _held(self) -> Iterator[None]:
        # Runs with the lock held, so the depth and holder are this owner's alone.
        self._depth += 1
        if self._depth == 1:
            operation, task_id = _SUBJECT.get() or (None, None)
            with self._observation:
                self._holder = (operation, task_id, current_thread().name, monotonic())
        try:
            yield
        finally:
            self._depth -= 1
            if self._depth == 0:
                with self._observation:
                    self._holder = None

    def run(
        self,
        operation: Callable[_Parameters, _Result],
        /,
        *args: _Parameters.args,
        **kwargs: _Parameters.kwargs,
    ) -> _Result:
        """Execute a callable while holding the process-local mutation boundary.

        Args:
            operation: Callable executed while holding the reentrant mutation boundary.

        Returns:
            The callable's result; its exceptions propagate.
        """
        with self.hold():
            return operation(*args, **kwargs)

    @contextmanager
    def hold(self) -> Iterator[None]:
        """Hold the single-writer boundary across a mutation session.

        Foreground callers use their request deadline or the default bound; background Tasks wait.

        Yields:
            Nothing; the gate is held until the block ends.

        Raises:
            GateBusy: Another thread held the gate throughout the bounded wait.
        """
        if _BACKGROUND.get():
            if not self._lock.acquire(blocking=False):
                thread = current_thread()
                operation, task_id = _SUBJECT.get() or (None, None)
                with self._observation:
                    self._waiters[thread] = (operation, task_id, thread.name, monotonic())
                try:
                    self._lock.acquire()
                finally:
                    with self._observation:
                        self._waiters.pop(thread, None)
        else:
            deadline = _DEADLINE.get()
            remaining = DEFAULT_WAIT_SECONDS if deadline is None else deadline - monotonic()
            got = self._lock.acquire(timeout=remaining) if remaining > 0 else False
            if not got and not self._lock.acquire(blocking=False):
                raise GateBusy(self.holder())
        try:
            with self._held():
                yield
        finally:
            self._lock.release()

    @contextmanager
    def try_hold(self, *, timeout_seconds: float = 30.0) -> Iterator[None]:
        """Wait for a bounded interval without stealing a live writer lock."""
        if timeout_seconds <= 0.0:
            raise ValueError("workspace mutation timeout must be positive")
        acquired = self._lock.acquire(timeout=timeout_seconds)
        if not acquired:
            raise GateBusy(self.holder())
        try:
            with self._held():
                yield
        finally:
            self._lock.release()
