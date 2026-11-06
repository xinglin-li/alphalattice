"""One synchronous mutation boundary for one local workspace process."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from threading import RLock, current_thread
from time import monotonic
from typing import ParamSpec, TypeVar

_Parameters = ParamSpec("_Parameters")
_Result = TypeVar("_Result")
_SUBJECT: ContextVar[tuple[str, str | None] | None] = ContextVar(
    "workspace_gate_subject", default=None
)
_DEADLINE: ContextVar[float | None] = ContextVar("workspace_gate_deadline", default=None)


class GateBusy(TimeoutError):
    """A request's wait for the gate reached its deadline; the wait changed nothing."""

    def __init__(self, holder: dict[str, object] | None = None) -> None:
        """Name the holder the request waited behind.

        Args:
            holder: The gate's holder when the wait ended, or None if it had just freed.
        """
        super().__init__("workspace mutation gate remained busy")
        self.holder = holder


@contextmanager
def request_deadline(deadline: float) -> Iterator[None]:
    """Bound every gate wait of this request, on this thread, by its monotonic `deadline`.

    Background work runs outside any request and keeps waiting as before.

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

    def holder(self) -> dict[str, object] | None:
        """Who holds the gate now, read without taking it.

        A wait behind the gate can say whom it waited for instead of guessing.

        Returns:
            The holder's operation and Task as its thread named them, the thread, and the
            seconds held; None when the gate is free.
        """
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

    @contextmanager
    def _held(self) -> Iterator[None]:
        # Runs with the lock held, so the depth and holder are this owner's alone.
        self._depth += 1
        if self._depth == 1:
            operation, task_id = _SUBJECT.get() or (None, None)
            self._holder = (operation, task_id, current_thread().name, monotonic())
        try:
            yield
        finally:
            self._depth -= 1
            if self._depth == 0:
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
        """Hold the single-writer boundary across a bounded mutation session.

        Inside a request's deadline, the wait ends there with GateBusy, naming the holder.

        Yields:
            Nothing; the gate is held until the block ends.

        Raises:
            GateBusy: A request's deadline passed while another thread held the gate.
        """
        deadline = _DEADLINE.get()
        if deadline is None:
            self._lock.acquire()
        else:
            remaining = deadline - monotonic()
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
            raise TimeoutError("workspace mutation gate remained busy")
        try:
            with self._held():
                yield
        finally:
            self._lock.release()
