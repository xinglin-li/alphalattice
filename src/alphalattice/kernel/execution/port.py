"""Task execution, scheduling and cancellation."""

from collections.abc import Callable, Iterator
from concurrent.futures import Future
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal, Protocol
from uuid import UUID

type ExecutionLane = Literal["interactive", "background"]
type PublicationCheck = Callable[[], AbstractContextManager[bool]]


_SELECTED_TASK: ContextVar[UUID | None] = ContextVar("execution_selected_task", default=None)


def preferred_task() -> UUID | None:
    """Return the interactive Task chosen by the current dispatcher."""
    return _SELECTED_TASK.get()


@contextmanager
def prefer_task(task_id: UUID | None) -> Iterator[None]:
    """Carry the chosen identity without changing durable Task admission."""
    reset = _SELECTED_TASK.set(task_id)
    try:
        yield
    finally:
        _SELECTED_TASK.reset(reset)


@dataclass(frozen=True, slots=True)
class ExecutionHeartbeat:
    """The callback and interval for a running Task's existing heartbeat record."""

    emit: Callable[[int], None]
    interval: float
    name: str


class ExecutionPort(Protocol):
    """Run an admitted callable under the owner's execution budget."""

    def submit[T](
        self, action: Callable[[], T], *, lane: ExecutionLane = "background"
    ) -> Future[T]:
        """Queue work in its lane and return its completion."""
        ...

    def running(
        self,
        execution_id: UUID,
        timeout: float | None,
        heartbeat: ExecutionHeartbeat,
        *,
        publication_check: PublicationCheck | None = None,
    ) -> AbstractContextManager[None]:
        """Own a running execution until its work and heartbeat settle."""
        ...

    def cancel(self, execution_id: UUID | None) -> None:
        """Stop the matching live execution after its authority accepts cancellation."""
        ...

    def register_pool[T](self, pool: T, *, name: str) -> T:
        """Register an existing pool with the owner for the pool's lifetime."""
        ...
