"""Cooperative stop scoped to one Host's Task execution."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_STOP: ContextVar[Callable[[], bool]] = ContextVar("host_stop", default=bool)


class HostStopping(Exception):
    """The Host asked its Task to stop at a safe point and retain its work."""

    failure_class = "HOST_STOP_REQUESTED"


def stop_predicate() -> Callable[[], bool]:
    """Capture the current Host stop for a child work-item thread."""
    return _STOP.get()


def stop_requested() -> bool:
    """Read only the stop of the Host that owns this execution."""
    return _STOP.get()()


def checkpoint() -> None:
    """Interrupt work only at a boundary that can be recovered safely."""
    if stop_requested():
        raise HostStopping("HOST_STOP_REQUESTED")


@contextmanager
def host_stop_scope(stopping: Callable[[], bool]) -> Iterator[None]:
    """Bind a Host's stop for the duration of one owned execution."""
    token = _STOP.set(stopping)
    try:
        yield
    finally:
        _STOP.reset(token)
