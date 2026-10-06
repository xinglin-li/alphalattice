"""One synchronous mutation boundary for one local workspace process."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from threading import RLock
from typing import ParamSpec, TypeVar

_Parameters = ParamSpec("_Parameters")
_Result = TypeVar("_Result")


class WorkspaceMutationGate:
    """Serialize local database mutations made by a single desktop host process.

    The selected database engine remains the owner of transaction isolation.
    This gate supplies the application-level single-writer ordering required by
    task events and data writes without serializing read-only connections.
    """

    def __init__(self) -> None:
        """Create one reentrant synchronous mutation boundary."""
        self._lock = RLock()

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
        with self._lock:
            return operation(*args, **kwargs)

    @contextmanager
    def hold(self) -> Iterator[None]:
        """Hold the single-writer boundary across a bounded mutation session."""
        with self._lock:
            yield

    @contextmanager
    def try_hold(self, *, timeout_seconds: float = 30.0) -> Iterator[None]:
        """Wait for a bounded interval without stealing a live writer lock."""
        if timeout_seconds <= 0.0:
            raise ValueError("workspace mutation timeout must be positive")
        acquired = self._lock.acquire(timeout=timeout_seconds)
        if not acquired:
            raise TimeoutError("workspace mutation gate remained busy")
        try:
            yield
        finally:
            self._lock.release()
