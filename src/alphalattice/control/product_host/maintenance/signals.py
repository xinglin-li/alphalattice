"""Non-blocking in-memory wake signal for Front Desk turns."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from threading import Event, Lock, Thread


class MaintenanceWakeController:
    """Keep Front Desk turn-time work to an O(1) timestamp comparison."""

    def __init__(self) -> None:
        """Create synchronized due-time and wake-event ownership."""
        self._lock = Lock()
        self._wake = Event()
        self._next_due_at: datetime | None = None

    @property
    def event(self) -> Event:
        """Read the wake event used by the background maintenance host.

        Returns:
            Retained thread wake event.
        """
        return self._wake

    def set_next_due(self, value: datetime | None) -> None:
        """Set or clear the next due time with explicit timezone awareness.

        Args:
            value: Optional timezone-aware due timestamp.

        Raises:
            ValueError: Supplied due timestamp is naive.
        """
        if value is not None and value.tzinfo is None:
            raise ValueError("maintenance due time must be timezone-aware")
        with self._lock:
            self._next_due_at = value.astimezone(UTC) if value is not None else None

    def seconds_until_due(self, now: datetime) -> float | None:
        """Read the non-negative delay until due work using an explicit aware clock.

        Args:
            now: Explicit timezone-aware observation timestamp.

        Returns:
            Remaining seconds, bounded below by zero, or None when disarmed.

        Raises:
            ValueError: Observation timestamp is naive.
        """
        if now.tzinfo is None:
            raise ValueError("maintenance clock must be timezone-aware")
        with self._lock:
            return (
                None
                if self._next_due_at is None
                else max(0.0, (self._next_due_at - now.astimezone(UTC)).total_seconds())
            )

    def signal_due_work(self, now: datetime) -> bool:
        """Signal the wake event only when the declared due time has been reached.

        Args:
            now: Explicit timezone-aware observation timestamp.

        Returns:
            Whether work was due and the event was signaled.

        Raises:
            ValueError: Observation timestamp is naive.
        """
        if now.tzinfo is None:
            raise ValueError("maintenance signal time must be timezone-aware")
        with self._lock:
            due = self._next_due_at is not None and now.astimezone(UTC) >= self._next_due_at
            if due:
                self._wake.set()
            return due

    def consume(self) -> bool:
        """Read and clear the pending wake signal.

        Returns:
            Whether a signal was present before clearing.
        """
        observed = self._wake.is_set()
        self._wake.clear()
        return observed


class MaintenanceBackgroundHost:
    """One coalescing wake event around a durable coordinator callback."""

    def __init__(
        self,
        *,
        wake: MaintenanceWakeController,
        run_once: Callable[[], None],
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Create a bounded maintenance worker with explicit wake and run-once owners.

        Args:
            wake: Retained due-time/wake controller.
            run_once: Deterministic single-cycle callback.
            clock: Optional explicit observed-time source.
        """
        self.wake = wake
        self.run_once = run_once
        self.clock = clock
        self._stop = Event()
        self._last_error_code: str | None = None
        self._thread = Thread(
            target=self._worker,
            name="workspace-maintenance-coordinator",
            daemon=True,
        )

    def start(self) -> None:
        """Start this retained maintenance worker thread."""
        self._thread.start()

    def close(self, *, timeout: float = 5.0) -> bool:
        """Request shutdown and report whether the worker has stopped.

        A provider call cannot be cancelled safely.  Callers must not close
        resources used by ``run_once`` until this method returns ``True`` (or
        ``wait_stopped`` subsequently returns).
        """
        self._stop.set()
        self.wake.event.set()
        self._thread.join(timeout=timeout)
        return not self._thread.is_alive()

    def wait_stopped(self) -> None:
        """Wait for a previously requested shutdown to finish."""
        self._thread.join()

    @property
    def last_error_code(self) -> str | None:
        """Read the bounded failure code retained by the maintenance worker.

        Returns:
            Last public worker failure code or None.
        """
        return self._last_error_code

    def _worker(self) -> None:
        while not self._stop.is_set():
            self.wake.event.wait(
                None if self.clock is None else self.wake.seconds_until_due(self.clock())
            )
            if self._stop.is_set():
                return
            due = self.clock is None or self.wake.signal_due_work(self.clock())
            signalled = self.wake.consume()
            if due and signalled:
                try:
                    self.run_once()
                    self._last_error_code = None
                except Exception:
                    # The durable coordinator owns details; the chat surface gets
                    # only a stable safe code and the host remains wakeable.
                    self._last_error_code = "workspace_maintenance.background_cycle_failed"
                    if self.clock is not None:
                        # A failing timed callback must not spin on an expired
                        # deadline. An explicit re-arm is needed after failure.
                        self.wake.set_next_due(None)
