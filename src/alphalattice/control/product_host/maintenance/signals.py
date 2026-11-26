"""Non-blocking in-memory wake signal for Front Desk turns, and the job's place on the one timer.

A controller's due time arms the timer its host was given (its Host's own, or
`kernel.execution.timers.TIMERS` outside a Host) once the host names the clock owner; the timer
sets the wake event at that time, and the host's worker blocks on the event alone.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from threading import Event, Lock, Thread

from alphalattice.capabilities.domain_contracts.bounded_memo import stage_digests
from alphalattice.capabilities.domain_contracts.source_identity import source_role_acquisition
from alphalattice.capabilities.domain_contracts.waits import CEILING
from alphalattice.kernel.execution import TIMERS, Timer, Timers


class MaintenanceWakeController:
    """Keep Front Desk turn-time work to an O(1) timestamp comparison."""

    def __init__(self) -> None:
        """Create synchronized due-time and wake-event ownership."""
        self._lock = Lock()
        self._wake = Event()
        self._next_due_at: datetime | None = None
        self._timer: Timer | None = None

    def bind(self, name: str, clock: Callable[[], datetime], timers: Timers = TIMERS) -> None:
        """Declare this job on the one timer, read by ``clock``: each due time set from now on
        arms it, and the timer sets the wake event when that time comes."""
        with self._lock:
            self._timer = timers.job(name, clock, self._wake.set)
            self._timer.arm(self._next_due_at)

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
            if self._timer is not None:
                self._timer.arm(self._next_due_at)

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

    def arm_if_disarmed(self, due: datetime) -> None:
        """Set a deadline on an idle handoff without resetting a scheduled retry."""
        if due.tzinfo is None:
            raise ValueError("maintenance due time must be timezone-aware")
        with self._lock:
            if self._next_due_at is None:
                self._next_due_at = due.astimezone(UTC)
                if self._timer is not None:
                    self._timer.arm(self._next_due_at)
                else:  # no clock owner: the worker reads it as due at once
                    self._wake.set()

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


class MaintenanceBackgroundHost:
    """One coalescing wake event around a durable coordinator callback."""

    def __init__(
        self,
        *,
        wake: MaintenanceWakeController,
        run_once: Callable[[], None],
        clock: Callable[[], datetime] | None = None,
        timers: Timers = TIMERS,
    ) -> None:
        """Create a bounded maintenance worker with explicit wake and run-once owners.

        Args:
            wake: Retained due-time/wake controller.
            run_once: Deterministic single-cycle callback.
            clock: Optional explicit observed-time source.
            timers: The timer that wakes it at each due time: its Host's own, else the process's.
        """
        self.wake = wake
        self.run_once = run_once
        self.clock = clock
        if clock is not None:
            wake.bind("workspace-maintenance-coordinator", clock, timers)
        self._stop = Event()
        self._last_error_code: str | None = None
        self._failures = 0
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
            # The one timer sets the event at the due time, a signal at once, close to stop; an
            # hour with none of them runs nothing and blocks again.
            if not self.wake.event.wait(CEILING):
                continue  # no signal within the hour: nothing is read, nothing cleared
            if self._stop.is_set():
                return
            # Cleared only after the wait returned a signal, and before the due read: a timer
            # signal that lands while the clock is read stays set for the next pass.
            self.wake.event.clear()
            left = None if self.clock is None else self.wake.seconds_until_due(self.clock())
            due = self.clock is None or left == 0
            if due:
                try:
                    # A cycle hashes each payload and observes each source once.
                    with stage_digests(), source_role_acquisition():
                        self.run_once()
                    self._last_error_code, self._failures = None, 0
                except Exception:
                    # The durable coordinator owns details; the chat surface gets
                    # only a stable safe code and the host remains wakeable.
                    self._last_error_code = "workspace_maintenance.background_cycle_failed"
                    self._failures += 1
                    if self.clock is not None and self.wake.seconds_until_due(self.clock()) == 0:
                        # Work the failed run left due now is tried again after a bounded
                        # backoff, never spun on; a time it armed, or its disarm, stands.
                        delay = min(30.0 * 2 ** min(self._failures - 1, 7), 3600.0)
                        self.wake.set_next_due(self.clock() + timedelta(seconds=delay))
