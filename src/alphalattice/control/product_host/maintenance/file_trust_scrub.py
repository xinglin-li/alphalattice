"""The Host's idle job for re-proving its retained file trust records."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from threading import Event

from alphalattice.control.workspace_runtime.verified_facts import FileTrustScrub
from alphalattice.interface.local_application.dispatcher import LocalBackgroundDispatcher
from alphalattice.kernel.config import current as process_config
from alphalattice.kernel.resources import set_idle_thread_priority
from alphalattice.kernel.shared_kernel.environment import file_trust

from .signals import MaintenanceBackgroundHost, MaintenanceWakeController


class IdleFileTrustScrub:
    """Let the verifier re-prove files on the Host's idle wake without taking a Task worker."""

    def __init__(self, workspace: Path, dispatcher: LocalBackgroundDispatcher) -> None:
        """Bind the workspace's trust record to the dispatcher clock and idle wake."""
        self.dispatcher = dispatcher
        self.clock = dispatcher.clock
        self.scrub = FileTrustScrub(
            workspace,
            clock=self.clock,
            period=timedelta(days=process_config().settings.file_scrub_days),
        )
        self.wake = MaintenanceWakeController()
        self._stopping = Event()
        self._prioritized = False
        self.host = MaintenanceBackgroundHost(wake=self.wake, run_once=self._run, clock=self.clock)

    def start(self) -> None:
        """Arm the cadence and start the scrub's dedicated worker."""
        self.idle()
        self.host.start()

    def idle(self) -> None:
        """Signal this Host's existing idle handoff."""
        if not self._stopping.is_set() and file_trust():
            self.wake.set_next_due(self.scrub.next_due)
            self.wake.signal_due_work(self.clock())

    def close(self, timeout: float | None) -> bool:
        """Stop at a file boundary while the workspace's writer lease still holds."""
        self._stopping.set()
        quiet = self.host.close(timeout=0.0 if timeout is None else timeout)
        if not quiet and timeout is None:
            self.host.wait_stopped()
            return True
        return quiet

    def _available(self) -> bool:
        return not self._stopping.is_set() and self.dispatcher.tasks_idle()

    def _run(self) -> None:
        if not self._prioritized:
            set_idle_thread_priority()
            self._prioritized = True
        # Disarm before checking work so the next idle callback can always resume a yield.
        self.wake.set_next_due(None)
        try:
            complete = self.scrub.run(self._available)
        except Exception:
            self.idle()  # leave the cycle due for the maintenance worker's bounded retry
            raise
        if complete or self._available():
            self.idle()
