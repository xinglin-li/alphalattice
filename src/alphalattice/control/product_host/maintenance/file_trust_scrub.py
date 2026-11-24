"""The Host's idle job for re-proving its retained file trust records."""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from threading import Event

from alphalattice.capabilities.domain_contracts.environment import file_trust
from alphalattice.control.workspace_runtime.verified_facts import FileTrustScrub
from alphalattice.interface.local_application.dispatcher import LocalBackgroundDispatcher
from alphalattice.kernel.config import current as process_config
from alphalattice.kernel.execution import TIMERS, Timers

from .signals import MaintenanceBackgroundHost, MaintenanceWakeController


class IdleFileTrustScrub:
    """Let the verifier re-prove files on the Host's idle wake without taking a Task worker."""

    def __init__(
        self, workspace: Path, dispatcher: LocalBackgroundDispatcher, timers: Timers = TIMERS
    ) -> None:
        """Bind the workspace's trust record to the dispatcher clock, idle wake and timer."""
        self.dispatcher = dispatcher
        self.clock = dispatcher.clock
        self.scrub = FileTrustScrub(
            workspace,
            clock=self.clock,
            period=timedelta(days=process_config().settings.file_scrub_days),
        )
        self.wake = MaintenanceWakeController()
        self._stopping = Event()
        self.host = MaintenanceBackgroundHost(
            wake=self.wake, run_once=self._run, clock=self.clock, timers=timers
        )

    def start(self) -> None:
        """Start the worker. The first dispatcher idle handoff arms its cadence."""
        self.host.start()

    def idle(self) -> None:
        """Signal this Host's existing idle handoff."""
        if not self._stopping.is_set() and file_trust():
            self.wake.arm_if_disarmed(self.scrub.next_due)

    def close(self, timeout: float | None) -> bool:
        """Stop at a file boundary while the workspace's writer lease still holds."""
        self._stopping.set()
        quiet = self.host.close(timeout=0.0 if timeout is None else timeout)
        if not quiet and timeout is None:
            self.host.wait_stopped()
            return True
        return quiet

    def _available(self) -> bool:
        return not self._stopping.is_set() and self.dispatcher.foreground_idle()

    def _publish(self, action: Callable[[], None]) -> bool:
        return not self._stopping.is_set() and self.dispatcher.publish_when_idle(action)

    def _run(self) -> None:
        # At its thread's own priority: a lowered thread that holds the interpreter's lock, or
        # the dispatcher's, on a busy machine stalls every request thread of this Host. The
        # scrub yields to foreground work by `_available` instead.
        # Disarm before checking work so the next idle callback can always resume a yield.
        self.wake.set_next_due(None)
        try:
            complete = self.scrub.run(self._available, publish=self._publish)
        except Exception:
            self.idle()  # leave the cycle due for the maintenance worker's bounded retry
            raise
        if complete:
            self.wake.set_next_due(self.scrub.next_due)
        elif self._available():
            self.idle()
