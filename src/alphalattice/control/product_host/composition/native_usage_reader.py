"""Read native usage for the served workspace at explicit research milestones."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any

from alphalattice.interface.local_application.cli_contract import (
    REQUEST_PROVENANCE,
    RequestProvenance,
)
from alphalattice.interface.local_application.failure_codes import (
    owner_failure_code,
    safe_failure_code,
)
from alphalattice.interface.local_application.native_bridge import (
    HOSTS,
    NativeBridgeError,
    NativeResearchBinding,
    clear_usage_checkpoints,
    read_session_usage,
    session_project,
    usage_reading,
)
from alphalattice.interface.local_application.native_setup import (
    admitted_session_project,
    autobind_root,
)
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchRequestDocument,
)

if TYPE_CHECKING:
    from alphalattice.control.product_host.composition.portfolio_research_operations import (
        PortfolioResearchOperations,
    )


class NativeUsageReader:
    """Read the bound Sessions' usage when research asks, never on a timer.

    A milestone -- a goal's take or submission, an answer's submission -- or a Goal or Team page
    opening asks. Each Session bound to this workspace whose reading is on is read from its own
    file and the children that file records (`read_session_usage`); no other Session's file is
    opened. One read runs at a time, a Session read within `RECENT` is not read again, and a
    failure is named in `state`, never a refusal of research.
    """

    RECENT = timedelta(seconds=10)

    def __init__(self, operations: PortfolioResearchOperations) -> None:
        """Retain the served workspace's binding, Goal and observation owners.

        Args:
            operations: This Host's admitted operation and Goal owners.
        """
        self.operations = operations
        self.workspace = operations.workspace_session.workspace.resolve()
        self.clock = operations.dispatcher.clock
        self._reading = Lock()
        self._read_at: dict[tuple[str, str], datetime] = {}
        self._status = "NOT_BOUND"
        self._reason: str | None = None
        self._progress: dict[str, object] = {}

    def state(self) -> dict[str, object]:
        """The last reading's standing, never native lifecycle or completeness proof.

        Returns:
            Bounded status and failure code without participant ids or file paths.
        """
        return {
            "status": "UNAVAILABLE"
            if self._reason is not None and self._reason != "native_bridge.usage_catching_up"
            else self._status,
            "reason": self._reason,
            **self._progress,
        }

    def _bindings(self) -> tuple[tuple[Path, NativeResearchBinding], ...]:
        found: list[tuple[Path, NativeResearchBinding]] = []
        projects: set[tuple[Path, str]] = set()
        for host in HOSTS:
            try:
                projects.add((session_project(self.workspace, host), host))
            except NativeBridgeError as error:
                if str(error) != "native_bridge.project_declaration_missing":
                    self._remember_failure(safe_failure_code(str(error)))
            # A workspace kept outside its project: the projects this Host admitted it from.
            projects.update((admitted, host) for admitted in self.operations.native_projects)
        for candidate, host in sorted(projects, key=lambda item: (str(item[0]), item[1])):
            try:
                project = admitted_session_project(self.workspace, candidate, host)
                bindings, diagnostics = NativeResearchBinding.binding_entries(project)
                for diagnostic in diagnostics:
                    self._reason = self._reason or safe_failure_code(diagnostic["failure_code"])
                found.extend(
                    (project, binding)
                    for binding in bindings
                    if binding.host == host and binding.workspace.resolve() == self.workspace
                )
            except NativeBridgeError as error:
                if str(error) != "native_bridge.project_declaration_missing":
                    self._remember_failure(safe_failure_code(str(error)))
            except Exception as error:
                self._reason = self._reason or (
                    owner_failure_code(error) or "native_bridge.lead_usage_read_failed"
                )
        # The bindings the Host made itself, kept in the workspace.
        root = autobind_root(self.workspace)
        try:
            held, _diagnostics = NativeResearchBinding.binding_entries(root)
            found.extend(
                (root, binding) for binding in held if binding.workspace.resolve() == self.workspace
            )
        except Exception as error:
            self._reason = self._reason or (
                owner_failure_code(error) or "native_bridge.lead_usage_read_failed"
            )
        return tuple(found)

    def _remember_failure(self, reason: str | None) -> None:
        self._reason = self._reason or reason

    def read_binding(self, project: Path, binding: NativeResearchBinding) -> dict[str, object]:
        """Read one bound Session now, filed under the Goal its Session holds.

        Args:
            project: The admitted project holding the binding.
            binding: The exact bound Session of this workspace.

        Returns:
            The reading's receipt, one entry per participant.
        """
        provenance = RequestProvenance(vendor=binding.host, session=binding.session_id)
        goal = self.operations.goals.attributed_goal(provenance)
        goal_id = None if goal is None else str(goal.goal_id)

        def publish(document: dict[str, Any]) -> dict[str, Any]:
            # Pin even an explicitly absent Goal through filing. The same-store RLock
            # prevents a concurrent Goal take from changing file_event's fallback binding.
            with (
                self.operations.workspace_session.mutation_gate.hold(),
                self.operations.goals.store.lock,
            ):
                current = self.operations.goals.attributed_goal(provenance)
                if (None if current is None else str(current.goal_id)) != goal_id or (
                    NativeResearchBinding.read(project, session=(binding.host, binding.session_id))
                    != binding
                ):
                    return {
                        "status": "REFUSED",
                        "failure_code": "native_bridge.event_scope_invalid",
                    }
                token = REQUEST_PROVENANCE.set(
                    None if goal_id is None else RequestProvenance(goal_id=goal_id)
                )
                try:
                    return self.operations.execute(
                        PortfolioResearchRequestDocument(
                            operation="EVENT_DECLARE", event=document
                        ).to_operation_request(),
                        caller="EXTERNAL_AUTOMATION",
                    )
                finally:
                    REQUEST_PROVENANCE.reset(token)

        self._read_at[(binding.host, binding.session_id)] = self.clock()
        receipt = read_session_usage(project, binding, publish=publish, goal_id=goal_id)
        self._progress = {
            key: receipt[key]
            for key in ("read_through", "total_bytes", "bytes_read", "rescan")
            if key in receipt
        }
        self._status = "OFF" if receipt.get("status") == "SKIPPED" else "OBSERVING"
        self._reason = None if self._status == "OFF" else safe_failure_code(receipt.get("reason"))
        return receipt

    def read(self) -> dict[str, object]:
        """Read every Session bound to this workspace whose reading is on, once.

        Returns:
            ``READ`` with one receipt per Session, ``RECENT`` for one read moments ago,
            ``OFF`` or ``NOT_BOUND``; ``BUSY`` while another read runs.
        """
        if not self._reading.acquire(blocking=False):
            return {"status": "BUSY", "sessions": []}
        try:
            previous_reason = self._reason
            self._reason = None
            self._progress = {}
            if not usage_reading(self.workspace):
                # The person's switch: no binding is even looked for.
                clear_usage_checkpoints(self.workspace)
                self._status = "OFF"
                self._progress = {}
                return {"status": "OFF", "sessions": []}
            found = self._bindings()
            active = tuple(item for item in found if item[1].usage != "OFF")
            if not active:
                self._status = "NOT_BOUND" if not found else "OFF"
                self._progress = {}
                return {"status": self._status, "sessions": []}
            self._status = "OBSERVING"
            pending_reason, pending_progress = self._reason, {}
            now = self.clock()
            sessions: list[dict[str, object]] = []
            for project, binding in active:
                key = (binding.host, binding.session_id)
                last = self._read_at.get(key)
                if last is not None and now - last < self.RECENT:
                    if pending_reason in {None, "native_bridge.usage_catching_up"}:
                        pending_reason = previous_reason or pending_reason
                    sessions.append({"host": binding.host, "status": "RECENT"})
                    continue
                try:
                    self._progress = {}
                    receipt = self.read_binding(project, binding)
                except Exception as error:
                    receipt = {
                        "status": "UNAVAILABLE",
                        "reason": owner_failure_code(error)
                        or "native_bridge.lead_usage_read_failed",
                    }
                if pending_reason in {None, "native_bridge.usage_catching_up"} and receipt.get(
                    "status"
                ) not in {"DELIVERED", "SKIPPED"}:
                    pending_reason = safe_failure_code(receipt.get("reason"))
                    pending_progress = self._progress
                sessions.append({"host": binding.host, **receipt})
            if pending_reason is not None:
                self._reason, self._progress = pending_reason, pending_progress
            return {"status": "READ", "sessions": sessions}
        finally:
            self._reading.release()
