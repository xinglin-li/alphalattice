"""Read native usage for the served workspace at explicit research milestones."""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from time import monotonic
from typing import TYPE_CHECKING, Any, Final

from alphalattice.interface.local_application.cli_contract import (
    REQUEST_PROVENANCE,
    RequestProvenance,
)
from alphalattice.interface.local_application.failure_codes import (
    owner_failure_code,
    safe_failure_code,
)
from alphalattice.interface.local_application.native_bridge import (
    CHILD_USAGE_BINDING_UNVERIFIED as UNVERIFIED,
)
from alphalattice.interface.local_application.native_bridge import (
    HOSTS,
    NativeBridgeError,
    NativeResearchBinding,
    autobind_root,
    clear_usage_checkpoints,
    read_session_usage,
    session_project,
    usage_reading,
)
from alphalattice.interface.local_application.native_setup import admitted_session_project
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchRequestDocument,
)

if TYPE_CHECKING:
    from alphalattice.control.product_host.composition.portfolio_research_operations import (
        PortfolioResearchOperations,
    )


UNVERIFIED_LIMIT: Final = 8
CATCHING_UP: Final = "native_bridge.usage_catching_up"
PROGRESS: Final = ("read_through", "total_bytes", "bytes_read", "rescan")


def own_standing(receipt: dict[str, Any]) -> dict[str, object]:
    """One Session's standing from its own reading, its unverified children named apart.

    A child whose own record does not name this Session as its parent contributes to
    `unverified_count`; only a bounded sample is listed under `unverified`.
    The parent check stays as
    `read_session_usage` makes it. Any other participant's failure is the Session's reason.
    """
    if receipt.get("status") == "SKIPPED":
        return {"status": "OFF", "reason": None}
    members = [m for m in receipt.get("participants") or () if isinstance(m, dict)]
    unverified = [str(m.get("agent_id")) for m in members if m.get("reason") == UNVERIFIED]
    failed = [
        m for m in members if m.get("status") != "DELIVERED" and m.get("reason") != UNVERIFIED
    ]
    reason = safe_failure_code(
        (failed[0] if failed else {}).get("reason") if members else receipt.get("reason")
    )
    status = "UNAVAILABLE" if reason not in {None, CATCHING_UP} else "OBSERVING"
    return {
        "status": status,
        "reason": reason,
        **{key: receipt[key] for key in PROGRESS if key in receipt},
        **(
            {"unverified": unverified[:UNVERIFIED_LIMIT], "unverified_count": len(unverified)}
            if unverified
            else {}
        ),
    }


class NativeUsageReader:
    """Read the bound Sessions' usage when research asks, never on a timer.

    A milestone -- a goal's take or submission, an answer's submission -- or a Goal or Team page
    opening asks. Each Session bound to this workspace whose reading is on is read from its own
    file and the children that file records (`read_session_usage`); no other Session's file is
    opened. One read runs at a time, a Session read within `RECENT` is not read again, and a
    failure is named in `state`, never a refusal of research.
    """

    RECENT = 10.0

    def __init__(self, operations: PortfolioResearchOperations) -> None:
        """Retain the served workspace's binding, Goal and observation owners.

        Args:
            operations: This Host's admitted operation and Goal owners.
        """
        self.operations = operations
        self.workspace = operations.workspace_session.workspace.resolve()
        self._reading = Lock()
        self._read_at: dict[tuple[str, str], tuple[float, dict[str, object]]] = {}
        self._sessions: dict[tuple[str, str], dict[str, object]] = {}
        self._status = "NOT_BOUND"
        self._reason: str | None = None
        self._progress: dict[str, object] = {}

    def state(self) -> dict[str, object]:
        """The last reading's standing, never native lifecycle or completeness proof.

        Returns:
            Bounded status and failure code without file paths, and under `sessions` each
            read Session's own (`own_standing`), keyed as `<host>:<session_id>`.
            Equal identifiers on different hosts never share an observation.
        """
        return {
            "status": "UNAVAILABLE"
            if self._reason is not None and self._reason != CATCHING_UP
            else self._status,
            "reason": self._reason,
            **self._progress,
            "sessions": {f"{host}:{sid}": own for (host, sid), own in self._sessions.items()},
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
        # Admission and the Host's root may name the same binding; differing authority stays.
        return tuple(dict.fromkeys(found))

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

        receipt = read_session_usage(project, binding, publish=publish, goal_id=goal_id)
        self._read_at[(binding.host, binding.session_id)] = (monotonic(), receipt)
        own = self._sessions[binding.host, binding.session_id] = own_standing(receipt)
        self._progress = {key: own[key] for key in PROGRESS if key in own}
        self._status = "OFF" if own["status"] == "OFF" else "OBSERVING"
        self._reason = None if self._status == "OFF" else safe_failure_code(own["reason"])
        return receipt

    def _retain(self, active: tuple[tuple[Path, NativeResearchBinding], ...]) -> None:
        """Discard switched-off or removed bindings from both observation caches."""
        keys = {(binding.host, binding.session_id) for _, binding in active}
        self._sessions = {key: own for key, own in self._sessions.items() if key in keys}
        self._read_at = {key: read for key, read in self._read_at.items() if key in keys}

    def read(self) -> dict[str, object]:
        """Read every Session bound to this workspace whose reading is on, once.

        Returns:
            ``READ`` with one receipt per Session, ``RECENT`` for one read moments ago,
            ``OFF`` or ``NOT_BOUND``; ``BUSY`` while another read runs.
        """
        if not self._reading.acquire(blocking=False):
            return {"status": "BUSY", "sessions": []}
        try:
            self._reason = None
            self._progress = {}
            if not usage_reading(self.workspace):
                # The person's switch: no binding is even looked for.
                clear_usage_checkpoints(self.workspace)
                self._status = "OFF"
                self._progress, self._sessions = {}, {}
                self._read_at.clear()
                return {"status": "OFF", "sessions": []}
            found = self._bindings()
            active = tuple(item for item in found if item[1].usage != "OFF")
            self._retain(active)
            if not active:
                self._status = "NOT_BOUND" if not found else "OFF"
                self._progress = {}
                return {"status": self._status, "sessions": []}
            self._status = "OBSERVING"
            pending_reason, pending_progress = self._reason, {}
            now = monotonic()
            sessions: list[dict[str, object]] = []
            for project, binding in active:
                key = (binding.host, binding.session_id)
                last = self._read_at.get(key)
                if last is not None and now - last[0] < self.RECENT:
                    receipt = last[1]
                    sessions.append({"host": binding.host, "status": "RECENT"})
                else:
                    try:
                        self._progress = {}
                        receipt = self.read_binding(project, binding)
                    except Exception as error:
                        receipt = {
                            "status": "UNAVAILABLE",
                            "reason": owner_failure_code(error)
                            or "native_bridge.lead_usage_read_failed",
                        }
                        self._read_at[key] = (monotonic(), receipt)
                        self._sessions[binding.host, binding.session_id] = own_standing(receipt)
                    sessions.append({"host": binding.host, **receipt})
                # A Session's own failure is the workspace's; its unverified child is not.
                own = self._sessions.get(key) or own_standing(receipt)
                if pending_reason in {None, CATCHING_UP} and own["reason"] is not None:
                    pending_reason = safe_failure_code(own["reason"])
                    pending_progress = {k: own[k] for k in PROGRESS if k in own}
            self._status = (
                "OFF"
                if all(own["status"] == "OFF" for own in self._sessions.values())
                else "OBSERVING"
            )
            if pending_reason is not None:
                self._reason, self._progress = pending_reason, pending_progress
            return {"status": "READ", "sessions": sessions}
        finally:
            self._reading.release()
