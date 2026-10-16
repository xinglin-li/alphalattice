"""Whether this workspace may reach the network, and what decided it (C1 rule 7).

Network access is a workspace control, typed and visible: ``NETWORK_ACCESS`` reads it and names
what decided it, ``NETWORK_ACCESS_SET`` sets it, and every refusal for want of it can say why.
The operator's ``ALPHALATTICE_NETWORK_DISABLED=1`` forces the process offline (probes and QA
scenes rely on it); nothing else in the environment opens it. A person sets the control in
Settings; ``=0`` opened it before the control had a page and is not read any more (V13).
A first-use goal's agent sets it under the person's delegation (OP19): that setting carries its
delegation and its end (``until``), and reads closed after it with no write (V452). The person's
own setting, theirs or their yes an agent relayed, always wins: a delegation never writes over it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final, Literal

from alphalattice.control.workspace_runtime.content_store import replace_shared_file
from alphalattice.kernel.shared_kernel.environment import (
    OFFLINE_SWITCH,
    offline,
    running_held_offline,
)

NETWORK_SWITCH = OFFLINE_SWITCH
CONTROL_PATH = Path("runtime") / "network-access.json"
_SCHEMA = "network-access"
_USE: Final = {
    "PERSON": "AlphaLattice may use the network for this step: you turned it on in this "
    "workspace's settings.",
    "RELAYED_YES": "AlphaLattice may use the network for this step, under your yes for daily "
    "updates; say stop to turn it off.",
    "FIRST_USE_DELEGATION": "AlphaLattice may use the network for this step, under the first "
    "use's delegation, which ends with its goal.",
}


@dataclass(frozen=True, slots=True)
class NetworkAccess:
    """Effective network permission and the rule that decided it.

    Attributes:
        allowed: Whether a product network read may proceed.
        decided_by: Winning run hold, operator switch, workspace control or closed default.
    """

    allowed: bool
    decided_by: Literal[
        "RUN_HELD_OFFLINE", "OPERATOR_OFFLINE_SWITCH", "WORKSPACE_CONTROL", "DEFAULT"
    ]
    set_by: dict[str, str] | None = None
    """The delegation that set the control and its end, when a person's goal set it."""
    persons: bool = False
    """Whether the person's own setting is the control: theirs, or their yes an agent relayed."""

    def use(self) -> dict[str, str]:
        """Whose setting lets a step use the network, and the one line said each time it does."""
        under = (
            "FIRST_USE_DELEGATION"
            if self.set_by and "delegation" in self.set_by
            else "RELAYED_YES"
            if self.set_by
            else "PERSON"
        )
        return {"under": under, "words": _USE[under]}

    def body(
        self, *, for_refusal: bool = False, delegation: str | None = None
    ) -> dict[str, object]:
        """Render effective permission and the available typed workspace-control action.

        Args:
            for_refusal: Whether a network-dependent step has actually been refused.
            delegation: Active authority validated by the calling first-use owner.

        Returns:
            Network-access read model; held-offline decisions expose no set request.
        """
        words = {
            "RUN_HELD_OFFLINE": "A research run holds its reads offline while it runs. "
            "Wait for that run to finish; changing the workspace control cannot lift its hold.",
            "OPERATOR_OFFLINE_SWITCH": (
                f"Tell the person in one line that the operator's {NETWORK_SWITCH}=1 "
                "setting keeps network off and blocks network-dependent work."
                if for_refusal
                else f"{NETWORK_SWITCH}=1 keeps this process offline. "
                "Network-dependent steps refuse; steps with already admitted recorded "
                "or captured sources may proceed. Follow the plan's next action; workspace "
                "settings cannot lift the switch."
            ),
            "WORKSPACE_CONTROL": (
                "The person turned this workspace's network off, and only they turn it on. SEC "
                "acquisition stops before its next request; a data update already fetching "
                "finishes the stage it is in, and `task cancel <task-id>` stops it now."
                if self.persons and not self.allowed
                else "This workspace's network control decides."
            ),
            "DEFAULT": "No workspace control is set, so the network stays off.",
        }
        # After the person's own setting, their stop included, no delegation is offered.
        delegated = (
            delegation is not None
            and self.decided_by in {"DEFAULT", "WORKSPACE_CONTROL"}
            and not self.persons
        )
        return {
            "status": "NETWORK_ACCESS",
            "network_allowed": self.allowed,
            "decided_by": self.decided_by,
            "detail": (
                "The first-use goal delegates the network decision to its agent "
                "for the goal's hours."
                if delegated and not self.allowed
                else words[self.decided_by]
            ),
            "next_action": (
                "STOP_AND_TELL_PERSON_NETWORK_IS_HELD_OFFLINE"
                if self.decided_by == "OPERATOR_OFFLINE_SWITCH" and for_refusal
                else "FOLLOW_THE_PLAN_NEXT_ACTION"
                if self.decided_by == "OPERATOR_OFFLINE_SWITCH"
                else "WAIT_FOR_THE_OFFLINE_RUN_TO_FINISH"
                if self.decided_by == "RUN_HELD_OFFLINE"
                else "RETRY_THE_REFUSED_STEP"
                if self.allowed
                else "SET_NETWORK_UNDER_FIRST_USE_DELEGATION"
                if delegated
                else "ASK_A_PERSON_TO_ALLOW_NETWORK_ACCESS"
            ),
            "next_requests": {
                "set": {
                    "operation": "NETWORK_ACCESS_SET",
                    "network_enabled": not self.allowed,
                }
            }
            if self.decided_by not in {"OPERATOR_OFFLINE_SWITCH", "RUN_HELD_OFFLINE"}
            else {},
            **({"set_by": self.set_by} if self.set_by else {}),
            **({"delegation": delegation} if delegated else {}),
        }


def _control(workspace: Path, now: datetime) -> tuple[bool | None, dict[str, str] | None, bool]:
    """The control's setting, the delegation that set it, and whether it is the person's own."""
    path = workspace / CONTROL_PATH
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, None, False
    except (OSError, ValueError):
        return False, None, False  # An unreadable control fails closed.
    # Anything but the two mappings the writer writes reads as closed (OP5, V271): a person's
    # (version 1), or a delegation's, open only until its end (version 2, OP19), or the
    # person's yes an agent relayed (version 3, OP23).
    keys = {1: {"schema", "version", "network_enabled"}}
    keys[2] = keys[1] | {"delegation", "until"}
    keys[3] = keys[1] | {"relayed_by"}
    if (
        not isinstance(record, dict)
        or record.get("schema") != _SCHEMA
        or type(record.get("version")) is not int
        or set(record) != keys.get(record["version"])
        or not isinstance(record["network_enabled"], bool)
    ):
        return False, None, False
    if record["version"] == 1:
        return bool(record["network_enabled"]), None, True
    if record["version"] == 3:
        return bool(record["network_enabled"]), {"relayed_by": str(record["relayed_by"])}, True
    try:
        until = datetime.fromisoformat(str(record["until"]))
    except ValueError:
        return False, None, False
    if until.tzinfo is None or not isinstance(record["delegation"], str):
        return False, None, False
    set_by = {"delegation": record["delegation"], "until": until.isoformat()}
    return bool(record["network_enabled"]) and now < until, set_by, False


def persons_stop(workspace: Path | None, now: datetime) -> bool:
    """Whether the person's own setting holds the control off: their stop.

    No delegation lifts it, so its way on is their yes.
    """
    if workspace is None:
        return False
    control, _set_by, persons = _control(workspace, now)
    return persons and control is False


def network_access(
    workspace: Path | None = None,
    environment: Mapping[str, str] | None = None,
    now: datetime | None = None,
) -> NetworkAccess:
    """What decides network access for ``workspace``, the operator's offline switch first.

    Who set the workspace's control is named whatever decides, a hold or the switch included:
    it is a fact about the control (OP19, U70). A workspace's control is read on the caller's
    clock, the Host's: a delegation's end is the Host's instant, so the machine's never decides.
    """
    if workspace is None:
        control, set_by, persons = None, None, False
    elif now is None:
        raise TypeError("network_access reads a workspace control on the caller's clock: pass now")
    else:
        control, set_by, persons = _control(workspace, now)
    if running_held_offline():
        return NetworkAccess(False, "RUN_HELD_OFFLINE", set_by, persons)
    if offline(environment):
        return NetworkAccess(False, "OPERATOR_OFFLINE_SWITCH", set_by, persons)
    if control is not None:
        return NetworkAccess(control, "WORKSPACE_CONTROL", set_by, persons)
    return NetworkAccess(False, "DEFAULT")


def set_network_access(
    workspace: Path,
    *,
    enabled: bool,
    delegation: str | None = None,
    until: datetime | None = None,
    relayed_by: str | None = None,
    now: datetime,
) -> NetworkAccess:
    """Atomically persist the typed workspace setting and read effective permission.

    Args:
        workspace: Physical workspace root containing the control or writer-lock file.
        enabled: Requested workspace network setting; offline holds still take precedence.
        delegation: The person's delegation that sets it (a first-use goal), with ``until``.
        until: When that delegation's setting ends; after it the control reads closed.
        relayed_by: The agent session that relayed the person's yes, when one did.
        now: The Host's clock, on which ``until`` was computed and is read back.

    Returns:
        Effective access after applying higher-priority run and operator offline holds; a
        delegation's setting over the person's own is not written, and answers theirs.
    """
    if delegation is not None and _control(workspace, now)[2]:
        return network_access(workspace, now=now)
    record: dict[str, object] = {"schema": _SCHEMA, "version": 1, "network_enabled": enabled}
    if delegation is not None and until is not None:
        record.update(version=2, delegation=delegation, until=until.isoformat())
    elif relayed_by is not None:
        record.update(version=3, relayed_by=relayed_by)
    path = workspace / CONTROL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(".json.tmp")
    staged.write_text(json.dumps(record), "utf-8")
    replace_shared_file(staged, path)  # every request reads the control (V477)
    return network_access(workspace, now=now)


__all__ = [
    "CONTROL_PATH",
    "NETWORK_SWITCH",
    "NetworkAccess",
    "network_access",
    "persons_stop",
    "set_network_access",
]
