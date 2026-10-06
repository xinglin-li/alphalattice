"""Whether this workspace may reach the network, and what decided it (C1 rule 7).

Network access is a workspace control, typed and visible: ``NETWORK_ACCESS`` reads it and names
what decided it, ``NETWORK_ACCESS_SET`` sets it, and every refusal for want of it can say why.
The operator's ``ALPHALATTICE_NETWORK_DISABLED=1`` forces the process offline (probes and QA
scenes rely on it); nothing else in the environment opens it. A person sets the control in
Settings; ``=0`` opened it before the control had a page and is not read any more (V13).
A first-use goal's agent sets it under the person's delegation (OP19): that setting carries its
delegation and its end (``until``), and reads closed after it with no write (V452).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from alphalattice.control.workspace_runtime.content_store import replace_shared_file
from alphalattice.kernel.shared_kernel.environment import (
    OFFLINE_SWITCH,
    offline,
    running_held_offline,
)

NETWORK_SWITCH = OFFLINE_SWITCH
CONTROL_PATH = Path("runtime") / "network-access.json"
_SCHEMA = "network-access"


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

    def body(self) -> dict[str, object]:
        """Render effective permission and the available typed workspace-control action.

        Returns:
            Network-access read model; held-offline decisions expose no set request.
        """
        words = {
            "RUN_HELD_OFFLINE": "A research run holds its reads offline while it runs. "
            "Wait for that run to finish; changing the workspace control cannot lift its hold.",
            "OPERATOR_OFFLINE_SWITCH": f"{NETWORK_SWITCH}=1 keeps this process offline. "
            "Ask the person who starts the Host to remove that switch from its launch "
            "environment and restart the idle Host; workspace settings cannot lift it.",
            "WORKSPACE_CONTROL": "This workspace's network control decides.",
            "DEFAULT": "No workspace control is set, so the network stays off.",
        }
        return {
            "status": "NETWORK_ACCESS",
            "network_allowed": self.allowed,
            "decided_by": self.decided_by,
            "detail": words[self.decided_by],
            "next_action": (
                "RESTART_WITHOUT_OPERATOR_OFFLINE_SWITCH"
                if self.decided_by == "OPERATOR_OFFLINE_SWITCH"
                else "WAIT_FOR_THE_OFFLINE_RUN_TO_FINISH"
                if self.decided_by == "RUN_HELD_OFFLINE"
                else "RETRY_THE_REFUSED_STEP"
                if self.allowed
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
        }


def _control(workspace: Path, now: datetime) -> tuple[bool | None, dict[str, str] | None]:
    """The control's setting and, when a delegation set it, that delegation and its end."""
    path = workspace / CONTROL_PATH
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, None
    except (OSError, ValueError):
        return False, None  # An unreadable control fails closed.
    # Anything but the two mappings the writer writes reads as closed (OP5, V271): a person's
    # (version 1), or a delegation's, open only until its end (version 2, OP19).
    keys = {1: {"schema", "version", "network_enabled"}}
    keys[2] = keys[1] | {"delegation", "until"}
    if (
        not isinstance(record, dict)
        or record.get("schema") != _SCHEMA
        or type(record.get("version")) is not int
        or set(record) != keys.get(record["version"])
        or not isinstance(record["network_enabled"], bool)
    ):
        return False, None
    if record["version"] == 1:
        return bool(record["network_enabled"]), None
    try:
        until = datetime.fromisoformat(str(record["until"]))
    except ValueError:
        return False, None
    if until.tzinfo is None or not isinstance(record["delegation"], str):
        return False, None
    set_by = {"delegation": record["delegation"], "until": until.isoformat()}
    return bool(record["network_enabled"]) and now < until, set_by


def network_access(
    workspace: Path | None = None,
    environment: Mapping[str, str] | None = None,
    now: datetime | None = None,
) -> NetworkAccess:
    """What decides network access for ``workspace``, the operator's offline switch first.

    Who set the workspace's control is named whatever decides, a hold or the switch included:
    it is a fact about the control (OP19, U70).
    """
    control, set_by = (
        (None, None) if workspace is None else _control(workspace, now or datetime.now(UTC))
    )
    if running_held_offline():
        return NetworkAccess(False, "RUN_HELD_OFFLINE", set_by)
    if offline(environment):
        return NetworkAccess(False, "OPERATOR_OFFLINE_SWITCH", set_by)
    if control is not None:
        return NetworkAccess(control, "WORKSPACE_CONTROL", set_by)
    return NetworkAccess(False, "DEFAULT")


def set_network_access(
    workspace: Path,
    *,
    enabled: bool,
    delegation: str | None = None,
    until: datetime | None = None,
) -> NetworkAccess:
    """Atomically persist the typed workspace setting and read effective permission.

    Args:
        workspace: Physical workspace root containing the control or writer-lock file.
        enabled: Requested workspace network setting; offline holds still take precedence.
        delegation: The person's delegation that sets it (a first-use goal), with ``until``.
        until: When that delegation's setting ends; after it the control reads closed.

    Returns:
        Effective access after applying higher-priority run and operator offline holds.
    """
    record: dict[str, object] = {"schema": _SCHEMA, "version": 1, "network_enabled": enabled}
    if delegation is not None and until is not None:
        record.update(version=2, delegation=delegation, until=until.isoformat())
    path = workspace / CONTROL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(".json.tmp")
    staged.write_text(json.dumps(record), "utf-8")
    replace_shared_file(staged, path)  # every request reads the control (V477)
    return network_access(workspace)


__all__ = [
    "CONTROL_PATH",
    "NETWORK_SWITCH",
    "NetworkAccess",
    "network_access",
    "set_network_access",
]
