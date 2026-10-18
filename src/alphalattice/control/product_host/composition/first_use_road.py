"""The road a first-use sentence takes once the project's first use is complete.

Owned by the Host's goals (the first use's `FIRST_USE` goal): its goal open asks this road
first, and `first-use prepare` relays the answer. A completed first use whose strategy runs forward
turns a sentence for a later date into that workspace's daily update, never a second first use:
the same sentence serves both roads, so the road is decided from state, not from wording.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from alphalattice.control.product_host.composition.research_workspace import (
    ResearchWorkspaceError,
    read_research_workspace_manifest,
)
from alphalattice.control.product_host.publication.goals import GoalStore
from alphalattice.interface.local_application.cli_contract import (
    client_refusal,
    entry,
    join,
    refusal_words,
    shell,
)
from alphalattice.interface.local_application.goals import (
    NEW_RESEARCH_SCOPE,
    GoalDeclaration,
    target_sessions,
)
from alphalattice.interface.local_application.native_bridge import (
    NativeBridgeError,
    NativeResearchBinding,
    session_project,
)

CODE = "local_client.first_use_completed"


def _candidates(selected: Path) -> Iterator[Path]:
    """Wherever this project's first use may stand, the selected workspace first.

    Then each workspace the project's sessions are bound to, and each under its `workspaces/`.
    """
    yield selected
    try:  # the project up from the workspace, else the folder the Host serves from
        project = session_project(selected, None)
    except NativeBridgeError:
        try:
            project = session_project(Path.cwd(), None)
        except NativeBridgeError:
            return
    with_errors = NativeResearchBinding.binding_entries(project)[0]
    yield from (binding.workspace for binding in with_errors)
    folder = project / "workspaces"
    yield from (sorted(p for p in folder.iterdir() if p.is_dir()) if folder.is_dir() else ())


def completed_first_use(selected: Path) -> tuple[Path, str] | None:
    """The workspace whose first use is complete, and the package of its forward strategy.

    None when the project holds no first use that completed with a strategy running forward.
    """
    seen: set[Path] = set()
    for workspace in map(Path.resolve, _candidates(selected)):
        if workspace in seen:
            continue
        seen.add(workspace)
        try:
            manifest = read_research_workspace_manifest(workspace)
            active = [u.strategy_package_id for u in manifest.decision_updates or ()]
            store = GoalStore(workspace / "artifacts", manifest.workspace_id)
            heads = (store.head(goal_id) for goal_id in store.goal_ids()) if active else ()
            first = next((h for h in heads if h and h.declaration.kind == "FIRST_USE"), None)
        except (ResearchWorkspaceError, ValueError, OSError):
            continue  # not a workspace, or one this read cannot open: never this road's
        if first is not None and first.state == "COMPLETE":
            return workspace, active[0]
    return None


def daily_road(selected: Path, declaration: GoalDeclaration) -> dict[str, Any] | None:
    """A first use refused by name where the project's first use is complete; else None.

    The refusal offers the daily update planned for the sentence's date in that workspace.

    Args:
        selected: The workspace this Host serves, where the first use would open.
        declaration: The first use's declaration: its date names the positions' entry session.

    Returns:
        The refusal, naming the workspace and the plan's command; in that same workspace, the
        plan ready to send.
    """
    if declaration.kind != "FIRST_USE" or declaration.scope == NEW_RESEARCH_SCOPE:
        return None  # not a first use, or the person's yes to separate research
    found = completed_first_use(selected) if declaration.target_date else None
    if found is None:
        return None
    workspace, package = found
    plan = {"operation": "RESEARCH_UPDATE_PLAN", "strategy_package_id": package}
    named = target_sessions(declaration.target_date)  # type: ignore[arg-type]
    plan["positions_date"] = str(named["entry_session"])
    words = client_refusal(CODE)
    command = (*entry(workspace), "research-update", "plan", "--package", package)
    same = workspace == selected.resolve()
    return {
        "status": words.outcome,
        "failure_code": CODE,
        "detail": words.detail,
        "next_action": words.next_action,
        "workspace": str(workspace),
        "strategy_package_id": package,
        "next_command": join([*command, "--date", plan["positions_date"]], shell()),
        **({"next_requests": {"plan": plan}} if same else {}),
    }


def first_use_command(workspace: str | Path, objective: str, named: str) -> str:
    """The first use's command for a sentence and date, quoted through the shared CLI grammar."""
    fields = ("first-use", "prepare", "--sentence", objective, "--date", named)
    return join([*entry(workspace, fields, full=True)], shell())


def undated(workspace: Path, declaration: GoalDeclaration, now: datetime) -> dict[str, Any] | None:
    """A first use refused for a date the calendar cannot place; else None.

    The refusal offers the nearest date's command. Any other goal gets None.
    """
    if declaration.kind != "FIRST_USE" or declaration.target_date is None:
        return None
    dated = target_sessions(declaration.target_date, now, for_preparation=True)
    if not (code := dated.get("failure_code")):
        return None
    nearest = str(dated["nearest_date"])
    return {
        "status": "REFUSED",
        "failure_code": code,
        "target_date": dated["target_date"],
        "nearest_date": dated["nearest_date"],
        "next_command": first_use_command(workspace, declaration.objective, nearest),
        **refusal_words(str(code) + ":" + nearest),
    }
