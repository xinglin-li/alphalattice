"""Find authenticated Hosts without coupling the transport to the native bridge."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from itertools import islice
from pathlib import Path
from typing import Any, Final

from alphalattice.interface.local_application.cli_contract import (
    ClientRefusal,
    client_failure,
    worded,
)
from alphalattice.interface.local_application.failure_codes import LocalResearchClientError

WORKSPACE_SCAN_LIMIT: Final = 64
"""Maximum direct project workspaces checked; overflow is reported as partial."""


def serving_workspaces(project: Path, make_client: Callable[..., Any]) -> dict[str, Any]:
    """Find authenticated Hosts under this project's immediate workspaces, without switching."""
    folder = project.resolve() / "workspaces"
    try:
        with os.scandir(folder) as entries:
            candidates = list(islice(entries, WORKSPACE_SCAN_LIMIT + 1))
    except FileNotFoundError:
        return {"status": "ABSENT", "workspaces": [], "unverified": 0}
    except OSError as error:
        return {"status": "UNREADABLE", "workspaces": [], "reason": type(error).__name__}
    found, unverified = [], 0
    for candidate in candidates[:WORKSPACE_SCAN_LIMIT]:
        try:
            workspace = Path(candidate.path)
            if not candidate.is_dir(follow_symlinks=False) or workspace.is_junction():
                continue
            if not (workspace / "research-workspace.json").is_file():
                continue
            host = make_client(workspace, timeout=1)
            response = host.request(
                {"operation": "OPERATION_LIST", "schema_operation": "WORKSPACE_SHOW"}
            )
            if response.get("status") == "REFUSED" or response.get("refused"):
                unverified += 1
            else:
                found.append(str(workspace.resolve()))
        except (OSError, LocalResearchClientError):
            unverified += 1
    return {
        "status": "PARTIAL" if len(candidates) > WORKSPACE_SCAN_LIMIT else "CHECKED",
        "workspaces": sorted(found),
        "unverified": unverified,
    }


def serving_workspace_notice(project: Path, make_client: Callable[..., Any]) -> str:
    """Disclose discovered Hosts and any incomplete workspace scan."""
    result = serving_workspaces(project, make_client)
    notices = []
    if result["workspaces"]:
        notices.append(
            str(worded("cli.serving_workspaces")).replace(
                "{subject}", ", ".join(result["workspaces"])
            )
        )
    if result["status"] in {"PARTIAL", "UNREADABLE"}:
        notices.append(
            str(worded("cli.serving_workspaces_incomplete")).replace(
                "{subject}", str(result.get("reason") or result["status"])
            )
        )
    return "" if not notices else " " + " ".join(notices)


def failure_with_hosts(
    code: str,
    document: Mapping[str, Any],
    cause: BaseException | None,
    request_timeout: float,
    answered: bool,
    workspace: Path,
    make_client: Callable[..., Any],
) -> tuple[ClientRefusal, dict[str, Any]]:
    """Add authenticated Host discovery to the existing no-Host refusal."""
    refusal, facts = client_failure(code, document, cause, request_timeout, answered, workspace)
    if code == "local_client.service_not_running":
        refusal = refusal._replace(
            detail=refusal.detail + serving_workspace_notice(Path.cwd(), make_client)
        )
    return refusal, facts
