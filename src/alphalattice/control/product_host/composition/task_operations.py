"""Actor-neutral task operations, loaded at their first request."""

from __future__ import annotations

from typing import TYPE_CHECKING

from alphalattice.control.product_host.composition.plain_refusals import refused
from alphalattice.control.product_host.composition.portfolio_research_operations import (
    _TASK_PAGE_ROWS,
    _runner_retry_reason,
)
from alphalattice.control.product_host.composition.research_experiments import (
    TASK_KIND as RESEARCH_EXPERIMENT_TASK_KIND,
)
from alphalattice.interface.local_application.cli_contract import REQUEST_PROVENANCE

if TYPE_CHECKING:
    from alphalattice.interface.local_application import portfolio_research as requests

    from . import portfolio_research_operations as host
    from .operation_route_families import WorkspaceOperationCall


def execute(
    call: WorkspaceOperationCall[host.PortfolioResearchOperations],
) -> dict[str, object] | None:
    """Dispatch a Task or supervised-run operation."""
    operations, request, caller = call.operations, call.request, call.caller
    match request.operation:
        case "STATUS":
            assert request.task_id is not None
            result = operations.status(request.task_id, wait_seconds=request.wait_seconds)
            if request.status_section == "storage":
                return {**result, "storage": operations.storage.status()}
            return result
        case "TASK_RECOVERY":
            assert request.task_id is not None
            return operations.recovery_view(request.task_id)
        case "TASK_GUARDIAN":
            return operations.guardian()
        case "TASK_INCIDENTS":
            return operations.supervisor.incidents()
        case "TASK_REMEDIATE":
            return operations.supervisor.remediate(
                request,
                selected_by="USER_COMMAND"
                if REQUEST_PROVENANCE.get() is None
                else "AGENT_PROPOSAL",
            )
        case "RECOVER":
            return _recover(operations, request, caller=caller)
        case "TASKS":
            return operations.tasks(
                agent_session=request.agent_session,
                limit=request.history_limit or _TASK_PAGE_ROWS,
                cursor=request.history_cursor,
            )
        case "CANCEL":
            return _cancel(operations, request, caller=caller)
    return None


def _recover(
    operations: host.PortfolioResearchOperations,
    request: requests.PortfolioResearchOperationRequest,
    caller: requests.OperationCaller,
) -> dict[str, object]:
    assert request.task_id is not None
    before = operations.status(request.task_id)
    stale = operations._stale_confirmation(request, before)
    if stale is not None:
        return stale
    confirmed = request.expected_task_hash
    retry: dict[str, object] | None = None
    if before["lifecycle"] == "BLOCKED":
        # Reopen the confirmed BLOCKED version only for a repairable artifact.
        record = operations.workspace_session.task_control_registry.task(request.task_id)
        raised = _runner_retry_reason(record) is not None
        if record.task_kind != RESEARCH_EXPERIMENT_TASK_KIND and not raised:
            return {**before, "disposition": "NOT_RECOVERY_REQUIRED"}
        if confirmed is None:
            return {
                "status": "REFUSED",
                "failure_code": "local_application.expected_task_hash_required",
                "task_id": before["task_id"],
                "task_record_hash": before["task_record_hash"],
                # The view that offers it bound to this version.
                "next_requests": {
                    "recovery": {
                        "operation": "TASK_RECOVERY",
                        "task_id": str(request.task_id),
                    }
                },
            }
        # Task Control checks the confirmed version again at the write;
        # a Task that moved during plan validation is refused as stale.
        retry = (
            operations._reopen_runner_block(record, confirmed)
            if raised
            else operations.experiments.retry_blocked(
                request.task_id,
                now=operations.dispatcher.clock(),
                expected_task_hash=confirmed,
            )
        )
        if retry.get("failure_code") == "task_control.recovery_version_stale":
            return {
                "status": "REFUSED",
                "failure_code": "local_application.confirmation_stale",
                "task_id": str(retry["task_id"]),
                "lifecycle": retry["lifecycle"],
                "task_record_hash": retry["task_record_hash"],
                "refused_at": "TASK_CONTROL",
            }
        if retry.get("disposition") != "RETRY_ADMITTED":
            return {**before, **retry}
        # Resume confirms the reopened version, the confirmed version's successor.
        confirmed = str(retry["task_record_hash"])
        before = operations.status(request.task_id)
    if before["lifecycle"] not in {"RECOVERY_REQUIRED", "QUEUED"}:
        return {**before, "disposition": "NOT_RECOVERY_REQUIRED"}
    if before["resume_refusal"] is not None:
        return operations._resume_refused(before)
    if operations.recover_task is None:
        return {
            **before,
            **refused(
                "local_application.task_recovery_not_configured",
                task_id=str(request.task_id),
            ),
        }
    resumed = operations.recover_task(request.task_id, expected_task_hash=confirmed)
    if request.task_id not in resumed:
        failure = operations.dispatcher.failure(request.task_id)
        if failure:
            return {
                **operations.status(request.task_id),
                "status": "REFUSED",
                "failure_code": failure,
                "detail": (
                    "The resume was not handed to the worker; the Task's recovery "
                    "view says what stopped it and what may resume it."
                ),
                "next_requests": {
                    "recovery": {
                        "operation": "TASK_RECOVERY",
                        "task_id": str(request.task_id),
                    }
                },
            }
    return {
        **operations.status(request.task_id),
        **({"retry": retry} if retry is not None else {}),
        "resumed_task_ids": [str(v) for v in resumed],
    }


def _cancel(
    operations: host.PortfolioResearchOperations,
    request: requests.PortfolioResearchOperationRequest,
    caller: requests.OperationCaller,
) -> dict[str, object]:
    assert request.task_id is not None
    stale = operations._stale_confirmation(request, operations.status(request.task_id))
    if stale is not None:
        return stale
    return operations.cancel(request.task_id, expected_task_hash=request.expected_task_hash)


def route(call: WorkspaceOperationCall[host.PortfolioResearchOperations]) -> dict[str, object]:
    """Serve a task family operation."""
    result = execute(call)
    assert result is not None
    return result
