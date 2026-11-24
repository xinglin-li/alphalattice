"""Actor-neutral data operations, loaded at their first request."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from alphalattice.control.product_host.composition.plain_refusals import explain
from alphalattice.control.product_host.maintenance.data_update import WorkspaceDataUpdateCommand
from alphalattice.interface.local_application.failure_codes import located_failure

if TYPE_CHECKING:
    from alphalattice.interface.local_application import portfolio_research as requests

    from . import portfolio_research_operations as host
    from .operation_route_families import WorkspaceOperationCall


def execute(
    call: WorkspaceOperationCall[host.PortfolioResearchOperations],
) -> dict[str, object] | None:
    """Dispatch a workspace data and storage operation."""
    operations, request, caller = call.operations, call.request, call.caller
    match request.operation:
        case "DATA_ISSUES":
            return cast(
                dict[str, object],
                operations.data_issues.readback(
                    limit=request.history_limit or 25,
                    cursor=request.history_cursor,
                    delegated_by=operations._data_decisions(caller),
                ),
            )
        case "DATA_ISSUE_REVOKE":
            assert request.data_issue_grant_hash is not None
            return cast(
                dict[str, object],
                operations.data_issues.revoke(
                    grant_hash=request.data_issue_grant_hash, caller=caller
                ),
            )
        case "DATA_ISSUE_CONFIRM" | "DATA_ISSUE_PREVIEW" | "DATA_ISSUE_DELEGATE":
            return cast(
                dict[str, object],
                operations.data_issues.operate(request, caller, operations._data_decisions(caller)),
            )
        case "STORAGE_CAP_SHOW":
            return operations.storage.cap_readback()
        case "STORAGE_CAP_SET":
            assert request.storage_cap_bytes is not None
            return operations.storage.set_cap(request.storage_cap_bytes, caller=caller)
        case "STORAGE_READBACK":
            return operations.storage.readback()
        case "STORAGE_PLAN":
            return operations.storage.plan()
        case "STORAGE_CONFIRM":
            assert request.storage_plan_hash is not None
            return operations.storage.confirm(request.storage_plan_hash, caller=caller)
        case "STORAGE_PIN":
            assert request.input_binding_hash is not None and type(request.input_pinned) is bool
            return operations.storage.pin(
                request.input_binding_hash, pinned=request.input_pinned, caller=caller
            )
        case "STORAGE_EVIDENCE_REBUILD":
            assert request.evidence_index_id is not None
            return operations.storage.rebuild_evidence_index(
                request.evidence_index_id, caller=caller
            )
        case "DATA_CHANGE_CONFIRM":
            return _data_confirm(operations, request, caller=caller)
        case "DATA_UPDATE_PLAN" | "DATA_UPDATE_RUN" | "DATA_UPDATE_READBACK":
            return _data_update(operations, request, caller=caller)
    return None


def _data_confirm(
    operations: host.PortfolioResearchOperations,
    request: requests.PortfolioResearchOperationRequest,
    caller: requests.OperationCaller,
) -> dict[str, object]:
    if operations.data_update is None or request.update_plan_hash is None:
        raise ValueError("workspace_data_update.not_configured")
    admitted = operations.data_update.confirm(request.update_plan_hash, caller=caller)
    return {
        "status": "APPROVED",
        "task_id": str(admitted.task_id),
        "lifecycle": admitted.lifecycle,
        "next_action": "DATA_UPDATE_RUN",
        "next_requests": {
            "run": {
                "operation": "DATA_UPDATE_RUN",
                "update_plan_hash": request.update_plan_hash,
            }
        },
    }


def _data_update(
    operations: host.PortfolioResearchOperations,
    request: requests.PortfolioResearchOperationRequest,
    caller: requests.OperationCaller,
) -> dict[str, object]:
    try:
        if operations.data_update is None:
            raise ValueError("workspace_data_update.not_configured")
        if request.operation == "DATA_UPDATE_PLAN":
            return operations.data_update.plan(recovery_task_id=request.recovery_task_id)
        if request.operation == "DATA_UPDATE_READBACK":
            return operations._data_update_readback(request.task_id)
        assert request.update_plan_hash is not None
        plan = operations.data_update.prepare(request.update_plan_hash)
        reused = operations.data_update.reusable(plan)
        if reused is not None:
            # follow the Task that sealed the reused update.
            sealed = operations.data_update.receipt_task(reused)
            return {
                "status": "REUSED_EXACT",
                "task_id": None,
                **(
                    {}
                    if sealed is None
                    else {
                        "publication_task_id": str(sealed),
                        "next_requests": {
                            "read": {
                                "operation": "DATA_UPDATE_READBACK",
                                "task_id": str(sealed),
                            }
                        },
                    }
                ),
                "receipt": reused.model_dump(mode="json"),
            }
        submitted = operations.dispatcher.submit(
            WorkspaceDataUpdateCommand(operations.data_update, plan)
        )
        # Read the Task's current state after any admitted reopening.
        answer = operations._run_answer(submitted, plan)
        return {
            **answer,
            # A refused admission's words, a provider's wait among them.
            **(
                explain(submitted.refusal_detail, **operations._network())
                if submitted.refusal_detail
                else {}
            ),
            # The update it started, read by its Task.
            **(
                {
                    "next_requests": {
                        **cast(dict[str, Any], answer.get("next_requests") or {}),
                        "show": {
                            "operation": "DATA_UPDATE_READBACK",
                            "task_id": str(submitted.task_id),
                        },
                    }
                }
                if submitted.task_id
                else {}
            ),
        }
    except FileNotFoundError:
        return {
            "status": "REFUSED",
            "failure_code": "workspace_data_update.artifact_missing",
        }
    except (ValueError, KeyError) as error:
        update_refusal: dict[str, object] = {
            "status": "REFUSED",
            **located_failure(error, "workspace_data_update.refused"),
        }
        if update_refusal["failure_code"] == "workspace_data_update.plan_required":
            # A plan past its hour or never kept here: plan the update again.
            update_refusal["next_requests"] = {"replan": {"operation": "DATA_UPDATE_PLAN"}}
        return update_refusal


def route(call: WorkspaceOperationCall[host.PortfolioResearchOperations]) -> dict[str, object]:
    """Serve a data family operation."""
    result = execute(call)
    assert result is not None
    return result
