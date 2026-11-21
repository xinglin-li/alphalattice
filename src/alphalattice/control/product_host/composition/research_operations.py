"""Actor-neutral research operations, loaded at their first request."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import ValidationError

from alphalattice.control.product_host.composition.plain_refusals import (
    explain,
    invalid_plan_document,
    refused,
)
from alphalattice.control.product_host.composition.portfolio_research_operations import (
    _PREPARATION_IN_FLIGHT,
)
from alphalattice.control.product_host.data_preparation.application import (
    WorkspacePreparationCommand,
)
from alphalattice.control.task_control.contracts import TaskLifecycle
from alphalattice.interface.local_application.failure_codes import (
    located_failure,
    owner_failure_code,
)

if TYPE_CHECKING:
    from alphalattice.interface.local_application import portfolio_research as requests

    from . import portfolio_research_operations as host


def execute(call: host.WorkspaceOperationCall) -> dict[str, object] | None:
    """Dispatch a workspace research, preparation or experiment operation."""
    operations, request, caller = call
    match request.operation:
        case "RESEARCH_STRATEGY_CONTROLS":
            return operations.research_strategies.controls()
        case (
            "RESEARCH_STRATEGY_PLAN"
            | "RESEARCH_STRATEGY_PREPARE"
            | "RESEARCH_STRATEGY_READBACK"
            | "RESEARCH_STRATEGY_INSTALL"
        ):
            return _strategy(operations, request, caller=caller)
        case "MODEL_TRAINING_INPUT_PLAN":
            assert request.research_input_id is not None and request.component_id is not None
            try:
                return operations.model_training_inputs.plan(
                    request.research_input_id,
                    request.input_binding_hash,
                    request.component_id,
                    **(
                        {"model_lifecycle": request.model_lifecycle}
                        if request.model_lifecycle
                        else {}
                    ),
                )
            except ValueError as error:
                code = owner_failure_code(error)
                if code != "model_training.component_selection_invalid":
                    raise
                return {
                    **refused(code),
                    "next_requests": {
                        "valid_values": {"operation": "RESEARCH_STRATEGY_CONTROLS"},
                    },
                }
        case "MODEL_TRAINING_INPUT_PREPARE":
            return _training_input(operations, request, caller=caller)
        case "MODEL_TRAINING_INPUT_READBACK":
            assert request.task_id is not None
            return operations.model_training_inputs.readback(request.task_id)
        case "RESEARCH_INPUTS":
            # Discovery isolates unreadable publications; exact selection stays strict.
            return operations.input_capture.revisions.versions()
        case "RESEARCH_INPUT_PLAN":
            assert request.research_input_id is not None
            return operations.input_capture.plan(request.research_input_id)
        case "RESEARCH_INPUT_CONFIRM":
            return _input_confirm(operations, request, caller=caller)
        case "RESEARCH_INPUT_READBACK":
            assert request.task_id is not None
            return operations.input_capture.readback(request.task_id)
        case "WORKSPACE_PREPARE_READBACK":
            return operations._preparation_readback(request.task_id)
        case "WORKSPACE_PREPARE_PLAN":
            return operations.goals.prepare(request, caller, operations.preparation.plan)
        case "WORKSPACE_PREPARE_CONFIRM":
            return _prepare_confirm(operations, request, caller=caller)
    return None


def _strategy(
    operations: host.PortfolioResearchOperations,
    request: requests.PortfolioResearchOperationRequest,
    caller: requests.OperationCaller,
) -> dict[str, object]:
    try:
        if request.operation == "RESEARCH_STRATEGY_PLAN":
            assert request.experiment_document is not None
            return operations.research_strategies.plan(request.experiment_document)
        if request.operation == "RESEARCH_STRATEGY_PREPARE":
            assert request.experiment_plan_hash is not None
            return operations.research_strategies.prepare(
                request.experiment_plan_hash, caller=caller, dispatcher=operations.dispatcher
            )
        assert request.task_id is not None
        if request.operation == "RESEARCH_STRATEGY_INSTALL":
            return operations._install(request.task_id)
        return operations.research_strategies.readback(request.task_id)
    except ValidationError as error:
        return invalid_plan_document(error, request, operations.experiments.manifest)
    except (ValueError, OSError, KeyError) as error:
        if (
            str(error) == "research_strategy.preview_required"
            and request.experiment_plan_hash is not None
        ):
            return {
                **refused("research_strategy.preview_required"),
                "next_requests": operations.research_strategies.replan_requests(
                    request.experiment_plan_hash
                ),
            }
        refusal: dict[str, object] = {
            "status": "REFUSED",
            **located_failure(error, "research_strategy.refused"),
        }
        if str(refusal.get("failure_code", "")).startswith(
            (
                "research_strategy.risk_parent_support_incomplete",
                "research_strategy.risk_history_insufficient",
            )
        ):
            recovery = operations.research_strategies.controls(
                (request.experiment_document or {}).get("input_binding_hash")
            )
            refusal["risk_windows"] = recovery["risk_windows"]
            refusal["next_requests"] = recovery["next_requests"]
        return refusal


def _training_input(
    operations: host.PortfolioResearchOperations,
    request: requests.PortfolioResearchOperationRequest,
    caller: requests.OperationCaller,
) -> dict[str, object]:
    assert request.experiment_plan_hash is not None
    try:
        return operations.model_training_inputs.prepare(
            request.experiment_plan_hash, caller=caller, dispatcher=operations.dispatcher
        )
    except ValueError as error:
        if str(error) != "model_training.preview_required":
            raise
        return {
            **refused("model_training.preview_required"),
            "next_requests": operations.model_training_inputs.replan_requests(
                request.experiment_plan_hash
            ),
        }


def _input_confirm(
    operations: host.PortfolioResearchOperations,
    request: requests.PortfolioResearchOperationRequest,
    caller: requests.OperationCaller,
) -> dict[str, object]:
    assert request.research_input_plan_hash is not None
    try:
        return operations.input_capture.confirm(
            request.research_input_plan_hash, caller=caller, dispatcher=operations.dispatcher
        )
    except ValueError as error:
        if str(error) != "research_input.preview_required":
            raise
        return {
            **refused("research_input.preview_required"),
            "next_requests": operations.input_capture.replan_requests(
                request.research_input_plan_hash
            ),
        }


def _prepare_confirm(
    operations: host.PortfolioResearchOperations,
    request: requests.PortfolioResearchOperationRequest,
    caller: requests.OperationCaller,
) -> dict[str, object]:
    assert request.preparation_plan_hash is not None
    try:
        operations.preparation.require_confirmation_caller(
            request.preparation_plan_hash,
            caller=caller,
            grant_hash=request.data_issue_grant_hash,
            **operations.goals.preparation_scope(),
        )
    except ValueError as error:
        if str(error) != "workspace_preparation.preview_stale":
            raise
        return {
            **refused("workspace_preparation.preview_stale"),
            "next_requests": {"replan": {"operation": "WORKSPACE_PREPARE_PLAN"}},
        }
    previous = next(
        (
            t
            for t in operations.preparation.tasks()
            if t.input.payload["plan"]["plan_hash"] == request.preparation_plan_hash
            and t.lifecycle.value != "CANCELLED"
        ),
        None,
    )
    if previous is not None:
        tid = str(previous.task_id)
        reused = {
            TaskLifecycle.SUCCEEDED: ("REUSED_EXACT", None, tid),
            **dict.fromkeys(_PREPARATION_IN_FLIGHT, ("REUSED_IN_FLIGHT", tid)),
        }.get(previous.lifecycle)
        if reused is not None:
            return dict(zip(("status", "task_id", "publication_task_id"), reused, strict=False))
    sent = operations.dispatcher.submit(
        WorkspacePreparationCommand(
            operations.preparation,
            request.preparation_plan_hash,
            caller=caller,
            grant_hash=request.data_issue_grant_hash,
        )
    )
    return {
        **operations._run_answer(sent),
        **explain(sent.refusal_detail or "", **operations._network()),
        **{
            None: {},
            "workspace_preparation.preview_required": {
                "next_requests": {"replan": {"operation": "WORKSPACE_PREPARE_PLAN"}}
            },
        }.get(sent.refusal_detail, {}),
    }
