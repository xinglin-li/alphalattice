"""Actor-neutral Feature operations, loaded when the Host receives a Feature request."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from alphalattice.control.product_host.composition.plain_refusals import (
    explain,
    invalid_plan_document,
    refused,
)
from alphalattice.control.product_host.research_authoring.feature_research import (
    ResearchFeatureDefinitions,
)
from alphalattice.control.product_host.research_authoring.timing import binding_temporal_scope
from alphalattice.foundation.feature_engine.catalog.service import FeatureCatalogCrudError
from alphalattice.interface.local_application.failure_codes import public_failure

if TYPE_CHECKING:
    from alphalattice.interface.local_application import portfolio_research as requests

    from . import portfolio_research_operations as host
    from .operation_route_families import WorkspaceOperationCall


def execute(
    call: WorkspaceOperationCall[host.PortfolioResearchOperations],
) -> dict[str, object] | None:
    """Dispatch a Feature preparation or trial operation."""
    operations, request, caller = call.operations, call.request, call.caller
    match request.operation:
        case "FEATURE_CATALOG_BUILD":
            assert request.feature_plan_hash is not None
            return operations.feature_builds.build(
                request.feature_plan_hash,
                caller=caller,
                dispatcher=operations.dispatcher,
                preprocess=request.feature_output == "PREPROCESSED_VALUES",
            )
        case "FEATURE_TRIAL" | "FEATURE_TRIAL_READBACK" | "FEATURE_TRIALS":
            return _feature_trial(operations, request, caller=caller)
        case "FEATURE_CATALOG_BUILD_READBACK":
            return _feature_build(operations, request, caller=caller)
        case "FEATURE_CATALOG_CONTROLS" | "FEATURE_CATALOG_PLAN" | "FEATURE_CATALOG_READBACK":
            return _feature_catalog(operations, request, caller=caller)
    return None


def _feature_trial(
    operations: host.PortfolioResearchOperations,
    request: requests.PortfolioResearchOperationRequest,
    caller: requests.OperationCaller,
) -> dict[str, object]:
    try:
        if request.operation == "FEATURE_TRIALS":
            return operations.trials.listing()
        if request.operation == "FEATURE_TRIAL_READBACK":
            assert request.feature_trial_id is not None
            return operations.trials.readback(request.feature_trial_id)
        assert request.feature_plan_hash is not None and request.task_id is not None
        return operations.trials.start(request.feature_plan_hash, request.task_id, caller=caller)
    except ValueError as error:
        code = public_failure(error, "feature_trial.refused")
        return refused(
            code,
            task_id=str(request.task_id or "") or None,
            # A study the trial cannot run against names the way on.
            trial=operations.trials.way_on(request.feature_plan_hash)
            if code == "feature_trial.study_not_from_factor_evidence"
            and request.feature_plan_hash is not None
            else None,
        )


def _feature_build(
    operations: host.PortfolioResearchOperations,
    request: requests.PortfolioResearchOperationRequest,
    caller: requests.OperationCaller,
) -> dict[str, object]:
    assert request.task_id is not None
    build = operations.feature_builds.readback(request.task_id)
    # The readback checked the Task's contract; its input is the build's source.
    binding = str(
        operations.workspace_session.task_control_registry.task(request.task_id).input.payload[
            "input_binding_hash"
        ]
    )
    # What the build's source can claim about time, from its input's Panel.
    return {
        **build,
        "temporal_scope": binding_temporal_scope(
            operations.workspace_session.workspace,
            binding,
            window_start=None,
            window_end=None,
        ),
    }


def _feature_catalog(
    operations: host.PortfolioResearchOperations,
    request: requests.PortfolioResearchOperationRequest,
    caller: requests.OperationCaller,
) -> dict[str, object]:
    definitions = ResearchFeatureDefinitions(
        operations.workspace_session.workspace,
    )
    try:
        if request.operation == "FEATURE_CATALOG_CONTROLS":
            assert request.input_binding_hash is not None
            return {
                **definitions.controls(request.input_binding_hash, request.feature_plan_hash),
                # trial controls precede the factor build.
                **operations.experiments.prerequisites_of(
                    "FEATURE_TRIAL", binding_hash=request.input_binding_hash
                ),
            }
        if request.operation == "FEATURE_CATALOG_PLAN":
            assert request.feature_document is not None
            with operations.workspace_session.mutation_gate.hold():
                plan = definitions.plan(request.feature_document)
            return _with_trial(definitions.readback(plan.plan_hash))
        assert request.feature_plan_hash is not None
        return _with_trial(definitions.readback(request.feature_plan_hash))
    except ValidationError as error:
        return invalid_plan_document(error, request, operations.experiments.manifest)
    except FeatureCatalogCrudError as error:
        return {
            "status": "REFUSED",
            "failure_code": error.failure_code,
            "message": str(error),
        }
    except (ValueError, OSError, KeyError) as error:
        named = request.input_binding_hash or str(
            (request.feature_document or {}).get("input_binding_hash") or ""
        )
        held = [v.binding_hash for v in operations.experiments.manifest.experiment_inputs or ()]
        if isinstance(error, FileNotFoundError) and named and named not in held:
            # refused bindings offer their field and admitted controls.
            return {
                "status": "REFUSED",
                "failure_code": "feature_research.input_binding_unresolved",
                "fields": [["input_binding_hash"]],
                "expected": {"input_binding_hash": held},
                **explain("feature_research.input_binding_unresolved"),
                "next_requests": {
                    **{
                        f"controls:{binding[:12]}": {
                            "operation": "FEATURE_CATALOG_CONTROLS",
                            "input_binding_hash": binding,
                        }
                        for binding in held
                    },
                    "inputs": {"operation": "WORKSPACE_SHOW"},
                },
            }
        return {
            "status": "REFUSED",
            "failure_code": public_failure(
                error, "feature_research.definition_or_source_unavailable"
            ),
        }


def _with_trial(readback: dict[str, Any]) -> dict[str, Any]:
    """A planned feature's readback with its trial offered, the study left to choose.

    What a trial runs against is stated before any build, so an agent holding only Factor
    studies learns it before it builds rather than at the trial's refusal.
    """
    return {
        **readback,
        "trial_baseline": (
            "A completed Alpha study handed off from a Factor study on this input, or the "
            "Portfolio study built on one."
        ),
        "next_requests": {
            **readback["next_requests"],
            "trial": {
                "operation": "FEATURE_TRIAL",
                "feature_plan_hash": readback["plan_hash"],
                "task_id": None,
            },
        },
    }


def route(call: WorkspaceOperationCall[host.PortfolioResearchOperations]) -> dict[str, object]:
    """Serve a feature family operation."""
    result = execute(call)
    assert result is not None
    return result
