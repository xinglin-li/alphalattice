"""Actor-neutral portfolio operations, loaded on request."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, cast

from alphalattice.control.product_host.composition.decision_advancement import (
    DecisionAdvancementCommand,
)
from alphalattice.control.product_host.composition.plain_refusals import explain, refused
from alphalattice.control.product_host.composition.portfolio_research_operations import (
    reused_read,
)
from alphalattice.control.product_host.composition.portfolio_updates import (
    PortfolioUpdateCommand,
    update_refusal,
)
from alphalattice.control.product_host.composition.research_update_automation import dated_plan
from alphalattice.control.product_host.composition.strategy_calibration import (
    StrategyCalibrationCommand,
)
from alphalattice.control.product_host.composition.strategy_scoring import StrategyScoreCommand
from alphalattice.control.product_host.composition.task_kinds import (
    PORTFOLIO_UPDATE_TASK_KIND,
    STRATEGY_CALIBRATION_TASK_KIND,
    STRATEGY_SCORE_TASK_KIND,
)
from alphalattice.control.workspace_runtime.content_store import ContentAddressedStoreError
from alphalattice.interface.local_application.failure_codes import (
    located_failure,
    public_failure,
    safe_failure_code,
)
from alphalattice.interface.local_application.portfolio_research import LocalApplicationError
from alphalattice.investment.portfolio_strategy_lab.application.advancement_task import (
    ADVANCEMENT_TASK_KIND,
)

if TYPE_CHECKING:
    from alphalattice.control.product_host.composition.portfolio_research_operations import (
        PortfolioResearchOperations,
    )
    from alphalattice.interface.local_application.dispatcher import LocalApplicationCommand
    from alphalattice.interface.local_application.portfolio_research import (
        OperationCaller,
        PortfolioResearchOperationRequest,
    )


def recovery_command(operations: PortfolioResearchOperations, kind: str) -> LocalApplicationCommand:
    """Build the Portfolio family's command after recovery selects a Task kind."""
    if kind == ADVANCEMENT_TASK_KIND:
        assert operations.research_updates is not None
        return DecisionAdvancementCommand(operations.research_updates)
    if kind == PORTFOLIO_UPDATE_TASK_KIND:
        assert operations.updates is not None
        return PortfolioUpdateCommand(operations.updates)
    if kind == STRATEGY_CALIBRATION_TASK_KIND:
        assert operations.calibration is not None
        return StrategyCalibrationCommand(operations.calibration)
    assert kind == STRATEGY_SCORE_TASK_KIND and operations.scoring is not None
    return StrategyScoreCommand(operations.scoring)


def execute(
    operations: PortfolioResearchOperations,
    request: PortfolioResearchOperationRequest,
    *,
    caller: OperationCaller,
) -> dict[str, object]:
    """Dispatch a Portfolio study, update or installed result operation."""
    assert operations.service is not None
    match request.operation:
        case "RESEARCH_UPDATE_AUTOMATION_READBACK":
            automation = operations.automation
            assert automation is not None
            return operations._automation_answer(automation.readback())
        case "RESEARCH_UPDATE_AUTOMATION_CONFIGURE":
            assert caller != "SERVICE_AUTOMATION"
            if caller == "EXTERNAL_AUTOMATION":
                # A person turns it on or off (PERSON_ONLY).
                raise ValueError("research_update.human_confirmation_required")
            if (
                type(request.automation_enabled) is not bool
                or request.automation_package_ids is None
            ):
                raise ValueError("research_update.automation_fields_invalid")
            automation = operations.automation
            assert automation is not None
            return operations._automation_answer(
                automation.configure(
                    enabled=request.automation_enabled,
                    package_ids=request.automation_package_ids,
                    chosen_by=caller,
                )
            )
        case "RESEARCH_UPDATE_PLAN" | "RESEARCH_UPDATE_RUN" | "RESEARCH_UPDATE_READBACK":
            try:
                if operations.research_updates is None:
                    raise ValueError("research_update.not_installed")
                if request.operation == "RESEARCH_UPDATE_PLAN":
                    plan, now = operations.research_updates.plan, operations.dispatcher.clock()
                    return dated_plan(plan, request, now)
                if request.operation == "RESEARCH_UPDATE_READBACK":
                    chosen = operations._strategy_task(request)
                    if isinstance(chosen, dict):
                        return chosen
                    return operations._positions(
                        operations._research_update_readback(chosen), caller
                    )
                assert request.update_plan_hash is not None
                plan = operations.research_updates.resuming(request.update_plan_hash)
                reused_task = operations.research_updates.reusable(plan)
                if reused_task is not None:
                    return {
                        "status": "REUSED_EXACT",
                        "task_id": None,
                        "publication_task_id": str(reused_task),
                    }
                pending = operations.research_updates.in_flight(plan)
                if pending is not None:
                    return {
                        "status": "REUSED_IN_FLIGHT",
                        "task_id": str(pending.task_id),
                        "lifecycle": pending.lifecycle,
                    }
                return operations._run_answer(
                    operations.dispatcher.submit(
                        DecisionAdvancementCommand(operations.research_updates, plan)
                    ),
                    plan.data_plan,
                )
            except (ValueError, KeyError, FileNotFoundError) as error:
                body = update_refusal(
                    error,
                    "research_update.refused",
                    installed_packages=tuple(operations._packages),
                    strategy_package_id=request.strategy_package_id,
                )
                if (
                    body["failure_code"] == "portfolio_update.not_installed"
                    and request.strategy_package_id is not None
                    and request.strategy_package_id in operations._packages
                    and operations.activations is not None
                ):
                    body["activation"] = operations._activation_offer(request.strategy_package_id)
                return operations._inputs_way(body)
        case "PORTFOLIO_UPDATE_PLAN" | "PORTFOLIO_UPDATE_RUN" | "PORTFOLIO_UPDATE_READBACK":
            assert operations.updates is not None
            try:
                if request.operation == "PORTFOLIO_UPDATE_PLAN":
                    assert request.strategy_package_id is not None
                    return operations.updates.plan(
                        request.strategy_package_id,
                        request.prepared_input_hash,
                        None
                        if request.observed_through is None
                        else date.fromisoformat(request.observed_through),
                    )
                if request.operation == "PORTFOLIO_UPDATE_READBACK":
                    chosen = operations._strategy_task(request)
                    if isinstance(chosen, dict):
                        return chosen
                    return operations._positions(operations.updates.readback(chosen))
                assert request.update_plan_hash is not None
                plan = operations.updates.prepare(request.update_plan_hash)
                reused = operations.updates.reusable(plan)
                if reused is not None:
                    return {
                        "status": "REUSED_EXACT",
                        "task_id": None,
                        "publication_hash": reused.content_hash,
                        **reused_read(
                            operations.workspace_session.task_control_registry,
                            plan.plan_hash,
                            "PORTFOLIO_UPDATE_READBACK",
                        ),
                    }
                admitted = operations.dispatcher.submit(
                    PortfolioUpdateCommand(operations.updates, plan)
                )
                return {
                    "status": admitted.disposition,
                    "task_id": str(admitted.task_id) if admitted.task_id else None,
                    "lifecycle": admitted.lifecycle,
                    "failure_code": admitted.refusal_detail,
                }
            except FileNotFoundError:
                return refused("portfolio_update.artifact_missing")
            except (ValueError, KeyError) as error:
                return update_refusal(
                    error,
                    "portfolio_update.refused",
                    strategy_package_id=request.strategy_package_id,
                )
        case (
            "STRATEGY_CALIBRATION_PLAN"
            | "STRATEGY_CALIBRATION_RUN"
            | "STRATEGY_CALIBRATION_READBACK"
        ):
            assert operations.calibration is not None
            try:
                if request.operation == "STRATEGY_CALIBRATION_PLAN":
                    assert (
                        request.strategy_package_id is not None
                        and request.score_snapshot_hash is not None
                    )
                    return operations.calibration.plan(
                        request.strategy_package_id, request.score_snapshot_hash
                    )
                if request.operation == "STRATEGY_CALIBRATION_READBACK":
                    chosen = operations._strategy_task(request)
                    if isinstance(chosen, dict):
                        return chosen
                    return operations.calibration.readback(chosen)
                assert request.calibration_plan_hash is not None
                plan = operations.calibration.prepare(request.calibration_plan_hash)
                reused = operations.calibration.reusable(plan)
                if reused is not None:
                    return {
                        "status": "REUSED_EXACT",
                        "task_id": None,
                        # The record its reuse is, by its identity (OP18).
                        "calibration_input_hash": reused.content_hash,
                        "input": reused.model_dump(mode="json"),
                        "readout": reused.readout(),
                        **reused_read(
                            operations.workspace_session.task_control_registry,
                            plan.plan_hash,
                            "STRATEGY_CALIBRATION_READBACK",
                        ),
                    }
                result = operations.dispatcher.submit(
                    StrategyCalibrationCommand(operations.calibration, plan)
                )
                return {
                    "status": result.disposition,
                    "task_id": str(result.task_id) if result.task_id else None,
                    "lifecycle": result.lifecycle,
                    "failure_code": result.refusal_detail,
                }
            except FileNotFoundError:
                return {
                    "status": "REFUSED",
                    "failure_code": "portfolio_calibration.artifact_missing",
                }
            except (ValueError, KeyError) as error:
                if (
                    str(error) == "portfolio_calibration.plan_required"
                    and request.calibration_plan_hash is not None
                ):
                    return {
                        **refused("portfolio_calibration.plan_required"),
                        "next_requests": operations.calibration.replan_requests(
                            request.calibration_plan_hash
                        ),
                    }
                failure = located_failure(error, "portfolio_calibration.refused")
                return {
                    "status": "REFUSED",
                    **failure,
                    **explain(str(failure["failure_code"])),
                }
        case "STRATEGY_SCORE_PLAN" | "STRATEGY_SCORE_RUN" | "STRATEGY_SCORE_READBACK":
            assert operations.scoring is not None
            try:
                if request.operation == "STRATEGY_SCORE_PLAN":
                    assert request.strategy_package_id is not None
                    return operations.scoring.plan(
                        request.strategy_package_id,
                        date.fromisoformat(request.formation_session)
                        if request.formation_session
                        else None,
                        component_id=request.component_id,
                    )
                if request.operation == "STRATEGY_SCORE_READBACK":
                    chosen = operations._strategy_task(request)
                    if isinstance(chosen, dict):
                        return chosen
                    return operations.scoring.readback(chosen)
                assert request.score_plan_hash is not None
                plan = operations.scoring.prepare(request.score_plan_hash)
                reused = operations.scoring.reusable(plan)
                if reused is not None:
                    return {
                        "status": "REUSED_EXACT",
                        "task_id": None,
                        # The score it reuses, which a calibration plan takes (OP18).
                        "score_snapshot_hash": reused.snapshot_hash,
                        "score": reused.model_dump(mode="json"),
                        **reused_read(
                            operations.workspace_session.task_control_registry,
                            plan.plan_hash,
                            "STRATEGY_SCORE_READBACK",
                        ),
                    }
                result = operations.dispatcher.submit(
                    StrategyScoreCommand(operations.scoring, plan)
                )
                return {
                    "status": result.disposition,
                    "task_id": str(result.task_id) if result.task_id else None,
                    "lifecycle": result.lifecycle,
                    "failure_code": result.refusal_detail,
                }
            except FileNotFoundError:
                return {"status": "REFUSED", "failure_code": "strategy_score.artifact_missing"}
            except (ValueError, KeyError) as error:
                if (
                    str(error) == "strategy_score.plan_required"
                    and request.score_plan_hash is not None
                ):
                    return {
                        **refused("strategy_score.plan_required"),
                        "next_requests": operations.scoring.replan_requests(
                            request.score_plan_hash
                        ),
                    }
                failure = (
                    # The Alpha store says it in a sentence, inside the study's closure: a
                    # stored artifact missing, stale or tampered.
                    {"failure_code": "strategy_score.alpha_artifact_unreadable"}
                    if type(error).__name__ == "AlphaDevelopmentArtifactReadbackError"
                    and safe_failure_code(str(error)) is None
                    else located_failure(error, "strategy_score.refused")
                )
                return {
                    "status": "REFUSED",
                    **failure,
                    **explain(str(failure["failure_code"])),
                }
        case "CONTROLS" | "PLAN" | "RUN":
            try:
                if request.operation == "CONTROLS":
                    return operations.controls(request.strategy_package_id)
                assert request.spec is not None
                return (
                    operations.plan(request.spec)
                    if request.operation == "PLAN"
                    else operations.run(request.spec)
                )
            except LocalApplicationError as error:
                code = str(error)
                if code not in {
                    "strategy_book.strategy_package_required",
                    "research_workspace.strategy_not_installed",
                } and not code.startswith("local_application.strategy_package_not_installed:"):
                    raise
                return refused(code, installed_packages=operations._packages)
        case "RESULTS":
            return operations.results(request.task_id)
        case "REPORT":
            assert request.result_hash is not None
            try:
                return operations.report(request.result_hash, request.portfolio_session)
            except ContentAddressedStoreError as error:
                rows = cast("list[dict[str, object]]", operations.results()["results"])
                listed = {str(v["result_hash"]) for v in rows}
                return refused(
                    "portfolio_research.result_not_found"
                    if (
                        str(error).partition(":")[0] == "content_store.artifact_missing"
                        and request.result_hash not in listed
                    )
                    else public_failure(error, "portfolio_research.report_unavailable")
                )
            except LocalApplicationError as error:
                if str(error) != "portfolio_application.session_outside_report":
                    raise
                return refused(
                    public_failure(error, "portfolio_research.report_unavailable"),
                    result_hash=request.result_hash,
                )
        case "COMPARE":
            assert request.left_result_hash is not None
            assert request.right_result_hash is not None
            return operations.compare(request.left_result_hash, request.right_result_hash)
        case "FREEZE":
            assert request.result_hash is not None
            return operations.freeze(request.result_hash)
        case "FINALIZATION":
            assert request.candidate_hash is not None
            return operations.finalization(request.candidate_hash)
        case "EXPORT":
            assert request.result_hash is not None
            return operations.export(request.result_hash)
    raise ValueError("portfolio_research.operation_unknown")
