"""A stopped Task's durable declaration continues through the real CLI (V615)."""

import gc
import json
import shutil
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from alphalattice.control.product_host.composition.research_workspace import (
    manifest_fields_hash,
    publish_research_workspace_manifest,
)
from alphalattice.control.product_host.data_preparation.research_strategy import (
    PLAN_FIELDS,
    ResearchStrategyPlan,
    implementation_hash,
)
from alphalattice.control.product_host.research_authoring.frozen_portfolio import (
    FrozenPortfolioPreparationRequest,
)
from alphalattice.control.task_control.contracts import (
    ResearchGoal,
    ResearchPlan,
    TaskInputEnvelope,
    WorkItemDefinition,
)
from alphalattice.control.task_control.registry import LEDGER_REBUILT_DETAIL, LEDGER_REBUILT_NEXT
from alphalattice.evidence.alternative_evidence.runtime.task_adapter import (
    AlternativeEvidenceDocumentTaskAdapter,
)
from alphalattice.interface.local_application.cli import main
from tests.portfolio_strategy_lab.strategy_dates_support import date_host


@pytest.mark.parametrize("view", ["full", "compact"])
def test_a_ledger_rebuilt_preparation_keeps_its_declaration_in_every_way_on(
    tmp_path, capsys, monkeypatch, view
):
    """regression (V615): a genuine lost ledger is rebuilt from the owner's admitted
    request. Recovery/status/readback preserve its declaration after preview loss. The CLI
    saved-answer and offered-action routes reach the public planner with those exact values.
    The planner port records only the declaration; this continuation test runs no studies.
    """
    workspace = tmp_path / "workspace"
    try:
        with date_host(workspace) as live:
            publish_research_workspace_manifest(workspace, live.workspace_manifest)
            owner = live.operations.research_strategies
            declaration = FrozenPortfolioPreparationRequest(
                input_binding_hash="b" * 64,
                alpha_task_ids=(uuid4(), uuid4()),
                risk_task_id=uuid4(),
                unavailable_return_policy="quarantine_listings",
            )
            plan = ResearchStrategyPlan.create(
                workspace_manifest_hash=manifest_fields_hash(live.workspace_manifest, PLAN_FIELDS),
                request=declaration,
                implementation_hash=implementation_hash(),
            )
            admitted = owner.admit(plan, "HUMAN")
            registry = live.session.task_control_registry
            database = registry.database_path
            for path in database.parent.glob(f"{database.name}*"):
                path.unlink()
            type(registry)(database, gate=live.session.mutation_gate)
            (task,) = registry.rebuild_from_requests(
                observed_at=datetime(2026, 10, 3, 13, tzinfo=UTC)
            )
            assert task.task_id == admitted.task_id
            assert task.failure_code == "task_control.ledger_rebuilt"
            assert not list((workspace / "runtime/plan-previews/research-strategy").glob("*.json"))
            expected = {
                "operation": "RESEARCH_STRATEGY_PLAN",
                "experiment_document": declaration.model_dump(mode="json"),
            }
            for read in (
                live.operations.recovery_view(task.task_id),
                live.operations.status(task.task_id),
                owner.readback(task.task_id),
            ):
                assert read["next_requests"]["replan"] == expected
            recovery = live.operations.recovery_view(task.task_id)
            assert recovery["stop"]["detail"] == LEDGER_REBUILT_DETAIL
            assert recovery["verified_stage_count"] == 0
            assert all(stage["evidence"] == [] for stage in recovery["stages"])
            offered = next(a for a in recovery["actions"] if a["action"] == "REPLAN")
            assert offered["available"] and not offered["admits"]
            assert offered["expected_effect"] == LEDGER_REBUILT_NEXT
            row = next(
                r for r in live.operations.tasks()["tasks"] if r["task_id"] == str(task.task_id)
            )
            assert row["detail"] == LEDGER_REBUILT_DETAIL
            assert row["stop_next"] == LEDGER_REBUILT_NEXT
            assert live.operations.status(task.task_id)["detail"] == LEDGER_REBUILT_DETAIL

            planned = []

            def record(document):
                planned.append(document)
                return {
                    "status": "PLANNED",
                    "plan_hash": plan.plan_hash,
                    "declaration": document,
                    "source_start": "2026-06-29",
                    "source_end": "2026-06-30",
                    "fit_calls": 0,
                    "portfolio_calls": 0,
                    "economic_scope": "POST_OBSERVED_LOCAL_QA_NOT_HOLDOUT_RELEASE",
                    "expected_economic_support": {
                        "start": "2026-06-29",
                        "end": "2026-06-30",
                        "count": 2,
                    },
                    "claim": (
                        "PREPARE_LOCAL_RESEARCH_AUTHORITY; "
                        "install and Portfolio PLAN/RUN remain separate"
                    ),
                    "next_requests": {
                        "prepare": {
                            "operation": "RESEARCH_STRATEGY_PREPARE",
                            "experiment_plan_hash": plan.plan_hash,
                        }
                    },
                }

            monkeypatch.setattr(owner, "plan", record)

            def cli(*arguments):
                code = main(
                    ["--workspace", str(workspace), "--view", view, *arguments],
                    serve=lambda _: 99,
                )
                captured = capsys.readouterr()
                assert not captured.err
                return code, json.loads(captured.out)

            saved = tmp_path / "stopped.json"
            code, answer = cli("recovery", "show", str(task.task_id), "--output", str(saved))
            assert code == 2 and answer["data"]["lifecycle"] == "BLOCKED"
            assert answer["data"]["stop"]["detail"] == LEDGER_REBUILT_DETAIL
            original = registry.task(task.task_id)
            for arguments in (
                ("strategy", "plan", "--from", str(saved)),
                ("request", "--from", str(saved), "--action", "replan"),
            ):
                code, answer = cli(*arguments)
                assert code == 0 and answer["data"]["status"] == "PLANNED", answer
            assert planned == [expected["experiment_document"]] * 2
            assert registry.task(task.task_id) == original
            assert registry.tasks() == (original,)
    finally:
        gc.collect()
        if workspace.exists():
            assert workspace.resolve().is_relative_to(tmp_path.resolve())
            shutil.rmtree(workspace)


def test_an_evidence_task_without_its_book_selector_names_the_choice_before_replan(
    tmp_path, capsys
):
    """contract (V615/TE12): an issuer-only Task never supplies a default book in place
    of an absent durable selector. Its unavailable action says what the record lacks and
    gives a filled, usable history request without changing the stopped Task.
    """
    workspace = tmp_path / "workspace"
    try:
        with date_host(workspace) as live:
            registry = live.session.task_control_registry
            envelope = TaskInputEnvelope.create(
                task_kind=AlternativeEvidenceDocumentTaskAdapter.task_kind,
                input_schema_id="synthetic-evidence-input",
                payload={"scope_hash": "a" * 64, "run_hash": "b" * 64},
            )
            goal = ResearchGoal.create(
                goal_kind="PREPARE_EVIDENCE",
                input_hash=envelope.input_hash,
                deliverable_kind="EvidencePackets",
                summary="Synthetic issuer scope",
            )
            plan = ResearchPlan.create(
                goal_hash=goal.goal_hash,
                workflow_definition_hash="c" * 64,
                verifier_catalog_hash="d" * 64,
                work_items=(
                    WorkItemDefinition.create(
                        stage_id="prepare",
                        dependency_ids=(),
                        verifier_id="synthetic.verify",
                    ),
                ),
            )
            task = registry.admit(
                input_envelope=envelope,
                goal=goal,
                plan=plan,
                observed_at=datetime(2026, 10, 3, 12, tzinfo=UTC),
            ).record
            database = registry.database_path
            for path in database.parent.glob(f"{database.name}*"):
                path.unlink()
            type(registry)(database, gate=live.session.mutation_gate)
            registry.rebuild_from_requests(observed_at=datetime(2026, 10, 3, 13, tzinfo=UTC))
            saved = tmp_path / "evidence-stop.json"
            assert (
                main(
                    [
                        "--workspace",
                        str(workspace),
                        "--view",
                        "full",
                        "recovery",
                        "show",
                        str(task.task_id),
                        "--output",
                        str(saved),
                    ],
                    serve=lambda _: 99,
                )
                == 2
            )
            answer = json.loads(capsys.readouterr().out)
            action = next(v for v in answer["data"]["actions"] if v["action"] == "REPLAN")
            assert action["available"] is False
            assert action["expected_effect"] != LEDGER_REBUILT_NEXT
            assert "book selector" in action["reason"] and "history" in action["reason"]
            assert answer["next_requests"]["books"] == {"operation": "RESEARCH_HISTORY"}
            assert "replan" not in answer["next_requests"]
            row = next(
                r for r in live.operations.tasks()["tasks"] if r["task_id"] == str(task.task_id)
            )
            assert row["detail"] == LEDGER_REBUILT_DETAIL
            assert "stop_next" not in row
            original = registry.task(task.task_id)
            assert (
                main(
                    [
                        "--workspace",
                        str(workspace),
                        "--view",
                        "full",
                        "request",
                        "--from",
                        str(saved),
                        "--action",
                        "books",
                    ],
                    serve=lambda _: 99,
                )
                == 0
            )
            continued = json.loads(capsys.readouterr().out)
            assert continued["status"] == "AVAILABLE", continued
            assert registry.task(task.task_id) == original
    finally:
        gc.collect()
        if workspace.exists():
            assert workspace.resolve().is_relative_to(tmp_path.resolve())
            shutil.rmtree(workspace)
