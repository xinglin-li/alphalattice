"""First-use preparation uses real numerical owners, with recorded source adapters."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from alphalattice.control.product_host.composition.application_session import (
    WorkspaceApplicationSession,
)
from alphalattice.control.product_host.composition.local_web_session import LocalPortfolioWebSession
from alphalattice.control.product_host.composition.research_workspace import (
    ResearchWorkspaceManifest,
    publish_research_workspace_manifest,
    read_research_workspace_manifest,
)
from alphalattice.control.product_host.data_preparation.application import (
    WorkspacePreparationApplication,
)
from alphalattice.control.product_host.research_authoring.factor_inputs import read_factor_bundle
from alphalattice.control.task_control.contracts import TaskLifecycle
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchOperationRequest,
)
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchRequestDocument as PortfolioResearchAgentRequest,
)
from tests.portfolio_strategy_lab.local_web_support import InstalledAgent, _json
from tests.researcher_methodology_surface.real_workspace import (
    OBSERVED_AT,
    _source_loader_for,
)


def test_initialization_creates_real_factor_inputs_and_reopens_without_strategies(fresh_prepared):
    root, _task_id, view = fresh_prepared
    manifest = read_research_workspace_manifest(root)
    assert manifest.strategy_installation == "NOT_INSTALLED"
    assert manifest.default_strategy_package_id is None
    assert manifest.data_update is not None and len(manifest.experiment_inputs) == 1
    bundle = read_factor_bundle(root, manifest.experiment_inputs[0].binding_hash)
    assert bundle.database_snapshot_hash is not None
    with WorkspaceApplicationSession.acquire(root) as session:
        app = WorkspacePreparationApplication(session, clock=lambda: OBSERVED_AT)
        assert app.readback() == view
        assert app.plan()["status"] == "ALREADY_PREPARED"


def test_new_preparation_plan_and_exact_task_reuse_ignore_the_operator_cap(
    tmp_path: Path,
) -> None:
    """V680: real previews and admission retain one plan/Task at 10 and 20 GiB; no run."""
    from alphalattice.control.workspace_runtime.storage.capacity import StorageCapStore
    from tests.workspace_maintenance.local_data_provider import recording_provider

    publish_research_workspace_manifest(
        tmp_path, ResearchWorkspaceManifest.research_only("operator-cap-isolation")
    )
    with WorkspaceApplicationSession.acquire(tmp_path) as session:
        app = WorkspacePreparationApplication(
            session,
            clock=lambda: OBSERVED_AT,
            provider=recording_provider(),
            source_loader=_source_loader_for(("AAA", "BBB")),
        )
        store = StorageCapStore(tmp_path)
        previews = []
        for gib in (10, 20):
            store.write(gib * 1024**3, chosen_by="HUMAN", chosen_at=OBSERVED_AT)
            previews.append(app.plan())
        assert previews[0]["plan_hash"] == previews[1]["plan_hash"]
        tasks = []
        readings = []
        for gib, preview in zip((10, 20), previews, strict=True):
            store.write(gib * 1024**3, chosen_by="HUMAN", chosen_at=OBSERVED_AT)
            task = session.task_control_registry.task(
                app.confirm(preview["plan_hash"], caller="HUMAN").task_id
            )
            tasks.append(task)
            readings.append(
                {
                    "cap_gib": gib,
                    "plan_hash": task.input.payload["plan"]["plan_hash"],
                    "input_hash": task.input.input_hash,
                    "goal_hash": task.goal.goal_hash,
                    "workflow_hash": task.plan.plan_hash,
                    "record_hash": task.record_hash,
                }
            )
        assert tasks[0] == tasks[1]
        assert tasks[0].lifecycle is TaskLifecycle.QUEUED
        (tmp_path / "cap-identity-pairs.json").write_text(
            json.dumps(readings, indent=2) + "\n", encoding="utf-8", newline="\n"
        )


@pytest.mark.parametrize("budget_defect", ["cap", "resealed_derivation", "policy_hash"])
def test_historical_frozen_budget_is_fully_checked_but_never_grants_current_capacity(
    tmp_path: Path,
    budget_defect: str,
) -> None:
    """V680: legacy outer/inner seals and derivation hold; admission reads the live cap."""
    from uuid import uuid4

    from alphalattice.control.product_host.storage.contracts import resolve_storage_budget
    from alphalattice.control.task_control.contracts import TaskExecution
    from alphalattice.control.task_control.runner import StageDisposition
    from alphalattice.control.workspace_runtime.storage.capacity import StorageCapStore
    from alphalattice.kernel.shared_kernel.identity import canonical_hash
    from tests.workspace_maintenance.local_data_provider import recording_provider

    publish_research_workspace_manifest(
        tmp_path, ResearchWorkspaceManifest.research_only("legacy-cap")
    )
    with WorkspaceApplicationSession.acquire(tmp_path) as session:
        app = WorkspacePreparationApplication(
            session,
            clock=lambda: OBSERVED_AT,
            provider=recording_provider(),
            source_loader=_source_loader_for(("AAA", "BBB")),
        )
        plan = app.plan()
        store = StorageCapStore(tmp_path)
        store.write(20 * 1024**3, chosen_by="HUMAN", chosen_at=OBSERVED_AT)
        assert app.plan()["plan_hash"] == plan["plan_hash"]
        task = session.task_control_registry.task(
            app.confirm(plan["plan_hash"], caller="HUMAN").task_id
        )
        execution_id = uuid4()
        execution = TaskExecution.from_identity(
            execution_id=execution_id,
            task_id=task.task_id,
            graph_thread_id=f"workspace-task:{task.task_id}:{execution_id}",
            worker_instance_id=uuid4(),
            compatibility=app.compatibility(task),
            started_at=OBSERVED_AT,
            last_heartbeat_at=OBSERVED_AT,
            checkpoint_disposition="active",
        )
        freeze, prepare = task.plan.work_items[:2]
        fresh = app.execute_stage(task=task, execution=execution, work_item=freeze)
        assert fresh.disposition is StageDisposition.READY
        path = app.telemetry.path(task.task_id, freeze.stage_id)
        legacy = json.loads(path.read_text("utf-8"))
        estimate = legacy.pop("input_estimate")
        assert not any("cap" in key or "budget" in key for key in estimate)
        legacy["budget_upper_bound"] = resolve_storage_budget(
            active_listing_count=estimate["active_listing_count"],
            research_session_count=estimate["research_session_count"],
        ).model_dump(mode="json")
        legacy["content_hash"] = canonical_hash(
            {key: value for key, value in legacy.items() if key != "content_hash"}
        )
        app.telemetry.replace_json(path, legacy)
        sealed = path.read_bytes()
        repeated = app.execute_stage(task=task, execution=execution, work_item=freeze)
        assert repeated.disposition is StageDisposition.READY
        assert (
            app.verify_stage(
                task=task, execution=execution, work_item=freeze, evidence=repeated.evidence
            )
            == repeated.evidence
        )
        store.write(1, chosen_by="HUMAN", chosen_at=OBSERVED_AT)
        blocked = app.execute_stage(task=task, execution=execution, work_item=prepare)
        assert blocked.failure_code == "storage.managed_capacity_exceeded"
        store.write("auto", chosen_by="HUMAN", chosen_at=OBSERVED_AT)
        assert (
            app.execute_stage(task=task, execution=execution, work_item=freeze).evidence
            == repeated.evidence
        )
        assert path.read_bytes() == sealed
        bad_receipt = json.loads(sealed)
        budget = bad_receipt["budget_upper_bound"]
        if budget_defect == "policy_hash":
            budget["policy_hash"] = "0" * 64
        else:
            budget["managed_cap_bytes"] += 1024**3
            if budget_defect == "resealed_derivation":
                budget["cleanup_target_bytes"] = int(budget["managed_cap_bytes"] * 0.8)
                budget["high_water_bytes"] = int(budget["managed_cap_bytes"] * 0.9)
                budget["policy_hash"] = canonical_hash(
                    {key: value for key, value in budget.items() if key != "policy_hash"}
                )
        bad_receipt["content_hash"] = canonical_hash(
            {key: value for key, value in bad_receipt.items() if key != "content_hash"}
        )
        app.telemetry.replace_json(path, bad_receipt)
        inner_refusal = app.execute_stage(task=task, execution=execution, work_item=freeze)
        assert inner_refusal.failure_code == "workspace_preparation.stage_tampered"
        with pytest.raises(ValueError, match=r"^workspace_preparation\.stage_tampered$"):
            app.verify_stage(
                task=task, execution=execution, work_item=freeze, evidence=repeated.evidence
            )
        assert path.read_bytes() != sealed
        assert json.loads(path.read_text("utf-8")) == bad_receipt

        # Keep the separate outer-tamper refusal without resealing the stage.
        legacy["budget_upper_bound"]["managed_cap_bytes"] += 1
        app.telemetry.replace_json(path, legacy)
        tampered = app.execute_stage(task=task, execution=execution, work_item=freeze)
        assert tampered.failure_code == "workspace_preparation.stage_tampered"


def test_stage_file_replace_outlives_a_momentary_sharing_violation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Publication retries preserve bytes, clean staging and reach Task failure codes."""

    from alphalattice.kernel.shared_kernel import persistence
    from alphalattice.kernel.shared_kernel.domain.errors import WorkspaceConflictError
    from tests.workspace_maintenance.local_data_provider import recording_provider

    publish_research_workspace_manifest(tmp_path, ResearchWorkspaceManifest.research_only("retry"))
    with WorkspaceApplicationSession.acquire(tmp_path) as session:
        app = WorkspacePreparationApplication(
            session,
            clock=lambda: OBSERVED_AT,
            provider=recording_provider(),
            source_loader=_source_loader_for(("AAA", "BBB")),
        )
        admission = app.confirm(app.plan()["plan_hash"], caller="HUMAN")
        task = session.task_control_registry.task(admission.task_id)
        target = app._path(task.task_id, "progress")
        calls = []
        real_replace = os.replace

        def flaky(source, destination):
            if Path(destination).parent == target.parent:
                calls.append(Path(destination))
                if len(calls) < 3:
                    raise PermissionError("temporary sharing violation")
            real_replace(source, destination)

        monkeypatch.setattr(persistence.os, "replace", flaky)
        monkeypatch.setattr(persistence.time, "sleep", lambda _: None)
        values = app._save(task, "progress", {"completed": 1}, progress=True)
        assert len(calls) == 3
        assert app._load(task.task_id, "progress") == values
        assert list(target.parent.iterdir()) == [target]
        before = target.read_bytes()

        def refused(source, destination):
            if Path(destination).parent == target.parent:
                calls.append(Path(destination))
                raise PermissionError("persistent sharing violation")
            real_replace(source, destination)

        calls.clear()
        monkeypatch.setattr(persistence.os, "replace", refused)
        with pytest.raises(WorkspaceConflictError) as blocked:
            app._save(task, "progress", {"completed": 2}, progress=True)
        assert blocked.value.failure.code == "catalog.replace_blocked"
        assert isinstance(blocked.value.__cause__, PermissionError)
        assert len(calls) == len(persistence.DURABLE_REPLACE_DELAYS) + 1
        assert target.read_bytes() == before
        assert list(target.parent.iterdir()) == [target]

        # Drive the real Task runner only as far as the refused source receipt.
        # No data download or Feature build is necessary to prove the boundary.
        app.execute(task.task_id)
        failed = session.task_control_registry.task(task.task_id)
        assert failed.lifecycle is TaskLifecycle.BLOCKED
        assert failed.failure_code == "catalog.replace_blocked"
        assert not (tmp_path / "market-data.duckdb").exists()
        assert list(target.parent.iterdir()) == [target]
        assert target.read_bytes() == before


def test_an_agent_resumes_a_granted_preparation_from_its_own_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """regression (V484, the user's review): a preparation Task an agent started under a
    person's grant could not be resumed from its own answer: the readback offers its confirm by
    the plan, and the caller check asked for the grant again, though the Task holds it. An agent's
    confirm of that plan runs under the Task's own grant, which the validator checks again (a
    stopped Task is re-marked after the same check); a new admission still names its grant, and a
    person's confirm uses none."""

    from alphalattice.control.product_host.composition.portfolio_research_operations import (
        PortfolioResearchOperations,
    )
    from tests.workspace_maintenance.local_data_provider import recording_provider

    grant = "9" * 64
    publish_research_workspace_manifest(tmp_path, ResearchWorkspaceManifest.research_only("grant"))
    with WorkspaceApplicationSession.acquire(tmp_path) as session:
        app = WorkspacePreparationApplication(
            session,
            clock=lambda: OBSERVED_AT,
            provider=recording_provider(),
            source_loader=_source_loader_for(("AAA", "BBB")),
        )
        asked: list[str | None] = []

        def person_granted(_plan_hash: str, grant_hash: str | None) -> bool:
            # The validator stands for a person's grant over this plan; its own checks of a
            # grant's term and scope are unchanged, and what it is asked is what this holds.
            asked.append(grant_hash)
            return grant_hash == grant

        monkeypatch.setattr(app, "delegated_resume_allowed", person_granted)
        plan_hash = app.plan()["plan_hash"]
        with pytest.raises(ValueError, match="human_confirmation_required"):
            app.confirm(plan_hash, caller="EXTERNAL_AUTOMATION")
        assert asked == [None]
        admitted = app.confirm(plan_hash, caller="EXTERNAL_AUTOMATION", grant_hash=grant)

        class Dispatcher:
            """A dispatcher a confirm of the plan's own Task never reaches."""

            def submit(self, *_args: object, **_kwargs: object) -> None:
                raise AssertionError("the plan's own Task was submitted again")

        operations = PortfolioResearchOperations.__new__(PortfolioResearchOperations)
        operations.preparation = app  # type: ignore[assignment]
        operations.dispatcher = Dispatcher()  # type: ignore[assignment]
        asked.clear()
        answer = operations._workspace_operation(
            PortfolioResearchOperationRequest(
                operation="WORKSPACE_PREPARE_CONFIRM", preparation_plan_hash=plan_hash
            ),
            caller="EXTERNAL_AUTOMATION",
        )
        assert answer == {"status": "REUSED_IN_FLIGHT", "task_id": str(admitted.task_id)}
        assert asked == [grant]
        with pytest.raises(ValueError, match="human_confirmation_does_not_use_delegation"):
            app.confirm(plan_hash, caller="HUMAN", grant_hash=grant)


@pytest.mark.parametrize(
    ("failure_code", "decisions_ready"),
    [
        ("data.truth_review_required", False),
        ("data.truth_review_required", True),
        ("catalog.replace_blocked", False),
    ],
)
def test_a_preparation_with_pending_truth_decisions_offers_issues_before_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_code: str,
    decisions_ready: bool,
) -> None:
    """A truth-review stop reads its permitted choices before offering the same confirm.

    The readback uses the decision owner's standing. Other stops retain their confirmation.
    This checks a real admitted Task without acquiring data or deciding a case.
    """
    from alphalattice.control.product_host.data_preparation.remediation import (
        WorkspaceDataIssueApplication,
    )
    from alphalattice.control.task_control.runner import StageDisposition, StageExecutionResult
    from tests.workspace_maintenance.local_data_provider import recording_provider

    publish_research_workspace_manifest(
        tmp_path, ResearchWorkspaceManifest.research_only("truth-review")
    )
    manifest_before = read_research_workspace_manifest(tmp_path)
    provider = recording_provider()
    decision_reads: list[bool] = []

    def current_decisions_ready(_owner: WorkspaceDataIssueApplication) -> bool:
        decision_reads.append(decisions_ready)
        return decisions_ready

    monkeypatch.setattr(
        WorkspaceDataIssueApplication, "current_decisions_ready", current_decisions_ready
    )
    with WorkspaceApplicationSession.acquire(tmp_path) as session:
        app = WorkspacePreparationApplication(
            session,
            clock=lambda: OBSERVED_AT,
            provider=provider,
            source_loader=_source_loader_for(("AAA", "BBB")),
        )
        plan = app.plan()
        admitted = app.confirm(plan["plan_hash"], caller="HUMAN")
        monkeypatch.setattr(
            app,
            "execute_stage",
            lambda **_kwargs: StageExecutionResult(
                StageDisposition.BLOCKED, failure_code=failure_code
            ),
        )
        app.execute(admitted.task_id)
        blocked = session.task_control_registry.task(admitted.task_id)
        assert blocked.lifecycle is TaskLifecycle.BLOCKED
        view = app.readback(admitted.task_id)
        pending = failure_code == "data.truth_review_required" and not decisions_ready
        assert view["status"] == "BLOCKED"
        assert view["task_id"] == str(admitted.task_id)
        assert view["plan_hash"] == plan["plan_hash"]
        assert view["failure_code"] == failure_code
        assert view["execution_binding_changed"] is False
        assert view["next_action"] == ("DATA_ISSUES" if pending else None)
        assert view["next_requests"] == (
            {"issues": {"operation": "DATA_ISSUES"}}
            if pending
            else {
                "confirm": {
                    "operation": "WORKSPACE_PREPARE_CONFIRM",
                    "preparation_plan_hash": plan["plan_hash"],
                }
            }
        )
        assert view["confirmation_available"] is (False if pending else None)
        assert decision_reads == (
            [decisions_ready] if failure_code == "data.truth_review_required" else []
        )
        assert (
            session.task_control_registry.task(admitted.task_id).record_hash == blocked.record_hash
        )
        assert app.tasks() == (blocked,)
        assert read_research_workspace_manifest(tmp_path) == manifest_before
        assert provider.calls == []
        assert not (tmp_path / "market-data.duckdb").exists()


def test_a_delegated_data_decision_carries_its_preparation_past_a_storage_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """requirement (V484's follow-up, end to end): a preparation stops for a data decision; a
    person delegates the decision; the agent decides it under the grant and confirms the
    successor preparation; a storage failure stops the successor, whose readback words the stop
    (`catalog.replace_blocked`) and offers its resume; and the agent's `preparation
    confirm --from <its readback>`, naming no grant, resumes it under the Task's own grant, which
    the real validator checks again (its term, its preparation, the decision's actor and choice),
    and the successor ends SUCCEEDED. No stand-in replaces the validator."""

    from uuid import UUID

    from alphalattice.interface.local_application.cli import main as client_main
    from alphalattice.interface.local_application.client import LocalResearchClient
    from alphalattice.kernel.data.calendar import materialize_calendar_schedule
    from alphalattice.kernel.shared_kernel import persistence
    from tests.researcher_methodology_surface.real_workspace import (
        AS_OF,
        HISTORY_START,
        SeededWalkProvider,
    )

    symbols = tuple(f"F{i:03d}" for i in range(120))
    schedule = materialize_calendar_schedule(
        ("XNAS",), start=HISTORY_START, end=AS_OF, as_of_timestamp=OBSERVED_AT
    )
    sessions = tuple(value["session_date"] for value in schedule.to_pylist())
    moved = sessions[len(sessions) // 2]

    class UnexplainedMove(SeededWalkProvider):
        """One listing's close doubles for one session, with no event that explains it."""

        def _closes(self, symbol: str, end: Any) -> dict[Any, float]:
            closes = super()._closes(symbol, end)
            if symbol == symbols[0] and moved in closes:
                closes[moved] *= 2.0
            return closes

    root = tmp_path / "delegated"
    live = LocalPortfolioWebSession.from_workspace(root, clock=lambda: OBSERVED_AT)
    live.data_provider = UnexplainedMove(symbols, sessions)
    live.data_source_loader = _source_loader_for(symbols)
    with live:
        registry, app = live.session.task_control_registry, live.operations.preparation
        plan = _json(live, "/api/workspace/preparation/plan", method="POST", payload={})
        first = _json(
            live,
            "/api/workspace/preparation/confirm",
            method="POST",
            payload={"preparation_plan_hash": plan["plan_hash"]},
        )
        live.dispatcher.drain_for_tests(timeout=900)
        stopped = registry.task(UUID(first["task_id"]))
        assert (stopped.lifecycle, stopped.failure_code) == (
            TaskLifecycle.BLOCKED,
            "data.truth_review_required",
        ), live.dispatcher.failure(stopped.task_id)

        # The person delegates the decision, on the Workbench's route.
        issues = _json(live, "/api/workspace/data-issues")
        offers = {
            name: value
            for name, value in issues["next_requests"].items()
            if name.startswith(f"delegate:{stopped.task_id}:")
        }
        offer = offers[min(offers, key=lambda name: ("retain" not in name, name))]
        delegated = _json(
            live,
            "/api/workspace/data-issues/delegate",
            method="POST",
            payload={key: value for key, value in offer.items() if key != "operation"},
        )
        assert delegated["status"] == "DELEGATED" and not delegated["effect_applied"]
        grant = delegated["grant"]["grant_hash"]

        # The agent decides it under the grant, then plans and confirms the successor.
        agent = LocalResearchClient(root)
        decided = agent.request(delegated["next_requests"]["confirm"])
        assert decided.get("failure_code") is None, decided
        successor_plan = agent.request({"operation": "WORKSPACE_PREPARE_PLAN"})
        confirmed = agent.request(
            {
                "operation": "WORKSPACE_PREPARE_CONFIRM",
                "preparation_plan_hash": successor_plan["plan_hash"],
                "data_issue_grant_hash": grant,
            }
        )
        successor = UUID(str(confirmed["task_id"]))
        assert successor != stopped.task_id

        # A storage failure stops the successor: its records cannot be replaced.
        records = app._path(successor, "progress").parent
        real_replace = os.replace

        def refused(source: Any, destination: Any) -> None:
            if Path(destination).parent == records:
                raise PermissionError("persistent sharing violation")
            real_replace(source, destination)

        monkeypatch.setattr(persistence.os, "replace", refused)
        monkeypatch.setattr(persistence.time, "sleep", lambda _: None)
        live.dispatcher.drain_for_tests(timeout=900)
        halted = registry.task(successor)
        assert (halted.lifecycle, halted.failure_code) == (
            TaskLifecycle.BLOCKED,
            "catalog.replace_blocked",
        ), live.dispatcher.failure(successor)
        monkeypatch.setattr(persistence.os, "replace", real_replace)

        # Its readback says what stopped it and offers the confirm that resumes it; the agent
        # resumes it from that answer, naming no grant.
        held = agent.request({"operation": "WORKSPACE_PREPARE_READBACK", "task_id": str(successor)})
        assert held["failure_code"] == "catalog.replace_blocked", held
        assert "another program" in held["detail"], held
        assert held["next_requests"]["confirm"]["operation"] == "WORKSPACE_PREPARE_CONFIRM", held
        readback = tmp_path / "successor.json"
        readback.write_text(json.dumps(held), encoding="utf-8")
        resumed = tmp_path / "resumed.json"
        code = client_main(
            [
                "--workspace",
                str(root),
                "preparation",
                "confirm",
                "--from",
                str(readback),
                "--output",
                str(resumed),
            ],
            serve=lambda _: pytest.fail("the confirm must use the running Host"),
        )
        answer = json.loads(resumed.read_text(encoding="utf-8"))
        assert code in {0, 3} and answer["task_id"] == str(successor), (code, answer)
        live.dispatcher.drain_for_tests(timeout=900)
        assert registry.task(successor).lifecycle is TaskLifecycle.SUCCEEDED, (
            live.dispatcher.failure(successor)
        )


def test_optional_progress_delivery_never_governs_the_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Task-bound Feature progress is telemetry: a failed delivery is counted with its
    typed cause and the work goes on; a file that cannot be read, is malformed or is
    another Task's is never promoted, and the last valid observation is kept."""

    from uuid import uuid4

    from alphalattice.control.observation_runtime.telemetry.progress import (
        WorkProgressUpdate,
        WorkspaceProgressPublisher,
    )
    from alphalattice.control.task_control.contracts import TaskExecution
    from alphalattice.kernel.shared_kernel import persistence
    from alphalattice.kernel.shared_kernel.domain.errors import WorkspaceConflictError
    from alphalattice.kernel.shared_kernel.identity import canonical_hash
    from tests.workspace_maintenance.local_data_provider import recording_provider

    publish_research_workspace_manifest(
        tmp_path, ResearchWorkspaceManifest.research_only("telemetry")
    )
    with WorkspaceApplicationSession.acquire(tmp_path) as session:
        app = WorkspacePreparationApplication(
            session,
            clock=lambda: OBSERVED_AT,
            provider=recording_provider(),
            source_loader=_source_loader_for(("AAA", "BBB")),
        )
        task = session.task_control_registry.task(
            app.confirm(app.plan()["plan_hash"], caller="HUMAN").task_id
        )
        execution_id = uuid4()
        execution = TaskExecution.from_identity(
            execution_id=execution_id,
            task_id=task.task_id,
            graph_thread_id=f"workspace-task:{task.task_id}:{execution_id}",
            worker_instance_id=uuid4(),
            compatibility=app.compatibility(task),
            started_at=OBSERVED_AT,
            last_heartbeat_at=OBSERVED_AT,
            checkpoint_disposition="active",
        )
        publisher = WorkspaceProgressPublisher(tmp_path / "artifacts")
        sink = app._bound_progress_sink(task, execution, "prepare_features", publisher)
        update = WorkProgressUpdate(
            operation_id="build",
            stage_id="base_feature_materialization",
            status="RUNNING",
            completed_units=3,
            total_units=10,
            unit_name="listings",
        )
        sidecar = app._path(task.task_id, "work-progress")

        # A healthy delivery: both copies written, the readback names the observation.
        assert sink(update) is not None
        first = app.readback()["work_progress"]
        assert first["availability"] == "NOT_CURRENT"  # queued, not this execution's
        assert (first["completed_units"], first["total_units"]) == (3, 10)
        assert first["delivery"] == {
            "attempts": 1,
            "delivered": 1,
            "failures": 0,
            "last_failure": None,
            "last_delivered_at": OBSERVED_AT.isoformat(),
        }

        # The sidecar replace is refused for good: the sink still returns the projection,
        # the workspace projection is still published, the failure is counted with its
        # typed code, and the earlier valid sidecar stays what the readback shows.
        real_replace = os.replace

        def refuse_sidecar(source, destination):
            if Path(destination) == sidecar:
                raise PermissionError("persistent sharing violation")
            real_replace(source, destination)

        monkeypatch.setattr(persistence.os, "replace", refuse_sidecar)
        monkeypatch.setattr(persistence.time, "sleep", lambda _: None)
        heartbeat = publisher.read().heartbeat_sequence
        assert sink(update.model_copy(update={"completed_units": 4})) is not None
        assert publisher.read().heartbeat_sequence == heartbeat + 1
        degraded = app.readback()["work_progress"]
        assert degraded["completed_units"] == 3, "the last valid sidecar is retained"
        assert degraded["delivery"]["failures"] == 1
        assert degraded["delivery"]["last_failure"] == {
            "target": "TASK_SIDECAR",
            "code": "catalog.replace_blocked",
            "type": "WorkspaceConflictError",
            "at": OBSERVED_AT.isoformat(),
        }
        assert list(sidecar.parent.glob("*")) and all(
            not name.name.startswith("tmp") for name in sidecar.parent.iterdir()
        ), "no temporary survives a refused replace"
        monkeypatch.setattr(persistence.os, "replace", real_replace)

        # The workspace projection itself refuses: nothing is written, nothing raised.
        def refuse_publish(_update):
            raise OSError("projection store unavailable")

        monkeypatch.setattr(publisher, "publish", refuse_publish)
        assert sink(update.model_copy(update={"completed_units": 5})) is None
        assert app.readback()["work_progress"]["delivery"]["last_failure"] == {
            "target": "WORKSPACE_PROJECTION",
            "code": None,
            "type": "OSError",
            "at": OBSERVED_AT.isoformat(),
        }
        assert app.readback()["work_progress"]["delivery"]["failures"] == 2

        # Read failures: a torn or foreign file is never promoted; the last valid
        # observation this process wrote stays beside an UNREADABLE answer.
        valid = sidecar.read_bytes()
        sidecar.write_bytes(b"{not json")
        unreadable = app.readback()["work_progress"]
        assert unreadable["availability"] == "UNREADABLE"
        assert unreadable["cause"] == "JSONDecodeError"
        assert unreadable["last_valid"]["completed_units"] == 3
        assert unreadable["last_valid"]["stage"] == "prepare_features"
        assert unreadable["delivery"]["failures"] == 2
        forged = json.loads(valid)
        forged["task_id"] = str(uuid4())
        forged["content_hash"] = canonical_hash(
            {k: v for k, v in forged.items() if k != "content_hash"}
        )
        sidecar.write_text(json.dumps(forged), encoding="utf-8")
        assert app.readback()["work_progress"]["availability"] == "UNBOUND"
        malformed = json.loads(valid)
        malformed["projection"] = {"stage_id": "x"}
        malformed["content_hash"] = canonical_hash(
            {k: v for k, v in malformed.items() if k != "content_hash"}
        )
        sidecar.write_text(json.dumps(malformed), encoding="utf-8")
        assert app.readback()["work_progress"]["availability"] == "UNREADABLE"
        assert app.readback()["work_progress"]["cause"] == "projection_invalid"
        tampered = json.loads(valid)
        tampered["projection"]["completed_units"] = 9
        sidecar.write_text(json.dumps(tampered), encoding="utf-8")
        assert app.readback()["work_progress"]["cause"] == "content_hash"

        # Correctly sealed documents (inner projection hash and outer content hash both
        # consistent) whose values break the update owner's invariants -- a zero total, a
        # naive instant, negative counts -- are a typed invalid read: no numeric work, no
        # exception through the readback, never BOUND, and the last valid observation is
        # not replaced by them; a torn file afterwards still returns that observation.
        def sealed(**changes):
            document = json.loads(valid)
            projection = {k: v for k, v in document["projection"].items() if k != "projection_hash"}
            projection.update(changes)
            projection["projection_hash"] = canonical_hash(projection)
            document["projection"] = projection
            document["content_hash"] = canonical_hash(
                {k: v for k, v in document.items() if k != "content_hash"}
            )
            sidecar.write_text(json.dumps(document), encoding="utf-8")

        for malformed_values in (
            {"completed_units": 0, "total_units": 0, "percent_complete": 0.0},
            {"updated_at": "2026-08-02T06:30:00"},
            {"completed_units": -1, "total_units": -2, "percent_complete": 50.0},
        ):
            sealed(**malformed_values)
            invalid = app.readback()["work_progress"]
            assert invalid["availability"] == "UNREADABLE"
            assert invalid["cause"] == "projection_invalid"
            retained = invalid["last_valid"]
            assert (retained["completed_units"], retained["total_units"]) == (3, 10)
            assert invalid["last_valid"]["age_seconds"] >= 0.0
        sidecar.write_bytes(b"{not json")
        torn = app.readback()["work_progress"]
        assert torn["cause"] == "JSONDecodeError"
        assert torn["last_valid"]["completed_units"] == 3, "no malformed file poisoned the fallback"
        sidecar.write_bytes(valid)
        assert app.readback()["work_progress"]["completed_units"] == 3

        # Existence itself failing is said so, not raised through the readback.
        def refuse_stat(self):
            if self == sidecar:
                raise PermissionError("metadata unavailable")
            return real_exists(self)

        real_exists = Path.exists
        monkeypatch.setattr(Path, "exists", refuse_stat)
        assert app.readback()["work_progress"]["cause"] == "PermissionError"
        monkeypatch.setattr(Path, "exists", real_exists)
        # The durable stage record keeps its fail-closed write regardless.
        progress_path = app._path(task.task_id, "progress")

        def refuse_progress(source, destination):
            if Path(destination) == progress_path:
                raise PermissionError("persistent sharing violation")
            real_replace(source, destination)

        monkeypatch.setattr(persistence.os, "replace", refuse_progress)
        with pytest.raises(WorkspaceConflictError) as blocked:
            app._save(task, "progress", {"phase": "prepare_features"}, progress=True)
        assert blocked.value.failure.code == "catalog.replace_blocked"
        assert not (tmp_path / "market-data.duckdb").exists()


def test_listing_activity_is_bound_bounded_and_optional(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The onboarding runner's listing units reach the readback while the chunk is still
    running: the denominator is written before the first chunk, every unit transition
    is kept beside the Task bound to its execution and stage (acquired, assessed,
    admitted, failed -- each as the runner recorded it), the snapshot is bounded and
    labelled, a refused delivery is counted and the stage goes on, and after the stage
    the rows stay as history, never as current activity."""

    from alphalattice.control.product_host.data_preparation.application import (
        TELEMETRY_REPLACE_DELAYS,
    )
    from alphalattice.foundation.market_data_ops.sources.providers import ProviderFetchError
    from alphalattice.kernel.shared_kernel import persistence
    from tests.workspace_maintenance.local_data_provider import recording_provider

    symbols = ("AAA", "BBB", "CCC", "FAIL")
    publish_research_workspace_manifest(tmp_path, ResearchWorkspaceManifest.research_only("units"))
    provider = recording_provider(symbols=symbols)
    fetch = provider.fetch_daily
    seen: dict[tuple[str, ...], dict[str, Any]] = {}
    sidecar_path: list[Path] = []
    refused: list[int] = []
    real_replace = os.replace

    def refuse_second_sidecar_write(source, destination):
        # The second delivery of the listing snapshot (at the tenth transition) is refused
        # for good, through the whole telemetry retry; the unit that caused it is not
        # slowed, and the chunk-boundary delivery carries its rows as well.
        if sidecar_path and Path(destination) == sidecar_path[0]:
            refused.append(1)
            if 2 <= len(refused) <= 2 + len(TELEMETRY_REPLACE_DELAYS):
                raise PermissionError("persistent sharing violation")
        real_replace(source, destination)

    monkeypatch.setattr(persistence.os, "replace", refuse_second_sidecar_write)
    monkeypatch.setattr(persistence.time, "sleep", lambda _: None)
    with WorkspaceApplicationSession.acquire(tmp_path) as session:
        app = WorkspacePreparationApplication(
            session,
            clock=lambda: OBSERVED_AT,
            provider=provider,
            source_loader=_source_loader_for(symbols),
        )
        task = session.task_control_registry.task(
            app.confirm(app.plan()["plan_hash"], caller="HUMAN").task_id
        )
        sidecar_path.append(app._path(task.task_id, "listing-activity"))

        def observe_from_the_network_edge(symbols_requested, **kwargs):
            # Read the owners as the page would, while the runner is still inside its
            # first chunk: the denominator is already there, the earlier units too. (The
            # fixture's adjusted-close read goes through this same fetch; the first read
            # of each symbol is the one kept.)
            seen.setdefault(tuple(symbols_requested), app.readback())
            if symbols_requested == ("FAIL",):
                raise ProviderFetchError("data.provider_fetch_failed", "fixture", retryable=False)
            return fetch(symbols_requested, **kwargs)

        provider.fetch_daily = observe_from_the_network_edge  # type: ignore[method-assign]
        app.execute(task.task_id)

        # Before the first unit: the captured denominator, nothing hydrated, no snapshot
        # yet (the delivery counters say so).
        first = seen[("AAA",)]
        assert first["progress"] == {
            "task_id": str(task.task_id),
            "input_hash": task.input.input_hash,
            "stage": "progress",
            "phase": "prepare_data",
            "candidates": 4,
            "raw_ready": 0,
            "quality_eligible": 0,
            "failed": 0,
            "retry_after_at": None,
            "content_hash": first["progress"]["content_hash"],
        }
        assert first["listing_activity"] == {
            "availability": "NONE",
            "delivery": {
                "attempts": 0,
                "delivered": 0,
                "failures": 0,
                "last_failure": None,
                "last_delivered_at": None,
            },
        }
        # At the fourth unit (three listings done, 9 transitions observed): the snapshot
        # is BOUND to this execution and stage inside the still-running chunk. Its one
        # delivery so far (at the fifth transition) carries five rows and the counts at
        # that boundary -- two listings with raw bars while the chunk record still says
        # none: the headline advances within the chunk, the durable record waits for the
        # chunk. The four transitions after it wait for the next delivery.
        during = seen[("FAIL",)]["listing_activity"]
        assert during["availability"] == "BOUND"
        assert during["stage"] == "prepare_data"
        assert during["execution_id"] == str(
            session.task_control_registry.task(task.task_id).latest_execution_id
        )
        assert (during["observed"], during["retained"], during["dropped"]) == (5, 5, 0)
        assert (during["delivery"]["attempts"], during["delivery"]["delivered"]) == (1, 1)
        assert during["counts"] == {
            "candidates": 4,
            "raw_ready": 2,
            "quality_eligible": 2,
            "feature_ready": 1,
            "failed": 0,
        }
        assert seen[("FAIL",)]["progress"]["raw_ready"] == 0, "the chunk record waits"
        assert all(r["noted_at"] and r["observed_at"] for r in during["rows"])

        # After the stage: every transition as the runner recorded it, in order, with its
        # origin; the failed unit with its code and no origin; retained as history.
        after = app.readback()
        activity = after["listing_activity"]
        assert activity["availability"] == "NOT_CURRENT"
        assert (activity["observed"], activity["retained"], activity["dropped"]) == (10, 10, 0)
        rows = activity["rows"]
        assert [(r["symbol"], r["state"], r["origin"]) for r in rows] == [
            ("AAA", "RAW_READY", "ACQUIRED"),
            ("AAA", "QUALITY_ELIGIBLE", "LOCAL"),
            ("AAA", "FEATURE_READY", "LOCAL"),
            ("BBB", "RAW_READY", "ACQUIRED"),
            ("BBB", "QUALITY_ELIGIBLE", "LOCAL"),
            ("BBB", "FEATURE_READY", "LOCAL"),
            ("CCC", "RAW_READY", "ACQUIRED"),
            ("CCC", "QUALITY_ELIGIBLE", "LOCAL"),
            ("CCC", "FEATURE_READY", "LOCAL"),
            ("FAIL", "RAW_FAILED", None),
        ]
        assert rows[0]["raw_through"] is not None and rows[0]["observed_at"].startswith("2026-")
        assert rows[-1]["failure_code"] == "data.provider_fetch_failed"
        assert all(len(r["listing_id"]) > 8 for r in rows), "the exact listing, not only a symbol"
        assert activity["age_seconds"] == 0.0
        # The second delivery (at the tenth transition) was refused through the telemetry
        # budget and counted; the chunk boundary delivered all ten rows with the counts the
        # runner itself reported for the chunk: three listings with raw bars, three
        # eligible and admitted, one failed, kept apart.
        assert (activity["delivery"]["attempts"], activity["delivery"]["delivered"]) == (3, 2)
        assert activity["delivery"]["failures"] == 1
        assert activity["delivery"]["last_failure"]["target"] == "LISTING_ACTIVITY"
        assert activity["delivery"]["last_failure"]["code"] == "catalog.replace_blocked"
        assert activity["counts"] == {
            "candidates": 4,
            "raw_ready": 3,
            "quality_eligible": 3,
            "feature_ready": 3,
            "failed": 1,
        }

        # A malformed snapshot is a typed invalid read beside the last valid one; the
        # retained rows are not lost to it.
        sidecar_path[0].write_bytes(b"{not json")
        unreadable = app.readback()["listing_activity"]
        assert unreadable["availability"] == "UNREADABLE"
        assert unreadable["cause"] == "JSONDecodeError"
        assert len(unreadable["last_valid"]["rows"]) == 10
        json_rows = json.loads(json.dumps(after["listing_activity"]["rows"]))
        assert json_rows == rows
        # The stage itself moved on and stopped at the fixture's own population floor,
        # untouched by the telemetry.
        stopped = session.task_control_registry.task(task.task_id)
        assert (stopped.lifecycle.value, stopped.failure_code) == (
            "BLOCKED",
            "feature.baseline_qualified_population_insufficient",
        )


def test_listing_counts_follow_each_listings_own_transition(tmp_path: Path) -> None:
    """The counts move by what the affected listing left and entered, never by the
    destination alone: a retained listing whose tail audit fails while it is still
    PENDING adds a failure and revokes no other listing's eligibility; one that fails
    after its quality admission gives that admission back; resumed listings depart from
    the state the run loaded them in, which the seed already counted."""

    from datetime import datetime
    from uuid import uuid4

    from alphalattice.control.task_control.contracts import TaskExecution
    from alphalattice.foundation.market_data_ops.runtime.universe_onboarding import (
        CurrentUniverseOnboardingOutcome,
        CurrentUniverseOnboardingStatus,
        ListingUnitObservation,
    )
    from tests.workspace_maintenance.local_data_provider import recording_provider

    publish_research_workspace_manifest(tmp_path, ResearchWorkspaceManifest.research_only("count"))
    with WorkspaceApplicationSession.acquire(tmp_path) as session:
        app = WorkspacePreparationApplication(
            session,
            clock=lambda: OBSERVED_AT,
            provider=recording_provider(),
            source_loader=_source_loader_for(("AAA", "BBB")),
        )
        task = session.task_control_registry.task(
            app.confirm(app.plan()["plan_hash"], caller="HUMAN").task_id
        )
        execution_id = uuid4()
        execution = TaskExecution.from_identity(
            execution_id=execution_id,
            task_id=task.task_id,
            graph_thread_id=f"workspace-task:{task.task_id}:{execution_id}",
            worker_instance_id=uuid4(),
            compatibility=app.compatibility(task),
            started_at=OBSERVED_AT,
            last_heartbeat_at=OBSERVED_AT,
            checkpoint_disposition="active",
        )
        # The store at stage entry: X FEATURE_READY, V QUALITY_ELIGIBLE, W RAW_READY, the
        # retained Y and the fresh Z still PENDING; one more candidate untouched.
        seed = CurrentUniverseOnboardingOutcome(
            onboarding_id="o",
            status=CurrentUniverseOnboardingStatus.RUNNING,
            candidates=6,
            raw_ready=3,
            quality_eligible=2,
            feature_ready=1,
            failed=0,
        )
        delivery = app._bound_listing_observer(task, execution, "prepare_data", seed)
        at = datetime.fromisoformat("2026-08-02T06:30:00+00:00")

        def announce(symbol: str, state: str, run_start: str, origin: str | None) -> None:
            delivery.observe(
                ListingUnitObservation(
                    listing_id=symbol.lower() + "-listing",
                    symbol=symbol,
                    state=state,
                    observed_at=at,
                    run_start_state=run_start,
                    origin=origin,
                    failure_code="data.retained_action_evidence_not_reusable"
                    if state == "AUDIT_FAILED"
                    else None,
                )
            )

        counts = lambda: (  # noqa: E731 - a compact view of the four moving counts
            delivery.counts["raw_ready"],
            delivery.counts["quality_eligible"],
            delivery.counts["feature_ready"],
            delivery.counts["failed"],
        )
        # The retained Y fails its tail audit before any raw or quality admission: one
        # failure, and X keeps its eligibility (the supervisor's counterexample).
        announce("Y", "AUDIT_FAILED", "PENDING", None)
        assert counts() == (3, 2, 1, 1)
        # Z is acquired, admitted by the quality gate, then fails its audit: the admission
        # it held is given back, its raw bars stay counted.
        announce("Z", "RAW_READY", "PENDING", "ACQUIRED")
        announce("Z", "QUALITY_ELIGIBLE", "PENDING", "LOCAL")
        assert counts() == (4, 3, 1, 1)
        announce("Z", "AUDIT_FAILED", "PENDING", None)
        assert counts() == (4, 2, 1, 2)
        # Resumed work departs from the state the run loaded it in: W's raw bars were in the
        # seed and stay counted when the quality gate rejects it; V's eligibility was in the
        # seed and is unchanged when it is admitted for Features.
        announce("W", "QUALITY_INELIGIBLE", "RAW_READY", "LOCAL")
        assert counts() == (4, 2, 1, 3)
        announce("V", "FEATURE_READY", "QUALITY_ELIGIBLE", "LOCAL")
        assert counts() == (4, 2, 2, 3)
        assert delivery.counts["candidates"] == 6
        # The chunk boundary agrees with the runner's own count of the same units.
        boundary = CurrentUniverseOnboardingOutcome(
            onboarding_id="o",
            status=CurrentUniverseOnboardingStatus.RUNNING,
            candidates=6,
            raw_ready=4,
            quality_eligible=2,
            feature_ready=2,
            failed=3,
        )
        delivery.flush(boundary)
        assert app.readback()["listing_activity"]["counts"] == delivery.counts
        assert delivery.delivery["attempts"] == 2  # the sixth unit's delivery, the boundary


def test_listing_delivery_stays_cheap_under_persistent_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The listing delivery runs synchronously in the runner's thread, so its cost when the
    sidecar cannot be replaced is a runner cost. Under a persistent refusal and a fixed
    clock, ten observations must cost at most two bounded attempts (one per five units),
    each waiting no more than the telemetry replace budget; the observations stay retained
    and counted, and the first delivery that succeeds afterwards carries all of them with
    the counts advanced. No file is written by a refused attempt; no real sleep happens."""

    import dataclasses
    from datetime import datetime
    from uuid import uuid4

    from alphalattice.control.product_host.data_preparation import application as owner
    from alphalattice.control.task_control.contracts import TaskExecution
    from alphalattice.foundation.market_data_ops.runtime.universe_onboarding import (
        CurrentUniverseOnboardingOutcome,
        CurrentUniverseOnboardingStatus,
        ListingUnitObservation,
    )
    from alphalattice.kernel.shared_kernel import persistence
    from tests.workspace_maintenance.local_data_provider import recording_provider

    publish_research_workspace_manifest(tmp_path, ResearchWorkspaceManifest.research_only("cost"))
    sleeps: list[float] = []
    replaces: list[Path] = []
    refusing = [True]
    real_replace = os.replace

    def replace(source, destination):
        if refusing[0] and Path(destination).name == "listing-activity.json":
            replaces.append(Path(destination))
            raise PermissionError("persistent sharing violation")
        real_replace(source, destination)

    monkeypatch.setattr(persistence.os, "replace", replace)
    monkeypatch.setattr(persistence.time, "sleep", sleeps.append)
    with WorkspaceApplicationSession.acquire(tmp_path) as session:
        app = WorkspacePreparationApplication(
            session,
            clock=lambda: OBSERVED_AT,
            provider=recording_provider(),
            source_loader=_source_loader_for(("AAA", "BBB")),
        )
        task = session.task_control_registry.task(
            app.confirm(app.plan()["plan_hash"], caller="HUMAN").task_id
        )
        execution_id = uuid4()
        execution = TaskExecution.from_identity(
            execution_id=execution_id,
            task_id=task.task_id,
            graph_thread_id=f"workspace-task:{task.task_id}:{execution_id}",
            worker_instance_id=uuid4(),
            compatibility=app.compatibility(task),
            started_at=OBSERVED_AT,
            last_heartbeat_at=OBSERVED_AT,
            checkpoint_disposition="active",
        )
        seed = CurrentUniverseOnboardingOutcome(
            onboarding_id="o",
            status=CurrentUniverseOnboardingStatus.RUNNING,
            candidates=12,
            raw_ready=2,
            quality_eligible=1,
            feature_ready=1,
            failed=0,
        )
        delivery = app._bound_listing_observer(task, execution, "prepare_data", seed)
        assert delivery.counts == {
            "candidates": 12,
            "raw_ready": 2,
            "quality_eligible": 1,
            "feature_ready": 1,
            "failed": 0,
        }, "seeded from the retained progress, not from zero"
        at = datetime.fromisoformat("2026-08-02T06:30:00+00:00")
        units = [
            ("F001", "RAW_READY", "ACQUIRED"),
            ("F001", "QUALITY_ELIGIBLE", "LOCAL"),
            ("F001", "FEATURE_READY", "LOCAL"),
            ("F002", "RAW_READY", "RETAINED"),
            ("F002", "QUALITY_INELIGIBLE", "LOCAL"),
            ("F003", "RAW_FAILED", None),
            ("F004", "RAW_READY", "ACQUIRED"),
            ("F004", "QUALITY_ELIGIBLE", "LOCAL"),
            ("F004", "AUDIT_FAILED", None),
            ("F005", "RAW_READY", "ACQUIRED"),
        ]
        for symbol, state, origin in units:
            delivery.observe(
                ListingUnitObservation(
                    listing_id=symbol.lower() + "-listing",
                    symbol=symbol,
                    state=state,
                    observed_at=at,
                    run_start_state="PENDING",
                    origin=origin,
                    failure_code="data.x" if state.endswith("FAILED") else None,
                )
            )
        # Two attempts (at the fifth and tenth unit), each within the telemetry budget; every
        # observation retained and counted; nothing written.
        assert delivery.delivery["attempts"] == 2
        assert delivery.delivery["failures"] == 2
        assert delivery.delivery["delivered"] == 0
        assert len(sleeps) == 2 * len(owner.TELEMETRY_REPLACE_DELAYS)
        assert sum(sleeps) <= 0.2
        assert len(delivery.rows) == 10 and delivery.observed == 10
        assert not delivery.path.exists()
        assert delivery.counts == {
            "candidates": 12,
            "raw_ready": 2 + 4,
            "quality_eligible": 1 + 2 - 1,
            "feature_ready": 1 + 1,
            "failed": 0 + 3,
        }, "raw availability, quality and failures are advanced separately"
        # The refusal lifts: the chunk-boundary delivery carries all retained work, and the
        # runner's own counts at that boundary reconcile the replayed ones.
        refusing[0] = False
        boundary = CurrentUniverseOnboardingOutcome(
            onboarding_id="o",
            status=CurrentUniverseOnboardingStatus.RUNNING,
            candidates=12,
            raw_ready=6,
            quality_eligible=2,
            feature_ready=2,
            failed=3,
        )
        delivery.flush(boundary)
        assert (delivery.delivery["attempts"], delivery.delivery["delivered"]) == (3, 1)
        activity = app.readback()["listing_activity"]
        assert activity["availability"] == "NOT_CURRENT"  # not this execution's Task run
        assert (activity["observed"], activity["retained"], activity["dropped"]) == (10, 10, 0)
        assert [r["symbol"] for r in activity["rows"]][:3] == ["F001", "F001", "F001"]
        assert activity["counts"] == delivery.counts
        assert all(datetime.fromisoformat(r["noted_at"]) == OBSERVED_AT for r in activity["rows"])
        assert activity["delivery"]["failures"] == 2
        # A boundary with nothing new does not write again; a differing owner count does.
        delivery.flush(boundary)
        assert delivery.delivery["attempts"] == 3
        delivery.flush(dataclasses.replace(boundary, failed=4))
        assert delivery.delivery["attempts"] == 4
        assert app.readback()["listing_activity"]["counts"]["failed"] == 4


def test_missing_source_authority_refuses_before_task_or_market_database(tmp_path: Path):
    publish_research_workspace_manifest(
        tmp_path, ResearchWorkspaceManifest.research_only("offline")
    )
    with WorkspaceApplicationSession.acquire(tmp_path) as session:
        app = WorkspacePreparationApplication(session, clock=lambda: OBSERVED_AT)
        plan = app.plan()
        assert plan["confirmation_available"] is False
        assert plan["source_access_failure"] == "workspace_preparation.source_access_not_admitted"
        assert "existing_inputs" not in app.last_plan.model_dump(mode="json")
        with pytest.raises(ValueError, match="source_access_not_admitted"):
            app.confirm(plan["plan_hash"], caller="HUMAN")
        assert not session.task_control_registry.tasks()
        assert not (tmp_path / "market-data.duckdb").exists()


def test_a_preview_past_its_hour_is_refused_by_the_host_that_planned_it(tmp_path: Path):
    """regression (V536, the user's review at a84e523f): an unconfirmed preview confirmed 61
    minutes after its plan was accepted by the Host that planned it, which returned the plan its
    memory held, and refused `preview_required` by a restarted one. Every Host reads it through
    the plan store's hour. The confirm reaches the workspace's missing source access only once
    it holds the plan, so that refusal marks the plan found."""

    from datetime import timedelta

    publish_research_workspace_manifest(
        tmp_path, ResearchWorkspaceManifest.research_only("offline")
    )
    clock = [OBSERVED_AT]
    with WorkspaceApplicationSession.acquire(tmp_path) as session:
        app = WorkspacePreparationApplication(session, clock=lambda: clock[0])
        plan_hash = app.plan()["plan_hash"]
        with pytest.raises(ValueError, match="source_access_not_admitted"):
            app.confirm(plan_hash, caller="HUMAN")
        clock[0] = OBSERVED_AT + timedelta(minutes=61)
        assert app.last_plan is not None and app.last_plan.plan_hash == plan_hash
        restarted = WorkspacePreparationApplication(session, clock=lambda: clock[0])
        for host in (app, restarted):
            with pytest.raises(ValueError, match="preview_required"):
                host.confirm(plan_hash, caller="HUMAN")
        assert not session.task_control_registry.tasks()


def test_a_preview_past_its_hour_offers_the_preparation_again(tmp_path: Path):
    """regression (V543, the sweep of S1): a confirm past the preview's hour was refused
    `workspace_preparation.preview_required` with its words and no request; it offers planning
    the preparation again."""

    from datetime import timedelta

    clock = [OBSERVED_AT]
    with LocalPortfolioWebSession.from_workspace(tmp_path, clock=lambda: clock[0]) as live:
        plan = _json(live, "/api/workspace/preparation/plan", method="POST", payload={})
        clock[0] = OBSERVED_AT + timedelta(minutes=61)
        refused = _json(
            live,
            "/api/workspace/preparation/confirm",
            method="POST",
            payload={"preparation_plan_hash": plan["plan_hash"]},
        )
        assert refused["failure_code"] == "workspace_preparation.preview_required", refused
        assert refused["next_requests"] == {"replan": {"operation": "WORKSPACE_PREPARE_PLAN"}}
        assert not live.session.task_control_registry.tasks()


def test_a_preparation_answer_names_its_own_plan_whatever_its_owner_holds_last(
    tmp_path: Path, monkeypatch
):
    """regression (V534, the user's review at a84e523f): the answer was built from the owner's
    last plan after this one was kept, so a concurrent plan that replaced it in between gave
    this answer the other plan's hash, which its confirm would have run. The answer names its
    own plan."""

    from types import SimpleNamespace

    publish_research_workspace_manifest(
        tmp_path, ResearchWorkspaceManifest.research_only("offline")
    )
    with WorkspaceApplicationSession.acquire(tmp_path) as session:
        app = WorkspacePreparationApplication(session, clock=lambda: OBSERVED_AT)
        kept = []
        remember = app._previews.remember
        other = SimpleNamespace(
            plan_hash="f" * 64,
            target_session=OBSERVED_AT.date(),
            existing_inputs=None,
            predecessor_task_id=None,
        )

        def concurrent(plan):
            kept.append(plan.plan_hash)
            entry = remember(plan)
            app.last_plan = other  # another request's plan, arriving in between
            return entry

        monkeypatch.setattr(app._previews, "remember", concurrent)
        answer = app.plan()
        assert answer["plan_hash"] == kept[0] != other.plan_hash
        assert "f" * 64 not in json.dumps(answer, default=str)


def test_empty_start_and_actor_boundary_without_strategy_or_data_authority(tmp_path: Path):
    with pytest.raises(ValueError, match="operation_hash_invalid"):
        PortfolioResearchOperationRequest(
            operation="STORAGE_CONFIRM", storage_plan_hash="../outside"
        )
    with LocalPortfolioWebSession.from_workspace(tmp_path, clock=lambda: OBSERVED_AT) as live:
        projection = _json(live, "/api/session")
        assert projection["strategy"] is None and projection["installed_strategies"] == []
        assert live.application is None and live.service is None
        assert live.review is not None and not live.review.has_evidence_authority
        readback = _json(live, "/api/workspace/preparation")
        assert readback["status"] == "INITIALIZATION_REQUIRED"
        assert readback["progress"] is None and readback["work_progress"] is None
        # The entry page learns this with the session: the same readback, one request.
        assert _json(live, "/api/session?context=1")["research_context"]["preparation"] == readback
        assert _json(live, "/api/workspace/storage")["managed_bytes"] == 0
        assert _json(live, "/api/experiments/controls")["status"] == "RESEARCH_INPUT_NOT_ADMITTED"
        plan = _json(live, "/api/workspace/preparation/plan", method="POST", payload={})
        bridge = InstalledAgent(live.operations)
        refused = json.loads(
            bridge.invoke(
                PortfolioResearchAgentRequest(
                    operation="WORKSPACE_PREPARE_CONFIRM",
                    preparation_plan_hash=plan["plan_hash"],
                )
            )
        )
        assert "human_confirmation_required" in str(refused)
        assert (
            _json(
                live,
                "/api/workspace/preparation/confirm",
                method="POST",
                payload={"preparation_plan_hash": plan["plan_hash"]},
            )["failure_code"]
            == "workspace_preparation.source_access_not_admitted"
        )
        assert (
            _json(live, "/api/results")["failure_code"]
            == "research_workspace.strategy_not_installed"
        )
        assert not live.session.task_control_registry.tasks()
        assert not (tmp_path / "market-data.duckdb").exists()
        manifest_bytes = (tmp_path / "research-workspace.json").read_bytes()
    with LocalPortfolioWebSession.from_workspace(tmp_path, clock=lambda: OBSERVED_AT) as reopened:
        assert _json(reopened, "/api/session")["workspace_id"] == projection["workspace_id"]
    assert (tmp_path / "research-workspace.json").read_bytes() == manifest_bytes
    # Unknown existing contents must not be silently adopted as an empty workspace.
    unknown = tmp_path / "unknown"
    unknown.mkdir()
    (unknown / "market-data.duckdb").write_bytes(b"not a database")
    with pytest.raises(ValueError, match="nonempty_manifest_absent"):
        LocalPortfolioWebSession.from_workspace(unknown)


def test_explicit_qualified_data_binding_creates_its_manifest_without_manual_json(tmp_path):
    from alphalattice.control.product_host.maintenance.data_update import (
        bind_existing_data_workspace,
        inspect_existing_data_workspace,
    )
    from tests.researcher_methodology_surface.real_workspace import build_real_risk_workspace

    data = build_real_risk_workspace(
        tmp_path / "known-data", symbols=tuple(f"F{i:03d}" for i in range(120))
    )
    root = data.workspace
    assert not (root / "research-workspace.json").exists()
    database = (root / "market-data.duckdb").read_bytes()
    inspection = inspect_existing_data_workspace(root)
    assert inspection["status"] == "QUALIFIED_LOCAL_INPUTS"
    assert inspection["next_action"] == "BIND_EXISTING_DATA_WORKSPACE"
    assert not (root / "research-workspace.json").exists()
    assert (root / "market-data.duckdb").read_bytes() == database
    manifest = bind_existing_data_workspace(root)
    assert manifest.strategy_installation == "NOT_INSTALLED"
    assert manifest.data_update is not None and not manifest.experiment_inputs
    assert (root / "market-data.duckdb").read_bytes() == database
    assert bind_existing_data_workspace(root) == manifest

    class NoSources:
        def __getattribute__(self, name):
            raise AssertionError(f"qualified local preparation accessed sources: {name}")

    start = LocalPortfolioWebSession.from_workspace(root, clock=lambda: OBSERVED_AT)
    start.data_provider = NoSources()
    with start as live:
        perform = live.operations.preparation._perform

        def interrupt_before_publication(task, plan, stage, execution):
            if stage == "publish_inputs":
                raise ValueError("test_retry_existing_local_preparation")
            return perform(task, plan, stage, execution)

        live.operations.preparation._perform = interrupt_before_publication
        plan = _json(live, "/api/workspace/preparation/plan", method="POST", payload={})
        assert plan["source_mode"] == "REUSE_QUALIFIED_LOCAL_DATA_NO_DOWNLOAD", plan
        assert plan["confirmation_available"] is True
        sent = _json(
            live,
            "/api/workspace/preparation/confirm",
            method="POST",
            payload={
                "preparation_plan_hash": plan["plan_hash"],
            },
        )
        live.dispatcher.drain_for_tests(timeout=240)
        task = _json(live, "/api/status?task_id=" + sent["task_id"])
        assert task["lifecycle"] == "BLOCKED", task["latest_failure_code"]
        live.operations.preparation._perform = perform
        retried = _json(
            live,
            "/api/workspace/preparation/confirm",
            method="POST",
            payload={
                "preparation_plan_hash": plan["plan_hash"],
            },
        )
        assert retried["task_id"] == sent["task_id"]
        live.dispatcher.drain_for_tests(timeout=240)
        task = _json(live, "/api/status?task_id=" + sent["task_id"])
        assert task["lifecycle"] == "SUCCEEDED", task["latest_failure_code"]
        assert _json(live, "/api/experiments/controls")["status"] == "READY"
        assert (
            _json(live, "/api/workspace/preparation/plan", method="POST", payload={})["status"]
            == "ALREADY_PREPARED"
        )
    unknown = tmp_path / "unknown-data"
    unknown.mkdir()
    (unknown / "notes.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="existing_data_required"):
        bind_existing_data_workspace(unknown)
    assert not (unknown / "research-workspace.json").exists()


@pytest.mark.parametrize("source_kind", ["absent", "legacy", "corrupt"])
def test_source_inspection_is_read_only_and_explains_missing_qualification(
    tmp_path, monkeypatch, source_kind
):
    from alphalattice.control.product_host.maintenance.data_update import (
        inspect_existing_data_workspace,
    )
    from alphalattice.control.workspace_runtime.database import WorkspaceDatabase
    from alphalattice.control.workspace_runtime.writer_lease import WorkspaceWriterLease

    root = tmp_path / "source"
    if source_kind != "absent":
        root.mkdir()
        if source_kind == "legacy":
            with WorkspaceDatabase(root).connect(read_only=False) as connection:
                connection.execute("CREATE TABLE old_bars (session_date DATE)")
        else:
            (root / "market-data.duckdb").write_bytes(b"not a database")
    before = {p.name: p.read_bytes() for p in root.glob("*") if p.is_file()}
    monkeypatch.setattr(
        WorkspaceWriterLease, "acquire", lambda *_: pytest.fail("inspection took a writer lease")
    )
    result = inspect_existing_data_workspace(root)
    assert result["status"] == "NOT_QUALIFIED"
    assert (
        result["failure_code"]
        == {
            "absent": "workspace_data_update.existing_data_required",
            "legacy": "workspace_data_update.initialization_required",
            "corrupt": "workspace_data_update.source_unreadable",
        }[source_kind]
    )
    assert result["message"] and result["next_action"]
    assert {p.name: p.read_bytes() for p in root.glob("*") if p.is_file()} == before
    if source_kind == "absent":
        assert not root.exists()


def test_cancel_requested_during_hydration_is_honoured_at_the_chunk_boundary(tmp_path):
    """Task Control's cancel, requested while the runner waits on the Provider.

    The request is accepted at once (Task Control's own store, not the
    runner's), the running chunk finishes and persists its units, the stage
    stops CANCELLED at the next chunk boundary -- the runner's cancel
    granularity is a chunk, no shorter -- and the Human's retry hydrates
    nothing again: the persisted units are reused. Nothing is held once the
    session has closed.
    """

    from datetime import timedelta

    from alphalattice.control.workspace_runtime.database import live_workspace_connections
    from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
    from tests.workspace_maintenance.local_data_provider import recording_provider

    publish_research_workspace_manifest(tmp_path, ResearchWorkspaceManifest.research_only("cancel"))
    provider = recording_provider()
    market = MarketDataRepository(tmp_path)
    with WorkspaceApplicationSession.acquire(tmp_path) as session:
        app = WorkspacePreparationApplication(
            session,
            clock=lambda: OBSERVED_AT,
            provider=provider,
            source_loader=_source_loader_for(("AAA", "BBB")),
        )
        task = app.confirm(app.plan()["plan_hash"], caller="HUMAN")
        accepted: list[str] = []
        fetch = provider.fetch_daily

        def cancel_while_fetching(symbols, **kwargs):
            # The runner is at its network edge; the request lands on Task
            # Control's store and is accepted before the fetch returns.
            if not accepted:
                current = session.task_control_registry.task(task.task_id)
                session.task_control_registry.request_cancel(
                    task_id=task.task_id,
                    expected_task_hash=current.record_hash,
                    observed_at=app.clock(),
                )
                accepted.append(session.task_control_registry.task(task.task_id).lifecycle.value)
            return fetch(symbols, **kwargs)

        provider.fetch_daily = cancel_while_fetching  # type: ignore[method-assign]
        app.execute(task.task_id)
        assert accepted == ["CANCEL_REQUESTED"]
        assert session.task_control_registry.task(task.task_id).lifecycle.value == "CANCELLED"
        # Honoured at the chunk boundary: the chunk's units are persisted and kept.
        disclosure = market.latest_current_universe_onboarding_disclosure(
            market_profile_id="us-current-index-research"
        )
        assert disclosure is not None
        assert disclosure["state_counts"] == {"FEATURE_READY": 2}
        hydrated = len(provider.calls)
        assert {call[0] for call in provider.calls} == {("AAA",), ("BBB",)}

        retry = WorkspacePreparationApplication(
            session,
            clock=lambda: OBSERVED_AT + timedelta(days=2),
            provider=provider,
            source_loader=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("retry must use its original captured membership")
            ),
        )
        proposed = retry.plan()
        assert proposed["resume_from_cancelled_task"] == str(task.task_id)
        resumed = retry.confirm(proposed["plan_hash"], caller="HUMAN")
        retry.execute(resumed.task_id)
        # The retry's data stage completed from the persisted units -- nothing
        # was hydrated again -- and the Task moved on to the Feature stage,
        # where this two-name fixture stops at its own population floor; that
        # stop is the fixture's size, not the cancellation's.
        assert not [call for call in provider.calls[hydrated:] if call[0] != ("SPY",)]
        assert provider.calls[hydrated:]  # the Feature stage's market reference, its own fetch
        progress = retry._load(resumed.task_id, "progress")
        assert progress is not None and progress["phase"] == "prepare_features"
        after = session.task_control_registry.task(resumed.task_id)
        assert (after.lifecycle.value, after.failure_code) == (
            "BLOCKED",
            "feature.baseline_qualified_population_insufficient",
        )
        # The readback tells one Task's story: asked for A it stays A (cancelled, A's own
        # progress record, no input of its own), unasked it discovers the latest (B), and a
        # Task that does not exist or is not a preparation is a typed refusal that names
        # the latest without becoming it.
        from uuid import uuid4

        from alphalattice.control.task_control.contracts import (
            ResearchGoal,
            ResearchPlan,
            TaskInputEnvelope,
            WorkItemDefinition,
        )

        selected = retry.readback(task.task_id)
        assert (selected["task_id"], selected["status"], selected["selected"]) == (
            str(task.task_id),
            "CANCELLED",
            True,
        )
        assert selected["latest_task_id"] == str(resumed.task_id)
        assert selected["progress"]["phase"] == "prepare_data"
        assert selected["inputs"] == [] and selected["published_binding_hash"] is None
        discovered = retry.readback()
        assert (discovered["task_id"], discovered["selected"]) == (str(resumed.task_id), False)
        missing = retry.readback(uuid4())
        assert (missing["status"], missing["failure_code"], missing["latest_task_id"]) == (
            "REFUSED",
            "workspace_preparation.task_not_found",
            str(resumed.task_id),
        )
        envelope = TaskInputEnvelope.create(
            task_kind="qa_other_kind", input_schema_id="qa-other-input", payload={"qa": 1}
        )
        goal = ResearchGoal.create(
            goal_kind="QA_OTHER",
            input_hash=envelope.input_hash,
            deliverable_kind="Nothing",
            summary="A Task of another kind, admitted for the refusal only.",
        )
        foreign = session.task_control_registry.admit(
            input_envelope=envelope,
            goal=goal,
            plan=ResearchPlan.create(
                goal_hash=goal.goal_hash,
                workflow_definition_hash="1" * 64,
                verifier_catalog_hash="2" * 64,
                work_items=(
                    WorkItemDefinition.create(
                        stage_id="only", dependency_ids=(), verifier_id="qa.only"
                    ),
                ),
            ),
            observed_at=retry.clock(),
        ).record
        mismatch = retry.readback(foreign.task_id)
        assert (mismatch["status"], mismatch["failure_code"], mismatch["task_kind"]) == (
            "REFUSED",
            "workspace_preparation.task_kind_mismatch",
            "qa_other_kind",
        )
        assert retry.readback()["task_id"] == str(resumed.task_id), "not the latest preparation"
    assert live_workspace_connections(market.path) is None
    # The writer lease went with the session: another session takes it at once.
    with WorkspaceApplicationSession.acquire(tmp_path):
        pass


@pytest.mark.parametrize(
    ("terminal", "failure_code", "decisions_ready"),
    [
        ("CANCELLED", None, None),
        ("BLOCKED", "data.preparation_fixture_fix_required", None),
        ("BLOCKED", "data.truth_review_required", False),
        ("BLOCKED", "data.truth_review_required", True),
    ],
)
def test_cancelled_preparation_reuses_captured_scope_without_new_discovery(
    tmp_path, monkeypatch, terminal, failure_code, decisions_ready
):
    from datetime import timedelta

    from alphalattice.control.product_host.data_preparation import application as preparation_owner
    from alphalattice.control.product_host.data_preparation.remediation import (
        WorkspaceDataIssueApplication,
    )
    from alphalattice.control.task_control.runner import StageDisposition, StageExecutionResult
    from tests.workspace_maintenance.local_data_provider import recording_provider

    publish_research_workspace_manifest(tmp_path, ResearchWorkspaceManifest.research_only("cancel"))
    provider = recording_provider()
    with WorkspaceApplicationSession.acquire(tmp_path) as session:

        def cancel_before_raw_work(app):
            perform = app._perform

            def wrapped(task, plan, stage, execution):
                if stage == "prepare_data":
                    if terminal == "BLOCKED":
                        return StageExecutionResult(
                            StageDisposition.BLOCKED,
                            failure_code=failure_code,
                        )
                    current = session.task_control_registry.task(task.task_id)
                    session.task_control_registry.request_cancel(
                        task_id=task.task_id,
                        expected_task_hash=current.record_hash,
                        observed_at=app.clock(),
                    )
                return perform(task, plan, stage, execution)

            app._perform = wrapped

        first = WorkspacePreparationApplication(
            session,
            clock=lambda: OBSERVED_AT,
            provider=provider,
            source_loader=_source_loader_for(("AAA", "BBB")),
        )
        cancel_before_raw_work(first)
        original = first.plan()
        task = first.confirm(original["plan_hash"], caller="HUMAN")
        first.execute(task.task_id)
        assert session.task_control_registry.task(task.task_id).lifecycle.value == terminal
        if terminal == "BLOCKED":
            monkeypatch.setattr(preparation_owner, "_implementation", lambda: "f" * 64)
        if decisions_ready is not None:
            monkeypatch.setattr(
                WorkspaceDataIssueApplication,
                "current_decisions_ready",
                lambda _owner: decisions_ready,
            )
            # The same registered implementation-change seam also holds truth recovery:
            # unresolved cases still lead to issues; only completed choices offer preview.
            stopped = session.task_control_registry.task(task.task_id)
            view = first.readback(task.task_id)
            assert view["execution_binding_changed"] is True
            assert view["failure_code"] == "data.truth_review_required"
            assert view["task_id"] == str(task.task_id)
            assert view["plan_hash"] == original["plan_hash"]
            assert view["confirmation_available"] is False
            assert view["next_action"] == (
                "WORKSPACE_PREPARE_PLAN" if decisions_ready else "DATA_ISSUES"
            )
            assert view["next_requests"] == (
                {} if decisions_ready else {"issues": {"operation": "DATA_ISSUES"}}
            )
            assert (
                session.task_control_registry.task(task.task_id).record_hash == stopped.record_hash
            )
            assert first.tasks() == (stopped,)
            assert provider.calls == []
            return

        def no_discovery(**_kwargs):
            raise AssertionError("retry must use its original captured membership")

        retry = WorkspacePreparationApplication(
            session,
            clock=lambda: OBSERVED_AT + timedelta(days=2),
            provider=provider,
            source_loader=no_discovery,
        )
        cancel_before_raw_work(retry)
        proposed = retry.plan()
        assert proposed["predecessor_task_id"] == str(task.task_id)
        assert proposed["resume_from_cancelled_task"] == (
            str(task.task_id) if terminal == "CANCELLED" else None
        )
        assert proposed["target_session"] == original["target_session"]
        resumed = retry.confirm(proposed["plan_hash"], caller="HUMAN")
        assert (
            resumed.task_id != task.task_id
        )  # Cancellation is terminal; Human requested a new task.
        retry.execute(resumed.task_id)
        assert session.task_control_registry.task(resumed.task_id).lifecycle.value == terminal
        assert provider.calls == []
        assert (
            retry._load(resumed.task_id, "freeze_sources")["candidate"]
            == first._load(task.task_id, "freeze_sources")["candidate"]
        )
        continuations = WorkspaceDataIssueApplication(session, retry.clock).readback()[
            "continuations"
        ]
        assert [item["task_id"] for item in continuations] == (
            [str(resumed.task_id)] if terminal == "BLOCKED" else []
        )


@pytest.mark.parametrize("lifecycle", ["QUEUED", "RUNNING", "CANCEL_REQUESTED"])
def test_a_plan_confirmed_while_its_task_runs_answers_that_task(lifecycle: str) -> None:
    """regression (V430, found by V410's survey): the confirm compared the lifecycle with
    `IN_PROGRESS`, which no Task holds, so a plan confirmed again while its Task ran was
    submitted again; every lifecycle still on its way answers the Task it has (a deferred one
    is not on its way, V506)."""

    from types import SimpleNamespace
    from uuid import uuid4

    from alphalattice.control.product_host.composition.portfolio_research_operations import (
        PortfolioResearchOperations,
    )
    from alphalattice.control.task_control.contracts import TaskLifecycle
    from alphalattice.interface.local_application.portfolio_research import (
        PortfolioResearchOperationRequest,
    )

    task_id = uuid4()
    running = SimpleNamespace(
        task_id=task_id,
        lifecycle=TaskLifecycle(lifecycle),
        input=SimpleNamespace(payload={"plan": {"plan_hash": "a" * 64}}),
    )

    class Preparation:
        """The preparation owner, holding the one Task of the plan."""

        def require_confirmation_caller(self, *_args: object, **_kwargs: object) -> None:
            return None

        def tasks(self) -> list[SimpleNamespace]:
            return [running]

    class Dispatcher:
        """A dispatcher no confirm of an in-flight plan may reach."""

        def submit(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("an in-flight plan was submitted again")

    operations = PortfolioResearchOperations.__new__(PortfolioResearchOperations)
    operations.preparation = Preparation()  # type: ignore[assignment]
    operations.dispatcher = Dispatcher()  # type: ignore[assignment]
    answer = operations._workspace_operation(
        PortfolioResearchOperationRequest(
            operation="WORKSPACE_PREPARE_CONFIRM", preparation_plan_hash="a" * 64
        ),
        caller="HUMAN",
    )
    assert answer == {"status": "REUSED_IN_FLIGHT", "task_id": str(task_id)}


def test_a_cancelled_successor_without_a_verified_stage_gives_its_source_back() -> None:
    """regression (V510, the user's review at 645fa9f4): a preparation A blocked on its truth
    review was continued by a successor B, and B was cancelled before it started. Every
    successor counted as superseding its source, so the next plan dropped A, found no checkpoint
    in B and was refused `workspace_preparation.existing_data_requires_explicit_binding`, while
    A's checkpoint still verified. A successor supersedes only while it can resume; one cancelled
    before any stage verified gives its source back. Its plan and the data issues' readback read
    the one rule."""

    import inspect
    from types import SimpleNamespace
    from uuid import uuid4

    from alphalattice.control.product_host.data_preparation import application, remediation
    from alphalattice.control.product_host.data_preparation.remediation import (
        superseded_preparations,
    )
    from alphalattice.control.task_control.contracts import TaskLifecycle, WorkItemLifecycle

    source = uuid4()
    items: dict[object, tuple[SimpleNamespace, ...]] = {}
    registry = SimpleNamespace(work_items=lambda task_id: items[task_id])

    def successor(lifecycle: TaskLifecycle, *stages: WorkItemLifecycle) -> SimpleNamespace:
        task = SimpleNamespace(
            task_id=uuid4(),
            task_kind="workspace_preparation",
            lifecycle=lifecycle,
            input=SimpleNamespace(payload={"source_task_id": str(source)}),
        )
        items[task.task_id] = tuple(SimpleNamespace(lifecycle=stage) for stage in stages)
        return task

    def held(task: SimpleNamespace) -> frozenset[str]:
        return superseded_preparations(registry, [task])  # type: ignore[arg-type]

    pending, verified = WorkItemLifecycle.PENDING, WorkItemLifecycle.VERIFIED
    assert held(successor(TaskLifecycle.CANCELLED, WorkItemLifecycle.CANCELLED)) == frozenset()
    assert held(successor(TaskLifecycle.CANCELLED, verified, pending)) == {str(source)}
    for lifecycle in TaskLifecycle:
        if lifecycle is not TaskLifecycle.CANCELLED:
            assert held(successor(lifecycle, pending)) == {str(source)}, lifecycle
    other = successor(TaskLifecycle.RUNNING, pending)
    other.task_kind = "workspace_data_update"
    assert held(other) == frozenset()
    # One rule: the plan and the data issues' readback both read it, and neither keeps its own.
    for owner in (
        application.WorkspacePreparationApplication.plan,
        remediation.WorkspaceDataIssueApplication.readback,
    ):
        code = inspect.getsource(owner)
        assert "superseded_preparations(" in code and "source_task_id" not in code, owner


def test_a_deferred_preparation_confirmed_again_goes_to_its_owner(tmp_path, monkeypatch) -> None:
    """regression (V506, the user's review at 244900d8): V430 counted a deferred preparation as
    in flight, so the `resume` its readback offers was answered REUSED_IN_FLIGHT before and
    after its retry time alike, and never reached the owner that judges that time (V375). Its
    plan confirmed again goes to the owner through the dispatcher, and the owner's answer, a
    refusal before the retry time among them, is the confirm's, worded."""

    from types import SimpleNamespace
    from uuid import uuid4

    from alphalattice.control.product_host.data_preparation.application import (
        WorkspacePreparationCommand,
    )
    from alphalattice.control.task_control.contracts import TaskLifecycle
    from alphalattice.interface.local_application.portfolio_research import (
        PortfolioResearchOperationRequest,
    )

    deferred = SimpleNamespace(
        task_id=uuid4(),
        lifecycle=TaskLifecycle.DEFERRED,
        input=SimpleNamespace(payload={"plan": {"plan_hash": "a" * 64}}),
    )
    sent: list[object] = []

    def submit(command: object) -> SimpleNamespace:
        """The owner refuses the plan before its retry time, as V375 has it."""
        sent.append(command)
        return SimpleNamespace(
            disposition="REFUSED_INVALID_COMMAND",
            task_id=None,
            lifecycle=None,
            refusal_detail="workspace_preparation.retry_not_due",
        )

    with LocalPortfolioWebSession.from_workspace(tmp_path, clock=lambda: OBSERVED_AT) as live:
        monkeypatch.setattr(live.operations.preparation, "tasks", lambda: [deferred])
        monkeypatch.setattr(
            live.operations.preparation, "require_confirmation_caller", lambda *_a, **_k: None
        )
        monkeypatch.setattr(live.dispatcher, "submit", submit)
        answer = live.operations._workspace_operation(
            PortfolioResearchOperationRequest(
                operation="WORKSPACE_PREPARE_CONFIRM", preparation_plan_hash="a" * 64
            ),
            caller="HUMAN",
        )
    (command,) = sent
    assert isinstance(command, WorkspacePreparationCommand) and command.plan_hash == "a" * 64
    assert answer is not None and answer["status"] == "REFUSED_INVALID_COMMAND", answer
    assert answer["failure_code"] == "workspace_preparation.retry_not_due"
    assert "`retry_after_at`" in str(answer["detail"])
