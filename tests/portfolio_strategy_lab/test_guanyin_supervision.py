"""Guanyin's Supervisor and Recovery Center on a live Host (GY2)."""

from __future__ import annotations


def test_the_supervisor_opens_an_incident_the_task_read_names_and_the_recovery_center_acts(
    live,
) -> None:
    """requirement (GY2, WK): the Supervisor finds a Task owed a recovery, the Task's read names
    its open incident so a follower wakes, the Recovery Center performs only a remedy the Host
    offers, through its existing operation, and records the attempt; the incident resolves once
    the Task has ended."""

    from uuid import uuid4

    from alphalattice.interface.local_application.portfolio_research import (
        PortfolioResearchOperationRequest,
    )
    from tests.workspace_task_runner.task_control_support import compatibility, task_contract

    registry = live.session.task_control_registry
    now = live.dispatcher.clock()
    envelope, goal, plan = task_contract(salt="supervised")
    task = registry.admit(input_envelope=envelope, goal=goal, plan=plan, observed_at=now).record
    started = registry.start_next(
        compatibility=compatibility(plan), worker_instance_id=uuid4(), observed_at=now
    )
    assert started is not None
    stopped = registry.mark_recovery_required(
        task_id=task.task_id, failure_code="TASK_EXECUTION_INTERRUPTED", observed_at=now
    )
    supervisor = live.operations.supervisor
    (opened,) = supervisor.supervise_once()
    assert opened.code == "task_runtime.recovery_required" and opened.state == "OPEN"
    assert supervisor.supervise_once() == ()  # the same incident is not opened twice
    read = live.operations.status(task.task_id)
    assert read["incident"]["key"] == opened.key
    assert read["incident"]["code"] == "task_runtime.recovery_required"
    ops = live.operations

    def remediate(remedy: str) -> dict:  # type: ignore[type-arg]
        return ops.execute(
            PortfolioResearchOperationRequest(
                operation="TASK_REMEDIATE",
                task_id=task.task_id,
                incident_key=opened.key,
                remedy=remedy,  # type: ignore[arg-type]
                expected_task_hash=stopped.record_hash,
            ),
            caller="HUMAN",
        )

    refused = remediate("RECOVER")  # no command of this Host runs a Factor Task of this kind
    assert refused["failure_code"] == "guanyin.remedy_not_offered"
    assert [r["action"] for r in refused["offered"]] == ["CANCEL"]
    attempted = remediate("CANCEL")
    assert attempted["status"] == "REMEDY_ATTEMPTED", attempted
    attempt = attempted["attempt"]
    assert (attempt["action"], attempt["selected_by"]) == ("CANCEL", "USER_COMMAND")
    # A stopped Task's cancellation is requested at once and acknowledged by the Host.
    assert attempt["lifecycle_after"] in {"CANCEL_REQUESTED", "CANCELLED"}
    # The receipt names the Task and where it stands, its outcome the Task's.
    assert attempted["task_id"] == str(task.task_id)
    assert attempted["lifecycle"] == attempt["lifecycle_after"]
    supervisor.supervise_once()
    (kept,) = [r for r in supervisor.store.records() if r.key == opened.key]
    assert kept.state == "RESOLVED" and len(kept.attempts) == 1
    assert kept.resolved_task_record_hash == registry.task(task.task_id).record_hash
    assert live.operations.status(task.task_id)["incident"] is None
    listed = ops.execute(PortfolioResearchOperationRequest(operation="TASK_INCIDENTS"))
    assert listed["open"] == 0 and listed["incidents"][0]["key"] == opened.key


def test_a_succeeded_linked_successor_does_not_reopen_the_stopped_source_incident(live) -> None:
    """An explicit successful re-plan resolves incident attention without changing its source."""
    from datetime import timedelta
    from uuid import uuid4

    from alphalattice.control.task_control.contracts import TaskEvidence, TaskStageReceipt
    from alphalattice.interface.local_application.portfolio_research import (
        PortfolioResearchOperationRequest,
    )
    from tests.workspace_task_runner.task_control_support import (
        compatibility,
        digest,
        task_contract,
    )

    registry = live.session.task_control_registry
    supervisor = live.operations.supervisor
    now = live.dispatcher.clock()
    source_input, source_goal, source_plan = task_contract(salt="attention-source")
    source = registry.admit(
        input_envelope=source_input,
        goal=source_goal,
        plan=source_plan,
        observed_at=now,
    ).record
    source_execution = registry.start_next(
        compatibility=compatibility(source_plan),
        worker_instance_id=uuid4(),
        observed_at=now + timedelta(seconds=1),
    )
    assert source_execution is not None
    stopped = registry.mark_recovery_required(
        task_id=source.task_id,
        failure_code="TASK_EXECUTION_INTERRUPTED",
        observed_at=now + timedelta(seconds=2),
    )
    (opened,) = supervisor.supervise_once()
    assert opened.task_id == str(source.task_id) and opened.state == "OPEN"
    request = {"operation": "FACTOR_PLAN", "research_input_id": "replanned-input"}
    registry.record_recovery_link(
        source_task_id=source.task_id,
        source_record_hash=stopped.record_hash,
        admission_request=request,
        observed_at=now + timedelta(seconds=3),
    )
    child_input, child_goal, child_plan = task_contract(salt="attention-successor")
    child = registry.admit(
        input_envelope=child_input,
        goal=child_goal,
        plan=child_plan,
        observed_at=now + timedelta(seconds=4),
    ).record
    registry.record_recovery_link(
        source_task_id=source.task_id,
        source_record_hash=stopped.record_hash,
        admission_request=request,
        successor_task_id=child.task_id,
        observed_at=now + timedelta(seconds=5),
    )
    child_execution = registry.start_next(
        compatibility=compatibility(child_plan),
        worker_instance_id=uuid4(),
        observed_at=now + timedelta(seconds=6),
    )
    assert child_execution is not None
    running_child, execution = child_execution
    assert running_child.task_id == child.task_id

    observed_at = now + timedelta(seconds=7)
    for definition in child_plan.work_items:
        item = registry.begin_work_item(
            task_id=child.task_id,
            execution_id=execution.execution_id,
            stage_id=definition.stage_id,
            observed_at=observed_at,
        )
        evidence = tuple(
            TaskEvidence(
                evidence_kind=kind,
                reference=f"playpen://task-evidence/{child.task_id}/{definition.stage_id}/{kind}",
                content_hash=digest(f"{child.task_id}:{definition.stage_id}:{kind}"),
            )
            for kind in definition.required_evidence_kinds
        )
        registry.mark_ready(
            task_id=child.task_id,
            execution_id=execution.execution_id,
            stage_id=definition.stage_id,
            evidence=evidence,
            observed_at=observed_at + timedelta(seconds=1),
        )
        registry.verify_work_item(
            TaskStageReceipt.from_identity(
                receipt_id=uuid4(),
                task_id=child.task_id,
                execution_id=execution.execution_id,
                stage_id=definition.stage_id,
                work_item_definition_hash=item.definition_hash,
                verifier_id=definition.verifier_id,
                evidence=evidence,
                status="VERIFIED",
                failure_code=None,
                observed_at=observed_at + timedelta(seconds=2),
            )
        )
        observed_at += timedelta(seconds=3)

    assert registry.task(child.task_id).lifecycle.value == "SUCCEEDED"
    assert registry.task(source.task_id).record_hash == stopped.record_hash
    assert supervisor.supervise_once() == ()

    assert registry.task(source.task_id).lifecycle.value == "RECOVERY_REQUIRED"
    assert supervisor.supervise_once() == ()
    assert registry.task(source.task_id).record_hash == stopped.record_hash
    listed = live.operations.execute(PortfolioResearchOperationRequest(operation="TASK_INCIDENTS"))
    assert listed["open"] == 0
    assert [row["key"] for row in listed["incidents"]] == [opened.key]
    assert listed["incidents"][0]["state"] == "RESOLVED"
    assert listed["incidents"][0]["resolved_task_record_hash"] == stopped.record_hash


def test_a_remedys_receipt_answers_for_the_task_it_acted_on(tmp_path) -> None:
    """regression (the user's review at 244900d8): the receipt nested the Task a recovery
    resumed in `answer`, so its outcome read OK and `--wait` ended at once. A recovery or a
    cancellation lifts the Task and its lifecycle, a refused one its code, so the receipt's
    outcome is the Task's and its owner's; a re-plan leaves the Task as it was."""

    from datetime import UTC, datetime
    from types import SimpleNamespace
    from uuid import uuid4

    from alphalattice.control.product_host.composition.task_supervision import TaskSupervisor
    from alphalattice.interface.local_application.cli_contract import outcome_of
    from alphalattice.interface.local_application.portfolio_research import (
        PortfolioResearchOperationRequest,
    )

    task_id = uuid4()
    owned = str(task_id)
    answers: list[dict[str, object]] = []
    after = {"lifecycle": "RUNNING", "verified_stage_count": 1}

    class Record:
        """An open incident on the Task, its attempts kept as the store keeps them."""

        task_id = owned
        state = "OPEN"
        attempts: tuple[object, ...] = ()

        def changed(self, **_values: object) -> Record:
            return self

        def model_dump(self, **_kwargs: object) -> dict[str, object]:
            return {"key": "b" * 64}

    class Store:
        def get(self, _key: str) -> Record:
            return Record()

        def put(self, _record: object) -> None:
            return None

    def action(name: str, operation: str) -> dict[str, object]:
        return {"action": name, "operation": operation, "available": True, "reason": "offered"}

    operations = SimpleNamespace(
        workspace_session=SimpleNamespace(workspace=tmp_path),
        recovery_view=lambda _t: {
            "actions": [action("RECOVER", "RECOVER"), action("REPLAN", "EXPERIMENT_PLAN")],
            "task_record_hash": "a" * 64,
            "next_requests": {"replan": {"operation": "EXPERIMENT_PLAN"}},
        },
        execute=lambda _request, caller: answers.pop(0),
        status=lambda _t: after,
        dispatcher=SimpleNamespace(clock=lambda: datetime(2026, 10, 2, 16, tzinfo=UTC)),
    )
    supervisor = TaskSupervisor(operations)  # type: ignore[arg-type]
    supervisor.store = Store()  # type: ignore[assignment]

    def remediate(remedy: str) -> dict[str, object]:
        request = PortfolioResearchOperationRequest(
            operation="TASK_REMEDIATE",
            task_id=task_id,
            incident_key="b" * 64,
            remedy=remedy,  # type: ignore[arg-type]
            expected_task_hash="a" * 64,
        )
        return supervisor.remediate(request, selected_by="USER_COMMAND")

    answers.append({"task_id": str(task_id), "lifecycle": "RUNNING"})
    resumed = remediate("RECOVER")
    assert (resumed["task_id"], resumed["lifecycle"]) == (str(task_id), "RUNNING")
    assert outcome_of(resumed) == "PENDING" and "failure_code" not in resumed
    replanned = remediate("REPLAN")
    assert "task_id" not in replanned and "lifecycle" not in replanned
    assert outcome_of(replanned) == "OK"
    answers.append({"status": "REFUSED", "failure_code": "local_application.confirmation_stale"})
    after["lifecycle"] = "RECOVERY_REQUIRED"
    stale = remediate("RECOVER")
    assert stale["failure_code"] == "local_application.confirmation_stale"
    assert outcome_of(stale) == "REFUSED"
