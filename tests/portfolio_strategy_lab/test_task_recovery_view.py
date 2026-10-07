"""Card 34: the recovery view of one Task, read from its owners over the real Host.

Every case boots the actual composition and drives it through the routes the
workbench and the CLI use. The synthetic Portfolio resolver stands in for
numerical work; interruption, cancellation and an owner's stop are produced at
the owners' own boundaries (a raising resolution, the cancel route inside the
runner's claim gap, the adapter's BLOCKED disposition), never by editing Task
Control. The view is Guanyin at G0: it explains and names what the owners permit,
and every mutation in these tests goes through the existing CANCEL and RECOVER
operations, bound to the exact Task version the confirmation was made against
and enforced by Task Control's own transaction -- the interleavings below move
the Task between the operation's entry and that boundary.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4

import pytest

from alphalattice.control.product_host.composition.local_web_session import (
    LocalPortfolioWebSession,
)
from alphalattice.control.product_host.composition.task_recovery import (
    TaskRecoveryView,
    build_task_recovery_view,
)
from alphalattice.control.task_control.contracts import (
    TaskExecution,
    TaskHeartbeatSignal,
    TaskLifecycle,
)
from alphalattice.control.task_control.registry import TaskBoardSnapshot
from alphalattice.control.task_control.runner import (
    StageDisposition,
    StageExecutionResult,
    TaskControlRunner,
    TaskHeartbeatReader,
    TaskHeartbeatReadout,
    _SqliteHeartbeatStore,
    heartbeat_store_path,
)
from alphalattice.interface.local_application.dispatcher import LocalBackgroundDispatcher
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchOperationRequest,
    PortfolioResearchRequestDocument,
)
from alphalattice.investment.portfolio_strategy_lab.application.task import (
    PortfolioResearchTaskAdapter,
)
from alphalattice.investment.portfolio_strategy_lab.application.tranche_book_execution import (
    EligiblePoolShort,
)
from tests.portfolio_strategy_lab.local_web_support import (
    _InterruptsOnce,
    _json,
    _manifest,
    _request,
    _resolved,
    _Resolver,
)


def _service(tmp_path: Path, resolver: Any, name: str = "qa-recovery") -> LocalPortfolioWebSession:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    return LocalPortfolioWebSession(
        workspace=workspace, workspace_manifest=_manifest(name), resolver=resolver
    )


def _view(live: LocalPortfolioWebSession, task_id: str) -> dict[str, Any]:
    body = _json(live, f"/api/tasks/recovery?task_id={task_id}")
    # The document is the typed contract, hash and all; the status block is the projection of
    # the same snapshot as the operation reader sees it (the dispatcher's mask), not a re-read,
    # and the offered requests stand beside it as navigation (V443).
    TaskRecoveryView.model_validate(
        {k: v for k, v in body.items() if k not in {"status", "next_requests"}}
    )
    assert body["status"]["task_id"] == task_id == body["task_id"]
    assert body["status"]["task_record_hash"] == body["task_record_hash"]
    assert "running_since" in body["status"]
    assert datetime.fromisoformat(body["status"]["last_activity_at"]).tzinfo is not None
    if body["status"]["running_since"] is not None:
        assert datetime.fromisoformat(body["status"]["running_since"]).tzinfo is not None
    assert body["guardian_mode"] == "G0_READ_ONLY"
    assert body["active_remediation_attempt_count"] == 0
    assert body["model_facts"] == "NOT_OBSERVED"
    assert body["guardian"] is None and body["guardian_availability"] == "NOT_PUBLISHED"
    assert body["health"]["task_id"] == task_id
    return body


def _only(readout: TaskHeartbeatReadout) -> TaskHeartbeatSignal | None:
    """The one signal a readout over one sidecar can hold."""

    assert len(readout.signals) <= 1 and readout.unreadable == ()
    return readout.signals[0] if readout.signals else None


def _actions(view: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["action"]: item for item in view["actions"]}


def test_interrupted_run_is_explained_preserved_and_resumed_as_the_same_task(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Interrupted work -> retained evidence -> explicit permitted recovery -> the same Task's
    readback, with a stale confirmation refused without mutation on the way."""

    interrupting = _InterruptsOnce(_Resolver(_resolved()))
    live = _service(tmp_path, interrupting)
    live.start()

    def cli_read(expected_exit: int, expected: dict[str, Any]) -> None:
        import json

        from alphalattice.interface.local_application.cli import main

        code = main(
            [
                "--workspace",
                str(live.workspace),
                "--view",
                "full",
                "recovery",
                "show",
                expected["task_id"],
            ],
            serve=lambda _: 99,
        )
        captured = capsys.readouterr()
        body = json.loads(captured.out)["data"]
        assert code == expected_exit and not captured.err
        assert body["task_record_hash"] == expected["task_record_hash"]
        assert body["status"]["lifecycle"] == body["lifecycle"] == expected["lifecycle"]

    try:
        admitted = _json(live, "/api/run", method="POST", payload={})
        task_id = admitted["task_id"]
        live.dispatcher.drain_for_tests()  # type: ignore[union-attr]
        assert interrupting.failures == 1
        pending_before_view = _json(live, "/api/decisions")
        assert any(
            decision["kind"] == "STOPPED_TASK" and decision["task_id"] == task_id
            for decision in pending_before_view["decisions"]
        )
        stopped = _view(live, task_id)
        cli_read(3, stopped)
        assert stopped["lifecycle"] == "RECOVERY_REQUIRED" and stopped["status"]["lifecycle"] == (
            "RECOVERY_REQUIRED"
        )
        assert stopped["attention"]["unresolved"] is True
        assert stopped["attention"]["resolution"] == "STOPPED"
        assert stopped["operation_running"] is False
        assert stopped["cancellation"] == "NOT_REQUESTED"
        # What stopped: Task Control's own code, explained, marked resumable by Task Control.
        assert stopped["stop"]["code"] == "TASK_EXECUTION_INTERRUPTED"
        status = _json(live, f"/api/status?task_id={task_id}")
        assert status["lifecycle"] == "RECOVERY_REQUIRED"
        assert status["attention"]["unresolved"] is True
        assert status["attention"]["resolution"] == "STOPPED"
        task_row = next(
            row for row in _json(live, "/api/tasks")["tasks"] if row["task_id"] == task_id
        )
        assert task_row["attention"]["unresolved"] is True
        assert any(
            decision["kind"] == "STOPPED_TASK" and decision["task_id"] == task_id
            for decision in _json(live, "/api/decisions")["decisions"]
        )
        assert status["detail"] and stopped["stop"]["code"] not in status["detail"]
        assert status["detail"].endswith(".")
        assert stopped["stop"]["recoverable"] is True
        assert "only the unverified stage again" in stopped["stop"]["detail"]
        assert stopped["worker_failure"]["failure_type"] == "RuntimeError"
        assert stopped["worker_failure"]["code"] == "qa.interrupted_mid_execution"
        # Unknown telemetry stays unknown: no execution is running, so no heartbeat is judged.
        assert stopped["liveness"]["status"] == "NOT_APPLICABLE"
        assert stopped["execution_binding_hash"] is not None  # the interrupted execution's
        # What is preserved: the plan's stage, unverified, with no invented evidence.
        assert stopped["verified_stage_count"] == 0 and stopped["total_stage_count"] == 1
        assert [s["lifecycle"] for s in stopped["stages"]] != ["VERIFIED"]
        assert all(s["evidence"] == [] and s["evidence_count"] == 0 for s in stopped["stages"])
        assert stopped["artifact_refs"] == []
        # What is permitted: cancel and resume, each with its scope and effect; a re-PLAN names
        # the installed Portfolio's own PLAN preview and RUN admission.
        actions = _actions(stopped)
        assert actions["CANCEL"]["available"] and actions["CANCEL"]["requires_confirmation"]
        assert "nothing is deleted or recomputed" in actions["CANCEL"]["scope"]
        assert "Requested is not acknowledged" in actions["CANCEL"]["expected_effect"]
        assert actions["RECOVER"]["available"] and actions["RECOVER"]["requires_confirmation"]
        assert actions["RECOVER"]["operation"] == "RECOVER" and actions["RECOVER"]["admits"]
        assert "no new Task and no new declaration" in actions["RECOVER"]["scope"]
        assert "An attempt" in actions["RECOVER"]["expected_effect"]
        assert "may stop again" in actions["RECOVER"]["expected_effect"]
        # V443: each one taking a version is offered bound to the Task and the version this
        # view read, so `recovery run --from` sends what was seen.
        bound = {"task_id": stopped["task_id"], "expected_task_hash": stopped["task_record_hash"]}
        version = stopped["task_record_hash"]
        assert stopped["next_requests"] == {
            "cancel": {"operation": actions["CANCEL"]["operation"], **bound},
            "recover": {"operation": "RECOVER", **bound},
            "replan": {
                **live.application.replan_request(
                    live.session.task_control_registry.task(UUID(task_id))
                ),
                "recovery_task_id": task_id,
                "recovery_task_hash": version,
            },
        }
        replan = live.operations.replans()[stopped["task_kind"]]  # type: ignore[union-attr]
        assert (replan.preview, replan.admitting) == ("PLAN", "RUN")
        assert actions["REPLAN"]["operation"] == "PLAN" and actions["REPLAN"]["admits"] is False
        assert "admits nothing" in actions["REPLAN"]["scope"]
        assert "Nothing is admitted until RUN is confirmed" in actions["REPLAN"]["expected_effect"]
        # The generic G0 pieces that are source-backed, and the ones that are not.
        assert stopped["health"]["status"] == "TERMINAL_DEFERRED"
        assert [i["incident_code"] for i in stopped["incidents"]] == ["TASK_EXECUTION_INTERRUPTED"]
        assert _json(live, f"/api/status?task_id={task_id}")["task_record_hash"] == version

        # A confirmation made against another version is refused at the entry and changes nothing.
        stale = _json(
            live,
            "/api/recover",
            method="POST",
            payload={"task_id": task_id, "expected_task_hash": "0" * 64},
        )
        assert stale["status"] == "REFUSED"
        assert stale["failure_code"] == "local_application.confirmation_stale"
        assert stale["refused_at"] == "OPERATION_ENTRY"
        assert stale["task_record_hash"] == version and stale["lifecycle"] == "RECOVERY_REQUIRED"
        unchanged = _view(live, task_id)
        assert unchanged["task_record_hash"] == version
        assert unchanged["lifecycle"] == "RECOVERY_REQUIRED"
        assert _json(live, "/api/results")["results"] == []
        assert len(_json(live, "/api/tasks")["tasks"]) == 1
        malformed = _json(
            live,
            "/api/cancel",
            method="POST",
            payload={"task_id": task_id, "expected_task_hash": "not-a-version"},
        )
        assert malformed["failure_code"] == "local_application.expected_task_hash_invalid"
        assert _view(live, task_id)["task_record_hash"] == version

        # The confirmed recovery of that exact version resumes the same Task through the
        # existing owner; it publishes once and is readable by its own identity.
        resumed = _json(
            live,
            "/api/recover",
            method="POST",
            payload={"task_id": task_id, "expected_task_hash": version},
        )
        assert resumed["resumed_task_ids"] == [task_id]
        live.dispatcher.drain_for_tests()  # type: ignore[union-attr]
        assert interrupting.failures == 1, "the interrupted stage ran once more, nothing else"
        done = _view(live, task_id)
        cli_read(0, done)
        assert done["lifecycle"] == "SUCCEEDED" and done["stop"] is None
        assert done["attention"]["unresolved"] is False
        assert done["attention"]["resolution"] == "NOT_STOPPED"
        assert not any(
            decision["kind"] == "STOPPED_TASK" and decision["task_id"] == task_id
            for decision in _json(live, "/api/decisions")["decisions"]
        )
        assert done["worker_failure"] is None
        assert done["task_record_hash"] != version
        assert done["verified_stage_count"] == done["total_stage_count"] == 1
        (stage,) = done["stages"]
        assert stage["lifecycle"] == "VERIFIED" and stage["evidence_count"] == len(
            stage["evidence"]
        )
        assert stage["evidence_count"] >= 1
        assert done["artifact_refs"] and set(done["artifact_refs"]) <= set(stage["evidence"])
        assert done["health"]["status"] == "TERMINAL_SUCCEEDED" and done["incidents"] == []
        actions = _actions(done)
        assert not actions["CANCEL"]["available"] and "has ended" in actions["CANCEL"]["reason"]
        assert not actions["RECOVER"]["available"]
        assert (
            "only an interrupted or parked queued Task can be resumed"
            in (actions["RECOVER"]["reason"])
        )
        assert actions["REPLAN"]["available"]
        results = _json(live, "/api/results")["results"]
        assert [r["task_id"] for r in results] == [task_id], "published once, by identity"
        report = _json(live, f"/api/report?result_hash={results[0]['result_hash']}")
        assert report["originating_task_id"] == task_id
        # Resuming again is not a second run: the owner answers with its disposition.
        again = _json(live, "/api/recover", method="POST", payload={"task_id": task_id})
        assert again["disposition"] == "NOT_RECOVERY_REQUIRED"
        assert len(_json(live, "/api/results")["results"]) == 1
    finally:
        live.stop()


def test_a_confirmed_recovery_is_refused_at_task_control_when_the_task_moved_meanwhile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The supervisor's interleaving: RECOVER confirmed against version A passes the entry
    check; before the queued command reaches Task Control, another real recovery runs and
    interrupts the Task again at version B. The confirmed resume is refused inside
    `restart_recovery`, starts nothing from B, and says so as the dispatcher's own fact."""

    interrupting = _InterruptsOnce(_Resolver(_resolved()), interruptions=2)
    live = _service(tmp_path, interrupting, "qa-race-recover")
    live.start()
    original = live.operations.recover_task  # type: ignore[union-attr]
    raced: list[str] = []

    def racing(task_id: UUID, *, expected_task_hash: str | None = None) -> tuple[UUID, ...]:
        if expected_task_hash is not None and not raced:
            raced.append(expected_task_hash)
            original(task_id)  # another actor's versionless recovery, in the gap
            live.dispatcher.drain_for_tests()  # type: ignore[union-attr]
        return original(task_id, expected_task_hash=expected_task_hash)

    monkeypatch.setattr(live.operations, "recover_task", racing)
    try:
        task_id = _json(live, "/api/run", method="POST", payload={})["task_id"]
        live.dispatcher.drain_for_tests()  # type: ignore[union-attr]
        version_a = _view(live, task_id)["task_record_hash"]
        accepted = _json(
            live,
            "/api/recover",
            method="POST",
            payload={"task_id": task_id, "expected_task_hash": version_a},
        )
        assert accepted["resumed_task_ids"] == [task_id], "enqueued: the entry check saw A"
        live.dispatcher.drain_for_tests()  # type: ignore[union-attr]
        assert raced == [version_a] and interrupting.failures == 2
        after = _view(live, task_id)
        version_b = after["task_record_hash"]
        assert version_b != version_a and after["lifecycle"] == "RECOVERY_REQUIRED"
        assert after["worker_failure"]["code"] == "task_control.recovery_version_stale"
        assert after["worker_failure"]["failure_type"] == "TaskVersionStale"
        assert "refused there; nothing was started or changed" in after["worker_failure"]["detail"]
        assert after["stop"]["code"] == "TASK_EXECUTION_INTERRUPTED", "Task Control's own fact"
        assert _json(live, "/api/results")["results"] == []
        assert interrupting.failures == 2, "the refused attempt never reached the resolver"
        # A confirmation against B is what starts the recovery.
        renewed = _json(
            live,
            "/api/recover",
            method="POST",
            payload={"task_id": task_id, "expected_task_hash": version_b},
        )
        assert renewed["resumed_task_ids"] == [task_id]
        live.dispatcher.drain_for_tests()  # type: ignore[union-attr]
        final = _view(live, task_id)
        assert final["lifecycle"] == "SUCCEEDED" and final["worker_failure"] is None
        assert len(_json(live, "/api/results")["results"]) == 1
    finally:
        live.stop()


def test_a_confirmed_cancel_is_refused_at_task_control_when_the_task_moved_meanwhile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CANCEL confirmed against A; between the entry check and the dispatcher's request,
    a real recovery moves the Task to B. The registry refuses the confirmed version and B is
    not cancelled; the refusal names Task Control, not the entry."""

    interrupting = _InterruptsOnce(_Resolver(_resolved()), interruptions=2)
    live = _service(tmp_path, interrupting, "qa-race-cancel")
    live.start()
    original = LocalBackgroundDispatcher.request_cancel
    raced: list[str] = []

    def racing(
        self: LocalBackgroundDispatcher, task_id: UUID, *, expected_task_hash: str | None = None
    ) -> bool:
        if expected_task_hash is not None and not raced:
            raced.append(expected_task_hash)
            live.resume(task_id)  # another actor's versionless recovery, in the gap
            live.dispatcher.drain_for_tests()  # type: ignore[union-attr]
        return original(self, task_id, expected_task_hash=expected_task_hash)

    monkeypatch.setattr(LocalBackgroundDispatcher, "request_cancel", racing)
    try:
        task_id = _json(live, "/api/run", method="POST", payload={})["task_id"]
        live.dispatcher.drain_for_tests()  # type: ignore[union-attr]
        version_a = _view(live, task_id)["task_record_hash"]
        refused = _json(
            live,
            "/api/cancel",
            method="POST",
            payload={"task_id": task_id, "expected_task_hash": version_a},
        )
        assert refused["status"] == "REFUSED"
        assert refused["failure_code"] == "local_application.confirmation_stale"
        assert refused["refused_at"] == "TASK_CONTROL"
        assert raced == [version_a]
        after = _view(live, task_id)
        version_b = after["task_record_hash"]
        assert version_b != version_a and refused["task_record_hash"] == version_b
        assert after["lifecycle"] == "RECOVERY_REQUIRED", "B was not cancelled"
        assert after["cancellation"] == "NOT_REQUESTED"
        # The confirmation renewed against B is what cancels; an idle owned Task finalizes.
        cancelled = _json(
            live,
            "/api/cancel",
            method="POST",
            payload={"task_id": task_id, "expected_task_hash": version_b},
        )
        assert cancelled["cancel_requested"] is True
        final = _view(live, task_id)
        assert final["lifecycle"] == "CANCELLED" and final["cancellation"] == "ACKNOWLEDGED"
        assert final["stop"]["code"] == "TASK_CANCELLED_AFTER_WRITER_STOPPED"
        assert _json(live, "/api/results")["results"] == []
    finally:
        live.stop()


def test_cancellation_requested_is_shown_apart_from_acknowledged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cancel route is driven inside the running stage; the view says REQUESTED with a
    live heartbeat and never ACKNOWLEDGED until the runner's own checkpoint says so."""

    resolver = _Resolver(_resolved())
    live = _service(tmp_path, resolver, "qa-cancel")
    live.start()
    inside: list[dict[str, Any]] = []
    resolve = resolver.resolve

    def resolve_then_cancel(**kwargs: Any) -> Any:
        value = resolve(**kwargs)
        if not inside:
            task_id = _json(live, "/api/tasks")["tasks"][0]["task_id"]
            running = _view(live, task_id)
            cancelled = _json(live, "/api/cancel", method="POST", payload={"task_id": task_id})
            inside.append({"running": running, "cancel": cancelled, "after": _view(live, task_id)})
        return value

    monkeypatch.setattr(resolver, "resolve", resolve_then_cancel)
    try:
        admitted = _json(live, "/api/run", method="POST", payload={})
        task_id = admitted["task_id"]
        live.dispatcher.drain_for_tests()  # type: ignore[union-attr]
        (seen,) = inside
        running, after = seen["running"], seen["after"]
        assert running["lifecycle"] == "RUNNING" and running["operation_running"] is True
        assert running["cancellation"] == "NOT_REQUESTED"
        assert running["liveness"]["status"] == "OBSERVED"
        assert running["health"]["status"] == "HEALTHY"
        assert _actions(running)["CANCEL"]["available"] is True
        assert seen["cancel"]["cancel_requested"] is True
        assert after["cancellation"] == "REQUESTED" and after["lifecycle"] == "CANCEL_REQUESTED"
        assert after["status"]["cancel_pending"] is True
        cancel = _actions(after)["CANCEL"]
        assert cancel["available"] is False and "not acknowledged" in cancel["reason"]
        assert "ACKNOWLEDGED" not in (after["cancellation"], after["stop"])
        final = _view(live, task_id)
        assert final["cancellation"] in {"ACKNOWLEDGED", "NOT_REQUESTED"}
        if final["cancellation"] == "ACKNOWLEDGED":
            assert final["lifecycle"] == "CANCELLED"
            assert final["stop"]["code"].startswith("TASK_CANCELLED")
            assert "acknowledged at a safe checkpoint" in final["stop"]["detail"]
            assert final["health"]["status"] == "TERMINAL_BLOCKED"
        assert final["operation_running"] is False
    finally:
        live.stop()


def test_a_queued_task_cancelled_before_execution_reads_cancelled_while_its_command_returns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Task Control's state and the dispatcher's operation-return mask stay apart: the cancel
    lands in the runner's claim gap (the command has not returned), the registry says
    CANCELLED / TASK_CANCELLED_BEFORE_START, and the view says exactly that -- never
    RUNNING, NOT_REQUESTED, HEALTHY -- while `operation_running` and the masked status block
    carry the dispatcher's fact."""

    live = _service(tmp_path, _Resolver(_resolved()), "qa-gap")
    live.start()
    run_next = TaskControlRunner.run_next
    inside: list[dict[str, Any]] = []

    def cancel_before_claim(runner: TaskControlRunner, **kwargs: Any) -> Any:
        task_id = str(kwargs["expected_task_id"])
        _json(live, "/api/cancel", method="POST", payload={"task_id": task_id})
        inside.append(_view(live, task_id))
        return run_next(runner, **kwargs)

    monkeypatch.setattr(TaskControlRunner, "run_next", cancel_before_claim)
    try:
        task_id = _json(live, "/api/run", method="POST", payload={})["task_id"]
        live.dispatcher.drain_for_tests()  # type: ignore[union-attr]
        (gap,) = inside
        assert gap["operation_running"] is True, "read while the command had not returned"
        assert gap["lifecycle"] == "CANCELLED" and gap["cancellation"] == "ACKNOWLEDGED"
        assert gap["stop"]["code"] == "TASK_CANCELLED_BEFORE_START"
        assert gap["health"]["status"] == "TERMINAL_BLOCKED"
        assert gap["liveness"]["status"] == "NOT_APPLICABLE"
        assert gap["status"]["task_record_hash"] == gap["task_record_hash"]
        assert _actions(gap)["CANCEL"]["available"] is False
        assert _actions(gap)["RECOVER"]["available"] is False
        final = _view(live, task_id)
        assert final["lifecycle"] == "CANCELLED" and final["operation_running"] is False
        assert _json(live, "/api/results")["results"] == []
    finally:
        live.stop()


def test_an_owner_stop_remains_a_stop_and_admits_no_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stage the domain owner refuses to complete blocks the Task: the view says so with
    the owner's code, offers no recovery, and RECOVER answers without touching it."""

    def refuse(self: Any, **kwargs: Any) -> StageExecutionResult:
        return StageExecutionResult(
            disposition=StageDisposition.BLOCKED,
            failure_code="portfolio_application.no_admissible_candidate",
        )

    monkeypatch.setattr(PortfolioResearchTaskAdapter, "execute_stage", refuse)
    live = _service(tmp_path, _Resolver(_resolved()), "qa-stop")
    live.start()
    try:
        admitted = _json(live, "/api/run", method="POST", payload={})
        task_id = admitted["task_id"]
        live.dispatcher.drain_for_tests()  # type: ignore[union-attr]
        view = _view(live, task_id)
        assert view["lifecycle"] == "BLOCKED"
        assert view["stop"]["code"] == "portfolio_application.no_admissible_candidate"
        assert view["stop"]["recoverable"] is False
        assert "Nothing was retried" in view["stop"]["detail"]
        assert view["health"]["status"] == "TERMINAL_BLOCKED"
        assert [i["incident_code"] for i in view["incidents"]] == [
            "portfolio_application.no_admissible_candidate"
        ]
        actions = _actions(view)
        assert actions["RECOVER"]["available"] is False
        assert actions["CANCEL"]["available"] is False
        assert actions["REPLAN"]["available"] is True
        # The PM's one read of every unfinished Task carries the same facts (GY, V78).
        guardian = _json(live, "/api/tasks/guardian")
        (entry,) = [item for item in guardian["tasks"] if item["task_id"] == task_id]
        assert entry["health"]["status"] == "TERMINAL_BLOCKED" and entry["lifecycle"] == "BLOCKED"
        assert [i["incident_code"] for i in entry["incidents"]] == [
            "portfolio_application.no_admissible_candidate"
        ]
        assert [action["action"] for action in entry["actions"]] == ["REPLAN"]
        assert entry["progress"]["total_stage_count"] == view["total_stage_count"]
        assert entry["next_requests"]["recovery"] == {
            "operation": "TASK_RECOVERY",
            "task_id": task_id,
        }
        # A read of the blocked Task answers REFUSED, so it says why and where it goes on, as a
        # refusal does (V424, AX14: it named neither).
        from alphalattice.interface.local_application.cli_contract import outcome_of

        status = _json(live, f"/api/status?task_id={task_id}")
        assert outcome_of(status) == "REFUSED"
        assert status["failure_code"] == "portfolio_application.no_admissible_candidate"
        assert isinstance(status["detail"], str) and status["detail"], status
        assert status["next_requests"]["recovery"] == {
            "operation": "TASK_RECOVERY",
            "task_id": task_id,
        }
        answer = _json(live, "/api/recover", method="POST", payload={"task_id": task_id})
        assert answer["disposition"] == "NOT_RECOVERY_REQUIRED" and answer["lifecycle"] == "BLOCKED"
        assert _json(live, "/api/results")["results"] == []
        assert _view(live, task_id)["task_record_hash"] == view["task_record_hash"]
    finally:
        live.stop()


def test_a_strategy_book_too_short_for_a_rebalance_stops_before_its_walk_in_words(
    tmp_path: Path,
) -> None:
    """requirement (V519, V500's class): a strategy book whose resolved scores leave a
    formation fewer tradable, scored names than a rebalance selects stops before its walk,
    by that session, never in the middle of it, and the read of the stopped Task says why in
    the door's words with the way on, as a refusal does (OP4)."""

    from dataclasses import replace

    import numpy as np

    from alphalattice.interface.local_application.cli_contract import outcome_of, refusal_words

    resolved = _resolved()
    (component,) = resolved.components
    short = component.formations[1]
    scores = np.where(np.arange(short.scores.size) < 20, short.scores, np.nan)
    formations = (component.formations[0], replace(short, scores=scores), *component.formations[2:])
    numerical = replace(resolved, components=(replace(component, formations=formations),))
    # The run reads this resolution verbatim, its admission the unchanged one.
    live = _service(tmp_path, _Resolver(resolved, numerical=numerical), "qa-short-pool")
    live.start()
    try:
        admitted = _json(live, "/api/run", method="POST", payload={})
        task_id = admitted["task_id"]
        live.dispatcher.drain_for_tests()  # type: ignore[union-attr]
        code = f"portfolio_strategy_lab.eligible_pool_short:{short.formation_session}"
        status = _json(live, f"/api/status?task_id={task_id}")
        assert outcome_of(status) == "REFUSED"
        assert status["failure_code"] == code, status
        assert status["detail"] == refusal_words(code)["detail"], status
        assert str(short.formation_session) in status["detail"]
        assert status["next_requests"]["recovery"] == {
            "operation": "TASK_RECOVERY",
            "task_id": task_id,
        }
        assert _json(live, "/api/results")["results"] == []
    finally:
        live.stop()


def test_unavailable_activity_telemetry_leaves_recovery_usable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The observation store cannot open; the view and the recovery operation still answer
    from Task Control and the dispatcher, and the feed says UNAVAILABLE beside them. The
    runner's heartbeat sidecar is then refused by the filesystem at its existence check
    (the supervisor's injection, which the route answered with 500): the same view still
    answers, says UNREADABLE and permits the same recovery of the same Task version."""

    workspace = tmp_path / "workspace"
    (workspace / "runtime" / "observations.sqlite").mkdir(parents=True)  # a directory, not a store
    interrupting = _InterruptsOnce(_Resolver(_resolved()))
    live = _service(tmp_path, interrupting, "qa-degraded")
    live.start()
    try:
        assert live.activity is not None and not live.activity.available
        admitted = _json(live, "/api/run", method="POST", payload={})
        task_id = admitted["task_id"]
        live.dispatcher.drain_for_tests()  # type: ignore[union-attr]
        stopped = _view(live, task_id)
        assert stopped["lifecycle"] == "RECOVERY_REQUIRED"
        assert _actions(stopped)["RECOVER"]["available"] is True
        feed = _json(live, f"/api/activity?watch={task_id}")
        assert feed["disposition"] == "UNAVAILABLE" and feed["items"] == []
        # The filesystem refuses the sidecar itself, at the exact path the reader checks.
        sidecar = live.operations.heartbeats.paths[0]
        assert sidecar == heartbeat_store_path(Path(live.session.runtime_path).resolve())  # type: ignore[union-attr]
        is_file = Path.is_file

        def refused(self: Path, *args: Any, **kwargs: Any) -> bool:
            if self == sidecar:
                raise PermissionError(13, "Permission denied", str(self))
            return is_file(self, *args, **kwargs)

        with monkeypatch.context() as patched:
            patched.setattr(Path, "is_file", refused)
            assert live.operations.heartbeats.read(uuid4()) == TaskHeartbeatReadout(
                signals=(), unreadable=("task_control.heartbeat_store_unreadable",)
            )
            refused_view = _view(live, task_id)
        assert refused_view["lifecycle"] == "RECOVERY_REQUIRED"
        assert refused_view["task_record_hash"] == stopped["task_record_hash"]
        assert refused_view["liveness"]["telemetry"] == "UNREADABLE"
        assert refused_view["liveness"]["status"] == "NOT_APPLICABLE"
        assert _actions(refused_view)["RECOVER"]["available"] is True
        assert _actions(refused_view) == _actions(stopped)
        restored = _view(live, task_id)
        assert restored["liveness"]["telemetry"] != "UNREADABLE"
        assert restored["task_record_hash"] == stopped["task_record_hash"]
        resumed = _json(
            live,
            "/api/recover",
            method="POST",
            payload={"task_id": task_id, "expected_task_hash": stopped["task_record_hash"]},
        )
        assert resumed["resumed_task_ids"] == [task_id]
        live.dispatcher.drain_for_tests()  # type: ignore[union-attr]
        assert _view(live, task_id)["lifecycle"] == "SUCCEEDED"
        assert len(_json(live, "/api/results")["results"]) == 1
    finally:
        live.stop()


class _Held:
    """A resolver that waits inside its numerical resolution until released, so a Task is
    genuinely RUNNING with a live runner heartbeat while the view is read."""

    def __init__(self, inner: _Resolver) -> None:
        self.inner = inner
        self.entered = threading.Event()
        self.release = threading.Event()

    def resolve_authorities(self, **kwargs: Any) -> Any:
        return self.inner.resolve_authorities(**kwargs)

    @property
    def strategy_catalog_hash(self) -> str:
        return self.inner.strategy_catalog_hash

    def installed_packages(self) -> Any:
        return self.inner.installed_packages()

    def resolve(self, **kwargs: Any) -> Any:
        self.entered.set()
        assert self.release.wait(60), "the test releases the held resolution"
        return self.inner.resolve(**kwargs)


def test_liveness_reads_the_runners_operational_heartbeat_not_only_the_durable_timestamp(
    tmp_path: Path,
) -> None:
    """The supervisor's reproduction: a Task held inside its resolver, the service clock
    advanced 60 s, and the runner's own second background heartbeat. Task Control's durable
    timestamp is the start (60 s old); the runner's sidecar signal (sequence 2, bound to
    this Task, execution and worker) is 0 s old, and the view reads OBSERVED by that
    signal -- the way `stale_active_tasks` reads it -- without instantiating a runner."""

    base = datetime(2026, 9, 15, 10, 0, tzinfo=UTC)
    offset = [0.0]
    held = _Held(_Resolver(_resolved()))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    live = LocalPortfolioWebSession(
        workspace=workspace,
        workspace_manifest=_manifest("qa-liveness"),
        resolver=held,
        clock=lambda: base + timedelta(seconds=offset[0]),
    )
    live.start()
    try:
        task_id = _json(live, "/api/run", method="POST", payload={})["task_id"]
        assert held.entered.wait(30), "the resolver was entered"
        registry = live.session.task_control_registry  # type: ignore[union-attr]
        execution_id = registry.task(UUID(task_id)).latest_execution_id
        assert execution_id is not None
        reader = TaskHeartbeatReader(live.session.runtime_path)  # type: ignore[union-attr]
        offset[0] = 60.0  # the service clock moves; nothing else does
        deadline = datetime.now(UTC) + timedelta(seconds=20)
        signal = _only(reader.read(execution_id))
        while (signal is None or signal.observed_at < base + timedelta(seconds=60)) and (
            datetime.now(UTC) < deadline
        ):
            threading.Event().wait(0.2)
            signal = _only(reader.read(execution_id))
        assert signal is not None and signal.sequence >= 2, "the runner's own later heartbeat"
        assert signal.task_id == UUID(task_id) and signal.execution_id == execution_id
        assert signal.observed_at == base + timedelta(seconds=60)
        view = _view(live, task_id)
        assert view["lifecycle"] == "RUNNING" and view["operation_running"] is True
        liveness = view["liveness"]
        assert liveness["status"] == "OBSERVED", liveness
        assert liveness["telemetry"] == "OPERATIONAL"
        assert liveness["last_heartbeat_source"] == "OPERATIONAL_SIGNAL"
        assert liveness["signal_sequence"] == signal.sequence
        assert liveness["age_seconds"] == 0.0 == liveness["operational_age_seconds"]
        assert datetime.fromisoformat(liveness["operational_at"]) == signal.observed_at
        durable = registry.execution(execution_id).last_heartbeat_at
        assert datetime.fromisoformat(liveness["durable_at"]) == durable
        assert durable == base, "the durable timestamp is the start, 60 s old, and not the verdict"
        for status in (
            view["status"],
            _json(live, "/api/status?task_id=" + task_id),
            next(t for t in _json(live, "/api/tasks")["tasks"] if t["task_id"] == task_id),
        ):
            assert datetime.fromisoformat(status["running_since"]) == base
            assert datetime.fromisoformat(status["last_activity_at"]) == base
        assert f"heartbeat #{signal.sequence}" in liveness["note"]
        assert view["health"]["status"] == "HEALTHY"
        assert view["health"]["heartbeat_age_seconds"] == 0.0
        assert view["incidents"] == []
    finally:
        held.release.set()
        live.dispatcher.drain_for_tests()  # type: ignore[union-attr]
        assert _view(live, task_id)["lifecycle"] == "SUCCEEDED"
        live.stop()


def test_stale_unreadable_or_unbound_telemetry_never_establishes_current_liveness(
    tmp_path: Path,
) -> None:
    """With the same held Task and the clock 60 s ahead, the builder is fed the runner's
    signal through the read-only reader over controlled sidecars: a fresh signal of a
    previous execution is not consulted (the store is keyed by execution), a signal for
    this execution from another worker is unbound and not counted, an unreadable sidecar is
    said so, and only a signal bound to this execution and worker reads as observed. A
    sidecar that cannot be read never hides the bound signal another sidecar holds, and
    when no bound signal can be read, unreadable is said before unbound. The reported age
    follows the later timestamp and names its source: a bound signal older than Task
    Control's durable timestamp is available, at its own age, and not the source. None of
    it changes lifecycle authority or the permitted actions."""

    base = datetime(2026, 9, 15, 11, 0, tzinfo=UTC)
    offset = [0.0]
    held = _Held(_Resolver(_resolved()))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    live = LocalPortfolioWebSession(
        workspace=workspace,
        workspace_manifest=_manifest("qa-liveness-bound"),
        resolver=held,
        clock=lambda: base + timedelta(seconds=offset[0]),
    )
    live.start()
    try:
        task_id = _json(live, "/api/run", method="POST", payload={})["task_id"]
        assert held.entered.wait(30)
        registry = live.session.task_control_registry  # type: ignore[union-attr]
        offset[0] = 60.0
        snapshot = registry.board(UUID(task_id))
        execution = snapshot.execution
        assert execution is not None and snapshot.projection.lifecycle is TaskLifecycle.RUNNING
        now = base + timedelta(seconds=60)
        controlled = tmp_path / "controlled-checkpoints.sqlite"  # a sidecar the test writes
        store = _SqliteHeartbeatStore(heartbeat_store_path(controlled))
        reader = TaskHeartbeatReader(controlled)
        kinds = frozenset({"portfolio_public_development_replay"})

        def readout(*signals: TaskHeartbeatSignal, unreadable: int = 0) -> TaskHeartbeatReadout:
            return TaskHeartbeatReadout(
                signals=signals,
                unreadable=("task_control.heartbeat_store_unreadable",) * unreadable,
            )

        def build(heartbeats: TaskHeartbeatReadout, at: TaskBoardSnapshot = snapshot) -> Any:
            return build_task_recovery_view(
                snapshot=at,
                running=True,
                worker_failure=None,
                heartbeats=heartbeats,
                recoverable_kinds=kinds,
                observed_at=now,
                replans=live.operations.replans(),  # type: ignore[union-attr]
            )

        # No signal for this execution: the durable timestamp alone, 60 s old, says not recent.
        assert reader.read(execution.execution_id) == readout()
        durable_only = build(readout())
        assert durable_only.liveness.status == "NOT_RECENT"
        assert durable_only.liveness.telemetry == "DURABLE_ONLY"
        assert durable_only.liveness.last_heartbeat_source == "DURABLE_TIMESTAMP"
        assert durable_only.liveness.age_seconds == 60.0
        assert durable_only.liveness.operational_age_seconds is None
        assert "no runner heartbeat is recorded for this execution" in durable_only.liveness.note
        assert durable_only.health.status == "LIVENESS_STALE"
        # A fresh signal of a previous execution of this Task is never this execution's.
        previous = uuid4()
        store.touch(
            TaskHeartbeatSignal.from_identity(
                task_id=execution.task_id,
                execution_id=previous,
                worker_instance_id=execution.worker_instance_id,
                sequence=9,
                observed_at=now,
            )
        )
        assert reader.read(execution.execution_id) == readout(), "keyed by execution, not Task"
        assert build(reader.read(execution.execution_id)).liveness.telemetry == "DURABLE_ONLY"
        # A signal for this execution from another worker is unbound and not counted.
        store.touch(
            TaskHeartbeatSignal.from_identity(
                task_id=execution.task_id,
                execution_id=execution.execution_id,
                worker_instance_id=uuid4(),
                sequence=3,
                observed_at=now,
            )
        )
        foreign = _only(reader.read(execution.execution_id))
        assert foreign is not None and foreign.worker_instance_id != execution.worker_instance_id
        unbound = build(reader.read(execution.execution_id))
        assert unbound.liveness.status == "NOT_RECENT" and unbound.liveness.telemetry == "UNBOUND"
        assert unbound.liveness.signal_sequence is None and unbound.liveness.operational_at is None
        assert unbound.liveness.last_heartbeat_source == "DURABLE_TIMESTAMP"
        assert "not bound to this execution's worker" in unbound.liveness.note
        # Beside a sidecar that could not be read, the unbound row settles nothing: the bound
        # signal may be in the store that could not be read, and that is what is said.
        hidden = build(readout(foreign, unreadable=1))
        assert hidden.liveness.telemetry == "UNREADABLE" and hidden.liveness.status == "NOT_RECENT"
        # Only the signal bound to this Task, execution and worker establishes liveness.
        store.touch(
            TaskHeartbeatSignal.from_identity(
                task_id=execution.task_id,
                execution_id=execution.execution_id,
                worker_instance_id=execution.worker_instance_id,
                sequence=4,
                observed_at=now - timedelta(seconds=3),
            )
        )
        bound_signal = _only(reader.read(execution.execution_id))
        assert bound_signal is not None and bound_signal.sequence == 4
        bound = build(reader.read(execution.execution_id))
        assert bound.liveness.status == "OBSERVED" and bound.liveness.telemetry == "OPERATIONAL"
        assert bound.liveness.last_heartbeat_source == "OPERATIONAL_SIGNAL"
        assert bound.liveness.signal_sequence == 4 and bound.liveness.age_seconds == 3.0
        assert bound.liveness.operational_age_seconds == 3.0
        assert (
            bound.liveness.note == "A heartbeat was recorded 3 s ago by the runner's heartbeat #4."
        )
        assert bound.health.status == "HEALTHY"
        # An unreadable sidecar is reported, not guessed; the store is read-only from here.
        garbage = heartbeat_store_path(tmp_path / "garbage-checkpoints.sqlite")
        garbage.write_bytes(b"not a database")
        assert TaskHeartbeatReader(tmp_path / "garbage-checkpoints.sqlite").read(
            execution.execution_id
        ) == readout(unreadable=1)
        unreadable = build(readout(unreadable=1))
        assert unreadable.liveness.telemetry == "UNREADABLE"
        assert unreadable.liveness.status == "NOT_RECENT"
        assert unreadable.liveness.last_heartbeat_source == "DURABLE_TIMESTAMP"
        assert "could not be read, so a signal for this execution may be hidden" in (
            unreadable.liveness.note
        )
        # The supervisor's two sidecars: a first SQLite store without the heartbeat table and
        # the second holding the bound signal. Read together, the first is reported unreadable
        # and the second's signal is exactly the one read alone; the view reads it as observed.
        tableless = tmp_path / "tableless-checkpoints.sqlite"
        with sqlite3.connect(heartbeat_store_path(tableless)) as first:
            first.execute("CREATE TABLE something_else (id INTEGER PRIMARY KEY)")
        first.close()
        combined = TaskHeartbeatReader(tableless, controlled).read(execution.execution_id)
        assert combined == readout(bound_signal, unreadable=1)
        assert combined.signals == (bound_signal,)
        second_owner = build(combined)
        assert second_owner.liveness.telemetry == "OPERATIONAL"
        assert second_owner.liveness.status == "OBSERVED"
        assert second_owner.liveness.signal_sequence == 4
        assert (
            build(TaskHeartbeatReader(tableless).read(execution.execution_id)).liveness.telemetry
            == "UNREADABLE"
        )
        # The reported age follows the later timestamp and names its source. Task Control's
        # durable timestamp moved to now (a stage boundary) while the runner's bound signal #5
        # is a minute old: the age is 0 s by the durable timestamp, the signal is available at
        # its own age of 60 s, and nothing says it refreshed. Older durable, newer signal: the
        # signal is the source.
        older_signal = TaskHeartbeatSignal.from_identity(
            task_id=execution.task_id,
            execution_id=execution.execution_id,
            worker_instance_id=execution.worker_instance_id,
            sequence=5,
            observed_at=now - timedelta(seconds=60),
        )
        durable_newer = TaskBoardSnapshot(
            task=snapshot.task,
            work_items=snapshot.work_items,
            execution=TaskExecution.from_identity(
                **execution.model_dump(exclude={"execution_hash"}) | {"last_heartbeat_at": now}
            ),
            projection=snapshot.projection,
        )
        by_durable = build(readout(older_signal), at=durable_newer)
        assert by_durable.liveness.status == "OBSERVED"
        assert by_durable.liveness.telemetry == "OPERATIONAL"
        assert by_durable.liveness.last_heartbeat_source == "DURABLE_TIMESTAMP"
        assert (
            by_durable.liveness.age_seconds == 0.0 and by_durable.liveness.last_heartbeat_at == now
        )
        assert by_durable.liveness.durable_at == now
        assert by_durable.liveness.operational_at == now - timedelta(seconds=60)
        assert by_durable.liveness.operational_age_seconds == 60.0
        assert by_durable.liveness.signal_sequence == 5
        assert by_durable.liveness.note == (
            "A heartbeat was recorded 0 s ago by Task Control's durable timestamp (a stage "
            "boundary); the runner's heartbeat #5 is older, 60 s ago."
        )
        assert by_durable.health.status == "HEALTHY"
        assert by_durable.health.heartbeat_age_seconds == 0.0
        by_signal = build(readout(older_signal))  # the durable timestamp is the start, 60 s old
        assert by_signal.liveness.last_heartbeat_source == "OPERATIONAL_SIGNAL"
        assert by_signal.liveness.status == "NOT_RECENT"
        assert by_signal.liveness.age_seconds == 60.0 == by_signal.liveness.operational_age_seconds
        assert by_signal.liveness.note.startswith(
            "No heartbeat for 60 s by the runner's heartbeat #5."
        )
        # Nothing above touched lifecycle authority or the permitted actions.
        for view in (durable_only, unbound, hidden, bound, unreadable, second_owner, by_durable):
            assert view.lifecycle == "RUNNING" and view.cancellation == "NOT_REQUESTED"
            actions = {a.action: a for a in view.actions}
            assert actions["CANCEL"].available is True and actions["RECOVER"].available is False
        missing = TaskHeartbeatReader(tmp_path / "never-written-checkpoints.sqlite")
        assert missing.read(execution.execution_id) == readout()
        assert not heartbeat_store_path(tmp_path / "never-written-checkpoints.sqlite").exists()
    finally:
        held.release.set()
        live.dispatcher.drain_for_tests()  # type: ignore[union-attr]
        live.stop()


def test_every_owner_that_offers_a_plan_is_its_stopped_tasks_re_plan(
    live: LocalPortfolioWebSession,
) -> None:
    """V188: each Task kind's re-PLAN is the one the owner that admits it declares
    beside its kind, collected from the owners the Host composes, never a
    hand-written map. A stopped training-input, research-strategy or Feature-build
    Task offers its owner's own PLAN preview, never `NONE`; each operation named is
    one the operation registry holds."""

    import json

    registry = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "src/alphalattice/interface/local_application/operations.json"
        ).read_text(encoding="utf-8")
    )
    registered = {op for ops in registry["commands"].values() for op in ops}
    operations = live.operations
    assert operations is not None
    replans = operations.replans()
    for owner in (
        operations.model_training_inputs,
        operations.research_strategies,
        operations.feature_builds,
    ):
        assert replans[owner.task_kind].preview is not None, owner.task_kind
    for kind, replan in replans.items():
        assert replan.task_kind == kind
        assert {replan.preview or replan.admitting, replan.admitting} <= registered, kind


def test_recovery_context_fields_follow_distinct_composed_preview_admission_pairs(
    live: LocalPortfolioWebSession,
) -> None:
    """The public request contract follows every composed two-press replan declaration."""
    operations = live.operations
    assert operations is not None
    pairs = {
        operation
        for replan in operations.replans().values()
        if replan.preview is not None and replan.preview != replan.admitting
        for operation in (replan.preview, replan.admitting)
    }
    assert pairs
    context_fields = {"recovery_task_id", "recovery_task_hash"}
    for operation in pairs:
        _required, allowed = PortfolioResearchOperationRequest.field_contract(cast(Any, operation))
        assert context_fields <= allowed, operation
    for operation in {"EXPERIMENT_VERIFY_ALL", "CRO_REVIEW"}:
        _required, allowed = PortfolioResearchOperationRequest.field_contract(cast(Any, operation))
        assert not context_fields <= allowed, operation


# Its last request raises an inner KeyError on purpose, to prove it is not read as an
# unknown Task (V449).
@pytest.mark.untyped_failure
def test_the_view_refuses_an_unknown_task_and_a_missing_query_typed(
    live: LocalPortfolioWebSession,
    monkeypatch,
    capsys,
) -> None:
    import json

    from alphalattice.interface.local_application.cli import main

    unknown = "00000000-0000-4000-8000-000000000000"
    for route in ("/api/status", "/api/tasks/recovery"):
        status, _, body = _request(live, route)
        assert status == 400 and b"handler_failed" not in body
        status, _, body = _request(live, route + "?task_id=not-a-uuid")
        assert status == 400
        assert json.loads(body)["failure_code"] == "local_web.task_id_invalid"
        status, _, body = _request(live, route + "?task_id=" + unknown)
        answer = json.loads(body)
        assert status == 404
        assert answer["failure_code"] == "task_control.task_not_found"
        assert answer["next_action"] == "DISCOVER_TASK_IN_CURRENT_WORKSPACE"
        assert "KeyError" not in body.decode()

    for noun in ("task", "recovery"):
        assert (
            main(
                ["--workspace", str(live.workspace), "--view", "full", noun, "show", unknown],
                serve=lambda _: 99,
            )
            == 2
        )
        answer = json.loads(capsys.readouterr().out)["data"]
        assert answer["failure_code"] == "task_control.task_not_found"
        assert answer["next_action"] == "DISCOVER_TASK_IN_CURRENT_WORKSPACE"

    def inner_failure(*args, **kwargs):
        raise KeyError("missing execution, not missing Task")

    monkeypatch.setattr(live.dispatcher, "status", inner_failure)
    status, _, body = _request(live, "/api/status?task_id=" + unknown)
    assert status == 400 and b"task_control.task_not_found" not in body


def test_a_cancelled_verification_sweep_offers_and_runs_its_replan(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CONTRACT: the cancelled sweep's saved recovery admits a new sweep, not a resume."""
    import json

    from alphalattice.interface.local_application.cli import main

    held = _Held(_Resolver(_resolved()))
    live = _service(tmp_path, held, "qa-sweep-replan")
    live.start()

    def cli(*arguments: str) -> tuple[int, dict[str, Any]]:
        code = main(
            ["--workspace", str(live.workspace), "--view", "full", *arguments], serve=lambda _: 99
        )
        captured = capsys.readouterr()
        assert not captured.err
        return code, json.loads(captured.out)

    try:
        _json(live, "/api/run", method="POST", payload={})
        assert held.entered.wait(30)
        task_id = _json(live, "/api/experiments/verify-all", method="POST", payload={})["task_id"]
        assert _view(live, task_id)["lifecycle"] == "QUEUED"
        _json(live, "/api/cancel", method="POST", payload={"task_id": task_id})
        stopped = tmp_path / "stopped.json"
        code, saved = cli("recovery", "show", task_id, "--output", str(stopped))
        assert code == 2 and saved["data"]["lifecycle"] == "CANCELLED", saved
        action = _actions(saved["data"])["REPLAN"]
        assert (action["operation"], action["available"], action["admits"]) == (
            "EXPERIMENT_VERIFY_ALL",
            True,
            True,
        ), action
        assert "new Task" in action["expected_effect"]
        assert saved["data"]["next_requests"]["replan"] == {"operation": "EXPERIMENT_VERIFY_ALL"}
        version = saved["data"]["task_record_hash"]
        code, admitted = cli("request", "--from", str(stopped), "--action", "replan")
        assert code == 3 and admitted["data"]["lifecycle"] == "QUEUED", admitted
        new_id = admitted["data"]["task_id"]
        assert new_id != task_id
        original = _view(live, task_id)
        assert original["lifecycle"] == "CANCELLED" and original["task_record_hash"] == version
        assert _view(live, new_id)["task_kind"] == original["task_kind"]
        held.release.set()
        live.dispatcher.drain_for_tests()
        assert _view(live, new_id)["lifecycle"] == "SUCCEEDED"
        assert _view(live, task_id)["lifecycle"] == "CANCELLED"
    finally:
        held.release.set()
        live.stop()


class _BlocksOnce:
    """A real typed Portfolio refusal once, then the same installed inputs resolve normally."""

    def __init__(self, inner: _Resolver) -> None:
        self.inner = inner
        self.block_next = True
        self.entered = threading.Event()
        self.release = threading.Event()

    @property
    def strategy_catalog_hash(self) -> str:
        return self.inner.strategy_catalog_hash

    def resolve_authorities(self, **kwargs: Any) -> Any:
        return self.inner.resolve_authorities(**kwargs)

    def installed_packages(self) -> Any:
        return self.inner.installed_packages()

    def resolve(self, **kwargs: Any) -> Any:
        if self.block_next:
            self.block_next = False
            raise EligiblePoolShort("portfolio_strategy_lab.eligible_pool_short:2024-01-03")
        self.entered.set()
        if not self.release.wait(timeout=30):
            raise TimeoutError("successor gate was not released by its test")
        return self.inner.resolve(**kwargs)


def test_a_replan_preview_and_admission_link_a_distinct_successor_across_restart(
    tmp_path: Path,
) -> None:
    """A blocked source stays pending through preview/admission until its child succeeds."""
    workspace_id = "qa-recovery-successor"
    resolver = _BlocksOnce(_Resolver(_resolved()))
    live = _service(tmp_path, resolver, workspace_id)
    live.start()

    def decision_for(task_id: str, session: LocalPortfolioWebSession) -> dict[str, Any] | None:
        return next(
            (
                decision
                for decision in _json(session, "/api/decisions")["decisions"]
                if decision.get("kind") == "STOPPED_TASK" and decision.get("task_id") == task_id
            ),
            None,
        )

    try:
        source_answer = _json(live, "/api/run", method="POST", payload={})
        source_id = source_answer["task_id"]
        live.dispatcher.drain_for_tests()  # type: ignore[union-attr]
        source_before = _view(live, source_id)
        assert source_before["lifecycle"] == "BLOCKED"
        source_hash = source_before["task_record_hash"]
        assert source_before["attention"]["unresolved"] is True
        assert decision_for(source_id, live) is not None

        recovery_request = source_before["next_requests"]["replan"]
        assert recovery_request["recovery_task_id"] == source_id
        assert recovery_request["recovery_task_hash"] == source_hash
        preview = live.operations.execute(  # type: ignore[union-attr]
            PortfolioResearchRequestDocument.model_validate(recovery_request).to_operation_request()
        )
        preview_request = preview["next_requests"]["run"]
        assert preview_request["recovery_task_id"] == source_id
        assert preview_request["recovery_task_hash"] == source_hash
        preview_links = live.session.task_control_registry.recovery_links(  # type: ignore[union-attr]
            UUID(source_id)
        )
        assert len(preview_links) == 1 and preview_links[0].successor_task_id is None
        assert "recovery_task_id" not in preview_links[0].admission_request
        assert "recovery_task_hash" not in preview_links[0].admission_request
        assert decision_for(source_id, live) is not None

        admitted = live.operations.execute(  # type: ignore[union-attr]
            PortfolioResearchRequestDocument.model_validate(preview_request).to_operation_request()
        )
        successor_id = admitted["task_id"]
        assert successor_id != source_id
        assert resolver.entered.wait(timeout=30)
        source_after_admission = _view(live, source_id)
        assert source_after_admission["task_record_hash"] == source_hash
        assert source_after_admission["lifecycle"] == "BLOCKED"
        assert source_after_admission["attention"]["unresolved"] is True
        assert decision_for(source_id, live) is not None

        resolver.release.set()
        live.dispatcher.drain_for_tests()  # type: ignore[union-attr]
        successor = _view(live, successor_id)
        assert successor["lifecycle"] == "SUCCEEDED"
        source_after_success = _view(live, source_id)
        assert source_after_success["task_record_hash"] == source_hash
        assert source_after_success["lifecycle"] == "BLOCKED"
        assert source_after_success["attention"]["unresolved"] is False
        assert source_after_success["attention"]["resolution"] == "SUCCESSOR_SUCCEEDED"
        assert source_after_success["attention"]["successor_task_id"] == successor_id
        assert decision_for(source_id, live) is None
        links = live.session.task_control_registry.recovery_links(  # type: ignore[union-attr]
            UUID(source_id)
        )
        successor_link = next(link for link in links if link.successor_task_id is not None)
        assert str(successor_link.successor_task_id) == successor_id

        live.stop()
        restarted = _service(tmp_path, _Resolver(_resolved()), workspace_id)
        restarted.start()
        try:
            refetched = _view(restarted, source_id)
            assert refetched["task_record_hash"] == source_hash
            assert refetched["attention"]["resolution"] == "SUCCESSOR_SUCCEEDED"
            assert refetched["attention"]["successor_task_id"] == successor_id
            assert decision_for(source_id, restarted) is None
        finally:
            restarted.stop()
    finally:
        resolver.release.set()
        live.stop()
