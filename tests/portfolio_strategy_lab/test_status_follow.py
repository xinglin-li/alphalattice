"""A follow learns when its Task moves on (binding plan N8): STATUS may wait on the Host for the
Task's next stage or lifecycle, and the CLI's `--wait` asks it to instead of polling.

For the UI line's suite: tests/portfolio_strategy_lab/test_status_follow.py.
"""

from __future__ import annotations

import itertools
import json
import shlex
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar
from uuid import uuid4

import pytest

from alphalattice.control.product_host.composition.portfolio_research_operations import (
    PortfolioResearchOperations,
)
from alphalattice.control.task_control.contracts import TaskLifecycle
from alphalattice.interface.local_application import client as client_module
from alphalattice.interface.local_application.cli import main
from alphalattice.interface.local_application.cli_contract import AGENT_HOST_WAITS, outcome_of
from alphalattice.interface.local_application.portfolio_research import (
    LocalApplicationError,
    PortfolioResearchOperationRequest,
)


@pytest.mark.parametrize("wait", [0, -1, 20.5, True, "5"])
def test_a_status_wait_outside_zero_to_twenty_seconds_is_refused(wait: Any) -> None:
    with pytest.raises(LocalApplicationError, match="wait_seconds_invalid"):
        PortfolioResearchOperationRequest(operation="STATUS", task_id=uuid4(), wait_seconds=wait)


def _projection(lifecycle: TaskLifecycle, verified: int) -> SimpleNamespace:
    return SimpleNamespace(lifecycle=lifecycle, verified_stage_count=verified)


def _host(sequence: list[SimpleNamespace]) -> SimpleNamespace:
    reads = iter(sequence)
    last = {"value": sequence[0]}

    def status(_task_id: object) -> SimpleNamespace:
        last["value"] = next(reads, last["value"])
        return last["value"]

    return SimpleNamespace(dispatcher=SimpleNamespace(status=status))


def test_a_waiting_status_answers_when_the_task_moves_on_or_its_wait_ends() -> None:
    running = _projection(TaskLifecycle.RUNNING, 0)
    host = _host([running, running, _projection(TaskLifecycle.RUNNING, 1)])
    begun = time.monotonic()
    moved = PortfolioResearchOperations._moved_on(host, uuid4(), running, 5.0)  # type: ignore[arg-type]
    assert moved.verified_stage_count == 1 and time.monotonic() - begun < 1.0

    begun = time.monotonic()
    still = PortfolioResearchOperations._moved_on(_host([running]), uuid4(), running, 0.3)  # type: ignore[arg-type]
    assert still is running and 0.3 <= time.monotonic() - begun < 1.0

    done = _projection(TaskLifecycle.SUCCEEDED, 2)
    begun = time.monotonic()
    assert PortfolioResearchOperations._moved_on(_host([done]), uuid4(), done, 5.0) is done  # type: ignore[arg-type]
    assert time.monotonic() - begun < 0.05


def test_the_follow_asks_the_host_to_wait_instead_of_polling(capsys, monkeypatch) -> None:
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "session-1")
    monkeypatch.delenv("CODEX_THREAD_ID", raising=False)
    task_id = str(uuid4())
    sent: list[dict[str, Any]] = []
    answers = iter(
        [
            {"status": "RUNNING", "lifecycle": "RUNNING", "task_id": task_id},
            {"status": "SUCCEEDED", "lifecycle": "SUCCEEDED", "task_id": task_id},
        ]
    )

    class _Client:
        workspace = Path("workspace")  # every client names one (V428)

        def request(self, document: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
            sent.append(document)
            return next(answers)

    admitted = {"status": "TASK_ADMITTED", "task_id": task_id, "lifecycle": "QUEUED"}
    begun = time.monotonic()
    final = client_module._follow(_Client(), admitted, 60)  # type: ignore[arg-type]
    assert final["lifecycle"] == "SUCCEEDED" and final["admission"] == admitted
    assert all(d["operation"] == "STATUS" and 0 < d["wait_seconds"] <= 20 for d in sent)
    assert len(sent) == 2 and time.monotonic() - begun < 0.2
    # Its first line tells the agent this call is the wait, and its host's way to wait for it.
    first = json.loads(capsys.readouterr().err.splitlines()[0])
    assert first["follow"]["operation"] == "STATUS" and "do not poll" in first["returns"]
    assert "run_in_background" in first["returns"]
    # On Codex it names both ways: the queued wake when idle, this one call under a goal.
    codex = AGENT_HOST_WAITS["codex"]
    assert "--notify codex-queue" in codex and "get_goal" in codex


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ({"status": "QUEUED", "task_id": "t"}, "PENDING"),
        ({"status": "RUNNING", "task_id": "t"}, "PENDING"),
        ({"lifecycle": "DEFERRED"}, "PENDING"),
        ({"lifecycle": "REVIEW_PENDING"}, "PENDING"),
        ({"status": "CANCEL_REQUESTED"}, "PENDING"),
        ({"lifecycle": "RECOVERY_REQUIRED"}, "PENDING"),
        ({"status": "BLOCKED"}, "REFUSED"),
        ({"lifecycle": "CANCELLED"}, "REFUSED"),
        ({"status": "FEATURE_TRIAL", "state": "RUNNING"}, "PENDING"),
        (
            {"status": "FEATURE_TRIAL", "state": "STOPPED", "stopped": {"failure_code": "x"}},
            "REFUSED",
        ),
        ({"status": "FEATURE_TRIAL", "state": "COMPLETED"}, "OK"),
        ({"status": "ADMITTED", "task_id": "t"}, "PENDING"),
        ({"status": "EXPERIMENT_PUBLISHED", "lifecycle": "SUCCEEDED"}, "OK"),
        ({"status": "SUCCEEDED", "lifecycle": "SUCCEEDED"}, "OK"),
    ],
)
def test_an_outcome_reads_the_task_state_wherever_the_answer_writes_it(
    body: dict[str, Any], expected: str
) -> None:
    """requirement (V144, V138): work that waits, runs or awaits a decision is pending, in
    ``lifecycle`` or in ``status`` as readbacks write it; a blocked or cancelled Task and a
    stopped trial are refused; nothing unfinished answers OK."""

    assert outcome_of(body) == expected


def test_a_wait_follows_the_task_the_answer_started_and_stops_for_a_decision() -> None:
    """requirement (V137): a Portfolio promotion names the new Alpha Task it started; the
    follow reads that Task, not the finished one the request was about, keeps the admission's
    next requests when the last read offers none, and stops at a review a wait cannot pass."""

    old, new = str(uuid4()), str(uuid4())
    sent: list[dict[str, Any]] = []
    answers = iter(
        [
            {"lifecycle": "RUNNING", "task_id": new},
            {"lifecycle": "REVIEW_PENDING", "task_id": new},
        ]
    )

    class _Client:
        workspace = Path("workspace")  # every client names one (V428)

        def request(self, document: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
            sent.append(document)
            return next(answers)

    promote = {"operation": "EXPERIMENT_PROMOTE", "task_id": old}
    requests = {
        "task": {"operation": "STATUS", "task_id": new},
        "alpha": {"operation": "EXPERIMENT_READBACK", "task_id": new},
        "promote": promote,
    }
    admitted = {
        "status": "UPSTREAM_PROMOTION_ADMITTED",
        "task_id": new,
        "follow_task_id": new,
        "promoted_from_task_id": old,
        "next_requests": requests,
    }
    final = client_module._follow(_Client(), admitted, 60)  # type: ignore[arg-type]
    assert [d["task_id"] for d in sent] == [new, new]
    assert final["lifecycle"] == "REVIEW_PENDING" and final["admission"] == admitted
    assert final["next_requests"] == requests
    assert outcome_of(final) == "PENDING"


class _Stub:
    """A client whose Host may be gone for a moment, as a restart leaves it."""

    workspace = Path("workspace")
    timeout = goal = None

    def __init__(self, answers: list[Any]) -> None:
        self.answers, self.sent, self.published = iter(answers), [], []

    def request(self, document: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        self.sent.append(document)
        answer = next(self.answers)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    def publish_event(self, document: dict[str, Any]) -> dict[str, Any]:
        self.published.append(document)
        return {"status": "RECORDED"}


def test_the_waiter_never_ends_on_a_timer_and_waits_out_a_host_restart(monkeypatch) -> None:
    """requirement (WK, V280): without --max-wait the follow waits until the Task ends, a
    Host that stopped or restarted included, and ends with one line: the event and its read."""

    task_id = str(uuid4())
    stub = _Stub(
        [
            {"status": "RUNNING", "lifecycle": "RUNNING", "task_id": task_id},
            client_module.LocalResearchClientError("local_client.service_not_running"),
            {"status": "REFUSED", "refused": "local_web.external_token_invalid"},
            {"status": "SUCCEEDED", "lifecycle": "SUCCEEDED", "task_id": task_id},
        ]
    )
    monkeypatch.setattr(client_module.time, "sleep", lambda _s: None)
    monkeypatch.setattr(client_module, "LocalResearchClient", lambda *a, **k: stub)
    admitted = {"status": "TASK_ADMITTED", "task_id": task_id, "lifecycle": "QUEUED"}
    final = client_module._follow(stub, admitted, None)  # type: ignore[arg-type]
    assert final["lifecycle"] == "SUCCEEDED" and outcome_of(final) == "OK"
    event = final["wait_event"]
    assert event["event"] == "ENDED" and event["task_id"] == task_id
    assert "task show" in event["read"] and task_id in event["read"]
    # The installed command and the workspace, as every printed command starts (V428, V429).
    assert event["read"].startswith("alphalattice ") and "--workspace" in event["read"]
    assert len(stub.sent) == 4


def test_a_follow_wakes_on_a_new_incident_and_not_on_one_it_began_with(monkeypatch) -> None:
    """requirement (GY2, WK): an incident the Supervisor opens while a Task runs wakes the agent
    following it; one open when the follow began is not news."""

    task_id = str(uuid4())
    old = {"key": "a" * 64, "code": "task_runtime.work_stalled"}
    new = {"key": "b" * 64, "code": "task_runtime.liveness_stale"}
    running = {"status": "RUNNING", "lifecycle": "RUNNING", "task_id": task_id}
    stub = _Stub([{**running, "incident": old}, {**running, "incident": new}])
    monkeypatch.setattr(client_module.time, "sleep", lambda _s: None)
    admitted = {
        "status": "TASK_ADMITTED",
        "task_id": task_id,
        "lifecycle": "QUEUED",
        "incident": old,
    }
    final = client_module._follow(stub, admitted, None)  # type: ignore[arg-type]
    assert final["wait_event"]["event"] == "INCIDENT" and final["incident"] == new
    assert len(stub.sent) == 2


def test_a_wait_on_each_stage_returns_as_a_stage_is_verified(monkeypatch) -> None:
    """A wait on each stage returns as a stage is verified."""

    task_id = str(uuid4())
    running = {"status": "RUNNING", "lifecycle": "RUNNING", "task_id": task_id}
    stub = _Stub(
        [
            {**running, "verified_stage_count": 2, "total_stage_count": 5},
            {**running, "verified_stage_count": 3, "total_stage_count": 5},
        ]
    )
    monkeypatch.setattr(client_module.time, "sleep", lambda _s: None)
    held = {**running, "verified_stage_count": 2, "total_stage_count": 5}
    final = client_module._follow(stub, held, None, each_stage=True)  # type: ignore[arg-type]
    event = final["wait_event"]
    assert (event["event"], event["verified_stage_count"]) == ("STAGE_VERIFIED", 3), event
    assert outcome_of(final) == "PENDING" and len(stub.sent) == 2
    ended = {"status": "SUCCEEDED", "lifecycle": "SUCCEEDED", "task_id": task_id}
    stub = _Stub([{**ended, "verified_stage_count": 5, "total_stage_count": 5}])
    final = client_module._follow(stub, held, None, each_stage=True)  # type: ignore[arg-type]
    assert final["wait_event"]["event"] == "ENDED"
    goal = SimpleNamespace(task_id=None, goal_id=str(uuid4()), max_wait=None, notify=None)
    with pytest.raises(client_module.LocalResearchClientError, match="each_stage_needs_task"):
        client_module._wait(stub, SimpleNamespace(**vars(goal), each_stage=True))  # type: ignore[arg-type]


def test_max_wait_is_the_waiters_only_timer(monkeypatch) -> None:
    """requirement (WK): --max-wait, set just under a command cap, ends the wait as pending;
    it is refused when not above zero."""

    task_id = str(uuid4())
    running = {"status": "RUNNING", "lifecycle": "RUNNING", "task_id": task_id}
    stub = _Stub(itertools.repeat(running))  # type: ignore[arg-type]
    admitted = {"status": "TASK_ADMITTED", "task_id": task_id, "lifecycle": "QUEUED"}
    final = client_module._follow(stub, admitted, 0.05)  # type: ignore[arg-type]
    assert final["wait_event"]["event"] == "MAX_WAIT_REACHED"
    assert final["wait_status"] == "MAX_WAIT_TASK_CONTINUES" and outcome_of(final) == "PENDING"
    with pytest.raises(client_module.LocalResearchClientError, match="max_wait_invalid"):
        client_module._follow(stub, admitted, 0)  # type: ignore[arg-type]


def test_a_goal_wait_ends_on_the_next_task_end_it_did_not_see_begin(monkeypatch) -> None:
    """requirement (WK, OP13): `activity wait --goal-id` ends on the first of the goal's Tasks
    to end or need a decision after it began, not on one that had ended before."""

    goal, old, new = str(uuid4()), str(uuid4()), str(uuid4())

    def narrative(*tasks: tuple[str, str]) -> dict[str, Any]:
        return {
            "status": "GOAL_NARRATIVE",
            "state": "OPEN",
            "record": {
                "tasks": [{"task_id": t, "state": s} for t, s in tasks],
                "conversation": [],
            },
        }

    stub = _Stub(
        [
            narrative((old, "SUCCEEDED")),
            narrative((old, "SUCCEEDED"), (new, "RUNNING")),
            narrative((old, "SUCCEEDED"), (new, "RECOVERY_REQUIRED")),
        ]
    )
    monkeypatch.setattr(client_module.time, "sleep", lambda _s: None)
    final = client_module._wait_for_goal(stub, goal, None)  # type: ignore[arg-type]
    event = final["wait_event"]
    assert (event["event"], event["task_id"], event["goal_id"]) == ("NEEDS_DECISION", new, goal)
    assert all(d == {"operation": "GOAL_NARRATIVE", "goal_id": goal} for d in stub.sent)


def test_a_goal_wait_ends_on_the_next_message_under_the_goal(monkeypatch) -> None:
    """A goal wait ends on the next message under the goal."""

    monkeypatch.delenv("CLAUDE_CODE_SESSION_ID", raising=False)
    monkeypatch.delenv("CODEX_THREAD_ID", raising=False)
    goal = str(uuid4())
    heard = {"observation_id": "o-1", "message_kind": "decision", "agent_id": "lead"}
    assigned = {
        "observation_id": "o-2",
        "message_kind": "assignment",
        "message_id": "assign-1",
        "agent_id": "lead",
        "recipient_id": "child",
    }

    def narrative(*messages: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": "GOAL_NARRATIVE",
            "state": "OPEN",
            "record": {"tasks": [], "conversation": list(messages)},
        }

    request = {
        "message_kind": "AGENT_BUNDLE_PREPARE",
        "message_id": "request-0",
        "input_channel": "PRODUCT_OPERATION",
        "agent_id": "lead",
    }
    stub = _Stub([narrative(heard), narrative(heard, request), narrative(heard, request, assigned)])
    monkeypatch.setattr(client_module.time, "sleep", lambda _s: None)
    event = client_module._wait_for_goal(stub, goal, None)["wait_event"]  # type: ignore[arg-type]
    assert len(stub.sent) == 3, "a product request's row ended the wait"
    assert (event["event"], event["message_kind"], event["message_id"]) == (
        "MESSAGE",
        "assignment",
        "assign-1",
    )
    assert (event["sender"], event["recipient"], event["goal_id"]) == ("lead", "child", goal)
    assert "goal show" in event["read"] and goal in event["read"]


def test_a_goal_wait_wakes_only_for_its_own_messages(monkeypatch) -> None:
    """A goal wait wakes only for its own messages."""

    goal = str(uuid4())
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "00000000-0000-4000-8000-0000000000aa")
    monkeypatch.delenv("CODEX_THREAD_ID", raising=False)
    lead = "00000000-0000-4000-8000-0000000000aa"
    own = {"observation_id": "o-1", "message_id": "m-1", "agent_id": lead, "recipient_id": "a"}
    between = {
        "observation_id": "o-2",
        "message_kind": "question",
        "message_id": "m-2",
        "agent_id": "a",
        "recipient_id": "b",
    }
    answer = {
        "observation_id": "o-3",
        "message_kind": "answer",
        "message_id": "m-3",
        "agent_id": "a",
        "reply_to": "m-1",
    }

    def narrative(*messages: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": "GOAL_NARRATIVE",
            "state": "OPEN",
            "record": {"tasks": [], "conversation": list(messages)},
        }

    stub = _Stub([narrative(own), narrative(own, between), narrative(own, between, answer)])
    monkeypatch.setattr(client_module.time, "sleep", lambda _s: None)
    event = client_module._wait_for_goal(stub, goal, None)["wait_event"]  # type: ignore[arg-type]
    assert (event["event"], event["message_id"]) == ("MESSAGE", "m-3"), event
    assert len(stub.sent) == 3, "the question between two subagents did not end the wait"


def test_every_waiter_ends_on_a_deferral_with_its_retry_time_and_resume(monkeypatch) -> None:
    """Every waiter ends on a deferral with its retry time and resume."""

    task_id = str(uuid4())
    resume = {"operation": "WORKSPACE_PREPARE_CONFIRM", "preparation_plan_hash": "a" * 64}
    read = {"operation": "WORKSPACE_PREPARE_READBACK", "task_id": task_id}
    deferred = {
        "lifecycle": "DEFERRED",
        "task_id": task_id,
        "retry_after_at": "2026-10-02T17:00:00+00:00",
        "next_requests": {"resume": resume, "read": read},
    }
    monkeypatch.setattr(client_module.time, "sleep", lambda _s: None)
    stub = _Stub([{"lifecycle": "RUNNING", "task_id": task_id}, deferred])
    admitted = {"status": "ADMITTED", "task_id": task_id, "lifecycle": "QUEUED"}
    final = client_module._follow(stub, admitted, None)  # type: ignore[arg-type]
    assert final["wait_event"]["event"] == "DEFERRED" and len(stub.sent) == 2
    assert final["retry_after_at"] == deferred["retry_after_at"]
    assert final["next_requests"] == {"resume": resume, "read": read}
    assert outcome_of(final) == "PENDING"
    # `activity wait --task` on a Task already deferred names the deferral, not an end.
    args = SimpleNamespace(
        task_id=task_id, goal_id=None, max_wait=None, notify=None, each_stage=False
    )
    waited = client_module._wait(_Stub([deferred]), args)  # type: ignore[arg-type]
    assert waited["wait_event"]["event"] == "DEFERRED"
    goal = str(uuid4())

    def narrative(state: str) -> dict[str, Any]:
        return {
            "status": "GOAL_NARRATIVE",
            "state": "OPEN",
            "record": {"tasks": [{"task_id": task_id, "state": state}], "conversation": []},
        }

    stub = _Stub([narrative("RUNNING"), narrative("DEFERRED")])
    woke = client_module._wait_for_goal(stub, goal, None)  # type: ignore[arg-type]
    assert (woke["wait_event"]["event"], woke["wait_event"]["task_id"]) == ("DEFERRED", task_id)


def test_a_deferred_tasks_status_names_its_retry_time_and_resume() -> None:
    """requirement (V507): the read a waiter ends on says when a deferred Task's plan may be
    sent again and by which request, as its owner's readback words them (V375), and names that
    readback; a kind with no deferral readback goes on by its recovery view."""

    task_id = uuid4()
    projection = SimpleNamespace(
        task_id=task_id,
        task_kind="workspace_preparation",
        lifecycle=TaskLifecycle.DEFERRED,
        goal_summary=None,
        current_stage="prepare_data",
        verified_stage_count=1,
        total_stage_count=3,
        running_since=None,
        last_activity_at=datetime(2026, 10, 2, 16, tzinfo=UTC),
        cancel_available=True,
        cancel_pending=False,
        queued_next_task_id=None,
        latest_failure_code="data.rate_limited",
        task_record_hash="h" * 64,
    )

    class Registry:
        """A registry whose work items this read does not need."""

        def task_with_work_items(self, task: object) -> None:
            raise KeyError(task)

    class Preparation:
        """The preparation owner's readback of its deferred Task."""

        def readback(self, _task_id: object) -> dict[str, Any]:
            return {
                "status": "DEFERRED",
                "failure_code": "data.rate_limited",
                "progress": {"retry_after_at": "2026-10-02T17:00:00+00:00"},
                "plan_hash": "a" * 64,
                "inputs": None,
            }

    host = PortfolioResearchOperations.__new__(PortfolioResearchOperations)
    host.dispatcher = SimpleNamespace(status=lambda _t: projection, failure=lambda _t: None)  # type: ignore[assignment]
    host.resume_refusal = None
    host.review = None
    host.workspace_session = SimpleNamespace(task_control_registry=Registry())  # type: ignore[assignment]
    host.preparation = Preparation()  # type: ignore[assignment]
    host.review = None  # no Evidence owner: a preparation Task's status reads none (EVRB)
    body = host.status(task_id)
    assert body["lifecycle"] == "DEFERRED"
    assert body["retry_after_at"] == "2026-10-02T17:00:00+00:00"
    assert body["next_requests"] == {
        "resume": {"operation": "WORKSPACE_PREPARE_CONFIRM", "preparation_plan_hash": "a" * 64},
        "read": {"operation": "WORKSPACE_PREPARE_READBACK", "task_id": str(task_id)},
    }
    assert "provider" in str(body["detail"])
    recovery = {"operation": "TASK_RECOVERY", "task_id": str(task_id)}
    assert host._deferred_way("research_experiment", task_id) == {
        "next_requests": {"recovery": recovery}
    }


def test_a_deferred_tasks_clock_stops_while_it_waits() -> None:
    """A deferred Task's recorded clock stops while it waits, including its listed row."""

    from alphalattice.control.task_control.timing import task_timing

    started = datetime(2026, 10, 2, 9, tzinfo=UTC)
    deferred = SimpleNamespace(
        lifecycle=TaskLifecycle.DEFERRED,
        admitted_at=started - timedelta(seconds=5),
        started_at=started,
        updated_at=started + timedelta(minutes=3),
    )
    later = started + timedelta(hours=4)
    timing = task_timing(deferred, (), now=later)  # type: ignore[arg-type]
    assert timing["running_seconds"] == 180.0, timing
    running = SimpleNamespace(**{**vars(deferred), "lifecycle": TaskLifecycle.RUNNING})
    assert task_timing(running, (), now=later)["running_seconds"] == 4 * 3600.0  # type: ignore[arg-type]


def test_a_remedy_that_resumed_its_task_is_followed_to_the_tasks_end(monkeypatch) -> None:
    """regression (V508, the user's review at 244900d8): `incident remediate --remedy RECOVER
    --wait` answered OK at once, the resumed Task's RUNNING nested in `answer`. The receipt
    names the Task and its lifecycle, so the wait follows the Task to its end."""

    task_id = str(uuid4())
    receipt = {
        "status": "REMEDY_ATTEMPTED",
        "task_id": task_id,
        "lifecycle": "RUNNING",
        "answer": {"task_id": task_id, "lifecycle": "RUNNING"},
    }
    assert outcome_of(receipt) == "PENDING"
    monkeypatch.setattr(client_module.time, "sleep", lambda _s: None)
    stub = _Stub([{"lifecycle": "SUCCEEDED", "task_id": task_id}])
    final = client_module._follow(stub, receipt, None)  # type: ignore[arg-type]
    assert final["wait_event"]["event"] == "ENDED" and stub.sent[0]["task_id"] == task_id


def test_the_compact_view_leaves_out_the_timing_no_decision_reads() -> None:
    """requirement (V280): half of a `task show` answer was its stage timings; the compact
    view leaves them out and says so, the full view and --output keep them."""

    stage = {"stage_id": "prepare_data", "seconds": 125.0}
    shown = client_module.compact_display(
        {
            "status": "RUNNING",
            "lifecycle": "RUNNING",
            "timing": {"stages": [stage] * 50},
            "stage_timing": stage,
        }
    )
    assert "timing" not in shown["data"] and "timing" in shown["omitted_sections"]
    assert shown["data"]["stage_timing"] == stage


@pytest.mark.parametrize(
    "overdue, language, timer_failure, operation",
    [
        (False, "en", None, "PLAN"),
        (True, "en", None, "PLAN"),
        (True, "zh", None, "PLAN"),
        (False, "en", "create", "PLAN"),
        (False, "en", "start", "PLAN"),
        (True, "en", None, "RUN"),
    ],
)
def test_a_pending_plan_explains_validation_once_without_changing_json_stdout(
    live, tmp_path, monkeypatch, capsys, overdue, language, timer_failure, operation
):
    """A scheduled validation message leaves the CLI answer intact and closes with the request."""
    from alphalattice.interface.local_application.cli import main

    scheduled, finished = [], []

    def timer(seconds, callback):
        scheduled.append(seconds)
        if timer_failure == "create":
            raise RuntimeError("timer unavailable")

        def start():
            if timer_failure == "start":
                raise RuntimeError("thread unavailable")
            if overdue:
                callback()

        return SimpleNamespace(
            start=start,
            cancel=lambda: finished.append("cancel"),
            join=lambda: finished.append("join"),
        )

    monkeypatch.setattr(client_module, "Timer", timer)
    request = tmp_path / "plan-request.json"
    request.write_text(json.dumps({"operation": operation, "spec": {}}), encoding="utf-8")
    main(
        [
            "--workspace",
            str(live.workspace),
            "--lang",
            language,
            "request",
            "--file",
            str(request),
        ],
        serve=lambda _: 99,
    )
    captured = capsys.readouterr()
    assert json.loads(captured.out)["operation"] == operation
    assert scheduled == [5.0] and finished == ([] if timer_failure else ["cancel", "join"])
    assert bool(captured.err) is overdue
    if overdue:
        message = json.loads(captured.err)
        assert (message["operation"], message["status"]) == (operation, "VALIDATING")
        assert message["elapsed_seconds"] >= 0 and message["detail"]
        assert any("\u4e00" <= letter <= "\u9fff" for letter in message["detail"]) is (
            language == "zh"
        )


class _Admitting(_Stub):
    """A client that admits one Task, then answers its STATUS reads."""

    def __init__(self, admitted: dict[str, Any], answers: list[Any], workspace: Any) -> None:
        super().__init__(answers)
        self.admitted, self.workspace = admitted, workspace

    def exchange(self, document: dict[str, Any]) -> tuple[dict[str, Any], bytes]:
        self.sent.append(document)
        return dict(self.admitted), json.dumps(self.admitted).encode("utf-8")

    def selected_url(self, document: dict[str, Any], body: dict[str, Any]) -> str:
        return ""

    def navigation(self, document: dict[str, Any], body: dict[str, Any]) -> dict[str, str]:
        return {}


def _wait_with_output(tmp_path: Any, monkeypatch: Any, answers: list[Any]) -> tuple[Any, str]:
    from alphalattice.interface.local_application.cli import main as client_main

    task_id = str(uuid4())
    admitted = {"status": "TASK_ADMITTED", "task_id": task_id, "lifecycle": "QUEUED"}
    stub = _Admitting(admitted, answers, tmp_path)
    monkeypatch.setattr(client_module, "LocalResearchClient", lambda *a, **k: stub)
    monkeypatch.setattr(client_module.time, "sleep", lambda _s: None)
    document = tmp_path / "request.json"
    document.write_text(json.dumps({"operation": "EXPERIMENT_RUN"}), encoding="utf-8")
    output = tmp_path / "answer.json"
    argv = ["--workspace", str(tmp_path), "request", "--file", str(document), "--wait"]
    client_main([*argv, "--output", str(output)], serve=lambda _: pytest.fail("no serve"))
    return output, task_id


def test_an_interrupted_wait_keeps_the_admitted_task_in_its_output(tmp_path, monkeypatch) -> None:
    """regression (V273, OP3, OP12): `--wait --output` wrote the answer only after the wait, so
    a Ctrl-C while waiting left no file and named the admitted Task nowhere. The admission is
    written first."""

    output = tmp_path / "answer.json"
    with pytest.raises(KeyboardInterrupt):
        _wait_with_output(tmp_path, monkeypatch, [KeyboardInterrupt()])
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["status"] == "TASK_ADMITTED" and saved["task_id"]


def test_a_finished_wait_replaces_the_admission_with_its_final_answer(
    tmp_path, monkeypatch, capsys
) -> None:
    """requirement (V273): the final answer takes the admission's place in `--output`, and the
    envelope names the file."""

    answers = [{"status": "SUCCEEDED", "lifecycle": "SUCCEEDED"}]
    output, task_id = _wait_with_output(tmp_path, monkeypatch, answers)
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["lifecycle"] == "SUCCEEDED" and saved["admission"]["task_id"] == task_id
    assert not output.with_name(output.name + ".tmp").exists()
    assert json.loads(capsys.readouterr().out)["output_file"] == str(output.resolve())


def test_a_goal_wait_ends_on_a_task_that_turned_blocked(monkeypatch) -> None:
    """A goal wait ends on a task that turned blocked."""

    goal, task = str(uuid4()), str(uuid4())

    def narrative(state: str) -> dict[str, Any]:
        return {
            "status": "GOAL_NARRATIVE",
            "state": "OPEN",
            "record": {"tasks": [{"task_id": task, "state": state}], "conversation": []},
        }

    stub = _Stub([narrative("RUNNING"), narrative("BLOCKED")])
    monkeypatch.setattr(client_module.time, "sleep", lambda _s: None)
    final = client_module._wait_for_goal(stub, goal, None)  # type: ignore[arg-type]
    event = final["wait_event"]
    assert (event["event"], event["task_id"], event["lifecycle"]) == ("STOPPED", task, "BLOCKED")
    assert "--workspace" in event["read"] and task in event["read"], event


def test_a_goal_wait_wakes_on_a_task_stopped_again_after_its_recovery(monkeypatch) -> None:
    """A goal wait wakes on a task stopped again after its recovery."""

    goal, task = str(uuid4()), str(uuid4())

    def narrative(state: str) -> dict[str, Any]:
        return {
            "status": "GOAL_NARRATIVE",
            "state": "OPEN",
            "record": {"tasks": [{"task_id": task, "state": state}], "conversation": []},
        }

    stub = _Stub(
        [
            narrative("RECOVERY_REQUIRED"),
            narrative("RECOVERY_REQUIRED"),
            narrative("RUNNING"),
            narrative("RECOVERY_REQUIRED"),
        ]
    )
    monkeypatch.setattr(client_module.time, "sleep", lambda _s: None)
    final = client_module._wait_for_goal(stub, goal, None)  # type: ignore[arg-type]
    event = final["wait_event"]
    assert (event["event"], event["task_id"], event["lifecycle"]) == (
        "NEEDS_DECISION",
        task,
        "RECOVERY_REQUIRED",
    )
    assert len(stub.sent) == 4


def test_a_bundles_submit_command_is_quoted_for_the_shell_in_use(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """A bundle's submit command is quoted for the shell in use."""

    from alphalattice.interface.local_application.cli_contract import entry

    body = {"status": "AGENT_BUNDLE_READY", "files": [{"name": "README.md", "text": "Read.\n"}]}
    monkeypatch.setenv("ALPHALATTICE_SHELL", "powershell")
    folder = tmp_path / "it's a bundle"
    written = client_module._write_bundle(folder, body, prefix=entry(tmp_path))["submit_command"]
    assert written.startswith("alphalattice "), written
    assert "'" + str(folder).replace("'", "''") + "'" in written, written
    monkeypatch.setenv("ALPHALATTICE_SHELL", "posix")
    again_folder = tmp_path / "it's again"
    again = client_module._write_bundle(again_folder, body, prefix=entry(tmp_path))
    assert again["submit_command"].startswith("alphalattice "), again["submit_command"]
    assert shlex.quote(str(again_folder)) in again["submit_command"], again["submit_command"]


def test_a_wait_on_a_trial_ends_when_a_step_needs_its_recovery(monkeypatch) -> None:
    """A wait on a trial ends when a step needs its recovery."""

    from alphalattice.control.product_host.composition.feature_trials import waiting_step

    task_id = str(uuid4())
    steps = [
        {"step": "FEATURE_BUILD", "task_id": str(uuid4()), "state": "SUCCEEDED"},
        {"step": "ALPHA_STUDY", "task_id": task_id, "state": "RECOVERY_REQUIRED"},
        {"step": "PORTFOLIO_STUDY", "state": "WAITING"},
    ]
    assert waiting_step(steps) == steps[1] and waiting_step(steps[:1]) is None
    trial = {"status": "FEATURE_TRIAL", "feature_trial_id": "trial-1", "state": "RUNNING"}
    recovery = {"operation": "TASK_RECOVERY", "task_id": task_id}
    waiting = {
        **trial,
        "steps": steps,
        "lifecycle": "RECOVERY_REQUIRED",
        "task_id": task_id,
        "next_requests": {"recovery": recovery},
    }
    stub = _Stub([{**trial, "steps": steps[:1]}, waiting])
    monkeypatch.setattr(client_module.time, "sleep", lambda _s: None)
    final = client_module._follow(stub, trial, None)  # type: ignore[arg-type]
    assert final["wait_event"]["event"] == "NEEDS_DECISION"
    assert final["next_requests"]["recovery"] == recovery
    assert len(stub.sent) == 2


def test_a_request_reads_a_completed_trial_once() -> None:
    """requirement (V461): a goal resolves the trial its reference names and the review packet
    another names, which reads the same trial again; inside one request a completed trial is
    read once, a running one each time, and outside a request every read is new."""

    from alphalattice.control.product_host.composition import feature_trials

    reads: list[str] = []
    states = {"done": "COMPLETED", "running": "RUNNING"}

    class Trials:
        def _readback(self, trial_id: str) -> dict[str, Any]:
            reads.append(trial_id)
            return {"state": states[trial_id], "comparison": {"read": len(reads)}}

    trials = Trials()
    readback = feature_trials.FeatureTrials.readback
    with feature_trials.trial_reads_once():
        first = readback(trials, "done")  # type: ignore[arg-type]
        first["comparison"]["read"] = 99  # a caller's change stays its own
        again = readback(trials, "done")  # type: ignore[arg-type]
        with feature_trials.trial_reads_once():  # an operation inside another joins its scope
            nested = readback(trials, "done")  # type: ignore[arg-type]
        readback(trials, "running")  # type: ignore[arg-type]
        readback(trials, "running")  # type: ignore[arg-type]
    assert reads == ["done", "running", "running"]
    assert again == nested == {"state": "COMPLETED", "comparison": {"read": 1}}
    readback(trials, "done")  # type: ignore[arg-type]
    assert reads == ["done", "running", "running", "done"]


def test_a_goal_wait_capped_before_its_first_read_is_pending(monkeypatch) -> None:
    """regression (V563, the user's review at de555b07): `activity wait --goal G --max-wait 1`
    whose first goal read met a stopped Host answered `MAX_WAIT_REACHED` as OK, exit 0, where
    the Task waiter answers PENDING, exit 3. Both are pending."""

    down = client_module.LocalResearchClientError("local_client.service_not_running")
    clock = {"now": 0.0}
    monkeypatch.setattr(client_module.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(client_module.time, "sleep", lambda s: clock.update(now=clock["now"] + s))
    goal_stub = _Stub(itertools.repeat(down))  # type: ignore[arg-type]
    monkeypatch.setattr(client_module, "LocalResearchClient", lambda *a, **k: goal_stub)
    capped = client_module._wait_for_goal(goal_stub, str(uuid4()), 0.05)  # type: ignore[arg-type]
    assert capped["wait_event"]["event"] == "MAX_WAIT_REACHED"
    assert outcome_of(capped) == "PENDING"
    task_stub = _Stub(itertools.repeat(down))  # type: ignore[arg-type]
    monkeypatch.setattr(client_module, "LocalResearchClient", lambda *a, **k: task_stub)
    admitted = {"status": "TASK_ADMITTED", "task_id": str(uuid4()), "lifecycle": "QUEUED"}
    followed = client_module._follow(task_stub, admitted, 0.05)  # type: ignore[arg-type]
    assert followed["wait_event"]["event"] == "MAX_WAIT_REACHED"
    assert outcome_of(followed) == "PENDING"


def test_a_read_followed_to_its_end_reads_its_selection_again(monkeypatch) -> None:
    """A read followed to its end reads its selection again."""

    from alphalattice.interface.local_application.client import continued

    task, child = str(uuid4()), str(uuid4())
    read_request = {
        "operation": "EXPERIMENT_READBACK",
        "task_id": task,
        "portfolio_session": "2026-09-01",
    }
    monkeypatch.setattr(client_module.time, "sleep", lambda _s: None)
    stub = _Stub(
        [{"status": "SUCCEEDED", "lifecycle": "SUCCEEDED", "task_id": task}],
    )
    shown = {
        "status": "RUNNING",
        "lifecycle": "RUNNING",
        "task_id": task,
        "read_request": read_request,
    }
    final = client_module._follow(stub, shown, None)  # type: ignore[arg-type]
    assert final["read_request"] == read_request and final["admission"] == shown
    allowed = frozenset({"task_id", "portfolio_session"})
    again = continued("EXPERIMENT_READBACK", final, {}, allowed)
    assert again["portfolio_session"] == "2026-09-01" and again["task_id"] == task
    moved = continued("EXPERIMENT_READBACK", final, {"portfolio_session": "2026-09-02"}, allowed)
    assert moved["portfolio_session"] == "2026-09-02"
    promoted = _Stub([{"status": "SUCCEEDED", "lifecycle": "SUCCEEDED", "task_id": child}])
    promotion = {**shown, "follow_task_id": child}
    final = client_module._follow(promoted, promotion, None)  # type: ignore[arg-type]
    assert "read_request" not in final and final["task_id"] == child


class _ScriptedHost:
    """A Host answering each operation as its owner does, by the fields the verb reads."""

    def __init__(self, workspace: Path, *, prerequisites: bool = True) -> None:
        self.workspace = workspace
        self.goal = None
        self.sent: list[dict[str, Any]] = []
        self.prerequisites = prerequisites
        self.previews = 0
        self.prepares = 0
        self.prepared_request: dict[str, Any] | None = None

    def activity(self) -> dict[str, Any]:
        return {"observer": {"codex_queue": {"present": True}}}

    def request(self, document: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        self.sent.append(document)
        book = {"result_hash": "b" * 64}
        match document["operation"]:
            case "CONTROLS":
                return {
                    "status": "CONTROLS",
                    "template": {"strategy_package_id": "pkg", "lookback": 20},
                    "next_requests": {"preview": {"operation": "PLAN", "spec": {}}},
                }
            case "RUN":
                assert document["spec"] == {"strategy_package_id": "pkg", "lookback": 20}
                return {"status": "ADMITTED", "disposition": "ADMITTED", "task_id": "t-book"}
            case "STATUS":
                return {"status": "SUCCEEDED", "lifecycle": "SUCCEEDED", **document}
            case "RESULTS":
                # The book's Task names no result; the saved results do, by their Task.
                return {
                    "results": [
                        {"task_id": "other", "result_hash": "a" * 64},
                        {"task_id": "t-book", **book},
                    ]
                }
            case "REPORT":
                assert document["result_hash"] == book["result_hash"]
                return {
                    "result_hash": book["result_hash"],
                    "review_selector": book,
                    "next_requests": {
                        "review": {"operation": "EVIDENCE_CRO", **book},
                        "evidence_preview": {"operation": "EVIDENCE_PREVIEW", **book},
                    },
                }
            case "RESEARCH_UPDATE_READBACK":
                selector = {"update_task_id": document["task_id"], **PUBLICATION}
                return {
                    "status": "PUBLISHED",
                    "update": {"target_session": "2026-10-12"},
                    "review_selector": selector,
                    "next_requests": {
                        "evidence_preview": {"operation": "EVIDENCE_PREVIEW", **selector},
                        "review": {"operation": "EVIDENCE_CRO", **selector},
                    },
                }
            case "EVIDENCE_PREVIEW":
                self.previews += 1
                if not self.prerequisites:
                    return {
                        "status": "EVIDENCE_PREREQUISITES_MISSING",
                        "next_requests": {"setup": {"operation": "EVIDENCE_SETUP"}},
                    }
                subject = {k: v for k, v in document.items() if k != "operation"}
                subject["evidence_as_of"] = f"2026-10-12T12:{self.previews:02d}:00+00:00"
                subject["preparation_binding_hash"] = "c" * 64
                return {
                    "status": "EVIDENCE_PREPARATION_READY",
                    "coverage": {"unit_count": 2},
                    "next_requests": {"prepare": {"operation": "EVIDENCE_PREPARE", **subject}},
                }
            case "EVIDENCE_PREPARE" if not self.prepares:
                self.prepares += 1
                self.prepared_request = document
                return {"status": "ADMITTED", "task_id": "t-evidence", "lifecycle": "QUEUED"}
            case "EVIDENCE_PREPARE":
                assert document == self.prepared_request
                # The same preparation asked again: reused, its own units offered.
                subject = {k: v for k, v in document.items() if k != "operation"}
                units = {
                    f"analyst_bundle_{unit}": {
                        "operation": "AGENT_BUNDLE_PREPARE",
                        "agent_role": "ANALYST",
                        "evidence_unit_id": unit,
                        "task_id": "t-evidence",
                        "bundle_directory": None,
                        **subject,
                    }
                    for unit in ("u1", "u2")
                }
                return {"status": "REUSED_EXACT", "task_id": "t-evidence", "next_requests": units}
            case "WAKE_REGISTER":
                return {"status": "WAKE_REGISTERED", "task_id": document["task_id"]}
            case "AGENT_BUNDLE_PREPARE":
                return {
                    "status": "AGENT_BUNDLE_READY",
                    "agent_role": "ANALYST",
                    "bundle_directory": document["bundle_directory"],
                    "index": "README.md",
                    "files": [{"name": "README.md", "text": "Read the packet.\n"}],
                    "next_requests": {"submit": {"operation": "AGENT_ANSWER_SUBMIT"}},
                }
        raise AssertionError(document)


PUBLICATION = {"update_publication_hash": "p" * 64, "position_basis": "CONDITIONAL_ESTIMATE"}
"""A date's sealed publication, as an update's readback names it."""


def _review_args(root: Path, update: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        strategy_package_id="pkg", bundle_root=root, max_wait=None, update_task_id=update
    )


def test_a_book_review_runs_the_book_prepares_evidence_and_writes_each_analyst_bundle(
    tmp_path: Path,
) -> None:
    """requirement (AGENT-TIME verb 2, approved 2026-10-08): one call sends the requests the
    answers offer, in their order, following each Task to its end, and returns every Analyst
    bundle written with its answer path and submit command; each step stays a command."""
    host = _ScriptedHost(tmp_path / "workspace")
    answer = client_module._review_steps(host, _review_args(tmp_path / "analysts"))  # type: ignore[arg-type]

    assert answer["status"] == "BOOK_REVIEW_READY" and answer["next_action"] == "DISPATCH_ANALYSTS"
    assert [step["step"] for step in answer["steps"]] == [
        "controls",
        "book",
        "book_task",
        "book_result",
        "book_readback",
        "evidence_preview",
        "evidence",
        "evidence_task",
        "evidence_prepared",
        "analyst_bundle",
        "analyst_bundle",
    ]
    # Its bundles are the prepared run's own; a later preview would pack other units.
    assert host.previews == 1
    assert [bundle["unit"] for bundle in answer["analyst_bundles"]] == ["u1", "u2"]
    for bundle in answer["analyst_bundles"]:
        folder = Path(bundle["bundle_directory"])
        assert folder.parent == (tmp_path / "analysts").resolve()
        assert (folder / "README.md").read_text(encoding="utf-8") == "Read the packet.\n"
        assert bundle["answer_file"] == str(folder / "answer.json")
        assert bundle["submit_arguments"][-4:] == [
            "--dir",
            str(folder),
            "--file",
            bundle["answer_file"],
        ]
    assert answer["book"]["review_selector"] == {"result_hash": "b" * 64}
    assert answer["next_requests"] == {
        "review": {"operation": "EVIDENCE_CRO", "result_hash": "b" * 64}
    }
    assert {d["task_id"] for d in host.sent if d["operation"] == "STATUS"} == {
        "t-book",
        "t-evidence",
    }


def test_a_book_review_stops_at_the_first_answer_that_needs_another_step(tmp_path: Path) -> None:
    """requirement (AGENT-TIME verb 2): a missing prerequisite ends the call with that answer,
    its way on kept and the steps taken named; no bundle is written."""
    host = _ScriptedHost(tmp_path / "workspace", prerequisites=False)
    answer = client_module._review_steps(host, _review_args(tmp_path / "analysts"))  # type: ignore[arg-type]

    assert answer["status"] == "EVIDENCE_PREREQUISITES_MISSING"
    assert answer["next_requests"] == {"setup": {"operation": "EVIDENCE_SETUP"}}
    assert answer["book_review"]["stopped_at"] == "evidence_preview"
    assert not (tmp_path / "analysts").exists()


def test_a_review_of_a_dates_positions_binds_their_publication(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """A date review's wake resumes its exact preparation despite an advancing preview clock."""
    host = _ScriptedHost(tmp_path / "workspace")
    monkeypatch.setenv("CODEX_THREAD_ID", str(uuid4()))
    monkeypatch.setenv("ALPHALATTICE_SHELL", "posix")
    monkeypatch.setattr(client_module, "LocalResearchClient", lambda *a, **k: host)
    host.selected_url = lambda *_: ""  # type: ignore[attr-defined]
    line = [
        *("--workspace", str(host.workspace), "--view", "full", "strategy-book", "review"),
        *("--package", "pkg", "--dir", str(tmp_path / "analysts"), "--update", "t-update"),
        *("--notify", "codex-queue", "--output", str(tmp_path / "review.json")),
    ]
    assert main(line, serve=lambda _: 0) == 0
    capsys.readouterr()
    wake = next(d for d in host.sent if d["operation"] == "WAKE_REGISTER")
    assert wake["task_id"] == "t-evidence"
    assert main([*shlex.split(wake["wake_read"])[1:], "--view", "full"], serve=lambda _: 0) == 0
    answer = json.loads(capsys.readouterr().out)["data"]

    assert answer["status"] == "BOOK_REVIEW_READY"
    assert answer["positions"]["review_selector"]["update_publication_hash"] == "p" * 64
    previews = [d for d in host.sent if d["operation"] == "EVIDENCE_PREVIEW"]
    assert [d["update_publication_hash"] for d in previews] == ["p" * 64]
    assert not any(d["operation"] in {"CONTROLS", "RUN"} for d in host.sent)
    assert [bundle["unit"] for bundle in answer["analyst_bundles"]] == ["u1", "u2"]
    assert host.previews == 1 and host.prepares == 1
    assert answer["evidence"]["prepared_task_id"] == "t-evidence"


class _ReviewHost:
    """A Host answering a review's submissions, Evidence, dossier and offer as their owners do."""

    BOOK: ClassVar[dict[str, str]] = {"result_hash": "b" * 64}

    def __init__(self, workspace: Path, *, correct: str | None = None, places: int = 8) -> None:
        self.workspace = workspace
        self.goal = None
        self.sent: list[dict[str, Any]] = []
        self.correct = correct
        self.places, self.queued = places, set[str]()

    def request(self, document: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        self.sent.append(document)
        match document["operation"]:
            case "AGENT_ANSWER_SUBMIT" if len(self.queued) >= self.places:
                return {"status": "REFUSED", "failure_code": "REFUSED_QUEUE_FULL"}
            case "AGENT_ANSWER_SUBMIT":
                folder = Path(document["bundle_directory"])
                role = "CRO" if folder.name == "cro" else "ANALYST"
                if folder.name == self.correct:
                    return {
                        "status": "CORRECT",
                        "agent_role": role,
                        "problems": [{"item": 1, "text": "S9 is not an excerpt of this bundle."}],
                        "rounds_left": 2,
                    }
                task = f"t-{folder.name}"
                self.queued.add(task)
                return {
                    "status": "ACCEPTED",
                    "agent_role": role,
                    "task_id": task,
                    "task_lifecycle": "QUEUED",
                    "receipt": {"agent_role": role, "verdict": "ACCEPTED", "task_id": task},
                    "next_requests": {
                        "task": {"operation": "STATUS", "task_id": task},
                        "book": {"operation": "EVIDENCE_CRO", **self.BOOK},
                    },
                }
            case "STATUS":
                self.queued.discard(document["task_id"])
                return {"status": "SUCCEEDED", "lifecycle": "SUCCEEDED", **document}
            case "EVIDENCE_CRO":
                return {
                    "state": "ALTERNATIVE_EVIDENCE_READY_FOR_REVIEW",
                    "next_requests": {"dossier": {"operation": "CRO_REVIEW_DOSSIER", **self.BOOK}},
                }
            case "CRO_REVIEW_DOSSIER":
                return {
                    "status": "CRO_DOSSIER_READY",
                    "coverage_words": "Every holding was read.",
                    "next_requests": {
                        "cro_bundle": {
                            "operation": "AGENT_BUNDLE_PREPARE",
                            "agent_role": "CRO",
                            "bundle_directory": None,
                            **self.BOOK,
                        }
                    },
                }
            case "AGENT_BUNDLE_PREPARE":
                return {
                    "status": "AGENT_BUNDLE_READY",
                    "agent_role": "CRO",
                    "bundle_directory": document["bundle_directory"],
                    "index": "README.md",
                    "files": [{"name": "README.md", "text": "Read the dossier.\n"}],
                }
            case "CONTROLS":
                return {
                    "activation": {
                        "status": "INACTIVE",
                        "review_holdings": {"positions": []},
                        "next_requests": {"activate": {"operation": "STRATEGY_ACTIVATE"}},
                    }
                }
        raise AssertionError(document)


def _answers(root: Path, *names: str) -> Path:
    for name in names:
        (root / name).mkdir(parents=True)
        (root / name / "answer.json").write_text('{"findings": []}', encoding="utf-8")
    return root


def _continue_args(root: Path, **fields: Any) -> SimpleNamespace:
    return SimpleNamespace(
        bundle_root=root,
        cro_root=fields.get("cro_root"),
        strategy_package_id=fields.get("package"),
        max_wait=None,
    )


def test_review_continue_submits_the_analysts_and_writes_the_cros_bundle(tmp_path: Path) -> None:
    """requirement (AGENT-TIME verb 3): one call submits every Analyst answer within the Host's
    waiting places, follows each publication, and writes the CRO's offered bundle with its
    answer path and submit command."""
    host = _ReviewHost(tmp_path / "workspace", places=1)
    root = _answers(tmp_path / "analysts", "analyst-u1", "analyst-u2")
    answer = client_module._continue_steps(  # type: ignore[arg-type]
        host, _continue_args(root, cro_root=tmp_path / "cro")
    )

    assert answer["status"] == "CRO_BUNDLE_READY" and answer["next_action"] == "DISPATCH_CRO"
    # The second answer met a full queue: the first publication was followed, then it was sent.
    assert sum(d["operation"] == "AGENT_ANSWER_SUBMIT" for d in host.sent) == 3
    assert [receipt["task_id"] for receipt in answer["receipts"]] == [
        "t-analyst-u1",
        "t-analyst-u2",
    ]
    assert {d["task_id"] for d in host.sent if d["operation"] == "STATUS"} == {
        "t-analyst-u1",
        "t-analyst-u2",
    }
    bundle = answer["cro_bundle"]
    assert Path(bundle["bundle_directory"]) == (tmp_path / "cro").resolve()
    assert (tmp_path / "cro" / "README.md").read_text(encoding="utf-8") == "Read the dossier.\n"
    assert bundle["answer_file"] == str((tmp_path / "cro").resolve() / "answer.json")
    assert answer["dossier"] == {"coverage_words": "Every holding was read."}


def test_review_continue_names_each_correction_and_goes_no_further(tmp_path: Path) -> None:
    """requirement (AGENT-TIME verb 3): an answer the Host asks to correct is named with its
    problems; the accepted ones stand, and nothing past the submissions runs."""
    host = _ReviewHost(tmp_path / "workspace", correct="analyst-u2")
    root = _answers(tmp_path / "analysts", "analyst-u1", "analyst-u2")
    answer = client_module._continue_steps(host, _continue_args(root))  # type: ignore[arg-type]

    assert answer["status"] == "REVIEW_ANSWERS_NEED_CORRECTION"
    assert [c["bundle_directory"] for c in answer["corrections"]] == [str(root / "analyst-u2")]
    assert answer["corrections"][0]["problems"][0]["item"] == 1
    assert [r["task_id"] for r in answer["receipts"]] == ["t-analyst-u1"]
    assert not any(d["operation"] in {"STATUS", "EVIDENCE_CRO"} for d in host.sent)


def test_review_continue_publishes_the_cros_review_and_reads_the_activation_offer(
    tmp_path: Path,
) -> None:
    """requirement (AGENT-TIME verb 3): after the CRO's answer its publication is followed and
    the answer is Evidence's state with the strategy's activation offer and reviewed holdings."""
    host = _ReviewHost(tmp_path / "workspace")
    root = _answers(tmp_path, "cro") / "cro"
    answer = client_module._continue_steps(  # type: ignore[arg-type]
        host, _continue_args(root, package="pkg")
    )

    assert answer["status"] == "REVIEW_PUBLISHED" and answer["next_action"] == "OFFER_ACTIVATION"
    assert answer["evidence"]["state"] == "ALTERNATIVE_EVIDENCE_READY_FOR_REVIEW"
    assert answer["activation"]["review_holdings"] == {"positions": []}
    assert answer["next_requests"] == {"activate": {"operation": "STRATEGY_ACTIVATE"}}


class _FirstUseHost:
    """A Host answering a first use's goal, data issues and preparation as their owners do;
    ``stopped`` holds the statuses of a stopped preparation's data issues."""

    def __init__(self, workspace: Path, *, stopped: tuple[str, ...] = (), prepared: bool = False):
        self.workspace, self.goal, self.prepared = workspace, None, prepared
        self.stopped, self.resumes = list(stopped), 1
        self.sent: list[dict[str, Any]] = []

    def request(self, document: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        self.sent.append(document)
        match document["operation"]:
            case "GOAL_OPEN" if self.prepared:
                return {"status": "REFUSED", "failure_code": "goal.first_use_after_preparation"}
            case "GOAL_OPEN":
                return {"status": "OPENED", "goal_id": "g-1"}
            case "DATA_ISSUES":
                resume = {"operation": "WORKSPACE_PREPARE_PLAN"}
                offers = {
                    f"continue:t-{n}": {**resume, "recovery_task_id": f"t-{n}"}
                    for n in range(1, self.resumes + 1)
                    if self.stopped
                }
                confirm = {"confirm:c1:retain": {"operation": "DATA_ISSUE_CONFIRM"}}
                return {
                    "issues": [{"status": status} for status in self.stopped],
                    "next_requests": {**offers, **confirm},
                }
            case "WORKSPACE_PREPARE_PLAN" if self.prepared:
                return {"status": "ALREADY_PREPARED", "inputs": [{"input_id": "in-1"}]}
            case "WORKSPACE_PREPARE_PLAN":
                recovery = {k: v for k, v in document.items() if k.startswith("recovery_")}
                confirm = {"operation": "WORKSPACE_PREPARE_CONFIRM", "preparation_plan_hash": "p"}
                return {"status": "PLANNED", "next_requests": {"confirm": {**confirm, **recovery}}}
            case "WORKSPACE_PREPARE_CONFIRM":
                task = "t-resumed" if self.stopped else "t-1"
                return {"status": "ADMITTED", "task_id": task, "lifecycle": "QUEUED"}
            case "STATUS" if document["task_id"] == "t-1":
                self.stopped = ["AWAITING_CHOICE"]
                return {"lifecycle": "BLOCKED", "latest_failure_code": "data.truth_review_required"}
            case "STATUS":
                self.prepared = True
                return {"status": "SUCCEEDED", "lifecycle": "SUCCEEDED", **document}
        raise AssertionError(document)


def _first_use(host: _FirstUseHost, named: str = "2026-10-10") -> dict[str, Any]:
    args = SimpleNamespace(
        objective="Positions for Saturday.",
        target_date=named,
        max_wait=None,
        notify=None,
        output=None,
    )
    return client_module._first_use_steps(host, args)  # type: ignore[arg-type]


def test_a_first_use_opens_its_goal_from_the_sentence_and_stops_at_its_data_issues(
    tmp_path: Path,
) -> None:
    """requirement (fewer agent steps): one call opens the goal from the person's sentence and
    named date, prepares, and stops where the preparation needs data decisions."""
    host = _FirstUseHost(tmp_path / "workspace")
    stopped = _first_use(host)

    declaration = host.sent[0]["goal_declaration"]
    assert declaration["objective"] == "Positions for Saturday."
    assert declaration["target_date"] == "2026-10-10"
    assert "2026-10-12" in declaration["criteria"][0]["text"]
    assert host.goal == "g-1" and stopped["first_use"]["stopped_at"] == "issues"
    assert "confirm:c1:retain" in stopped["next_requests"]


@pytest.mark.parametrize(
    ("stopped", "resumes", "resumed"),
    [
        (("CONFIRMED_PENDING_REVALIDATION",) * 2, 1, True),
        (("CONFIRMED_PENDING_REVALIDATION", "OPTION_REFUSED"), 1, False),
        (("WAITING_FOR_RETRY",), 1, False),
        (("CONFIRMED_PENDING_REVALIDATION",), 2, False),
    ],
)
def test_a_first_use_resumes_its_stopped_preparation_only_once_every_decision_is_confirmed(
    tmp_path: Path, stopped: tuple[str, ...], resumes: int, resumed: bool
) -> None:
    """requirement (fewer agent steps): the same call again resumes the stopped preparation
    through its recovery link once every decision is confirmed; a refused or waiting decision,
    or two stopped preparations, stop with the issues for the agent."""
    host = _FirstUseHost(tmp_path / "workspace", stopped=stopped)
    host.resumes = resumes
    answer = _first_use(host)

    confirms = [d for d in host.sent if d["operation"] == "WORKSPACE_PREPARE_CONFIRM"]
    assert [d.get("recovery_task_id") for d in confirms] == (["t-1"] if resumed else [])
    assert (answer.get("status") == "ALREADY_PREPARED") is resumed
    assert (answer["first_use"].get("stopped_at") == "issues") is not resumed


def test_a_first_use_answer_lays_out_the_whole_first_use_and_what_it_did_not_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """requirement (plan first): every answer echoes the date's sessions, names what this
    session's setup lacks, carries each later step's command and names an unrecorded sentence."""
    monkeypatch.chdir(tmp_path)  # a project with no declaration for this session's host
    monkeypatch.delenv("CLAUDE_CODE_SESSION_ID", raising=False)
    monkeypatch.setenv("CODEX_THREAD_ID", "thread-1")
    answer = _first_use(_FirstUseHost(tmp_path / "workspace", prepared=True))

    # A Saturday names no session: the positions enter on Monday, decided at Friday's close.
    date_ = answer["first_use"]["date"]
    assert (date_["named_is_session"], date_["entry_session"]) == (False, "2026-10-12")
    assert date_["formation_session"] == "2026-10-09"
    road = answer["first_use"]["road"]
    assert [step["step"] for step in road] == [
        *("prepare", "strategy", "book", "activate", "review", "cro", "publish"),
        *("committee", "report"),
    ]
    assert answer["next_action"] == "BUILD_THE_STRATEGY"
    assert answer["next_command"] == road[1]["command"]
    asked = answer["first_use"]["ask_now"][0]
    assert "default budget" in asked and "retrieval model" in asked
    assert "SEC_USER_AGENT" not in asked
    assert "not recorded" in answer["first_use"]["sentence"]
    assert "project_declaration" in answer["first_use"]["setup"]["missing"]


@pytest.mark.parametrize(
    ("named", "read"),
    [
        ("2026-09-07", (False, "2026-09-08", "2026-09-04")),
        ("2026-10-09", (True, "2026-10-09", "2026-10-08")),
        ("2026-13-01", "local_client.first_use_date_invalid"),
        ("2099-01-02", "local_client.first_use_date_outside_calendar"),
    ],
)
def test_a_first_use_reads_the_named_date_by_the_exchange_calendars(
    tmp_path: Path, named: str, read: object
) -> None:
    """requirement (the person's date): a holiday enters on the next session, a session date on
    itself; a date that is not one, or that the calendars do not plan, is refused by name."""
    answer = _first_use(_FirstUseHost(tmp_path / "workspace", prepared=True), named)

    if isinstance(read, str):
        assert answer["failure_code"] == read and answer["detail"], answer
        return
    date_ = answer["first_use"]["date"]
    assert (date_["named_is_session"], date_["entry_session"], date_["formation_session"]) == read


WINDOW = {"start": "2022-07-01", "end": "2026-09-03", "formation_history": {"start": "2021-06-30"}}
"""The window a calibrated study names, with the history a Risk study needs before it."""
LAST = {"start": "2026-08-07", "end": "2026-09-04"}
"""A Risk template's default: the last formations only."""


class _BuildHost:
    """A Host answering a strategy's controls, training, studies and installation, admitting
    each as its owner does: training and the strategy's preparation only beside no unfinished
    Task, a study whenever."""

    def __init__(self, workspace: Path, *, risk_template: bool = True, covers: bool = True):
        self.workspace, self.goal, self.risk_template = workspace, None, risk_template
        self.covers = covers
        self.done: set[str] = set()
        self.unfinished: set[str] = set()

    def controls(self) -> dict[str, Any]:
        trained = {"t-train-G2", "t-train-G6"} <= self.done
        alphas = {"t-alpha-G2", "t-alpha-G6"} <= self.done
        covered = "t-risk" in self.done and self.covers
        step = "EXPERIMENT_CONTROLS" if trained else "MODEL_TRAINING_INPUT_PLAN"
        routes = {} if alphas else {f"component:{c}": c for c in ("G2", "G6")}
        plan = {"alpha_task_ids": ["t-alpha-G2", "t-alpha-G6"], "risk_task_id": "t-risk"}
        return {
            "risk_windows": [WINDOW] if alphas else [],
            "next_requests": {
                **{name: {"operation": step, "component_id": c} for name, c in routes.items()},
                **({} if covered else {"risk": {"operation": "EXPERIMENT_CONTROLS"}}),
                "plan": {
                    "operation": "RESEARCH_STRATEGY_PLAN",
                    "experiment_document": plan if covered else None,
                },
            },
        }

    def admit(self, task: str, *, exclusive: bool) -> dict[str, Any]:
        if task in self.done:
            return {"status": "REUSED_EXACT", "task_id": None, "publication_task_id": task}
        if exclusive and self.unfinished:
            return {"status": "REFUSED", "failure_code": "finish_or_recover_existing_task"}
        self.unfinished.add(task)
        return {"status": "ADMITTED", "task_id": task}

    def request(self, document: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        operation = document["operation"]
        offered = {
            "MODEL_TRAINING_INPUT_PLAN": ("prepare", "MODEL_TRAINING_INPUT_PREPARE"),
            "EXPERIMENT_PLAN": ("run", "EXPERIMENT_RUN"),
            "RESEARCH_STRATEGY_PLAN": ("prepare", "RESEARCH_STRATEGY_PREPARE"),
        }
        match operation:
            case "RESEARCH_STRATEGY_CONTROLS":
                return self.controls()
            case _ if operation in offered:
                name, next_operation = offered[operation]
                of = document.get("component_id") or document.get("experiment_document")
                request = {"operation": next_operation, "experiment_plan_hash": "p", "of": of}
                return {"status": "PLANNED", "next_requests": {name: request}}
            case "MODEL_TRAINING_INPUT_PREPARE":
                return self.admit(f"t-train-{document['of']}", exclusive=True)
            case "EXPERIMENT_CONTROLS" if "component_id" not in document and not self.risk_template:
                return {"status": "REFUSED", "failure_code": "risk_research.insufficient_history"}
            case "EXPERIMENT_CONTROLS":
                component = document.get("component_id")
                return {
                    "status": "READY",
                    "input_id": "in-1",
                    "template": {"experiment": {"kind": component or "risk", "sessions": LAST}},
                    "model_lifecycle": "LIGHT" if component else None,
                }
            case "EXPERIMENT_RUN":
                kind = document["of"]["experiment"]["kind"]
                task = "t-risk" if kind == "risk" else f"t-alpha-{kind}"
                return self.admit(task, exclusive=False)
            case "STATUS":
                self.unfinished.discard(document["task_id"])
                self.done.add(document["task_id"])
                return {"status": "SUCCEEDED", "lifecycle": "SUCCEEDED", **document}
            case "RESEARCH_STRATEGY_PREPARE":
                read = {"operation": "RESEARCH_STRATEGY_READBACK", "task_id": "t-strategy"}
                return {
                    **self.admit("t-strategy", exclusive=True),
                    "next_requests": {"readback": read},
                }
            case "RESEARCH_STRATEGY_READBACK":
                install = {"operation": "RESEARCH_STRATEGY_INSTALL", "task_id": "t-strategy"}
                return {"status": "SUCCEEDED", "next_requests": {"install_non_default": install}}
            case "RESEARCH_STRATEGY_INSTALL":
                packages = ["BAL", "G6"]
                return {
                    "status": "INSTALLED_NON_DEFAULT_RESEARCH",
                    "strategy_package_ids": packages,
                }
        raise AssertionError(document)


def _build(host: _BuildHost) -> dict[str, Any]:
    args = SimpleNamespace(max_wait=None, notify=None, output=None)
    return client_module._build_steps(host, args)  # type: ignore[arg-type]


def test_a_strategy_build_takes_each_offered_step_as_its_owner_admits_it_and_installs(
    tmp_path: Path,
) -> None:
    """requirement (fewer agent steps): one call trains, runs the Alpha and then the Risk study,
    each admitted beside no unfinished Task as training's owner requires, then plans from the
    filled declaration, prepares and installs, naming the book review next."""
    answer = _build(_BuildHost(tmp_path / "workspace"))

    assert answer["status"] == "STRATEGY_INSTALLED"
    assert [s["task_id"] for s in answer["studies"]] == ["t-alpha-G2", "t-alpha-G6", "t-risk"]
    assert answer["studies"][0]["model_lifecycle"] == "LIGHT"
    # The Risk study covers the named window from its formation history, not the default.
    assert answer["studies"][2]["sessions"] == {"start": "2021-06-30", "end": "2026-09-04"}
    # Each installed package's book run is offered; the agent chooses.
    assert answer["next_action"] == "RUN_THE_BOOK"
    assert {"book:BAL", "book:G6"} <= set(answer["next_requests"])


@pytest.mark.parametrize(
    ("risk_template", "covers", "stopped_at", "risk"),
    [(False, True, "study_controls", []), (True, False, "controls", [("t-risk", None)])],
)
def test_a_stopped_strategy_build_names_each_study_it_started_once(
    tmp_path: Path, risk_template: bool, covers: bool, stopped_at: str, risk: list[Any]
) -> None:
    """requirement (the guide's model disclosure): where a build stops, its answer keeps each
    study it started once, with its model lifecycle, for the agent to tell the person."""
    host = _BuildHost(tmp_path / "workspace", risk_template=risk_template, covers=covers)
    answer = _build(host)

    assert answer["strategy_build"]["stopped_at"] == stopped_at
    studies = answer["strategy_build"]["studies"]
    assert [(s["task_id"], s["model_lifecycle"]) for s in studies] == [
        ("t-alpha-G2", "LIGHT"),
        ("t-alpha-G6", "LIGHT"),
        *risk,
    ]
