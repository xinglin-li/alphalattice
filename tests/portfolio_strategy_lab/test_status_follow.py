"""A follow learns when its Task moves on (binding plan N8): STATUS may wait on the Host for the
Task's next stage or lifecycle, and the CLI's `--wait` asks it to instead of polling.

For the UI line's suite: tests/portfolio_strategy_lab/test_status_follow.py.
"""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

from alphalattice.control.product_host.composition.portfolio_research_operations import (
    PortfolioResearchOperations,
)
from alphalattice.control.task_control.contracts import TaskLifecycle
from alphalattice.interface.local_application import client as client_module
from alphalattice.interface.local_application.cli_contract import outcome_of
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


def test_the_follow_asks_the_host_to_wait_instead_of_polling() -> None:
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
    admitted = {
        "status": "UPSTREAM_PROMOTION_ADMITTED",
        "task_id": old,
        "follow_task_id": new,
        "next_requests": {"promote": promote},
    }
    final = client_module._follow(_Client(), admitted, 60)  # type: ignore[arg-type]
    assert [d["task_id"] for d in sent] == [new, new]
    assert final["lifecycle"] == "REVIEW_PENDING" and final["admission"] == admitted
    assert final["next_requests"] == {"promote": promote}
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
    """regression (V421, an outside review at 67d55a35): a coverage run's units are ready one by
    one, yet no waiter returned before the whole Task ended and the handoff told the lead to poll.
    `--each-stage` returns as the Task verifies a stage beyond those it held at the start, its
    end still ending the wait; it needs a Task."""

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
    """requirement (GR2, WK): an assignment made under a goal wakes its assignee's wait on the
    goal, and a reply its sender's, named in the one line; a message heard before does not.
    A waiter whose agent session is unknown wakes on every new message (V503)."""

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

    stub = _Stub([narrative(heard), narrative(heard), narrative(heard, assigned)])
    monkeypatch.setattr(client_module.time, "sleep", lambda _s: None)
    event = client_module._wait_for_goal(stub, goal, None)["wait_event"]  # type: ignore[arg-type]
    assert (event["event"], event["message_kind"], event["message_id"]) == (
        "MESSAGE",
        "assignment",
        "assign-1",
    )
    assert (event["sender"], event["recipient"], event["goal_id"]) == ("lead", "child", goal)
    assert "goal show" in event["read"] and goal in event["read"]


def test_a_goal_wait_wakes_only_for_its_own_messages(monkeypatch) -> None:
    """regression (V503, the user's review): the lead's `activity wait --goal` woke on a
    question one subagent sent another, which the lead could not act on. The waiter is its
    agent session: a message between two other agents never wakes it; one addressed to it, a
    reply to one it sent, or one addressed to no one does."""

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
    """regression (V507, the user's review at 244900d8): a data Task the provider deferred
    waits on its retry time and then on its plan sent again, yet `--wait`, `activity wait` and
    a goal's waiter read DEFERRED as running and never gave control back. Every waiter ends on
    it (`WAIT_EXITS`), the read it ends with naming the retry time and the request that resumes
    it."""

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

    from datetime import UTC, datetime

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
    host.workspace_session = SimpleNamespace(task_control_registry=Registry())  # type: ignore[assignment]
    host.preparation = Preparation()  # type: ignore[assignment]
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
    """regression (V520, the lifecycle sweep S2): a deferred preparation or update counted its
    wait for the provider as running, so its `running_seconds` grew for hours while nothing ran.
    Its clock stops at its last change, as a recovery's or a review's does."""

    from datetime import UTC, datetime, timedelta

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


def test_a_codex_wake_is_best_effort_and_an_undelivered_one_is_recorded(monkeypatch) -> None:
    """requirement (WK; the user, 2026-09-28: codex queue is 锦上添花): the queue is checked
    before the agent ends its turn, carries only the event and its read, is retried, and a
    wake it could not deliver is left beside the event in the Host's record."""

    import shutil
    import subprocess

    monkeypatch.setattr(shutil, "which", lambda _name: None)
    monkeypatch.setenv("CODEX_THREAD_ID", "00000000-0000-7000-8000-000000000001")
    with pytest.raises(client_module.LocalResearchClientError, match="codex_queue_unavailable"):
        client_module._codex_queue_ready()
    monkeypatch.setattr(shutil, "which", lambda _name: "codex")
    thread = client_module._codex_queue_ready()
    monkeypatch.setattr(client_module.time, "sleep", lambda _s: None)
    event = {"event": "ENDED", "task_id": str(uuid4()), "read": "alphalattice task show"}
    calls: list[list[str]] = []

    def failing(args: list[str], **_kwargs: Any) -> None:
        calls.append(args)
        raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(subprocess, "run", failing)
    stub = _Stub([])
    wake = client_module._queue_wake(stub, thread, event)  # type: ignore[arg-type]
    assert wake == {
        "channel": "codex-queue",
        "delivered": False,
        "failure": "CODEX_QUEUE_FAILED",
        "attempts": 3,
    }
    assert len(calls) == 3 and calls[0][:4] == ["codex", "queue", "--thread", thread]
    assert calls[0][5] == "Host event ENDED: read and verify it with alphalattice task show"
    (published,) = stub.published
    assert published["event_kind"] == "WAKE_UNDELIVERED"
    assert published["subject"] == {"task_id": event["task_id"]}
    monkeypatch.setattr(subprocess, "run", lambda *_a, **_k: None)
    assert client_module._queue_wake(stub, thread, event)["delivered"] is True  # type: ignore[arg-type]


def test_a_wait_capped_before_its_first_read_still_queues_its_wake(monkeypatch) -> None:
    """regression (V496, the user's review): `activity wait --task --max-wait --notify
    codex-queue` whose first read met a stopped Host returned at its cap before the tail that
    queues the wake, so the agent that ended its turn was never woken; every end of the wait,
    the cap before a first read included, queues it."""

    import shutil
    import subprocess

    monkeypatch.setattr(shutil, "which", lambda _name: "codex")
    monkeypatch.setenv("CODEX_THREAD_ID", "00000000-0000-7000-8000-000000000001")
    monkeypatch.setattr(client_module.time, "sleep", lambda _s: None)
    queued: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", lambda args, **_kwargs: queued.append(args))
    gone = client_module.LocalResearchClientError("local_client.service_not_running")
    stub = _Stub(itertools.repeat(gone))  # type: ignore[arg-type]
    args = SimpleNamespace(
        task_id=str(uuid4()), goal_id=None, max_wait=0.05, notify="codex-queue", each_stage=False
    )
    final = client_module._wait(stub, args)  # type: ignore[arg-type]
    assert final["wait_event"]["event"] == "MAX_WAIT_REACHED"
    assert final["wake"]["delivered"] is True and len(queued) == 1


def test_the_compact_view_leaves_out_the_timing_no_decision_reads() -> None:
    """requirement (V280): half of a `task show` answer was its stage timings; the compact
    view leaves them out and says so, the full view and --output keep them."""

    shown = client_module.compact_display(
        {"status": "RUNNING", "lifecycle": "RUNNING", "timing": {"stages": [{"s": 1.0}] * 50}}
    )
    assert "timing" not in shown["data"] and "timing" in shown["omitted_sections"]


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
    """regression (V440, an outside review at b13cb386): the goal waiter ended on a Task that
    succeeded, was cancelled or waits on a decision, never one that turned BLOCKED, which a Task
    waiter ends on at once; it reads each Task's state as the Task waiter does, and its read
    starts with the checkout's entry and the workspace."""

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
    """regression (V535, the user's review at a84e523f): the goal waiter remembered each Task's
    state from its start, so a Task that needed recovery when the wait began, recovered and
    needed it again slept until `MAX_WAIT_REACHED`, and its agent missed the new stop. Each
    Task is judged against the state the waiter last saw. A recovery that stops again between
    two reads is read as unchanged, the bound of a waiter that reads states."""

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
    """regression (V439, an outside review at b13cb386): a bundle's `submit_command` was joined
    with shlex, POSIX quoting PowerShell refuses as printed; it is quoted for the shell in use.
    Since V429 it starts with the installed `alphalattice`, which needs no call operator, and a
    path is quoted as the shell in use reads it: PowerShell doubles a single quote, POSIX
    closes and reopens it."""

    import shlex

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
    """regression (V449, an outside review at 368f0d6f): a trial whose step's Task needed
    recovery still read RUNNING, and `--wait` ran on to its cap, or for ever. The trial names
    the step's Task and its lifecycle and offers its recovery, so the wait ends there as a
    decision, the recovery kept."""

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


def test_every_wait_a_cap_ends_reads_pending() -> None:
    """regression (V563's class): every exit of a waiter that a cap ends carries `wait_status`,
    which the outcome reads as pending; none answers a cap as a completed read."""

    import ast

    tree = ast.parse(Path(client_module.__file__).read_text(encoding="utf-8"))
    exits: list[tuple[int, bool]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            named = {key.value for key in node.keys if isinstance(key, ast.Constant)}
            for key, value in zip(node.keys, node.values, strict=True):
                if (
                    isinstance(key, ast.Constant)
                    and key.value == "wait_event"
                    and isinstance(value, ast.Call)
                    and value.args
                    and isinstance(value.args[0], ast.Constant)
                    and value.args[0].value == "MAX_WAIT_REACHED"
                ):
                    exits.append((node.lineno, "wait_status" in named))
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "final"
            and len(node.args) > 1
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == "MAX_WAIT_REACHED"
        ):
            exits.append((node.lineno, any(kw.arg == "wait_status" for kw in node.keywords)))
    assert len(exits) >= 4 and all(pending for _line, pending in exits), exits


def test_a_read_followed_to_its_end_reads_its_selection_again(monkeypatch) -> None:
    """regression (V554, the user's review at de555b07): `study show T --session 2026-09-01
    --wait --output final.json` kept its day only under `admission.read_request`, so `study show
    --from final.json` read the newest day. A wait that ends on the Task its read selected
    keeps the read at the top, an explicit day still wins, and a wait that followed another
    Task (a promotion) never carries the parent's selection."""

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
