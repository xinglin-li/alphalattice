"""A capped wait observes the last interval through the CLI, with no wall-clock delay."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

from alphalattice.interface.local_application import cli, client


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        assert seconds >= 0
        self.sleeps.append(seconds)
        self.now += seconds


class _Host:
    goal = None
    timeout = 60

    def __init__(
        self,
        workspace: Path,
        clock: _Clock,
        kind: str,
        arrival: float | None,
        restart: bool = False,
    ) -> None:
        self.workspace, self.clock, self.kind = workspace, clock, kind
        self.arrival, self.restart = arrival, restart
        self.reads: list[float] = []
        self.requests: list[dict[str, Any]] = []
        self.task, self.trial, self.goal_id = (str(uuid4()) for _ in range(3))

    def request(self, document: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        self.requests.append(dict(document))
        if self.restart:
            self.restart = False
            self.clock.now = 0.9
            self.reads.append(self.clock.now)
            raise client.LocalResearchClientError("local_client.service_not_running")
        self.clock.sleep(float(document.get("wait_seconds", 0)))
        self.reads.append(self.clock.now)
        arrived = self.arrival is not None and self.clock.now >= self.arrival
        if self.kind == "goal":
            conversation = (
                [
                    {
                        "observation_id": "reply",
                        "message_id": "reply",
                        "message_kind": "answer",
                        "agent_id": "child",
                        "recipient_id": "lead",
                    }
                ]
                if arrived
                else []
            )
            return {
                "status": "GOAL_NARRATIVE",
                "state": "OPEN",
                "record": {"tasks": [], "conversation": conversation},
            }
        if self.kind == "committee":
            return {"status": "COMMITTEE_FLOOR", "for_you": [1] if arrived else []}
        if self.kind == "trial":
            return {
                "status": "FEATURE_TRIAL",
                "feature_trial_id": self.trial,
                "state": "COMPLETED" if arrived else "RUNNING",
            }
        return {
            "status": "SUCCEEDED" if arrived else "RUNNING",
            "task_id": self.task,
            "lifecycle": "SUCCEEDED" if arrived else "RUNNING",
        }

    def exchange(self, document: dict[str, Any]) -> tuple[dict[str, Any], None]:
        return self.request(document), None

    def selected_url(self, *_args: Any) -> None:
        return None


@pytest.mark.parametrize("kind", ["goal", "activity-task", "task", "trial", "committee"])
@pytest.mark.parametrize("arrival", [None, 0.5, 1.0])
def test_every_cli_wait_reads_at_its_cap(tmp_path, monkeypatch, capsys, kind, arrival) -> None:
    """every wait reaches its cap, sees the final interval and never extends it."""
    clock = _Clock()
    host = _Host(tmp_path, clock, "goal" if kind == "goal" else kind, arrival)
    monkeypatch.setattr(client.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(client.time, "sleep", clock.sleep)
    monkeypatch.setattr(client, "LocalResearchClient", lambda *_a, **_k: host)
    monkeypatch.setenv("CODEX_THREAD_ID", "lead")
    command = {
        "goal": ["activity", "wait", "--goal", host.goal_id],
        "activity-task": ["activity", "wait", "--task", host.task],
        "task": ["task", "show", host.task, "--wait"],
        "trial": ["trial", "show", host.trial, "--wait"],
        "committee": ["committee", "wait", "--update", host.task, "--role", "PM", "--key", "k"],
    }[kind]
    code = cli.main(
        ["--workspace", str(tmp_path), *command, "--max-wait", "1", "--view", "full"],
        serve=lambda _: pytest.fail("no serve"),
    )
    answer = json.loads(capsys.readouterr().out)
    expected = "MAX_WAIT_REACHED" if arrival is None else "MESSAGE" if kind == "goal" else "ENDED"
    got = answer["data"].get("wait_event", {}).get("event") or answer["data"].get("wait_status")
    assert (got or "ENDED") == expected  # a committee floor answers itself, with no event
    assert code == (3 if arrival is None else 0)
    if arrival is None or arrival == 1.0 or kind != "trial":
        assert host.reads[-1] == pytest.approx(1.0)
        assert clock.now == pytest.approx(1.0)
    else:
        assert arrival <= clock.now <= 1.0
    assert len(host.reads) >= 2
    assert all(t <= 1.0 for t in host.reads)


@pytest.mark.parametrize("kind", ["goal", "activity-task"])
@pytest.mark.parametrize("available", [False, True])
def test_a_restart_near_the_cap_gets_its_last_cli_read(
    tmp_path, monkeypatch, capsys, kind, available
) -> None:
    """a reconnect at 0.9 seconds still attempts its last read at one second."""
    clock = _Clock()
    host = _Host(tmp_path, clock, "goal" if kind == "goal" else kind, None, restart=True)
    request = host.request

    def read(document, timeout=None):
        if not available and host.reads:
            host.reads.append(clock.now)
            raise client.LocalResearchClientError("local_client.service_not_running")
        return request(document, timeout)

    monkeypatch.setattr(host, "request", read)
    monkeypatch.setattr(client.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(client.time, "sleep", clock.sleep)
    monkeypatch.setattr(client, "LocalResearchClient", lambda *_a, **_k: host)
    target = ["--goal", host.goal_id] if kind == "goal" else ["--task", host.task]
    code = cli.main(
        [
            "--workspace",
            str(tmp_path),
            "activity",
            "wait",
            *target,
            "--max-wait",
            "1",
            "--view",
            "full",
        ],
        serve=lambda _: pytest.fail("no serve"),
    )
    answer = json.loads(capsys.readouterr().out)
    assert code == 3 and answer["data"]["wait_event"]["event"] == "MAX_WAIT_REACHED"
    assert host.reads[0] == pytest.approx(0.9)
    assert host.reads[-1] == pytest.approx(1.0)
    assert clock.now == pytest.approx(1.0)
    assert clock.sleeps[0] == pytest.approx(0.1)


@pytest.mark.parametrize("cap", [0.05, 1.0, 2.0, 3.0, 17.0])
@pytest.mark.parametrize("delay", [0.25, 1.0, 2.0, 15.0, 30.0])
def test_the_shared_pause_reserves_a_final_read(monkeypatch, cap, delay) -> None:
    """All client polling/backoff delays share the cap rule, including sub-delay caps."""
    clock = _Clock()
    monkeypatch.setattr(client.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(client.time, "sleep", clock.sleep)
    reads = [clock.now]
    while client._sleep_before_read(delay, cap):
        reads.append(clock.now)
    assert reads[-1] == pytest.approx(cap)
    assert all(t <= cap for t in reads)
    assert len(reads) >= 2


@pytest.mark.parametrize("cap", [0.05, 1.0, 2.0])
@pytest.mark.parametrize("arrives", [False, True])
def test_the_host_status_poll_reads_at_its_cap(monkeypatch, cap, arrives) -> None:
    """The Host long poll already sleeps by remaining time and reads before timing out."""

    from alphalattice.control.product_host.composition import portfolio_research_operations as host
    from alphalattice.control.task_control.contracts import TaskLifecycle

    clock, reads = _Clock(), []
    running = SimpleNamespace(lifecycle=TaskLifecycle.RUNNING, verified_stage_count=0)
    ended = SimpleNamespace(lifecycle=TaskLifecycle.SUCCEEDED, verified_stage_count=1)

    def status(_task):
        reads.append(clock.now)
        return ended if arrives and clock.now >= cap else running

    monkeypatch.setattr(host, "monotonic", clock.monotonic)
    monkeypatch.setattr(host, "sleep", clock.sleep)
    owner = SimpleNamespace(dispatcher=SimpleNamespace(status=status))
    result = host.PortfolioResearchOperations._moved_on(owner, uuid4(), running, cap)
    assert result is (ended if arrives else running)
    assert reads[-1] == pytest.approx(cap) and clock.now == pytest.approx(cap)
    assert all(t <= cap for t in reads)
