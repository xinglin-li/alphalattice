"""A Codex lead's wake is held by the Host in the Task's journal, not by the turn that asked
for it, and sent from the activity's hooks and the start's replay."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from alphalattice.control.task_control.contracts import TaskEvidence, TaskStageReceipt
from alphalattice.interface.local_application.cli import main
from alphalattice.interface.local_application.client import LocalResearchClient
from alphalattice.interface.local_application.dispatcher import CommandAdmission
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchOperationRequest as Request,
)
from alphalattice.kernel.shared_kernel.project_layout import command_prefix
from tests.workspace_task_runner.task_control_support import compatibility, task_contract

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/run_alphalattice.py"
THREAD = "019a0000-0000-7000-8000-00000000c0de"


@pytest.fixture
def codex(tmp_path, monkeypatch):
    """A `codex` on PATH that records each call and exits with FAKE_CODEX_EXIT."""
    folder = tmp_path / "bin"
    folder.mkdir()
    calls = tmp_path / "codex-calls.jsonl"
    recorder = folder / "recorder.py"
    recorder.write_text(
        "import json, os, sys\n"
        "if sys.argv[1:] == ['queue', '--help']:\n"
        "    with open(os.environ['FAKE_CODEX_PROBES'], 'a') as out: out.write('probe\\n')\n"
        "    sys.exit(int(os.environ.get('FAKE_CODEX_PROBE_EXIT', '0')))\n"
        "with open(os.environ['FAKE_CODEX_CALLS'], 'a', encoding='utf-8') as out:\n"
        "    out.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "sys.exit(int(os.environ.get('FAKE_CODEX_EXIT', '0')))\n",
        encoding="utf-8",
    )
    if os.name == "nt":
        (folder / "codex.cmd").write_text(f'@"{sys.executable}" "{recorder}" %*\r\n')
    else:
        (folder / "codex").write_text(f'#!/bin/sh\nexec "{sys.executable}" "{recorder}" "$@"\n')
        (folder / "codex").chmod(0o755)
    monkeypatch.setenv("PATH", str(folder))
    monkeypatch.setenv("CODEX_THREAD_ID", THREAD)
    monkeypatch.setenv("FAKE_CODEX_CALLS", str(calls))
    monkeypatch.setenv("FAKE_CODEX_PROBES", str(tmp_path / "codex-probes.txt"))
    return SimpleNamespace(
        folder=folder,
        command=Path(shutil.which("codex")),
        probes=lambda: (tmp_path / "codex-probes.txt").read_text().splitlines(),
        calls=lambda: (
            [json.loads(line) for line in calls.read_text("utf-8").splitlines()]
            if calls.exists()
            else []
        ),
    )


def _task(live, salt: str):  # type: ignore[no-untyped-def]
    envelope, goal, plan = task_contract(salt=salt)
    return live.session.task_control_registry.admit(
        input_envelope=envelope, goal=goal, plan=plan, observed_at=live.clock()
    ).record


def _register(live, task, *, env=None) -> dict:  # type: ignore[no-untyped-def,type-arg]
    """The lead's call: a real CLI process that answers and exits before the Task ends."""
    done = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            *("--workspace", str(live.workspace), "--view", "full"),
            *("activity", "wait", "--task", str(task.task_id), "--notify", "codex-queue"),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        env=env,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    answer = json.loads(done.stdout)["data"]
    assert answer["status"] == "WAKE_REGISTERED" and answer["wake"]["state"] == "PENDING"
    return answer["wake"]


def _run(live, task, *, decision: bool = False) -> dict:  # type: ignore[no-untyped-def,type-arg]
    """Run the Task on the Host's dispatcher to its end, or to a decision; answer its wake."""
    registry = live.session.task_control_registry

    def execute(task_id, *, expected_task_hash=None):  # type: ignore[no-untyped-def]
        started, execution = registry.start_next(
            compatibility=compatibility(task.plan),
            worker_instance_id=uuid4(),
            observed_at=live.clock(),
            expected_task_id=task_id,
        )
        if decision:
            registry.mark_recovery_required(
                task_id=task_id, failure_code="TASK_EXECUTION_INTERRUPTED", observed_at=live.clock()
            )
            return
        for definition in started.plan.work_items:
            stage = {
                "task_id": task_id,
                "execution_id": execution.execution_id,
                "stage_id": definition.stage_id,
                "observed_at": live.clock(),
            }
            registry.begin_work_item(**stage)
            evidence = tuple(
                TaskEvidence(
                    evidence_kind=kind,
                    reference=f"playpen://wake/{definition.stage_id}/{kind}",
                    content_hash="a" * 64,
                )
                for kind in definition.required_evidence_kinds
            )
            registry.mark_ready(**stage, evidence=evidence)
            registry.verify_work_item(
                TaskStageReceipt.from_identity(
                    **stage,
                    receipt_id=uuid4(),
                    evidence=evidence,
                    status="VERIFIED",
                    failure_code=None,
                    work_item_definition_hash=definition.definition_hash,
                    verifier_id=definition.verifier_id,
                )
            )

    live.dispatcher.submit(
        SimpleNamespace(
            command_kind="factor_research",
            admit=lambda: CommandAdmission(task.task_id, "QUEUED"),
            execute=execute,
        )
    )
    live.dispatcher.drain_for_tests()
    live.activity.drain_wakes()
    (wake,) = registry.wake_registrations(task.task_id)
    return wake


def _restart(live) -> None:  # type: ignore[no-untyped-def]
    live.stop()
    live.start()
    live.activity.drain_wakes()


def test_the_host_sends_a_wake_once_after_the_cli_has_gone(live, codex) -> None:
    """Registered Tasks send one wake when they end or need a decision after the CLI exits."""
    task = _task(live, "ended")
    wake = _register(live, task)
    assert codex.calls() == []
    assert _run(live, task)["result"] == {
        "channel": "codex-queue",
        "delivered": True,
        "command": str(codex.command),
        "command_source": "CLIENT_PATH",
    }
    ((*queue, message),) = codex.calls()
    assert queue == ["queue", "--thread", THREAD, "--message"]
    # The line names the Task and what happened before the command that reads it.
    assert message.startswith("Host: ") and "finished" in message
    assert message.endswith(wake["read_command"])
    assert wake["read_command"].endswith(f"task show {task.task_id}")
    live.activity.command_returned("factor_research", task.task_id, None)
    live.activity.drain_wakes()
    assert len(codex.calls()) == 1

    stopped = _task(live, "decision")
    _register(live, stopped)
    assert _run(live, stopped, decision=True)["event"] == "NEEDS_DECISION"
    assert len(codex.calls()) == 2 and "needs a decision" in codex.calls()[-1][-1]
    # A stopped Task moves only on a request: a wake on it is refused, with its way on.
    sent = Request(
        operation="WAKE_REGISTER", task_id=stopped.task_id, wake_thread=THREAD, wake_read="read"
    )
    refused = live.operations.execute(sent)
    assert refused["failure_code"] == "task_control.wake_task_stopped:RECOVERY_REQUIRED"
    assert refused["next_requests"]["read"]["task_id"] == str(stopped.task_id)


def test_a_held_wake_outlives_a_restart_and_an_interrupted_send_reads_uncertain(
    live, codex
) -> None:
    """requirement (WAKE ruling 1 and 4, board 16:32): a pending wake survives a Host restart;
    a Task that ended while no hook saw it is sent by the start's replay; a send the Host did
    not finish is named uncertain at the next start and never sent again."""
    task = _task(live, "restart")
    _register(live, task)
    _restart(live)
    assert codex.calls() == []
    assert _run(live, task)["state"] == "DELIVERED"
    _restart(live)
    assert len(codex.calls()) == 1

    registry = live.session.task_control_registry
    unseen = _task(live, "unseen")
    _register(live, unseen)
    registry.request_cancel(
        task_id=unseen.task_id, expected_task_hash=unseen.record_hash, observed_at=live.clock()
    )  # straight to the registry: no hook sees it
    live.activity.drain_wakes()
    assert len(codex.calls()) == 1
    _restart(live)
    assert "stopped" in codex.calls()[-1][-1]  # a cancelled Task ended, not finished

    interrupted = _task(live, "interrupted")
    held = _register(live, interrupted)
    registry.mark_recovery_required(
        task_id=interrupted.task_id,
        failure_code="TASK_EXECUTION_INTERRUPTED",
        observed_at=live.clock(),
    )
    assert registry.claim_wake(
        held["registration_id"],
        interrupted.task_id,
        event="NEEDS_DECISION",
        lifecycle="RECOVERY_REQUIRED",
        observed_at=live.clock(),
    )
    _restart(live)
    page = LocalResearchClient(live.workspace).activity(watch=(str(interrupted.task_id),))
    (wake,) = page["tasks"][str(interrupted.task_id)]["wakes"]
    assert (wake["state"], wake["result"]["failure"]) == (
        "UNDELIVERED",
        "CODEX_WAKE_DELIVERY_UNCERTAIN",
    )
    assert wake["next_action"] == "READ_THE_SAME_TASK"
    assert len(codex.calls()) == 2


def test_a_wake_the_host_cannot_send_is_named_in_the_tasks_activity_and_not_retried(
    live, codex, monkeypatch, tmp_path
) -> None:
    """A command lost after admission names the delivery failure and never retries it."""
    empty = tmp_path / "empty"
    empty.mkdir()
    for salt, path, failure in (
        ("missing", empty, "CODEX_COMMAND_MISSING"),
        ("failed", codex.folder, "CODEX_QUEUE_FAILED"),
    ):
        monkeypatch.setenv("PATH", str(codex.folder))
        task = _task(live, salt)
        _register(live, task)
        monkeypatch.setenv("PATH", str(path))
        monkeypatch.setenv("FAKE_CODEX_EXIT", "17")
        parked = codex.command.with_suffix(".held")
        if salt == "missing":
            codex.command.rename(parked)
        wake = _run(live, task)
        if salt == "missing":
            parked.rename(codex.command)
        assert (wake["state"], wake["result"]["failure"]) == ("UNDELIVERED", failure)
        live.activity.command_returned("factor_research", task.task_id, None)
        live.activity.drain_wakes()
        page = LocalResearchClient(live.workspace).activity(watch=(str(task.task_id),))
        assert any(
            row["task_id"] == str(task.task_id)
            and row["payload"].get("disposition") == "WAKE_UNDELIVERED"
            and row["payload"].get("failure_code") == failure
            for row in page["items"]
        )
    assert len(codex.calls()) == 1


def test_a_missing_queue_refuses_before_the_host_holds_a_wake(live, codex, monkeypatch):
    """Missing background wakes refuse admission and offer an in-turn wait."""
    monkeypatch.setenv("PATH", "")
    task = _task(live, "queue-unavailable")
    client = LocalResearchClient(live.workspace)
    assert client.activity()["observer"]["codex_queue"]["present"] is False
    registration = {
        "operation": "WAKE_REGISTER",
        "task_id": str(task.task_id),
        "wake_thread": THREAD,
        "wake_read": "task show",
    }
    answer = client.request(registration)
    assert answer["failure_code"] == "local_client.codex_queue_unavailable"
    assert answer["next_action"] == "WAIT_IN_THE_TURN" and answer["detail"]
    assert client.request(answer["next_requests"]["read"])["task_id"] == str(task.task_id)
    assert live.session.task_control_registry.wake_registrations(task.task_id) == ()
    assert codex.calls() == []
    monkeypatch.setenv("FAKE_CODEX_PROBE_EXIT", "17")
    refused = client.request({**registration, "wake_codex_path": str(codex.command)})
    assert refused["failure_code"] == "local_client.codex_queue_unavailable"
    assert live.session.task_control_registry.wake_registrations(task.task_id) == ()
    monkeypatch.delenv("FAKE_CODEX_PROBE_EXIT")
    # The same Host receives the fresh client's PATH; it never restarts.
    wake = _register(live, task, env={**os.environ, "PATH": str(codex.folder)})
    assert wake["codex_path"] == str(codex.command)
    delivered = _run(live, task)["result"]
    assert delivered == {
        "channel": "codex-queue",
        "delivered": True,
        "command": str(codex.command),
        "command_source": "CLIENT_PATH",
    }
    assert len(codex.calls()) == 1 and len(codex.probes()) == 2


def test_concurrent_reads_probe_the_queue_once_per_host(live, codex):
    """Concurrent admission and health reads share one Host-scoped queue probe."""
    with ThreadPoolExecutor(max_workers=4) as pool:
        views = list(pool.map(lambda _: LocalResearchClient(live.workspace).activity(), range(8)))
    assert all(view["observer"]["codex_queue"]["present"] for view in views)
    task = _task(live, "probe-once")
    _register(live, task)
    assert len(codex.probes()) == 1 and codex.calls() == []
    _restart(live)
    LocalResearchClient(live.workspace).activity()
    assert len(codex.probes()) == 2


def _cli(arguments: list[str], capsys) -> dict:  # type: ignore[no-untyped-def,type-arg]
    """The answer's envelope, its `data` the owner's answer."""
    main(arguments, serve=lambda _args: pytest.fail("a wake uses the running Host"))
    return json.loads(capsys.readouterr().out)


def test_a_goal_wake_is_refused_and_offers_each_unfinished_task_its_own(
    live, codex, monkeypatch, capsys
) -> None:
    """requirement (WAKE ruling 2): a wake follows a Task. `--goal` with `--notify` is refused
    by name and offers each of the goal's unfinished Tasks its registration; `--each-stage`
    never wakes."""
    running = _task(live, "goal-running")
    monkeypatch.setenv("ALPHALATTICE_SHELL", "posix")
    original = LocalResearchClient.request

    def request(client, document=None, **options):  # type: ignore[no-untyped-def]
        if document and document["operation"] == "GOAL_NARRATIVE":
            tasks = [{"task_id": str(running.task_id), "state": "RUNNING"}]
            return {"record": {"tasks": [*tasks, {"task_id": str(uuid4()), "state": "SUCCEEDED"}]}}
        return original(client, document, **options)

    monkeypatch.setattr(LocalResearchClient, "request", request)
    wait = ["--workspace", str(live.workspace), "--view", "full", "activity", "wait"]
    goal = str(uuid4())
    refused = _cli([*wait, "--goal", goal, "--notify", "codex-queue"], capsys)["data"]
    assert refused["failure_code"] == "local_client.codex_notify_needs_task"
    offered = refused["next_requests"]
    assert set(offered) == {f"wake:{running.task_id}", "goal"}
    assert offered[f"wake:{running.task_id}"]["wake_thread"] == THREAD

    # The offer is sent as any other next request.
    sent = original(LocalResearchClient(live.workspace), offered[f"wake:{running.task_id}"])
    assert sent["status"] == "WAKE_REGISTERED" and sent["task_id"] == str(running.task_id)
    each = _cli(
        [*wait, "--task", str(running.task_id), "--each-stage", "--notify", "codex-queue"], capsys
    )
    assert each["failure_code"] == "local_client.each_stage_never_wakes"
    assert codex.calls() == []


_VERB_ARGUMENTS = {
    "strategy-book review": ("--package", "installed-book"),
    "review continue": (),
    "first-use prepare": ("--sentence", "Build me a book.", "--date", "2026-10-09"),
    "strategy build": (),
}
"""Each agent verb's own arguments beside its folder, where it takes one."""


@pytest.mark.parametrize("verb", sorted(_VERB_ARGUMENTS))
def test_an_agent_verb_registers_its_running_task_and_its_rerun_keeps_every_answer(
    live, codex, tmp_path, monkeypatch, capsys, verb
) -> None:
    """An agent verb registers its running task and its rerun keeps every answer."""
    task = _task(live, "verb")
    monkeypatch.setenv("ALPHALATTICE_SHELL", "posix")
    original = LocalResearchClient.request
    running = {"status": "ADMITTED", "task_id": str(task.task_id), "lifecycle": "RUNNING"}
    planned = {"operation": "WORKSPACE_PREPARE_CONFIRM", "preparation_plan_hash": "p"}
    trained = {"operation": "MODEL_TRAINING_INPUT_PREPARE", "experiment_plan_hash": "p"}
    route = {"operation": "MODEL_TRAINING_INPUT_PLAN", "component_id": "G2"}
    answers = {
        "ACTIVITY_LIST": {"observer": {"codex_queue": {"present": False}}},
        "CONTROLS": {"status": "CONTROLS", "template": {}},
        "RUN": running,
        "AGENT_ANSWER_SUBMIT": {**running, "status": "ACCEPTED"},
        "GOAL_OPEN": {"status": "REFUSED", "failure_code": "goal.first_use_after_preparation"},
        "DATA_ISSUES": {"issues": []},
        "WORKSPACE_PREPARE_PLAN": {"status": "PLANNED", "next_requests": {"confirm": planned}},
        "WORKSPACE_PREPARE_CONFIRM": running,
        "RESEARCH_STRATEGY_CONTROLS": {"next_requests": {"component:G2": route}},
        "MODEL_TRAINING_INPUT_PLAN": {"status": "PLANNED", "next_requests": {"prepare": trained}},
        "MODEL_TRAINING_INPUT_PREPARE": running,
    }

    def request(client, document=None, **options):  # type: ignore[no-untyped-def]
        if document and document["operation"] in answers:
            return answers[document["operation"]]
        return original(client, document, **options)

    monkeypatch.setattr(LocalResearchClient, "request", request)
    folder = tmp_path / "answer bundles"
    folder.mkdir()
    (folder / "answer.json").write_text("{}\n", encoding="utf-8")
    output = tmp_path / "receipt.json"
    line = [
        *("--workspace", str(live.workspace), "--view", "full", "--lang", "zh"),
        *verb.split(),
        *(("--dir", str(folder)) if verb.endswith(("review", "continue")) else ()),
        *_VERB_ARGUMENTS[verb],
        *("--notify", "codex-queue", "--max-wait", "900", "--output", str(output)),
    ]
    answer = _cli(line, capsys)["data"]
    assert answer["status"] == "WAKE_REGISTERED" and answer["task_id"] == str(task.task_id)
    saved = output.read_bytes()
    (wake,) = live.session.task_control_registry.wake_registrations(task.task_id)
    rerun = shlex.split(wake["read_command"])[len(command_prefix()) :]
    assert rerun[rerun.index("--output") + 1] == str(tmp_path.resolve() / "receipt.wake1.json")
    assert {"--lang", "zh", "--notify", "codex-queue", "--max-wait", *verb.split()} <= set(rerun)

    _cli(rerun, capsys)  # the real parser takes the Host's command back
    assert output.read_bytes() == saved
    assert json.loads((tmp_path / "receipt.wake1.json").read_text("utf-8"))["task_id"] == str(
        task.task_id
    )
    (again,) = live.session.task_control_registry.wake_registrations(task.task_id)
    assert again["registration_id"] == wake["registration_id"]
    following = shlex.split(again["read_command"])
    assert following[following.index("--output") + 1] == str(
        tmp_path.resolve() / "receipt.wake2.json"
    )
    assert codex.calls() == []


@pytest.mark.parametrize("verb", sorted(_VERB_ARGUMENTS))
def test_an_agent_verb_refuses_unavailable_wakes_before_any_work(
    live, codex, monkeypatch, capsys, tmp_path, verb
):
    """An unavailable wake stops an agent verb before it changes the workspace."""
    folder = tmp_path / "answer bundles"
    folder.mkdir()
    (folder / "answer.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("PATH", "")
    sent = []
    original = LocalResearchClient.request

    def request(client, document=None, **options):
        sent.append(document["operation"])
        return original(client, document, **options)

    monkeypatch.setattr(LocalResearchClient, "request", request)
    line = [
        "--workspace",
        str(live.workspace),
        "--view",
        "full",
        *verb.split(),
        *(("--dir", str(folder)) if verb.endswith(("review", "continue")) else ()),
        *_VERB_ARGUMENTS[verb],
        "--notify",
        "codex-queue",
    ]
    answer = _cli(line, capsys)
    assert answer["failure_code"] == "local_client.codex_queue_unavailable"
    assert answer["next_action"] == "WAIT_IN_THE_TURN" and answer["detail"]
    assert sent == ["ACTIVITY_LIST"] and codex.calls() == []
