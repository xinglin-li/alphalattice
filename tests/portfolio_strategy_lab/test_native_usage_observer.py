"""Automatic usage through real Host owners; fixture files confer no native hook credit."""

from __future__ import annotations

import inspect
import io
import json
import threading
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from alphalattice.control.product_host.composition.application_session import (
    WorkspaceApplicationSession,
)
from alphalattice.control.product_host.composition.entry import serve
from alphalattice.control.product_host.composition.local_web_session import NativeUsageObserver
from alphalattice.control.workspace_runtime.lock import WorkspaceLock
from alphalattice.interface.local_application import cli, native_setup
from alphalattice.interface.local_application.cli_contract import refusal_words
from alphalattice.interface.local_application.client import LocalResearchClient
from alphalattice.interface.local_application.native_bridge import (
    NativeResearchBinding,
    coordination_event,
)
from alphalattice.interface.local_application.native_observation_sequence import LOCK_NAME
from alphalattice.interface.local_application.native_setup import declare_project
from tests.workspace_task_runner.task_control_support import task_contract

ROLE = "alphalattice_risk"
SECRET = "SYNTHETIC-PRIVATE-NATIVE-TEXT-AND-ERROR"


def _response(name="first", *, input_tokens=20, output_tokens=3, host="codex"):
    if host == "claude-code":
        return [
            {
                "type": "assistant",
                "timestamp": datetime.now(UTC).isoformat(),
                "effort": "high",
                "message": {
                    "id": name,
                    "model": "synthetic-model",
                    "content": SECRET,
                    "usage": {
                        "input_tokens": input_tokens - 5,
                        "cache_read_input_tokens": 5,
                        "cache_creation_input_tokens": 0,
                        "output_tokens": output_tokens,
                    },
                },
            }
        ]
    return [
        {"type": "turn_context", "payload": {"model": "synthetic-model", "effort": "high"}},
        {
            "type": "token_usage_record",
            "timestamp": datetime.now(UTC).isoformat(),
            "payload": {
                "response_id": name,
                "usage": {
                    "input_tokens": input_tokens,
                    "cached_input_tokens": 5,
                    "cache_write_input_tokens": 0,
                    "output_tokens": output_tokens,
                },
            },
        },
    ]


def _rollout(scene, thread, *, parent=None, role=ROLE, suffix="a", responses=None, agent_path=None):
    """Write labelled native-file fixtures, never supplied hook or author observations."""
    if scene.host == "claude-code":
        directory = scene.native / "projects/fixture-project"
        path = (
            directory / scene.parent / "subagents" / f"agent-{thread}.jsonl"
            if parent is not None
            else directory / f"{thread}.jsonl"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "type": "user",
                "sessionId": parent or thread,
                "agentId": thread,
                "isSidechain": parent is not None,
                "message": {"content": SECRET},
            },
            *(responses if responses is not None else _response(host=scene.host)),
        ]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        if parent is not None:
            path.with_suffix(".meta.json").write_text(
                json.dumps(
                    {
                        "agentType": role,
                        "spawnDepth": 1,
                        "toolUseId": "synthetic-tool-use",
                        "description": SECRET,
                    }
                ),
                encoding="utf-8",
            )
        return path
    meta = {"id": thread, "cwd": str(scene.project), "base_instructions": SECRET}
    if parent is not None:
        meta["source"] = {
            "subagent": {"thread_spawn": {"parent_thread_id": parent, "agent_role": role}}
        }
        if agent_path is not None:
            meta["source"]["subagent"]["thread_spawn"]["agent_path"] = agent_path
    path = scene.native / "sessions/2026/10/06" / f"rollout-{suffix}-{thread}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"type": "session_meta", "payload": meta},
        {"type": "response_item", "payload": {"type": "message", "content": SECRET}},
        *(responses if responses is not None else _response()),
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


@pytest.fixture
def scene(live, tmp_path, monkeypatch, request):
    """Real Host/Goal/observation owners, with only the optional timer driven separately."""
    assert live.native_usage_observer is not None
    assert live.native_usage_observer.close()
    options = request.param if hasattr(request, "param") and isinstance(request.param, dict) else {}
    host = options.get("host", "codex")
    project = live.workspace.parent
    directory, declaration = (
        (".codex", "config.toml") if host == "codex" else (".claude", "settings.json")
    )
    target = project / directory
    target.mkdir(exist_ok=True)
    shipped = Path(__file__).resolve().parents[2] / directory / declaration
    (target / declaration).write_bytes(shipped.read_bytes())
    declare_project(project, host)
    cards = project / (".codex/agents" if host == "codex" else ".claude/agents")
    cards.mkdir(exist_ok=True)
    card = cards / f"{ROLE}.{'toml' if host == 'codex' else 'md'}"
    card.write_text(
        'model = "synthetic-model"\nmodel_reasoning_effort = "high"\n'
        if host == "codex"
        else "---\nname: alphalattice_risk\nmodel: synthetic-model\n"
        "effort: high\n---\nSynthetic card.\n"
    )
    parent = str(uuid4())
    monkeypatch.delenv("CODEX_THREAD_ID", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_SESSION_ID", raising=False)
    monkeypatch.setenv("CODEX_THREAD_ID" if host == "codex" else "CLAUDE_CODE_SESSION_ID", parent)
    native = tmp_path / "synthetic-native-home"
    monkeypatch.setenv("CODEX_HOME", str(native))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(native))
    monkeypatch.chdir(project)
    client = LocalResearchClient(live.workspace)
    usage = options.get("usage", "read") if options else getattr(request, "param", "read")
    assert client.bind_native_session(project, usage=usage)["status"] == "BOUND"
    observer = NativeUsageObserver(live.operations, live.activity)
    live.native_usage_observer = observer
    result = SimpleNamespace(
        live=live,
        client=client,
        project=project,
        native=native,
        host=host,
        parent=parent,
        observer=observer,
        card=card,
    )
    result.path = _rollout(result, parent)
    yield result


def _rows(scene):
    return [
        row
        for row in scene.client.read_external()["items"]
        if (row.get("payload") or {}).get("event_kind") == "NATIVE_AGENT_USAGE"
    ]


def _wait(read, predicate, *, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = read()
        if predicate(value):
            return value
        time.sleep(0.02)
    raise AssertionError("The bounded observer fixture did not reach its expected state.")


def _open_goal(scene, capsys):
    """Use the public CLI for declarations; takes use its real transport without usage callbacks."""
    path = scene.project / "synthetic-goal.json"
    path.write_text(
        json.dumps(
            {
                "operation": "GOAL_OPEN",
                "goal_id": str(uuid4()),
                "change_reason": "Synthetic observer fixture",
                "goal_declaration": {
                    "title": "Synthetic observer scope",
                    "objective": "Observe exact cumulative usage.",
                    "kind": "RESEARCH",
                    "criteria": [{"criterion_id": "usage", "text": "Read usage."}],
                    "deliverables": [
                        {"deliverable_id": "result", "kind": "RESULT", "description": "Receipt."}
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    assert (
        cli.main(
            [
                "--workspace",
                str(scene.live.workspace),
                "--view",
                "full",
                "request",
                "--file",
                str(path),
            ],
            serve=serve,
        )
        == 0
    )
    answer = json.loads(capsys.readouterr().out)
    return answer["data"]["goal_id"]


def _take(scene, goal_id):
    # cli.main's existing GOAL_TAKE usage callback would hide a broken observer Goal cache.
    answer = scene.client.request({"operation": "GOAL_TAKE", "goal_id": goal_id})
    assert answer["status"] == "GOAL_TAKEN"


def _assign(scene, child, *, message_id=None):
    binding = NativeResearchBinding.read(scene.project, session=(scene.host, scene.parent))
    assert binding is not None
    event = coordination_event(
        binding,
        kind="assignment",
        message=b"Synthetic usage assignment.",
        recipient_id=child,
        message_id=message_id,
    )
    assert scene.client.publish_native_event(scene.project, event)["status"] == "DELIVERED"


def _cycle(scene):
    """Drive the actual timer callback without replacing any owner or publisher."""
    scene.observer._cycle()


@contextmanager
def _source(scene, monkeypatch):
    """Select a labelled fixture's actual shell source without replacing any owner."""
    with monkeypatch.context() as selected:
        selected.delenv("CODEX_THREAD_ID", raising=False)
        selected.delenv("CLAUDE_CODE_SESSION_ID", raising=False)
        selected.setenv(
            "CODEX_THREAD_ID" if scene.host == "codex" else "CLAUDE_CODE_SESSION_ID",
            scene.parent,
        )
        yield


def _other_scene(scene, monkeypatch, *, host=None, parent=None, usage="read"):
    """Bind another real public Host source, preserving the first compatibility record."""
    other = SimpleNamespace(**vars(scene))
    other.host, other.parent = host or scene.host, parent or str(uuid4())
    directory, declaration = (
        (".codex", "config.toml") if other.host == "codex" else (".claude", "settings.json")
    )
    target = scene.project / directory
    target.mkdir(exist_ok=True)
    if not (target / declaration).exists():
        shipped = Path(__file__).resolve().parents[2] / directory / declaration
        (target / declaration).write_bytes(shipped.read_bytes())
    declare_project(scene.project, other.host)
    cards = target / "agents"
    cards.mkdir(exist_ok=True)
    other.card = cards / f"{ROLE}.{'toml' if other.host == 'codex' else 'md'}"
    if not other.card.exists():
        other.card.write_text(
            'model = "synthetic-model"\nmodel_reasoning_effort = "high"\n'
            if other.host == "codex"
            else "---\nname: alphalattice_risk\nmodel: synthetic-model\neffort: high\n---\n",
            encoding="utf-8",
        )
    with _source(other, monkeypatch):
        assert other.client.bind_native_session(scene.project, usage=usage)["status"] == "BOUND"
    other.path = _rollout(other, other.parent)
    return other


def _binding_path(scene):
    binding = NativeResearchBinding.read(scene.project, session=(scene.host, scene.parent))
    assert binding is not None
    return binding.record_path(scene.project)


@pytest.mark.parametrize("scene", [{"host": "codex"}, {"host": "claude-code"}], indirect=True)
def test_two_sessions_keep_owner_goals_usage_cache_and_unbind_independent(
    scene, monkeypatch, capsys
):
    first_path = _binding_path(scene)
    first_bytes = first_path.read_bytes()
    other = _other_scene(scene, monkeypatch)
    other_path, other_bytes = _binding_path(other), _binding_path(other).read_bytes()
    assert other_path == NativeResearchBinding.slot_path(scene.project, (other.host, other.parent))
    assert first_path.read_bytes() == first_bytes
    goals = []
    for owner in (scene, other):
        with _source(owner, monkeypatch):
            goal = _open_goal(owner, capsys)
            _take(owner, goal)
            child = str(uuid4())
            _rollout(owner, child, parent=owner.parent)
            _assign(owner, child)
            goals.append((owner.parent, child, goal))
    _cycle(scene)
    rows = _rows(scene)
    assert len(rows) == 4
    assert {
        (
            r["payload"]["subject"]["native_session_id"],
            r["payload"]["subject"]["native_agent_id"],
            r["payload"]["subject"]["goal_id"],
        )
        for r in rows
    } == {(parent, agent, goal) for parent, child, goal in goals for agent in (parent, child)}
    _cycle(scene)
    assert _rows(scene) == rows
    with _source(scene, monkeypatch):
        assert cli.main(["session", "unbind"], serve=serve) == 0
    assert not first_path.exists() and other_path.read_bytes() == other_bytes
    for owner in (scene, other):
        with owner.path.open("a", encoding="utf-8") as stream:
            stream.write(
                "".join(json.dumps(r) + "\n" for r in _response("growth", host=owner.host))
            )
    _cycle(scene)
    updated = _rows(scene)
    assert len(updated) == 5 and all(row in updated for row in rows)
    latest = max(updated, key=lambda row: row["ordinal"])["payload"]["subject"]
    assert (latest["native_session_id"], latest["goal_id"], latest["responses"]) == (
        other.parent,
        goals[1][2],
        "2",
    )
    _cycle(scene)
    assert _rows(scene) == updated and other_path.read_bytes() == other_bytes
    assert SECRET not in json.dumps(updated)


@pytest.mark.parametrize("scene", [{"host": "codex"}, {"host": "claude-code"}], indirect=True)
def test_same_session_text_on_other_host_has_its_own_goal_counts_and_cache(
    scene, monkeypatch, capsys
):
    other = _other_scene(
        scene,
        monkeypatch,
        host="claude-code" if scene.host == "codex" else "codex",
        parent=scene.parent,
    )
    other.path = _rollout(
        other, other.parent, responses=_response(output_tokens=7, host=other.host)
    )
    expected = set()
    for owner, count in ((scene, "3"), (other, "7")):
        with _source(owner, monkeypatch):
            goal = _open_goal(owner, capsys)
            _take(owner, goal)
            expected.add((owner.host, owner.parent, goal, count))
    before = {_binding_path(owner): _binding_path(owner).read_bytes() for owner in (scene, other)}
    _cycle(scene)
    rows = _rows(scene)
    assert len(rows) == 2 and len({row["source_id"] for row in rows}) == 2
    assert {
        (
            r["payload"]["subject"]["native_host"],
            r["payload"]["subject"]["native_session_id"],
            r["payload"]["subject"]["goal_id"],
            r["payload"]["subject"]["output_tokens"],
        )
        for r in rows
    } == expected
    _cycle(scene)
    assert _rows(scene) == rows
    assert all(path.read_bytes() == payload for path, payload in before.items())
    assert all("authorship_basis" not in row["payload"]["subject"] for row in rows)


@pytest.mark.parametrize("scene", ["off", {"host": "claude-code", "usage": "off"}], indirect=True)
def test_off_source_is_not_discovered_when_another_session_is_read(scene, monkeypatch):
    other = _other_scene(scene, monkeypatch)
    child = str(uuid4())
    child_path = _rollout(scene, child, parent=scene.parent)
    _assign(scene, child)
    opening, globbing = Path.open, Path.glob
    accessed = []

    def checked_open(path, *args, **kwargs):
        if path == scene.path or path == child_path or path == child_path.with_suffix(".meta.json"):
            accessed.append("open")
            pytest.fail("Another READ Session enabled this OFF source.")
        return opening(path, *args, **kwargs)

    def checked_glob(path, pattern, *args, **kwargs):
        if path.is_relative_to(scene.native) and any(
            key in pattern for key in (scene.parent, child)
        ):
            accessed.append("glob")
            pytest.fail("Another READ Session discovered this OFF source.")
        return globbing(path, pattern, *args, **kwargs)

    monkeypatch.setattr(Path, "open", checked_open)
    monkeypatch.setattr(Path, "glob", checked_glob)
    _cycle(scene)
    (row,) = _rows(scene)
    assert accessed == []
    assert row["payload"]["subject"]["native_session_id"] == other.parent
    assert scene.client.activity()["observer"]["native_usage"] == {
        "status": "OBSERVING",
        "reason": None,
    }


@pytest.mark.parametrize("scene", [{"host": "codex"}, {"host": "claude-code"}], indirect=True)
def test_bad_slot_does_not_hide_good_session_and_recovers_without_duplicate_usage(
    scene, monkeypatch
):
    other = _other_scene(scene, monkeypatch)
    path = _binding_path(other)
    original = path.read_bytes()
    first_bytes = _binding_path(scene).read_bytes()
    path.write_bytes(b"not-json " + SECRET.encode())
    _cycle(scene)
    (first,) = _rows(scene)
    assert first["payload"]["subject"]["native_session_id"] == scene.parent
    health = scene.client.activity()["observer"]["native_usage"]
    _assert_health(health, "native_bridge.binding_invalid")
    assert SECRET not in json.dumps(health)
    path.write_bytes(original)
    _cycle(scene)
    rows = _rows(scene)
    assert len(rows) == 2 and first in rows
    assert {r["payload"]["subject"]["native_session_id"] for r in rows} == {
        scene.parent,
        other.parent,
    }
    _cycle(scene)
    assert _rows(scene) == rows and _binding_path(scene).read_bytes() == first_bytes


@pytest.mark.parametrize("channel", ["http", "cli"])
def test_admitted_codex_child_message_keeps_parent_binding_goal_and_declared_actor(
    scene, monkeypatch, capsys, channel
):
    goal = _open_goal(scene, capsys)
    _take(scene, goal)
    child = str(uuid4())
    _rollout(scene, child, parent=scene.parent)
    _assign(scene, child)
    binding = NativeResearchBinding.read(scene.project, session=(scene.host, scene.parent))
    assert binding is not None
    before = binding.record_path(scene.project).read_bytes()
    child_source = SimpleNamespace(host="codex", parent=child)
    with _source(child_source, monkeypatch):
        if channel == "http":
            event = coordination_event(
                binding,
                kind="answer",
                message=b"Synthetic child answer.",
                agent_id=child,
                role=ROLE,
            )
            answer = scene.client.publish_native_event(scene.project, event)
        else:
            monkeypatch.setattr(native_setup, "ROOT", scene.project)
            monkeypatch.setattr(
                native_setup.sys,
                "argv",
                [
                    "native_research.py",
                    "message",
                    "--kind",
                    "answer",
                    "--agent-id",
                    child,
                    "--role",
                    ROLE,
                ],
            )
            monkeypatch.setattr(
                native_setup.sys,
                "stdin",
                SimpleNamespace(buffer=io.BytesIO(b"Synthetic child answer.")),
            )
            assert native_setup.main() == 0
            answer = json.loads(capsys.readouterr().out)
    assert answer["status"] == "DELIVERED"
    rows = [
        r
        for r in scene.client.read_external()["items"]
        if (r.get("payload") or {}).get("subject", {}).get("message_kind") == "answer"
    ]
    (row,) = rows
    subject = row["payload"]["subject"]
    assert (
        subject["native_session_id"],
        subject["native_agent_id"],
        subject["role"],
        subject["goal_id"],
    ) == (scene.parent, child, ROLE, goal)
    assert subject["input_channel"] == "ACTOR_DECLARED" and "authorship_basis" not in subject
    assert binding.record_path(scene.project).read_bytes() == before
    assert NativeResearchBinding.read(scene.project, session=("codex", child)) is None
    assert not any(
        "HOOK" in (r.get("payload") or {}).get("event_kind", "")
        for r in scene.client.read_external()["items"]
    )


@pytest.mark.parametrize("scene", [{"host": "codex"}, {"host": "claude-code"}], indirect=True)
def test_background_lead_arrives_without_hooks_and_stops_with_the_host(scene):
    """A retained background worker publishes source counts without a CLI usage request."""
    scene.observer.start()
    (row,) = _wait(lambda: _rows(scene), lambda rows: len(rows) == 1)
    subject = row["payload"]["subject"]
    assert (subject["native_agent_id"], subject["role"]) == (scene.parent, "research_lead")
    assert (subject["responses"], subject["input_tokens"], subject["output_tokens"]) == (
        "1",
        "15",
        "3",
    )
    assert (
        subject["input_channel"]
        == ("CODEX_SESSION_FILE" if scene.host == "codex" else "CLAUDE_CODE_SESSION_FILE")
        and row["authority"] == "AGENT_PROPOSAL"
    )
    assert "authorship_basis" not in subject
    assert SECRET not in json.dumps(row) and str(scene.path) not in json.dumps(row)
    assert {r["payload"]["event_kind"] for r in scene.client.read_external()["items"]} == {
        "NATIVE_AGENT_USAGE"
    }
    with scene.path.open("a", encoding="utf-8") as stream:
        stream.write(
            "".join(json.dumps(item) + "\n" for item in _response("later", host=scene.host))
        )
    rows = _wait(lambda: _rows(scene), lambda items: len(items) == 2)
    later = max(rows, key=lambda item: item["ordinal"])
    assert (
        later["payload"]["subject"]["responses"],
        later["payload"]["subject"]["output_tokens"],
    ) == ("2", "6")
    scene.live.stop()
    assert scene.observer.state()["status"] == "STOPPED"
    assert scene.live.native_usage_observer is None and scene.live.activity is None
    with WorkspaceApplicationSession.acquire(scene.live.workspace):
        pass


@pytest.mark.parametrize("scene", ["off", {"host": "claude-code", "usage": "off"}], indirect=True)
def test_off_never_discovers_opens_or_publishes_native_files(scene, monkeypatch):
    """Privacy is checked before native glob, header, count reads and publication."""
    opening, globbing = Path.open, Path.glob
    accessed = []

    def opening_checked(path, *args, **kwargs):
        if path.is_relative_to(scene.native):
            accessed.append("open")
            pytest.fail("OFF accessed native content.")
        return opening(path, *args, **kwargs)

    def globbing_checked(path, *args, **kwargs):
        if path.is_relative_to(scene.native):
            accessed.append("glob")
            pytest.fail("OFF discovered native files.")
        return globbing(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", opening_checked)
    monkeypatch.setattr(Path, "glob", globbing_checked)
    _cycle(scene)
    assert accessed == [] and _rows(scene) == []
    assert scene.client.activity()["observer"]["native_usage"] == {"status": "OFF", "reason": None}


@pytest.mark.parametrize(
    "scope", ["exact", "unassigned", "other_parent", "other_role", "alias", "old_binding"]
)
@pytest.mark.parametrize("scene", [{"host": "codex"}, {"host": "claude-code"}], indirect=True)
def test_child_requires_exact_assignment_and_actual_direct_parent_role(scene, scope):
    """An explicit assignment selects scope; first-record metadata proves the child tuple."""
    child = str(uuid4())
    _rollout(
        scene,
        child,
        parent=str(uuid4()) if scope == "other_parent" else scene.parent,
        role="unrelated_role" if scope == "other_role" else ROLE,
    )
    if scope != "unassigned":
        _assign(scene, "/root/named_risk" if scope == "alias" else child)
    if scope == "old_binding":
        assert (
            cli.main(["--workspace", str(scene.live.workspace), "session", "unbind"], serve=serve)
            == 0
        )
        assert scene.client.bind_native_session(scene.project)["status"] == "BOUND"
    _cycle(scene)
    rows = _rows(scene)
    assert {row["payload"]["subject"]["native_agent_id"] for row in rows} == (
        {scene.parent, child} if scope == "exact" else {scene.parent}
    )
    assert all("authorship_basis" not in row["payload"]["subject"] for row in rows)
    assert all(
        "HOOK" not in row["payload"]["event_kind"] for row in scene.client.read_external()["items"]
    )
    assert SECRET not in json.dumps(rows)


@pytest.mark.parametrize("scene", [{"host": "codex"}, {"host": "claude-code"}], indirect=True)
def test_retry_growth_replacement_and_role_pin_keep_latest_cumulative_snapshots(scene, capsys):
    """Unchanged reads retry exactly; changed sources and pins publish replacement snapshots."""
    goal_id = _open_goal(scene, capsys)
    _take(scene, goal_id)
    child = str(uuid4())
    _rollout(scene, child, parent=scene.parent)
    _assign(scene, child)
    _cycle(scene)
    first = _rows(scene)
    _cycle(scene)
    assert _rows(scene) == first and len(first) == 2
    with scene.path.open("a", encoding="utf-8") as stream:
        stream.write(
            "".join(
                json.dumps(row) + "\n"
                for row in _response("second", input_tokens=30, output_tokens=4, host=scene.host)
            )
        )
    _cycle(scene)
    grown = _rows(scene)
    assert len(grown) == 3
    latest = max(
        (row for row in grown if row["payload"]["subject"]["native_agent_id"] == scene.parent),
        key=lambda row: row["ordinal"],
    )
    assert (
        latest["payload"]["subject"]["responses"],
        latest["payload"]["subject"]["output_tokens"],
    ) == ("2", "7")
    scene.path = _rollout(
        scene,
        scene.parent,
        suffix="z",
        responses=_response("replacement", output_tokens=9, host=scene.host),
    )
    _cycle(scene)
    assert len(_rows(scene)) == 4
    scene.card.write_text(
        'model = "different-synthetic-model"\nmodel_reasoning_effort = "high"\n'
        if scene.host == "codex"
        else "---\nname: alphalattice_risk\nmodel: different-synthetic-model\n"
        "effort: high\n---\nSynthetic card.\n"
    )
    _cycle(scene)
    rows = _rows(scene)
    assert len(rows) == 5
    latest_child = max(
        (row for row in rows if row["payload"]["subject"]["native_agent_id"] == child),
        key=lambda row: row["ordinal"],
    )
    assert latest_child["payload"]["subject"]["pin_differs"] == "model"
    child_subject = latest_child["payload"]["subject"]
    assert len(child_subject) == 16
    assert child_subject["goal_id"] == goal_id and child_subject["efforts"] == "high"
    assert child_subject["input_channel"] == (
        "CODEX_SESSION_FILE" if scene.host == "codex" else "CLAUDE_CODE_SESSION_FILE"
    )
    assert child_subject["sample_time_kind"] == "LATEST_USAGE_RECORD_AT"
    assert child_subject["last_at"]
    assert "native_event_id" not in child_subject and "source_kind" not in child_subject
    _cycle(scene)
    assert _rows(scene) == rows
    record = scene.client.request({"operation": "GOAL_SHOW", "goal_id": goal_id})
    usages = record["record"]["session_usage"]
    assert usages[0]["aggregation"] == "NOT_COMBINED" and "by_model" not in usages[0]
    individual = {p["agent_id"]: p["models"][0] for p in usages[0]["participants"]}
    assert individual[scene.parent]["output_tokens"] == 9
    assert individual[child]["output_tokens"] == 3
    assert individual[child]["source_kind"] == child_subject["input_channel"]
    assert individual[child]["sample_time_kind"] == child_subject["sample_time_kind"]
    assert individual[child]["last_at"] == child_subject["last_at"]
    assert SECRET not in json.dumps(rows)


@pytest.mark.parametrize("scene", [{"host": "codex"}, {"host": "claude-code"}], indirect=True)
def test_exact_same_role_children_are_independent_of_parent_read_failure(scene, monkeypatch):
    """Exact recipients each retain own counts; a failed parent does not block either child."""
    children = [str(uuid4()), str(uuid4())]
    for child in [*children, str(uuid4())]:
        _rollout(scene, child, parent=scene.parent)
    for child in children:
        _assign(scene, child)
    opening = Path.open

    def unavailable(path, mode="r", *args, **kwargs):
        if path == scene.path and mode == "rb":
            raise OSError(SECRET)
        return opening(path, mode, *args, **kwargs)

    with monkeypatch.context() as failure:
        failure.setattr(Path, "open", unavailable)
        _cycle(scene)
        health = scene.client.activity()["observer"]["native_usage"]
        _assert_health(health, "native_bridge.lead_usage_read_failed")
    rows = _rows(scene)
    assert {r["payload"]["subject"]["native_agent_id"] for r in rows} == set(children)
    assert all(r["payload"]["subject"]["output_tokens"] == "3" for r in rows)
    _cycle(scene)
    recovered = _rows(scene)
    assert len(recovered) == 3 and all(row in recovered for row in rows)
    assert {r["payload"]["subject"]["native_agent_id"] for r in recovered} == {
        scene.parent,
        *children,
    }
    _cycle(scene)
    assert _rows(scene) == recovered
    assert SECRET not in json.dumps(rows) and SECRET not in json.dumps(health)


@pytest.mark.parametrize("scene", [{"host": "codex"}, {"host": "claude-code"}], indirect=True)
@pytest.mark.parametrize("participant", ["lead", "child"])
def test_incomplete_source_keeps_the_previous_snapshot_without_publishing_a_prefix(
    scene, participant
):
    """Malformed native counts stay visible as unavailable until a complete source recovers."""
    child = str(uuid4())
    path = scene.path
    if participant == "child":
        path = _rollout(scene, child, parent=scene.parent)
        _assign(scene, child)
    _cycle(scene)
    before = _rows(scene)
    original = path.read_bytes()
    with path.open("ab") as stream:
        stream.write(
            b'{"type":"assistant","message":{"usage":'
            if scene.host == "claude-code"
            else b'{"type":"token_usage_record","payload":'
        )
    _cycle(scene)
    assert _rows(scene) == before
    health = scene.client.activity()["observer"]["native_usage"]
    _assert_health(health, f"native_bridge.{participant}_usage_incomplete")
    path.write_bytes(
        original
        + "".join(
            json.dumps(r) + "\n" for r in _response("recovered", output_tokens=5, host=scene.host)
        ).encode()
    )
    _cycle(scene)
    recovered = _rows(scene)
    assert len(recovered) == len(before) + 1 and all(row in recovered for row in before)
    latest = max(recovered, key=lambda row: row["ordinal"])["payload"]["subject"]
    assert latest["native_agent_id"] == (scene.parent if participant == "lead" else child)
    assert (latest["responses"], latest["output_tokens"]) == ("2", "8")
    assert SECRET not in json.dumps(health)


@pytest.mark.parametrize("scene", [{"host": "claude-code"}], indirect=True)
def test_changed_claude_role_metadata_invalidates_the_unchanged_count_cache(scene):
    """A cached count source cannot hide a changed or unavailable exact child association."""
    child = "exact-claude-child"
    path = _rollout(scene, child, parent=scene.parent)
    _assign(scene, child)
    _cycle(scene)
    before = _rows(scene)
    assert len(before) == 2
    metadata = path.with_suffix(".meta.json")
    original = metadata.read_bytes()
    changed = json.loads(original)
    changed["agentType"] = "unrelated_role"
    metadata.write_text(json.dumps(changed), encoding="utf-8")
    _cycle(scene)
    assert _rows(scene) == before
    _assert_health(
        scene.client.activity()["observer"]["native_usage"],
        "native_bridge.child_usage_binding_unverified",
    )
    metadata.write_bytes(original)
    _cycle(scene)
    assert _rows(scene) == before
    assert scene.client.activity()["observer"]["native_usage"] == {
        "status": "OBSERVING",
        "reason": None,
    }


@pytest.mark.parametrize("scene", [{"host": "codex"}, {"host": "claude-code"}], indirect=True)
def test_changed_owner_goal_refreshes_unchanged_files_and_preserves_old_filing(scene, capsys):
    """A new owner Goal invalidates the successful stat cache without changing the old row."""
    first_goal, second_goal = _open_goal(scene, capsys), _open_goal(scene, capsys)
    _take(scene, first_goal)
    _cycle(scene)
    (first,) = _rows(scene)
    _take(scene, second_goal)
    _cycle(scene)
    rows = _rows(scene)
    assert len(rows) == 2 and first in rows
    assert {row["payload"]["subject"]["goal_id"] for row in rows} == {first_goal, second_goal}
    _cycle(scene)
    assert _rows(scene) == rows


@pytest.mark.parametrize("scene", [{"host": "codex"}, {"host": "claude-code"}], indirect=True)
def test_exact_child_assignment_does_not_follow_the_lead_to_another_goal(scene, capsys):
    """An unchanged child source needs a fresh exact assignment under the owner's new Goal."""
    first_goal, second_goal = _open_goal(scene, capsys), _open_goal(scene, capsys)
    child = str(uuid4())
    _rollout(scene, child, parent=scene.parent)
    _take(scene, first_goal)
    _assign(scene, child)
    _cycle(scene)
    first = _rows(scene)
    assert len(first) == 2
    (child_first,) = [r for r in first if r["payload"]["subject"]["native_agent_id"] == child]
    assert child_first["payload"]["subject"]["goal_id"] == first_goal
    _take(scene, second_goal)
    _cycle(scene)
    switched = _rows(scene)
    assert len(switched) == 3 and all(row in switched for row in first)
    assert [r for r in switched if r["payload"]["subject"]["native_agent_id"] == child] == [
        child_first
    ]
    _assign(scene, child)
    _cycle(scene)
    # The same declared message is an exact retry and retains its original Goal filing.
    assert _rows(scene) == switched
    _assign(scene, child, message_id="synthetic-fresh-goal-assignment")
    _cycle(scene)
    current = _rows(scene)
    child_rows = [r for r in current if r["payload"]["subject"]["native_agent_id"] == child]
    assert len(current) == 4 and child_first in child_rows
    assert {r["payload"]["subject"]["goal_id"] for r in child_rows} == {first_goal, second_goal}
    assert all(r["payload"]["subject"]["output_tokens"] == "3" for r in child_rows)
    _cycle(scene)
    assert _rows(scene) == current


def test_real_sequence_lock_refusal_remains_visible_and_retries_after_release(scene):
    """A busy real producer lock yields no cached success or invented observation receipt."""
    with WorkspaceLock(scene.project / ".codex" / LOCK_NAME):
        _cycle(scene)
        assert _rows(scene) == []
        health = scene.client.activity()["observer"]["native_usage"]
        _assert_health(health, "native_bridge.observation_busy")
    _cycle(scene)
    rows = _rows(scene)
    assert len(rows) == 1
    _cycle(scene)
    assert _rows(scene) == rows
    assert SECRET not in json.dumps(health)


def _read_barrier(scene, monkeypatch):
    """Delay only the real native file's first open, keeping every owner and parser unchanged."""
    entered, release = threading.Event(), threading.Event()
    opening = Path.open

    @contextmanager
    def held(path, mode, *args, **kwargs):
        with opening(path, mode, *args, **kwargs) as stream:
            entered.set()
            assert release.wait(5), "The fixture must release its native-file read."
            yield stream

    def intercept(path, mode="r", *args, **kwargs):
        if path == scene.path and mode == "rb" and not release.is_set():
            return held(path, mode, *args, **kwargs)
        return opening(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", intercept)
    return entered, release


@pytest.mark.parametrize("scene", [{"host": "codex"}, {"host": "claude-code"}], indirect=True)
def test_goal_changed_during_real_read_refuses_old_snapshot_then_recovers(
    scene, capsys, monkeypatch
):
    """The publishing owner rechecks the Goal resolved before reading native counts."""
    first_goal, second_goal = _open_goal(scene, capsys), _open_goal(scene, capsys)
    _take(scene, first_goal)
    entered, release = _read_barrier(scene, monkeypatch)
    scene.observer.start()
    try:
        assert entered.wait(5)
        _take(scene, second_goal)
        release.set()
        _wait(
            scene.observer.state,
            lambda state: state["reason"] == "native_bridge.event_scope_invalid",
        )
        assert _rows(scene) == []
        (row,) = _wait(lambda: _rows(scene), lambda rows: len(rows) == 1)
        assert row["payload"]["subject"]["goal_id"] == second_goal
    finally:
        release.set()
        assert scene.observer.close()


def test_bounded_stop_keeps_real_workspace_lease_until_native_read_finishes(scene, monkeypatch):
    """A live observer joins before ledger close and lease release, even after refused publish."""
    from alphalattice.control.product_host.composition.local_web_session import LocalWebSessionError

    entered, release = _read_barrier(scene, monkeypatch)
    scene.observer.start()
    try:
        assert entered.wait(5)
        with pytest.raises(LocalWebSessionError, match=r"local_web_session\.writer_still_live"):
            scene.live.stop(timeout=0)
        assert scene.live.session is not None and scene.live.activity is not None
        with pytest.raises(RuntimeError, match=r"^workspace runtime writer is already owned$"):
            WorkspaceApplicationSession.acquire(scene.live.workspace)
    finally:
        release.set()
        scene.live.stop()
    with WorkspaceApplicationSession.acquire(scene.live.workspace):
        pass


def test_constructor_filesystem_failure_is_safe_and_does_not_block_host(scene, monkeypatch):
    """A thin filesystem fault exercises optional construction; owners and responses stay real."""
    scene.live.stop()
    resolving = Path.resolve

    def resolve(path, *args, **kwargs):
        frame = inspect.currentframe()
        caller = frame.f_back if frame is not None else None
        constructing = caller is not None and isinstance(
            caller.f_locals.get("self"), NativeUsageObserver
        )
        if path == scene.live.workspace and constructing and caller.f_code.co_name == "__init__":
            raise OSError(SECRET)
        return resolving(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)
    scene.live.start()
    client = LocalResearchClient(scene.live.workspace)
    health = client.activity()["observer"]["native_usage"]
    _assert_health(health, "native_bridge.lead_usage_read_failed")
    workspace = client.request({"operation": "WORKSPACE_SHOW"})
    assert workspace["workspace_id"] == scene.live.workspace_id
    assert workspace["workspace_manifest_hash"] == scene.live.workspace_manifest.manifest_hash
    assert "failure_code" not in workspace
    assert SECRET not in json.dumps(health)


def _assert_health(health, code):
    assert health["status"] == "UNAVAILABLE" and health["reason"] == code
    words = refusal_words(code)
    assert health["detail"] == words["detail"]
    assert health["next_action"] == words["next_action"]
    assert set(health) == {"status", "reason", "detail", "next_action"}


@pytest.mark.parametrize(
    "role",
    [
        "alphalattice_data",
        "alphalattice_factor",
        "alphalattice_alpha",
        "alphalattice_risk",
        "alphalattice_portfolio",
        "alphalattice_evidence_analyst",
        "alphalattice_cro",
    ],
)
def test_codex_canonical_child_counts_grow_and_refuse_a_changed_identity(scene, capsys, role):
    """A native tool path is a count-file association, never hook or authorship proof."""
    card = scene.project / ".codex/agents" / (role + ".toml")
    card.write_text(
        'model = "synthetic-model"\nmodel_reasoning_effort = "high"\n', encoding="utf-8"
    )
    goal = _open_goal(scene, capsys)
    _take(scene, goal)
    child, task = str(uuid4()), "/root/synthetic_exact_child"
    path = _rollout(scene, child, parent=scene.parent, role=role, agent_path=task)
    _assign(scene, task)
    _cycle(scene)
    rows = _rows(scene)
    assert {row["payload"]["subject"]["native_agent_id"] for row in rows} == {scene.parent, task}
    first = next(row for row in rows if row["payload"]["subject"]["native_agent_id"] == task)
    assert first["payload"]["subject"]["role"] == role
    assert first["payload"]["subject"]["goal_id"] == goal
    assert first["payload"]["subject"]["input_channel"] == "CODEX_SESSION_FILE"
    assert "authorship_basis" not in first["payload"]["subject"]
    with path.open("a", encoding="utf-8") as stream:
        stream.write("".join(json.dumps(row) + "\n" for row in _response("later")))
    _cycle(scene)
    later = [row for row in _rows(scene) if row["payload"]["subject"]["native_agent_id"] == task]
    assert len(later) == 2
    assert later[-1]["payload"]["subject"]["responses"] == "2"
    assert later[-1]["payload"]["subject"]["output_tokens"] == "6"
    before = _rows(scene)
    # Replace the same named file with another exact native parent: the previous cache
    # cannot keep its path/identity association alive merely because the name stayed.
    _rollout(scene, child, parent=str(uuid4()), role=role, agent_path=task)
    _cycle(scene)
    assert _rows(scene) == before
    _assert_health(
        scene.client.activity()["observer"]["native_usage"],
        "native_bridge.child_usage_binding_unverified",
    )
    assert SECRET not in json.dumps(before)
    assert not any(
        "HOOK" in row["payload"]["event_kind"] for row in scene.client.read_external()["items"]
    )


def _retained_assignment(scene, child, *, reference=None):
    """Declare through the real public owner and retain its exact original observed row."""
    binding = NativeResearchBinding.read(scene.project, session=(scene.host, scene.parent))
    assert binding is not None
    receipt = scene.client.publish_native_event(
        scene.project,
        coordination_event(
            binding,
            kind="assignment",
            message=b"Read the retained synthetic child context.",
            recipient_id=child,
            reference=reference,
            message_id="synthetic-retained-before-rebind",
        ),
    )
    assert receipt["status"] == "DELIVERED"
    (observed,) = [
        row
        for row in scene.client.read_external()["items"]
        if row["observation_id"] == receipt["observation_id"]
    ]
    return receipt, observed


def _read_rebind(scene, capsys, goal, original_assignment):
    """A real same-Session unbind/rebind advances only the native observation checkpoint."""
    assert (
        cli.main(["--workspace", str(scene.live.workspace), "session", "unbind"], serve=serve) == 0
    )
    capsys.readouterr()
    assert scene.client.bind_native_session(scene.project, usage="read")["status"] == "BOUND"
    binding = NativeResearchBinding.read(scene.project, session=(scene.host, scene.parent))
    assert binding is not None and binding.usage == "READ"
    occurred = datetime.fromisoformat(original_assignment["occurred_at"])
    assert occurred.utcoffset() is not None
    assert binding.observation_started_at is not None
    assert occurred < binding.observation_started_at
    _take(scene, goal)
    retained = next(
        row
        for row in scene.client.read_external()["items"]
        if row["observation_id"] == original_assignment["observation_id"]
    )
    assert retained == original_assignment
    return binding


def _child_usage_rows(scene, child):
    return [row for row in _rows(scene) if row["payload"]["subject"]["native_agent_id"] == child]


@pytest.mark.parametrize(
    "role",
    [
        "alphalattice_data",
        "alphalattice_factor",
        "alphalattice_alpha",
        "alphalattice_risk",
        "alphalattice_portfolio",
        "alphalattice_evidence_analyst",
        "alphalattice_cro",
    ],
)
def test_background_closed_child_assignment_survives_same_goal_read_rebind(scene, capsys, role):
    """BEHAVIOUR: an exact closed Goal assignment still selects independent growing usage."""
    card = scene.project / ".codex/agents" / (role + ".toml")
    card.write_text(
        'model = "synthetic-model"\nmodel_reasoning_effort = "high"\n', encoding="utf-8"
    )
    goal = _open_goal(scene, capsys)
    _take(scene, goal)
    child, canonical = str(uuid4()), "/root/synthetic_rebound_child"
    path = _rollout(scene, child, parent=scene.parent, role=role, agent_path=canonical)
    receipt, original_assignment = _retained_assignment(scene, canonical)
    assignment_message_id = original_assignment["payload"]["subject"]["message_id"]
    assert original_assignment["payload"]["subject"]["goal_id"] == goal
    binding = NativeResearchBinding.read(scene.project, session=(scene.host, scene.parent))
    assert binding is not None
    closed = scene.client.publish_native_event(
        scene.project,
        coordination_event(
            binding,
            kind="decision",
            reply_to=assignment_message_id,
            message=b"The synthetic lead resolves this retained dispatch.",
            terminal_decision="COMPLETED",
            terminal_reason="Retain the original assignment; closing it is not a native Stop.",
        ),
    )
    assert closed["status"] == "DELIVERED"
    before = scene.client.request({"operation": "GOAL_SHOW", "goal_id": goal})["record"]
    assert before["open_assignments"] == []
    original_entries = [
        entry
        for entry in before["conversation"]
        if entry["observation_id"] in {receipt["observation_id"], closed["observation_id"]}
    ]
    assert len(original_entries) == 2
    terminal = next(e for e in original_entries if e["observation_id"] == closed["observation_id"])
    assert terminal["closure_source"] == "LEAD_TERMINAL_DECISION"
    assert terminal["assignment_packet_hash"] == receipt["packet_hash"]
    _read_rebind(scene, capsys, goal, original_assignment)
    scene.observer.start()
    try:
        (first,) = _wait(lambda: _child_usage_rows(scene, canonical), lambda rows: len(rows) == 1)
        subject = first["payload"]["subject"]
        assert (subject["native_session_id"], subject["native_agent_id"], subject["goal_id"]) == (
            scene.parent,
            canonical,
            goal,
        )
        assert (subject["role"], subject["model"], subject["efforts"]) == (
            role,
            "synthetic-model",
            "high",
        )
        assert (subject["responses"], subject["input_tokens"], subject["output_tokens"]) == (
            "1",
            "15",
            "3",
        )
        assert subject["input_channel"] == "CODEX_SESSION_FILE"
        with path.open("a", encoding="utf-8") as stream:
            stream.write(
                "".join(
                    json.dumps(row) + "\n"
                    for row in _response("later", input_tokens=30, output_tokens=4)
                )
            )
        rows = _wait(lambda: _child_usage_rows(scene, canonical), lambda items: len(items) == 2)
        later = max(rows, key=lambda row: row["ordinal"])
        assert first in rows and later["ordinal"] > first["ordinal"]
        subject = later["payload"]["subject"]
        assert (
            subject["responses"],
            subject["input_tokens"],
            subject["output_tokens"],
            subject["cache_read_tokens"],
            subject["cache_write_tokens"],
        ) == ("2", "40", "7", "10", "0")
        assert subject["goal_id"] == goal and subject["native_agent_id"] == canonical
        assert scene.observer.state()["status"] == "OBSERVING"
        after = scene.client.request({"operation": "GOAL_SHOW", "goal_id": goal})["record"]
        assert after["open_assignments"] == []
        assert [
            entry
            for entry in after["conversation"]
            if entry["observation_id"] in {receipt["observation_id"], closed["observation_id"]}
        ] == original_entries
        assert all("authorship_basis" not in row["payload"]["subject"] for row in rows)
        assert not any(
            "HOOK" in row["payload"]["event_kind"] for row in scene.client.read_external()["items"]
        )
        assert SECRET not in json.dumps(rows) and str(path) not in json.dumps(rows)
    finally:
        assert scene.observer.close()


def test_background_accepted_child_rebind_preserves_original_bundle_answer_context(scene, capsys):
    """CONTRACT: public accepted closure survives READ rebind without recapturing its basis."""
    goal = _open_goal(scene, capsys)
    _take(scene, goal)
    envelope, task_goal, plan = task_contract(salt="observer-accepted-child-read-rebind")
    task = scene.live.session.task_control_registry.admit(
        input_envelope=envelope,
        goal=task_goal,
        plan=plan,
        observed_at=datetime(2026, 10, 6, tzinfo=UTC),
    ).record
    directory = scene.project / "synthetic-rebound-risk-bundle"
    assert (
        cli.main(
            [
                "--workspace",
                str(scene.live.workspace),
                "bundle",
                "prepare",
                "--role",
                "RISK",
                "--task",
                str(task.task_id),
                "--dir",
                str(directory),
            ],
            serve=serve,
        )
        == 0
    )
    prepared = json.loads(capsys.readouterr().out)["data"]
    assert prepared["status"] == "AGENT_BUNDLE_READY"
    directory = Path(prepared["bundle_directory"])
    bundle_bytes = {
        row["name"]: (directory / row["name"]).read_bytes() for row in prepared["files"]
    }
    child, canonical = str(uuid4()), "/root/synthetic_accepted_rebound_child"
    path = _rollout(scene, child, parent=scene.parent, agent_path=canonical)
    receipt, original_assignment = _retained_assignment(
        scene, canonical, reference=prepared["bundle_reference"]
    )
    assignment_message_id = original_assignment["payload"]["subject"]["message_id"]
    authored = {
        "text": "Synthetic retained Task has no numerical report; preserve its exact reference.",
        "references": [str(task.task_id)],
    }
    request = {
        "operation": "AGENT_ANSWER_SUBMIT",
        "bundle_directory": str(directory),
        "agent_answer": authored,
    }
    accepted = scene.client.request(request)
    assert accepted["status"] == "ACCEPTED"
    assert accepted["recorded_agent"]["basis"] == "NOT_OBSERVED"
    closure = accepted["conversation"]["assignment_closure"]
    assert closure == {
        "status": "CLOSED",
        "source": "PRODUCT_ACCEPTED_ANSWER",
        "reason": "ACCEPTED_ANSWER",
        "message_id": assignment_message_id,
        "packet_hash": receipt["packet_hash"],
        "assigned_agent_id": canonical,
    }
    assert accepted["conversation"]["original_goal_id"] == goal
    assert accepted["conversation"]["original_session_id"] == scene.parent
    assert accepted["conversation"]["original_session_host"] == "codex"
    selected = scene.client.read_external(
        observation_id=accepted["conversation"]["observation_id"]
    )["accepted_answer"]
    assert selected["status"] == "AVAILABLE" and selected["contribution"] == authored
    assert (
        scene.client.request({"operation": "GOAL_SHOW", "goal_id": goal})["record"][
            "open_assignments"
        ]
        == []
    )
    _read_rebind(scene, capsys, goal, original_assignment)
    scene.observer.start()
    try:
        first = _wait(lambda: _child_usage_rows(scene, canonical), lambda rows: len(rows) == 1)
        with path.open("a", encoding="utf-8") as stream:
            stream.write("".join(json.dumps(row) + "\n" for row in _response("later")))
        rows = _wait(lambda: _child_usage_rows(scene, canonical), lambda items: len(items) == 2)
        assert first[0] in rows and rows[-1]["payload"]["subject"]["responses"] == "2"
        assert rows[-1]["payload"]["subject"]["output_tokens"] == "6"
        assert scene.observer.state()["status"] == "OBSERVING"
        retry = scene.client.request(request)
        assert retry["answer_reference"] == accepted["answer_reference"]
        assert retry["conversation"] == accepted["conversation"]
        assert (
            scene.client.read_external(observation_id=accepted["conversation"]["observation_id"])[
                "accepted_answer"
            ]
            == selected
        )
        assert {name: (directory / name).read_bytes() for name in bundle_bytes} == bundle_bytes
        assert (
            next(
                row
                for row in scene.client.read_external()["items"]
                if row["observation_id"] == original_assignment["observation_id"]
            )
            == original_assignment
        )
        assert (
            scene.client.request({"operation": "GOAL_SHOW", "goal_id": goal})["record"][
                "open_assignments"
            ]
            == []
        )
        assert SECRET not in json.dumps(rows)
        assert not any(
            "HOOK" in row["payload"]["event_kind"] for row in scene.client.read_external()["items"]
        )
    finally:
        assert scene.observer.close()


@pytest.mark.parametrize("scope", ["other_goal", "off"])
def test_retained_child_read_rebind_keeps_wrong_goal_and_off_exclusions(scene, capsys, scope):
    """BEHAVIOUR: relaxing age never relaxes the exact current Goal or OFF admission."""
    original_goal = _open_goal(scene, capsys)
    _take(scene, original_goal)
    child, canonical = str(uuid4()), "/root/synthetic_excluded_rebound_child"
    _rollout(scene, child, parent=scene.parent, agent_path=canonical)
    _receipt, original_assignment = _retained_assignment(scene, canonical)
    _read_rebind(scene, capsys, original_goal, original_assignment)
    if scope == "other_goal":
        current_goal = _open_goal(scene, capsys)
        _take(scene, current_goal)
    else:
        assert (
            cli.main(["--workspace", str(scene.live.workspace), "session", "unbind"], serve=serve)
            == 0
        )
        capsys.readouterr()
        assert scene.client.bind_native_session(scene.project, usage="off")["status"] == "BOUND"
    scene.observer.start()
    try:
        if scope == "other_goal":
            _wait(
                lambda: _rows(scene),
                lambda rows: any(
                    row["payload"]["subject"]["native_agent_id"] == scene.parent
                    and row["payload"]["subject"]["goal_id"] == current_goal
                    for row in rows
                ),
            )
        else:
            _wait(scene.observer.state, lambda state: state["status"] == "OFF")
            assert _rows(scene) == []
        assert _child_usage_rows(scene, canonical) == []
        assert (
            next(
                row
                for row in scene.client.read_external()["items"]
                if row["observation_id"] == original_assignment["observation_id"]
            )
            == original_assignment
        )
    finally:
        assert scene.observer.close()
