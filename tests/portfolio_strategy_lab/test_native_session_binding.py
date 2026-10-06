"""Prospective native attachment metadata is written by the served workspace Host."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

import pytest

from alphalattice.interface.local_application import cli
from alphalattice.interface.local_application.client import LocalResearchClient
from alphalattice.interface.local_application.native_bridge import (
    BINDING_NAME,
    NativeResearchBinding,
    coordination_event,
    deliver,
)
from alphalattice.interface.local_application.native_observation_sequence import SEQUENCE_NAME
from alphalattice.interface.local_application.native_setup import bind_session, files_unavailable
from tests.portfolio_strategy_lab.local_web_support import _json

ROOT = Path(__file__).resolve().parents[2]


def _lead_usage_file(tmp_path, monkeypatch):
    """A labelled synthetic host Session file, never a supplied usage claim."""
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "synthetic-claude"))
    path = tmp_path / "synthetic-claude/projects/fixture-project/fixture-parent.jsonl"
    path.parent.mkdir(parents=True)
    record = {
        "type": "assistant",
        "timestamp": datetime.now(UTC).isoformat(),
        "message": {
            "id": "synthetic-response-1",
            "model": "claude-sonnet-5-5",
            "role": "assistant",
            "content": [{"type": "text", "text": "SECRET-SYNTHETIC-CONVERSATION"}],
            "usage": {
                "input_tokens": 7,
                "output_tokens": 3,
                "cache_read_input_tokens": 2,
                "cache_creation_input_tokens": 1,
            },
        },
    }
    path.write_bytes((json.dumps(record) + "\n").encode())
    return path


def test_lead_usage_is_read_and_sequenced_by_the_host_without_a_child_stop(
    live,
    tmp_path,
    monkeypatch,
):
    """P2: only the admitted parent asks the Host to read its own usage; retry is idempotent.

    The agent cannot write bridge metadata, provide counts or publish conversation content.
    This live transport proves the reader separately from SubagentStop.
    """
    project = _project(live, "claude-code")
    _session(monkeypatch, "claude-code")
    _lead_usage_file(tmp_path, monkeypatch)
    client = LocalResearchClient(live.workspace)
    assert client.bind_native_session(project)["status"] == "BOUND_NOT_ATTACHED"
    opening = Path.open
    calling_thread = threading.get_ident()

    def host_only_metadata(path, mode="r", *args, **kwargs):
        if path.parent == project / ".codex" and any(flag in mode for flag in "wxa+"):
            assert threading.get_ident() != calling_thread
        return opening(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", host_only_metadata)
    first = client.publish_native_event(project, {"source": "native_usage_read"})
    second = client.publish_native_event(project, {"source": "native_usage_read"})
    assert first["status"] == second["status"] == "DELIVERED"
    items = client.read_external()["items"]
    assert len(items) == 1
    payload = items[0]["payload"]
    assert payload["event_kind"] == "NATIVE_AGENT_USAGE"
    subject = payload["subject"]
    assert subject["native_agent_id"] == subject["native_session_id"] == "fixture-parent"
    assert subject["role"] == "research_lead"
    assert (subject["input_tokens"], subject["output_tokens"], subject["responses"]) == (
        "7",
        "3",
        "1",
    )
    assert "SECRET-SYNTHETIC-CONVERSATION" not in json.dumps(items)


@pytest.mark.parametrize("problem", ("counts", "path", "child", "missing_file", "off"))
def test_lead_usage_request_accepts_no_claims_and_keeps_missing_usage_visible(
    live,
    tmp_path,
    monkeypatch,
    problem,
):
    """P2: the narrow read door names absence and admits neither supplied counts nor child ids."""
    project = _project(live, "claude-code")
    _session(monkeypatch, "claude-code")
    _lead_usage_file(tmp_path, monkeypatch)
    client = LocalResearchClient(live.workspace)
    assert (
        client.bind_native_session(project, usage="off" if problem == "off" else "read")["status"]
        == "BOUND_NOT_ATTACHED"
    )
    event = {"source": "native_usage_read"}
    if problem == "counts":
        event["input_tokens"] = 999
    elif problem == "path":
        event["transcript_path"] = "SECRET-PATH"
    elif problem == "child":
        _session(monkeypatch, "claude-code", "synthetic-child")
    elif problem == "missing_file":
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "absent-host"))
    answer = client.publish_native_event(project, event)
    if problem == "missing_file":
        assert answer == {
            "status": "UNAVAILABLE",
            "reason": "native_bridge.lead_usage_file_missing",
        }
    elif problem == "off":
        assert answer == {"status": "SKIPPED", "reason": "native_bridge.usage_disabled"}
    else:
        assert answer["status"] == "REFUSED"
    _session(monkeypatch, "claude-code")
    assert client.read_external()["items"] == []
    assert "SECRET-PATH" not in json.dumps(answer)


def _project(live, host: str) -> Path:
    project = live.workspace.parent
    if host == "codex":
        directory, declaration = ".codex", "config.toml"
    else:
        directory, declaration = ".claude", "settings.json"
        cards = project / directory / "agents"
        cards.mkdir(parents=True)
        card = "alphalattice_cro.md"
        (cards / card).write_bytes((ROOT / directory / "agents" / card).read_bytes())
    target = project / directory
    target.mkdir(exist_ok=True)
    (target / declaration).write_bytes((ROOT / directory / declaration).read_bytes())
    return project


def _session(monkeypatch, host: str, session: str | None = "fixture-parent") -> None:
    monkeypatch.delenv("CODEX_THREAD_ID", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_SESSION_ID", raising=False)
    if session is not None:
        monkeypatch.setenv(
            "CODEX_THREAD_ID" if host == "codex" else "CLAUDE_CODE_SESSION_ID", session
        )


@pytest.mark.parametrize("host", ("codex", "claude-code"))
def test_session_bind_uses_the_host_writer_and_never_claims_attachment(
    live, monkeypatch, capsys, host
):
    project = _project(live, host)
    _session(monkeypatch, host)
    monkeypatch.chdir(live.workspace)
    binding_path = project / ".codex" / BINDING_NAME
    opening = Path.open
    calling_thread = threading.get_ident()

    def agent_cannot_write(path, mode="r", *args, **kwargs):
        if path == binding_path and "x" in mode and threading.get_ident() == calling_thread:
            raise PermissionError(13, "fixture sandbox denial", str(path))
        return opening(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", agent_cannot_write)
    assert (
        cli.main(
            ["--lang", "zh", "--workspace", str(live.workspace), "session", "bind"],
            serve=lambda _: 99,
        )
        == 0
    )
    answer = json.loads(capsys.readouterr().out)
    assert answer["status"] == "BOUND_NOT_ATTACHED"
    assert "本地声明和绑定不能确认" in answer["detail"]
    assert answer["data"]["project"] == str(project)
    assert answer["data"]["host_trust"] == "NOT_CHECKED"
    assert answer["data"]["foreground_attachment"] == "NOT_PROVED"
    assert answer["data"]["attachment_preflight"]["failure_code"] == (
        "native_bridge.readiness_incomplete"
    )
    assert "complete_fresh_native_chain" in answer["data"]["attachment_preflight"]["missing"]
    assert (
        "actual_runtime_definitions_and_trust" in answer["data"]["attachment_preflight"]["missing"]
    )
    assert (
        "prospective_observation_checkpoint"
        not in answer["data"]["attachment_preflight"]["missing"]
    )
    binding = NativeResearchBinding.read(project)
    assert (binding.host, binding.session_id, binding.workspace) == (
        host,
        "fixture-parent",
        live.workspace,
    )
    assert binding.observation_started_at.tzinfo is not None
    before = binding_path.read_bytes()
    client = LocalResearchClient(live.workspace)
    assert client.bind_native_session(project)["status"] == "BOUND_NOT_ATTACHED"
    assert binding_path.read_bytes() == before
    _session(monkeypatch, host, "another-session")
    assert client.bind_native_session(project)["failure_code"] == (
        "native_bridge.existing_configuration_differs"
    )
    assert binding_path.read_bytes() == before
    assert _json(live, "/api/activity/external")["items"] == []


@pytest.mark.parametrize(
    "problem",
    ("outside", "symlink", "declaration_symlink", "payload_session", "no_session"),
)
def test_host_bind_refuses_unadmitted_paths_and_session_fields(live, monkeypatch, problem):
    project = _project(live, "codex")
    _session(monkeypatch, "codex")
    payload = {"project": str(project)}
    expected = "native_bridge.project_mismatch"
    if problem == "outside":
        payload["project"] = str(project.parent / "other-project")
    elif problem in {"symlink", "declaration_symlink"}:
        is_symlink = Path.is_symlink
        alias = project if problem == "symlink" else project / ".codex/config.toml"
        monkeypatch.setattr(Path, "is_symlink", lambda path: path == alias or is_symlink(path))
        expected = (
            "native_bridge.project_path_invalid"
            if problem == "symlink"
            else "native_bridge.configuration_path_invalid"
        )
    elif problem == "payload_session":
        payload["session_id"] = "invented"
        expected = "native_bridge.request_invalid"
    else:
        _session(monkeypatch, "codex", None)
        expected = "local_client.session_unnamed"
    answer, _ = LocalResearchClient(live.workspace)._exchange("/api/client/session/bind", payload)
    assert answer["status"] == "REFUSED"
    assert answer["failure_code"] == expected
    assert not (project / ".codex" / BINDING_NAME).exists()


def test_native_binding_write_failure_names_only_its_safe_path(live, monkeypatch):
    project = _project(live, "codex")
    _session(monkeypatch, "codex")
    binding_path = project / ".codex" / BINDING_NAME
    opening = Path.open

    def unavailable(path, mode="r", *args, **kwargs):
        if path == binding_path and "x" in mode:
            raise PermissionError(13, "secret fixture exception text", str(path))
        return opening(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", unavailable)
    answer = LocalResearchClient(live.workspace).bind_native_session(project)
    assert answer["failure_code"] == "native_bridge.files_unavailable:PermissionError"
    assert answer["path_category"] == "PROJECT_BINDING"
    assert answer["path"] == str(binding_path)
    assert "secret fixture exception text" not in json.dumps(answer)
    assert answer["next_action"] == "USE_THE_WORKSPACE_HOST_OR_CHECK_THE_NAMED_LOCAL_PATH"
    outside = PermissionError(13, "secret", str(project.parent / "outside-private-file"))
    assert "path" not in files_unavailable(outside, project=project)


@pytest.mark.parametrize("host", ("codex", "claude-code"))
def test_explicit_native_message_uses_host_sequence_and_keeps_declared_provenance(
    live, monkeypatch, host
):
    project = _project(live, host)
    _session(monkeypatch, host)
    client = LocalResearchClient(live.workspace)
    assert client.bind_native_session(project)["status"] == "BOUND_NOT_ATTACHED"
    binding = NativeResearchBinding.read(project)
    sequence_path = project / ".codex" / SEQUENCE_NAME
    opening = Path.open
    calling_thread = threading.get_ident()

    def agent_cannot_write(path, mode="r", *args, **kwargs):
        if path.parent == project / ".codex" and any(flag in mode for flag in "wxa+"):
            assert threading.get_ident() != calling_thread
        return opening(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", agent_cannot_write)
    connect = sqlite3.connect

    def host_connect(database, *args, **kwargs):
        if str(database) == str(sequence_path):
            assert threading.get_ident() != calling_thread
        return connect(database, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", host_connect)
    assigned = coordination_event(
        binding, kind="assignment", message=b"Read the fixture packet.", recipient_id="child"
    )
    first = deliver(project, binding, assigned, host_owned=True)
    assert first["status"] == "DELIVERED"
    assert deliver(project, binding, assigned, host_owned=True) == first
    answered = coordination_event(
        binding,
        agent_id="child",
        role="alphalattice_cro",
        kind="answer",
        message=b"The fixture has no findings.",
        reply_to=str(assigned["message_id"]),
    )
    assert deliver(project, binding, answered, host_owned=True)["status"] == "DELIVERED"
    assert sequence_path.is_file()
    rows = _json(live, "/api/activity/external")["items"]
    assert len(rows) == 2
    assert {row["payload"]["subject"]["native_host"] for row in rows} == {host}
    assert {row["payload"]["subject"]["input_channel"] for row in rows} == {"ACTOR_DECLARED"}
    assert {row["payload"]["event_kind"] for row in rows} == {"NATIVE_COORDINATION_MESSAGE"}
    assert all(row["authority"] == "AGENT_PROPOSAL" for row in rows)


def test_host_native_message_refuses_other_workspace_and_unselected_fields(live, monkeypatch):
    project = _project(live, "codex")
    _session(monkeypatch, "codex")
    other = project / "other-workspace"
    other.mkdir()
    bind_session(project, host="codex", session_id="fixture-parent", workspace=other)
    binding = NativeResearchBinding.read(project)
    event = coordination_event(binding, kind="plan", message=b"A scoped fixture plan.")
    client = LocalResearchClient(live.workspace)
    assert client.publish_native_event(project, event)["failure_code"] == (
        "native_bridge.workspace_mismatch"
    )
    assert not (project / ".codex" / SEQUENCE_NAME).exists()
    (project / ".codex" / BINDING_NAME).unlink()
    assert client.bind_native_session(project)["status"] == "BOUND_NOT_ATTACHED"
    assert (
        client.publish_native_event(project, {**event, "transcript": "not selected"})[
            "failure_code"
        ]
        == "native_bridge.event_invalid"
    )
    assert not (project / ".codex" / SEQUENCE_NAME).exists()
    refused = client.publish_native_event(project.parent / "outside-project", event)
    assert refused["failure_code"] == "native_bridge.project_mismatch"
    assert _json(live, "/api/activity/external")["items"] == []
