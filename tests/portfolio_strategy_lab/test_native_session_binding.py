"""Prospective native attachment metadata is written by the served workspace Host."""

from __future__ import annotations

import json
import os
import re
import shlex
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest

from alphalattice.control.product_host.composition.local_web_session import LocalPortfolioWebSession
from alphalattice.interface.local_application import cli
from alphalattice.interface.local_application.cli_contract import client_refusal
from alphalattice.interface.local_application.client import LocalResearchClient
from alphalattice.interface.local_application.native_bridge import (
    BINDING_NAME,
    NativeBridgeError,
    NativeResearchBinding,
)
from alphalattice.interface.local_application.native_setup import (
    PROJECT_DECLARATION_NAME,
    bind_session,
    declare_project,
    files_unavailable,
)
from tests.portfolio_strategy_lab.cli_support import _agent_project, _session_cli
from tests.portfolio_strategy_lab.local_web_support import _json, _manifest, _resolved, _Resolver

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


@pytest.mark.parametrize("problem", ("counts", "path", "child", "missing_file"))
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
        == "BOUND"
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
            "participants": [
                {
                    "agent_id": "fixture-parent",
                    "role": "research_lead",
                    "status": "UNAVAILABLE",
                    "reason": "native_bridge.lead_usage_file_missing",
                }
            ],
            "reason": "native_bridge.lead_usage_file_missing",
        }
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
    declare_project(project, host)
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
        if path.parent == project / ".codex" and any(flag in mode for flag in "wxa+"):
            assert threading.get_ident() != calling_thread
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
    assert answer["status"] == "BOUND"
    assert answer["detail"]
    assert answer["data"]["project"] == str(project)
    # No hook exists to trust or attach: the answer claims neither.
    assert "host_trust" not in answer["data"] and "foreground_attachment" not in answer["data"]
    assert answer["data"]["attachment_preflight"]["failure_code"] is None
    assert answer["data"]["attachment_preflight"]["missing"] == []
    assert "native_proof" not in answer["data"]["attachment_preflight"]
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
    assert client.bind_native_session(project)["status"] == "BOUND"
    assert binding_path.read_bytes() == before
    _session(monkeypatch, host, "another-session")
    assert client.bind_native_session(project)["status"] == "BOUND"
    another = NativeResearchBinding.read(project, session=(host, "another-session"))
    assert another.session_id == "another-session"
    assert another.workspace == binding.workspace
    assert NativeResearchBinding.read(project, session=(host, "fixture-parent")) == binding
    assert binding_path.read_bytes() == before
    assert _json(live, "/api/activity/external")["items"] == []


@pytest.mark.parametrize(
    "problem",
    (
        "outside",
        "symlink",
        "declaration_symlink",
        "marker_symlink",
        "marker_missing",
        "marker_corrupt",
        "marker_wrong_host",
        "payload_session",
        "no_session",
    ),
)
@pytest.mark.parametrize("host", ("codex", "claude-code"))
def test_host_bind_refuses_unadmitted_paths_and_session_fields(live, monkeypatch, problem, host):
    project = _project(live, host)
    _session(monkeypatch, host)
    directory = ".codex" if host == "codex" else ".claude"
    declaration = "config.toml" if host == "codex" else "settings.json"
    marker = project / directory / PROJECT_DECLARATION_NAME
    payload = {"project": str(project)}
    expected = "native_bridge.project_mismatch"
    if problem == "outside":
        payload["project"] = str(project.parent / "other-project")
    elif problem in {"symlink", "declaration_symlink", "marker_symlink"}:
        is_symlink = Path.is_symlink
        alias = (
            project
            if problem == "symlink"
            else marker
            if problem == "marker_symlink"
            else project / directory / declaration
        )
        monkeypatch.setattr(Path, "is_symlink", lambda path: path == alias or is_symlink(path))
        expected = (
            "native_bridge.project_path_invalid"
            if problem == "symlink"
            else "native_bridge.configuration_path_invalid"
        )
    elif problem.startswith("marker_"):
        expected = "native_bridge.configuration_path_invalid"
        if problem == "marker_missing":
            marker.unlink()
            expected = "native_bridge.project_declaration_missing"
        elif problem == "marker_corrupt":
            marker.write_bytes(b"{broken")
        else:
            document = json.loads(marker.read_bytes())
            document["host"] = "claude-code" if host == "codex" else "codex"
            marker.write_text(json.dumps(document), encoding="utf-8")
    elif problem == "payload_session":
        payload["session_id"] = "invented"
        expected = "native_bridge.request_invalid"
    else:
        _session(monkeypatch, host, None)
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
def test_two_named_sessions_keep_their_own_workspace_and_detach_only_their_slot(
    live, monkeypatch, capsys, host
):
    """Two named sessions keep their own workspace and detach only their slot."""
    project = _project(live, host)
    second_workspace = project / "second-workspace"
    second_workspace.mkdir()
    second_host = LocalPortfolioWebSession(
        workspace=second_workspace,
        workspace_manifest=_manifest("qa-native-second"),
        resolver=_Resolver(_resolved()),
    )
    second_host.start()
    try:
        monkeypatch.chdir(project)

        def run(session, *line):
            _session(monkeypatch, host, session)
            code = cli.main(list(line), serve=lambda _: 99)
            return code, json.loads(capsys.readouterr().out)

        assert run("fixture-first", "--workspace", str(live.workspace), "session", "bind")[0] == 0
        first = NativeResearchBinding.read(project, session=(host, "fixture-first"))
        first_before = first.record_path(project).read_bytes()
        assert (
            run("fixture-second", "--workspace", str(second_workspace), "session", "bind")[0] == 0
        )
        second = NativeResearchBinding.read(project, session=(host, "fixture-second"))
        second_before = second.record_path(project).read_bytes()
        assert second.workspace == second_workspace
        assert first.record_path(project).read_bytes() == first_before
        for session, workspace in (
            ("fixture-first", live.workspace),
            ("fixture-second", second_workspace),
        ):
            code, answer = run(session, "task", "list")
            assert (code, answer["outcome"]) == (0, "OK"), answer
            assert answer["context"]["workspace_from"] == "BINDING"
            assert answer["context"]["workspace"] == str(workspace)
        code, removed = run("fixture-first", "session", "unbind")
        assert (code, removed["data"]["status"]) == (0, "DETACHED")
        assert removed["data"]["session_id"] == "fixture-first"
        assert removed["data"]["research_unchanged"] is True
        assert second.record_path(project).read_bytes() == second_before
        assert NativeResearchBinding.read(project, session=(host, "fixture-first")) is None
        code, answer = run("fixture-second", "task", "list")
        assert (code, answer["context"]["workspace"]) == (0, str(second_workspace))
        code, refused = run("fixture-first", "session", "unbind")
        assert (code, refused["failure_code"]) == (
            2,
            "local_client.session_unbind_refused:native_bridge.session_mismatch",
        )
        assert second.record_path(project).read_bytes() == second_before
    finally:
        second_host.stop()


def test_two_hosts_with_the_same_session_text_have_separate_sources_and_detach(
    live, tmp_path, monkeypatch, capsys
):
    """P2-NH: a Session identifier alone grants no other host's binding or public source."""
    project = _project(live, "codex")
    _project(live, "claude-code")
    monkeypatch.chdir(project)
    client = LocalResearchClient(live.workspace)
    session = "fixture-shared-text"
    _usage_files(tmp_path, monkeypatch, session)
    bindings = {}
    for host in ("codex", "claude-code"):
        _session(monkeypatch, host, session)
        assert client.bind_native_session(project)["status"] == "BOUND"
        binding = NativeResearchBinding.read(project, session=(host, session))
        bindings[host] = binding
        assert (binding.host, binding.session_id) == (host, session)
        reading = client.publish_native_event(project, {"source": "native_usage_read"})
        assert reading["status"] == "DELIVERED", reading
    rows = client.read_external()["items"]
    assert len(rows) == 2
    assert {row["source_id"].partition(":")[0] for row in rows} == {
        "codex-native",
        "claude-code-native",
    }
    assert {row["payload"]["subject"]["native_host"] for row in rows} == {"codex", "claude-code"}
    other_before = bindings["claude-code"].record_path(project).read_bytes()
    _session(monkeypatch, "codex", session)
    assert cli.main(["session", "unbind"], serve=lambda _: 99) == 0
    answer = json.loads(capsys.readouterr().out)
    assert answer["data"]["host"] == "codex"
    assert bindings["claude-code"].record_path(project).read_bytes() == other_before
    assert NativeResearchBinding.read(project, session=("codex", session)) is None
    assert (
        NativeResearchBinding.read(project, session=("claude-code", session))
        == bindings["claude-code"]
    )
    assert len(client.read_external()["items"]) == 2


@pytest.mark.parametrize("host", ("codex", "claude-code"))
@pytest.mark.parametrize("changed", ("workspace", "usage", "roles", "workspace_alias"))
def test_exact_session_retry_keeps_checkpoint_and_refuses_changed_scope(
    live, monkeypatch, host, changed
):
    """A new slot does not weaken the same Session's immutable admitted scope."""
    project = _project(live, host)
    _session(monkeypatch, host)
    assert bind_session(project, host=host, session_id="fixture-parent", workspace=live.workspace)
    binding = NativeResearchBinding.read(project, session=(host, "fixture-parent"))
    before = binding.record_path(project).read_bytes()
    assert bind_session(project, host=host, session_id="fixture-parent", workspace=live.workspace)
    assert NativeResearchBinding.read(project, session=(host, "fixture-parent")) == binding
    assert binding.record_path(project).read_bytes() == before
    workspace, usage = live.workspace, "read"
    if changed == "workspace":
        workspace = project / "other-workspace"
        workspace.mkdir()
    elif changed == "usage":
        usage = "off"
    elif changed == "roles":
        if host == "claude-code":
            (project / ".claude/agents/alphalattice_cro.md").unlink()
        else:
            (project / ".codex/config.toml").write_text(
                '[agents.alphalattice_cro]\nconfig_file = "agents/alphalattice_cro.toml"\n',
                encoding="utf-8",
            )
    else:
        alias = project / "alias-child"
        alias.mkdir()
        workspace = alias / ".." / live.workspace.name
    code = (
        "binding_path_invalid" if changed == "workspace_alias" else "existing_configuration_differs"
    )
    with pytest.raises(NativeBridgeError, match=code):
        bind_session(
            project, host=host, session_id="fixture-parent", workspace=workspace, usage=usage
        )
    assert binding.record_path(project).read_bytes() == before


@pytest.mark.parametrize("relation", ("parented", "foreign", "off"))
def test_a_child_is_served_by_its_ancestry_but_never_reads_its_leads_usage(
    live, tmp_path, monkeypatch, relation
):
    """A Codex child's command works in its recorded parent's workspace (V568); it asks for no
    reading, which is the bound Session's own milestone, and a foreign or OFF binding never
    adopts it."""

    project = _project(live, "codex")
    parent, child = str(UUID(int=1001)), str(UUID(int=1002))
    home = tmp_path / "synthetic-codex-home"
    day = home / "sessions/2026/10/06"
    day.mkdir(parents=True)
    spawn = {
        "parent_thread_id": str(UUID(int=1999)) if relation == "foreign" else parent,
        "agent_role": "alphalattice_cro",
    }
    header = {"id": child, "source": {"subagent": {"thread_spawn": spawn}}}
    (day / f"rollout-2026-10-06T00-00-00-{child}.jsonl").write_text(
        json.dumps({"type": "session_meta", "payload": header}) + "\n", encoding="utf-8"
    )
    monkeypatch.setenv("CODEX_HOME", str(home))
    _session(monkeypatch, "codex", parent)
    client = LocalResearchClient(live.workspace)
    assert (
        client.bind_native_session(project, usage="off" if relation == "off" else "read")["status"]
        == "BOUND"
    )
    binding = NativeResearchBinding.read(project, session=("codex", parent))
    before = binding.record_path(project).read_bytes()
    _session(monkeypatch, "codex", child)
    answer = client.publish_native_event(project, {"source": "native_usage_read"})
    assert answer["status"] == "REFUSED"
    assert answer["failure_code"] == (
        "native_bridge.event_scope_invalid" if relation == "parented" else "native_bridge.not_bound"
    )
    assert client.read_external()["items"] == []
    assert binding.record_path(project).read_bytes() == before


def _usage_files(tmp_path, monkeypatch, session: str) -> None:
    """One synthetic usage file for ``session`` under each host's own root."""
    _lead_usage_file(tmp_path, monkeypatch)
    claude = tmp_path / "synthetic-claude/projects/fixture-project"
    (claude / "fixture-parent.jsonl").rename(claude / f"{session}.jsonl")
    home = tmp_path / "synthetic-codex"
    day = home / "sessions/2026/10/07"
    day.mkdir(parents=True)
    counts = {
        "input_tokens": 9,
        "cached_input_tokens": 2,
        "cache_write_input_tokens": 0,
        "output_tokens": 4,
    }
    records = [
        {"type": "session_meta", "payload": {"id": session, "source": "cli"}},
        {"type": "turn_context", "payload": {"model": "gpt-6-luna", "effort": "high"}},
        {
            "type": "token_usage_record",
            "timestamp": datetime.now(UTC).isoformat(),
            "payload": {"response_id": "r1", "usage": counts},
        },
    ]
    (day / f"rollout-2026-10-07T00-00-00-{session}.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )
    monkeypatch.setenv("CODEX_HOME", str(home))


@pytest.mark.parametrize("host", ("codex", "claude-code"))
def test_a_fresh_clone_binds_a_workspace_kept_outside_it_after_the_documented_configure(
    live, tmp_path, monkeypatch, capsys, host
):
    """A fresh clone binds a workspace kept outside it after the documented configure."""
    from scripts import native_research

    clone = tmp_path / "fresh-clone"
    (clone / ".codex").mkdir(parents=True)
    (clone / ".codex/config.toml").write_bytes((ROOT / ".codex/config.toml").read_bytes())
    if host == "claude-code":
        (clone / ".claude/agents").mkdir(parents=True)
        card = ".claude/agents/alphalattice_cro.md"
        (clone / card).write_bytes((ROOT / card).read_bytes())
        (clone / ".claude/settings.json").write_bytes((ROOT / ".claude/settings.json").read_bytes())
    python = clone / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    python.parent.mkdir(parents=True)
    python.touch()
    assert not live.workspace.resolve().is_relative_to(clone.resolve())
    monkeypatch.setattr(
        sys, "argv", ["native_research.py", "--project", str(clone), "configure", "--host", host]
    )
    assert native_research.main() == 0
    assert json.loads(capsys.readouterr().out)["status"] == "LOCAL_DECLARATIONS_VALIDATED"
    _session(monkeypatch, host)
    if host == "codex":
        _usage_files(tmp_path, monkeypatch, "fixture-parent")
    else:
        _lead_usage_file(tmp_path, monkeypatch)
    client = LocalResearchClient(live.workspace)
    bound = client.bind_native_session(clone)
    assert bound["status"] == "BOUND", bound
    assert bound["attachment_preflight"]["status"] == "READY"
    binding = NativeResearchBinding.read(clone)
    assert binding.workspace == live.workspace and binding.host == host
    reading = client.publish_native_event(clone, {"source": "native_usage_read"})
    assert reading["status"] == "DELIVERED", reading
    (row,) = client.read_external()["items"]
    assert row["payload"]["subject"]["native_agent_id"] == "fixture-parent"
    subject = row["payload"]["subject"]
    assert subject["native_agent_id"] == subject["native_session_id"] == "fixture-parent"
    # A page's request reads the same Session, found through the project it was admitted from.
    asked = client.request({"operation": "SESSION_USAGE_READ"})
    assert asked["status"] == "READ"
    assert [session["host"] for session in asked["sessions"]] == [host]
    # A project that declares nothing is not admitted for an outside workspace.
    plain = tmp_path / "undeclared"
    (plain / ".codex").mkdir(parents=True)
    _session(monkeypatch, host, "fixture-other")
    refused = client.bind_native_session(plain)
    assert refused["failure_code"] == "native_bridge.project_declaration_missing"
    assert not (plain / ".codex" / BINDING_NAME).exists()


def test_only_a_person_turns_usage_reading_off_and_then_nothing_is_read(
    live: LocalPortfolioWebSession,
) -> None:
    """requirement (privacy, FLOW-1): reading is on by default; a person's Settings switch
    turns it off for every Session of the workspace, an agent can read the switch but never
    set it, and with it off a page's or a milestone's request reads no Session file."""
    from alphalattice.interface.local_application.portfolio_research import (
        PortfolioResearchOperationRequest,
    )

    client = LocalResearchClient(live.workspace)
    shown = client.request({"operation": "USAGE_READING"})
    assert (shown["status"], shown["usage_reading"]) == ("USAGE_READING", "READ")
    refused = client.request({"operation": "USAGE_READING_SET", "usage_reading_enabled": False})
    assert refused["failure_code"] == "native_bridge.usage_reading_human_only"
    assert client.request({"operation": "USAGE_READING"})["usage_reading"] == "READ"

    def person(enabled: bool) -> dict:
        return live.operations.execute(
            PortfolioResearchOperationRequest(
                operation="USAGE_READING_SET", usage_reading_enabled=enabled
            ),
            caller="HUMAN",
        )

    assert person(False)["usage_reading"] == "OFF"
    assert client.request({"operation": "SESSION_USAGE_READ"}) == {
        "status": "OFF",
        "sessions": [],
    }
    assert person(True)["usage_reading"] == "READ"
    assert client.request({"operation": "SESSION_USAGE_READ"})["status"] == "NOT_BOUND"


@pytest.mark.parametrize("host", ("codex", "claude-code"))
def test_a_session_that_opens_a_goal_here_is_bound_without_a_separate_step(
    live: LocalPortfolioWebSession, monkeypatch, host
) -> None:
    """A session that opens a goal here is bound without a separate step."""
    declaration = {
        "title": "A study",
        "objective": "Does the signal survive costs?",
        "kind": "REVIEW",
        "criteria": [{"criterion_id": "done", "text": "The review is recorded."}],
        "deliverables": [{"deliverable_id": "result", "kind": "RESULT", "description": "It"}],
    }

    project = _project(live, host)
    _session(monkeypatch, host)
    assert NativeResearchBinding.read(project, session=(host, "fixture-parent")) is None
    client = LocalResearchClient(live.workspace)
    opened = client.request({"operation": "GOAL_OPEN", "goal_declaration": declaration})
    assert opened["status"] == "GOAL_SAVED", opened
    binding = NativeResearchBinding.read(project, session=(host, "fixture-parent"))
    assert binding is not None and binding.workspace == live.workspace.resolve()
    assert binding.usage == "READ"
    record = binding.record_path(project).read_bytes()
    again = client.request(
        {"operation": "GOAL_OPEN", "goal_declaration": {**declaration, "title": "Another"}}
    )
    assert again["status"] == "GOAL_SAVED", again
    assert binding.record_path(project).read_bytes() == record


@pytest.mark.parametrize("host", ("codex", "claude-code"))
def test_one_research_command_binds_the_session_and_opens_its_goal(
    live: LocalPortfolioWebSession, monkeypatch, capsys, host
) -> None:
    """One research command binds the session and opens its goal."""
    from alphalattice.interface.local_application.native_setup import autobind_root

    _session(monkeypatch, host, "fixture-lead")
    line = ["--workspace", str(live.workspace), "cpu-budget", "set", "--queue", "2"]
    assert cli.main(line, serve=lambda _: 99) == 0
    capsys.readouterr()
    root = autobind_root(live.workspace)
    binding = NativeResearchBinding.read(root, session=(host, "fixture-lead"))
    assert binding is not None and binding.workspace == live.workspace.resolve()
    assert binding.usage == "READ"
    client = LocalResearchClient(live.workspace)
    bound = [
        row
        for row in client.read_external()["items"]
        if row["payload"]["subject"].get("message_kind") == "session_bound"
    ]
    assert [row["payload"]["subject"]["native_session_id"] for row in bound] == ["fixture-lead"]
    goals = client.request({"operation": "GOAL_LIST"})["goals"]
    assert len(goals) == 1, goals
    record = client.request({"operation": "GOAL_SHOW", "goal_id": goals[0]["goal_id"]})["record"]
    requests = [
        row for row in record["conversation"] if row.get("input_channel") == "PRODUCT_OPERATION"
    ]
    # The Conversation shows the binding, then the request, both the Session's.
    assert [row["message_kind"] for row in requests] == ["session_bound", "CPU_BUDGET_SET"]
    assert requests[1]["agent_session"] == "fixture-lead"
    assert bound[0]["payload"]["subject"]["goal_id"] == goals[0]["goal_id"]
    # The fact names the goal the Session holds, as its Sessions row reads it (STOPS-1).
    held = client.request({"operation": "GOAL_SHOW", "goal_id": goals[0]["goal_id"]})
    assert bound[0]["payload"]["subject"]["reference"] == f"case:{held['goal']['goal_hash']}"
    record_bytes = binding.record_path(root).read_bytes()
    assert cli.main(line, serve=lambda _: 99) == 0
    capsys.readouterr()
    assert len(client.request({"operation": "GOAL_LIST"})["goals"]) == 1
    assert binding.record_path(root).read_bytes() == record_bytes


def test_a_session_bound_by_a_read_names_the_goal_its_first_request_opens(
    live: LocalPortfolioWebSession, monkeypatch, capsys
) -> None:
    """regression (STOPS-1, the Tech Lead's live check of 2026-10-08): a Session bound by its
    first read held no goal, and when its next request opened one its Sessions row still read
    "No research question is declared". The Host now names the goal once it is held."""
    _session(monkeypatch, "claude-code", "fixture-reader")
    assert (
        cli.main(["--workspace", str(live.workspace), "workspace", "show"], serve=lambda _: 99) == 0
    )
    line = ["--workspace", str(live.workspace), "cpu-budget", "set", "--queue", "2"]
    assert cli.main(line, serve=lambda _: 99) == 0
    capsys.readouterr()
    client = LocalResearchClient(live.workspace)
    [goal] = client.request({"operation": "GOAL_LIST"})["goals"]
    held = client.request({"operation": "GOAL_SHOW", "goal_id": goal["goal_id"]})["goal"]
    facts = [
        row["payload"]["subject"]
        for row in client.read_external()["items"]
        if row["payload"]["subject"].get("message_kind") == "session_bound"
    ]
    assert [fact.get("reference") for fact in facts] == [None, f"case:{held['goal_hash']}"]
    assert facts[1]["goal_id"] == goal["goal_id"]


# Additional imports for test_native_session_binding.py.


def test_a_workspace_left_out_is_the_bound_sessions_and_any_other_is_refused_in_words(
    live: LocalPortfolioWebSession,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Workspace selection uses this session's binding or answers the named absence."""

    project = _agent_project(tmp_path, project=live.workspace.parent)
    workspace = live.workspace
    monkeypatch.chdir(project / "notes" / "deep")
    run = _session_cli(monkeypatch, capsys)
    lead, other = str(uuid4()), str(uuid4())
    code, body = run("task", "list", session=lead)
    assert (code, body["failure_code"]) == (1, "local_client.workspace_unbound")
    assert ["alphalattice", "--workspace", "<dir>", "session", "bind"] in [
        shlex.split(command) for command in re.findall(r"`([^`]+)`", body["detail"])
    ]
    assert body["detail"] == client_refusal("local_client.workspace_unbound").detail
    assert body["next_action"] == client_refusal("local_client.workspace_unbound").next_action
    code, body = run("--workspace", str(workspace), "session", "bind", session=None)
    assert (code, body["failure_code"]) == (1, "local_client.session_unnamed")
    assert not (project / ".codex" / BINDING_NAME).exists()
    code, body = run("--workspace", str(workspace), "session", "bind", session=lead)
    assert (code, body["data"]["status"]) == (0, "BOUND")
    assert "foreground_attachment" not in body["data"], "no hook asks to attach"
    assert (body["data"]["project"], body["data"]["session_id"]) == (str(project), lead)
    code, body = run("task", "list", session=other)
    assert (code, body["failure_code"]) == (1, "local_client.workspace_unbound")
    assert "alphalattice --workspace <dir>" in body["detail"]
    assert body["detail"] == client_refusal("local_client.workspace_unbound").detail
    assert body["next_action"] == client_refusal("local_client.workspace_unbound").next_action
    code, body = run("--workspace", str(workspace), "session", "bind", session=other)
    assert (code, body["data"]["status"]) == (0, "BOUND")
    code, body = run("task", "list", session=other)
    assert body["outcome"] == "OK", body
    assert body["context"]["workspace"] == str(workspace.resolve())
    assert body["context"]["workspace_from"] == "BINDING"
    code, body = run("task", "list", session=None)
    assert (code, body["failure_code"]) == (1, "local_client.workspace_unbound")
    # The bound session's own served workspace, from below the project.
    code, body = run("task", "list", session=lead)
    assert body["outcome"] == "OK", body
    assert body["context"]["workspace"] == str(workspace.resolve())
    assert body["context"]["workspace_from"] == "BINDING"
    code, body = run("--workspace", str(tmp_path), "task", "list", session=lead)
    assert body["context"]["workspace_from"] == "OPTION"
    assert body["context"]["workspace"] == str(tmp_path.resolve())


def test_a_bound_session_is_printed_its_commands_clean_and_any_other_the_full_form(
    live: LocalPortfolioWebSession,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Every printed command is clean for a bound session and explicit for any other session."""

    project = _agent_project(tmp_path, project=live.workspace.parent)
    monkeypatch.chdir(project)
    run = _session_cli(monkeypatch, capsys)
    lead = str(uuid4())
    named = ("--workspace", str(live.workspace))
    assert run(*named, "session", "bind", session=lead)[0] == 0

    def printed(body: dict[str, Any]) -> list[str]:
        found: list[str] = []
        pending: list[object] = [body]
        while pending:
            value = pending.pop()
            if isinstance(value, dict):
                pending.extend(value.values())
            elif isinstance(value, list):
                pending.extend(value)
            elif isinstance(value, str) and value.startswith("alphalattice "):
                found.append(value)
        return found

    # An unknown Task is refused with its next requests, each printed as a command.
    asked = ("task", "show", str(uuid4()))
    for line in (asked, (*named, *asked)):
        code, body = run(*line, session=lead)
        assert code == 2, body
        commands = printed(body)
        assert commands and all("--workspace" not in command for command in commands), commands
        assert "--workspace" not in json.dumps(body)
    for session in (str(uuid4()), None):
        code, body = run(*named, *asked, session=session)
        commands = printed(body)
        assert commands and all(
            command.startswith(f"alphalattice --workspace {live.workspace.resolve()}")
            or command.startswith(f'alphalattice --workspace "{live.workspace.resolve()}"')
            or command.startswith(f"alphalattice --workspace '{live.workspace.resolve()}'")
            for command in commands
        ), commands


def test_session_unbind_removes_this_projects_binding_for_its_session_or_the_person(
    live: LocalPortfolioWebSession,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Unbinding affects only this project's selected session and never an outer project."""

    project = _agent_project(tmp_path, project=live.workspace.parent)
    workspace = live.workspace
    monkeypatch.chdir(project / "notes" / "deep")
    run = _session_cli(monkeypatch, capsys)
    first, second = str(uuid4()), str(uuid4())
    bind = ("--workspace", str(workspace), "session", "bind")
    assert run(*bind, session=first)[0] == 0
    before = (project / ".codex" / BINDING_NAME).read_bytes()
    code, body = run("session", "unbind", session=second)
    assert (code, body["failure_code"]) == (
        2,
        "local_client.session_unbind_refused:native_bridge.session_mismatch",
    )
    assert "other Sessions' bindings stay" in body["detail"]
    words = client_refusal("local_client.session_unbind_refused:native_bridge.session_mismatch")
    assert body["detail"] == words.detail
    assert body["next_action"] == words.next_action
    assert ["alphalattice", "--workspace", "<dir>", "session", "bind"] in [
        shlex.split(command) for command in re.findall(r"`([^`]+)`", body["detail"])
    ]
    assert run(*bind, session=second)[0] == 0
    assert (project / ".codex" / BINDING_NAME).read_bytes() == before
    code, body = run("session", "unbind", session=None)
    assert (code, body["failure_code"]) == (
        2,
        "local_client.session_unbind_refused:native_bridge.binding_ambiguous",
    )
    assert "alphalattice session unbind" in body["detail"]
    assert body["next_action"] == "RESOLVE_THE_NAMED_CAUSE_THEN_UNBIND"
    code, body = run("session", "unbind", session=second)
    assert (code, body["data"]["status"], body["data"]["session_id"]) == (0, "DETACHED", second)
    assert (project / ".codex" / BINDING_NAME).read_bytes() == before
    code, body = run("session", "unbind", session=second)
    assert (code, body["failure_code"]) == (
        2,
        "local_client.session_unbind_refused:native_bridge.session_mismatch",
    )
    code, body = run("session", "unbind", session=first)
    assert (code, body["data"]["status"], body["data"]["session_id"]) == (0, "DETACHED", first)
    assert run("session", "unbind", session=first)[1]["data"]["status"] == "NOT_BOUND"
    assert run(*bind, session=second)[0] == 0
    code, body = run("session", "unbind", session=None)
    assert (code, body["data"]["status"], body["data"]["session_id"]) == (0, "DETACHED", second)
    assert not (project / ".codex" / BINDING_NAME).exists()
    # An inner project's unbind, by the person, never reaches the outer project's binding.
    assert run(*bind, session=first)[0] == 0
    inner = _agent_project(project / "notes")
    monkeypatch.chdir(inner)
    code, body = run("session", "unbind", session=None)
    assert (code, body["data"]["status"], body["data"]["project"]) == (0, "NOT_BOUND", str(inner))
    assert (project / ".codex" / BINDING_NAME).exists()
