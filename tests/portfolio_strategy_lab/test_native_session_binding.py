"""Prospective native attachment metadata is written by the served workspace Host."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

import pytest

from alphalattice.control.product_host.composition.local_web_session import LocalPortfolioWebSession
from alphalattice.interface.local_application import cli
from alphalattice.interface.local_application.client import LocalResearchClient
from alphalattice.interface.local_application.native_bridge import (
    BINDING_NAME,
    NativeBridgeError,
    NativeResearchBinding,
    coordination_event,
    deliver,
)
from alphalattice.interface.local_application.native_observation_sequence import SEQUENCE_NAME
from alphalattice.interface.local_application.native_setup import (
    PROJECT_DECLARATION_NAME,
    bind_session,
    declare_project,
    files_unavailable,
)
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
    assert client.bind_native_session(project)["status"] == "BOUND"
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
    assert answer["data"]["host_trust"] == "NOT_CHECKED"
    assert answer["data"]["foreground_attachment"] == "NOT_REQUESTED"
    assert answer["data"]["attachment_preflight"]["failure_code"] is None
    assert answer["data"]["attachment_preflight"]["missing"] == []
    assert answer["data"]["attachment_preflight"]["native_proof"]["status"] == "NOT_REQUESTED"
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
def test_explicit_native_message_uses_host_sequence_and_keeps_declared_provenance(
    live, monkeypatch, host
):
    project = _project(live, host)
    _session(monkeypatch, host)
    client = LocalResearchClient(live.workspace)
    assert client.bind_native_session(project)["status"] == "BOUND"
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
    assert client.bind_native_session(project)["status"] == "BOUND"
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


@pytest.mark.parametrize("host", ("codex", "claude-code"))
def test_two_named_sessions_keep_their_own_workspace_and_detach_only_their_slot(
    live, monkeypatch, capsys, host
):
    """P2-NH: two fixture Sessions use separate served owners without a manual old detach.

    Environment labels exercise HTTP provenance; this is not live App/Claude evidence.
    """
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
    live, monkeypatch, capsys
):
    """P2-NH: a Session identifier alone grants no other host's binding or public source."""
    project = _project(live, "codex")
    _project(live, "claude-code")
    monkeypatch.chdir(project)
    client = LocalResearchClient(live.workspace)
    session = "fixture-shared-text"
    bindings = {}
    for host in ("codex", "claude-code"):
        _session(monkeypatch, host, session)
        assert client.bind_native_session(project)["status"] == "BOUND"
        binding = NativeResearchBinding.read(project, session=(host, session))
        bindings[host] = binding
        assert (binding.host, binding.session_id) == (host, session)
        event = coordination_event(binding, kind="plan", message=b"A public fixture plan.")
        assert client.publish_native_event(project, event)["status"] == "DELIVERED"
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
def test_child_public_message_uses_exact_native_ancestry_without_lead_or_closure_credit(
    live, tmp_path, monkeypatch, relation
):
    """Fixture native metadata admits a public sender; it proves no live Start or author.

    A child may speak as itself in its recorded parent's Goal. It may not read lead usage,
    claim a sibling or parent sender, or close the assignment with a forged lead decision.
    """
    from uuid import UUID, uuid4

    project = _project(live, "codex")
    parent, child = str(UUID(int=1001)), str(UUID(int=1002))
    home = tmp_path / "synthetic-codex-home"
    day = home / "sessions/2026/10/06"
    day.mkdir(parents=True)
    header = {
        "id": child,
        "agent_path": "/root/cro",
        "source": {
            "subagent": {
                "thread_spawn": {
                    "parent_thread_id": str(UUID(int=1999)) if relation == "foreign" else parent,
                    "agent_role": "alphalattice_cro",
                    "agent_path": "/root/cro",
                }
            }
        },
    }
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
    goal = client.request(
        {
            "operation": "GOAL_OPEN",
            "goal_id": str(uuid4()),
            "change_reason": "Labelled public child-message fixture",
            "goal_declaration": {
                "title": "Public child-message fixture",
                "objective": "Keep a scoped public reply without fabricating acceptance.",
                "kind": "RESEARCH",
                "criteria": [{"criterion_id": "read", "text": "Read the public reply."}],
                "deliverables": [
                    {
                        "deliverable_id": "result",
                        "kind": "RESULT",
                        "description": "The public record.",
                    }
                ],
            },
        }
    )["goal_id"]
    assert client.request({"operation": "GOAL_TAKE", "goal_id": goal})["status"] == "GOAL_TAKEN"
    assignment = coordination_event(
        binding, kind="assignment", recipient_id=child, message=b"Read the labelled fixture packet."
    )
    assert client.publish_native_event(project, assignment)["status"] == "DELIVERED"
    _session(monkeypatch, "codex", child)
    other_goal = None
    if relation == "parented":
        other_goal = client.request(
            {
                "operation": "GOAL_OPEN",
                "goal_id": str(uuid4()),
                "change_reason": "Labelled different child Goal",
                "goal_declaration": {
                    "title": "Different child Goal",
                    "objective": "Detect a wrongly reused default Goal.",
                    "kind": "RESEARCH",
                    "criteria": [{"criterion_id": "read", "text": "Read the public record."}],
                },
            }
        )["goal_id"]
        assert (
            client.request({"operation": "GOAL_TAKE", "goal_id": other_goal})["status"]
            == "GOAL_TAKEN"
        )
    reply = coordination_event(
        binding,
        agent_id=child,
        role="alphalattice_cro",
        kind="answer",
        reply_to=str(assignment["message_id"]),
        message=b"The labelled fixture has no submitted judgment.",
    )
    answer = client.publish_native_event(project, reply)
    if relation == "parented":
        assert answer["status"] == "DELIVERED", answer
        assert answer["goal_id"] == goal
        assert (
            client.publish_native_event(project, {"source": "native_usage_read"})["status"]
            == "REFUSED"
        )
        for forged in (
            coordination_event(binding, kind="decision", message=b"A forged lead statement."),
            coordination_event(
                binding,
                agent_id=str(UUID(int=1003)),
                role="alphalattice_cro",
                kind="answer",
                message=b"A forged sibling statement.",
            ),
            coordination_event(
                binding,
                kind="decision",
                reply_to=str(assignment["message_id"]),
                terminal_decision="COMPLETED",
                terminal_reason="A forged close.",
                message=b"A forged close.",
            ),
        ):
            assert client.publish_native_event(project, forged)["status"] == "REFUSED"
    else:
        assert answer["status"] == "REFUSED"
        assert answer["failure_code"] == "native_bridge.not_bound"
    _session(monkeypatch, "codex", parent)
    record = client.request({"operation": "GOAL_SHOW", "goal_id": goal})["record"]
    assert record["open_assignments"] == [
        {"message_id": assignment["message_id"], "recipient_id": child}
    ]
    assert [row["message_kind"] for row in record["conversation"]] == (
        ["assignment", "answer"] if relation == "parented" else ["assignment"]
    )
    if relation == "parented":
        spoken = record["conversation"][-1]
        assert spoken["agent_id"] == child and spoken["role"] == "alphalattice_cro"
        assert spoken["input_channel"] == "ACTOR_DECLARED"
        assert (
            client.request({"operation": "GOAL_SHOW", "goal_id": other_goal})["record"][
                "conversation"
            ]
            == []
        )
        rows = client.read_external()["items"]
        assert all(row["authority"] == "AGENT_PROPOSAL" for row in rows)
        assert not any("HOOK" in row["payload"]["event_kind"] for row in rows)
    assert binding.record_path(project).read_bytes() == before
