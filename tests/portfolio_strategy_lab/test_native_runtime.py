"""Retained native claims cannot substitute for owner-verified delivery authority."""

from __future__ import annotations

import errno
import json
import shutil
import tomllib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from alphalattice.interface.local_application.native_bridge import (
    JUDGMENT_ROLES,
    NativeResearchBinding,
    role_pin,
)
from alphalattice.interface.local_application.native_runtime import (
    MAX_CHAIN_CANDIDATES,
    definition_digest,
    native_proof_readiness,
    readiness,
    retained_history,
    runtime_definitions,
    validate_runtime_definitions,
)
from alphalattice.interface.local_application.native_setup import (
    bind_session,
    declare_project,
    native_proof_hooks,
)

ROOT = Path(__file__).resolve().parents[2]
GOAL = "00000000-0000-0000-0000-000000000001"


def _project(tmp_path, host="codex", *, native_proof=True):
    project = tmp_path / "project"
    workspace = project / "workspace"
    workspace.mkdir(parents=True)
    relative = ".codex/config.toml" if host == "codex" else ".claude/settings.json"
    declaration = project / relative
    declaration.parent.mkdir()
    declaration.write_bytes((ROOT / relative).read_bytes())
    declare_project(project, host)
    if native_proof:
        hooks = native_proof_hooks(project)
        if host == "claude-code":
            settings = json.loads(declaration.read_text())
            settings["hooks"] = hooks
            declaration.write_text(json.dumps(settings), encoding="utf-8")
        else:
            with declaration.open("a", encoding="utf-8") as stream:
                for event, groups in hooks.items():
                    hook = groups[0]["hooks"][0]
                    stream.write(
                        f'\n[[hooks.{event}]]\nmatcher = "^alphalattice_.*$"\n'
                        f'[[hooks.{event}.hooks]]\ntype = "command"\n'
                        f"command = {json.dumps(hook['command'])}\ntimeout = 5\n"
                    )
    return project, workspace


@pytest.mark.parametrize("hook_shape", ("none", "empty", "unrelated"))
@pytest.mark.parametrize("role", tuple(JUDGMENT_ROLES))
@pytest.mark.parametrize("host", ("codex", "claude-code"))
def test_default_session_readiness_never_reads_optional_native_proof(
    tmp_path, monkeypatch, host, role, hook_shape
):
    project, workspace = _project(tmp_path, host, native_proof=False)
    directory = ".codex" if host == "codex" else ".claude"
    declaration = project / directory / ("config.toml" if host == "codex" else "settings.json")
    shutil.copytree(ROOT / directory / "agents", project / directory / "agents")
    if host == "claude-code":
        settings = json.loads(declaration.read_bytes())
        if hook_shape != "none":
            settings["hooks"] = (
                {}
                if hook_shape == "empty"
                else {
                    "SubagentStart": [
                        {
                            "matcher": "Explore",
                            "hooks": [
                                {"type": "command", "command": "fixture-unrelated", "timeout": 19}
                            ],
                        }
                    ]
                }
            )
        declaration.write_bytes(json.dumps(settings, sort_keys=True).encode())
    elif hook_shape != "none":
        declaration.write_bytes(
            declaration.read_bytes()
            + (
                b"\n[hooks]\n"
                if hook_shape == "empty"
                else b'\n[[hooks.SubagentStart]]\nmatcher = "Explore"\n'
                b'[[hooks.SubagentStart.hooks]]\ntype = "command"\n'
                b'command = "fixture-unrelated"\ntimeout = 19\n'
            )
        )
    configured = declaration.read_bytes()
    card = JUDGMENT_ROLES[role][0]
    bind_session(project, host=host, session_id="parent", workspace=workspace)
    binding = NativeResearchBinding.read(project)
    assert binding is not None and card in binding.roles
    assert (binding.host, binding.session_id, binding.workspace) == (host, "parent", workspace)
    assert role_pin(project, host, card)["role_model"]

    def no_optional_read(*args, **kwargs):
        pytest.fail("Default readiness requested optional native proof")

    opening = Path.open

    def no_hook_definition_read(path, *args, **kwargs):
        if path == declaration:
            pytest.fail("Default readiness opened native hook definitions")
        return opening(path, *args, **kwargs)

    with monkeypatch.context() as default_reads:
        default_reads.setattr(Path, "open", no_hook_definition_read)
        answer = readiness(
            project, binding, None, requester=no_optional_read, read_external=no_optional_read
        )
    assert answer["status"] == "READY"
    assert answer["failure_code"] is None
    assert answer["missing"] == []
    assert answer["foreground_attachment"] == "NOT_REQUESTED"
    assert answer["native_proof"] == {
        "status": "NOT_REQUESTED",
        "host_trust": "NOT_CHECKED",
        "trust_changed": False,
    }
    assert answer["research_nonblocking"] and not answer["trust_changed"]
    assert declaration.read_bytes() == configured


def _rows(project, role, *, host="codex"):
    epoch = datetime(2026, 10, 6, tzinfo=UTC)
    card = JUDGMENT_ROLES[role][0] + ("_medium" if host == "claude-code" else "")
    fingerprint = definition_digest(project, host)

    def event(ordinal, kind, **subject):
        timestamp = (epoch + timedelta(seconds=ordinal)).isoformat()
        return {
            "ordinal": ordinal,
            "observation_id": str(ordinal) * 64,
            "authority": "AGENT_PROPOSAL",
            "occurred_at": timestamp,
            "observed_at": timestamp,
            "correlation_ids": ["parent"],
            "payload": {
                "event_kind": kind,
                "summary": "PRIVATE",
                "subject": {
                    "native_host": host,
                    "native_session_id": "parent",
                    "goal_id": GOAL,
                    **subject,
                },
            },
        }

    rows = [
        event(
            1,
            "NATIVE_SUBAGENT_START_HOOK",
            native_agent_id="child",
            role=card,
            hook_model="actual-model",
            native_definition_digest=fingerprint,
        ),
        event(
            2,
            "NATIVE_COORDINATION_MESSAGE",
            native_agent_id="parent",
            role="research_lead",
            message_kind="assignment",
            message_id="assignment",
            reference="a" * 64,
            recipient_id="child",
        ),
        event(
            3,
            "NATIVE_SUBAGENT_STOP_HOOK",
            native_agent_id="child",
            role=card,
            native_definition_digest=fingerprint,
        ),
        event(
            4,
            "NATIVE_COORDINATION_MESSAGE",
            native_agent_id="child",
            role=card,
            message_kind="answer",
            input_channel="PRODUCT_ACCEPTED_ANSWER",
            authorship_basis="HOOK",
            bundle_reference="a" * 64,
            answer_reference="b" * 64,
            reference="00000000-0000-0000-0000-000000000002",
            submitted_by="parent",
        ),
    ]
    for ordinal, agent, usage_role in ((5, "child", card), (6, "parent", "research_lead")):
        rows.append(
            event(
                ordinal,
                "NATIVE_AGENT_USAGE",
                native_agent_id=agent,
                role=usage_role,
                model="actual-model",
                input_channel="CODEX_SESSION_FILE"
                if host == "codex"
                else "CLAUDE_CODE_SESSION_FILE",
                last_at=(epoch + timedelta(seconds=ordinal)).isoformat(),
                responses="1",
                input_tokens="1",
                output_tokens="1",
            )
        )
    binding = NativeResearchBinding(
        "parent", project / "workspace", (card,), host=host, observation_started_at=epoch
    )
    return binding, rows


def _accepted_readback(role="RISK", host="codex"):
    """Synthetic public owner evidence, distinct from retained lifecycle claims."""
    return {
        "observation_id": "4" * 64,
        "accepted_answer": {
            "status": "AVAILABLE",
            "observation_id": "4" * 64,
            "native_host": host,
            "native_session_id": "parent",
            "native_agent_id": "child",
            "role": JUDGMENT_ROLES[role][0] + ("_medium" if host == "claude-code" else ""),
            "submitted_by": "parent",
            "task_id": "00000000-0000-0000-0000-000000000002",
            "bundle_reference": "a" * 64,
            "answer_reference": "b" * 64,
            "answer_digest": "e" * 64,
            "contribution": {"text": "PRIVATE owner contribution"},
        },
    }


@pytest.mark.parametrize("role", tuple(JUDGMENT_ROLES))
@pytest.mark.parametrize("host", ("codex", "claude-code"))
def test_every_role_retains_claims_but_owner_accepted_record_never_verifies_native_delivery(
    tmp_path, role, host
):
    project, _ = _project(tmp_path, host)
    binding, rows = _rows(project, role, host=host)
    runtime = {
        "status": "RUNTIME_PROJECT_HOOKS_TRUSTED",
        "host_trust": "TRUSTED",
        "active_session_attachment": "PROVED",
    }
    calls = []

    def read_external(*, observation_id):
        calls.append(observation_id)
        return _accepted_readback(role, host)

    answer = native_proof_readiness(
        project, binding, rows, runtime=runtime, goal_id=GOAL, read_external=read_external
    )
    assert answer["status"] == "REFUSED"
    assert answer["foreground_attachment"] == "NOT_PROVED"
    assert all(item["status"] == "NOT_PROVED" for item in answer["roles"].values())
    chain = answer["roles"][role]["chains"][0]
    assert chain["status"] == "retained_not_proved"
    assert chain["missing"] == ["native_event_delivery_unverified"]
    assert chain["accepted_answer"] == {"status": "AVAILABLE", "answer_digest": "e" * 64}
    assert answer["evidence"] == [chain]
    assert calls == ["4" * 64]
    assert "native_event_delivery_unverified" in answer["missing"]
    assert "PRIVATE" not in json.dumps(answer)


@pytest.mark.parametrize(
    "defect",
    (
        "legacy_checkpoint",
        "old_start",
        "changed_definition",
        "missing_assignment",
        "missing_answer",
        "actor_answer",
        "missing_stop",
        "missing_child_usage",
        "missing_lead_usage",
        "wrong_goal",
        "wrong_start_goal",
        "wrong_stop_goal",
        "wrong_usage_goal",
        "empty_child_usage",
        "wrong_source_task",
        "untrusted_runtime",
        "unattached_runtime",
        "missing_history",
        "retained_gap",
    ),
)
def test_no_configuration_or_old_or_partial_record_can_establish_readiness(tmp_path, defect):
    project, _ = _project(tmp_path)
    binding, rows = _rows(project, "RISK")
    runtime = {
        "status": "RUNTIME_PROJECT_HOOKS_TRUSTED",
        "host_trust": "TRUSTED",
        "active_session_attachment": "PROVED",
    }
    if defect == "legacy_checkpoint":
        binding = NativeResearchBinding(binding.session_id, binding.workspace, binding.roles)
    elif defect == "old_start":
        rows[0]["occurred_at"] = "2026-10-05T00:00:00+00:00"
    elif defect == "changed_definition":
        rows[0]["payload"]["subject"]["native_definition_digest"] = "f" * 64
    elif defect in {
        "missing_assignment",
        "missing_answer",
        "missing_stop",
        "missing_child_usage",
        "missing_lead_usage",
    }:
        rows.pop(
            {
                "missing_assignment": 1,
                "missing_answer": 3,
                "missing_stop": 2,
                "missing_child_usage": 4,
                "missing_lead_usage": 5,
            }[defect]
        )
    elif defect == "actor_answer":
        rows[3]["payload"]["subject"]["input_channel"] = "ACTOR_DECLARED"
    elif defect in {"wrong_goal", "wrong_start_goal", "wrong_stop_goal"}:
        rows[{"wrong_goal": 1, "wrong_start_goal": 0, "wrong_stop_goal": 2}[defect]]["payload"][
            "subject"
        ]["goal_id"] = "00000000-0000-0000-0000-000000000003"
    elif defect == "wrong_usage_goal":
        rows[4]["payload"]["subject"]["goal_id"] = "00000000-0000-0000-0000-000000000003"
    elif defect == "empty_child_usage":
        rows[4]["payload"]["subject"]["responses"] = "0"
    elif defect == "wrong_source_task":
        rows[3]["payload"]["subject"]["reference"] = "not-a-task"
    elif defect == "untrusted_runtime":
        runtime["status"] = "RUNTIME_HOOK_TRUST_UNPROVED"
    elif defect == "unattached_runtime":
        runtime["active_session_attachment"] = "NOT_PROVED"
    elif defect == "missing_history":
        rows = None
    elif defect == "retained_gap":
        rows.append({"ordinal": 7, "correlation_ids": ["parent"], "payload": None})
    answer = native_proof_readiness(
        project,
        binding,
        rows,
        runtime=runtime,
        goal_id=GOAL,
        read_external=lambda **_: _accepted_readback(),
    )
    assert answer["status"] == "REFUSED"
    assert answer["missing"]
    assert answer["research_nonblocking"] is True
    assert answer["trust_changed"] is False
    assert "PRIVATE" not in json.dumps(answer)
    expected_missing = {
        "legacy_checkpoint": "prospective_observation_checkpoint",
        "old_start": "fresh_same_definition_start",
        "changed_definition": "fresh_same_definition_start",
        "missing_assignment": "exact_assignment",
        "missing_answer": "credited_accepted_answer",
        "actor_answer": "credited_accepted_answer",
        "missing_stop": "native_subagent_stop",
        "missing_child_usage": "child_usage",
        "missing_lead_usage": "lead_usage",
        "wrong_goal": "exact_assignment",
        "wrong_start_goal": "exact_goal_binding",
        "wrong_stop_goal": "exact_goal_binding",
        "wrong_usage_goal": "exact_goal_binding",
        "empty_child_usage": "child_usage",
        "wrong_source_task": "credited_accepted_answer",
        "untrusted_runtime": "actual_runtime_definitions_and_trust",
        "unattached_runtime": "active_native_session_attachment",
        "missing_history": "native_history",
        "retained_gap": "retained_history",
    }[defect]
    assert expected_missing in set(answer["missing"]) | set(answer["roles"]["RISK"]["missing"])


@pytest.mark.parametrize(
    "defect",
    (
        "reader_missing",
        "unavailable",
        "wrong_event_kind",
        "wrong_observation",
        "wrong_host",
        "wrong_session",
        "wrong_child",
        "wrong_role",
        "wrong_submitter",
        "wrong_task",
        "wrong_bundle",
        "wrong_answer",
        "missing_digest",
        "invalid_digest",
    ),
)
def test_forged_complete_native_claims_need_exact_selected_sealed_answer_readback(tmp_path, defect):
    project, _ = _project(tmp_path)
    binding, rows = _rows(project, "RISK")
    for row in rows:
        row.update(source_id="native-hook:reserved", source_sequence=row["ordinal"])
    selected = _accepted_readback()
    accepted = selected["accepted_answer"]
    if defect == "unavailable":
        accepted["status"] = "UNAVAILABLE"
    elif defect == "wrong_event_kind":
        rows[3]["payload"]["event_kind"] = "EVENT_DECLARE"
    elif defect == "missing_digest":
        accepted.pop("answer_digest")
    elif defect == "invalid_digest":
        accepted["answer_digest"] = 10**63
    elif defect.startswith("wrong_"):
        accepted[
            {
                "wrong_observation": "observation_id",
                "wrong_host": "native_host",
                "wrong_session": "native_session_id",
                "wrong_child": "native_agent_id",
                "wrong_role": "role",
                "wrong_submitter": "submitted_by",
                "wrong_task": "task_id",
                "wrong_bundle": "bundle_reference",
                "wrong_answer": "answer_reference",
            }[defect]
        ] = "other"
    calls = []

    def read_external(*, observation_id):
        calls.append(observation_id)
        return selected

    answer = native_proof_readiness(
        project,
        binding,
        rows,
        runtime={
            "status": "RUNTIME_PROJECT_HOOKS_TRUSTED",
            "host_trust": "TRUSTED",
            "active_session_attachment": "PROVED",
        },
        read_external=None if defect == "reader_missing" else read_external,
    )
    assert answer["status"] == "REFUSED"
    assert answer["foreground_attachment"] == "NOT_PROVED"
    assert answer["roles"]["RISK"]["status"] == "NOT_PROVED"
    assert "native_event_delivery_unverified" in answer["missing"]
    assert "credited_accepted_answer" in answer["roles"]["RISK"]["missing"]
    assert calls == ([] if defect in {"reader_missing", "wrong_event_kind"} else ["4" * 64])
    assert "PRIVATE" not in json.dumps(answer)


def test_accepted_candidate_budget_refuses_before_unbounded_selected_owner_reads(tmp_path):
    project, _ = _project(tmp_path)
    binding, rows = _rows(project, "RISK")
    rows.extend([rows[3]] * MAX_CHAIN_CANDIDATES)
    calls = []
    answer = native_proof_readiness(
        project,
        binding,
        rows,
        runtime={"status": "NOT_CHECKED"},
        read_external=lambda **selector: calls.append(selector),
    )
    assert answer["roles"]["RISK"]["missing"] == ["native_evidence_projection_budget"]
    assert calls == []


def _hooks_response(project, workspace):
    config = project / ".codex/config.toml"
    declarations = tomllib.loads(config.read_text())["hooks"]
    return {
        "data": [
            {
                "cwd": str(cwd),
                "errors": [],
                "warnings": [],
                "hooks": [
                    {
                        "source": "project",
                        "sourcePath": str(config),
                        "matcher": "^alphalattice_.*$",
                        "eventName": native,
                        "enabled": True,
                        "trustStatus": "trusted",
                        "currentHash": "sha256:" + "a" * 64,
                        "handlerType": "command",
                        "isManaged": False,
                        "timeoutSec": 5,
                        "async": False,
                        "key": str(config) + ":" + wire + ":0:0",
                        "command": declarations[event][0]["hooks"][0]["command"],
                    }
                    for event, native, wire in (
                        ("SubagentStart", "subagentStart", "subagent_start"),
                        ("SubagentStop", "subagentStop", "subagent_stop"),
                    )
                ],
            }
            for cwd in (project, workspace)
        ]
    }


def test_exact_runtime_requester_reads_definitions_and_loaded_session_without_writes(tmp_path):
    project, workspace = _project(tmp_path)
    calls = []

    def request(method, params):
        calls.append((method, params))
        if method == "hooks/list":
            return _hooks_response(project, workspace)
        if method == "thread/read":
            return {
                "thread": {
                    "id": "parent",
                    "cwd": str(project),
                    "status": {"type": "idle"},
                    "title": "PRIVATE TITLE",
                }
            }
        return {"data": ["parent"], "nextCursor": None}

    answer = runtime_definitions(
        project, workspace, "codex", session_id="parent", requester=request
    )
    assert answer["status"] == "RUNTIME_PROJECT_HOOKS_TRUSTED"
    assert answer["active_session_attachment"] == "PROVED"
    assert [method for method, _ in calls] == ["hooks/list", "thread/read", "thread/loaded/list"]
    assert "PRIVATE" not in json.dumps(answer)
    assert answer["trust_changed"] is False


def _runtime_requester(raw, cwd, *, thread_id="parent", status="idle", loaded=("parent",)):
    """Labelled RPC transport fixtures; no live App, lifecycle or owner authority is supplied."""

    def request(method, params):
        if method == "hooks/list":
            return raw
        if method == "thread/read":
            assert params == {"threadId": "parent", "includeTurns": False}
            return {
                "thread": {
                    "id": thread_id,
                    "cwd": str(cwd),
                    "status": {"type": status},
                    "title": "PRIVATE TITLE",
                }
            }
        assert method == "thread/loaded/list"
        return {"data": list(loaded), "nextCursor": None}

    return request


@pytest.mark.parametrize("trusted_scope", ("project", "workspace"))
@pytest.mark.parametrize("parent_scope", ("project", "workspace"))
@pytest.mark.parametrize("status", ("idle", "active"))
def test_loaded_parent_requires_only_its_own_exact_definition_scope(
    tmp_path, trusted_scope, parent_scope, status
):
    """A data-workspace query need not discover hooks loaded at the actual parent's cwd."""
    project, workspace = _project(tmp_path)
    scopes = {"project": project, "workspace": workspace}
    raw = _hooks_response(project, workspace)
    empty_index = 1 if trusted_scope == "project" else 0
    raw["data"][empty_index]["hooks"] = []
    conservative = validate_runtime_definitions(raw, project=project, workspace=workspace)
    assert conservative["status"] == "RUNTIME_HOOK_TRUST_UNPROVED"
    assert conservative["definition_scope"] == "ALL_REQUESTED_CWDS"
    assert conservative["required_cwd"] is None
    answer = runtime_definitions(
        project,
        workspace,
        "codex",
        session_id="parent",
        requester=_runtime_requester(raw, scopes[parent_scope], status=status),
    )
    assert answer["status"] == (
        "RUNTIME_PROJECT_HOOKS_TRUSTED"
        if trusted_scope == parent_scope
        else "RUNTIME_HOOK_TRUST_UNPROVED"
    )
    assert answer["active_session_attachment"] == "PROVED"
    assert answer["connection"] == "EXACT_RUNTIME_RPC"
    assert answer["definition_scope"] == "REQUIRED_CWD"
    assert answer["required_cwd"] == str(scopes[parent_scope].resolve())
    assert {row["cwd"]: row["host_trust"] for row in answer["hooks"]} == {
        str(scopes[trusted_scope].resolve()): "TRUSTED",
        str(scopes["workspace" if trusted_scope == "project" else "project"].resolve()): "UNPROVED",
    }
    assert "PRIVATE" not in json.dumps(answer)
    assert answer["trust_changed"] is False


@pytest.mark.parametrize("defect", ("other_id", "other_cwd", "not_loaded", "not_active"))
def test_required_definition_scope_is_not_selected_before_loaded_parent_admission(tmp_path, defect):
    """A readable thread alone cannot narrow the conservative definition guard."""
    project, workspace = _project(tmp_path)
    raw = _hooks_response(project, workspace)
    raw["data"][1]["hooks"] = []
    answer = runtime_definitions(
        project,
        workspace,
        "codex",
        session_id="parent",
        requester=_runtime_requester(
            raw,
            tmp_path / "other" if defect == "other_cwd" else project,
            thread_id="other" if defect == "other_id" else "parent",
            status="notLoaded" if defect == "not_active" else "idle",
            loaded=() if defect == "not_loaded" else ("parent",),
        ),
    )
    assert answer["status"] == "NOT_CHECKED" and answer["host_trust"] == "NOT_CHECKED"
    assert answer["active_session_attachment"] == "NOT_PROVED"
    assert answer["attachment_diagnostic"] == "EXACT_SESSION_NOT_LOADED_IN_PROJECT"
    assert answer["definition_check"]["status"] == "RUNTIME_HOOK_TRUST_UNPROVED"
    assert answer["definition_check"]["definition_scope"] == "ALL_REQUESTED_CWDS"
    assert answer["definition_check"]["required_cwd"] is None
    assert "PRIVATE" not in json.dumps(answer)
    assert answer["trust_changed"] is False


def test_managed_proxy_loaded_thread_does_not_select_actual_app_definition_scope(
    tmp_path, monkeypatch
):
    """Even matching managed transport metadata does not admit an actual App requester."""
    from alphalattice.interface.local_application import native_runtime

    project, workspace = _project(tmp_path)
    raw = _hooks_response(project, workspace)
    raw["data"][1]["hooks"] = []
    closed = []

    class ManagedFixtureProxy:
        def __init__(self, cwd):
            assert cwd == project
            self.request = _runtime_requester(raw, project)

        def close(self):
            closed.append(True)

    monkeypatch.setattr(native_runtime, "ManagedRuntimeRPC", ManagedFixtureProxy)
    answer = runtime_definitions(project, workspace, "codex", session_id="parent")
    assert answer["status"] == "NOT_CHECKED" and answer["host_trust"] == "NOT_CHECKED"
    assert answer["active_session_attachment"] == "NOT_PROVED"
    assert answer["connection"] == "MANAGED_CONTROL_RPC"
    assert answer["attachment_diagnostic"] == "ACTIVE_APP_RPC_NOT_EXPOSED"
    assert answer["definition_check"]["definition_scope"] == "ALL_REQUESTED_CWDS"
    assert answer["definition_check"]["required_cwd"] is None
    assert closed == [True] and answer["trust_changed"] is False
    assert "PRIVATE" not in json.dumps(answer)


@pytest.mark.parametrize(
    "field, changed",
    (
        ("command", "PRIVATE COMMAND"),
        ("timeoutSec", 6),
        ("async", True),
        ("isManaged", True),
        ("handlerType", "prompt"),
        ("key", "different"),
        ("currentHash", "sha256:" + "b" * 64),
        ("currentHash", "invalid"),
        ("enabled", False),
        ("trustStatus", "untrusted"),
        ("sourcePath", "other_source"),
    ),
)
def test_selected_scope_retains_every_exact_definition_check(tmp_path, field, changed):
    """Selecting a cwd never lets hash trust override a mismatched actual definition."""
    project, workspace = _project(tmp_path)
    raw = _hooks_response(project, workspace)
    expected = dict.fromkeys(("subagentStart", "subagentStop"), "sha256:" + "a" * 64)
    assert (
        validate_runtime_definitions(
            raw,
            project=project,
            workspace=workspace,
            required_cwd=workspace,
            expected_hashes=expected,
        )["status"]
        == "RUNTIME_PROJECT_HOOKS_TRUSTED"
    )
    raw["data"][1]["hooks"][0][field] = (
        str(tmp_path / "other.toml") if field == "sourcePath" else changed
    )
    answer = validate_runtime_definitions(
        raw, project=project, workspace=workspace, required_cwd=workspace, expected_hashes=expected
    )
    assert answer["status"] == "RUNTIME_HOOK_TRUST_UNPROVED"
    assert answer["host_trust"] == "UNPROVED"
    assert answer["required_cwd"] == str(workspace.resolve())
    assert answer["hooks"][0]["host_trust"] == "TRUSTED"
    assert answer["hooks"][1]["host_trust"] == "UNPROVED"
    assert "PRIVATE COMMAND" not in json.dumps(answer)
    assert answer["trust_changed"] is False


def test_unselected_hashes_do_not_replace_the_selected_exact_host_hashes(tmp_path):
    """Hash aggregation follows the admitted scope; the direct default remains conservative."""
    project, workspace = _project(tmp_path)
    raw = _hooks_response(project, workspace)
    for hook in raw["data"][1]["hooks"]:
        hook["currentHash"] = "sha256:" + "b" * 64
    assert validate_runtime_definitions(raw, project=project, workspace=workspace)["status"] == (
        "RUNTIME_HOOK_TRUST_UNPROVED"
    )
    answer = runtime_definitions(
        project,
        workspace,
        "codex",
        session_id="parent",
        requester=_runtime_requester(raw, project),
    )
    assert answer["status"] == "RUNTIME_PROJECT_HOOKS_TRUSTED"
    assert answer["definition_hashes"] == dict.fromkeys(
        ("subagentStart", "subagentStop"), "sha256:" + "a" * 64
    )
    assert answer["hooks"][1]["definition_hashes"] == dict.fromkeys(
        ("subagentStart", "subagentStop"), "sha256:" + "b" * 64
    )
    assert answer["active_session_attachment"] == "PROVED"
    assert answer["trust_changed"] is False


def test_history_failure_and_pagination_cycle_are_unavailable_not_empty():
    assert retained_history(lambda **_: {"disposition": "UNAVAILABLE", "items": []})[0] is None
    calls = []

    def repeated(**cursor):
        calls.append(cursor)
        return {"items": [], "more": True, "oldest": 10, "epoch": "same"}

    rows, status = retained_history(repeated)
    assert rows is None and status["status"] == "UNAVAILABLE"
    assert calls == [{"before": None}, {"before": 10}]


def test_trusted_hashes_must_match_the_same_exact_host_definition(tmp_path):
    project, workspace = _project(tmp_path)
    raw = _hooks_response(project, workspace)
    for hook in raw["data"][1]["hooks"]:
        hook["currentHash"] = "sha256:" + "b" * 64
    assert (
        validate_runtime_definitions(raw, project=project, workspace=workspace)["status"]
        == "RUNTIME_HOOK_TRUST_UNPROVED"
    )


@pytest.mark.parametrize(
    "field, changed",
    (
        ("command", "PRIVATE COMMAND"),
        ("timeoutSec", 6),
        ("async", True),
        ("isManaged", True),
        ("handlerType", "prompt"),
        ("key", "different"),
    ),
)
def test_trusted_hash_never_overrides_a_different_actual_definition(tmp_path, field, changed):
    project, workspace = _project(tmp_path)
    raw = _hooks_response(project, workspace)
    assert (
        validate_runtime_definitions(raw, project=project, workspace=workspace)["status"]
        == "RUNTIME_PROJECT_HOOKS_TRUSTED"
    )
    raw["data"][0]["hooks"][0][field] = changed
    answer = validate_runtime_definitions(raw, project=project, workspace=workspace)
    assert answer["status"] == "RUNTIME_HOOK_TRUST_UNPROVED"
    assert "PRIVATE COMMAND" not in json.dumps(answer)


@pytest.mark.parametrize("host", ("codex", "claude-code"))
@pytest.mark.parametrize("linked", ("declaration", "host_directory"))
def test_original_host_declaration_links_never_establish_runtime_trust(tmp_path, host, linked):
    project, workspace = _project(tmp_path, host)
    declaration = project / (".codex/config.toml" if host == "codex" else ".claude/settings.json")
    raw = _hooks_response(project, workspace) if host == "codex" else None
    if raw is not None:
        assert (
            validate_runtime_definitions(raw, project=project, workspace=workspace)["status"]
            == "RUNTIME_PROJECT_HOOKS_TRUSTED"
        )
    assert len(definition_digest(project, host)) == 64
    target_directory = tmp_path / "ordinary-target"
    target = target_directory / declaration.name
    if linked == "declaration":
        target_directory.mkdir()
        target.write_bytes(declaration.read_bytes())
        declaration.unlink()
        link, destination = declaration, target
    else:
        declaration.parent.rename(target_directory)
        link, destination = declaration.parent, target_directory
    try:
        link.symlink_to(destination, target_is_directory=linked == "host_directory")
    except (OSError, NotImplementedError) as error:
        if (
            isinstance(error, OSError)
            and error.errno not in {errno.EPERM, errno.EACCES, errno.ENOSYS, errno.ENOTSUP}
            and getattr(error, "winerror", None) not in {50, 1314}
        ):
            raise
        pytest.skip(f"This platform cannot create this bounded symlink fixture: {error}")
    assert link.is_symlink()
    with pytest.raises(ValueError, match=r"native_hook\.definition_unreadable"):
        definition_digest(project, host)
    if raw is not None:
        # The host may report the ordinary resolved target: its matching hash and
        # handler fields cannot authorize a linked original declaration.
        for group in raw["data"]:
            for hook in group["hooks"]:
                source = str(target.resolve())
                hook["sourcePath"] = source
                wire = "subagent_start" if hook["eventName"] == "subagentStart" else "subagent_stop"
                hook["key"] = f"{source}:{wire}:0:0"
        answer = validate_runtime_definitions(raw, project=project, workspace=workspace)
        assert answer["status"] == "RUNTIME_HOOK_TRUST_UNPROVED"
        assert answer["host_trust"] == "UNPROVED"
    else:
        answer = runtime_definitions(project, workspace, host)
        assert answer["status"] == "NOT_CHECKED"
        assert answer["host_trust"] == "NOT_CHECKED"
    assert answer["trust_changed"] is False


def test_missing_local_declarations_are_a_named_missing_link(tmp_path):
    project, _ = _project(tmp_path)
    binding, rows = _rows(project, "RISK")
    (project / ".codex/config.toml").unlink()
    answer = native_proof_readiness(project, binding, rows, runtime={"status": "NOT_CHECKED"})
    assert answer["status"] == "REFUSED"
    assert "local_lifecycle_definitions" in answer["missing"]
    assert answer["local_definition_digest"] is None
