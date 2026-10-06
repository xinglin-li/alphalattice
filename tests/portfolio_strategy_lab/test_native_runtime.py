"""Retained native claims cannot substitute for owner-verified delivery authority."""

from __future__ import annotations

import json
import tomllib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from alphalattice.interface.local_application.native_bridge import (
    JUDGMENT_ROLES,
    NativeResearchBinding,
)
from alphalattice.interface.local_application.native_runtime import (
    MAX_CHAIN_CANDIDATES,
    definition_digest,
    readiness,
    retained_history,
    runtime_definitions,
    validate_runtime_definitions,
)

ROOT = Path(__file__).resolve().parents[2]
GOAL = "00000000-0000-0000-0000-000000000001"


def _project(tmp_path, host="codex"):
    project = tmp_path / "project"
    workspace = project / "workspace"
    workspace.mkdir(parents=True)
    relative = ".codex/config.toml" if host == "codex" else ".claude/settings.json"
    declaration = project / relative
    declaration.parent.mkdir()
    declaration.write_bytes((ROOT / relative).read_bytes())
    return project, workspace


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

    answer = readiness(
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
    answer = readiness(
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

    answer = readiness(
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
    answer = readiness(
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
    target_directory.mkdir()
    target = target_directory / declaration.name
    target.write_bytes(declaration.read_bytes())
    declaration.unlink()
    try:
        if linked == "declaration":
            declaration.symlink_to(target)
        else:
            declaration.parent.rmdir()
            declaration.parent.symlink_to(target_directory, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("This platform cannot create this bounded symlink fixture.")
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
    answer = readiness(project, binding, rows, runtime={"status": "NOT_CHECKED"})
    assert answer["status"] == "REFUSED"
    assert "local_lifecycle_definitions" in answer["missing"]
    assert answer["local_definition_digest"] is None
