"""The Session's own optional readings and CLI ordering: the lead and the children its own
files record, with no hook, message or always-running reader (FLOW-1)."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from alphalattice.control.product_host.composition.goals import GoalApplication
from alphalattice.control.product_host.composition.workspace_activity import WorkspaceActivity
from alphalattice.control.product_host.publication.goals import GoalStore
from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.interface.local_application import cli, client, native_bridge
from alphalattice.interface.local_application.activity import (
    ExternalActivityEventDocument,
    ExternalActivityReadQuery,
)
from alphalattice.interface.local_application.cli_contract import RequestProvenance
from alphalattice.interface.local_application.native_bridge import (
    BINDING_NAME,
    NativeResearchBinding,
    lead_readings,
    read_session_usage,
)
from alphalattice.interface.local_application.native_setup import declare_project
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchRequestDocument,
)

SECRET = "SYNTHETIC-PRIVATE-TRANSCRIPT-AND-ERROR"
AT = datetime(2026, 10, 6, tzinfo=UTC)


@pytest.fixture(params=["codex", "claude-code"])
def lead_scene(request, tmp_path, monkeypatch):
    """Real session parser, observation and Goal owners; a public local transport seam."""
    host = request.param
    parent = "synthetic-lead-" + uuid4().hex
    project = tmp_path / "project"
    workspace = project / "workspace"
    nested = project / "tools" / "nested"
    workspace.mkdir(parents=True)
    nested.mkdir(parents=True)
    (project / ".codex").mkdir()
    declare_project(project, host)
    binding_path = project / ".codex" / BINDING_NAME
    binding_path.write_text(
        json.dumps(
            {
                "host": host,
                "session_id": parent,
                "workspace": str(workspace),
                "roles": ["alphalattice_risk"],
                "usage": "READ",
            }
        ),
        encoding="utf-8",
        newline="\n",
    )
    monkeypatch.delenv("CODEX_THREAD_ID", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_SESSION_ID", raising=False)
    if host == "codex":
        config = tmp_path / "codex"
        monkeypatch.setenv("CODEX_HOME", str(config))
        monkeypatch.setenv("CODEX_THREAD_ID", parent)
        session = config / "sessions/2026/10/06" / f"rollout-synthetic-{parent}.jsonl"
        records = [
            {"type": "session_meta", "payload": {"id": parent}},
            {"type": "response_item", "payload": {"type": "message", "content": SECRET}},
            {"type": "turn_context", "payload": {"model": "gpt-6-luna", "effort": "high"}},
            {
                "type": "token_usage_record",
                "timestamp": AT.isoformat(),
                "payload": {
                    "response_id": "synthetic-response",
                    "usage": {
                        "input_tokens": 20,
                        "cached_input_tokens": 5,
                        "cache_write_input_tokens": 0,
                        "output_tokens": 3,
                    },
                },
            },
        ]
    else:
        config = tmp_path / "claude"
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config))
        monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", parent)
        session = config / "projects/p" / f"{parent}.jsonl"
        records = [
            {
                "type": "assistant",
                "timestamp": AT.isoformat(),
                "message": {
                    "id": "synthetic-response",
                    "model": "claude-sonnet-5-5",
                    "role": "assistant",
                    "content": [{"type": "text", "text": SECRET}],
                    "usage": {
                        "input_tokens": 15,
                        "cache_read_input_tokens": 5,
                        "cache_creation_input_tokens": 0,
                        "output_tokens": 3,
                    },
                },
            }
        ]
    session.parent.mkdir(parents=True)
    session.write_text("\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")
    observer = WorkspaceActivity(
        workspace=workspace,
        workspace_id="synthetic-lead-usage",
        gate=WorkspaceMutationGate(),
        instance="synthetic-instance",
        clock=lambda: AT,
    )
    goals = GoalApplication(
        GoalStore(workspace / "artifacts", "synthetic-lead-usage"),
        lambda: AT,
        lambda *_args: pytest.fail("This lead-usage seam performs no research reads."),
        lambda _task: AT,
        lambda _task: None,
        workspace=workspace,
    )
    opened = goals.operate(
        PortfolioResearchRequestDocument(
            operation="GOAL_OPEN",
            goal_id=str(uuid4()),
            change_reason="Synthetic declaration",
            goal_declaration={
                "title": "Synthetic lead usage",
                "objective": "Retain the lead's exact usage.",
                "kind": "RESEARCH",
                "criteria": [{"criterion_id": "read", "text": "Read it."}],
                "deliverables": [
                    {"deliverable_id": "result", "kind": "RESULT", "description": "Receipt."}
                ],
            },
        ).to_operation_request(),
        "HUMAN",
        None,
    )
    scene = SimpleNamespace(
        host=host,
        parent=parent,
        project=project,
        workspace=workspace,
        nested=nested,
        binding_path=binding_path,
        session=session,
        observer=observer,
        goals=goals,
        goal_id=opened["goal_id"],
        order=[],
        failure=None,
        answer={"status": "ACCEPTED"},
        last_request_goal=None,
    )

    def publish(document, native_provenance):
        if scene.failure == "delivery":
            # The Host's own observation owner refuses; nothing is filed.
            return {"status": "REFUSED", "failure_code": "activity.storage_unavailable"}
        filed, goal, owner_session = goals.file_event(
            ExternalActivityEventDocument.model_validate(document), native_provenance
        )
        answer = observer.admit_external_event(filed)
        if goal is not None:
            if answer["status"] == "APPENDED":
                return {
                    **answer,
                    **goals.record_event(goal, filed, answer["observation_id"], owner_session),
                }
            return {**answer, "goal_id": str(goal.goal_id)}
        return answer

    scene.publish = lambda document: publish(
        document, RequestProvenance(vendor=host, session=parent, goal_id=scene.goal_id)
    )

    class Host:
        def __init__(self, served, **kwargs):
            assert served.resolve() == workspace.resolve()
            self.workspace, self.goal = served, kwargs.get("goal")

        def publish_native_event(self, requested, event):
            scene.order.append("usage")
            assert requested == project.resolve() and event == {"source": "native_usage_read"}
            if scene.failure == "unexpected":
                raise RuntimeError(SECRET * 300)
            binding = NativeResearchBinding.read(requested)
            assert binding is not None and binding.workspace.resolve() == workspace.resolve()
            native_provenance = RequestProvenance(vendor=host, session=parent, goal_id=self.goal)
            goal = goals.attributed_goal(native_provenance)
            return read_session_usage(
                requested,
                binding,
                publish=lambda document: publish(document, native_provenance),
                goal_id=None if goal is None else str(goal.goal_id),
            )

        def exchange(self, document):
            scene.order.append(document["operation"])
            request = PortfolioResearchRequestDocument.model_validate(
                document
            ).to_operation_request()
            scene.last_request_goal = self.goal
            main_provenance = RequestProvenance(vendor=host, session=parent, goal_id=self.goal)
            if document["operation"] == "GOAL_TAKE":
                try:
                    answer = goals.operate(
                        request,
                        "EXTERNAL_AUTOMATION",
                        main_provenance,
                    )
                except ValueError as error:
                    answer = {"status": "REFUSED", "failure_code": str(error)}
            else:
                answer = scene.answer
            return answer, json.dumps(answer).encode()

        def selected_url(self, *_args):
            return None

    monkeypatch.setattr(client, "LocalResearchClient", Host)
    monkeypatch.chdir(nested)
    try:
        yield scene
    finally:
        observer.close()


def run(scene, capsys, document, *, goal=None):
    """Drive the public CLI's real before/after send callbacks through a request file."""
    path = scene.project / "request.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    code = cli.main(
        [
            "--workspace",
            str(scene.workspace),
            "--view",
            "full",
            *([] if goal is None else ["--goal", goal]),
            "request",
            "--file",
            str(path),
        ],
        serve=lambda _args: pytest.fail("No server is launched by this seam."),
    )
    captured = capsys.readouterr()
    return code, json.loads(captured.out), captured.err


def open_another_goal(scene):
    """Open another real Goal through the existing public owner, with its own identity."""
    original = scene.goals.store.head(UUID(scene.goal_id))
    assert original is not None
    return scene.goals.operate(
        PortfolioResearchRequestDocument(
            operation="GOAL_OPEN",
            goal_id=str(uuid4()),
            change_reason="Synthetic second Goal",
            goal_declaration=original.declaration.model_dump(mode="json"),
        ).to_operation_request(),
        "HUMAN",
        None,
    )["goal_id"]


def test_lead_reading_from_nested_cwd_is_independent_of_child_stop_and_idempotent(lead_scene):
    """BEHAVIOUR: one own-session reading needs no lifecycle event and retries one receipt."""
    scene = lead_scene
    first = lead_readings(scene.nested, dict(os.environ), workspace=scene.workspace)
    assert first[0]["status"] == "DELIVERED"
    again = lead_readings(scene.nested, dict(os.environ), workspace=scene.workspace)
    assert again == first
    (row,) = scene.observer.read_external(ExternalActivityReadQuery())["items"]
    assert row["payload"]["event_kind"] == "NATIVE_AGENT_USAGE"
    subject = row["payload"]["subject"]
    assert (subject["native_agent_id"], subject["role"]) == (scene.parent, "research_lead")
    assert (subject["responses"], subject["input_tokens"], subject["output_tokens"]) == (
        "1",
        "15",
        "3",
    )
    assert SECRET not in json.dumps(row) and str(scene.session) not in json.dumps(row)


def test_lead_workspace_mismatch_and_off_never_read_or_deliver(lead_scene, monkeypatch):
    """BEHAVIOUR: the actual workspace and privacy switch are checked before session reading."""
    scene = lead_scene

    def no_read(*_args, **_kwargs):
        pytest.fail("A refused workspace or OFF binding must not read usage.")

    monkeypatch.setattr(native_bridge, "read_session", no_read)
    for environment in (
        {},
        {"CODEX_THREAD_ID": "other-session"},
        {"CLAUDE_CODE_SESSION_ID": "other-session"},
        {"CODEX_THREAD_ID": scene.parent, "CLAUDE_CODE_SESSION_ID": scene.parent},
    ):
        assert lead_readings(scene.nested, environment, workspace=scene.workspace) == []
    assert lead_readings(scene.nested, dict(os.environ), workspace=scene.project) == [
        {"status": "UNAVAILABLE", "reason": "native_bridge.workspace_mismatch"}
    ]
    binding = json.loads(scene.binding_path.read_text(encoding="utf-8"))
    scene.binding_path.write_text(json.dumps({**binding, "usage": "OFF"}), encoding="utf-8")
    assert lead_readings(scene.nested, dict(os.environ), workspace=scene.workspace) == []
    owned = NativeResearchBinding.read(scene.project)
    assert owned is not None
    assert read_session_usage(
        scene.project, owned, publish=lambda _event: pytest.fail("OFF must not publish.")
    ) == {"status": "SKIPPED", "reason": "native_bridge.usage_disabled"}
    assert (
        scene.order == []
        and scene.observer.read_external(ExternalActivityReadQuery())["items"] == []
    )


@pytest.mark.parametrize("found", [False, True])
def test_goal_take_reads_after_the_successful_owner_take(lead_scene, capsys, found):
    """BEHAVIOUR: the owner must bind the Goal before its lead's reading enters the ledger."""
    scene = lead_scene
    code, answer, error = run(
        scene,
        capsys,
        {
            "operation": "GOAL_TAKE",
            "goal_id": scene.goal_id if found else str(uuid4()),
        },
    )
    assert (code, answer["status"]) == ((0, "GOAL_TAKEN") if found else (2, "REFUSED"))
    assert scene.order == (["GOAL_TAKE", "usage"] if found else ["GOAL_TAKE"])
    if found:
        (row,) = scene.observer.read_external(ExternalActivityReadQuery())["items"]
        assert row["payload"]["subject"]["goal_id"] == scene.goal_id
        (entry,) = scene.goals.store.attributed(UUID(scene.goal_id))
        assert entry["event_kind"] == "NATIVE_AGENT_USAGE" and entry["agent_id"] == scene.parent
        assert error == ""
    else:
        assert scene.observer.read_external(ExternalActivityReadQuery())["items"] == []


@pytest.mark.parametrize("operation", ["GOAL_SUBMIT", "AGENT_ANSWER_SUBMIT"])
def test_explicit_goal_receives_its_snapshot_while_the_parent_keeps_another_goal(
    lead_scene,
    capsys,
    operation,
):
    """BEHAVIOUR: CLI Goal provenance reaches usage and the request without rebinding its parent."""
    scene = lead_scene
    first_code, first_answer, first_error = run(
        scene, capsys, {"operation": "GOAL_TAKE", "goal_id": scene.goal_id}
    )
    assert (first_code, first_answer["status"], first_error) == (0, "GOAL_TAKEN", "")
    other = open_another_goal(scene)
    document = (
        {
            "operation": operation,
            "goal_id": other,
            "goal_submission": {
                "outcome": "PARTLY_ACHIEVED",
                "summary": "Synthetic exact Goal seam.",
                "criteria": [{"criterion_id": "read", "answer": "NOT_ASSESSED"}],
            },
        }
        if operation == "GOAL_SUBMIT"
        else {
            "operation": operation,
            "bundle_directory": str(scene.project / "bundle"),
            "agent_answer": {"text": "Synthetic exact Goal answer seam."},
        }
    )
    scene.order.clear()
    code, answer, error = run(scene, capsys, document, goal=other)
    assert code == 0 and answer["data"] == scene.answer and error == ""
    assert scene.order == ["usage", operation] and scene.last_request_goal == other
    held = scene.goals.attributed_goal(RequestProvenance(vendor=scene.host, session=scene.parent))
    assert held is not None and str(held.goal_id) == scene.goal_id
    rows = scene.observer.read_external(ExternalActivityReadQuery())["items"]
    assert {row["payload"]["subject"]["goal_id"] for row in rows} == {scene.goal_id, other}
    (entry,) = scene.goals.store.attributed(UUID(other))
    assert entry["event_kind"] == "NATIVE_AGENT_USAGE" and entry["agent_id"] == scene.parent
    assert (entry["responses"], entry["input_tokens"], entry["output_tokens"]) == ("1", "15", "3")
    assert run(scene, capsys, document, goal=other)[0] == 0
    assert scene.observer.read_external(ExternalActivityReadQuery())["items"] == rows
    assert scene.goals.store.attributed(UUID(other)) == (entry,)


def test_a_later_goal_take_gets_a_new_snapshot_with_unchanged_session_counts(lead_scene, capsys):
    """BEHAVIOUR: exact snapshots retry within one Goal and retain both original Goal bindings."""
    scene = lead_scene
    assert run(scene, capsys, {"operation": "GOAL_TAKE", "goal_id": scene.goal_id})[0] == 0
    (first,) = scene.observer.read_external(ExternalActivityReadQuery())["items"]
    other = open_another_goal(scene)
    code, answer, error = run(scene, capsys, {"operation": "GOAL_TAKE", "goal_id": other})
    assert (code, answer["status"], error) == (0, "GOAL_TAKEN", "")
    rows = scene.observer.read_external(ExternalActivityReadQuery())["items"]
    assert len(rows) == 2 and first in rows
    by_goal = {row["payload"]["subject"]["goal_id"]: row for row in rows}
    assert set(by_goal) == {scene.goal_id, other}
    original, current = by_goal[scene.goal_id], by_goal[other]
    assert original["source_sequence"] != current["source_sequence"]
    assert original["observation_id"] != current["observation_id"]
    for name in (
        "model",
        "responses",
        "input_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "output_tokens",
        "last_at",
    ):
        assert original["payload"]["subject"][name] == current["payload"]["subject"][name]
    assert current["payload"]["subject"]["native_agent_id"] == scene.parent
    assert run(scene, capsys, {"operation": "GOAL_TAKE", "goal_id": other})[0] == 0
    assert scene.observer.read_external(ExternalActivityReadQuery())["items"] == rows
    for goal_id in (scene.goal_id, other):
        (entry,) = scene.goals.store.attributed(UUID(goal_id))
        assert entry["event_kind"] == "NATIVE_AGENT_USAGE" and entry["agent_id"] == scene.parent


@pytest.mark.parametrize("change", ["timestamp", "effort"])
def test_a_valid_later_response_block_forms_a_new_snapshot_without_counting_it_twice(
    lead_scene,
    capsys,
    change,
):
    """BEHAVIOUR: legitimate source metadata changes do not collide with the earlier snapshot."""
    scene = lead_scene
    assert run(scene, capsys, {"operation": "GOAL_TAKE", "goal_id": scene.goal_id})[0] == 0
    (first,) = scene.observer.read_external(ExternalActivityReadQuery())["items"]
    records = [json.loads(line) for line in scene.session.read_text(encoding="utf-8").splitlines()]
    later = records[-1]
    updates = []
    if change == "timestamp":
        later["timestamp"] = (AT + timedelta(seconds=7)).isoformat()
    elif scene.host == "codex":
        updates.append(
            {"type": "turn_context", "payload": {"model": "gpt-6-luna", "effort": "xhigh"}}
        )
    else:
        later["effort"] = "xhigh"
    updates.append(later)
    with scene.session.open("a", encoding="utf-8") as stream:
        stream.write("\n".join(json.dumps(row) for row in updates) + "\n")
    receipt = lead_readings(
        scene.nested, dict(os.environ), workspace=scene.workspace, goal=scene.goal_id
    )
    assert receipt[0]["status"] == "DELIVERED"
    rows = scene.observer.read_external(ExternalActivityReadQuery())["items"]
    assert len(rows) == 2 and first in rows
    (current,) = [row for row in rows if row["observation_id"] != first["observation_id"]]
    before, after = first["payload"]["subject"], current["payload"]["subject"]
    assert before["goal_id"] == after["goal_id"] == scene.goal_id
    for name in (
        "responses",
        "input_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "output_tokens",
    ):
        assert before[name] == after[name]
    assert after["responses"] == "1"
    if change == "timestamp":
        assert after["last_at"] == later["timestamp"] and after["last_at"] != before["last_at"]
    else:
        assert after["efforts"] == "xhigh" and after["efforts"] != before.get("efforts")
    assert (
        lead_readings(scene.nested, dict(os.environ), workspace=scene.workspace, goal=scene.goal_id)
        == receipt
    )
    assert scene.observer.read_external(ExternalActivityReadQuery())["items"] == rows
    assert all(row["payload"]["event_kind"] == "NATIVE_AGENT_USAGE" for row in rows)
    assert SECRET not in json.dumps(rows)
    goal = scene.goals.store.head(UUID(scene.goal_id))
    assert goal is not None
    (session_usage,) = scene.goals.record(goal)["session_usage"]
    assert session_usage["aggregation"] == "NOT_COMBINED" and "by_model" not in session_usage
    (participant,) = session_usage["participants"]
    (model,) = participant["models"]
    assert (model["responses"], model["input_tokens"], model["output_tokens"]) == (1, 15, 3)


@pytest.mark.parametrize("operation", ["GOAL_SUBMIT", "AGENT_ANSWER_SUBMIT"])
@pytest.mark.parametrize(
    "failure,reason",
    [
        (None, None),
        ("missing", "native_bridge.lead_usage_file_missing"),
        ("read", "native_bridge.lead_usage_read_failed"),
        ("delivery", "activity.storage_unavailable"),
        ("unexpected", "native_bridge.lead_usage_read_failed"),
    ],
)
def test_submit_reads_before_send_and_usage_failure_preserves_cli_output(
    lead_scene,
    capsys,
    monkeypatch,
    operation,
    failure,
    reason,
):
    """BEHAVIOUR: optional lead observation never blocks the exact successful owner answer."""
    scene = lead_scene
    scene.failure = failure
    if failure == "missing":
        scene.session.unlink()
    elif failure == "read":

        def unreadable(*_args, **_kwargs):
            raise OSError(SECRET * 300)

        monkeypatch.setattr(native_bridge, "read_session", unreadable)
    document = (
        {
            "operation": operation,
            "goal_id": scene.goal_id,
            "goal_submission": {
                "outcome": "PARTLY_ACHIEVED",
                "summary": "Synthetic completion seam.",
                "criteria": [
                    {"criterion_id": "read", "answer": "NOT_ASSESSED", "note": "Synthetic only."}
                ],
            },
        }
        if operation == "GOAL_SUBMIT"
        else {
            "operation": operation,
            "bundle_directory": str(scene.project / "bundle"),
            "agent_answer": {"text": "Synthetic accepted answer seam."},
        }
    )
    code, answer, error = run(scene, capsys, document)
    assert code == 0 and answer["status"] == "ACCEPTED" and answer["data"] == scene.answer
    assert scene.order == ["usage", operation]
    assert SECRET not in error and len(error) < 3000
    rows = scene.observer.read_external(ExternalActivityReadQuery())["items"]
    if failure is None:
        assert error == "" and len(rows) == 1
    else:
        assert rows == []
        diagnostic = json.loads(error)["native_observation"]
        assert diagnostic["status"] == "UNAVAILABLE" and diagnostic["phase"] == "LEAD_USAGE"
        assert diagnostic["operation"] == operation and diagnostic["reason"] == reason
        assert diagnostic["detail"] and diagnostic["next_action"]


def _write_native(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8", newline="\n"
    )


def _own_child(scene):
    child = str(uuid4()) if scene.host == "codex" else "synthetic-child-" + uuid4().hex
    records = [json.loads(line) for line in scene.session.read_text("utf-8").splitlines()]
    if scene.host == "codex":
        path = scene.session.with_name(scene.session.name.replace(scene.parent, child))
        records[0] = {
            "type": "session_meta",
            "payload": {
                "id": child,
                "source": {
                    "subagent": {
                        "thread_spawn": {
                            "parent_thread_id": scene.parent,
                            "agent_role": "alphalattice_risk",
                        }
                    }
                },
            },
        }
    else:
        path = scene.session.parent / scene.parent / "subagents" / f"agent-{child}.jsonl"
        records.insert(
            0,
            {
                "type": "user",
                "sessionId": scene.parent,
                "agentId": child,
                "isSidechain": True,
                "message": {"content": SECRET},
            },
        )
    _write_native(path, records)
    if scene.host == "codex":
        # The lead's own rollout names the child it started; nothing else selects it.
        item = {"type": "SubAgentActivity", "kind": "started", "agent_thread_id": child}
        with scene.session.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {"type": "event_msg", "payload": {"type": "item_completed", "item": item}}
                )
                + "\n"
            )
    if scene.host == "claude-code":
        # The host's retained sidecar supplies role and direct spawn depth; neither is
        # inferred from the filename, assigned role or a card's model pin.
        path.with_suffix(".meta.json").write_text(
            json.dumps(
                {
                    "agentType": "alphalattice_risk",
                    "spawnDepth": 1,
                    "toolUseId": "synthetic-tool-use",
                    "description": SECRET,
                }
            ),
            encoding="utf-8",
            newline="\n",
        )
    return child, path


def _read_owned_usage(scene, participant, child=None):
    """One reading of the bound Session; the named participant's own entry from it."""
    binding = NativeResearchBinding.read(scene.project)
    assert binding is not None
    result = read_session_usage(
        scene.project, binding, publish=scene.publish, goal_id=scene.goal_id
    )
    if result["status"] == "SKIPPED":
        return result
    agent = scene.parent if participant == "lead" else child
    return next((m for m in result["participants"] if m.get("agent_id") == agent), None)


def _rows(scene, agent):
    """The observations filed for one participant."""
    return [
        row
        for row in scene.observer.read_external(ExternalActivityReadQuery())["items"]
        if row["payload"]["subject"]["native_agent_id"] == agent
    ]


def test_a_child_usage_snapshot_needs_exact_parent_binding_but_no_lifecycle_hook(lead_scene):
    """BEHAVIOUR: own child counts are real and idempotent without claiming hook authorship."""
    scene = lead_scene
    child, path = _own_child(scene)
    first = _read_owned_usage(scene, "child", child)
    assert first == {
        "agent_id": child,
        "role": "alphalattice_risk",
        "status": "DELIVERED",
        "models": 1,
    }
    assert _read_owned_usage(scene, "child", child) == first
    (row,) = _rows(scene, child)
    assert row["payload"]["event_kind"] == "NATIVE_AGENT_USAGE"
    assert row["authority"] == "AGENT_PROPOSAL"
    subject = row["payload"]["subject"]
    assert (subject["native_session_id"], subject["native_agent_id"], subject["role"]) == (
        scene.parent,
        child,
        "alphalattice_risk",
    )
    assert (subject["responses"], subject["input_tokens"], subject["output_tokens"]) == (
        "1",
        "15",
        "3",
    )
    assert subject["input_channel"] == (
        "CODEX_SESSION_FILE" if scene.host == "codex" else "CLAUDE_CODE_SESSION_FILE"
    )
    assert "native_hook_event" not in subject and "authorship_basis" not in subject
    assert SECRET not in json.dumps(row) and str(path) not in json.dumps(row)
    assert subject["goal_id"] == scene.goal_id


def test_a_child_usage_delivery_failure_is_visible_and_retry_keeps_the_same_reading(lead_scene):
    """BEHAVIOUR: unavailable optional transport leaves no invented delivery or native text."""
    scene = lead_scene
    child, _path = _own_child(scene)
    scene.failure = "delivery"
    result = _read_owned_usage(scene, "child", child)
    assert result["status"] == "UNAVAILABLE"
    assert result["reason"] == "activity.storage_unavailable"
    assert scene.observer.read_external(ExternalActivityReadQuery())["items"] == []
    assert SECRET not in json.dumps(result)
    scene.failure = None
    first = _read_owned_usage(scene, "child", child)
    assert first["status"] == "DELIVERED" and _read_owned_usage(scene, "child", child) == first
    assert len(_rows(scene, child)) == 1


@pytest.mark.parametrize("lead_scene", ["codex"], indirect=True)
@pytest.mark.parametrize("mismatch", ["parent", "role", "missing_parent", "missing_role", "id"])
def test_a_codex_child_usage_refuses_a_missing_or_conflicting_recorded_binding(
    lead_scene, mismatch
):
    """BEHAVIOUR: a nominated id cannot substitute for its own exact recorded parent and role."""
    scene = lead_scene
    child, path = _own_child(scene)
    records = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
    meta = records[0]["payload"]
    spawn = meta["source"]["subagent"]["thread_spawn"]
    if mismatch == "parent":
        spawn["parent_thread_id"] = "another-parent"
    elif mismatch == "role":
        spawn["agent_role"] = "alphalattice_factor"
    elif mismatch == "missing_parent":
        del spawn["parent_thread_id"]
    elif mismatch == "missing_role":
        del spawn["agent_role"]
    else:
        meta["id"] = "another-child"
    _write_native(path, records)
    assert _read_owned_usage(scene, "child", child) == {
        "agent_id": child,
        "status": "UNAVAILABLE",
        "reason": "native_bridge.child_usage_binding_unverified",
    }
    assert _rows(scene, child) == []


@pytest.mark.parametrize("lead_scene", ["claude-code"], indirect=True)
def test_a_claude_child_usage_reads_only_the_bound_parents_own_subagent_file(lead_scene):
    """BEHAVIOUR: the same parent id under another native project names no child of it."""
    scene = lead_scene
    child, path = _own_child(scene)
    foreign = scene.session.parent.parent / "other-project" / scene.parent / "subagents" / path.name
    foreign.parent.mkdir(parents=True)
    foreign.write_bytes(path.read_bytes())
    path.unlink()
    assert _read_owned_usage(scene, "child", child) is None
    assert _rows(scene, child) == []


@pytest.mark.parametrize("participant", ["lead", "child"])
@pytest.mark.parametrize("model_count", [32, 33])
def test_an_optional_usage_snapshot_publishes_all_allowed_models_or_no_overflow_rows(
    lead_scene, participant, model_count
):
    """BEHAVIOUR: model overflow cannot publish a plausible truncated set of cumulative totals."""
    scene = lead_scene
    child, path = _own_child(scene) if participant == "child" else (None, scene.session)
    existing = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
    records = existing[:1] if scene.host == "codex" or participant == "child" else []
    for number in range(model_count):
        model = f"synthetic-model-{number}"
        if scene.host == "codex":
            records.extend(
                [
                    {"type": "turn_context", "payload": {"model": model, "effort": "high"}},
                    {
                        **existing[-1],
                        "payload": {
                            **existing[-1]["payload"],
                            "response_id": f"synthetic-response-{number}",
                        },
                    },
                ]
            )
        else:
            records.append(
                {
                    **existing[-1],
                    "message": {
                        **existing[-1]["message"],
                        "id": f"synthetic-response-{number}",
                        "model": model,
                    },
                }
            )
    _write_native(path, records)
    result = _read_owned_usage(scene, participant, child)
    rows = _rows(scene, scene.parent if participant == "lead" else child)
    if model_count == 33:
        assert result["status"] == "UNAVAILABLE" and result["read_limit"] == "models"
        assert result["reason"] == f"native_bridge.{participant}_usage_read_failed"
        assert "models" not in result and rows == []
    else:
        assert result["status"] == "DELIVERED" and result["models"] == 32
        assert len(rows) == 32
        assert {row["payload"]["subject"]["model"] for row in rows} == {
            f"synthetic-model-{number}" for number in range(32)
        }
        assert all(row["payload"]["event_kind"] == "NATIVE_AGENT_USAGE" for row in rows)
        assert all(row["payload"]["subject"]["responses"] == "1" for row in rows)
        assert all(row["payload"]["subject"]["input_tokens"] == "15" for row in rows)
        assert all(row["payload"]["subject"]["output_tokens"] == "3" for row in rows)
    assert SECRET not in json.dumps(result) + json.dumps(rows)


@pytest.mark.parametrize("participant", ["lead", "child"])
@pytest.mark.parametrize("limit", ["max_bytes", "max_line_bytes"])
def test_an_overbound_native_usage_scan_publishes_none_of_its_complete_prefix(
    lead_scene, monkeypatch, participant, limit
):
    """BEHAVIOUR: both byte limits reject complete prefix counts before the observation owner."""
    scene = lead_scene
    child, path = _own_child(scene) if participant == "child" else (None, scene.session)
    records = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
    records.append({"type": "user", "payload": {"content": SECRET * 1000}})
    _write_native(path, records)
    original = path.read_bytes()
    # Calibrate only the public byte policy, keeping the real parser and publication owners.
    monkeypatch.setattr(
        native_bridge,
        "USAGE_READ_BYTES" if limit == "max_bytes" else "USAGE_LINE_BYTES",
        1024,
    )
    result = _read_owned_usage(scene, participant, child)
    assert result["status"] == "UNAVAILABLE" and result["read_limit"] == limit
    assert result["reason"] == f"native_bridge.{participant}_usage_read_failed"
    assert "models" not in result
    assert _rows(scene, scene.parent if participant == "lead" else child) == []
    assert path.read_bytes() == original and SECRET not in json.dumps(result)


def test_child_usage_off_keeps_no_usage_or_claimed_native_author(lead_scene):
    """BEHAVIOUR: an OFF binding preserves privacy even when no native file can be located."""
    scene = lead_scene
    child, path = _own_child(scene)
    value = json.loads(scene.binding_path.read_text("utf-8"))
    scene.binding_path.write_text(json.dumps({**value, "usage": "OFF"}), encoding="utf-8")
    scene.session.unlink()
    path.unlink()
    assert _read_owned_usage(scene, "child", child) == {
        "status": "SKIPPED",
        "reason": "native_bridge.usage_disabled",
    }
    assert scene.observer.read_external(ExternalActivityReadQuery())["items"] == []
