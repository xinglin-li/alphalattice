"""The Session's own optional readings and CLI ordering: the lead and the children its own
files record, with no hook, message or always-running reader."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from alphalattice.control.product_host.composition.goals import GoalApplication
from alphalattice.control.product_host.composition.workspace_activity import WorkspaceActivity
from alphalattice.control.product_host.publication.goals import GoalStore
from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.foundation.data_layer import history
from alphalattice.interface.local_application import cli, client
from alphalattice.interface.local_application.activity import (
    ExternalActivityEventDocument,
    ExternalActivityReadQuery,
)
from alphalattice.interface.local_application.cli_contract import RequestProvenance
from alphalattice.interface.local_application.client import lead_readings
from alphalattice.interface.local_application.native_bridge import (
    BINDING_NAME,
    NativeResearchBinding,
    read_session_usage,
    set_usage_reading,
)
from alphalattice.interface.local_application.native_setup import declare_project, unbind_session
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchRequestDocument,
)

SECRET = "SYNTHETIC-PRIVATE-TRANSCRIPT-AND-ERROR"
AT = datetime(2026, 10, 6, tzinfo=UTC)


@pytest.fixture(params=["codex", "claude-code"])
def lead_scene(request, tmp_path, monkeypatch, recording_history):
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
    monkeypatch.setenv("CODEX_THREAD_ID" if host == "codex" else "CLAUDE_CODE_SESSION_ID", parent)
    session = project / "never-read-native.jsonl"
    session.write_text(SECRET, encoding="utf-8")
    observer = WorkspaceActivity(
        workspace=workspace,
        workspace_id=recording_history,
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

    scene.telemetry = [_sample(scene, parent)]

    def observed(vendor, session_id, *, workspace_id):
        assert workspace_id == "fixture-workspace"
        return [
            row
            for row in scene.telemetry
            if row["vendor"] == vendor
            and session_id in (row["conversation_id"], row["session_id"], row["parent_agent_id"])
        ]

    monkeypatch.setattr(history, "agent_telemetry", observed)

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
                workspace_id="fixture-workspace",
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
    declared = original.declaration.model_dump(mode="json")
    return scene.goals.operate(
        PortfolioResearchRequestDocument(
            operation="GOAL_OPEN",
            change_reason="Synthetic second Goal",
            goal_declaration={**declared, "title": f"{declared['title']} (another)"},
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
    assert again[0]["status"] == "DELIVERED"
    for reading in (first[0], again[0]):
        assert reading["source"] == "DUCKLAKE_OTLP"
        (member,) = reading["participants"]
        assert (member["agent_id"], member["role"], member["status"], member["models"]) == (
            scene.parent,
            "research_lead",
            "DELIVERED",
            1,
        )
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

    monkeypatch.setattr(history, "agent_telemetry", no_read)
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
    later = scene.telemetry[0].copy()
    later["event_id"] = "later"
    if change == "timestamp":
        later["time_ns"] = int((AT + timedelta(seconds=7)).timestamp() * 1e9)
    else:
        later["effort"] = "xhigh"
    scene.telemetry.append(later)
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
        assert (
            after["last_at"] == (AT + timedelta(seconds=7)).isoformat()
            and after["last_at"] != before["last_at"]
        )
    else:
        assert after["efforts"] == "xhigh" and after["efforts"] != before.get("efforts")
    (warm,) = lead_readings(
        scene.nested, dict(os.environ), workspace=scene.workspace, goal=scene.goal_id
    )
    assert warm["status"] == "DELIVERED" and warm["source"] == "DUCKLAKE_OTLP"
    for reading in (receipt[0], warm):
        assert reading["source"] == "DUCKLAKE_OTLP"
        (member,) = reading["participants"]
        assert (member["agent_id"], member["role"], member["status"], member["models"]) == (
            scene.parent,
            "research_lead",
            "DELIVERED",
            1,
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
        ("missing", "native_bridge.lead_usage_not_observed"),
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
        scene.telemetry.clear()
    elif failure == "read":

        def unreadable(*_args, **_kwargs):
            raise OSError(SECRET * 300)

        monkeypatch.setattr(history, "agent_telemetry", unreadable)
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


def _sample(scene, agent):
    return {
        "event_id": agent,
        "vendor": scene.host,
        "signal": "logs",
        "name": "api_request",
        "conversation_id": scene.parent if scene.host == "codex" else None,
        "session_id": scene.parent if scene.host == "claude-code" else None,
        "agent_id": agent,
        "parent_agent_id": None,
        "agent_role": None,
        "model": "gpt-6-luna" if scene.host == "codex" else "claude-sonnet-5-5",
        "effort": "high",
        "response_id": "synthetic-response",
        "time_ns": int(AT.timestamp() * 1e9),
        "input_tokens": 20 if scene.host == "codex" else 15,
        "uncached_input_tokens": 15,
        "cache_read_tokens": 5,
        "cache_write_tokens": 0,
        "output_tokens": 3,
        "model_seconds": 0.25,
    }


def _own_child(scene):
    child = str(uuid4())
    row = _sample(scene, child) | {
        "parent_agent_id": scene.parent,
        "agent_role": "alphalattice_risk",
    }
    scene.telemetry.append(row)
    return child, scene.session


def _read_owned_usage(scene, participant, child=None):
    """One reading of the bound Session; the named participant's own entry from it."""
    binding = NativeResearchBinding.read(scene.project)
    assert binding is not None
    result = read_session_usage(
        scene.project,
        binding,
        workspace_id="fixture-workspace",
        publish=scene.publish,
        goal_id=scene.goal_id,
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
    """Explicit native lineage supplies counts, never member-key or authorship authority."""
    scene = lead_scene
    child, path = _own_child(scene)
    first = _read_owned_usage(scene, "child", child)
    warm = _read_owned_usage(scene, "child", child)
    assert first == warm
    assert (first["agent_id"], first["role"], first["status"], first["models"]) == (
        child,
        "alphalattice_risk",
        "DELIVERED",
        1,
    )
    assert first["specialist_verified"] is False
    (row,) = _rows(scene, child)
    subject = row["payload"]["subject"]
    assert (
        row["payload"]["event_kind"] == "NATIVE_AGENT_USAGE"
        and row["authority"] == "AGENT_PROPOSAL"
    )
    assert (subject["native_session_id"], subject["native_agent_id"], subject["observed_role"]) == (
        scene.parent,
        child,
        "alphalattice_risk",
    )
    assert subject["specialist_verified"] == "false" and "role" not in subject
    assert (subject["responses"], subject["input_tokens"], subject["output_tokens"]) == (
        "1",
        "15",
        "3",
    )
    assert subject["input_channel"] == (
        "CODEX_OTLP" if scene.host == "codex" else "CLAUDE_CODE_OTLP"
    )
    assert subject["goal_id"] == scene.goal_id
    assert "native_hook_event" not in subject and "authorship_basis" not in subject
    assert SECRET not in json.dumps(row) and str(path) not in json.dumps(row)
    assert set_usage_reading(scene.workspace, enabled=False)["usage_reading"] == "OFF"
    assert _read_owned_usage(scene, "child", child)["status"] == "SKIPPED"
    assert set_usage_reading(scene.workspace, enabled=True)["usage_reading"] == "READ"
    assert _read_owned_usage(scene, "child", child)["status"] == "DELIVERED"
    assert (
        unbind_session(scene.project, session_id=scene.parent, host=scene.host)["status"]
        == "DETACHED"
    )
    assert not (scene.workspace / "runtime/native-usage").exists()


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
    warm = _read_owned_usage(scene, "child", child)
    assert first["status"] == "DELIVERED" and warm == first
    assert "bytes_read" not in warm
    assert len(_rows(scene, child)) == 1


@pytest.mark.parametrize("lead_scene", ["codex"], indirect=True)
@pytest.mark.parametrize("mismatch", ["parent", "role", "missing_parent", "missing_role", "id"])
def test_a_codex_child_usage_requires_its_recorded_parent_and_verifies_its_role_separately(
    lead_scene, mismatch
):
    scene = lead_scene
    child, _path = _own_child(scene)
    row = scene.telemetry[-1]
    if mismatch == "parent":
        row["parent_agent_id"] = "another-parent"
    elif mismatch == "missing_parent":
        row["parent_agent_id"] = None
    elif mismatch == "role":
        row["agent_role"] = "alphalattice_factor"
    elif mismatch == "missing_role":
        row["agent_role"] = None
    else:
        row["agent_id"] = "another-child"
    reading = _read_owned_usage(scene, "child", child)
    if mismatch in {"role", "missing_role"}:
        role = "alphalattice_factor" if mismatch == "role" else None
        assert (reading["agent_id"], reading["role"], reading["status"], reading["models"]) == (
            child,
            role,
            "DELIVERED",
            1,
        )
        assert reading["specialist_verified"] is False
        (filed,) = _rows(scene, child)
        subject = filed["payload"]["subject"]
        assert subject["specialist_verified"] == "false" and "role" not in subject
        assert subject.get("observed_role") == role
        assert (subject["responses"], subject["input_tokens"], subject["output_tokens"]) == (
            "1",
            "15",
            "3",
        )
    else:
        assert reading is None or (reading["status"], reading["reason"]) == (
            "UNAVAILABLE",
            "native_bridge.child_usage_binding_unverified",
        )
        assert _rows(scene, child) == []


@pytest.mark.parametrize("lead_scene", ["claude-code"], indirect=True)
def test_a_claude_child_usage_reads_only_the_bound_parents_own_subagent_file(lead_scene):
    """Another conversation supplies no counts to this bound session."""
    scene = lead_scene
    child, _path = _own_child(scene)
    scene.telemetry[-1].update(session_id="another-session", parent_agent_id=None)
    assert _read_owned_usage(scene, "child", child) is None
    assert _rows(scene, child) == []


@pytest.mark.parametrize("participant", ["lead", "child"])
@pytest.mark.parametrize(
    "model_count,missing",
    [
        pytest.param(32, None, id="all-models"),
        pytest.param(33, None, id="overflow"),
        pytest.param(1, "input_tokens", id="missing-input"),
        pytest.param(1, "output_tokens", id="missing-output"),
    ],
)
def test_an_optional_usage_snapshot_publishes_all_allowed_models_or_no_overflow_rows(
    lead_scene, participant, model_count, missing
):
    """A complete snapshot publishes all its models; overflow or incomplete counts files none."""
    scene = lead_scene
    child, _path = _own_child(scene) if participant == "child" else (None, scene.session)
    agent = child or scene.parent
    own = next(row for row in scene.telemetry if row["agent_id"] == agent)
    scene.telemetry[:] = [row for row in scene.telemetry if row["agent_id"] != agent]
    scene.telemetry.extend(
        own
        | {
            "model": f"synthetic-model-{n}",
            "response_id": f"r{n}",
            "event_id": f"event-{agent}-{n}",
        }
        for n in range(model_count)
    )
    if missing:
        scene.telemetry.append(
            own | {"response_id": "incomplete", "event_id": "incomplete", missing: None}
        )
    result = _read_owned_usage(scene, participant, child)
    rows = _rows(scene, scene.parent if participant == "lead" else child)
    if missing:
        assert result["status"] == "UNAVAILABLE"
        assert result["reason"] == f"native_bridge.{participant}_usage_incomplete"
        assert "models" not in result and rows == []
    elif model_count == 33:
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
