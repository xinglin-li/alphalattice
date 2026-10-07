"""Card 31: product operations become visible activity, proved against the real Host.

The Host tests boot the actual composition (workspace session, dispatcher,
observation ledger, loopback HTTP) and drives it the way a person or Codex does:
through the maintained CLI and the same routes the workbench reads. The synthetic
Portfolio resolver stands in for numerical work. The receipt-only test exercises
the public observer without admitting or running a scientific Task.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from alphalattice.control.observation_runtime.adapters import safe_observation_draft
from alphalattice.control.observation_runtime.contracts import (
    ObservationAuthority,
    ObservationRetentionClass,
)
from alphalattice.control.observation_runtime.ledger import ObservationStorageError
from alphalattice.control.observation_runtime.policy import (
    EXTERNAL_ACTIVITY_FIELDS,
    EXTERNAL_ACTIVITY_MAX_INLINE_BYTES,
    EXTERNAL_ACTIVITY_SCHEMA,
    ObservationSchemaPolicy,
    default_observation_policies,
)
from alphalattice.control.product_host.composition.local_web_session import (
    LocalPortfolioWebSession,
)
from alphalattice.control.product_host.composition.workspace_activity import WorkspaceActivity
from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.interface.local_application.activity import (
    OBSERVED_OPERATIONS,
    READ_OPERATIONS,
    RECORDS_ITSELF,
    ActivityReadQuery,
)
from alphalattice.interface.local_application.cli_contract import refusal_words
from alphalattice.interface.local_application.client import LocalResearchClient
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchRequestDocument,
)
from tests.portfolio_strategy_lab.local_web_support import _json, _request

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/run_alphalattice.py"


def test_answer_receipt_preserves_sealed_refs_without_creating_a_scientific_task(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    observer = WorkspaceActivity(
        workspace=workspace,
        workspace_id="synthetic-answer-receipt",
        gate=WorkspaceMutationGate(),
        instance="synthetic-answer-receipt",
        clock=lambda: datetime(2026, 10, 5, 1, 10, tzinfo=UTC),
    )
    request = PortfolioResearchRequestDocument.model_validate(
        {
            "operation": "AGENT_ANSWER_SUBMIT",
            "bundle_directory": str(tmp_path / "bundle"),
            "agent_answer": {"text": "A bounded interpretation.", "references": []},
        }
    ).to_operation_request()
    try:
        span = observer.entered(request, caller="EXTERNAL_AUTOMATION")
        assert span is not None
        observer.returned(
            span,
            {
                "answer_reference": "a" * 64,
                "bundle_reference": "b" * 64,
                "text": "Owner prose is not retained in operation receipts.",
            },
        )
        page = observer.read(ActivityReadQuery())
        requested, returned = page["items"]
        assert requested["payload"]["phase"] == "REQUESTED"
        assert returned["payload"]["phase"] == "RETURNED"
        assert returned["payload"]["subject"] == {
            "answer_reference": "a" * 64,
            "bundle_reference": "b" * 64,
        }
        assert returned["payload"].get("status") is None
        assert returned["task_id"] is None
        assert page["tasks"] == {}
        assert "Owner prose" not in json.dumps(page)
    finally:
        observer.close()


@pytest.mark.parametrize("status", ["NOT_BOUND", "OFF", "OBSERVING", "STOPPED", "UNAVAILABLE"])
def test_native_usage_callback_exposes_only_owner_status_and_safe_reason(tmp_path, status):
    """Usage health exposes no callback identity, native path, content or supplied prose."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    calls = []

    def usage_state():
        calls.append("read")
        return {
            "status": status,
            "reason": "activity.observer_failed",
            "session_id": "PRIVATE-NATIVE-SESSION",
            "path": "C:/PRIVATE-NATIVE-SESSION/rollout.jsonl",
            "conversation": "PRIVATE-NATIVE-CONVERSATION",
            "detail": "PRIVATE-CALLBACK-DETAIL",
            "next_action": "PRIVATE-CALLBACK-ACTION",
        }

    observer = WorkspaceActivity(
        workspace=workspace,
        workspace_id="synthetic-usage-state",
        gate=WorkspaceMutationGate(),
        instance="synthetic-usage-state",
        native_usage_state=usage_state,
    )
    try:
        state = observer.observer_state()
        assert calls == ["read"]
        usage = state["native_usage"]
        assert usage["status"] == status and usage["reason"] == "activity.observer_failed"
        assert set(usage) <= {"status", "reason", "detail", "next_action"}
        words = refusal_words("activity.observer_failed")
        for field in ("detail", "next_action"):
            if field in usage:
                assert usage[field] == words[field]
        assert "PRIVATE-" not in json.dumps(state)
        assert state["status"] == state["recording"] == "OK"
        assert state["missing_observations"] == state["appends"] == 0
    finally:
        observer.close()


@pytest.mark.parametrize(
    "status,reason,expected_reason",
    [
        ("INVENTED_READY", None, None),
        ("OBSERVING", "PRIVATE-CALLBACK-REASON", "activity.failure_detail_withheld"),
    ],
)
def test_native_usage_callback_refuses_unknown_status_or_untyped_reason(
    tmp_path, status, reason, expected_reason
):
    """Unknown states and arbitrary reasons cannot become owner health or private text."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    observer = WorkspaceActivity(
        workspace=workspace,
        workspace_id="synthetic-usage-state",
        gate=WorkspaceMutationGate(),
        instance="synthetic-usage-state",
        native_usage_state=lambda: {"status": status, "reason": reason},
    )
    try:
        state = observer.observer_state()
        usage = state["native_usage"]
        assert usage["status"] == ("UNAVAILABLE" if status == "INVENTED_READY" else status)
        assert usage["reason"] == expected_reason
        assert "INVENTED_READY" not in json.dumps(state)
        assert "PRIVATE-CALLBACK-REASON" not in json.dumps(state)
        assert state["status"] == state["recording"] == "OK"
        assert state["missing_observations"] == 0
    finally:
        observer.close()


@pytest.mark.untyped_failure
def test_native_usage_callback_failure_is_isolated_from_recording_health(tmp_path):
    """A failed health read reveals its safe code without changing recording or counters."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    def failed_state():
        raise RuntimeError("PRIVATE-NATIVE-CALLBACK-EXCEPTION")

    observer = WorkspaceActivity(
        workspace=workspace,
        workspace_id="synthetic-usage-state",
        gate=WorkspaceMutationGate(),
        instance="synthetic-usage-state",
        native_usage_state=failed_state,
    )
    try:
        for _ in range(2):
            state = observer.observer_state()
            assert state["native_usage"]["status"] == "UNAVAILABLE"
            assert state["native_usage"]["reason"] == "activity.observer_failed"
            assert state["status"] == state["recording"] == "OK"
            assert state["missing_observations"] == state["appends"] == 0
            assert "PRIVATE-NATIVE-CALLBACK-EXCEPTION" not in json.dumps(state)
    finally:
        observer.close()


def _cli(live: LocalPortfolioWebSession, *arguments: str) -> tuple[int, dict[str, Any]]:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--workspace",
            str(live.workspace),
            "--view",
            "full",
            *arguments,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert live.web.application.external_token not in result.stdout  # type: ignore[union-attr]
    return result.returncode, json.loads(result.stdout)


def _items(page: dict[str, Any], **where: object) -> list[dict[str, Any]]:
    rows = []
    for item in page["items"]:
        payload = item.get("payload") or {}
        facts = {**payload, "schema_kind": item["schema_kind"]}
        if all(facts.get(key) == value for key, value in where.items()):
            rows.append(item)
    return rows


def _event(sequence: int, **overrides: object) -> dict[str, Any]:
    document: dict[str, Any] = {
        "event_kind": "NATIVE_TURN_STARTED",
        "producer_id": "codex-adapter",
        "producer_session": "session-a",
        "producer_sequence": sequence,
        "occurred_at": datetime(2026, 9, 14, 12, 0, sequence, tzinfo=UTC).isoformat(),
        "summary": "Main PM asked the Analyst for a bounded factor screen.",
        "subject": {},
    }
    document.update(overrides)
    return document


@pytest.mark.parametrize("character", ("x", "中", "😀"), ids=("ascii", "cjk", "astral"))
def test_full_external_document_fits_ledger_and_preserves_old_retries(live, character):
    client = LocalResearchClient(live.workspace)
    old = _event(0)
    first = client.publish_event(old)
    assert first["status"] == "APPENDED"
    document = _event(
        1,
        event_kind="A" * 64,
        producer_id="p" * 64,
        producer_session="s" * 64,
        producer_sequence=2**63 - 1,
        summary=character * 4000,
        subject={f"s{index:02d}" + "k" * 61: character * 200 for index in range(16)},
        correlation_ids=[character * 128 for _ in range(8)],
    )
    admitted = client.publish_event(document)
    assert admitted["status"] == "APPENDED" and admitted["summary_truncated"] is True
    assert client.publish_event(document)["status"] == "REUSED_EXACT"
    stored = _json(live, "/api/activity/external")["items"]
    assert len(stored) == 2
    narrow, widest = stored
    original_policy = ObservationSchemaPolicy(
        EXTERNAL_ACTIVITY_SCHEMA, 1, EXTERNAL_ACTIVITY_FIELDS, max_inline_bytes=4096
    )
    assert default_observation_policies().policy(EXTERNAL_ACTIVITY_SCHEMA, 1).policy_hash == (
        original_policy.policy_hash
    )
    assert widest["payload"]["summary"] == character * 500
    assert widest["payload"]["subject"] == document["subject"]
    size = len(json.dumps(widest["payload"], sort_keys=True, separators=(",", ":")).encode())
    assert size <= EXTERNAL_ACTIVITY_MAX_INLINE_BYTES
    # The read projection does not expose transport policy versions. Check the
    # public adapter that builds the ledger draft without changing the payload.
    for item, expected_version in ((narrow, 1), (widest, 1 if size <= 4096 else 2)):
        draft = safe_observation_draft(
            policies=default_observation_policies(),
            schema_kind=EXTERNAL_ACTIVITY_SCHEMA,
            occurred_at=datetime.now(UTC),
            source_kind="EXTERNAL_CLIENT",
            source_id="test-source",
            source_sequence=0,
            payload=item["payload"],
            authority=ObservationAuthority.AGENT_PROPOSAL,
            retention_class=ObservationRetentionClass.TRANSIENT_OPERATIONAL,
        )
        assert draft.schema_version == expected_version
        if expected_version == 1:
            assert draft.policy_hash == original_policy.policy_hash
    replay = client.publish_event(old)
    assert replay["status"] == "REUSED_EXACT"
    assert replay["observation_id"] == first["observation_id"]
    for changes, fields in (
        ({"producer_sequence": 2**63}, [["producer_sequence"]]),
        ({"subject": {"s": character * 201}}, [[]]),
        ({"summary": character * 4001}, [["summary"]]),
    ):
        refused = client.publish_event({**document, **changes})
        assert refused["status"] == "REFUSED"
        assert refused["failure_code"] == "activity.event_invalid"
        assert refused["fields"] == fields
    assert len(_json(live, "/api/activity/external")["items"]) == 2


def test_every_operation_in_the_vocabulary_is_classified_once() -> None:
    """A new operation must be placed deliberately: recorded activity or a read."""

    from alphalattice.interface.local_application.operations import OPERATIONS

    vocabulary = frozenset(OPERATIONS)
    parts = (READ_OPERATIONS, OBSERVED_OPERATIONS, RECORDS_ITSELF)
    assert sum(len(part) for part in parts) == len(frozenset().union(*parts))
    assert vocabulary == frozenset().union(*parts)
    assert {"PLAN", "RUN", "EXPERIMENT_PLAN", "EXPERIMENT_RUN", "CANCEL"} <= OBSERVED_OPERATIONS
    assert {"STATUS", "TASKS", "RESULTS", "REPORT", "EXPERIMENT_READBACK"} <= READ_OPERATIONS


def test_cli_preview_refusal_and_run_become_activity_with_the_real_task_and_result(
    live: LocalPortfolioWebSession, tmp_path: Path
) -> None:
    """The whole card-31 flow: PLAN without a Task, a typed refusal, an admitted RUN,
    its real Task lifecycle and the owner-verified result, all read the way the
    workbench and the CLI read them, from one bounded cursor."""

    start = _json(live, "/api/activity")
    assert start["disposition"] == "TAIL" and start["items"] == []
    assert start["observer"]["status"] == "OK"
    assert start["workspace_id"] == live.workspace_id
    epoch, cursor = start["epoch"], start["cursor"]

    # A PLAN is a preview: recorded, no Task, no invented lifecycle.
    spec = tmp_path / "spec.yaml"
    spec.write_text("{}", encoding="utf-8")
    code, planned = _cli(live, "strategy-book", "preview", "--file", str(spec))
    assert code == 0 and planned["data"]["spec_hash"]
    page = _json(live, f"/api/activity?after={cursor}")
    assert page["disposition"] == "CONTINUED" and page["epoch"] == epoch
    requested, returned = _items(page, operation="PLAN")
    assert requested["payload"]["phase"] == "REQUESTED"
    assert returned["payload"]["phase"] == "RETURNED"
    for item in (requested, returned):
        assert item["payload"]["caller"] == "EXTERNAL_AUTOMATION"
        assert item["task_id"] is None and item["run_id"] is None
        assert item["authority"] == "OPERATIONAL_ASSERTION"
        assert "spec" not in item["payload"]["subject"]  # content stays with its owner
    assert requested["payload"]["operation_ref"] == returned["payload"]["operation_ref"]
    assert returned["payload"]["subject"]["spec_hash"] == planned["data"]["spec_hash"]
    assert "next_read" not in returned["payload"]  # a preview has no readable object yet
    assert page["tasks"] == {}
    cursor = page["cursor"]

    # A meaningful refusal is recorded as the owner's own refusal, and nothing else.
    code, refused = _cli(live, "study", "run", "--plan", "a" * 64)
    assert code == 2 and refused["data"]["failure_code"] == "research_experiment.preview_required"
    page = _json(live, f"/api/activity?after={cursor}")
    (refusal,) = _items(page, operation="EXPERIMENT_RUN", phase="RETURNED")
    assert refusal["payload"]["status"] == "REFUSED"
    assert refusal["payload"]["failure_code"] == "research_experiment.preview_required"
    assert refusal["payload"]["subject"] == {"experiment_plan_hash": "a" * 64}
    assert refusal["task_id"] is None and page["tasks"] == {}
    assert not live.session.task_control_registry.tasks()  # type: ignore[union-attr]
    cursor = page["cursor"]

    # An admitted RUN names its real Task; ADMITTED is not a result.
    code, sent = _cli(live, "strategy-book", "run", "--file", str(spec))
    assert code == 3 and sent["data"]["disposition"] == "ADMITTED"
    task_id = sent["data"]["task_id"]
    page = _json(live, f"/api/activity?after={cursor}")
    (admitted,) = _items(page, operation="RUN", phase="RETURNED")
    assert admitted["payload"]["status"] == "ADMITTED"
    assert admitted["payload"]["task_id"] == task_id == admitted["task_id"]
    assert admitted["payload"]["task_lifecycle"] == "QUEUED"
    assert admitted["payload"]["next_read"] == {"operation": "STATUS", "task_id": task_id}
    assert admitted["run_id"] == f"local-web:{task_id}"
    assert task_id in admitted["correlation_ids"]
    assert not _items(page, schema_kind="ArtifactVerificationObserved")
    assert page["tasks"][task_id]["task_id"] == task_id
    assert page["tasks"][task_id]["lifecycle"] in {"QUEUED", "RUNNING", "SUCCEEDED"}
    cursor = page["cursor"]

    # Once the command returns, Task Control and the artifact owner are recorded.
    live.dispatcher.drain_for_tests()  # type: ignore[union-attr]
    page = _json(live, f"/api/activity?after={cursor}&watch={task_id}")
    (transition,) = _items(page, schema_kind="TaskControlTransition")
    assert transition["authority"] == "TASK_CONTROL_ASSERTION"
    assert transition["payload"]["task_lifecycle"] == "SUCCEEDED"
    assert transition["payload"]["task_class"] == "portfolio_public_development_replay"
    assert transition["payload"]["disposition"] == "COMMAND_RETURNED"
    assert "failure_code" not in transition["payload"]  # a clean return persists no failure
    assert "failure_type" not in transition["payload"]
    (artifact,) = _items(page, schema_kind="ArtifactVerificationObserved")
    assert artifact["authority"] == "ARTIFACT_ASSERTION"
    assert artifact["payload"]["artifact_kind"] == "PortfolioResearchResult"
    result_hash = artifact["payload"]["artifact_hash"]
    assert _json(live, "/api/results")["results"][0]["result_hash"] == result_hash
    assert _json(live, f"/api/report?result_hash={result_hash}")["result_hash"] == result_hash
    assert page["tasks"][task_id]["lifecycle"] == "SUCCEEDED"
    assert page["tasks"][task_id]["worker_failure"] is None
    assert [item["ordinal"] for item in page["items"]] == sorted(
        item["ordinal"] for item in page["items"]
    )
    cursor = page["cursor"]

    # The CLI reads the same feed through the client route, and reads record nothing.
    code, mirrored = _cli(live, "activity", "list", "--after", start["cursor"], "--limit", "100")
    assert code == 0
    assert [i["observation_id"] for i in mirrored["data"]["items"]][-2:] == [
        transition["observation_id"],
        artifact["observation_id"],
    ]
    head = _json(live, "/api/activity")["head"]
    for _ in range(3):
        _json(live, "/api/tasks")
        _json(live, f"/api/status?task_id={task_id}")
        _json(live, "/api/results")
    quiet = _json(live, f"/api/activity?after={cursor}")
    assert quiet["items"] == [] and quiet["head"] == head
    assert quiet["read_cost"]["observations"] == 0
    assert quiet["read_cost"]["elapsed_ms"] < 1000
    # Read cost of the whole flow's page: measured, bounded, reported.
    whole = _json(live, f"/api/activity?after={start['cursor']}&limit=100")
    assert whole["read_cost"]["observations"] == len(whole["items"]) == 8
    assert not whole["more"] and whole["unavailable"] == 0


def test_an_agent_session_and_a_named_goal_ride_with_their_requests(
    live: LocalPortfolioWebSession, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """requirement (LAWS OP13, V281): a request carries the agent session its command ran in
    and the goal it names, as provenance in its activity rows and never as a request field, so
    a result outlives its session; a session that cannot be told apart is not guessed."""

    spec = tmp_path / "spec.yaml"
    spec.write_text("{}", encoding="utf-8")
    declaration = tmp_path / "goal.json"
    declaration.write_text(
        json.dumps(
            {
                "title": "Why the update skipped names",
                "objective": "Find why the last update skipped names",
                "kind": "DATA",
                "criteria": [{"criterion_id": "cause", "text": "The cause, with evidence"}],
            }
        ),
        encoding="utf-8",
    )
    code, opened = _cli(live, "goal", "open", "--file", str(declaration))
    assert code == 0, opened
    goal = opened["data"]["goal_id"]
    sessions = (
        ({"CLAUDE_CODE_SESSION_ID": "00000000-0000-4000-8000-000000000001"}, "claude-code"),
        ({"CODEX_THREAD_ID": "00000000-0000-7000-8000-000000000002"}, "codex"),
    )
    for environment, vendor in sessions:
        cursor = _json(live, "/api/activity")["cursor"]
        with monkeypatch.context() as m:
            for name, value in environment.items():
                m.setenv(name, value)
            code, _planned = _cli(
                live, "--goal", goal, "strategy-book", "preview", "--file", str(spec)
            )
        assert code == 0
        page = _json(live, f"/api/activity?after={cursor}")
        rows = _items(page, operation="PLAN")
        assert [row["payload"]["phase"] for row in rows] == ["REQUESTED", "RETURNED"]
        for row in rows:
            subject = row["payload"]["subject"]
            assert subject["agent_vendor"] == vendor
            assert subject["agent_session"] == next(iter(environment.values()))
            assert subject["goal_id"] == goal
            assert not {"agent_vendor", "agent_session", "goal_id"} & set(
                row["payload"]["request_fields"]
            )
    cursor = _json(live, "/api/activity")["cursor"]
    with monkeypatch.context() as m:
        for environment, _vendor in sessions:
            for name, value in environment.items():
                m.setenv(name, value)
        code, _planned = _cli(live, "strategy-book", "preview", "--file", str(spec))
    assert code == 0
    for row in _items(_json(live, f"/api/activity?after={cursor}"), operation="PLAN"):
        assert not {"agent_vendor", "agent_session", "goal_id"} & set(row["payload"]["subject"])


def test_the_recent_read_finds_what_just_happened_by_session_and_goal(
    live: LocalPortfolioWebSession, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """requirement (OP13, V281): one read answers 帮我找到刚才的结果, whichever session asks:
    the newest requests grouped by agent session and goal, newest first."""

    spec = tmp_path / "spec.yaml"
    spec.write_text("{}", encoding="utf-8")
    for name, value in (("CLAUDE_CODE_SESSION_ID", "c-1"), ("CODEX_THREAD_ID", "t-1")):
        with monkeypatch.context() as m:
            m.setenv(name, value)
            code, _planned = _cli(live, "strategy-book", "preview", "--file", str(spec))
        assert code == 0
    code, recent = _cli(live, "activity", "recent")
    assert code == 0, recent
    groups = recent["data"]["groups"]
    assert [group["agent"] for group in groups[:2]] == [
        {"vendor": "codex", "session_id": "t-1"},
        {"vendor": "claude-code", "session_id": "c-1"},
    ]
    assert [item["operation"] for item in groups[1]["items"]] == ["PLAN"]
    code, refused = _cli(live, "activity", "recent", "--limit", "0")
    assert code == 2 and refused["failure_code"] == "local_client.request_invalid"
    assert refused["data"]["fields"] == [["limit"]]


def test_every_refusal_is_counted_by_operation_code_caller_and_vendor_and_nothing_else(
    live: LocalPortfolioWebSession, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """requirement (AC, LAWS OP14): every refusal is counted durably, reads included, by day,
    operation, code, caller kind and agent vendor, never by session, request or text, so it
    outlives the transient rows; a request that does not meet its schema counts under the
    operation it named; an answer that is no refusal counts nothing; `activity refusals`
    reads the counts, the most frequent first."""

    session = "00000000-0000-4000-8000-000000000001"
    unknown = "00000000-0000-4000-8000-000000000000"
    malformed = tmp_path / "malformed.yaml"
    malformed.write_text("operation: STATUS\ntask_id: not-a-task\n", encoding="utf-8")
    spec = tmp_path / "spec.yaml"
    spec.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", session)
    for _ in range(2):
        code, refused = _cli(live, "task", "show", unknown)
        assert code == 2 and refused["failure_code"] == "task_control.task_not_found", refused
    code, invalid = _cli(live, "request", "--file", str(malformed))
    assert code != 0 and invalid["failure_code"] == "local_client.request_invalid", invalid
    code, _planned = _cli(live, "strategy-book", "preview", "--file", str(spec))
    assert code == 0  # an answer that is no refusal counts nothing
    code, counted = _cli(live, "activity", "refusals", "--days", "7")
    assert code == 0, counted
    data = counted["data"]
    assert data["status"] == "ACTIVITY_REFUSALS" and data["days"] == 7
    rows = {(row["operation"], row["failure_code"]): row for row in data["refusals"]}
    assert set(rows) == {
        ("STATUS", "task_control.task_not_found"),
        ("STATUS", "local_client.request_invalid"),
    }
    status = rows[("STATUS", "task_control.task_not_found")]
    assert status["count"] == 2 and data["total"] == 3
    assert status["caller"] == "EXTERNAL_AUTOMATION" and status["vendor"] == "claude-code"
    assert set(status) == {
        "day",
        "operation",
        "failure_code",
        "caller",
        "vendor",
        "count",
        "first_at",
        "last_at",
    }
    assert data["refusals"][0] == status  # the most frequent first
    # The counts keep no session; the envelope's context names the caller's own (CG4).
    assert session not in json.dumps(data) and unknown not in json.dumps(data)
    code, refused = _cli(live, "activity", "refusals", "--days", "0")
    assert code != 0  # a window outside 1 to 3660 days is refused at its field


def test_a_malformed_agent_session_or_goal_is_refused_at_the_boundary() -> None:
    """requirement (OP13): the provenance headers are read at the Host's boundary and a
    malformed one is refused by name, never recorded as someone's session."""

    from alphalattice.interface.local_application.cli_contract import (
        AGENT_SESSION_HEADER,
        AGENT_VENDOR_HEADER,
        GOAL_HEADER,
        agent_provenance_headers,
        request_provenance,
    )

    assert request_provenance({}) is None
    goal = str(uuid4())
    read = request_provenance(agent_provenance_headers({"CODEX_THREAD_ID": "t-1"}, goal))
    assert read is not None and (read.vendor, read.session, read.goal_id) == ("codex", "t-1", goal)
    session = "local_web.agent_session_invalid"
    for headers, code in (
        ({AGENT_VENDOR_HEADER: "codex"}, session),
        ({AGENT_VENDOR_HEADER: "other", AGENT_SESSION_HEADER: "s"}, session),
        ({AGENT_VENDOR_HEADER: "codex", AGENT_SESSION_HEADER: "a b"}, session),
        ({GOAL_HEADER: "not-a-goal"}, "local_web.goal_header_invalid"),
    ):
        with pytest.raises(ValueError, match=code):
            request_provenance(headers)
    assert agent_provenance_headers({"CLAUDE_CODE_SESSION_ID": "a", "CODEX_THREAD_ID": "b"}) == {}


def test_external_events_carry_the_boundary_assigned_trust_and_refuse_spoofing(
    live: LocalPortfolioWebSession, tmp_path: Path
) -> None:
    client = LocalResearchClient(live.workspace)
    before = _json(live, "/api/activity")["cursor"]

    first = client.publish_event(_event(0))
    assert first["status"] == "APPENDED" and first["authority"] == "AGENT_PROPOSAL"
    assert first["source_id"] == "codex-adapter:session-a" and first["task_verified"] is False
    replay = client.publish_event(_event(0))
    assert replay["status"] == "REUSED_EXACT"
    assert replay["observation_id"] == first["observation_id"]
    collision = client.publish_event(_event(0, summary="a different claim, same sequence"))
    assert collision["status"] == "REFUSED"
    assert collision["failure_code"] == "observation.source_sequence_collision"

    page = _json(live, f"/api/activity?after={before}")
    (declared,) = page["items"]
    assert declared["schema_kind"] == "ExternalActivityObserved"
    assert declared["source_kind"] == "EXTERNAL_CLIENT"
    assert declared["authority"] == "AGENT_PROPOSAL"  # assigned here, not by the client
    assert declared["payload"]["event_kind"] == "NATIVE_TURN_STARTED"
    assert declared["payload"]["summary_truncated"] is False
    assert declared["task_id"] is None

    # The document cannot name its own trust, caller or source.
    for field in ("authority", "caller", "source_kind", "claim_level"):
        spoofed = client.publish_event(_event(1, **{field: "HOST_DECISION"}))
        assert spoofed["status"] == "REFUSED"
        assert spoofed["failure_code"] == "activity.event_invalid"
        assert [field] in spoofed["fields"]
    assert client.publish_event(_event(1, event_kind="lower case"))["failure_code"] == (
        "activity.event_invalid"
    )
    secret = client.publish_event(_event(1, subject={"api_key_hint": "x"}))
    assert secret["failure_code"] == "observation.redaction_failed"
    assert client.publish_event(_event(1, summary="x" * 4001))["failure_code"] == (
        "activity.event_invalid"
    )

    # A Task reference is verified against Task Control before it is recorded.
    unknown = client.publish_event(_event(1, subject={"task_id": str(uuid4())}))
    assert (
        unknown["status"] == "REFUSED" and unknown["failure_code"] == "activity.event_task_unknown"
    )
    spec = tmp_path / "spec.yaml"
    spec.write_text("{}", encoding="utf-8")
    task_id = _cli(live, "strategy-book", "run", "--file", str(spec))[1]["data"]["task_id"]
    live.dispatcher.drain_for_tests()  # type: ignore[union-attr]
    long_summary = "word " * 300
    verified = client.publish_event(_event(1, subject={"task_id": task_id}, summary=long_summary))
    assert verified["status"] == "APPENDED" and verified["task_verified"] is True
    assert verified["summary_truncated"] is True
    page = _json(live, f"/api/activity?after={page['cursor']}")
    declared_about_task = _items(page, event_kind="NATIVE_TURN_STARTED")[-1]
    assert declared_about_task["task_id"] == task_id
    assert declared_about_task["run_id"] is None  # declared about a Task, not part of its run
    assert len(declared_about_task["payload"]["summary"]) == 500
    assert declared_about_task["payload"]["task_verified"] is True
    assert page["tasks"][task_id]["lifecycle"] == "SUCCEEDED"

    # The event and the feed are the client's operations, on the external-client route: a
    # browser origin, a session token or a foreign workspace never reach the observer (V266).
    port = live.web.bound_port  # type: ignore[union-attr]
    status, _headers, body = _request(
        live,
        "/api/client/operations",
        method="POST",
        payload={"operation": "EVENT_DECLARE", "event": _event(2)},
        origin=f"http://127.0.0.1:{port}",
    )
    assert status == 403
    assert json.loads(body)["refused"] == "local_web.external_browser_origin_not_admitted"
    status, _headers, body = _request(
        live,
        "/api/client/operations",
        method="POST",
        payload={"operation": "ACTIVITY_LIST"},
        token=None,
        cookie=None,
    )
    assert status == 403 and json.loads(body)["refused"].startswith("local_web.external_")
    foreign = LocalResearchClient(live.workspace)
    foreign.workspace = live.workspace.parent / "another-workspace"
    assert "refused" in foreign.publish_event(_event(2))
    assert _json(live, "/api/activity")["head"] == page["head"]  # nothing was recorded


def test_observer_failure_degrades_visibly_without_failing_or_repeating_work(
    live: LocalPortfolioWebSession, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    activity = live.activity
    assert activity is not None
    healthy_emit = activity._port.emit

    def broken_emit(_draft: object) -> None:
        raise ObservationStorageError("observation.storage_busy", "fixture: store locked")

    monkeypatch.setattr(activity._port, "emit", broken_emit)
    spec = tmp_path / "spec.yaml"
    spec.write_text("{}", encoding="utf-8")
    code, sent = _cli(live, "strategy-book", "run", "--file", str(spec))
    assert code == 3 and sent["data"]["disposition"] == "ADMITTED"  # result preserved
    task_id = sent["data"]["task_id"]
    live.dispatcher.drain_for_tests()  # type: ignore[union-attr]
    status = _json(live, f"/api/status?task_id={task_id}")
    assert status["lifecycle"] == "SUCCEEDED" and status["worker_failure"] is None
    assert live.dispatcher.executions == 1  # type: ignore[union-attr]

    page = _json(live, "/api/activity")
    assert page["items"] == []  # nothing could be written
    assert page["observer"]["status"] == "DEGRADED"
    assert page["observer"]["recording"] == "FAILING"
    assert page["observer"]["last_failure_code"] == "observation.storage_busy"
    assert page["observer"]["last_failure_type"] == "ObservationStorageError"
    assert page["observer"]["missing_observations"] >= 3  # requested, returned, command return
    assert page["observer"]["first_failure_at"]
    assert page["observer"]["claim"] == "OBSERVER_STATE_NOT_EXECUTION_STATE"
    assert "fixture: store locked" not in json.dumps(page)  # no failure prose leaves the owner
    assert page["tasks"] == {}
    assert _json(live, f"/api/activity?watch={task_id}")["tasks"][task_id]["lifecycle"] == (
        "SUCCEEDED"
    )

    # An unexpected hook failure stays on the dispatcher, never on the Task.
    monkeypatch.setattr(activity._port, "emit", healthy_emit)
    monkeypatch.setattr(
        live.dispatcher, "on_command_returned", lambda *_a: (_ for _ in ()).throw(RuntimeError("x"))
    )
    reused = _cli(live, "strategy-book", "run", "--file", str(spec))[1]["data"]
    assert reused["disposition"] == "REUSED_EXACT"  # exact reuse admits no second Task
    other = tmp_path / "other.yaml"
    other.write_text("top_k: 30\n", encoding="utf-8")
    second = _cli(live, "strategy-book", "run", "--file", str(other))[1]["data"]
    assert second["disposition"] == "ADMITTED"
    live.dispatcher.drain_for_tests()  # type: ignore[union-attr]
    status = _json(live, f"/api/status?task_id={second['task_id']}")
    assert status["lifecycle"] == "SUCCEEDED" and status["worker_failure"] is None
    assert live.dispatcher.worker_alive  # type: ignore[union-attr]
    assert live.dispatcher.observer_failures == 1  # type: ignore[union-attr]
    page = _json(live, "/api/activity")
    # A later successful append does not conceal what was lost: the gap stays
    # DEGRADED while current recording is reported healthy again.
    assert page["observer"]["status"] == "DEGRADED"
    assert page["observer"]["recording"] == "OK"
    assert page["observer"]["hook_failures"] == 1
    assert page["observer"]["hook_failure_type"] == "RuntimeError"
    assert page["observer"]["missing_observations"] >= 4
    assert _items(page, operation="RUN", phase="RETURNED", status="ADMITTED")
    assert live.dispatcher.executions == 2  # type: ignore[union-attr]


def test_cursor_reconnect_restart_and_typed_read_refusals(
    live: LocalPortfolioWebSession, tmp_path: Path
) -> None:
    client = LocalResearchClient(live.workspace)
    for sequence in range(3):
        assert client.publish_event(_event(sequence))["status"] == "APPENDED"
    tail = _json(live, "/api/activity?limit=2")
    assert tail["disposition"] == "TAIL" and tail["more"] and len(tail["items"]) == 2
    assert tail["items"][-1]["payload"]["producer_sequence"] == 2
    epoch = tail["epoch"]

    # A client that missed pages continues from where it stopped, in order, once.
    early = f"{epoch}:1"
    page = _json(live, f"/api/activity?after={early}&limit=1")
    assert page["disposition"] == "CONTINUED" and page["more"]
    assert [i["ordinal"] for i in page["items"]] == [2]
    again = _json(live, f"/api/activity?after={early}&limit=1")
    assert again["items"] == page["items"]  # idempotent: the same cursor, the same rows
    rest = _json(live, f"/api/activity?after={page['cursor']}&limit=50")
    assert [i["ordinal"] for i in rest["items"]] == [3] and not rest["more"]

    # A cursor from another store is reset with a fresh tail, not continued.
    reset = _json(live, f"/api/activity?after={'f' * 32}:1")
    assert reset["disposition"] == "RESET" and reset["epoch"] == epoch
    assert [i["ordinal"] for i in reset["items"]] == [1, 2, 3]
    beyond = _json(live, f"/api/activity?after={epoch}:99")
    assert beyond["disposition"] == "RESET"

    for query, code in (
        ("after=not-a-cursor", "activity.cursor_invalid"),
        ("limit=0", "activity.limit_invalid"),
        ("limit=201", "activity.limit_invalid"),
        ("watch=not-a-task", "activity.watch_invalid"),
        ("watch=" + ",".join(str(uuid4()) for _ in range(17)), "activity.watch_invalid"),
        ("cursor=1", "activity.query_field_unknown"),
    ):
        status, _headers, body = _request(live, f"/api/activity?{query}")
        assert status == 400, query
        assert json.loads(body)["refused"] == code

    # A service restart keeps the store: the old cursor continues, new work is
    # attributed to the new launch, and the client connection is re-issued.
    old_instance = client.connection.instance
    live.stop()
    live.start()
    client = LocalResearchClient(live.workspace)
    assert client.connection.instance != old_instance
    resumed = _json(live, f"/api/activity?after={rest['cursor']}")
    assert resumed["disposition"] == "CONTINUED" and resumed["items"] == []
    assert resumed["epoch"] == epoch
    spec = tmp_path / "spec.yaml"
    spec.write_text("{}", encoding="utf-8")
    assert _cli(live, "strategy-book", "preview", "--file", str(spec))[0] == 0
    after_restart = _json(live, f"/api/activity?after={rest['cursor']}")
    sources = {item["source_id"] for item in after_restart["items"]}
    assert sources == {f"local-web:{client.connection.instance}"}
    assert client.activity(after=rest["cursor"])["items"] == after_restart["items"]


def test_declared_sessions_are_read_from_their_owner_not_the_feed_tail(
    live: LocalPortfolioWebSession, tmp_path: Path
) -> None:
    """Round 88: the Team page's sessions readback. The feed's tail is bounded by
    the newest rows of every kind, so the product's own operations push a team's
    events out of it; the external read walks one kind and pages back by ordinal."""

    client = LocalResearchClient(live.workspace)
    for sequence in range(3):
        session = "session-a" if sequence < 2 else "session-b"
        assert (
            client.publish_event(_event(sequence, producer_session=session))["status"] == "APPENDED"
        )
    spec = tmp_path / "spec.yaml"
    spec.write_text("{}", encoding="utf-8")
    for _ in range(2):
        assert (
            _cli(live, "strategy-book", "preview", "--file", str(spec))[0] == 0
        )  # product operations after them

    tail = _json(live, "/api/activity?limit=2")
    assert tail["disposition"] == "TAIL" and tail["more"]
    assert {item["schema_kind"] for item in tail["items"]} == {"ProductOperationObserved"}

    declared = _json(live, "/api/activity/external?limit=2")
    assert declared["disposition"] == "TAIL" and declared["epoch"] == tail["epoch"]
    assert declared["more"] and declared["oldest"] == declared["items"][0]["ordinal"]
    assert [i["schema_kind"] for i in declared["items"]] == ["ExternalActivityObserved"] * 2
    assert [i["payload"]["producer_sequence"] for i in declared["items"]] == [1, 2]
    assert [i["payload"]["producer_session"] for i in declared["items"]] == [
        "session-a",
        "session-b",
    ]
    assert "tasks" not in declared  # no Task join: a declared event names its Task as a flag

    older = _json(live, f"/api/activity/external?limit=2&before={declared['oldest']}")
    assert [i["payload"]["producer_sequence"] for i in older["items"]] == [0]
    assert not older["more"] and older["oldest"] == older["items"][0]["ordinal"]
    whole = _json(live, "/api/activity/external")
    assert [i["payload"]["producer_sequence"] for i in whole["items"]] == [0, 1, 2]
    assert "accepted_answer" not in whole
    selected_row = whole["items"][0]
    selected = _json(
        live, f"/api/activity/external?observation_id={selected_row['observation_id']}"
    )
    assert selected["observation_id"] == selected_row["observation_id"]
    assert selected["epoch"] == whole["epoch"]
    assert "items" not in selected and "head" not in selected and "more" not in selected
    assert selected["accepted_answer"]["status"] == "UNAVAILABLE"
    assert selected["accepted_answer"]["reason"] == "activity.accepted_answer_unavailable"
    assert "contribution" not in selected["accepted_answer"]
    missing = _json(live, f"/api/activity/external?observation_id={'f' * 64}")
    assert missing["observation_id"] == "f" * 64 and "items" not in missing
    assert missing["accepted_answer"]["status"] == "UNAVAILABLE"
    assert missing["accepted_answer"]["reason"] == "activity.accepted_answer_unavailable"
    assert "contribution" not in missing["accepted_answer"]
    assert _json(live, "/api/activity/external")["items"] == whole["items"]

    for query, code in (
        ("limit=0", "activity.limit_invalid"),
        ("limit=201", "activity.limit_invalid"),
        ("before=0", "activity.cursor_invalid"),
        ("before=x", "activity.cursor_invalid"),
        ("after=1", "activity.query_field_unknown"),
        ("observation_id=x", "activity.observation_id_invalid"),
        (f"observation_id={'A' * 64}", "activity.observation_id_invalid"),
        (f"observation_id={'a' * 64}&limit=1", "activity.query_selection_conflict"),
        (f"observation_id={'a' * 64}&before=1", "activity.query_selection_conflict"),
        (
            f"observation_id={'a' * 64}&observation_id={'b' * 64}",
            "activity.query_parameter_invalid:observation_id",
        ),
    ):
        status, _headers, body = _request(live, f"/api/activity/external?{query}")
        assert status == 400, query
        assert json.loads(body)["refused"] == code


def test_shutdown_closes_the_observer_after_the_worker(
    live: LocalPortfolioWebSession, tmp_path: Path
) -> None:
    spec = tmp_path / "spec.yaml"
    spec.write_text("{}", encoding="utf-8")
    task_id = _cli(live, "strategy-book", "run", "--file", str(spec))[1]["data"]["task_id"]
    ledger_path = live.workspace / "runtime" / "observations.sqlite"
    live.stop()
    assert live.activity is None and live.dispatcher is None and live.web is None
    assert ledger_path.is_file()
    assert not (live.workspace / "runtime" / "observation-storage-failures").exists()
    # The command's return was recorded before the store closed.
    live.start()
    page = _json(live, "/api/activity?limit=100")
    returned = _items(page, schema_kind="TaskControlTransition")
    assert returned and returned[-1]["task_id"] == task_id
    assert returned[-1]["payload"]["task_lifecycle"] == "SUCCEEDED"


def test_seam_fixture_round_trips_one_native_shaped_event_and_one_product_event(
    live: LocalPortfolioWebSession, tmp_path: Path
) -> None:
    """The interface checkpoint: the labelled fixture's native-shaped document is
    admitted through the public client and read back exactly as the fixture says,
    beside one real product operation. A fixture, not native proof."""

    fixture = json.loads(
        Path(__file__).with_name("workspace_activity_seam_fixture.json").read_text("utf-8")
    )
    assert fixture["claim"].startswith("FIXTURE_NOT_NATIVE_PROOF")
    client = LocalResearchClient(live.workspace)
    before = client.activity()["cursor"]
    admitted = client.publish_event(fixture["native_event_document"])
    assert admitted["status"] == "APPENDED"
    spec = tmp_path / "spec.yaml"
    spec.write_text("{}", encoding="utf-8")
    task_id = _cli(live, "strategy-book", "run", "--file", str(spec))[1]["data"]["task_id"]

    page = client.activity(after=before)
    declared = next(i for i in page["items"] if i["schema_kind"] == "ExternalActivityObserved")
    expected = fixture["expected_declared_item"]
    assert {key: declared[key] for key in expected} == expected
    (product,) = [
        i
        for i in page["items"]
        if i["schema_kind"] == "ProductOperationObserved" and i["payload"]["phase"] == "RETURNED"
    ]
    example = fixture["product_operation_item_example"]
    assert product["source_kind"] == example["source_kind"]
    assert product["authority"] == example["authority"]
    assert product["task_id"] == task_id and product["run_id"] == f"local-web:{task_id}"
    assert set(product["payload"]) == set(example["payload"])
    assert product["payload"]["next_read"] == {"operation": "STATUS", "task_id": task_id}
    assert product["payload"]["status"] == "ADMITTED" and product["payload"]["caller"] == (
        "EXTERNAL_AUTOMATION"
    )
    # Every refusal the fixture documents is one the owners actually raise.
    source_root = Path(__file__).resolve().parents[2] / "src/alphalattice"
    owners = "".join(
        (source_root / name).read_text("utf-8")
        for name in (
            "interface/local_application/activity.py",
            "interface/local_application/failure_codes.py",
            "interface/local_application/web.py",
            "control/product_host/composition/workspace_activity.py",
            "control/product_host/composition/local_web_session.py",
            "control/product_host/composition/portfolio_research_operations.py",
            "control/observation_runtime/ledger.py",
            "control/observation_runtime/policy.py",
        )
    )
    for code, _why in fixture["typed_refusals"]:
        assert f'"{code}"' in owners, code
    live.dispatcher.drain_for_tests()  # type: ignore[union-attr]


def test_observer_callbacks_that_raise_never_change_the_operation_or_its_refusal(
    live: LocalPortfolioWebSession, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Entry, return and failure steps are stepped over when the observer itself
    is broken: the owner's answer, a refusal and a Task's execution are untouched,
    and the breakage is counted at the operation owner."""

    from alphalattice.interface.local_application.activity import observed_operation
    from alphalattice.interface.local_application.portfolio_research import (
        PortfolioResearchOperationRequest,
    )

    activity, operations = live.activity, live.operations
    assert activity is not None and operations is not None

    def broken(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("observer bug")

    def broken_entry(request: object, *, caller: object) -> None:
        if observed_operation(getattr(request, "operation", "")):
            raise RuntimeError("observer bug")

    monkeypatch.setattr(activity, "entered", broken_entry)
    monkeypatch.setattr(activity, "returned", broken)
    monkeypatch.setattr(activity, "failed", broken)
    spec = tmp_path / "spec.yaml"
    spec.write_text("{}", encoding="utf-8")
    code, sent = _cli(live, "strategy-book", "run", "--file", str(spec))
    assert code == 3 and sent["data"]["disposition"] == "ADMITTED"  # entry breakage: unobserved run
    live.dispatcher.drain_for_tests()  # type: ignore[union-attr]
    assert _json(live, f"/api/status?task_id={sent['data']['task_id']}")["lifecycle"] == "SUCCEEDED"
    assert operations.observer_failures == 1  # `entered` raised; nothing else was called

    monkeypatch.setattr(
        activity, "entered", lambda *_a, **_k: object()
    )  # a span the owner passes on
    reused = operations.execute(PortfolioResearchOperationRequest(operation="RUN", spec={}))
    assert reused["disposition"] == "REUSED_EXACT"  # `returned` raised; the answer is intact
    with pytest.raises(
        ValueError,
        match=re.escape("research_update.human_confirmation_required"),
    ):
        operations.execute(
            PortfolioResearchOperationRequest(
                operation="RESEARCH_UPDATE_AUTOMATION_CONFIGURE",
                automation_enabled=True,
                automation_package_ids=(),
            ),
            caller="EXTERNAL_AUTOMATION",
        )  # `failed` raised; the original refusal is what the caller receives
    assert operations.observer_failures == 3
    assert operations.last_observer_failure == "RuntimeError"
    state = _json(live, "/api/activity")["observer"]
    assert state["status"] == "DEGRADED" and state["entry_failures"] == 3
    assert state["entry_failure_type"] == "RuntimeError"
    assert "observer bug" not in json.dumps(state)


# It raises an untyped failure on purpose, to prove its text stays out (V449).
@pytest.mark.untyped_failure
def test_only_typed_failure_codes_and_exception_classes_are_persisted(
    live: LocalPortfolioWebSession, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refusal prose, exception messages and worker error text never reach the ledger."""

    operations = live.operations
    assert operations is not None
    before = _json(live, "/api/activity")["cursor"]
    original = type(operations)._execute

    def leaking(_self: object, request: object, **kwargs: object) -> dict[str, object]:
        raise KeyError("private path C:/secret/spec.yaml")

    monkeypatch.setattr(type(operations), "_execute", leaking)
    status, _headers, body = _request(live, "/api/plan", method="POST", payload={"spec": {}})
    assert status == 400 and "KeyError" in json.loads(body)["refused"]  # the caller's own answer
    monkeypatch.setattr(
        type(operations),
        "_execute",
        lambda _self, request, **kwargs: {
            "status": "REFUSED",
            "failure_code": "not a code: C:/secret",
        },
    )
    assert _json(live, "/api/plan", method="POST", payload={"spec": {}})["status"] == "REFUSED"
    monkeypatch.setattr(type(operations), "_execute", original)

    spec = tmp_path / "spec.yaml"
    spec.write_text("{}", encoding="utf-8")
    task_id = _cli(live, "strategy-book", "run", "--file", str(spec))[1]["data"]["task_id"]
    monkeypatch.setattr(
        live.dispatcher, "failure", lambda _task_id: "RuntimeError: secret worker detail"
    )
    live.dispatcher.drain_for_tests()  # type: ignore[union-attr]

    page = _json(live, f"/api/activity?after={before}&limit=100")
    text = json.dumps(page["items"])
    assert "secret" not in text and "private path" not in text
    (failed,) = _items(page, operation="PLAN", phase="FAILED")
    assert failed["payload"]["failure_type"] == "KeyError"
    assert failed["payload"]["failure_code"] == "activity.failure_detail_withheld"
    (refused,) = _items(page, operation="PLAN", phase="RETURNED", status="REFUSED")
    assert refused["payload"]["failure_code"] == "activity.failure_detail_withheld"
    (transition,) = _items(page, schema_kind="TaskControlTransition")
    assert transition["task_id"] == task_id
    assert transition["payload"]["failure_type"] == "RuntimeError"
    assert transition["payload"]["task_lifecycle"] == "SUCCEEDED"
    # The projection join carries the class alongside the caller-visible text.
    assert page["tasks"][task_id]["worker_failure_type"] == "RuntimeError"


def test_artifact_is_verified_through_the_owner_readback_not_the_index(
    live: LocalPortfolioWebSession, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = live.service
    assert service is not None
    spec = tmp_path / "spec.yaml"
    spec.write_text("{}", encoding="utf-8")
    task_id = _cli(live, "strategy-book", "run", "--file", str(spec))[1]["data"]["task_id"]
    live.dispatcher.drain_for_tests()  # type: ignore[union-attr]
    page = _json(live, "/api/activity?limit=100")
    (artifact,) = _items(page, schema_kind="ArtifactVerificationObserved")
    opened = service.open_result(artifact["payload"]["artifact_hash"])
    assert opened.result_hash == artifact["payload"]["artifact_hash"]
    assert str(service.originating_task(opened.result_hash)) == task_id == artifact["task_id"]

    # An index entry whose artifact cannot be opened records no availability.
    def unreadable(_result_hash: str) -> None:
        raise ValueError("portfolio_ledger.result_unreadable")

    monkeypatch.setattr(service, "open_result", unreadable)
    other = tmp_path / "other.yaml"
    other.write_text("top_k: 30\n", encoding="utf-8")
    cursor = page["cursor"]
    second = _cli(live, "strategy-book", "run", "--file", str(other))[1]["data"]["task_id"]
    live.dispatcher.drain_for_tests()  # type: ignore[union-attr]
    page = _json(live, f"/api/activity?after={cursor}&limit=100")
    assert _items(page, schema_kind="TaskControlTransition")[-1]["task_id"] == second
    assert not _items(page, schema_kind="ArtifactVerificationObserved")
    assert page["tasks"][second]["lifecycle"] == "SUCCEEDED"  # the Task itself is untouched
    assert page["observer"]["status"] == "DEGRADED"
    assert page["observer"]["last_failure_code"] == "activity.artifact_readback_failed"
    assert page["observer"]["last_failure_type"] == "ValueError"


def test_a_sessions_task_listing_reads_each_artifact_from_its_owner_a_page_at_a_time(
    live: LocalPortfolioWebSession, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """requirement (U54): TASKS names an agent session and lists the Tasks it submitted, newest
    first, a page at a time; each row names its submitter, its final state and the artifact its
    owner opens when the listing answers, unavailable when it cannot -- never the feed's."""

    service = live.service
    assert service is not None
    submitted: dict[str, list[str]] = {"c-54": [], "t-54": []}
    runs = (
        ("CLAUDE_CODE_SESSION_ID", "c-54", "{}"),
        ("CLAUDE_CODE_SESSION_ID", "c-54", "top_k: 30\n"),
        ("CODEX_THREAD_ID", "t-54", "top_k: 20\n"),
    )
    for index, (variable, session, text) in enumerate(runs):
        spec = tmp_path / f"spec-{index}.yaml"
        spec.write_text(text, encoding="utf-8")
        with monkeypatch.context() as m:
            m.setenv(variable, session)
            sent = _cli(live, "strategy-book", "run", "--file", str(spec))[1]["data"]
        assert sent["disposition"] == "ADMITTED", sent
        submitted[session].append(sent["task_id"])
        live.dispatcher.drain_for_tests()  # type: ignore[union-attr]

    page = _json(live, "/api/tasks?agent_session=c-54")
    assert [row["task_id"] for row in page["tasks"]] == submitted["c-54"][::-1]
    assert page["next_cursor"] is None
    for row in page["tasks"]:
        assert row["submitted_by"] == {"vendor": "claude-code", "session": "c-54", "goal_id": None}
        assert row["final_state"] == "SUCCEEDED"
        artifact = row["artifact"]
        assert (artifact["artifact_kind"], artifact["availability"]) == (
            "PortfolioResearchResult",
            "AVAILABLE",
        )
        assert str(service.originating_task(artifact["artifact_hash"])) == row["task_id"]
    code, listed = _cli(live, "task", "list", "--session-id", "t-54")
    assert code == 0 and [row["task_id"] for row in listed["data"]["tasks"]] == submitted["t-54"]
    assert _json(live, "/api/tasks?agent_session=nobody")["tasks"] == []
    whole = _json(live, "/api/tasks")
    assert "next_cursor" not in whole and len(whole["tasks"]) == 3

    # A page at a time, by the cursor the previous page answered.
    first = _json(live, "/api/tasks?agent_session=c-54&history_limit=1")
    assert [row["task_id"] for row in first["tasks"]] == [submitted["c-54"][1]]
    assert first["next_cursor"] == submitted["c-54"][1]
    last = _json(
        live, f"/api/tasks?agent_session=c-54&history_limit=1&history_cursor={first['next_cursor']}"
    )
    assert [row["task_id"] for row in last["tasks"]] == [submitted["c-54"][0]]
    assert last["next_cursor"] is None
    _status, _headers, body = _request(
        live, f"/api/tasks?agent_session=c-54&history_cursor={submitted['t-54'][0]}"
    )
    assert json.loads(body)["failure_code"] == "tasks.cursor_moved_reload"

    # An artifact its owner cannot open now is unavailable, with the owner's code.
    def unreadable(_result_hash: str) -> None:
        raise ValueError("portfolio_ledger.result_unreadable")

    monkeypatch.setattr(service, "open_result", unreadable)
    (row,) = _json(live, "/api/tasks?agent_session=t-54")["tasks"]
    assert row["final_state"] == "SUCCEEDED"
    assert row["artifact"] == {
        "artifact_kind": "PortfolioResearchResult",
        "artifact_hash": None,
        "availability": "UNAVAILABLE",
        "failure_code": "portfolio_ledger.result_unreadable",
    }

    # A damaged derived projection is rebuilt from its canonical Task. If that exact Task is
    # unreadable too, TASKS and Activity keep their healthy peer and name the unreadable ID
    # without making up lifecycle or progress.
    import duckdb

    from alphalattice.control.task_control.registry import resolve_task_control_database

    control_database = resolve_task_control_database(live.workspace)

    def damage_projection(task_id: str) -> None:
        connection = duckdb.connect(str(control_database))
        try:
            connection.execute(
                "UPDATE workspace_task_projection SET projection_json = '{' WHERE task_id = ?",
                [task_id],
            )
        finally:
            connection.close()

    def remove_projection(task_id: str) -> None:
        connection = duckdb.connect(str(control_database))
        try:
            connection.execute("DELETE FROM workspace_task_projection WHERE task_id = ?", [task_id])
        finally:
            connection.close()

    target, peer = submitted["c-54"]
    damage_projection(target)
    rebuilt = _json(live, "/api/tasks?agent_session=c-54")
    assert {row["task_id"] for row in rebuilt["tasks"]} == {target, peer}
    assert not rebuilt.get("refusals")
    assert all(row["final_state"] == "SUCCEEDED" for row in rebuilt["tasks"])

    remove_projection(target)
    remove_projection(peer)
    connection = duckdb.connect(str(control_database))
    try:
        connection.execute(
            "UPDATE workspace_task SET record_json = '{' WHERE task_id = ?", [target]
        )
    finally:
        connection.close()

    partial = _json(live, "/api/tasks?agent_session=c-54")
    assert [row["task_id"] for row in partial["tasks"]] == [peer]
    (refusal,) = partial["refusals"]
    assert (refusal["task_id"], refusal["status"], refusal["failure_code"]) == (
        target,
        "REFUSED",
        "task_control.projection_unavailable",
    )
    assert refusal["detail"] and refusal["next_action"]
    assert {route["operation"] for route in refusal["next_requests"].values()} == {
        "WORKSPACE_SHOW",
        "WORKSPACE_BACKUPS",
    }
    assert not {"lifecycle", "current_stage", "verified_stage_count", "total_stage_count"} & set(
        refusal
    )

    activity = _json(live, f"/api/activity?watch={target},{peer}")
    assert activity["tasks"][peer]["lifecycle"] == "SUCCEEDED"
    joined_refusal = activity["tasks"][target]
    assert (joined_refusal["status"], joined_refusal["failure_code"]) == (
        "REFUSED",
        "task_control.projection_unavailable",
    )
    assert joined_refusal["detail"] and joined_refusal["next_action"]
    assert {route["operation"] for route in joined_refusal["next_requests"].values()} == {
        "WORKSPACE_SHOW",
        "WORKSPACE_BACKUPS",
    }
    assert not {"lifecycle", "current_stage", "verified_stage_count", "total_stage_count"} & set(
        joined_refusal
    )


def test_task_and_activity_lists_name_an_unreadable_queue_dependency_without_blaming_a_peer(
    live: LocalPortfolioWebSession, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A damaged queued head blocks cache reconstruction without assigning it to the peer."""
    import duckdb

    from alphalattice.control.task_control.contracts import (
        ResearchGoal,
        ResearchPlan,
        TaskInputEnvelope,
        WorkItemDefinition,
    )
    from alphalattice.control.task_control.registry import resolve_task_control_database

    monkeypatch.setenv("ALPHALATTICE_NETWORK_DISABLED", "1")
    session_id = "queue-authority-peer"
    spec = tmp_path / "queue-authority-peer.yaml"
    spec.write_text("{}", encoding="utf-8")
    with monkeypatch.context() as env:
        env.setenv("CLAUDE_CODE_SESSION_ID", session_id)
        admitted = _cli(live, "strategy-book", "run", "--file", str(spec))[1]["data"]
    peer_id = admitted["task_id"]
    live.dispatcher.drain_for_tests()  # type: ignore[union-attr]

    assert live.session is not None
    registry = live.session.task_control_registry
    envelope = TaskInputEnvelope.create(
        task_kind="factor_research",
        input_schema_id="factor-research.confirmed-mandate",
        payload={"purpose": "queue-authority-refusal-regression"},
    )
    goal = ResearchGoal.create(
        goal_kind="RUN_FACTOR_RESEARCH",
        input_hash=envelope.input_hash,
        deliverable_kind="ResearchDeskFactorInput",
        summary="Hold one queued Task while a peer is read back.",
    )
    plan = ResearchPlan.create(
        goal_hash=goal.goal_hash,
        workflow_definition_hash="a" * 64,
        verifier_catalog_hash="b" * 64,
        work_items=(
            WorkItemDefinition.create(
                stage_id="resolve_inputs",
                dependency_ids=(),
                verifier_id="task-control.resolve_inputs",
            ),
        ),
    )
    head = registry.admit(
        input_envelope=envelope,
        goal=goal,
        plan=plan,
        observed_at=live.clock(),
    ).record
    assert head.lifecycle.value == "QUEUED" and str(head.task_id) != peer_id

    database = resolve_task_control_database(live.workspace)
    with duckdb.connect(str(database)) as connection:
        connection.execute(
            "UPDATE workspace_task SET record_json = '{' WHERE task_id = ?",
            [str(head.task_id)],
        )
        connection.execute("DELETE FROM workspace_task_projection WHERE task_id = ?", [peer_id])

    expected_requests = {
        "workspace": {"operation": "WORKSPACE_SHOW"},
        "backups": {"operation": "WORKSPACE_BACKUPS"},
    }

    def assert_queue_refusal(answer: dict[str, Any]) -> None:
        assert answer["status"] == "REFUSED"
        assert answer["failure_code"] == "task_control.database_authority_unreadable"
        assert answer["queue_head_task_id"] == str(head.task_id)
        assert answer["detail"]
        assert answer["next_requests"] == expected_requests
        assert not {
            "task_id",
            "tasks",
            "refusals",
            "lifecycle",
            "current_stage",
            "verified_stage_count",
            "total_stage_count",
        } & set(answer)
        assert peer_id not in json.dumps(answer)

    try:
        code, task_answer = _cli(live, "task", "list", "--session-id", session_id)
        assert code == 2
        assert_queue_refusal(task_answer["data"])

        code, activity_answer = _cli(live, "activity", "list", "--watch", json.dumps([peer_id]))
        assert code == 2
        assert_queue_refusal(activity_answer["data"])
    finally:
        with duckdb.connect(str(database)) as connection:
            connection.execute(
                "UPDATE workspace_task SET record_json = ? WHERE task_id = ?",
                [head.model_dump_json(), str(head.task_id)],
            )


def test_unopenable_activity_store_degrades_observation_not_the_product(tmp_path: Path) -> None:
    from tests.portfolio_strategy_lab.local_web_support import _manifest, _resolved, _Resolver

    workspace = tmp_path / "workspace"
    (workspace / "runtime" / "observations.sqlite").mkdir(parents=True)  # a directory, not a store
    session = LocalPortfolioWebSession(
        workspace=workspace,
        workspace_manifest=_manifest("qa-degraded"),
        resolver=_Resolver(_resolved()),
    )
    session.start()
    try:
        assert session.activity is not None and not session.activity.available
        assert session.activity.store_failure == "observation.storage_open_failed"
        spec = tmp_path / "spec.yaml"
        spec.write_text("{}", encoding="utf-8")
        code, sent = _cli(session, "strategy-book", "run", "--file", str(spec))
        assert code == 3 and sent["data"]["disposition"] == "ADMITTED"
        session.dispatcher.drain_for_tests()  # type: ignore[union-attr]
        task_id = sent["data"]["task_id"]
        assert _json(session, f"/api/status?task_id={task_id}")["lifecycle"] == "SUCCEEDED"
        result_hash = _json(session, "/api/results")["results"][0]["result_hash"]
        assert (
            _json(session, f"/api/report?result_hash={result_hash}")["result_hash"] == result_hash
        )
        page = _json(session, f"/api/activity?watch={task_id}")
        assert page["disposition"] == "UNAVAILABLE" and page["items"] == []
        assert page["cursor"] is None and page["epoch"] is None
        assert page["observer"]["status"] == "UNAVAILABLE"
        assert page["observer"]["store_failure"] == "observation.storage_open_failed"
        assert page["tasks"][task_id]["lifecycle"] == "SUCCEEDED"  # Task truth is still served
        event = LocalResearchClient(session.workspace).publish_event(_event(0))
        assert (
            event["status"] == "REFUSED" and event["failure_code"] == "activity.storage_unavailable"
        )
        assert event["detail"] and event["next_action"]
        assert event["next_requests"] == {
            "workspace": {"operation": "WORKSPACE_SHOW"},
            "backups": {"operation": "WORKSPACE_BACKUPS"},
        }
        code, cli_page = _cli(session, "activity", "list")
        assert code == 0 and cli_page["data"]["disposition"] == "UNAVAILABLE"
    finally:
        session.stop()


def test_task_join_reads_only_the_referenced_projections(
    live: LocalPortfolioWebSession, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = live.session.task_control_registry  # type: ignore[union-attr]
    monkeypatch.setattr(
        type(registry),
        "latest_safe_projections",
        lambda _self: pytest.fail("the whole projection table was loaded for one page"),
    )
    spec = tmp_path / "spec.yaml"
    spec.write_text("{}", encoding="utf-8")
    task_id = _cli(live, "strategy-book", "run", "--file", str(spec))[1]["data"]["task_id"]
    live.dispatcher.drain_for_tests()  # type: ignore[union-attr]
    absent = str(uuid4())
    page = _json(live, f"/api/activity?watch={task_id},{absent}")
    # An explicitly watched ID with no canonical record stays named as unknown, not silently
    # dropped or given a guessed lifecycle. Only the held Task counts as a joined projection.
    assert set(page["tasks"]) == {task_id, absent}
    assert page["tasks"][absent]["status"] == "REFUSED"
    assert page["tasks"][absent]["failure_code"] == "task_control.projection_unavailable"
    assert "lifecycle" not in page["tasks"][absent]
    assert page["read_cost"]["projections"] == 1
    assert registry.safe_projections_for(()) == ()
