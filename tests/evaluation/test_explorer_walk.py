"""The path explorer's grading of recorded answers: no Host and no model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from devtools.evaluation.explorer import Explorer, mutations
from devtools.evaluation.forms import Scenario


def test_alias_mistakes_change_the_alias_and_keep_the_members_words() -> None:
    """An alias mutation changes the referenced value without replacing its surrounding words."""
    message = {"kind": "CHALLENGE", "text": "H1 conflicts with T2.", "targets": ["H1"]}
    request = {"operation": "COMMITTEE_SUBMIT", "committee_message": message}
    changed = next(value for kind, value, _ in mutations(request, []) if kind == "alias-number")
    assert changed["committee_message"] == {**message, "text": "1 conflicts with T2."}
    assert request["committee_message"] == message
    reply = next(value for kind, value, _ in mutations(request, []) if kind == "reply")
    assert reply["committee_message"] == {**message, "reply_to": "M9999"}


def test_the_walk_names_each_failure_an_agent_would_meet_with_its_part(tmp_path: Path) -> None:
    """requirement: a refused offer, a self-offer that fails the same way, a forbidden act and
    an unreached route are each named once, with the loop part it broke in."""
    start = {
        "outcome": "OK",
        "detail": "Ask the person to restart the Host.",
        "next_requests": {"replan": {"operation": "WORKSPACE_PREPARE_PLAN"}},
        "data": {"next_commands": {"preflight": "setup --preflight"}},
    }
    stuck = {"outcome": "REFUSED", "failure_code": "setup.refused", "next_action": "RUN_IT"}
    stuck["next_commands"] = {"preflight": "setup --preflight"}

    def call(argv: list[str]) -> dict[str, Any]:
        refused = {"outcome": "REFUSED", "failure_code": "x.recovery_request_not_offered"}
        return refused if argv[0] == "request" else start

    scenario = Scenario(
        "walk",
        "a walk",
        "2026-10-10T15:00:00+00:00",
        (),
        (("REACHED",),),
        "agent",
        ("restart_host",),
    )
    grade, trace = Explorer(call, lambda _line: stuck, scenario, tmp_path).explore([["start"]])
    parts = {issue["reason"].split(" ")[0]: issue["loop_part"] for issue in grade["reasons"]}
    assert parts == {
        "forbidden": "feedback",
        "`replan`": "feedback",
        "`preflight`": "feedback",
        "route": "system",
    }
    assert not grade["passed"] and trace.events[0].attributes["alphalattice.offers"] == ["replan"]


@pytest.mark.parametrize(
    "ending",
    [
        *("CANCEL", "WORKSPACE_PREPARE_CANCEL", "GOAL_ABANDON", "HOST_STOP"),
        *("cancelled", "resend", "resolved"),
    ],
)
def test_a_positive_walk_keeps_endings_and_cancelled_replans_optional(tmp_path, ending):
    """Ending offers remain visible, but are not continuations of the requested work."""
    cancelled = ending in {"cancelled", "resend", "resolved"}
    offers = (
        {"replan": {"operation": "MODEL_PLAN"}}
        if cancelled
        else {"cancel": {"operation": ending}, "task": {"operation": "STATUS"}}
    )
    if ending == "resolved":
        offers = {"successor": {"operation": "STATUS"}}
    start = {"outcome": "OK", "next_requests": offers}
    if cancelled:
        start.update(
            data={"lifecycle": "CANCELLED"},
            next_action=("SEND_THE_OFFERED_REQUEST" if ending == "resend" else "REPORT_THE_STOP"),
        )

    if ending == "resolved":
        start["next_action"] = "READ_THE_TASK_WITH_TASK_SHOW"

    def call(argv):
        return start if argv == ["start"] else {"outcome": "OK", "data": {"lifecycle": "SUCCEEDED"}}

    terminal = ending in {"cancelled", "resend"}
    state = "CANCELLED" if terminal else "SUCCEEDED"
    route = Scenario("w", "w", "2026-10-17T02:44:00Z", (), ((state,),), "agent", ())
    grade, trace = Explorer(call, lambda _: {}, route, tmp_path).explore([["start"]])
    assert grade["passed"] == (ending != "resend") and len(trace.events) == (1 if terminal else 2)
    assert trace.events[0].attributes["alphalattice.offers"] == list(offers)
    if ending == "resend":
        assert grade["reasons"][0]["loop_part"] == "feedback"
    if not terminal:
        key = "successor" if ending == "resolved" else "task"
        assert trace.events[-1].attributes["alphalattice.path"].endswith(" > " + key)


def test_a_wait_past_its_time_box_while_the_task_runs_is_waited_again(tmp_path: Path) -> None:
    """regression (a merge set under load, 2026-10-15): one wait's time box ended the path."""
    wait = {"task_id": "t", "lifecycle": "RUNNING", "wait_event": {"event": "MAX_WAIT_REACHED"}}
    answers = [{"outcome": "OK", "data": {"lifecycle": "SUCCEEDED"}}]
    answers += [{"outcome": "PENDING", "data": data} for data in (wait, {"task_id": "t"})]
    route = Scenario("w", "w", "2026-10-15T16:00:00+00:00", (), (("SUCCEEDED",),), "agent", ())
    grade, trace = Explorer(lambda _: answers.pop(), lambda _: {}, route, tmp_path, 1).explore([[]])
    assert grade["passed"] and len(trace.events) == 3, grade


@pytest.mark.parametrize(
    "recovery",
    [
        "filled",
        "located",
        "located-read",
        "owned-read",
        "empty-read",
        "usage",
        "missing",
        "generic",
        "generic-offer",
        "loop",
        "wrong-read",
        "wrong-binding",
        "wrong-write",
        "changed-hash",
        "unavailable",
        "walk",
        "activity-good",
        "activity-hash",
        "activity-operation",
        "activity-phase",
        "activity-status",
        "activity-empty",
        "tasks-good",
        "tasks-id",
        "tasks-kind",
        "tasks-lifecycle",
        "tasks-empty",
        "packages-good",
        "packages-id",
        "packages-empty",
        "schema-good",
        "schema-operation",
        "schema-fields",
        "schema-empty",
        "schema-duplicate",
    ],
)
def test_an_injected_refusal_recovers_only_the_original_bound_step(
    tmp_path: Path, recovery: str, monkeypatch
) -> None:
    """A named refusal needs a filled recovery that admits the original operation and bindings."""
    request = {
        "operation": "COMMITTEE_SUBMIT",
        "update_task_id": "task-1",
        "committee_key": "a" * 32,
    }
    required = "plan_hash" if recovery == "changed-hash" else "update_task_id"
    if recovery == "changed-hash":
        request = {"operation": "MODEL_RUN", "plan_hash": "a" * 64}
    discovery = recovery.startswith(("activity-", "tasks-", "packages-", "schema-"))
    read_data = {}
    if recovery.startswith("schema-"):
        required = "evidence_setup"
        request = {"operation": "EVIDENCE_INSTALL", required: "task-1"}
        read_data = (
            {
                "operation_schema": {
                    "properties": {
                        "operation": {
                            "const": "TASKS"
                            if recovery == "schema-operation"
                            else request["operation"]
                        },
                        **({} if recovery == "schema-fields" else {required: {"type": "string"}}),
                    }
                }
            }
            if recovery != "schema-empty"
            else {}
        )
    elif recovery.startswith("packages-"):
        required = "strategy_package_id"
        request = {"operation": "RESEARCH_UPDATE_PLAN", required: "installed"}
        read_data = {
            "installed_strategies": []
            if recovery == "packages-empty"
            else [{"strategy_id": "other" if recovery == "packages-id" else "installed"}]
        }
    elif recovery.startswith("activity-"):
        required = "update_plan_hash"
        request = {"operation": "RESEARCH_UPDATE_RUN", required: "a" * 64}
        read_data = {
            "items": []
            if recovery == "activity-empty"
            else [
                {
                    "payload": {
                        "phase": "REQUESTED" if recovery == "activity-phase" else "RETURNED",
                        "status": "REFUSED" if recovery == "activity-status" else "PLANNED",
                        "operation": "DATA_UPDATE_PLAN"
                        if recovery == "activity-operation"
                        else "RESEARCH_UPDATE_PLAN",
                        "subject": {
                            required: "b" * 64 if recovery == "activity-hash" else "a" * 64
                        },
                    }
                }
            ]
        }
    elif recovery.startswith("tasks-"):
        read_data = {
            "tasks": []
            if recovery == "tasks-empty"
            else [
                {
                    "task_id": "task-2" if recovery == "tasks-id" else "task-1",
                    "task_kind": "portfolio_run"
                    if recovery == "tasks-kind"
                    else "portfolio_public_watermark_advancement",
                    "lifecycle": "RUNNING" if recovery == "tasks-lifecycle" else "SUCCEEDED",
                }
            ]
        }
    sent: list[dict[str, Any]] = []
    repaired = dict(request)
    if recovery in {"wrong-read", "located-read", "owned-read", "empty-read"}:
        repaired = {"operation": "COMMITTEE_READ", "update_task_id": "task-1"}
    elif discovery:
        repaired = {
            "operation": "WORKSPACE_SHOW"
            if recovery.startswith("packages-")
            else "OPERATION_LIST"
            if recovery.startswith("schema-")
            else "ACTIVITY_LIST"
            if recovery.startswith("activity-")
            else "TASKS"
        }
        if recovery.startswith("schema-"):
            repaired["schema_operation"] = request["operation"]
    elif recovery in {"wrong-binding", "wrong-write"}:
        repaired["update_task_id"] = "task-2"
    elif recovery == "changed-hash":
        repaired["plan_hash"] = "b" * 64

    def call(argv: list[str]) -> dict[str, Any]:
        if argv == ["start"]:
            return {"outcome": "OK", "next_requests": {"submit": request}}
        document = json.loads(Path(argv[-1]).read_text(encoding="utf-8"))
        sent.append(document)
        if discovery and document == repaired:
            return {"operation": document["operation"], "outcome": "OK", "data": read_data}
        if document.get(required) and argv[1] == "--file" and recovery != "schema-duplicate":
            return {
                "operation": document["operation"],
                "outcome": "UNAVAILABLE" if recovery == "unavailable" else "OK",
                "data": {"message_id": "M1", "update_task_id": "task-1"}
                if recovery != "empty-read"
                else {},
                **({"next_requests": {"again": request}} if recovery == "wrong-write" else {}),
            }
        answer = {
            "operation": request["operation"],
            "outcome": "REFUSED",
            "failure_code": "local_client.request_invalid"
            if recovery
            in {"generic", "generic-offer", "located", "located-read", "empty-read", "usage"}
            else "portfolio_research.operation_field_required:" + required,
        }
        if recovery in {"located", "located-read", "empty-read"} or discovery:
            answer["reasons"] = {required: "required"}
        if recovery == "owned-read":
            answer.update(
                failure_code="committee.message_refused:TARGET",
                rule="TARGET",
                detail="Choose an alias from the floor.",
            )
        if recovery == "usage" or argv[1] == "--flie":
            answer.update(
                failure_code="local_client.usage_invalid",
                usage_error={"observed": required if recovery == "usage" else "--flie"},
            )
        if recovery not in {"missing", "generic"}:
            answer["next_requests"] = {"repair": document if recovery == "loop" else repaired}
        return answer

    scenario = Scenario(
        "injection", "continue the request", "2026-10-10T15:00:00+00:00", (), (), "agent", ()
    )
    explorer = Explorer(
        call,
        lambda _: {},
        scenario,
        tmp_path,
        request_fields={request["operation"]: {"required": [required], "allowed": list(request)}},
        mutation_owners={request["operation"]: "committee.py", "COMMITTEE_READ": "committee.py"},
        read_operations={
            "COMMITTEE_READ",
            "ACTIVITY_LIST",
            "TASKS",
            "WORKSPACE_SHOW",
            "OPERATION_LIST",
        },
        mutation_budget=120,
        mutation_kinds={"drop-field"} if recovery == "walk" else None,
    )
    if recovery == "walk":
        explorer.mutation_budget, explorer.mutation_limit = 0, 2
        grade, trace = explorer.explore([["start"]])
        assert grade["passed"] and sent.count(request) == 2
        assert (
            sum(bool(e.attributes.get("alphalattice.corrected_original")) for e in trace.events)
            == 1
        )
    else:
        explorer.inject(
            request,
            "committee",
            **(
                {"baseline": {"outcome": "OK", "data": {"task_id": "held"}}}
                if recovery == "schema-duplicate"
                else {"kinds": {"drop-field"}}
            ),
        )
    clean = recovery in {
        "schema-good",
        "filled",
        "located",
        "located-read",
        "owned-read",
        "usage",
        "walk",
        "activity-good",
        "tasks-good",
        "packages-good",
    }
    assert bool(explorer.findings) != clean
    assert len(explorer.mutation_checks) == (2 if recovery == "walk" else 1)
    assert explorer.mutation_checks[0]["rule_named"] == (
        recovery not in {"generic", "generic-offer"}
    )
    assert explorer.mutation_checks[0]["words_state_rule"] == (
        (recovery in {"located", "located-read", "owned-read", "empty-read", "usage"} or discovery)
        and recovery != "schema-duplicate"
    )
    assert explorer.mutation_checks[0]["owner"] == "committee.py"
    assert (required in sent[0]) == (recovery == "schema-duplicate") and request[required] == (
        "installed"
        if recovery.startswith("packages-")
        else "a" * 64
        if recovery == "changed-hash" or recovery.startswith("activity-")
        else "task-1"
    )
    if clean:
        assert sent[-1] == request
    elif recovery == "loop":
        assert len(sent) <= 3
    if recovery == "filled":
        sampled, preceding = [], [*explorer.mutation_checks, {"mutation": "earlier"}]
        for number, checks in enumerate(([], preceding, preceding[::-1])):
            explorer.mutation_checks = checks
            before = len(sent)
            state = f"committee/00000000-0000-0000-0000-{number:012d}"
            explorer.inject(request, state)
            sampled.append((explorer.mutation_checks[-1]["mutation"], sent[before]))
        assert sampled[0] == sampled[1] == sampled[2] and sampled[0][1] != request
        explorer.mutation_limit = 2
        for cost in (1, 1000):
            clock = iter(range(0, 100000, cost))
            monkeypatch.setattr(
                "devtools.evaluation.explorer.time.monotonic", lambda clock=clock: next(clock)
            )
            explorer.mutation_checks, explorer.mutation_seconds = [], 0
            for _ in range(3):
                explorer.inject(request, "committee", kinds={"drop-field"})
            assert [row["mutation"] for row in explorer.mutation_checks] == ["drop-field"] * 2


@pytest.mark.parametrize(
    "state", ["SUCCEEDED", "RUNNING", "other-task", "stopped-read", "wrong-read"]
)
def test_a_retired_recovery_offer_is_settled_only_by_its_completed_source(tmp_path, state):
    """A correction keeps its Task and read; old recovery ends only after its source completes."""
    original = {"operation": "RESEARCH_UPDATE_PLAN", "recovery_task_id": "source"}
    read = {"operation": "TASK_RECOVERY", "task_id": "source"}
    code = "portfolio_research.recovery_task_id_invalid"
    if state in {"stopped-read", "wrong-read"}:
        original = {"operation": "TASK_RECOVERY", "task_id": "source"}
        read = {"operation": "TASKS"}
    refused = {
        "outcome": "REFUSED",
        "failure_code": code,
        "detail": code,
        "next_requests": {"recovery": read},
    }

    def call(argv):
        request = json.loads(Path(argv[-1]).read_text())
        if state in {"stopped-read", "wrong-read"} and request == original:
            return {
                "operation": "TASK_SHOW" if state == "wrong-read" else "TASK_RECOVERY",
                "outcome": "REFUSED",
                "failure_code": "execution.memory_insufficient",
                "data": {"task_id": "source", "lifecycle": "BLOCKED", "read_request": original},
            }
        if request != read:
            return refused
        if state in {"stopped-read", "wrong-read"}:
            return {"outcome": "OK", "data": {"tasks": [{"task_id": "source"}]}}
        return {
            "operation": "TASK_RECOVERY",
            "outcome": "OK",
            "data": {
                "task_id": "other" if state == "other-task" else "source",
                "lifecycle": "SUCCEEDED" if state == "other-task" else state,
            },
        }

    scenario = Scenario("recovery", "replan", "2026-10-10T15:00:00+00:00", (), (), "agent", ())
    explorer = Explorer(
        call,
        lambda _: {},
        scenario,
        tmp_path,
        read_operations={"TASK_RECOVERY", "TASKS"},
        mutation_owners={"TASK_RECOVERY": "tasks"},
        mutation_budget=120,
    )
    field = "task_id" if state in {"stopped-read", "wrong-read"} else "recovery_task_id"
    explorer.inject(original, "recovery", kinds={"value-change"}, fields={field})
    assert bool(explorer.findings) == (state not in {"SUCCEEDED", "stopped-read"})


@pytest.mark.parametrize(
    "identity",
    [
        *("same", "update-plan", "different", "missing", "cancelled", "cancelled-read"),
        *("recovered-same", "recovered-task", "recovered-message"),
        *("read-same", "read-update", "read-message", "read-empty"),
    ],
)
def test_a_duplicate_is_clean_only_when_it_reuses_the_original_identity(
    tmp_path: Path, identity: str
) -> None:
    """A duplicate must recover its original admission without creating another one."""
    request = {"operation": "MODEL_RUN", "plan_hash": "a" * 64}
    key = "message_id" if identity == "recovered-message" else "task_id"
    baseline = {"operation": "MODEL_RUN", "outcome": "PENDING", "data": {key: "original"}}
    if identity.startswith("cancelled"):
        baseline["data"]["lifecycle"] = "CANCELLED"
        if identity == "cancelled-read":
            request["operation"] = "COMMITTEE_READ"
    if identity == "update-plan":
        request = {"operation": "RESEARCH_UPDATE_PLAN"}
        key = "update_plan_hash"
        baseline = {"outcome": "OK", "data": {key: "original"}}
    if identity.startswith("read-"):
        request = {"operation": "COMMITTEE_SUBMIT", "update_task_id": "task-1"}
        baseline = {"outcome": "OK", "data": {"message_id": "original"}}
    sent: list[dict[str, Any]] = []

    def call(argv: list[str]) -> dict[str, Any]:
        sent.append(json.loads(Path(argv[-1]).read_text(encoding="utf-8")))
        if identity.startswith("read-"):
            if len(sent) == 1:
                return {
                    "outcome": "REFUSED",
                    "failure_code": "committee.message_refused:STANCE_IN",
                    "next_requests": {
                        "floor": {"operation": "COMMITTEE_READ", "update_task_id": "task-1"}
                    },
                }
            return {
                "outcome": "OK",
                "data": {}
                if identity == "read-empty"
                else {
                    "update_task_id": "other" if identity == "read-update" else "task-1",
                    "messages": [{"id": "other" if identity == "read-message" else "original"}],
                },
            }
        if identity.startswith("recovered-") and len(sent) == 1:
            return {
                "outcome": "REFUSED",
                "failure_code": "task_control.queue_full",
                "detail": "same admission",
                "next_requests": {"retry": request},
            }
        data = (
            {}
            if identity == "missing"
            else {
                key: "original"
                if identity in {"same", "recovered-same", "update-plan"}
                else "different"
            }
        )
        return {"operation": request["operation"], "outcome": "PENDING", "data": data}

    scenario = Scenario(
        "duplicate", "run the plan", "2026-10-10T15:00:00+00:00", (), (), "agent", ()
    )
    explorer = Explorer(
        call,
        lambda _: {},
        scenario,
        tmp_path,
        mutation_budget=120,
        read_operations={"COMMITTEE_READ"},
        mutation_owners={"COMMITTEE_SUBMIT": "committee.py", "COMMITTEE_READ": "committee.py"},
    )
    explorer.inject(request, "model", baseline=baseline, kinds={"duplicate"})
    if identity == "cancelled":
        assert sent == [] and explorer.findings[0]["loop_part"] == "agent"
    elif identity.startswith("read-"):
        assert sent == [request, {"operation": "COMMITTEE_READ", "update_task_id": "task-1"}]
    else:
        assert sent == [request] * (2 if identity.startswith("recovered-") else 1)
    assert bool(explorer.findings) == (
        identity not in {"same", "recovered-same", "read-same", "update-plan", "cancelled-read"}
    )
    assert len(explorer.mutation_checks) == 1
