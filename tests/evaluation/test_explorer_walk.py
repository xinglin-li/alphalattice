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
    ],
)
def test_an_injected_refusal_recovers_only_the_original_bound_step(
    tmp_path: Path, recovery: str
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
    sent: list[dict[str, Any]] = []
    repaired = dict(request)
    if recovery in {"wrong-read", "located-read", "owned-read", "empty-read"}:
        repaired = {"operation": "COMMITTEE_READ", "update_task_id": "task-1"}
    elif recovery in {"wrong-binding", "wrong-write"}:
        repaired["update_task_id"] = "task-2"
    elif recovery == "changed-hash":
        repaired["plan_hash"] = "b" * 64

    def call(argv: list[str]) -> dict[str, Any]:
        if argv == ["start"]:
            return {"outcome": "OK", "next_requests": {"submit": request}}
        document = json.loads(Path(argv[-1]).read_text(encoding="utf-8"))
        sent.append(document)
        if document.get(required):
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
        if recovery in {"located", "located-read", "empty-read"}:
            answer["reasons"] = {required: "required"}
        if recovery == "owned-read":
            answer.update(
                failure_code="committee.message_refused:TARGET",
                rule="TARGET",
                detail="Choose an alias from the floor.",
            )
        if recovery == "usage":
            answer.update(
                failure_code="local_client.usage_invalid", usage_error={"observed": required}
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
        read_operations={"COMMITTEE_READ"},
        mutation_budget=120,
    )
    if recovery == "walk":
        grade, trace = explorer.explore([["start"]])
        assert grade["passed"] and sent.count(request) == 2
        assert (
            sum(bool(e.attributes.get("alphalattice.corrected_original")) for e in trace.events)
            == 1
        )
    else:
        explorer.inject(request, "committee", kinds={"drop-field"})
    clean = recovery in {"filled", "located", "located-read", "owned-read", "usage", "walk"}
    assert bool(explorer.findings) != clean
    assert len(explorer.mutation_checks) == (2 if recovery == "walk" else 1)
    assert explorer.mutation_checks[0]["rule_named"] == (
        recovery not in {"generic", "generic-offer"}
    )
    assert explorer.mutation_checks[0]["words_state_rule"] == (
        recovery in {"located", "located-read", "owned-read", "empty-read", "usage"}
    )
    assert explorer.mutation_checks[0]["owner"] == "committee.py"
    assert required not in sent[0] and request[required] == (
        "a" * 64 if recovery == "changed-hash" else "task-1"
    )
    if clean:
        assert sent[-1] == request
    elif recovery == "loop":
        assert len(sent) <= 3


@pytest.mark.parametrize(
    "identity",
    [
        "same",
        "different",
        "missing",
        "recovered-same",
        "recovered-task",
        "recovered-message",
        "read-same",
        "read-update",
        "read-message",
        "read-empty",
    ],
)
def test_a_duplicate_is_clean_only_when_it_reuses_the_original_identity(
    tmp_path: Path, identity: str
) -> None:
    """A duplicate must recover its original admission without creating another one."""
    request = {"operation": "MODEL_RUN", "plan_hash": "a" * 64}
    key = "message_id" if identity == "recovered-message" else "task_id"
    baseline = {"operation": "MODEL_RUN", "outcome": "PENDING", "data": {key: "original"}}
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
            else {key: "original" if identity in {"same", "recovered-same"} else "different"}
        )
        return {"operation": "MODEL_RUN", "outcome": "PENDING", "data": data}

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
    if identity.startswith("read-"):
        assert sent == [request, {"operation": "COMMITTEE_READ", "update_task_id": "task-1"}]
    else:
        assert sent == [request] * (2 if identity.startswith("recovered-") else 1)
    assert bool(explorer.findings) == (identity not in {"same", "recovered-same", "read-same"})
    assert len(explorer.mutation_checks) == 1
