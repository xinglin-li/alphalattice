"""Goals: the Host's ledger of what agents are asked to achieve (LAWS OP13, GR1).

Metadata integrity, the exact-read reference seam, the Host's record by session and the
submission check; numerical owners are stand-ins, never duplicated here.
"""

from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest

from alphalattice.control.product_host.composition.goals import GoalApplication
from alphalattice.control.product_host.publication.goals import GoalStore
from alphalattice.interface.local_application.activity import ExternalActivityEventDocument
from alphalattice.interface.local_application.cli_contract import RequestProvenance
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchOperationRequest as Request,
)
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchRequestDocument,
)

NOW = datetime(2026, 9, 15, tzinfo=UTC)
DECLARATION = {
    "title": "A study",
    "objective": "Does the signal survive costs?",
    "kind": "RESEARCH",
    "scope": "One admitted historical input; full declared support",
    "criteria": [{"criterion_id": "folds", "text": "Inspect all folds, including negative ones"}],
    "deliverables": [
        {"deliverable_id": "evidence", "kind": "RESULT", "description": "The exact result"}
    ],
    "budget": {"maximum_tasks": 2, "maximum_numerical_calls": 10, "maximum_model_calls": 2},
    "research": {
        "purpose": "NEW_RESEARCH",
        "comparison_design": "Same target and costs; no retuning",
        "required_stages": ["ALPHA", "RISK"],
    },
}
SESSION = RequestProvenance(vendor="claude-code", session="00000000-0000-4000-8000-000000000001")
SCRIPT = Path(__file__).resolve().parents[2] / "scripts/run_alphalattice.py"


@pytest.fixture
def goal_app(tmp_path):
    calls = []
    tasks: dict[UUID, tuple[str, str, datetime | None]] = {}
    body = {
        "status": "EXPERIMENT_PUBLISHED",
        "task_id": str(UUID(int=2)),
        "program": {"program_hash": "a" * 64, "kind": "alpha.model-development"},
        "evidence": {"evidence_hash": "b" * 64},
        "result": {"candidates": [{"score": -1}]},
        "next_requests": {
            "continue": {"operation": "EXPERIMENT_DRAFT", "task_id": str(UUID(int=2))}
        },
    }

    def read(request, caller):
        calls.append((request.operation, caller))
        assert request.operation not in {"EXPERIMENT_RUN", "RUN"}
        return deepcopy(body)

    app = GoalApplication(
        GoalStore(tmp_path, "workspace"),
        lambda: NOW,
        read,
        lambda _: NOW + timedelta(seconds=10),
        tasks.get,
        workspace=tmp_path,
    )
    opened = app.operate(
        Request(
            operation="GOAL_OPEN",
            goal_id=UUID(int=1),
            goal_declaration=DECLARATION,
            change_reason="Register before execution",
        ),
        "HUMAN",
    )
    return app, opened, calls, body, tasks


def _attach(app, opened, reference_id="alpha"):
    return app.operate(
        Request(
            operation="GOAL_ATTACH",
            goal_hash=opened["goal_hash"],
            change_reason="Attach exact result",
            goal_reference={
                "reference_id": reference_id,
                "label": "Alpha result",
                "stage": "ALPHA",
                "request": {"operation": "EXPERIMENT_READBACK", "task_id": str(UUID(int=2))},
            },
        ),
        "EXTERNAL_AUTOMATION",
    )


def _submission(**overrides: Any) -> dict[str, Any]:
    document: dict[str, Any] = {
        "outcome": "ACHIEVED",
        "summary": "The signal survives costs on every fold.",
        "criteria": [{"criterion_id": "folds", "answer": "MET", "evidence": ["alpha"]}],
        "deliverables": [{"deliverable_id": "evidence", "references": ["alpha"]}],
    }
    document.update(overrides)
    return document


def test_a_bound_session_names_neither_its_goal_nor_its_hash(goal_app):
    """regression (V391): AX13's agents typed a goal id or hash on 43 of 120 calls although
    goal open had bound their session. A bound session's goal and head are the defaults: it
    attaches, shows and submits naming neither; a request with no session and no id is refused
    with words, not run against some goal."""

    app = goal_app[0]
    taken = app.operate(
        Request(operation="GOAL_TAKE", goal_id=UUID(int=1)), "EXTERNAL_AUTOMATION", SESSION
    )
    assert taken["bound_session"]["session_id"] == SESSION.session
    attached = app.operate(
        Request(
            operation="GOAL_ATTACH",
            change_reason="Attach exact result",
            goal_reference={
                "reference_id": "alpha",
                "label": "Alpha result",
                "stage": "ALPHA",
                "request": {"operation": "EXPERIMENT_READBACK", "task_id": str(UUID(int=2))},
            },
        ),
        "EXTERNAL_AUTOMATION",
        SESSION,
    )
    assert attached["goal_id"] == str(UUID(int=1)) and attached["revision"] == 2
    shown = app.operate(Request(operation="GOAL_SHOW"), "EXTERNAL_AUTOMATION", SESSION)
    assert shown["goal_hash"] == attached["goal_hash"]
    submitted = app.operate(
        Request(operation="GOAL_SUBMIT", goal_submission=_submission()),
        "EXTERNAL_AUTOMATION",
        SESSION,
    )
    assert submitted.get("failure_code") is None and submitted["goal_id"] == str(UUID(int=1))
    with pytest.raises(ValueError, match=r"goal\.goal_id_required"):
        app.operate(Request(operation="GOAL_SHOW"), "EXTERNAL_AUTOMATION", None)


def test_a_held_reference_declared_again_is_refused_with_its_way_on(goal_app):
    """regression (V385): AX13 run 5's submission declared again the `trial` and `review` it had
    attached and cited, and was refused `goal.reference_id_already_used` with no field, ids or
    next request. The refusal names each entry and the held ids, and offers the submission
    without the entries that only declared a held reference again, its citations kept; one
    naming another result under a held id gets no offer, and an attach names its field too."""

    app, opened = goal_app[:2]
    held = _attach(app, opened)
    again = {
        "reference_id": "alpha",
        "label": "Alpha result",
        "stage": "ALPHA",
        "request": {"operation": "EXPERIMENT_READBACK", "task_id": str(UUID(int=2))},
    }
    refused = app.operate(
        Request(
            operation="GOAL_SUBMIT",
            goal_id=UUID(int=1),
            goal_submission=_submission(references=[again]),
        ),
        "EXTERNAL_AUTOMATION",
    )
    assert refused["failure_code"] == "goal.reference_id_already_used"
    assert refused["fields"] == [["goal_submission", "references", 0, "reference_id"]]
    assert refused["expected"] == {"attached_reference_ids": ["alpha"]} and refused["detail"]
    offered = refused["next_requests"]["submit"]
    assert offered["goal_hash"] == held["goal_hash"]
    assert offered["goal_submission"]["references"] == []
    assert offered["goal_submission"]["criteria"][0]["evidence"] == ["alpha"]
    other = {**again, "request": {"operation": "EXPERIMENT_READBACK", "task_id": str(UUID(int=3))}}
    conflict = app.operate(
        Request(
            operation="GOAL_SUBMIT",
            goal_id=UUID(int=1),
            goal_submission=_submission(references=[other]),
        ),
        "EXTERNAL_AUTOMATION",
    )
    assert conflict["failure_code"] == "goal.reference_id_already_used"
    assert "next_requests" not in conflict
    attached = app.operate(
        Request(
            operation="GOAL_ATTACH",
            goal_hash=held["goal_hash"],
            change_reason="Attach again",
            goal_reference=again,
        ),
        "EXTERNAL_AUTOMATION",
    )
    assert attached["fields"] == [["goal_reference", "reference_id"]]
    assert app.store.head(UUID(int=1)).goal_hash == held["goal_hash"]  # nothing was saved
    resent = app.operate(
        PortfolioResearchRequestDocument.model_validate(offered).to_operation_request(),
        "EXTERNAL_AUTOMATION",
    )
    assert resent.get("failure_code") != "goal.reference_id_already_used"


def test_a_revision_conflict_reads_the_goals_latest_revision_by_its_id() -> None:
    """regression (V527, the user's review at b97f7d07): `goal revise --from old-goal.json`
    after another session revised the goal was refused `goal.revision_conflict_read_latest`
    with no words and no request: `goal show --from` the refusal lacked the goal, the old answer
    read the stale revision, and the session's own goal may be another. The refusal says what
    happened and offers the goal's latest revision by its id alone, never the stale hash.
    AX-Dogfood2 met the same refusal when one session reused its own earlier completion form:
    a revision conflict does not establish that another session changed the goal."""

    from types import SimpleNamespace

    from alphalattice.control.product_host.composition.portfolio_research_operations import (
        goal_refusal,
    )

    goal = uuid4()
    request = SimpleNamespace(operation="GOAL_REVISE", goal_id=goal, goal_hash="a" * 64)
    answer = goal_refusal(ValueError("goal.revision_conflict_read_latest"), request)  # type: ignore[arg-type]
    assert answer["failure_code"] == "goal.revision_conflict_read_latest"
    assert answer["detail"] == (
        "This goal changed after the revision this request started "
        "from, so nothing was changed. Read its latest revision by the goal's id "
        "(`next_requests.show`), then make the change on that revision."
    )
    assert answer["next_requests"] == {"show": {"operation": "GOAL_SHOW", "goal_id": str(goal)}}


def test_the_goal_schema_offers_a_declaration_to_edit(goal_app):
    """regression (V378): AX13's agent asked `goal schema --declaration goal.yaml` and was
    sent to experiment and feature controls, which declare no goal; the schema's answer carries
    the shortest declaration, valid as it stands, which the declaration loader writes and reads
    back."""

    from alphalattice.interface.local_application.goals import GoalDeclaration
    from alphalattice.protocols.research_authoring.selection import (
        dump_declaration,
        load_safe_yaml_document,
    )

    app = goal_app[0]
    template = app.operate(Request(operation="GOAL_SCHEMA"), "HUMAN")["template"]
    written = dump_declaration(template)
    declaration = GoalDeclaration.model_validate(load_safe_yaml_document(written))
    assert declaration.kind == "RESEARCH" and declaration.criteria[0].criterion_id == "done"


def test_a_formula_factors_trial_and_packet_are_goal_evidence(goal_app):
    """regression (V371): a goal could not cite a formula factor's trial or review packet, so
    AX11 marked its review criterion MET on a text file listing its exports; both are exact
    reads a goal attaches under DATA_FEATURES, and another stage is refused."""

    app, head, calls, _body, _tasks = goal_app
    for reference_id, request in (
        (
            "packet",
            {
                "operation": "FEATURE_REVIEW",
                "feature_plan_hash": "c" * 64,
                "feature_factor_id": "formula_x",
            },
        ),
        ("trial", {"operation": "FEATURE_TRIAL_READBACK", "feature_trial_id": "d" * 64}),
    ):
        reference = {"reference_id": reference_id, "label": reference_id, "request": request}
        with pytest.raises(ValueError, match="reference_operation_not_admitted"):
            app.operate(
                Request(
                    operation="GOAL_ATTACH",
                    goal_hash=head["goal_hash"],
                    change_reason="Another stage",
                    goal_reference={**reference, "stage": "ALPHA"},
                ),
                "HUMAN",
            )
        head = app.operate(
            Request(
                operation="GOAL_ATTACH",
                goal_hash=head["goal_hash"],
                change_reason="The exact read",
                goal_reference={**reference, "stage": "DATA_FEATURES"},
            ),
            "EXTERNAL_AUTOMATION",
        )
    assert {"FEATURE_REVIEW", "FEATURE_TRIAL_READBACK"} <= {op for op, _caller in calls}
    held = {ref.reference_id: ref.stage for ref in app.store.load(head["goal_hash"]).references}
    assert held == {"packet": "DATA_FEATURES", "trial": "DATA_FEATURES"}


def test_goal_revision_reopen_narrative_no_work_and_tamper(goal_app):
    app, opened, calls, body, _tasks = goal_app
    assert not calls and opened["status"] == "GOAL_SAVED" and opened["state"] == "OPEN"
    attached = _attach(app, opened)
    revised = app.operate(
        Request(
            operation="GOAL_REVISE",
            goal_id=UUID(int=1),
            goal_hash=attached["goal_hash"],
            goal_declaration={**DECLARATION, "title": "Readable title"},
            change_reason="Title only",
        ),
        "HUMAN",
    )
    assert len(calls) == 1  # revisions never invoke execution, PLAN or even another report
    retry = app.operate(
        Request(
            operation="GOAL_REVISE",
            goal_id=UUID(int=1),
            goal_hash=attached["goal_hash"],
            goal_declaration={**DECLARATION, "title": "Readable title"},
            change_reason="Title only",
        ),
        "HUMAN",
    )
    assert retry["goal_hash"] == revised["goal_hash"]
    old = app.store.load(opened["goal_hash"])
    new = app.store.load(revised["goal_hash"])
    assert old.declaration.title == "A study" and new.declaration.title == "Readable title"
    assert new.intent_registered_at == old.intent_registered_at
    assert new.references[0].intent_relation == "QUESTION_RECORDED_BEFORE_TASK_ADMISSION"
    with pytest.raises(ValueError, match="revision_conflict"):
        app.operate(
            Request(
                operation="GOAL_REVISE",
                goal_id=UUID(int=1),
                goal_hash=opened["goal_hash"],
                goal_declaration=DECLARATION,
                change_reason="Stale write",
            ),
            "HUMAN",
        )
    reopened = GoalApplication(
        GoalStore(app.store.content.root.parents[1], "workspace"),
        lambda: NOW,
        app.read,
        app.admitted_at,
        app.task_facts,
        workspace=app.workspace,
    )
    assert reopened.store.load(new.goal_hash) == new
    report = app.operate(Request(operation="GOAL_EXPORT", goal_hash=new.goal_hash), "HUMAN")
    assert report["gaps"] == [{"stage": "RISK", "failure_code": "goal.stage_not_evidenced"}]
    assert report["portfolio_deliveries"] == [] and "Readable title" in report["html"]
    shown = app.operate(Request(operation="GOAL_SHOW", goal_hash=old.goal_hash), "HUMAN")
    assert shown["goal"]["declaration"]["title"] == "A study"
    assert app.operate(Request(operation="GOAL_SHOW", goal_id=UUID(int=1)), "HUMAN")[
        "goal_hash"
    ] == (new.goal_hash)
    body["evidence"]["evidence_hash"] = "c" * 64
    assert app.readback(new, "HUMAN")["references"][0]["state"] == "UNAVAILABLE"
    path = app.store.content.root / "revisions" / f"{new.goal_hash}.json"
    path.write_text(path.read_text().replace("Readable title", "Tampered title"))
    with pytest.raises(ValueError, match="tampered"):
        app.store.load(new.goal_hash)


def test_goal_reference_cannot_execute_or_mislabel_and_a_note_rereads_it(goal_app):
    app, opened, _calls, _body, _tasks = goal_app
    with pytest.raises(ValueError, match="exact_review_required"):
        app.operate(
            Request(
                operation="GOAL_ATTACH",
                goal_hash=opened["goal_hash"],
                change_reason="No implicit latest book",
                goal_reference={
                    "reference_id": "review",
                    "label": "Review",
                    "stage": "EVIDENCE_CRO",
                    "request": {
                        "operation": "EVIDENCE_CRO_EXPORT",
                        "review_publication_hash": "a" * 64,
                    },
                },
            ),
            "HUMAN",
        )
    assert not _calls
    for op, stage in (("EXPERIMENT_RUN", "ALPHA"), ("EXPERIMENT_READBACK", "RISK")):
        with pytest.raises(ValueError, match=r"not_admitted|stage_mismatch"):
            app.operate(
                Request(
                    operation="GOAL_ATTACH",
                    goal_hash=opened["goal_hash"],
                    change_reason="Bad reference",
                    goal_reference={
                        "reference_id": "bad",
                        "label": "Bad",
                        "stage": stage,
                        "request": {"operation": op, "task_id": str(UUID(int=2))},
                    },
                ),
                "HUMAN",
            )
    attached = _attach(app, opened)
    note = {
        "statement_id": "reviewed",
        "kind": "DECISION",
        "attribution": "Risk reviewer",
        "text": "No independent validation",
        "evidence": ["alpha"],
        "disposition": "INSUFFICIENT",
    }
    _calls.clear()
    accepted = app.operate(
        Request(
            operation="GOAL_NOTE",
            goal_hash=attached["goal_hash"],
            change_reason="File attributed review",
            goal_statement=note,
        ),
        "EXTERNAL_AUTOMATION",
    )
    assert app.store.load(accepted["goal_hash"]).statements[0].text == note["text"]
    assert _calls == [("EXPERIMENT_READBACK", "EXTERNAL_AUTOMATION")]
    _body["evidence"]["evidence_hash"] = "e" * 64
    with pytest.raises(ValueError, match="reference_changed"):
        app.operate(
            Request(
                operation="GOAL_NOTE",
                goal_hash=accepted["goal_hash"],
                change_reason="Never cache evidence across requests",
                goal_statement={**note, "statement_id": "reviewed-again"},
            ),
            "EXTERNAL_AUTOMATION",
        )


def _team_event(sequence: int, text: str = "A message.", **subject: str) -> Any:
    return ExternalActivityEventDocument(
        event_kind="NATIVE_COORDINATION_MESSAGE",
        producer_id="claude-code-native",
        producer_session="team-session",
        producer_sequence=sequence,
        occurred_at=NOW,
        summary=text,
        subject={"native_host": "claude-code", "native_session_id": SESSION.session, **subject},
    )


def _filed(app: GoalApplication, document: Any) -> dict[str, Any]:
    filed, goal, session = app.file_event(document, None)
    assert goal is not None
    return app.record_event(goal, filed, f"observation-{document.producer_sequence}", session)


def test_team_messages_are_filed_under_the_goal_their_session_holds(goal_app):
    """requirement (GR2, OP13): Team and the goal are one record. A message is filed under the
    goal its session holds at receipt, and a replay under the goal of its first receipt; an
    assignment seals the packet its assignee is handed, a reply keeps that packet's hash, and
    an assignment no reply closes is missing at submission. The goal is the Host's word."""

    app, opened, _calls, _body, _tasks = goal_app
    _attach(app, opened)
    app.operate(Request(operation="GOAL_TAKE", goal_id=UUID(int=1)), "EXTERNAL_AUTOMATION", SESSION)
    lead = {"native_agent_id": SESSION.session, "role": "research_lead"}
    assignment = _team_event(
        1,
        "Challenge the fold evidence.",
        message_kind="assignment",
        message_id="assign-1",
        recipient_id="child",
        goal_id=str(uuid4()),
        **lead,
    )
    filed, _goal, _session = app.file_event(assignment, None)
    assert filed.subject["goal_id"] == str(UUID(int=1))
    assigned = _filed(app, assignment)
    assert assigned["goal_id"] == str(UUID(int=1)) and len(assigned["packet_hash"]) == 64
    _filed(app, _team_event(2, "Costs first.", message_kind="decision", message_id="n-1", **lead))
    unanswered = app.operate(
        Request(operation="GOAL_SUBMIT", goal_id=UUID(int=1), goal_submission=_submission()),
        "EXTERNAL_AUTOMATION",
    )
    assert unanswered["status"] == "INCOMPLETE"
    assert unanswered["missing"] == [
        {"code": "goal.assignment_unanswered", "message_id": "assign-1", "recipient_id": "child"}
    ]
    replied = _filed(
        app,
        _team_event(
            3,
            "The folds hold; costs are thin.",
            native_agent_id="child",
            role="alphalattice_cro",
            message_kind="answer",
            message_id="answer-1",
            reply_to="assign-1",
        ),
    )
    assert replied["packet_hash"] == assigned["packet_hash"]
    record = app.record(app.store.head(UUID(int=1)))
    kinds = [m["message_kind"] for m in record["conversation"]]
    assert kinds == ["assignment", "decision", "answer"]
    assert record["open_assignments"] == [] and record["message_count"] == 3
    page = app.operate(Request(operation="GOAL_EXPORT", goal_id=UUID(int=1)), "HUMAN")["html"]
    assert "<h2>Conversation</h2>" in page and "The folds hold; costs are thin." in page
    assert record["request_count"] == 0 and record["sessions"] == [
        {"vendor": "claude-code", "session_id": SESSION.session}
    ]
    follow_up = app.operate(
        Request(
            operation="GOAL_OPEN",
            goal_declaration={**DECLARATION, "title": "A follow-up"},
            change_reason="The next question",
        ),
        "EXTERNAL_AUTOMATION",
        SESSION,
    )
    replayed, _goal, _session = app.file_event(assignment, None)
    assert replayed.subject["goal_id"] == str(UUID(int=1))
    _later, held, _session = app.file_event(
        _team_event(4, message_kind="plan", message_id="plan-1", **lead), None
    )
    assert held is not None and str(held.goal_id) == follow_up["goal_id"]
    sealed = app.operate(
        Request(operation="GOAL_SUBMIT", goal_id=UUID(int=1), goal_submission=_submission()),
        "EXTERNAL_AUTOMATION",
    )
    assert sealed["status"] == "COMPLETE"
    with pytest.raises(ValueError, match="closed_open_a_follow_up"):
        app.file_event(
            _team_event(5, message_kind="plan", message_id="plan-2", **lead),
            RequestProvenance(goal_id=str(UUID(int=1))),
        )
    alone, nothing, _session = app.file_event(
        _team_event(
            6,
            native_session_id="unbound-session",
            native_agent_id="unbound-session",
            role="research_lead",
            message_kind="plan",
            message_id="plan-3",
        ),
        None,
    )
    assert nothing is None and "goal_id" not in alone.subject


def _reading(sequence: int, kind: str = "NATIVE_AGENT_USAGE", **subject: str) -> Any:
    return ExternalActivityEventDocument(
        event_kind=kind,
        producer_id="claude-code-native",
        producer_session="team-session",
        producer_sequence=sequence,
        occurred_at=NOW,
        summary="A reading.",
        subject={"native_host": "claude-code", "native_session_id": SESSION.session, **subject},
    )


def test_a_goal_record_gives_its_sessions_models_and_tokens(goal_app):
    """requirement (AU, V300; LAWS ID7): a goal's record gives each participant's latest
    reading by model, the bridge's mark where a reading differs from the role card, and each
    session's totals by model; an older reading of the same agent and model is replaced."""

    app, opened, _calls, _body, _tasks = goal_app
    _attach(app, opened)
    app.operate(Request(operation="GOAL_TAKE", goal_id=UUID(int=1)), "EXTERNAL_AUTOMATION", SESSION)
    child = {"native_agent_id": "child", "role": "alphalattice_cro"}
    counts = {
        "responses": "2",
        "input_tokens": "6",
        "cache_read_tokens": "2000",
        "cache_write_tokens": "400",
        "output_tokens": "47",
    }
    _filed(app, _reading(1, "NATIVE_SUBAGENT_STOP_HOOK", native_hook_event="SubagentStop", **child))
    old = {"model": "claude-sonnet-5", "efforts": "high", "pin_differs": "model"}
    _filed(app, _reading(2, **old, **counts, **child))
    _filed(app, _reading(3, **old, **{**counts, "responses": "3", "output_tokens": "50"}, **child))
    lead = {"native_agent_id": SESSION.session, "role": "research_lead"}
    _filed(app, _reading(4, model="claude-opus-5-5", efforts="max", **counts, **lead))

    record = app.record(app.store.head(UUID(int=1)))
    (session,) = record["session_usage"]
    assert (session["vendor"], session["session_id"]) == ("claude-code", SESSION.session)
    by_agent = {p["agent_id"]: p for p in session["participants"]}
    assert by_agent["child"]["pin_differs"] == ["model"]
    assert by_agent["child"]["models"] == [
        {
            "model": "claude-sonnet-5",
            "efforts": ["high"],
            "responses": 3,
            "input_tokens": 6,
            "cache_read_tokens": 2000,
            "cache_write_tokens": 400,
            "output_tokens": 50,
            "last_at": None,
        }
    ]
    assert by_agent[SESSION.session]["pin_differs"] == []
    assert [(m["model"], m["responses"], m["output_tokens"]) for m in session["by_model"]] == [
        ("claude-opus-5-5", 2, 47),
        ("claude-sonnet-5", 3, 50),
    ]
    assert record["event_count"] == 4 and record["message_count"] == 0


def test_goal_request_fields_and_other_operations_remain_distinct():
    installed = {
        "result_hash": "a" * 64,
        "report_hash": "b" * 64,
        "readouts": {"cumulative_net_wealth": 0.9},
        "window": {"selected_session_count": 42},
    }
    assert GoalApplication._summary(installed) == installed
    request = PortfolioResearchRequestDocument(
        operation="GOAL_LIST", history_limit=5
    ).to_operation_request()
    assert request.operation == "GOAL_LIST"
    with pytest.raises(ValueError, match="not_allowed"):
        PortfolioResearchRequestDocument(
            operation="EXPERIMENT_RUN", experiment_plan_hash="a" * 64, goal_hash="b" * 64
        )
    with pytest.raises(ValueError, match="not_allowed"):
        PortfolioResearchRequestDocument(operation="GOAL_SHOW", goal_hash="b" * 64, spec={})


def test_changed_objective_does_not_relabel_old_intent_and_comparison_refusal_is_evidence(
    goal_app,
):
    app, opened, _calls, body, _tasks = goal_app
    attached = _attach(app, opened)
    revised = app.operate(
        Request(
            operation="GOAL_REVISE",
            goal_id=UUID(int=1),
            goal_hash=attached["goal_hash"],
            goal_declaration={**DECLARATION, "objective": "A different post-observation question"},
            change_reason="Changed research design",
        ),
        "HUMAN",
    )
    assert app.store.load(revised["goal_hash"]).references[0].intent_relation == "POST_HOC"
    assert (
        app.store.load(attached["goal_hash"]).references[0].intent_relation
        == "QUESTION_RECORDED_BEFORE_TASK_ADMISSION"
    )
    body.clear()
    body.update(
        status="REFUSED", failure_code="portfolio_research.comparison_completed_books_required"
    )
    outcome = app.operate(
        Request(
            operation="GOAL_ATTACH",
            goal_hash=revised["goal_hash"],
            change_reason="Preserve incompatibility",
            goal_reference={
                "reference_id": "incompatible",
                "label": "Not comparable",
                "stage": "PORTFOLIO",
                "request": {
                    "operation": "EXPERIMENT_COMPARE",
                    "left_task_id": str(UUID(int=2)),
                    "right_task_id": str(UUID(int=3)),
                },
            },
        ),
        "HUMAN",
    )
    row = app.readback(app.store.load(outcome["goal_hash"]), "HUMAN")["references"][1]
    assert row["state"] == "VERIFIED_REFUSAL"
    assert row["summary"]["owner_refusal"] == body
    assert "dimensions" not in row["summary"]


def test_goal_keeps_descriptive_alpha_comparison_fold_evidence(goal_app):
    app, opened, _calls, body, _tasks = goal_app
    body.clear()
    body.update(
        status="COMPARABLE",
        disposition="DESCRIPTIVE_ALPHA_COMPARISON_NO_SELECTION",
        left={"task_id": str(UUID(int=2)), "candidate_id": "ridge-alpha-1"},
        right={"task_id": str(UUID(int=3)), "candidate_id": "ridge-alpha-100"},
        declared_parameter_difference={"left": {"alpha": 1.0}, "right": {"alpha": 100.0}},
        score_support={"scored_row_count": 12, "common_surface_row_count": 12},
        folds=[
            {
                "fold_commitment_hash": "a" * 64,
                "left": {"status": "DEVELOPMENT_EVALUATED", "metrics": {"rank_ic": 0.1}},
                "right": {"status": "DEVELOPMENT_EVALUATED", "metrics": {"rank_ic": 0.2}},
            }
        ],
        limitations=["NO_STATISTICAL_SIGNIFICANCE_OR_SELECTION_CLAIM"],
    )
    outcome = app.operate(
        Request(
            operation="GOAL_ATTACH",
            goal_hash=opened["goal_hash"],
            change_reason="Attach exact descriptive Alpha comparison",
            goal_reference={
                "reference_id": "alpha-comparison",
                "label": "Ridge regularization comparison",
                "stage": "ALPHA",
                "request": {
                    "operation": "EXPERIMENT_ALPHA_COMPARE",
                    "left_task_id": str(UUID(int=2)),
                    "left_candidate_id": "ridge-alpha-1",
                    "right_task_id": str(UUID(int=3)),
                    "right_candidate_id": "ridge-alpha-100",
                },
            },
        ),
        "HUMAN",
    )
    row = app.readback(app.store.load(outcome["goal_hash"]), "HUMAN")["references"][0]
    assert row["state"] == "VERIFIED_READBACK"
    assert row["summary"]["disposition"] == "DESCRIPTIVE_ALPHA_COMPARISON_NO_SELECTION"
    assert row["summary"]["folds"] == body["folds"]
    assert row["summary"]["score_support"] == body["score_support"]


def test_goal_show_reuses_identical_owner_read_but_verifies_every_reference(goal_app):
    app, opened, calls, _body, _tasks = goal_app
    attached = _attach(app, opened)
    duplicated = _attach(app, attached, reference_id="alpha-again")
    calls.clear()

    report = app.operate(Request(operation="GOAL_SHOW", goal_hash=duplicated["goal_hash"]), "HUMAN")

    assert [row["state"] for row in report["references"]] == [
        "VERIFIED_READBACK",
        "VERIFIED_READBACK",
    ]
    assert report["verification_counts"] == {
        "request_scoped": True,
        "reference_verification_count": 2,
        "owner_read_count": 1,
        "owner_read_cache_hits": 1,
    }
    assert calls == [("EXPERIMENT_READBACK", "HUMAN")]


def test_goal_reference_verifies_only_selected_owner_and_preserves_full_readback(
    goal_app, monkeypatch
):
    """regression (V683): opening one saved reference read every reference's owner."""
    app, opened, calls, _body, _tasks = goal_app
    original_read = app.read
    comparison = {
        "status": "REFUSED",
        "failure_code": "portfolio_research.comparison_completed_books_required",
    }

    def read(request, caller):
        if request.operation == "EXPERIMENT_COMPARE":
            calls.append((request.operation, caller))
            return deepcopy(comparison)
        return original_read(request, caller)

    monkeypatch.setattr(app, "read", read)
    attached = _attach(app, opened)
    duplicated = _attach(app, attached, reference_id="alpha-again")
    complete = app.operate(
        Request(
            operation="GOAL_ATTACH",
            goal_hash=duplicated["goal_hash"],
            change_reason="Retain the unrelated comparison refusal",
            goal_reference={
                "reference_id": "comparison",
                "label": "Unrelated comparison",
                "stage": "PORTFOLIO",
                "request": {
                    "operation": "EXPERIMENT_COMPARE",
                    "left_task_id": str(UUID(int=2)),
                    "right_task_id": str(UUID(int=3)),
                },
            },
        ),
        "HUMAN",
    )
    calls.clear()

    selected = app.operate(
        Request(
            operation="GOAL_REFERENCE",
            goal_hash=complete["goal_hash"],
            goal_reference_id="alpha-again",
        ),
        "HUMAN",
    )

    assert selected["status"] == "GOAL_REFERENCE_READBACK"
    assert selected["goal_hash"] == complete["goal_hash"]
    assert selected["reference"]["reference"]["reference_id"] == "alpha-again"
    assert selected["reference"]["state"] == "VERIFIED_READBACK"
    assert selected["evidence_verification"] == "SELECTED_REFERENCE"
    assert "gaps" not in selected and "references" not in selected
    assert selected["verification_counts"] == {
        "request_scoped": True,
        "reference_verification_count": 1,
        "owner_read_count": 1,
        "owner_read_cache_hits": 0,
    }
    assert calls == [("EXPERIMENT_READBACK", "HUMAN")]

    calls.clear()
    whole = app.operate(Request(operation="GOAL_SHOW", goal_hash=complete["goal_hash"]), "HUMAN")
    assert whole["evidence_verification"] == "COMPLETE"
    assert selected["reference"] == whole["references"][1]
    assert whole["verification_counts"] == {
        "request_scoped": True,
        "reference_verification_count": 3,
        "owner_read_count": 2,
        "owner_read_cache_hits": 1,
    }
    assert calls == [("EXPERIMENT_READBACK", "HUMAN"), ("EXPERIMENT_COMPARE", "HUMAN")]

    calls.clear()
    refused = app.operate(
        Request(
            operation="GOAL_REFERENCE",
            goal_hash=complete["goal_hash"],
            goal_reference_id="comparison",
        ),
        "HUMAN",
    )
    assert calls == [("EXPERIMENT_COMPARE", "HUMAN")]
    assert refused["reference"] == whole["references"][2]
    assert refused["reference"]["state"] == "VERIFIED_REFUSAL"
    assert refused["reference"]["summary"]["owner_refusal"] == comparison


@pytest.mark.parametrize(
    ("change", "expected_code"),
    [
        ("identity", "goal.reference_changed_reattach_explicitly"),
        ("refusal", "research_experiment.publication_evidence_mismatch"),
    ],
)
def test_goal_reference_rechecks_identity_and_preserves_named_owner_unavailability(
    goal_app, change, expected_code
):
    app, opened, calls, body, _tasks = goal_app
    attached = _attach(app, opened)
    request = Request(
        operation="GOAL_REFERENCE",
        goal_hash=attached["goal_hash"],
        goal_reference_id="alpha",
    )
    assert app.operate(request, "HUMAN")["reference"]["state"] == "VERIFIED_READBACK"
    if change == "identity":
        body["evidence"]["evidence_hash"] = "c" * 64
    else:
        body.clear()
        body.update(status="REFUSED", failure_code=expected_code)
    calls.clear()

    selected = app.operate(request, "HUMAN")

    assert calls == [("EXPERIMENT_READBACK", "HUMAN")]
    assert selected["reference"]["state"] == "UNAVAILABLE"
    assert selected["reference"]["failure_code"] == expected_code
    assert "summary" not in selected["reference"]
    assert selected["verification_counts"] == {
        "request_scoped": True,
        "reference_verification_count": 1,
        "owner_read_count": 1,
        "owner_read_cache_hits": 0,
    }
    whole = app.operate(Request(operation="GOAL_SHOW", goal_hash=attached["goal_hash"]), "HUMAN")
    assert selected["reference"] == whole["references"][0]
    assert whole["gaps"][0] == {"reference_id": "alpha", "failure_code": expected_code}


@pytest.mark.parametrize("reference_id", ["absent", "alpha-again"])
def test_goal_reference_requires_a_reference_in_the_exact_saved_revision(goal_app, reference_id):
    app, opened, calls, _body, _tasks = goal_app
    attached = _attach(app, opened)
    _attach(app, attached, reference_id="alpha-again")
    calls.clear()

    with pytest.raises(ValueError, match=r"^goal\.reference_not_found$"):
        app.operate(
            Request(
                operation="GOAL_REFERENCE",
                goal_hash=attached["goal_hash"],
                goal_reference_id=reference_id,
            ),
            "HUMAN",
        )

    assert calls == []


def test_goal_narrative_reads_the_saved_revision_without_any_owner_read(goal_app):
    app, opened, calls, body, _tasks = goal_app
    attached = _attach(app, opened)
    noted = app.operate(
        Request(
            operation="GOAL_NOTE",
            goal_hash=attached["goal_hash"],
            change_reason="Record what the evidence leaves open",
            goal_statement={
                "statement_id": "open-question",
                "kind": "DECISION",
                "attribution": "Researcher",
                "text": "Costs remain unmeasured",
                "evidence": ["alpha"],
                "disposition": "OPEN",
            },
        ),
        "HUMAN",
    )
    calls.clear()
    body["evidence"]["evidence_hash"] = "c" * 64  # the owner has moved on since the attach

    narrative = app.operate(
        Request(operation="GOAL_NARRATIVE", goal_hash=noted["goal_hash"]), "HUMAN"
    )

    assert calls == []
    assert narrative["status"] == "GOAL_NARRATIVE"
    assert narrative["evidence_verification"] == "NOT_PERFORMED"
    assert "gaps" not in narrative and "verification_counts" not in narrative
    assert [row["state"] for row in narrative["references"]] == ["SAVED_NOT_VERIFIED"]
    assert narrative["goal"]["statements"][0]["text"] == "Costs remain unmeasured"
    assert narrative["open_choices"][0]["statement_id"] == "open-question"
    assert narrative["statement_context"]["open-question"]["design"] == "CURRENT_DESIGN"
    assert narrative["head_hash"] == noted["goal_hash"]
    assert narrative["next_requests"]["show"] == {
        "operation": "GOAL_SHOW",
        "goal_id": str(UUID(int=1)),
    }

    verified = app.operate(Request(operation="GOAL_SHOW", goal_hash=noted["goal_hash"]), "HUMAN")
    assert calls == [("EXPERIMENT_READBACK", "HUMAN")]
    assert verified["evidence_verification"] == "COMPLETE"
    assert verified["references"][0]["state"] == "UNAVAILABLE"
    assert verified["gaps"][0]["reference_id"] == "alpha"
    for key in ("goal", "statement_context", "head_hash", "outcome", "open_choices", "record"):
        assert verified[key] == narrative[key]
    listed = app.operate(Request(operation="GOAL_LIST"), "HUMAN")["goals"]
    assert [(row["state"], row["outcome"]) for row in listed] == [("OPEN", None)]

    path = app.store.content.root / "revisions" / f"{noted['goal_hash']}.json"
    path.write_text(path.read_text().replace("Costs remain", "Costs never"))
    with pytest.raises(ValueError, match="tampered"):
        app.operate(Request(operation="GOAL_NARRATIVE", goal_hash=noted["goal_hash"]), "HUMAN")


def test_a_submission_is_checked_against_the_record_and_names_what_is_missing(goal_app):
    """requirement (OP13): the Host checks only what it can decide, against its own record of
    the goal's Tasks, and answers each missing item; a complete record seals the goal."""

    app, opened, _calls, _body, tasks = goal_app
    attached = _attach(app, opened)
    goal = app.store.load(attached["goal_hash"])
    done, running, stuck = UUID(int=11), UUID(int=12), UUID(int=13)
    tasks.update(
        {
            done: ("alpha_research.development", "SUCCEEDED", None),
            running: ("alpha_research.development", "RUNNING", None),
            stuck: ("alpha_research.development", "RECOVERY_REQUIRED", None),
        }
    )
    for task in (done, running, stuck):
        app.attribute(
            goal,
            Request(operation="EXPERIMENT_RUN", experiment_plan_hash="p" * 64),
            {"status": "ADMITTED", "task_id": str(task)},
            SESSION,
        )
    shown = app.operate(Request(operation="GOAL_SHOW", goal_id=UUID(int=1)), "HUMAN")
    assert shown["record"]["request_count"] == 3
    assert shown["record"]["sessions"] == [{"vendor": "claude-code", "session_id": SESSION.session}]
    listed = app.operate(Request(operation="GOAL_LIST"), "HUMAN")["goals"]
    assert listed[0]["sessions"] == shown["record"]["sessions"], "U23: a row names its sessions"
    # U54: an open goal has no deliverables and no check yet; a session lists its own goals.
    assert listed[0]["deliverables"] is None and listed[0]["completion"] is None
    mine = app.operate(Request(operation="GOAL_LIST", agent_session=SESSION.session), "HUMAN")
    assert [row["goal_id"] for row in mine["goals"]] == [str(UUID(int=1))]
    assert mine["goals"][0]["sessions"] == shown["record"]["sessions"]
    others = app.operate(Request(operation="GOAL_LIST", agent_session="nobody"), "HUMAN")
    assert others["goals"] == [] and others["next_cursor"] is None

    first = app.operate(
        Request(
            operation="GOAL_SUBMIT",
            goal_id=UUID(int=1),
            goal_submission=_submission(
                criteria=[{"criterion_id": "folds", "answer": "MET"}],
                deliverables=[],
                findings=[{"finding_id": "f", "text": "Cited", "evidence": ["nowhere"]}],
            ),
        ),
        "EXTERNAL_AUTOMATION",
    )
    assert first["status"] == "INCOMPLETE"
    assert {
        (
            m["code"],
            m.get("task_id")
            or m.get("criterion_id")
            or m.get("deliverable_id")
            or m.get("reference_id"),
        )
        for m in first["missing"]
    } == {
        ("goal.criterion_evidence_required", "folds"),
        ("goal.deliverable_missing", "evidence"),
        ("goal.reference_absent", "nowhere"),
        ("goal.task_running", str(running)),
        ("goal.task_unaccounted", str(stuck)),
    }
    assert app.store.head(UUID(int=1)).state == "OPEN"  # nothing was sealed

    tasks[running] = ("alpha_research.development", "SUCCEEDED", None)
    sealed = app.operate(
        Request(
            operation="GOAL_SUBMIT",
            goal_id=UUID(int=1),
            goal_submission=_submission(
                problems=[
                    {"problem_id": "stuck", "text": "Needs recovery", "task_ids": [str(stuck)]}
                ],
                files=[
                    {
                        "reference_id": "notes",
                        "name": "notes.md",
                        "media_type": "text/markdown",
                        "text": "# Decision notes",
                    }
                ],
            ),
        ),
        "EXTERNAL_AUTOMATION",
    )
    assert sealed["status"] == "COMPLETE" and sealed["outcome"] == "ACHIEVED"
    closed = app.store.head(UUID(int=1))
    assert closed.state == "COMPLETE" and closed.submission.files[0].text == "# Decision notes"
    assert {t.task_id for t in closed.completion.tasks} == {done, running, stuck}
    assert closed.completion.sessions[0].session_id == SESSION.session
    # U54: a complete goal's row names its deliverables and when the Host checked it.
    (row,) = app.operate(Request(operation="GOAL_LIST", agent_session=SESSION.session), "HUMAN")[
        "goals"
    ]
    assert row["deliverables"] == [
        {
            "deliverable_id": "evidence",
            "kind": "RESULT",
            "description": "The exact result",
            "reference_count": 1,
        }
    ]
    assert row["completion"] == {"checked_at": closed.completion.checked_at.isoformat()}
    for operation, extra in (
        ("GOAL_SUBMIT", {"goal_id": UUID(int=1), "goal_submission": _submission()}),
        ("GOAL_ABANDON", {"goal_id": UUID(int=1), "change_reason": "Too late"}),
    ):
        with pytest.raises(ValueError, match="closed"):
            app.operate(Request(operation=operation, **extra), "HUMAN")
    with pytest.raises(ValueError, match="closed"):
        _attach(app, sealed, reference_id="late")
    with pytest.raises(ValueError, match="closed"):
        app.attributed_goal(RequestProvenance(goal_id=str(UUID(int=1))))


def test_goal_task_fact_refetches_its_offset_update_time_after_restart(goal_app):
    """A goal reads the current Task Control clock each time and preserves its UTC offset."""

    app, opened, _calls, _body, tasks = goal_app
    task_id = UUID(int=31)
    first = datetime.fromisoformat("2026-10-05T13:14:15+05:45")
    goal = app.store.load(opened["goal_hash"])
    tasks[task_id] = ("factor_research", "QUEUED", first)
    app.attribute(
        goal,
        Request(operation="EXPERIMENT_RUN", experiment_plan_hash="p" * 64),
        {"status": "ADMITTED", "task_id": str(task_id)},
        SESSION,
    )

    expected = {
        "task_id": str(task_id),
        "kind": "factor_research",
        "state": "QUEUED",
        "updated_at": first.isoformat(),
    }
    shown = app.operate(Request(operation="GOAL_SHOW", goal_id=goal.goal_id), "HUMAN")
    assert shown["record"]["tasks"] == [expected]

    refreshed = datetime.fromisoformat("2026-10-06T02:03:04-07:00")
    tasks[task_id] = ("factor_research", "SUCCEEDED", refreshed)
    expected.update(state="SUCCEEDED", updated_at=refreshed.isoformat())
    refetched = app.operate(Request(operation="GOAL_SHOW", goal_id=goal.goal_id), "HUMAN")
    assert refetched["record"]["tasks"] == [expected]

    reopened = GoalApplication(
        GoalStore(app.store.content.root.parents[1], "workspace"),
        lambda: NOW,
        app.read,
        app.admitted_at,
        app.task_facts,
        workspace=app.workspace,
    )
    readback = reopened.operate(Request(operation="GOAL_SHOW", goal_id=goal.goal_id), "HUMAN")
    assert readback["record"]["tasks"] == [expected]


def test_a_complete_goal_without_task_update_time_reads_its_legacy_seal(goal_app):
    """A pre-field COMPLETE payload has no clock key, so its existing goal hash still proves."""

    app, opened, _calls, _body, tasks = goal_app
    attached = _attach(app, opened)
    task_id = UUID(int=32)
    tasks[task_id] = ("factor_research", "SUCCEEDED", None)
    goal = app.store.load(attached["goal_hash"])
    app.attribute(
        goal,
        Request(operation="EXPERIMENT_RUN", experiment_plan_hash="p" * 64),
        {"status": "ADMITTED", "task_id": str(task_id)},
        SESSION,
    )
    sealed = app.operate(
        Request(
            operation="GOAL_SUBMIT",
            goal_id=goal.goal_id,
            goal_submission=_submission(),
        ),
        "EXTERNAL_AUTOMATION",
    )
    assert sealed["status"] == "COMPLETE"
    path = app.store.content.root / "revisions" / f"{sealed['goal_hash']}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    (legacy_task,) = payload["completion"]["tasks"]
    assert legacy_task == {
        "task_id": str(task_id),
        "kind": "factor_research",
        "state": "SUCCEEDED",
    }

    reopened = GoalApplication(
        GoalStore(app.store.content.root.parents[1], "workspace"),
        lambda: NOW,
        app.read,
        app.admitted_at,
        app.task_facts,
        workspace=app.workspace,
    )
    loaded = reopened.store.load(sealed["goal_hash"])
    assert loaded.state == "COMPLETE" and loaded.completion.tasks[0].updated_at is None
    shown = reopened.operate(Request(operation="GOAL_SHOW", goal_hash=sealed["goal_hash"]), "HUMAN")
    assert shown["goal_hash"] == sealed["goal_hash"]
    assert shown["goal"]["completion"]["tasks"] == [legacy_task]
    assert shown["record"]["tasks"] == [legacy_task]


def test_goal_task_fact_uses_the_live_registry_record_update_time(live):
    """The product composition hands the Goal owner Task Control's persisted TaskRecord clock."""

    from tests.workspace_task_runner.task_control_support import task_contract

    observed_at = datetime.fromisoformat("2026-10-05T13:14:15+05:45")
    refreshed = datetime.fromisoformat("2026-10-06T02:03:04-07:00")
    envelope, task_goal, plan = task_contract(salt="goal-task-current-update-time")
    registry = live.session.task_control_registry
    task = registry.admit(
        input_envelope=envelope,
        goal=task_goal,
        plan=plan,
        observed_at=observed_at,
    ).record
    canonical = registry.task(task.task_id)
    app = live.operations.goals
    goal_id = uuid4()
    opened = app.operate(
        Request(
            operation="GOAL_OPEN",
            goal_id=goal_id,
            goal_declaration=DECLARATION,
            change_reason="Register before execution",
        ),
        "HUMAN",
    )
    goal = app.store.load(opened["goal_hash"])
    app.attribute(
        goal,
        Request(operation="EXPERIMENT_RUN", experiment_plan_hash="p" * 64),
        {"status": "ADMITTED", "task_id": str(task.task_id)},
        SESSION,
    )

    expected = {
        "task_id": str(canonical.task_id),
        "kind": canonical.task_kind,
        "state": canonical.lifecycle.value,
        "updated_at": canonical.updated_at.isoformat(),
    }
    shown = app.operate(Request(operation="GOAL_SHOW", goal_id=goal_id), "HUMAN")
    assert shown["record"]["tasks"] == [expected]

    registry.request_cancel(
        task_id=task.task_id,
        expected_task_hash=canonical.record_hash,
        observed_at=refreshed,
    )
    updated = registry.task(task.task_id)
    assert updated.updated_at.isoformat() == refreshed.isoformat()
    expected.update(state=updated.lifecycle.value, updated_at=updated.updated_at.isoformat())
    refetched = app.operate(Request(operation="GOAL_SHOW", goal_id=goal_id), "HUMAN")
    assert refetched["record"]["tasks"] == [expected]

    # A second Host composes a fresh TaskControl registry from the same workspace root.
    live.stop()
    live.start()
    restarted_registry = live.session.task_control_registry
    restarted_task = restarted_registry.task(task.task_id)
    assert restarted_task == updated
    restarted = live.operations.goals.operate(
        Request(operation="GOAL_SHOW", goal_id=goal_id), "HUMAN"
    )
    assert restarted["record"]["tasks"] == [expected]


def test_goal_task_fact_rejects_a_naive_update_time():
    from alphalattice.interface.local_application.goals import GoalTaskFact

    with pytest.raises(ValueError, match=r"goal\.task_updated_at_must_be_aware"):
        GoalTaskFact(
            task_id=UUID(int=33),
            kind="factor_research",
            state="QUEUED",
            updated_at=datetime(2026, 10, 6, 2, 3, 4),
        )


def test_an_unmet_objective_completes_but_an_answer_that_contradicts_it_does_not(goal_app):
    """requirement (OP13): complete is the record's, not the objective's; NOT_ACHIEVED seals,
    and ACHIEVED beside a criterion NOT_MET, or NOT_ASSESSED without a reason, is missing."""

    app, opened, _calls, _body, _tasks = goal_app
    _attach(app, opened)
    contradicted = app.operate(
        Request(
            operation="GOAL_SUBMIT",
            goal_id=UUID(int=1),
            goal_submission=_submission(
                criteria=[{"criterion_id": "folds", "answer": "NOT_MET", "evidence": ["alpha"]}]
            ),
        ),
        "EXTERNAL_AUTOMATION",
    )
    assert [m["code"] for m in contradicted["missing"]] == ["goal.outcome_contradicts_criteria"]
    unreasoned = app.operate(
        Request(
            operation="GOAL_SUBMIT",
            goal_id=UUID(int=1),
            goal_submission=_submission(
                outcome="PARTLY_ACHIEVED",
                criteria=[{"criterion_id": "folds", "answer": "NOT_ASSESSED"}],
            ),
        ),
        "EXTERNAL_AUTOMATION",
    )
    assert [m["code"] for m in unreasoned["missing"]] == ["goal.criterion_reason_required"]
    sealed = app.operate(
        Request(
            operation="GOAL_SUBMIT",
            goal_id=UUID(int=1),
            goal_submission=_submission(
                outcome="NOT_ACHIEVED",
                summary="Costs erase the signal.",
                criteria=[{"criterion_id": "folds", "answer": "NOT_MET", "evidence": ["alpha"]}],
            ),
        ),
        "EXTERNAL_AUTOMATION",
    )
    assert sealed["status"] == "COMPLETE" and sealed["outcome"] == "NOT_ACHIEVED"


def _cli(live: Any, *arguments: str) -> tuple[int, dict[str, Any]]:
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
    return result.returncode, json.loads(result.stdout)


def test_a_session_takes_a_goal_and_its_requests_are_recorded_under_it(
    live, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """requirement (OP13, V282): a goal opened in an agent session binds it, so each request
    the session sends is the goal's record with no agent effort; a closed goal takes none."""

    declaration = tmp_path / "goal.json"
    declaration.write_text(
        json.dumps({**DECLARATION, "kind": "DATA", "research": None}), encoding="utf-8"
    )
    spec = tmp_path / "spec.yaml"
    spec.write_text("{}", encoding="utf-8")
    code, refused = _cli(live, "goal", "take", str(uuid4()))
    assert code == 2 and refused["data"]["failure_code"] == "goal.not_found"
    continuing = tmp_path / "continuing.json"
    research = {**DECLARATION["research"], "purpose": "CONTINUATION"}  # type: ignore[dict-item]
    continuing.write_text(json.dumps({**DECLARATION, "research": research}), encoding="utf-8")
    code, refused = _cli(live, "goal", "open", "--file", str(continuing))
    assert refused["data"]["failure_code"] == "goal.continuation_requires_parent_goal"
    # V320: the refusal names the field and the way on.
    assert refused["data"]["fields"] == [["research", "purpose"], ["parent_goal_id"]]
    assert refused["data"]["next_requests"]["goals"]["operation"] == "GOAL_LIST"
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", SESSION.session)
    monkeypatch.setenv("ALPHALATTICE_SHELL", "posix")
    code, opened = _cli(live, "goal", "open", "--file", str(declaration))
    assert code == 0, opened
    goal_id = opened["data"]["goal_id"]
    assert opened["data"]["bound_session"] == {
        "vendor": "claude-code",
        "session_id": SESSION.session,
    }
    assert opened["data"]["goal_prompt"].startswith("/goal ")
    assert len(opened["data"]["goal_prompt"]) <= 4000
    # The handoff's first command runs as written: it names the workspace every command needs,
    # quoted as the CLI quotes its commands (V492, the user's review).
    import shlex

    take = opened["data"]["goal_prompt"].split("First run `", 1)[1].split("`", 1)[0]
    assert shlex.split(take) == [
        "alphalattice",
        "--workspace",
        str(live.workspace),
        "goal",
        "take",
        goal_id,
    ]
    code, _planned = _cli(live, "strategy-book", "preview", "--file", str(spec))
    assert code == 0
    code, shown = _cli(live, "goal", "show", goal_id)
    assert code == 0 and shown["data"]["record"]["request_count"] == 1
    assert shown["data"]["record"]["sessions"] == [
        {"vendor": "claude-code", "session_id": SESSION.session}
    ]
    code, abandoned = _cli(live, "goal", "abandon", goal_id, "--reason", "Superseded")
    assert code == 0 and abandoned["data"]["state"] == "ABANDONED"
    code, _planned = _cli(live, "strategy-book", "preview", "--file", str(spec))
    assert code == 0  # a closed goal binds no more work: the request runs unattributed
    code, shown = _cli(live, "goal", "show", goal_id)
    assert shown["data"]["record"]["request_count"] == 1
    code, named = _cli(live, "--goal", goal_id, "strategy-book", "preview", "--file", str(spec))
    assert code == 2 and named["data"]["failure_code"] == "goal.closed_open_a_follow_up"
    monkeypatch.delenv("CLAUDE_CODE_SESSION_ID")
    code, refused = _cli(live, "goal", "take", goal_id)
    assert code == 2 and refused["data"]["failure_code"] == "goal.closed_open_a_follow_up"


def test_a_team_message_reaches_its_goal_through_the_host(
    live, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """requirement (GR2, V283): the Host's event ingress files what the native bridge delivers
    under the goal its session holds; the receipt names the goal and an assignment's packet, a
    reply binds that packet, and the Team feed's rows keep the goal."""

    from alphalattice.interface.local_application.native_bridge import (
        BINDING_NAME,
        NativeResearchBinding,
        coordination_event,
        deliver,
    )
    from tests.portfolio_strategy_lab.local_web_support import _json

    declaration = tmp_path / "goal.json"
    declaration.write_text(
        json.dumps({**DECLARATION, "kind": "REVIEW", "research": None}), encoding="utf-8"
    )
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", SESSION.session)
    code, opened = _cli(live, "goal", "open", "--file", str(declaration))
    assert code == 0, opened
    goal_id = opened["data"]["goal_id"]
    project = tmp_path / "project"
    (project / ".codex").mkdir(parents=True)
    (project / ".codex" / BINDING_NAME).write_text(
        json.dumps(
            {
                "session_id": SESSION.session,
                "workspace": str(live.workspace),
                "roles": ["alphalattice_cro"],
                "host": "claude-code",
            }
        ),
        encoding="utf-8",
    )
    binding = NativeResearchBinding.read(project)
    assert binding is not None
    assigned = deliver(
        project,
        binding,
        coordination_event(
            binding,
            agent_id=SESSION.session,
            role="research_lead",
            kind="assignment",
            message_id="assign-1",
            message=b"Challenge the fold evidence.",
            recipient_id="child",
        ),
    )
    assert assigned["status"] == "DELIVERED" and assigned["goal_id"] == goal_id
    answered = deliver(
        project,
        binding,
        coordination_event(
            binding,
            agent_id="child",
            role="alphalattice_cro",
            kind="answer",
            message_id="answer-1",
            message=b"The folds hold; costs are thin.",
            reply_to="assign-1",
        ),
    )
    assert answered["goal_id"] == goal_id
    assert answered["packet_hash"] == assigned["packet_hash"]
    code, shown = _cli(live, "goal", "show", goal_id)
    record = shown["data"]["record"]
    assert [(m["message_kind"], m["agent_id"]) for m in record["conversation"]] == [
        ("assignment", SESSION.session),
        ("answer", "child"),
    ]
    assert record["open_assignments"] == [] and record["request_count"] == 0
    rows = _json(live, "/api/activity/external")["items"]
    assert [row["payload"]["subject"]["goal_id"] for row in rows] == [goal_id, goal_id]


def test_the_workbench_reaches_what_the_ui_pass_reads(live, tmp_path: Path):
    """requirement (V321, UI-P D10): the goals a person keeps, the saved studies' sweep and
    Guanyin's incidents answer on the Local Web, each route its registered operation, a write
    behind the session as every write is; an agent's own goal steps stay on the client route."""

    from tests.portfolio_strategy_lab.local_web_support import _json, _request

    body = {"goal_declaration": DECLARATION, "change_reason": "Register before execution"}
    denied, _, _ = _request(live, "/api/goals/open", method="POST", payload=body, token=None)
    assert denied == 403
    opened = _json(live, "/api/goals/open", method="POST", payload=body)
    goal_id = opened["goal_id"]
    assert opened["goal_prompt"].startswith("/goal ")
    assert goal_id in {row["goal_id"] for row in _json(live, "/api/goals")["goals"]}
    shown = _json(live, "/api/goals/show?goal_id=" + goal_id)
    assert shown["goal"]["declaration"]["title"] == DECLARATION["title"]
    assert "declaration" in _json(live, "/api/goals/schema")["schemas"]
    statement = {
        "statement_id": "costs",
        "kind": "DECISION",
        "attribution": "PM",
        "text": "Costs first",
        "disposition": "OPEN",
    }
    noted = _json(
        live,
        "/api/goals/note",
        method="POST",
        payload={
            "goal_id": goal_id,
            "goal_hash": shown["goal_hash"],
            "goal_statement": statement,
            "change_reason": "A decision",
        },
    )
    assert noted.get("status") != "REFUSED", noted
    assert _json(live, "/api/goals/narrative?goal_id=" + goal_id).get("status") != "REFUSED"
    assert "incidents" in _json(live, "/api/tasks/incidents")
    swept = _json(live, "/api/experiments/verify-all", method="POST", payload={})
    assert swept.get("failure_code") != "local_web.route_not_found", swept
    status, _, _ = _request(live, "/api/goals/take", method="POST", payload={"goal_id": goal_id})
    assert status == 404


def test_the_case_page_routes_answer_from_goals_and_never_admit_a_task(live):
    """The research case page keeps working over goals until the UI pass (U23)."""

    from types import SimpleNamespace

    from alphalattice.interface.local_application.client import LocalResearchClient
    from tests.portfolio_strategy_lab.local_web_support import _json, _request

    tasks = tuple(live.operations.workspace_session.task_control_registry.tasks())
    document = {
        "title": "A study",
        "question": "Does the signal survive costs?",
        "purpose": "NEW_RESEARCH",
        "scope": "One admitted historical input; full declared support",
        "comparison_design": "Same target and costs; no retuning",
        "criteria": ["Inspect all folds, including negative results"],
        "required_stages": ["ALPHA", "RISK"],
        "budget": {"maximum_tasks": 2, "maximum_numerical_calls": 10, "maximum_model_calls": 2},
    }
    body = {"case_id": str(UUID(int=7)), "case_document": document, "change_reason": "Question"}
    denied, _, _ = _request(
        live, "/api/research/case/save", method="POST", payload=body, token=None
    )
    assert denied == 403
    saved = _json(live, "/api/research/case/save", method="POST", payload=body)
    assert saved["status"] == "CASE_SAVED" and saved["case_id"] == str(UUID(int=7))
    read = _json(live, "/api/research/case?case_hash=" + saved["case_hash"])
    assert read["status"] == "CASE_READBACK"
    assert read["case"]["document"] == {**document, "constraints": []}
    assert read["case"]["submitted_by"] == "HUMAN" and read["case_hash"] == saved["case_hash"]
    narrative = _json(live, "/api/research/case/narrative?case_hash=" + saved["case_hash"])
    assert narrative["status"] == "CASE_NARRATIVE" and narrative["case"] == read["case"]
    listed = _json(live, "/api/research/cases")["cases"]
    assert [(row["case_hash"], row["question"], row["purpose"]) for row in listed] == [
        (saved["case_hash"], document["question"], "NEW_RESEARCH")
    ]
    goal = live.operations.execute(Request(operation="GOAL_SHOW", goal_hash=saved["case_hash"]))
    assert goal["goal"]["declaration"]["kind"] == "RESEARCH"
    assert tuple(live.operations.workspace_session.task_control_registry.tasks()) == tasks
    client = object.__new__(LocalResearchClient)
    client.connection = SimpleNamespace(url=live.url)
    url = client.selected_url({"operation": "GOAL_SHOW", "goal_hash": saved["case_hash"]}, goal)
    assert "/#" in url and saved["case_hash"] in url
    assert client.navigation({"operation": "GOAL_SHOW"}, goal)["kind"] == "goal"


def test_an_open_goal_offers_its_completion_to_fill(goal_app):
    """requirement (V419, V426; an outside review at 3fa785fd and 97b65a25): a completion
    repeated what the Host holds, and once written could complete another goal: `goal show`
    answers the open goal's completion to fill, a whole GOAL_SUBMIT request bound to the goal
    and its revision, each slot by its id and the goal's references as the evidence to cite.
    Left as it stands it is refused; a later revision refuses it; written, it completes its own
    goal whatever goal the sending session holds."""

    import yaml

    from alphalattice.interface.local_application.goals import GoalSubmission

    app, opened, *_ = goal_app
    attached = _attach(app, opened)
    show = Request(operation="GOAL_SHOW", goal_id=UUID(int=1))
    text = app.operate(show, "HUMAN")["yaml"]
    assert "#   alpha: Alpha result (ALPHA)" in text
    assert "# Inspect all folds, including negative ones" in text
    template = yaml.safe_load(text)
    assert (template["operation"], template["goal_id"]) == ("GOAL_SUBMIT", str(UUID(int=1)))
    assert template["goal_hash"] == attached["goal_hash"]
    filled = template["goal_submission"]
    assert [row["criterion_id"] for row in filled["criteria"]] == ["folds"]
    assert [row["deliverable_id"] for row in filled["deliverables"]] == ["evidence"]
    with pytest.raises(ValueError, match="outcome"):
        GoalSubmission.model_validate(filled)

    def written(document: dict[str, Any]) -> Request:
        judged = document["goal_submission"]
        judged.update(outcome="PARTLY_ACHIEVED", summary="The signal survives costs.")
        judged["criteria"][0].update(answer="MET", evidence=["alpha"])
        judged["deliverables"][0]["references"] = ["alpha"]
        return PortfolioResearchRequestDocument.model_validate(document).to_operation_request()

    # A later revision refuses the earlier one's completion.
    _attach(app, {"goal_hash": attached["goal_hash"]}, reference_id="beta")
    with pytest.raises(ValueError, match="revision_conflict"):
        app.operate(written(template), "EXTERNAL_AUTOMATION")
    # A session holding another goal, with criteria of the same ids, completes this one.
    app.operate(
        Request(
            operation="GOAL_OPEN",
            goal_id=UUID(int=5),
            goal_declaration=DECLARATION,
            change_reason="Another goal, held by the session",
        ),
        "EXTERNAL_AUTOMATION",
        SESSION,
    )
    current = yaml.safe_load(app.operate(show, "HUMAN")["yaml"])
    sealed = app.operate(written(current), "EXTERNAL_AUTOMATION", SESSION)
    assert sealed["goal_id"] == str(UUID(int=1)), sealed
    assert app.store.head(UUID(int=5)).state == "OPEN"


def test_a_closed_goal_refuses_with_its_standing_and_the_follow_up_start() -> None:
    """regression (V437, AX14's finding): a second submit to a completed goal was refused
    `goal.closed_open_a_follow_up` with no words and no way on; the refusal stays, and it names
    the goal's standing (its show request, by the goal the request named) and where a follow-up
    starts (the declaration template)."""

    from alphalattice.control.product_host.composition.portfolio_research_operations import (
        goal_refusal,
    )

    request = Request(operation="GOAL_SUBMIT", goal_id=UUID(int=1), goal_submission={})
    refused = goal_refusal(ValueError("goal.closed_open_a_follow_up"), request)
    assert (refused["status"], refused["failure_code"]) == (
        "REFUSED",
        "goal.closed_open_a_follow_up",
    )
    assert "parent_goal_id" in str(refused["detail"])
    assert refused["next_requests"] == {
        "show": {"operation": "GOAL_SHOW", "goal_id": str(UUID(int=1))},
        "schema": {"operation": "GOAL_SCHEMA"},
    }
    held = goal_refusal(
        ValueError("goal.closed_open_a_follow_up"),
        Request(operation="GOAL_SUBMIT", goal_submission={}),
    )
    assert held["next_requests"]["show"] == {"operation": "GOAL_SHOW"}  # the session's own


def test_an_open_goals_requests_leave_their_documents_to_give(goal_app, tmp_path) -> None:
    """regression (V441, an outside review at bbb1b9e8): the offered submit, revise and resubmit
    held their open fields as text placeholders, which the client binds, so a valid `goal submit
    --from goal.json --file submission.yaml` was refused `bound_reference_override`; the open
    fields are None, and the goal and its revision stay bound."""

    from alphalattice.interface.local_application import client

    app, opened, *_ = goal_app
    attached = _attach(app, opened)
    shown = app.operate(Request(operation="GOAL_SHOW", goal_id=UUID(int=1)), "HUMAN")
    offered = shown["next_requests"]
    assert offered["submit"]["goal_submission"] is None
    assert offered["revise"]["goal_declaration"] is None
    assert offered["revise"]["change_reason"] is None
    saved = tmp_path / "goal.json"
    saved.write_text(json.dumps(shown), encoding="utf-8")
    sent = client.continuation(
        "GOAL_SUBMIT",
        saved,
        {"goal_submission": _submission()},
        frozenset({"goal_id", "goal_hash", "goal_submission"}),
    )
    assert sent == {
        "operation": "GOAL_SUBMIT",
        "goal_id": str(UUID(int=1)),
        "goal_hash": attached["goal_hash"],
        "goal_submission": _submission(),
    }


def test_a_goals_head_keeps_a_missing_or_corrupt_revisions_refusal(goal_app):
    """CONTRACT: a kept head's lost revision offers recovery, not an invalid-pointer claim."""
    from alphalattice.control.workspace_runtime.content_store import ContentAddressedStoreError
    from alphalattice.interface.local_application.cli_contract import refusal_words

    app, opened, calls, _body, _tasks = goal_app
    identity = opened["goal_hash"]
    revision = app.store.content.root / "revisions" / f"{identity}.json"
    original = revision.read_bytes()
    revision.unlink()
    request = Request(operation="GOAL_SHOW", goal_id=UUID(int=1))
    with pytest.raises(ContentAddressedStoreError) as missing:
        app.operate(request, "HUMAN")
    assert str(missing.value) == f"content_store.artifact_missing:{identity}"
    assert "restore a backup" in refusal_words(str(missing.value))["detail"]
    revision.write_bytes(b"{}")
    with pytest.raises(ContentAddressedStoreError, match=r"^content_store\.artifact_tampered$"):
        app.operate(request, "HUMAN")
    revision.write_bytes(original)
    assert app.operate(request, "HUMAN")["goal_hash"] == identity
    # A corrupt head pointer is still the head owner's distinct refusal.
    pointer = app.store.content.root / "heads" / f"{UUID(int=1)}.json"
    pointer.write_bytes(b"{}")
    with pytest.raises(ValueError, match=r"^goal\.head_invalid$"):
        app.operate(request, "HUMAN")
    assert not calls


def test_goal_list_keeps_readable_rows_and_scopes_only_provable_head_refusals(goal_app):
    """A corrupt head cannot hide other goals or disclose an unproved session association."""
    app, _opened, _calls, _body, _tasks = goal_app
    other_session = RequestProvenance(
        vendor="claude-code", session="00000000-0000-4000-8000-000000000004"
    )

    def open_goal(goal_id: UUID, provenance: RequestProvenance) -> dict[str, Any]:
        result = app.operate(
            Request(
                operation="GOAL_OPEN",
                goal_id=goal_id,
                goal_declaration=DECLARATION,
                change_reason="Register before execution",
            ),
            "HUMAN",
            provenance,
        )
        app.attribute(
            app.store.load(result["goal_hash"]),
            Request(operation="GOAL_SHOW", goal_id=goal_id),
            {"status": "AVAILABLE"},
            provenance,
        )
        return result

    damaged_id, unreadable_attribution_id, other_id = UUID(int=2), UUID(int=3), UUID(int=4)
    open_goal(damaged_id, SESSION)
    unreadable_attribution = open_goal(unreadable_attribution_id, SESSION)
    other = open_goal(other_id, other_session)

    clean = app.operate(Request(operation="GOAL_LIST"), "HUMAN")
    assert "refused" not in clean
    assert {row["goal_id"] for row in clean["goals"]} == {
        str(UUID(int=1)),
        str(damaged_id),
        str(unreadable_attribution_id),
        str(other_id),
    }

    head = app.store.content.root / "heads" / f"{damaged_id}.json"
    head.write_bytes(b"[]")  # valid JSON, wrong pointer shape
    attribution = (
        app.store.content.root / "attribution" / str(unreadable_attribution_id) / "00000000.json"
    )
    attribution.write_bytes(b"[]")  # valid JSON, wrong attribution entry shape

    all_goals = app.operate(Request(operation="GOAL_LIST"), "HUMAN")
    assert {row["goal_id"] for row in all_goals["goals"]} == {
        str(UUID(int=1)),
        str(other_id),
    }
    refusals = {item["goal_id"]: item for item in all_goals["refused"]}
    assert set(refusals) == {str(damaged_id), str(unreadable_attribution_id)}
    assert refusals[str(damaged_id)]["item"] == "HEAD"
    assert refusals[str(unreadable_attribution_id)]["item"] == "SESSION_ATTRIBUTION"
    assert refusals[str(unreadable_attribution_id)]["failure_code"] == ("goal.attribution_invalid")
    assert "session association is inferred" in refusals[str(unreadable_attribution_id)]["detail"]

    refused_page = app.operate(
        Request(
            operation="GOAL_LIST",
            history_limit=1,
            history_cursor=other["goal_hash"],
        ),
        "HUMAN",
    )
    assert refused_page["goals"] == []
    assert refused_page["next_cursor"] == unreadable_attribution["goal_hash"]
    following_page = app.operate(
        Request(
            operation="GOAL_LIST",
            history_limit=1,
            history_cursor=refused_page["next_cursor"],
        ),
        "HUMAN",
    )
    assert [row["goal_id"] for row in following_page["goals"]] == [str(UUID(int=1))]
    assert following_page["next_cursor"] is None

    mine = app.operate(Request(operation="GOAL_LIST", agent_session=SESSION.session), "HUMAN")
    assert mine["goals"] == []
    assert len(mine["refused"]) == 1
    refusal = mine["refused"][0]
    assert (refusal["status"], refusal["goal_id"], refusal["item"], refusal["failure_code"]) == (
        "REFUSED",
        str(damaged_id),
        "HEAD",
        "goal.head_invalid",
    )
    assert "No prior revision is inferred" in refusal["detail"]
    assert refusal["next_requests"]["backups"]["operation"] == "WORKSPACE_BACKUPS"
    assert refusal["next_requests"]["schema"]["operation"] == "GOAL_SCHEMA"

    theirs = app.operate(
        Request(operation="GOAL_LIST", agent_session=other_session.session), "HUMAN"
    )
    assert [row["goal_id"] for row in theirs["goals"]] == [str(other_id)]
    assert "refused" not in theirs
    assert theirs["goals"][0]["goal_hash"] == other["goal_hash"]


@pytest.mark.parametrize(
    "parent_sender,recipient,flood,wakes",
    [
        (SESSION.session, None, False, True),
        (SESSION.session, None, True, True),
        ("peer-a", None, False, False),
        (SESSION.session, "peer-b", False, False),
    ],
)
def test_a_reply_past_the_shown_conversation_reaches_only_its_addressee(
    live,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    parent_sender: str,
    recipient: str | None,
    flood: bool,
    wakes: bool,
) -> None:
    """CONTRACT: full-ledger replies, explicit recipients and between-poll arrivals.

    The real bridge files the assignment and fifty later messages. The real CLI waiter
    must hear the sender's reply even when its parent, or the reply itself, left the shown
    tail; a reply addressed to another agent does not wake the lead.
    """
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    from alphalattice.interface.local_application.native_bridge import (
        BINDING_NAME,
        NativeResearchBinding,
        coordination_event,
        deliver,
    )
    from tests.portfolio_strategy_lab.local_web_support import _json

    monkeypatch.delenv("CODEX_THREAD_ID", raising=False)
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", SESSION.session)
    declaration = tmp_path / "goal.json"
    declaration.write_text(
        json.dumps({**DECLARATION, "kind": "REVIEW", "research": None}),
        encoding="utf-8",
        newline="\n",
    )
    code, opened = _cli(live, "goal", "open", "--file", str(declaration))
    assert code == 0, opened
    goal_id = opened["data"]["goal_id"]
    project = tmp_path / "project"
    (project / ".codex").mkdir(parents=True)
    (project / ".codex" / BINDING_NAME).write_text(
        json.dumps(
            {
                "session_id": SESSION.session,
                "workspace": str(live.workspace),
                "roles": ["alphalattice_cro"],
                "host": "claude-code",
            }
        ),
        encoding="utf-8",
        newline="\n",
    )
    binding = NativeResearchBinding.read(project)
    assert binding is not None

    def send(sender: str, kind: str, ident: str, **options: str) -> dict[str, Any]:
        answer = deliver(
            project,
            binding,
            coordination_event(
                binding,
                agent_id=sender,
                role="research_lead" if sender == SESSION.session else "alphalattice_cro",
                kind=kind,
                message_id=ident,
                message=ident.encode(),
                **options,
            ),
        )
        assert answer["status"] == "DELIVERED", answer
        return answer

    assigned = send(
        parent_sender,
        "assignment" if parent_sender == SESSION.session else "question",
        "old-assignment",
        recipient_id="child",
    )
    for index in range(50):
        send("peer-a", "question", f"later-{index}", recipient_id="peer-b")
    code, shown = _cli(live, "goal", "show", goal_id)
    assert code == 0 and shown["data"]["record"]["message_count"] == 51
    assert all(m["message_id"] != "old-assignment" for m in shown["data"]["record"]["conversation"])

    first_read, arrivals_done = Event(), Event()
    narrative = live.operations.goals.narrative

    def read_after_arrivals(goal):
        # Synchronize at the public owner's read, without replacing its result or the waiter.
        if first_read.is_set():
            assert arrivals_done.wait(20)
        body = narrative(goal)
        first_read.set()
        return body

    monkeypatch.setattr(live.operations.goals, "narrative", read_after_arrivals)
    with ThreadPoolExecutor(max_workers=1) as pool:
        waiting = pool.submit(_cli, live, "activity", "wait", "--goal", goal_id, "--max-wait", "8")
        try:
            assert first_read.wait(20)
            replied = send(
                "child",
                "answer",
                "deep-reply",
                reply_to="old-assignment",
                **({"recipient_id": recipient} if recipient else {}),
            )
            if parent_sender == SESSION.session:
                assert replied["packet_hash"] == assigned["packet_hash"]
            else:
                assert replied.get("packet_hash") is None
            if flood:
                for index in range(50):
                    send("peer-a", "question", f"after-reply-{index}", recipient_id="peer-b")
        finally:
            arrivals_done.set()
        code, final = waiting.result(timeout=30)
    event = final["data"]["wait_event"]
    assert (
        (code, event["event"]) == (0, "MESSAGE")
        if wakes
        else (code, event["event"]) == (3, "MAX_WAIT_REACHED")
    ), final
    if wakes:
        assert (event["message_id"], event["recipient"]) == ("deep-reply", SESSION.session)
    record = live.operations.goals.narrative(live.operations.goals.store.head(UUID(goal_id)))[
        "record"
    ]
    reply = next(m for m in record["conversation"] if m["message_id"] == "deep-reply")
    assert reply["recipient_id"] == (recipient or parent_sender)
    assert record["open_assignments"] == []
    assert len(record["conversation"]) == record["message_count"] == (102 if flood else 52)
    rows = _json(live, "/api/activity/external?limit=200")["items"]
    kept = next(
        r["payload"]["subject"]
        for r in rows
        if r["payload"]["subject"].get("message_id") == "deep-reply"
    )
    assert kept["recipient_id"] == reply["recipient_id"], "Team reads the same resolved recipient"
