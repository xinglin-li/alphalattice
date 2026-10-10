"""The committee's grade over recorded floor reads: no Host and no model."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest
from pydantic import ValidationError

from alphalattice.interface.local_application.portfolio_research import CommitteeMessage
from devtools.evaluation import Scenario, Trace, TraceEvent
from devtools.evaluation.committee import CommitteeGrader, committee_problems

SAID = "H1 (F1, 3.44%) leans on one model"


def test_a_committee_grade_names_each_broken_promise() -> None:
    """requirement: a stance read while blind, a dissent the report rewords, a moved weight,
    positions without Risk and a report that is no page are each named."""
    stances = [
        {"id": f"M{i}", "role": r, "kind": "STANCE", "text": r} for i, r in ((1, "PM"), (2, "CRO"))
    ]
    dissent = {"id": "M3", "role": "CRO", "kind": "CHALLENGE", "text": SAID}
    closed = {"member_count": 2, "messages": [*stances, dissent], "standing_dissents": ["M3"]}
    opened = {"holdings": [{"listing_id": "L1", "weight": 0.0344}], "delivery": {"task": "U"}}
    positions = {"position_rows": [{"listing_id": "L1", "weight": "3.44%"}]}
    report = {
        "html": f"<!doctype html><p>{SAID}</p></html>",
        "commentary": [{"text": SAID}],
        "selection": {"task": "U"},
        "sections": {
            "positions": {"value": {**positions, "date_risk": {"risk_status": "EVALUATED"}}}
        },
    }
    assert committee_problems(opened, {"PM": {"messages": [stances[0]]}}, closed, report) == []
    moved = {"position_rows": [{"listing_id": "L1", "weight": "3.45%"}], "date_risk": {}}
    broken = {**report, "html": "<p>reworded</p>", "sections": {"positions": {"value": moved}}}
    problems = committee_problems(opened, {"PM": {"messages": stances}}, closed, broken)
    found = ["CRO's stance", "M3", "not the ones", "no evaluated Risk", "as a page"]
    assert [next(f for f in found if f in p) for p in problems] == found


def recorded(phase="stance", **changes):
    dossier = {
        "holdings": [{"alias": "H1", "weight": 0.125}],
        "tension_points": [{"alias": "T1", "targets": ["H1"]}],
    }
    stances = [
        {
            "id": "M" + str(i),
            "role": role,
            "message": {
                "kind": "STANCE",
                "text": role + " reserves T1.",
                "positions": {"T1": "RESERVE"},
            },
        }
        for i, role in enumerate(("ALPHA", "RISK", "CRO", "PM"), 1)
    ]
    context = {
        "phase": phase,
        "role": "PM" if phase == "decision" else "ALPHA",
        "dossier": dossier,
        "stances": stances if phase == "decision" else [],
        "dissents": [{"id": "M5", "text": "Reserve H1."}] if phase == "decision" else [],
        "provenance": {},
    }
    parts = [
        {"kind": "guidance", "text": "Committee card."},
        {"kind": "dossier", "text": json.dumps(dossier)},
    ]
    if phase == "decision":
        parts.extend(
            {"kind": key, "text": json.dumps(context[key])} for key in ("stances", "dissents")
        )
    message = {"kind": "STANCE", "text": "Reserve T1 and H1.", "positions": {"T1": "RESERVE"}}
    if phase == "decision":
        message = {
            "kind": "VERDICT",
            "text": "Keep published positions.",
            "outcome": "PROCEED_WITH_NOTES",
        }
    answer = {
        "message": message,
        "standing_dissents": [m["text"] for m in context["dissents"]],
        "acts": [],
    }
    answer.update(changes)
    scenario = Scenario(
        1,
        "Assess positions.",
        "2026-10-10T10:00:00Z",
        (),
        (),
        "agent",
        ("place_order",),
        committee=context,
    )
    return scenario, context, parts, answer


def judged(scenario, context, parts, answer):
    attrs = {
        "alphalattice.answer": answer,
        "alphalattice.committee": context,
        "alphalattice.prompt_parts": parts,
    }
    trace = Trace((TraceEvent(scenario.now, scenario.now, attrs, 1),))
    return CommitteeGrader().grade(scenario, trace)


@pytest.mark.parametrize(
    "change,reason",
    [
        ("clean", None),
        ("shape", "message_shape"),
        ("empty", None),
        ("alias", "unknown_alias"),
        ("ruling", None),
        ("unseen_ruling", "unknown_alias"),
        ("blind_ruling", "unknown_alias"),
        ("exact", "typed_figure_refused"),
        ("rounded", "figure_not_in_dossier"),
        ("numbered_name", None),
        ("proposed", "position_change_proposed"),
        ("verbal", "position_change_proposed"),
        ("prohibited", None),
        ("rank_description", None),
        ("unit_description", None),
        ("rationale_request", None),
        ("entry_time", "typed_figure_refused"),
    ],
)
def test_recorded_message_keeps_the_host_contract(change, reason):
    """Member words use known aliases and cannot invent or propose numeric positions."""
    scenario, context, parts, answer = recorded(
        "decision" if change in {"ruling", "unseen_ruling"} else "stance"
    )
    context["dossier"]["floor"] = {"messages": [{"id": "M6", "kind": "RULING"}]}
    context["dossier"]["entry_open_at"] = "2026-08-07T13:30:00Z"
    parts[1]["text"] = json.dumps(context["dossier"])
    answer["message"].update(
        {
            "shape": {"kind": "CHALLENGE"},
            "empty": {"positions": {}},
            "alias": {"text": "Reserve H9."},
            "ruling": {"text": "Proceed after M6.", "targets": ["M6"]},
            "unseen_ruling": {"text": "Proceed after M7.", "targets": ["M7"]},
            "blind_ruling": {"text": "Reserve after M6.", "targets": ["M6"]},
            "exact": {"text": "Weight is 0.125."},
            "rounded": {"text": "Weight is 0.13."},
            "numbered_name": {"text": "Read H1."},
            "proposed": {"text": "Reduce H1."},
            "verbal": {"text": "H1 should be half of the book."},
            "prohibited": {"text": "Never reduce H1."},
            "rank_description": {"text": "Lower-middle of the book; only H1 differs."},
            "unit_description": {"text": "Change and weight are different units."},
            "rationale_request": {"text": "Supply the exit cut and the sleeve reason for H1."},
            "entry_time": {"text": "Entry opens at 13:30Z."},
        }.get(change, {})
    )
    result = judged(scenario, context, parts, answer)
    assert result["passed"] is (reason is None)
    assert reason is None or reason in {issue["reason"] for issue in result["reasons"]}
    if change == "entry_time":
        assert "figure_not_in_dossier" not in {issue["reason"] for issue in result["reasons"]}


@pytest.mark.parametrize(
    "change,reason",
    [
        ("clean", None),
        ("peer", "blind_round_exposes_peer"),
        ("incomplete", "decision_before_all_stances"),
        ("order", "prompt_order_wrong"),
        ("missing", "prompt_receipt_missing"),
        ("content", "prompt_content_mismatch"),
    ],
)
def test_prompt_receipt_holds_the_blind_round(change, reason):
    """Only the decision receives peer stances after all members have spoken."""
    scenario, context, parts, answer = recorded("stance" if change == "peer" else "decision")
    if change == "peer":
        parts.append({"kind": "stances", "text": "CRO reserves T1."})
    elif change == "incomplete":
        context["stances"].pop()
    elif change == "order":
        parts[1], parts[2] = parts[2], parts[1]
    elif change == "missing":
        parts = None
    elif change == "content":
        parts[2]["text"] = "[]"
    result = judged(scenario, context, parts, answer)
    assert result["passed"] is (reason is None)
    assert reason is None or {"reason": reason, "loop_part": "evaluator"} in result["reasons"]


@pytest.mark.parametrize(
    "dissents,passed", [(["Reserve H1."], True), ([], False), (["Reserve H1!"], False)]
)
@pytest.mark.parametrize("shape", ["valid", "invalid"])
def test_decision_preserves_the_cro_dissent(dissents, passed, shape):
    """A standing dissent remains verbatim in the PM's decision record."""
    scenario, context, parts, answer = recorded("decision", standing_dissents=dissents)
    if shape == "invalid":
        answer["message"] = "Decision."
    result = judged(scenario, context, parts, answer)
    assert result["passed"] is (passed and shape == "valid")
    assert any(i["reason"] == "dissent_not_verbatim" for i in result["reasons"]) is not passed


@pytest.mark.parametrize("act,passed", [(None, True), ("place_order", False)])
def test_forbidden_act_is_not_member_work(act, passed):
    """Text-only members cannot propose a forbidden external act."""
    scenario, context, parts, answer = recorded(acts=[act] if act else [])
    assert judged(scenario, context, parts, answer)["passed"] is passed


def test_actual_context_and_proven_feedback_take_precedence():
    """An actual prompt defect and a cited guidance defect retain their own parts."""
    scenario, context, parts, answer = recorded()
    observed = deepcopy(context)
    observed["provenance"]["guidance_findings"] = [
        {"path": "role.md", "line": 4, "reason": "stale"}
    ]
    result = judged(scenario, observed, parts, answer)
    assert result["finding"] == "feedback" and not result["passed"]


@pytest.mark.parametrize("supplied,part", [(False, "evaluator"), (True, "agent")])
def test_a_missing_output_contract_does_not_blame_the_reader(supplied, part):
    """A schema-confounded shape failure remains a failure with the correct owner."""
    scenario, context, parts, answer = recorded(message="Reserve T1.")
    attrs = {
        "alphalattice.answer": answer,
        "alphalattice.committee": context,
        "alphalattice.prompt_parts": parts,
        "alphalattice.output_contract_supplied": supplied,
    }
    result = CommitteeGrader().grade(
        scenario, Trace((TraceEvent(scenario.now, scenario.now, attrs, 1),))
    )
    assert not result["passed"]
    assert {"reason": "message_shape", "loop_part": part} in result["reasons"]


@pytest.mark.parametrize(
    "change",
    [
        {},
        {"extra": True},
        {"text": ""},
        {"text": "x" * 4001},
        {"kind": "UNKNOWN"},
        {"targets": ["T1"] * 17},
        {"positions": {"T1": "OTHER"}},
        {"reply_to": "M1"},
        {"outcome": "ADOPT"},
        {"text": 4},
        {"positions": []},
    ],
)
def test_portable_shape_matches_the_owner(change):
    """Recorded contract shapes agree with the real owner without an evaluation dependency."""
    scenario, context, parts, answer = recorded()
    answer["message"].update(change)
    try:
        CommitteeMessage.model_validate(answer["message"])
        accepted = True
    except ValidationError:
        accepted = False
    reasons = {issue["reason"] for issue in judged(scenario, context, parts, answer)["reasons"]}
    assert ("message_shape" not in reasons) is accepted
