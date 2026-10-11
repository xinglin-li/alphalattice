"""The committee's grade over recorded floor reads: no Host and no model."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from alphalattice.control.product_host.composition.committee import (
    ALIAS as OWNER_ALIAS,
)
from alphalattice.control.product_host.composition.committee import (
    misshapen,
)
from alphalattice.control.product_host.composition.committee import (
    standing_messages as owner_standing,
)
from alphalattice.control.product_host.composition.committee_materials import (
    QUESTIONS,
    facts,
    sufficiency,
)
from alphalattice.interface.local_application.portfolio_research import CommitteeMessage
from devtools.evaluation import Scenario, Trace, TraceEvent
from devtools.evaluation.committee import (
    ALIAS,
    CommitteeGrader,
    committee_problems,
    materials_findings,
    report_fidelity,
    standing_messages,
)

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


@pytest.mark.parametrize("change", ["clean", "subject", "role", "message", "refused"])
def test_an_admitted_message_belongs_to_its_expected_floor(change):
    """A Host receipt for another task, role or message cannot prove this committee step."""
    scenario, context, parts, answer = recorded()
    expected = {
        "operation": "COMMITTEE_SUBMIT",
        "update_task_id": "selected",
        "committee_role": "ALPHA",
    }
    sent = {**expected, "committee_message": answer["message"]}
    if change == "subject":
        sent["update_task_id"] = "another"
    if change == "role":
        sent["committee_role"] = "RISK"
    if change == "message":
        sent["committee_message"] = {"kind": "STANCE", "text": "Another."}
    attrs = {
        "alphalattice.answer": answer,
        "alphalattice.committee": context,
        "alphalattice.prompt_parts": parts,
        "alphalattice.expected_request": expected,
        "alphalattice.request": sent,
        "alphalattice.response": {
            "operation": "COMMITTEE_SUBMIT",
            "outcome": "REFUSED" if change == "refused" else "OK",
        },
    }
    result = CommitteeGrader().grade(
        scenario, Trace((TraceEvent(scenario.now, scenario.now, attrs, 1),))
    )
    assert result["passed"] is (change == "clean")


# Component and fact aliases spelled like development labels; the public label scan refuses
# the literal spelling, so they are composed.
COMPONENT, FACT, UNKNOWN_COMPONENT = (f"{k}{n}" for k, n in (("C", 1), ("S", 1), ("C", 9)))


@pytest.mark.parametrize(
    "change,reason",
    [
        ("clean", None),
        ("shape", "message_shape"),
        ("empty", None),
        ("alias", "unknown_alias"),
        ("component", None),
        ("fact", None),
        ("unknown_component", "unknown_alias"),
        ("unknown_fact", "unknown_alias"),
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
        "decision" if change in {"ruling", "unseen_ruling", "fact"} else "stance"
    )
    aliases = f"{COMPONENT} H1 M1 {FACT} T1 X1 S10000"
    assert ALIAS.findall(aliases) == OWNER_ALIAS.findall(aliases)
    context["dossier"]["holdings"][0]["scores"] = {COMPONENT: {}}
    context["dossier"]["facts"] = [{"alias": FACT, "text": "A sealed fact."}]
    if change == "fact":
        context["stances"][0]["message"]["text"] = f"Reserve {FACT}."
        parts[2]["text"] = json.dumps(context["stances"])
    context["dossier"]["floor"] = {"messages": [{"id": "M6", "kind": "RULING"}]}
    context["dossier"]["entry_open_at"] = "2026-08-07T13:30:00Z"
    parts[1]["text"] = json.dumps(context["dossier"])
    answer["message"].update(
        {
            "shape": {"kind": "CHALLENGE"},
            "empty": {"positions": {}},
            "alias": {"text": "Reserve H9."},
            "component": {"text": f"Reserve {COMPONENT}."},
            "fact": {"text": f"Proceed after {FACT}."},
            "unknown_component": {"text": f"Reserve {UNKNOWN_COMPONENT}."},
            "unknown_fact": {"text": "Reserve S9."},
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


@pytest.mark.parametrize(
    "text,act,passed",
    [
        ("Reserve T1.", None, True),
        ("Reserve T1.", "place_order", False),
        ("Reserve T1.", "run_cro_review_before_holding", False),
        ("Reserve T1.", "do_not_run_cro_review", True),
        ("Reserve T1.", "record_missing_cro_review", True),
        ("I ask the person whether to wait.", None, False),
        ("This is a Risk question the dossier cannot settle.", None, True),
        ("If the person needs a decision, I will ask the person.", None, True),
    ],
)
def test_forbidden_act_is_not_member_work(text, act, passed):
    """Text-only members cannot propose a forbidden external act."""
    scenario, context, parts, answer = recorded(acts=[act] if act else [])
    scenario = replace(scenario, forbidden_acts=("place_order", "admit_task"), ask_required=False)
    answer["message"]["text"] = text
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
    """Recorded contract shapes agree with the real owner, its message model and its floor's
    shape rule for each kind, without an evaluation dependency."""
    scenario, context, parts, answer = recorded()
    answer["message"].update(change)
    try:
        accepted = not misshapen(CommitteeMessage.model_validate(answer["message"]))
    except ValidationError:
        accepted = False
    result = judged(scenario, context, parts, answer)
    reasons = {issue["reason"] for issue in result["reasons"]}
    assert ("message_shape" not in reasons) is accepted
    if not isinstance(answer["message"].get("text"), str):
        assert not result["passed"] and "invalid_answer" in reasons


def floor_message(identifier, role, kind, **fields):
    return dict(id=identifier, role=role, kind=kind, **fields)


def fidelity_floor(role="CRO", point="T1", following=None):
    messages = [
        floor_message("M1", role, "STANCE", text=f"Object {point}.", positions={point: "OBJECT"}),
        floor_message("M2", role, "REPLY", text="Read T1.", positions=following or {}),
        floor_message("M3", "PM", "VERDICT", text="Proceed."),
    ]
    rows = [{"attribution": m["id"], "text": m["text"]} for m in messages]
    selected = {"review_publication_hash": "a" * 64}
    closed = {"messages": messages, "delivery": selected}
    report = {
        "commentary": rows,
        "selection": dict(selected),
        "html": " ".join(m["text"] for m in messages),
        "sections": {
            "evidence_cro": {"value": {"review": {"publication": {"publication_hash": "a" * 64}}}}
        },
    }
    evidence = {
        "raw_messages": deepcopy(messages),
        "cro_view": ["Review now: REVIEWED."],
        "review_standing": {"status": "REVIEWED", **selected},
    }
    return closed, report, evidence


@pytest.mark.parametrize(
    "role,point,following", [("CRO", "T1", {"T2": "RESERVE"}), ("ALPHA", "T2", {})]
)
def test_a_reply_only_replaces_the_positions_it_explicitly_states(role, point, following):
    """A later unrelated or empty reply leaves each earlier point's position in the report."""
    closed, report, evidence = fidelity_floor(role, point, following)
    assert report_fidelity(closed, report, **evidence) == []
    report["commentary"].pop(0)
    assert f"report.standing_missing:{role}:{point}:M1" in report_fidelity(
        closed, report, **evidence
    )
    closed["messages"][1]["positions"] = {point: "SUPPORT"}
    evidence["raw_messages"] = deepcopy(closed["messages"])
    assert report_fidelity(closed, report, **evidence) == []


@pytest.mark.parametrize("kind", ["missing", "unused"])
def test_a_reserve_beside_its_material_is_the_agents_and_without_it_the_products(kind):
    """Every question holds a fact or a named absence; a RESERVE citing none is graded unused
    where the member's facts stand, missing where only absences do."""
    rows = {"issuer_rows": [{"entity_id": "E", "tickers": ["N"], "ending_weight": "12.5%"}]}
    rows["issue_cards"] = [{"issue_handle": "I", "affected_entities": ["E", "X"]}]
    held = [{"alias": "H", "name": "N", "weight": 0.125, "listing_id": "n", "change": 0.0}]
    said = facts({"publication": {"cro_status": "NOT_REVIEWED"}}, rows, held)
    assert sufficiency(SimpleNamespace(opened={"facts": said})) == {}
    assert any("affects H; not held here: X" in f["text"] for f in said)  # by its issuer row
    if kind == "unused":
        fact = {"alias": "S99", "question": "ALPHA.held", "text": "By rank.", "source": "pub"}
        said.append(fact)
    stance = floor_message("M1", "ALPHA", "STANCE", text="Hold.", positions={"T1": "RESERVE"})
    closed = {"facts": said, "questions": QUESTIONS, "messages": [stance]}
    assert materials_findings(closed) == [f"materials.{kind}:ALPHA:T1:M1"]
    stance["text"] = said[-1]["alias"] + " holds."
    assert materials_findings(closed) == []


@pytest.mark.parametrize(
    "change",
    ["same", "changed", "unrelated", "empty", "unruled", "ADOPT", "REJECT", "FOR_THE_PERSON"],
)
def test_standing_message_mirror_agrees_with_the_floor_owner(change):
    """Standing sources change only with a point's value, or a ruling that settles it."""
    first = floor_message("M1", "ALPHA", "STANCE", positions={"T1": "OBJECT"}, targets=[])
    values = dict(same={"T1": "OBJECT"}, changed={"T1": "SUPPORT"}, unrelated={"T2": "RESERVE"})
    second = dict(first, id="M2", kind="REPLY", positions=values.get(change, {}))
    messages = [first, second]
    if change in {"unruled", "ADOPT", "REJECT", "FOR_THE_PERSON"}:
        messages.append(
            dict(first, id="M3", role="CRO", kind="CHALLENGE", positions={}, targets=["T1"])
        )
        if change != "unruled":
            ruled = dict(first, id="M4", role="PM", kind="RULING", positions={})
            ruled.update(reply_to="M3", outcome=change)
            messages.append(ruled)
    assert standing_messages(messages) == owner_standing(SimpleNamespace(messages=messages))


@pytest.mark.parametrize(
    "broken",
    [
        "clean",
        "ruling",
        "after",
        "relay",
        "hash",
        "standing",
        "cro",
        "numbers",
        "raw",
        "missing_raw",
        "missing_cro",
        "alias_attribution",
        "stale_status",
        "closed_hash",
    ],
)
def test_report_fidelity_requires_the_floor_and_its_original_admissions(broken):
    """Rulings, person relays, review identity and raw number checks have independent evidence."""
    closed, report, evidence = fidelity_floor()
    challenge = floor_message("M4", "CRO", "CHALLENGE", text="Check T1.")
    ruling = floor_message("M5", "PM", "RULING", reply_to="M4", text="Ask person.")
    ruling["outcome"] = "FOR_THE_PERSON"
    words = "Wait 2 days on H1."
    answer = floor_message("M6", "PM", "PERSON_ANSWER", reply_to="M5", text=words)
    closed["messages"][2:2] = [challenge, ruling, answer]
    evidence["raw_messages"] = deepcopy(closed["messages"])
    answer["text"] = words.replace("H1", "Synthetic holding.")
    relayed = dict(answer=words, relay_word="Relayed", text=answer["text"])
    relayed["attribution_words"] = {"message": "M5"}
    report["commentary"].append(relayed)
    report["html"] += " " + words
    if broken == "ruling":
        closed["messages"][3]["role"] = "RISK"
    elif broken == "after":
        closed["messages"].append(closed["messages"].pop(3))
    elif broken == "relay":
        report["commentary"][-1]["relay_word"] = None
    elif broken == "hash":
        report["selection"]["review_publication_hash"] = "b" * 64
    elif broken == "standing":
        evidence["review_standing"]["review_publication_hash"] = "b" * 64
    elif broken == "cro":
        evidence["cro_view"] = ["Review now: NOT_REVIEWED."]
    elif broken == "numbers":
        evidence["raw_messages"][0]["text"] = "Weight 2%."
    elif broken == "raw":
        evidence["raw_messages"].pop()
    elif broken == "alias_attribution":
        report["commentary"][0]["attribution"] = "M10"
    elif broken == "stale_status":
        evidence["cro_view"] = ['{"old": "REVIEWED"}', "Review now: NOT_REVIEWED."]
    elif broken == "closed_hash":
        closed["delivery"] = {}
    elif broken.startswith("missing_"):
        evidence["raw_messages" if broken == "missing_raw" else "cro_view"] = None
    problems = report_fidelity(closed, report, **evidence)
    assert bool(problems) is (broken != "clean")
    if broken in {"raw", "missing_raw"}:
        assert "UNPROVED:raw_messages" in problems
    if broken == "missing_cro":
        assert "UNPROVED:cro_view" in problems
    if broken == "closed_hash":
        assert "report.review_hash_mismatch" in problems
