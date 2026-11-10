"""Each native gate's verdict comes from one leg's own graded results, by the lane's rules."""

import hashlib
import json

import pytest

from devtools.evaluation.leg_report import GRADES, gate_verdicts

TEST_SOURCE = "src/devtools/evaluation/leg_report.py"
pytestmark = pytest.mark.proves("evaluation.native_gate_verdicts_follow_the_release_lane_rules")
SHA = "a" * 40


def leg(
    run,
    *,
    state="DONE",
    host="claude-code",
    previous="0.2.3",
    budget="COMPLETE",
    upgrade="ALL EQUAL",
    cancel=True,
):
    grades = {}
    for name in (*GRADES, "fresh/budget.json"):
        (run / name).parent.mkdir(parents=True, exist_ok=True)
        (run / name).write_text(json.dumps(dict(held=True, status=budget)), encoding="utf-8")
        grades[name] = hashlib.sha256((run / name).read_bytes()).hexdigest()
    row = dict(state=state, source=SHA, host=host, stop=None, previous=previous, grades=grades)
    (run / "leg.json").write_text(json.dumps(row), encoding="utf-8")
    released_cancel = dict(held=cancel, task="t")
    summary = dict(
        candidate_sha=SHA,
        held=cancel,  # acceptance's whole verdict: the upgrade and the released cancel
        upgrade=dict(held=True, book_verdict=upgrade, released_cancel=released_cancel),
    )
    (run / "acceptance-summary.json").write_text(json.dumps(summary), encoding="utf-8")


@pytest.mark.parametrize(
    ("change", "verdicts"),
    [
        ({}, {"native-claude-code": True, "resource-budget": True, "upgrade-axis": True}),
        ({"host": "codex", "previous": None}, {"native-codex": True, "resource-budget": True}),
        (
            {"state": "STOPPED"},
            {"native-claude-code": False, "resource-budget": False, "upgrade-axis": True},
        ),
        (
            {"budget": "INCOMPLETE"},
            {"native-claude-code": True, "resource-budget": False, "upgrade-axis": True},
        ),
        (
            {"upgrade": "DIFFERENT"},
            {"native-claude-code": True, "resource-budget": True, "upgrade-axis": False},
        ),
        (  # the upgrade held and its book matched, but the released cancel failed
            {"cancel": False},
            {"native-claude-code": True, "resource-budget": True, "upgrade-axis": False},
        ),
    ],
)
def test_each_gate_reads_its_own_result_from_the_leg(tmp_path, change, verdicts):
    leg(tmp_path, **change)
    assert gate_verdicts(tmp_path, SHA) == verdicts


@pytest.mark.parametrize("fault", ["ungraded", "absent", "other-source"])
def test_a_result_the_leg_did_not_grade_is_a_failed_gate(tmp_path, fault):
    leg(tmp_path)
    target = tmp_path / "fresh/summary.json"
    if fault == "ungraded":  # a result beside the leg proves nothing unless the leg graded it
        row = json.loads((tmp_path / "leg.json").read_text(encoding="utf-8"))
        del row["grades"]["fresh/summary.json"]
        (tmp_path / "leg.json").write_text(json.dumps(row), encoding="utf-8")
    elif fault == "absent":
        target.unlink()
    found = gate_verdicts(tmp_path, SHA if fault != "other-source" else "b" * 40)
    assert found["native-claude-code"] is False and found["resource-budget"] is False
    assert found["upgrade-axis"] is (fault != "other-source")
