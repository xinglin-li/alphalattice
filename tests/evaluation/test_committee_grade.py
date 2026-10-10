"""The committee's grade over recorded floor reads: no Host and no model."""

from __future__ import annotations

from devtools.evaluation.committee import committee_problems

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
