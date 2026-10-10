"""The path explorer's grading of recorded answers: no Host and no model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from devtools.evaluation.explorer import Explorer
from devtools.evaluation.forms import Scenario


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
