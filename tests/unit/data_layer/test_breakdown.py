"""A goal's time from its traces: its waterfall, the agent's inferred time, the untraced share."""

from __future__ import annotations

import json
from typing import Any

import pytest

from alphalattice.kernel.data_layer.breakdown import goal_trace
from tests._support.history_fixtures import native_first_use

TEST_SOURCE = "src/alphalattice/kernel/data_layer/breakdown.py"
S = 1_000_000  # microseconds in a second


def _row(
    kind: str, span: str, phase: str, start: int, end: int | None = None, **more: Any
) -> dict[str, Any]:
    """One log row: a started row at `start`, or an ended row from `start` to `end`."""
    parent, trace, actor = more.pop("parent", None), more.pop("trace", span), more.pop("actor", "a")
    ended = phase != "started"
    return {
        "type": kind,
        "phase": phase,
        "span_id": span,
        "trace_id": trace,
        "parent_span_id": parent,
        "actor_id": actor,
        "occurred_us": end if ended else start,
        "started_us": start if ended else None,
        "attrs": json.dumps(more),
    }


@pytest.mark.proves("data_layer.goal_trace_breaks_down_a_goals_time")
def test_two_agents_calls_read_apart_with_their_gaps_stages_and_untraced_share() -> None:
    """Each session's calls read apart, with the gap between two of its calls as inferred agent
    time; a call whose steps leave a planted gap is flagged, one naming no session is counted, a
    live step counts no seconds, and each stage sums its steps."""
    s1, s2 = {"gen_ai.conversation.id": "s1"}, {"gen_ai.conversation.id": "s2"}
    calls: list[tuple[Any, ...]] = [  # (span, phase, start, end, attributes); its own trace
        ("c1", "started", 0, None, s1 | {"command": "goal open"}),
        ("c1", "completed", 0, 2 * S, s1 | {"command": "goal open", "exit_code": 0}),
        ("c2", "completed", 1 * S, 3 * S, s2 | {"command": "goal show"}),
        ("c3", "completed", 7 * S, 17 * S, s1 | {"command": "run"}),
        ("c4", "completed", 20 * S, 21 * S, {"unattributed": True, "command": "network show"}),
    ]
    steps: list[tuple[Any, ...]] = [  # (type, span, phase, start, end, parent, trace, attrs)
        ("host.request", "h1", "completed", 0, 2 * S, "c1", "c1", {"operation": "GOAL_OPEN"}),
        ("product.operation", "p1", "completed", S // 2, S, "h1", "c1", {"operation": "GOAL_OPEN"}),
        ("host.request", "h2", "completed", 1 * S, 3 * S, "c2", "c2", {"operation": "GOAL_SHOW"}),
        ("host.request", "h3", "completed", 7 * S, 9 * S, "c3", "c3", {"operation": "RUN"}),
        ("worker.call", "w3", "started", 8 * S, None, "h3", "c3", {"call": "m:f"}),
        ("host.request", "h4", "completed", 20 * S, 21 * S, "c4", "c4", {"operation": "NETWORK"}),
    ]
    called = [
        _row("cli.call", span, phase, lo, hi, **attrs) for span, phase, lo, hi, attrs in calls
    ]
    stepped = [
        _row(kind, span, phase, lo, hi, parent=parent, trace=trace, **attrs)
        for kind, span, phase, lo, hi, parent, trace, attrs in steps
    ]
    trace = goal_trace(called, stepped)
    assert [(c["name"], c["conversation"]) for c in trace["calls"]] == [
        ("goal open", "s1"),
        ("goal show", "s2"),
        ("run", "s1"),
        ("network show", None),
    ]
    assert [
        (gap["gen_ai.conversation.id"], gap["seconds"], gap["inferred"]) for gap in trace["agent"]
    ] == [("s1", 5.0, True)]
    run = trace["calls"][2]
    assert (run["untraced_seconds"], run.get("untraced_flagged")) == (8.0, True)
    assert [(s["name"], s["depth"], s["ended_us"]) for s in run["steps"]] == [
        ("RUN", 1, 9 * S),
        ("m:f", 2, None),
    ]
    breakdown = trace["breakdown"]
    assert (breakdown["agent_seconds"], breakdown["call_seconds"]) == (5.0, 15.0)
    assert (breakdown["untraced_flagged_calls"], breakdown["unattributed_calls"]) == (1, 1)
    assert (breakdown["untraced_seconds"], breakdown["untraced_share"]) == (8.0, round(8 / 15, 6))
    stages = {(s["type"], s["name"]): (s["seconds"], s["steps"]) for s in breakdown["stages"]}
    assert stages[("host.request", "GOAL_OPEN")] == (2.0, 1)
    assert stages[("product.operation", "GOAL_OPEN")] == (0.5, 1)
    assert ("worker.call", "m:f") not in stages  # live: no seconds yet
    assert trace["window"] == {"started_us": 0, "ended_us": 21 * S}


@pytest.mark.proves("data_layer.native_events_split_an_agents_gap")
def test_a_conversations_native_records_split_its_gap_and_leave_the_remainder_unattributed() -> (
    None
):
    """A conversation's gap splits by its records into waits, tools and model, then the rest."""
    s1, s2 = {"gen_ai.conversation.id": "s1"}, {"gen_ai.conversation.id": "s2"}
    calls = [
        _row("cli.call", "c1", "completed", 0, 2 * S, **s1),
        _row("cli.call", "c2", "completed", 0, 1 * S, **s2),
        _row("cli.call", "c3", "completed", 7 * S, 9 * S, **s1),
        _row("cli.call", "c4", "completed", 4 * S, 5 * S, **s2),
    ]
    trace = goal_trace(calls, [], native=native_first_use("s1", 2 * S, 7 * S))
    other, split = trace["agent"]  # in the order their next calls start
    assert (split["gen_ai.conversation.id"], split["seconds"], split["inferred"]) == (
        "s1",
        5.0,
        False,
    )
    seconds = {
        kind: split[f"{kind}_seconds"] for kind in ("person", "tool", "model", "unattributed")
    }
    assert seconds == pytest.approx({"person": 0.9, "tool": 0.7, "model": 2.3, "unattributed": 1.1})
    assert split["unattributed"] is True
    assert [s["gen_ai.agent.id"] for s in split["spans"] if "gen_ai.agent.id" in s] == ["a1"]
    assert all(2 * S <= s["started_us"] < s["ended_us"] <= 7 * S for s in split["spans"])
    assert (other["inferred"], other["unattributed_seconds"]) == (True, 3.0)
    breakdown = trace["breakdown"]
    assert breakdown["agent_split"] == "NATIVE_EVENTS"
    assert (breakdown["agent_seconds"], breakdown["unattributed_seconds"]) == pytest.approx(
        (8.0, 4.1)
    )
    assert (breakdown["model_seconds"], breakdown["person_seconds"]) == pytest.approx((2.3, 0.9))
    assert goal_trace(calls, [])["breakdown"]["agent_split"] == "INFERRED_FROM_CALL_TIMES"
