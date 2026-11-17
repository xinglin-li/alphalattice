"""A goal's time, read from its traces: each CLI call's waterfall and where the time went.

A goal is many traces, one per CLI call naming it (`cli.call`). Under each call are the Host's
steps (`host.request`, `product.operation`, `worker.call` and any other step type), each a row
with its span, parent, start and end. The agent's own time is between its calls: the gap from one
call's end to the next call of the same conversation is one `invoke_agent` span, inferred from
the call times. Where the agent's own OpenTelemetry records of that conversation are at hand
(`native`, joined by the vendor's session id), the gap splits into model requests, the agent's
own tools outside the product and waits on the person; only the remainder stays one agent span,
marked unattributed. The view never depends on them: with none, the whole gap is inferred. A call
whose direct steps leave more than `UNTRACED_BOUND` of its wall untraced is flagged. A step not
yet ended is live: its end is None, and it counts no seconds yet.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Final

UNTRACED_BOUND: Final = 0.2
"""The share of a call's wall its direct steps may leave untraced before the call is flagged."""
SCHEMA: Final = "alphalattice.goal-trace.v1"
NATIVE: Final = {
    "claude_code.api_request": "model",
    "codex.api_request": "model",
    "claude_code.tool_result": "tool",
    "codex.tool_result": "tool",
    "claude_code.tool_decision": "person",
    "codex.tool_decision": "person",
    "claude_code.user_prompt": "person",
    "codex.user_prompt": "person",
}
"""Each agent's native event that places its time, by its class: a model request and a tool run
end at the event and last its `duration_ms`; a wait on the person ends at the prompt the person
sent or the permission they decided (a decision whose `source` is the user's), and runs from the
agent's previous record. Any other event places nothing."""
_CLASSES: Final = ("person", "tool", "model")
"""The order a moment two classes claim is counted in: the person's wait first."""


def _attrs(row: Mapping[str, Any]) -> dict[str, Any]:
    value = row.get("attrs") or "{}"
    return dict(json.loads(value) if isinstance(value, str) else value)


def _spans(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Each span once: its start (from a started row or an ended one) and, once ended, its end."""
    spans: dict[str, dict[str, Any]] = {}
    for row in rows:
        span = spans.setdefault(
            row["span_id"],
            {
                "type": row["type"],
                "span_id": row["span_id"],
                "trace_id": row["trace_id"],
                "parent_span_id": row["parent_span_id"],
                "actor_id": row["actor_id"],
                "started_us": row["occurred_us"],
                "ended_us": None,
                "attrs": {},
            },
        )
        span["attrs"] |= _attrs(row)
        if row["phase"] in ("completed", "failed", "cancelled"):
            span |= {"started_us": row["started_us"], "ended_us": row["occurred_us"]}
            span["failed"] = row["phase"] == "failed"
    return spans


def _covered(intervals: Iterable[tuple[int, int]], lo: int, hi: int) -> int:
    """Microseconds of [lo, hi] the intervals cover, overlaps counted once."""
    end, total = lo, 0
    for start, stop in sorted(intervals):
        start, stop = max(lo, start), min(hi, stop)
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


def _name(span: Mapping[str, Any]) -> str:
    attrs = span["attrs"]
    return str(attrs.get("operation") or attrs.get("call") or attrs.get("command") or span["type"])


def goal_trace(
    calls: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    *,
    native: Iterable[Mapping[str, Any]] = (),
    untraced_bound: float = UNTRACED_BOUND,
) -> dict[str, Any]:
    """The goal's waterfall and breakdown from its call rows, its calls' step rows and its
    conversations' native records (`conversation`, `name`, `at_us`, `duration_ms`, `source`,
    `agent_id`), if any."""
    called = sorted(_spans(calls).values(), key=lambda span: span["started_us"])
    stepped = _spans(steps)
    children: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for span in sorted(stepped.values(), key=lambda span: span["started_us"]):
        children[span["parent_span_id"]].append(span)
    stages: dict[tuple[str, str], dict[str, Any]] = {}
    waterfall, flagged, untraced_us = [], [], 0

    def tree(parent: str, depth: int) -> list[dict[str, Any]]:
        out = []
        for child in children.get(parent, ()):
            row = {key: child[key] for key in ("type", "started_us", "ended_us")}
            out.append(row | {"name": _name(child), "depth": depth})
            if child["ended_us"] is not None:
                stage = stages.setdefault(
                    (child["type"], _name(child)),
                    {"type": child["type"], "name": _name(child), "seconds": 0.0, "steps": 0},
                )
                stage["seconds"] += (child["ended_us"] - child["started_us"]) / 1e6
                stage["steps"] += 1
            out.extend(tree(child["span_id"], depth + 1))
        return out

    for call in called:
        attrs, lo, hi = call["attrs"], call["started_us"], call["ended_us"]
        row = {
            "type": "cli.call",
            "span_id": call["span_id"],
            "trace_id": call["trace_id"],
            "name": _name(call),
            "conversation": attrs.get("gen_ai.conversation.id"),
            "unattributed": bool(attrs.get("unattributed")),
            "started_us": lo,
            "ended_us": hi,
            "exit_code": attrs.get("exit_code"),
            "steps": tree(call["span_id"], 1),
        }
        if hi is not None and hi > lo:
            direct = [
                (child["started_us"], child["ended_us"] or hi)
                for child in children.get(call["span_id"], ())
            ]
            gap = hi - lo - _covered(direct, lo, hi)
            untraced_us += gap
            row["untraced_seconds"] = gap / 1e6
            if gap > untraced_bound * (hi - lo):
                flagged.append(call["span_id"])
                row["untraced_flagged"] = True
        waterfall.append(row)
    agents = _agent_gaps(called, native)
    ended = [call["ended_us"] for call in called if call["ended_us"] is not None]
    return {
        "schema": SCHEMA,
        "window": {
            "started_us": called[0]["started_us"] if called else None,
            "ended_us": max(ended) if ended else None,
        },
        "calls": waterfall,
        "agent": agents,
        "breakdown": _breakdown(called, agents, waterfall, stages, untraced_us, untraced_bound)
        | {"untraced_flagged_calls": len(flagged)},
    }


def _breakdown(
    called: Sequence[Mapping[str, Any]],
    agents: Sequence[Mapping[str, Any]],
    waterfall: Sequence[Mapping[str, Any]],
    stages: Mapping[tuple[str, str], Mapping[str, Any]],
    untraced_us: int,
    untraced_bound: float,
) -> dict[str, Any]:
    """Where the goal's time went: the agent's split, the calls, each stage, the untraced share."""
    call_us = sum(call["ended_us"] - call["started_us"] for call in called if call["ended_us"])
    return {
        "agent_seconds": round(sum(gap["seconds"] for gap in agents), 6),
        "agent_split": (
            "NATIVE_EVENTS"
            if any(not gap["inferred"] for gap in agents)
            else "INFERRED_FROM_CALL_TIMES"
        ),
        **{
            f"{name}_seconds": round(sum(gap.get(f"{name}_seconds", 0) for gap in agents), 6)
            for name in (*_CLASSES, "unattributed")
        },
        "call_seconds": round(call_us / 1e6, 6),
        "stages": sorted(
            ({**stage, "seconds": round(stage["seconds"], 6)} for stage in stages.values()),
            key=lambda stage: -stage["seconds"],
        ),
        "untraced_seconds": round(untraced_us / 1e6, 6),
        "untraced_share": round(untraced_us / call_us, 6) if call_us else None,
        "untraced_bound": untraced_bound,
        "unattributed_calls": sum(1 for row in waterfall if row["unattributed"]),
    }


def _agent_gaps(
    called: Sequence[Mapping[str, Any]], native: Iterable[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Each conversation's time between consecutive calls: one `invoke_agent` span, split by the
    conversation's native records where it has them."""
    records: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in native:
        records[str(record["conversation"])].append(record)
    last: dict[str, Mapping[str, Any]] = {}
    gaps = []
    for call in called:
        conversation = call["attrs"].get("gen_ai.conversation.id")
        if conversation is None:
            continue
        before = last.get(conversation)
        ended = None if before is None else before["ended_us"]
        if ended is not None and call["started_us"] > ended:
            gap = {
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.conversation.id": conversation,
                "inferred": True,
                "started_us": ended,
                "ended_us": call["started_us"],
                "seconds": (call["started_us"] - ended) / 1e6,
            }
            gaps.append(gap | _split(records.get(conversation, ()), ended, call["started_us"]))
        last[conversation] = call
    return gaps


def _split(records: Iterable[Mapping[str, Any]], lo: int, hi: int) -> dict[str, Any]:
    """The gap [lo, hi] as its native records place it: each record's span in its class, the
    seconds of each class (a moment counted once, the person's wait first) and the remainder,
    unattributed: the whole gap when no record falls in it."""
    spans, previous = [], lo
    for record in sorted(records, key=lambda record: int(record["at_us"])):
        at, name = int(record["at_us"]), str(record["name"])
        kind = NATIVE.get(name)
        if not lo < at <= hi or kind is None:
            continue
        source = str(record.get("source") or "")
        if name.endswith("tool_decision") and not source.startswith("user"):
            kind = None  # decided by configuration or a hook: no wait on the person
        start = previous if kind == "person" else at - int(1000 * (record.get("duration_ms") or 0))
        if kind is not None and start < at:
            spans.append(
                {
                    "class": kind,
                    "name": name,
                    "started_us": max(lo, start),
                    "ended_us": at,
                    **({"gen_ai.agent.id": record["agent_id"]} if record.get("agent_id") else {}),
                }
            )
        previous = at
    if not spans:
        return {"unattributed_seconds": (hi - lo) / 1e6, "unattributed": True}
    split: dict[str, Any] = {"inferred": False, "spans": spans}
    claimed: list[tuple[int, int]] = []
    for kind in _CLASSES:
        mine = [(s["started_us"], s["ended_us"]) for s in spans if s["class"] == kind]
        alone = _covered(mine + claimed, lo, hi) - _covered(claimed, lo, hi)
        split[f"{kind}_seconds"] = alone / 1e6
        claimed += mine
    split["unattributed_seconds"] = (hi - lo - _covered(claimed, lo, hi)) / 1e6
    split["unattributed"] = split["unattributed_seconds"] > 0
    return split
