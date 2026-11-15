"""Spans recorded through the OpenTelemetry SDK land in the Host's log as step events."""

from __future__ import annotations

import json

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.trace import Status, StatusCode

from alphalattice.kernel.data_layer import create_data_root
from alphalattice.kernel.data_layer.log import HostLog
from alphalattice.kernel.data_layer.spans import LogSpanExporter, tracer_provider

TEST_SOURCE = "src/alphalattice/kernel/data_layer/spans.py"
T0 = 1_794_700_000_000_000
AGENT = {"alphalattice.actor_kind": "agent", "alphalattice.actor_id": "claude"}


@pytest.fixture
def log(tmp_path):
    """An open Host log on a new data root."""
    create_data_root(tmp_path / "data")
    with HostLog.open(tmp_path / "data", now_us=T0) as opened:
        yield opened


def _steps(log):
    """The step events the spans wrote, in order."""
    sql = "SELECT * FROM event WHERE phase IS NOT NULL ORDER BY seq"
    return log.read(sql)


@pytest.mark.proves("data_layer.spans_are_step_events")
def test_each_ended_span_is_its_step_event(log):
    """A span is appended at its end as the event its name registers: its trace, span, parent,
    start, end and phase, its actor and node, and its other attributes as the event's."""
    tracer = tracer_provider(log).get_tracer("research")
    with (
        tracer.start_as_current_span("agent.turn", attributes=AGENT) as turn,
        tracer.start_as_current_span("tool.call", attributes=AGENT | {"tool": "grep"}) as call,
    ):
        call.set_status(Status(StatusCode.ERROR))
    inner, outer = _steps(log)
    assert (inner["type"], inner["phase"], outer["type"], outer["phase"]) == (
        "tool.call",
        "failed",
        "agent.turn",
        "completed",
    )
    assert inner["trace_id"] == outer["trace_id"] == f"{turn.get_span_context().trace_id:032x}"
    assert inner["parent_span_id"] == outer["span_id"] == f"{turn.get_span_context().span_id:016x}"
    assert (inner["actor_kind"], inner["actor_id"], inner["node"]) == (
        "agent",
        "claude",
        "research",
    )
    assert json.loads(inner["attrs"]) == {"tool": "grep"}
    assert (
        outer["started_us"] <= inner["started_us"] <= inner["occurred_us"] <= outer["occurred_us"]
    )


@pytest.mark.proves("data_layer.spans_are_step_events")
def test_a_span_the_log_refuses_is_counted_never_raised(log):
    """A span whose name is no registered type, or that lacks a required attribute, is counted by
    its refusal code; the instrumented code goes on and nothing is written."""
    exporter = LogSpanExporter(log)
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("research")
    for name in ("made.up", "tool.call"):
        with tracer.start_as_current_span(name):
            pass
    assert dict(exporter.refused) == {
        "data_layer.event_type_unregistered": 1,
        "data_layer.event_attrs_missing": 1,
    }
    assert _steps(log) == []
