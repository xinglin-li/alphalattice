"""Spans recorded through the OpenTelemetry SDK land in the Host's log as step events."""

from __future__ import annotations

import http.cookiejar
import json
import urllib.request
from pathlib import Path

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.trace import Status, StatusCode

from alphalattice.control.task_control.child import run_in_child
from alphalattice.interface.local_application.cli_contract import agent_provenance_headers
from alphalattice.interface.local_application.web import (
    SESSION_HEADER,
    LocalWebApplication,
    LocalWebService,
)
from alphalattice.kernel.data_layer import create_data_root, history
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
    tracer = tracer_provider(LogSpanExporter(log)).get_tracer("research")
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


TRACE, CLIENT_SPAN = "4bf92f3577b34da6a3ce929d0e0e4736", "00f067aa0ba902b7"


def read_bytes(path: str, cancelled) -> int:
    """A worker's data read: the size of the file at `path`."""
    return len(Path(path).read_bytes())


@pytest.mark.proves("data_layer.a_clients_trace_reaches_its_workers")
def test_a_cli_trace_runs_through_a_host_step_into_its_worker_as_one_tree(tmp_path):
    """A request whose environment names a trace is one Host step in that trace, and the worker
    call it makes is the step's child, attributed to the Host's workspace and the baggage's
    trial; both are read back from the log, and a read the workbench polls opens no span."""
    (tmp_path / "bars").write_bytes(b"x" * 300)
    web = LocalWebApplication()
    web.add_asset("/workbench.html", b"page", "text/html")  # where the launch sends a browser

    @web.route("POST", "/api/read", mutates=True)
    def read(_query, _payload):
        return {"bytes": run_in_child(f"{__name__}:read_bytes", {"path": str(tmp_path / "bars")})}

    @web.route("GET", "/api/poll")
    def poll(_query, _payload):
        return {}

    environ = {  # the client claims workspace w2; the Host serving w1 binds its own
        "TRACEPARENT": f"00-{TRACE}-{CLIENT_SPAN}-01",
        "BAGGAGE": "alphalattice.trial_id=t7,alphalattice.workspace_id=w2",
    }
    assert history.open_history("w1", "one", tmp_path / "data") is None
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}), urllib.request.HTTPCookieProcessor(jar)
    )
    try:
        with LocalWebService(web) as service:
            base = service.url.rstrip("/")
            opener.open(service.launch_url, timeout=10).read()
            opener.open(f"{base}/api/poll", timeout=10).read()
            headers = {SESSION_HEADER: web.session_token, "Origin": base}
            headers |= agent_provenance_headers(environ) | {"Content-Type": "application/json"}
            sent = urllib.request.Request(f"{base}/api/read", b"{}", headers, method="POST")
            answer = json.loads(opener.open(sent, timeout=120).read())
    finally:
        history.close_history()
    with HostLog.open(tmp_path / "data") as log:
        rows = log.read(
            "SELECT type, trace_id, span_id, parent_span_id, workspace_id, attrs FROM event"
            " WHERE phase IS NOT NULL ORDER BY started_us"
        )
    assert answer == {"bytes": 300}
    (step, call) = [dict(row) for row in rows]
    assert (step["type"], step["trace_id"], step["parent_span_id"]) == (
        "host.request",
        TRACE,
        CLIENT_SPAN,
    )
    assert (call["type"], call["trace_id"], call["parent_span_id"]) == (
        "worker.call",
        TRACE,
        step["span_id"],
    )
    assert step["workspace_id"] == call["workspace_id"] == "w1"
    assert json.loads(call["attrs"]) == {"call": f"{__name__}:read_bytes", "trial_id": "t7"}
    assert json.loads(step["attrs"])["route"] == "POST /api/read"
