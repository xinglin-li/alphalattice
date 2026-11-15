"""The W3C context a step's spans carry across a process: its trace, and what it works for."""

from __future__ import annotations

import json

import pytest
from opentelemetry import context as current

from alphalattice.kernel.data_layer.spans import RowsExporter, carrier, context_of, tracer_provider

TEST_SOURCE = "src/alphalattice/kernel/data_layer/spans.py"
TRACE, CLIENT_SPAN = "4bf92f3577b34da6a3ce929d0e0e4736", "00f067aa0ba902b7"


@pytest.mark.proves("data_layer.baggage_attributes_every_span")
def test_a_span_below_w3c_headers_is_their_child_and_works_for_their_baggage():
    """A span started in the context W3C headers carry is the client span's child and works for
    the trial the baggage names and the workspace its caller binds, whatever workspace the
    baggage claims; the span's own context writes back out as headers in the same trace."""
    exporter = RowsExporter()
    tracer = tracer_provider(exporter).get_tracer("worker")
    headers = {
        "traceparent": f"00-{TRACE}-{CLIENT_SPAN}-01",
        "baggage": "alphalattice.trial_id=t7,alphalattice.workspace_id=w2",
    }
    token = current.attach(context_of(headers, workspace_id="w1", selection_id=None))
    try:
        with tracer.start_as_current_span("worker.call", attributes={"call": "m:f"}):
            sent = carrier()
    finally:
        current.detach(token)
    (row,) = exporter.rows
    assert (row["trace_id"], row["parent_span_id"]) == (TRACE, CLIENT_SPAN)
    assert row["workspace_id"] == "w1"
    assert json.loads(row["attrs"]) == {"call": "m:f", "trial_id": "t7"}
    assert sent["traceparent"].split("-")[1:3] == [TRACE, row["span_id"]]
    assert {"alphalattice.trial_id=t7", "alphalattice.workspace_id=w1"} <= set(
        sent["baggage"].split(",")
    )
