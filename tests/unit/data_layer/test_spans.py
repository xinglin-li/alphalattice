"""The W3C context a step's spans carry across a process: its trace, and what it works for."""

from __future__ import annotations

import json
from typing import cast

import pytest
from opentelemetry import context as current
from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Event, ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.util.instrumentation import InstrumentationScope
from opentelemetry.trace import SpanContext, Status, StatusCode, TraceFlags, TraceState

from alphalattice.kernel.data_layer import spans
from alphalattice.kernel.data_layer.spans import (
    RowsExporter,
    carrier,
    context_of,
    fact_event,
    logger_provider,
    step_event,
    tracer_provider,
)

TEST_SOURCE = "src/alphalattice/kernel/data_layer/spans.py"
TRACE, CLIENT_SPAN = "4bf92f3577b34da6a3ce929d0e0e4736", "00f067aa0ba902b7"


@pytest.mark.proves("data_layer.configured_export_scrubs_secrets")
@pytest.mark.parametrize("include_content", [False, True])
def test_the_external_exporter_scrubs_secrets_and_respects_content_choice(
    include_content, monkeypatch
):
    """The local record stays independent while remote spans contain no credentials."""
    monkeypatch.delenv("ALPHALATTICE_NETWORK_DISABLED", raising=False)
    local, remote = RowsExporter(), InMemorySpanExporter()
    processor_type = getattr(spans, "ExternalSpanProcessor", None)
    assert processor_type is not None, "external span export needs its secret-scanning processor"
    processor = processor_type(remote, include_content=include_content)
    provider = tracer_provider(local, remote=processor)
    secret = "sk-" + "A" * 24
    with provider.get_tracer("host").start_as_current_span(
        "worker.call",
        attributes={
            "call": "read",
            "note": secret,
            "prompt": "private prompt",
            "gen_ai.input.messages": "private messages",
            "gen_ai.retrieval.query.text": secret,
            "gen_ai.usage.input_tokens": 7,
            secret: "safe value",
        },
    ) as span:
        span.add_event("worker.detail", {"credential": "sensitive", "summary": "safe"})
    assert provider.force_flush()
    (sent,) = remote.get_finished_spans()
    assert sent.attributes["note"] == "[redacted]"
    assert sent.attributes["[redacted]"] == "safe value"
    assert ("prompt" in sent.attributes) is include_content
    assert ("gen_ai.input.messages" in sent.attributes) is include_content
    assert ("gen_ai.retrieval.query.text" in sent.attributes) is include_content
    assert sent.attributes["gen_ai.usage.input_tokens"] == 7
    if include_content:
        assert sent.attributes["prompt"] == "private prompt"
        assert sent.attributes["gen_ai.retrieval.query.text"] == "[redacted]"
    assert sent.events[0].attributes["credential"] == "[redacted]"
    assert json.loads(local.rows[0]["attrs"])["note"] == "[redacted]"
    assert json.loads(local.rows[0]["attrs"])["prompt"] == "private prompt"
    raw = dict(local.rows[0]) | {
        "attrs": json.dumps({"call": "read", "note": secret, "token": secret})
    }
    local_worker = json.loads(processor.log_row(raw)["attrs"])
    assert local_worker == {"call": "read", "note": "[redacted]", "token": ""}
    processor.worker_rows(local.rows)
    assert provider.force_flush()
    returned = remote.get_finished_spans()[1]
    assert returned.context.trace_id == sent.context.trace_id
    assert returned.context.span_id == sent.context.span_id
    assert returned.attributes["note"] == "[redacted]"
    context = SpanContext(
        sent.context.trace_id,
        sent.context.span_id,
        False,
        TraceFlags(1),
        TraceState([("vendor", secret)]),
    )
    processor.on_end(
        ReadableSpan(
            "worker.call",
            context=context,
            resource=Resource({"service.name": secret}),
            instrumentation_scope=InstrumentationScope(secret),
            events=[Event("worker.detail", {"password": "sensitive"})],
            status=Status(StatusCode.ERROR, secret),
            start_time=sent.start_time,
            end_time=sent.end_time,
        )
    )
    assert provider.force_flush()
    protected = remote.get_finished_spans()[2]
    assert protected.resource.attributes["service.name"] == "[redacted]"
    assert protected.instrumentation_scope.name == "[redacted]"
    assert not protected.context.trace_state
    assert protected.events[0].attributes["password"] == "[redacted]"
    assert protected.status.description == ("[redacted]" if include_content else None)
    provider.shutdown()


@pytest.mark.proves("data_layer.baggage_attributes_every_span")
def test_a_span_below_w3c_headers_is_their_child_and_works_for_their_baggage():
    """A span under W3C headers is the client span's child, works for their baggage and the bound
    workspace, writes its context back out, and is logged live as started; a log record is a fact
    in the span current where it is emitted (its text the body), or in no span outside one."""
    exporter, kept = RowsExporter(), InMemoryLogRecordExporter()
    tracer = tracer_provider(exporter).get_tracer("worker")
    logger = logger_provider(kept).get_logger("host")
    headers = {
        "traceparent": f"00-{TRACE}-{CLIENT_SPAN}-01",
        "baggage": "alphalattice.trial_id=t7,alphalattice.workspace_id=w2",
    }
    own = {"alphalattice.actor_kind": "agent", "alphalattice.actor_id": "a1"}
    token = current.attach(context_of(headers, workspace_id="w1", selection_id=None))
    try:
        with tracer.start_as_current_span("worker.call", attributes={"call": "m:f"}) as opened:
            sent = carrier()
            span = cast(ReadableSpan, opened)  # the SDK's span, read as the exporter reads it
            live = step_event(span, started=True)
            logger.emit(event_name="agent.activity", body="Read it.", attributes=own | {"k": 1})
    finally:
        current.detach(token)
    logger.emit(event_name="request.refused", attributes={"operation": "S"}, timestamp=5_000)
    (row,) = exporter.rows
    assert (row["trace_id"], row["parent_span_id"]) == (TRACE, CLIENT_SPAN)
    assert row["workspace_id"] == "w1"
    assert json.loads(row["attrs"]) == {"call": "m:f", "trial_id": "t7"}
    assert sent["traceparent"].split("-")[1:3] == [TRACE, row["span_id"]]
    assert {"alphalattice.trial_id=t7", "alphalattice.workspace_id=w1"} <= set(
        sent["baggage"].split(",")
    )
    started_at = cast(int, span.start_time) // 1000
    assert (live["phase"], live["span_id"], live["started_us"], live["occurred_us"]) == (
        "started",
        row["span_id"],
        None,
        started_at,
    )
    (inside, inside_text), (outside, outside_text) = map(fact_event, kept.get_finished_logs())
    assert (inside["phase"], inside["span_id"], inside["actor_id"], inside_text) == (
        None,
        row["span_id"],
        "a1",
        "Read it.",
    )
    assert json.loads(inside["attrs"]) == {"k": 1, "trial_id": "t7"}  # its baggage stamped
    assert (outside["span_id"], outside["occurred_us"], outside_text) == (None, 5, None)
