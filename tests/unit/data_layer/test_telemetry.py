"""Native agent OTLP is numerical observation, never retained conversation content."""

from __future__ import annotations

import gzip
import json

import pytest
from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import ExportLogsServiceRequest
from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import ExportMetricsServiceRequest
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

from alphalattice.kernel.data_layer import telemetry

TEST_SOURCE = "src/alphalattice/kernel/data_layer/telemetry.py"


@pytest.mark.proves("data_layer.agent_otlp_drops_content_before_storage")
@pytest.mark.parametrize("name_slot", ("known", "attribute", "event", "span", "metric"))
def test_an_otlp_batch_keeps_attribution_and_counts_but_never_content(name_slot):
    """Protobuf and JSON take the same allowlist before any writer sees the records."""
    request = ExportLogsServiceRequest()
    resource = request.resource_logs.add()
    resource.resource.attributes.add(key="service.name").value.string_value = "claude-code"
    record = resource.scope_logs.add().log_records.add(time_unix_nano=123456789)
    record.body.string_value = "PRIVATE_BODY"
    values = {
        "event.name": "claude_code.api_request",
        "session.id": "session-1",
        "agent_id": "agent-1",
        "alphalattice.goal_id": "goal-1",
        "alphalattice.workspace_id": "FORGED-WORKSPACE",
        "alphalattice.trial_id": "trial-1",
        "model": "claude-haiku-5-5",
        "input_tokens": 10,
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
        "output_tokens": 3,
        "duration_ms": 250,
        "prompt": "PRIVATE_PROMPT",
        "tool_parameters": "PRIVATE_TOOL",
        "gen_ai.input.messages": "PRIVATE_OPT_IN",
        "unknown": "PRIVATE_UNKNOWN",
    }
    for key, value in values.items():
        attribute = record.attributes.add(key=key)
        if isinstance(value, int):
            attribute.value.int_value = value
        else:
            attribute.value.string_value = value
    from google.protobuf.json_format import MessageToJson

    if name_slot != "known":
        canary = "PRIVATE_SINGLE_TOKEN"
        if name_slot in {"attribute", "event"}:
            record.attributes[0].value.string_value = canary if name_slot == "attribute" else ""
            if name_slot == "event":
                record.event_name = canary
            signal, request = "logs", request
        elif name_slot == "span":
            request = ExportTraceServiceRequest()
            request.resource_spans.add().scope_spans.add().spans.add(name=canary)
            signal = "traces"
        else:
            request = ExportMetricsServiceRequest()
            metric = request.resource_metrics.add().scope_metrics.add().metrics.add(name=canary)
            metric.gauge.data_points.add(as_int=1)
            signal = "metrics"
        rows = telemetry.decode(
            signal, request.SerializeToString(), "application/x-protobuf", workspace_id="w1"
        )
        assert rows[0]["name"] == "unplaced"
        assert canary not in json.dumps(rows)
        return

    encoded = MessageToJson(request).encode()
    binary = telemetry.decode(
        "logs", request.SerializeToString(), "application/x-protobuf", workspace_id="w1"
    )
    assert binary == telemetry.decode("logs", encoded, "application/json", workspace_id="w1")
    assert binary == telemetry.decode(
        "logs", gzip.compress(encoded), "application/json", workspace_id="w1", encoding="gzip"
    )
    for body, kind, options in (
        (b"not protobuf", "application/x-protobuf", {}),
        (encoded, "text/plain", {}),
        (gzip.compress(encoded), "application/json", {"encoding": "gzip", "maximum": 10}),
    ):
        with pytest.raises(ValueError, match=r"agent_telemetry\.payload_invalid"):
            telemetry.decode("logs", body, kind, workspace_id="w1", **options)
    (row,) = binary
    other = telemetry.decode("logs", encoded, "application/json", workspace_id="w2")
    assert other[0]["event_id"] != row["event_id"]
    assert row["session_id"] == "session-1"
    assert row["conversation_id"] is None
    assert row["agent_id"] == "agent-1"
    assert row["goal_id"] == "goal-1"
    assert row["workspace_id"] == "w1"
    assert row["trial_id"] == "trial-1"
    assert row["parent_agent_id"] is None  # absence never invents lineage
    assert row["input_tokens"] == 10 and row["output_tokens"] == 3
    assert row["model_seconds"] == 0.25
    assert "PRIVATE_" not in json.dumps(binary)
    assert row["model"] == "claude-haiku-5-5"
    observed = telemetry.usage(binary)
    assert observed["models"][0]["input_tokens"] == 10
    assert observed["model_seconds"] == 0.25 and observed["lineage"] == []
    assert observed["incomplete"] == 0
    # Counter snapshots and repeated trace evidence cannot double an API request.
    assert (
        telemetry.usage([*binary, row | {"signal": "metrics"}, row | {"signal": "traces"}])
        == observed
    )
    # OTLP/JSON context IDs are hex, unlike protobuf's generic JSON encoding.
    from google.protobuf.json_format import MessageToDict

    trace = ExportTraceServiceRequest()
    resource_spans = trace.resource_spans.add()
    resource_spans.resource.CopyFrom(resource.resource)
    span = resource_spans.scope_spans.add().spans.add(
        name="claude_code.api_request",
        trace_id=b"t" * 16,
        span_id=b"s" * 8,
        parent_span_id=b"p" * 8,
        start_time_unix_nano=100,
        end_time_unix_nano=200,
    )
    span.attributes.extend(record.attributes)
    span.attributes.add(key="parent_agent_id").value.string_value = "parent-agent"
    span.events.add(name="PRIVATE_EVENT")
    payload = MessageToDict(trace)
    json_span = payload["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    for key, value in (("traceId", b"t" * 16), ("spanId", b"s" * 8), ("parentSpanId", b"p" * 8)):
        json_span[key] = value.hex()
    traced = telemetry.decode(
        "traces", trace.SerializeToString(), "application/x-protobuf", workspace_id="w1"
    )
    assert traced == telemetry.decode(
        "traces", json.dumps(payload).encode(), "application/json", workspace_id="w1"
    )
    assert traced[0]["trace_id"] == (b"t" * 16).hex()
    assert traced[0]["parent_span_id"] == (b"p" * 8).hex()
    assert telemetry.usage(traced)["lineage"] == [("parent-agent", "agent-1")]
    assert "PRIVATE_" not in json.dumps(traced)
    metrics = ExportMetricsServiceRequest()
    metric = (
        metrics.resource_metrics.add()
        .scope_metrics.add()
        .metrics.add(name="claude_code.token.usage")
    )
    point = metric.sum.data_points.add(time_unix_nano=123, as_int=10)
    point.attributes.extend(record.attributes)
    counted = telemetry.decode(
        "metrics", metrics.SerializeToString(), "application/x-protobuf", workspace_id="w1"
    )
    assert len(counted) == 1 and counted[0]["metric_value"] == 10
    assert telemetry.usage(counted)["models"] == []
    assert telemetry.usage(counted)["model_seconds"] is None
