"""Native OTLP decoding: only attribution, numerical usage and explicit lineage survive.

No body, span event, status message, resource payload or unknown attribute is returned to
the writer. The protocol package is already in the product's locked exporter closure.
"""

from __future__ import annotations

import base64
import json
import math
import re
import zlib
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, Final

from alphalattice.kernel.data_layer.chain import canonical, digest
from alphalattice.kernel.shared_kernel import telemetry_names as telemetry

FIELDS: Final = {
    "conversation_id": ("conversation.id", telemetry.GEN_AI_CONVERSATION_ID),
    "session_id": ("session.id",),
    "agent_id": ("agent_id", "agent.id", telemetry.GEN_AI_AGENT_ID),
    "agent_role": ("agent_role", "agent.role", "agent_type"),
    "parent_agent_id": ("parent_agent_id", "parent.agent.id"),
    "goal_id": (telemetry.ALPHALATTICE_GOAL_ID,),
    "trial_id": (telemetry.ALPHALATTICE_TRIAL_ID,),
    "model": ("gen_ai.response.model", telemetry.GEN_AI_REQUEST_MODEL, "model"),
    "effort": ("gen_ai.request.reasoning_effort", "reasoning_effort"),
    "response_id": ("gen_ai.response.id", "request_id", "request.id"),
    "input_tokens": (telemetry.GEN_AI_USAGE_INPUT_TOKENS, "input_tokens"),
    "cache_read_tokens": ("gen_ai.usage.cache_read.input_tokens", "cache_read_tokens"),
    "cache_write_tokens": ("gen_ai.usage.cache_write.input_tokens", "cache_creation_tokens"),
    "output_tokens": (telemetry.GEN_AI_USAGE_OUTPUT_TOKENS, "output_tokens"),
}
"""The entire attribute allowlist; opt-in attributes have no entry."""
COUNTS: Final = frozenset(
    {"input_tokens", "cache_read_tokens", "cache_write_tokens", "output_tokens"}
)
_IDENTIFIER = re.compile(r"[\w./:@+\-]{1,256}", re.ASCII)
_EVENT_NAMES: Final = frozenset(
    {
        "codex.api_request",
        "codex.sse_event",
        "codex.user_prompt",
        "codex.tool_decision",
        "codex.tool_result",
        "codex.agent_communication",
        "claude_code.api_request",
        "claude_code.tool_result",
        "claude_code.subagent_completed",
    }
)
"""The literal native event identifiers admitted from every OTLP name slot."""
_METRIC_NAMES: Final = frozenset({"claude_code.token.usage"})
_IDS: Final = {"traceId": 16, "spanId": 8, "parentSpanId": 8}


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and _IDENTIFIER.fullmatch(value) else None


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _attributes(rows: list[dict[str, Any]]) -> dict[str, Any]:
    # Inspect only admitted keys; an arbitrary JSON/object value is never copied.
    admitted = {key for keys in FIELDS.values() for key in keys} | {
        "event.name",
        "service.name",
        "duration_ms",
    }
    return {
        row["key"]: next(iter(row.get("value", {}).values()), None)
        for row in rows
        if row.get("key") in admitted
    }


def decode(
    signal: str,
    body: bytes,
    content_type: str,
    *,
    workspace_id: str,
    encoding: str = "identity",
    maximum: int = 4 * 1024 * 1024,
) -> list[dict[str, Any]]:
    """Decode one bounded HTTP batch into the only columns the catalog may receive.

    Protobuf and OTLP/JSON share the same generated messages, including validation. Bad
    payloads raise ValueError without putting payload bytes in the exception or log.
    """
    from google.protobuf.json_format import MessageToDict, ParseDict  # lazy: received OTLP only
    from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import (  # lazy: OTLP only
        ExportLogsServiceRequest,
    )
    from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import (  # lazy: OTLP only
        ExportMetricsServiceRequest,
    )
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (  # lazy: OTLP only
        ExportTraceServiceRequest,
    )

    kinds = {
        "logs": (ExportLogsServiceRequest, "resourceLogs", "scopeLogs", "logRecords"),
        "traces": (ExportTraceServiceRequest, "resourceSpans", "scopeSpans", "spans"),
        "metrics": (ExportMetricsServiceRequest, "resourceMetrics", "scopeMetrics", "metrics"),
    }
    if signal not in kinds:
        raise ValueError("agent_telemetry.signal_invalid")
    kind, resources, scopes, records = kinds[signal]
    message = kind()
    try:
        if encoding == "gzip":
            decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
            body = decompressor.decompress(body, maximum + 1)
            if not decompressor.eof or decompressor.unused_data:
                raise ValueError("compressed_body")
        elif encoding != "identity":
            raise ValueError("encoding")
        if len(body) > maximum:
            raise ValueError("body_bound")
        if content_type == "application/json":
            document = json.loads(body)
            # OTLP/JSON uses hex for context IDs; protobuf's generic JSON mapper uses
            # base64. Translate only the protocol's exact context slots before ParseDict.
            for resource in document.get(resources, []):
                for scope in resource.get(scopes, []):
                    for record in scope.get(records, []):
                        for key, size in _IDS.items():
                            if record.get(key):
                                value = record[key]
                                if not isinstance(value, str) or not re.fullmatch(
                                    f"[a-fA-F0-9]{{{size * 2}}}", value
                                ):
                                    raise ValueError("context_id")
                                record[key] = base64.b64encode(bytes.fromhex(value)).decode("ascii")
            ParseDict(document, message)
        elif content_type == "application/x-protobuf":
            message.ParseFromString(body)
        else:
            raise ValueError("content_type")
        document = MessageToDict(message)
    except Exception:
        raise ValueError("agent_telemetry.payload_invalid") from None
    rows = []
    for resource in document.get(resources, []):
        attrs = _attributes(resource.get("resource", {}).get("attributes", []))
        for scope in resource.get(scopes, []):
            for record in scope.get(records, []):
                points = [record]
                if signal == "metrics":
                    data = next(
                        (
                            record[kind]
                            for kind in (
                                "sum",
                                "gauge",
                                "histogram",
                                "exponentialHistogram",
                                "summary",
                            )
                            if kind in record
                        ),
                        {},
                    )
                    points = data.get("dataPoints", [])
                for point in points:
                    row = _row(signal, record, point, attrs)
                    row["workspace_id"] = workspace_id
                    # Identical exports have one identity. Ordinal preserves repeated records
                    # within a batch; raw discarded content never participates in identity.
                    row["event_id"] = digest(canonical([len(rows), row]))
                    rows.append(row)
    return rows


def _row(signal: str, record: Mapping[str, Any], point: Mapping[str, Any], resource: dict) -> dict:
    attrs = resource | _attributes(point.get("attributes", []))
    service = _text(attrs.get("service.name"))
    names = (
        (record.get("name"),)
        if signal == "metrics"
        else (
            attrs.get("event.name"),
            record.get("eventName"),
            record.get("name"),
            record.get("body", {}).get("stringValue"),
        )
    )
    admitted = _METRIC_NAMES if signal == "metrics" else _EVENT_NAMES
    name = "unplaced"
    for candidate in names:
        if not isinstance(candidate, str):
            continue
        if service in {"claude-code", "claude_code"} and not candidate.startswith("claude_code."):
            candidate = "claude_code." + candidate
        if candidate in admitted:
            name = candidate
            break
    vendor = (
        "claude-code"
        if service in {"claude-code", "claude_code"} or name.startswith("claude_code.")
        else "codex"
        if service == "codex" or name.startswith("codex.") or "conversation.id" in attrs
        else None
    )
    row: dict[str, Any] = {"vendor": vendor, "signal": signal, "name": name}
    for field, keys in FIELDS.items():
        value = next((attrs[key] for key in keys if key in attrs), None)
        number = _number(value) if field in COUNTS else None
        if (
            field in COUNTS
            and value is not None
            and (number is None or not number.is_integer() or number >= 2**63)
        ):
            raise ValueError("agent_telemetry.payload_invalid")
        row[field] = (
            int(number)
            if number is not None and number.is_integer() and number < 2**63
            else (None if field in COUNTS else _text(value))
        )
    for field, key in (
        ("trace_id", "traceId"),
        ("span_id", "spanId"),
        ("parent_span_id", "parentSpanId"),
    ):
        value = point.get(key)
        raw = base64.b64decode(value, validate=True) if value else b""
        if raw and len(raw) != _IDS[key]:
            raise ValueError("agent_telemetry.payload_invalid")
        row[field] = raw.hex() if raw and any(raw) else None
    row["time_ns"] = int(point.get("timeUnixNano", point.get("startTimeUnixNano", 0)))
    if not 0 <= row["time_ns"] < 2**63:
        raise ValueError("agent_telemetry.payload_invalid")
    duration = _number(attrs.get("duration_ms"))
    start, end = point.get("startTimeUnixNano"), point.get("endTimeUnixNano")
    seconds = (
        duration / 1000
        if duration is not None
        else (max(0, int(end) - int(start)) / 1e9 if start and end else None)
    )
    inference = signal != "metrics" and (
        name.endswith(".api_request")
        or (signal == "traces" and any(key.startswith("gen_ai.usage.") for key in attrs))
    )
    row["model_seconds"] = seconds if inference else None
    row["metric_value"] = _number(point.get("asInt", point.get("asDouble")))
    return row


def usage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-response usage, without adding cumulative metrics to their log/trace events.

    Prefer numerical log records; traces supply counts only when there are no such logs.
    Where the vendor supplies a response id, later observations replace that response's
    counts. A record without counts is never a zero-token response. Missing attribution
    and explicit parent links remain as observed, never inferred from export batch order.
    """
    counted = [
        row
        for row in rows
        if any(row.get(field) is not None for field in COUNTS) and row["signal"] != "metrics"
    ]

    def source(row):
        return (row["vendor"], row["conversation_id"], row["session_id"], row["agent_id"])

    with_logs = {source(row) for row in counted if row["signal"] == "logs"}
    counted = [row for row in counted if row["signal"] == "logs" or source(row) not in with_logs]
    responses: dict[tuple, dict[str, Any]] = {}
    timed = [row for row in rows if row.get("model_seconds") is not None]
    timed_logs = {source(row) for row in timed if row["signal"] == "logs"}
    timed = [row for row in timed if row["signal"] == "logs" or source(row) not in timed_logs]
    times = {(*source(row), row.get("response_id") or row["event_id"]): row for row in timed}
    incomplete = 0
    for row in counted:
        if (
            row.get("model") is None
            or any(row.get(field) is None for field in COUNTS)
            or ("uncached_input_tokens" in row and row["uncached_input_tokens"] is None)
        ):
            incomplete += 1
            continue
        responses[*source(row), row.get("response_id") or row["event_id"]] = row
    models = []
    for model in sorted({row["model"] for row in responses.values()}):
        own = [row for row in responses.values() if row["model"] == model]
        stamps = sorted(
            datetime.fromtimestamp(row["time_ns"] / 1e9, UTC).isoformat()
            for row in own
            if row["time_ns"]
        )
        inputs = [row.get("uncached_input_tokens", row["input_tokens"]) for row in own]
        models.append(
            {
                "model": model,
                "efforts": tuple(sorted({row["effort"] for row in own if row["effort"]})),
                "responses": len(own),
                "input_tokens": sum(inputs),
                "cache_read_tokens": sum(row["cache_read_tokens"] or 0 for row in own),
                "cache_write_tokens": sum(row["cache_write_tokens"] or 0 for row in own),
                "output_tokens": sum(row["output_tokens"] for row in own),
                "first_at": stamps[0] if stamps else None,
                "last_at": stamps[-1] if stamps else None,
            }
        )
    return {
        "models": models,
        "incomplete": incomplete,
        "model_seconds": sum(row["model_seconds"] for row in times.values()) if times else None,
        "lineage": sorted(
            {
                (row["parent_agent_id"], row["agent_id"])
                for row in rows
                if row["parent_agent_id"] and row["agent_id"]
            }
        ),
    }
