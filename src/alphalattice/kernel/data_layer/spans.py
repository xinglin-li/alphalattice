"""Spans into the log, through the OpenTelemetry SDK and one exporter of ours.

The exporter writes each ended span as a step event in the Host's batch.

A span's name is its event type (a registered work type such as `tool.call`); its status gives the
phase, its times the step's start and end, its context the trace, span and parent. The attributes
under `alphalattice.` give the event's actor, node, workspace and subject; the rest are its attrs.
The log exporter stays local; a separately configured exporter may receive scrubbed spans.

A trace crosses processes as W3C trace context: `carrier` writes the current context as the two
headers `traceparent` and `baggage`, and `context_of` reads them back. Baggage names the workspace,
trial and selection a step works for (`BAGGAGE_KEYS`); every span started under it carries them as
attributes, so a worker's span is attributed with no argument of ours. A Host worker records its
spans into a `RowsExporter` and hands the rows back with its answer, for the Host to append.

The SDK is imported here and nowhere on the CLI's start path; only a caller that records spans
imports this module.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Final

from opentelemetry import baggage
from opentelemetry.baggage.propagation import W3CBaggagePropagator
from opentelemetry.context import Context
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Event, ReadableSpan, Span, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.sdk.util.instrumentation import InstrumentationScope
from opentelemetry.trace import Link, SpanContext, Status, StatusCode, TraceFlags
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer.admit import NUMERIC, secrets_in
from alphalattice.kernel.data_layer.log import HostLog, envelope
from alphalattice.kernel.shared_kernel.environment import offline

PREFIX: Final = "alphalattice."
BAGGAGE_KEYS: Final = ("workspace_id", "trial_id", "selection_id")
"""What a step works for, carried as baggage under PREFIX into every span below it."""
_ENVELOPE: Final = frozenset(
    {"subject_kind", "subject_id", "workspace_id", "actor_kind", "actor_id", "node"}
)
"""The PREFIX attributes the step event's envelope takes; any other stays an attribute."""
_PROPAGATORS: Final = (TraceContextTextMapPropagator(), W3CBaggagePropagator())
_LOG = logging.getLogger(__name__)
_CONTENT = re.compile(
    r"(?i)(?:^|[._-])(prompt|completion|content|messages?|result|response|input|output|body|text|stacktrace|system_instructions)(?:$|[._-])"
)


def step_event(span: ReadableSpan) -> dict[str, Any]:
    """The step event an ended span is recorded as."""
    attrs = dict(span.attributes or {})
    own = {key[len(PREFIX) :]: str(attrs.pop(key)) for key in list(attrs) if key.startswith(PREFIX)}
    attrs |= {key: value for key, value in own.items() if key not in _ENVELOPE}
    context, parent = span.get_span_context(), span.parent
    assert context is not None and span.start_time is not None and span.end_time is not None
    return envelope(
        span.name,
        (own.get("subject_kind"), own.get("subject_id")),
        attrs,
        phase="failed" if span.status.status_code is StatusCode.ERROR else "completed",
        workspace=own.get("workspace_id"),
        actor=(own.get("actor_kind", "host"), own.get("actor_id", "host")),
        node=own.get(
            "node", span.instrumentation_scope.name if span.instrumentation_scope else "host"
        ),
        trace=f"{context.trace_id:032x}",
        span=f"{context.span_id:016x}",
        parent_span=None if parent is None else f"{parent.span_id:016x}",
        started_us=span.start_time // 1000,
        occurred_us=span.end_time // 1000,
    )


class LogSpanExporter(SpanExporter):
    """Appends each ended span to the Host's log.

    A span the log refuses is counted by its code (`refused`) and logged, never raised into the
    instrumented code.
    """

    def __init__(self, log: HostLog) -> None:
        """Write into `log`, the Host's one write path."""
        self._log = log
        self.refused: Counter[str] = Counter()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Append each span as its step event; FAILURE when the log refused any."""
        return self.write(step_event(span) for span in spans)

    def write(self, rows: Iterable[Mapping[str, Any]]) -> SpanExportResult:
        """Append step events another process's spans became, as `export` appends its own."""
        result = SpanExportResult.SUCCESS
        for row in rows:
            try:
                self._log.append(row)
            except (DataError, sqlite3.Error) as error:
                code = error.failure.code if isinstance(error, DataError) else type(error).__name__
                self.refused[code] += 1
                _LOG.warning("span %s not recorded: %s (%s)", row.get("type"), code, error)
                result = SpanExportResult.FAILURE
        return result

    def shutdown(self) -> None:
        """Commit what the spans wrote."""
        self._log.flush()


class RowsExporter(SpanExporter):
    """Keeps each ended span as its step event, for a worker to hand back with its answer."""

    def __init__(self) -> None:
        """Start with no rows."""
        self.rows: list[dict[str, Any]] = []

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Keep each span's step event."""
        self.rows.extend(step_event(span) for span in spans)
        return SpanExportResult.SUCCESS


class BaggageAttributes(SpanProcessor):
    """Gives a starting span the workspace, trial and selection its context's baggage names."""

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        """Copy each of BAGGAGE_KEYS the baggage holds into the span's attributes."""
        for key in BAGGAGE_KEYS:
            value = baggage.get_baggage(PREFIX + key, parent_context)
            if value is not None:
                span.set_attribute(PREFIX + key, str(value))


def _scrub(
    value: Any, name: str = "", *, include_content: bool = False, for_log: bool = False
) -> Any:
    """Apply the log admission's credential scan to values sent outside the machine."""
    if isinstance(value, Mapping):
        return {
            str(_scrub(str(key), for_log=for_log)): _scrub(
                item, str(key), include_content=include_content, for_log=for_log
            )
            for key, item in value.items()
            if include_content or str(key) in NUMERIC or not _CONTENT.search(str(key))
        }
    if isinstance(value, (list, tuple)):
        return tuple(
            _scrub(item, name, include_content=include_content, for_log=for_log) for item in value
        )
    if isinstance(value, bytes) or secrets_in({name: value}):
        return "" if for_log and secrets_in({name: "[redacted]"}) else "[redacted]"
    return value


def _without_trace_state(context: SpanContext) -> SpanContext:
    """Keep trace identity and flags; vendor trace-state values are not admitted content."""
    return SpanContext(context.trace_id, context.span_id, context.is_remote, context.trace_flags)


class ExternalSpanProcessor(SpanProcessor):
    """Scan each remote span before the SDK's bounded batch exporter receives it."""

    def __init__(self, exporter: SpanExporter, *, include_content: bool = False) -> None:
        """Send scrubbed spans to the supplied exporter, omitting content unless selected."""
        self._batch = BatchSpanProcessor(exporter)
        self._include_content = include_content

    def scrub_span(self, span: ReadableSpan, *, for_log: bool = False) -> ReadableSpan:
        """Return an admitted copy without changing the original span."""
        include_content = for_log or self._include_content
        clean = lambda value, name="": _scrub(
            value, name, include_content=include_content, for_log=for_log
        )
        resource = span.resource
        scope = span.instrumentation_scope
        return ReadableSpan(
            name=clean(span.name),
            context=_without_trace_state(span.context) if span.context else None,
            parent=_without_trace_state(span.parent) if span.parent else None,
            resource=Resource(clean(resource.attributes), schema_url=clean(resource.schema_url))
            if resource
            else None,
            attributes=clean(span.attributes or {}),
            events=[
                Event(clean(event.name), clean(event.attributes or {}), event.timestamp)
                for event in span.events
                if include_content or not _CONTENT.search(event.name)
            ],
            links=[
                Link(_without_trace_state(link.context), clean(link.attributes or {}))
                for link in span.links
            ],
            kind=span.kind,
            instrumentation_scope=InstrumentationScope(
                clean(scope.name),
                clean(scope.version),
                clean(scope.schema_url),
                clean(scope.attributes or {}),
            )
            if scope
            else None,
            status=Status(
                span.status.status_code,
                clean(span.status.description) if include_content else None,
            ),
            start_time=span.start_time,
            end_time=span.end_time,
        )

    def on_end(self, span: ReadableSpan) -> None:
        """Scan an ended span before queuing it for external export."""
        if offline():
            return
        try:
            self._batch.on_end(self.scrub_span(span))
        except Exception as error:
            _LOG.warning("external span was not exported: %s", type(error).__name__)

    def shutdown(self) -> None:
        """Flush and stop the SDK's batch processor."""
        self._batch.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Ask the SDK to send its queued spans within the supplied bound."""
        return self._batch.force_flush(timeout_millis)

    def log_row(self, row: Mapping[str, Any]) -> dict[str, Any]:
        """Remove secrets before a returned worker event reaches the unchanged log exporter."""
        attrs = _scrub(json.loads(row["attrs"]), include_content=True, for_log=True)
        return dict(row) | {
            "attrs": json.dumps(attrs, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        }

    def worker_rows(self, rows: Iterable[Mapping[str, Any]]) -> None:
        """Export worker spans returned to the Host as their recorded event rows."""
        for row in rows:
            try:
                trace = int(row["trace_id"], 16)
                context = SpanContext(trace, int(row["span_id"], 16), False, TraceFlags(1))
                parent_id = row.get("parent_span_id")
                parent = (
                    SpanContext(trace, int(parent_id, 16), False, TraceFlags(1))
                    if parent_id
                    else None
                )
                attributes = json.loads(row["attrs"])
                for field in _ENVELOPE:
                    if row.get(field) is not None:
                        attributes[PREFIX + field] = row[field]
                self.on_end(
                    ReadableSpan(
                        str(row["type"]),
                        context=context,
                        parent=parent,
                        attributes=attributes,
                        instrumentation_scope=InstrumentationScope(str(row["node"])),
                        status=Status(
                            StatusCode.ERROR if row["phase"] == "failed" else StatusCode.UNSET
                        ),
                        start_time=int(row["started_us"]) * 1000,
                        end_time=int(row["occurred_us"]) * 1000,
                    )
                )
            except (KeyError, TypeError, ValueError) as error:
                _LOG.warning("worker span was not exported: %s", type(error).__name__)


class RecordedSpanProcessor(SpanProcessor):
    """Export remotely only after the same span is admitted by the local record."""

    def __init__(self, exporter: SpanExporter, remote: ExternalSpanProcessor) -> None:
        """Keep the existing local exporter and the opted-in external processor."""
        self._exporter, self._remote = exporter, remote

    def on_end(self, span: ReadableSpan) -> None:
        """Record an admitted copy and send the span only when the log accepts it."""
        if not span.context or not span.context.trace_flags.sampled:
            return
        try:
            clean = self._remote.scrub_span(span, for_log=True)
            if self._exporter.export((clean,)) is SpanExportResult.SUCCESS:
                self._remote.on_end(span)
        except Exception as error:
            _LOG.warning("span was not recorded or exported: %s", type(error).__name__)

    def shutdown(self) -> None:
        """Drain remote spans before closing the local exporter."""
        self._remote.shutdown()
        self._exporter.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Send queued remote spans within the supplied bound."""
        return self._remote.force_flush(timeout_millis)


def tracer_provider(
    exporter: SpanExporter, *, remote: ExternalSpanProcessor | None = None
) -> TracerProvider:
    """Record locally and, only when supplied, send scrubbed spans to a remote exporter."""
    provider = TracerProvider()
    provider.add_span_processor(BaggageAttributes())
    provider.add_span_processor(
        SimpleSpanProcessor(exporter) if remote is None else RecordedSpanProcessor(exporter, remote)
    )
    return provider


def carrier(context: Context | None = None) -> dict[str, str]:
    """The W3C headers of `context` (the current one by default): traceparent and baggage."""
    headers: dict[str, str] = {}
    for propagator in _PROPAGATORS:
        propagator.inject(headers, context=context)
    return headers


def context_of(headers: Mapping[str, str], **works_for: str | None) -> Context:
    """The context W3C headers carry, with `works_for` (any of BAGGAGE_KEYS) bound over what their
    baggage names: the caller's own identity, never a client's claim."""
    context: Context | None = None
    for propagator in _PROPAGATORS:
        context = propagator.extract(headers, context=context)
    assert context is not None
    for key, value in works_for.items():
        if value is not None:
            context = baggage.set_baggage(PREFIX + key, value, context)
    return context
