"""Spans into the log, through the OpenTelemetry SDK and one exporter of ours.

The exporter writes each ended span as a step event in the Host's batch.

A span's name is its event type (a registered work type such as `tool.call`); its status gives the
phase, its times the step's start and end, its context the trace, span and parent. The attributes
under `alphalattice.` give the event's actor, node, workspace and subject; the rest are its attrs.
The log exporter stays local; a separately configured exporter may receive scrubbed spans.

A trace crosses processes as W3C trace context: `carrier` writes the current context as the two
headers `traceparent` and `baggage`, and `context_of` reads them back. Baggage names the workspace,
trial, selection and goal a step works for and the caller it acts for (`BAGGAGE_KEYS`);
OpenTelemetry's baggage processors stamp them on every span and log record under it, so a worker's
span is attributed with no argument of ours. A Host worker records its
spans into a `RowsExporter` and hands the rows back with its answer, for the Host to append.

The SDK is imported here and nowhere on the CLI's start path; only a caller that records spans
imports this module.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final, cast

from opentelemetry import baggage
from opentelemetry.baggage.propagation import W3CBaggagePropagator
from opentelemetry.context import Context
from opentelemetry.processor.baggage import BaggageLogProcessor, BaggageSpanProcessor
from opentelemetry.sdk._logs import LoggerProvider, ReadableLogRecord
from opentelemetry.sdk._logs.export import (
    LogRecordExporter,
    LogRecordExportResult,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Event, ReadableSpan, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.sdk.util.instrumentation import InstrumentationScope
from opentelemetry.trace import (
    Link,
    NonRecordingSpan,
    SpanContext,
    Status,
    StatusCode,
    TraceFlags,
    set_span_in_context,
)
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from alphalattice.capabilities.data.errors import DataError
from alphalattice.capabilities.domain_contracts import telemetry_names as telemetry
from alphalattice.capabilities.domain_contracts.environment import offline
from alphalattice.foundation.data_layer.admit import NUMERIC, secrets_in
from alphalattice.foundation.data_layer.log import HostLog, envelope

if TYPE_CHECKING:
    from opentelemetry.sdk.metrics import MeterProvider

PREFIX: Final = telemetry.ALPHALATTICE_ATTRIBUTE_PREFIX
_BAGGAGE_ATTRIBUTES: Final = (
    telemetry.ALPHALATTICE_WORKSPACE_ID,
    telemetry.ALPHALATTICE_TRIAL_ID,
    telemetry.ALPHALATTICE_SELECTION_ID,
    telemetry.ALPHALATTICE_GOAL_ID,
    telemetry.ALPHALATTICE_ACTOR_KIND,
    telemetry.ALPHALATTICE_ACTOR_ID,
)
BAGGAGE_KEYS: Final = tuple(key.removeprefix(PREFIX) for key in _BAGGAGE_ATTRIBUTES)
"""What a step works for and who it acts for, carried as baggage under PREFIX into every span and
log record below it."""
_STAMPED: Final = frozenset(_BAGGAGE_ATTRIBUTES)
_ENVELOPE: Final = frozenset(
    key.removeprefix(PREFIX)
    for key in (
        telemetry.ALPHALATTICE_SUBJECT_KIND,
        telemetry.ALPHALATTICE_SUBJECT_ID,
        telemetry.ALPHALATTICE_WORKSPACE_ID,
        telemetry.ALPHALATTICE_ACTOR_KIND,
        telemetry.ALPHALATTICE_ACTOR_ID,
        telemetry.ALPHALATTICE_NODE,
    )
)
"""The PREFIX attributes the step event's envelope takes; any other stays an attribute."""
_PROPAGATORS: Final = (TraceContextTextMapPropagator(), W3CBaggagePropagator())
_LOG = logging.getLogger(__name__)
_CONTENT = re.compile(
    r"(?i)(?:^|[._-])(prompt|completion|content|messages?|result|response|input|output|body|text|stacktrace|system_instructions)(?:$|[._-])"
)


def _envelope(attributes: Mapping[str, Any] | None, scope: Any) -> dict[str, Any]:
    """The event fields a span's or a log record's attributes give: those under PREFIX name its
    actor, node, workspace and subject, and every other attribute (a sequence as a list) is kept."""
    attrs = {key: list(v) if isinstance(v, tuple) else v for key, v in (attributes or {}).items()}
    own = {key[len(PREFIX) :]: str(attrs.pop(key)) for key in list(attrs) if key.startswith(PREFIX)}
    attrs |= {key: value for key, value in own.items() if key not in _ENVELOPE}
    return {
        "subject": (own.get("subject_kind"), own.get("subject_id")),
        "attrs": attrs,
        "workspace": own.get("workspace_id"),
        "actor": (own.get("actor_kind", "host"), own.get("actor_id", "host")),
        "node": own.get("node", scope.name if scope is not None else "host"),
    }


def step_event(
    span: ReadableSpan, *, started: bool = False, text: str | None = None
) -> dict[str, Any]:
    """The step event an ended span is recorded as; with `started`, the event a live span writes
    as it starts, so a reader sees the step before it ends."""
    context, parent = span.get_span_context(), span.parent
    assert context is not None and span.start_time is not None
    assert started or span.end_time is not None
    fields = _envelope(span.attributes, span.instrumentation_scope)
    row: dict[str, Any] = envelope(
        span.name,
        fields.pop("subject"),
        fields.pop("attrs"),
        text=text,
        phase=(
            "started"
            if started
            else "failed"
            if span.status.status_code is StatusCode.ERROR
            else "completed"
        ),
        trace=f"{context.trace_id:032x}",
        span=f"{context.span_id:016x}",
        parent_span=None if parent is None else f"{parent.span_id:016x}",
        started_us=None if started else span.start_time // 1000,
        occurred_us=span.start_time // 1000 if started else cast(int, span.end_time) // 1000,
        **fields,
    )
    return row


def fact_event(record: ReadableLogRecord) -> tuple[dict[str, Any], str | None]:
    """The fact a log record is recorded as, and its text: its event name is the event's type, a
    text body the event's text, its trace and span the span it happened in (a new trace outside
    one)."""
    log = record.log_record
    fields = _envelope(log.attributes, record.instrumentation_scope)
    text = log.body if isinstance(log.body, str) else None
    row: dict[str, Any] = envelope(
        cast(str, log.event_name),
        fields.pop("subject"),
        fields.pop("attrs"),
        text=text,
        trace=f"{log.trace_id:032x}" if log.trace_id else None,
        span=f"{log.span_id:016x}" if log.span_id else None,
        occurred_us=None if log.timestamp is None else log.timestamp // 1000,
        **fields,
    )
    return row, text


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
        ok = all([self.append(row) for row in rows])
        return SpanExportResult.SUCCESS if ok else SpanExportResult.FAILURE

    def append(
        self, row: Mapping[str, Any], text: str | None = None, *, raising: bool = False
    ) -> bool:
        """Append one event; one the log refuses is counted by its code and logged, and raised
        only to a writer that asks (`raising`): the Host's own rows, whose writer must know."""
        try:
            self._log.append(row, text)
        except (DataError, sqlite3.Error) as error:
            code = error.failure.code if isinstance(error, DataError) else type(error).__name__
            self.refused[code] += 1
            _LOG.warning("event %s not recorded: %s (%s)", row.get("type"), code, error)
            if raising:
                raise
            return False
        return True

    def shutdown(self) -> None:
        """Commit what the spans wrote."""
        self._log.flush()


class LogFactExporter(LogRecordExporter):
    """Appends each log record to the Host's log as a fact, through the span exporter's append, so
    a refused record is counted beside a refused span."""

    def __init__(self, spans: LogSpanExporter) -> None:
        """Write through `spans`, the Host log's exporter."""
        self._spans = spans
        self._here = threading.local()

    def export(self, batch: Sequence[ReadableLogRecord]) -> LogRecordExportResult:
        """Append each record as its fact; FAILURE when the log refused any, whose refusal the
        emitting thread reads back (`refusal`)."""
        self._here.refused = self._here.kept = None
        for record in batch:
            row, text = fact_event(record)
            try:
                self._spans.append(row, text, raising=True)
            except (DataError, sqlite3.Error) as error:
                self._here.refused = error
            else:
                self._here.kept = row["event_id"]
        failed = self._here.refused is not None
        return LogRecordExportResult.FAILURE if failed else LogRecordExportResult.SUCCESS

    def refusal(self) -> DataError | sqlite3.Error | None:
        """The log's refusal of the last record this thread emitted, once; None when it was kept.
        A record is exported as it is emitted, on the emitting thread."""
        refused, self._here.refused = getattr(self._here, "refused", None), None
        return cast("DataError | sqlite3.Error | None", refused)

    def kept(self) -> str | None:
        """The event id of the last record this thread emitted and the log kept, once."""
        kept, self._here.kept = getattr(self._here, "kept", None), None
        return cast("str | None", kept)

    def shutdown(self) -> None:
        """Nothing of its own: the span exporter commits the log."""

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Nothing waits: each record is appended as it is emitted."""
        return True


class RowsExporter(SpanExporter):
    """Keeps each ended span as its step event, for a worker to hand back with its answer."""

    def __init__(self) -> None:
        """Start with no rows."""
        self.rows: list[dict[str, Any]] = []

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Keep each span's step event."""
        self.rows.extend(step_event(span) for span in spans)
        return SpanExportResult.SUCCESS


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

    def __init__(
        self, exporter: SpanExporter, *, include_content: bool = False, loopback: bool = False
    ) -> None:
        """Send scrubbed spans to the supplied exporter, omitting content unless selected; a
        `loopback` receiver is on this machine, no egress, so it receives them offline too."""
        self._batch = BatchSpanProcessor(exporter)
        self._include_content = include_content
        self._loopback = loopback

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
        if offline() and not self._loopback:
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


class LatencyHistograms(SpanProcessor):
    """Each ended Host request's seconds into the semantic conventions' HTTP server duration, with
    their advised bounds, recorded in the request span's own context, so the meter's trace-based
    exemplar filter links a slow point to its trace and span."""

    def __init__(self, meters: MeterProvider) -> None:
        """Record into `meters`, which this processor shuts down with its tracer provider."""
        from opentelemetry.instrumentation._semconv import (  # lazy: a configured meter's
            HTTP_DURATION_HISTOGRAM_BUCKETS_NEW,
        )
        from opentelemetry.semconv.attributes.http_attributes import (  # lazy: a configured meter's
            HTTP_REQUEST_METHOD,
            HTTP_ROUTE,
        )
        from opentelemetry.semconv.metrics.http_metrics import (  # lazy: a configured meter's
            HTTP_SERVER_REQUEST_DURATION,
        )

        self._meters, self._keys = meters, (HTTP_REQUEST_METHOD, HTTP_ROUTE)
        self._requests = meters.get_meter(__name__).create_histogram(
            HTTP_SERVER_REQUEST_DURATION,
            unit="s",
            description="Duration of HTTP server requests.",
            explicit_bucket_boundaries_advisory=HTTP_DURATION_HISTOGRAM_BUCKETS_NEW,
        )

    def on_end(self, span: ReadableSpan) -> None:
        """Record a Host request's duration, its exemplar the span itself."""
        if span.name != telemetry.SPAN_HOST_REQUEST or span.context is None:
            return
        if span.start_time is None or span.end_time is None:
            return
        method, _, route = str((span.attributes or {}).get(telemetry.ROUTE, "")).partition(" ")
        self._requests.record(
            (span.end_time - span.start_time) / 1e9,
            dict(zip(self._keys, (method, route), strict=True)),
            context=set_span_in_context(NonRecordingSpan(span.context)),
        )

    def shutdown(self) -> None:
        """Send the last points and stop the meters' reader."""
        self._meters.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Send the recorded points within the supplied bound."""
        return self._meters.force_flush(timeout_millis)


def tracer_provider(
    exporter: SpanExporter,
    *,
    remote: ExternalSpanProcessor | None = None,
    meters: MeterProvider | None = None,
) -> TracerProvider:
    """Record locally and, only when supplied, send scrubbed spans to a remote exporter and each
    Host request's seconds to `meters`."""
    provider = TracerProvider()
    provider.add_span_processor(BaggageSpanProcessor(_STAMPED.__contains__))
    provider.add_span_processor(
        SimpleSpanProcessor(exporter) if remote is None else RecordedSpanProcessor(exporter, remote)
    )
    if meters is not None:
        provider.add_span_processor(LatencyHistograms(meters))
    return provider


def logger_provider(exporter: LogRecordExporter) -> LoggerProvider:
    """A logger provider whose one exporter is `exporter`: each record a fact in the Host's log as
    it is emitted. Nothing leaves the machine."""
    provider = LoggerProvider()
    provider.add_log_record_processor(BaggageLogProcessor(_STAMPED.__contains__))
    provider.add_log_record_processor(SimpleLogRecordProcessor(exporter))
    return provider


def carrier(context: Context | None = None) -> dict[str, str]:
    """The W3C headers of `context` (the current one by default): traceparent and baggage."""
    headers: dict[str, str] = {}
    for propagator in _PROPAGATORS:
        propagator.inject(headers, context=context)
    return headers


def context_of(headers: Mapping[str, str], **works_for: str | None) -> Context:
    """The context W3C headers carry, with `works_for` (any of BAGGAGE_KEYS) bound over what their
    baggage names: the caller's own identity, never a client's claim; a key bound to None is
    removed, so a client cannot claim it either."""
    context: Context | None = None
    for propagator in _PROPAGATORS:
        context = propagator.extract(headers, context=context)
    assert context is not None
    for key, value in works_for.items():
        if value is None:
            context = baggage.remove_baggage(PREFIX + key, context)
        else:
            context = baggage.set_baggage(PREFIX + key, value, context)
    return context
