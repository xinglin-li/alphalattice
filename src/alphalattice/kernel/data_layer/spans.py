"""Spans into the log, through the OpenTelemetry SDK and one exporter of ours.

The exporter writes each ended span as a step event in the Host's batch.

A span's name is its event type (a registered work type such as `tool.call`); its status gives the
phase, its times the step's start and end, its context the trace, span and parent. The attributes
under `alphalattice.` give the event's actor, node, workspace and subject; the rest are its attrs.
No exporter sends anything off the machine: `tracer_provider` installs this one alone.

A trace crosses processes as W3C trace context: `carrier` writes the current context as the two
headers `traceparent` and `baggage`, and `context_of` reads them back. Baggage names the workspace,
trial and selection a step works for (`BAGGAGE_KEYS`); every span started under it carries them as
attributes, so a worker's span is attributed with no argument of ours. A Host worker records its
spans into a `RowsExporter` and hands the rows back with its answer, for the Host to append.

The SDK is imported here and nowhere on the CLI's start path; only a caller that records spans
imports this module.
"""

from __future__ import annotations

import logging
import sqlite3
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Final

from opentelemetry import baggage
from opentelemetry.baggage.propagation import W3CBaggagePropagator
from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.trace import StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer.log import HostLog, envelope

PREFIX: Final = "alphalattice."
BAGGAGE_KEYS: Final = ("workspace_id", "trial_id", "selection_id")
"""What a step works for, carried as baggage under PREFIX into every span below it."""
_ENVELOPE: Final = frozenset(
    {"subject_kind", "subject_id", "workspace_id", "actor_kind", "actor_id", "node"}
)
"""The PREFIX attributes the step event's envelope takes; any other stays an attribute."""
_PROPAGATORS: Final = (TraceContextTextMapPropagator(), W3CBaggagePropagator())
_LOG = logging.getLogger(__name__)


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


def tracer_provider(exporter: SpanExporter) -> TracerProvider:
    """A tracer provider whose one exporter is `exporter`: a log's, or the rows a worker hands
    back. Nothing leaves the machine."""
    provider = TracerProvider()
    provider.add_span_processor(BaggageAttributes())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
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
