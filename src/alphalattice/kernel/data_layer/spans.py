"""Spans into the log, through the OpenTelemetry SDK and one exporter of ours.

The exporter writes each ended span as a step event in the Host's batch.

A span's name is its event type (a registered work type such as `tool.call`); its status gives the
phase, its times the step's start and end, its context the trace, span and parent. The attributes
under `alphalattice.` give the event's actor, node, workspace and subject; the rest are its attrs.
No exporter sends anything off the machine: `tracer_provider` installs this one alone.

The SDK is imported here and nowhere on the CLI's start path; only a caller that records spans
imports this module.
"""

from __future__ import annotations

import logging
import sqlite3
from collections import Counter
from collections.abc import Sequence
from typing import Any, Final

from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.trace import StatusCode

from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer.log import HostLog, envelope

PREFIX: Final = "alphalattice."
_LOG = logging.getLogger(__name__)


def step_event(span: ReadableSpan) -> dict[str, Any]:
    """The step event an ended span is recorded as."""
    attrs = dict(span.attributes or {})
    own = {key[len(PREFIX) :]: str(attrs.pop(key)) for key in list(attrs) if key.startswith(PREFIX)}
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
        result = SpanExportResult.SUCCESS
        for span in spans:
            try:
                self._log.append(step_event(span))
            except (DataError, sqlite3.Error) as error:
                code = error.failure.code if isinstance(error, DataError) else type(error).__name__
                self.refused[code] += 1
                _LOG.warning("span %s not recorded: %s (%s)", span.name, code, error)
                result = SpanExportResult.FAILURE
        return result

    def shutdown(self) -> None:
        """Commit what the spans wrote."""
        self._log.flush()


def tracer_provider(log: HostLog) -> TracerProvider:
    """A tracer provider whose one exporter writes into `log`; nothing leaves the machine."""
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(LogSpanExporter(log)))
    return provider
