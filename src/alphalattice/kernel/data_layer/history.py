"""The research history the Host records: its data root opened at start, and the loop's events.

The Host opens the machine's data root when it starts (`open_history`): `ALPHALATTICE_DATA_ROOT`,
else the application data folder (`%LOCALAPPDATA%/AlphaLattice` on Windows,
`$XDG_DATA_HOME/alphalattice` elsewhere). It holds the log's one writer and the reading session
for its life. The history is the Host's optional infrastructure, as its activity store is: a Host
that finds another holding the data root (HISTORY_DEFERRED), or whose log will not open (a killed
Host's files not yet released, a damaged log), records nothing, logs why and serves as before, since
a research path never waits on the history.

The research paths call `record` beside what they write today. Outside a Host, or while the history
is deferred, `record` writes nothing and returns None: only the Host writes the log. Each event is a
small row in the step's own transaction; no artifact is kept, copied or hashed for its sake.

The open history also holds the Host's tracer (`spans.tracer_provider` over its log). `step` opens
one span for one step the Host serves, in the trace its client's W3C headers name, and `traced`
appends the spans a worker handed back with its answer. Both do nothing while the history is
deferred.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final

from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer.log import HostLog
from alphalattice.kernel.data_layer.reading import reading_session
from alphalattice.kernel.data_layer.root import create_data_root

DATA_ROOT_ENV: Final = "ALPHALATTICE_DATA_ROOT"
HISTORY_DEFERRED: Final = "history: deferred (another Host holds the data root)"
HISTORY_UNAVAILABLE: Final = "history: deferred (the data root's log or views did not open)"
EVIDENCE_TYPE: Final = "research_evidence"
"""The artifact type of a trial's execution evidence, registered by the Host as its own."""
_EVIDENCE_TEXT: Final = "What one research trial's Desk run did: its program, inputs and artifacts."
STEPS: Final = {
    "host.request": (["operation"], "One request a client sent the Host: its operation and route."),
    "worker.call": (["call"], "One call a Host worker ran for the step that sent it."),
}
"""The Host's own step types: each span `step` and a worker open is one of these events."""

_lock = threading.Lock()
_once = threading.Lock()
_open: dict[str, Any] = {}


def data_root() -> Path:
    """The machine's data root: `ALPHALATTICE_DATA_ROOT`, else the application data folder."""
    override = os.environ.get(DATA_ROOT_ENV, "").strip()
    if override:
        return Path(override)
    if os.name == "nt":
        local = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(local) / "AlphaLattice"
    data = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(data) / "alphalattice"


def open_history(workspace_id: str, name: str, root: Path | None = None) -> str | None:
    """Open the data root's log and reading views for this Host, and record its workspace.

    Returns None when the history records, else why it does not (HISTORY_DEFERRED or
    HISTORY_UNAVAILABLE, logged with the error). Opening an open history again records the
    workspace only.
    """
    with _lock:
        if "log" not in _open:
            where = data_root() if root is None else root
            try:
                create_data_root(where)
                log = HostLog.open(where)
            except (DataError, sqlite3.Error, OSError) as error:
                held = (
                    isinstance(error, DataError) and error.failure.code == "data_layer.host_running"
                )
                reason = HISTORY_DEFERRED if held else HISTORY_UNAVAILABLE
                logging.getLogger(__name__).warning("%s at %s: %s", reason, where, error)
                return reason
            try:
                reading = reading_session(where)
            except Exception as error:  # DuckDB, an optional extra, raises its own errors too
                log.close()  # the data root's lock goes with it, so the next start can open it
                reason = HISTORY_UNAVAILABLE
                logging.getLogger(__name__).warning("%s at %s: %s", reason, where, error)
                return reason
            from alphalattice.kernel.data_layer.spans import (  # lazy: the SDK, a Host's alone
                LogSpanExporter,
                tracer_provider,
            )

            exporter = LogSpanExporter(log)  # the SDK is imported by a Host alone
            _open.update(log=log, reading=reading, workspace=workspace_id, exporter=exporter)
            _open["tracer"] = tracer_provider(exporter)
            spec: dict[str, Any] = {
                "storage": "none",  # the evidence stays in the workspace's store until row 8a
                "table": None,
                "measures": [],
                "keep": "while_active",
            }
            define("artifact_type", EVIDENCE_TYPE, _EVIDENCE_TEXT, spec)
            for step_name, (required, text) in STEPS.items():
                kind = {"work": True, "subject": None, "required": required, "text": "none"}
                define("event_type", step_name, text, kind | {"durable": False})
        log = _open["log"]
    known = log.read("SELECT 1 FROM workspace WHERE workspace_id = ?", (workspace_id,))
    subject = ("workspace", workspace_id)
    if known:
        log.record("workspace.opened", subject, workspace=workspace_id)
    else:
        log.record("workspace.created", subject, {"name": name}, workspace=workspace_id)
    return None


def close_history() -> None:
    """Commit the open batch, close the reading session and release the data root."""
    with _lock:
        reading = _open.pop("reading", None)
        log = _open.pop("log", None)
        _open.pop("workspace", None)
        provider = _open.pop("tracer", None)
        _open.pop("exporter", None)
    if provider is not None:
        provider.shutdown()  # every span ended before the log closes
    if reading is not None:
        reading.close()
    if log is not None:
        log.close()


def record(
    type_: str,
    subject: tuple[str | None, str | None] = (None, None),
    attrs: Mapping[str, Any] | None = None,
    **more: Any,
) -> int | None:
    """Append one research event through the Host's log, in the Host's workspace unless `more`
    names another; None outside a Host or when deferred."""
    log = _open.get("log")
    if log is None:
        return None
    more.setdefault("workspace", _open["workspace"])
    return int(log.record(type_, subject, attrs, **more))


def recording() -> bool:
    """Whether this process is a Host whose history records."""
    return "log" in _open


def record_new(
    exists: str,
    args: tuple[Any, ...],
    type_: str,
    subject: tuple[str | None, str | None] = (None, None),
    attrs: Mapping[str, Any] | None = None,
    **more: Any,
) -> int | None:
    """`record`, unless the row `exists` selects is already in the log; the check and the append
    are one step for this process, the log's one writer. None when it exists or nothing records."""
    with _once:
        log = _open.get("log")
        if log is None or log.read(exists, args):
            return None
        return record(type_, subject, attrs, **more)


def define(family: str, name: str, text: str, spec: Mapping[str, Any] | None = None) -> None:
    """Register `family/name` version 1 once, as the Host's own definition."""
    key = {"family": family, "name": name, "version": 1}
    body = key | {"spec": dict(spec or {}), "status": "standard", "owner": "host", "text": text}
    words = json.dumps(body, sort_keys=True)
    exists = "SELECT 1 FROM definition WHERE family = ? AND name = ? AND version = 1"
    subject = ("definition", f"{family}/{name}")
    record_new(
        exists, (family, name), "definition.registered", subject, key, text=words, node="host"
    )


@contextmanager
def step(name: str, headers: Mapping[str, str], **attributes: str) -> Iterator[None]:
    """One span for one step the Host serves, in the trace `headers` (W3C) name, working for the
    Host's workspace whatever the baggage names. Nothing while the history is deferred."""
    provider = _open.get("tracer")
    if provider is None:
        yield
        return
    from opentelemetry import context  # lazy: the SDK, once open

    from alphalattice.kernel.data_layer.spans import context_of  # lazy: the SDK, once open

    # attached, not only the parent: the step's baggage is what its worker calls carry on
    token = context.attach(context_of(headers, workspace_id=_open["workspace"]))
    try:
        with provider.get_tracer("host").start_as_current_span(
            name, attributes=attributes, record_exception=False
        ):
            yield
    finally:
        context.detach(token)


def traced(rows: Iterable[Mapping[str, Any]] | None) -> None:
    """Append the spans a worker handed back, through the Host's span exporter: one the log
    refuses is counted, never raised into the step. Nothing while deferred."""
    exporter = _open.get("exporter")
    if exporter is not None and rows:
        exporter.write(rows)


def trace_carrier() -> dict[str, str]:
    """The W3C headers of the step this thread serves, for a worker call to join its trace;
    none outside a Host, while deferred, or outside a step."""
    if "tracer" not in _open:
        return {}
    from opentelemetry import trace  # lazy: the SDK, once open

    from alphalattice.kernel.data_layer.spans import carrier  # lazy: the SDK, once open

    return carrier() if trace.get_current_span().get_span_context().is_valid else {}
