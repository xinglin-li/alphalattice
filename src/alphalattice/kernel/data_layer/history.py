"""The machine's log as this process holds it, and the research history the Host records in it.

A process writes the log of the data root it holds (`held`): `ALPHALATTICE_DATA_ROOT`, else the
application data folder (`%LOCALAPPDATA%/AlphaLattice` on Windows, `$XDG_DATA_HOME/alphalattice`
elsewhere). It takes the root's lock at its first write, a Task's or the Host's, and keeps it until
`close_history`. Only one process holds a data root: another is refused by name
(`data_layer.host_running`), with the way on, a second data root or the other Host closed. A copy
of a workspace (a model sandbox, a saved-object readback) opens on a data root of its own
(`own_root`), so its Tasks never enter the machine's history; a restored workspace's first open is a
workspace like any other.

The Host opens its workspace's history when it starts (`open_history`): the held log, the reading
views and its workspace's row. A Host whose log or views will not open does not start: its Tasks
are written to that log. Research paths call `record` beside what they write today; it writes only
in a Host whose history is open (`recording`), and returns None elsewhere. Each event is a small
row in the step's own transaction; no artifact is kept, copied or hashed for its sake.

The open history also holds the Host's tracer (`spans.tracer_provider` over its log). `step` opens
one span for one step the Host serves, in the trace its client's W3C headers name, and `traced`
appends the spans a worker handed back with its answer. Both do nothing outside a Host whose
history is open.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlsplit

from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer.admit import secrets_in
from alphalattice.kernel.data_layer.log import HostLog, uuid7
from alphalattice.kernel.data_layer.reading import experience, goal_notes, reading_session
from alphalattice.kernel.data_layer.root import create_data_root
from alphalattice.kernel.shared_kernel.environment import offline

DATA_ROOT_ENV: Final = "ALPHALATTICE_DATA_ROOT"
OTLP_ENDPOINT_ENV: Final = "ALPHALATTICE_OTLP_ENDPOINT"
OTLP_CONTENT_ENV: Final = "ALPHALATTICE_OTLP_INCLUDE_CONTENT"
EVIDENCE_TYPE: Final = "research_evidence"
"""The artifact type of a trial's execution evidence, registered by the Host as its own."""
_EVIDENCE_TEXT: Final = "What one research trial's Desk run did: its program, inputs and artifacts."
STEPS: Final = {
    "host.request": (["operation"], "One request a client sent the Host: its operation and route."),
    "worker.call": (["call"], "One call a Host worker ran for the step that sent it."),
}
"""The Host's own step types: each span `step` and a worker open is one of these events."""

_lock = threading.RLock()
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


def held(root: Path | None = None) -> HostLog:
    """This process's log: its data root's, opened at the first call and held until
    `close_history`. Refused by name when another process holds the root
    (`data_layer.host_running`, with the way on), its log does not open, or this process holds
    another root (`data_layer.root_held`): a process writes one data root's log."""
    with _lock:
        where = data_root() if root is None else root
        log: HostLog | None = _open.get("log")
        if log is not None:
            if Path(_open["root"]) != Path(where):
                raise DataError(
                    f"this process holds the data root {_open['root']}, not {where}",
                    code="data_layer.root_held",
                )
            return log
        try:
            create_data_root(where)
            log = HostLog.open(where)
        except DataError as error:
            if error.failure.code != "data_layer.host_running":
                raise
            raise DataError(
                f"another AlphaLattice Host holds the data root {where}; close it, or start this "
                f"one with {DATA_ROOT_ENV} set to a folder of its own",
                code="data_layer.host_running",
            ) from error
        except (sqlite3.Error, OSError) as error:
            raise DataError(
                f"the log at {where} did not open ({error}); restore it from its last backup",
                code="data_layer.log_unavailable",
            ) from error
        _open.update(log=log, root=where)
        return log


def holding() -> HostLog | None:
    """The log this process holds, if it holds the data root named now; it takes none."""
    log: HostLog | None = _open.get("log")
    return log if log is not None and Path(_open["root"]) == data_root() else None


def workspace_on_log(
    folder: Path, name: str, workspace_id: str | None = None, root: Path | None = None
) -> str:
    """The held log's id for the workspace in `folder`, recording it when it is new.

    The Host names its workspace's id; it is recorded with the folder it was opened from, which
    belongs to that workspace alone from then on. Without an id, the folder's workspace is the
    one the log knows by that folder, or a new one.
    """
    log, where = held(root), str(folder.resolve())
    with log.unit() as con:
        if workspace_id is None:
            known = con.execute(
                "SELECT workspace_id FROM workspace WHERE folder = ? AND status = 'active'",
                (where,),
            ).fetchone()
            if known is not None:
                return str(known[0])
            workspace_id = uuid7()
        subject, place = ("workspace", workspace_id), {"folder": where}
        if con.execute(
            "SELECT 1 FROM workspace WHERE workspace_id = ?", (workspace_id,)
        ).fetchone():
            log.record("workspace.opened", subject, place, workspace=workspace_id)
        else:
            log.record("workspace.created", subject, place | {"name": name}, workspace=workspace_id)
    return workspace_id


def open_history(
    workspace_id: str,
    name: str,
    folder: Path,
    root: Path | None = None,
    *,
    can_export: Callable[[], bool] | None = None,
) -> None:
    """Open this Host's history: the held log, its reading views and its workspace's row.

    Refused by name when the log is held elsewhere or does not open, and
    `data_layer.history_unavailable` when the views do not; either way the data root is released.
    Opening an open history again records the workspace only.
    """
    with _lock:
        held(root)
        if "reading" not in _open:
            where = data_root() if root is None else root
            try:
                _open["reading"] = reading_session(where)
            except Exception as error:  # DuckDB raises its own errors too
                close_history()  # the data root's lock goes with it, so the next start can open it
                raise DataError(
                    f"the reading views at {where} did not open ({error})",
                    code="data_layer.history_unavailable",
                ) from error
            from alphalattice.kernel.data_layer.spans import (  # lazy: the SDK, a Host's alone
                LogSpanExporter,
                tracer_provider,
            )

            _open["exporter"] = LogSpanExporter(_open["log"])
            remote = _remote_exporter(can_export)
            _open["tracer"] = tracer_provider(_open["exporter"], remote=remote)
            if remote is not None:
                _open["remote"] = remote
        _open["workspace"] = workspace_on_log(folder, name, workspace_id, root)
        _open["hosts"] = _open.get("hosts", 0) + 1
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
        define(
            "event_type",
            "experience.read",
            "A read of recorded trial experience.",
            {
                "work": False,
                "subject": None,
                "required": ["count"],
                "text": "none",
                "durable": False,
            },
        )


def _remote_exporter(can_export: Callable[[], bool] | None) -> Any:
    """The opt-in OTLP span processor `OTLP_ENDPOINT_ENV` names, or None: off with a warning when
    the endpoint is not a plain HTTP URL, the Host is offline, or the exporter is unavailable;
    each export is sent only while `can_export` allows the workspace's network."""
    endpoint = os.environ.get(OTLP_ENDPOINT_ENV, "").strip()
    if not endpoint:
        return None
    from alphalattice.kernel.data_layer.spans import (  # lazy: the SDK, a Host's alone
        ExternalSpanProcessor,
    )

    try:
        address = urlsplit(endpoint)
        if (
            address.scheme not in {"http", "https"}
            or not address.hostname
            or address.username is not None
            or address.password is not None
            or address.query
            or address.fragment
            or secrets_in({"endpoint": endpoint})
        ):
            raise ValueError("endpoint must be an HTTP URL without credentials or query")
        if offline():
            logging.getLogger(__name__).warning("OTLP export is off: the Host is offline")
            return None
        from opentelemetry.exporter.otlp.proto.http import (  # lazy: opt-in export
            trace_exporter,
        )
        from opentelemetry.sdk.trace.export import (  # lazy: opt-in export
            SpanExportResult,
        )

        class NetworkBoundExporter(trace_exporter.OTLPSpanExporter):
            def export(self, spans):
                try:
                    if offline() or can_export is None or not can_export():
                        logging.getLogger(__name__).warning(
                            "OTLP export is off: workspace network is not allowed"
                        )
                        return SpanExportResult.FAILURE
                except Exception as error:
                    logging.getLogger(__name__).warning(
                        "OTLP export is off: network control unavailable (%s)",
                        type(error).__name__,
                    )
                    return SpanExportResult.FAILURE
                return super().export(spans)

        return ExternalSpanProcessor(
            NetworkBoundExporter(endpoint=endpoint),
            include_content=os.environ.get(OTLP_CONTENT_ENV) == "1",
        )
    except (OSError, ValueError) as error:
        logging.getLogger(__name__).warning(
            "OTLP export is off: endpoint or exporter unavailable (%s)", type(error).__name__
        )
        return None


@contextmanager
def own_root(folder: Path) -> Iterator[Path]:
    """Hold `folder` as this process's data root while a copy of a workspace opens, so the copy's
    Tasks go to a log of its own and never the machine's; the root held before comes back after.
    Refused by name (`data_layer.root_held`) while this process's Host records."""
    with _lock:
        if recording():
            raise DataError(
                f"this process's Host records in {_open['root']}; a copy opens in another process",
                code="data_layer.root_held",
            )
        aside, before = dict(_open), os.environ.get(DATA_ROOT_ENV)
        _open.clear()
        os.environ[DATA_ROOT_ENV] = str(folder)
    try:
        yield folder
    finally:
        with _lock:
            close_history()
            _open.update(aside)
            if before is None:
                os.environ.pop(DATA_ROOT_ENV, None)
            else:
                os.environ[DATA_ROOT_ENV] = before


def release_history() -> None:
    """A Host's stop: its events are committed, and its history closes with the last Host this
    process opened on it."""
    with _lock:
        hosts = _open.get("hosts", 0) - 1
        if hosts > 0:
            _open["hosts"] = hosts
            _open["log"].flush()
            return
        close_history()


def close_history() -> None:
    """Commit the open batch, close the reading session and release the data root."""
    with _lock:
        _open.pop("hosts", None)
        _open.pop("remote", None)
        reading = _open.pop("reading", None)
        log = _open.pop("log", None)
        _open.pop("workspace", None)
        _open.pop("root", None)
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
    names another; None outside a Host whose history is open."""
    if not recording():
        return None
    more.setdefault("workspace", _open["workspace"])
    return int(_open["log"].record(type_, subject, attrs, **more))


def recording() -> bool:
    """Whether this process is a Host whose history records."""
    return "workspace" in _open


def read_experience(**filters: Any) -> dict[str, list[dict[str, Any]]] | None:
    """Read the open Host's experience and record the read; None if unavailable."""
    with _lock:
        reading = _open.get("reading")
        if reading is None:
            return None
        _open["log"].flush()
        trials = experience(reading, **filters)
        notes = goal_notes(reading, **filters)
        record("experience.read", attrs={"count": len(trials)})
        _open["log"].flush()
        return {"trials": trials, "notes": notes}


def hypothesis_goal(hypothesis: str, *, expected_goal: str | None = None) -> str | None:
    """Resolve an explicitly selected hypothesis in the current Host's log."""
    with _lock:
        log = _open.get("log")
        if log is None:
            raise ValueError("experience.history_unavailable")
        rows = log.read(
            "SELECT workspace_id, goal_id FROM hypothesis WHERE hypothesis_id = ?", (hypothesis,)
        )
        if not rows:
            raise ValueError("experience.hypothesis_unknown")
        workspace, goal = rows[0]
        if workspace != _open["workspace"]:
            raise ValueError("experience.hypothesis_workspace")
        if expected_goal is not None and goal != expected_goal:
            raise ValueError("experience.hypothesis_goal")
        return goal


def record_hypothesis(goal_id: str, name: str, text: str, attribution: str) -> str:
    """Return the one recorded Host-issued id for a goal statement, including on retry."""
    with _once:
        log = _open.get("log")
        if log is None:
            raise ValueError("experience.history_unavailable")
        rows = log.read(
            "SELECT e.subject_id, x.text, e.actor_id FROM event e "
            "LEFT JOIN event_text x ON x.seq=e.seq "
            "WHERE e.workspace_id=? AND e.type='hypothesis.stated' "
            "AND json_extract(e.attrs, '$.goal_id')=? "
            "AND json_extract(e.attrs, '$.statement_id')=?",
            (_open["workspace"], goal_id, name),
        )
        if rows:
            if len(rows) != 1 or rows[0][1] != text or rows[0][2] != attribution:
                raise ValueError("goal.duplicate_identifier")
            return str(rows[0][0])
        hypothesis_id = str(uuid7())
        record(
            "hypothesis.stated",
            ("hypothesis", hypothesis_id),
            {"goal_id": goal_id, "statement_id": name},
            text=text,
            actor=("agent", attribution),
        )
        log.flush()
        return hypothesis_id


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
        if not recording() or _open["log"].read(exists, args):
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


def definition_id(family: str, name: str) -> str | None:
    """The lineage `name` names in `family`, by any name it held: what a recipe or a row keys by,
    never the name. None outside a Host, while deferred, or when nothing registers it."""
    log = _open.get("log")
    found = (
        None
        if log is None
        else log.read(
            "SELECT definition_id FROM definition_name WHERE family = ? AND name = ?",
            (family, name),
        )
    )
    return str(found[0][0]) if found else None


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
        remote = _open.get("remote")
        for row in rows:
            if remote is None:
                exporter.write((row,))
                continue
            try:
                admitted = remote.log_row(row)
            except (KeyError, TypeError, ValueError) as error:
                logging.getLogger(__name__).warning(
                    "worker span was not recorded: %s", type(error).__name__
                )
                continue
            if exporter.write((admitted,)).name == "SUCCESS":
                remote.worker_rows((row,))


def trace_carrier() -> dict[str, str]:
    """The W3C headers of the step this thread serves, for a worker call to join its trace;
    none outside a Host, while deferred, or outside a step."""
    if "tracer" not in _open:
        return {}
    from opentelemetry import trace  # lazy: the SDK, once open

    from alphalattice.kernel.data_layer.spans import carrier  # lazy: the SDK, once open

    return carrier() if trace.get_current_span().get_span_context().is_valid else {}
