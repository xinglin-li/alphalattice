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

The open history also holds the Host's StorePort (`store`), the one writer of the catalog's world
tables, opened at its first use, so a Host that admits nothing never opens the catalog.

The open history also holds the Host's tracer (`spans.tracer_provider` over its log). `step` opens
one span for one step the Host serves, in the trace its client's W3C headers name, and `traced`
appends the spans a worker handed back with its answer. Both do nothing outside a Host whose
history is open.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import sqlite3
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final
from urllib.parse import urlsplit

from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer.admit import secrets_in
from alphalattice.kernel.data_layer.breakdown import goal_trace as traced_goal
from alphalattice.kernel.data_layer.log import HostLog, envelope, uuid7
from alphalattice.kernel.data_layer.reading import experience, goal_notes, reading_session
from alphalattice.kernel.data_layer.root import create_data_root
from alphalattice.kernel.shared_kernel.environment import offline

if TYPE_CHECKING:
    from alphalattice.kernel.data_layer.store import StorePort

DATA_ROOT_ENV: Final = "ALPHALATTICE_DATA_ROOT"
OTLP_ENDPOINT_ENV: Final = "ALPHALATTICE_OTLP_ENDPOINT"
OTLP_CONTENT_ENV: Final = "ALPHALATTICE_OTLP_INCLUDE_CONTENT"
EVIDENCE_TYPE: Final = "research_evidence"
"""The artifact type of a trial's execution evidence, registered by the Host as its own."""
_EVIDENCE_TEXT: Final = "What one research trial's Desk run did: its program, inputs and artifacts."
STEPS: Final = {
    "host.request": (
        ["operation"],
        "none",
        "One request a client sent the Host: its operation and route.",
    ),
    "worker.call": (["call"], "none", "One call a Host worker ran for the step that sent it."),
    "product.operation": (
        ["operation", "caller"],
        "optional",
        "One product operation the Host served; a record past the attributes' bound is the text.",
    ),
    "cli.call": (
        ["command"],
        "none",
        "One CLI invocation: the command an agent or the person ran.",
    ),
}
"""The Host's own step types, by required attributes and text: each span `step`, `begin` and a
worker open is one of these events."""
FACTS: Final = {
    "task.transition": (
        "task",
        [],
        "optional",
        "A Task Control transition: a wake or a command's return.",
    ),
    "artifact.verified": ("artifact", [], "optional", "A result's artifact, verified."),
    "agent.activity": (
        "agent_event",
        ["event_kind"],
        "optional",
        "What an agent's client reported; its summary is the text.",
    ),
    "person.question": (
        None,
        ["code", "state"],
        "none",
        "A question asked of the person, or answered.",
    ),
    "clock.correlated": (
        None,
        [],
        "optional",
        "One correlation of a source's clock with the Host's.",
    ),
    "request.refused": (
        None,
        ["operation", "failure_code", "caller"],
        "none",
        "A request the Host refused.",
    ),
}
"""The Host's own fact types, by subject kind, required attributes and text: each log record `fact`
emits is one of these events."""

_EPOCH: Final = datetime(1970, 1, 1, tzinfo=UTC)
_lock = threading.RLock()
CALLS_KEPT: Final = 1024
"""The live CLI calls, and the conversations' last calls, the Host keeps for links."""
_calls: dict[str, list[str]] = {}
_conversations: dict[str, str] = {}
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


def store() -> StorePort | None:
    """The Host's StorePort over its data root, opened at the first call and kept until the history
    closes; None outside a Host whose history is open, and without the data extra."""
    with _lock:
        port: StorePort | None = _open.get("store")
        if port is None and recording() and _open.get("reading") is not None:
            from alphalattice.kernel.data_layer.store import StorePort  # lazy: the data extra

            port = _open["store"] = StorePort.open(Path(_open["root"]), _open["log"])
        return port


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
                ExternalSpanProcessor,
                LogFactExporter,
                LogSpanExporter,
                logger_provider,
                tracer_provider,
            )

            _open["exporter"] = LogSpanExporter(_open["log"])
            remote = _remote(ExternalSpanProcessor, can_export)
            _open["tracer"] = tracer_provider(_open["exporter"], remote=remote)
            if remote is not None:
                _open["remote"] = remote
            _open["facts"] = LogFactExporter(_open["exporter"])
            _open["logger"] = logger_provider(_open["facts"])
        _open["workspace"] = workspace_on_log(folder, name, workspace_id, root)
        _open["hosts"] = _open.get("hosts", 0) + 1
        spec: dict[str, Any] = {
            "storage": "none",  # the evidence stays in the workspace's store until row 8a
            "table": None,
            "measures": [],
            "keep": "while_active",
        }
        define("artifact_type", EVIDENCE_TYPE, _EVIDENCE_TEXT, spec)
        for step_name, (required, words, text) in STEPS.items():
            kind = {"work": True, "subject": None, "required": required, "text": words}
            define("event_type", step_name, text, kind | {"durable": False})
        for fact_name, (about, required, words, text) in FACTS.items():
            kind = {"work": False, "subject": about, "required": required, "text": words}
            define("event_type", fact_name, text, kind | {"durable": False})
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


def _remote(processor: Any, can_export: Callable[[], bool] | None) -> Any:
    """The opted-in OTLP export (`ALPHALATTICE_OTLP_ENDPOINT`), scrubbed by `processor` and sent
    only while the workspace's network control allows it; None when it is off, with why logged."""
    endpoint = os.environ.get(OTLP_ENDPOINT_ENV, "").strip()
    if not endpoint:
        return None
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

        return processor(
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
    """Commit the open batch, close the reading session and release the data root; the live
    calls and the conversations' last calls go with it, so a history opened again links nothing
    to a call of the one before."""
    with _lock:
        _calls.clear()
        _conversations.clear()
        _open.pop("hosts", None)
        _open.pop("remote", None)
        port = _open.pop("store", None)
        reading = _open.pop("reading", None)
        log = _open.pop("log", None)
        _open.pop("workspace", None)
        _open.pop("root", None)
        provider = _open.pop("tracer", None)
        logger = _open.pop("logger", None)
        _open.pop("exporter", None)
        _open.pop("facts", None)
    if logger is not None:
        logger.shutdown()
    if provider is not None:
        provider.shutdown()  # every span ended before the log closes
    if port is not None:
        port.close()
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


def flush() -> None:
    """Commit the Host's open batch, so a reply never names an event a crash could still lose;
    nothing when this process holds no log."""
    log = _open.get("log")
    if log is not None:
        log.flush()


def opened_log() -> tuple[HostLog, str] | None:
    """The Host's log and its workspace while its history records; None outside a Host."""
    return (_open["log"], str(_open["workspace"])) if recording() else None


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
def step(
    name: str,
    headers: Mapping[str, str],
    *,
    caller: tuple[str, str] | None = None,
    goal: str | None = None,
    **attributes: str,
) -> Iterator[None]:
    """One span for one step the Host serves, in the trace `headers` (W3C) name, working for the
    Host's workspace and `goal` and acting for `caller` (an agent's session), whatever the baggage
    names; every span and record under it carries them. Nothing while the history is deferred."""
    provider = _open.get("tracer")
    if provider is None:
        yield
        return
    from opentelemetry import context  # lazy: the SDK, once open

    from alphalattice.kernel.data_layer.spans import context_of  # lazy: the SDK, once open

    # attached, not only the parent: the step's baggage is what its worker calls carry on
    actor = caller or (None, None)
    bound = context_of(
        headers,
        workspace_id=_open["workspace"],
        goal_id=goal,
        actor_kind=actor[0],
        actor_id=actor[1],
    )
    token = context.attach(bound)
    try:
        with provider.get_tracer("host").start_as_current_span(
            name, attributes=attributes, record_exception=False
        ):
            yield
    finally:
        context.detach(token)


def call(
    ids: tuple[str, str, str | None],
    started_us: int,
    attributes: Mapping[str, Any],
    *,
    caller: tuple[str, str] | None,
    conversation: str | None = None,
    ended: tuple[int, int] | None = None,
) -> None:
    """Record one CLI invocation's span, `ids` its (trace, span, parent): `started` the first time
    the Host sees it, so a reader sees it live, and completed with `ended` (its end and exit code).
    A conversation's consecutive calls are linked, each to the one before. The caller is the
    actor; a call naming none stays the Host's, marked unattributed. Nothing outside a Host."""
    exporter = _open.get("exporter")
    if exporter is None:
        return
    trace, span, parent = ids
    with _lock:
        links = _calls.pop(span, None) if ended is not None else None
        if ended is None and span in _calls:
            return
        if links is None:
            previous = _conversations.pop(conversation, None) if conversation else None
            links = [] if previous is None else [previous]
            if conversation:
                _conversations[conversation] = f"{trace}-{span}"
                _bound(_conversations)
        if ended is None:
            _calls[span] = links
            _bound(_calls)
    attrs = _attributes(attributes) | ({"links": links} if links else {})
    attrs |= {} if caller else {"unattributed": True}
    if ended is not None:
        attrs["exit_code"] = ended[1]
    exporter.append(
        envelope(
            "cli.call",
            attrs=attrs,
            phase="started" if ended is None else "completed",
            workspace=_open["workspace"],
            actor=caller or ("host", "host"),
            node="cli",
            trace=trace,
            span=span,
            parent_span=parent,
            started_us=started_us,
            occurred_us=started_us if ended is None else ended[0],
        )
    )


def _bound(kept: dict[str, Any]) -> None:
    """Keep the newest CALLS_KEPT entries: a call whose end never comes is forgotten in time."""
    while len(kept) > CALLS_KEPT:
        kept.pop(next(iter(kept)))


def begin(
    name: str,
    attributes: Mapping[str, Any],
    text: str | None = None,
    *,
    workspace: str | None = None,
) -> dict[str, Any] | None:
    """Start one step the Host ends later (`finish`), a child of the current step, and record its
    started row now with exactly `attributes`, so a reader sees it live; None outside a Host.
    The step carries that row's `event_id`. The Host writes both rows itself, so no attributes
    merge or flatten; a row the log refuses raises its refusal (`DataError`), so the writer never
    counts it kept."""
    exporter = _open.get("exporter")
    if exporter is None:
        return None
    step = _new_step(name, workspace)
    row = envelope(name, attrs=attributes, text=text, phase="started", **_step_fields(step))
    exporter.append(row | {"occurred_us": step["started_us"]}, text, raising=True)
    return step | {"event_id": row["event_id"]}


def finish(
    step: Mapping[str, Any] | None,
    name: str,
    attributes: Mapping[str, Any],
    text: str | None = None,
    *,
    failed: bool,
    workspace: str | None = None,
) -> str | None:
    """Record the ended row of a step `begin` started, with exactly `attributes`; failed is its
    phase. A step this Host did not begin (None) is a new one, ended now, of type `name`. The
    ended row's event id; None outside a Host. A row the log refuses raises its refusal
    (`DataError`)."""
    exporter = _open.get("exporter")
    if exporter is None:
        return None
    step = step or _new_step(name, workspace)
    phase = "failed" if failed else "completed"
    fields = _step_fields(step) | {"started_us": step["started_us"]}
    row = envelope(step["name"], attrs=attributes, text=text, phase=phase, **fields)
    exporter.append(row, text, raising=True)
    return str(row["event_id"])


def _new_step(name: str, workspace: str | None) -> dict[str, Any]:
    """A step of type `name` starting now, a child of the current step (a new trace outside one),
    in `workspace` (the Host's by default)."""
    from opentelemetry import trace  # lazy: the SDK, once open

    current = trace.get_current_span().get_span_context()
    return {
        "name": name,
        "trace": f"{current.trace_id:032x}" if current.is_valid else secrets.token_hex(16),
        "span": secrets.token_hex(8),
        "parent_span": f"{current.span_id:016x}" if current.is_valid else None,
        "workspace": workspace or _open["workspace"],
        "node": "host",
        "started_us": time.time_ns() // 1000,
    }


def _step_fields(step: Mapping[str, Any]) -> dict[str, Any]:
    return {key: step[key] for key in ("trace", "span", "parent_span", "workspace", "node")}


def fact(
    name: str,
    attributes: Mapping[str, Any],
    *,
    subject: tuple[str, str] | None = None,
    text: str | None = None,
    actor: tuple[str, str] | None = None,
    at: datetime | None = None,
    workspace: str | None = None,
) -> str | None:
    """Emit one log record, a fact in the span current here (a new trace outside one), in the
    Host's workspace, with `attributes` as they are (nested values kept); `text` is its body, `at`
    when it happened (now by default). Its event id; None outside a Host. A record the log refuses
    raises its refusal, so the writer never counts it kept."""
    provider = _open.get("logger")
    if provider is None:
        return None
    own = {"workspace_id": workspace or _open["workspace"]}
    if subject is not None:
        own |= {"subject_kind": subject[0], "subject_id": subject[1]}
    if actor is not None:
        own |= {"actor_kind": actor[0], "actor_id": actor[1]}
    provider.get_logger("host").emit(
        event_name=name,
        body=text,
        attributes={**attributes, **{f"alphalattice.{key}": value for key, value in own.items()}},
        timestamp=None if at is None else (at - _EPOCH) // timedelta(microseconds=1) * 1000,
    )
    if (refused := _open["facts"].refusal()) is not None:
        raise refused
    kept: str | None = _open["facts"].kept()
    return kept


def _attributes(values: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """OpenTelemetry attributes of `values`: a nested map flattened to dotted keys, None dropped,
    a list kept when it holds only strings, numbers or booleans."""
    out: dict[str, Any] = {}
    for key, value in values.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            out |= _attributes(value, f"{name}.")
        elif isinstance(value, list | tuple):
            if all(isinstance(item, str | int | float | bool) for item in value):
                out[name] = list(value)
        elif value is not None:
            out[name] = value
    return out


def goal_trace(
    goal_id: str,
    sessions: Iterable[str] = (),
    native: Iterable[Mapping[str, Any]] = (),
    *,
    workspace: str | None = None,
) -> dict[str, Any] | None:
    """A goal's waterfall and time breakdown, read from the log: each CLI call `workspace` (else
    the Host's) recorded naming the goal or made by one of `sessions` (those that worked for it),
    and the steps under each call, its agents' gaps split by their conversations' `native` records
    where there are any (`breakdown.goal_trace`). A session working in another workspace on the
    same log adds nothing. None outside a Host."""
    log = _open.get("log")
    if log is None:
        return None
    where = workspace or _open["workspace"]
    named = sorted(set(sessions)) or [""]  # a session id is never empty
    columns = (
        "type, phase, span_id, trace_id, parent_span_id, actor_id, occurred_us, started_us, attrs"
    )
    calls = [
        dict(row)
        for row in log.read(
            f"SELECT {columns} FROM event WHERE type = 'cli.call' AND workspace_id = ?"
            " AND (json_extract(attrs, '$.goal_id') = ?"
            " OR json_extract(attrs, '$.goal_id') IS NULL"  # its sessions' calls naming no goal
            " AND json_extract(attrs, '$.\"gen_ai.conversation.id\"') IN"
            f" ({', '.join('?' * len(named))})) ORDER BY seq",
            (where, goal_id, *named),
        )
    ]
    traces = sorted({row["trace_id"] for row in calls})
    steps: list[dict[str, Any]] = []
    for at in range(0, len(traces), 500):
        chunk = traces[at : at + 500]
        marks = ", ".join("?" * len(chunk))
        steps += map(
            dict,
            log.read(
                f"SELECT {columns} FROM event WHERE workspace_id = ? AND trace_id IN ({marks})"
                " AND phase IS NOT NULL AND type <> 'cli.call' ORDER BY seq",
                (where, *chunk),
            ),
        )
    return traced_goal(calls, steps, native=native)


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
