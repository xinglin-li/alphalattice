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
from contextlib import closing, contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final
from urllib.parse import urlsplit

import alphalattice.kernel.data_layer.derived_files as derived_files
import alphalattice.kernel.data_layer.experiences as experiences
import alphalattice.kernel.data_layer.project as project
from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer.admit import secrets_in
from alphalattice.kernel.data_layer.breakdown import goal_trace as traced_goal
from alphalattice.kernel.data_layer.log import HostLog, attach, envelope, uuid7
from alphalattice.kernel.data_layer.reading import LogReader, reading_session
from alphalattice.kernel.data_layer.root import DATA_ROOT_ENV, create_data_root, data_root
from alphalattice.kernel.shared_kernel.environment import offline

if TYPE_CHECKING:
    from alphalattice.kernel.data_layer.store import StorePort

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
_read_lock = threading.Lock()
_open: dict[str, Any] = {}
_attached: dict[str, tuple[Path, Any]] = {}
"""Outside a Host, the log attached read-only for derived file reads, with its root."""


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
    closes; None outside a Host whose history is open, and without the data extra. Its opening
    reads the log, so it takes the log's guard before this module's lock, in the order a writer
    holding a log unit takes them when it records: the other order deadlocks the first open from
    a worker thread against a writer's unit."""
    port: StorePort | None = _open.get("store")
    log: HostLog | None = _open.get("log")
    if port is not None or log is None:
        return port
    with log.reading(), _lock:
        port = _open.get("store")
        if (
            port is None
            and _open.get("log") is log
            and recording()
            and _open.get("reading") is not None
        ):
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
            try:  # the views' DuckDB, checked here without the log, which only the reader opens
                session = reading_session(where, log=False)
                if session is not None:
                    session.close()
                _open["reading"] = None if session is None else LogReader(where)
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
        import alphalattice.kernel.data_layer.arrays as arrays  # lazy: numpy, BLAKE3 and Parquet

        # a copy's lanes move onto this log before anything reads them, its workers included
        where, log = Path(_open["root"]), _open["log"]
        arrays.import_carried(where, log, folder, workspace=_open["workspace"])
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
    # a note a stopped Host left unrendered, or a renderer this release changed, is rendered now
    threading.Thread(target=render_experiences, name="experience-render", daemon=True).start()


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
    with _read_lock:
        if reading is not None:
            reading.close()
        outcome_reading = _open.pop("outcome_reading", None)
        if outcome_reading is not None:
            outcome_reading.close()
        attached = _attached.pop("log", None)
        if attached is not None:
            attached[1].close()
    if log is not None:
        log.close()


def write_execution_outcome(
    content_hash: str, table: Any, *, upgrade_owner: tuple[Any, Path] | None = None
) -> str:
    """Admit one immutable outcome chunk through the Host's StorePort and log its profile."""
    import pyarrow as pa  # lazy: the optional data extra, as DuckDB

    from alphalattice.kernel.data_layer.store import stage  # lazy: the data extra is optional

    with _once:
        port = store()  # first open takes the log guard before history's lock
        with _lock:
            log, root = _open.get("log"), _open.get("root")
            if log is None or root is None:
                raise DataError(
                    "the Host's data root is unavailable; open its history before publishing",
                    code="data_layer.history_unavailable",
                )
            if upgrade_owner is not None and (
                log is not upgrade_owner[0] or root.resolve() != upgrade_owner[1].resolve()
            ):
                raise DataError(
                    "the upgrade and outcome writer do not own the same data root",
                    code="upgrade.refused",
                )
            if port is None:
                raise DataError(
                    "the Host's StorePort is unavailable", code="data_layer.history_unavailable"
                )
        type_name = "execution_outcome"
        exists = log.read(
            "SELECT 1 FROM definition WHERE family = 'artifact_type' AND name = ?", (type_name,)
        )
        if not exists:
            key = {"family": "artifact_type", "name": type_name, "version": 1}
            spec = {
                "storage": "table",
                "table": "results.execution_outcome",
                "measures": ["n_obs"],
                "keep": "while_active",
            }
            body = key | {
                "spec": spec,
                "status": "standard",
                "owner": "host",
                "text": "A sealed execution-outcome chunk.",
            }
            log.record(
                "definition.registered",
                ("definition", "artifact_type/" + type_name),
                key,
                text=json.dumps(body, sort_keys=True),
                node="host",
            )
        profile = _execution_outcome_profile(log.read, content_hash)
        artifact = profile[0] if profile else uuid7()
        if profile and profile[1] != "present":
            raise DataError("the outcome body was collected", code="data_layer.outcome_collected")
        if not profile:
            current = log.read(
                "SELECT max(version) FROM definition WHERE family = 'artifact_type' AND name = ?",
                (type_name,),
            )
            attrs = {
                "type": type_name,
                "type_version": int(current[0][0]),  # the registered one: no version defaults
                "identity_kind": "content",
                "storage": "table",
                "meaning": "The realized execution outcomes for a sealed chunk.",
                "content_digest": content_hash,
                "row_count": table.num_rows,
            }
            log.record(
                "artifact.produced",
                ("artifact", artifact),
                attrs,
                workspace=_open["workspace"],
                node="host",
            )
            log.flush()
        count = port.lake.execute(
            "SELECT count(*) FROM lake.results.execution_outcome WHERE artifact_id = ?",
            (artifact,),
        ).fetchone()[0]
        if count == 0:
            with stage(
                root,
                "execution_outcome",
                None,
                table.append_column("artifact_id", pa.repeat(pa.scalar(artifact), table.num_rows)),
            ) as staged:
                port.commit(staged)
        elif count != table.num_rows:
            raise DataError("the outcome body is incomplete", code="data_layer.outcome_incomplete")
        else:
            columns = ", ".join('"' + name.replace('"', '""') + '"' for name in table.column_names)
            held = (
                port.lake.execute(
                    f"SELECT {columns} FROM lake.results.execution_outcome WHERE artifact_id = ? "
                    "ORDER BY formation_session, listing_id",
                    (artifact,),
                )
                .to_arrow_table()
                .cast(table.schema)
            )
            if not held.equals(table):
                raise DataError(
                    "the outcome identity is already bound to different values",
                    code="data_layer.outcome_incomplete",
                )
        if not log.read("SELECT 1 FROM measure WHERE artifact_id = ?", (artifact,)):
            log.record(
                "result.recorded",
                ("artifact", artifact),
                {"measures": [["n_obs", 1, {}, table.num_rows]]},
                workspace=_open["workspace"],
                node="host",
            )
            log.flush()
        return artifact


def _execution_outcome_profile(read: Callable[..., Any], content_hash: str) -> Any:
    """Find the one registered artifact for an old chunk digest, including after a restart."""
    rows = read(
        "SELECT a.artifact_id, a.body_state FROM artifact a JOIN definition d"
        " ON d.definition_id = a.type_id AND d.version = a.type_version"
        " WHERE d.family = 'artifact_type' AND d.name = 'execution_outcome'"
        " AND a.content_digest = ?",
        (content_hash,),
    )
    if len(rows) > 1:
        raise DataError(
            "the outcome digest names multiple artifacts", code="data_layer.outcome_incomplete"
        )
    return rows[0] if rows else None


def execution_outcome_artifact_id(content_hash: str) -> str:
    """Find the registered artifact named by a legacy outcome content digest."""
    log = _open.get("log")
    if log is not None:
        profile = _execution_outcome_profile(log.read, content_hash)
    else:
        with closing(attach(_open.get("root") or data_root())) as log_reader:
            profile = _execution_outcome_profile(
                lambda sql, args: log_reader.execute(sql, args).fetchall(), content_hash
            )
    if profile is None or profile[1] != "present":
        raise DataError("the outcome body is not in the catalog", code="data_layer.outcome_missing")
    return str(profile[0])


def derived_file_body(table: str, digest: str) -> Path | None:
    """The body the log names for `table`'s file partition with `digest`, or None
    (derived_files.found): read on the Host's log, or outside a Host on its data root's, attached
    read-only once per process and root."""
    log, root = _open.get("log"), _open.get("root")
    if log is not None and root is not None:
        return derived_files.found(log.read, Path(root), table, digest)
    where = data_root()
    with _read_lock:
        held = _attached.get("log")
        if held is None or held[0] != where:
            if held is not None:
                held[1].close()
            held = _attached["log"] = (where, attach(where, shared=True))
        reader = held[1]
        return derived_files.found(
            lambda sql, args: reader.execute(sql, args).fetchall(), where, table, digest
        )


def keep_derived_file(table: str, *, recipe_key: str, part: str, digest: str, source: Path) -> None:
    """Hand the file `source` to the Host's StorePort to keep as a derived partition, rooted by
    the Host's workspace, on its own worker (StorePort.keep_file_later): the caller never waits
    on the keep. Refused by name outside a Host, which alone writes the data root."""
    port = store()
    if port is None or not recording():
        raise DataError(
            f"only the Host keeps {table}'s partitions in the data root; open this workspace in"
            " the Host, which keeps it, or read it from the workspace's own file",
            code="data_layer.derived_files_need_host",
        )
    port.keep_file_later(
        table,
        recipe_key=recipe_key,
        part=part,
        digest=digest,
        source=source,
        rooted_by=str(_open["workspace"]),
    )


def read_execution_outcome(content_hash: str, columns: list[str] | None = None) -> Any:
    """Read the outcome body through the data root's catalog, including from non-Host processes."""
    with _read_lock:
        # The Host's general reader can predate its catalog. Open this reader on first
        # outcome access, then retain it for the Host's lifetime.
        connection = _open.get("outcome_reading")
        owned = not recording()
        if connection is None:
            connection = reading_session(_open.get("root") or data_root(), log=False)
        if connection is None:
            raise DataError(
                "the outcome catalog is unavailable", code="data_layer.history_unavailable"
            )
        try:
            artifact = execution_outcome_artifact_id(content_hash)
            # Discovery can precede the catalog: retain a reader only for a held outcome.
            if not owned:
                _open["outcome_reading"] = connection
            selected = (
                "* EXCLUDE (artifact_id)"
                if columns is None
                else ", ".join('"' + name.replace('"', '""') + '"' for name in columns)
            )
            table = connection.execute(
                f"SELECT {selected} FROM lake.results.execution_outcome "
                "WHERE artifact_id = ? ORDER BY formation_session, listing_id",
                (artifact,),
            ).to_arrow_table()
        finally:
            if owned or connection is not _open.get("outcome_reading"):
                connection.close()
    if table.num_rows == 0:
        raise DataError("the outcome body is not in the catalog", code="data_layer.outcome_missing")
    return table if columns is None else table.select(columns)


def keep_array(
    columns: Mapping[str, Any] | None = None,
    *,
    packed: bytes | None = None,
    durable: bool = True,
) -> str:
    """Keep one array through the door (`arrays`) and return the digest a document cites: in a
    Host, on its log and held by its workspace; in another process, on the log it holds (`held`),
    held by none. `durable=False` leaves its profile in the log's batch for the caller to commit
    (`flush`) before it writes anything citing the digest."""
    import alphalattice.kernel.data_layer.arrays as arrays  # lazy: numpy, BLAKE3 and Parquet

    with _held_log() as found:
        current = found
    log, workspace, root = current if current is not None else (held(), None, data_root())
    if packed is not None:
        return arrays.keep_packed(root, log, packed, workspace=workspace, durable=durable)
    assert columns is not None
    return arrays.keep(root, log, columns, workspace=workspace, durable=durable)


def read_array(content_digest: str, *, packed: bool = False) -> Any:
    """The array a digest names, checked against its profile and proved: its flat columns, or its
    C-order bytes with `packed`; from any process."""
    import alphalattice.kernel.data_layer.arrays as arrays  # lazy: numpy, BLAKE3 and Parquet

    with _array_reader() as (root, read):
        return (arrays.read_packed if packed else arrays.read_columns)(root, read, content_digest)


def array_profile(content_digest: str) -> dict[str, Any] | None:
    """The profile of the array a digest names (its kind, shape and dtypes, its artifact_id and
    body state); None when the door holds none."""
    import alphalattice.kernel.data_layer.arrays as arrays  # lazy: numpy, BLAKE3 and Parquet

    with _array_reader() as (_root, read):
        return arrays.profile(read, content_digest)


def arrays_lacking(digests: Iterable[str]) -> list[str]:
    """The digests the door holds no array for, in one read of the log."""
    import alphalattice.kernel.data_layer.arrays as arrays  # lazy: numpy, BLAKE3 and Parquet

    with _array_reader() as (_root, read):
        return [digest for digest in digests if arrays.profile(read, digest) is None]


def move_carried(folder: Path, workspace_id: str) -> int:
    """A Host's admission of a workspace copy: the lanes it carries move onto the log this process
    holds, held by the workspace, before admission reads them. Nothing when it carries none."""
    import alphalattice.kernel.data_layer.arrays as arrays  # lazy: numpy, BLAKE3 and Parquet

    if not (folder / arrays.CARRIED).is_dir():
        return 0
    log = held()
    held_id = workspace_on_log(folder, folder.name, workspace_id)
    return arrays.import_carried(Path(_open["root"]), log, folder, workspace=held_id)


def carry_arrays(workspace: Path, copy: Path, *, root: Path | None = None) -> int:
    """Write the arrays the log (`root`'s when named) holds for `workspace` into its copy `copy`,
    in the old lane form; the Host that admits the copy (`move_carried`) moves them onto the log
    its process holds.
    Returns the arrays carried.

    contract: carried lanes retire when a copy carries its own segment of the log, as carried
    Tasks do (`carry_tasks`)."""
    import alphalattice.kernel.data_layer.arrays as arrays  # lazy: numpy, BLAKE3 and Parquet

    where = data_root() if root is None else root
    log = holding()
    if log is not None and Path(_open["root"]) == Path(where):
        return _carry(arrays, log.read, where, workspace, copy)
    if not (where / "machine" / "log.sqlite").is_file():
        return 0
    with closing(attach(where)) as con:
        return _carry(arrays, _query(con), where, workspace, copy)


def _carry(arrays: Any, read: Callable[..., Any], root: Path, workspace: Path, copy: Path) -> int:
    known = read(
        "SELECT workspace_id FROM workspace WHERE folder = ? AND status = 'active'",
        (str(workspace.resolve()),),
    )
    return int(arrays.carry(read, root, str(known[0][0]), copy)) if known else 0


def _query(con: sqlite3.Connection) -> Callable[[str, tuple[Any, ...]], list[Any]]:
    return lambda sql, args: con.execute(sql, args).fetchall()


@contextmanager
def _array_reader() -> Iterator[tuple[Path, Callable[..., Any]]]:
    """The data root and a query on its log: the log this process holds, else a read-only
    attachment for this read."""
    log: HostLog | None = _open.get("log")
    if log is not None:
        yield Path(_open["root"]), log.read
        return
    where = data_root()
    if not (where / "machine" / "log.sqlite").is_file():
        yield where, lambda _sql, _args: []  # a root with no log yet holds no array
        return
    try:
        con = attach(where)
    except sqlite3.Error as error:
        raise DataError(
            f"the log at {where} cannot be read ({error}); start the Host on its data root",
            code="data_layer.history_unavailable",
        ) from error
    with closing(con):
        yield where, _query(con)


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
    if type_ == "trial" and more.get("phase") in {"completed", "failed", "cancelled"}:
        try:
            with _lock:
                port = store()
                native = (
                    {"status": "UNMEASURED"}
                    if port is None
                    else port.trial_agent_usage(
                        str(subject[1]), workspace_id=str(more.get("workspace", _open["workspace"]))
                    )
                )
        except Exception as error:
            logging.getLogger(__name__).warning(
                "agent_telemetry.trial_unmeasured: %s", type(error).__name__
            )
            native = {"status": "UNMEASURED"}
        attrs = dict(attrs or {}) | {"agent_telemetry": native}
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


def receive_agent_telemetry(rows: list[dict[str, Any]]) -> int:
    """The Host's existing log lifetime alone may open the catalog's writer, lazily."""
    with _lock:
        port = store()
        if port is None:
            raise DataError(
                "The Host has no writable catalog; start a Host with the data extra",
                code="agent_telemetry.host_unavailable",
            )
        return port.telemetry_batch(rows)


def agent_telemetry(host: str, session: str, *, workspace_id: str) -> list[dict[str, Any]] | None:
    """Read the serving Host's exact workspace/vendor/session; absent means unmeasured."""
    with _lock:
        port = store()
        return (
            None if port is None else port.agent_telemetry(host, session, workspace_id=workspace_id)
        )


def unplaced_agent_telemetry(*, workspace_id: str) -> int | None:
    """Count this Host workspace's records without a conversation/session identity."""
    with _lock:
        port = store()
        return None if port is None else port.unplaced_telemetry(workspace_id=workspace_id)


def read_experience(**filters: Any) -> dict[str, list[dict[str, Any]]] | None:
    """Read the open Host's experience and record the read; None if unavailable."""
    with _lock:
        reading = _open.get("reading")
        if reading is None:
            return None
        _open["log"].flush()
        trials = reading.read("experience", **filters)
        notes = reading.read("goal_notes", **filters)
        record("experience.read", attrs={"count": len(trials)})
        _open["log"].flush()
        return {"trials": trials, "notes": notes}


@contextmanager
def _held_log() -> Iterator[tuple[HostLog, str | None, Path] | None]:
    """The held log under its guard alone, with its workspace and root as they were then; None
    without a log, or when it closed meanwhile. This module's lock is not taken: holding both,
    a caller would wait on one while holding the other against `open_history` (the lock, then
    the guard) or `store()` (the guard, then the lock)."""
    log: HostLog | None = _open.get("log")
    if log is None:
        yield None
        return
    with log.reading():
        workspace, root = _open.get("workspace"), _open.get("root")
        current = _open.get("log") is log and root is not None
        yield (log, workspace, Path(root)) if current else None


def record_review(
    layer: str,
    unit: str,
    verdict: str,
    text: str,
    attribution: str,
    trials: list[str] | None = None,
) -> dict[str, Any]:
    """Record an agent's review note in the Host's log (`experiences.record_note`), committed before
    the reply; its experience is rendered after, off the agent's path. Returns the note's id and
    the aggregates it holds."""
    with _held_log() as held_log:
        if held_log is None or held_log[1] is None:
            raise ValueError("experience.history_unavailable")
        log, workspace, _ = held_log
        seq = experiences.record_note(
            log,
            layer,
            unit,
            verdict,
            text,
            workspace=workspace,
            actor=("agent", attribution),
            trials=trials,
        )
        log.flush()
        note_id, attrs = log.read("SELECT event_id, attrs FROM event WHERE seq = ?", (seq,))[0]
    threading.Thread(target=render_experiences, name="experience-render", daemon=True).start()
    return {"note_id": note_id, **json.loads(attrs)}


def render_experiences() -> list[str]:
    """Render each review note still awaiting its experience, and each experience an older
    renderer wrote (`experiences.render_pending`); the experiences recorded, none without a log."""
    with _held_log() as held_log:
        if held_log is None:
            return []
        log, _, root = held_log
        made = experiences.render_pending(log, root / "blobs")
        log.flush()
        return made


def hypothesis_goal(hypothesis: str, *, expected_goal: str | None = None) -> str | None:
    """Resolve an explicitly selected hypothesis in the current Host's log."""
    with _lock:
        log = _open.get("log")
        if log is None:
            raise ValueError("experience.history_unavailable")
        rows = log.read(
            "SELECT workspace_id, goal_id FROM hypothesis_scope WHERE hypothesis_id = ?",
            (hypothesis,),
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


def version(family: str, name: str) -> int | None:
    """The registered version of `family/name` new work is made under, its current one; None
    outside a Host, while deferred, or when nothing registers it."""
    log = _open.get("log")
    found = (
        None
        if log is None
        else log.read(
            "SELECT version FROM definition_current WHERE family = ? AND name = ?", (family, name)
        )
    )
    return int(found[0][0]) if found else None


def selection(window: Mapping[str, Any], *, actor: tuple[str, str]) -> str | None:
    """The id of the Host workspace's selection of `window` (its cutoff_us, universe name,
    valid_from, valid_to and spec): the one it made of an equal window, compared column by column,
    else a new one, named by a UUIDv7. Its universe is registered as a lineage first. None outside
    a Host, or while deferred."""
    if window.get("universe") is not None:
        define("universe", str(window["universe"]), "A set of listings research reads, as named.")
    with _once:
        if not recording():
            return None
        with _open["log"].unit() as con:
            same = project.window(con, _open["workspace"], window)
            held = con.execute(project.SAME_WINDOW, same).fetchone()
            if held is not None:
                return str(held[0])
            made = uuid7()
            record("selection.made", ("selection", made), window, actor=actor)
            return made


def recipe(
    method: str, params: Mapping[str, Any], reads: Iterable[str] = (), *, actor: tuple[str, str]
) -> str | None:
    """The key of the recipe of `method`, at its registered version, with `params`, reading the
    artifacts `reads` names: the Host fills its upstream from their recipes. Defined, with a
    UUIDv7, at its first use. None outside a Host, or while deferred."""
    with _once:
        if not recording():
            return None
        with _open["log"].unit() as con:
            found = con.execute(
                "SELECT definition_id, version FROM definition_current"
                " WHERE family = 'method' AND name = ?",
                (method,),
            ).fetchone()
            if found is None:
                raise DataError(
                    f"no method definition is named {method}",
                    code="data_layer.method_unregistered",
                )
            read = sorted(set(reads))
            key = project.recipe_key(found[0], found[1], params, project.upstream(con, read))
            if con.execute("SELECT 1 FROM recipe WHERE recipe_key = ?", (key,)).fetchone():
                return key
            attrs = {"method": method, "method_version": found[1], "params": dict(params)}
            attrs |= {"recipe_key": key} | ({"reads": read} if read else {})
            record("recipe.defined", ("recipe", uuid7()), attrs, actor=actor)
            return key


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
