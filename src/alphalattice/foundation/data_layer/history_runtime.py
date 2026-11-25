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
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import ExitStack, closing, contextmanager, nullcontext
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final
from urllib.parse import urlsplit

import alphalattice.foundation.data_layer.experiences as experiences
import alphalattice.foundation.data_layer.governance as governance
import alphalattice.foundation.data_layer.knowledge as knowledge
import alphalattice.foundation.data_layer.project as project
import alphalattice.foundation.data_layer.recall as recall
import alphalattice.foundation.data_layer.return_series as return_series
from alphalattice.capabilities.data.errors import DataError
from alphalattice.capabilities.domain_contracts import telemetry_names as telemetry
from alphalattice.capabilities.domain_contracts.environment import offline
from alphalattice.foundation.data_layer.admit import secrets_in
from alphalattice.foundation.data_layer.breakdown import goal_trace as traced_goal
from alphalattice.foundation.data_layer.log import HostLog, attach, await_workers, envelope, uuid7
from alphalattice.foundation.data_layer.reading import (
    CONDITION_TYPE,
    CatalogReader,
    condition_events,
    reading_session,
)
from alphalattice.foundation.data_layer.root import (
    DATA_ROOT_ENV,
    create_data_root,
    data_root,
)
from alphalattice.kernel.clock import wall_time_ns
from alphalattice.kernel.config import Change, process_override
from alphalattice.kernel.config import current as process_config
from alphalattice.kernel.execution import interruptible

if TYPE_CHECKING:
    from alphalattice.foundation.data_layer.knowledge import Retrieved
    from alphalattice.foundation.data_layer.store import StorePort

OTLP_ENDPOINT_ENV: Final = "OTEL_EXPORTER_OTLP_ENDPOINT"
OTLP_TRACES_ENDPOINT_ENV: Final = "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
OTLP_CONTENT_ENV: Final = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
EVIDENCE_TYPE: Final = "research_evidence"
"""The artifact type of a trial's execution evidence, registered by the Host as its own."""


_EVIDENCE_TEXT: Final = "What one research trial's Desk run did: its program, inputs and artifacts."
_RETURN_SERIES_TEXT: Final = "A trial's headline monthly return series, kept with its profile."
_CONDITION_TEXT: Final = "A market's condition at a selection's cutoff, as registered metrics."
STEPS: Final = {
    telemetry.SPAN_GEN_AI_RETRIEVAL: (
        [],
        "none",
        "One source-scoped hybrid retrieval.",
    ),
    telemetry.SPAN_HOST_REQUEST: (
        [telemetry.OPERATION],
        "none",
        "One request a client sent the Host: its operation and route.",
    ),
    telemetry.SPAN_WORKER_CALL: (
        [telemetry.CALL],
        "none",
        "One call a Host worker ran for the step that sent it.",
    ),
    telemetry.SPAN_PRODUCT_OPERATION: (
        [telemetry.OPERATION, telemetry.CALLER],
        "optional",
        "One product operation the Host served; a record past the attributes' bound is the text.",
    ),
    telemetry.SPAN_CLI_CALL: (
        [telemetry.COMMAND],
        "none",
        "One CLI invocation: the command an agent or the person ran.",
    ),
    telemetry.GEN_AI_OPERATION_SEARCH_MEMORY: (
        [telemetry.GEN_AI_OPERATION_NAME, telemetry.GEN_AI_MEMORY_STORE_ID],
        "none",
        "One read of an agent memory store, as the gen_ai conventions name it.",
    ),
}
"""The Host's own step types, by required attributes and text: each span `step`, `begin` and a
worker open is one of these events."""
FACTS: Final = {
    telemetry.EVENT_TASK_TRANSITION: (
        "task",
        [],
        "optional",
        "A Task Control transition: a wake or a command's return.",
    ),
    telemetry.EVENT_ARTIFACT_VERIFIED: (
        "artifact",
        [],
        "optional",
        "A result's artifact, verified.",
    ),
    telemetry.EVENT_AGENT_ACTIVITY: (
        "agent_event",
        [telemetry.EVENT_KIND],
        "optional",
        "What an agent's client reported; its summary is the text.",
    ),
    telemetry.EVENT_PERSON_QUESTION: (
        None,
        [telemetry.CODE, telemetry.STATE],
        "none",
        "A question asked of the person, or answered.",
    ),
    telemetry.EVENT_CLOCK_CORRELATED: (
        None,
        [],
        "optional",
        "One correlation of a source's clock with the Host's.",
    ),
    telemetry.EVENT_REQUEST_REFUSED: (
        None,
        [telemetry.OPERATION, telemetry.FAILURE_CODE, telemetry.CALLER],
        "none",
        "A request the Host refused.",
    ),
}
"""The Host's own fact types, by subject kind, required attributes and text: each log record `fact`
emits is one of these events."""

_EPOCH: Final = datetime(1970, 1, 1, tzinfo=UTC)
_lock = threading.RLock()
"""Protect the in-memory handles and state. Release before using the log or StorePort."""
_opening = threading.RLock()
"""Serialize creation of an unpublished log, independently of the Host lifecycle."""
_lifecycle = threading.RLock()
"""Serialize Host start, last close and process-local root switches."""
CALLS_KEPT: Final = 1024
"""The live CLI calls, and the conversations' last calls, the Host keeps for links."""
_calls: dict[str, list[str]] = {}
_conversations: dict[str, str] = {}
_once = threading.Lock()
_read_lock = threading.Lock()
_open: dict[str, Any] = {}
_catalog_surfaces: dict[Path, CatalogReader] = {}
_catalog_lock = threading.Lock()


def held(root: Path | None = None) -> HostLog:
    """This process's log: its data root's, opened at the first call and held until
    `close_history`. Refused by name when another process holds the root
    (`data_layer.host_running`, with the way on), its log does not open, or this process holds
    another root (`data_layer.root_held`): a process writes one data root's log. The Task workers
    of a Host that held the root and ended are waited for first (`await_workers`): they may still
    hold its log for a moment."""
    _, log = _log_at(root)
    if log is not None:
        return log
    with _opening:
        where, log = _log_at(root)
        if log is not None:
            return log
        try:
            create_data_root(where)
            await_workers(where)
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
        with _lock:
            _open.update(log=log, root=where)
        return log


def _log_at(root: Path | None) -> tuple[Path, HostLog | None]:
    """Resolve the root and check the published log under the state lock."""
    with _lock:
        where = data_root() if root is None else root
        log: HostLog | None = _open.get("log")
        if log is not None and Path(_open["root"]) != Path(where):
            raise DataError(
                f"this process holds the data root {_open['root']}, not {where}",
                code="data_layer.root_held",
            )
        return where, log


def holding() -> HostLog | None:
    """The log this process holds, if it holds the data root named now; it takes none."""
    with _lock:
        log: HostLog | None = _open.get("log")
        return log if log is not None and Path(_open["root"]) == data_root() else None


def host_duckdb(root: Path) -> Any | None:
    """The loaded DuckDB instance held by this root's open Host history."""
    with _lock:
        active = _open.get("root")
        if active is not None and active.resolve() == root.resolve():
            return _open.get("duckdb")
        return None


def store() -> StorePort | None:
    """The Host's StorePort over its data root, opened at the first call and kept until the history
    closes; None outside a Host whose history is open, and without the data extra. Its opening
    reads the log, so it takes the log's guard before this module's lock, in the order a writer
    holding a log unit takes them when it records: the other order deadlocks the first open from
    a worker thread against a writer's unit. Once open, outside both, the Host's start collects the
    bodies nothing holds (`StorePort.collect_unheld`); a restore's own opening never does."""
    with _lock:
        port: StorePort | None = _open.get("store")
        log: HostLog | None = _open.get("log")
    if port is not None or log is None:
        return port
    with log.reading():
        with _lock:
            if _open.get("log") is not log:
                return None
            port = _open.get("store")
            if port is not None or not recording() or _open.get("reading") is None:
                return port
            root = Path(_open["root"])
        from alphalattice.foundation.data_layer.store import StorePort  # lazy: the data extra

        port = StorePort.open(root, log, owner=host_duckdb(root))
        with _lock:
            _open["store"] = port
    # The collector takes the backup's lock first, so run it outside the log guard.
    port.started["unheld"] = port.collect_unheld()
    return port


@contextmanager
def catalog_session(root: Path | None = None) -> Iterator[Any | None]:
    """A cursor on the Host catalog or another root's read-only catalog."""
    active = (_open.get("root") or data_root()).resolve()
    where = active if root is None else root.resolve()
    port = store() if where == active else None
    if port is not None:
        cursor = port.lake.cursor()
    else:
        with _catalog_lock:
            surface = _catalog_surface(where)
            cursor = None if surface is None else surface.cursor()
    if cursor is None:
        yield None
        return
    try:
        with interruptible(cursor.interrupt):
            yield cursor
    finally:
        cursor.close()


def _catalog_surface(root: Path) -> CatalogReader | None:
    """The process's catalog outside a Host, opened under the catalog lock."""
    surface = _catalog_surfaces.get(root)
    if surface is None and (root / "machine" / "catalog.sqlite").is_file():
        connection = reading_session(root, log=False)
        if connection is not None:
            surface = _catalog_surfaces[root] = CatalogReader(connection)
    return surface


@contextmanager
def catalog_projection(
    stage: str, query: str, parameters: Sequence[Any], *, root: Path | None = None
) -> Iterator[Any | None]:
    """Read one shaped result on the held catalog. Materialize it inside this context."""
    with _catalog_lock:
        host_root = _open.get("root") if recording() else None
        where = (root or host_root or data_root()).resolve()
        if host_root is not None and host_root.resolve() == where:
            surface = _open.get("catalog_reader")
            if surface is None and (port := _open.get("store")) is not None:
                # A catalog made on the first write was absent when the Host opened.
                surface = _open["catalog_reader"] = CatalogReader(port.lake.cursor())
        else:
            surface = _catalog_surface(where)
    if surface is None:
        yield None
    else:
        with surface.projection(stage, query, parameters) as result:
            yield result


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
                "SELECT workspace_id, status FROM workspace"
                " WHERE folder = ? AND status IN ('active', 'missing')",
                (where,),
            ).fetchone()
            if known is not None:
                if known[1] == "missing":  # its folder holds it again
                    governance.record_workspace_status(log, str(known[0]), "active")
                return str(known[0])
            workspace_id = uuid7()
        subject, place = ("workspace", workspace_id), {"folder": where}
        status = con.execute(
            "SELECT status FROM workspace WHERE workspace_id = ?", (workspace_id,)
        ).fetchone()
        if status is not None:
            if status[0] == "deleted":
                raise DataError(
                    f"Workspace {workspace_id} was deleted. Create a new workspace to continue.",
                    code="data_layer.workspace_deleted",
                )
            if status[0] == "missing":
                governance.record_workspace_status(log, workspace_id, "active")
            log.record("workspace.opened", subject, place, workspace=workspace_id)
        else:
            log.record("workspace.created", subject, place | {"name": name}, workspace=workspace_id)
    return workspace_id


def mark_missing() -> list[str]:
    """Reconcile workspace folders on the serving Host and return the IDs first marked missing."""
    opened = opened_log()
    return governance.reconcile_workspaces(opened[0]) if opened is not None else []


def removed_since(
    age: timedelta, receipt: str, *, now_us: int | None = None
) -> list[tuple[str, list[dict[str, Any]]]]:
    """The workspaces missing or deleted for longer than `age` on the log's clock (or at `now_us`),
    each with the receipts from its current removal episode. Deletion preserves the episode's
    earlier missing timestamp and receipts. None outside a recording Host."""
    opened = opened_log()
    if opened is None:
        return []
    log = opened[0]
    before = (log.clock() if now_us is None else now_us) - age // timedelta(microseconds=1)
    rows = log.read(
        """WITH removed AS (
             SELECT w.workspace_id, coalesce(
               (SELECT CASE WHEN json_extract(prev.attrs, '$.status') = 'missing'
                            THEN prev.seq END
                FROM event prev
                WHERE w.status = 'deleted' AND prev.subject_kind = 'workspace'
                  AND prev.subject_id = w.workspace_id AND prev.type = 'workspace.status_changed'
                  AND prev.seq < w.status_seq ORDER BY prev.seq DESC LIMIT 1),
               w.status_seq) AS missing_seq
             FROM workspace w WHERE w.status IN ('missing', 'deleted')
           )
           SELECT w.workspace_id, r.attrs FROM removed w JOIN event e ON e.seq = w.missing_seq
           LEFT JOIN event r ON r.type = ? AND r.subject_kind = 'workspace'
             AND r.subject_id = w.workspace_id AND r.seq > w.missing_seq
           WHERE e.recorded_us < ? ORDER BY w.workspace_id, r.seq""",
        (receipt, before),
    )
    removed: dict[str, list[dict[str, Any]]] = {}
    for workspace_id, attrs in rows:
        receipts = removed.setdefault(str(workspace_id), [])
        if attrs is not None:
            receipts.append(json.loads(attrs))
    return list(removed.items())


def admit_removal(
    workspace_id: str, receipt: str, attrs: Mapping[str, Any], *, resumed: bool = False
) -> bool:
    """Admit a removed workspace's files to go when its folder is confirmed absent.

    Record the receipt durably on the first try. A resumed try already has its receipt.
    Return False outside a recording Host."""
    opened = opened_log()
    if opened is None:
        return False
    log = opened[0]
    with log.unit() as con:
        row = con.execute(
            "SELECT status, folder FROM workspace WHERE workspace_id = ?", (workspace_id,)
        ).fetchone()
        if row is None or row[0] not in ("missing", "deleted"):
            return False
        if governance.folder_present(row[1]) is not False:
            return False
        if not resumed:
            log.record(receipt, ("workspace", workspace_id), dict(attrs), workspace=workspace_id)
    log.flush()
    return True


def _capture_message_content() -> bool:
    """Whether the OpenTelemetry GenAI capture mode permits content on spans."""
    return process_config().get(OTLP_CONTENT_ENV, "").strip().lower() in {
        "span_only",
        "span_and_event",
    }


def _remote(processor: Any, can_export: Callable[[], bool] | None) -> Any:
    """Scrub and send opted-in OTLP traces while the workspace's network control allows it."""
    config = process_config()
    traces = config.secret(OTLP_TRACES_ENDPOINT_ENV)
    endpoint = traces.reveal().strip() if traces is not None else ""
    if not endpoint:
        base = config.secret(OTLP_ENDPOINT_ENV)
        base_endpoint = base.reveal().strip() if base is not None else ""
        if not base_endpoint:
            return None
        endpoint = f"{base_endpoint.removesuffix('/')}/v1/traces"
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
            NetworkBoundExporter(endpoint=endpoint), include_content=_capture_message_content()
        )
    except (OSError, ValueError) as error:
        logging.getLogger(__name__).warning(
            "OTLP export is off: endpoint or exporter unavailable (%s)", type(error).__name__
        )
        return None


@contextmanager
def own_root(folder: Path) -> Iterator[Path]:
    """Hold `folder` as this process's data root while a copy of a workspace opens, so the copy's
    Tasks go to a log of its own and never the machine's; the root held before comes back after,
    and the catalog read on `folder` outside a Host closes with the hold, so a folder made again
    at that path is read afresh. Refused by name (`data_layer.root_held`) while this process's
    Host records."""
    with ExitStack() as settings:
        with _lifecycle, _opening, _lock:
            if recording():
                raise DataError(
                    f"this process's Host records in {_open['root']}; "
                    "a copy opens in another process",
                    code="data_layer.root_held",
                )
            aside = dict(_open)
            settings.enter_context(
                process_override(
                    Change("borrow-history-root", {DATA_ROOT_ENV: str(folder)}), native=True
                )
            )
            _open.clear()
        try:
            yield folder
        finally:
            with _lifecycle, _opening:
                _close_history()
                with _lock:
                    _open.update(aside)
                    settings.close()
            with _catalog_lock:
                surface = _catalog_surfaces.pop(folder.resolve(), None)
            if surface is not None:
                surface.close()


def release_history() -> None:
    """A Host's stop: its events are committed, and its history closes with the last Host this
    process opened on it."""
    with _lifecycle:
        with _lock:
            hosts = _open.get("hosts", 0) - 1
            log = _open.get("log")
            if hosts > 0:
                _open["hosts"] = hosts
        if hosts > 0:
            log.flush()
        else:
            _close_history()


def close_history() -> None:
    """Commit the open batch, close the reading session and release the data root; the live
    calls and the conversations' last calls go with it, so a history opened again links nothing
    to a call of the one before."""
    with _lifecycle:
        _close_history()


def _close_history() -> None:
    """Detach the Host's handles after its guarded reads, then close them outside both locks."""
    with _opening:
        with _lock:
            log = _open.get("log")
        with log.reading() if log is not None else nullcontext(), _lock:
            _calls.clear()
            _conversations.clear()
            _open.pop("hosts", None)
            _open.pop("remote", None)
            port = _open.pop("store", None)
            with _catalog_lock:
                catalog_reader = _open.pop("catalog_reader", None)
            reading = _open.pop("reading", None)
            duckdb = _open.pop("duckdb", None)
            _open.pop("log", None)
            _open.pop("workspace", None)
            _open.pop("portfolio_catalog_workspaces", None)
            _open.pop("root", None)
            provider = _open.pop("tracer", None)
            logger = _open.pop("logger", None)
            _open.pop("exporter", None)
            _open.pop("facts", None)
    if logger is not None:
        logger.shutdown()
    if provider is not None:
        provider.shutdown()  # every span ended before the log closes
    if catalog_reader is not None:
        catalog_reader.close()
    if port is not None:
        port.close()
    if duckdb is not None:
        duckdb.close()
    recall.close()
    with _read_lock:
        if reading is not None:
            reading.close()
    if log is not None:
        log.close()


def _hold_existing_result(log: HostLog, artifact: str) -> None:
    """Keep a shared result for the current workspace when it does not already hold it."""
    workspace = _open["workspace"]
    if workspace is not None and not log.read(
        "SELECT 1 FROM live_ref WHERE artifact_id = ? AND root_kind = 'workspace' AND root_id = ?",
        (artifact, workspace),
    ):
        log.record("artifact.held", ("artifact", artifact), {}, workspace=workspace, node="host")
        log.flush()


def _result_writer(upgrade_owner: tuple[Any, Path] | None) -> tuple[Any, Path, Any]:
    """Resolve the held StorePort and confirm an upgrade uses the same owner."""
    port = store()
    with _lock:
        log, root = _open.get("log"), _open.get("root")
        if log is None or root is None or port is None:
            raise DataError("Host data root is unavailable", code="data_layer.history_unavailable")
        if upgrade_owner is not None and (
            log is not upgrade_owner[0] or root.resolve() != upgrade_owner[1].resolve()
        ):
            raise DataError("upgrade owns a different data root", code="upgrade.refused")
    return log, root, port


def _produce_result_artifact(
    log: HostLog,
    artifact: str,
    kind: str,
    digest: str,
    metadata: Mapping[str, Any],
    row_count: int,
    meaning: str,
    *,
    workspace: str | None,
    held: bool | None = None,
) -> None:
    current = log.read(
        "SELECT max(version) FROM definition WHERE family = 'artifact_type' AND name = ?", (kind,)
    )
    attrs = {
        "type": kind,
        "type_version": int(current[0][0]),
        "identity_kind": "content",
        "storage": "table",
        "meaning": meaning,
        "content_digest": digest,
        "shape": dict(metadata),
        "row_count": row_count,
    }
    if held is not None:
        attrs["held"] = held
    log.record(
        "artifact.produced",
        ("artifact", artifact),
        attrs,
        workspace=workspace,
        node="host",
    )
    log.flush()


def _record_result_count(log: HostLog, artifact: str, count: int) -> None:
    """Record the count once after the artifact's body is committed."""
    if not log.read("SELECT 1 FROM measure WHERE artifact_id = ?", (artifact,)):
        log.record(
            "result.recorded",
            ("artifact", artifact),
            {"measures": [["n_obs", 1, {}, count]]},
            workspace=_open["workspace"],
            node="host",
        )
        log.flush()


def kept_file_workspace(folder: Path) -> str | None:
    """The id of the registered workspace whose folder is `folder`, or None when the log holds
    none: its kept files (`kept_files`) are named by it."""
    with _array_reader() as (_root, read):
        rows = read(
            "SELECT workspace_id FROM workspace WHERE folder = ? AND status = 'active'",
            (str(folder.resolve()),),
        )
    return str(rows[0][0]) if len(rows) == 1 else None


def read_kept_file(workspace_id: str, name: str) -> bytes | None:
    """The workspace's kept file `name`, proved, or None when it is not kept."""
    import alphalattice.foundation.data_layer.kept_files as kept_files  # lazy: the array door

    with _array_reader() as (root, read):
        found: bytes | None = kept_files.read(root, read, workspace_id, name)
    return found


def kept_file_digest(workspace_id: str, name: str) -> str | None:
    """The digest of the workspace's kept file `name`, or None when it is not kept."""
    import alphalattice.foundation.data_layer.kept_files as kept_files  # lazy: the array door

    with _array_reader() as (_root, read):
        digest: str | None = kept_files.digest_of(read, workspace_id, name)
    return digest


def kept_file_names(workspace_id: str, folder: str, *, by_write: bool = False) -> list[str]:
    """The names of the workspace's kept files directly in `folder`: sorted, or `by_write`, the
    one written last at the end."""
    import alphalattice.foundation.data_layer.kept_files as kept_files  # lazy: the array door

    with _array_reader() as (_root, read):
        found: list[str] = kept_files.names(read, workspace_id, folder, by_write=by_write)
    return found


def kept_files_held(workspace_id: str) -> dict[str, str]:
    """Every kept file of the workspace: its name and its newest digest."""
    import alphalattice.foundation.data_layer.kept_files as kept_files  # lazy: the array door

    with _array_reader() as (_root, read):
        held_files: dict[str, str] = kept_files.held(read, workspace_id)
    return held_files


def keep_workspace_file(
    workspace_id: str, name: str, data: bytes, *, durable: bool = True, rewrite: bool = False
) -> str:
    """Keep `data` as the workspace's file `name` on the log this process holds, the one its kept
    files are read from (`_array_reader`): a Host's, else its data root's, opened (`held`).
    Returns its digest. `rewrite`: `kept_files.keep`'s."""
    import alphalattice.foundation.data_layer.kept_files as kept_files  # lazy: the array door

    log = _kept_file_log()
    digest: str = kept_files.keep(
        Path(_open["root"]), log, workspace_id, name, data, durable=durable, rewrite=rewrite
    )
    return digest


def remove_workspace_file(workspace_id: str, name: str) -> None:
    """Remove the workspace's kept file `name` on the log its kept files are read from."""
    import alphalattice.foundation.data_layer.kept_files as kept_files  # lazy: the array door

    kept_files.remove(_kept_file_log(), workspace_id, name)


def _kept_file_log() -> HostLog:
    """The log a kept file is written to: the one this process holds, whatever its root, as reads
    take it (`_array_reader`); else the data root's, opened and held."""
    log: HostLog | None = _open.get("log")
    return log if log is not None else held()


def carry_arrays(workspace: Path, copy: Path, *, root: Path | None = None) -> int:
    """Write the arrays the log (`root`'s when named) holds for `workspace` into its copy `copy`,
    in the old lane form; the Host that admits the copy (`move_carried`) moves them onto the log
    its process holds.
    Returns the arrays carried.

    contract: carried lanes retire when a copy carries its own segment of the log, as carried
    Tasks do (`carry_tasks`)."""
    import alphalattice.foundation.data_layer.arrays as arrays  # lazy: numpy, BLAKE3 and Parquet

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
    if not known:
        return 0
    workspace_id = str(known[0][0])
    count = int(arrays.carry(read, root, workspace_id, copy))
    rows = read(
        "SELECT a.content_digest, json_extract(a.shape, '$.layout'), b.content_digest,"
        " b.body_state, b.type_id = (SELECT definition_id FROM definition_name"
        " WHERE family = 'artifact_type' AND name = 'array'), held.artifact_id"
        " FROM live_ref l JOIN artifact a ON a.artifact_id = l.artifact_id"
        " JOIN definition_name d ON d.definition_id = a.type_id"
        " LEFT JOIN event_artifact u ON u.seq = a.produced_seq AND u.role = 'used'"
        " LEFT JOIN artifact b ON b.artifact_id = u.artifact_id"
        " LEFT JOIN live_ref held ON held.artifact_id = b.artifact_id"
        " AND held.root_kind = 'workspace' AND held.root_id = l.root_id"
        " WHERE l.root_kind = 'workspace' AND l.root_id = ?"
        " AND d.family = 'artifact_type' AND d.name = 'alpha_part'"
        " ORDER BY a.content_digest, json_extract(a.shape, '$.layout')",
        (workspace_id,),
    )
    target = copy / arrays.CARRIED / "alpha-parts.json"
    if rows or target.exists():
        keys: set[tuple[Any, Any]] = set()
        for row in rows:
            if (
                (row[0], row[1]) in keys
                or not row[0]
                or not row[1]
                or not row[2]
                or row[3] != "present"
                or row[4] != 1
                or row[5] is None
            ):
                raise DataError(
                    "an Alpha part relation cannot be carried", code="data_layer.part_invalid"
                )
            keys.add((row[0], row[1]))
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps([list(row[:3]) for row in rows], separators=(",", ":")).encode()
        staged = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        staged.write_bytes(payload)
        os.replace(staged, target)
    _carry_kept_files(read, workspace_id, copy / arrays.CARRIED)
    return count


def _carry_kept_files(read: Callable[..., Any], workspace_id: str, carried: Path) -> None:
    """The workspace's kept files' index into its copy's carried arrays, whose bodies `arrays.carry`
    wrote beside it, in the order of their newest writes; the copy's Host names them under its own
    id at admission."""
    import alphalattice.foundation.data_layer.kept_files as kept_files  # lazy: the array door

    files = kept_files.held(read, workspace_id)
    target = carried / kept_files.CARRIED
    if not files and not target.exists():
        return
    carried.mkdir(parents=True, exist_ok=True)
    staged = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    staged.write_bytes(
        json.dumps([[n, d] for n, d in files.items()], separators=(",", ":")).encode()
    )
    os.replace(staged, target)


def _import_kept_files(log: HostLog, folder: Path, workspace: str) -> None:
    """Name the kept files a copy carries under `workspace`, in their carried write order, once
    their carried bodies are kept. A name the log already has any event for under `workspace` is
    not named again: the index stays in the copy and is read at every start, so it applies once,
    and a write or removal since then stands. An index naming a body the copy did not carry is
    refused by name."""
    import alphalattice.foundation.data_layer.arrays as arrays  # lazy: numpy, BLAKE3 and Parquet
    import alphalattice.foundation.data_layer.kept_files as kept_files  # lazy: the array door

    index = folder / arrays.CARRIED / kept_files.CARRIED
    if index.is_symlink():
        raise DataError("the copy's kept-file index is a link", code="data_layer.kept_file_missing")
    if not index.is_file():
        return
    rows = json.loads(index.read_text(encoding="utf-8"))
    known = kept_files.known(log.read, workspace)
    for name, digest in rows:
        if name in known:
            continue
        if arrays.profile(log.read, digest) is None:
            raise DataError(
                f"the copy names its file {name} by a body it does not carry",
                code="data_layer.kept_file_missing",
            )
        kept_files.index(log, workspace, name, digest)


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
    opened = opened_log()
    if opened is None:
        return None
    log, workspace = opened
    if type_ == "trial" and more.get("phase") in {"completed", "failed", "cancelled"}:
        try:
            port = store()
            native = (
                {"status": "UNMEASURED"}
                if port is None
                else port.trial_agent_usage(
                    str(subject[1]), workspace_id=str(more.get("workspace", workspace))
                )
            )
        except Exception as error:
            logging.getLogger(__name__).warning(
                "agent_telemetry.trial_unmeasured: %s", type(error).__name__
            )
            native = {"status": "UNMEASURED"}
        attrs = dict(attrs or {}) | {"agent_telemetry": native}
    more.setdefault("workspace", workspace)
    return int(log.record(type_, subject, attrs, **more))


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
    with _lock:
        return (_open["log"], str(_open["workspace"])) if recording() else None


def receive_agent_telemetry(rows: list[dict[str, Any]]) -> int:
    """The Host's existing log lifetime alone may open the catalog's writer, lazily."""
    port = store()
    if port is None:
        raise DataError(
            "The Host has no writable catalog; start a Host with the data extra",
            code="agent_telemetry.host_unavailable",
        )
    return port.telemetry_batch(rows)


def agent_telemetry(host: str, session: str, *, workspace_id: str) -> list[dict[str, Any]] | None:
    """Read the serving Host's exact workspace/vendor/session; absent means unmeasured."""
    port = store()
    return None if port is None else port.agent_telemetry(host, session, workspace_id=workspace_id)


def unplaced_agent_telemetry(*, workspace_id: str) -> int | None:
    """Count this Host workspace's records without a conversation/session identity."""
    port = store()
    return None if port is None else port.unplaced_telemetry(workspace_id=workspace_id)


def read_experience(**filters: Any) -> dict[str, list[dict[str, Any]]] | None:
    """Read the open Host's experience and record the read; None if unavailable."""
    with _held_log() as opened:
        reading = _open.get("reading")
        if opened is None or reading is None:
            return None
        log, _, _ = opened
        log.flush()
        trials = reading.read("experience", **filters)
        notes = reading.read("goal_notes", **filters)
        record("experience.read", attrs={"count": len(trials)})
        log.flush()
        return {"trials": trials, "notes": notes}


@contextmanager
def _held_log() -> Iterator[tuple[HostLog, str | None, Path] | None]:
    """Borrow the log under its guard, with the workspace and root for that lifetime."""
    with _lock:
        log: HostLog | None = _open.get("log")
    if log is None:
        yield None
        return
    with log.reading():
        with _lock:
            workspace, root = _open.get("workspace"), _open.get("root")
            current = _open.get("log") is log and root is not None
        yield (log, workspace, Path(root)) if current else None


def start_log_reader() -> None:
    """Start the held Host's one reader before a Task needs its first view."""
    with _held_log() as opened:
        if opened is not None and (reader := _open.get("reading")) is not None:
            reader.start_ahead()


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
    render_in_background()
    return {"note_id": note_id, **json.loads(attrs)}


def render_in_background() -> None:
    """Render the notes awaiting their experience (`render_experiences`) off the caller's path."""
    threading.Thread(target=render_experiences, name="experience-render", daemon=True).start()


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
    with _held_log() as opened:
        if opened is None:
            raise ValueError("experience.history_unavailable")
        log, current_workspace, _ = opened
        rows = log.read(
            "SELECT workspace_id, goal_id FROM hypothesis_scope WHERE hypothesis_id = ?",
            (hypothesis,),
        )
        if not rows:
            raise ValueError("experience.hypothesis_unknown")
        workspace, goal = rows[0]
        if workspace != current_workspace:
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


def define(family: str, name: str, text: str, spec: Mapping[str, Any] | None = None) -> str | None:
    """Register a Host definition once and return its lineage, or None while recording defers."""
    key = {"family": family, "name": name, "version": 1}
    body = key | {"spec": dict(spec or {}), "status": "standard", "owner": "host", "text": text}
    exists = "SELECT 1 FROM definition WHERE family = ? AND name = ? AND version = 1"
    with _once:
        if not recording():
            return None
        with _open["log"].unit() as con:
            if con.execute(exists, (family, name)).fetchone() is None:
                record(
                    "definition.registered",
                    ("definition", f"{family}/{name}"),
                    key,
                    text=json.dumps(body, sort_keys=True),
                    node="host",
                )
            return project.require_lineage(con, family, name)


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
        define("universe", str(window["universe"]), project.UNIVERSE_TEXT)
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
    _record_condition(made, window, actor)
    return made


def _record_condition(selection: str, window: Mapping[str, Any], actor: tuple[str, str]) -> None:
    """Record the market's condition at a new selection's cutoff (reading.market_condition) as
    one `market_condition` artifact of that selection, its metrics as measures; nothing when the
    selection names no universe, and a warning, never a refusal, when world cannot be read, so a
    trial never waits on it."""
    reading = _open.get("reading")
    universe = (
        None
        if window.get("universe") is None
        else definition_lineage("universe", str(window["universe"]))
    )
    if reading is None or universe is None:
        return
    cutoff = int(window["cutoff_us"])
    try:
        with _held_log() as opened:
            if opened is None or reading is not _open.get("reading"):
                return
            values = reading.read("market_condition", universe=universe, cutoff_us=cutoff)
    except DataError as error:
        logging.getLogger(__name__).warning("market_condition.unread: %s", error)
        return
    kind = version("artifact_type", CONDITION_TYPE)  # the Host registers it at its open
    if kind is None:
        return
    produced, result = condition_events(selection, universe, cutoff, values, kind)
    artifact = ("artifact", uuid7())
    record("artifact.produced", artifact, produced, actor=actor)
    record("result.recorded", artifact, result, actor=actor)


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


def record_return_series(
    trial: str,
    months: Sequence[str],
    returns: Sequence[float],
    *,
    actor: tuple[str, str] = ("host", "host"),
) -> str | None:
    """Keep `trial`'s headline monthly return series with its profile, as one inline
    `return_series` artifact (return_series.encode's body); the class read's effective N reads
    the newest one a trial holds. Refused by name for a malformed series or a trial the log does
    not hold; None outside a Host."""
    return_series.encode(months, returns)  # refused by name before the log is read
    log = _open.get("log")
    if log is None or not recording():
        return None
    found = log.read("SELECT recipe_key FROM trial WHERE trial_id = ?", (trial,))
    if not found:
        raise DataError(f"no trial {trial!r} is in the log", code="data_layer.trial_unknown")
    kind = version("artifact_type", return_series.TYPE)  # the Host registers it at its open
    if kind is None:
        return None
    produced, body = return_series.produced(trial, found[0][0], months, returns, kind)
    artifact = uuid7()
    record("artifact.produced", ("artifact", artifact), produced, body=body, actor=actor)
    return artifact


def definition_lineage(family: str, name: str) -> str | None:
    """Resolve any held name to its stored id in the open Host log."""
    log = _open.get("log")
    return None if log is None else log.lineage(family, name)


def read_experience_class(
    *, trial: str, width: str = "method_version", cutoff_us: int | None = None
) -> dict[str, Any] | None:
    """Every trial of `trial`'s experiment class at `width`, as known at `cutoff_us`, with the
    class's spread and search size (`reading.experience_class`), read by the open Host as one
    gen_ai `search_memory` span over the experience store; None if unavailable. The query and
    the records stay off the span: the conventions make both opt-in."""
    return _search_memory(
        "experience_class",
        lambda read: len(read["members"]),
        trial=trial,
        width=width,
        cutoff_us=cutoff_us,
    )


def recall_experiences(
    *, query: str, workspace: str, cutoff_us: int, superseded: bool | None = False, limit: int = 5
) -> dict[str, Any] | None:
    """The experiences most like `query` in `workspace` as known at `cutoff_us`
    (`recall.recall`, one gen_ai.retrieval span), each hit with its text and, for each trial its
    experience is on, that trial's class as known then (`read_experience_class`); None while the
    history is not open."""
    with _lock:
        log, root = _open.get("log"), _open.get("root")
        if log is None or root is None or _open.get("reading") is None:
            return None
    # Flushed after `_lock` is let go: a registry write holds the log's guard and then asks
    # `held()` for `_lock`, so flushing under `_lock` would take the two locks in the other order.
    log.flush()
    found, trials = recall.recall(
        log,
        Path(root),
        query=query,
        workspace=workspace,
        cutoff_us=cutoff_us,
        superseded=superseded,
        limit=limit,
    )
    classes: dict[str, dict[str, Any] | None] = {}
    for named in dict.fromkeys(trial for held in trials.values() for trial in held):
        classes[named] = read_experience_class(trial=named, cutoff_us=cutoff_us)
    return {
        "score_type": found.score_type,
        "hits": [
            {
                "experience_id": item.hit.source_id,
                "version": int(item.hit.version),
                "score": item.hit.score,
                "text": item.text,
                "classes": {trial: classes[trial] for trial in trials[item.hit.source_id]},
            }
            for item in found.hits
        ],
    }


def index_references(
    *, workspace: str, revisions: Sequence[tuple[str, int]]
) -> recall.ReferenceIndex | None:
    """Chunk and embed the reference revisions `workspace` published (`recall.index_references`);
    None while the history is not open."""
    with _lock:
        log = _open.get("log")
        if log is None or _open.get("reading") is None:
            return None
    # Flushed after `_lock` is let go: a registry write holds the log's guard and then asks
    # `held()` for `_lock`, so flushing under `_lock` would take the two locks in the other order.
    log.flush()
    return recall.index_references(log, workspace=workspace, revisions=revisions)


def search_references(
    query: str,
    *,
    workspace: str,
    cutoff_us: int,
    ranges: tuple[tuple[str, int, int], ...],
    groups: tuple[tuple[str, ...], ...],
    depth: int | None,
    limit: int,
) -> Retrieved | None:
    """The reference chunks of `workspace` most like `query` (`recall.search_references`, one
    gen_ai.retrieval span); None while the history is not open."""
    with _lock:
        log, root = _open.get("log"), _open.get("root")
        if log is None or root is None or _open.get("reading") is None:
            return None
    # Flushed after `_lock` is let go: a registry write holds the log's guard and then asks
    # `held()` for `_lock`, so flushing under `_lock` would take the two locks in the other order.
    log.flush()
    return recall.search_references(
        log,
        Path(root),
        query,
        workspace=workspace,
        cutoff_us=cutoff_us,
        ranges=ranges,
        groups=groups,
        depth=depth,
        limit=limit,
    )


def read_experience_brief(
    *, goal: str, budget: int, cutoff_us: int | None = None
) -> dict[str, Any] | None:
    """What to know before new work on `goal` as known at `cutoff_us`, fitted to `budget`
    tokens (`reading.brief`), read by the open Host as one gen_ai `search_memory` span counting
    the items it took; None if unavailable."""
    return _search_memory(
        "brief",
        lambda read: sum(len(part["items"]) for part in read["parts"]),
        goal=goal,
        budget=budget,
        cutoff_us=cutoff_us,
    )


def _search_memory(
    name: str, count: Callable[[dict[str, Any]], int], **filters: Any
) -> dict[str, Any] | None:
    """The reader's `name` read, as one gen_ai `search_memory` client span over the experience
    store, its record count `count` of the answer; the query and the records stay off it, the
    conventions making both opt-in. None while the history is not open."""
    with _held_log() as opened:
        reading = _open.get("reading")
        if opened is None or reading is None:
            return None
        log, _, _ = opened
        log.flush()
        provider = _open.get("tracer")
        if provider is None:
            found: dict[str, Any] = reading.read(name, **filters)
            return found
        from opentelemetry.trace import SpanKind  # lazy: the SDK, once open

        attributes = {
            telemetry.GEN_AI_OPERATION_NAME: telemetry.GEN_AI_OPERATION_SEARCH_MEMORY,
            telemetry.GEN_AI_MEMORY_STORE_ID: "experience",
        }
        with provider.get_tracer("host").start_as_current_span(
            telemetry.GEN_AI_OPERATION_SEARCH_MEMORY, kind=SpanKind.CLIENT, attributes=attributes
        ) as span:
            found = reading.read(name, **filters)
            span.set_attribute(telemetry.GEN_AI_MEMORY_RECORD_COUNT, count(found))
        log.flush()
        return found


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

    from alphalattice.foundation.data_layer.spans import context_of  # lazy: the SDK, once open

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
    attrs = _attributes(attributes) | ({telemetry.LINKS: links} if links else {})
    attrs |= {} if caller else {telemetry.UNATTRIBUTED: True}
    if ended is not None:
        attrs[telemetry.EXIT_CODE] = ended[1]
    exporter.append(
        envelope(
            telemetry.SPAN_CLI_CALL,
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
        "started_us": wall_time_ns() // 1000,
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
    own = {telemetry.ALPHALATTICE_WORKSPACE_ID: workspace or _open["workspace"]}
    if subject is not None:
        own |= {
            telemetry.ALPHALATTICE_SUBJECT_KIND: subject[0],
            telemetry.ALPHALATTICE_SUBJECT_ID: subject[1],
        }
    if actor is not None:
        own |= {
            telemetry.ALPHALATTICE_ACTOR_KIND: actor[0],
            telemetry.ALPHALATTICE_ACTOR_ID: actor[1],
        }
    provider.get_logger("host").emit(
        event_name=name,
        body=text,
        attributes={**attributes, **own},
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
            f"SELECT {columns} FROM event WHERE type = '{telemetry.SPAN_CLI_CALL}'"
            f" AND workspace_id = ? AND (json_extract(attrs, '$.\"{telemetry.GOAL_ID}\"') = ?"
            f" OR json_extract(attrs, '$.\"{telemetry.GOAL_ID}\"') IS NULL"
            f" AND json_extract(attrs, '$.\"{telemetry.GEN_AI_CONVERSATION_ID}\"') IN"
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
                f"SELECT {columns} FROM event WHERE workspace_id = ?"
                f" AND trace_id IN ({marks}) AND phase IS NOT NULL"
                f" AND type <> '{telemetry.SPAN_CLI_CALL}' ORDER BY seq",
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

    from alphalattice.foundation.data_layer.spans import carrier  # lazy: the SDK, once open

    return carrier() if trace.get_current_span().get_span_context().is_valid else {}


@contextmanager
def _retrieval_span(query: str) -> Iterator[None]:
    """One store retrieval's span; the query text rides on it only where content is admitted."""
    attributes = {telemetry.GEN_AI_OPERATION_NAME: telemetry.GEN_AI_OPERATION_RETRIEVAL}
    if _capture_message_content():
        attributes[telemetry.GEN_AI_RETRIEVAL_QUERY_TEXT] = query
    with step(telemetry.SPAN_GEN_AI_RETRIEVAL, {}, **attributes):
        yield


knowledge.RETRIEVAL_SPAN = _retrieval_span  # the store's searches run in the Host's span
