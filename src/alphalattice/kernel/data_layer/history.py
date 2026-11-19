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
import uuid
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from contextlib import closing, contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final
from urllib.parse import urlsplit

import alphalattice.kernel.data_layer.derived_files as derived_files
import alphalattice.kernel.data_layer.experiences as experiences
import alphalattice.kernel.data_layer.project as project
import alphalattice.kernel.data_layer.return_series as return_series
from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer.admit import secrets_in
from alphalattice.kernel.data_layer.breakdown import goal_trace as traced_goal
from alphalattice.kernel.data_layer.log import HostLog, attach, envelope, uuid7
from alphalattice.kernel.data_layer.reading import (
    CONDITION_METRICS,
    CONDITION_TYPE,
    LogReader,
    condition_events,
    reading_session,
)
from alphalattice.kernel.data_layer.root import (
    DATA_ROOT_ENV,
    create_data_root,
    data_root,
    seed_rows,
)
from alphalattice.kernel.shared_kernel.environment import offline

if TYPE_CHECKING:
    from alphalattice.kernel.data_layer.store import StorePort

OTLP_ENDPOINT_ENV: Final = "ALPHALATTICE_OTLP_ENDPOINT"
OTLP_CONTENT_ENV: Final = "ALPHALATTICE_OTLP_INCLUDE_CONTENT"
EVIDENCE_TYPE: Final = "research_evidence"
"""The artifact type of a trial's execution evidence, registered by the Host as its own."""


def alpha_array_door() -> bool:
    """Select Alpha's array reader; the old reader remains available until row U."""
    mode = os.environ.get("ALPHALATTICE_ARRAY_STORE", "door")
    if mode not in {"door", "legacy"}:
        raise DataError(
            "the Alpha array store is invalid", code="alpha_research.array_store_invalid"
        )
    return mode == "door"


_EVIDENCE_TEXT: Final = "What one research trial's Desk run did: its program, inputs and artifacts."
_RETURN_SERIES_TEXT: Final = "A trial's headline monthly return series, kept with its profile."
_CONDITION_TEXT: Final = "A market's condition at a selection's cutoff, as registered metrics."
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
    "search_memory": (
        ["gen_ai.operation.name", "gen_ai.memory.store.id"],
        "none",
        "One read of an agent memory store, as the gen_ai conventions name it.",
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
_catalog_surfaces: dict[Path, Any] = {}
_catalog_lock = threading.Lock()
_reads: dict[str, set[str]] = {}
"""The file partitions (by table, their digests) the Host read since their last mark."""


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
    a worker thread against a writer's unit. Once open, outside both, the Host's start collects the
    bodies nothing holds (`StorePort.collect_unheld`); a restore's own opening never does."""
    port: StorePort | None = _open.get("store")
    log: HostLog | None = _open.get("log")
    if port is not None or log is None:
        return port
    opened = None
    with log.reading(), _lock:
        port = _open.get("store")
        if (
            port is None
            and _open.get("log") is log
            and recording()
            and _open.get("reading") is not None
        ):
            from alphalattice.kernel.data_layer.store import StorePort  # lazy: the data extra

            port = opened = _open["store"] = StorePort.open(Path(_open["root"]), _open["log"])
    if opened is not None:  # the collector takes the backup's lock first, so never under the guard
        opened.started["unheld"] = opened.collect_unheld()
    return port


@contextmanager
def catalog_session() -> Iterator[Any | None]:
    """A cursor on this process's attached catalog, without attaching its SQLite log."""
    port = store()
    if port is not None:
        cursor = port.lake.cursor()
    else:
        root = data_root()
        with _catalog_lock:
            surface = _catalog_surfaces.get(root)
            if surface is None and (root / "machine" / "catalog.sqlite").is_file():
                surface = _catalog_surfaces[root] = reading_session(root, log=False)
        cursor = None if surface is None else surface.cursor()
    try:
        yield cursor
    finally:
        if cursor is not None:
            cursor.close()


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
        _import_alpha_parts(where, log, folder, _open["workspace"])
        spec: dict[str, Any] = {
            "storage": "none",  # the evidence stays in the workspace's store until row 8a
            "table": None,
            "measures": [],
            "keep": "while_active",
        }
        define("artifact_type", EVIDENCE_TYPE, _EVIDENCE_TEXT, spec)
        kept = {"storage": "inline", "table": None, "measures": [], "keep": "forever"}
        define("artifact_type", return_series.TYPE, _RETURN_SERIES_TEXT, kept)
        for metric, (metric_spec, words) in CONDITION_METRICS.items():
            define("metric", metric, words, metric_spec)
        condition = {"storage": "none", "table": None, "keep": "forever"}
        condition["measures"] = sorted(CONDITION_METRICS)
        define("artifact_type", CONDITION_TYPE, _CONDITION_TEXT, condition)
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
    if log is not None:
        _mark_reads(log)
    with _read_lock:
        if reading is not None:
            reading.close()
        outcome_reading = _open.pop("outcome_reading", None)
        if outcome_reading is not None:
            outcome_reading.close()
        benchmark_reading = _open.pop("benchmark_reading", None)
        if benchmark_reading is not None:
            benchmark_reading.close()
        attached = _attached.pop("log", None)
        if attached is not None:
            attached[1].close()
    if log is not None:
        log.close()


def _register_result_definitions(
    log: HostLog,
    kind: str,
    measures: set[str],
) -> None:
    """Install the seed's result definitions in a data root created before they shipped."""
    needed = {("artifact_type", kind), *(("metric", name) for name in measures)}
    present = log.read(
        "SELECT family, name FROM definition WHERE family IN ('artifact_type', 'metric')"
        " AND version = 1"
    )
    missing = needed.difference((row[0], row[1]) for row in present)
    if not missing:
        return
    for family, name, version, _, _, _, spec, text, *_ in seed_rows():
        if (family, name) in missing:
            key = {"family": family, "name": name, "version": version}
            body = key | {
                "spec": json.loads(spec),
                "status": "standard",
                "owner": "host",
                "text": text,
            }
            log.record(
                "definition.registered",
                ("definition", f"{family}/{name}"),
                key,
                text=json.dumps(body, sort_keys=True),
                node="host",
            )


type AlphaDevelopmentMeasures = list[tuple[str, int, dict[str, str], float | int | None]]


def record_alpha_development_summary(report_hash: str, measures: AlphaDevelopmentMeasures) -> None:
    """Record the measures of a sealed Alpha report under its own durable profile."""
    opened = opened_log()
    if opened is not None:
        log, workspace = opened
        with log.unit():
            _record_alpha_summary(log, workspace, report_hash, measures)
        log.flush()


def _record_alpha_summary(
    log: HostLog,
    workspace: str,
    report_hash: str,
    measures: AlphaDevelopmentMeasures,
) -> None:
    kind = "alpha_development_summary"
    if log.read(
        "SELECT 1 FROM artifact WHERE type_id = ("
        " SELECT definition_id FROM definition_name WHERE family = 'artifact_type' AND name = ?"
        ") AND content_digest = ?",
        (kind, report_hash),
    ):
        return
    _register_result_definitions(log, kind, {name for name, *_ in measures})
    artifact = ("artifact", uuid7())
    log.record(
        "artifact.produced",
        artifact,
        {
            "type": kind,
            "type_version": 1,
            "identity_kind": "content",
            "storage": "none",
            "meaning": "The measures of a sealed Alpha candidate development report.",
            "content_digest": report_hash,
        },
        workspace=workspace,
        node="host",
    )
    log.record(
        "result.recorded", artifact, {"measures": measures}, workspace=workspace, node="host"
    )


_ALPHA_CHUNK = "alpha_development_chunk"
_ALPHA_LANES = ("target", "economic_return", "score")
_ALPHA_CHUNK_PROFILES = (
    "SELECT content_digest, artifact_id, body_state, shape, row_count FROM artifact"
    " WHERE type_id = (SELECT definition_id FROM definition_name"
    " WHERE family = 'artifact_type' AND name = ?) AND content_digest IN ({slots})"
)
_ALPHA_BREAKDOWN = """
    WITH named AS (
        SELECT artifact_id, metric_id, metric_version, value,
               json_transform(dims, '{"row_index":"BIGINT","listing_id":"VARCHAR",
                   "formation_session":"DATE","feature_available":"BOOLEAN",
                   "outcome_available":"BOOLEAN","feature_row_hash":"VARCHAR",
                   "causal_outcome_row_hash":"VARCHAR","availability":"VARCHAR"}') AS d
        FROM lake.results.breakdown
        WHERE artifact_id BETWEEN ? AND ? AND list_contains(?::VARCHAR[], artifact_id)
    ), lane AS (SELECT ? AS target, ? AS economic_return, ? AS score)
    SELECT artifact_id, d.row_index AS row_index,
           max(d.listing_id) AS listing_id,
           max(d.formation_session) AS formation_session,
           bool_or(d.feature_available) AS feature_available,
           bool_or(d.outcome_available) AS outcome_available,
           max(d.feature_row_hash) AS feature_row_hash,
           max(d.causal_outcome_row_hash) AS causal_outcome_row_hash,
           max(d.availability) AS availability,
           max(value) FILTER (WHERE metric_id = lane.target) AS target,
           max(value) FILTER (WHERE metric_id = lane.economic_return)
               AS economic_return,
           max(value) FILTER (WHERE metric_id = lane.score) AS score,
           count(*) AS n_lanes,
           bool_and(metric_version = 1) AS v1,
           bit_or(CASE metric_id
               WHEN lane.target THEN 1
               WHEN lane.economic_return THEN 2
               WHEN lane.score THEN 4
               ELSE 8
           END) AS lanes
    FROM named CROSS JOIN lane
    GROUP BY artifact_id, d.row_index ORDER BY artifact_id, row_index
"""


def _alpha_chunk_profiles(
    read: Callable[..., Any],
    digests: Iterable[str],
) -> dict[str, Any]:
    named = tuple(dict.fromkeys(digests))
    if not named:
        return {}
    slots = ", ".join("?" for _ in named)
    rows = read(
        _ALPHA_CHUNK_PROFILES.format(slots=slots),
        (_ALPHA_CHUNK, *named),
    )
    profiles = {str(row[0]): row for row in rows}
    if len(profiles) != len(rows):
        raise ValueError("alpha_research.development_breakdown_profile_repeated")
    return profiles


def _alpha_lane_ids(read: Callable[..., Any]) -> dict[str, str]:
    names = tuple(f"alpha_development_{lane}" for lane in _ALPHA_LANES)
    registered = dict(
        read(
            "SELECT name, definition_id FROM definition_name"
            " WHERE family = 'metric' AND name IN (?, ?, ?)",
            names,
        )
    )
    return {lane: registered[name] for lane, name in zip(_ALPHA_LANES, names, strict=True)}


def _alpha_breakdown_rows(
    chunks: list[tuple[str, Any]], lane_ids: Mapping[str, str], lake: Any
) -> Any:
    """Encode a stage's dimensions and numeric lanes in one columnar projection."""
    import pyarrow as pa  # lazy: the data extra is optional
    import pyarrow.compute as pc  # lazy: the data extra is optional

    types = {
        "availability": pa.string(),
        "causal_outcome_row_hash": pa.string(),
        "feature_available": pa.bool_(),
        "feature_row_hash": pa.string(),
        "formation_session": pa.date32(),
        "listing_id": pa.string(),
        "outcome_available": pa.bool_(),
        "row_index": pa.int64(),
    }
    batches = []
    for artifact, table in chunks:
        count = table.num_rows
        nulls = {name: pa.nulls(count, type_) for name, type_ in types.items()}
        dims = {
            name: pc.cast(table[name], type_).combine_chunks()
            if name in table.column_names
            else nulls[name]
            for name, type_ in types.items()
        }
        anchor = nulls | {"row_index": dims["row_index"]}
        for name, metric in lane_ids.items():
            if name not in table.column_names:
                continue
            fields = anchor if name == "economic_return" else dims
            batches.append(
                pa.table(
                    {
                        "artifact_id": pa.repeat(pa.scalar(artifact), count),
                        "metric_id": pa.repeat(pa.scalar(metric), count),
                        "metric_version": pa.repeat(pa.scalar(1, pa.int32()), count),
                        "dimensions": pa.StructArray.from_arrays(
                            list(fields.values()), names=list(fields)
                        ),
                        "value": pc.cast(table[name], pa.float64()),
                    }
                )
            )
    with closing(lake.cursor()) as cursor:
        cursor.register("alpha_lane", pa.concat_tables(batches))
        try:
            return cursor.sql(
                "SELECT artifact_id, metric_id, metric_version,"
                " json_merge_patch('{}', to_json(dimensions))::VARCHAR AS dims, value"
                " FROM alpha_lane"
            ).to_arrow_table()
        finally:
            cursor.unregister("alpha_lane")


def write_alpha_development_chunks(
    chunks: Mapping[str, tuple[str, Any]],
    *,
    summaries: Mapping[str, AlphaDevelopmentMeasures] | None = None,
) -> None:
    """Record profiles and measures, then admit the stage's new detail in one snapshot."""
    if not (chunks or summaries) or not recording():
        return
    from alphalattice.kernel.data_layer.store import stage  # lazy: the data extra is optional

    with _once:
        port = store()
        if port is None:
            raise DataError(
                "the Host's data root is unavailable for Alpha development evidence",
                code="data_layer.history_unavailable",
            )
        log, workspace = port.log, _open["workspace"]
        profiles = _alpha_chunk_profiles(log.read, chunks)
        artifacts: dict[str, str] = {}
        with log.unit():
            for report_hash, measures in (summaries or {}).items():
                _record_alpha_summary(log, workspace, report_hash, measures)
            _register_result_definitions(
                log, _ALPHA_CHUNK, {f"alpha_development_{name}" for name in _ALPHA_LANES}
            )
            for digest, (seal, table) in chunks.items():
                shape = {
                    "arrow_schema": table.schema.serialize().to_pybytes().hex(),
                    "metadata_hash": seal,
                }
                profile = profiles.get(digest)
                if profile is not None:
                    if profile[2] != "present" or json.loads(profile[3]) != shape:
                        raise ValueError(
                            f"alpha_research.development_breakdown_profile_invalid: {digest}"
                        )
                    if profile[4] != table.num_rows:
                        raise ValueError(
                            f"alpha_research.development_breakdown_count_changed: {digest}"
                        )
                    artifacts[digest] = str(profile[1])
                    continue
                artifact = artifacts[digest] = uuid7()
                log.record(
                    "artifact.produced",
                    ("artifact", artifact),
                    {
                        "type": _ALPHA_CHUNK,
                        "type_version": 1,
                        "identity_kind": "content",
                        "storage": "table",
                        "meaning": "Sealed Alpha development evidence rows.",
                        "content_digest": digest,
                        "row_count": table.num_rows,
                        "shape": shape,
                        "held": True,
                    },
                    workspace=workspace,
                    node="host",
                )
        log.flush()
        ids = [artifacts[digest] for digest in profiles]
        existing = (
            dict(
                port.lake.execute(
                    "SELECT artifact_id, count(*) FROM lake.results.breakdown"
                    " WHERE list_contains(?::VARCHAR[], artifact_id) GROUP BY artifact_id",
                    (ids,),
                ).fetchall()
            )
            if ids
            else {}
        )
        pending = []
        for digest, (_, table) in chunks.items():
            artifact = artifacts[digest]
            held_rows = existing.get(artifact)
            if held_rows is None:
                pending.append((artifact, table))
            elif held_rows != table.num_rows * (1 + ("economic_return" in table.column_names)):
                raise ValueError(f"alpha_research.development_breakdown_count_changed: {digest}")
        if pending:
            rows = _alpha_breakdown_rows(pending, _alpha_lane_ids(log.read), port.lake)
            with stage(port.root, "breakdown", None, rows) as staged:
                port.commit(staged)


def read_alpha_development_chunks(digests: Iterable[str]) -> dict[str, Any]:
    """Read one projected breakdown for the supplied sealed chunks."""
    import pyarrow as pa  # lazy: the data extra is optional
    import pyarrow.compute as pc  # lazy: the data extra is optional

    named = tuple(dict.fromkeys(digests))
    if not named:
        return {}
    log = _open.get("log")
    if log is None:
        with closing(attach(data_root())) as reader:
            read = _query(reader)
            profiles = _alpha_chunk_profiles(read, named)
            lane_ids = _alpha_lane_ids(read) if profiles else {}
    else:
        profiles = _alpha_chunk_profiles(log.read, named)
        lane_ids = _alpha_lane_ids(log.read) if profiles else {}
    for digest in named:
        if digest not in profiles or profiles[digest][2] != "present":
            raise FileNotFoundError(f"Alpha development breakdown {digest}")
    with catalog_session() as cursor:
        if cursor is None:
            raise DataError(
                "the Alpha evidence catalog is unavailable", code="data_layer.history_unavailable"
            )
        ids = sorted(str(profile[1]) for profile in profiles.values())
        params = [ids[0], ids[-1], ids, *(lane_ids[name] for name in _ALPHA_LANES)]
        rows = cursor.sql(_ALPHA_BREAKDOWN, params=params).to_arrow_table()
    result = {}
    for digest, (_, artifact, _, raw_shape, expected) in profiles.items():
        shape = json.loads(raw_shape)
        schema = pa.ipc.read_schema(pa.BufferReader(bytes.fromhex(shape["arrow_schema"])))
        selected = rows.filter(pc.equal(rows["artifact_id"], artifact))
        has_returns = "economic_return" in schema.names
        lanes = 4 if "score" in schema.names else 3 if has_returns else 1
        if (
            selected.num_rows != expected
            or not pc.all(selected["v1"]).as_py()
            or not pc.all(pc.equal(selected["n_lanes"], 1 + has_returns)).as_py()
            or not pc.all(pc.equal(selected["lanes"], lanes)).as_py()
        ):
            raise ValueError(f"alpha_research.development_breakdown_count_changed: {digest}")
        result[digest] = (
            selected.select(schema.names)
            .cast(schema)
            .replace_schema_metadata(
                {b"alphalattice.metadata_hash": shape["metadata_hash"].encode()}
            )
        )
    return result


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
                "held": _open["workspace"] is not None,  # its workspace keeps it (governance)
            }
            log.record(
                "artifact.produced",
                ("artifact", artifact),
                attrs,
                workspace=_open["workspace"],
                node="host",
            )
            log.flush()
        elif _open["workspace"] is not None and not log.read(
            "SELECT 1 FROM live_ref WHERE artifact_id = ? AND root_kind = 'workspace'"
            " AND root_id = ?",
            (artifact, _open["workspace"]),
        ):  # another workspace's equal outcome: this one keeps it too
            log.record(
                "artifact.held",
                ("artifact", artifact),
                {},
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
        _record_result_count(log, artifact, table.num_rows)
        return artifact


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
        held = derived_files.found(log.read, Path(root), table, digest)
        if held is not None:
            with _read_lock:  # its last read, marked in one batch (_mark_reads)
                _reads.setdefault(table, set()).add(digest)
        return held
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


def root_derived_files(
    table: str,
    digests: Collection[str],
    *,
    unread_for: timedelta,
    verified: Callable[[str], bool],
) -> dict[str, int] | None:
    """In the Host, make the file partitions with these digests its workspace's only roots of
    `table`, evict what has gone unread for `unread_for` where `verified` holds its source, and
    collect what no workspace roots (StorePort.root_files), the reads noted marked first. None
    outside a Host, whose roots wait for the Host's next call; nothing it keeps is lost by that."""
    port = store()
    if port is None or not recording():
        return None
    _mark_reads(_open["log"])
    return port.root_files(
        table, str(_open["workspace"]), digests, unread_for=unread_for, verified=verified
    )


def _mark_reads(log: HostLog) -> None:
    """Mark the file partitions the Host read since the last call as read now, in its log."""
    with _read_lock:
        reads = dict(_reads)
        _reads.clear()
    for table, digests in reads.items():
        log.read_cached(table, sorted(digests))


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


def _portfolio_result_profile(read: Callable[..., Any], kind: str, digest: str) -> Any:
    rows = read(
        "SELECT a.artifact_id, a.body_state, a.shape FROM artifact a JOIN definition d"
        " ON d.definition_id = a.type_id AND d.version = a.type_version"
        " WHERE d.family = 'artifact_type' AND d.name = ?"
        " AND a.content_digest = ?",
        (kind, digest),
    )
    if len(rows) > 1:
        raise DataError("duplicate benchmark profile", code="data_layer.benchmark_invalid")
    return rows[0] if rows else None


def require_portfolio_result_store() -> None:
    """Refuse Portfolio publication before any artifact when the Host has no results writer."""
    if holding() is None or store() is None:
        raise DataError("Host data root is unavailable", code="data_layer.history_unavailable")


def write_portfolio_result(
    kind: str,
    digest: str,
    metadata: Mapping[str, Any],
    table: Any,
    *,
    result_table: str,
    upgrade_owner: tuple[Any, Path] | None = None,
) -> str:
    """Admit one Portfolio artifact's rows and retain its lineage in its profile."""
    import pyarrow as pa  # lazy: only catalog result admission needs the data extra

    from alphalattice.kernel.data_layer.store import stage  # lazy: the data extra is optional

    if result_table not in {"series", "holding"}:
        raise ValueError("portfolio result table is invalid")
    selected, order = (
        ("name, session_date, value", "name, session_date")
        if result_table == "series"
        else ("as_of, listing_id, weight, shares", "as_of, listing_id")
    )

    with _once:
        port = store()  # opening the log precedes the history lock
        with _lock:
            log, root = _open.get("log"), _open.get("root")
            if log is None or root is None or port is None:
                raise DataError(
                    "Host data root is unavailable", code="data_layer.history_unavailable"
                )
            if upgrade_owner is not None and (
                log is not upgrade_owner[0] or root.resolve() != upgrade_owner[1].resolve()
            ):
                raise DataError("upgrade owns a different data root", code="upgrade.refused")
        if not log.read(
            "SELECT 1 FROM definition WHERE family = 'artifact_type' AND name = ?",
            (kind,),
        ):
            key = {"family": "artifact_type", "name": kind, "version": 1}
            body = key | {
                "spec": {
                    "storage": "table",
                    "table": "results." + result_table,
                    "measures": ["n_obs"],
                    "keep": "while_active",
                },
                "status": "standard",
                "owner": "host",
                "text": "A sealed Portfolio series artifact.",
            }
            log.record(
                "definition.registered",
                ("definition", "artifact_type/" + kind),
                key,
                text=json.dumps(body, sort_keys=True),
                node="host",
            )
        profile = _portfolio_result_profile(log.read, kind, digest)
        artifact = profile[0] if profile else uuid7()
        if profile and (profile[1] != "present" or json.loads(profile[2]) != dict(metadata)):
            raise DataError("benchmark profile differs", code="data_layer.benchmark_invalid")
        if not profile:
            current = log.read(
                "SELECT max(version) FROM definition WHERE family = 'artifact_type' AND name = ?",
                (kind,),
            )
            log.record(
                "artifact.produced",
                ("artifact", artifact),
                {
                    "type": kind,
                    "type_version": int(current[0][0]),
                    "identity_kind": "content",
                    "storage": "table",
                    "meaning": "The sealed series of a Portfolio artifact.",
                    "content_digest": digest,
                    "shape": dict(metadata),
                    "row_count": table.num_rows,
                },
                workspace=_open["workspace"],
                node="host",
            )
            log.flush()
        count = port.lake.execute(
            f"SELECT count(*) FROM lake.results.{result_table} WHERE artifact_id = ?", (artifact,)
        ).fetchone()[0]
        if count == 0:
            with stage(
                root,
                result_table,
                None,
                table.append_column("artifact_id", pa.repeat(pa.scalar(artifact), table.num_rows)),
            ) as staged:
                port.commit(staged)
        elif count != table.num_rows:
            raise DataError("benchmark body is incomplete", code="data_layer.benchmark_invalid")
        else:
            held = (
                port.lake.execute(
                    f"SELECT {selected} FROM lake.results.{result_table}"
                    f" WHERE artifact_id = ? ORDER BY {order}",
                    (artifact,),
                )
                .to_arrow_table()
                .cast(table.schema)
            )
            if not held.equals(table):
                raise DataError(
                    "benchmark identity has different values", code="data_layer.benchmark_invalid"
                )
        _record_result_count(log, artifact, table.num_rows)
        return str(artifact)


def read_portfolio_result(
    kind: str, digest: str, *, result_table: str
) -> tuple[dict[str, Any], Any]:
    """Read a registered Portfolio profile and its catalog rows."""
    if result_table not in {"series", "holding"}:
        raise ValueError("portfolio result table is invalid")
    with _read_lock:
        profile = _portfolio_read_profile(kind, digest)
        connection, owned = _portfolio_reader()
        try:
            artifact = str(uuid.UUID(str(profile[0])))
            table = connection.sql(
                f"EXECUTE portfolio_{result_table}('{artifact}')"
            ).to_arrow_table()
        finally:
            if owned:
                connection.close()
    return json.loads(profile[2]), table


def _portfolio_read_profile(kind: str, digest: str) -> Any:
    log = _open.get("log")
    if log is not None:
        profile = _portfolio_result_profile(log.read, kind, digest)
    else:
        with closing(attach(_open.get("root") or data_root())) as reader:
            profile = _portfolio_result_profile(
                lambda sql, args: reader.execute(sql, args).fetchall(), kind, digest
            )
    if profile is None or profile[1] != "present":
        raise DataError(
            f"portfolio result {digest} is not in the catalog", code="data_layer.benchmark_missing"
        )
    return profile


def _portfolio_reader() -> tuple[Any, bool]:
    """Prepare point reads once per held connection; its owner closes it."""
    owned = not recording()
    connection = _open.get("benchmark_reading")
    if connection is None:
        connection = reading_session(_open.get("root") or data_root(), log=False)
        if connection is None:
            raise DataError("portfolio catalog unavailable", code="data_layer.history_unavailable")
        for statement in (
            "PREPARE portfolio_series AS SELECT name, session_date, value FROM lake.results.series"
            " WHERE artifact_id = ? ORDER BY name, session_date",
            "PREPARE portfolio_holding AS SELECT as_of, listing_id, weight, shares"
            " FROM lake.results.holding WHERE artifact_id = ? ORDER BY as_of, listing_id",
            "PREPARE portfolio_pair AS SELECT 'series' AS kind, name, session_date, value,"
            " NULL::DATE AS as_of, NULL::VARCHAR AS listing_id, NULL::DOUBLE AS weight,"
            " NULL::DOUBLE AS shares FROM lake.results.series WHERE artifact_id = ? UNION ALL"
            " SELECT 'holding', NULL, NULL, NULL, as_of,"
            " CAST(listing_id AS VARCHAR), weight, shares"
            " FROM lake.results.holding WHERE artifact_id = ?",
        ):
            connection.execute(statement)
        if not owned:
            _open["benchmark_reading"] = connection
    return connection, owned


def read_portfolio_pair(series_digest: str, holding_digest: str) -> tuple[Any, Any]:
    """Read a ledger's series and its report's holdings in one catalog query."""
    import pyarrow.compute as pc  # lazy: only catalog result reads need the data extra

    with _read_lock:
        series = _portfolio_read_profile("portfolio_execution_series", series_digest)
        holding = _portfolio_read_profile("portfolio_report_holding", holding_digest)
        connection, owned = _portfolio_reader()
        try:
            series_id, holding_id = str(uuid.UUID(str(series[0]))), str(uuid.UUID(str(holding[0])))
            table = connection.sql(
                f"EXECUTE portfolio_pair('{series_id}', '{holding_id}')"
            ).to_arrow_table()
        finally:
            if owned:
                connection.close()
    first = table.filter(pc.equal(table["kind"], "series")).select(
        ["name", "session_date", "value"]
    )
    second = table.filter(pc.equal(table["kind"], "holding")).select(
        ["as_of", "listing_id", "weight", "shares"]
    )
    return (
        json.loads(series[2]),
        first.sort_by([("name", "ascending"), ("session_date", "ascending")]),
    ), (
        json.loads(holding[2]),
        second.sort_by([("as_of", "ascending"), ("listing_id", "ascending")]),
    )


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


def array_body_path(content_digest: str) -> Path:
    """The array owner's body path for a sealed-source cache; read through the door."""
    import alphalattice.kernel.data_layer.arrays as arrays  # lazy: numpy, BLAKE3 and Parquet

    with _array_reader() as (root, read):
        known = arrays.profile(read, content_digest)
        if known is None or known["body_state"] != "present":
            raise DataError(f"no array is named {content_digest}", code="data_layer.array_missing")
        return root / "blobs" / known["artifact_id"]


def _alpha_part_link(read: Callable[..., Any], content_hash: str, layout: str) -> Any:
    """The checked relation and array profile, read together from the log."""
    rows = read(
        "SELECT a.artifact_id, u.artifact_id, b.content_digest, b.body_state,"
        " b.type_id = (SELECT definition_id FROM definition_name"
        " WHERE family = 'artifact_type' AND name = 'array'), b.shape"
        " FROM artifact a JOIN definition_name d ON d.definition_id = a.type_id"
        " LEFT JOIN event_artifact u ON u.seq = a.produced_seq AND u.role = 'used'"
        " LEFT JOIN artifact b ON b.artifact_id = u.artifact_id"
        " WHERE d.family = 'artifact_type' AND d.name = 'alpha_part'"
        " AND a.content_digest = ? AND json_extract(a.shape, '$.layout') = ?",
        (content_hash, layout),
    )
    if not rows:
        return None
    if (
        len(rows) != 1
        or rows[0][1] is None
        or rows[0][2] is None
        or rows[0][3] != "present"
        or rows[0][4] != 1
        or rows[0][5] is None
    ):
        raise DataError(
            "the Alpha part's array relation is invalid", code="data_layer.part_invalid"
        )
    return rows[0]


def _alpha_part_body(read: Callable[..., Any], content_hash: str, layout: str) -> str | None:
    link = _alpha_part_link(read, content_hash, layout)
    return None if link is None else str(link[2])


def keep_alpha_part(
    content_hash: str,
    layout: str,
    payload: bytes,
    *,
    capacity: Callable[[int], None] | None = None,
    durable: bool = True,
    upgrade_owner: tuple[HostLog, Path, str | None] | None = None,
) -> str:
    """Keep a legacy-format body and relate it to its unchanged content and layout identities."""
    import alphalattice.kernel.data_layer.arrays as arrays  # lazy: numpy, BLAKE3 and Parquet

    with _once:
        if upgrade_owner is not None:
            log, root, workspace = upgrade_owner
        else:
            with _held_log() as found:
                current = found
            log, workspace, root = current if current is not None else (held(), None, data_root())
        link = _alpha_part_link(log.read, content_hash, layout)
        if link is not None:
            existing = str(link[2])
            # The first body for this semantic identity remains authoritative and is held here.
            body = arrays.read_packed(root, log.read, existing)
            if (
                arrays.keep_packed(
                    root, log, body, workspace=workspace, durable=False, capacity=capacity
                )
                != existing
            ):
                raise DataError("the Alpha part body changed", code="data_layer.part_invalid")
            arrays._hold(log, str(link[0]), workspace)
            if durable:
                log.flush()
            return existing
        described = arrays.describe_packed(payload)
        digest = arrays._keep(root, log, described, workspace, False, capacity)
        name = "alpha_part"
        versions = log.read(
            "SELECT version FROM definition_current WHERE family = 'artifact_type' AND name = ?",
            (name,),
        )
        if not versions:
            key = {"family": "artifact_type", "name": name, "version": 1}
            definition = key | {
                "spec": {"storage": "none", "table": None, "measures": [], "keep": "while_active"},
                "status": "standard",
                "owner": "host",
                "text": "An Alpha part's content and layout identities and the array it uses.",
            }
            log.record(
                "definition.registered",
                ("definition", "artifact_type/" + name),
                key,
                text=json.dumps(definition, sort_keys=True),
                node="host",
            )
            version = 1
        else:
            version = int(versions[0][0])
        log.record(
            "artifact.produced",
            ("artifact", uuid7()),
            {
                "type": name,
                "type_version": version,
                "identity_kind": "content",
                "storage": "none",
                "meaning": "An Alpha part's content and layout identities.",
                "content_digest": content_hash,
                "shape": {"layout": layout},
                "used": [described.artifact_id],
                "held": workspace is not None,
            },
            workspace=workspace,
            node="host",
        )
        if durable:
            log.flush()
        return digest


def read_alpha_part(content_hash: str, layout: str) -> bytes:
    """Resolve the typed Alpha part relation and read its verified packed body."""
    import alphalattice.kernel.data_layer.arrays as arrays  # lazy: numpy, BLAKE3 and Parquet

    with _array_reader() as (root, read):
        link = _alpha_part_link(read, content_hash, layout)
        if link is None:
            raise DataError("the Alpha part is missing", code="data_layer.part_missing")
        digest, artifact, shape = str(link[2]), str(link[1]), link[5]

        def profile(_sql: str, args: tuple[Any, ...]) -> list[Any]:
            if args != (digest, arrays.TYPE):
                raise DataError("the Alpha array profile changed", code="data_layer.part_invalid")
            return [(artifact, "present", shape)]

        return arrays.read_packed(root, profile, digest, packed_only=True)


def alpha_part_body_digest(content_hash: str, layout: str) -> str | None:
    """The packed-body digest linked to a content and layout pair, if upgraded."""
    with _array_reader() as (_root, read):
        return _alpha_part_body(read, content_hash, layout)


def alpha_part_body_path(content_hash: str, layout: str) -> Path:
    """The linked array's source path for the existing sealed-source verifier."""
    with _array_reader() as (root, read):
        link = _alpha_part_link(read, content_hash, layout)
        if link is None:
            raise DataError("the Alpha part is missing", code="data_layer.part_missing")
        return root / "blobs" / str(link[1])


def array_held_by(content_hash: str, workspace: str | None) -> bool:
    """Whether an array is present and, when named, held by this workspace."""
    import alphalattice.kernel.data_layer.arrays as arrays  # lazy: numpy, BLAKE3 and Parquet

    with _array_reader() as (_root, read):
        profile = arrays.profile(read, content_hash)
        if profile is None or profile["body_state"] != "present":
            return False
        return workspace is None or bool(
            read(
                "SELECT 1 FROM live_ref WHERE artifact_id = ? AND root_kind = 'workspace'"
                " AND root_id = ?",
                (profile["artifact_id"], workspace),
            )
        )


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
    root = Path(_open["root"])
    count = arrays.import_carried(root, log, folder, workspace=held_id)
    _import_alpha_parts(root, log, folder, held_id)
    return count


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
    return count


def _import_alpha_parts(root: Path, log: HostLog, folder: Path, workspace: str) -> None:
    """Restore carried Alpha content/layout relations after the carried arrays are admitted."""
    import alphalattice.kernel.data_layer.arrays as arrays  # lazy: numpy, BLAKE3 and Parquet

    source = folder / arrays.CARRIED / "alpha-parts.json"
    if source.is_symlink():
        raise DataError("carried Alpha part relations are a link", code="data_layer.part_invalid")
    try:
        payload = source.read_bytes()
    except FileNotFoundError:
        return
    except OSError as error:
        raise DataError(
            "carried Alpha part relations are unreadable", code="data_layer.part_invalid"
        ) from error
    try:
        rows = json.loads(payload)
        if not isinstance(rows, list):
            raise ValueError("not a list")
        seen: set[tuple[str, str]] = set()
        for row in rows:
            if not isinstance(row, list) or len(row) != 3:
                raise ValueError("not a part relation")
            content, layout, digest = row
            if (
                not isinstance(content, str)
                or not isinstance(digest, str)
                or any(
                    len(v) != 64 or any(c not in "0123456789abcdef" for c in v)
                    for v in (content, digest)
                )
                or not isinstance(layout, str)
                or not layout
                or (content, layout) in seen
            ):
                raise ValueError("invalid or duplicate part relation")
            seen.add((content, layout))
            body = arrays.read_packed(root, log.read, digest, packed_only=True)
            if (
                keep_alpha_part(content, layout, body, upgrade_owner=(log, root, workspace))
                != digest
            ):
                raise ValueError("part relation differs from the held one")
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        raise DataError(
            "carried Alpha part relations are invalid", code="data_layer.part_invalid"
        ) from error


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
        with _lock:
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
    with _lock:
        reading = _open.get("reading")
        if reading is None:
            return None
        _open["log"].flush()
        provider = _open.get("tracer")
        if provider is None:
            found: dict[str, Any] = reading.read(name, **filters)
            return found
        from opentelemetry.trace import SpanKind  # lazy: the SDK, once open

        attributes = {
            "gen_ai.operation.name": "search_memory",
            "gen_ai.memory.store.id": "experience",
        }
        with provider.get_tracer("host").start_as_current_span(
            "search_memory", kind=SpanKind.CLIENT, attributes=attributes
        ) as span:
            found = reading.read(name, **filters)
            span.set_attribute("gen_ai.memory.record.count", count(found))
        _open["log"].flush()
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
