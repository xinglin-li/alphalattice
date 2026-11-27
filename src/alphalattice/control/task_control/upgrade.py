"""A 0.2.7 workspace's Task ledgers moved onto the log: the upgrade's step, verify and switch for
task-requests and task-control, behind the registry's seam until the data-root upgrade runner
registers them.

The 0.2.7 store's shape is also what a copy of a workspace carries its Tasks in (`carry_tasks`)
and what a restored backup is written as: the copy's or the restore's first open moves them onto
the log its process holds, through the same import.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import closing, contextmanager
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any, Protocol

from alphalattice.capabilities.domain_contracts import telemetry_names as telemetry
from alphalattice.capabilities.domain_contracts.identity import canonical_hash
from alphalattice.control.workspace_runtime.database import open_workspace_database
from alphalattice.foundation.data_layer.chain import canonical
from alphalattice.foundation.data_layer.history import data_root, held, holding
from alphalattice.foundation.data_layer.log import attach, uuid7
from alphalattice.foundation.data_layer.project import registry_us
from alphalattice.foundation.data_layer.upgrade import Family
from alphalattice.kernel.clock import utc_now
from alphalattice.kernel.identity import blake3_filename

from .contracts import (
    TaskExecution,
    TaskHeartbeatSignal,
    TaskLifecycle,
    TaskRecord,
    WorkItemLifecycle,
    WorkItemState,
)
from .ledger import REQUESTS_DIRECTORY, SubmittingAgent, TaskAdmissionRequest, read_frozen

TASK_CONTROL_DATABASE_FILENAME = "research-task-control.duckdb"
"""The sole durable Task Control store name inside a workspace."""
TASK_CONTROL_DIRECTORY = "runtime"
"""Where under a workspace's root its Task Control store lives, beside the Host's own state."""

LEDGER_REBUILT_DETAIL = (
    "The ledger rebuild could not recover this Task's execution or verified progress."
)
"""What rebuilding a missing Task row preserves and cannot recover."""
LEDGER_REBUILT_NEXT = (
    "Re-PLAN keeps the request for a new plan or Task; its owner chooses reusable sealed "
    "evidence and work to redo."
)
"""The offered new admission, distinct from resuming the rebuilt Task."""


class _Registry(Protocol):
    """The registry the import writes through: its log unit, its event writers and its reads."""

    database_path: Path
    workspace_id: str

    def _write(self, operation: Callable[[Any], Any], *, control_only: bool = False) -> Any: ...
    def _read(self, operation: Callable[[Any], Any]) -> Any: ...
    def _record(self, type_: str, task_id: Any, attrs: dict[str, Any], **more: Any) -> None: ...
    def _admitted(self, record: TaskRecord, items: Any, **values: Any) -> None: ...
    def _started(self, execution: TaskExecution) -> None: ...
    def _resolved(
        self, task_id: Any, command_id: object, at: str | None, resolution: str
    ) -> None: ...
    def _event(self, connection: Any, **values: Any) -> None: ...
    @staticmethod
    def _verified_record(stored_id: str, document: str) -> TaskRecord: ...


@contextmanager
def log_of(database_path: Path, root: Path | None = None) -> Iterator[tuple[Any, str | None]]:
    """The log's connection and the id it knows `database_path`'s workspace by (None when it
    knows none), read without taking a data root: the held log's in its holder, a read-only
    attach elsewhere or of the data root `root` names, nothing when there is no log."""
    folder = str(database_path.resolve().parent.parent)
    query = "SELECT workspace_id FROM workspace WHERE folder = ?"
    log = holding() if root is None else None
    if log is not None:
        with log.reading() as connection:
            row = connection.execute(query, [folder]).fetchone()
            yield connection, None if row is None else str(row[0])
        return
    root = data_root() if root is None else root
    if not (root / "machine" / "log.sqlite").is_file():
        yield None, None
        return
    with closing(attach(root)) as connection:
        row = connection.execute(query, [folder]).fetchone()
        yield connection, None if row is None else str(row[0])


def _store(workspace: Path) -> Path:
    """Where a 0.2.7 workspace kept its Task Control store."""
    return workspace / TASK_CONTROL_DIRECTORY / TASK_CONTROL_DATABASE_FILENAME


RETIRED_DIRECTORY = "retired-0.2.7"
"""Where under `runtime/` a 0.2.7 Task Control store and its frozen requests are set aside once
the log holds them."""
_IMPORTED_KEYS: dict[str, tuple[tuple[str, ...], str]] = {
    "workspace_task_work_item": (("task_id", "stage_id"), "state_json"),
    "workspace_task_execution": (("execution_id",), "execution_json"),
    "workspace_task_event": (("event_id",), "details_json"),
    "workspace_task_command": (("command_id",), "command_json"),
    "workspace_task_stage_receipt": (("receipt_hash",), "receipt_json"),
}
"""Each imported 0.2.7 table: the columns its rows are found by on the log, and the sealed
document each must hold as the store held it."""
_SEALED_AS: dict[str, Callable[[str], object]] = {
    "workspace_task_work_item": WorkItemState.model_validate_json,
    "workspace_task_execution": TaskExecution.model_validate_json,
}
"""The documents the import writes again through their models, compared as their models read
them: an older store spells a field added since by its absence (a work item's `failure_cause`),
which its hash leaves out; the rest are moved as their text and compared as JSON."""
_0_2_7_TABLES = ("workspace_task", "workspace_task_projection", *_IMPORTED_KEYS)


def sealed_equal(table: str, logged: str, stored: str) -> bool:
    """Whether the log's document and the 0.2.7 store's are the same sealed document; one that
    does not read as its model is not."""
    read: Callable[[str], object] = _SEALED_AS.get(table, json.loads)
    try:
        return bool(read(logged) == read(stored))
    except ValueError:
        return False


STORE_0_2_7: tuple[str, ...] = (
    "CREATE TABLE workspace_task (task_id VARCHAR PRIMARY KEY, task_kind VARCHAR NOT NULL,"
    " input_hash VARCHAR NOT NULL, plan_hash VARCHAR NOT NULL, lifecycle VARCHAR NOT NULL,"
    " record_json VARCHAR NOT NULL, admitted_at TIMESTAMP NOT NULL, updated_at TIMESTAMP NOT"
    " NULL, admission_sequence BIGINT, submitted_by_json VARCHAR)",
    "CREATE TABLE workspace_task_work_item (task_id VARCHAR NOT NULL, stage_id VARCHAR NOT NULL,"
    " ordinal INTEGER NOT NULL, lifecycle VARCHAR NOT NULL, state_json VARCHAR NOT NULL,"
    " updated_at TIMESTAMP NOT NULL, PRIMARY KEY (task_id, stage_id))",
    "CREATE TABLE workspace_task_execution (execution_id VARCHAR PRIMARY KEY, task_id VARCHAR"
    " NOT NULL, graph_thread_id VARCHAR NOT NULL UNIQUE, worker_instance_id VARCHAR NOT NULL,"
    " compatibility_hash VARCHAR NOT NULL, execution_json VARCHAR NOT NULL, checkpoint_seen"
    " BOOLEAN NOT NULL DEFAULT FALSE, started_at TIMESTAMP NOT NULL, last_heartbeat_at"
    " TIMESTAMP NOT NULL)",
    "CREATE TABLE workspace_task_event (event_id VARCHAR PRIMARY KEY, task_id VARCHAR NOT NULL,"
    " execution_id VARCHAR, sequence BIGINT NOT NULL, kind VARCHAR NOT NULL, details_json"
    " VARCHAR NOT NULL, recorded_at TIMESTAMP NOT NULL, UNIQUE (task_id, sequence))",
    "CREATE TABLE workspace_task_command (command_id VARCHAR PRIMARY KEY, task_id VARCHAR NOT"
    " NULL, command_hash VARCHAR NOT NULL UNIQUE, command_json VARCHAR NOT NULL, accepted_at"
    " TIMESTAMP NOT NULL, resolved_at TIMESTAMP, resolution VARCHAR)",
    "CREATE TABLE workspace_task_stage_receipt (receipt_hash VARCHAR PRIMARY KEY, task_id"
    " VARCHAR NOT NULL, execution_id VARCHAR NOT NULL, stage_id VARCHAR NOT NULL, receipt_json"
    " VARCHAR NOT NULL, observed_at TIMESTAMP NOT NULL, UNIQUE (task_id, stage_id))",
    "CREATE TABLE workspace_task_projection (task_id VARCHAR PRIMARY KEY, task_record_hash"
    " VARCHAR NOT NULL, projection_json VARCHAR NOT NULL, updated_at TIMESTAMP NOT NULL)",
)
"""A 0.2.7 workspace's Task Control store, as its registry created it: the shape the import
reads and a restored backup is written in, so the import moves both onto the log alike."""


def _clock(us: str, name: str) -> str:
    """The log's microseconds as the 0.2.7 store's naive UTC text, under its 0.2.7 name."""
    seconds = f"strftime('%Y-%m-%d %H:%M:%S', {us} / 1000000, 'unixepoch')"
    fraction = f"printf('.%06d', {us} % 1000000)"
    return f"CASE WHEN {us} IS NOT NULL THEN {seconds} || {fraction} END AS {name}"


_AS_0_2_7 = {
    "workspace_task": "SELECT task_id, task_kind, input_hash, plan_hash, lifecycle, record_json,"
    f" {_clock('admitted_us', 'admitted_at')}, {_clock('updated_us', 'updated_at')},"
    " admission_sequence, submitted_by_json FROM workspace_task w WHERE workspace_id = ?",
    "workspace_task_work_item": "SELECT w.task_id, w.stage_id, w.ordinal, w.lifecycle,"
    f" w.state_json, {_clock('w.updated_us', 'updated_at')} FROM workspace_task_work_item w",
    "workspace_task_execution": "SELECT w.execution_id, w.task_id, w.graph_thread_id,"
    " w.worker_instance_id, w.compatibility_hash, w.execution_json, 0 AS checkpoint_seen,"
    f" {_clock('w.started_us', 'started_at')},"
    f" {_clock('coalesce(h.last_heartbeat_us, w.started_us)', 'last_heartbeat_at')}"
    " FROM workspace_task_execution w LEFT JOIN workspace_task_heartbeat_at h USING (execution_id)",
    "workspace_task_event": "SELECT w.event_id, w.task_id, w.execution_id, w.sequence, w.kind,"
    f" w.details_json, {_clock('w.recorded_us', 'recorded_at')} FROM workspace_task_event w",
    "workspace_task_command": "SELECT w.command_id, w.task_id, w.command_hash, w.command_json,"
    f" {_clock('w.accepted_us', 'accepted_at')}, {_clock('w.resolved_us', 'resolved_at')},"
    " w.resolution FROM workspace_task_command w",
    "workspace_task_stage_receipt": "SELECT w.receipt_hash, w.task_id, w.execution_id,"
    f" w.stage_id, w.receipt_json, {_clock('w.observed_us', 'observed_at')}"
    " FROM workspace_task_stage_receipt w",
    "workspace_task_projection": "SELECT w.task_id, w.task_record_hash, w.projection_json,"
    f" {_clock('w.updated_us', 'updated_at')} FROM workspace_task_projection w",
}
"""Each table of a workspace's Tasks on the log, read in its 0.2.7 shape."""


def create_0_2_7_store(store: Path) -> None:
    """An empty Task Control store in the 0.2.7 shape, for a restore to fill."""
    store.parent.mkdir(parents=True, exist_ok=True)
    connection = open_workspace_database(store, read_only=False)
    try:
        for create in STORE_0_2_7:
            connection.execute(create)
    finally:
        connection.close()


def task_rows(database_path: Path, root: Path | None = None) -> dict[str, list[dict[str, Any]]]:
    """A workspace's Tasks on the log (`root`'s when named), each table's rows in its 0.2.7
    shape, in one order."""
    with log_of(database_path, root) as (connection, workspace_id):
        if workspace_id is None:
            return {table: [] for table in _AS_0_2_7}
        found: dict[str, list[dict[str, Any]]] = {}
        for table, sql in _AS_0_2_7.items():
            scoped = (
                sql
                if table == "workspace_task"
                else f"{sql} JOIN workspace_task t ON t.task_id = w.task_id AND t.workspace_id = ?"
            )
            cursor = connection.execute(scoped, [workspace_id])
            names = [column[0] for column in cursor.description]
            rows = [dict(zip(names, values, strict=True)) for values in cursor.fetchall()]
            found[table] = sorted(rows, key=lambda row: repr(sorted(row.items())))
        return found


def carry_tasks(workspace: Path, copy: Path, *, root: Path | None = None) -> int:
    """Write the Tasks the log (`root`'s when named) holds for `workspace` into its copy `copy`,
    with their frozen requests, as the store a 0.2.7 workspace kept: the copy's first open moves
    them onto the log its process holds, as it moves a 0.2.7 workspace's. A copy that already has
    such a store (an unopened 0.2.7 workspace's, copied whole) keeps it. Returns the Tasks
    carried.

    Contract: the carried store takes the 0.2.7 shape only because the upgrade import reads it;
    it retires when a copy carries its own segment of the log.
    """
    source, target = _store(workspace), _store(copy)
    if target.is_file():
        return 0
    rows = task_rows(source, root)
    if not rows["workspace_task"]:
        return 0
    with log_of(source, root) as (connection, workspace_id):
        texts = connection.execute(
            "SELECT t.text FROM event e JOIN event_text t ON t.seq = e.seq"
            " WHERE e.type = 'task.requested' AND e.workspace_id = ?",
            [workspace_id],
        ).fetchall()
    requests = target.parent / REQUESTS_DIRECTORY
    requests.mkdir(parents=True, exist_ok=True)
    for (text,) in texts:
        name = canonical_hash(
            TaskAdmissionRequest.model_validate_json(text).model_dump(mode="json")
        )
        (requests / f"{blake3_filename(name)}.json").write_bytes(text.encode("utf-8"))
    # Everything the store's Tasks bring goes first; the store is published last, filled under
    # another name and moved into place whole: a kill leaves no store, never a partial one or one
    # without its telemetry, and the next carry completes what is missing.
    _carry_telemetry(workspace, copy, root)
    filling = target.with_name(f"{target.name}.carrying")
    filling.unlink(missing_ok=True)
    create_0_2_7_store(filling)
    store = open_workspace_database(filling, read_only=False)
    try:
        for table, found in rows.items():
            for row in found:
                columns = ", ".join(row)
                marks = ", ".join("?" * len(row))
                store.execute(
                    f"INSERT INTO {table} ({columns}) VALUES ({marks})", list(row.values())
                )
    finally:
        store.close()
    os.replace(filling, target)
    return len(rows["workspace_task"])


def _carry_telemetry(workspace: Path, copy: Path, root: Path | None) -> None:
    """Write the Task telemetry the log holds for `workspace` into `copy` as the files a 0.2.7
    workspace kept them in, each at its source's place: a phase JSONL whole, a snapshot as its
    latest document. The copy's readers read them there, and its first record moves them onto
    the log its process holds (`import_telemetry`). A file the copy already has is kept: each is
    written whole under another name, so one that exists is complete."""
    with log_of(_store(workspace), root) as (connection, workspace_id):
        if workspace_id is None:
            return
        rows = connection.execute(
            "SELECT json_extract(e.attrs, '$.source'), t.text FROM event e JOIN event_text t"
            f" ON t.seq = e.seq WHERE e.type = '{telemetry.SPAN_TASK_TELEMETRY}'"
            " AND e.workspace_id = ? ORDER BY e.seq",
            [workspace_id],
        ).fetchall()
    held, base = {}, workspace.resolve()
    for source, text in rows:
        if Path(source).is_relative_to(base):
            held.setdefault(Path(source).relative_to(base), []).append(text)
    for relative, texts in held.items():
        target = copy / relative
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        body = "".join(text + "\n" for text in texts) if target.suffix == ".jsonl" else texts[-1]
        partial = target.with_name(target.name + ".carrying")
        partial.write_text(body, encoding="utf-8")
        os.replace(partial, target)


_Rows = dict[str, list[dict[str, Any]]]
"""Each table's rows by name, each row by column."""


def _read_0_2_7(store: Path) -> _Rows:
    """Every row of a 0.2.7 Task Control store, by table, read without writing it."""
    connection = open_workspace_database(store, read_only=True)
    try:
        present = {
            str(row[0])
            for row in connection.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            ).fetchall()
        }
        rows: dict[str, list[dict[str, Any]]] = {}
        for table in _0_2_7_TABLES:
            if table not in present:
                continue
            cursor = connection.execute(f"SELECT * FROM {table}")
            names = [column[0] for column in cursor.description]
            rows[table] = [dict(zip(names, values, strict=True)) for values in cursor.fetchall()]
        return rows
    finally:
        connection.close()


def _admission_order(row: dict[str, Any]) -> tuple[bool, int, object, str]:
    """A 0.2.7 Task row's place in admission order, the log's own: unordered rows first."""
    sequence = row.get("admission_sequence")
    return sequence is not None, sequence or 0, row["admitted_at"], str(row["task_id"])


def stored_records(store: Path) -> tuple[TaskRecord, ...] | None:
    """The Tasks a 0.2.7 store not yet moved onto the log holds, in admission order: until the
    move, it is their authority. None when the workspace has no such store."""
    if not store.is_file():
        return None
    rows = sorted(_read_0_2_7(store).get("workspace_task", ()), key=_admission_order)
    return tuple(TaskRecord.model_validate_json(row["record_json"]) for row in rows)


def _by_task(rows: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {table: {} for table in _IMPORTED_KEYS}
    for table in _IMPORTED_KEYS:
        for row in rows.get(table, ()):
            grouped[table].setdefault(str(row["task_id"]), []).append(row)
    return grouped


def _stored_time(value: object) -> str:
    """A 0.2.7 store's naive UTC timestamp as the log's rows keep a registry clock."""
    if isinstance(value, datetime):
        return value.isoformat(sep=" ", timespec="microseconds")
    return str(value)


def import_ledgers(registry: _Registry) -> list[str]:
    """Move a 0.2.7 workspace's Task ledgers onto the log: step, verify, switch.

    These are the upgrade's three parts for task-requests and task-control, run here, behind
    this seam, until the data-root upgrade runner registers them. The step imports every row of
    the 0.2.7 store and every frozen request in one unit of the log, so a kill leaves all of it
    or none; a Task already on the log is not imported again, so a kill after the unit resumes
    at verify.
    Verify reads every imported record and frozen text back from the log. The switch then sets
    the store and the requests aside under `retired-0.2.7/`; nothing is deleted. A switch a kill
    interrupted completes first, into its recorded directory, even when the store has moved.
    """
    store, runtime = registry.database_path, registry.database_path.parent
    requests = runtime / REQUESTS_DIRECTORY
    retired = (store, store.with_name(f"{store.name}.wal"), requests)
    if not store.is_file() and not requests.is_dir():
        set_aside(runtime, retired)  # the rest of a set a kill left part-way, if any
        return []
    rows = _read_0_2_7(store) if store.is_file() else {}
    frozen = read_frozen(runtime)
    registry._write(lambda connection: _import(registry, connection, rows, frozen))
    unequal = _verify_import(registry, rows, frozen)
    if unequal:
        return unequal
    set_aside(runtime, retired)
    return []


def set_aside(runtime: Path, paths: Iterable[Path]) -> None:
    """Move one set of a 0.2.7 workspace's retired files (a database with its -wal and -shm, a
    store with its requests) out of its live root, under `retired-0.2.7/`, the upgrade's keeping
    of them: nothing is deleted, checkpointed or rewritten.

    The set's directory is chosen and recorded before its first move: `retired-0.2.7/` itself,
    or a dated directory beside a set kept there before (a carried copy of a workspace that moved
    once), so nothing is replaced. A move interrupted part-way resumes into the recorded
    directory, so the set is never split; the record goes once the set has moved."""
    paths = list(paths)
    kept = runtime / RETIRED_DIRECTORY
    record = kept / f"{paths[0].name}.set.json"
    if record.is_file():
        retired = runtime / json.loads(record.read_text("utf-8"))["directory"]
    else:
        moved = [path for path in paths if path.exists()]
        if not moved:
            return
        retired = kept
        if any((kept / path.name).exists() for path in moved):
            retired = kept / utc_now().strftime("%Y%m%dT%H%M%S%fZ")
        retired.mkdir(parents=True, exist_ok=True)
        chosen = record.with_name(f"{record.name}.tmp")
        where = retired.relative_to(runtime).as_posix()
        chosen.write_text(json.dumps({"directory": where}), "utf-8")
        os.replace(chosen, record)
    for path in paths:
        if path.exists():
            os.replace(path, retired / path.name)
    record.unlink()


def _import(
    registry: _Registry,
    connection: sqlite3.Connection,
    rows: _Rows,
    frozen: tuple[tuple[TaskAdmissionRequest, str], ...],
) -> None:
    """Every Task of the 0.2.7 store, each as the events that make it; then each frozen request
    whose row the store lost; then the listing projection the store held."""
    present = {
        str(row[0])
        for row in connection.execute(
            "SELECT task_id FROM workspace_task WHERE workspace_id = ?", [registry.workspace_id]
        ).fetchall()
    }
    text_of = {str(request.task_id): text for request, text in frozen}
    of_task = _by_task(rows)
    tasks = sorted(rows.get("workspace_task", ()), key=_admission_order)
    for row in tasks:
        if str(row["task_id"]) not in present:
            _import_task(registry, row, of_task, text_of.get(str(row["task_id"])))
            present.add(str(row["task_id"]))
    taken = {row.get("admission_sequence") for row in tasks} - {None}
    _rebuild_lost(registry, connection, frozen, present, taken)
    for projection in rows.get("workspace_task_projection", ()):
        connection.execute(  # working state: the derived listing, as the store held it
            "INSERT OR IGNORE INTO workspace_task_projection VALUES (?, ?, ?, ?)",
            [
                projection["task_id"],
                projection["task_record_hash"],
                projection["projection_json"],
                registry_us(_stored_time(projection["updated_at"])),
            ],
        )


def _import_task(
    registry: _Registry,
    row: dict[str, Any],
    of_task: dict[str, dict[str, list[dict[str, Any]]]],
    frozen: str | None,
) -> None:
    """One 0.2.7 Task as its events, in the order its registry wrote them."""
    task_id = str(row["task_id"])
    record = registry._verified_record(task_id, row["record_json"])
    items = [
        WorkItemState.model_validate_json(item["state_json"])
        for item in sorted(
            of_task["workspace_task_work_item"].get(task_id, ()), key=lambda i: i["ordinal"]
        )
    ]
    by = row.get("submitted_by_json")
    registry._admitted(
        record,
        items,
        admission_sequence=row.get("admission_sequence"),
        submitted_by=None if by is None else SubmittingAgent.model_validate_json(by),
        frozen=frozen,
    )
    for execution in sorted(
        of_task["workspace_task_execution"].get(task_id, ()), key=lambda e: e["started_at"]
    ):
        registry._started(TaskExecution.model_validate_json(execution["execution_json"]))
    for note in sorted(
        of_task["workspace_task_event"].get(task_id, ()), key=lambda n: n["sequence"]
    ):
        attrs = {
            "event_id": note["event_id"],
            "execution_id": note["execution_id"],
            "sequence": note["sequence"],
            "kind": note["kind"],
            "recorded_at": _stored_time(note["recorded_at"]),
        }
        registry._record("task.noted", record.task_id, attrs, text=note["details_json"])
    for receipt in of_task["workspace_task_stage_receipt"].get(task_id, ()):
        attrs = {
            "receipt_hash": receipt["receipt_hash"],
            "execution_id": receipt["execution_id"],
            "stage_id": receipt["stage_id"],
            "observed_at": _stored_time(receipt["observed_at"]),
        }
        registry._record("stage.verified", record.task_id, attrs, text=receipt["receipt_json"])
    for command in sorted(
        of_task["workspace_task_command"].get(task_id, ()), key=lambda c: c["accepted_at"]
    ):
        accepted = {
            "command_id": command["command_id"],
            "command_hash": command["command_hash"],
            "accepted_at": _stored_time(command["accepted_at"]),
        }
        text = command["command_json"]
        registry._record("command.accepted", record.task_id, accepted, text=text)
        if command["resolution"] is not None:
            at = command["resolved_at"]
            resolved = None if at is None else _stored_time(at)
            registry._resolved(
                record.task_id, command["command_id"], resolved, command["resolution"]
            )


def _rebuild_lost(
    registry: _Registry,
    connection: sqlite3.Connection,
    frozen: tuple[tuple[TaskAdmissionRequest, str], ...],
    present: set[str],
    taken: set[Any],
) -> None:
    """A request whose row the store lost is listed again: what it was admitted to do reads
    again and its progress is not claimed; the same request admits it anew."""
    now = utc_now()
    for request, text in frozen:
        if str(request.task_id) in present:
            continue
        record = TaskRecord.from_identity(
            task_id=request.task_id,
            task_kind=request.task_kind,
            input=request.input,
            goal=request.goal,
            plan=request.plan,
            lifecycle=TaskLifecycle.BLOCKED,
            active_work_item_id=None,
            latest_execution_id=None,
            admitted_at=request.admitted_at,
            started_at=None,
            updated_at=now,
            failure_code="task_control.ledger_rebuilt",
            version=1,
        )
        items = [
            WorkItemState.from_identity(
                task_id=record.task_id,
                stage_id=definition.stage_id,
                definition_hash=definition.definition_hash,
                lifecycle=WorkItemLifecycle.PENDING,
                attempt_count=0,
                evidence=(),
                failure_code=None,
                started_at=None,
                updated_at=now,
                version=1,
            )
            for definition in record.plan.work_items
        ]
        sequence = request.admission_sequence
        registry._admitted(
            record,
            items,
            admission_sequence=None if sequence in taken else sequence,
            submitted_by=request.submitted_by,
            frozen=text,
        )
        registry._event(
            connection,
            task_id=record.task_id,
            execution_id=None,
            kind="task.rebuilt_from_request",
            details={
                "admission_sequence": request.admission_sequence,
                "detail": LEDGER_REBUILT_DETAIL,
                "next_detail": LEDGER_REBUILT_NEXT,
            },
            observed_at=now,
        )
        present.add(str(record.task_id))


def _verify_import(
    registry: _Registry,
    rows: _Rows,
    frozen: tuple[tuple[TaskAdmissionRequest, str], ...],
) -> list[str]:
    """Refuse the switch by name unless the log holds every 0.2.7 row and frozen text."""

    def read(connection: sqlite3.Connection) -> list[str]:
        unequal: list[str] = []
        for row in rows.get("workspace_task", ()):
            found = connection.execute(
                "SELECT record_json FROM workspace_task WHERE task_id = ? AND workspace_id = ?",
                [str(row["task_id"]), registry.workspace_id],
            ).fetchone()
            if found is None or json.loads(found[0]) != json.loads(row["record_json"]):
                unequal.append(str(row["task_id"]))
        for table, (keys, document) in _IMPORTED_KEYS.items():
            where = " AND ".join(f"{key} = ?" for key in keys)
            for row in rows.get(table, ()):
                found = connection.execute(
                    f"SELECT {document} FROM {table} WHERE {where}", [row[k] for k in keys]
                ).fetchone()
                if found is None or not sealed_equal(table, found[0], row[document]):
                    unequal.append(f"{table}:{'/'.join(str(row[k]) for k in keys)}")
        for request, text in frozen:
            held_text = connection.execute(
                "SELECT t.text FROM event e JOIN event_text t ON t.seq = e.seq"
                " WHERE e.type = 'task.requested' AND e.subject_id = ?",
                [str(request.task_id)],
            ).fetchone()
            if held_text is None or held_text[0] != text:
                unequal.append(f"request:{request.task_id}")
        return unequal

    return list(registry._read(read))


HEARTBEAT_LEASE_SECONDS = 30.0
"""The runner's existing default stale threshold; readers may still ask for another threshold."""


def read_heartbeat(path: Path, execution_id: object) -> TaskHeartbeatSignal | None:
    """The latest signal for this runner and execution, from its wait projection, read-only."""
    with log_of(path) as (connection, _):
        if connection is None:
            return None
        row = connection.execute(
            "SELECT match_attrs FROM wait WHERE purpose = 'task.heartbeat'"
            " AND json_extract(match_attrs, '$.source') = ?"
            " AND json_extract(match_attrs, '$.signal.execution_id') = ?"
            " ORDER BY started_seq DESC LIMIT 1",
            [str(path.resolve()), str(execution_id)],
        ).fetchone()
    return None if row is None else TaskHeartbeatSignal.model_validate(json.loads(row[0])["signal"])


def write_heartbeat(path: Path, signal: TaskHeartbeatSignal) -> None:
    """Renew a step's time wait in the held log's unit; the sealed signal is its match data.

    The latest signal remains readable after the wait resolves. Expiry readers retain their
    existing business/operational max and caller threshold, including the strict greater-than
    boundary. This deadline records that default; it does not create a scheduler or a new clock.
    """
    log = held()
    with log.unit() as connection:
        row = connection.execute(
            "SELECT workspace_id FROM workspace_task WHERE task_id = ?", [str(signal.task_id)]
        ).fetchone()
        if row is None:
            raise ValueError(f"task_control.heartbeat_task_unbound:{signal.task_id}")
        source = str(path.resolve())
        previous = connection.execute(
            "SELECT wait_id, match_attrs FROM wait WHERE purpose = 'task.heartbeat'"
            " AND json_extract(match_attrs, '$.source') = ?"
            " AND json_extract(match_attrs, '$.signal.execution_id') = ?"
            " AND resolved_seq IS NULL ORDER BY started_seq DESC LIMIT 1",
            [source, str(signal.execution_id)],
        ).fetchone()
        if previous is not None:
            if json.loads(previous[1])["signal"] == signal.model_dump(mode="json"):
                return
            log.record(
                "wait.resolved", ("wait", previous[0]), workspace=row[0], node="task_control"
            )
        observed_us = int(signal.observed_at.timestamp() * 1_000_000)
        log.record(
            "wait.started",
            ("wait", uuid7()),
            {
                "task_id": str(signal.task_id),
                "kind": "time",
                "purpose": "task.heartbeat",
                "fires_us": observed_us + int(HEARTBEAT_LEASE_SECONDS * 1_000_000),
                "match": {"source": source, "signal": signal.model_dump(mode="json")},
            },
            workspace=row[0],
            occurred_us=observed_us,
            node="task_control",
        )


def _stored_heartbeats(path: Path) -> tuple[TaskHeartbeatSignal, ...]:
    """Read every old signal without changing or repairing the old SQLite file."""
    if not path.is_file():
        return ()
    try:
        with closing(
            sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=5)
        ) as con:
            if not con.execute("SELECT 1 FROM workspace_task_heartbeat LIMIT 1").fetchall():
                return ()  # an older empty schema has no signal_json column
            rows = con.execute("SELECT signal_json FROM workspace_task_heartbeat").fetchall()
        return tuple(TaskHeartbeatSignal.model_validate_json(row[0]) for row in rows)
    except (OSError, sqlite3.Error, ValueError) as error:
        raise ValueError("task_control.heartbeat_store_unreadable") from error


def _heartbeat_step(path: Path) -> None:
    """A live old lease is refused by execution id, before any signal is copied."""
    signals, log = _stored_heartbeats(path), held()
    with log.unit() as connection:
        for signal in signals:
            row = connection.execute(
                "SELECT lifecycle FROM workspace_task WHERE task_id = ?", [str(signal.task_id)]
            ).fetchone()
            if row is None:
                raise ValueError(f"task_control.heartbeat_task_unbound:{signal.task_id}")
            if row[0] in ("RUNNING", "CANCEL_REQUESTED"):
                raise ValueError(f"task_control.live_lease_present:{signal.execution_id}")
        for signal in signals:
            write_heartbeat(path, signal)


def _heartbeat_verify(path: Path, _runner: object = None) -> bool:
    """ALL EQUAL for every old execution, including its worker, sequence, seal and instant."""
    return all(
        read_heartbeat(path, signal.execution_id) == signal for signal in _stored_heartbeats(path)
    )


def _set_aside(paths: tuple[Path, ...]) -> None:
    """Leave the log the one reader by setting the verified old files aside.

    No file is deleted. A kill between renames leaves each file at its old or retired name;
    the next step checks whichever old files remain and never replaces a retired generation:
    a later one (a carried copy's, which the log already holds) is kept in a dated directory
    beside it, as `set_aside` keeps a store's.
    """
    held().flush()  # the conversion is durable before the old reader is set aside
    generation = utc_now().strftime("%Y%m%dT%H%M%S%fZ")
    files = tuple(
        file
        for path in paths
        for file in (
            (path, path.with_name(path.name + "-wal"), path.with_name(path.name + "-shm"))
            if path.suffix == ".sqlite"
            else (path,)
        )
    )
    # The main file first, then its abandoned WAL side files.
    for path in files:
        if not path.exists():
            continue
        target = path.parent / RETIRED_DIRECTORY / path.name
        if target.exists():
            target = target.parent / generation / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(path, target)


def heartbeat_family(path: Path) -> Family:
    """The task-heartbeats registration in the upgrade runner's step/verify form: its step is the
    import, which sets the old file aside once its copy is ALL EQUAL."""
    return Family(
        "task-heartbeats",
        lambda _runner: import_heartbeats(path),
        partial(_heartbeat_verify, path),
    )


def import_heartbeats(path: Path) -> None:
    """The carrier's first-use import seam, and the family's restart-safe step."""
    if not path.is_file():
        return
    _heartbeat_step(path)
    if not _heartbeat_verify(path):
        raise ValueError("task_control.heartbeat_import_unequal")
    _set_aside((path,))


def _stored_telemetry(path: Path) -> tuple[dict[str, Any], ...]:
    """The old reader's valid rows; an invalid source stays present and refuses the switch."""
    if not path.is_file():
        return ()
    try:
        raw = path.read_text(encoding="utf-8")
        values = (
            [json.loads(line) for line in raw.splitlines()]
            if path.suffix == ".jsonl"
            else [json.loads(raw)]
        )
    except (OSError, ValueError) as error:
        raise ValueError(f"task_control.telemetry_import_unreadable:{path.name}") from error
    expected = path.stem if path.suffix == ".jsonl" else path.parent.name
    if any(not isinstance(row, dict) or row.get("task_id") != expected for row in values):
        raise ValueError(f"task_control.telemetry_import_unreadable:{path.name}")
    return tuple(values)


def read_task_telemetry(
    path: Path, task_id: object, *, latest: bool = False
) -> tuple[dict[str, Any], ...]:
    """The exact Task-bound documents carried by this source's spans, in append order."""
    with log_of(path) as (connection, _):
        if connection is None:
            return ()
        order = " ORDER BY e.seq DESC LIMIT 1" if latest else " ORDER BY e.seq"
        rows = connection.execute(
            "SELECT t.text FROM event e JOIN event_text t USING (seq)"
            f" WHERE e.type = '{telemetry.SPAN_TASK_TELEMETRY}' AND e.subject_id = ?"
            " AND json_extract(e.attrs, '$.source') = ?" + order,
            [str(task_id), str(path.resolve())],
        ).fetchall()
    return tuple(json.loads(row[0]) for row in rows)


def record_task_telemetry(path: Path, row: Mapping[str, Any]) -> None:
    """One readout is one local SDK span through the data-layer span owner.

    The source's structured readout is the admitted event text, so large span summaries stay
    intact without expanding the small attribute envelope. No runtime, exporter thread or
    remote destination is added. Historical clocks come from the document, never from import.
    """
    from opentelemetry.sdk.trace import ReadableSpan  # lazy: the SDK, only while recording
    from opentelemetry.sdk.trace.id_generator import RandomIdGenerator  # lazy: span recording
    from opentelemetry.trace import (  # lazy: span recording, never a timing read
        SpanContext,
        TraceFlags,
        get_current_span,
    )

    from alphalattice.foundation.data_layer.spans import (  # lazy: the SDK, only while recording
        step_event,
    )

    log = held()
    task_id = str(row["task_id"])
    found = log.read("SELECT workspace_id FROM workspace_task WHERE task_id = ?", (task_id,))
    if not found:
        raise ValueError(f"task_control.telemetry_task_unbound:{task_id}")
    with log.unit():
        _register_telemetry(log)
    projection = row.get("projection", row)
    # A progress snapshot is an instantaneous observation. Only a completed phase supplies
    # elapsed work; counting every snapshot since started_at would count that work repeatedly.
    instant = (
        row["started_at"]
        if "wall_seconds" in row
        else projection.get("updated_at", row.get("written_at"))
    )
    start = datetime.fromisoformat(str(instant))
    if start.utcoffset() is None:
        raise ValueError("task_control.telemetry_clock_naive")
    end = start + timedelta(seconds=float(row["wall_seconds"])) if "wall_seconds" in row else start
    ids, parent = RandomIdGenerator(), get_current_span().get_span_context()
    context = SpanContext(
        parent.trace_id if parent.is_valid else ids.generate_trace_id(),
        ids.generate_span_id(),
        False,
        TraceFlags(TraceFlags.SAMPLED),
    )
    span = ReadableSpan(
        telemetry.SPAN_TASK_TELEMETRY,
        context=context,
        parent=parent if parent.is_valid else None,
        attributes={
            telemetry.ALPHALATTICE_SUBJECT_KIND: "task",
            telemetry.ALPHALATTICE_SUBJECT_ID: task_id,
            telemetry.ALPHALATTICE_WORKSPACE_ID: str(found[0][0]),
            telemetry.ALPHALATTICE_NODE: "task_control",
            telemetry.SOURCE: str(path.resolve()),
            telemetry.CHANNEL: path.stem if path.suffix != ".jsonl" else "stage.phase",
        },
        start_time=int(start.timestamp() * 1_000_000) * 1000,
        end_time=int(end.timestamp() * 1_000_000) * 1000,
    )
    text = json.dumps(dict(row), sort_keys=True, separators=(",", ":"))
    log.append(step_event(span, text=text), text)


def _register_telemetry(log: Any) -> None:
    """Register once in the caller's log unit, including concurrent first phase completions."""
    if not log.read(
        "SELECT 1 FROM definition WHERE family = 'event_type' AND name = ?",
        (telemetry.SPAN_TASK_TELEMETRY,),
    ):
        definition = {"family": "event_type", "name": telemetry.SPAN_TASK_TELEMETRY, "version": 1}
        log.record(
            "definition.registered",
            ("definition", f"event_type/{telemetry.SPAN_TASK_TELEMETRY}"),
            definition,
            text=canonical(
                definition
                | {
                    "status": "standard",
                    "text": "A Task phase readout or optional progress observation.",
                    "spec": {
                        "work": True,
                        "subject": "task",
                        "required": [telemetry.CHANNEL, telemetry.SOURCE],
                        "text": "required",
                        "durable": False,
                    },
                }
            ).decode(),
            node="task_control",
        )


def _logged(path: Path, task_id: object) -> tuple[dict[str, Any], ...]:
    """What the log holds of `path`'s source in the form its file keeps: a phase JSONL's every
    readout, a snapshot's latest document (a carried copy's file holds only that one)."""
    return read_task_telemetry(path, task_id, latest=path.suffix != ".jsonl")


def _telemetry_step(paths: tuple[Path, ...]) -> None:
    """Copy every old phase/snapshot once, retaining its exact readout and source clock."""
    # Parse all sources first: a bad source cannot retire another file or leave half a source.
    sources = [(path, _stored_telemetry(path)) for path in paths]
    with held().unit():
        for path, rows in sources:
            if not rows:
                continue
            logged = _logged(path, rows[0]["task_id"])
            if logged != rows[: len(logged)]:
                raise ValueError(f"task_control.telemetry_import_conflict:{path.name}")
            for row in rows[len(logged) :]:
                record_task_telemetry(path, row)


def _telemetry_verify(paths: tuple[Path, ...], _runner: object = None) -> bool:
    """ALL EQUAL for every stored phase and snapshot, without a missing-field default."""
    for path in paths:
        rows = _stored_telemetry(path)
        if rows and _logged(path, rows[0]["task_id"]) != rows:
            return False
    return True


def telemetry_family(paths: tuple[Path, ...]) -> Family:
    """The task-telemetry registration in the upgrade runner's step/verify form: its step is the
    import, which sets the old files aside once their copies are ALL EQUAL.

    Paths name only stage phase JSONL and work-progress/listing-activity snapshots. Durable
    preparation stage evidence sharing TaskTelemetry's JSON utility is outside this family.
    """
    return Family(
        "task-telemetry",
        lambda _runner: import_telemetry(paths),
        partial(_telemetry_verify, paths),
    )


def import_telemetry(paths: tuple[Path, ...]) -> None:
    """The carrier's first-use import seam, and the family's restart-safe step."""
    existing = tuple(path for path in paths if path.is_file())
    if not existing:
        return
    _telemetry_step(existing)
    if not _telemetry_verify(existing):
        raise ValueError("task_control.telemetry_import_unequal")
    _set_aside(existing)
