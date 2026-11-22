"""Each event type's projection: the rows of log.sqlite that derive from the event log.

A projection reads the event's own fields, its attributes and its machine text; never a
clock, a file or another store. `log.replay` also supplies the retained IDs for historical
registrations. `differences` names any projected table that
does not. The meaning of each type is the 0.2.8 reference's (tools/records/2026-11/2026-11-14/
schemas/project.py). An undeclared status move retains the event and records its refusal. Other
projection refusals roll the event back with its unit.
"""

from __future__ import annotations

import json
import math
import re
import secrets
import sqlite3
import struct
import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timedelta
from typing import Any, Final

from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer.admit import secrets_in
from alphalattice.kernel.data_layer.chain import CHAIN_VERSION, canonical, digest
from alphalattice.kernel.shared_kernel import telemetry_names as telemetry

ENDED: Final = ("completed", "failed", "cancelled")
ARTIFACT_PROFILE_TEXT_VERSION: Final = 2
"""The artifact.produced version whose optional machine text is the profile's JSON shape."""
ADMISSION_SUMMARY_TEXT_VERSION: Final = 2
"""The admission version whose optional machine text is its JSON provider list."""
BOUND_TEXT_EVENTS: Final = (
    ("artifact.produced", ARTIFACT_PROFILE_TEXT_VERSION),
    ("admission", ADMISSION_SUMMARY_TEXT_VERSION),
)
RECIPE_ENCODING: Final = 1
"""The version of recipe_key's typed binary encoding, its first byte; a new encoding is a new
version, so a key never changes meaning."""
PROJECTED: Final = (
    "definition_lineage",
    "definition_name",
    "definition_version",
    "definition_ref",
    "workspace",
    "goal",
    "task",
    "status_refusal",
    "task_step",
    "wait",
    "recipe",
    "recipe_upstream",
    "selection",
    "hypothesis",
    "trial",
    "experience_made",
    "experience_version",
    "trial_experience",
    "prediction",
    "artifact",
    "measure",
    "event_artifact",
    "tag",
    "live_ref",
    "book",
    "ledger_entry",
    "setting",
    "chain_checkpoint",
    "chain_head",
    "daily_span",
    "knowledge_source",
    "knowledge_source_version",
    "knowledge_workspace_source",
    "knowledge_snapshot",
    "knowledge_snapshot_revision",
)
"""Every table replay rebuilds from the events alone."""
MACHINE_TEXT: Final = frozenset(
    {
        "definition.registered",
        "registry.seeded",
        "task.requested",
        "task.saved",
        "work_item.saved",
        "task.noted",
        telemetry.SPAN_TASK_TELEMETRY,
        "execution.started",
        "stage.verified",
        "command.accepted",
        "knowledge.revision_published",
        "knowledge.snapshot_published",
    }
)
"""The types whose text is never forgotten: a definition's, which replay rebuilds the registry
from; a Task's frozen request, what it was admitted to do; and the sealed record each Task
Control event carries, which its view reads."""
TASK_STATE: Final = {
    "QUEUED": "requested",
    "RUNNING": "running",
    "SUCCEEDED": "completed",
    "CANCELLED": "cancelled",
}
"""A Task Control lifecycle's state in `task`. Review, deferral, requested cancellation, recovery,
and a blocked task that an explicit recovery may reopen all wait."""
REFS: Final = {
    "feature_set": (("method", "method", "method_version"),),
    "artifact_type": (("measures", "metric", None),),
}
"""What a family's spec names of other definitions: (spec field, family, version field). Each
becomes a definition_ref row, so a check guards what the JSON alone would not."""

Con = sqlite3.Connection
Row = sqlite3.Row
Attrs = dict[str, Any]
Projection = Callable[[Con, Row, Attrs, str | None], None]
HANDLERS: Final[dict[str, Projection]] = {}
"""Each projected event type's handler; it reads the event, its attributes and its text only."""


def on(*types: str) -> Callable[[Projection], Projection]:
    """Register a handler as the projection of `types`."""

    def register(handler: Projection) -> Projection:
        for name in types:
            HANDLERS[name] = handler
        return handler

    return register


def project(
    con: sqlite3.Connection,
    event: sqlite3.Row,
    text: str | None,
    legacy_identity: Callable[[str, str], str] | None = None,
) -> None:
    """Apply one written event's projection, in the transaction that wrote it."""
    a = json.loads(event["attrs"])
    for family, name, identity in a.get("definition_ids", ()):
        _insert_lineage(con, family, name, identity, event["seq"])
    if legacy_identity is not None and "definition_ids" not in a:
        for family, name in definition_names(event["type"], event["phase"], event["node"], a, text):
            if lineage(con, family, name) is None:
                _insert_lineage(con, family, name, legacy_identity(family, name), event["seq"])
    handler = HANDLERS.get(event["type"])
    if handler is not None:
        try:
            handler(con, event, a, text)
        except sqlite3.IntegrityError as error:
            prefix = "status move refused: "
            if error.sqlite_errorcode != sqlite3.SQLITE_CONSTRAINT_TRIGGER or not str(
                error
            ).startswith(prefix):
                raise
            _status_refusal(con, event, str(error).removeprefix(prefix))
    for role in ("used", "produced"):  # any step names what it used and produced
        for artifact in a.get(role, ()):
            con.execute(
                "INSERT OR IGNORE INTO event_artifact VALUES (?, ?, ?)",
                (event["seq"], role, artifact),
            )
    for key, values in a.get("tags", {}).items():  # any event may tag its subject
        named = _tag_key(con, key, None)
        for value in values if isinstance(values, list) else [values]:
            _tag(con, event, named, value)
    if event["phase"] in ENDED:  # the day's totals, in the step's own transaction
        _daily(con, event, a)


def _status_refusal(con: Con, e: Row, name: str) -> None:
    con.execute("INSERT INTO status_refusal VALUES (?, ?)", (e["seq"], name))


def recipe_key(method_id: str, version: int, params: Any, upstream: Iterable[str] = ()) -> str:
    """The key that finds an action already defined: BLAKE3 over RECIPE_ENCODING's typed binary
    form of its method's lineage and registered version, its parameters and the keys of the
    recipes it reads, sorted. Never a name, so a renamed method keeps every recipe's key, and
    never JSON text, so 5 and 5.0 are one parameter."""
    value = [method_id, version, params, sorted(upstream)]
    return digest(bytes([RECIPE_ENCODING]) + _typed(value))


def _typed(value: Any) -> bytes:
    """One value's tagged binary form: an integral number is its integer, any other float its
    IEEE-754 double; a map's names are sorted; each string and number is length-prefixed."""
    if value is None:
        return b"N"
    if isinstance(value, bool):
        return b"T" if value else b"F"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _refuse("value_not_finite", f"a recipe or a dimension holds no {value}")
        if not value.is_integer():
            return b"D" + struct.pack(">d", value)
        value = int(value)
    if isinstance(value, int):
        return b"I" + _sized(value.to_bytes(value.bit_length() // 8 + 1, "big", signed=True))
    if isinstance(value, str):
        return b"S" + _sized(value.encode())
    if isinstance(value, list | tuple):
        return b"L" + struct.pack(">I", len(value)) + b"".join(map(_typed, value))
    if isinstance(value, Mapping) and all(isinstance(name, str) for name in value):
        named = (_typed(name) + _typed(value[name]) for name in sorted(value))
        return b"M" + struct.pack(">I", len(value)) + b"".join(named)
    raise _refuse("value_untyped", f"a recipe or a dimension holds no {type(value).__name__}")


def _sized(data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + data


def dims(value: Any) -> str:
    """A measure's dimensions in their one form: the canonical JSON of a map of names to strings,
    booleans or numbers, an integral number as its integer; '{}' for the whole artifact."""
    if not isinstance(value, Mapping):
        raise _refuse("measure_dims_untyped", f"a measure's dims are a map of names, not {value!r}")
    for named in value.values():
        if not isinstance(named, str | bool | int | float):
            raise _refuse("value_untyped", f"a dimension holds no {type(named).__name__}")
        _typed(named)  # a value with no typed form, such as NaN, is refused here
    return canonical(
        {
            name: int(v) if isinstance(v, float) and v.is_integer() else v
            for name, v in value.items()
        }
    ).decode()


INPUT_ENCODING: Final = 1
"""The version of input_digest's binary form; a change to the form is a new version."""


def input_digest(
    con: sqlite3.Connection, depends: Mapping[str, Iterable[str]], cutoff_us: int
) -> str:
    """Whether a cached partition is still valid: BLAKE3 over INPUT_ENCODING's form of the
    admissions of each dataset it depends on whose rows in its parts (`depends`: a dataset and its
    parts, 'all' for an unpartitioned one) are visible at `cutoff_us`, the cutoff itself when it
    falls inside one of their knowledge ranges, else their latest knowledge time, and their lag-rule
    versions. It reads the admissions' `parts`, never the data: a correction, which is a new
    admission, changes the digest of exactly the parts it touched, and a later admission never
    reaches an earlier cutoff's digest."""
    wanted = {dataset: set(parts) for dataset, parts in depends.items()}
    found = con.execute(
        "SELECT a.admission_id, a.dataset, a.parts, json_extract(e.attrs, '$.lag_rule'),"
        " json_extract(e.attrs, '$.lag_rule_version') FROM admission a JOIN event e"
        " ON e.seq = a.admission_seq"
    ).fetchall()
    rows, ranges = [], []
    for admission, dataset, parts, rule, version in found:
        seen = [
            (p["part"], p["min_known_asof_us"], p["max_known_asof_us"])
            for p in json.loads(parts or "[]")
            if p["part"] in wanted.get(dataset, ()) and p["min_known_asof_us"] <= cutoff_us
        ]
        if seen:
            rows.append((admission, dataset, rule or "", int(version or 0)))
            ranges += [(low, high) for _, low, high in seen]
    rows.sort()  # one order whatever the log's: by admission id
    inside = any(low <= cutoff_us <= high for low, high in ranges)
    effective = cutoff_us if inside or not ranges else max(high for _, high in ranges)
    data = bytearray([INPUT_ENCODING]) + struct.pack(">qI", effective, len(rows))
    for row in rows:  # column by column within each admission: its id, dataset, lag rule, version
        for text in row[:3]:
            data += _sized(str(text).encode())
        data += struct.pack(">q", row[3])
    return digest(bytes(data))


def _sized(data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + data


def uuid7(now_ns: int | None = None) -> str:
    """A UUIDv7 (RFC 9562): 48 bits of Unix milliseconds, then 74 random bits."""
    ms = (time.time_ns() if now_ns is None else now_ns) // 1_000_000
    value = (ms & (1 << 48) - 1) << 80 | 0x7 << 76 | secrets.randbits(12) << 64
    return str(uuid.UUID(int=value | 0b10 << 62 | secrets.randbits(62)))


def lineage(con: sqlite3.Connection, family: str, name: str) -> str | None:
    """The lineage that holds or held `name` in `family`; None when no lineage ever did."""
    found = con.execute(
        "SELECT definition_id FROM definition_name WHERE family = ? AND name = ?", (family, name)
    ).fetchone()
    return None if found is None else str(found[0])


def require_lineage(con: sqlite3.Connection, family: str, name: str) -> str:
    """The lineage `name` names in `family`, refused by name when none does."""
    found = lineage(con, family, name)
    if found is None:
        raise _refuse(f"{family}_unregistered", f"no {family} definition is named {name}")
    return found


def definition_names(
    type_: str, phase: str | None, node: str, attrs: Attrs, text: str | None
) -> Iterable[tuple[str, str]]:
    """Names this event can register, explicitly or when the projection first sees them."""
    if type_ == "definition.registered":
        definition = json.loads(text or "{}")
        if definition.get("version") == 1:
            yield definition["family"], definition["name"]
    elif type_ == "task.requested":
        yield "event_type", attrs["type"]
    elif type_ == "wait.started" and attrs.get("event_type") is not None:
        yield "event_type", attrs["event_type"]
    if phase in ENDED:
        yield "node", node


def _insert_lineage(con: Con, family: str, name: str, identity: str, seq: int) -> None:
    con.execute(
        "INSERT INTO definition_lineage VALUES (?, ?, ?, ?, ?)", (identity, family, name, seq, seq)
    )
    con.execute("INSERT INTO definition_name VALUES (?, ?, ?, ?)", (family, name, identity, seq))


def register(
    con: Con, family: str, name: str, version: int, *fields: Any, identity: str | None = None
) -> str:
    """Register a version of a named lineage, with its supplied id on a seed's first version."""
    found = lineage(con, family, name)
    if found is None:
        if version != 1 or identity is None:
            raise _refuse(
                "definition_unregistered", f"{family} {name} has no version 1 registration id"
            )
        found = identity
        _insert_lineage(con, family, name, found, fields[-3])  # registered_seq
    con.execute(
        f"INSERT INTO definition_version VALUES (?, ?, {', '.join('?' * len(fields))})",
        (found, version, *fields),
    )
    spec = json.loads(fields[3])
    for field, referred, version_field in REFS.get(family, ()):
        names = spec.get(field)
        for named in names if isinstance(names, list) else [] if names is None else [names]:
            ref_version = spec.get(version_field) if version_field else None
            con.execute(
                "INSERT INTO definition_ref VALUES (?, ?, ?, ?, ?)",
                (found, version, field, require_lineage(con, referred, named), ref_version),
            )
    return found


def _refuse(code: str, message: str) -> DataError:
    return DataError(message, code=f"data_layer.{code}")


def _json(value: Any) -> str | None:
    return None if value is None else canonical(value).decode()


def _one(cursor: sqlite3.Cursor, code: str, message: str) -> None:
    if cursor.rowcount != 1:
        raise _refuse(code, message)


def _tag_key(con: sqlite3.Connection, key: str, version: int | None) -> tuple[str, int]:
    """The vocabulary's lineage and the version a tag is given under: as named, else its current."""
    found = con.execute(
        "SELECT definition_id, max(version) FROM definition WHERE family = 'tag' AND name = ?"
        " AND (version = ? OR (? IS NULL AND status IN ('experimental', 'standard')))",
        (key, version, version),
    ).fetchone()
    if found is None or found[0] is None:
        raise _refuse("tag_unregistered", f"no tag definition registers {key}")
    return str(found[0]), int(found[1])


def _tag(con: sqlite3.Connection, e: sqlite3.Row, key: tuple[str, int], value: Any) -> None:
    con.execute(
        "INSERT OR IGNORE INTO tag (subject_kind, subject_id, key_id, tag_version, value, seq)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (e["subject_kind"], e["subject_id"], *key, str(value), e["seq"]),
    )


def _daily(con: sqlite3.Connection, e: sqlite3.Row, a: dict[str, Any]) -> None:
    cost = a.get("cost_usd")
    con.execute(
        "INSERT INTO daily_span VALUES (date(? / 1000000, 'unixepoch'), ?, ?, ?, ?, ?, 1, ?, ?, ?,"
        " ?, ?, ?) ON CONFLICT (day, actor_kind, actor_id, node_id, type_id, outcome) DO UPDATE SET"
        " n = n + 1, micros = micros + excluded.micros, tokens_in = tokens_in + excluded.tokens_in,"
        " tokens_out = tokens_out + excluded.tokens_out,"
        " cost_microusd = cost_microusd + excluded.cost_microusd,"
        " bytes_read = bytes_read + excluded.bytes_read,"
        " bytes_written = bytes_written + excluded.bytes_written",
        (
            e["occurred_us"],
            e["actor_kind"],
            e["actor_id"],
            require_lineage(con, "node", e["node"]),
            require_lineage(con, "event_type", e["type"]),
            e["phase"],
            e["occurred_us"] - e["started_us"],
            int(a.get("gen_ai.usage.input_tokens", 0)),
            int(a.get("gen_ai.usage.output_tokens", 0)),
            0 if cost is None else round(cost * 1_000_000),
            int(a.get("bytes_read", 0)),
            int(a.get("bytes_written", 0)),
        ),
    )


@on("definition.registered")
def _definition(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    d = json.loads(text or "{}")
    fields = [d.get("status", "experimental"), d.get("relation"), _json(d.get("conversion"))]
    fields += [_json(d.get("spec", {})), d["text"], d.get("code_ref"), _json(d.get("golden"))]
    fields += [d.get("owner", e["node"]), e["seq"], e["seq"], None]
    register(con, d["family"], d["name"], d["version"], *fields)


def _owned(con: Con, e: Row, family: str, name: str, version: int | None) -> str:
    """The lineage a definition event names, refused unless its owner, the person or the host
    acts: the owner of the version named, else of the newest."""
    found = con.execute(
        "SELECT definition_id, owner FROM definition WHERE family = ? AND name = ?"
        " AND (version = ? OR ? IS NULL) ORDER BY version DESC LIMIT 1",
        (family, name, version, version),
    ).fetchone()
    if found is None or (found[1] != e["node"] and e["actor_kind"] not in ("person", "host")):
        raise _refuse(
            "definition_not_owner",
            "a definition is moved or renamed by its owner, the person or the host",
        )
    return str(found[0])


@on("definition.status_changed")
def _definition_status(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    found = _owned(con, e, a["family"], a["name"], a["version"])
    con.execute(
        "UPDATE definition_version SET status = ?, status_seq = ?, status_reason = ?"
        " WHERE definition_id = ? AND version = ? AND status <> ?",
        (a["status"], e["seq"], a.get("reason"), found, a["version"], a["status"]),
    )


@on("definition.renamed")
def _definition_renamed(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    """A lineage takes a new name and keeps its id; every name it held still names it."""
    found = _owned(con, e, a["family"], a["name"], None)
    if lineage(con, a["family"], a["to"]) is not None:
        raise _refuse("definition_name_taken", f"{a['family']} {a['to']} is or was another's name")
    con.execute(
        "UPDATE definition_lineage SET name = ?, named_seq = ? WHERE definition_id = ?",
        (a["to"], e["seq"], found),
    )
    con.execute(
        "INSERT INTO definition_name VALUES (?, ?, ?, ?)", (a["family"], a["to"], found, e["seq"])
    )


@on("workspace.created")
def _workspace(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    _folder_moves(con, e["subject_id"], a.get("folder"))
    con.execute(
        "INSERT INTO workspace"
        " (workspace_id, name, folder, status, created_seq, status_seq)"
        " VALUES (?, ?, ?, 'active', ?, ?)",
        (e["subject_id"], a["name"], a.get("folder"), e["seq"], e["seq"]),
    )


@on("workspace.opened")
def _workspace_opened(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    _folder_moves(con, e["subject_id"], a.get("folder"))
    con.execute(
        "UPDATE workspace SET opened_seq = ?, folder = coalesce(?, folder) WHERE workspace_id = ?",
        (e["seq"], a.get("folder"), e["subject_id"]),
    )


def _folder_moves(con: Con, workspace: str, folder: str | None) -> None:
    """A folder is the last seen of one workspace: another that held it holds no folder now."""
    if folder is not None:
        con.execute(
            "UPDATE workspace SET folder = NULL WHERE folder = ? AND workspace_id <> ?",
            (folder, workspace),
        )


@on("workspace.status_changed")
def _workspace_status(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    """A deleted workspace holds nothing: its live references go, and what no other root holds is
    collected by the age rule (data_layer.governance)."""
    moved = con.execute(
        "UPDATE workspace SET status = ?, status_seq = ? WHERE workspace_id = ? AND status <> ?",
        (a["status"], e["seq"], e["subject_id"], a["status"]),
    )
    if moved.rowcount and a["status"] == "deleted":
        con.execute(
            "DELETE FROM live_ref WHERE root_kind = 'workspace' AND root_id = ?", (e["subject_id"],)
        )


@on("setting.changed")
def _setting(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    con.execute(
        "INSERT OR REPLACE INTO setting VALUES (?, ?, ?, ?, ?)",
        (a["scope"], a["key"], _json(a.get("value")), a.get("secret_ref"), e["seq"]),
    )


@on("goal.set")
def _goal(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    con.execute(
        "INSERT INTO goal VALUES (?, ?, ?, 'open', ?, ?, NULL, NULL)",
        (e["subject_id"], e["workspace_id"], a["title"], a.get("deadline_us"), e["seq"]),
    )


@on("goal.closed")
def _goal_closed(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    con.execute(
        "UPDATE goal SET state = ?, closed_seq = ? WHERE goal_id = ? AND state <> ?",
        (a["state"], e["seq"], e["subject_id"], a["state"]),
    )


@on("task.requested")
def _task_requested(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    """A Task enters QUEUED, or a 0.2.7 Task the upgrade imports at the lifecycle its store held
    (its first save then moves nothing): an admission is not a move."""
    task = e["subject_id"]
    parent, goal, _ = home = task_parent(e, a, lambda g: _goal_held(con, g, e["workspace_id"]))
    held = "SELECT workspace_id FROM task_workspace WHERE task_id = ?"
    found = None if parent is None else con.execute(held, (parent,)).fetchone()
    if parent is not None and (found is None or found[0] != e["workspace_id"]):
        raise _refuse("task_workspace_disagrees", "a sub-task's parent is of its workspace")
    if goal is not None and not _goal_held(con, goal, e["workspace_id"]):
        raise _refuse("task_workspace_disagrees", "a task's goal is of its workspace")
    con.execute(  # its record follows in the same unit (`task.saved`)
        "INSERT INTO task (task_id, parent_task_id, goal_id, workspace_id, type_id, state,"
        " trace_id, product_version, requested_seq, input_hash, plan_hash, lifecycle,"
        " admitted_us, updated_us, admission_sequence, submitted_by_json, state_seq, lifecycle_seq)"
        " VALUES (?, ?, ?, ?, ?, 'requested', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            task,
            *home,
            require_lineage(con, "event_type", a["type"]),
            e["trace_id"],
            a["product_version"],
            e["seq"],
            a["input_hash"],
            a["plan_hash"],
            a.get("lifecycle", "QUEUED"),
            registry_us(a["admitted_at"]),
            registry_us(a["admitted_at"]),
            a.get("admission_sequence"),
            _json(a.get("submitted_by")),
            e["seq"],
            e["seq"],
        ),
    )


def task_parent(
    e: Row, a: Attrs, held: Callable[[str], bool]
) -> tuple[str | None, str | None, str | None]:
    """A Task's one parent, (parent task, goal, workspace): version 2 names its parent task or
    goal. Version 1 named neither; its goal is the one its submitting agent declared, when `held`
    finds that goal in the Task's own workspace, and its workspace otherwise."""
    if e["type_version"] >= 2:
        parent, goal = a.get("parent_task_id"), a.get("goal_id")
    else:
        parent, goal = None, (a.get("submitted_by") or {}).get("goal_id")
        goal = goal if goal is not None and held(goal) else None
    if parent is not None:
        return parent, None, None
    return None, goal, None if goal is not None else e["workspace_id"]


def _goal_held(con: Con, goal: str, workspace: str) -> bool:
    """Whether the log holds `goal` in `workspace`."""
    held = "SELECT 1 FROM goal WHERE goal_id = ? AND workspace_id = ?"
    return con.execute(held, (goal, workspace)).fetchone() is not None


@on("task.saved")
def _task_saved(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    _one(
        _task_status(
            con,
            e["subject_id"],
            e["seq"],
            TASK_STATE.get(a["lifecycle"], "waiting"),
            attempt=True,
            lifecycle=a["lifecycle"],
            record=text,
            updated=registry_us(a["updated_at"]),
        ),
        "task_unknown",
        "a saved record names a Task the log was never asked for",
    )


def _task_status(
    con: Con,
    task: str,
    seq: int,
    state: str,
    *,
    attempt: bool = False,
    lifecycle: str | None = None,
    record: str | None = None,
    updated: int | None = None,
) -> sqlite3.Cursor:
    """Both status guards judge the one update before its attempt and saved record can move."""
    return con.execute(
        "UPDATE task SET state = :state,"
        " state_seq = CASE WHEN state <> :state THEN :seq ELSE state_seq END,"
        " attempts = attempts + (:attempt AND :state = 'running' AND state <> 'running'),"
        " started_seq = CASE WHEN :state = 'running' THEN coalesce(started_seq, :seq)"
        " ELSE started_seq END,"
        " ended_seq = CASE WHEN state = :state THEN ended_seq"
        " WHEN :state IN ('completed', 'failed', 'cancelled') THEN :seq END,"
        " lifecycle = coalesce(:lifecycle, lifecycle),"
        " lifecycle_seq = CASE WHEN lifecycle <> :lifecycle THEN :seq ELSE lifecycle_seq END,"
        " record_json = CASE WHEN :lifecycle IS NULL THEN record_json ELSE :record END,"
        " updated_us = coalesce(:updated, updated_us) WHERE task_id = :task",
        {
            "state": state,
            "task": task,
            "seq": seq,
            "attempt": attempt,
            "lifecycle": lifecycle,
            "record": record,
            "updated": updated,
        },
    )


@on("work_item.saved")
def _work_item_saved(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    """A work item's save is its own home (workspace_task_work_item reads it): its first save
    names its place; each later one moves it only as Task Control declares."""
    registry_us(a["updated_at"])
    prior = con.execute(
        "SELECT json_extract(attrs, '$.lifecycle') FROM event WHERE type = 'work_item.saved'"
        " AND subject_id = ? AND json_extract(attrs, '$.stage_id') = ? AND seq < ?"
        " ORDER BY seq DESC LIMIT 1",
        (e["subject_id"], a["stage_id"], e["seq"]),
    ).fetchone()
    if "ordinal" in a:  # its first save, at the Task's admission, names its place
        if prior is not None:
            raise _refuse("work_item_repeated", "a work item's place is named once")
        return
    if prior is None:
        raise _refuse(
            "work_item_unknown", "a saved work item names a stage its Task was not admitted with"
        )
    if prior[0] != a["lifecycle"] and a["lifecycle"] not in WORK_ITEM_MOVES.get(prior[0], ()):
        raise _refuse(
            "work_item_move_undeclared", "a work item moves only as Task Control declares"
        )


@on("task.noted")
def _task_noted(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    registry_us(a["recorded_at"])  # the note is its event: workspace_task_event reads it


@on("execution.started")
def _execution(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    registry_us(a["started_at"])  # the start is its event: workspace_task_execution reads it


@on("stage.verified")
def _stage_verified(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    registry_us(a["observed_at"])
    con.execute(  # the verified stage is the Task's completed step; its event carries the receipt
        "INSERT OR REPLACE INTO task_step VALUES (?, ?, ?, ?)",
        (e["subject_id"], a["stage_id"], _json({"receipt": a["receipt_hash"]}), e["seq"]),
    )


@on("command.accepted")
def _command_accepted(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    registry_us(a["accepted_at"])  # the command is its event: workspace_task_command reads it


@on("command.resolved")
def _command_resolved(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    registry_us(a["resolved_at"])
    accepted = con.execute(
        "SELECT 1 FROM event WHERE type = 'command.accepted' AND subject_id = ?"
        " AND json_extract(attrs, '$.command_id') = ?",
        (e["subject_id"], a["command_id"]),
    ).fetchone()
    if accepted is None:
        raise _refuse("command_unknown", "a resolution names a command the Task never accepted")


# A work item's declared moves; VERIFIED and CANCELLED are final.
WORK_ITEM_MOVES: Final[dict[str, frozenset[str]]] = {
    "PENDING": frozenset({"IN_PROGRESS", "CANCELLED"}),
    "IN_PROGRESS": frozenset({"READY_FOR_VERIFICATION", "PENDING", "BLOCKED", "CANCELLED"}),
    "READY_FOR_VERIFICATION": frozenset({"VERIFIED", "PENDING", "BLOCKED", "CANCELLED"}),
    "BLOCKED": frozenset({"PENDING", "CANCELLED"}),
}
_REGISTRY_TIME: Final = re.compile(r"\d{4}-\d\d-\d\d[ T]\d\d:\d\d:\d\d(\.\d{6})?(\+00:00|Z)?")
_EPOCH: Final = datetime(1970, 1, 1)


def registry_us(value: object) -> int:
    """A Task Control time in integer microseconds. The registry writes UTC ISO text to the
    microsecond, and a 0.2.7 store's import its naive UTC; the views read each by this rule:
    the second at characters 1-19, then six fraction digits or none."""
    if not isinstance(value, str) or not _REGISTRY_TIME.fullmatch(value):
        raise _refuse("task_time_unreadable", "a Task Control time is UTC ISO, to the microsecond")
    whole = datetime.strptime(value[:19].replace("T", " "), "%Y-%m-%d %H:%M:%S")
    fraction = int(value[20:26]) if value[19:20] == "." else 0
    return (whole - _EPOCH) // timedelta(seconds=1) * 1_000_000 + fraction


@on("task")
def _task(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    phase = e["phase"]
    state = "running" if phase == "started" else phase
    _task_status(con, e["subject_id"], e["seq"], state, attempt=phase == "started")


@on("step.completed")
def _step(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    con.execute(
        "INSERT OR REPLACE INTO task_step VALUES (?, ?, ?, ?)",
        (e["subject_id"], a["step"], _json(a.get("output")), e["seq"]),
    )


@on("wait.started")
def _wait(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    con.execute(
        "INSERT INTO wait VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)",
        (
            e["subject_id"],
            a["task_id"],
            a["kind"],
            None
            if a.get("event_type") is None
            else require_lineage(con, "event_type", a["event_type"]),
            _json(a.get("match")),
            a.get("fires_us"),
            a["purpose"],
            e["seq"],
        ),
    )
    # A watchdog's lease waits for its deadline, not the Task: its work is still running.
    if a["purpose"] != "task.heartbeat":
        con.execute(
            "UPDATE task SET state = 'waiting', state_seq = ?"
            " WHERE task_id = ? AND state = 'running'",
            (e["seq"], a["task_id"]),
        )


@on("wait.resolved")
def _wait_resolved(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    resolved = con.execute(
        "UPDATE wait SET resolved_seq = ?, resolved_by_seq = ?"
        " WHERE wait_id = ? AND resolved_seq IS NULL",
        (e["seq"], a.get("by_seq"), e["subject_id"]),
    )
    if not resolved.rowcount:
        return
    task = con.execute(  # a task resumes only when its last ordinary wait resolves
        "SELECT task.task_id, task.lifecycle FROM task JOIN wait ON wait.task_id = task.task_id"
        " WHERE task.state = 'waiting' AND wait.wait_id = ? AND wait.purpose <> 'task.heartbeat'"
        " AND NOT EXISTS (SELECT 1 FROM wait w"
        " WHERE w.task_id = task.task_id AND w.resolved_seq IS NULL"
        " AND w.purpose <> 'task.heartbeat')",
        (e["subject_id"],),
    ).fetchone()
    if task is not None:
        if task[1] in {"REVIEW_PENDING", "BLOCKED", "CANCEL_REQUESTED", "RECOVERY_REQUIRED"}:
            _status_refusal(con, e, "task.state")
        else:
            _task_status(con, task[0], e["seq"], "running")


SAME_WINDOW: Final = (
    "SELECT selection_id FROM selection WHERE workspace_id = ? AND cutoff_us = ?"
    " AND universe_id IS ? AND valid_from IS ? AND valid_to IS ? AND spec = ?"
)
"""The workspace's selection of an equal window, column by column."""


def window(con: sqlite3.Connection, workspace: str | None, a: Mapping[str, Any]) -> tuple[Any, ...]:
    """A selection's window as its columns, SAME_WINDOW's arguments: the universe by its lineage."""
    universe = a.get("universe")
    universe_id = None if universe is None else require_lineage(con, "universe", universe)
    spec = _json(a.get("spec", {}))
    return (workspace, a["cutoff_us"], universe_id, a.get("valid_from"), a.get("valid_to"), spec)


@on("selection.made")
def _selection(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    columns = window(con, e["workspace_id"], a)
    held = con.execute(SAME_WINDOW, columns).fetchone()
    if held is not None:
        raise _refuse(
            "selection_split", f"the workspace already made selection {held[0]} of this window"
        )
    con.execute(
        "INSERT INTO selection VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (e["subject_id"], *columns, e["seq"]),
    )


def upstream(con: sqlite3.Connection, reads: Iterable[str]) -> list[str]:
    """The recipes a recipe reads: those of the artifacts it reads, sorted; an artifact the log
    does not hold is refused by name."""
    reads = sorted(set(reads))
    found = dict(
        con.execute(
            f"SELECT artifact_id, recipe_key FROM artifact WHERE artifact_id IN"
            f" ({', '.join('?' * len(reads))})",
            reads,
        ).fetchall()
    )
    if unknown := [artifact for artifact in reads if artifact not in found]:
        raise _refuse(
            "recipe_reads_unknown", f"a recipe reads artifacts the log holds, not {unknown}"
        )
    return sorted({key for key in found.values() if key is not None})


@on("recipe.defined")
def _recipe(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    if "upstream" in a:
        raise _refuse(
            "recipe_upstream_given",
            "the Host fills a recipe's upstream from the artifacts it reads",
        )
    reads = upstream(con, a.get("reads", ()))
    method = require_lineage(con, "method", a["method"])
    key = recipe_key(method, a["method_version"], a["params"], reads)
    if a.get("recipe_key") != key:
        raise _refuse(
            "recipe_key_mismatch",
            "a recipe's key is the digest of its method's lineage, version, parameters and"
            " upstream",
        )
    held = con.execute("SELECT recipe_id FROM recipe WHERE recipe_key = ?", (key,)).fetchone()
    if held is not None:
        raise _refuse("recipe_split", f"recipe {held[0]} already defines this approach")
    con.execute(
        "INSERT INTO recipe VALUES (?, ?, ?, ?, ?, ?)",
        (e["subject_id"], key, method, a["method_version"], _json(a["params"]), e["seq"]),
    )
    con.executemany(
        "INSERT INTO recipe_upstream VALUES (?, ?, ?)",
        [(key, position, read) for position, read in enumerate(reads)],
    )


@on("hypothesis.stated")
def _hypothesis(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    """A hypothesis's one parent: its goal when the log holds it in the hypothesis's workspace,
    else that workspace."""
    goal = a.get("goal_id")
    if goal is not None and not _goal_held(con, goal, e["workspace_id"]):
        goal = None
    con.execute(
        "INSERT INTO hypothesis VALUES (?, ?, ?, 'open', ?, ?)",
        (e["subject_id"], goal, None if goal else e["workspace_id"], e["seq"], e["seq"]),
    )


@on("hypothesis.standing_changed")
def _hypothesis_standing(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    con.execute(
        "UPDATE hypothesis SET standing = ?, standing_seq = ?"
        " WHERE hypothesis_id = ? AND standing <> ?",
        (a["standing"], e["seq"], e["subject_id"], a["standing"]),
    )
    if a.get("trial_id"):  # the transition keeps the standing it produced
        _one(
            con.execute(
                "UPDATE trial SET standing_after = ? WHERE trial_id = ? AND hypothesis_id = ?",
                (a["standing"], a["trial_id"], e["subject_id"]),
            ),
            "hypothesis_trial_unknown",
            "the trial that moved a standing tests that hypothesis",
        )


@on("review.completed")
def _reviewed(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    """A review's note closes what it reviews: an ended trial or task is reviewed from its first
    note on (one with none awaits review), and an open goal is done. Its experience is made from
    the note later (`experience.recorded`)."""
    kind, unit, home = e["subject_kind"], e["subject_id"], e["workspace_id"]
    if kind in ("trial", "task"):
        _of_workspace(con, kind, unit, home)
        _one(
            con.execute(
                f"UPDATE {kind} SET reviewed_seq = coalesce(reviewed_seq, ?)"
                f" WHERE {kind}_id = ? AND ended_seq IS NOT NULL",
                (e["seq"], unit),
            ),
            "review_before_its_end",
            f"a {kind} is reviewed once it has ended",
        )
    elif kind == "goal":
        if not _goal_held(con, unit, home):
            raise _refuse("review_unknown_goal", "a review names a goal of its workspace")
        con.execute(
            "UPDATE goal SET state = 'done', closed_seq = ? WHERE goal_id = ? AND state = 'open'",
            (e["seq"], unit),
        )
    trials = a.get("trials", [])
    if trials and (
        kind != "task"
        or con.execute("SELECT type_id FROM task WHERE task_id = ?", (unit,)).fetchone()[0]
        != lineage(con, "event_type", "review")
    ):
        raise _refuse("review_trials_need_a_review_task", "only a review task's note names trials")
    for trial in trials:
        _of_workspace(con, "trial", trial, home)
        ended = "SELECT 1 FROM trial WHERE trial_id = ? AND ended_seq IS NOT NULL"
        if con.execute(ended, (trial,)).fetchone() is None:
            raise _refuse(
                "review_trial_unknown", "a review names trials of the log that have ended"
            )


_SCOPE: Final = {
    "trial": "SELECT workspace_id FROM trial_scope WHERE trial_id = ?",
    "task": "SELECT workspace_id FROM task_workspace WHERE task_id = ?",
}


def _of_workspace(con: Con, kind: str, unit: str, workspace: str | None) -> None:
    """A review names a trial or task of its own workspace: another's is refused before the note
    closes it or names it, so no review crosses a workspace's authority."""
    found = con.execute(_SCOPE[kind], (unit,)).fetchone()
    if found is not None and found[0] != workspace:
        raise _refuse("review_workspace_disagrees", f"a review names a {kind} of its workspace")


def _note_maker(con: Con, note_seq: int) -> tuple[str, str, str | None]:
    """The layer a review's note is of (trial, task or goal), its id and the task that made it."""
    found = con.execute(
        "SELECT subject_kind, subject_id FROM event WHERE seq = ? AND type = 'review.completed'",
        (note_seq,),
    ).fetchone()
    if found is None or found[0] not in ("trial", "task", "goal"):
        raise _refuse(
            "experience_note_unknown", "an experience renders a review of a trial, task or goal"
        )
    kind, unit = found
    if kind == "trial":
        return (
            kind,
            unit,
            con.execute("SELECT task_id FROM trial WHERE trial_id = ?", (unit,)).fetchone()[0],
        )
    return kind, unit, unit if kind == "task" else None


@on("experience.recorded")
def _experience(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    """Version 1 makes an experience of its note's layer and points the layer at it, marking the
    one it replaces; each later version renders the same note again under the same id."""
    experience, version, note = e["subject_id"], a["version"], a["note_seq"]
    kind, unit, task = _note_maker(con, note)
    if a.get("task_id") != task:
        raise _refuse(
            "experience_parent_disagrees", "an experience's parent is the task its note reviewed"
        )
    if version == 1:
        current = con.execute(
            f"SELECT experience_id FROM {kind} WHERE {kind}_id = ?", (unit,)
        ).fetchone()[0]
        con.execute(
            "INSERT INTO experience_made VALUES (?, ?, ?, NULL)", (experience, task, e["seq"])
        )
        if current is not None:
            con.execute(
                "UPDATE experience_made SET replaced_seq = ? WHERE experience_id = ?",
                (e["seq"], current),
            )
        con.execute(f"UPDATE {kind} SET experience_id = ? WHERE {kind}_id = ?", (experience, unit))
        reviewed = con.execute(  # a review task's experience names the trials its note reviewed
            "SELECT j.value FROM event, json_each(event.attrs, '$.trials') j WHERE event.seq = ?",
            (note,),
        ).fetchall()
        con.executemany(
            "INSERT INTO trial_experience VALUES (?, ?, 'reviewed', ?)",
            [(trial, experience, e["seq"]) for (trial,) in reviewed],
        )
    else:
        last = con.execute(
            "SELECT max(version), min(note_seq) FROM experience_version WHERE experience_id = ?",
            (experience,),
        ).fetchone()
        if last[0] != version - 1 or last[1] != note:
            raise _refuse(
                "experience_version_out_of_order",
                "a version follows the last of its experience, of its note",
            )
    con.execute(
        "INSERT INTO experience_version VALUES (?, ?, ?, ?, ?, ?, ?)",
        (experience, version, note, a["body"], a["size"], a["renderer_version"], e["seq"]),
    )


@on("trial.registered")
def _trial_registered(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    """A trial stores neither workspace nor goal: its selection's workspace is its own, and its
    task and hypothesis lie in it; a goal it names (as one version did) is its hypothesis's. A
    disagreement is refused."""
    selection = con.execute(
        "SELECT workspace_id FROM selection WHERE selection_id = ?", (a["selection_id"],)
    ).fetchone()
    if selection is None or selection[0] != e["workspace_id"]:
        raise _refuse(
            "trial_workspace_disagrees", "a trial names a selection made in its workspace"
        )
    task = a.get("task_id")
    if task is not None:
        home = con.execute("SELECT workspace_id FROM task_workspace WHERE task_id = ?", (task,))
        if (found := home.fetchone()) is None or found[0] != e["workspace_id"]:
            raise _refuse("trial_workspace_disagrees", "a trial's task is of its workspace")
    stated = (None, None)
    if a.get("hypothesis_id"):
        stated = con.execute(
            "SELECT workspace_id, goal_id FROM hypothesis_scope WHERE hypothesis_id = ?",
            (a["hypothesis_id"],),
        ).fetchone()
        if stated is None or stated[0] != e["workspace_id"]:
            raise _refuse(
                "trial_workspace_disagrees", "a trial tests a hypothesis of its workspace"
            )
    if "goal_id" in a and a["goal_id"] != stated[1]:
        raise _refuse("trial_goal_disagrees", "a trial works for its hypothesis's goal")
    con.execute(
        "INSERT INTO trial (trial_id, task_id, hypothesis_id, selection_id,"
        " recipe_key, seeds, product_version, model, context, state, registered_seq, state_seq)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'registered', ?, ?)",
        (
            e["subject_id"],
            task,
            a.get("hypothesis_id"),
            a["selection_id"],
            a["recipe_key"],
            _json(a["seeds"]),
            a["product_version"],
            a.get("model"),
            _json(a.get("context", {})),
            e["seq"],
            e["seq"],
        ),
    )


@on("trial")
def _trial(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    unregistered = "a trial runs and ends after its registration"
    if e["phase"] == "started":
        _one(
            con.execute(
                "UPDATE trial SET state = 'running',"
                " state_seq = CASE WHEN state <> 'running' THEN ? ELSE state_seq END"
                " WHERE trial_id = ?",
                (e["seq"], e["subject_id"]),
            ),
            "trial_unregistered",
            unregistered,
        )
        return
    headline = a.get("result_artifact")
    if headline is not None:
        owner = con.execute(
            "SELECT trial_id FROM artifact WHERE artifact_id = ?", (headline,)
        ).fetchone()
        if owner is None or owner[0] != e["subject_id"]:
            raise _refuse(
                "trial_headline_not_its_own", "a completion names an artifact this trial produced"
            )
    _one(
        con.execute(
            "UPDATE trial SET state = :state,"
            " state_seq = CASE WHEN state <> :state THEN :seq ELSE state_seq END,"
            " ended_seq = CASE WHEN state <> :state THEN :seq ELSE ended_seq END,"
            " result_artifact = coalesce(:headline, result_artifact),"
            " candidates = coalesce(:candidates, candidates) WHERE trial_id = :trial",
            {
                "state": e["phase"],
                "seq": e["seq"],
                "headline": headline,
                "candidates": a.get("candidates"),
                "trial": e["subject_id"],
            },
        ),
        "trial_unregistered",
        unregistered,
    )


@on("admission")
def _admission_summary(_con: Con, e: Row, a: Attrs, text: str | None) -> None:
    """Validate an admission's provider profile when it uses the bound-text form."""
    if e["type_version"] != ADMISSION_SUMMARY_TEXT_VERSION:
        return
    provider, named = a["provider"], e["text_digest"]
    # contract: keep the attribute form while old admission writers are still in use.
    if named is None and not isinstance(provider, dict):
        return
    if text is None:
        raise _refuse("event_text_missing", "an admission summary keeps its provider list")
    if not isinstance(provider, dict) or provider.get("digest") != named:
        raise _refuse("event_text_unbound", "the provider digest must match the admission text")
    try:
        providers = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _refuse("event_attrs_form", "an admission's provider list is not JSON") from exc
    count = provider.get("count")
    if (
        not isinstance(providers, list)
        or type(count) is not int
        or count != len(providers)
        or not all(isinstance(name, str) for name in providers)
    ):
        raise _refuse("event_attrs_form", "an admission needs the named count of provider names")
    if secrets_in(providers):
        raise _refuse(
            "event_secret", "an admission provider name would make a credential permanent"
        )


@on("artifact.produced")
def _artifact(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    """An artifact's profile: what it means, how it is identified and where its body is."""
    trial = a.get("trial_id")
    if trial and con.execute("SELECT 1 FROM trial WHERE trial_id = ?", (trial,)).fetchone() is None:
        raise _refuse(
            "artifact_trial_unregistered", "a trial's artifact follows the trial's registration"
        )
    kind = require_lineage(con, "artifact_type", a["type"])
    values = [e["subject_id"], a["identity_kind"], kind, a["type_version"]]
    values += [a["meaning"], a.get("recipe_key"), a.get("selection_id"), trial]
    values += [_json(_artifact_shape(e, a, text)), a.get("valid_from"), a.get("valid_to")]
    values += [a.get("row_count"), a.get("size_bytes"), a.get("content_digest")]
    state = "never_stored" if a["storage"] == "none" else "present"
    con.execute(
        "INSERT INTO artifact (artifact_id, identity_kind, type_id, type_version, meaning,"
        " recipe_key,"
        " selection_id, trial_id, shape, valid_from, valid_to, row_count, size_bytes,"
        f" content_digest, storage, body_state, produced_seq) VALUES ({', '.join('?' * 17)})",
        (*values, a["storage"], state, e["seq"]),
    )
    if a["storage"] != "none":  # a kept trial keeps what it produces later too
        con.execute(
            "INSERT OR IGNORE INTO live_ref SELECT ?, 'trial', trial_id, ? FROM trial"
            " WHERE trial_id = ? AND kept = 1",
            (e["subject_id"], e["seq"], trial),
        )
    if a.get("held"):  # the workspace that keeps it holds it from its first write
        _hold(con, e)


def _artifact_shape(e: Row, a: Attrs, text: str | None) -> Any:
    """Project the v2 shape from bound machine text and preserve historical shapes."""
    if e["type_version"] != ARTIFACT_PROFILE_TEXT_VERSION or e["text_digest"] is None:
        # contract: keep the attribute form while logs can replay producers without profile text.
        return a.get("shape")
    try:
        shape = json.loads(text) if text is not None else None
    except json.JSONDecodeError as exc:
        raise _refuse("artifact_profile_invalid", "artifact profile text is not JSON") from exc
    if not isinstance(shape, dict) or "shape" in a:
        raise _refuse(
            "artifact_profile_invalid", "an artifact profile needs one JSON shape in its bound text"
        )
    if hits := secrets_in(shape):
        raise _refuse("event_secret", f"{', '.join(hits)} would make a credential permanent")
    return shape


@on("result.recorded")
def _result(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    for metric, version, named, value in a["measures"]:
        con.execute(
            "INSERT INTO measure VALUES (?, ?, ?, ?, ?, ?)",
            (
                e["subject_id"],
                require_lineage(con, "metric", metric),
                version,
                dims(named),
                value,
                e["seq"],
            ),
        )
    owner = con.execute(
        "SELECT trial_id FROM artifact WHERE artifact_id = ?", (e["subject_id"],)
    ).fetchone()
    if owner is None:
        raise _refuse("result_artifact_unknown", "measures follow their artifact's profile")
    if owner[0]:  # the trial is the artifact's own, never a caller's claim
        con.execute(
            "UPDATE trial SET reads_legacy = max(reads_legacy, ?) WHERE trial_id = ?",
            (int(bool(a.get("reads_legacy"))), owner[0]),
        )


@on("artifact.collected")
def _collected(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    moved = con.execute(
        "UPDATE artifact SET body_state = 'collected', collected_seq = ? WHERE artifact_id = ?"
        " AND body_state <> 'collected'",
        (e["seq"], e["subject_id"]),
    )
    if moved.rowcount:
        con.execute("DELETE FROM artifact_body WHERE artifact_id = ?", (e["subject_id"],))
        con.execute("DELETE FROM live_ref WHERE artifact_id = ?", (e["subject_id"],))


@on("prediction.registered")
def _prediction(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    """A prediction's one parent: its trial, of the event's workspace, else that workspace."""
    trial = a.get("trial_id")
    if trial is not None:
        held = con.execute(
            "SELECT workspace_id FROM trial_scope WHERE trial_id = ?", (trial,)
        ).fetchone()
        if held is None or held[0] != e["workspace_id"]:
            raise _refuse(
                "prediction_workspace_disagrees", "a prediction names a trial of its workspace"
            )
    con.execute(
        "INSERT INTO prediction (prediction_id, workspace_id, trial_id, target, horizon_us,"
        " confidence, registered_seq) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            e["subject_id"],
            None if trial is not None else e["workspace_id"],
            trial,
            _json(a["target"]),
            a["horizon_us"],
            a["confidence"],
            e["seq"],
        ),
    )


@on("outcome.observed")
def _outcome(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    if e["subject_kind"] == "prediction":
        con.execute(
            "UPDATE prediction SET outcome_seq = ? WHERE prediction_id = ? AND outcome_seq IS NULL",
            (e["seq"], e["subject_id"]),
        )


@on("prediction.scored")
def _scored(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    con.execute(
        "UPDATE prediction SET score = ?, score_metric_id = ?, score_metric_version = ?,"
        " outcome_seq = coalesce(outcome_seq, ?) WHERE prediction_id = ?",
        (
            a["score"],
            require_lineage(con, "metric", a["metric"]),
            a["metric_version"],
            e["seq"],
            e["subject_id"],
        ),
    )


@on("trial.kept")
def _kept(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    con.execute("UPDATE trial SET kept = 1 WHERE trial_id = ?", (e["subject_id"],))
    con.execute(
        "INSERT OR IGNORE INTO live_ref SELECT artifact_id, 'trial', trial_id, ? FROM artifact"
        " WHERE trial_id = ? AND body_state = 'present'",
        (e["seq"], e["subject_id"]),
    )


@on("book.activated")
def _book(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    con.execute(
        "INSERT INTO book (book_id, workspace_id, mode, state, holding_artifact,"
        " activated_seq, state_seq) VALUES (?, ?, ?, 'active', ?, ?, ?)",
        (
            e["subject_id"],
            e["workspace_id"],
            a["mode"],
            a.get("holding_artifact"),
            e["seq"],
            e["seq"],
        ),
    )
    if a.get("holding_artifact"):
        con.execute(
            "INSERT OR IGNORE INTO live_ref VALUES (?, 'book', ?, ?)",
            (a["holding_artifact"], e["subject_id"], e["seq"]),
        )


@on("book.rebalanced")
def _book_rebalanced(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    _one(
        con.execute(
            "UPDATE book SET holding_artifact = ? WHERE book_id = ? AND state = 'active'",
            (a["holding_artifact"], e["subject_id"]),
        ),
        "book_not_active",
        "only an active book follows new holdings",
    )
    con.execute("DELETE FROM live_ref WHERE root_kind = 'book' AND root_id = ?", (e["subject_id"],))
    con.execute(
        "INSERT INTO live_ref VALUES (?, 'book', ?, ?)",
        (a["holding_artifact"], e["subject_id"], e["seq"]),
    )


@on("book.retired")
def _book_retired(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    moved = con.execute(
        "UPDATE book SET state = 'retired', retired_seq = ?, state_seq = ?"
        " WHERE book_id = ? AND state <> 'retired'",
        (e["seq"], e["seq"], e["subject_id"]),
    )
    if moved.rowcount:
        con.execute(
            "DELETE FROM live_ref WHERE root_kind = 'book' AND root_id = ?", (e["subject_id"],)
        )


@on("artifact.held")
def _held(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    """A workspace keeps an artifact's body: one live_ref per workspace, however often it holds."""
    _holdable(con, e)
    _hold(con, e)


@on("artifact.pinned")
def _pinned(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    """The person's pin holds an artifact's body: one live_ref per pin."""
    _holdable(con, e)
    con.execute(
        "INSERT OR IGNORE INTO live_ref VALUES (?, 'pin', ?, ?)",
        (e["subject_id"], a["pin_id"], e["seq"]),
    )


@on("artifact.unpinned")
def _unpinned(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    con.execute(
        "DELETE FROM live_ref WHERE artifact_id = ? AND root_kind = 'pin' AND root_id = ?",
        (e["subject_id"], a["pin_id"]),
    )


def _holdable(con: Con, e: Row) -> None:
    """A hold names an artifact whose body is present: a collected one is gone for good."""
    known = con.execute(
        "SELECT body_state FROM artifact WHERE artifact_id = ?", (e["subject_id"],)
    ).fetchone()
    if known is None:
        raise _refuse("artifact_held_unknown", "a hold follows its artifact's profile")
    if known[0] == "collected":
        raise _refuse("artifact_held_collected", "a collected body is gone; produce it again")


def _hold(con: Con, e: Row) -> None:
    if e["workspace_id"] is None:
        raise _refuse("artifact_holder_unnamed", "a workspace holds an artifact; name it")
    con.execute(
        "INSERT OR IGNORE INTO live_ref VALUES (?, 'workspace', ?, ?)",
        (e["subject_id"], e["workspace_id"], e["seq"]),
    )


@on("publication.released")
def _publication(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    con.execute(
        "INSERT OR IGNORE INTO live_ref VALUES (?, 'publication', ?, ?)",
        (e["subject_id"], e["event_id"], e["seq"]),
    )


@on("ledger.posted")
def _ledger(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    for line, (account, asset_kind, asset_id, quantity_n) in enumerate(a["lines"], 1):
        con.execute(
            "INSERT INTO ledger_entry VALUES (?, ?, ?, ?, ?, ?, ?)",
            (e["seq"], line, e["subject_id"], account, asset_kind, asset_id, quantity_n),
        )
    if con.execute("SELECT 1 FROM ledger_unbalanced WHERE seq = ?", (e["seq"],)).fetchone():
        raise _refuse("ledger_unbalanced", "the lines of one posting sum to zero for each asset")


@on("tag.added")
def _tag_added(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    _tag(con, e, _tag_key(con, a["key"], a.get("version")), a["value"])


@on("tag.removed")
def _tag_removed(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    key_id, _version = _tag_key(con, a["key"], a.get("version"))
    con.execute(
        "DELETE FROM tag WHERE subject_kind = ? AND subject_id = ? AND key_id = ? AND value = ?",
        (e["subject_kind"], e["subject_id"], key_id, str(a["value"])),
    )


@on("record.forgotten")
def _forgotten(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    """Forget an event's text only when replay does not need the text to rebuild it."""
    target = con.execute(
        "SELECT type, type_version, text_digest FROM event WHERE seq = ?", (a["seq"],)
    ).fetchone()
    if (
        target is None
        or target[0] in MACHINE_TEXT
        or (tuple(target)[:2] in BOUND_TEXT_EVENTS and target[2])
    ):
        raise _refuse(
            "record_unforgettable",
            "machine text rebuilds definitions, Task records, artifact profiles"
            " and admission summaries",
        )
    con.execute("DELETE FROM event_text WHERE seq = ?", (a["seq"],))  # the event and digest stay


@on("chain.checkpointed")
def _checkpoint(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    con.execute(
        "INSERT INTO chain_checkpoint VALUES (?, ?, ?, ?, ?, ?)",
        (
            a["seq"],
            a["kind"],
            a["head"],
            e["occurred_us"],
            a.get("note"),
            a.get("chain_version", CHAIN_VERSION),
        ),
    )


@on("knowledge.revision_published")
def _knowledge_revision(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    """Rebuild source identity, version and workspace membership in the publication's unit."""
    payload = json.loads(text or "{}")
    source, version = a["source_id"], a["version_id"]
    con.execute("INSERT OR IGNORE INTO knowledge_source VALUES (?, ?)", (source, a["kind"]))
    con.execute(
        "UPDATE knowledge_source_version SET superseded_us = ? WHERE source_id = ?"
        " AND version_id != ? AND known_us < ?"
        " AND (superseded_us IS NULL OR superseded_us > ?)",
        (a["known_us"], source, version, a["known_us"], a["known_us"]),
    )
    con.execute(
        "INSERT OR IGNORE INTO knowledge_source_version VALUES (?, ?, ?, ?,"
        " (SELECT min(boundary) FROM ("
        " SELECT known_us AS boundary FROM knowledge_source_version"
        " WHERE source_id = ? AND known_us > ?"
        " UNION ALL SELECT CAST(json_extract(attrs, '$.known_us') AS INTEGER)"
        " FROM event WHERE type = 'knowledge.source_retired' AND subject_id = ?"
        " AND CAST(json_extract(attrs, '$.known_us') AS INTEGER) >= ?)),"
        " ?, ?, ?)",
        (
            version,
            source,
            a["version"],
            a["known_us"],
            source,
            a["known_us"],
            source,
            a["known_us"],
            a["body_digest"],
            e["seq"] if a["kind"] == "reference" else None,
            e["seq"],
        ),
    )
    con.execute(
        "INSERT INTO knowledge_workspace_source VALUES (?, ?, ?, ?, ?, ?)",
        (
            a["membership_id"],
            e["workspace_id"],
            a["document_id"],
            a["revision"],
            version,
            canonical(payload["metadata"]).decode(),
        ),
    )


@on("knowledge.source_retired")
def _knowledge_retired(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    """End only versions current at the authority's retirement time."""
    con.execute(
        "UPDATE knowledge_source_version SET superseded_us = ? WHERE source_id = ?"
        " AND known_us <= ? AND (superseded_us IS NULL OR superseded_us > ?)",
        (a["known_us"], e["subject_id"], a["known_us"], a["known_us"]),
    )


@on("knowledge.snapshot_published")
def _knowledge_snapshot(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    """Rebuild the snapshot's ordered, constrained membership from its sealed text."""
    con.execute(
        "INSERT INTO knowledge_snapshot VALUES (?, ?, ?, ?)",
        (e["subject_id"], a["reference_id"], e["workspace_id"], e["seq"]),
    )
    con.execute(
        "INSERT INTO knowledge_snapshot_revision"
        " SELECT json_extract(value, '$.membership_id'), ?, json_extract(value, '$.version_id'),"
        " CAST(key AS INTEGER) FROM json_each(?, '$.members')",
        (e["subject_id"], text),
    )


def differences(a: sqlite3.Connection, b: sqlite3.Connection) -> list[str]:
    """The projected tables whose rows differ between two logs; ALL EQUAL is an empty list."""

    def rows(con: sqlite3.Connection, table: str) -> list[tuple[Any, ...]]:
        return sorted((tuple(row) for row in con.execute(f"SELECT * FROM {table}")), key=repr)

    return [table for table in PROJECTED if rows(a, table) != rows(b, table)]
