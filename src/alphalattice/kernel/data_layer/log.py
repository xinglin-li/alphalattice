"""The Host's one write path into log.sqlite, and the lock that makes it the only one.

One Host per machine holds the data root's lock (machine/host.lock) and is the only writer; any
other process reads through `attach`, read-only, and a second `HostLog.open` is refused by name
with the holder it found. Every row of log.sqlite comes from `HostLog.append`: it admits the event
(admit.py), then writes the event, its text, the chain's new head, its projection and an inline
artifact's body as one unit, all of it or none of it.

Ordinary events commit in batches: a batch commits at BATCH_SIZE events, BATCH_SECONDS after its
first, at `flush` and at `close`. Each event is its own savepoint inside the batch, so a refused
event leaves the batch's others whole. An event whose type is durable in the registry commits at
once, with the batch before it. Every commit is under synchronous FULL, so what has committed
survives a lost machine, not only a killed process. A crash loses at most the open batch, never
part of an event, and the chain verifies after it. `append` returns before an ordinary event's
batch commits, so the Host calls `flush` before it returns anything to an agent: a reply never
names an event a crash could still lose.

The log keeps its own clock (`clock`): a live event's recorded_us is the larger of the wall clock
and the last value plus one microsecond, so it never steps back when the wall clock does.

Each event type's projection is project.py's; `replay` copies the events alone into an empty log
through the same write and rebuilds every projected table ALL EQUAL.
"""

from __future__ import annotations

import json
import os
import secrets
import sqlite3
import sys
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import IO, Any, Final

from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer.admit import admit
from alphalattice.kernel.data_layer.chain import (
    CHAIN_VERSION,
    ENVELOPE,
    LINKS,
    canonical,
    digest,
    fold,
    fold_segments,
)
from alphalattice.kernel.data_layer.project import ENDED, project
from alphalattice.kernel.data_layer.root import build_log

BATCH_SIZE: Final = 64
"""Events per batch commit: appends per second level off from 64 (12.7k at 64, 13.1k at 256).
FULL costs a tenth of that against NORMAL (14.6k against 16.6k at 64 on the workshop PC)."""
BATCH_SECONDS: Final = 0.2
"""The longest an ordinary event waits for its batch's commit."""

LOCK_NAME: Final = "host.lock"
LOCK_AT: Final = 1 << 30
"""The byte a Windows Host locks: past any name the lock file holds."""


def uuid7(now_ns: int | None = None) -> str:
    """A UUIDv7 (RFC 9562): 48 bits of Unix milliseconds, then 74 random bits."""
    ms = (time.time_ns() if now_ns is None else now_ns) // 1_000_000
    value = (ms & (1 << 48) - 1) << 80 | 0x7 << 76 | secrets.randbits(12) << 64
    return str(uuid.UUID(int=value | 0b10 << 62 | secrets.randbits(62)))


def envelope(
    type_: str,
    subject: tuple[str | None, str | None] = (None, None),
    attrs: Mapping[str, Any] | None = None,
    *,
    text: str | None = None,
    phase: str | None = None,
    workspace: str | None = None,
    actor: tuple[str, str] = ("host", "host"),
    node: str = "kernel",
    trace: str | None = None,
    span: str | None = None,
    parent_span: str | None = None,
    cause: int | None = None,
    started_us: int | None = None,
    occurred_us: int | None = None,
) -> dict[str, Any]:
    """A new event's envelope, with a UUIDv7 id and, when it carries text, the text's digest."""
    now = time.time_ns() // 1000
    at = now if occurred_us is None else occurred_us
    return {
        "event_id": uuid7(),
        "type": type_,
        "type_version": 1,
        "phase": phase,
        "occurred_us": at,
        "started_us": (at if started_us is None else started_us) if phase in ENDED else None,
        "recorded_us": now,
        "actor_kind": actor[0],
        "actor_id": actor[1],
        "node": node,
        "workspace_id": workspace,
        "subject_kind": subject[0],
        "subject_id": subject[1],
        "trace_id": trace or uuid.uuid4().hex,
        "span_id": span or (secrets.token_hex(8) if phase else None),
        "parent_span_id": parent_span,
        "cause_seq": cause,
        "attrs": canonical(dict(attrs or {})).decode(),
        "text_digest": None if text is None else digest(text.encode()),
        "legacy": 0,
    }


class HostLog:
    """The machine's one writer of log.sqlite; open it with `HostLog.open`."""

    def __init__(self, con: sqlite3.Connection, lock: IO[bytes], *, batch: int, seconds: float):
        """Hold `con` and the data root's `lock`; `open` is the way to make one."""
        self._con, self._lock_file = con, lock
        self._batch, self._seconds = batch, seconds
        self._guard = threading.RLock()
        self._pending = 0
        self._timer: threading.Timer | None = None
        last = con.execute("SELECT recorded_us FROM event ORDER BY seq DESC LIMIT 1").fetchone()
        self._last_us = -1 if last is None else int(last[0])

    @classmethod
    def open(
        cls,
        root: Path,
        *,
        batch: int = BATCH_SIZE,
        seconds: float = BATCH_SECONDS,
        now_us: int | None = None,
    ) -> HostLog:
        """Take the data root's lock and open its log as the only writer.

        It verifies the chain from its last checkpoint, seeds a new log with `registry.seeded`,
        and checkpoints the head in a month that has none. Refused by name:
        `data_layer.host_running` when another Host holds the lock, `data_layer.chain_broken` when
        the log no longer matches its head.
        """
        lock = _take_lock(root / "machine" / LOCK_NAME)
        try:
            con = sqlite3.connect(
                root / "machine" / "log.sqlite", isolation_level=None, check_same_thread=False
            )
            con.row_factory = sqlite3.Row
            con.execute("PRAGMA foreign_keys = ON")
            con.execute("PRAGMA synchronous = FULL")
            if not verify_from_checkpoint(con):
                con.close()
                raise DataError(
                    f"{root / 'machine' / 'log.sqlite'} does not match its chain head; restore "
                    "it from its last backup",
                    code="data_layer.chain_broken",
                )
        except BaseException:
            lock.close()
            raise
        log = cls(con, lock, batch=batch, seconds=seconds)
        if con.execute("SELECT 1 FROM event LIMIT 1").fetchone() is None:
            count = con.execute("SELECT count(*) FROM definition_version").fetchone()[0]
            log.append(envelope("registry.seeded", attrs={"count": count}), None)
        log._checkpoint_month(time.time_ns() // 1000 if now_us is None else now_us)
        log.flush()
        return log

    def _checkpoint_month(self, now_us: int) -> None:
        """Keep the head at its moment once a month, so start-up folds at most a month of events."""
        month = "strftime('%Y-%m', ? / 1000000, 'unixepoch')"
        newest = "SELECT max(at_us) FROM chain_checkpoint WHERE kind = 'month'"
        if self._con.execute(
            f"SELECT {month} = {month}", (now_us, self._newest(newest))
        ).fetchone()[0]:
            return
        seq, head = self._con.execute("SELECT seq, head FROM chain_head WHERE id = 1").fetchone()
        attrs = {"kind": "month", "seq": seq, "head": head, "chain_version": CHAIN_VERSION}
        self.append(envelope("chain.checkpointed", attrs=attrs, occurred_us=now_us), None)

    def _newest(self, sql: str) -> int:
        found = self._con.execute(sql).fetchone()[0]
        return -1 if found is None else int(found)

    def append(
        self,
        row: Mapping[str, Any],
        text: str | None = None,
        body: bytes | None = None,
        *,
        live: bool = True,
    ) -> int:
        """Admit one event and write it as one unit; returns its seq.

        The unit is the event, its text, the chain, its projection and an inline artifact's body.
        A durable type is committed on return; an ordinary one with its batch, so a caller that
        replies on it flushes first. A live event's recorded_us is the log's clock; replay
        (live=False) keeps the one it had.
        """
        with self._guard:
            con = self._con
            if live:
                row = {**row, "recorded_us": self.clock()}
            spec = _spec(con, row)
            admit(row, text, spec, _superseded(con), live=live, body=body)
            durable = bool(spec.get("durable"))
            if self._pending == 0:
                con.execute("BEGIN IMMEDIATE")
                self._arm()
            con.execute("SAVEPOINT event")
            try:
                seq = _write(con, row, text, body, live=live)
            except BaseException:
                con.execute("ROLLBACK TO event")
                con.execute("RELEASE event")
                if self._pending == 0:
                    self._end("ROLLBACK")
                raise
            con.execute("RELEASE event")
            self._pending += 1
            if durable or self._pending >= self._batch:
                self._commit()
            return seq

    def clock(self, at_least: int = 0) -> int:
        """The log's next moment, in microseconds: the larger of the wall clock, the last moment
        plus one and `at_least`. It never steps back, so neither does anything stamped with it."""
        with self._guard:
            self._last_us = max(time.time_ns() // 1000, self._last_us + 1, at_least)
            return self._last_us

    def record(
        self,
        type_: str,
        subject: tuple[str | None, str | None] = (None, None),
        attrs: Mapping[str, Any] | None = None,
        *,
        text: str | None = None,
        body: bytes | None = None,
        **more: Any,
    ) -> int:
        """Append a new event of `type_`, its envelope made by `envelope`."""
        return self.append(envelope(type_, subject, attrs, text=text, **more), text, body)

    def flush(self) -> None:
        """Commit the open batch, if any; the Host's reply waits for it."""
        with self._guard:
            self._commit()

    def read(self, sql: str, args: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        """Rows as the Host sees them, its open batch included."""
        with self._guard:
            return self._con.execute(sql, args).fetchall()

    def queue_deletion(self, handle: str, kind: str, error: str) -> None:
        """Record a file another process holds open, or count one more failed try at it.

        `deletion_queue` is the Host's working state, not a projection: no event writes it.
        """
        with self._guard:
            self._con.execute(
                "INSERT INTO deletion_queue VALUES (?, ?, ?, 1, ?) ON CONFLICT (handle) DO UPDATE"
                " SET attempts = attempts + 1, last_error = excluded.last_error",
                (handle, kind, self.clock(), error),
            )

    def dequeue_deletion(self, handle: str) -> None:
        """Forget a queued file once it is gone."""
        with self._guard:
            self._con.execute("DELETE FROM deletion_queue WHERE handle = ?", (handle,))

    def close(self) -> None:
        """Commit the open batch and release the data root's lock."""
        with self._guard:
            self._commit()
            self._con.close()
            self._lock_file.close()

    def __enter__(self) -> HostLog:
        """The log itself; leaving the block closes it."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close the log, committing its batch and releasing the lock."""
        self.close()

    def _arm(self) -> None:
        self._timer = threading.Timer(self._seconds, self.flush)
        self._timer.daemon = True
        self._timer.start()

    def _commit(self) -> None:
        if self._pending:
            self._end("COMMIT")

    def _end(self, statement: str) -> None:
        self._con.execute(statement)
        self._pending = 0
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None


def attach(root: Path) -> sqlite3.Connection:
    """A read-only connection to the data root's log, for any process but the Host."""
    uri = f"file:{(root / 'machine' / 'log.sqlite').as_posix()}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    con.row_factory = sqlite3.Row
    return con


def verify_from_checkpoint(con: sqlite3.Connection) -> bool:
    """Whether the events after the last checkpoint fold from its head to the stored head."""
    stored = con.execute("SELECT head, chain_version FROM chain_head WHERE id = 1").fetchone()
    if stored is None:
        return con.execute("SELECT 1 FROM event LIMIT 1").fetchone() is None
    last = con.execute("SELECT seq, head FROM chain_checkpoint ORDER BY seq DESC LIMIT 1")
    start, head = last.fetchone() or (0, "")
    events = con.execute("SELECT * FROM event WHERE seq > ? ORDER BY seq", (start,))
    return bool(fold(head, events, stored[1]) == stored[0])


def verify_whole(con: sqlite3.Connection) -> bool:
    """Whether the whole log, each segment under its checkpoint's encoding, folds to the head."""
    stored = con.execute("SELECT head, chain_version FROM chain_head WHERE id = 1").fetchone()
    if stored is None:
        return con.execute("SELECT 1 FROM event LIMIT 1").fetchone() is None
    bounds = con.execute("SELECT seq, chain_version FROM chain_checkpoint ORDER BY seq").fetchall()
    events = con.execute("SELECT * FROM event ORDER BY seq")
    return bool(fold_segments(events, [(b[0], b[1]) for b in bounds], stored[1]) == stored[0])


def hold(path: Path) -> IO[bytes] | None:
    """An exclusive hold on `path`, kept while the returned handle is open; None when another
    handle, in this process or another, holds it, or when the file it locked is no longer the one
    at `path` (`drop` removed it between the open and the lock). The holder's death releases it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b", buffering=0)
    try:
        lock_byte(handle)
        if os.path.samestat(os.fstat(handle.fileno()), os.stat(path)):
            return handle
    except OSError:
        pass  # held by another, or removed after the open: no hold either way
    handle.close()
    return None


def drop(handle: IO[bytes], path: Path) -> None:
    """Give up a hold and remove its file, so no later `hold` takes a file no longer at `path`.

    POSIX removes the file while it is still held. Windows cannot remove an open file, so the
    hold ends first; a handle that opened the file meanwhile keeps it, since the removal fails.
    """
    if sys.platform != "win32":
        path.unlink(missing_ok=True)
        handle.close()
        return
    handle.close()
    try:
        path.unlink(missing_ok=True)
    except PermissionError:
        return  # another handle opened it after the hold ended: the file is that holder's now


def _take_lock(path: Path) -> IO[bytes]:
    """The data root's lock, held by its open handle for the Host's life.

    The locked byte lies past the file's end (LOCK_AT), so the holder's name at its start stays
    readable to the process the refusal names it to.
    """
    handle = hold(path)
    if handle is None:
        holder = path.read_bytes()[:512].decode(errors="replace").strip() or "an unnamed process"
        raise DataError(
            f"another Host ({holder}) holds this data root and is its only writer; read it "
            "through data_layer.log.attach, or stop that Host first",
            code="data_layer.host_running",
        )
    handle.truncate(0)
    handle.write(f"pid {os.getpid()}, {Path(sys.argv[0]).name or 'python'}".encode())
    return handle


def lock_byte(handle: IO[bytes]) -> None:
    """Lock one byte of `handle`'s file without waiting; OSError when another handle has it."""
    if sys.platform == "win32":
        import msvcrt  # lazy: the module exists only on Windows

        handle.seek(LOCK_AT)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl  # lazy: the module exists only on POSIX

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _spec(con: sqlite3.Connection, row: Mapping[str, Any]) -> dict[str, Any]:
    found = con.execute(
        "SELECT spec FROM definition WHERE family = 'event_type' AND name = ? AND version = ?",
        (row["type"], row["type_version"]),
    ).fetchone()
    if found is None:
        raise DataError(
            f"{row['type']} v{row['type_version']} is not a registered event type",
            code="data_layer.event_type_unregistered",
        )
    spec: dict[str, Any] = json.loads(found[0])
    return spec


def _superseded(con: sqlite3.Connection) -> Callable[[str, Any, Any], bool]:
    def superseded(family: str, name: Any, version: Any) -> bool:
        sql = "SELECT 1 FROM definition WHERE family = ? AND name = ? AND version = ?"
        found = con.execute(f"{sql} AND status = 'superseded'", (family, name, version))
        return found.fetchone() is not None

    return superseded


def _write(
    con: sqlite3.Connection,
    row: Mapping[str, Any],
    text: str | None,
    body: bytes | None,
    *,
    live: bool,
) -> int:
    """The unit `append` commits: the event, its text, the chain, the projection, the body."""
    columns = [column for column in ENVELOPE if column in row]
    insert = f"INSERT INTO event ({', '.join(columns)}) VALUES ({', '.join('?' * len(columns))})"
    seq = con.execute(insert, [row[column] for column in columns]).lastrowid
    assert seq is not None  # an INSERT's rowid
    if text is not None:
        con.execute("INSERT INTO event_text VALUES (?, ?)", (seq, text))
    stored = con.execute("SELECT * FROM event WHERE seq = ?", (seq,)).fetchone()
    _fold(con, stored)
    project(con, stored, text)
    _body(con, stored, body, live=live)
    return seq


def _fold(con: sqlite3.Connection, event: sqlite3.Row) -> None:
    head = con.execute("SELECT head FROM chain_head WHERE id = 1").fetchone()
    new = LINKS[CHAIN_VERSION](head[0] if head else "", event)
    con.execute(
        "INSERT OR REPLACE INTO chain_head VALUES (1, ?, ?, ?)", (event["seq"], new, CHAIN_VERSION)
    )


def _body(con: sqlite3.Connection, event: sqlite3.Row, body: bytes | None, *, live: bool) -> None:
    attrs = json.loads(event["attrs"])
    inline = event["type"] == "artifact.produced" and attrs.get("storage") == "inline"
    if body is not None and not inline:
        raise DataError(
            "only an inline artifact carries a body", code="data_layer.artifact_body_unexpected"
        )
    if inline and body is None and live:
        raise DataError(
            "an inline artifact is written with its body", code="data_layer.artifact_body_missing"
        )
    if body is not None:
        con.execute("INSERT INTO artifact_body VALUES (?, ?)", (event["subject_id"], body))


def replay(source: sqlite3.Connection, path: Path) -> sqlite3.Connection:
    """A new log at `path` holding `source`'s events, each written again through the one write.

    Nothing but the events, their texts and inline bodies is copied, one event at a time; every
    projected table is rebuilt from them, so `project.differences(source, replayed)` is empty
    when the log is the only truth. A forgotten text is admitted as it was once (live=False).
    """
    con = sqlite3.connect(path, isolation_level=None)
    con.row_factory = sqlite3.Row
    build_log(con)
    con.execute("PRAGMA foreign_keys = ON")
    rows = source.cursor()
    rows.row_factory = sqlite3.Row  # whatever rows the caller's connection makes
    rows.execute(
        "SELECT e.*, t.text, b.body FROM event e LEFT JOIN event_text t ON t.seq = e.seq"
        " LEFT JOIN artifact_body b ON e.type = 'artifact.produced' AND b.artifact_id ="
        " e.subject_id ORDER BY e.seq"
    )
    con.execute("BEGIN IMMEDIATE")
    try:
        for event in rows:
            row = {column: event[column] for column in ENVELOPE}
            text, body = event["text"], event["body"]
            admit(row, text, _spec(con, row), _superseded(con), live=False, body=body)
            _write(con, row, text, body, live=False)
        con.execute("COMMIT")
    except BaseException:
        con.execute("ROLLBACK")
        con.close()
        raise
    return con
