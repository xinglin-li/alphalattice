"""The Host's one write path into log.sqlite, and the lock that makes it the only one.

One Host per data root holds its lock (machine/host.lock) and is the only writer; any
other process reads through `attach`, read-only, and a second `HostLog.open` is refused by name
with the holder it found. Every row of log.sqlite comes from `HostLog.append`: it admits the event
(admit.py), then writes the event, its text, the chain's new head, its projection and an inline
artifact's body as one unit, all of it or none of it.

Ordinary events commit in batches: a batch commits at BATCH_SIZE events, BATCH_SECONDS after its
first, at `flush` and at `close`. Each event is its own savepoint inside the batch, so a refused
event leaves the batch's others whole. An event whose type is durable in the registry commits at
once, with the batch before it. The log runs WAL under synchronous NORMAL, so every commit
survives a killed process; a commit that holds a durable event also syncs the WAL, as FULL would,
so it and every commit before it survive a lost machine. A lost machine loses at most the
ordinary commits after the last durable one, never part of an event or a gap before a kept one,
and the chain verifies after it. `append` returns before an ordinary event's
batch commits, so the Host calls `flush` before it returns anything to an agent: a reply never
names an event a crash could still lose. An event may carry what its writer does once its fate is
known (`committed`): called with True after the commit that holds it, with False when it is rolled
back. Every action of a commit runs, though one raises; the first error is raised after them.

The log keeps its own clock (`clock`): a live event's recorded_us is the larger of the wall clock
and the last value plus one microsecond, so it never steps back when the wall clock does.

Each event type's projection lives in project.py. `replay` rebuilds the projection through the
same write. Replay decodes historical first registrations and reads minted ids from the events.
"""

from __future__ import annotations

import json
import os
import secrets
import sqlite3
import sys
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import IO, Any, Final

from alphalattice.kernel.clock import wall_time_ns
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
from alphalattice.kernel.data_layer.project import (
    BOUND_TEXT_EVENTS,
    ENDED,
    definition_names,
    lineage,
    project,
    require_lineage,
)
from alphalattice.kernel.data_layer.project import uuid7 as uuid7
from alphalattice.kernel.data_layer.root import build_log, seed_rows, sqlite_read_only_uri

BATCH_SIZE: Final = 64
"""Events per batch commit: appends per second level off from 64 (12.7k at 64, 13.1k at 256).
A WAL sync costs about 0.35 to 0.6 ms a commit against 0.01 ms without one (the workshop PC),
so only a durable event's commit pays it."""
BATCH_SECONDS: Final = 0.2
"""The longest an ordinary event waits for its batch's commit."""

COMMIT_TIMER: Final = "host-log-commit"
"""The name of the thread that commits an open batch once it is `BATCH_SECONDS` old."""
LOCK_NAME: Final = "host.lock"
ATTRS_CHARACTERS: Final = 4096
"""The most characters an event's attributes hold (log.sql's CHECK on `attrs`)."""
LOCK_AT: Final = 1 << 30
"""The byte a Windows Host locks: past any name the lock file holds."""


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
    version: int | None = None,
) -> dict[str, Any]:
    """A new event's envelope, of its type's `version` as given, else the one registered when the
    log writes it (`HostLog.append`), with a UUIDv7 id and, when it carries text, the text's
    digest. A finished step names its start: one with none is refused by name, never counted as
    no time."""
    if phase in ENDED and started_us is None:
        raise DataError(
            f"a finished {type_} step names when it started",
            code="data_layer.step_unstarted",
        )
    now = wall_time_ns() // 1000
    at = now if occurred_us is None else occurred_us
    return {
        "event_id": uuid7(),
        "type": type_,
        "type_version": version,
        "phase": phase,
        "occurred_us": at,
        "started_us": started_us if phase in ENDED else None,
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
        self._guard = RLock()
        con.create_function("uuid7", 0, uuid7)
        self.knowledge_cache_lock = threading.Lock()
        # Claim an embedding miss before inference, outside the SQLite writer. Every cache
        # reader of this Host shares it, so two workspaces never encode the same miss at once.
        self._pending = 0
        self._units = 0  # how many `unit`s hold the transaction open; none commits inside one
        self._owed = False  # a durable event was appended inside a unit: commit when it ends
        self._durable = False  # the open transaction holds a durable event: its commit syncs
        self._wal = con.execute("PRAGMA database_list").fetchone()[2] + "-wal"
        self._timer: threading.Timer | None = None
        self._committed: list[Callable[[bool], None]] = []  # each open-batch event's, in order
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
            con.execute("PRAGMA synchronous = NORMAL")
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
            text = None
            if _current(con, "registry.seeded") == 2:
                text = canonical(_seed_definition_ids(con)).decode()
            log.record("registry.seeded", attrs={"count": count}, text=text)
        log._checkpoint_month(wall_time_ns() // 1000 if now_us is None else now_us)
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
        committed: Callable[[bool], None] | None = None,
        durable: bool = False,
    ) -> int:
        """Admit one event and write it as one unit; returns its seq. `committed` is called under
        the log's guard with True once the event has committed, or False once it is rolled back.
        `durable` commits an ordinary type's event as a durable one: an event that no other record
        carries, so nothing could append it again.

        The unit is the event, its text, the chain, its projection and an inline artifact's body.
        A durable type is committed on return; an ordinary one with its batch, so a caller that
        replies on it flushes first. A live event's recorded_us is the log's clock; replay
        (live=False) keeps the one it had.
        """
        with self._guard:
            con = self._con
            if live:
                row = {**row, "recorded_us": self.clock()}
            if row.get("type_version") is None:
                row = {**row, "type_version": _current(con, row["type"])}
            spec = _spec(con, row)
            attrs = admit(row, text, spec, _superseded(con), live=live, body=body)
            durable = durable or bool(spec.get("durable"))
            if not con.in_transaction:
                con.execute("BEGIN IMMEDIATE")
                self._arm()
            con.execute("SAVEPOINT event")
            try:
                if live:
                    row = _mint_definition_ids(con, row, attrs, text)
                seq = _write(con, row, text, body, live=live)
            except BaseException:
                con.execute("ROLLBACK TO event")
                con.execute("RELEASE event")
                if self._pending == 0 and not self._units:
                    self._end("ROLLBACK")
                raise
            con.execute("RELEASE event")
            self._pending += 1
            self._durable = self._durable or durable
            if committed is not None:
                self._committed.append(committed)
            if self._units:
                self._owed = self._owed or durable
            elif durable or self._pending >= self._batch:
                self._commit()
            return seq

    @contextmanager
    def unit(self) -> Iterator[sqlite3.Connection]:
        """Reads and the events they decide, as one unit: all of it or none of it.

        The log's guard is held throughout, so no other writer appends between a read and the
        event it decided; the connection yielded sees the open batch. An exception rolls back
        every event appended inside, and a durable one among them commits when the unit ends.
        The Host's working tables (no event writes them) may be written inside it too.
        """
        with self._guard:
            con = self._con
            if not con.in_transaction:
                con.execute("BEGIN IMMEDIATE")
                self._arm()
            pending, actions, durable = self._pending, len(self._committed), self._durable
            con.execute("SAVEPOINT unit")
            self._units += 1
            try:
                yield con
            except BaseException:
                con.execute("ROLLBACK TO unit")
                con.execute("RELEASE unit")
                self._pending, self._durable = pending, durable
                dropped, self._committed = self._committed[actions:], self._committed[:actions]
                self._units -= 1
                _run(dropped, committed=False)
                if not self._units:
                    self._owed = False
                    if self._pending == 0:
                        self._end("ROLLBACK")
                raise
            con.execute("RELEASE unit")
            self._units -= 1
            if not self._units and (self._owed or self._pending >= self._batch):
                self._owed = False
                self._end("COMMIT")
            elif not self._units and self._pending == 0:
                self._end("COMMIT")  # the working tables it wrote, with no event to batch

    @contextmanager
    def reading(self) -> Iterator[sqlite3.Connection]:
        """The Host's connection for several reads that must see one state, open batch included."""
        with self._guard:
            yield self._con

    def lineage(self, family: str, name: str) -> str | None:
        """The Host's id for a name, including one registered in its open batch."""
        with self.reading() as con:
            return lineage(con, family, name)

    def require_lineage(self, family: str, name: str) -> str:
        """Resolve a held name or refuse the missing registration by family and name."""
        with self.reading() as con:
            return require_lineage(con, family, name)

    def clock(self, at_least: int = 0) -> int:
        """The log's next moment, in microseconds: the larger of the wall clock, the last moment
        plus one and `at_least`. It never steps back, so neither does anything stamped with it."""
        with self._guard:
            self._last_us = max(wall_time_ns() // 1000, self._last_us + 1, at_least)
            return self._last_us

    def record(
        self,
        type_: str,
        subject: tuple[str | None, str | None] = (None, None),
        attrs: Mapping[str, Any] | None = None,
        *,
        text: str | None = None,
        body: bytes | None = None,
        committed: Callable[[bool], None] | None = None,
        **more: Any,
    ) -> int:
        """Append a new event of `type_`, its envelope made by `envelope`."""
        row = envelope(type_, subject, attrs, text=text, **more)
        if (type_, row["type_version"]) in BOUND_TEXT_EVENTS:
            with self.unit() as con:
                self._register_bound_text(con, type_, row["type_version"])
                return self.append(row, text, body, committed=committed)
        return self.append(row, text, body, committed=committed)

    def _register_bound_text(self, con: sqlite3.Connection, kind: str, version: int) -> None:
        """Install a compatible machine-text version through the log at first use."""
        key = {"family": "event_type", "name": kind, "version": version}
        if con.execute(
            "SELECT 1 FROM definition WHERE family = ? AND name = ? AND version = ?",
            ("event_type", kind, version),
        ).fetchone():
            return
        spec = _spec(con, {"type": kind, "type_version": 1}) | {"text": "optional"}
        description = {
            "artifact.produced": "An artifact profile may carry its shape as machine text.",
            "admission": "An admission may carry its complete provider list as machine text.",
        }[kind]
        self.record(
            "definition.registered",
            ("definition", f"event_type/{kind}"),
            key,
            text=canonical(
                key
                | {
                    "spec": spec,
                    "status": "standard",
                    "relation": "equal",
                    "text": description,
                }
            ).decode(),
        )

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

    def cached(
        self,
        table: str,
        recipe_key: str,
        parts: Sequence[tuple[str, str, int, int]],
        *,
        compute_ms: int,
        body: str | None = None,
    ) -> None:
        """Record the derived partitions of `table` written for these inputs, each (part,
        input_digest, rows, size_bytes), in one unit; `body` is a file partition's handle (one
        part), None for catalog rows. `cache_entry` is the Host's working state, not a projection:
        no event writes it, and losing it loses only the cache."""
        with self.unit() as con:
            now = self.clock()
            con.executemany(
                "INSERT INTO cache_entry VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT DO UPDATE SET row_count = excluded.row_count,"
                " size_bytes = excluded.size_bytes, compute_ms = excluded.compute_ms,"
                " computed_us = excluded.computed_us, body = excluded.body",
                [
                    (recipe_key, table, part, digest, rows, size, compute_ms, now, now, body)
                    for part, digest, rows, size in parts
                ],
            )

    def rooted(
        self,
        workspace_id: str,
        table: str,
        parts: Sequence[tuple[str, str, str]],
        *,
        exactly: bool = False,
    ) -> int:
        """Root the partitions of `table` a workspace's reads take, each (recipe_key, part,
        input_digest), in `cache_root`. With `exactly`, they become its only roots of `table`: a
        partition it no longer reads loses its root, and is collected once no workspace roots it.
        The roots dropped."""
        with self.unit() as con:
            con.execute(
                "CREATE TEMP TABLE IF NOT EXISTS rooting (recipe_key TEXT, part TEXT,"
                " input_digest TEXT, PRIMARY KEY (recipe_key, part, input_digest)) WITHOUT ROWID"
            )  # keyed, so the exact drop below looks each root up rather than scanning them all
            con.execute("DELETE FROM temp.rooting")
            con.executemany("INSERT OR IGNORE INTO temp.rooting VALUES (?, ?, ?)", parts)
            dropped = 0
            if exactly:
                dropped = con.execute(
                    "DELETE FROM cache_root WHERE workspace_id = ? AND derived_table = ?"
                    " AND NOT EXISTS (SELECT 1 FROM temp.rooting t WHERE t.recipe_key ="
                    " cache_root.recipe_key AND t.part = cache_root.part"
                    " AND t.input_digest = cache_root.input_digest)",
                    (workspace_id, table),
                ).rowcount
            con.execute(
                "INSERT OR IGNORE INTO cache_root SELECT ?, *, ? FROM temp.rooting",
                (table, workspace_id),
            )
            con.execute("DELETE FROM temp.rooting")
        return dropped

    def collect_cached(self, table: str) -> None:
        """Drop the cache entries of `table`'s partitions no workspace roots."""
        with self._guard:
            self._con.execute(
                "DELETE FROM cache_entry WHERE derived_table = ? AND NOT EXISTS (SELECT 1 FROM"
                " cache_root r WHERE r.derived_table = cache_entry.derived_table AND r.recipe_key"
                " = cache_entry.recipe_key AND r.part = cache_entry.part AND r.input_digest ="
                " cache_entry.input_digest)",
                (table,),
            )

    def cache_hit(self, table: str, recipe_key: str, part: str, input_digest: str) -> bool:
        """Whether a derived partition of `table` is cached for exactly these inputs; a hit marks
        its last read, which eviction reads."""
        with self._guard:
            found = self._con.execute(
                "UPDATE cache_entry SET last_read_us = ? WHERE recipe_key = ? AND derived_table = ?"
                " AND part = ? AND input_digest = ?",
                (self.clock(), recipe_key, table, part, input_digest),
            )
            return found.rowcount == 1

    def uncached(self, table: str, digests: Sequence[str]) -> None:
        """Drop the cache entries of `table`'s partitions with these input digests. Their roots
        stay, so a workspace that roots one keeps it again at its next read."""
        with self.unit() as con:
            con.execute(
                "DELETE FROM cache_entry WHERE derived_table = ?"
                " AND input_digest IN (SELECT value FROM json_each(?))",
                (table, json.dumps(sorted(digests))),
            )

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
        self._timer.daemon, self._timer.name = True, COMMIT_TIMER
        self._timer.start()

    def _commit(self) -> None:
        if self._units:  # a flush inside a unit commits when the unit ends, never before it
            self._owed = True
        elif self._pending:
            self._end("COMMIT")

    def _end(self, statement: str) -> None:
        self._con.execute(statement)
        durable = statement == "COMMIT" and self._durable
        self._pending, self._durable = 0, False
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        actions, self._committed = self._committed, []
        try:  # the batch has ended: a failed sync is raised to its appender, never left open
            if durable:
                _sync(self._wal)
        finally:
            _run(actions, committed=statement == "COMMIT")


def _sync(wal: str) -> None:
    """Sync the log's WAL to the disk, as synchronous FULL does at each commit: what committed
    before it survives a lost machine. A WAL a checkpoint has removed was synced into the log."""
    try:
        handle = os.open(wal, os.O_RDWR | getattr(os, "O_BINARY", 0))
    except FileNotFoundError:
        return
    try:
        os.fsync(handle)
    finally:
        os.close(handle)


def _run(actions: list[Callable[[bool], None]], *, committed: bool) -> None:
    """Each event's action, told its event's fate; one that raises stops none after it, and the
    first error is raised once all have run."""
    failed: Exception | None = None
    for action in actions:
        try:
            action(committed)
        except Exception as error:
            failed = failed or error
    if failed is not None:
        raise failed


def attach(root: Path, *, shared: bool = False) -> sqlite3.Connection:
    """A read-only connection to the data root's log, for any process but the Host; `shared`, one
    any thread may use under its holder's lock."""
    uri = sqlite_read_only_uri((root / "machine" / "log.sqlite").absolute())
    con = sqlite3.connect(uri, uri=True, check_same_thread=not shared)
    con.row_factory = sqlite3.Row
    return con


def verify_from_checkpoint(con: sqlite3.Connection) -> bool:
    """Whether the tail folds to the stored head and its machine profiles match their digests."""
    stored = con.execute("SELECT head, chain_version FROM chain_head WHERE id = 1").fetchone()
    if stored is None:
        return con.execute("SELECT 1 FROM event LIMIT 1").fetchone() is None
    last = con.execute("SELECT seq, head FROM chain_checkpoint ORDER BY seq DESC LIMIT 1")
    start, head = last.fetchone() or (0, "")
    events = con.execute("SELECT * FROM event WHERE seq > ? ORDER BY seq", (start,))
    return fold(head, events, stored[1]) == stored[0] and _verify_profile_text(con, start)


def verify_whole(con: sqlite3.Connection) -> bool:
    """Whether the chain folds to the head and its machine profiles match their digests."""
    stored = con.execute("SELECT head, chain_version FROM chain_head WHERE id = 1").fetchone()
    if stored is None:
        return con.execute("SELECT 1 FROM event LIMIT 1").fetchone() is None
    bounds = con.execute("SELECT seq, chain_version FROM chain_checkpoint ORDER BY seq").fetchall()
    events = con.execute("SELECT * FROM event ORDER BY seq")
    folded = fold_segments(events, [(b[0], b[1]) for b in bounds], stored[1])
    return folded == stored[0] and _verify_profile_text(con, 0)


def _verify_profile_text(con: sqlite3.Connection, after: int) -> bool:
    """Whether each bound machine profile in the verified range kept its named text."""
    versions = ", ".join("(?, ?)" for _ in BOUND_TEXT_EVENTS)
    keys = tuple(value for event in BOUND_TEXT_EVENTS for value in event)
    rows = con.execute(
        "SELECT e.text_digest, t.text FROM event e LEFT JOIN event_text t USING (seq)"
        f" WHERE e.seq > ? AND (e.type, e.type_version) IN ({versions})"
        " AND e.text_digest IS NOT NULL",
        (after, *keys),
    )
    return all(text is not None and digest(text.encode()) == expected for expected, text in rows)


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


def _current(con: sqlite3.Connection, type_: str) -> int:
    """The version of event type `type_` new events are written under: its current one."""
    found = con.execute(
        "SELECT version FROM definition_current WHERE family = 'event_type' AND name = ?",
        (type_,),
    ).fetchone()
    if found is None:
        raise DataError(
            f"{type_} is not a registered event type", code="data_layer.event_type_unregistered"
        )
    return int(found[0])


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


def _mint_definition_ids(
    con: sqlite3.Connection, row: Mapping[str, Any], attrs: dict[str, Any], text: str | None
) -> Mapping[str, Any]:
    if "definition_ids" in attrs:
        raise DataError("the Host mints definition ids", code="data_layer.definition_id_reserved")
    names = definition_names(row["type"], row.get("phase"), row["node"], attrs, text)
    minted = [
        (family, name, uuid7())
        for family, name in dict.fromkeys(names)
        if lineage(con, family, name) is None
    ]
    if minted:
        return {**row, "attrs": canonical({**attrs, "definition_ids": minted}).decode()}
    return row


def _write(
    con: sqlite3.Connection,
    row: Mapping[str, Any],
    text: str | None,
    body: bytes | None,
    *,
    live: bool,
    legacy_identity: Callable[[str, str], str] | None = None,
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
    project(con, stored, text, legacy_identity)
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


def _seed_definition_ids(con: sqlite3.Connection) -> list[tuple[str, str, str]]:
    """The ids the Host records in the seed event, in stable order."""
    rows = con.execute(
        "SELECT n.family, n.name, n.definition_id FROM definition_name n"
        " JOIN definition_lineage l USING (definition_id)"
        " WHERE n.named_seq = l.registered_seq AND l.registered_seq = 1 ORDER BY n.family, n.name"
    )
    return [(family, name, identity) for family, name, identity in rows]


def _legacy_lineage(family: str, first_name: str) -> str:
    """Decode the original key of a first registration written without an id."""
    return digest(canonical(["definition", family, first_name]))


def replay(source: sqlite3.Connection, path: Path) -> sqlite3.Connection:
    """A new log at `path` holding `source`'s events, each written again through the one write.

    Copy events, texts and inline bodies. The seed event carries the original seed ids, and
    the event that first named a later lineage carries its id. In a v1-seeded log, recover the
    ids of older registrations from their original names and use the recorded ids where present.
    """
    seeded = source.execute(
        "SELECT e.type_version, t.text, e.attrs FROM event e LEFT JOIN event_text t USING (seq)"
        " WHERE e.seq = 1 AND e.type = 'registry.seeded'"
    ).fetchone()
    seed_version = 1 if seeded is None else seeded[0]
    seed_count = None if seeded is None else json.loads(seeded[2]).get("count")
    if seeded is not None and (type(seed_count) is not int or seed_count < 1):
        raise DataError(
            "registry.seeded has no positive definition count", code="data_layer.chain_broken"
        )
    legacy_identity = _legacy_lineage if seed_version == 1 else None
    if legacy_identity is None:
        identities = {(family, name): identity for family, name, identity in json.loads(seeded[1])}
    else:
        identities = {
            (family, name): legacy_identity(family, name)
            for family, name, version, *_ in seed_rows(historical_count=seed_count)
            if version == 1
        }
    con = sqlite3.connect(path, isolation_level=None)
    con.row_factory = sqlite3.Row
    try:
        build_log(con, seed_ids=identities, seed_version=seed_version)
        actual = con.execute("SELECT count(*) FROM definition_version").fetchone()[0]
        if seed_count is not None and actual != seed_count:
            raise DataError(
                f"registry.seeded records {seed_count} definitions but replay rebuilt {actual}",
                code="data_layer.chain_broken",
            )
    except BaseException:
        con.close()
        raise
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
            _write(con, row, text, body, live=False, legacy_identity=legacy_identity)
        con.execute("COMMIT")
    except BaseException:
        con.execute("ROLLBACK")
        con.close()
        raise
    return con
