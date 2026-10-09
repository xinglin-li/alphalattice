"""Physical DuckDB ownership for one local workspace.

DuckDB keeps one database instance per file per process and refuses to open
the same file in another access mode while any connection to that instance is
alive. The instance's threads are the operator's, set on each connection
(`reader_threads`), never in the configuration an opener must match. Every
opener of a workspace file therefore goes through this owner, which tracks the
live connections it handed out, decides which instance mode serves a request,
and keeps two promises:

- a request for read-only access never receives a writable connection. On a
  read-only instance the engine enforces that; on a writable instance the
  handle runs every statement inside a ``READ ONLY`` transaction the engine
  refuses to write in;
- a request for the other access mode waits, bounded, for the connections of
  the current instance to close instead of meeting the engine's refusal, and
  wakes deterministically when the last one closes.
"""

from __future__ import annotations

import json
import os
import secrets
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from typing import Any, NamedTuple, cast

import duckdb

from alphalattice.control.workspace_runtime.reader_threads import connect_duckdb
from alphalattice.kernel.shared_kernel.spans import span

WORKSPACE_MARKET_DATA_DATABASE_FILENAME = "market-data.duckdb"
"""The market/feature/sector store. Named so other owners can refuse it by name."""

RETENTION_WAIT_SECONDS = 120.0
"""How long a request waits for another thread's connections in the other access mode."""

_CONNECTION_REFUSED_RETRY_SECONDS = 0.05
"""Re-check cadence while DuckDB still holds an instance this owner no longer tracks."""


def _connect(path: Path, *, read_only: bool) -> duckdb.DuckDBPyConnection:
    return connect_duckdb(path, read_only=read_only)


_SEAL_RECORD_SUFFIX = ".seal.json"
"""Beside the store, like its WAL: the seal epoch and the file as its last close left it."""

_EPOCHS: dict[Path, tuple[str, bool]] = {}
"""Per file with an open instance: the seal epoch its seals vouch under, and whether the
instance opened read only (`seal_epoch`)."""


def _fingerprint(path: Path) -> list[list[int] | None]:
    """The store file's and its WAL's size and mtime, as any program writing them leaves them."""
    stamps: list[list[int] | None] = []
    for item in (path, Path(f"{path}.wal")):
        try:
            stat = item.stat()
        except FileNotFoundError:
            stamps.append(None)
        else:
            stamps.append([stat.st_size, stat.st_mtime_ns])
    return stamps


def _seal_record(path: Path) -> dict[str, Any]:
    try:
        record = json.loads(Path(f"{path}{_SEAL_RECORD_SUFFIX}").read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return {}
    return record if isinstance(record, dict) else {}


def _write_seal_record(
    path: Path, *, epoch: str, fingerprint: list[list[int] | None] | None
) -> None:
    target = Path(f"{path}{_SEAL_RECORD_SUFFIX}")
    staged = target.with_name(f"{target.name}.tmp")
    staged.write_text(json.dumps({"epoch": epoch, "fingerprint": fingerprint}), encoding="utf-8")
    os.replace(staged, target)


def seal_epoch(path: Path) -> str:
    """The epoch whose seals the store's contents vouch for, a random token.

    A seal is derived state its own writers keep current in their transactions (the Feature
    year seals). Only an edit none of them made can leave one stale, and while an instance holds
    the file only this process can write it. So the store's file is fingerprinted, by size and
    mtime with its WAL, when its last writable instance closes, and the next instance's open
    compares: a file otherwise than its last close left it -- another program wrote it -- or no
    close recorded at all (a crash) takes a new token, and no seal of an earlier one vouches
    again. A read-only open records nothing; one that finds the file moved holds a token of its
    own. With no instance open, no seal vouches.
    """
    held = _EPOCHS.get(Path(path).resolve())
    return held[0] if held is not None else secrets.token_hex(16)


def _fresh_connection(path: Path, *, read_only: bool) -> duckdb.DuckDBPyConnection:
    """Open the process's instance of the file (a metadata read; a checkpoint on close).

    The open first reads the file's fingerprint against its last recorded close (`seal_epoch`),
    and clears the record until this instance's close records it again.
    """

    before = _fingerprint(path)
    with span("read", "duckdb_instance_open"):
        raw = _connect(path, read_only=read_only)
    try:
        record = _seal_record(path)
        epoch = record.get("epoch")
        if not isinstance(epoch, str) or record.get("fingerprint") != before:
            epoch = secrets.token_hex(16)
        if not read_only:
            _write_seal_record(path, epoch=epoch, fingerprint=None)
    except BaseException:
        raw.close()
        raise
    _EPOCHS[path] = (epoch, read_only)
    return raw


def _attached_connection(path: Path, *, read_only: bool) -> duckdb.DuckDBPyConnection:
    """One more connection to the instance this owner already tracks."""

    return _connect(path, read_only=read_only)


class WorkspaceConnection:
    """One connection handed out by this owner.

    It behaves as the ``duckdb.DuckDBPyConnection`` it wraps, and adds what
    the engine cannot do on its own: a top-level handle reports its close so
    the instance's lifetime is known, and a read-only handle on a writable
    instance is *guarded*. A guarded handle lives inside a ``READ ONLY``
    transaction from creation to close -- a fresh one before each statement,
    so every statement still sees the latest committed state as autocommit
    reads do on a read-only instance -- and the engine refuses every write
    inside it. What the engine's read-only transaction cannot stop is
    refused here, by the engine's own classification of the statements it
    is asked to run: transaction control (which would end the transaction
    and leave a later statement or relation writable), attaching or
    detaching databases, loading extensions, copying or exporting databases,
    vacuuming and procedure calls (a forced checkpoint aborts other
    transactions).

    Every call goes through one dispatch (``_call``), keyed by the engine's
    public surface as installed: the methods that take SQL -- ``execute``,
    ``executemany``, ``sql``, ``query``, ``from_query``, positionally or as
    ``query=``, text or a statement object -- are classified before the
    transaction is armed; the doors that build a statement from something
    other than SQL (a frame, a file, a table name) are armed; the method
    forms of refused statements (``begin``/``commit``/``rollback``,
    ``checkpoint``, ``install_extension``/``load_extension``) are refused by
    the same code; everything else (fetching, registering a frame, session
    functions and types) passes through, and a method that returns the
    engine's connection hands back the guarded handle instead. A relation
    built on the handle runs inside the same transaction; writing a
    relation to a file is an export, not a database write, and is not
    refused. ``snapshot()`` keeps one transaction open across several
    statements for a consistent read.
    """

    __slots__ = ("_closed", "_guarded", "_instance", "_owner", "_path", "_pinned", "_raw", "_top")

    def __init__(
        self,
        raw: duckdb.DuckDBPyConnection,
        *,
        path: Path,
        instance: _Instance,
        guarded: bool,
        top_level: bool,
    ) -> None:
        """Wrap an engine connection and arm its read-only guard when requested.

        Args:
            raw: Engine connection wrapped by this physical database owner.
            path: Physical file path owned by the caller.
            instance: Tracked physical database instance shared by its live handles.
            guarded: Whether each statement requires the read-only transaction guard.
            top_level: Whether closing this handle releases tracked instance ownership.
        """
        self._raw = raw
        self._path = path
        self._instance = instance
        self._guarded = guarded
        self._top = top_level
        self._owner = threading.get_ident()
        self._pinned = False
        self._closed = False
        if guarded:
            self._raw.execute("BEGIN TRANSACTION READ ONLY")

    # -- the read-only guard ---------------------------------------------

    def _end_transaction(self) -> None:
        # COMMIT ends a read-only transaction whether it is live or aborted,
        # keeps the session's registered views and temporary tables, and is
        # refused only when no transaction is open.
        with suppress(duckdb.TransactionException):
            self._raw.execute("COMMIT")

    def _arm(self) -> None:
        if not self._guarded or self._pinned:
            return
        self._end_transaction()
        self._raw.execute("BEGIN TRANSACTION READ ONLY")

    def _refuse_unguardable(self, query: object) -> None:
        """Refuse, by the engine's classification, what its read-only transaction cannot stop.

        The exposed methods accept SQL text (one or several statements) or a
        statement object the engine extracted earlier; both are classified
        here before anything reaches the transaction.
        """

        if not self._guarded:
            return
        if isinstance(query, str):
            statements = self._raw.extract_statements(query)
        elif isinstance(query, duckdb.Statement):
            statements = [query]
        else:
            return
        for statement in statements:
            if statement.type in _UNGUARDABLE_STATEMENTS:
                raise RuntimeError(
                    f"workspace_database.read_only_handle_refuses_statement:{statement.type.name}"
                )

    @contextmanager
    def snapshot(self) -> Iterator[WorkspaceConnection]:
        """Read several statements from one consistent snapshot."""
        if self._guarded:
            self._arm()
            self._pinned = True
            try:
                yield self
            finally:
                self._pinned = False
                self._arm()
            return
        self._raw.execute("BEGIN TRANSACTION")
        try:
            yield self
        except BaseException:
            with suppress(duckdb.Error):
                self._raw.execute("ROLLBACK")
            raise
        else:
            self._raw.execute("COMMIT")

    # -- the one dispatch ------------------------------------------------

    def _call(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        """Run one of the engine's methods on this handle under the guard."""

        attribute = getattr(self._raw, name)
        if self._guarded:
            refused = _REFUSED_METHODS.get(name)
            if refused is not None:
                raise RuntimeError(
                    f"workspace_database.read_only_handle_refuses_statement:{refused}"
                )
            if name in _SQL_DOORS:
                self._refuse_unguardable(args[0] if args else kwargs.get("query"))
            if name in _SQL_DOORS or name in _STATEMENT_DOORS:
                self._arm()
        try:
            result = attribute(*args, **kwargs)
        except duckdb.TransactionException as error:
            # A relation used on this handle may have aborted the transaction
            # outside any door; re-arm and try once more, as the statement
            # would simply run on a read-only instance.
            if not self._guarded or self._pinned or "aborted" not in str(error):
                self._recover()
                raise
            self._arm()
            result = attribute(*args, **kwargs)
        except duckdb.Error:
            self._recover()
            raise
        return self if result is self._raw else result

    def _recover(self) -> None:
        # A refused write or a conversion error aborts the read-only
        # transaction; on a read-only instance the next statement would
        # simply run, so the guarded handle re-arms a clean one.
        if self._guarded and not self._pinned:
            self._arm()

    # -- statements ------------------------------------------------------

    def execute(self, query: Any, *args: Any, **kwargs: Any) -> WorkspaceConnection:
        """Execute one statement through the access-mode guard.

        Args:
            query: SQL or engine statement passed through the guarded dispatch.

        Returns:
            This wrapped connection for subsequent result access.

        Raises:
            RuntimeError: A guarded reader requests a refused operation.
        """
        self._call("execute", (query, *args), kwargs)
        return self

    def executemany(self, query: Any, *args: Any, **kwargs: Any) -> WorkspaceConnection:
        """Execute repeated parameter bindings through the access-mode guard.

        Args:
            query: SQL or engine statement passed through the guarded dispatch.

        Returns:
            This wrapped connection for subsequent result access.

        Raises:
            RuntimeError: A guarded reader requests a refused operation.
        """
        self._call("executemany", (query, *args), kwargs)
        return self

    def begin(self) -> WorkspaceConnection:
        """Begin an engine transaction through the guarded dispatch.

        Returns:
            This wrapped connection.

        Raises:
            RuntimeError: Transaction control is refused on a guarded read-only handle.
        """
        self._call("begin", (), {})
        return self

    def commit(self) -> WorkspaceConnection:
        """Commit an engine transaction through the guarded dispatch.

        Returns:
            This wrapped connection.

        Raises:
            RuntimeError: Transaction control is refused on a guarded read-only handle.
        """
        self._call("commit", (), {})
        return self

    def rollback(self) -> WorkspaceConnection:
        """Roll back an engine transaction through the guarded dispatch.

        Returns:
            This wrapped connection.

        Raises:
            RuntimeError: Transaction control is refused on a guarded read-only handle.
        """
        self._call("rollback", (), {})
        return self

    def cursor(self) -> WorkspaceConnection:
        """Its own transaction context on the same instance; closed with this connection."""
        return self._child(self._guarded)

    def duplicate(self) -> WorkspaceConnection:
        """Create another cursor with the same read-only guard policy.

        Returns:
            Child wrapped connection; the caller closes it independently.
        """
        return self._child(self._guarded)

    def _child(self, guarded: bool) -> WorkspaceConnection:
        return WorkspaceConnection(
            self._raw.cursor(),
            path=self._path,
            instance=self._instance,
            guarded=guarded,
            top_level=False,
        )

    def close(self) -> None:
        """Close once and release tracked top-level ownership even if engine close raises."""
        if self._closed:
            return
        self._closed = True
        if not self._top:
            self._raw.close()
            return
        # Untracked first: the engine closes the instance with its last
        # connection (a checkpoint, then the file), and an opener that
        # attached to this handle meanwhile would meet a dying instance and
        # the engine's file lock. Openers take the fresh path instead, and
        # one that meets the lock while a close is in flight waits for it.
        # The handle leaves the books and its close is counted in flight in
        # one critical section (`_release_top_level`), so no opener can read
        # "untracked, and nothing closing" between the two; the engine's
        # close itself runs outside the lock, and the count comes down --
        # and the waiters are woken -- whether it returns or raises.
        counted = _release_top_level(self)
        try:
            with span("write", "duckdb_instance_close"):
                self._raw.close()
        finally:
            if counted:
                _closed(self._path)

    def __enter__(self) -> WorkspaceConnection:
        """Enter the connection context.

        Returns:
            This wrapped connection.
        """
        return self

    def __exit__(self, *exc: object) -> None:
        """Close the handle when its context exits; propagate the context exception."""
        self.close()

    def __del__(self) -> None:
        """Attempt to close an abandoned connection while suppressing destructor failures."""
        with suppress(Exception):
            self.close()

    def __getattr__(self, name: str) -> Any:
        # Everything else is the engine's, reached through the same dispatch:
        # fetching a pending result does not touch the transaction, and no
        # caller ever receives the unguarded connection.
        """Resolve engine attributes while preserving guarded method dispatch.

        Args:
            name: Engine attribute name to resolve through the guarded wrapper.

        Returns:
            Engine attribute, or a callable routed through the access-mode guard.

        Raises:
            AttributeError: The name is a dunder attribute or the engine has no such attribute.
        """
        if name.startswith("__"):
            raise AttributeError(name)
        attribute = getattr(self._raw, name)
        if not callable(attribute):
            return attribute

        def guarded_call(*args: Any, **kwargs: Any) -> Any:
            return self._call(name, args, kwargs)

        return guarded_call


_SQL_DOORS = frozenset({"execute", "executemany", "sql", "query", "from_query"})
"""The engine's methods that take SQL: text or a statement object, positionally or as ``query=``."""

_STATEMENT_DOORS = frozenset(
    {
        "append",
        "from_arrow",
        "from_csv_auto",
        "from_df",
        "from_parquet",
        "read_csv",
        "read_json",
        "read_parquet",
        "table",
        "table_function",
        "values",
        "view",
    }
)
"""Doors that build a statement from something other than SQL; armed, nothing to classify."""

_REFUSED_METHODS = {
    "begin": "TRANSACTION",
    "commit": "TRANSACTION",
    "rollback": "TRANSACTION",
    "checkpoint": "CALL",
    "install_extension": "LOAD",
    "load_extension": "LOAD",
}
"""Method forms of the refused statements, named by the statement type they stand for."""

_UNGUARDABLE_STATEMENTS = frozenset(
    {
        duckdb.StatementType.TRANSACTION,
        duckdb.StatementType.ATTACH,
        duckdb.StatementType.DETACH,
        duckdb.StatementType.LOAD,
        duckdb.StatementType.EXTENSION,
        duckdb.StatementType.COPY_DATABASE,
        duckdb.StatementType.EXPORT,
        duckdb.StatementType.VACUUM,
        duckdb.StatementType.CALL,
    }
)
"""What a read-only transaction does not stop, so a guarded handle refuses it."""


@dataclass
class _Instance:
    """The connections alive on the process's instance of one file."""

    read_only: bool
    handles: set[WorkspaceConnection] = field(default_factory=set)

    def held_by(self, thread: int) -> bool:
        return any(handle._owner == thread for handle in self.handles)


@dataclass
class _Retention:
    root: WorkspaceConnection
    holds: list[tuple[object, bool]]
    """The scopes holding this thread's retention, innermost last, with their access modes.

    Scopes may end out of order (a runner's unit ends while the caller's
    hold, taken inside it, continues), so each removes its own entry and the
    root closes with the last one.
    """

    @property
    def innermost_read_only(self) -> bool:
        return self.holds[-1][1]


_STATE = threading.Condition()
_INSTANCES: dict[Path, _Instance] = {}
_HELD = "workspace_database.access_mode_held_by_another_thread"
_UNTRACKED = "workspace_database.access_mode_held_outside_this_owner"
_OTHER_PROCESS = "workspace_database.file_held_by_another_process"
_UNKNOWN_HOLDER = "workspace_database.file_held_by_an_unknown_holder"


def _is_file_lock_refusal(error: duckdb.IOException) -> bool:
    text = str(error)
    return "process cannot access the file" in text or "Could not set lock" in text


def _lock_refusal(error: duckdb.IOException, path: Path) -> str | None:
    """How a file-lock refusal is answered, the same on the fresh and the attached path:
    None to wait (bounded, on the caller's one deadline), or the refusal to raise by name.
    The engine names the holding process when it can: this process's own pid is an
    instance of ours going away (or one this owner does not track) and is waited for;
    any other pid is another process, refused at once. A message that names no process
    is decided by this owner's knowledge alone: while an engine close of one of this
    owner's connections to the file is in flight (`_CLOSING`) the lock is that dying
    connection's and is waited for; with no close in flight the holder is unknown and the
    request is refused as such -- never retried on a guess about timing. Must be called
    under `_STATE`."""
    text = str(error)
    if "already open in" in text:
        return None if f"(PID {os.getpid()})" in text else _OTHER_PROCESS
    return None if _CLOSING.get(path) else _UNKNOWN_HOLDER


_WRITERS_WAITING: dict[Path, int] = {}
_RETENTIONS: dict[tuple[Path, int], _Retention] = {}
_CLOSING: dict[Path, int] = {}
"""Per file, the engine closes of this owner's top-level connections in flight right now:
counted in the same critical section that untracks the handle, counted down when the
engine's close returns. Every close is counted, not the last tracked handle's alone: the
engine releases the file with whichever connection drops the last reference, and two
handles closing at once may do so in either order -- the first to leave the books can be
the last to let go of the file, after the other's mark would have been lifted."""


def _release_top_level(handle: WorkspaceConnection) -> bool:
    """Take the handle off the books and count its engine close as in flight, in one
    critical section; when it was the instance's last, the instance leaves the books too.
    True says the caller owes `_closed` when the engine's close returns; False, that the
    handle was not on the books and nothing is owed."""
    with _STATE:
        instance = _INSTANCES.get(handle._path)
        if instance is None or handle not in instance.handles:
            return False
        instance.handles.discard(handle)
        if not instance.handles:
            del _INSTANCES[handle._path]
        _CLOSING[handle._path] = _CLOSING.get(handle._path, 0) + 1
        _STATE.notify_all()
        return True


def _closed(path: Path) -> None:
    """The engine's close of one connection returned (or raised): counted down, and the
    openers waiting on the file are woken when no close is in flight any more."""
    with _STATE:
        _CLOSING[path] -= 1
        if not _CLOSING[path]:
            del _CLOSING[path]
            if path not in _INSTANCES:
                # The file as this owner's last writable close leaves it (`seal_epoch`).
                epoch, read_only = _EPOCHS.pop(path, ("", True))
                if not read_only:
                    _write_seal_record(path, epoch=epoch, fingerprint=_fingerprint(path))
        _STATE.notify_all()


def _acquire(path: Path, *, read_only: bool, wait_seconds: float) -> WorkspaceConnection:
    """A top-level connection in the requested access mode, waiting when it must.

    - no live instance: open one in the requested mode (unless a writer is
      waiting for the file, which a new reader lets go first);
    - a live instance of the same mode: attach to it;
    - a writable instance and a read-only request: attach with the guard;
    - a read-only instance and a writable request: wait, bounded, for its
      connections to close -- unless they are this thread's own, which no
      wait can release.
    """

    me = threading.get_ident()
    deadline: float | None = None
    waiting_as_writer = False

    def wait(refusal: str) -> None:
        nonlocal deadline
        if deadline is None:
            deadline = monotonic() + wait_seconds
        remaining = deadline - monotonic()
        if remaining <= 0.0:
            raise RuntimeError(refusal)
        slice_ = _CONNECTION_REFUSED_RETRY_SECONDS if refusal == _UNTRACKED else remaining
        _STATE.wait(min(remaining, slice_))

    with _STATE:
        try:
            while True:
                instance = _INSTANCES.get(path)
                if instance is None:
                    if read_only and _WRITERS_WAITING.get(path, 0):
                        wait(_HELD)
                        continue
                    try:
                        raw = _fresh_connection(path, read_only=read_only)
                    except duckdb.ConnectionException:
                        # DuckDB still holds an instance in the other mode
                        # that this owner does not track (a connection opened
                        # elsewhere, or one not yet collected); it goes away
                        # when that connection closes.
                        wait(_UNTRACKED)
                        continue
                    except duckdb.IOException as error:
                        # A file lock: this process's own instance still going
                        # away, an instance this owner does not track, or another
                        # process -- decided by `_lock_refusal`, waited for on the
                        # one deadline or refused by name; nothing changed.
                        if not _is_file_lock_refusal(error):
                            raise
                        refusal = _lock_refusal(error, path)
                        if refusal is None:
                            wait(_UNTRACKED)
                            continue
                        raise RuntimeError(refusal) from error
                    instance = _Instance(read_only=read_only)
                    _INSTANCES[path] = instance
                    return _register(instance, raw, path, guarded=False)
                if instance.read_only == read_only:
                    if read_only and _WRITERS_WAITING.get(path, 0) and not instance.held_by(me):
                        wait(_HELD)
                        continue
                    raw = _attach(path, read_only=read_only, wait=wait)
                    if raw is None:
                        continue
                    return _register(instance, raw, path, guarded=False)
                if not instance.read_only:
                    raw = _attach(path, read_only=False, wait=wait)
                    if raw is None:
                        continue
                    return _register(instance, raw, path, guarded=True)
                if instance.held_by(me):
                    raise RuntimeError("workspace_database.access_mode_conflict_on_this_thread")
                if not waiting_as_writer:
                    _WRITERS_WAITING[path] = _WRITERS_WAITING.get(path, 0) + 1
                    waiting_as_writer = True
                wait(_HELD)
        finally:
            if waiting_as_writer:
                _WRITERS_WAITING[path] -= 1
                if not _WRITERS_WAITING[path]:
                    del _WRITERS_WAITING[path]
                _STATE.notify_all()


def _attach(
    path: Path, *, read_only: bool, wait: Callable[[str], None]
) -> duckdb.DuckDBPyConnection | None:
    """One more connection to the instance this owner tracks, or None after waiting when
    the engine refuses it for an instance of this process: the tracked instance's engine
    object going away (its last handle closed an instant ago and the file lock is not
    released yet) or one this owner does not track. The caller re-reads its bookkeeping
    and tries again within the bound. A tracked instance is not proof that every lock on
    the file is this process's: a lock the engine names with another pid is refused by
    name here exactly as on the fresh path."""
    try:
        return _attached_connection(path, read_only=read_only)
    except duckdb.ConnectionException:
        wait(_UNTRACKED)
        return None
    except duckdb.IOException as error:
        if not _is_file_lock_refusal(error):
            raise
        refusal = _lock_refusal(error, path)
        if refusal is None:
            wait(_UNTRACKED)
            return None
        raise RuntimeError(refusal) from error


def _register(
    instance: _Instance, raw: duckdb.DuckDBPyConnection, path: Path, *, guarded: bool
) -> WorkspaceConnection:
    handle = WorkspaceConnection(raw, path=path, instance=instance, guarded=guarded, top_level=True)
    instance.handles.add(handle)
    return handle


def open_workspace_database(
    path: Path, *, read_only: bool, wait_seconds: float = RETENTION_WAIT_SECONDS
) -> duckdb.DuckDBPyConnection:
    """Open one connection to a workspace DuckDB file in the requested access mode.

    Inside a retention held by the calling thread the connection is a cursor
    of the retained instance: its own transaction context, closed by the
    caller as any connection would be, with no metadata re-read or checkpoint
    per open; a read-only request stays read-only there (the guard above),
    and a writable request inside a read-only unit is refused by name.
    Outside a retention the request is served as ``_acquire`` describes.
    """
    resolved = path.resolve()
    me = threading.get_ident()
    with _STATE:
        retention = _RETENTIONS.get((resolved, me))
    if retention is not None:
        if not read_only and retention.innermost_read_only:
            raise RuntimeError("workspace_database.read_only_retention_cannot_write")
        root = retention.root
        return cast(
            duckdb.DuckDBPyConnection,
            root._child(read_only and not root._instance.read_only),
        )
    return cast(
        duckdb.DuckDBPyConnection,
        _acquire(resolved, read_only=read_only, wait_seconds=wait_seconds),
    )


@contextmanager
def retain_workspace_database(
    path: Path, *, read_only: bool, wait_seconds: float = RETENTION_WAIT_SECONDS
) -> Iterator[None]:
    """Keep one workspace database instance open for a bounded unit of work.

    A retention is per thread and per file, re-entrant for the same thread
    with the innermost declared access mode governing what the unit may open
    (a writable unit cannot nest inside a read-only one), and released when
    the last scope holding it exits -- normally or by exception, and in any
    order, since a caller may take its hold inside a unit that ends first.
    It changes no transaction boundary: each operation inside still opens its
    own cursor and commits or rolls back on its own. The engine checkpoints
    when the instance's last connection closes. A read-only retention is a
    lock (a writer of another thread waits for it) and must not span a network
    wait. A writable one is none: every other thread attaches to its instance
    without waiting, so a stage may keep one across its network edges while
    its units still release the write gate at each edge.
    """
    resolved = path.resolve()
    key = (resolved, threading.get_ident())
    scope = object()
    with _STATE:
        retention = _RETENTIONS.get(key)
    if retention is None:
        root = _acquire(resolved, read_only=read_only, wait_seconds=wait_seconds)
        with _STATE:
            retention = _Retention(root, [(scope, read_only)])
            _RETENTIONS[key] = retention
    else:
        if not read_only and retention.innermost_read_only:
            raise RuntimeError("workspace_database.read_only_retention_cannot_write")
        retention.holds.append((scope, read_only))
    try:
        yield
    finally:
        with _STATE:
            retention.holds = [hold for hold in retention.holds if hold[0] is not scope]
            last = not retention.holds
            if last:
                del _RETENTIONS[key]
        if last:
            retention.root.close()


def retained_workspace_database(path: Path) -> bool:
    """Whether the calling thread currently retains this file (for tests and probes)."""
    with _STATE:
        return (path.resolve(), threading.get_ident()) in _RETENTIONS


class LiveConnections(NamedTuple):
    """What this owner currently tracks for one file (for tests and probes)."""

    read_only: bool
    handles: int
    writers_waiting: int


def live_workspace_connections(path: Path) -> LiveConnections | None:
    """The tracked instance's mode, top-level connection count and waiting writers."""
    resolved = path.resolve()
    with _STATE:
        instance = _INSTANCES.get(resolved)
        if instance is None:
            return None
        return LiveConnections(
            instance.read_only, len(instance.handles), _WRITERS_WAITING.get(resolved, 0)
        )


def checkpoint_workspace_database(connection: duckdb.DuckDBPyConnection) -> bool:
    """Fold the write-ahead log into the file now, unless a live reader still needs it.

    A guarded reader keeps a READ ONLY transaction open between its
    statements, and one that began before this writer's last update or
    schema change still reads what that change replaced: the engine refuses
    to checkpoint over it (its "other write transactions active"), and
    forcing it would abort the reader. The log is durable as written and the
    engine folds it at the next unblocked checkpoint or when the instance
    closes, so the writer's unit goes on; False says the fold was deferred.
    Any other refusal is the writer's own and is raised.
    """
    try:
        connection.execute("CHECKPOINT")
    except duckdb.TransactionException as error:
        if "other write transactions active" not in str(error):
            raise
        return False
    return True


class WorkspaceDatabase:
    """Own one database path and explicit read/write connection modes."""

    def __init__(self, workspace: Path) -> None:
        """Create the physical workspace directory and select its market-data database path.

        Args:
            workspace: Physical workspace root used for the market-data database.
        """
        self.workspace = workspace.resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.path = self.workspace / WORKSPACE_MARKET_DATA_DATABASE_FILENAME

    def connect(self, *, read_only: bool) -> duckdb.DuckDBPyConnection:
        """Open the workspace database in the explicit requested access mode.

        Args:
            read_only: Requested access mode; a read-only request cannot acquire write authority.

        Returns:
            Tracked connection; the caller is responsible for closing it.
        """
        return open_workspace_database(self.path, read_only=read_only)

    @contextmanager
    def retain(
        self, *, read_only: bool, wait_seconds: float = RETENTION_WAIT_SECONDS
    ) -> Iterator[None]:
        """Retain this database for a bounded unit of work on the calling thread."""
        with retain_workspace_database(self.path, read_only=read_only, wait_seconds=wait_seconds):
            yield

    def seal_epoch(self) -> str:
        """The epoch whose seals this store's contents vouch for (`seal_epoch`)."""
        return seal_epoch(self.path)

    @contextmanager
    def transaction(self) -> Iterator[duckdb.DuckDBPyConnection]:
        """Yield a writable transaction, commit on success and roll back on failure.

        Yields:
            Writable workspace database connection; always closed on exit.
        """
        connection = self.connect(read_only=False)
        try:
            connection.execute("BEGIN TRANSACTION")
            yield connection
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    @contextmanager
    def read_transaction(self) -> Iterator[duckdb.DuckDBPyConnection]:
        """One read-only connection whose statements all see one consistent snapshot."""
        connection = self.connect(read_only=True)
        try:
            with cast(WorkspaceConnection, connection).snapshot():
                yield connection
        finally:
            connection.close()


class WorkspaceRepository:
    """Share only physical connection identity across domain repositories."""

    def __init__(self, workspace: Path | WorkspaceDatabase) -> None:
        """Reuse a supplied database owner or construct one from a workspace path.

        Args:
            workspace: Workspace root or shared physical database owner.
        """
        self.database = (
            workspace if isinstance(workspace, WorkspaceDatabase) else WorkspaceDatabase(workspace)
        )
        self.workspace = self.database.workspace
        self.path = self.database.path

    def _connect(self, *, read_only: bool = False) -> duckdb.DuckDBPyConnection:
        return self.database.connect(read_only=read_only)


__all__ = [
    "LiveConnections",
    "WorkspaceConnection",
    "WorkspaceDatabase",
    "WorkspaceRepository",
    "checkpoint_workspace_database",
    "live_workspace_connections",
    "open_workspace_database",
    "retain_workspace_database",
    "retained_workspace_database",
]
