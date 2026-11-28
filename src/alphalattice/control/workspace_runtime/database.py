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

While a Host serves (`keep_workspace_databases`), each file's instance is that session's: opened
once, in the mode every use the session declares needs (a file it declares it writes opens
writable at its first use, and its reads take guarded handles), kept between its connections, its
writes checkpointed into the file whenever none is open, and let go only where another reader
needs the file -- another instance's attach, a writer the session did not declare meeting a kept
read-only instance, a call in another process that declares it opens the file (`hand_off`), a
read or replacement of its bytes the instance would refuse (the storage port's `outside`) -- or
when the session ends. Each file's table schemas are read once and kept until a writer changes
them (`table_columns`).
"""

from __future__ import annotations

import json
import os
import re
import secrets
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager, suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any, Final, NamedTuple, cast
from uuid import uuid4

import duckdb

from alphalattice.capabilities.domain_contracts.domain.errors import WorkspaceConflictError
from alphalattice.capabilities.domain_contracts.environment import ducklake_extensions
from alphalattice.capabilities.domain_contracts.persistence import (
    DURABLE_REPLACE_DELAYS,
    replace_with_retry,
)
from alphalattice.capabilities.domain_contracts.spans import span
from alphalattice.control.workspace_runtime.reader_threads import (
    configure_duckdb,
    connect_duckdb,
    install_blake3_sql,
)
from alphalattice.kernel import storage
from alphalattice.kernel.execution import interruptible

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


def store_fingerprint(path: Path) -> list[list[int] | None]:
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
    staged = target.with_name(f"{target.name}.{uuid4().hex}.tmp")
    staged.write_text(json.dumps({"epoch": epoch, "fingerprint": fingerprint}), encoding="utf-8")
    try:  # a process scanning the record holds it for a moment, which Windows refuses
        replace_with_retry(staged, target, delays=DURABLE_REPLACE_DELAYS)
    finally:
        staged.unlink(missing_ok=True)


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
    held = _EPOCHS.get(_book_key(Path(path)))
    return held[0] if held is not None else secrets.token_hex(16)


def _fresh_connection(path: Path, *, read_only: bool) -> duckdb.DuckDBPyConnection:
    """Open the process's instance of the file (a metadata read; a checkpoint on close).

    The open first reads the file's fingerprint against its last recorded close (`seal_epoch`),
    and clears the record until this instance's close records it again.
    """

    before = store_fingerprint(path)
    with span("read", "duckdb_instance_open"):
        raw = _connect(path, read_only=read_only)
    try:
        # Layered Feature views persist across closes. Reattach to an already
        # open instance through _attached_connection, which inherits the function.
        install_blake3_sql(raw)
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
    _changed(path)  # the file may have changed since this process last held it
    return raw


def _attached_connection(path: Path, *, read_only: bool) -> duckdb.DuckDBPyConnection:
    """One more connection to the instance this owner already tracks, which the engine attaches
    to it: no instance of the file opens."""
    with storage.attaching():
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

    __slots__ = (
        "__weakref__",
        "_closed",
        "_ddl",
        "_guarded",
        "_instance",
        "_on_use",
        "_owner",
        "_path",
        "_pinned",
        "_raw",
        "_root",
        "_since",
        "_top",
        "_transacted",
        "_view",
    )

    BLAKE3_SQL_INSTALLED: Final = True
    """Every instance this owner opens installs the digest at its open (`_fresh_connection`),
    and its connections share it, so `install_blake3_sql` asks no catalog for them."""

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
        self._ddl = False  # a schema change since this handle's last revision mark
        # A guarded snapshot's: the file's schema revision when it began; None while each statement
        # sees the file's latest catalog (`table_columns`).
        self._view: int | None = None
        # A writer's: a transaction statement ran (the engine's parser says so) since the engine
        # last reported no transaction open on this handle (`in_transaction`).
        self._transacted = False
        # A writer's: the file's schema revision when its last transaction statement ran, None
        # when that statement also changed the schema. While the revision stays there, no schema
        # statement ran on the file since, so its transaction sees the latest committed catalog.
        self._since: int | None = None
        self._root: duckdb.DuckDBPyConnection | None = None  # the instance's, closed after this
        # What runs before the first statement that names its relation (`bind_on_use`).
        self._on_use: dict[str, Callable[[], None]] = {}
        if guarded:
            self._raw.execute("BEGIN TRANSACTION READ ONLY")

    # -- the read-only guard ---------------------------------------------

    def _end_transaction(self) -> None:
        # COMMIT ends a read-only transaction whether it is live or aborted,
        # keeps the session's registered views and temporary tables, and is
        # refused only when no transaction is open.
        self._raw.complete()
        with suppress(duckdb.TransactionException):
            self._raw.execute("COMMIT")

    def _arm(self) -> None:
        if not self._guarded or self._pinned:
            return
        self._end_transaction()
        self._raw.execute("BEGIN TRANSACTION READ ONLY")

    def attach_market_catalog(self, root: Path) -> None:
        """Attach the installed read-only market catalog once to this workspace instance."""
        if root == self._instance.catalog_named and self._instance.catalog_attached:
            return
        with self._instance.catalog_lock:
            self._market_root(root)
            if self._instance.catalog_attached:
                return
            if self._pinned or self.in_transaction():
                raise RuntimeError("workspace_database.market_catalog_attach_in_transaction")
            if self._guarded:
                self._end_transaction()
            try:
                self._attach_catalog()
            finally:
                if self._guarded:
                    self._arm()

    def _market_root(self, named: Path) -> None:
        """Hold `named` as the one catalog root this instance reads market rows from, refused by
        name, whether or not a transaction is open, when the instance reads another root's or
        the root has no catalog. Called under the instance's catalog lock."""
        if named == self._instance.catalog_named:  # every open asks: no path resolution once held
            return
        root = named.resolve()
        if self._instance.catalog_root == root:
            return
        if self._instance.catalog_root is not None:
            raise ValueError("workspace_database.market_catalog_root_changed")
        if not (root / "machine" / "catalog.sqlite").is_file():
            raise ValueError("workspace_database.market_catalog_unavailable")
        self._instance.catalog_root = root
        self._instance.catalog_named = named

    def _attach_catalog(self) -> None:
        """Load the catalog's extensions into the instance and attach its catalog: about a
        tenth of a second, since the engine checks each extension's signature at every load.
        Called under the instance's catalog lock, outside any transaction on this handle."""
        root = cast(Path, self._instance.catalog_root)
        catalog_name = (root / "machine" / "catalog.sqlite").as_posix().replace("'", "''")
        data_path = (root / "data").as_posix().replace("'", "''")
        for _, binary in ducklake_extensions():
            self._raw.execute(f"LOAD '{binary}'").complete()
        self._raw.execute(
            f"ATTACH 'ducklake:sqlite:{catalog_name}' AS workspace_market"
            f" (DATA_PATH '{data_path}/', READ_ONLY)"
        ).complete()
        self._instance.catalog_attached = True

    def bind_on_use(self, binds: Mapping[str, Callable[[], None]]) -> None:
        """Run each relation's bind on this handle before its first statement whose text names
        that relation, and every one still pending before a statement that begins a transaction:
        a handle pays only for the relations it reads, and none is bound inside a transaction
        its rollback would undo."""
        self._on_use = dict(binds)

    def _bind_used(self, args: tuple[Any, ...], kwargs: dict[str, Any], *, now: bool) -> None:
        """Run the pending `bind_on_use` binds this call's SQL or names reach. Before a
        transaction begins each pending bind is tried: one that fails stays pending, so a
        transaction that never reads its relation is not refused for it, and its first read
        inside the transaction runs it again and refuses there."""
        if not self._on_use:
            return
        names: Iterable[str] = self._on_use
        if not now:
            texts = [
                text.lower()
                for value in (*args, *kwargs.values())
                for text in (
                    value
                    if isinstance(value, str)
                    else value.query
                    if isinstance(value, duckdb.Statement)
                    else "",
                )
                if text
            ]
            now = any(_BEGINS.search(text) for text in texts)
            if not now:
                names = [name for name in names if any(name in text for text in texts)]
        for name in list(names):
            bind = self._on_use.pop(name, None)  # a bind's own statement may run another
            if bind is None:
                continue
            if not now:
                bind()
                continue
            try:
                bind()
            except (ValueError, RuntimeError, duckdb.Error):
                self._on_use.setdefault(name, bind)  # its own read refuses, if one comes

    def market_fact(self, fact: str, probe: Callable[[], bool]) -> bool:
        """Whether `fact` holds of this file's market tables, `probed` until it does, once per
        instance: a fact named here, once committed, is never undone while the file is open (its
        version index installed, its selection published). A probe in a transaction may see its
        own uncommitted change, so only one outside it is kept."""
        if fact in self._instance.market_facts:
            return True
        found = probe()
        if found and not self.in_transaction():
            self._instance.market_facts.add(fact)
        return found

    def hold_market_selection(
        self,
        cache: str,
        selection: Callable[[Any], object],
        build: Callable[[duckdb.DuckDBPyConnection, bool], bool],
        root: Path,
    ) -> bool:
        """Keep the published market selection of the catalog at `root` in the instance's
        in-memory database `cache`, attached once: `build` refills it, in a transaction of its
        own, when the `selection` read there differs from the one it holds. `build` is told
        whether the instance has attached the catalog, and answers False, having written
        nothing, when the refill must read it: the catalog is attached then and the refill runs
        again, so an instance whose selection this process has read already attaches none. A
        connection in a transaction cannot refill it and answers False unless it holds that
        selection already; its reads then take the live rows."""
        with self._instance.catalog_lock:
            self._market_root(root)
            held = self._instance.market_selection
            if self._instance.market_cache_attached and selection(self) == held:
                return True
            if self._pinned or self.in_transaction():
                return False
            if self._guarded:
                self._end_transaction()
            try:
                if not self._instance.market_cache_attached:
                    self._raw.execute(f"ATTACH ':memory:' AS {cache} (READ_WRITE)").complete()
                    self._instance.market_cache_attached = True
                held, filled = self._refill(selection, build)
                if not filled:  # with the catalog attached, `build` fills the copy or raises
                    self._attach_catalog()
                    held, _ = self._refill(selection, build)
                self._instance.market_selection = held
            finally:
                if self._guarded:
                    self._arm()
        return True

    def _refill(
        self,
        selection: Callable[[Any], object],
        build: Callable[[duckdb.DuckDBPyConnection, bool], bool],
    ) -> tuple[object, bool]:
        """One refill transaction (`hold_market_selection`): the selection read in it, and whether
        `build` filled the copy, committed, or answered that it needs the catalog, rolled back."""
        self._raw.execute("BEGIN TRANSACTION")
        try:
            held = selection(self._raw)
            filled = build(self._raw, self._instance.catalog_attached)
            self._raw.execute("COMMIT" if filled else "ROLLBACK")
        except BaseException:
            self._raw.execute("ROLLBACK")
            raise
        return held, filled

    def _refuse_unguardable(
        self, query: object
    ) -> tuple[set[duckdb.StatementType], duckdb.Statement | None]:
        """Refuse, by the engine's classification, what its read-only transaction cannot stop.

        The exposed methods accept SQL text (one or several statements) or a
        statement object the engine extracted earlier; both are classified
        here before anything reaches the transaction. A single text statement carries
        its parsed form into execution.
        """

        if isinstance(query, str):
            statements = self._raw.extract_statements(query)
        elif isinstance(query, duckdb.Statement):
            statements = [query]
        else:
            return set(), None
        for statement in statements:
            if not self._guarded and statement.type == duckdb.StatementType.LOAD:
                raise RuntimeError(
                    "workspace_database.runtime_extension_refused: "
                    "use the product's installed data dependencies; runtime extension installation "
                    "and loading through a workspace connection are unavailable"
                )
            if self._guarded and statement.type in _UNGUARDABLE_STATEMENTS:
                raise RuntimeError(
                    f"workspace_database.read_only_handle_refuses_statement:{statement.type.name}"
                )
        single = statements[0] if isinstance(query, str) and len(statements) == 1 else None
        return {statement.type for statement in statements}, single

    def _began(self, began: bool) -> None:
        """A guarded snapshot began on this handle, reading the file's catalog now, or ended."""
        with _STATE:
            self._view = _REVISIONS.get(self._path, 0) if began else None

    def in_transaction(self) -> bool:
        """Whether the engine holds a transaction open on this writer, however it was begun.

        Only a transaction statement opens one, so the engine is asked only after one ran: in
        autocommit each statement runs in a transaction of its own and two reads of the id
        differ; an open transaction answers both with its own, and an aborted one refuses them.
        """
        if self._guarded or not self._transacted:
            return False
        self._raw.complete()
        try:
            first = self._raw.execute("SELECT txid_current()").fetchone()
            held: bool = first == self._raw.execute("SELECT txid_current()").fetchone()
        except duckdb.TransactionException:
            return True
        self._transacted = held
        return held

    def _mark(self, kinds: set[duckdb.StatementType]) -> None:
        """After a statement ran: a schema change moves the file's schema revision at once, and
        again when its transaction ends, so a schema read between the two is not kept."""
        if kinds & _SCHEMA_STATEMENTS:
            self._ddl = self._transacted or duckdb.StatementType.TRANSACTION in kinds
            _changed(self._path)
        elif self._ddl and duckdb.StatementType.TRANSACTION in kinds:
            self._ddl = False
            _changed(self._path)

    @contextmanager
    def snapshot(self) -> Iterator[WorkspaceConnection]:
        """Read several statements from one consistent snapshot."""
        self._bind_used((), {}, now=True)
        if self._guarded:
            self._arm()
            self._pinned = True
            self._began(True)
            try:
                yield self
            finally:
                self._pinned = False
                self._began(False)
                self._arm()
            return
        self._raw.execute("BEGIN TRANSACTION")
        self._began_writing(set())
        try:
            yield self
        except BaseException:
            with suppress(duckdb.Error):
                self._raw.execute("ROLLBACK")
            raise
        else:
            self._raw.execute("COMMIT")
        finally:
            self._mark({duckdb.StatementType.TRANSACTION})

    # -- the one dispatch ------------------------------------------------

    def _call(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        """Run one of the engine's methods on this handle under the guard."""

        self._bind_used(args, kwargs, now=name == "begin")
        attribute = getattr(self._raw, name)
        if not self._guarded and name in {"install_extension", "load_extension"}:
            raise RuntimeError(
                "workspace_database.runtime_extension_refused: "
                "use the product's installed data dependencies; runtime extension installation "
                "and loading through a workspace connection are unavailable"
            )
        # DuckDB 2.0 keeps execute()'s result until exhaustion, complete() or abort().
        # Complete before new work: abort() can fail a transaction whose statement writes.
        if name not in _RESULT_METHODS:
            with interruptible(self._raw.interrupt):
                self._raw.complete()
        kinds: set[duckdb.StatementType] = set()
        if name in _SQL_DOORS:
            kinds, parsed = self._refuse_unguardable(args[0] if args else kwargs.get("query"))
            if parsed is not None:
                if args:
                    args = (parsed, *args[1:])
                else:
                    kwargs = {**kwargs, "query": parsed}
        elif name in {"begin", "commit", "rollback"}:
            kinds = {duckdb.StatementType.TRANSACTION}
        if self._guarded:
            refused = _REFUSED_METHODS.get(name)
            if refused is not None:
                raise RuntimeError(
                    f"workspace_database.read_only_handle_refuses_statement:{refused}"
                )
            if name in _SQL_DOORS or name in _STATEMENT_DOORS:
                self._arm()
        try:
            with interruptible(self._raw.interrupt):
                result = attribute(*args, **kwargs)
        except duckdb.TransactionException as error:
            # A relation used on this handle may have aborted the transaction
            # outside any door; re-arm and try once more, as the statement
            # would simply run on a read-only instance.
            if not self._guarded or self._pinned or "aborted" not in str(error):
                self._recover()
                raise
            self._arm()
            with interruptible(self._raw.interrupt):
                result = attribute(*args, **kwargs)
        except duckdb.Error:
            self._recover()
            raise
        finally:
            if not self._guarded:
                self._mark(kinds)
                if duckdb.StatementType.TRANSACTION in kinds:
                    self._began_writing(kinds)
        return self if result is self._raw else result

    def _began_writing(self, kinds: set[duckdb.StatementType]) -> None:
        """A transaction statement ran on this writer: note the file's schema revision then."""
        self._transacted = True
        with _STATE:
            self._since = None if kinds & _SCHEMA_STATEMENTS else _REVISIONS.get(self._path, 0)

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
        if self._ddl:
            self._mark({duckdb.StatementType.TRANSACTION})
        if not self._top:
            try:
                self._raw.complete()
            finally:
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
        counted, kept = _release_top_level(self)
        try:
            with span("write", "duckdb_instance_close"):
                try:
                    if kept is not None and kept.root is None:
                        # opened while this handle still holds the instance, which it then keeps
                        kept.root = _attached_connection(self._path, read_only=kept.read_only)
                finally:
                    try:
                        self._raw.complete()
                    finally:
                        try:
                            self._raw.close()
                        finally:
                            if self._root is not None:
                                self._root.close()
        finally:
            if counted:
                _closed(self._path, kept)

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


_BEGINS = re.compile(r"\b(?:begin|start\s+transaction)\b")
"""A statement text that opens a transaction (`bind_on_use`)."""

_SQL_DOORS = frozenset({"execute", "executemany", "sql", "query", "from_query"})
"""The engine's methods that take SQL: text or a statement object, positionally or as ``query=``."""

_RESULT_METHODS = frozenset(
    {
        "fetchone",
        "fetchmany",
        "fetchall",
        "fetchdf",
        "fetch_df",
        "df",
        "fetchnumpy",
        "fetch_arrow_table",
        "fetch_record_batch",
        "fetch_df_chunk",
        "fetch_arrow_reader",
        "to_arrow_table",
        "to_arrow_reader",
        "arrow",
        "pl",
        "torch",
        "tf",
        "complete",
        "abort",
    }
)
"""Result readers retain the pending execute() result; every other call starts new work."""

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

_SCHEMA_STATEMENTS = frozenset(
    {duckdb.StatementType.CREATE, duckdb.StatementType.ALTER, duckdb.StatementType.DROP}
)
"""The statements that change a table's schema, which move the file's schema revision."""


@dataclass
class _Instance:
    """The connections alive on the process's instance of one file, and the owner's own root
    connection while the file is held (`hold_workspace_database`)."""

    read_only: bool
    handles: set[WorkspaceConnection] = field(default_factory=set)
    root: duckdb.DuckDBPyConnection | None = None
    catalog_root: Path | None = None
    """The one catalog root the instance reads market rows from, attached or not."""
    catalog_lock: threading.Lock = field(default_factory=threading.Lock)
    market_facts: set[str] = field(default_factory=set)
    catalog_named: Path | None = None  # `catalog_root` as its first reader named it
    catalog_attached: bool = False
    """Whether the instance has attached the catalog at `catalog_root`: only once one of its
    reads must read the catalog itself (`hold_market_selection`)."""
    market_cache_attached: bool = False
    market_selection: object = None
    """The published market selection held by the instance's in-memory copy
    (`hold_market_selection`)."""

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
_OUTSIDE_HELD = "workspace_database.outside_access_meets_this_threads_connection"


def _is_file_lock_refusal(error: duckdb.IOException) -> bool:
    text = str(error)
    return "process cannot access the file" in text or "Could not set lock" in text


def _letting_go(error: duckdb.Error) -> bool:
    """The engine's other form of an instance of this process still letting go: a new instance
    meets the file the dying one still holds by name ("Unique file handle conflict"), raised as
    a binder error or, by DuckDB 2.0, as a resource in use."""
    return "Unique file handle conflict" in str(error)


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
_HOLDS: dict[Path, int] = {}
"""Per file, the scopes holding its instance open between its connections
(`hold_workspace_database`)."""
_KEEPS: list[int] = [0]
"""The sessions keeping every file's instance for the process (`keep_workspace_databases`)."""
_REVISIONS: dict[Path, int] = {}
"""Per file, its schema revision: moved by a schema change and by each fresh instance."""
_HANDOFFS: dict[Path | None, int] = {}
"""Per file, the calls running in another process that open it (`hand_off`) and the accesses to
its bytes outside the engine (`_outside`), under None those that may open every file: while any
runs, the file's instance is not kept idle."""
_WRITES: dict[Path, int] = {}
"""Per file, the sessions keeping it that declare they write it (`keep_workspace_databases`):
its instance opens writable at its first use where `_WRITABLE_SHARES_READS`."""
_WRITABLE_SHARES_READS = os.name != "nt"
"""Whether another handle can read a file while a writable instance holds it. On Windows DuckDB
opens a writable store with no sharing, so no other handle opens it then, and the byte proofs
read under a live connection need a read-only instance: there a declared write opens nothing
early, the first writer reopens the file as on any other session, and a read unit meeting the
kept writable instance lets it go and opens the file read-only."""
_IDLE: dict[Path, _Instance] = {}
"""Per held file with no connection open, the instance kept for the next: its root connection
alone holds it, its writes checkpointed into the file."""
_CLOSING: dict[Path, int] = {}
"""Per file, the engine closes of this owner's top-level connections in flight right now:
counted in the same critical section that untracks the handle, counted down when the
engine's close returns. Every close is counted, not the last tracked handle's alone: the
engine releases the file with whichever connection drops the last reference, and two
handles closing at once may do so in either order -- the first to leave the books can be
the last to let go of the file, after the other's mark would have been lifted."""


_DEVICE_PATH = "workspace_database.device_path_unsupported"


def _book_key(path: Path) -> Path:
    """The file's one key in these books, which is also the path the engine opens: resolved, a
    Windows extended drive-letter or UNC path (``\\\\?\\D:\\``, ``\\\\?\\UNC\\``) in its ordinary
    spelling, so it names the instance an ordinary one opened. Any other device path (a volume
    GUID's) has no ordinary spelling, and the engine would open it as another file relative to
    the working directory, so it is refused by name."""
    text = str(path.resolve())
    if text.startswith("\\\\?\\UNC\\"):
        return Path("\\\\" + text[8:])
    if text.startswith("\\\\?\\") and text[4:5].isalpha() and text[5:7] == ":\\":
        return Path(text[4:])
    if text.startswith(("\\\\?\\", "\\\\.\\")):
        raise RuntimeError(
            f"{_DEVICE_PATH}: {text} is a device path, which DuckDB opens as a relative file; "
            "give the folder a drive letter or a UNC name and open the file through it"
        )
    return Path(text)


def _held(path: Path) -> bool:
    """Whether the file's instance is kept between its connections: a hold on it, or a session
    keeping every file. Must be called under `_STATE`."""
    return bool(_HOLDS.get(path)) or bool(_KEEPS[0])


def _writes_early(path: Path) -> bool:
    """Whether the file's first use opens it writable for a write its session declares: only
    where a writable instance still lets other handles read it (`_WRITABLE_SHARES_READS`), and
    only a file this process may write, since one it may not (a read-only copy) takes no write.
    Must be called under `_STATE`."""
    return _WRITABLE_SHARES_READS and bool(_WRITES.get(path)) and os.access(path, os.W_OK)


def _handed_out(path: Path) -> bool:
    """Whether a call in another process, or an access outside the engine, may open the file
    now, so its instance is not kept idle. Must be called under `_STATE`."""
    return bool(_HANDOFFS.get(None)) or bool(_HANDOFFS.get(path))


def _changed(path: Path) -> None:
    """The file's schema may have changed: its kept schemas are read again."""
    with _STATE:
        _REVISIONS[path] = _REVISIONS.get(path, 0) + 1


def _release_top_level(handle: WorkspaceConnection) -> tuple[bool, _Instance | None]:
    """Take the handle off the books and count its engine close as in flight, in one
    critical section; when it was the instance's last, the instance leaves the books too.
    True says the caller owes `_closed` when the engine's close returns; False, that the
    handle was not on the books and nothing is owed. The instance comes back when the file is
    held: the close keeps it for the next opener (`_closed`), through a root connection of the
    owner's own, so the handle's close neither checkpoints nor releases the file."""
    with _STATE:
        instance = _INSTANCES.get(handle._path)
        if instance is None or handle not in instance.handles:
            return False, None
        instance.handles.discard(handle)
        kept = None
        if not instance.handles:
            del _INSTANCES[handle._path]
            if _held(handle._path) and not _handed_out(handle._path):
                kept = instance
            else:  # the root closes after this handle, whose cursor it may be
                handle._root, instance.root = instance.root, None
        _CLOSING[handle._path] = _CLOSING.get(handle._path, 0) + 1
        _STATE.notify_all()
        return True, kept


def _let_go(path: Path) -> None:
    """Close the file's idle instance, if one is kept: the engine checkpoints and releases the
    file, and the close is counted as any other (`_closed`). A close of the file still in flight
    is waited for first, since it may yet keep its instance idle."""
    with _STATE:
        _STATE.wait_for(lambda: not _CLOSING.get(path))
        idle = _IDLE.pop(path, None)
        if idle is None or idle.root is None:
            return
        _CLOSING[path] = _CLOSING.get(path, 0) + 1
    try:
        with span("write", "duckdb_instance_close"):
            idle.root.close()
    finally:
        _closed(path)


def _closed(path: Path, kept: _Instance | None = None) -> None:
    """The engine's close of one connection returned (or raised): the instance it `kept` is idle
    for the next opener while the file is still held, no call runs in another process and no
    opener took the file meanwhile, else its root closes now, as this close's last step. Then the
    close is counted down, and the openers waiting on the file are woken when no close is in
    flight any more. A writable instance is kept only once its writes are checkpointed into the
    file, so the file at rest holds them as a close would leave it; inside a held unit they are
    checkpointed once, at the unit's end (`hold_workspace_database`), and stay in the engine's
    durable log until then."""
    root = None if kept is None else kept.root
    folded = root is None or cast(_Instance, kept).read_only or _unit_holds(path) or _folded(root)
    with _STATE:
        idle = (
            root is not None
            and folded
            and _held(path)
            and not _handed_out(path)
            and path not in _INSTANCES
            and path not in _IDLE
        )
        if idle:
            _IDLE[path] = cast(_Instance, kept)
    try:
        if root is not None and not idle:
            root.close()
    finally:
        _counted_down(path)


def _unit_holds(path: Path) -> bool:
    """Whether a unit holds the file now (`hold_workspace_database`): its writes are checkpointed
    at its end, not at each close or explicit checkpoint within it."""
    with _STATE:
        return bool(_HOLDS.get(path))


def _fold(path: Path) -> None:
    """Checkpoint the file's kept instance once a held unit ends while the session still keeps
    it: the instance leaves the books as for a close (`_let_go`), and `_closed` folds its writes
    and keeps it idle again. A close of the file still in flight is waited for first."""
    with _STATE:
        _STATE.wait_for(lambda: not _CLOSING.get(path))
        idle = _IDLE.pop(path, None)
        if idle is None or idle.root is None:
            return
        _CLOSING[path] = _CLOSING.get(path, 0) + 1
    _closed(path, idle)


def _folded(root: duckdb.DuckDBPyConnection) -> bool:
    """Whether the kept instance's writes are now checkpointed into its file; False when the
    engine defers or refuses it, and the instance is then closed, which folds them."""
    try:
        return checkpoint_workspace_database(root)
    except duckdb.Error:
        return False


def _counted_down(path: Path) -> None:
    """One engine close fewer in flight; the last seals the file as its last writable close
    leaves it, unless an instance of it is still open or kept, and wakes its waiting openers."""
    with _STATE:
        _CLOSING[path] -= 1
        if not _CLOSING[path]:
            del _CLOSING[path]
            if path not in _INSTANCES and path not in _IDLE:
                # The file as this owner's last writable close leaves it (`seal_epoch`).
                epoch, read_only = _EPOCHS.pop(path, ("", True))
                if not read_only:
                    # The open cleared the record, so a close that can't write it reads as a crash
                    # (`seal_epoch`): seals are re-derived, and the Task that closed goes on.
                    with suppress(WorkspaceConflictError):
                        _write_seal_record(path, epoch=epoch, fingerprint=store_fingerprint(path))
        _STATE.notify_all()


def _acquire(
    path: Path, *, read_only: bool, wait_seconds: float, unit: bool = False
) -> WorkspaceConnection:
    """A top-level connection in the requested access mode, waiting when it must.

    - no live instance: open one in the requested mode, or writable for a file a
      keeping session declares it writes, whose reads then take the guard (unless
      a writer is waiting for the file, which a new reader lets go first);
    - a kept writable instance and a read-only unit proving the file's bytes
      (`retain_workspace_database`) where a writable instance shares nothing
      (`_WRITABLE_SHARES_READS`): let it go and open read-only, since the unit's byte
      proofs read the file under its connection;
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
            connection: WorkspaceConnection | None = None
            while connection is None:
                instance = _INSTANCES.get(path)
                # DuckDB can join the same engine even after the last handle left our
                # books. It still owns its attached catalogs until that close returns.
                if instance is None and _CLOSING.get(path):
                    wait(_HELD)
                    continue
                idle = _IDLE.get(path)
                if instance is None and idle is not None:
                    if idle.read_only and not read_only:  # a writer: the file reopens writable
                        _let_go(path)
                        continue
                    if unit and read_only and not idle.read_only and not _WRITABLE_SHARES_READS:
                        _let_go(path)  # a read unit: its byte proofs need the file shared
                        continue
                    del _IDLE[path]
                    _INSTANCES[path] = idle
                    # the instance takes the threads applied now, as a fresh open would
                    raw = configure_duckdb(cast(duckdb.DuckDBPyConnection, idle.root).cursor())
                    connection = _register(
                        idle, raw, path, guarded=read_only and not idle.read_only
                    )
                    break
                if instance is None:
                    if read_only and _WRITERS_WAITING.get(path, 0):
                        wait(_HELD)
                        continue
                    opened_read_only = read_only and not _writes_early(path)
                    raw = _open(path, read_only=opened_read_only, wait=wait, fresh=True)
                    if raw is None:
                        continue
                    instance = _Instance(read_only=opened_read_only)
                    _INSTANCES[path] = instance
                    connection = _register(
                        instance, raw, path, guarded=read_only and not opened_read_only
                    )
                    break
                if instance.read_only == read_only:
                    if read_only and _WRITERS_WAITING.get(path, 0) and not instance.held_by(me):
                        wait(_HELD)
                        continue
                    raw = _open(path, read_only=read_only, wait=wait)
                    if raw is None:
                        continue
                    connection = _register(instance, raw, path, guarded=False)
                    break
                if not instance.read_only:
                    raw = _open(path, read_only=False, wait=wait)
                    if raw is None:
                        continue
                    connection = _register(instance, raw, path, guarded=True)
                    break
                if instance.held_by(me):
                    raise RuntimeError("workspace_database.access_mode_conflict_on_this_thread")
                if not waiting_as_writer:
                    _WRITERS_WAITING[path] = _WRITERS_WAITING.get(path, 0) + 1
                    waiting_as_writer = True
                wait(_HELD)
            return connection
        finally:
            if waiting_as_writer:
                _WRITERS_WAITING[path] -= 1
                if not _WRITERS_WAITING[path]:
                    del _WRITERS_WAITING[path]
                _STATE.notify_all()


def _open(
    path: Path, *, read_only: bool, wait: Callable[[str], None], fresh: bool = False
) -> duckdb.DuckDBPyConnection | None:
    """A connection to the file, or None after waiting when the engine refuses it for an
    instance of this process; the caller re-reads its bookkeeping and tries again within the
    bound. A fresh open starts the process's instance; otherwise it is one more connection to
    the instance this owner tracks.

    The engine refuses an instance of this process going away (its last handle closed an
    instant ago) in three forms, each waited for: another access mode still held
    (ConnectionException), its file lock (IOException), or its unique file handle
    (BinderException or ResourceInUseException, `_letting_go`). A lock the engine names with
    another pid is refused by name (`_lock_refusal`); a tracked instance is not proof that every
    lock is this process's.
    """
    try:
        if fresh:
            return _fresh_connection(path, read_only=read_only)
        return _attached_connection(path, read_only=read_only)
    except duckdb.ConnectionException:
        wait(_UNTRACKED)
        return None
    except (duckdb.BinderException, duckdb.ResourceInUseException) as error:
        if not _letting_go(error):
            raise
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


_CONNECTION_BINDS: list[Callable[[duckdb.DuckDBPyConnection, Path], None]] = []
"""What each connection a `WorkspaceDatabase` opens runs first (`bind_each_connection`)."""


def bind_each_connection(bind: Callable[[duckdb.DuckDBPyConnection, Path], None]) -> None:
    """Run `bind(connection, path)` on each connection a `WorkspaceDatabase` opens from now on,
    before its caller's first statement: an owner whose relations read through per-connection
    views (the market version index) binds them for every repository reading the file, whichever
    opened the connection. A direct open (`open_workspace_database`: a backup, the maintenance
    registry, world's publish) reads the stored tables."""
    if bind not in _CONNECTION_BINDS:
        _CONNECTION_BINDS.append(bind)


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
    resolved = _book_key(path)
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
    path: Path,
    *,
    read_only: bool,
    wait_seconds: float = RETENTION_WAIT_SECONDS,
    proves_bytes: bool = True,
) -> Iterator[None]:
    """Keep one workspace database instance open for a bounded unit of work.

    A read-only unit is taken to prove the file's own bytes under its connection: where a writable
    instance shares nothing (`_WRITABLE_SHARES_READS`), it lets a kept writable instance go and
    opens read-only. A unit whose byte proofs read other files only says so (`proves_bytes=False`)
    and attaches to the kept instance, so the writer after it opens nothing.

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
    resolved = _book_key(path)
    key = (resolved, threading.get_ident())
    scope = object()
    with _STATE:
        retention = _RETENTIONS.get(key)
    if retention is None:
        root = _acquire(resolved, read_only=read_only, wait_seconds=wait_seconds, unit=proves_bytes)
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


@contextmanager
def hold_workspace_database(path: Path) -> Iterator[None]:
    """Keep the file's instance open between its connections for a unit of work, so a unit that
    opens and closes the file call by call opens the instance once (a metadata read) and
    checkpoints it once, at the end. Each opener still receives a connection of its own, a
    cursor of the owner's root connection on the instance (the engine's per-thread connection),
    with its own transactions; the access-mode rules are unchanged. The kept instance is closed,
    and the file released, when the last scope holding it ends; before another DuckDB instance of
    this process attaches the file (`release_workspace_database`); while a call runs in another
    process (`hand_off`); and when a writer meets a kept read-only instance. While a Host's
    session keeps every file (`keep_workspace_databases`), the last scope's end leaves the
    instance kept, its writes then checkpointed into the file."""
    resolved = _book_key(path)
    with _STATE:
        _HOLDS[resolved] = _HOLDS.get(resolved, 0) + 1
    try:
        yield
    finally:
        with _STATE:
            _HOLDS[resolved] -= 1
            last = not _HOLDS[resolved]
            if last:
                del _HOLDS[resolved]
            # a Host's session keeps the file past the unit's end (`keep_workspace_databases`)
            let_go = last and not _KEEPS[0]
        if let_go:
            _let_go(resolved)
        elif last:
            _fold(resolved)


def release_workspace_database(path: Path) -> None:
    """Close the file's kept instance (`hold_workspace_database`), as another DuckDB instance of
    this process must before it attaches the file; the next opener opens it again."""
    _let_go(_book_key(path))


def hand_off(opens: Iterable[Path] | None = None) -> Callable[[], None]:
    """A call another process runs, which opens the files `opens` names (None: any file this
    process holds): each of their kept instances is closed now, and none is kept until the
    returned end is called (once; a later call does nothing). The other files stay kept. A close
    in flight is waited for first: it keeps no instance once the call is counted, and closes its
    own."""
    keys: tuple[Path | None, ...] = (
        (None,) if opens is None else tuple(dict.fromkeys(_book_key(Path(path)) for path in opens))
    )
    with _STATE:
        for key in keys:
            _HANDOFFS[key] = _HANDOFFS.get(key, 0) + 1
    return _handing_out(keys)


def _handing_out(keys: tuple[Path | None, ...]) -> Callable[[], None]:
    """Let go of the kept instances of a hand-off already counted under `keys` (None: every
    file), once no close of theirs is in flight; its end, once."""
    with _STATE:
        _STATE.wait_for(
            lambda: not (_CLOSING if None in keys else [key for key in keys if _CLOSING.get(key)])
        )
        idle = [path for path in _IDLE if None in keys or path in keys]
    ended = threading.Event()

    def end() -> None:
        with _STATE:
            if not ended.is_set():
                ended.set()
                for key in keys:
                    _HANDOFFS[key] -= 1
                    if not _HANDOFFS[key]:
                        del _HANDOFFS[key]

    try:
        for path in idle:
            _let_go(path)
    except BaseException:
        end()
        raise
    return end


def _outside(paths: tuple[os.PathLike[str], ...], replacing: bool) -> Callable[[], None]:
    """The storage port's access to `paths` outside the engine (`storage.outside`): the kept
    instance of each that would refuse it is let go, and none is kept until the end. Windows
    refuses another handle's read of a file a writable instance holds, and every platform's
    replacement leaves a kept instance on the old contents, so a read-only instance stays for a
    read, and every instance goes for a replacement. An instance with connections open is waited
    for, bounded, as a request for the other access mode waits (`_acquire`): its last close keeps
    nothing, and the access begins once it has closed. One the calling thread holds is refused by
    name, since no wait can release it. An instance whose last close is in flight is in neither
    book yet still holds the file: its mode is its open's (`_EPOCHS`), and the access waits for
    the close, which keeps nothing while the access is counted. A process keeping no instance
    answers at once."""
    with _STATE:
        if not _IDLE and not _INSTANCES and not _CLOSING:
            return lambda: None
        names = {path.name for path in (*_IDLE, *_INSTANCES, *_CLOSING)}
    candidates = dict.fromkeys(_book_key(Path(path)) for path in paths if Path(path).name in names)
    keys: list[Path] = []
    me = threading.get_ident()
    deadline = monotonic() + RETENTION_WAIT_SECONDS
    with _STATE:
        for key in candidates:
            instance = _IDLE.get(key) or _INSTANCES.get(key)
            if instance is not None:
                read_only = instance.read_only
            elif _CLOSING.get(key):
                read_only = _EPOCHS.get(key, ("", False))[1]
            else:
                continue
            if not (replacing or (os.name == "nt" and not read_only)):
                continue
            keys.append(key)
            _HANDOFFS[key] = _HANDOFFS.get(key, 0) + 1
        try:
            while live := [key for key in keys if key in _INSTANCES]:
                if any(_INSTANCES[key].held_by(me) for key in live):
                    raise RuntimeError(_OUTSIDE_HELD)
                remaining = deadline - monotonic()
                if remaining <= 0.0:
                    raise RuntimeError(_HELD)
                _STATE.wait(remaining)
        except BaseException:
            for key in keys:
                _HANDOFFS[key] -= 1
                if not _HANDOFFS[key]:
                    del _HANDOFFS[key]
            raise
    if not keys:
        return lambda: None
    return _handing_out(tuple(keys))


storage.on_outside(_outside)


def keep_workspace_databases(writes: Iterable[Path] = ()) -> Callable[[], None]:
    """Keep every workspace file's instance for this process, as a Host serves: a file is opened
    once and kept between its connections until the returned end is called (once; a later call
    does nothing), or the process ends. A file of `writes`, one the session writes, opens
    writable at its first use where another handle can still read it (`_WRITABLE_SHARES_READS`),
    so a later write finds the instance it needs and reads take guarded handles on it. Each
    place another reader needs the file still lets it go (`hold_workspace_database`), and the
    next opener opens it again."""
    written = tuple(dict.fromkeys(_book_key(Path(path)) for path in writes))
    with _STATE:
        _KEEPS[0] += 1
        for path in written:
            _WRITES[path] = _WRITES.get(path, 0) + 1
    ended = threading.Event()

    def end() -> None:
        with _STATE:
            if ended.is_set():
                return
            ended.set()
            _KEEPS[0] -= 1
            for path in written:
                _WRITES[path] -= 1
                if not _WRITES[path]:
                    del _WRITES[path]
            idle = [] if _KEEPS[0] else [path for path in _IDLE if not _HOLDS.get(path)]
        for path in idle:
            _let_go(path)

    return end


def _let_go_every_file() -> None:
    """The storage sessions' teardown: every kept instance closes, its file sealed."""
    with _STATE:
        idle = list(_IDLE)
    for path in idle:
        _let_go(path)


storage.on_teardown(_let_go_every_file)


def install_workspace_schema(
    connection: duckdb.DuckDBPyConnection,
    owner: str,
    install: Callable[[], None],
    *,
    tables: tuple[str, ...],
    sequences: tuple[str, ...],
) -> None:
    """Install a repository's own declarations once at the committed revision."""
    if not isinstance(connection, WorkspaceConnection) or connection.in_transaction():
        install()
        return

    def revision() -> int:
        with _STATE:
            return _REVISIONS.get(connection._path, 0)

    catalog_input = (connection._path, owner, revision())

    def shape(
        catalog_input: tuple[Path, str, int] = catalog_input,
    ) -> frozenset[tuple[str, str | None]]:
        with span("schema", "installed_workspace_shape"):
            rows = connection.execute(
                "SELECT table_name, column_name FROM duckdb_columns()"
                " WHERE database_name = current_database() AND schema_name = 'main'"
                " AND table_name = ANY(?)"
                " AND table_name IN (SELECT table_name FROM duckdb_tables()"
                " WHERE database_name = current_database() AND schema_name = 'main')"
                " UNION ALL SELECT sequence_name, NULL FROM duckdb_sequences()"
                " WHERE database_name = current_database() AND schema_name = 'main'"
                " AND sequence_name = ANY(?)",
                [list(tables), list(sequences)],
            ).fetchall()
        return frozenset(rows)

    def measured(
        catalog_input: tuple[Path, str, int] = catalog_input,
    ) -> frozenset[tuple[str, str | None]]:
        with span("schema", "install_workspace"):
            install()
        return shape()

    def recheck(installed: frozenset[tuple[str, str | None]]) -> frozenset[tuple[str, str | None]]:
        return installed if installed <= shape() else measured()

    storage.schema(
        str(connection._path),
        ("install", owner),
        catalog_input[2],
        measured,
        installed_revision=revision,
        recheck=recheck,
    )


def _workspace_catalog_read[T](
    connection: duckdb.DuckDBPyConnection,
    key: str | tuple[str, str],
    read: Callable[[], T],
) -> T:
    """Share reads of the latest committed workspace catalog at its schema revision."""
    if not isinstance(connection, WorkspaceConnection):
        return read()
    with _STATE:
        revision = _REVISIONS.get(connection._path, 0)
    # A writer's open transaction reads its own catalog once a schema statement ran on the file
    # after it began (its own uncommitted change among them); until then it is the latest one.
    private = (
        connection.in_transaction() and connection._since != revision
    ) or connection._view not in (None, revision)
    catalog_input = (
        connection._path,
        key,
        revision,
        connection._view,
        id(connection) if private else None,
    )

    def measured(catalog_input: tuple[object, ...] = catalog_input) -> T:
        with span("schema", "workspace_catalog"):
            # The census uses the catalog identity, including the private writer or pinned view.
            return read()

    if private:
        return measured()
    return storage.schema(str(catalog_input[0]), key, revision, measured)


def catalog_answer[T](
    connection: duckdb.DuckDBPyConnection, question: str, read: Callable[[], T]
) -> T:
    """`read`'s answer about the connection's catalog, by its `question` name: a workspace
    file's is kept as `table_columns` keeps its columns, until a schema change moves its
    revision."""
    return _workspace_catalog_read(connection, ("question", question), read)


def table_info(
    connection: duckdb.DuckDBPyConnection, table: str
) -> tuple[tuple[int, str, str, bool, str | None, bool], ...]:
    """Read the engine's table_info rows. A missing relation retains the engine's refusal."""

    def read() -> tuple[tuple[int, str, str, bool, str | None, bool], ...]:
        return tuple(connection.execute("SELECT * FROM pragma_table_info(?)", [table]).fetchall())

    return _workspace_catalog_read(connection, ("table_info", table), read)


def table_columns(connection: duckdb.DuckDBPyConnection, table: str) -> tuple[tuple[str, str], ...]:
    """`table`'s columns in the connection's database, as (name, type) in their order; empty when
    it holds no such table. A workspace file's are read once and kept until a schema change moves
    its revision (the storage port's schema cache), for a connection that sees the file's latest
    committed catalog. Any other reads its own catalog and leaves the shared entry alone: a writer
    the engine reports holding a transaction open (its uncommitted schema changes are its own
    until the commit), and a guarded snapshot pinned before a schema change."""

    def read() -> tuple[tuple[str, str], ...]:
        rows = connection.execute(
            "SELECT column_name, data_type FROM duckdb_columns()"
            " WHERE database_name = current_database() AND schema_name = 'main'"
            " AND table_name = ? ORDER BY column_index",
            [table],
        ).fetchall()
        return tuple((str(name), str(kind)) for name, kind in rows)

    return _workspace_catalog_read(connection, table, read)


def retained_workspace_database(path: Path) -> bool:
    """Whether the calling thread currently retains this file (for tests and probes)."""
    with _STATE:
        return (_book_key(path), threading.get_ident()) in _RETENTIONS


class LiveConnections(NamedTuple):
    """What this owner currently tracks for one file (for tests and probes)."""

    read_only: bool
    handles: int
    writers_waiting: int


def live_workspace_connections(path: Path) -> LiveConnections | None:
    """The tracked instance's mode, top-level connection count and waiting writers."""
    resolved = _book_key(path)
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
    Inside a held unit the fold is the unit's, once at its end (`hold_workspace_database`), and
    this one is deferred to it. Any other refusal is the writer's own and is raised.
    """
    if isinstance(connection, WorkspaceConnection) and _unit_holds(connection._path):
        return False
    try:
        connection.execute("CHECKPOINT")
    except duckdb.TransactionException as error:
        if "other write transactions active" not in str(error):
            raise
        return False
    return True


_STORED_TIMES: Final = (
    ("data_remediation_failure_receipt", "observed_at"),
    ("workspace_maintenance_cycle", "updated_at"),
    ("sector_reference_receipt", "observed_at"),
    ("feature_panel_snapshot_manifest", "created_at"),
    ("feature_input_admission", "created_at"),
    ("feature_input_agent_case", "created_at"),
    ("feature_universe_rebuild_requirement", "created_at"),
    ("universe_source_observation", "observed_at"),
    ("current_universe_onboarding", "updated_at"),
    ("current_universe_hydration_deferred", "created_at"),
    ("provider_adjusted_series_revision", "observed_at"),
    ("provider_attempt", "observed_at"),
    ("action_audit_receipt", "observed_at"),
    ("bar_revision", "observed_at"),
    ("corporate_action_revision", "observed_at"),
)
"""The stored times the workspace's latest-row reads order by, each a naive UTC TIMESTAMP."""


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
        connection = open_workspace_database(self.path, read_only=read_only)
        try:
            for bind in _CONNECTION_BINDS:
                bind(connection, self.path)
        except BaseException:
            connection.close()
            raise
        return connection

    @contextmanager
    def retain(
        self,
        *,
        read_only: bool,
        wait_seconds: float = RETENTION_WAIT_SECONDS,
        proves_bytes: bool = True,
    ) -> Iterator[None]:
        """Retain this database for a bounded unit of work on the calling thread."""
        with retain_workspace_database(
            self.path, read_only=read_only, wait_seconds=wait_seconds, proves_bytes=proves_bytes
        ):
            yield

    def hold(self) -> AbstractContextManager[None]:
        """Keep this database's instance open between its connections
        (`hold_workspace_database`)."""
        return hold_workspace_database(self.path)

    def seal_epoch(self) -> str:
        """The epoch whose seals this store's contents vouch for (`seal_epoch`)."""
        return seal_epoch(self.path)

    def latest_stored_time(self) -> datetime | None:
        """The latest stored time the workspace's latest-row reads order by, over the tables this
        store holds, read from its committed rows; None for a store with none, and for a file no
        DuckDB reader opens, whose rows no latest-row read returns either (its owner refuses it)."""
        if not self.path.is_file():
            return None
        try:
            connection = open_workspace_database(self.path, read_only=True)
        except duckdb.IOException:
            return None
        try:
            listed = connection.execute(
                "SELECT list(table_name) FROM duckdb_tables() WHERE schema_name = 'main'"
            ).fetchone()
            held = set(listed[0] or ()) if listed is not None else set()
            reads = [
                f"SELECT max({column}) AS stamp FROM {table}"
                for table, column in _STORED_TIMES
                if table in held
            ]
            if not reads:
                return None
            union = " UNION ALL ".join(reads)
            latest = connection.execute(f"SELECT max(stamp) FROM ({union}) AS stored").fetchone()
        finally:
            connection.close()
        return None if latest is None or latest[0] is None else latest[0].replace(tzinfo=UTC)

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

    @contextmanager
    def _catalog_read(self, failure_code: str) -> Iterator[duckdb.DuckDBPyConnection]:
        """Read a required catalog without changing its schema or creating storage.

        Args:
            failure_code: The domain owner's registered refusal for unavailable schema.

        Yields:
            The physical owner's guarded read-only connection.

        Raises:
            ValueError: A required catalog table or column cannot be read.
        """
        connection = self._connect(read_only=True)
        try:
            yield connection
        except duckdb.CatalogException as error:
            raise ValueError(failure_code) from error
        finally:
            connection.close()


__all__ = [
    "LiveConnections",
    "WorkspaceConnection",
    "WorkspaceDatabase",
    "WorkspaceRepository",
    "catalog_answer",
    "checkpoint_workspace_database",
    "hand_off",
    "hold_workspace_database",
    "install_workspace_schema",
    "keep_workspace_databases",
    "live_workspace_connections",
    "open_workspace_database",
    "release_workspace_database",
    "retain_workspace_database",
    "retained_workspace_database",
    "table_columns",
    "table_info",
]
