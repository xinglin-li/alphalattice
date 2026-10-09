"""One connection owner per workspace file: retained instances, guarded reads, mode handover.

A daily update of 472 listings opened the workspace database about 20,000
times -- 43 per listing -- and every open re-read the file's metadata while
every write close checkpointed it (F8/F13 of the 2026-09-11 record). Retaining
the instance for a unit of work removes that cost. What it must not change is
here: each operation keeps its own transaction, a failure still rolls back on
its own, a read-only request never receives a writable connection (also
inside a writable retention, where the engine cannot enforce it by instance
mode), the instance is released on exception, and a request for the other
access mode waits -- bounded, and for every live connection, not only for a
retention -- instead of meeting DuckDB's configuration refusal.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic

import duckdb
import pytest

from alphalattice.control.product_host.composition.application_session import (
    WorkspaceApplicationSession,
)
from alphalattice.control.workspace_runtime.database import (
    LiveConnections,
    WorkspaceDatabase,
    checkpoint_workspace_database,
    live_workspace_connections,
    open_workspace_database,
    retain_workspace_database,
    retained_workspace_database,
)


def _prepared(tmp_path: Path) -> WorkspaceDatabase:
    database = WorkspaceDatabase(tmp_path)
    with database.transaction() as connection:
        connection.execute("CREATE TABLE unit (id INTEGER PRIMARY KEY, note VARCHAR)")
        connection.execute("INSERT INTO unit VALUES (1, 'seed')")
    return database


def _rows(database: WorkspaceDatabase) -> list[tuple[int, str]]:
    connection = database.connect(read_only=True)
    try:
        return connection.execute("SELECT id, note FROM unit ORDER BY id").fetchall()
    finally:
        connection.close()


def _until(predicate: Callable[[], bool], *, seconds: float = 10.0) -> None:
    """Wait for an observable state (a definite condition, not a timing guess)."""

    deadline = monotonic() + seconds
    while not predicate():
        assert monotonic() < deadline, "the expected state was not reached"
        threading.Event().wait(0.005)


def test_retained_unit_keeps_independent_transactions_and_rolls_back_failures(
    tmp_path: Path,
) -> None:
    """Retained unit keeps independent transactions and rolls back failures."""

    database = _prepared(tmp_path)
    with pytest.raises(RuntimeError, match="second unit"), database.retain(read_only=False):
        assert retained_workspace_database(database.path)
        with database.transaction() as first:
            first.execute("INSERT INTO unit VALUES (2, 'first')")
        with (
            pytest.raises(RuntimeError, match="second unit"),
            database.transaction() as second,
        ):
            second.execute("INSERT INTO unit VALUES (3, 'second')")
            raise RuntimeError("second unit fails after writing")
        # A read inside the unit is served by the same instance and sees
        # only what committed.
        assert _rows(database) == [(1, "seed"), (2, "first")]
        raise RuntimeError("the second unit's caller fails too")
    assert not retained_workspace_database(database.path)
    assert live_workspace_connections(database.path) is None
    assert _rows(database) == [(1, "seed"), (2, "first")]


def test_read_only_request_inside_a_writable_retention_cannot_write(tmp_path: Path) -> None:
    """Read only request inside a writable retention cannot write."""

    database = _prepared(tmp_path)
    ordinary = database.connect(read_only=True)
    try:
        with pytest.raises(duckdb.Error):
            ordinary.execute("INSERT INTO unit VALUES (2, 'ordinary')")
    finally:
        ordinary.close()
    with database.retain(read_only=False):
        reader = database.connect(read_only=True)
        try:
            with pytest.raises(duckdb.TransactionException, match="read-only"):
                reader.execute("INSERT INTO unit VALUES (3, 'inside writable retention')")
            assert reader.execute("SELECT count(*) FROM unit").fetchone() == (1,)
            with pytest.raises(duckdb.TransactionException, match="read-only"):
                reader.executemany("INSERT INTO unit VALUES (?, ?)", [(5, "many")])
            child = reader.cursor()
            try:
                with pytest.raises(duckdb.TransactionException, match="read-only"):
                    child.execute("DELETE FROM unit")
            finally:
                child.close()
        finally:
            reader.close()
        with database.transaction() as writer:
            writer.execute("INSERT INTO unit VALUES (6, 'the unit itself')")
    assert _rows(database) == [(1, "seed"), (6, "the unit itself")]


def test_guarded_handle_cannot_leave_its_read_only_transaction(tmp_path: Path) -> None:
    """Guarded handle cannot leave its read only transaction."""

    database = _prepared(tmp_path)
    with database.retain(read_only=False):
        reader = database.connect(read_only=True)
        try:
            for statement in (
                "COMMIT; INSERT INTO unit VALUES (2, 'after a commit')",
                "COMMIT",
                "ROLLBACK",
                "BEGIN TRANSACTION",
                "ATTACH ':memory:' AS elsewhere",
                "CHECKPOINT",
                "FORCE CHECKPOINT",
                "INSTALL json",
                "VACUUM",
            ):
                with pytest.raises(RuntimeError, match="read_only_handle_refuses_statement"):
                    reader.execute(statement)
            for method in (reader.begin, reader.commit, reader.rollback):
                with pytest.raises(RuntimeError, match="refuses_statement:TRANSACTION"):
                    method()
            # A relation built on the handle inserts inside the transaction
            # the consumer cannot end: refused by the engine.
            relation = reader.sql("SELECT 7 AS id, 'relation' AS note")
            with pytest.raises(duckdb.TransactionException, match="read-only"):
                relation.insert_into("unit")
            # Methods that return the engine's connection return the handle.
            import pandas as pd

            registered = reader.register("frame", pd.DataFrame({"id": [8], "note": ["frame"]}))
            with pytest.raises(duckdb.TransactionException, match="read-only"):
                registered.execute("INSERT INTO unit SELECT id, note FROM frame")
            assert registered.execute("SELECT count(*) FROM frame").fetchone() == (1,)
            twin = reader.duplicate()
            try:
                with pytest.raises(duckdb.TransactionException, match="read-only"):
                    twin.execute("INSERT INTO unit VALUES (9, 'duplicate')")
            finally:
                twin.close()
            # Session-local statements a read-only reader may issue still work.
            reader.execute("SET threads = 2")
            reader.execute("CREATE TEMP TABLE scratch AS SELECT 1 AS one")
            assert reader.execute("SELECT one FROM scratch").fetchone() == (1,)
            assert reader.execute("SELECT count(*) FROM unit").fetchone() == (1,)
        finally:
            reader.close()
    assert _rows(database) == [(1, "seed")]


def test_guarded_reader_sees_each_statement_fresh_and_a_snapshot_holds(tmp_path: Path) -> None:
    """A guarded reader sees each statement's latest committed state and an explicit read
    transaction holds one snapshot."""

    database = _prepared(tmp_path)
    with database.retain(read_only=False):
        reader = database.connect(read_only=True)
        try:
            assert reader.execute("SELECT count(*) FROM unit").fetchone() == (1,)
            with database.transaction() as writer:
                writer.execute("INSERT INTO unit VALUES (2, 'while reading')")
            assert reader.execute("SELECT count(*) FROM unit").fetchone() == (2,)
        finally:
            reader.close()
        with database.read_transaction() as snapshot:
            assert snapshot.execute("SELECT count(*) FROM unit").fetchone() == (2,)
            with database.transaction() as writer:
                writer.execute("INSERT INTO unit VALUES (3, 'after the snapshot')")
            assert snapshot.execute("SELECT count(*) FROM unit").fetchone() == (2,)
            with pytest.raises(duckdb.TransactionException, match="read-only"):
                snapshot.execute("INSERT INTO unit VALUES (4, 'in a snapshot')")
        assert _rows(database) == [(1, "seed"), (2, "while reading"), (3, "after the snapshot")]


def test_read_only_retention_refuses_a_write_instead_of_upgrading(tmp_path: Path) -> None:
    """requirement: a read-only unit never receives a writable connection, even nested."""

    database = _prepared(tmp_path)
    with database.retain(read_only=True):
        assert _rows(database) == [(1, "seed")]
        with pytest.raises(RuntimeError, match="read_only_retention_cannot_write"):
            database.connect(read_only=False)
        with (
            pytest.raises(RuntimeError, match="read_only_retention_cannot_write"),
            database.retain(read_only=False),
        ):
            pass
    # A read-only unit nested inside a writable one governs its own requests.
    with database.retain(read_only=False):
        with (
            database.retain(read_only=True),
            pytest.raises(RuntimeError, match="read_only_retention_cannot_write"),
        ):
            database.connect(read_only=False)
        with database.transaction() as connection:
            connection.execute("INSERT INTO unit VALUES (2, 'after')")
    assert _rows(database) == [(1, "seed"), (2, "after")]


def test_session_read_boundary_holds_each_store_once_and_writers_queue_behind_it(
    tmp_path: Path,
) -> None:
    """Session read boundary holds each store once and writers queue behind it."""

    database = _prepared(tmp_path)
    task_store = tmp_path / "runtime" / "research-task-control.duckdb"
    done: list[str] = []
    errors: list[BaseException] = []

    def market_writer() -> None:
        try:
            with (
                session.mutation_gate.hold(),
                database.retain(read_only=False),
                database.transaction() as connection,
            ):
                connection.execute("INSERT INTO unit VALUES (2, 'after the boundary')")
        except BaseException as error:  # pragma: no cover - reported through errors
            errors.append(error)
        done.append("market")

    def task_writer() -> None:
        try:
            session.task_control_registry.reconcile_after_writer_acquisition(
                observed_at=datetime.now(UTC)
            )
        except BaseException as error:  # pragma: no cover - reported through errors
            errors.append(error)
        done.append("task")

    with WorkspaceApplicationSession.acquire(tmp_path) as session:
        writers = [threading.Thread(target=market_writer), threading.Thread(target=task_writer)]
        with session.reads():
            assert live_workspace_connections(database.path) == (True, 1, 0)
            assert live_workspace_connections(task_store) == (True, 1, 0)
            for thread in writers:
                thread.start()
            # The writers are queued at the gate, not at the engine: neither
            # store gains a waiting writer, and the boundary's reads go on.
            threading.Event().wait(0.3)
            assert done == []
            assert live_workspace_connections(database.path) == (True, 1, 0)
            connection = open_workspace_database(database.path, read_only=True)
            try:
                assert connection.execute("SELECT count(*) FROM unit").fetchone()[0] == 1
                assert live_workspace_connections(database.path) == (True, 1, 0)
            finally:
                connection.close()
            assert session.task_control_registry.tasks() == ()
            assert live_workspace_connections(task_store) == (True, 1, 0)
            with pytest.raises(RuntimeError, match="read_only_retention_cannot_write"):
                open_workspace_database(database.path, read_only=False)
            with pytest.raises(RuntimeError, match="access_mode_conflict_on_this_thread"):
                open_workspace_database(task_store, read_only=False)
            assert done == []
        for thread in writers:
            thread.join(timeout=20)
        assert errors == []
        assert sorted(done) == ["market", "task"]
        assert _rows(database) == [(1, "seed"), (2, "after the boundary")]
        assert live_workspace_connections(database.path) is None
        assert not retained_workspace_database(database.path)

    # An unprepared workspace has no market store yet: the boundary holds the
    # Task store alone and the market store's later creation is not refused.
    with WorkspaceApplicationSession.acquire(tmp_path / "fresh") as session:
        with session.reads():
            assert live_workspace_connections(tmp_path / "fresh" / "market-data.duckdb") is None
            assert session.task_control_registry.tasks() == ()
        with retain_workspace_database(tmp_path / "fresh" / "market-data.duckdb", read_only=False):
            pass


def test_writer_waits_for_an_ordinary_reader_and_wakes_when_it_closes(tmp_path: Path) -> None:
    """Writer waits for an ordinary reader and wakes when it closes."""

    database = _prepared(tmp_path)
    reader_open = threading.Event()
    release_reader = threading.Event()
    reader_seen: list[object] = []

    def reader() -> None:
        connection = open_workspace_database(database.path, read_only=True)
        try:
            reader_seen.append(connection.execute("SELECT count(*) FROM unit").fetchone()[0])
            reader_open.set()
            release_reader.wait(timeout=10)
        finally:
            connection.close()

    with database.retain(read_only=False):
        pass
    assert live_workspace_connections(database.path) is None
    holder = threading.Thread(target=reader)
    holder.start()
    assert reader_open.wait(timeout=10)
    assert live_workspace_connections(database.path) == (True, 1, 0)

    with (
        pytest.raises(RuntimeError, match="access_mode_held_by_another_thread"),
        database.retain(read_only=False, wait_seconds=0.2),
    ):
        pass

    acquired = threading.Event()
    writer_error: list[BaseException] = []

    def writer() -> None:
        try:
            with database.retain(read_only=False):
                with database.transaction() as connection:
                    connection.execute("INSERT INTO unit VALUES (2, 'after the reader')")
                acquired.set()
        except BaseException as error:  # pragma: no cover - reported through writer_error
            writer_error.append(error)
            acquired.set()

    waiting = threading.Thread(target=writer)
    waiting.start()
    _until(lambda: live_workspace_connections(database.path) == (True, 1, 1))
    assert not acquired.is_set()
    release_reader.set()
    holder.join(timeout=10)
    assert acquired.wait(timeout=10)
    waiting.join(timeout=10)
    assert writer_error == []
    assert reader_seen == [1]
    assert live_workspace_connections(database.path) is None
    assert _rows(database) == [(1, "seed"), (2, "after the reader")]


def test_reader_during_a_writable_unit_reads_at_once_and_cannot_write(tmp_path: Path) -> None:
    """Reader during a writable unit reads at once and cannot write."""

    database = _prepared(tmp_path)
    entered = threading.Event()
    release_writer = threading.Event()
    reader_done = threading.Event()
    seen: list[object] = []

    def writer() -> None:
        with database.retain(read_only=False):
            with database.transaction() as connection:
                connection.execute("INSERT INTO unit VALUES (2, 'held')")
            entered.set()
            release_writer.wait(timeout=10)
        reader_done.wait(timeout=10)
        with database.retain(read_only=False), database.transaction() as connection:
            connection.execute("INSERT INTO unit VALUES (3, 'second unit')")

    holder = threading.Thread(target=writer)
    holder.start()
    assert entered.wait(timeout=10)
    connection = open_workspace_database(database.path, read_only=True, wait_seconds=0.2)
    try:
        seen.append(connection.execute("SELECT count(*) FROM unit").fetchone()[0])
        with pytest.raises(duckdb.TransactionException, match="read-only"):
            connection.execute("INSERT INTO unit VALUES (9, 'reader')")
        release_writer.set()
        _until(lambda: live_workspace_connections(database.path) == (False, 1, 0))
        seen.append(connection.execute("SELECT count(*) FROM unit").fetchone()[0])
    finally:
        connection.close()
    reader_done.set()
    holder.join(timeout=10)
    assert seen == [2, 2]
    assert live_workspace_connections(database.path) is None
    assert _rows(database) == [(1, "seed"), (2, "held"), (3, "second unit")]


def test_an_idle_guarded_reader_defers_the_writer_flush_instead_of_stopping_it(
    tmp_path: Path,
) -> None:
    """An idle guarded reader defers the writer flush instead of stopping it."""

    database = _prepared(tmp_path)
    log = database.path.with_name(database.path.name + ".wal")
    with database.retain(read_only=False):
        reader = open_workspace_database(database.path, read_only=True, wait_seconds=0.2)
        try:
            assert reader.execute("SELECT count(*) FROM unit").fetchone()[0] == 1
            with database.transaction() as connection:
                connection.execute("UPDATE unit SET note = 'revised' WHERE id = 1")
            writer = database.connect(read_only=False)
            try:
                with pytest.raises(duckdb.TransactionException, match="other write transactions"):
                    writer.execute("CHECKPOINT")
                assert checkpoint_workspace_database(writer) is False
                # The unit goes on writing, and the reader still reads.
                with database.transaction() as connection:
                    connection.execute("INSERT INTO unit VALUES (2, 'after the deferral')")
                assert reader.execute("SELECT count(*) FROM unit").fetchone()[0] == 2
            finally:
                writer.close()
        finally:
            reader.close()
        writer = database.connect(read_only=False)
        try:
            assert checkpoint_workspace_database(writer) is True
        finally:
            writer.close()
        assert not log.exists() or log.stat().st_size == 0
    assert _rows(database) == [(1, "revised"), (2, "after the deferral")]


def test_new_reader_lets_a_waiting_writer_take_the_instance_first(tmp_path: Path) -> None:
    """New reader lets a waiting writer take the instance first."""

    database = _prepared(tmp_path)
    first_open = threading.Event()
    release_first = threading.Event()
    writer_acquired = threading.Event()
    late_seen: list[object] = []
    late_served_on: list[tuple[bool, int] | None] = []

    def first_reader() -> None:
        connection = open_workspace_database(database.path, read_only=True)
        try:
            first_open.set()
            release_first.wait(timeout=10)
        finally:
            connection.close()

    def writer() -> None:
        with database.retain(read_only=False):
            writer_acquired.set()
            with database.transaction() as connection:
                connection.execute("INSERT INTO unit VALUES (2, 'writer')")
            threading.Event().wait(0.05)

    def late_reader() -> None:
        connection = open_workspace_database(database.path, read_only=True)
        try:
            late_served_on.append(live_workspace_connections(database.path))
            late_seen.append(connection.execute("SELECT count(*) FROM unit").fetchone()[0])
            with pytest.raises(duckdb.TransactionException, match="read-only"):
                connection.execute("INSERT INTO unit VALUES (9, 'late reader')")
        finally:
            connection.close()

    first = threading.Thread(target=first_reader)
    first.start()
    assert first_open.wait(timeout=10)
    waiting_writer = threading.Thread(target=writer)
    waiting_writer.start()
    # The writer is registered as waiting before the late reader arrives, so
    # the reader deterministically takes the waiting branch.
    _until(lambda: live_workspace_connections(database.path) == (True, 1, 1))
    late = threading.Thread(target=late_reader)
    late.start()
    _until(lambda: late.is_alive() and not late_seen)
    release_first.set()
    first.join(timeout=10)
    assert writer_acquired.wait(timeout=10)
    waiting_writer.join(timeout=10)
    late.join(timeout=10)
    # Served on the writer's instance (mode False), never by extending the
    # readers' instance the writer was waiting for.
    assert late_served_on and late_served_on[0] is not None and late_served_on[0].read_only is False
    assert late_seen in ([1], [2])
    assert live_workspace_connections(database.path) is None
    assert _rows(database) == [(1, "seed"), (2, "writer")]


def test_same_thread_cannot_wait_for_its_own_read_only_connection(tmp_path: Path) -> None:
    """requirement: a request that could only be satisfied by this thread is refused, not hung."""

    database = _prepared(tmp_path)
    connection = database.connect(read_only=True)
    try:
        with pytest.raises(RuntimeError, match="access_mode_conflict_on_this_thread"):
            database.connect(read_only=False)
    finally:
        connection.close()
    with database.transaction() as writer:
        writer.execute("INSERT INTO unit VALUES (2, 'after closing')")
    assert _rows(database) == [(1, "seed"), (2, "after closing")]


def test_guard_holds_through_keyword_and_statement_doors(tmp_path: Path) -> None:
    """The read-only guard classifies every exposed SQL keyword and statement entry."""

    database = _prepared(tmp_path)
    with database.retain(read_only=False):
        reader = database.connect(read_only=True)
        try:
            bypass = "COMMIT; INSERT INTO unit VALUES (2, 'bypass'); SELECT * FROM unit"
            with pytest.raises(RuntimeError, match="refuses_statement:TRANSACTION"):
                reader.from_query(bypass)
            with pytest.raises(RuntimeError, match="refuses_statement:TRANSACTION"):
                reader.from_query(query=bypass)
            statements = reader.extract_statements("COMMIT; INSERT INTO unit VALUES (2, 'object')")
            with pytest.raises(RuntimeError, match="refuses_statement:TRANSACTION"):
                reader.execute(statements[0])
            with pytest.raises(RuntimeError, match="refuses_statement:TRANSACTION"):
                reader.execute(query=statements[0])
            with pytest.raises(RuntimeError, match="refuses_statement:TRANSACTION"):
                reader.from_query(statements[0])
            with pytest.raises(RuntimeError, match="refuses_statement:TRANSACTION"):
                reader.sql(query="COMMIT")
            with pytest.raises(RuntimeError, match="refuses_statement:TRANSACTION"):
                reader.query("ROLLBACK")
            with pytest.raises(RuntimeError, match="refuses_statement:CALL"):
                reader.checkpoint()
            with pytest.raises(RuntimeError, match="refuses_statement:LOAD"):
                reader.install_extension("json")
            with pytest.raises(RuntimeError, match="refuses_statement:LOAD"):
                reader.load_extension("json")
            with pytest.raises(duckdb.TransactionException, match="read-only"):
                reader.execute(statements[1])
            with pytest.raises(duckdb.TransactionException, match="read-only"):
                reader.from_query(statements[1]).fetchall()
            with pytest.raises(duckdb.TransactionException, match="read-only"):
                reader.executemany(query="INSERT INTO unit VALUES (?, ?)", parameters=[(3, "kw")])
            # Valid reads on every door still work after the refusals.
            assert reader.execute(query="SELECT count(*) FROM unit").fetchone() == (1,)
            assert reader.sql(query="SELECT count(*) FROM unit").fetchone() == (1,)
            assert reader.from_query("SELECT count(*) FROM unit").fetchone() == (1,)
        finally:
            reader.close()
    # Persisted state, read back on a fresh instance after every connection
    # of the writable unit has closed: still the one seed row.
    assert live_workspace_connections(database.path) is None
    assert _rows(database) == [(1, "seed")]


def test_child_handles_end_with_their_parent_and_the_instance_is_released(
    tmp_path: Path,
) -> None:
    """Child handles end with their parent and the instance is released."""

    database = _prepared(tmp_path)
    with database.retain(read_only=False):
        kept = database.connect(read_only=True)
        assert kept.execute("SELECT count(*) FROM unit").fetchone() == (1,)
    assert live_workspace_connections(database.path) is None
    with pytest.raises(duckdb.ConnectionException):
        kept.execute("SELECT 1")
    kept.close()
    kept.close()
    assert _rows(database) == [(1, "seed")]


def test_an_instance_of_this_process_still_letting_go_is_waited_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An instance of this process still letting go is waited for."""

    from alphalattice.control.workspace_runtime import database as broker

    database = _prepared(tmp_path)

    def lock_message(pid: int) -> str:
        return (
            f'IO Error: Cannot open file "{database.path}": The process cannot access the file '
            "because it is being used by another process.\n\nFile is already open in \n"
            f"python.exe (PID {pid})"
        )

    attempts: list[str] = []
    real_connect = broker._connect

    def refuse_first(path: Path, *, read_only: bool) -> duckdb.DuckDBPyConnection:
        attempts.append("ro" if read_only else "rw")
        if len(attempts) == 1:
            raise duckdb.IOException(lock_message(os.getpid()))
        return real_connect(path, read_only=read_only)

    # The fresh path: nothing tracked, the engine refuses once with this process's own
    # lock, the broker waits and opens on the retry.
    monkeypatch.setattr(broker, "_connect", refuse_first)
    with database.transaction() as writer:
        writer.execute("INSERT INTO unit VALUES (2, 'after the wait')")
    assert attempts == ["rw", "rw"]
    monkeypatch.undo()
    # The attached path: another thread holds a writable instance; this thread's read-only
    # request attaches (guarded); the engine refuses that attach once the same way.
    attached_attempts: list[int] = []
    real_attached = broker._attached_connection

    def attach_refusing_first(path: Path, *, read_only: bool) -> duckdb.DuckDBPyConnection:
        attached_attempts.append(1)
        if len(attached_attempts) == 1:
            raise duckdb.IOException(lock_message(os.getpid()))
        return real_attached(path, read_only=read_only)

    monkeypatch.setattr(broker, "_attached_connection", attach_refusing_first)
    entered, release = threading.Event(), threading.Event()

    def writer_thread() -> None:
        with database.retain(read_only=False):
            entered.set()
            release.wait(timeout=10)

    holder = threading.Thread(target=writer_thread)
    holder.start()
    assert entered.wait(timeout=10)
    reader = open_workspace_database(database.path, read_only=True, wait_seconds=2.0)
    try:
        assert reader.execute("SELECT count(*) FROM unit").fetchone()[0] == 2
    finally:
        reader.close()
        release.set()
        holder.join(timeout=10)
    assert len(attached_attempts) == 2
    monkeypatch.undo()
    # The engine's plain sharing violation (no process named) while this owner is closing
    # the file's last connection is that dying instance's lock: waited for, then served
    # once the close returns and lifts the mark.
    attempts.clear()
    plain = (
        f'IO Error: Cannot open file "{database.path}": The process cannot access the file '
        "because it is being used by another process."
    )
    resolved = database.path.resolve()

    def refuse_while_closing(path: Path, *, read_only: bool) -> duckdb.DuckDBPyConnection:
        attempts.append("ro" if read_only else "rw")
        if broker._CLOSING.get(path):
            raise duckdb.IOException(plain)
        return real_connect(path, read_only=read_only)

    monkeypatch.setattr(broker, "_connect", refuse_while_closing)
    with broker._STATE:
        broker._CLOSING[resolved] = 1
    threading.Timer(0.15, broker._closed, args=(resolved,)).start()
    reader = open_workspace_database(database.path, read_only=True, wait_seconds=2.0)
    try:
        assert reader.execute("SELECT count(*) FROM unit").fetchone()[0] == 2
    finally:
        reader.close()
    assert attempts[0] == "ro" and attempts[-1] == "ro" and len(attempts) >= 2
    assert not broker._CLOSING
    monkeypatch.undo()
    # Refused by name, at once, with no wait consumed: a lock named with another pid is
    # another process; a plain violation with no close in flight is a holder this owner
    # cannot name -- never retried on a guess about timing.
    monkeypatch.setattr(
        broker,
        "_connect",
        lambda path, *, read_only: (_ for _ in ()).throw(duckdb.IOException(lock_message(1))),
    )
    with pytest.raises(RuntimeError, match="file_held_by_another_process"):
        open_workspace_database(database.path, read_only=True, wait_seconds=0.2)
    monkeypatch.undo()
    monkeypatch.setattr(
        broker,
        "_connect",
        lambda path, *, read_only: (_ for _ in ()).throw(duckdb.IOException(plain)),
    )
    started = monotonic()
    with pytest.raises(RuntimeError, match="file_held_by_an_unknown_holder"):
        open_workspace_database(database.path, read_only=True, wait_seconds=2.0)
    assert monotonic() - started < 0.5
    monkeypatch.undo()
    assert live_workspace_connections(database.path) is None
    assert _rows(database) == [(1, "seed"), (2, "after the wait")]


def test_the_attached_path_refuses_another_process_by_name_and_waits_finitely(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The attached path refuses another process by name and waits finitely."""

    from alphalattice.control.workspace_runtime import database as broker

    database = _prepared(tmp_path)

    def lock_message(pid: int) -> str:
        return (
            f'IO Error: Cannot open file "{database.path}": The process cannot access the file '
            "because it is being used by another process.\n\nFile is already open in \n"
            f"python.exe (PID {pid})"
        )

    waits: list[str] = []
    entered, release = threading.Event(), threading.Event()

    def writer_thread() -> None:
        with database.retain(read_only=False):
            entered.set()
            release.wait(timeout=10)

    holder = threading.Thread(target=writer_thread)
    holder.start()
    assert entered.wait(timeout=10)
    try:
        # Another pid named on the attached path: refused by name, no wait consumed.
        monkeypatch.setattr(
            broker,
            "_attached_connection",
            lambda path, *, read_only: (_ for _ in ()).throw(duckdb.IOException(lock_message(1))),
        )
        real_wait_slice = broker._CONNECTION_REFUSED_RETRY_SECONDS
        started = monotonic()
        with pytest.raises(RuntimeError, match="file_held_by_another_process"):
            open_workspace_database(database.path, read_only=True, wait_seconds=2.0)
        assert monotonic() - started < real_wait_slice + 0.5
        # This process's own lock that never lets go: waited for within the bound, then
        # refused as untracked -- finite, and the holder's transaction is untouched.
        attempts: list[int] = []

        def own_lock_forever(path: Path, *, read_only: bool) -> duckdb.DuckDBPyConnection:
            attempts.append(1)
            waits.append("attempt")
            raise duckdb.IOException(lock_message(os.getpid()))

        monkeypatch.setattr(broker, "_attached_connection", own_lock_forever)
        started = monotonic()
        with pytest.raises(RuntimeError, match="held_outside_this_owner"):
            open_workspace_database(database.path, read_only=True, wait_seconds=0.3)
        assert 0.3 <= monotonic() - started < 3.0
        assert len(attempts) >= 2
    finally:
        monkeypatch.undo()
        release.set()
        holder.join(timeout=10)
    assert live_workspace_connections(database.path) is None
    assert _rows(database) == [(1, "seed")]


def test_a_closing_instance_is_untracked_first_and_openers_wait_for_the_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A closing instance is untracked first and openers wait for the close."""

    from alphalattice.control.workspace_runtime import database as owner

    database = _prepared(tmp_path)
    seen: list[LiveConnections | None] = []
    real_fresh = owner._fresh_connection

    class _Recording:
        """The raw connection, with the owner's view recorded at its close."""

        def __init__(self, raw: duckdb.DuckDBPyConnection) -> None:
            self._raw = raw

        def close(self) -> None:
            seen.append(live_workspace_connections(database.path))
            self._raw.close()

        def __getattr__(self, name: str) -> object:
            return getattr(self._raw, name)

    monkeypatch.setattr(
        owner,
        "_fresh_connection",
        lambda path, *, read_only: _Recording(real_fresh(path, read_only=read_only)),
    )
    connection = database.connect(read_only=False)
    assert live_workspace_connections(database.path) == LiveConnections(False, 1, 0)
    connection.close()
    assert seen == [None], "the engine closed the instance after the owner stopped tracking it"
    monkeypatch.setattr(owner, "_fresh_connection", real_fresh)

    # The lock met while a close is in flight is waited for; the same lock
    # with no close in flight is another process's, refused at once.
    locked = duckdb.IOException(
        'IO Error: Cannot open file "x": The process cannot access the file because '
        "it is being used by another process."
    )
    attempts: list[int] = []

    def locked_then_open(path: Path, *, read_only: bool) -> duckdb.DuckDBPyConnection:
        attempts.append(len(attempts))
        if owner._CLOSING.get(path):
            raise locked
        return real_fresh(path, read_only=read_only)

    monkeypatch.setattr(owner, "_fresh_connection", locked_then_open)
    opened: list[duckdb.DuckDBPyConnection] = []
    resolved = database.path.resolve()
    with owner._STATE:
        owner._CLOSING[resolved] = 1
    opener = threading.Thread(
        target=lambda: opened.append(database.connect(read_only=True)), daemon=True
    )
    opener.start()
    _until(lambda: len(attempts) >= 3)
    assert opened == [] and opener.is_alive()
    owner._closed(resolved)
    opener.join(timeout=10.0)
    assert not opener.is_alive() and len(opened) == 1
    assert opened[0].execute("SELECT count(*) FROM unit").fetchone() == (1,)
    opened[0].close()
    monkeypatch.setattr(
        owner, "_fresh_connection", lambda path, *, read_only: (_ for _ in ()).throw(locked)
    )
    with pytest.raises(RuntimeError, match="file_held_by_an_unknown_holder"):
        database.connect(read_only=True)


def test_the_handle_leaves_the_books_and_the_file_is_marked_closing_in_one_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Closing a handle atomically untracks it and marks its database instance as closing."""

    from alphalattice.control.workspace_runtime import database as owner

    database = _prepared(tmp_path)
    resolved = database.path.resolve()
    closing_entered, release_close = threading.Event(), threading.Event()
    seen: list[tuple[LiveConnections | None, bool]] = []
    real_fresh = owner._fresh_connection

    class _Blocking:
        """The raw connection whose close waits: the engine's close in flight."""

        def __init__(self, raw: duckdb.DuckDBPyConnection) -> None:
            self._raw = raw

        def close(self) -> None:
            with owner._STATE:
                seen.append(
                    (live_workspace_connections(database.path), bool(owner._CLOSING.get(resolved)))
                )
            closing_entered.set()
            release_close.wait(timeout=10)
            self._raw.close()

        def __getattr__(self, name: str) -> object:
            return getattr(self._raw, name)

    locked = duckdb.IOException(
        f'IO Error: Cannot open file "{database.path}": The process cannot access the file '
        "because it is being used by another process."
    )
    attempts: list[int] = []

    def fresh(path: Path, *, read_only: bool) -> duckdb.DuckDBPyConnection:
        if not read_only:
            return _Blocking(real_fresh(path, read_only=read_only))
        attempts.append(1)
        if owner._CLOSING.get(path):
            raise locked
        return real_fresh(path, read_only=read_only)

    monkeypatch.setattr(owner, "_fresh_connection", fresh)
    last = database.connect(read_only=False)
    closer = threading.Thread(target=last.close, daemon=True)
    closer.start()
    assert closing_entered.wait(timeout=10)
    assert seen == [(None, True)], (
        "untracked and marked closing in one step, before the engine's close"
    )
    opened: list[duckdb.DuckDBPyConnection] = []
    opener = threading.Thread(
        target=lambda: opened.append(
            open_workspace_database(database.path, read_only=True, wait_seconds=5.0)
        ),
        daemon=True,
    )
    opener.start()
    _until(lambda: len(attempts) >= 2)
    assert opened == [] and opener.is_alive(), "the opener waits on the close in flight"
    release_close.set()
    closer.join(timeout=10)
    opener.join(timeout=10)
    assert not opener.is_alive() and len(opened) == 1
    assert not owner._CLOSING, "the mark is lifted when the close returns"
    assert opened[0].execute("SELECT count(*) FROM unit").fetchone() == (1,)
    opened[0].close()
    assert live_workspace_connections(database.path) is None


def test_two_handles_closing_at_once_keep_the_file_marked_until_the_last_close_returns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two handles closing at once keep the file marked until the last close returns."""

    from alphalattice.control.workspace_runtime import database as owner

    database = _prepared(tmp_path)
    resolved = database.path.resolve()
    closing_entered, release_close = threading.Event(), threading.Event()
    real_fresh = owner._fresh_connection

    class _Blocking:
        """The first handle's raw connection: its engine close waits, in flight."""

        def __init__(self, raw: duckdb.DuckDBPyConnection) -> None:
            self._raw = raw

        def close(self) -> None:
            closing_entered.set()
            release_close.wait(timeout=10)
            self._raw.close()

        def __getattr__(self, name: str) -> object:
            return getattr(self._raw, name)

    locked = duckdb.IOException(
        f'IO Error: Cannot open file "{database.path}": The process cannot access the file '
        "because it is being used by another process."
    )
    attempts: list[int] = []

    def fresh(path: Path, *, read_only: bool) -> duckdb.DuckDBPyConnection:
        if not read_only:
            return _Blocking(real_fresh(path, read_only=read_only))
        attempts.append(1)
        if owner._CLOSING.get(path):
            raise locked
        return real_fresh(path, read_only=read_only)

    monkeypatch.setattr(owner, "_fresh_connection", fresh)
    first = database.connect(read_only=False)  # the fresh handle; its close will block
    second = database.connect(read_only=False)  # attached: a real, fast close
    assert live_workspace_connections(database.path) == LiveConnections(False, 2, 0)
    first_closer = threading.Thread(target=first.close, daemon=True)
    first_closer.start()
    assert closing_entered.wait(timeout=10)
    with owner._STATE:
        assert live_workspace_connections(database.path) == LiveConnections(False, 1, 0)
        assert owner._CLOSING.get(resolved) == 1, "the first close counts, not the last handle's"
    second.close()  # the last handle on the books: fast, its own count comes down
    with owner._STATE:
        assert live_workspace_connections(database.path) is None
        assert owner._CLOSING.get(resolved) == 1, "the first close is in flight and still counted"
    opened: list[duckdb.DuckDBPyConnection] = []
    failures: list[BaseException] = []

    def open_reader() -> None:
        try:
            opened.append(open_workspace_database(database.path, read_only=True, wait_seconds=5.0))
        except BaseException as error:  # the refusal, if any, is the finding
            failures.append(error)

    opener = threading.Thread(target=open_reader, daemon=True)
    opener.start()
    _until(lambda: len(attempts) >= 2)
    assert opened == [] and failures == [] and opener.is_alive(), (
        "the opener waits on the close still in flight instead of refusing an unknown holder"
    )
    release_close.set()
    first_closer.join(timeout=10)
    opener.join(timeout=10)
    assert failures == [] and len(opened) == 1 and not opener.is_alive()
    assert not owner._CLOSING, "no close in flight once both returned"
    assert opened[0].execute("SELECT count(*) FROM unit").fetchone() == (1,)
    opened[0].close()
    assert live_workspace_connections(database.path) is None


def test_another_process_in_the_other_mode_is_refused_by_name(tmp_path: Path) -> None:
    """Another process in the other mode is refused by name."""

    database = _prepared(tmp_path)
    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(Path("src").resolve())!r})
        from pathlib import Path
        from alphalattice.control.workspace_runtime.database import open_workspace_database
        try:
            read_only = sys.argv[1] == "ro"
            connection = open_workspace_database(Path({str(database.path)!r}), read_only=read_only)
            print("opened", connection.execute("SELECT count(*) FROM unit").fetchone()[0])
            connection.close()
        except RuntimeError as error:
            print("refused", error)
        """
    )

    def other_process(mode: str) -> str:
        result = subprocess.run(
            [sys.executable, "-c", script, mode], capture_output=True, text=True, timeout=120
        )
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()

    # The engine names the holding process when the OS lets it; under a restricted account (a
    # sandbox) it cannot, and the owner answers that the holder is unknown. Both refuse the file.
    held = {
        "refused workspace_database.file_held_by_another_process",
        "refused workspace_database.file_held_by_an_unknown_holder",
    }
    with database.retain(read_only=False):
        assert other_process("ro") in held
    reader = database.connect(read_only=True)
    try:
        assert other_process("rw") in held
        assert other_process("ro") == "opened 1"
    finally:
        reader.close()
    with database.transaction() as writer:
        writer.execute("INSERT INTO unit VALUES (2, 'after the refusals')")
    assert _rows(database) == [(1, "seed"), (2, "after the refusals")]
