"""StorePort on a real data root and DuckLake catalog: the write order, its one repair, start-up
collection, the deletion queue, the live_ref guard and the writer's fences."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import UTC, date, datetime, timedelta
from functools import partial
from pathlib import Path

import duckdb
import pyarrow as pa
import pytest

from alphalattice.capabilities.data.errors import DataError
from alphalattice.capabilities.domain_contracts import spawn
from alphalattice.foundation.data_layer import create_data_root, governance
from alphalattice.foundation.data_layer import log as host
from alphalattice.foundation.data_layer import store as storeport
from alphalattice.foundation.data_layer.chain import canonical
from alphalattice.foundation.data_layer.log import HostLog, envelope
from alphalattice.foundation.data_layer.store import (
    RESULTS_KEYS,
    WORLD_KEYS,
    Derivation,
    StorePort,
    as_columns,
    delete,
    owed_events,
    repair,
    stage,
)
from alphalattice.foundation.data_layer.telemetry import decode
from alphalattice.kernel.config import Change, override

TEST_SOURCE = "src/alphalattice/foundation/data_layer/store.py"
LISTING, MOVED = str(uuid.UUID(int=7)), str(uuid.UUID(int=8))
WRITTEN = {"source_observed_at": "2024-02-01 00:00:00+00"}  # a source's own write
CHILD = {
    "env": os.environ | {"PYTHONPATH": str(Path(host.__file__).parents[3])},
    "stdin": subprocess.DEVNULL,
    "text": True,
    "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0),  # no window on Windows
}


@pytest.fixture
def root(tmp_path):
    """A new data root with its catalog, whose log registers the providers yf and calendar."""
    create_data_root(tmp_path / "data")
    with HostLog.open(tmp_path / "data") as log:
        for name in ("yf", "calendar"):
            key = {"family": "provider", "name": name, "version": 1}
            text = canonical(key | {"text": f"The {name} provider."}).decode()
            log.record("definition.registered", ("definition", f"provider/{name}"), key, text=text)
    return tmp_path / "data"


def arrow(rows: list[dict]) -> pa.Table:
    """An admission's rows as the Arrow table StorePort stages."""
    return pa.table(as_columns(rows))


def bars(count: int, close: float = 1.0, day: int = 1) -> pa.Table:
    """`count` daily bars of one listing from 2024-01-01, their source writing them on February
    `day`."""
    days = (f"2024-{d // 28 + 1:02d}-{d % 28 + 1:02d}" for d in range(count))
    bar = {"listing_id": LISTING, "close": close, "source_observed_at": f"2024-02-{day:02d}"}
    return arrow([bar | {"session_date": d} for d in days])


def outcome(day: str, simple_return: float | None = 0.1) -> dict:
    """One execution outcome of artifact art-1, formed on `day`: its 19 fields and its artifact."""
    at = f"{day} 14:30:00+00"
    rows = {"artifact_id": "art-1", "listing_id": LISTING, "symbol": "AAA", "sequence": 1}
    rows |= dict.fromkeys(("formation_session", "entry_session", "holding_end_session"), day)
    rows |= dict.fromkeys(("formation_close_at", "entry_open_at", "holding_end_open_at"), at)
    rows |= dict.fromkeys(("entry_status", "holding_end_status"), "verified")
    rows |= {"entry_open_split_adjusted": 1.0, "holding_end_open_split_adjusted": None}
    rows |= {"period_dividend_split_adjusted": 0.0, "simple_return": simple_return}
    rows |= {"entry_source_row_hash": "e", "holding_end_source_row_hash": "h", "row_hash": "r"}
    return rows | {"actual_session_span": 2}


def seed(root: Path, log: HostLog, sessions: list[tuple[str, str, bool]]) -> StorePort:
    """A Host whose log holds lag rule yf-close v1 (one hour) and whose world holds the listing
    on XNYS and its sessions, as (date, close, half day); returns its open StorePort."""
    rule = {"family": "lag_rule", "name": "yf-close", "version": 1, "text": "an hour after"}
    attrs = {key: rule[key] for key in ("family", "name", "version")}
    text = canonical(rule | {"spec": {"lag_seconds": 3600}}).decode()
    log.record("definition.registered", ("definition", "yf-close"), attrs, text=text)
    port = StorePort.open(root, log)
    calendar = log.require_lineage("provider", "calendar")
    market_id = port.market_id("XNYS", calendar)
    days = [
        {
            "market_id": market_id,
            "session_date": day,
            "close_utc": f"{day} {close}+00",
            "half_day": half,
        }
        for day, close, half in sessions
    ]
    for table, rows in (
        ("listing", [{"listing_id": LISTING, "market_id": market_id}]),
        ("session", days),
    ):
        with stage(root, table, calendar, arrow(rows)) as staged:
            port.commit(staged)
    return port


def owe(port: StorePort) -> dict:
    """A catalog commit that carries its owed event, with no event in the log: the crash."""
    attrs = {
        "provider_id": port.log.require_lineage("provider", "yf"),
        "dataset": "raw_bar",
        "row_count": 1,
        "known_us": 1,
    }
    owed = envelope("admission.committed", ("admission", "adm-1"), attrs, workspace="w1")
    owed |= {"actor_id": "store", "occurred_us": 123}  # the writer's own
    info = canonical(owed).decode().replace("'", "''")
    for sql in (
        "BEGIN",
        "CREATE TABLE lake.main.bar AS SELECT 'adm-1' AS admission_id, 10.5 AS close",
        f"CALL lake.set_commit_message('host', 'admission', extra_info => '{info}')",
        "COMMIT",
    ):
        port.lake.execute(sql).complete()
    return owed


def _telemetry_during_task_admission(root, monkeypatch, rows):
    """Admit a Task while the Host's shared DuckLake instance is writing telemetry."""
    from alphalattice.control.product_host.composition import application_session
    from alphalattice.foundation.data_layer import history

    workspace = root.parent / "workspace"
    workspace.mkdir()
    now = datetime(2025, 1, 1, tzinfo=UTC)
    application_session.CpuBudgetStore(workspace / "runtime").write(
        1, chosen_by="HUMAN", chosen_at=now
    )
    with history.own_root(root):
        application_session.WorkspaceApplicationSession.open_history("w1", workspace, lambda: now)
        port = history.store()
        assert port is not None
        settings = "SELECT current_setting('threads'), current_setting('async_threads')"
        opened = port.lake.execute(settings).fetchone()
        monkeypatch.setattr(application_session, "apply_reader_threads", int)
        resume = threading.Event()

        def admit():
            with application_session.TASK_LEASES.lease(uuid.uuid4(), 1, 1):
                return application_session._apply_task_reader_policy()

        with ThreadPoolExecutor(max_workers=1) as pool:
            execute = duckdb.DuckDBPyConnection.execute

            def at_transaction(connection, sql, *args, **kwargs):
                if sql.startswith("INSERT INTO lake.telemetry.event"):
                    try:
                        assert pool.submit(admit).result(timeout=5) == 1
                    except TimeoutError as error:
                        raise AssertionError("Task admission stalled during telemetry") from error
                    finally:
                        resume.set()
                if sql.startswith("SET threads TO ?"):
                    # DuckDB's native resize can get stuck here. Model that wait so
                    # the red replay releases the transaction and its test thread.
                    assert resume.wait(10)
                    raise AssertionError("Task admission tried to resize the shared lake")
                return execute(connection, sql, *args, **kwargs)

            with monkeypatch.context() as transaction:
                transaction.setattr(duckdb.DuckDBPyConnection, "execute", at_transaction)
                assert port.telemetry_batch(rows) == 1
        (record,) = port.agent_telemetry("codex", "c", workspace_id="w1")
        assert record["output_tokens"] == 3
        assert port.lake.execute(settings).fetchone() == opened == (1, 1)


@pytest.mark.proves("data_layer.agent_otlp_is_one_catalog_batch")
@pytest.mark.parametrize(
    "case",
    (
        "conversation",
        "workspace",
        "stopping",
        "unfinished-result",
        "held-reader-on-reopen",
        "failed-commit",
        "concurrent-batches",
        "daily-admission",
        "task-admission",
    ),
)
def test_native_observations_commit_once_and_read_only_their_conversation(root, monkeypatch, case):
    """One sanitized export survives restart, without duplicate delivery or invented scope, and
    a Host stop that meets a batch in the catalog's session waits for it to commit."""

    def batch(session, workspace="w1"):
        attributes = (
            [{"key": "conversation.id", "value": {"stringValue": session}}] if session else []
        )
        attributes += [
            {"key": "model", "value": {"stringValue": "synthetic-model"}},
            {"key": "alphalattice.trial_id", "value": {"stringValue": "trial-1"}},
            {"key": "gen_ai.usage.input_tokens", "value": {"intValue": "15"}},
            {"key": "gen_ai.usage.cache_read.input_tokens", "value": {"intValue": "5"}},
            {"key": "gen_ai.usage.cache_write.input_tokens", "value": {"intValue": "0"}},
            {"key": "gen_ai.usage.output_tokens", "value": {"intValue": "3"}},
        ]
        return decode(
            "logs",
            json.dumps(
                {
                    "resourceLogs": [
                        {
                            "resource": {
                                "attributes": [
                                    {"key": "service.name", "value": {"stringValue": "codex"}}
                                ]
                            },
                            "scopeLogs": [
                                {"logRecords": [{"timeUnixNano": "123", "attributes": attributes}]}
                            ],
                        }
                    ]
                }
            ).encode(),
            "application/json",
            workspace_id=workspace,
        )

    if case == "task-admission":
        _telemetry_during_task_admission(root, monkeypatch, batch("c"))
        return

    if case in {"held-reader-on-reopen", "failed-commit"}:
        from alphalattice.foundation.data_layer import history, history_runtime, reading
        from alphalattice.foundation.data_layer import root as catalog_owner

        if case == "held-reader-on-reopen":
            catalog_owner.catalog(root)
            with closing(sqlite3.connect(root / "machine/catalog.sqlite")) as catalog:
                assert catalog.execute("PRAGMA journal_mode = DELETE").fetchone() == ("delete",)
        else:
            monkeypatch.setattr(
                catalog_owner, "enable_catalog_wal", lambda root: None, raising=False
            )
        driver = """
import sys
from pathlib import Path
import alphalattice.control.workspace_runtime.reader_threads
from alphalattice.foundation.data_layer import reading

requests, answers = sys.stdin.buffer, sys.stdout.buffer
matches = reading.catalog_matches

def hold_after_attach(connection, *, name):
    connection.execute("BEGIN").complete()
    matches(connection, name=name)
    count = f"SELECT count(*) FROM {name}.telemetry.event"
    assert connection.execute(count).fetchall() == [(0,)]
    reading._send(answers, "holding the catalog snapshot")
    assert requests.read(1) == b"r"
    assert connection.execute(count).fetchall() == [(0,)]
    connection.execute("ROLLBACK").complete()

reading.catalog_matches = hold_after_attach
reading.serve(Path(sys.argv[1]), requests, answers)
"""
        start = reading.child_process.start_piped

        def hold_reader(args):
            if args == [sys.executable, "-m", reading.__name__, str(root)]:
                args = [sys.executable, "-c", driver, str(root)]
            return start(args)

        monkeypatch.setattr(reading.child_process, "start_piped", hold_reader)
        workspace = root.parent / "workspace"
        workspace.mkdir()
        history.open_history("w1", "workspace", workspace, root=root)
        try:
            history_runtime.start_log_reader()
            port = history.store()
            assert port is not None
            reader = history_runtime._open["reading"]
            child = reader._child
            assert child is not None and child.stdin is not None and child.stdout is not None
            reading._send(child.stdin, ("experience", {}))
            held = reader._answers.submit(reading._receive, child.stdout)
            assert held.result(timeout=20) == "holding the catalog snapshot"
            if case == "failed-commit":
                with pytest.raises(
                    duckdb.TransactionException, match='"COMMIT": database is locked'
                ):
                    port.telemetry_batch(batch("c"))
            else:
                try:
                    assert port.telemetry_batch(batch("c")) == 1
                except duckdb.TransactionException as error:
                    pytest.fail(f"the ahead reader blocked telemetry: {error}")
            child.stdin.write(b"r")
            child.stdin.flush()
            answer = reader._answers.submit(reading._receive, child.stdout)
            assert answer.result(timeout=20) == (True, [])
            rows = port.agent_telemetry("codex", "c", workspace_id="w1")
            assert len(rows) == (0 if case == "failed-commit" else 1)
        finally:
            history.close_history()
        return

    if case == "stopping":  # the batch holds the session's writer at its clock, then commits
        entered, release, done = threading.Event(), threading.Event(), {}
        with HostLog.open(root) as log:
            port, clock = StorePort.open(root, log), log.clock

            def held() -> int:
                entered.set()
                release.wait(30)
                return clock()

            monkeypatch.setattr(log, "clock", held)
            call = threading.Thread(
                target=lambda: done.update(rows=port.telemetry_batch(batch("c")))
            )
            call.start()
            assert entered.wait(30)
            stop = threading.Thread(target=port.close)
            stop.start()
            stop.join(0.5)
            assert stop.is_alive()  # the close waits: DuckDB refuses to close a session in use
            release.set()
            call.join(30)
            stop.join(30)
        assert done == {"rows": 1} and not stop.is_alive()
        return
    with HostLog.open(root) as log:
        port = StorePort.open(root, log)
        try:
            if case == "concurrent-batches":
                starting = threading.Barrier(2)

                def send(session):
                    starting.wait(timeout=10)
                    return port.telemetry_batch(batch(session))

                with ThreadPoolExecutor(max_workers=2) as pool:
                    assert list(pool.map(send, ("c1", "c2"))) == [1, 1]
                for session in ("c1", "c2"):
                    assert len(port.agent_telemetry("codex", session, workspace_id="w1")) == 1
                return
            if case == "daily-admission":
                entered = threading.Event()

                def send():
                    entered.set()
                    return port.telemetry_batch(batch("c"))

                with (
                    ThreadPoolExecutor(max_workers=1) as pool,
                    stage(
                        root, "raw_bar", log.require_lineage("provider", "yf"), bars(1)
                    ) as staged,
                ):
                    with port.admitting():
                        port.commit(staged)
                        pending = pool.submit(send)
                        assert entered.wait(10)
                        with pytest.raises(TimeoutError):
                            pending.result(timeout=0.2)
                    assert pending.result(timeout=10) == 1
                assert port.lake.execute("SELECT count(*) FROM lake.world.raw_bar").fetchall() == [
                    (1,)
                ]
                assert len(port.agent_telemetry("codex", "c", workspace_id="w1")) == 1
                return
            if case == "unfinished-result":
                pending = port.lake.execute("SELECT i FROM range(2) AS test_rows(i) ORDER BY i")
                assert pending.fetchone() == (0,)
                try:
                    assert port.telemetry_batch(batch("conversation-1")) == 1
                    (row,) = port.agent_telemetry("codex", "conversation-1", workspace_id="w1")
                    assert row["output_tokens"] == 3
                    assert port.unplaced_telemetry(workspace_id="w1") == 0
                    assert (
                        port.trial_agent_usage("trial-1", workspace_id="w1")["output_tokens"] == 3
                    )
                    assert pending.fetchall() == [(1,)]
                except duckdb.ResourceInUseException as error:
                    pytest.fail(f"telemetry used a connection with an unfinished result: {error}")
                finally:
                    pending.complete()
                return
            if case == "workspace":
                first, second = batch("conversation-1"), batch("conversation-1", "w2")
                # The catalog's idle check can precede an incoming observation.
                port.maintain_idle()
                assert port.telemetry_batch(first) == 1
                assert port.unplaced_telemetry(workspace_id="w1") == 0
                assert port.telemetry_batch(second) == 1
                assert port.telemetry_batch(second) == 0
                assert port.telemetry_batch(batch(None, "w2")) == 1
                for workspace in ("w1", "w2"):
                    (row,) = port.agent_telemetry("codex", "conversation-1", workspace_id=workspace)
                    assert row["workspace_id"] == workspace
                assert port.agent_telemetry("codex", "conversation-1", workspace_id="w3") == []
                assert port.unplaced_telemetry(workspace_id="w1") == 0
                assert port.unplaced_telemetry(workspace_id="w2") == 1
                assert port.trial_agent_usage("trial-1", workspace_id="w1")["output_tokens"] == 3
                assert port.trial_agent_usage("trial-1", workspace_id="w2")["output_tokens"] == 6
                return
            assert port.telemetry_batch(batch("conversation-1")) == 1
            assert port.telemetry_batch(batch("conversation-1")) == 0
            assert port.telemetry_batch(batch("conversation-2") + batch(None)) == 2
            (row,) = port.agent_telemetry("codex", "conversation-1", workspace_id="w1")
            assert row["uncached_input_tokens"] == 10 and row["output_tokens"] == 3
            assert row["agent_id"] is None and row["parent_agent_id"] is None
            assert port.agent_telemetry("claude-code", "conversation-1", workspace_id="w1") == []
            assert port.unplaced_telemetry(workspace_id="w1") == 1
            totals = port.trial_agent_usage("trial-1", workspace_id="w1")
            assert totals["status"] == "MEASURED"
            assert totals["input_tokens"] == 30 and totals["output_tokens"] == 9
            assert port.trial_agent_usage("absent", workspace_id="w1") == {
                "status": "UNMEASURED",
                "incomplete": 0,
            }
            from alphalattice.foundation.data_layer import history
            from alphalattice.interface.local_application.native_bridge import (
                NativeResearchBinding,
                read_session_usage,
            )

            monkeypatch.setattr(
                history,
                "agent_telemetry",
                lambda host, session, *, workspace_id: port.agent_telemetry(
                    host, session, workspace_id=workspace_id
                ),
            )
            posted = []

            def publish(document):
                posted.append(document)
                return {"status": "APPENDED"}

            receipt = read_session_usage(
                root,
                NativeResearchBinding("conversation-1", root, ()),
                workspace_id="w1",
                publish=publish,
            )
            assert receipt["status"] == "DELIVERED"
            assert receipt["member_claims"] == "UNMEASURED" and receipt["lineage"] == []
            assert posted[0]["subject"]["input_tokens"] == "10"
            assert posted[0]["subject"]["input_channel"] == "CODEX_OTLP"
        finally:
            port.close()
        port = StorePort.open(root, log)
        try:
            assert len(port.agent_telemetry("codex", "conversation-1", workspace_id="w1")) == 1
            assert port.unplaced_telemetry(workspace_id="w1") == 1
        finally:
            port.close()


@pytest.mark.proves("data_layer.repair_appends_each_owed_event_once")
def test_the_next_start_repairs_an_owed_event_once_even_below_the_floor(root):
    """A commit's missing event is appended once at the next start, with its writer's envelope and
    its snapshot, while staging is refused below the free-space floor; repair again adds none."""
    with HostLog.open(root) as log:
        port = StorePort.open(root, log)
        owed = owe(port)
        port.close()
        with (
            pytest.raises(DataError, match="free-space floor") as refused,
            stage(
                root, "raw_bar", log.require_lineage("provider", "yf"), bars(1), floor_bytes=1 << 62
            ),
        ):
            pass
        port = StorePort.open(root, log)
        again = repair(log, port.lake)
        kept = log.read(
            "SELECT type_version, workspace_id, actor_id, occurred_us, trace_id FROM event"
            " WHERE event_id = ?",
            (owed["event_id"],),
        )
        admission = log.read("SELECT provider_id, row_count, catalog_snapshot FROM admission")
        snapshot = port.lake.execute("SELECT max(snapshot_id) FROM lake.snapshots()").fetchone()
        port.close()
    assert refused.value.failure.code == "data_layer.below_free_floor"
    assert (port.started["repaired"], again) == ([owed["event_id"]], [])
    assert tuple(kept[0]) == (1, "w1", "store", 123, owed["trace_id"])
    assert tuple(admission[0]) == (json.loads(owed["attrs"])["provider_id"], 1, snapshot[0])


KILL = """
import json, os, sys
from pathlib import Path
import pyarrow as pa
from alphalattice.control.workspace_runtime import reader_threads
from alphalattice.kernel import storage
from alphalattice.foundation.data_layer import store
from alphalattice.foundation.data_layer.log import HostLog
storage.install_driver(reader_threads.DRIVER)  # this process's entry installs its DuckDB driver
root, point, batch = Path(sys.argv[1]), sys.argv[2], json.loads(Path(sys.argv[3]).read_text())
lag = batch["lag"] and tuple(batch["lag"])
if batch.get("array"):
    import numpy as np
    from alphalattice.foundation.data_layer import arrays
    log = HostLog.open(root)
    if point == "commit":  # the body staged, its profile not committed
        log.flush = lambda: os._exit(9)
    else:  # the profile committed, the body staged
        arrays.os.replace = lambda *a: os._exit(9)
    arrays.keep(root, log, {"value": np.array(batch["rows"])}, workspace=None)
elif batch.get("owner"):
    from datetime import date, datetime
    from alphalattice.foundation.data_layer import history
    rows = batch["rows"]
    for row in rows:
        row.pop("artifact_id")
        for name in ("formation_session", "entry_session", "holding_end_session"):
            row[name] = date.fromisoformat(row[name])
        for name in ("formation_close_at", "entry_open_at", "holding_end_open_at"):
            row[name] = datetime.fromisoformat(row[name])
    values = pa.Table.from_pylist(rows)
    index = values.schema.get_field_index("holding_end_open_split_adjusted")
    values = values.set_column(
        index, "holding_end_open_split_adjusted", pa.nulls(len(rows), pa.float64())
    )
    history.open_history("crash-fixture", "crash", root.parent / "workspace", root)
    history.store()
    setattr(store.StorePort if point == "commit" else store, point, lambda *a: os._exit(9))
    history.write_execution_outcome("a" * 64, values)
elif batch.get("market"):
    from alphalattice.foundation.market_data_ops.sources.sanitization import sanitize_payload
    from alphalattice.foundation.market_data_ops.storage import world
    from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
    from alphalattice.foundation.data_layer import history
    from tests.workspace_maintenance.acquisition_manifest import NOW, acquisition_manifest
    workspace, manifest = root.parent / "workspace", acquisition_manifest()
    market = MarketDataRepository(workspace, universe_reader="workspace")
    market.bootstrap(manifest)  # journaled, for the Host's first publish
    history.open_history("w1", "workspace", workspace, root=root)
    world.publish_pending(market.path, "w1")  # the Host's open: the bars below are a later delta
    if point == "write":  # the write and its journal row committed, its publish not begun
        world.publish = lambda *a, **k: os._exit(9)
    elif point == "catalog":  # the publish's batches written, its catalog commit not made
        store.StorePort._close_block = lambda *a: os._exit(9)
    else:  # the catalog committed and its events appended, the journal not yet emptied
        summary = world._record_admission_summary
        world._record_admission_summary = lambda *a: (summary(*a), os._exit(9))
    days = [{"session_date": day, "open": 100.0, "high": 110.0, "low": 99.0, "close": 101.0}
            | {"volume": 1_000_000} for day in batch["rows"]]
    payload = sanitize_payload(manifest, "fixture", dict.fromkeys(("AAPL", "MSFT"), days),
                               ("AAPL", "MSFT"))
    market.apply_validated_batch(manifest, payload, ingestion_id="day-0", observed_at=NOW)
elif batch["table"] == "results":
    from types import SimpleNamespace

    with HostLog.open(root) as log:
        port = store.StorePort.open(root, log)
        tables = {table: pa.table(store.as_columns(rows)) for table, rows in batch["rows"].items()}
        with store.stage(root, tables) as staged:
            if point == "commit":
                execute = port.lake.execute
                inserts = []

                def before_second_insert(sql, *args):
                    if sql.startswith("INSERT INTO lake.results."):
                        if inserts:
                            os._exit(9)
                        inserts.append(sql)
                    return execute(sql, *args)

                port.lake = SimpleNamespace(execute=before_second_insert)
            else:
                store.repair = lambda *a: os._exit(9)
            port.commit(staged)
else:
    with HostLog.open(root) as log:
        port = store.StorePort.open(root, log)
        yf = log.require_lineage("provider", "yf")
        rows = pa.table(store.as_columns(batch["rows"]))
        with store.stage(root, batch["table"], yf, rows, lag=lag) as staged:
            setattr(store.StorePort if point == "commit" else store, point, lambda *a: os._exit(9))
            port.commit(staged)
"""
DAYS = [f"{2000 + d // 336}-{d % 336 // 28 + 1:02d}-{d % 28 + 1:02d}" for d in range(1500)]
BATCHES = {
    "observed": (
        "raw_bar",
        None,
        [{"listing_id": LISTING, "session_date": d} | WRITTEN for d in DAYS],
    ),
    "results": (
        "execution_outcome",
        None,
        [outcome(d, None if i % 3 else 0.1) for i, d in enumerate(DAYS)],
    ),
    "results set": (
        "results",
        None,
        {
            "execution_outcome": [outcome(day) for day in DAYS],
            "breakdown": [
                {
                    "artifact_id": "art-2" if day == DAYS[-1] else "art-1",
                    "metric_id": "metric-1",
                    "metric_version": 1,
                    "dims": json.dumps({"session": day}, separators=(",", ":")),
                    "value": 0.1,
                }
                for day in DAYS
            ],
        },
    ),
    "array": ("array", None, [1.0, -0.0]),
    "market": ("raw_bar", None, ["2026-07-27", "2026-07-28", "2026-07-29", "2026-07-30"]),
}


@pytest.mark.proves("data_layer.write_order_recovers_from_a_kill_at_each_point")
@pytest.mark.parametrize(
    ("batch", "point", "rows", "repaired"),
    [
        pytest.param("observed", "commit", 0, False, id="observed after the files"),
        pytest.param("observed", "repair", 1500, True, id="observed after the catalog commit"),
        pytest.param("observed", "delete", 1500, True, id="observed after the event"),
        pytest.param("results", "commit", 0, False, id="results after the files"),
        pytest.param("results", "repair", 1500, True, id="results after the catalog commit"),
        pytest.param("results", "delete", 1500, True, id="results after the event"),
        pytest.param("results set", "commit", 0, False, id="table set before catalog commit"),
        pytest.param("results set", "repair", 1500, True, id="table set after catalog commit"),
        pytest.param("array", "commit", 0, False, id="array after its staged body"),
        pytest.param("array", "repair", 1, False, id="array after its profile commit"),
        pytest.param("market", "write", 8, True, id="market write before its publish"),
        pytest.param("market", "catalog", 8, True, id="market publish before catalog commit"),
        pytest.param("market", "journal", 8, False, id="market publish before journal clears"),
    ],
)
def test_a_kill_at_each_point_of_the_write_order_recovers_at_start(
    root, batch, point, rows, repaired, monkeypatch
):
    """A Host killed at each write-order point of an observed, results or array write leaves no
    half write: the next start finds all of it or none, its event once, and no staging left."""
    # A legacy batch takes observed's kill path: its lag enters only inside the catalog commit,
    # and its rows are the lag-rule test's. After an array's rename nothing is left to repair.
    table, lag, given = BATCHES[batch]
    sent = root.parent / "batch.json"  # a file: 1,500 rows exceed a Windows command line
    sent.write_text(
        json.dumps(
            {
                "table": table,
                "lag": lag,
                "rows": given,
                "owner": batch == "results",
                "array": batch == "array",
                "market": batch == "market",
            }
        )
    )
    child = spawn.run(
        [sys.executable, "-c", KILL, str(root), point, str(sent)], **CHILD, timeout=120
    )
    assert child.returncode == 9
    if batch == "array":
        _an_array_after_a_kill(root, point, given)
        return
    if batch == "market":
        _a_market_publish_after_a_kill(root, rows, repaired)
        return
    assert len(list((root / "staging").iterdir())) == 2  # the folder and its dead producer's lock
    orphans = []
    if batch == "results set" and point == "commit":
        orphans = [path.relative_to(root).as_posix() for path in (root / "data").rglob("*.parquet")]
        assert orphans  # the first table wrote files before the second insert was stopped
    schema = "world" if table in WORLD_KEYS else "results"
    with HostLog.open(root) as log:
        port = StorePort.open(root, log)
        tables = tuple(given) if batch == "results set" else (table,)
        found = tuple(
            port.lake.execute(f"SELECT count(*) FROM lake.{schema}.{name}").fetchone()[0]
            for name in tables
        )
        events = log.read(
            "SELECT event_id FROM event e JOIN admission a ON a.admission_id = e.subject_id"
            " WHERE e.type = 'admission.committed' AND a.dataset = ?",
            (table,),
        )
        if batch == "results set":
            snapshots = port.lake.execute(
                "SELECT snapshot_id FROM lake.snapshots() WHERE commit_message = 'admission'"
            ).fetchall()
            records = log.read("SELECT attrs FROM event WHERE type = 'admission.committed'")
            assert len(records) == len(snapshots) == int(rows > 0)
            if rows:
                attrs = json.loads(records[0][0])
                assert attrs["catalog_snapshot"] == snapshots[0][0]
                assert attrs["tables"] == {name: rows for name in tables}
                assert attrs["row_count"] == rows * len(tables)
                assert attrs["artifact_ids"] == ["art-1", "art-2"]
        port.close()
    assert (found, len(events)) == ((rows,) * len(tables), int(rows > 0))
    assert port.started["repaired"] == [row[0] for row in events if repaired]
    staging, *collected = port.started["collected"]
    assert staging.startswith("staging/") and set(collected) == set(orphans)
    assert all(not (root / path).exists() for path in orphans)
    assert list((root / "staging").iterdir()) == []
    if batch == "results":
        from datetime import datetime

        from alphalattice.foundation.data_layer import history, history_execution_outcomes

        records = [{k: v for k, v in row.items() if k != "artifact_id"} for row in given]
        for row in records:
            for name in ("formation_session", "entry_session", "holding_end_session"):
                row[name] = date.fromisoformat(row[name])
            for name in ("formation_close_at", "entry_open_at", "holding_end_open_at"):
                row[name] = datetime.fromisoformat(row[name])
        expected = pa.Table.from_pylist(records)
        index = expected.schema.get_field_index("holding_end_open_split_adjusted")
        expected = expected.set_column(
            index, "holding_end_open_split_adjusted", pa.nulls(len(records), pa.float64())
        )
        history.open_history("crash-fixture", "crash", root.parent / "workspace", root)
        try:
            if point == "commit":
                # A concurrent log unit must be able to enter history while the first
                # outcome opens StorePort: the log guard precedes history's lock.
                from concurrent.futures import ThreadPoolExecutor
                from threading import Event

                unit_held, opening = Event(), Event()
                store = history_execution_outcomes.store

                def first_store():
                    opening.set()
                    return store()

                def log_unit():
                    with history._open["log"].unit():
                        unit_held.set()
                        assert opening.wait(5)
                        acquired = history._lock.acquire(timeout=2)
                        if acquired:
                            history._lock.release()
                        return acquired

                monkeypatch.setattr(history_execution_outcomes, "store", first_store)
                with ThreadPoolExecutor(max_workers=2) as pool:
                    unit = pool.submit(log_unit)
                    assert unit_held.wait(5)
                    writer = pool.submit(history.write_execution_outcome, "a" * 64, expected)
                    assert unit.result(timeout=10), "first outcome inverted the log/history locks"
                    artifact = writer.result(timeout=30)
            else:
                artifact = history.write_execution_outcome("a" * 64, expected)
            actual = history.read_execution_outcome("a" * 64).select(expected.column_names)
            assert actual.cast(expected.schema).equals(expected)
            assert (
                len(
                    history._open["log"].read(
                        "SELECT 1 FROM artifact WHERE artifact_id = ?", (artifact,)
                    )
                )
                == 1
            )
            assert (
                len(
                    history._open["log"].read(
                        "SELECT 1 FROM measure WHERE artifact_id = ?", (artifact,)
                    )
                )
                == 1
            )
        finally:
            history.close_history()


def _a_market_publish_after_a_kill(root, rows, published):
    """Start after a kill in a market write's publish, as the Host's open does: whatever the
    killed publish left, world holds the workspace's bars once, the batch's commit is recorded
    once, and the journal is empty. A publish killed before its catalog commit is made by the
    open (`published`); one killed after it is judged `same` there and commits nothing again."""
    from alphalattice.foundation.data_layer import history
    from alphalattice.foundation.market_data_ops.storage import world

    workspace = root.parent / "workspace"
    history.open_history("w1", "workspace", workspace, root=root)
    try:
        port = history.store()

        def committed() -> int:
            return port.log.read(
                "SELECT count(*) FROM event e JOIN admission a ON a.admission_id = e.subject_id"
                " WHERE e.type = 'admission.committed' AND a.dataset = 'raw_bar'"
            )[0][0]

        before = committed()
        world.publish_pending(workspace / "market-data.duckdb", "w1")
        events = committed()
        found = port.lake.execute("SELECT count(*) FROM lake.world.raw_bar").fetchone()[0]
    finally:
        history.release_history()
    with closing(duckdb.connect(str(workspace / "market-data.duckdb"), read_only=True)) as db:
        journal = db.execute("SELECT count(*) FROM market_delta").fetchone()[0]
    assert (found, before, events, journal) == (rows, int(not published), 1, 0)


def _an_array_after_a_kill(root, point, given):
    """Start after a kill in an array's write: a body whose profile never committed is collected,
    one whose profile committed is placed, and the lane reads back whole or is missing by name."""
    import numpy as np

    from alphalattice.foundation.data_layer import arrays

    lane = {"value": np.array(given)}
    artifact, kept = arrays.describe(lane).artifact_id, point != "commit"
    with HostLog.open(root) as log:
        port = StorePort.open(root, log)
        port.close()
        assert port.started["placed"] == ([f"blobs/{artifact}"] if point == "repair" else [])
        collected = [handle.split("/")[1].startswith(".") for handle in port.started["collected"]]
        assert (collected, port.started["waiting"]) == ([] if kept else [True], [])
        assert [path.name for path in (root / "blobs").iterdir()] == ([artifact] if kept else [])
        if kept:
            assert (
                arrays.read_packed(root, log.read, arrays.digest(lane)) == lane["value"].tobytes()
            )
        else:
            with pytest.raises(DataError) as refused:
                arrays.read_packed(root, log.read, arrays.digest(lane))
            assert refused.value.failure.code == "data_layer.array_missing"


@pytest.mark.proves("data_layer.start_collects_what_no_write_owns")
def test_start_collects_orphan_staging_and_unnamed_data_files(root, monkeypatch):
    """Start collects staging no producer holds and data files no snapshot names, keeps a held
    producer's and every named file, and a producer paused before its lock commits or is refused."""
    with (
        HostLog.open(root) as log,
        stage(root, "raw_bar", log.require_lineage("provider", "yf"), bars(300)) as staged,
    ):
        port = StorePort.open(root, log)
        port.commit(staged)
        port.lake.execute("CALL ducklake_flush_inlined_data('lake')").complete()  # to files
        port.close()
    named = sorted((root / "data").rglob("*.parquet"))
    (root / "staging" / "adm-crashed").mkdir()
    orphan = named[0].parent / f"ducklake-{uuid.UUID(int=1)}.parquet"
    orphan.write_bytes(b"PAR1")
    with (
        HostLog.open(root) as log,
        stage(root, "raw_bar", log.require_lineage("provider", "yf"), bars(3, 2.0, day=2)) as live,
    ):
        first = StorePort.open(root, log)  # a Host restarting while a producer stages
        first.commit(live)
        first.close()
        second = StorePort.open(root, log)
        rows = second.lake.execute("SELECT count(*) FROM lake.world.raw_bar").fetchone()[0]
        second.close()
    data = orphan.relative_to(root).as_posix()
    assert (first.started["collected"], second.started["collected"]) == (
        ["staging/adm-crashed", data],
        [],
    )
    assert (sorted((root / "data").rglob("*.parquet")), rows) == (named, 303)
    with HostLog.open(root) as log:
        starts, lock = [], host.lock_byte

        def start_then_lock(handle):  # a Host starts while a producer pauses before its lock
            if not starts:
                starts.append(None)  # first: the start's own probe locks at once
                starts[0] = StorePort.open(root, log)
                starts[0].close()
            lock(handle)

        monkeypatch.setattr(host, "lock_byte", start_then_lock)
        try:
            with stage(
                root, "raw_bar", log.require_lineage("provider", "yf"), bars(1, 3.0, day=3)
            ) as late:
                port = StorePort.open(root, log)
                port.commit(late)
                port.close()
            outcome = "committed"
        except DataError as refused:
            outcome = refused.failure.code
    # POSIX removes the lock under the start's hold, so the producer's lock is on a removed file;
    # Windows cannot remove the file the producer has open, so the producer keeps it
    taken = "committed" if sys.platform == "win32" else "data_layer.staging_taken"
    assert (starts[0].started["collected"], outcome) == ([], taken)
    assert list((root / "staging").iterdir()) == []


@pytest.mark.proves("data_layer.held_files_wait_in_the_deletion_queue")
@pytest.mark.skipif(sys.platform != "win32", reason="only Windows refuses to delete an open file")
def test_a_held_file_waits_in_the_queue_and_goes_at_a_later_start(root):
    """A staging file another handle holds open is not deleted: its folder waits in the deletion
    queue with the error and its count of tries, and goes at the first start after it is let go."""
    (root / "staging" / "adm-held").mkdir()
    held = (root / "staging" / "adm-held" / "raw_bar.parquet").open("wb")
    with HostLog.open(root) as log:
        for _ in range(2):
            StorePort.open(root, log).close()
        row = log.read("SELECT kind, attempts, last_error FROM deletion_queue")[0]
        held.close()
        port = StorePort.open(root, log)
        port.close()
        assert log.read("SELECT count(*) FROM deletion_queue")[0][0] == 0
    assert port.started["collected"] == ["staging/adm-held"]
    assert (row["kind"], row["attempts"]) == ("staging", 2)
    assert row["last_error"].startswith("PermissionError")


@pytest.mark.proves("data_layer.live_ref_bodies_are_never_collected")
def test_a_body_a_live_ref_holds_is_never_deleted(root):
    """A blob whose artifact a live_ref holds is refused by name and stays; an unheld one goes."""
    profile = {"type": "model", "type_version": 1, "meaning": "a fitted model", "storage": "blob"}
    with HostLog.open(root) as log:
        for artifact in ("held", "free"):
            log.record(
                "artifact.produced",
                ("artifact", artifact),
                profile | {"identity_kind": "content", "content_digest": f"blake3:{artifact}"},
            )
        seq = log.read("SELECT max(seq) FROM event")[0][0]
    with closing(sqlite3.connect(root / "machine" / "log.sqlite")) as con, con:
        con.execute("INSERT INTO live_ref VALUES ('held', 'book', 'b1', ?)", (seq,))  # a book's pin
    for artifact in ("held", "free"):
        (root / "blobs" / artifact).write_bytes(b"model")
    with HostLog.open(root) as log:
        with pytest.raises(DataError, match="held by a live reference") as refused:
            delete(root, log, "blobs/held")
        assert delete(root, log, "blobs/free")
    assert refused.value.failure.code == "data_layer.live_ref_held"
    assert sorted(path.name for path in (root / "blobs").iterdir()) == ["held"]


@pytest.mark.proves("data_layer.admission_is_read_as_of_before_and_after_a_correction")
def test_an_admission_is_read_as_of_before_and_after_its_correction(root, monkeypatch):
    """One admission's rows share the known_at its event names, which rises though the wall clock
    steps back; the as-of read sees them before their correction and the correction after it."""
    root = root.parent / "o'brien" / "data"
    create_data_root(root)
    wall = [host.wall_time_ns()]
    monkeypatch.setattr(host, "wall_time_ns", lambda: wall[0])
    with HostLog.open(root) as log:
        key = {"family": "provider", "name": "yf", "version": 1}
        log.record(
            "definition.registered",
            ("definition", "provider/yf"),
            key,
            text=canonical(key | {"text": "The yf provider."}).decode(),
        )
        port = StorePort.open(root, log)
        with stage(root, "raw_bar", log.require_lineage("provider", "yf"), bars(3)) as first:
            port.commit(first)
        wall[0] -= 3_600 * 10**9  # an hour back
        with stage(
            root, "raw_bar", log.require_lineage("provider", "yf"), bars(1, close=2.0, day=2)
        ) as correction:
            port.commit(correction)
        a, b = (
            dict(log.read("SELECT admission_id, known_us FROM admission"))[s.admission_id]
            for s in (first, correction)
        )
        stamps = port.lake.execute(
            "SELECT count(DISTINCT known_at), max(epoch_us(known_at)) FROM lake.world.raw_bar"
            " GROUP BY admission_id ORDER BY 2"
        ).fetchall()
        read = "SELECT count(*), max(close) FROM raw_bar_asof(?) WHERE listing_id = ?"
        seen = [port.lake.execute(read, (cutoff, LISTING)).fetchone() for cutoff in (a - 1, a, b)]
        superseded = port.lake.execute("SELECT by_admission_id FROM lake.world.superseded")
        replaced = superseded.fetchall()
        recorded = [row[0] for row in log.read("SELECT recorded_us FROM event ORDER BY seq")]
        port.close()
    assert (stamps, b > a) == ([(1, a), (1, b)], True)
    assert recorded == sorted(set(recorded))
    assert seen == [(0, None), (3, 1.0), (3, 2.0)]
    assert replaced == [(correction.admission_id,)]


def _one_snapshot(root, refused: bool) -> None:
    """Admissions in one block share one snapshot, each keeping its own admission and event, one
    whose rows all drop against the block's own writes included; a refusal inside commits those
    made before it."""
    with HostLog.open(root) as log:
        port = StorePort.open(root, log)
        yf = log.require_lineage("provider", "yf")
        before = port.lake.execute("SELECT max(snapshot_id) FROM lake.snapshots()").fetchone()[0]
        made = []

        def block() -> None:
            with port.admitting():
                for close, day in ((1.0, 1), (2.0, 2), (0.5, 1)):  # the last is older: dropped
                    with stage(root, "raw_bar", yf, bars(2, close=close, day=day)) as staged:
                        port.commit(staged)
                    made.append(staged.admission_id)
                if refused:  # a provider no lineage registers
                    with stage(root, "raw_bar", str(uuid.UUID(int=9)), bars(1)) as unknown:
                        port.commit(unknown)

        if refused:
            with pytest.raises(DataError, match="no provider lineage"):
                block()
        else:
            block()
        snapshots = port.lake.execute(
            "SELECT snapshot_id FROM lake.snapshots() WHERE snapshot_id > ?", [before]
        ).fetchall()
        info = port.lake.execute(
            "SELECT commit_extra_info FROM lake.snapshots() WHERE snapshot_id > ?", [before]
        ).fetchone()[0]
        admitted = [
            tuple(row) for row in log.read("SELECT admission_id, catalog_snapshot FROM admission")
        ]
        rows = port.lake.execute(
            "SELECT admission_id, count(*) FROM lake.world.raw_bar GROUP BY ALL ORDER BY ALL"
        ).fetchall()
        newest = port.lake.execute("SELECT max(close) FROM raw_bar_asof(?)", [2**62]).fetchone()
        replaced = port.lake.execute("SELECT DISTINCT by_admission_id FROM lake.world.superseded")
        superseded = replaced.fetchall()
        port.close()
    assert len(snapshots) == 1
    assert [event["subject_id"] for event in owed_events(info)] == made
    assert sorted(admitted) == sorted((made_id, snapshots[0][0]) for made_id in made)
    assert rows == sorted((made_id, 2) for made_id in made[:2])
    assert newest == (2.0,) and superseded == [(made[1],)]


def _published_once_they_commit(root) -> None:
    """A record naming a block's rows is published once a block that opened commits them: never
    for one rolled back, and a block whose transaction never opened leaves none to join."""
    with HostLog.open(root) as log:
        port = StorePort.open(root, log)
        yf = log.require_lineage("provider", "yf")
        published: list[int] = []

        def block(interrupted: bool) -> list[int]:
            with port.admitting():
                with stage(root, "raw_bar", yf, bars(1)) as staged:
                    port.commit(staged)
                count = "SELECT count(*) FROM admission"
                port.after_commit(lambda: published.append(log.read(count)[0][0]))
                inside = list(published)
                if interrupted:
                    raise KeyboardInterrupt
            return inside

        port.lake.execute("BEGIN")  # an open transaction refuses the next block's
        with pytest.raises(duckdb.TransactionException), port.admitting():
            pass
        port.lake.execute("ROLLBACK")
        with pytest.raises(KeyboardInterrupt):
            block(interrupted=True)
        rolled_back = list(published)
        inside = block(interrupted=False)
        port.close()
    assert (rolled_back, inside, published) == ([], [], [1])


@pytest.mark.proves("data_layer.one_block_of_admissions_is_one_snapshot")
@pytest.mark.parametrize(
    "scenario",
    [
        pytest.param(partial(_one_snapshot, refused=False), id="whole"),
        pytest.param(partial(_one_snapshot, refused=True), id="a-refusal-inside"),
        pytest.param(_published_once_they_commit, id="a-record-naming-its-rows"),
    ],
)
def test_a_block_of_admissions_is_one_snapshot_each_keeping_its_own_admission(root, scenario):
    """A block's admissions commit as one snapshot, each keeping its own admission and event, and
    what names their rows is published only once it commits."""
    scenario(root)


FEATURES = storeport.feature_table(str(uuid.UUID(int=8)))


def factors(digest: str, *names: str, days=("2024-01-02", "2024-01-03")) -> pa.Table:
    """A feature set's derived rows of two listings on `days`, one column per factor."""
    rows = [
        {"recipe_key": "r1", "input_digest": digest, "listing_id": listing, "session_date": day}
        | {name: 0.5 for name in names}
        for listing in (LISTING, MOVED)
        for day in days
    ]
    return arrow(rows)  # the catalog casts each listing's UUID and each session's DATE


def parts(digest: str, days=("2024-01-02", "2024-01-03")) -> pa.Table:
    """The partitions `factors` fills on `days`, each listing's month, under `digest`."""
    named = [f"{day[:7]}/{listing}" for listing in (LISTING, MOVED) for day in sorted(set(days))]
    named = list(dict.fromkeys(named))
    return pa.table({"part": named, "input_digest": [digest] * len(named)})


def roots(*tables: pa.Table) -> list[tuple[str, str, str]]:
    """Recipe r1's roots of the partitions in `tables`."""
    return [("r1", row["part"], row["input_digest"]) for t in tables for row in t.to_pylist()]


@pytest.mark.proves("data_layer.storeport_writes_each_derived_partition_once")
def test_a_derived_partition_is_written_once_and_valid_only_for_its_inputs(root):
    """StorePort writes each derived partition once, valid for its inputs alone, and keeps it
    while a workspace roots it."""
    with HostLog.open(root) as log:
        port = StorePort.open(root, log)
        derive = partial(port.derive, FEATURES, recipe_key="r1", compute_ms=5)
        written = [derive(factors("d1", "momentum"), parts=parts("d1"))]
        written.append(derive(factors("d1", "momentum"), parts=parts("d1")))  # held already
        day = ("2024-01-04",)  # other inputs, and a column the table widens by
        written.append(derive(factors("d3", "momentum", "value", days=day), parts=parts("d3", day)))
        late = ("2023-12-29",)
        written.append(derive(factors("d1", "momentum", days=late), parts=parts("d1", late)))
        recomputed = factors("d2", "momentum").append_column("row_hash", pa.array(["h"] * 4))
        written.append(derive(recomputed, parts=parts("d2"), rooted_by="w1"))  # beside d1's
        log.rooted("w2", FEATURES, roots(parts("d1"), parts("d3", day), parts("d1", late)))
        kept = roots(parts("d1", late))
        dropped = log.rooted("w2", FEATURES, kept, exactly=True)  # w2 reads 2024 no more
        collected = port.collect(FEATURES)
        rooted = log.read(
            "SELECT workspace_id, input_digest FROM cache_root WHERE part = ?",
            (f"2024-01/{LISTING}",),
        )
        held = port.lake.execute(
            f"SELECT count(*), count(value), count(DISTINCT year(session_date)),"
            f" count(*) FILTER (WHERE input_digest = 'd2'), any_value(typeof(row_hash))"
            f" FROM lake.derived.{FEATURES}"
        ).fetchall()[0]
        hits = [log.cache_hit(FEATURES, "r1", f"2024-01/{LISTING}", d) for d in ("d1", "d2")]
        hits.append(log.cache_hit(FEATURES, "r1", f"2023-12/{LISTING}", "d1"))
        hits.append(log.cache_hit("factor_cov", "r1", f"2024-01/{LISTING}", "d2"))  # another table
        refused = []
        for table, rows, named in (
            ("feature_values", factors("d1", "momentum"), parts("d1")),
            (FEATURES, factors("d1", "momentum"), parts("d2")),
            (FEATURES, factors("d1", "momentum", days=("2023-12-29", "2024-01-02")), parts("d1")),
        ):
            with pytest.raises(DataError) as error:
                port.derive(table, rows, recipe_key="r1", parts=named, compute_ms=1)
            refused.append(error.value.failure.code)

        def risk(digest: str, recipe: str = "r1") -> list[Derivation]:
            """A Risk stage's two tables on one day under `digest`, derived as one snapshot."""
            day = pa.table({"part": ["2024-01-05"], "input_digest": [digest]})
            row = {"recipe_key": "r1", "input_digest": digest, "as_of": "2024-01-05", "value": 0.1}
            cov, specific = {"factor_i": "m", "factor_j": "m"}, {"listing_id": LISTING}
            return [
                Derivation("factor_cov", arrow([row | cov]), "r1", day, 1),
                Derivation("specific_risk", arrow([row | specific]), recipe, day, 1),
            ]

        snapshots = "SELECT count(*) FROM lake.snapshots()"
        before = port.lake.execute(snapshots).fetchall()[0][0]
        batched = [port.derive_set(risk("k1"), rooted_by="w3"), port.derive_set(risk("k1"))]
        taken = port.lake.execute(snapshots).fetchall()[0][0] - before
        with pytest.raises(DataError) as error:  # k2's second names another recipe: none written
            port.derive_set(risk("k2", recipe="r2"))
        refused.append(error.value.failure.code)
        risk_held = port.lake.execute(
            "SELECT (SELECT count(*) FROM lake.derived.factor_cov),"
            " (SELECT count(*) FROM lake.derived.specific_risk)"
        ).fetchall()[0]
        risk_rooted = log.read("SELECT count(*) FROM cache_root WHERE workspace_id = 'w3'")
        owed = repair(log, port.lake)  # a derived commit owes no event
        port.close()
    assert batched == [[1, 1], [0, 0]] and taken == 1 and risk_held == (1, 1)
    assert risk_rooted[0][0] == 2  # both tables' partitions, rooted together
    assert written == [4, 0, 2, 2, 4] and (dropped, collected) == (4, 6) and owed == []
    assert [tuple(row) for row in rooted] == [("w1", "d2")]  # derive rooted it under the writer
    assert held == (6, 0, 2, 4, "VARCHAR")  # d2's 2024 and d1's rooted 2023 alone
    assert hits == [False, True, True, False]  # the rooted inputs, its table alone
    mixed = ["data_layer.derived_rows_mixed"] * 2
    assert refused == ["data_layer.derived_table_not_admitted", *mixed, mixed[0]]


def _check_maintenance_keeps_owed_snapshots(root, log, port):
    """A week on, maintenance expires no snapshot from an owed event's on, so its repair still
    finds it; once repaired, it expires the rest and deletes the collected partition's files."""
    later = datetime.now(UTC) + timedelta(days=8)
    owed = owe(port)["event_id"]
    derive = partial(port.derive, FEATURES, recipe_key="r1", compute_ms=5, rooted_by="w1")
    derive(factors("d1", "momentum"), parts=parts("d1"))
    derive(factors("d2", "momentum"), parts=parts("d2"))
    files = partial(lambda: sum(f.stat().st_size for f in (root / "data").rglob("*.parquet")))
    held = port.maintain(later)  # merges the two derived files into one
    before = files()
    log.rooted("w1", FEATURES, roots(parts("d2")), exactly=True)
    port.collect(FEATURES)  # d1's file no longer read
    repaired = repair(log, port.lake)
    freed = port.maintain(later)
    after = files()
    read = port.lake.execute(
        f"SELECT count(*), any_value(input_digest) FROM lake.derived.{FEATURES}"
    ).fetchone()
    assert held["expired"] > 0 and (held["deleted"], repaired) == (2, [owed])  # the merged two
    assert freed["expired"] > 0 and freed["deleted"] > 0 and after < before
    assert read == (4, "d2")


@pytest.mark.proves("data_layer.maintenance_bounds_the_catalog")
@pytest.mark.parametrize("listings", [None, 108, 1080], ids=["owed", "inlined", "parquet"])
def test_maintenance_bounds_catalog_and_preserves_its_readers(root, listings, monkeypatch):
    """A real catalog keeps recovery snapshots and equal as-of rows through maintenance."""
    with HostLog.open(root) as log, closing(StorePort.open(root, log)) as port:
        if listings is None:
            _check_maintenance_keeps_owed_snapshots(root, log, port)
        else:
            _check_idle_maintenance_preserves_world_rows(root, log, port, listings, monkeypatch)


def _check_idle_maintenance_preserves_world_rows(root, log, port, listings, monkeypatch):
    """The idle write and age triggers compact both physical forms without moving as-of rows."""
    now = datetime.now(UTC)
    image = "SELECT * FROM raw_bar_asof(epoch_us(?)) ORDER BY listing_id, session_date"
    at = datetime(2024, 4, 1, tzinfo=UTC)
    run = port.lake.execute
    run("SET threads TO 6").complete()
    maintain = port.maintain

    def maintain_at_open_width(when):
        assert run("SELECT current_setting('threads')").fetchone() == (6,)
        return maintain(when)

    monkeypatch.setattr(port, "maintain", maintain_at_open_width)
    provider = log.require_lineage("provider", "yf")
    for day in range(126):
        run(
            "INSERT INTO lake.world.raw_bar BY NAME "
            " SELECT ? AS provider_id, md5(i::VARCHAR)::UUID AS listing_id,"
            " (?::DATE + ?::INTEGER) AS session_date,"
            " md5(i::VARCHAR || ':' || session_date::VARCHAR)::UUID AS version_id,"
            " i::DOUBLE AS close, ? AS known_asof, false AS retracted,"
            " 'synthetic-maintenance' AS admission_id"
            " FROM range(?) t(i) ORDER BY listing_id",
            (provider, date(2024, 1, 1), day, at, listings),
        ).complete()
        port.maintain_idle(now)
    catalog = (root / "machine/catalog.sqlite").as_uri()
    with closing(sqlite3.connect(catalog + "?mode=ro", uri=True)) as metadata:
        active, filed = metadata.execute(
            "SELECT count(*), coalesce(sum(record_count), 0) FROM ducklake_data_file"
            " WHERE end_snapshot IS NULL"
        ).fetchone()
    assert active <= 2 + storeport.IDLE_MAINTENANCE_WRITES
    assert 0 <= 126 * listings - filed < storeport.IDLE_MAINTENANCE_WRITES * listings
    before = run(image, (at,)).to_arrow_table()
    before_cutoff = run(image, (at - timedelta(seconds=1),)).to_arrow_table()
    old = run("SELECT count(*) FROM lake.snapshots()").fetchone()[0]
    port.maintain_idle(now + timedelta(days=8))
    assert run("SELECT count(*) FROM lake.snapshots()").fetchone()[0] == 1
    assert old > 1 and before.num_rows == 126 * listings and before_cutoff.num_rows == 0
    assert run(image, (at,)).to_arrow_table().equals(before)
    assert run(image, (at - timedelta(seconds=1),)).to_arrow_table().equals(before_cutoff)
    assert run("SELECT current_setting('threads')").fetchone() == (6,)


@pytest.mark.proves("data_layer.legacy_rows_are_known_from_their_lag_rule")
def test_a_legacy_batch_is_known_from_its_lag_rule_and_the_new_columns_round_trip(root):
    """Every catalog table is admitted; a legacy row is known its rule's lag after its market's
    first close on or after its valid date; a batch moving a listing's market or naming an
    unregistered scheme is refused; typed columns read back whole."""
    assert WORLD_KEYS["session"] == ("market_id", "session_date")
    bar = {"listing_id": LISTING, "session_date": "2024-01-05", "adjusted_close": 0.5}
    bar |= WRITTEN
    action = {"listing_id": LISTING, "effective_date": "2024-01-06", "kind": "split"}
    action |= {"action_id": "s1", "provisional": True, "provenance": "derived"} | WRITTEN
    symbol = {"listing_id": LISTING, "symbol": "AAA", "valid_from": "2024-01-05"}
    rule, noon = ("yf-close", 1), "2024-01-07 12:00:00+00"  # Saturday's action counts from Monday
    sector = {"scheme_id": str(uuid.UUID(int=9)), "listing_id": LISTING}
    sector |= {"valid_from": "2024-01-05", "sector_code": "45"}
    batches = [
        ("raw_bar", [bar, bar | {"session_date": "2024-01-08"}], rule),  # a half day's close
        ("corporate_action", [action, action | {"action_id": "s2", "published_at": noon}], rule),
        ("provider_symbol", [symbol, symbol | {"symbol": "B", "valid_to": "2024-01-08"}], rule),
        ("execution_outcome", [outcome("2024-01-02"), outcome("2024-01-03", None)], None),
        ("raw_bar", [bar], ("yf-close", 2)),
        ("raw_bar", [bar | {"session_date": "2024-01-09"}], rule),
        ("listing", [{"listing_id": MOVED, "market": "XNYS"}], None),
        ("listing", [{"listing_id": MOVED, "market": "XNAS"}], None),
        ("raw_bar", [bar | {"listing_id": MOVED}], rule),
        ("execution_outcome", [outcome("2024-01-02")], None),
        ("sector", [sector], None),  # gics is no registered sector_scheme lineage
    ]
    refused = []
    with HostLog.open(root) as log:
        port = seed(
            root, log, [("2024-01-05", "21:00:00", False), ("2024-01-08", "18:00:00", True)]
        )
        calendar = log.require_lineage("provider", "calendar")
        yf = log.require_lineage("provider", "yf")
        for table, rows, lag in batches:
            if table == "listing":
                rows = [
                    {
                        "listing_id": row["listing_id"],
                        "market_id": port.market_id(row["market"], yf),
                    }
                    for row in rows
                ]
            with stage(
                root,
                table,
                None if table in RESULTS_KEYS else yf,
                arrow(rows),
                lag=lag,
            ) as staged:
                try:
                    port.commit(staged)
                except DataError as error:
                    refused.append(error.failure.code)
        read = port.lake.execute
        known = [
            read(
                "SELECT count(known_at), list(strftime(known_asof, '%d %H') ORDER BY known_asof)"
                f" FROM lake.world.{table}"
            ).fetchone()
            for table in ("raw_bar", "corporate_action", "provider_symbol")
        ]
        carried = read(
            "SELECT (SELECT list(DISTINCT adjusted_close) FROM lake.world.raw_bar),"
            " provisional, provenance FROM lake.world.corporate_action"
        ).fetchone()
        outcomes = read(
            "SELECT simple_return, holding_end_open_split_adjusted, epoch_us(formation_close_at)"
            " FROM lake.results.execution_outcome ORDER BY 3"
        ).fetchall()
        ten_pm = 1_704_492_000_000_000  # 2024-01-05 22:00 UTC, the first bar's known_asof
        seen = [
            read(f"SELECT count(*) FROM raw_bar_asof({c})").fetchone()[0]
            for c in (ten_pm - 1, ten_pm)
        ]
        tables = read(
            "SELECT schema_name, list(table_name ORDER BY table_name) FROM duckdb_tables()"
            " WHERE database_name = 'lake' GROUP BY ALL"
        ).fetchall()
        columns = read(
            "SELECT column_name, data_type, is_nullable FROM duckdb_columns() WHERE database_name"
            " = 'lake' AND table_name = 'execution_outcome' AND column_name <> 'artifact_id'"
            " ORDER BY column_index"
        ).fetchall()
        admitted = log.read(
            "SELECT json_extract(attrs, '$.lag_rule_version'), json_extract(attrs, '$.parts')"
            " FROM event WHERE type = 'admission.committed' AND json_extract(attrs, '$.dataset')"
            " = 'raw_bar'"
        )
        lineage = log.read(
            "SELECT provider_id, json_extract(e.attrs, '$.artifact_ids') FROM admission a"
            " JOIN event e ON e.seq = a.admission_seq WHERE dataset = 'execution_outcome'"
        )
        market_id = read("SELECT market_id FROM lake.world.market WHERE mic = 'XNYS'").fetchone()[0]
        assert uuid.UUID(str(market_id)).version == 7
        sessions = read(
            "SELECT market_id, session_date, version_id, close_utc FROM lake.world.session"
            " ORDER BY session_date"
        ).fetchall()
        for label, mic, changed in (("same-code", "XNYS", False), ("new-code", "XNEW", True)):
            renamed = {"market_id": str(market_id), "mic": mic}
            if changed:
                renamed["valid_from"] = "2024-02-01"
            with stage(root, "market", calendar, arrow([renamed])) as staged:
                assert port.commit(staged).rows == int(changed), label
            assert (
                read(
                    "SELECT market_id, session_date, version_id, close_utc FROM lake.world.session"
                    " ORDER BY session_date"
                ).fetchall()
                == sessions
            ), label
        renamed_at = read(
            "SELECT epoch_us(known_asof) FROM lake.world.market WHERE mic = 'XNEW'"
        ).fetchone()[0]
        assert [
            read("SELECT mic FROM market_asof(?) WHERE market_id = ?", [at, market_id]).fetchone()[
                0
            ]
            for at in (renamed_at - 1, renamed_at)
        ] == ["XNYS", "XNEW"]
        with pytest.raises(DataError, match="no current binding") as retired:
            port.market_id("XNYS", calendar)
        assert retired.value.failure.code == "data_layer.market_mic_ambiguous"
        with stage(
            root, "raw_bar", yf, arrow([bar | {"session_date": "2024-01-06"}]), lag=rule
        ) as staged:
            assert port.commit(staged).rows == 1
        assert read(
            "SELECT strftime(known_asof, '%d %H') FROM lake.world.raw_bar"
            " WHERE session_date = DATE '2024-01-06'"
        ).fetchone() == ("08 19",)
        orphan = {
            "market_id": str(uuid.UUID(int=99)),
            "session_date": "2024-01-05",
            "half_day": False,
        }
        with (
            stage(root, "session", calendar, arrow([orphan])) as staged,
            pytest.raises(DataError, match="is not registered"),
        ):
            port.commit(staged)
        reassigned = host.uuid7()
        with stage(
            root,
            "market",
            calendar,
            arrow([{"market_id": reassigned, "mic": "XNYS", "valid_from": "2024-03-01"}]),
        ) as staged:
            port.commit(staged)
        assert port.market_id("XNYS", calendar) == reassigned
        with stage(
            root,
            "market",
            calendar,
            arrow([{"market_id": host.uuid7(), "mic": "XNYS", "valid_from": "2024-03-01"}]),
        ) as staged:
            port.commit(staged)
        with pytest.raises(DataError, match="names several recorded markets"):
            port.market_id("XNYS", calendar)
        port.close()
    schemas = dict(tables)
    kept = ("superseded", "rewritten")  # the versions' own records, keyed by no fact
    assert sorted(WORLD_KEYS) == [table for table in schemas["world"] if table not in kept]
    assert set(RESULTS_KEYS) <= set(schemas["results"])
    at = "TIMESTAMP WITH TIME ZONE"  # 0.2.7's types in its field order; only its doubles are null
    types = ["VARCHAR"] * 2 + ["INTEGER"] + ["DATE", at] * 3 + ["SMALLINT"] + ["VARCHAR"] * 2
    types += ["DOUBLE"] * 4 + ["VARCHAR"] * 3
    assert [(kind, null) for _, kind, null in columns] == [(t, t == "DOUBLE") for t in types]
    assert {name for name, *_ in columns} == set(outcome("2024-01-02")) - {"artifact_id"}
    assert known == [  # the closed symbol B is its open version, then its closing one
        (0, ["05 22", "08 19"]),
        (0, ["07 12", "08 19"]),
        (0, ["05 22", "05 22", "08 19"]),
    ]
    assert carried == ([0.5], True, "derived")
    assert outcomes == [(0.1, None, 1_704_205_800_000_000), (None, None, 1_704_292_200_000_000)]
    assert seen == [0, 1]
    assert [tuple(row) for row in lineage] == [(None, '["art-1"]')]
    part = {"part": "2024", "min_known_asof_us": ten_pm, "max_known_asof_us": 1_704_740_400_000_000}
    assert [(version, json.loads(parts)) for version, parts in admitted] == [(1, [part])]
    assert refused == [
        "data_layer.lag_rule_unknown",
        "data_layer.legacy_anchor_missing",
        "data_layer.legacy_anchor_missing",
        "data_layer.result_key_present",
        "data_layer.lineage_unregistered",
    ]


A, B, C, D = ("a" * 64, "b" * 64, "c" * 64, "d" * 64)  # four bodies' digests


def document(doc_id: str, digest: str, available: str, **facts: str) -> dict:
    """One legacy evidence document, as the evidence admission stages it."""
    row = {"doc_id": doc_id, "source_ref": doc_id, "doc_digest": digest}
    row |= {"byte_count": 1, "media_type": "text/html", "title": "t", "document_type": "10-K"}
    row |= {"entity_id": "AAA", "availability_basis": "OFFICIAL_ACCEPTANCE"}
    return row | {"available_at": available, "right_of_use": "SEC_PUBLIC_OFFICIAL_ACCESS"} | facts


@pytest.mark.proves("data_layer.a_legacy_document_is_known_from_its_availability")
def test_a_legacy_document_is_known_from_its_availability_never_its_filing_date(root):
    """A filing is known at its acceptance, not its filing date's midnight; a recorded import at
    its capture; a held document's other 0.2.7 body at its own availability; and a body observed
    now over a legacy one no earlier than its admission."""
    filed, accepted = "2024-01-05 00:00:00+00", "2024-01-05 21:30:00+00"
    filing = document(str(uuid.UUID(int=1)), A, accepted, published_at=filed)
    filing |= {"published_precision": "DATE", "accepted_at": accepted}
    capture = "2024-03-01 10:00:00+00"
    recorded = document(str(uuid.UUID(int=2)), B, capture, published_at=filed)
    recorded |= {"availability_basis": "RECORDED_IMPORT_CAPTURE_ACCEPTANCE_UNKNOWN"}
    rule = {"family": "lag_rule", "name": "evidence-availability", "version": 1, "text": "at once"}
    with HostLog.open(root) as log:
        provider = log.require_lineage("provider", "yf")
        filing["provider_id"] = recorded["provider_id"] = provider
        attrs = {key: rule[key] for key in ("family", "name", "version")}
        text = canonical(rule | {"spec": {"lag_seconds": 0}}).decode()
        log.record("definition.registered", ("definition", "evidence"), attrs, text=text)
        port = StorePort.open(root, log)
        restated = "2024-02-01 00:00:00+00"  # the filing's other body, in 0.2.7 from then
        for rows in ([filing, recorded], [filing | {"doc_digest": C, "available_at": restated}]):
            with stage(
                root,
                "evidence_doc",
                provider,
                arrow(rows),
                lag=("evidence-availability", 1),
            ) as s:
                port.commit(s)
        before = log.clock()
        with stage(
            root,
            "evidence_doc",
            provider,
            arrow([recorded | {"doc_digest": D}]),
        ) as s:
            port.commit(s)  # observed now, over the legacy import
        admitted = log.clock()
        read = port.lake.execute
        known = read(
            "SELECT doc_digest, epoch_us(known_asof), lag_rule FROM lake.world.evidence_doc"
            " ORDER BY doc_digest"
        ).fetchall()
        seen = {
            at: read(
                f"SELECT list(doc_digest ORDER BY doc_digest) FROM evidence_doc_asof({at})"
            ).fetchone()[0]
            for at in (
                1_704_412_800_000_000,
                1_704_490_200_000_000,
                1_706_745_600_000_000,
                1_709_287_200_000_000,
            )
        }
        port.close()
    acceptance, restated, captured = (
        1_704_490_200_000_000,
        1_706_745_600_000_000,
        1_709_287_200_000_000,
    )
    assert known[:3] == [
        (A, acceptance, "evidence-availability"),
        (B, captured, "evidence-availability"),
        (C, restated, "evidence-availability"),  # legacy over legacy: its own availability
    ]
    assert known[3][0] == D and before < known[3][1] <= admitted and known[3][2] is None
    assert seen == {
        1_704_412_800_000_000: None,  # the filing date's midnight: nothing is public yet
        acceptance: [A],
        restated: [C],  # the filing's other body, from its own availability
        captured: [B, C],  # the observed body is known only now, after every cutoff here
    }


HELD = {"listing_id": LISTING, "session_date": "2024-01-05", "open": 0.0, "close": 5.0} | WRITTEN
LATER = {"source_observed_at": "2024-02-02 00:00:00+00"}
UNTIMED = {name: value for name, value in HELD.items() if name not in WRITTEN}
FUTURE = HELD | {"session_date": "2099-01-02"}  # known from a close still to come
SYMBOL = {"listing_id": LISTING, "symbol": "AAA", "valid_from": "2024-01-05"}
ACTION = {
    "listing_id": LISTING,
    "effective_date": "2024-01-05",
    "kind": "dividend",
    "action_id": "cash",
    "amount": 0.5,
    "provisional": False,
    "provenance": "provider-observation",
} | WRITTEN
RULE, MOVED_RULE = ("yf-close", 1), ("yf-close", 2)
SAME = (None, 0, 0, {}, 1, 0)


@pytest.mark.proves("data_layer.a_correction_is_known_from_its_admission")
@pytest.mark.parametrize(
    ("table", "held", "staged", "added", "expected"),
    [
        pytest.param("raw_bar", ([HELD], None), ([HELD], None), None, SAME, id="again"),
        pytest.param(
            "raw_bar",
            ([HELD], None),
            ([HELD | {"adjusted_close": None}], None),
            None,
            SAME,
            id="a column absent, then null",
        ),
        pytest.param(
            "raw_bar",
            ([HELD], None),
            ([HELD | {"retracted": False}], None),
            None,
            SAME,
            id="retracted absent, then false",
        ),
        pytest.param(
            "raw_bar",
            ([HELD], None),
            ([HELD | {"close": 5, "open": -0.0}], None),
            None,
            SAME,
            id="a float, then an integer and -0",
        ),
        pytest.param(
            "raw_bar", ([HELD], None), ([HELD], None), "dividend", SAME, id="a column added later"
        ),
        pytest.param(
            "raw_bar",
            ([HELD], None),
            ([HELD | {"close": 6.0} | LATER], None),
            None,
            (None, 1, 0, {}, 2, 1),
            id="a correction",
        ),
        pytest.param(
            "raw_bar",
            ([HELD], None),
            ([HELD | {"retracted": True} | LATER], None),
            None,
            (None, 1, 0, {}, 2, 1),
            id="a retraction",
        ),
        pytest.param(
            "raw_bar",
            ([HELD], None),
            ([HELD | LATER], None),
            None,
            (None, 0, 1, {}, 1, 0),
            id="a rewrite of the held value",
        ),
        pytest.param(
            "corporate_action",
            ([ACTION], None),
            ([ACTION | LATER], None),
            None,
            (None, 0, 0, {}, 1, 0),
            id="a rewrite touches no new part",
        ),
        pytest.param(
            "raw_bar",
            ([HELD], None),
            ([UNTIMED | {"close": 6.0}], None),
            None,
            ("admission_source_time_missing", 0, 0, {}, 1, 0),
            id="no source time",
        ),
        pytest.param(
            "raw_bar",
            ([HELD], None),
            ([HELD | {"close": 6.0, "source_observed_at": "2024-01-31 00:00:00+00"}], None),
            None,
            (None, 0, 0, {"older": 1}, 1, 0),
            id="an all-dropped batch",
        ),
        pytest.param(
            "raw_bar",
            ([HELD], RULE),
            ([HELD | {"close": 6.0} | LATER], RULE),
            None,
            (None, 1, 0, {}, 2, 1),
            id="a legacy correction",
        ),
        pytest.param(
            "raw_bar",
            ([HELD], RULE),
            ([HELD | {"close": 6.0} | LATER], MOVED_RULE),
            None,
            ("lag_rule_version_moved", 0, 0, {}, 1, 0),
            id="another lag version",
        ),
        pytest.param(
            "raw_bar",
            ([FUTURE], RULE),
            ([FUTURE | {"close": 6.0} | LATER], None),
            None,
            ("admission_out_of_order", 0, 0, {}, 1, 0),
            id="known before the held version",
        ),
        pytest.param(
            "provider_symbol",
            ([SYMBOL | {"valid_to": "2024-01-05"}], RULE),
            ([SYMBOL | {"valid_to": "2024-01-08"}], RULE),
            None,
            (None, 1, 0, {}, 3, 2),
            id="an interval pair",
        ),
    ],
)
def test_one_fact_is_one_version_however_it_is_staged(
    root, monkeypatch, table, held, staged, added, expected
):
    """A staged row is a new version only when its conformed content differs from its key's newest
    one: each held version is superseded once, a new one is known from its admission, and an
    all-dropped batch is recorded with its drops once, though its commit is retried."""
    with HostLog.open(root) as log:
        port = seed(
            root,
            log,
            [
                ("2024-01-05", "21:00:00", False),
                ("2024-01-08", "18:00:00", True),
                ("2099-01-02", "21:00:00", False),
            ],
        )
        rule = {"family": "lag_rule", "name": "yf-close", "version": 2}
        text = canonical(
            rule
            | {"text": "two hours after", "spec": {"lag_seconds": 7200}}
            | {"relation": "not_comparable"}
        )
        log.record("definition.registered", ("definition", "yf-close/2"), rule, text=text.decode())
        with stage(
            root, table, log.require_lineage("provider", "yf"), arrow(held[0]), lag=held[1]
        ) as first:
            port.commit(first)
        if added:  # a nullable column the table gains after the held version
            port.lake.execute(f"ALTER TABLE lake.world.{table} ADD COLUMN {added} DOUBLE")
        refused, done, admission = None, None, None
        try:
            with stage(
                root, table, log.require_lineage("provider", "yf"), arrow(staged[0]), lag=staged[1]
            ) as second:
                admission = second.admission_id
                if expected[3] or table == "corporate_action":  # retry after its owed event

                    def failing(*_):
                        monkeypatch.setattr(storeport, "repair", repair)
                        raise OSError("the repair failed")

                    monkeypatch.setattr(storeport, "repair", failing)
                    with pytest.raises(OSError, match="the repair failed"):
                        port.commit(second)
                done = port.commit(second)
        except DataError as error:
            refused = error.failure.code.removeprefix("data_layer.")
        read = port.lake.execute
        versions = read(f"SELECT count(*) FROM lake.world.{table}").fetchone()[0]
        replaced, most = read(
            "SELECT count(*), coalesce(max(n), 0) FROM"
            " (SELECT count(*) AS n FROM lake.world.superseded GROUP BY version_id)"
        ).fetchone()
        events = log.read(
            "SELECT json_extract(attrs, '$.dropped'), json_extract(attrs, '$.known_us') FROM event"
            " WHERE type = 'admission.committed' AND subject_id = ?",
            (admission,),
        )
        with closing(sqlite3.connect(root / "machine" / "log.sqlite")) as other:  # committed only
            kept = other.execute(
                "SELECT count(*) FROM event WHERE type = 'admission.committed' AND subject_id = ?",
                (admission,),
            ).fetchone()[0]
        if table == "corporate_action":
            parts = log.read(
                "SELECT json_extract(attrs, '$.parts'), json_extract(attrs, '$.rewritten')"
                " FROM event WHERE type = 'admission.committed' AND subject_id = ?",
                (admission,),
            )
            assert [(json.loads(row[0]), row[1]) for row in parts] == [([], 1)]
        known = {int(at) for _, at in events}
        unknown = read(
            f"SELECT count(*) FROM lake.world.{table} WHERE admission_id = ?"
            " AND epoch_us(known_asof) NOT IN (SELECT unnest(?::BIGINT[]))",
            (admission, sorted(known)),
        ).fetchone()[0]
        port.close()
    got = (done.rows, done.rewritten, done.dropped) if done else (0, 0, {})
    assert (refused, *got, versions, replaced) == expected
    if (
        done and not done.rows and not done.rewritten and done.dropped
    ):  # no catalog commit carries it
        assert kept == 1, "an all-dropped admission's judgment is committed before its staging goes"
    assert most <= 1
    assert [json.loads(dropped or "{}") for dropped, _ in events] == (
        [expected[3]] if any(expected[1:4]) or table == "corporate_action" else []
    )
    assert unknown == 0


@pytest.mark.proves("data_layer.a_source_supersedes_only_with_a_later_write")
def test_an_interval_keeps_its_history_and_a_source_supersedes_only_with_a_later_write(root):
    """A closed interval is written as its open and closing versions, a later change of its end is
    known from its admission, and a bar is superseded only by a row its source wrote after its
    value's last write, a later write of the same value moving that time without a version."""
    symbol = {"listing_id": LISTING, "symbol": "AAA", "valid_from": "2024-01-05"}
    bar = {"listing_id": LISTING, "session_date": "2024-01-05", "close": 1.0}
    bar |= {"source_observed_at": "2024-02-01 00:00:00+00", "source_payload_digest": "original"}
    bars = [
        bar,
        bar | {"close": 0.5, "source_observed_at": "2024-01-31 00:00:00+00"},  # older
        bar | {"close": 0.7},  # the same write time, another value
        bar | {"close": 2.0, "source_observed_at": "2024-02-02 00:00:00+00"},  # later
        bar
        | {"close": 2.0, "source_observed_at": "2024-03-01 00:00:00+00"}
        | {"source_payload_digest": "next"},  # metadata still leaves this a rewrite, not a version
        bar | {"close": 3.0, "source_observed_at": "2024-02-15 00:00:00+00"},  # before the rewrite
        bar
        | {"close": 2.0, "source_observed_at": "2024-03-01 00:00:00+00"}
        | {"source_payload_digest": "next"},  # the same again
    ]
    rule = ("yf-close", 1)
    with HostLog.open(root) as log:
        port = seed(
            root, log, [("2024-01-05", "21:00:00", False), ("2024-01-08", "18:00:00", True)]
        )
        with stage(
            root,
            "provider_symbol",
            log.require_lineage("provider", "yf"),
            arrow([symbol | {"valid_to": "2024-01-05"}]),
            lag=rule,
        ) as s:
            first = port.commit(s)
        with stage(
            root,
            "provider_symbol",
            log.require_lineage("provider", "yf"),
            arrow([symbol | {"valid_to": "2024-01-08"}]),
            lag=rule,
        ) as s:
            moved = (s.admission_id, port.commit(s))
        done = []
        for row in bars:
            with stage(
                root, "raw_bar", log.require_lineage("provider", "yf"), arrow([row]), lag=rule
            ) as staged:
                done.append(port.commit(staged))
        known = dict(log.read("SELECT admission_id, known_us FROM admission"))[moved[0]]
        read = "SELECT list(valid_to ORDER BY valid_to) FROM lake.world.provider_symbol x"
        read += " WHERE known_asof <= make_timestamptz(?) AND NOT EXISTS (SELECT 1 FROM"
        read += " lake.world.superseded s WHERE s.version_id = x.version_id"
        read += " AND s.superseded_asof <= make_timestamptz(?))"
        ten_pm, half_day = 1_704_492_000_000_000, 1_704_740_400_000_000  # 01-05 22:00, 01-08 19:00
        ends = [
            port.lake.execute(read, (cutoff, cutoff)).fetchone()[0]
            for cutoff in (ten_pm - 1, ten_pm, half_day, known - 1, known)
        ]
        closes = port.lake.execute("SELECT close FROM raw_bar_latest").fetchall()
        versions = port.lake.execute("SELECT list(close ORDER BY close) FROM lake.world.raw_bar")
        versions = versions.fetchone()[0]
        port.close()
    assert (first.rows, moved[1].rows) == (2, 1)
    assert ends == [
        None,
        [date(2024, 1, 5)],
        [date(2024, 1, 5)],
        [date(2024, 1, 5)],
        [date(2024, 1, 8)],
    ]
    assert [(d.snapshot is None, d.rows, d.dropped, d.rewritten) for d in done] == [
        (False, 1, {}, 0),
        (True, 0, {"older": 1}, 0),
        (True, 0, {"equal_time": 1}, 0),
        (False, 1, {}, 0),
        (False, 0, {}, 1),
        (True, 0, {"older": 1}, 0),
        (True, 0, {}, 0),
    ]
    assert (closes, versions) == ([(2.0,)], [1.0, 2.0])


STAGE = r"""
import sys, uuid
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from contextlib import closing
from alphalattice.foundation.data_layer.log import attach
from alphalattice.foundation.data_layer.project import require_lineage
from alphalattice.foundation.data_layer.store import as_columns, stage
rows = [{"listing_id": str(uuid.UUID(int=i % 500)), "session_date": f"{2000 + i // 500}-01-01",
         "close": float(i), "source_observed_at": "2024-02-01 00:00:00+00"}
        for i in range(100_000)]
rows = pa.table(as_columns(rows))
root = Path(sys.argv[1])
with closing(attach(root)) as con:
    provider = require_lineage(con, "provider", "yf")

write_table = pq.write_table

def held_write(*args, **kwargs):
    print("staging", flush=True)
    if sys.stdin.readline() != "append committed\n":
        raise RuntimeError("staging released before the append committed")
    return write_table(*args, **kwargs)

pq.write_table = held_write
with stage(root, "raw_bar", provider, rows) as staged:
    print(staged.row_count, flush=True)
"""


@pytest.mark.proves("data_layer.heavy_work_runs_beside_the_writer")
def test_an_append_waits_no_longer_while_a_large_admission_is_staged(root):
    """The Host commits an append while another process holds a 100,000-row Parquet stage."""
    with (
        HostLog.open(root) as log,
        spawn.InteractiveSession(timeout=120, stop_scope="tree") as session,
    ):
        child = session.start(
            [sys.executable, "-c", STAGE, str(root)],
            **(CHILD | {"stdin": subprocess.PIPE}),
            stdout=subprocess.PIPE,
        )
        assert child.stdout is not None
        assert child.stdout.readline() == "staging\n"
        sequence = log.record("outcome.observed", attrs={"n": 1})
        log.flush()
        with closing(host.attach(root)) as reader:
            event = reader.execute("SELECT seq FROM event WHERE seq = ?", (sequence,)).fetchone()
            assert event is not None and event["seq"] == sequence
        session.send("append committed\n")
        assert child.stdout.readline() == "100000\n"
        assert child.wait(timeout=120) == 0
        session.check()


BLOB = {"type": "model", "type_version": 1, "meaning": "a fitted model", "identity_kind": "content"}


def _produced(log: HostLog, artifact: str, storage: str) -> None:
    """Record `artifact` with a `storage` body: a blob's file, an inline body or no rows yet."""
    attrs = BLOB | {"storage": storage, "content_digest": f"blake3:{artifact}"}
    body = b"model" if storage == "inline" else None
    log.record("artifact.produced", ("artifact", artifact), attrs, body=body)


def _holds(log: HostLog, holder: str) -> None:
    """Give artifact `a` the holder a case names, by its root's own events."""
    log.record("workspace.created", ("workspace", "w1"), {"name": "w1"})
    if holder in ("workspace", "deleted workspace"):
        log.record("artifact.held", ("artifact", "a"), {}, workspace="w1")
    if holder in ("pin", "unpinned"):
        log.record("artifact.pinned", ("artifact", "a"), {"pin_id": "p1"})
    if holder == "unpinned":
        log.record("artifact.unpinned", ("artifact", "a"), {"pin_id": "p1"})
    if holder == "deleted workspace":
        log.record("workspace.status_changed", ("workspace", "w1"), {"status": "deleted"})


_REMOVAL = {
    "missing within grace": "missing",
    "missing after grace": "deleted",
    "unreadable after grace": "missing",
    "returned within grace": "active",
    "returned after deletion": "deleted",
    "collection twice": "deleted",
    "interrupted after deletion": "deleted",
}


def _removed_workspace(root, monkeypatch, case, kept):
    from alphalattice.control.product_host.storage.backup import REMOVED_GRACE
    from alphalattice.foundation.data_layer import history_runtime

    folder, other = root.parent / "removed", root.parent / "shared"
    unreadable = case == "unreadable after grace"
    status_sql = "SELECT status FROM workspace WHERE workspace_id = 'w1'"
    other.mkdir()
    folders = {"w1": folder, "w2": other}
    owners = {"a": "w1", "b": "w1"}
    if unreadable:
        folders["w3"] = root.parent / "also-removed"
        owners["c"] = "w3"
    with HostLog.open(root) as log, closing(StorePort.open(root, log)) as port:
        for workspace, path in folders.items():
            log.record(
                "workspace.created",
                ("workspace", workspace),
                {"name": workspace, "folder": str(path)},
            )
        for artifact, workspace in owners.items():
            _produced(log, artifact, "blob")
            (root / "blobs" / artifact).write_bytes(b"model")
            log.record("artifact.held", ("artifact", artifact), {}, workspace=workspace)
        log.record("artifact.held", ("artifact", "b"), {}, workspace="w2")
        monkeypatch.setattr(history_runtime, "opened_log", lambda: (log, "w2"))
        assert history_runtime.mark_missing() == (["w1", "w3"] if unreadable else ["w1"])
        if case == "returned within grace":
            folder.mkdir()
        wall = host.wall_time_ns()
        days = -1 if kept and not unreadable else 1
        elapsed = (REMOVED_GRACE + timedelta(days=days)) // timedelta(microseconds=1)
        monkeypatch.setattr(host, "wall_time_ns", lambda: wall + elapsed * 1000)
        if unreadable:
            stat = Path.stat

            def deny_traversal(path, **kwargs):
                if path == folder:
                    raise PermissionError("fixture workspace cannot be traversed")
                return stat(path, **kwargs)

            monkeypatch.setattr(Path, "stat", deny_traversal)
        if case == "interrupted after deletion":

            def interrupted(*_):
                raise RuntimeError("before collecting bodies")

            with monkeypatch.context() as scoped:
                scoped.setattr(governance, "unheld", interrupted)
                with pytest.raises(RuntimeError, match="before collecting bodies"):
                    port.collect_unheld()
            with closing(host.attach(root)) as reader:
                assert reader.execute(status_sql).fetchone()[0] == "deleted"
            assert (root / "blobs" / "a").is_file()

        try:
            collected = port.collect_unheld()
        except PermissionError as error:
            pytest.fail(f"workspace collection stopped at an unreadable folder: {error}")
        assert collected == (["c"] if unreadable else ([] if kept else ["a"]))
        assert log.read(status_sql)[0][0] == _REMOVAL[case]
        refs = log.read("SELECT artifact_id, root_id FROM live_ref ORDER BY artifact_id, root_id")
        assert [tuple(row) for row in refs] == (
            [("a", "w1"), ("b", "w1"), ("b", "w2")] if kept else [("b", "w2")]
        )
        assert (root / "blobs" / "a").is_file() is kept
        assert (root / "blobs" / "b").is_file()
        if unreadable:
            assert (
                log.read("SELECT status FROM workspace WHERE workspace_id = 'w3'")[0][0]
                == "deleted"
            )
            assert not (root / "blobs" / "c").exists()
            assert not history_runtime.admit_removal("w1", "workspace.backups_pruned", {})
        if not kept:
            eligible = history_runtime.removed_since(REMOVED_GRACE, "workspace.backups_pruned")
            assert eligible == [("w1", [])]
        if case == "returned after deletion":
            folder.mkdir()
            monkeypatch.setattr(history_runtime, "held", lambda *_: log)
            with pytest.raises(DataError) as refused:
                history_runtime.workspace_on_log(folder, folder.name, "w1")
            assert refused.value.failure.code == "data_layer.workspace_deleted"
        if case in ("collection twice", "returned after deletion"):
            tip = log.read("SELECT max(seq) FROM event")[0][0]
            assert port.collect_unheld() == []
            assert log.read("SELECT max(seq) FROM event")[0][0] == tip


@pytest.mark.proves("data_layer.a_body_goes_only_when_nothing_holds_it")
@pytest.mark.parametrize(
    ("holder", "unheld", "retention_days", "kept"),
    [
        pytest.param("workspace", timedelta(0), None, True, id="held by its workspace"),
        *(
            pytest.param(case, timedelta(0), None, status != "deleted", id=case)
            for case, status in _REMOVAL.items()
        ),
        pytest.param("pin", timedelta(0), None, True, id="held by a pin"),
        pytest.param("none", timedelta(days=7), None, True, id="held by the window"),
        pytest.param("none", timedelta(days=7), "30", True, id="held by the configured window"),
        pytest.param("none", timedelta(0), None, False, id="collected when none holds it"),
        pytest.param("unpinned", timedelta(0), None, False, id="collected once unpinned"),
        pytest.param(
            "deleted workspace", timedelta(0), None, False, id="collected once unheld by delete"
        ),
    ],
    ids=str,
)
def test_the_collector_takes_a_body_only_when_no_reference_pin_or_window_holds_it(
    root, monkeypatch, holder, unheld, retention_days, kept
):
    """The collector takes a body no live reference, pin or window holds: its event first, then
    its file; a held one, or one younger than the window, stays."""
    monkeypatch.setattr(governance, "UNHELD", unheld)
    if holder in _REMOVAL:
        _removed_workspace(root, monkeypatch, holder, kept)
        return
    with HostLog.open(root) as log:
        _produced(log, "a", "blob")
        _holds(log, holder)
        (root / "blobs" / "a").write_bytes(b"model")
        port = StorePort.open(root, log)
        if retention_days is not None:
            in_two_weeks = log.clock() + timedelta(days=14) // timedelta(microseconds=1)
            monkeypatch.setattr(log, "clock", lambda: in_two_weeks)
        with override(Change("test-retention", {"ALPHALATTICE_RETENTION_DAYS": retention_days})):
            collected = port.collect_unheld()
        port.close()
        state = log.read("SELECT body_state FROM artifact WHERE artifact_id = 'a'")[0][0]
    assert collected == ([] if kept else ["a"])
    assert (state, (root / "blobs" / "a").exists()) == (
        ("present", True) if kept else ("collected", False)
    )


@pytest.mark.proves("data_layer.every_body_kind_is_collected_and_a_crash_finishes_at_start")
def test_each_body_kind_goes_and_a_crash_after_the_event_is_finished_at_the_next_start(
    root, monkeypatch
):
    """A blob, an inline body and a table's rows are collected; the event commits before a body
    goes, so a crash between them leaves the body queued, removed at the next start, and a
    collected body is never held again."""
    monkeypatch.setattr(governance, "UNHELD", timedelta(0))
    with HostLog.open(root) as log:
        port = StorePort.open(root, log)
        with stage(root, "execution_outcome", None, arrow([outcome("2024-01-02")])) as staged:
            port.commit(staged)
        for artifact, storage in (("b", "blob"), ("i", "inline"), ("art-1", "table")):
            _produced(log, artifact, storage)
        (root / "blobs" / "b").write_bytes(b"model")
        monkeypatch.setattr(storeport, "remove_bodies", lambda *_: None)  # a crash: none removed
        collected = port.collect()
        port.close()
        queued = [row[0] for row in log.read("SELECT handle FROM deletion_queue ORDER BY 1")]
        monkeypatch.undo()
        port = StorePort.open(root, log)
        rows = port.lake.execute("SELECT count(*) FROM lake.results.execution_outcome").fetchone()
        port.close()
        with pytest.raises(DataError) as refused:
            log.record("artifact.held", ("artifact", "b"), {})
        left = log.read("SELECT count(*) FROM deletion_queue")[0][0]
        inline = log.read("SELECT count(*) FROM artifact_body")[0][0]
    assert (collected, queued) == (3, ["blobs/b", "results/art-1"])
    assert (rows[0], left, inline, (root / "blobs" / "b").exists()) == (0, 0, 0, False)
    assert sorted(port.started["collected"]) == ["blobs/b", "results/art-1"]
    assert refused.value.failure.code == "data_layer.artifact_held_collected"
