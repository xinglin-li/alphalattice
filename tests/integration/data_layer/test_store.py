"""StorePort on a real data root and DuckLake catalog: the write order, its one repair, start-up
collection, the deletion queue, the live_ref guard and the writer's fences."""

from __future__ import annotations

import json
import os
import sqlite3
import statistics
import subprocess
import sys
import uuid
from contextlib import closing
from datetime import date
from pathlib import Path
from time import perf_counter

import pytest

from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer import create_data_root
from alphalattice.kernel.data_layer import log as host
from alphalattice.kernel.data_layer.chain import canonical
from alphalattice.kernel.data_layer.log import HostLog, envelope
from alphalattice.kernel.data_layer.project import definition_id
from alphalattice.kernel.data_layer.store import (
    RESULTS_KEYS,
    WORLD_KEYS,
    StorePort,
    delete,
    repair,
    stage,
)

TEST_SOURCE = "src/alphalattice/kernel/data_layer/store.py"
LISTING, MOVED = str(uuid.UUID(int=7)), str(uuid.UUID(int=8))
YF, CALENDAR = definition_id("provider", "yf"), definition_id("provider", "calendar")
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


def bars(count: int, close: float = 1.0) -> list[dict]:
    """`count` daily bars of one listing from 2024-01-01."""
    days = (f"2024-{day // 28 + 1:02d}-{day % 28 + 1:02d}" for day in range(count))
    return [{"listing_id": LISTING, "session_date": day, "close": close} for day in days]


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
    days = [
        {"market": "XNYS", "session_date": day, "close_utc": f"{day} {close}+00", "half_day": half}
        for day, close, half in sessions
    ]
    for table, rows in (
        ("listing", [{"listing_id": LISTING, "market": "XNYS"}]),
        ("session", days),
    ):
        with stage(root, table, CALENDAR, rows) as staged:
            port.commit(staged)
    return port


def owe(port: StorePort) -> dict:
    """A catalog commit that carries its owed event, with no event in the log: the crash."""
    attrs = {"provider_id": YF, "dataset": "raw_bar", "row_count": 1, "known_us": 1}
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
            stage(root, "raw_bar", YF, bars(1), floor_bytes=1 << 62),
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
    assert tuple(admission[0]) == (YF, 1, snapshot[0])


KILL = """
import json, os, sys
from pathlib import Path
from alphalattice.kernel.data_layer import store
from alphalattice.kernel.data_layer.log import HostLog
root, point, batch = Path(sys.argv[1]), sys.argv[2], json.loads(Path(sys.argv[3]).read_text())
lag = batch["lag"] and tuple(batch["lag"])
with HostLog.open(root) as log:
    port = store.StorePort.open(root, log)
    yf = store.definition_id("provider", "yf")
    with store.stage(root, batch["table"], yf, batch["rows"], lag=lag) as staged:
        setattr(store.StorePort if point == "commit" else store, point, lambda *a: os._exit(9))
        port.commit(staged)
"""
DAYS = [f"{2000 + d // 336}-{d % 336 // 28 + 1:02d}-{d % 28 + 1:02d}" for d in range(1500)]
BATCHES = {
    "observed": ("raw_bar", None, [{"listing_id": LISTING, "session_date": d} for d in DAYS]),
    "legacy": (
        "raw_bar",
        ["yf-close", 1],
        [{"listing_id": LISTING, "session_date": d} for d in DAYS],
    ),
    "results": (
        "execution_outcome",
        None,
        [outcome(d, None if i % 3 else 0.1) for i, d in enumerate(DAYS)],
    ),
}


@pytest.mark.proves("data_layer.write_order_recovers_from_a_kill_at_each_point")
@pytest.mark.parametrize("batch", list(BATCHES), ids=list(BATCHES))
@pytest.mark.parametrize(
    ("point", "rows", "repaired"),
    [
        pytest.param("commit", 0, False, id="after the files"),
        pytest.param("repair", 1500, True, id="after the catalog commit"),
        pytest.param("delete", 1500, False, id="after the event"),
    ],
)
def test_a_kill_at_each_point_of_the_write_order_recovers_at_start(
    root, batch, point, rows, repaired
):
    """A Host killed at each point of an observed, legacy or results write order leaves no half
    snapshot: the next start finds all its rows or none, its event once, and no staging left."""
    table, lag, given = BATCHES[batch]
    if lag:
        with HostLog.open(root) as log:
            seed(root, log, [("2005-01-03", "21:00:00", False)]).close()
    sent = root.parent / "batch.json"  # a file: 1,500 rows exceed a Windows command line
    sent.write_text(json.dumps({"table": table, "lag": lag, "rows": given}))
    child = subprocess.run(
        [sys.executable, "-c", KILL, str(root), point, str(sent)], **CHILD, timeout=120
    )
    assert child.returncode == 9
    assert len(list((root / "staging").iterdir())) == 2  # the folder and its dead producer's lock
    schema = "world" if table in WORLD_KEYS else "results"
    with HostLog.open(root) as log:
        port = StorePort.open(root, log)
        found = port.lake.execute(f"SELECT count(*) FROM lake.{schema}.{table}").fetchone()[0]
        events = log.read(
            "SELECT event_id FROM event e JOIN admission a ON a.admission_id = e.subject_id"
            " WHERE e.type = 'admission.committed' AND a.dataset = ?",
            (table,),
        )
        port.close()
    assert (found, len(events)) == (rows, int(rows > 0))
    assert port.started["repaired"] == [row[0] for row in events if repaired]
    assert [handle.split("/")[0] for handle in port.started["collected"]] == ["staging"]
    assert list((root / "staging").iterdir()) == []


@pytest.mark.proves("data_layer.start_collects_what_no_write_owns")
def test_start_collects_orphan_staging_and_unnamed_data_files(root, monkeypatch):
    """Start collects staging no producer holds and data files no snapshot names, keeps a held
    producer's and every named file, and a producer paused before its lock commits or is refused."""
    with HostLog.open(root) as log, stage(root, "raw_bar", YF, bars(300)) as staged:
        port = StorePort.open(root, log)
        port.commit(staged)
        port.lake.execute("CALL ducklake_flush_inlined_data('lake')").complete()  # to files
        port.close()
    named = sorted((root / "data").rglob("*.parquet"))
    (root / "staging" / "adm-crashed").mkdir()
    orphan = named[0].parent / f"ducklake-{uuid.UUID(int=1)}.parquet"
    orphan.write_bytes(b"PAR1")
    with HostLog.open(root) as log, stage(root, "raw_bar", YF, bars(3, 2.0)) as live:
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
            with stage(root, "raw_bar", YF, bars(1, 3.0)) as late:
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
    profile = {"type": "model", "meaning": "a fitted model", "storage": "blob"}
    with HostLog.open(root) as log:
        for artifact in ("held", "free"):
            log.record(
                "artifact.produced", ("artifact", artifact), profile | {"identity_kind": "content"}
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
    wall = [host.time.time_ns()]
    monkeypatch.setattr(host, "time", type("Wall", (), {"time_ns": staticmethod(lambda: wall[0])}))
    with HostLog.open(root) as log:
        key = {"family": "provider", "name": "yf", "version": 1}
        log.record(
            "definition.registered",
            ("definition", "provider/yf"),
            key,
            text=canonical(key | {"text": "The yf provider."}).decode(),
        )
        port = StorePort.open(root, log)
        with stage(root, "raw_bar", YF, bars(3)) as first:
            port.commit(first)
        wall[0] -= 3_600 * 10**9  # an hour back
        with stage(root, "raw_bar", YF, bars(1, close=2.0)) as correction:
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


@pytest.mark.proves("data_layer.legacy_rows_are_known_from_their_lag_rule")
def test_a_legacy_batch_is_known_from_its_lag_rule_and_the_new_columns_round_trip(root):
    """Every catalog table is admitted; a legacy row is known its rule's lag after its market's
    first close on or after its valid date; a batch moving a listing's market or naming an
    unregistered scheme is refused; typed columns read back whole."""
    bar = {"listing_id": LISTING, "session_date": "2024-01-05", "adjusted_close": 0.5}
    action = {"listing_id": LISTING, "effective_date": "2024-01-06", "kind": "split"}
    action |= {"action_id": "s1", "provisional": True, "provenance": "derived"}
    symbol = {"listing_id": LISTING, "symbol": "AAA", "valid_from": "2024-01-05"}
    rule, noon = ("yf-close", 1), "2024-01-07 12:00:00+00"  # Saturday's action counts from Monday
    sector = {"scheme_id": definition_id("sector_scheme", "gics"), "listing_id": LISTING}
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
        for table, rows, lag in batches:
            with stage(root, table, YF, rows, lag=lag) as staged:
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
    part = {"part": "2024", "min_known_asof_us": ten_pm, "max_known_asof_us": 1_704_740_400_000_000}
    assert [(version, json.loads(parts)) for version, parts in admitted] == [(1, [part])]
    assert refused == [
        "data_layer.lag_rule_unknown",
        "data_layer.legacy_anchor_missing",
        "data_layer.legacy_anchor_missing",
        "data_layer.result_key_present",
        "data_layer.lineage_unregistered",
    ]


@pytest.mark.proves("data_layer.a_correction_is_known_from_its_admission")
def test_a_batch_again_adds_nothing_and_a_correction_is_known_from_its_admission(root):
    """A batch admitted again adds nothing, and a legacy correction of a held key is known only
    from its admission."""
    bar = {"listing_id": LISTING, "session_date": "2024-01-05", "close": 1.0}
    later = bar | {"session_date": "2099-01-02"}  # known from a close still to come
    batches = [
        [bar, later],
        [bar, later],
        [bar | {"close": 2.0, "published_at": "2024-01-05 21:30:00+00"}],
        [bar | {"close": 2.0, "retracted": True}],
    ]
    with HostLog.open(root) as log:
        sessions = [("2024-01-05", "21:00:00", False), ("2099-01-02", "21:00:00", False)]
        port = seed(root, log, sessions)
        admitted = []
        for rows in batches:
            with stage(root, "raw_bar", YF, rows, lag=("yf-close", 1)) as staged:
                admitted.append((staged.admission_id, port.commit(staged)))
        with (
            stage(root, "raw_bar", YF, [later | {"close": 3.0}]) as observed,
            pytest.raises(DataError, match="known_asof order") as late,
        ):
            port.commit(observed)
        known = dict(log.read("SELECT admission_id, known_us FROM admission"))
        fixed, withdrawn = (known[admission] for admission, _ in admitted[2:])
        read = "SELECT list(close ORDER BY session_date) FROM raw_bar_asof(?)"
        ten_pm = 1_704_492_000_000_000  # 2024-01-05 22:00 UTC: the first bar's lag after its close
        seen = [
            port.lake.execute(read, (cutoff,)).fetchone()[0]
            for cutoff in (ten_pm, fixed - 1, fixed, withdrawn)
        ]
        versions = port.lake.execute("SELECT count(*) FROM lake.world.raw_bar").fetchone()[0]
        port.close()
    assert [done.snapshot is None for _, done in admitted] == [False, True, False, False]
    assert seen == [[1.0], [1.0], [2.0], None]
    assert versions == 4
    assert late.value.failure.code == "data_layer.admission_out_of_order"


@pytest.mark.proves("data_layer.a_source_supersedes_only_with_a_later_write")
def test_an_interval_keeps_its_history_and_a_source_supersedes_only_with_a_later_write(root):
    """A closed interval is written as its open and closing versions, a later change of its end is
    known from its admission, and a bar is superseded only by a row its source wrote after its
    value's last write, a later write of the same value moving that time without a version."""
    symbol = {"listing_id": LISTING, "symbol": "AAA", "valid_from": "2024-01-05"}
    bar = {"listing_id": LISTING, "session_date": "2024-01-05", "close": 1.0}
    bar |= {"source_observed_at": "2024-02-01 00:00:00+00"}
    bars = [
        bar,
        bar | {"close": 0.5, "source_observed_at": "2024-01-31 00:00:00+00"},  # older
        bar | {"close": 0.7},  # the same write time, another value
        bar | {"close": 2.0, "source_observed_at": "2024-02-02 00:00:00+00"},  # later
        bar | {"close": 2.0, "source_observed_at": "2024-03-01 00:00:00+00"},  # rewritten
        bar | {"close": 3.0, "source_observed_at": "2024-02-15 00:00:00+00"},  # before the rewrite
        bar | {"close": 2.0, "source_observed_at": "2024-03-01 00:00:00+00"},  # the same again
    ]
    rule = ("yf-close", 1)
    with HostLog.open(root) as log:
        port = seed(
            root, log, [("2024-01-05", "21:00:00", False), ("2024-01-08", "18:00:00", True)]
        )
        with stage(
            root, "provider_symbol", YF, [symbol | {"valid_to": "2024-01-05"}], lag=rule
        ) as s:
            first = port.commit(s)
        with stage(
            root, "provider_symbol", YF, [symbol | {"valid_to": "2024-01-08"}], lag=rule
        ) as s:
            moved = (s.admission_id, port.commit(s))
        done = []
        for row in bars:
            with stage(root, "raw_bar", YF, [row], lag=rule) as staged:
                done.append(port.commit(staged))
        known = dict(log.read("SELECT admission_id, known_us FROM admission"))[moved[0]]
        read = "SELECT list(valid_to ORDER BY valid_to) FROM lake.world.provider_symbol x"
        read += " WHERE known_asof <= make_timestamptz(?) AND NOT EXISTS (SELECT 1 FROM"
        read += " lake.world.superseded s WHERE s.value_digest = x.value_digest"
        read += " AND s.admission_id = x.admission_id AND s.superseded_asof <= make_timestamptz(?))"
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


STAGE = """
import sys, uuid
from pathlib import Path
from alphalattice.kernel.data_layer.store import definition_id, stage
rows = [{"listing_id": str(uuid.UUID(int=i % 500)), "session_date": f"{2000 + i // 500}-01-01",
         "close": float(i)} for i in range(100_000)]
print("hashing", flush=True)
with stage(Path(sys.argv[1]), "raw_bar", definition_id("provider", "yf"), rows) as staged:
    print(staged.row_count, flush=True)
"""


@pytest.mark.proves("data_layer.heavy_work_runs_beside_the_writer")
def test_an_append_waits_no_longer_while_a_large_admission_is_staged(root):
    """Staging digests rows in the process that has them, never under the writer: an append and
    its flush made while 100,000 rows are digested and written wait as an ordinary one does."""
    with HostLog.open(root) as log:

        def wait() -> float:
            began = perf_counter()
            log.record("outcome.observed", attrs={"n": 1})
            log.flush()
            return perf_counter() - began

        ordinary = [wait() for _ in range(100)]
        child = subprocess.Popen([sys.executable, "-c", STAGE, str(root)], **CHILD, stdout=-1)
        assert child.stdout.readline() == "hashing\n"
        during = []
        while child.poll() is None:
            during.append(wait())
        assert child.stdout.read() == "100000\n"
    assert len(during) > 20
    assert statistics.median(during) <= 2 * statistics.median(ordinary) + 0.002
    assert statistics.quantiles(during, n=10)[-1] <= 2 * max(ordinary) + 0.005
