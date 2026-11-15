"""StorePort on a real data root and DuckLake catalog: the write order, its one repair, start-up
collection, the deletion queue, the live_ref guard and the writer's fences."""

from __future__ import annotations

import os
import sqlite3
import statistics
import subprocess
import sys
import uuid
from contextlib import closing
from pathlib import Path
from time import perf_counter

import pytest

from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer import create_data_root
from alphalattice.kernel.data_layer import log as host
from alphalattice.kernel.data_layer.chain import canonical
from alphalattice.kernel.data_layer.log import HostLog, envelope
from alphalattice.kernel.data_layer.store import StorePort, delete, repair, stage

TEST_SOURCE = "src/alphalattice/kernel/data_layer/store.py"
LISTING = str(uuid.UUID(int=7))
CHILD = {
    "env": os.environ | {"PYTHONPATH": str(Path(host.__file__).parents[3])},
    "stdin": subprocess.DEVNULL,
    "text": True,
    "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0),  # no window on Windows
}


@pytest.fixture
def root(tmp_path):
    """A new data root with its catalog."""
    create_data_root(tmp_path / "data")
    return tmp_path / "data"


def bars(count: int, close: float = 1.0) -> list[dict]:
    """`count` daily bars of one listing from 2024-01-01."""
    days = (f"2024-{day // 28 + 1:02d}-{day % 28 + 1:02d}" for day in range(count))
    return [{"listing_id": LISTING, "session_date": day, "close": close} for day in days]


def owe(port: StorePort) -> dict:
    """A catalog commit that carries its owed event, with no event in the log: the crash."""
    attrs = {"provider": "yf", "dataset": "raw_bar", "row_count": 1, "known_us": 1}
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
            stage(root, "raw_bar", "yf", bars(1), floor_bytes=1 << 62),
        ):
            pass
        port = StorePort.open(root, log)
        again = repair(log, port.lake)
        kept = log.read(
            "SELECT type_version, workspace_id, actor_id, occurred_us, trace_id FROM event"
            " WHERE event_id = ?",
            (owed["event_id"],),
        )
        admission = log.read("SELECT provider, row_count, catalog_snapshot FROM admission")
        snapshot = port.lake.execute("SELECT max(snapshot_id) FROM lake.snapshots()").fetchone()
        port.close()
    assert refused.value.failure.code == "data_layer.below_free_floor"
    assert (port.started["repaired"], again) == ([owed["event_id"]], [])
    assert tuple(kept[0]) == (1, "w1", "store", 123, owed["trace_id"])
    assert tuple(admission[0]) == ("yf", 1, snapshot[0])


KILL = """
import os, sys, uuid
from pathlib import Path
from alphalattice.kernel.data_layer import store
from alphalattice.kernel.data_layer.log import HostLog
root, point = Path(sys.argv[1]), sys.argv[2]
days = (f"{2000 + d // 336}-{d % 336 // 28 + 1:02d}-{d % 28 + 1:02d}" for d in range(1500))
rows = [{"listing_id": str(uuid.UUID(int=7)), "session_date": day, "close": 1.0} for day in days]
with HostLog.open(root) as log:
    port = store.StorePort.open(root, log)
    with store.stage(root, "raw_bar", "yf", rows) as staged:
        setattr(store.StorePort if point == "commit" else store, point, lambda *a: os._exit(9))
        port.commit(staged)
"""


@pytest.mark.proves("data_layer.write_order_recovers_from_a_kill_at_each_point")
@pytest.mark.parametrize(
    ("point", "rows", "repaired"),
    [
        pytest.param("commit", 0, False, id="after the files"),
        pytest.param("repair", 1500, True, id="after the catalog commit"),
        pytest.param("delete", 1500, False, id="after the event"),
    ],
)
def test_a_kill_at_each_point_of_the_write_order_recovers_at_start(root, point, rows, repaired):
    """A Host killed at each point of the write order leaves no half snapshot: the next start finds
    all the rows or none, the owed event once if its snapshot committed, and no staging left."""
    child = subprocess.run([sys.executable, "-c", KILL, str(root), point], **CHILD, timeout=120)
    assert child.returncode == 9
    assert len(list((root / "staging").iterdir())) == 2  # the folder and its dead producer's lock
    with HostLog.open(root) as log:
        port = StorePort.open(root, log)
        found = port.lake.execute("SELECT count(*) FROM lake.world.raw_bar").fetchone()[0]
        events = log.read("SELECT event_id FROM event WHERE type = 'admission.committed'")
        port.close()
    assert (found, len(events)) == (rows, int(rows > 0))
    assert port.started["repaired"] == [row[0] for row in events if repaired]
    assert [handle.split("/")[0] for handle in port.started["collected"]] == ["staging"]
    assert list((root / "staging").iterdir()) == []


@pytest.mark.proves("data_layer.start_collects_what_no_write_owns")
def test_start_collects_orphan_staging_and_unnamed_data_files(root, monkeypatch):
    """Start collects staging no producer holds and data files no snapshot names, keeps a held
    producer's and every named file, and a producer paused before its lock commits or is refused."""
    with HostLog.open(root) as log, stage(root, "raw_bar", "yf", bars(300)) as staged:
        port = StorePort.open(root, log)
        port.commit(staged)
        port.lake.execute("CALL ducklake_flush_inlined_data('lake')").complete()  # to files
        port.close()
    named = sorted((root / "data").rglob("*.parquet"))
    (root / "staging" / "adm-crashed").mkdir()
    orphan = named[0].parent / f"ducklake-{uuid.UUID(int=1)}.parquet"
    orphan.write_bytes(b"PAR1")
    with HostLog.open(root) as log, stage(root, "raw_bar", "yf", bars(3, 2.0)) as live:
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
            with stage(root, "raw_bar", "yf", bars(1, 3.0)) as late:
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
    wall = [host.time.time_ns()]
    monkeypatch.setattr(host, "time", type("Wall", (), {"time_ns": staticmethod(lambda: wall[0])}))
    with HostLog.open(root) as log:
        port = StorePort.open(root, log)
        with stage(root, "raw_bar", "yf", bars(3)) as first:
            port.commit(first)
        wall[0] -= 3_600 * 10**9  # an hour back
        with stage(root, "raw_bar", "yf", bars(1, close=2.0)) as correction:
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


STAGE = """
import sys, uuid
from pathlib import Path
from alphalattice.kernel.data_layer.store import stage
rows = [{"listing_id": str(uuid.UUID(int=i % 500)), "session_date": f"{2000 + i // 500}-01-01",
         "close": float(i)} for i in range(100_000)]
print("hashing", flush=True)
with stage(Path(sys.argv[1]), "raw_bar", "yf", rows) as staged:
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
