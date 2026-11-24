"""What StorePort refuses before it touches a file: a handle outside its folders, a row set it
cannot admit; and the collector's order against a backup and the catalog."""

from __future__ import annotations

import json
import threading
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from functools import partial
from types import SimpleNamespace

import pyarrow as pa
import pytest

from alphalattice.capabilities.data.errors import DataError
from alphalattice.foundation.data_layer import governance
from alphalattice.foundation.data_layer.store import (
    OWED_EVENTS_VERSION,
    admissible,
    as_columns,
    collect_unheld,
    delete,
    expirable,
    owed_events,
    remove_bodies,
)

TEST_SOURCE = "src/alphalattice/foundation/data_layer/store.py"


@pytest.mark.proves("data_layer.store_deletes_only_its_own_files")
@pytest.mark.parametrize(
    "handle",
    [
        pytest.param("machine/log.sqlite", id="the log itself"),
        pytest.param("staging", id="a top folder, not a file in it"),
        pytest.param("staging/../machine/log.sqlite", id="a climb out of staging"),
        pytest.param("notes.txt", id="a file at the root"),
    ],
)
def test_a_handle_outside_staging_data_and_blobs_is_refused(tmp_path, handle):
    """Only a file or folder inside staging/, data/ or blobs/ can be deleted; anything else is
    refused by name before the disk is touched."""
    (tmp_path / "machine").mkdir()
    (tmp_path / "machine" / "log.sqlite").write_bytes(b"x")
    with pytest.raises(DataError, match="names no file under") as refused:
        delete(tmp_path, None, handle)  # refused before the log is read
    assert refused.value.failure.code == "data_layer.handle_outside_root"
    assert (tmp_path / "machine" / "log.sqlite").read_bytes() == b"x"


ROW = {"listing_id": str(uuid.UUID(int=7)), "session_date": "2024-01-02"}
WRITTEN = {"source_observed_at": "2024-01-03 00:00:00+00"}
NO_TIME = {"source_observed_at": None}


def arrow(rows: list[dict]) -> pa.Table:
    """An admission's rows as the Arrow table StorePort stages."""
    return pa.table(as_columns(rows))


@pytest.mark.proves("data_layer.stage_refuses_what_it_cannot_admit")
@pytest.mark.parametrize(
    ("table", "provider", "rows", "legacy", "code"),
    [
        pytest.param(
            "raw_bar",
            "yf",
            arrow([ROW | WRITTEN, ROW | WRITTEN | {"close": 2.0}]),
            False,
            "admission_key_repeated",
            id="key twice",
        ),
        pytest.param(
            "alpha_score", "yf", arrow([ROW]), False, "world_table_not_admitted", id="not admitted"
        ),
        pytest.param(
            "session", "yf", arrow([ROW]), True, "legacy_table_unanchored", id="legacy, no date"
        ),
        pytest.param(
            "raw_bar",
            None,
            arrow([ROW | WRITTEN]),
            False,
            "admission_lineage_invalid",
            id="world, no provider",
        ),
        pytest.param(
            "execution_outcome",
            "yf",
            arrow([ROW | {"artifact_id": "a", "formation_session": "2024-01-02"}]),
            False,
            "admission_lineage_invalid",
            id="results, a provider",
        ),
    ],
)
def test_staging_refuses_a_repeated_key_and_an_unadmitted_table(
    table, provider, rows, legacy, code
):
    """Two versions of one key in one admission, a table StorePort does not admit or a legacy
    batch with no valid date are refused by name."""
    with pytest.raises(DataError) as refused:
        admissible(table, provider, rows, legacy=legacy)
    assert refused.value.failure.code == f"data_layer.{code}"


@pytest.mark.proves("data_layer.a_document_is_keyed_by_its_doc_id")
def test_a_legacy_document_batch_is_keyed_by_its_doc_id_with_no_market():
    """A legacy evidence document batch, which names no market, is staged; two bodies of one
    doc_id in one admission are refused by name, since a doc_id's versions are admitted apart."""
    doc = {"doc_id": str(uuid.UUID(int=1)), "doc_digest": "a" * 64}
    doc |= {"available_at": "2024-01-05 21:30:00+00"}
    other = doc | {"doc_id": str(uuid.UUID(int=2)), "doc_digest": "b" * 64}
    staged = admissible("evidence_doc", "sec", arrow([doc, other]), legacy=True)
    with pytest.raises(DataError) as refused:
        admissible("evidence_doc", "sec", arrow([doc, doc | {"doc_digest": "c" * 64}]), legacy=True)
    assert staged.column("doc_id").to_pylist() == [doc["doc_id"], other["doc_id"]]
    assert refused.value.failure.code == "data_layer.admission_key_repeated"


@pytest.mark.proves("data_layer.a_timed_row_needs_its_source_time")
@pytest.mark.parametrize(
    "rows",
    [
        pytest.param([ROW], id="absent"),
        pytest.param([ROW | WRITTEN, ROW | {"session_date": "2024-01-03"} | NO_TIME], id="null"),
    ],
)
def test_a_timed_row_without_its_source_time_is_refused(rows):
    """A raw_bar or corporate_action row without its source's own write time is refused by name,
    since nothing could order it against a held value."""
    for table in ("raw_bar", "corporate_action"):
        with pytest.raises(DataError) as refused:
            admissible(table, "yf", arrow(rows))
        assert refused.value.failure.code == "data_layer.admission_source_time_missing"


@pytest.mark.proves("data_layer.a_world_row_carries_its_providers_lineage")
def test_a_world_row_is_staged_with_its_providers_lineage_and_a_results_row_as_it_came():
    """A world row whose key names a provider is staged with the lineage it was admitted under;
    other world rows and a results row are staged as they came."""
    world = [
        sorted(admissible(table, "yf", arrow([ROW | WRITTEN])).column_names)
        for table in ("raw_bar", "listing")
    ]
    assert world == [
        ["listing_id", "provider_id", "session_date", "source_observed_at"],
        ["listing_id", "session_date", "source_observed_at"],
    ]
    outcome = ROW | {"artifact_id": "a", "formation_session": "2024-01-02"}
    assert admissible("execution_outcome", None, arrow([outcome])).to_pylist() == [outcome]


@pytest.mark.proves("data_layer.staging_keeps_every_rows_columns")
def test_staging_keeps_a_column_only_a_later_row_carries():
    """A field only a later row of an admission carries is kept as a column, empty in the rows
    without it, as an interval's closing version carries its end."""
    closing = ROW | {"valid_to": "2024-01-09"}
    assert as_columns([ROW, closing]) == {
        "listing_id": [ROW["listing_id"]] * 2,
        "session_date": ["2024-01-02"] * 2,
        "valid_to": [None, "2024-01-09"],
    }


NOW = datetime(2026, 11, 17, tzinfo=UTC)


@pytest.mark.proves("data_layer.expiry_keeps_what_repair_reads")
@pytest.mark.parametrize(
    ("admitted", "held", "expired"),
    [
        pytest.param(None, set(), [1, 2, 3], id="the old ones but the newest"),
        pytest.param("e2", {"e2"}, [1, 2, 3], id="an admission the log holds"),
        pytest.param("e2", set(), [1], id="an owed admission stops expiry"),
    ],
)
def test_expiry_takes_old_snapshots_short_of_the_newest_and_an_owed_admission(
    admitted, held, expired
):
    """Snapshots older than a week expire, never the newest or a young one, and none from an
    admission whose event the log lacks on."""
    old, young = NOW - timedelta(days=8), NOW - timedelta(days=1)
    snapshots = [(1, old, None), (2, old, admitted), (3, old, None), (4, young, None)]
    snapshots.append((5, old, None))  # the newest by id, whatever its time
    assert expirable(snapshots, held, NOW) == expired


class _Log:
    """The Host's log as the collector uses it: `found` its unheld bodies and queued handles, the
    events and deletion queue it is given kept in order."""

    def __init__(self, found=(), queued=()):
        self.found, self.queued, self.done = list(found), list(queued), []

    def clock(self):
        return 10**15

    @contextmanager
    def unit(self):
        yield None

    def read(self, sql, args=()):
        if "FROM workspace w" in sql:
            return []
        return self.queued if "deletion_queue" in sql else self.found

    def record(self, type_, subject, attrs):
        self.done.append((type_, subject[1]))

    def queue_deletion(self, handle, kind, error):
        self.done.append(("queued", handle, kind))

    def dequeue_deletion(self, handle):
        self.done.append(("dequeued", handle))

    def flush(self):
        self.done.append(("flushed",))


class _Lake:
    """A catalog session that answers its results tables and keeps each statement it runs."""

    def __init__(self):
        self.ran = []

    def execute(self, sql, args=None):
        self.ran.append(" ".join(sql.split()[:3]))
        return SimpleNamespace(
            fetchall=lambda: [("alpha_score",), ("execution_outcome",)], complete=lambda: None
        )

    def register(self, name, table):
        self.ran.append(f"register {table.num_rows}")

    def unregister(self, name):
        self.ran.append("unregister")


@pytest.mark.proves("data_layer.a_backup_copy_excludes_collection")
def test_a_collection_decides_nothing_while_a_backup_copies(tmp_path):
    """A collection waits for the lock a backup holds while it copies, and decides its bodies
    only once the copy lets it go."""
    log = _Log(found=[("a", "inline")])
    collect = partial(collect_unheld, tmp_path, log, _Lake(), threading.Lock())
    with governance.COPYING:  # as backup_data_root holds it while it copies
        collector = threading.Thread(target=collect)
        collector.start()
        collector.join(timeout=0.3)
        during = list(log.done)
    collector.join()
    assert during == []
    assert log.done == [("artifact.collected", "a"), ("flushed",)]


@pytest.mark.proves("data_layer.collected_rows_leave_in_one_catalog_commit")
def test_collected_rows_leave_every_results_table_in_one_commit_then_the_queue(tmp_path):
    """The rows of each queued collected artifact leave every results table in one catalog
    commit, and only then leave the deletion queue."""
    log = _Log(queued=[("results/x", "rows"), ("results/y", "rows")])
    lake = _Lake()
    removed = remove_bodies(tmp_path, log, lake)
    assert lake.ran == [
        "SELECT table_name FROM",
        "register 2",
        "BEGIN",
        "CALL lake.set_commit_message('host', 'collect',",
        "DELETE FROM lake.results.alpha_score",
        "DELETE FROM lake.results.execution_outcome",
        "COMMIT",
        "unregister",
    ]
    assert log.done == [("dequeued", "results/x"), ("dequeued", "results/y")]
    assert removed == {"collected": ["results/x", "results/y"], "waiting": []}


@pytest.mark.proves("data_layer.an_owed_list_is_read_whatever_its_version")
@pytest.mark.parametrize(
    ("info", "read"),
    [
        pytest.param({"event_id": "e1", "attrs": "{}"}, ["e1"], id="one-envelope-written-before"),
        pytest.param(
            {"version": OWED_EVENTS_VERSION, "events": [{"event_id": "e1"}, {"event_id": "e2"}]},
            ["e1", "e2"],
            id="a-block-s-list",
        ),
        pytest.param({"version": 99, "events": []}, None, id="a-later-version-is-refused"),
    ],
)
def test_an_admission_snapshot_owes_each_event_its_info_lists(info, read):
    """A snapshot's commit info is read as the events it owes, in order: a list of them, or one
    envelope as a commit before the list wrote it; a version this Host does not know is refused
    by name."""
    if read is None:
        with pytest.raises(DataError, match=r"does not read"):
            owed_events(json.dumps(info))
    else:
        assert [event["event_id"] for event in owed_events(json.dumps(info))] == read
