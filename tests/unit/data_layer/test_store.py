"""What StorePort refuses before it touches a file: a handle outside its folders, a row set it
cannot admit."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pyarrow as pa
import pytest

from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer.store import admissible, as_columns, delete, expirable

TEST_SOURCE = "src/alphalattice/kernel/data_layer/store.py"


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
