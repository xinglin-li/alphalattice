"""What StorePort refuses before it touches a file: a handle outside its folders, a row set it
cannot admit."""

from __future__ import annotations

import uuid

import pytest

from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer.store import as_columns, delete, digested

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


@pytest.mark.proves("data_layer.stage_refuses_what_it_cannot_admit")
@pytest.mark.parametrize(
    ("table", "rows", "legacy", "code"),
    [
        pytest.param(
            "raw_bar", [ROW, ROW | {"close": 2.0}], False, "admission_key_repeated", id="key twice"
        ),
        pytest.param("alpha_score", [ROW], False, "world_table_not_admitted", id="not admitted"),
        pytest.param("session", [ROW], True, "legacy_table_unanchored", id="legacy, no date"),
    ],
)
def test_staging_refuses_a_repeated_key_and_an_unadmitted_table(table, rows, legacy, code):
    """Two versions of one key in one admission, a table StorePort does not admit, or a legacy
    batch with no valid date are refused by name; a world row is digested with its provider where
    its key has one, and a results row is kept as it is."""
    with pytest.raises(DataError) as refused:
        digested(table, "yf", rows, legacy=legacy)
    assert refused.value.failure.code == f"data_layer.{code}"
    world = [
        sorted(row) for table in ("raw_bar", "listing") for row in digested(table, "yf", [ROW])
    ]
    assert world == [
        ["listing_id", "provider", "session_date", "value_digest"],
        ["listing_id", "session_date", "value_digest"],
    ]
    outcome = ROW | {"artifact_id": "a", "formation_session": "2024-01-02"}
    assert digested("execution_outcome", "yf", [outcome]) == [outcome]


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
