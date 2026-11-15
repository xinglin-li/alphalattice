"""What StorePort refuses before it touches a file: a handle outside its folders, a row set it
cannot admit."""

from __future__ import annotations

import uuid

import pytest

from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer.store import delete, digested

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
    ("table", "rows", "code"),
    [
        pytest.param(
            "raw_bar", [ROW, ROW | {"close": 2.0}], "admission_key_repeated", id="a key twice"
        ),
        pytest.param("sector", [ROW], "world_table_not_admitted", id="a table it does not admit"),
    ],
)
def test_staging_refuses_a_repeated_key_and_an_unadmitted_table(table, rows, code):
    """Two versions of one key in one admission, or a world table StorePort does not admit, are
    refused by name; a row set it admits is digested with its provider."""
    with pytest.raises(DataError) as refused:
        digested(table, "yf", rows)
    assert refused.value.failure.code == f"data_layer.{code}"
    assert [sorted(row) for row in digested("raw_bar", "yf", rows[:1])] == [
        ["listing_id", "provider", "session_date", "value_digest"]
    ]
