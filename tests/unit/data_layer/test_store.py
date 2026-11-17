"""What StorePort refuses before it touches a file: a handle outside its folders, a row set it
cannot admit."""

from __future__ import annotations

import datetime
import uuid

import pyarrow as pa
import pytest

from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer.chain import canonical, digest
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


def arrow(rows: list[dict]) -> pa.Table:
    """An admission's rows as the Arrow table StorePort stages."""
    return pa.table(as_columns(rows))


@pytest.mark.proves("data_layer.stage_refuses_what_it_cannot_admit")
@pytest.mark.parametrize(
    ("table", "rows", "legacy", "code"),
    [
        pytest.param(
            "raw_bar",
            arrow([ROW, ROW | {"close": 2.0}]),
            False,
            "admission_key_repeated",
            id="key twice",
        ),
        pytest.param(
            "alpha_score", arrow([ROW]), False, "world_table_not_admitted", id="not admitted"
        ),
        pytest.param(
            "session", arrow([ROW]), True, "legacy_table_unanchored", id="legacy, no date"
        ),
        pytest.param(
            "raw_bar",
            arrow([ROW | {"session_date": datetime.date(2024, 1, 2)}]),
            False,
            "admission_column_unsupported",
            id="a date column",
        ),
    ],
)
def test_staging_refuses_a_repeated_key_and_an_unadmitted_table(table, rows, legacy, code):
    """Two versions of one key in one admission, a table StorePort does not admit, a legacy
    batch with no valid date or a column canonical() has no form for are refused by name; a
    world row is digested with its provider's lineage where its key has one."""
    with pytest.raises(DataError) as refused:
        digested(table, "yf", rows, legacy=legacy)
    assert refused.value.failure.code == f"data_layer.{code}"
    world = [sorted(digested(table, "yf", arrow([ROW])).column_names) for table in WORLD]
    assert world == [
        ["listing_id", "provider_id", "session_date", "value_digest"],
        ["listing_id", "session_date", "value_digest"],
    ]
    outcome = ROW | {"artifact_id": "a", "formation_session": "2024-01-02"}
    assert digested("execution_outcome", "yf", arrow([outcome])).to_pylist() == [outcome]


WORLD = ("raw_bar", "listing")
BAR = ROW | {"open": 1.25, "close": 2.5, "volume": 1200.0}
SYMBOL = {"listing_id": ROW["listing_id"], "symbol": "AAA", "valid_from": "2024-01-02"}


@pytest.mark.proves("data_layer.stage_digest_is_canonicals")
@pytest.mark.parametrize(
    ("table", "row", "later", "legacy"),
    [
        pytest.param("raw_bar", BAR | {"adjusted_close": 2.4}, {}, False, id="float"),
        pytest.param("raw_bar", BAR | {"adjusted_close": None}, {}, False, id="null"),
        pytest.param("raw_bar", BAR, {}, False, id="adjusted close absent"),
        pytest.param(
            "raw_bar", ROW | {"volume": 1200, "shares": -(2**62)}, {}, False, id="integer"
        ),
        pytest.param(
            "raw_bar",
            ROW | {"listing_id": '上海 Ünï \\ "q"\t\x01'},
            {},
            False,
            id="non-ASCII, escaped text",
        ),
        pytest.param(
            "raw_bar",
            ROW | {"session_date": "2024-02-29", "ok": True},
            {},
            False,
            id="date string, boolean",
        ),
        pytest.param(
            "raw_bar",
            BAR | {"open": 1e-05, "high": 1e16, "low": -0.0, "close": 123456789.0},
            {},
            False,
            id="float forms",
        ),
        pytest.param(
            "raw_bar",
            BAR | {"open": float("nan"), "high": float("-inf")},
            {},
            False,
            id="nan and infinity",
        ),
        pytest.param(
            "provider_symbol",
            SYMBOL | {"valid_to": "2024-01-09"},
            {"valid_to": None},
            True,
            id="legacy interval, one with an end",
        ),
    ],
)
def test_a_world_rows_digest_is_its_canonical_digest(table, row, later, legacy):
    """Each digest built over an admission's columns is blake3 over canonical() of its row, byte
    for byte, for every column type a row can carry."""
    other = row | {"valid_from" if legacy else "session_date": "2023-01-03"} | later
    staged = digested(table, "yf", arrow([row, other]), legacy=legacy)
    content = [
        {name: value for name, value in given.items() if name != "source_observed_at"}
        | {"provider_id": "yf"}
        for given in (row, other)
    ]
    assert staged.column("value_digest").to_pylist() == [
        digest(canonical(given)) for given in content
    ]
    if legacy:
        assert staged.column("open_digest").to_pylist() == [
            digest(canonical(given | {"valid_to": None})) if given["valid_to"] else None
            for given in content
        ]


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


@pytest.mark.proves("data_layer.a_source_time_is_not_content")
def test_a_source_time_moves_no_digest_and_a_closed_interval_carries_its_open_one():
    """The source's write time stays out of a row's value digest, so a later write of the same
    values is the same version; a legacy interval with an end also carries its open version's
    digest, the digest of the same row with no end."""
    bar = ROW | {"close": 1.0, "source_observed_at": "2024-02-01 00:00:00+00"}
    later = bar | {"source_observed_at": "2024-03-01 00:00:00+00"}
    first, again = (digested("raw_bar", "yf", arrow([row])).to_pylist()[0] for row in (bar, later))
    symbol = {"provider_id": "yf", "symbol": "AAA", "valid_from": "2024-01-05"}
    symbol |= {"listing_id": ROW["listing_id"]}
    closed, still_open = (
        digested(
            "provider_symbol", "yf", arrow([symbol | {"valid_to": end}]), legacy=True
        ).to_pylist()
        for end in ("2024-01-08", None)
    )
    assert first["value_digest"] == again["value_digest"]
    assert first["source_observed_at"] == bar["source_observed_at"]
    assert closed[0]["open_digest"] == still_open[0]["value_digest"]
    assert "open_digest" not in still_open[0]
